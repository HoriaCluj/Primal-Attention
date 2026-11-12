"""Run the repository's main_combo.py unchanged, adding a per-head J and KKT-residual log.

The repository logs only the batch-level KSVD term (train_loss_ksvd in log.txt). This wrapper
replaces engine_primal.train_one_epoch with a version that calls the original and then, on the
main process only, evaluates the Primal layer on a fixed set of validation images and appends
one row per head to <output_dir>/J_log.csv:

  epoch                   -1 = before training, k = after epoch k (0-based, as in log.txt)
  J_mean, J_abs_mean      per-head J over the fixed images; J_abs_mean shows heads cancelling
  Jbar_sq                 (mean over heads and images of J)^2: what eta multiplies in the loss
  rho_mean, rho_median    per-direction KKT residual rho_l, averaged over directions and images
  residual_aggregate      (||G_e||^2 + ||G_r||^2) / (||W_e||^2 + ||W_r||^2), mean over images
  lambda_abs_min/max      range of |Lambda| for the head; lambda_near_zero counts |lambda| < 1e-3

The formulas follow the code's objective (engine_primal.py:44-52): Lambda enters squared and token
sums are means, so G_e = Phi'_q^T E Lambda^2 / N - W_r and G_r = Phi'_k^T R Lambda^2 / N - W_e,
with Phi'_q = Phi_q F_X^T (see analysis/timeseries_eta_ablation/PHASE1_REPORT.md, Section 3).

The diagnostics draw no random numbers: the images are chosen with a private NumPy RNG, use the
deterministic evaluation transform, and the model runs in eval mode without autocast. Training
is therefore identical with or without this wrapper.

Usage, from PrimalAttention_CV/ (works under torch.distributed.launch too):
    python /path/to/train_with_j_log.py main_combo.py <main_combo.py arguments>
Environment: J_LOG_IMAGES (default 32), J_LOG_BATCH (default 16).

If a job dies after an epoch's row is written but before that epoch's checkpoint is saved, the
resumed run logs that epoch again; keep the last row per (epoch, head) when analysing.
"""
import csv
import os
import runpy
import sys

import numpy as np
import torch
import torch.nn.functional as F

LAMBDA_NEAR_ZERO = 1e-3
FIELDS = ["epoch", "head", "J_mean", "J_abs_mean", "Jbar_sq", "rho_mean", "rho_median",
          "residual_aggregate", "lambda_abs_min", "lambda_abs_max", "lambda_near_zero", "n_images"]


def primal_attention(model):
    inner = model.module if hasattr(model, "module") else model
    return inner.blocks[-1].attn  # models_combo.py:23-34 puts the Primal layer last


def primal_factors(attn, x):
    """Phi_q, Phi_k (B,H,N,p), F_X^T (B,H,p,n) and E, R (B,H,N,s) from the layer input x (B,N,C).

    Mirrors PrimalAttention.forward and gen_weights (models_primal.py:51-80).
    """
    B, N, C = x.shape
    H, p = attn.num_heads, attn.head_dim
    qk = attn.qk(x).reshape(B, N, 2, H, p).permute(2, 0, 3, 1, 4)
    phi_q, phi_k = F.normalize(qk[0], p=2, dim=-1), F.normalize(qk[1], p=2, dim=-1)
    fx_t = x.transpose(-2, -1).reshape(B, H, p, N)
    if attn.embed_len > attn.low_rank * attn.rank_multi:
        idx = torch.linspace(0, N - 1, attn.low_rank * attn.rank_multi, dtype=int)
        fx_t = fx_t[:, :, :, idx]
    e = phi_q @ (fx_t @ attn.we.type_as(fx_t))   # escore = Phi_q W_{e|X}
    r = phi_k @ (fx_t @ attn.wr.type_as(fx_t))
    return phi_q, phi_k, fx_t, e, r


def ksvd_diagnostics(phi_q, phi_k, fx_t, we, wr, lam):
    """Per-image, per-head J (B,H), residuals rho (B,H,s) and aggregate residual (B,H)."""
    pq, pk = phi_q @ fx_t, phi_k @ fx_t           # Phi'_q, Phi'_k: (B,H,N,n)
    n_tokens = pq.shape[2]
    e, r = pq @ we, pk @ wr
    l2 = (lam ** 2)[None, :, None, :]
    j = 0.5 * (e ** 2 * l2).sum(-1).mean(2) + 0.5 * (r ** 2 * l2).sum(-1).mean(2) - (we * wr).sum((-1, -2))[None]
    g_e = pq.transpose(-1, -2) @ (e * l2) / n_tokens - wr
    g_r = pk.transpose(-1, -2) @ (r * l2) / n_tokens - we
    num = (g_e ** 2).sum(-2) + (g_r ** 2).sum(-2)
    den = (we ** 2).sum(-2) + (wr ** 2).sum(-2)
    return j, num / (den[None] + 1e-12), num.sum(-1) / den[None].sum(-1)


class JLogger:
    def __init__(self, args):
        from datasets import build_transform
        from torchvision.datasets import ImageFolder

        self.path = os.path.join(args.output_dir, "J_log.csv")
        dataset = ImageFolder(os.path.join(args.data_path, "val"), transform=build_transform(False, args))
        count = min(int(os.environ.get("J_LOG_IMAGES", 32)), len(dataset))
        indices = np.random.RandomState(0).permutation(len(dataset))[:count]
        self.images = torch.stack([dataset[int(i)][0] for i in indices])
        self.batch = int(os.environ.get("J_LOG_BATCH", 16))

    @torch.no_grad()
    def log(self, model, epoch):
        attn = primal_attention(model)
        device = attn.we.device
        captured = {}
        handle = attn.register_forward_pre_hook(lambda _m, inputs: captured.__setitem__("x", inputs[0]))
        was_training = model.training
        model.eval()
        js, rhos, aggregates = [], [], []
        try:
            for start in range(0, len(self.images), self.batch):
                model(self.images[start:start + self.batch].to(device))
                phi_q, phi_k, fx_t, _, _ = primal_factors(attn, captured["x"].float())
                j, rho, agg = ksvd_diagnostics(phi_q.double(), phi_k.double(), fx_t.double(),
                                               attn.we.double(), attn.wr.double(), attn.Lambda.double())
                js.append(j.cpu()); rhos.append(rho.cpu()); aggregates.append(agg.cpu())
        finally:
            handle.remove()
            model.train(was_training)
        j, rho, agg = torch.cat(js), torch.cat(rhos), torch.cat(aggregates)
        lam = attn.Lambda.detach().abs().cpu()
        jbar_sq = j.mean().item() ** 2
        new_file = not os.path.exists(self.path)
        with open(self.path, "a", newline="") as handle_csv:
            writer = csv.writer(handle_csv)
            if new_file:
                writer.writerow(FIELDS)
            for h in range(j.shape[1]):
                writer.writerow([epoch, h, j[:, h].mean().item(), j[:, h].abs().mean().item(), jbar_sq,
                                 rho[:, h].mean().item(), rho[:, h].median().item(), agg[:, h].mean().item(),
                                 lam[h].min().item(), lam[h].max().item(), int((lam[h] < LAMBDA_NEAR_ZERO).sum()),
                                 j.shape[0]])


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    script = os.path.abspath(sys.argv[1])
    sys.argv = sys.argv[1:]                      # the repository script sees its own arguments
    sys.path.insert(0, os.path.dirname(script))

    import engine_primal
    import utils

    original = engine_primal.train_one_epoch
    state = {}

    def train_one_epoch(model, *pargs, **kwargs):
        args = kwargs["args"]
        epoch = pargs[4] if len(pargs) > 4 else kwargs["epoch"]
        logging = utils.is_main_process() and bool(args.output_dir)
        if logging and "logger" not in state:
            state["logger"] = JLogger(args)
            if epoch == 0:
                state["logger"].log(model, -1)
        stats = original(model, *pargs, **kwargs)
        if logging:
            state["logger"].log(model, epoch)
        return stats

    engine_primal.train_one_epoch = train_one_epoch  # main_combo.py imports it from engine_primal
    runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
    main()
