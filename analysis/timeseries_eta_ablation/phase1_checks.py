"""Phase 1 numerical checks for the eta-ablation brief. Read-only w.r.t. the repo.

Loads a checkpoint from an unmodified `main.py` run and verifies, on real test
batches, the facts PHASE1_REPORT.md relies on:

  1. Forward hooks capture everything needed, and the captured F_X reproduces the
     layer's own scores E and R.
  2. The hand-written J equals the value computed by models/loss.py.
  3. The KKT residual formula equals autograd's gradient of the code's J.
  4. The rank cap of K holds, and how long the float64 SVDs take.

Usage (from anywhere):
  .venv/bin/python analysis/timeseries_eta_ablation/phase1_checks.py <run_dir>
where <run_dir> holds configuration.json, normalization.pickle and checkpoints/.
"""
import json
import os
import pickle
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CODE = os.path.join(REPO, "PrimalAttention_TimeSeries", "Primal.+Trans.")
sys.path.insert(0, CODE)
os.chdir(CODE)

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from data_loader.data import Normalizer, data_factory  # noqa: E402
from data_loader.dataset import ClassiregressionDataset, collate_superv  # noqa: E402
from models.loss import get_loss_module  # noqa: E402
from models.model_factory import model_factory  # noqa: E402


class Capture:
    """Forward hooks only; nothing in the model is modified."""

    def __init__(self, model):
        self.d = {}
        layer0 = model.encoder.attn_layers[0].attention
        layer1 = model.encoder.attn_layers[1].attention
        self.handles = [
            # SoftmaxAttention never stores attn; its nn.Dropout receives it.
            layer0.inner_attention.dropout.register_forward_hook(self._softmax),
            layer1.register_forward_pre_hook(self._tokens),
            layer1.inner_attention.register_forward_hook(self._primal),
        ]
        self.layer1 = layer1

    def _softmax(self, _m, inputs, _o):
        self.d["attn0"] = inputs[0].detach()  # (B, H, N, N)

    def _tokens(self, _m, inputs):
        self.d["x"] = inputs[0].detach()  # (B, N, d_model)

    def _primal(self, _m, _i, output):
        _, s, lam = output
        self.d.update(E=s[0], R=s[1], We=s[2], Wr=s[3], phi_q=s[4], phi_k=s[5], lam=lam)

    def fx(self):
        """F_X exactly as AttentionLayer.forward builds it: (B, H, n, p)."""
        x = self.d["x"]
        B, L, _ = x.shape
        H, p, n = self.layer1.n_heads, self.layer1.d_keys, self.layer1.len
        idx = torch.linspace(0, L - 1, n, dtype=int)
        return x.transpose(-2, -1).reshape(B, H, p, L)[:, :, :, idx].transpose(-2, -1)

    def close(self):
        for h in self.handles:
            h.remove()


def main(run_dir):
    config = json.load(open(os.path.join(run_dir, "configuration.json")))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cls = data_factory[config["data_class"]]
    train = cls(config["data_dir"], pattern=config["pattern"], n_proc=-1, config=config)
    test = cls(config["data_dir"], pattern=config["val_pattern"], n_proc=-1, config=config)
    norm = Normalizer(**pickle.load(open(os.path.join(run_dir, "normalization.pickle"), "rb")))
    test.feature_df.loc[test.all_IDs] = norm.normalize(test.feature_df.loc[test.all_IDs])

    model = model_factory(config, train)
    ckpt = torch.load(os.path.join(run_dir, "checkpoints", "model_last.pth"),
                      map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    print("checkpoint epoch:", ckpt["epoch"], "| device:", device)

    loader = DataLoader(ClassiregressionDataset(test, test.all_IDs), batch_size=8, shuffle=False,
                        collate_fn=lambda b: collate_superv(b, max_len=model.max_len))
    X, y, masks, _ = next(iter(loader))
    cap = Capture(model)
    with torch.no_grad():
        pred, scores, lams = model(X.to(device), masks.to(device))
    d = cap.d
    F_X = cap.fx()
    print("shapes:", {k: tuple(v.shape) for k, v in d.items()}, "F_X", tuple(F_X.shape))

    # 1. Captured F_X must reproduce the layer's own scores.
    We_x = torch.einsum("bhnp,hns->bhps", F_X, d["We"].type_as(F_X))
    Wr_x = torch.einsum("bhnp,hns->bhps", F_X, d["Wr"].type_as(F_X))
    E = torch.einsum("bhNp,bhps->bhNs", d["phi_q"], We_x)
    R = torch.einsum("bhNp,bhps->bhNs", d["phi_k"], Wr_x)
    print("1. relative error E: %.2e  R: %.2e" % (
        ((E - d["E"]).norm() / d["E"].norm()).item(), ((R - d["R"]).norm() / d["R"].norm()).item()))
    a0 = d["attn0"]
    print("   softmax attn", tuple(a0.shape), "row-sum range [%.6f, %.6f]" % (
        a0.sum(-1).min().item(), a0.sum(-1).max().item()))
    print("   phi_q norm range [%.6f, %.6f]" % (
        d["phi_q"].norm(dim=-1).min().item(), d["phi_q"].norm(dim=-1).max().item()))

    # 2. Hand-written J against models/loss.py.
    N = d["E"].shape[2]
    lam = d["lam"]
    B = X.shape[0]
    term_e = 0.5 * ((d["E"] * lam[None, :, None, :]) ** 2).sum(-1).mean(2)  # (B, H)
    term_r = 0.5 * ((d["R"] * lam[None, :, None, :]) ** 2).sum(-1).mean(2)
    trace = (d["We"] * d["Wr"]).sum((-1, -2))  # (H,)
    J_head = term_e + term_r - trace[None, :]  # (B, H)
    J_mean = J_head.mean(1)
    loss_module = get_loss_module(config)
    with torch.no_grad():
        _, _, ksvd = loss_module(lams, pred, y.to(device), config["eta"], scores)
    print("2. max |J_mean^2 - loss.py| = %.2e  (loss.py mean %.4e)" % (
        (J_mean ** 2 - ksvd).abs().max().item(), ksvd.mean().item()))
    print("   J per head, sequence 0:", [round(v, 3) for v in J_head[0].tolist()])
    print("   J averaged over heads (sequence 0): %.4f" % J_mean[0].item())
    print("   lambda range [%.3f, %.3f], negative entries: %d / %d" % (
        lam.min().item(), lam.max().item(), (lam < 0).sum().item(), lam.numel()))

    # 3. KKT residual formula against autograd, one sequence.
    We = d["We"].detach().clone().requires_grad_(True)
    Wr = d["Wr"].detach().clone().requires_grad_(True)
    Fx = F_X[:1].detach()
    Pq = torch.einsum("bhNp,bhnp->bhNn", d["phi_q"][:1], Fx)  # Phi'_q, (1, H, N, n)
    Pk = torch.einsum("bhNp,bhnp->bhNn", d["phi_k"][:1], Fx)
    L2 = lam.detach() ** 2
    Ei = torch.einsum("bhNn,hns->bhNs", Pq, We)
    Ri = torch.einsum("bhNn,hns->bhNs", Pk, Wr)
    Jh = 0.5 * (Ei ** 2 * L2[None, :, None, :]).sum(-1).mean(2) \
        + 0.5 * (Ri ** 2 * L2[None, :, None, :]).sum(-1).mean(2) \
        - (We * Wr).sum((-1, -2))[None, :]
    ge_ad, gr_ad = torch.autograd.grad(Jh.sum(), [We, Wr])
    # Code's J: Lambda enters squared, sum over i becomes a mean over N.
    ge = torch.einsum("bhNn,bhNs->bhns", Pq, Ei.detach() * L2[None, :, None, :]) / N - Wr.detach()
    gr = torch.einsum("bhNn,bhNs->bhns", Pk, Ri.detach() * L2[None, :, None, :]) / N - We.detach()
    print("3. residual formula vs autograd, relative error G_e: %.2e  G_r: %.2e" % (
        ((ge[0] - ge_ad).norm() / ge_ad.norm()).item(), ((gr[0] - gr_ad).norm() / gr_ad.norm()).item()))
    num = (ge[0] ** 2).sum((-1, -2)) + (gr[0] ** 2).sum((-1, -2))
    den = (We ** 2).sum((-1, -2)) + (Wr ** 2).sum((-1, -2))
    print("   aggregate residual, by head:", [round(v, 3) for v in (num / den).tolist()])

    # 4. Rank cap and SVD timing, float64 on CPU.
    p, n = model.encoder.attn_layers[1].attention.d_keys, model.encoder.attn_layers[1].attention.len
    # Convert captured factors to float64 BEFORE multiplying, so K carries no float32 rounding.
    Fx64 = F_X[0].double().cpu()  # (H, n, p)
    Pq64 = d["phi_q"][0].double().cpu() @ Fx64.transpose(-1, -2)  # (H, N, n)
    Pk64 = d["phi_k"][0].double().cpu() @ Fx64.transpose(-1, -2)
    K = Pq64 @ Pk64.transpose(-1, -2)  # (H, N, N)
    def rank(m, tol=1e-10):
        s = torch.linalg.svdvals(m)
        return (s > tol * s[..., :1]).sum(-1).tolist()
    print("   rank by head (tol 1e-10): F_X", rank(Fx64), "| phi_q", rank(d["phi_q"][0].double().cpu()),
          "| phi_k", rank(d["phi_k"][0].double().cpu()), "| Phi'_q", rank(Pq64), "| Phi'_k", rank(Pk64))
    print("   rank of F_X head 0 at tol 1e-6/1e-4/1e-2:", [rank(Fx64[0], t) for t in (1e-6, 1e-4, 1e-2)])
    s_fx = torch.linalg.svdvals(Fx64[0])
    print("   sigma(F_X) head 0, indices 0,4,9,19,29,39,49,63:",
          ["%.3g" % s_fx[i].item() for i in (0, 4, 9, 19, 29, 39, 49, 63)])
    s_k = torch.linalg.svdvals(K[0])
    print("   sigma(K)/sigma_1 head 0, indices 0,9,19,29,31,33,39,49,63,64:",
          ["%.2e" % (s_k[i] / s_k[0]).item() for i in (0, 9, 19, 29, 31, 33, 39, 49, 63, 64)])
    uniq = torch.unique(F_X[0, 0].cpu(), dim=0).shape[0]
    print("   distinct F_X rows, head 0:", uniq, "of", F_X.shape[2])
    t0 = time.time()
    sv = torch.linalg.svdvals(K)
    t1 = time.time()
    sv0 = torch.linalg.svdvals(a0[0].double().cpu())
    t2 = time.time()
    ranks = (sv > 1e-10 * sv[:, :1]).sum(-1).tolist()
    print("4. K", tuple(K.shape), "numerical rank by head:", ranks, "| cap min(N,p,n) =", min(N, p, n))
    print("   sigma_1 by head:", [round(v, 2) for v in sv[:, 0].tolist()])
    print("   SVD seconds per 8-head set: K %.3f, softmax %.3f" % (t1 - t0, t2 - t1))
    sig_pred = N / (lam.detach().cpu().double() ** 2)
    print("   N/lambda^2, head 0, sorted first 5:",
          [round(v, 1) for v in torch.sort(sig_pred[0], descending=True)[0][:5].tolist()])
    print("   sigma(K), head 0, first 5:", [round(v, 1) for v in sv[0, :5].tolist()])
    cap.close()


if __name__ == "__main__":
    main(sys.argv[1])
