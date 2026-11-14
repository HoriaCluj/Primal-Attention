"""Singular-value spectra of every attention matrix of one trained model (Figure 1 of the paper).

For each fixed evaluation image, head and layer, the attention matrix (197 x 197, class token
included) is formed in float64 and its singular values computed. Per (layer, matrix) the script
keeps sums and sums of squares of the normalized cumulative curves over images, per head, so the
plotting script (figure1_plots.py) can form means and standard deviations at any grouping, plus
per-image, per-head summary metrics. Nothing in the repository is modified: matrices are read
with forward hooks.

Matrices:
  softmax layers   'softmax'      the post-softmax attention (input of timm Attention.attn_drop)
  Primal layer     'K_full'       K = Phi_q F_X^T F_X Phi_k^T, the layer's full kernel (used for the eta comparison)
                   'E_L_Rt'       E Lambda R^T, the paper's s-direction reconstruction
                   'E_L2N_Rt'     E (Lambda^2/N) R^T, the same with the code's scaling (engine_primal.py:44-52)
                   'softmax_qk'   softmax(q k^T / sqrt(p)) from the Primal layer's own q and k
  Phi_q, Phi_k, F_X, E and R are rebuilt with train_with_j_log.primal_factors (checked against the
  model's returned escore here, and against engine_primal's J by check_j_logger.py).

Curves per matrix (sv sorted descending):
  c      cumsum(sv) / sum(sv)            the paper's "normalized cumulative singular value"
  c2     cumsum(sv^2) / sum(sv^2)        explained energy
  c64    cumsum(sv[:64]) / sum(sv[:64])  first 64 values only; comparable between rank-64 Primal and 197-rank softmax
  lrs    log10(sv / sv_1), clamped at 1e-16
Metrics per (image, head): stable_rank sum(sv^2)/sv_1^2, effective_rank exp(entropy of sv/sum(sv)),
rank_1e-10 and rank_1e-14 (count of sv > tol * sv_1), sigma_1, c_at_0 (= sv_1 / sum(sv)).

Run from anywhere (CPU works; about 1-3 min per model per 100 images on a laptop CPU):
    python figure1_spectra.py --model primal_plus --checkpoint <run>/checkpoint.pth --name primal_eta0.05_seed0 \
        --data-path <imagenet-100> --images-per-class 10 --out <spectra dir>
--model: primal_plus | deit (checkpoint from main.py) | deit_official (--checkpoint is the official weights URL).
Writes <out>/<name>.npz and <out>/<name>_images.txt.
"""
import argparse
import copy
import json
import os
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
CV = os.path.normpath(os.path.join(HERE, "..", "..", "PrimalAttention_CV"))
sys.path[:0] = [HERE, CV]

from train_with_j_log import primal_factors  # noqa: E402

FIRST = 64          # head dimension p: the rank bound of K_full and of the reconstructions
CURVES = ("c", "c2", "c64", "lrs")
METRICS = ("stable_rank", "effective_rank", "rank_1e-10", "rank_1e-14", "sigma_1", "c_at_0")


def load_model(kind, checkpoint):
    import models          # noqa: F401  (repo's DeiT builders)
    import models_combo
    if kind == "deit_official":
        state = torch.hub.load_state_dict_from_url(checkpoint, map_location="cpu", check_hash=True)
    else:
        state = torch.load(checkpoint, map_location="cpu")
    ckpt_args = state.get("args")
    weights = state["model"]
    num_classes = weights["head.weight"].shape[0]
    if kind == "primal_plus":
        a = ckpt_args
        assert a.num_ksvd_layer == 1, "only the last layer is expected to be Primal"
        model = models_combo.primal_small_patch16_224(low_rank=a.low_rank, rank_multi=a.rank_multi,
                                                      num_ksvd_layer=a.num_ksvd_layer, num_classes=num_classes,
                                                      drop_path_rate=0.0)
    else:
        model = models.deit_small_patch16_224(num_classes=num_classes)
    model.load_state_dict(weights)   # strict: any mismatch is an error
    return model.eval(), ckpt_args, state.get("epoch")


def image_list(data_path, per_class):
    """The first `per_class` files (sorted by name) of every class in <data_path>/val."""
    from torchvision.datasets import ImageFolder
    dataset = ImageFolder(os.path.join(data_path, "val"))   # samples are sorted by class, then file name
    taken, chosen = {}, []
    for path, label in dataset.samples:
        if taken.get(label, 0) < per_class:
            chosen.append((path, label))
            taken[label] = taken.get(label, 0) + 1
    return chosen


def matrix_stats(sv):
    """sv (..., N) float64, descending. Returns curves (..., N or 64) and metrics (...)."""
    total = sv.sum(-1, keepdim=True)
    energy = (sv ** 2).sum(-1, keepdim=True)
    head = sv[..., :FIRST]
    p = sv / total
    curves = {
        "c": sv.cumsum(-1) / total,
        "c2": (sv ** 2).cumsum(-1) / energy,
        "c64": head.cumsum(-1) / head.sum(-1, keepdim=True),
        "lrs": torch.log10((sv / sv[..., :1]).clamp_min(1e-16)),
    }
    metrics = {
        "stable_rank": energy[..., 0] / sv[..., 0] ** 2,
        "effective_rank": torch.exp(-(p * torch.log(p.clamp_min(1e-300))).sum(-1)),
        "rank_1e-10": (sv > 1e-10 * sv[..., :1]).sum(-1).double(),
        "rank_1e-14": (sv > 1e-14 * sv[..., :1]).sum(-1).double(),
        "sigma_1": sv[..., 0],
        "c_at_0": sv[..., 0] / total[..., 0],
    }
    return curves, metrics


class Accumulator:
    """Per (layer, matrix): curve sums and sums of squares over images, per head; metrics per image and head."""

    def __init__(self):
        self.sums, self.metrics, self.count = {}, {}, {}

    def add(self, layer, name, mat, svd_device):
        sv = torch.linalg.svdvals(mat.to(svd_device, torch.float64)).cpu()   # (B, H, N)
        curves, metrics = matrix_stats(sv)
        key = "L{:02d}.{}".format(layer, name)
        for c in CURVES:
            s, q = curves[c].sum(0), (curves[c] ** 2).sum(0)               # (H, len)
            if key + "." + c in self.sums:
                self.sums[key + "." + c][0] += s
                self.sums[key + "." + c][1] += q
            else:
                self.sums[key + "." + c] = [s, q]
        for m in METRICS:
            self.metrics.setdefault(key + "." + m, []).append(metrics[m])
        self.count[key] = self.count.get(key, 0) + sv.shape[0]

    def arrays(self):
        out = {}
        for k, (s, q) in self.sums.items():
            out[k + ".sum"] = s.numpy()
            out[k + ".sumsq"] = q.numpy()
        for k, v in self.metrics.items():
            out[k + ".per_image"] = torch.cat(v).numpy()                    # (images, H)
        for k, n in self.count.items():
            out[k + ".n_images"] = np.array(n)
        return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", required=True, choices=["primal_plus", "deit", "deit_official"])
    parser.add_argument("--checkpoint", required=True, help="checkpoint.pth, or the official weights URL")
    parser.add_argument("--name", required=True, help="run name, e.g. primal_eta0.05_seed0 (output file name)")
    parser.add_argument("--data-path", required=True, help="folder with val/<class>/<images>")
    parser.add_argument("--images-per-class", type=int, default=10)
    parser.add_argument("--out", required=True)
    parser.add_argument("--batch", type=int, default=50)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--svd-device", default=None, help="default: same as --device")
    cli = parser.parse_args()
    svd_device = cli.svd_device or cli.device
    os.makedirs(cli.out, exist_ok=True)
    started = time.time()

    from timm.models.vision_transformer import Attention as SoftmaxAttention
    from PIL import Image
    from datasets import build_transform

    model, ckpt_args, epoch = load_model(cli.model, cli.checkpoint)
    model.to(cli.device)
    tf_args = SimpleNamespace(input_size=getattr(ckpt_args, "input_size", 224),
                              eval_crop_ratio=getattr(ckpt_args, "eval_crop_ratio", 0.875))
    transform = build_transform(False, tf_args)   # the repo's eval transform: resize 256, center crop 224
    images = image_list(cli.data_path, cli.images_per_class)
    with open(os.path.join(cli.out, cli.name + "_images.txt"), "w") as f:
        f.write("\n".join(os.path.relpath(p, cli.data_path) for p, _ in images) + "\n")

    # Hooks: softmax layers through attn_drop's input; the Primal layer's input and returned scores.
    captured, handles, softmax_layers, primal_layer = {}, [], [], None
    for i, block in enumerate(model.blocks):
        if isinstance(block.attn, SoftmaxAttention):
            softmax_layers.append(i)
            handles.append(block.attn.attn_drop.register_forward_hook(
                lambda _m, inputs, _out, i=i: captured.__setitem__(i, inputs[0])))
        else:
            primal_layer = i
            handles.append(block.attn.register_forward_pre_hook(
                lambda _m, inputs: captured.__setitem__("x", inputs[0])))
            handles.append(block.attn.register_forward_hook(
                lambda _m, _inputs, out: captured.__setitem__("escore", out[1][0])))
    primal64 = copy.deepcopy(model.blocks[primal_layer].attn).double() if primal_layer is not None else None

    acc = Accumulator()
    correct, max_rel_err = 0, 0.0
    for start in range(0, len(images), cli.batch):
        chunk = images[start:start + cli.batch]
        batch = torch.stack([transform(Image.open(p).convert("RGB")) for p, _ in chunk]).to(cli.device)
        labels = torch.tensor([l for _, l in chunk])
        captured.clear()
        with torch.no_grad():
            out = model(batch)
            logits = out[0] if isinstance(out, tuple) else out
            correct += int((logits.argmax(1).cpu() == labels).sum())   # sanity check that weights and labels match
            for i in softmax_layers:
                acc.add(i, "softmax", captured[i], svd_device)
            if primal_layer is not None:
                x = captured["x"].double()
                n_tokens = x.shape[1]
                phi_q, phi_k, fx_t, e, r = primal_factors(primal64, x)
                ref = captured["escore"].double()
                max_rel_err = max(max_rel_err, ((e - ref).norm() / ref.norm()).item())
                lam = primal64.Lambda[None, :, None, :]                       # (1, H, 1, s)
                rt = r.transpose(-1, -2)
                qk = primal64.qk(x).reshape(x.shape[0], n_tokens, 2, primal64.num_heads, primal64.head_dim)
                q, k = qk.permute(2, 0, 3, 1, 4)
                acc.add(primal_layer, "K_full", phi_q @ fx_t @ fx_t.transpose(-1, -2) @ phi_k.transpose(-1, -2), svd_device)
                acc.add(primal_layer, "E_L_Rt", (e * lam) @ rt, svd_device)
                acc.add(primal_layer, "E_L2N_Rt", (e * lam ** 2 / n_tokens) @ rt, svd_device)
                acc.add(primal_layer, "softmax_qk", torch.softmax(q @ k.transpose(-1, -2) * primal64.head_dim ** -0.5, -1),
                        svd_device)
        print("{}: {}/{} images, {:.0f} s".format(cli.name, start + len(chunk), len(images), time.time() - started),
              flush=True)
    for h in handles:
        h.remove()

    if primal_layer is not None:
        assert max_rel_err < 1e-4, "rebuilt E differs from the model's escore: {:.2e}".format(max_rel_err)
    meta = {
        "name": cli.name, "model": cli.model, "checkpoint": cli.checkpoint, "epoch": epoch,
        "eta": getattr(ckpt_args, "eta", None) if cli.model == "primal_plus" else None,
        "data_path": cli.data_path, "n_images": len(images), "images_per_class": cli.images_per_class,
        "softmax_layers": softmax_layers, "primal_layer": primal_layer,
        "primal_escore_max_rel_err": max_rel_err if primal_layer is not None else None,
        "top1_on_these_images": 100.0 * correct / len(images),
        "torch": torch.__version__, "seconds": round(time.time() - started, 1),
    }
    arrays = acc.arrays()
    if primal_layer is not None:
        arrays["primal.Lambda"] = model.blocks[primal_layer].attn.Lambda.detach().cpu().double().numpy()
    np.savez_compressed(os.path.join(cli.out, cli.name + ".npz"), meta=json.dumps(meta), **arrays)
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
