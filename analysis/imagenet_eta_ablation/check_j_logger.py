"""Check train_with_j_log.py against the repository's own code, on CPU, in a few seconds.

1. E and R rebuilt from the hooked layer input equal the scores the Primal layer returns.
2. The head- and image-mean of the logger's per-image J equals engine_primal.py's J (before squaring).
3. The residual G_e, G_r equals autograd's gradient of the per-image J with respect to W_e, W_r.

    /home/horia/envs/primal-cv-local/bin/python analysis/imagenet_eta_ablation/check_j_logger.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CV = os.path.join(HERE, "..", "..", "PrimalAttention_CV")
sys.path[:0] = [HERE, CV]

import torch  # noqa: E402

import models_combo  # noqa: E402
from train_with_j_log import ksvd_diagnostics, primal_attention, primal_factors  # noqa: E402

torch.manual_seed(0)
model = models_combo.primal_small_patch16_224(low_rank=96, rank_multi=10, num_ksvd_layer=1,
                                              num_classes=100, drop_path_rate=0.1).eval()
with torch.no_grad():
    model.blocks[-1].attn.Lambda.uniform_(0.2, 1.0)   # keep J away from trivial values
attn = primal_attention(model)
captured = {}
attn.register_forward_pre_hook(lambda _m, inputs: captured.__setitem__("x", inputs[0]))
images = torch.randn(3, 3, 224, 224)
with torch.no_grad():
    _, score_list, lambda_list = model(images)
    phi_q, phi_k, fx_t, e, r = primal_factors(attn, captured["x"])
print("number of params: {:,}".format(sum(p.numel() for p in model.parameters())))

# 1. Rebuilt scores against the returned scores (models_primal.py:85).
err_e = ((e - score_list[0][0]).norm() / score_list[0][0].norm()).item()
err_r = ((r - score_list[0][1]).norm() / score_list[0][1].norm()).item()
print("1. relative error, rebuilt E: {:.1e}  R: {:.1e}".format(err_e, err_r))

# 2. engine_primal.py:46-50, before squaring, on the same batch.
s = score_list[0]
lam_diag = torch.diag_embed(lambda_list[0])
loss_e = torch.mean((torch.einsum('...nd,...ds->...ns', s[0], lam_diag.unsqueeze(0).repeat(s[0].size(0), 1, 1, 1))).norm(dim=-1, p=2) ** 2) / 2
loss_r = torch.mean((torch.einsum('...nd,...ds->...ns', s[1], lam_diag.unsqueeze(0).repeat(s[1].size(0), 1, 1, 1))).norm(dim=-1, p=2) ** 2) / 2
loss_trace = torch.einsum('...ps,...pd->...sd', s[2], s[3].type_as(s[2])).mean(dim=0).trace()
j_engine = (loss_e + loss_r - loss_trace).item()
j, rho, agg = ksvd_diagnostics(phi_q.double(), phi_k.double(), fx_t.double(),
                               attn.we.detach().double(), attn.wr.detach().double(), attn.Lambda.detach().double())
print("2. J from engine_primal.py: {:.6f} | logger mean over heads and images: {:.6f} | abs diff {:.1e}".format(
    j_engine, j.mean().item(), abs(j_engine - j.mean().item())))

# 3. Residuals against autograd, for image 0.
we = attn.we.detach().double().clone().requires_grad_(True)
wr = attn.wr.detach().double().clone().requires_grad_(True)
pq, pk = (phi_q[:1].double() @ fx_t[:1].double()), (phi_k[:1].double() @ fx_t[:1].double())
l2 = (attn.Lambda.detach().double() ** 2)[None, :, None, :]
j0 = (0.5 * ((pq @ we) ** 2 * l2).sum(-1).mean(2) + 0.5 * ((pk @ wr) ** 2 * l2).sum(-1).mean(2)
      - (we * wr).sum((-1, -2))[None])
g_e_auto, g_r_auto = torch.autograd.grad(j0.sum(), [we, wr])
n_tokens = pq.shape[2]
g_e = pq.transpose(-1, -2) @ ((pq @ we.detach()) * l2) / n_tokens - wr.detach()
g_r = pk.transpose(-1, -2) @ ((pk @ wr.detach()) * l2) / n_tokens - we.detach()
print("3. relative error, G_e vs autograd: {:.1e}  G_r: {:.1e}".format(
    ((g_e[0] - g_e_auto).norm() / g_e_auto.norm()).item(), ((g_r[0] - g_r_auto).norm() / g_r_auto.norm()).item()))
num = (g_e[0] ** 2).sum(-2) + (g_r[0] ** 2).sum(-2)
den = (we.detach() ** 2).sum(-2) + (wr.detach() ** 2).sum(-2)
print("   aggregate residual from the logger vs recomputed, image 0: max abs diff {:.1e}".format(
    (agg[0] - num.sum(-1) / den.sum(-1)).abs().max().item()))
print("   shapes: J {}, rho {}, aggregate {}".format(tuple(j.shape), tuple(rho.shape), tuple(agg.shape)))
