"""Kernel formation, spectral metrics and KSVD diagnostics.

The formulas follow the code's regularizer (models/loss.py:37-43), which differs from
the paper: Lambda enters squared and token sums are means, so Lambda_eff = Lambda^2 / N
and sigma_l corresponds to N / lambda_l^2. See PHASE1_REPORT.md, Section 3.
"""
import numpy as np
import torch

LAMBDA_NEAR_ZERO = 1e-3


def eval_indices(n_total, k=128, seed=0):
    """Fixed evaluation subset. Uses a private RNG, so training randomness is untouched."""
    return np.random.RandomState(seed).permutation(n_total)[:min(k, n_total)].tolist()


def feature_products(phi_q, phi_k, fx):
    """Phi'_q and Phi'_k, shape (B, H, N, n). Inputs must already be the wanted dtype."""
    fxt = fx.transpose(-1, -2)
    return phi_q @ fxt, phi_k @ fxt


def ksvd_diagnostics(pq, pk, we, wr, lam):
    """Per-sequence J and stationarity residuals for the code's objective.

    Returns J (B, H), E and R (B, H, N, s), rho (B, H, s) and aggregate residual (B, H).
    """
    n_tokens = pq.shape[2]
    e, r = pq @ we, pk @ wr
    l2 = (lam ** 2)[None, :, None, :]
    j_e = 0.5 * (e ** 2 * l2).sum(-1).mean(2)
    j_r = 0.5 * (r ** 2 * l2).sum(-1).mean(2)
    trace = (we * wr).sum((-1, -2))[None, :]
    j = j_e + j_r - trace
    g_e = pq.transpose(-1, -2) @ (e * l2) / n_tokens - wr
    g_r = pk.transpose(-1, -2) @ (r * l2) / n_tokens - we
    num = (g_e ** 2).sum(-2) + (g_r ** 2).sum(-2)  # (B, H, s)
    den = (we ** 2).sum(-2) + (wr ** 2).sum(-2)  # (H, s)
    rho = num / (den[None] + 1e-12)
    aggregate = num.sum(-1) / den[None].sum(-1)
    return j, e, r, rho, aggregate


def rank_at(sv, tol):
    return (sv > tol * sv[..., :1]).sum(-1)


def spectral_metrics(sv, s):
    """sv is (..., M), sorted descending. Returns a dict of tensors with shape (...)."""
    total = sv.sum(-1)
    energy = (sv ** 2).sum(-1)
    p = sv / total[..., None]
    entropy = -(p * torch.log(p.clamp_min(1e-300))).sum(-1)
    return {
        "stable_rank": energy / sv[..., 0] ** 2,
        "effective_rank": torch.exp(entropy),
        "c_s": sv[..., :s].sum(-1) / total,
        "c2_s": (sv[..., :s] ** 2).sum(-1) / energy,
        "sigma_1": sv[..., 0],
        "rank_1e-10": rank_at(sv, 1e-10).double(),
        "rank_1e-14": rank_at(sv, 1e-14).double(),
    }


def curves(sv):
    """Per-matrix cumulative curves, all (..., M)."""
    return {
        "c": sv.cumsum(-1) / sv.sum(-1, keepdim=True),
        "c2": (sv ** 2).cumsum(-1) / (sv ** 2).sum(-1, keepdim=True),
        "log10_rel_sigma": torch.log10((sv / sv[..., :1]).clamp_min(1e-300)),
    }


def captured_energy(e, r, k):
    """||Q_E^T K Q_R||_F^2 / ||K||_F^2, shape (B, H)."""
    q_e = torch.linalg.qr(e, mode="reduced")[0]
    q_r = torch.linalg.qr(r, mode="reduced")[0]
    inner = q_e.transpose(-1, -2) @ k @ q_r
    return (inner ** 2).sum((-1, -2)) / (k ** 2).sum((-1, -2))


def lambda_match(lam, sv, n_tokens):
    """Compare N/lambda^2 with the nearest singular value of K, measured in log ratio.

    lam is (H, s), sv is (B, H, M). Returns predicted, matched, index and relative
    error, each (B, H, s), plus a (H, s) mask that is True for near-zero lambda.
    """
    b = sv.shape[0]
    flagged = lam.abs() < LAMBDA_NEAR_ZERO
    pred = (n_tokens / lam ** 2)[None].expand(b, -1, -1)
    valid = sv > 1e-14 * sv[..., :1]
    log_gap = (torch.log(pred)[..., None] - torch.log(sv.clamp_min(1e-300))[:, :, None, :]).abs()
    log_gap = log_gap.masked_fill(~valid[:, :, None, :], float("inf"))
    index = log_gap.argmin(-1)
    matched = sv[:, :, None, :].expand(-1, -1, lam.shape[1], -1).gather(-1, index[..., None])[..., 0]
    rel_error = (matched - pred).abs() / pred
    return pred, matched, index, rel_error, flagged
