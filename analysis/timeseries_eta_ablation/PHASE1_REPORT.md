# Phase 1 report: η ablation on Primal.+Trans

Status: reconnaissance and smoke test finished. **The sweep has not been started; it is waiting for review.**
Paths below are relative to `PrimalAttention_TimeSeries/Primal.+Trans./` unless stated. No original code,
configs or checkpoints were modified. Numerical checks are in `phase1_checks.py` (this folder).

**Scope of the numbers.** The Section 3 checks use one smoke-test checkpoint (UWaveGestureLibrary, η = 0.1,
seed 1, epoch 38) and, for the kernel and residual values, one test sequence. They validate the method. They say
nothing about the effect of η.

## Summary

- The proposed dataset is **UWaveGestureLibrary** (N = 315, p = 64, s = 40), with **Handwriting** as backup.
- Estimated sweep time on this machine is **about 2 to 2.5 hours**: 12 runs at the repo's 100 epochs, on the GeForce MX450.
- The code's regularizer differs from the paper's: **Λ enters squared and the sum over tokens is a mean**. So
  `σ_l` corresponds to `N/λ_l²`, not `1/λ_l`. The residual formula is adapted and matches autograd (relative error about 4e-7).
- All the requested tensors can be captured with forward hooks, with no source changes.
- Three items in the brief need adjusting; see "Decisions for review" at the end.

## 1. Primal.+Trans on UEA: launch and config

There is no config file. Settings come from CLI flags in `bashes/run_primal_trans.sh` (one line per dataset) with
defaults in `options.py`. Run `main.py` from `Primal.+Trans./`. UWaveGestureLibrary is line 34 of the script.

| Setting | Value for UWave | Source |
|---|---|---|
| `s` (`--low_rank`) | 40 (default 20) | script:34, `options.py:16` |
| `eta` | 0.1 (default 0.1) | script:34, `options.py:14` |
| `rank_multi` | 10 (default 10) | script:34, `options.py:18` |
| epochs | 100 (default 400) | script:34, `options.py:126` |
| batch size | 16 (default 128) | script:34, `options.py:140` |
| optimizer, lr | RAdam, 1e-3 | script:34, `options.py:130-131` |
| weight decay | 0 (`l2_reg` 0, `global_reg` off) | `options.py:142-144`, `main.py:160-168` |
| architecture | d_model 512, 8 heads, 2 layers, FFN 2048, dropout 0.1 | `options.py:161-169` |

The two layers are built explicitly: layer 0 is hard-coded `"softmax"` (`models/Primalformer.py:113-117`) and
layer 1 uses `--model primal_cos` (`:123-127`).

**Seeds and run names.** `main.py:31-36` sets every seed to 666. `--seed N` then re-seeds torch only
(`main.py:51-52`), so NumPy and `random` stay at 666. The smoke test confirms `--seed` is accepted. Output folders
are named from `--name`, `eta`, `s` and `rank_multi` (`running.py:76`), so each seed needs its own `--name`.

## 2. Implementation details

All in `models/Primal_Attention.py`. `B` = batch, `H` = heads, `N` = tokens.

| Object | Shape per head | Code |
|---|---|---|
| `p = d_keys` | 512 // 8 = **64** | `:64` |
| `n` | `min(N, s·rank_multi)` | `:79` |
| `W_e`, `W_r` | `n × s`, orthogonal init, shared across all sequences | `:82-83` |
| `F_X` | `n × p` | `:107-109` |
| `Φ_q`, `Φ_k` | `N × p`, L2-normalized (`F.normalize`) | `:43`, `:49-50` |
| `E`, `R` | `N × s` | `:52-53` |
| Output | `[E R]` → `Linear(2s, 64)`, then `Linear(512, 512)` across heads | `:40`, `:55`, `:75` |

How `F_X` is built:

```python
indices = torch.linspace(0, L-1, self.len, dtype=int)        # :107
x = x.transpose(-2,-1).reshape(B, H, self.d_keys, L)         # :108
x = x[:, :, :, indices].transpose(1, 2)                      # :109
```

- It uses the layer's input tokens, split into 8 contiguous 64-dimensional slices, one per head. There is no
  `W_q`/`W_k` projection and no normalization.
- The token selection is **evenly spaced and deterministic**, with no randomness.
- It is recomputed on every forward pass and is part of the autograd graph, so `J` also sends gradients into layer 0 through `F_X`.

## 3. Regularizer and numerical verification

`models/loss.py:37-43`. Per head, the code computes:

```
J_code = (1/2N) Σ_i ||Λ e_i||² + (1/2N) Σ_j ||Λ r_j||² − Tr(W_eᵀ W_r)
```

The value is averaged over heads, squared, then added as `loss_ce + eta * J²` (`:41-43`). Using `J²` and the
head mean both match the paper.

**Adapted formulas.** Writing `Λ̃ = Λ²/N`, the code's objective has the paper's form, so:

```
G_e = Φ'_qᵀ E Λ²/N − W_r
G_r = Φ'_kᵀ R Λ²/N − W_e
σ_l ↔ N / λ_l²
```

| Check (`phase1_checks.py`) | Result |
|---|---|
| `E`, `R` rebuilt from hooked `Φ`, `F_X`, `W` vs the layer's own | relative error 0 |
| Hand-written `J²` vs `models/loss.py` | max abs. difference 9.7e-8 (`loss.py` mean 6.4e-2) |
| `G_e`, `G_r` vs autograd of `J_code` | relative error 3.9e-7, 4.1e-7 |
| Softmax rows sum to 1, `‖φ‖ = 1` | yes |

## 4. Λ

`nn.Parameter(uniform_(Tensor(H, s)))`, so `U[0,1)` per entry (`Primal_Attention.py:38`). **There is no positivity
mechanism.** Because it enters squared, its sign does not matter. The checkpoint had 8 of 320 entries negative, range
[-0.003, 0.938]. Near-zero entries make `N/λ²` explode (largest value seen: 4.8e15), so `lambda_vs_sigma.png`
must flag or exclude directions with very small `|λ|`.

At η = 0, `Λ` receives zero gradient and should stay at its random initial values, so `J` and the `Λ`
comparison are not meaningful for that arm.

## 5. Figure 1

**The repo has no spectrum-analysis code.** A search for `svd`, `svdvals`, singular values, explained variance and
cumulative curves found only unrelated generic fairseq matches in `PrimalAttention_NLP`. So it cannot be determined
from the repo which matrix the paper decomposed for the Primal layer, whether values were squared, or how results were
aggregated. Figure 1's caption also refers to DeiT-Small/16 on ImageNet-1K, not this model.

## 6. Hooks (implemented and verified)

No source changes are needed. See `Capture` in `phase1_checks.py`.

| Target | Hook | Captures |
|---|---|---|
| `encoder.attn_layers[1].attention` | forward pre-hook | input tokens, from which `F_X` is rebuilt using `:107-109` |
| `…[1].attention.inner_attention` | forward hook | output list gives `E`, `R`, `W_e`, `W_r`, `Φ_q`, `Φ_k`; `Λ` |
| `…[0].attention.inner_attention.dropout` | forward hook on the `nn.Dropout` | post-softmax matrix (`:26-27`), shape `(B,H,N,N)` |

For the over-training logs, `main.main` has no callback. The plan is a wrapper in `analysis/` that wraps
`SupervisedRunner.train_epoch` at runtime and evaluates a fixed batch in eval mode.

## 7. Datasets and selection (Section 4.1)

`p = 64` for every dataset. `N` is the longest **training** sequence (`models/model_factory.py:21`). Shorter sequences
are zero-padded and longer test sequences are clipped (`data_loader/dataset.py:33-36`). All 10 paper datasets are present
locally. `s` and `rank_multi` are the script's values. "Attn map" is one layer-0 matrix at batch 16 (`N²/2048` MiB).

| Dataset | N | Train / test | Channels | Classes | s, rank_multi | n | N > p | Steps/epoch | Epochs | Attn map |
|---|---:|---|---:|---:|---|---:|:--:|---:|---:|---:|
| JapaneseVowels | 26 (padded) | 270 / 370 | 12 | 9 | 30, 5 | 26 | No | 17 | 100 | under 1 MiB |
| FaceDetection | 62 | 5890 / 3524 | 144 | 2 | 40, 5 | 62 | No | 369 | 100 | 2 MiB |
| SpokenArabicDigits | 93 (padded) | 6599 / 2199 | 13 | 10 | 30, 5 | 93 | Yes | 413 | 100 | 4 MiB |
| PEMS-SF | 144 | 267 / 173 | 963 | 7 | 40, 5 | 144 | Yes | 17 | 400 | 10 MiB |
| **Handwriting** | 152 | 150 / 850 | 3 | 26 | 20, 5 | 100 | Yes | 10 | 100 | 11 MiB |
| **UWaveGestureLibrary** | 315 | 120 / 320 | 3 | 8 | 40, 10 | 315 | Yes | 8 | 100 | 48 MiB |
| Heartbeat | 405 | 204 / 205 | 61 | 2 | 40, 10 | 400 | Yes | 13 | 100 | 80 MiB |
| SelfRegulationSCP1 | 896 | 268 / 293 | 6 | 2 | 30, 10 | 300 | Yes | 17 | 100 | 392 MiB |
| SelfRegulationSCP2 | 1152 | 200 / 180 | 7 | 2 | 20, 10 | 200 | Yes | 13 | 100 | 648 MiB |
| EthanolConcentration | 1751 | 261 / 263 | 3 | 4 | 40, 10 | 400 | Yes | 17 | 100 | 1.46 GiB (out of memory on this GPU) |

Eliminated: JapaneseVowels and FaceDetection have `N ≤ p`. SpokenArabicDigits is excluded in practice because it is
padded (attention is not masked) and costs about 50× as many steps per epoch as UWave. SCP1, SCP2 and Ethanol are too
large for 2 GB at the repo's batch size, or too tight. PEMS-SF needs 400 epochs.

**Proposal: UWaveGestureLibrary.** It has true fixed-length sequences, the model learns on it (paper: 88.4%), and it
fits comfortably in memory. Note that `n = N = 315`, so `F_X` uses every token.

**Backup: Handwriting.** It is the cheapest option and has `n = 100 < N`, so `F_X` really subsamples. Its weakness is that
the paper's accuracy is only 28.7%, so η would be tested on a poorly fitting model.

## 8. Hardware and timing

- GPU: NVIDIA GeForce MX450, 2048 MiB, CUDA available. RAM: about 3.7 GB under WSL2. 8 CPU cores.
- Smoke test: unmodified `main.py`, UWave, script settings, 38 epochs (304 steps), η = 0.1, `--seed 1`.

| Measured | Value |
|---|---|
| Train time | 3.12 s/epoch (0.39 s/step) |
| Validation pass (320 sequences) | 3.40 s |
| Total runtime, 38 epochs | 210 s |
| Peak GPU memory (sampled each second) | 1017 MiB of 2048 |
| float64 SVDs, 8 heads, one sequence, CPU | 0.054 s (K) + 0.061 s (softmax) |

| Estimated | Value |
|---|---|
| One run, 100 epochs (52 validations) | about 9 min |
| 12 runs, sequential | about 1.8 hours |
| Spectral analysis, 12 checkpoints × 128 sequences | about 15 to 25 min (SVDs alone: about 15 s per run) |
| Over-training diagnostics | not measured; allowing 10 to 20% |
| **Total** | **about 2 to 2.5 hours** |

The GPU can only hold one run at a time. RAM is tight, so spectra should be computed per batch and not accumulated.

## 9. Findings that affect the interpretation

**K's spectrum is extremely steep, which limits what `c(k)` can show.** `K = Φ_q F_Xᵀ F_X Φ_kᵀ` contains `F_X` twice,
so its singular-value decay is compounded. On the checkpoint, head 0, relative to `σ_1`:

| Index (0-based) | 0 | 9 | 19 | 29 | 39 | 49 | 63 | 64 |
|---|---|---|---|---|---|---|---|---|
| `σ_i/σ_1` | 1 | 1.8e-5 | 4.4e-8 | 3.7e-10 | 7.0e-12 | 2.5e-13 | 5.8e-15 | 2.3e-16 |

On a linear axis, `c(k)` will sit near 1 after a handful of indices, which may hide differences between η values. I
recommend a log-scale figure as well, with stable and effective rank as the main metrics.

**The brief's rank sanity check is weak here.** At tolerance `1e-10·σ_1` the numerical rank reads 28 to 33, even though
`F_X`, `Φ_q`, `Φ_k` and `Φ'` are all rank 64 and K's cap is 64. It correctly stays ≤ 64, but it cannot detect a failure
of the cap. Index 64 drops to machine precision, so I propose also reporting rank at `1e-14`. Factors must be converted
to float64 **before** multiplying.

**KKT conditions are per sequence, but the parameters are shared.** `W_e`, `W_r` and `Λ` are fixed across sequences
while `F_X` changes with each one, so `G = 0` cannot generally hold for all sequences at once. Large residuals are
expected even for η > 0. On the one sequence checked, the aggregate residual was 39 to 373 by head.

**Heads can cancel in `J`.** Only the head-mean is penalized. On the checked sequence, per-head `J` ranged from -0.22 to
0.81, with mean 0.18. Section 5.3's per-head logging will show this.

**"Best" accuracy is selected on the test set, and the gap is large.** `--val_pattern TEST` makes the test split the
validation set, and the README's metric is best accuracy across epochs. In the smoke test, best was 86.25% at epoch 2,
and the last epoch (38) was 75.0%, with training loss reaching 0.0009 by epoch 7.

## 10. Other paper/code discrepancies

| Item | Paper | Code |
|---|---|---|
| Λ in `J` | `eᵀΛe` | `‖Λe‖²`, i.e. `eᵀΛ²e` |
| Token aggregation in `J` | sum over N | mean over N |
| Λ positivity | "positive diagonal" | unconstrained |
| `F_X` tokens | "uniformly sampling" | deterministic evenly spaced |
| SpokenArabicDigits length | 83 (Table 2) | 93 for training (test maximum is 83) |
| PEMS-SF η | grid {0.1, 0.2, 0.5} | script uses 0.0 |

Causal (`cumsum`) code exists only in `PrimalAttention_RL`, so nothing needs excluding on the time-series path.

## Decisions for review

1. **Dataset.** Confirm UWaveGestureLibrary (backup Handwriting).
2. **`F_X` spread (brief 4.3).** The spread over 5 `F_X` samplings cannot be computed: the code has no sampling
   randomness, and on UWave `n = N`. Proposal: report it as not applicable.
3. **Checkpoint for spectra.** Proposal: use the fixed final epoch (`model_last.pth`) for every run, so test-set selection
   cannot differ across η, and report both final and best accuracy.
4. **`λ` comparison and rank check.** Proposal: use `N/λ²` in place of `1/λ` (brief 5.3), with near-zero `λ` flagged, and
   add the `1e-14` rank alongside `1e-10`.
