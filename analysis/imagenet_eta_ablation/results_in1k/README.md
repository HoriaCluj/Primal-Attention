# ImageNet-1K: DeiT-S vs Primal.+DeiT-S at η = 0.05 and η = 0.5

Three full 300-epoch runs on ImageNet-1K (1000 classes, 1,281,167 train / 50,000 val images), one seed each (seed 0).
Run on VSC wICE, 4× H100 80 GB, 24–26 Nov 2026, credit account `lp_robustness`. All three ended with exit code 0, and no
epoch has a NaN.

## Results

| Run | Slurm job | Wall time | Final top-1 | Best top-1 (epoch, 0-based) | Final top-5 | Paper (Table 7a) |
|---|---|---|---|---|---|---|
| DeiT-Small/16 | 62869857 | 13:51 h | **79.92** | 79.95 (293) | 94.94 | 79.8 |
| Primal.+DeiT-S, η = 0.05 | 62869858 | 15:55 h | **79.66** | 79.73 (296) | 94.78 | 79.8 (η = 0.05) |
| Primal.+DeiT-S, η = 0.5 | 62869861 | 14:17 h | **79.65** | 79.84 (293) | 94.86 | – |

Top-1 (%) during training:

| Epoch | 1 | 5 | 10 | 20 | 50 | 100 | 150 | 200 | 250 | 300 |
|---|---|---|---|---|---|---|---|---|---|---|
| DeiT-S | 0.34 | 7.83 | 22.97 | 44.78 | 61.92 | 68.72 | 71.95 | 75.80 | 78.64 | 79.92 |
| η = 0.05 | 0.33 | 9.16 | 23.50 | 43.23 | 61.15 | 67.98 | 71.70 | 75.25 | 78.09 | 79.66 |
| η = 0.5 | 0.30 | 9.19 | 23.96 | 43.07 | 60.97 | 67.82 | 71.85 | 75.46 | 78.30 | 79.65 |

- All three runs land within 0.15 points of the paper's 79.8 %. Unlike the paper (where Primal.+ equals DeiT-S), both
  Primal.+ runs end 0.27 points below DeiT-S. With one seed this is within typical run-to-run noise.
- **η makes no difference on ImageNet-1K** (79.66 vs 79.65 final). This differs from ImageNet-100, where accuracy rose
  with η (74.0 → 75.1, see [../REPORT.md](../REPORT.md)). There is no full η = 0 run on ImageNet-1K, so we can't say
  whether the regularizer helps at all here.
- Primal is ahead of DeiT for the first ~10 epochs (e.g. 14.15 vs 12.08 at epoch 7 for η = 0.5), then falls behind from
  about epoch 20 and stays 0.2–0.8 points behind.

## What the regularizer does: Λ collapses completely

Per-epoch diagnostics of the Primal layer (the last layer, 6 heads × s = 96 directions) on 32 fixed validation images,
from `J_log.csv` and `W_log.csv`. "Λ < 1e-3" counts all 576 values. Epochs here are 0-based, as in the CSVs and
`log.txt`; the top-1 table above counts from 1.

| Epoch | max \|λ\| η = 0.05 | max \|λ\| η = 0.5 | Λ < 1e-3, η = 0.05 | Λ < 1e-3, η = 0.5 | KKT residual (both) |
|---|---|---|---|---|---|
| init | 1.0 | 1.0 | 2 | 2 | 2–3.4 |
| 10 | 0.43 | 0.47 | 160 | 86 | 1.3–1.5 |
| 30 | 0.11 | 0.11 | 332 | 224 | 1.000 |
| 40 | 0.055 | 0.057 | 407 | 307 | 1.000 |
| 50 | 0.026 | 0.029 | 511 | 452 | 1.000 |
| 80 | 2.4e-6 | 3.4e-7 | 576 | 576 | 1.000 |
| 299 | 2.0e-8 | 2.9e-9 | 576 | 576 | 1.000 |

1. **Every Λ value goes to zero by epoch 80**, in both runs. On ImageNet-100 only 29–59 of 96 per head collapsed.
2. **The KKT residual is exactly 1 from about epoch 20–30.** With Λ = 0 the gradients reduce to `G_e = −W_r` and
   `G_r = −W_e`, which gives residual 1 by definition. The stationarity conditions are not met at all.
3. **J̄² → 0 only through cancellation between heads.** With Λ = 0, each head's `J = −Tr(W_eᵀW_r)`. At epoch 299
   (η = 0.05) these are −0.11, +0.15, +0.10, −0.04, −0.16 and +0.06. The loss squares the head mean
   (`PrimalAttention_CV/engine_primal.py:44-52`), which is about −1e-4, so J̄² ≈ 2e-8. The KSVD term in the training loss
   ends at about 3e-9 (η = 0.05) and 2e-9 (η = 0.5).
4. **η barely changes the Λ trajectory.** max |λ| is practically the same at epochs 30/40/50 for a 10× larger η, so
   something other than η drives the collapse.
5. W_e and W_r: Frobenius norm ends at 4.0–4.1 (η = 0.05) and 3.2 (η = 0.5); effective rank 76.6/77.5 and 74.7/75.2 of
   96. Both runs dip to an effective rank of about 54 around epoch 6, then recover.

**Likely cause (a hypothesis, not yet tested): weight decay on Λ.**
- Λ is not used in the forward pass (`models_primal.py:85`: the output uses only `escore` and `rscore`), so the
  cross-entropy gives it no gradient. Only `η · J²` and weight decay act on it.
- timm 0.4.12 (`add_weight_decay`) exempts only 1-D parameters and biases. Λ has shape 6 × 96, so it gets AdamW weight
  decay 0.05 at lr 1e-3 (5e-4 × 1024 / 512).
- Decay alone shrinks Λ by about `exp(−lr · wd · 1251 steps)` ≈ 0.55 per 10 epochs. The observed factor between epochs
  30–40 and 40–50 is 0.50 and 0.47.
- ImageNet-1K has 10× more steps per epoch than ImageNet-100. That fits a full collapse here and a partial one there.
- Once Λ is small, the gradient of J² with respect to Λ is also proportional to Λ, so nothing pulls it back. The code
  also doesn't keep Λ positive (the paper requires a positive diagonal Λ).

A cheap test: rerun on ImageNet-100 with Λ excluded from weight decay, and optionally with J² penalised per head instead of
on the head mean. Both can be done from `train_with_j_log.py` without editing the authors' code.

Not done yet: attention spectra for these checkpoints (the paper's Figure 1 is on ImageNet-1K) and figures for these runs.

## Files

Each run folder has:
- `log.txt`: one JSON line per epoch from the training code (`test_acc1`, `test_acc5`, `test_loss`, `train_lr`;
  DeiT: `train_loss`; Primal: `train_loss_total`, `train_loss_ce`, `train_loss_ksvd`).
- Primal only:
  - `J_log.csv`: per epoch and head: J, |J|, J̄², KKT residuals, Λ range and the count of |λ| < 1e-3.
  - `W_log.csv`: per epoch and head: ‖W_e‖, ‖W_r‖, Tr(W_eᵀW_r), cosine, effective ranks.
  - `W_dirs.csv.gz` (gzipped, 18 MB unpacked): per epoch, head and direction: |λ_l|, column norms of W_e and W_r, and the
    norms of the `concate_weight` columns that read e_l and r_l.
  - Column definitions are in the docstring of [../train_with_j_log.py](../train_with_j_log.py). Epoch −1 is before
    training.

Checkpoints (`checkpoint.pth` = epoch 299 and `best_checkpoint.pth`, about 350 MB each) are not in git. They are on wICE in
`/data/leuven/398/vsc39861/primal_fig1/runs_in1k/<run>/`. Ask Waseem for a copy.

## Setup and how to reproduce

- Job: [../jobs/train_in1k.slurm](../jobs/train_in1k.slurm), with 1 node, 4 H100, 64 cores, 20 h limit. It resumes from
  the last checkpoint when resubmitted and copies the logs to `$VSC_DATA` after every job. From `jobs/`:
  ```
  sbatch --account=<credit account> --export=ALL,MODEL=deit train_in1k.slurm
  sbatch --account=<credit account> --export=ALL,MODEL=primal,ETA=0.05 train_in1k.slurm
  sbatch --account=<credit account> --export=ALL,MODEL=primal,ETA=0.5 train_in1k.slurm
  ```
  Pass `--account` on the command line: an `SBATCH_ACCOUNT` set in `~/.bashrc` overrides `#SBATCH --account`.
- Flags:
  - DeiT: `main.py --model deit_small_patch16_224`.
  - Primal: `main_combo.py --model primal_small_patch16_224 --low-rank 96 --rank-multi 10 --num-ksvd-layer 1 --eta <η>`,
    run through `train_with_j_log.py`. That means Primal attention in the last layer only (Primal.+), s = 96.
  - Both: `--batch-size 256` per GPU (1024 total), `--data-set IMNET --seed 0 --num_workers 15`. Everything else is the
    code's default: 300 epochs, AdamW, lr 5e-4 × 1024/512, weight decay 0.05, 5 warmup epochs, DeiT augmentation.
- Data: ImageNet-1K from Hugging Face `ILSVRC/imagenet-1k`, unpacked with
  [../convert_imagenet1k.py](../convert_imagenet1k.py) ([../jobs/get_imagenet1k.slurm](../jobs/get_imagenet1k.slurm)).
  Label order is checked against the sorted synset ids.
- Cost: about 150k credits per wall hour on gpu_h100, so 2.1–2.4M per run and about 6.6M for all three. The 2-epoch smoke
  tests ([../jobs/smoke_in1k.slurm](../jobs/smoke_in1k.slurm)) measured ~159 s/epoch. The full runs averaged 166–191 s.
- Status during a run: [../jobs/status_in1k.sh](../jobs/status_in1k.sh) on a wICE login node.
- Differences from the paper: H100 instead of V100, a newer PyTorch, one seed, and the final epoch reported next to the
  best epoch.
