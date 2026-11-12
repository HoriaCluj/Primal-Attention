# Plan: Figure 1 spectra of Primal.+DeiT-Small/16 on ImageNet-100, varying only η

**Goal.** Measure the per-layer attention spectra of Figure 1 for Primal.+DeiT-Small/16 trained with different η, the coefficient of the KSVD term,
on the VSC wICE cluster. Everything except η follows the paper and the repository.

**Scope change.** The budget is 2,000,000 credits. One ImageNet-1K run costs about 4.55M, so the ablation uses **ImageNet-100**, the paper's
second ImageNet dataset (Table 7a), at about 0.45M per run. Figure 1 itself is on ImageNet-1K, so the new spectra can only be compared
with it qualitatively. The official DeiT-S on ImageNet-1K is still used to calibrate the measurement against the published figure.

**Sources.** Evidence and file references are in `RECON_REPORT.md`. Draft job scripts are in `jobs/`. Nothing has been run or
submitted. Every cluster fact comes from documentation and must be confirmed on wICE.

## Summary

| Decision | Choice |
|---|---|
| Dataset | **ImageNet-100**, built from ImageNet-1K with the 100-class list the repo's README points to |
| Primal.+ | the repo's ImageNet-100 launch command (`bashes/run_combo_small_rank96_layer1.sh`): `s = 96`, `rank_multi = 10`, 1 Primal layer, 300 epochs, total batch 1024; only η changes |
| η values | **{0, 0.05, 0.5}**, one seed each |
| DeiT-S baseline on ImageNet-100 | optional, decided after the three Primal.+ runs (about 0.47M credits) |
| Calibration | official DeiT-S on the ImageNet-1K val set, **on the laptop**, no credits |
| Hardware | `gpu_h100`: 4 × H100 80 GB (the paper used 4 × V100) |
| Cost | **unknown until the smoke test**: about 150,000 credits per wall-hour. The three runs fit the budget only if an H100 is at least about 6× faster per step than the paper's V100 rate (see 4.4) |

## 4.1 Experimental design

### Dataset

**ImageNet-100.** `datasets.py:66-69` loads it as an `ImageFolder` with 100 classes from `<data-path>/train` and `<data-path>/val`.

- **Classes:** the exact 100 classes come from the class list in `danielchyeh/ImageNet-100-Pytorch` (README line 22). It must be the same list the authors used,
  or the accuracy comparison with Table 7a is meaningless.
- **Built from ImageNet-1K:** the 100 class folders are copied out of ImageNet-1K. That gives about 1,300 training images per class (about 126,700
  in total, to be confirmed) and 50 val images per class (5,000).
- **Comparison targets (Table 7a):** DeiT-S 74.2%, Primal.+ 75.7% top-1 at η = 0.05.

What changes compared with the original Figure 1 goal:

- The spectra are measured on models trained on 100 classes, not 1,000. Their shape can be compared with Figure 1, but not
  value by value.
- The η conclusion applies to a 10× smaller dataset.

### Models

**Primal.+DeiT-Small/16.** Use `main_combo.py` with exactly the repo's ImageNet-100 command:

```
--model primal_small_patch16_224 --data-set IMNET-100 --low-rank 96 --rank-multi 10 --num-ksvd-layer 1 --eta <η>
```

All other settings are left at their defaults, which equal upstream DeiT's (`RECON_REPORT.md`, item 5). `s = 96` is fixed.

**DeiT-Small/16 baseline.** The official checkpoint is trained on ImageNet-1K, so it is **not** a fair baseline for models trained on
ImageNet-100. It is used only to calibrate the measurement against the published Figure 1 (stage S1). A matching baseline means training
DeiT-S on ImageNet-100 with `main.py` (`bashes/run_deit_small.sh`), about 0.47M credits. That decision is taken after the three Primal.+
runs, from the credits actually left.

### η values

| η | Why |
|---|---|
| 0 | No KSVD regularizer. Λ is then untrained; the architecture alone sets the rank cap |
| **0.05** | The paper's value; this run is also the sanity check against Table 7a's 75.7% |
| 0.5 | 10× the paper's value, to reveal any trend |

### Seeds

**One seed (0, the code default) per η.** A second seed per η costs another 0.45M, which the budget does not allow.

The consequence must be stated in any result: with one seed, η-to-η differences cannot be separated from seed-to-seed variation.
In the earlier time-series sweep, seed spread was often as large as the η effect.

### Deviations from the paper

| Deviation | Reason |
|---|---|
| ImageNet-100 instead of ImageNet-1K for the η ablation | budget: 2M credits; one ImageNet-1K run costs about 4.55M |
| H100 instead of V100 | the user's choice, to avoid the queue. The hardware changes run time and cost, not the experiment |
| 15 data-loader workers per GPU instead of 10 | wICE allows **16 CPU cores per H100** [VSC docs], and 4 processes × (1 + 15 workers) uses exactly those 64 cores. This changes loading speed only, not the model, the augmentations or the schedule |
| 4 × 256 batch layout | matches the paper's 4 GPUs and upstream DeiT-S. The repo's scripts use 2 × 512; the total is 1024 in both, so lr is unchanged |
| Official DeiT-S used only for calibration | it is trained on ImageNet-1K, not ImageNet-100 |
| Software stack | **torch 2.4.1 (CUDA 12.1) and torchvision 0.19.1 instead of the repo's torch 1.7.1 / torchvision 0.8.2**, which cannot run on H100. timm stays at the repo's 0.4.12. `check_env.py` confirmed that the repo's transforms, Mixup, optimizer, scheduler, loss scaler and KSVD loss run unchanged, and that checkpoints load across both versions. Kernel-level numerics differ slightly between versions |
| `torchrun` instead of `torch.distributed.launch` | the old launcher is deprecated in torch 2.x; both set the same environment variables that `utils.init_distributed_mode()` reads |
| η values other than 0.05 | the intended change |
| Final-epoch checkpoint for spectra | the code's `best_checkpoint.pth` is chosen on the val split, which is also the reported test split |

## 4.2 Measurement protocol

1. **Images.**
   - Primal.+ runs: 1,000 ImageNet-100 val images, the first 10 files (sorted by name) of each class.
   - Calibration (S1): 1,000 ImageNet-1K val images, the first file of each class.

   Both use the repo's eval transform (resize 256, center crop 224). Each list is saved once and reused.
2. **Matrices**, captured with forward hooks, so no repo code changes:

   | Layer | Matrix | How it's captured |
   |---|---|---|
   | Softmax layers (0-10 for Primal.+; 0-11 for DeiT) | post-softmax attention, 197 × 197, class token included | hook on the input of each timm `Attention.attn_drop`; confirm in timm 0.4.12 during S0 |
   | Primal layer 11 | (a) **full kernel** `K = Φ_q F_Xᵀ F_X Φ_kᵀ` | recompute q and k from `attn.qk` on the hooked input, then check against the returned `escore` |
   | | (b) **s-direction reconstruction**: `E Λ Rᵀ` and the code-consistent `E (Λ²/N) Rᵀ` | from the returned scores and Λ |
   | | (c) **softmax from the layer's own q and k**: `softmax(qkᵀ/8)` | from the recomputed q and k |

   (a) is used for the η comparison. All three have their spectra reported, to show which one resembles the published Figure 1(c). (a) and (b) have rank ≤ 64,
   because `p = 64 < s = 96`.
3. **SVD** in float64, per image, per head, per layer. Factors are converted to float64 **before** they are multiplied.
4. **Curves.** σ-based `c(k) = Σ_{l≤k} σ_l / Σ σ` and σ²-based `c2(k)`. Every figure states which it uses. The calibration (S1) picks the one
   that reproduces the published DeiT-S curves.
5. **Aggregation.** Per layer, average the curves over images and heads; keep the per-head std. The paper's panels show mean ± std over
   the 4 layers of each panel. Panel (c) is drawn both with and without the Primal layer in the 8-11 mean.
6. **Figures.**
   - The paper's layout: (a) layers 0-3, (b) 4-7, (c) 8-11 plus L[11], (d) heatmap of layers × index, for each η (and for DeiT-S if trained).
   - **One plot per layer (0-11)**, with one curve per η.
7. **Rank-independent summaries per layer and model.** Stable rank `Σσ²/σ₁²`; effective rank `exp(-Σ p_l log p_l)` with `p_l = σ_l/Σσ`; a
   cumulative curve normalized by the first 64 singular values, for comparing the rank-64 Primal kernel with 197-rank softmax layers; numerical rank at
   1e-10 and 1e-14.
8. **J during training.** `train_with_j_log.py` wraps `engine_primal.train_one_epoch`, with no source edits, and writes `J_log.csv`:
   per-head J, `(J̄)²`, the KKT residuals and the Λ range, on 32 fixed val images, before training and after every epoch. It is checked
   against the repo's own J and against autograd by `check_j_logger.py`. This confirms that larger η actually lowers J. Note that the code's J is one mean over batch,
   heads and tokens (`RECON_REPORT.md`, item 3).
9. **Accuracy.** Report final-epoch and best top-1 for every run (`log.txt`).

## 4.3 Sanity checks

- **S1, measurement calibration on the laptop, no credits.** Run the protocol on the official DeiT-S with the ImageNet-1K val set (6.4 GB):
  a forward pass of a 22M-parameter model, which the laptop's 2 GB GPU can do in small batches. Compare with panels (a)-(d) of the
  published figure (`pics/heatmap.jpg`).
  - Fix the σ vs σ² choice.
  - Check that the index-0 value of panel (a) is about 0.16.
  - Check that the L[11] curve reaches 1 at index 196.

  This also tests the spectra script before it runs on wICE.
- **S2, sanity training run.** Primal.+ at η = 0.05 on ImageNet-100. Accept it if top-1 is close to Table 7a's 75.7%. Then run η = 0 and 0.5.

## 4.4 Compute and logistics on wICE

### Hardware

**`gpu_h100`: one job = one node, 4 × H100 80 GB, 64 cores (16 per GPU, the documented maximum), `--mem=748800m`
(187,200 MiB per GPU), 3-day walltime** [VSC docs]. Chosen over V100 to avoid the queue. The training is single-node, so the H100 nodes'
lack of a fast interconnect does not matter.

Two consequences:
- **Software:** the repo's torch 1.7.1 cannot run on H100 (no `sm_90` support), so the environment uses torch 2.4.1 (see the deviations table).
- **Cost:** an H100 is charged about 9.6× a V100 per GPU-minute, so the H100 is only affordable if it is also much faster.

### Run time and the break-even speed

The paper gives 2,330 s per 1K steps for Primal.+ on one V100 at batch 256, which is 2.33 s per step and about 24 h per ImageNet-100 run
(`RASampler` gives 123 steps per epoch at total batch 1024, `samplers.py:35`, so 36,900 steps for 300 epochs). The H100's speed on this
code is unknown until the smoke test. It depends on the GPU and on whether 15 CPU workers can decode and augment 256 images per step fast enough.

| H100 speed vs the paper's V100 rate | s/step | Wall time per run | Credits per run | Three runs |
|---|---:|---:|---:|---:|
| 4× | 0.58 | 6.0 h | about 900,000 | about 2.7M: **over budget** |
| 6× | 0.39 | 4.0 h | about 600,000 | about 1.8M: just fits |
| 8× | 0.29 | 3.0 h | about 450,000 | about 1.35M: same cost as V100 |
| 12× | 0.19 | 2.0 h | about 300,000 | about 0.9M |

**Decision rule after the smoke test:** if the measured time is above about 0.39 s per step on one H100, the three runs don't fit in 2M credits.
Then either switch back to V100 (the earlier version of this plan) or drop to two η values.

### Credits

**Formula** [VSC docs]:

```
credits = (GPU rate × GPUs + CPU rate × cores) × minutes
```

Rates per minute: H100 569.444, Genoa core 3.472.

| Item | Credits |
|---|---:|
| One 4-H100 job (4 GPUs + 64 cores) | about **150,000 per wall-hour** |
| Smoke test (S0): 1 H100 + 16 cores, 1 h | about 37,500 |
| Spectra jobs: 1 H100 + 16 cores | about 37,500 per hour (a few hours in total) |
| One Primal.+ run | 150,000 × measured hours (see the table above) |
| *Optional:* DeiT-S on ImageNet-100 | about the same as one Primal.+ run |

**Smoke-test result (job 62653652, 1 H100):** about 0.2–0.24 s per step, so a run is expected to take about 3–3.7 h and about
450,000–555,000 credits; three runs about 1.35–1.67M. **wICE charges for time used** (936 credits for 90 s, against a 37,440 quote for the
hour requested), so `train_primal.slurm` asks for `06:00:00` as a safety limit at no extra cost. `sam-quote` shows the requested maximum,
about 900,000 per run. `sam-quote` before every new submission, and `sam-balance` after every run, are part of the order of work below.

### Walltime handling

`main_combo.py` saves `checkpoint.pth` every epoch and resumes from it with `--resume`, restoring epoch, optimizer, scheduler, EMA and scaler
(`main_combo.py:420-435`). `jobs/train_primal.slurm` works as follows:

- It resumes automatically if `checkpoint.pth` exists.
- It keeps a validated copy of the checkpoint, in case the job is killed mid-save.
- It exits immediately if the run has finished (epoch 299 in `log.txt`).

`jobs/submit_chain.sh` queues `N_JOBS` jobs per η with `--dependency=afterany`. Once `--time` covers a whole run, use `N_JOBS=1`. A spare job that starts after the run has finished exits in seconds.

### Storage

| What | Where | Size |
|---|---|---|
| ImageNet-1K (source for ImageNet-100, and val set for S1) | the laptop, `data/imagenet/` (git-ignored) | about 150 GB |
| ImageNet-100 | built on the laptop in `data/imagenet-100/` (git-ignored), then uploaded to `$VSC_SCRATCH/imagenet-100` | about 15 GB (estimate) |
| Run outputs (`checkpoint.pth`, `best_checkpoint.pth`, `log.txt`) | `$VSC_SCRATCH/primal_fig1/runs/<run>` | about 0.7 GB per run |
| Spectra (CSVs) and figures | scratch | under 1 GB |
| Code, environment, logs | `$VSC_DATA` (backed up) | a few GB |

**Copy-back:** after every finished run, `jobs/train_primal.slurm` copies `log.txt` and the final checkpoints to `$VSC_DATA/primal_fig1/`. Scratch is not
backed up and may be cleaned.

### Job scripts (drafts in `jobs/`, not submitted)

| File | Purpose |
|---|---|
| `common.sh` | paths for all scripts; edit the placeholders once |
| `env_setup.sh` | conda env with python 3.8, torch 2.4.1 (CUDA 12.1), torchvision 0.19.1, timm 0.4.12, numpy < 1.24; runs `check_env.py` at the end |
| `make_smoke_subset.sh` | symlinked 100-class × 130-image subset of ImageNet-100 for the smoke test (no copying) |
| `smoke_h100.slurm` | S0: 1 H100 on `gpu_h100`, 1 h (there is no H100 debug partition); 2 epochs on the subset, then a resume test; reports s/step and peak memory |
| `train_primal.slurm` | S2/S3: 4 × H100 DDP on `gpu_h100` via `torchrun`, one η per chain, auto-resume |
| `train_deit.slurm` | optional DeiT-S baseline on ImageNet-100, same settings |
| `submit_chain.sh` | queues the dependency chains |
| `spectra.slurm` | spectra for all finished runs |
| `deit_official_spectra.slurm` | S1 on wICE instead of the laptop, if preferred |

The spectra scripts call `figure1_spectra.py` (in this folder), which is **still to be written** and tested locally during S1. All scripts carry
`--account=<CREDIT_ACCOUNT>` and path placeholders.

## Order of work

1. **Data access.** Get ImageNet-1K (image-net.org account, or a copy your group already holds) and the ImageNet-100 class list.
2. **Build ImageNet-100 locally** into `data/imagenet-100/{train,val}`, with val images sorted into class folders.
3. **S1 on the laptop:** write and test `figure1_spectra.py` on the official DeiT-S and the ImageNet-1K val set.
4. **wICE setup:** clone the repo, run `env_setup.sh`, upload ImageNet-100 to `$VSC_SCRATCH`, run `make_smoke_subset.sh`.
5. **S0 smoke test**, then apply the decision rule in 4.4 with the measured speed and set `--time`.
6. **S2:** η = 0.05. Check accuracy against 75.7%.
7. **S3:** η = 0 and 0.5.
8. Spectra, figures and report. Decide on the optional DeiT-S baseline from the credits left.

## 4.5 Outputs

On `$VSC_SCRATCH/primal_fig1/`, copied to `$VSC_DATA/primal_fig1/` after each stage:

- **`runs/<model>_eta<η>_seed<s>/`:** `checkpoint.pth` (final), `best_checkpoint.pth`, `log.txt` (per-epoch train and test stats), `J_log.csv`.
- **`spectra/`:** `images.txt` (the evaluation files); `spectra_per_head.csv` (model, η, layer, head, image, σ and σ² metrics);
  `curves.csv` (mean and std curves per model, layer and candidate); `summary.csv` (stable rank, effective rank, numerical rank per layer).
- **`figures/`:** `fig1_<model>.png` (4-panel layout), `per_layer/layer_<ℓ>.png` (12 files, one curve per η),
  `primal_candidates.png` ((a), (b) and (c) against the published L[11]), `heatmap_<model>.png`, `ranks_vs_eta.png`, `J_over_training.png`,
  `accuracy_vs_eta.png`.
- **`REPORT.md`:** calibration result (S1), sanity run (S2), the η comparison with its single-seed caveat, and the deviations table.

## 4.6 Open questions for the user and supervisor

1. **Data:** does the group already hold a licensed ImageNet-1K copy? Is the danielchyeh class list the one the authors used for Table 7a?
2. **Scope:** is ImageNet-100 acceptable for the supervisor, given that Figure 1 is on ImageNet-1K?
3. **Baseline:** should the remaining credits go to a DeiT-S baseline on ImageNet-100 (about 0.47M), or be kept as a margin?
4. **Seeds:** is one seed per η acceptable, given that the result cannot separate η from seed noise?
5. **Primal-layer matrix:** if S1 does not show which matrix the paper plotted, should the authors be asked? The full kernel (a) is used for the
   η comparison either way.
