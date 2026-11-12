# Reconnaissance: Figure 1 of Primal-Attention (DeiT-Small/16 on ImageNet)

Answers to Section 3 of `figure1_eta_ablation_plan_brief.md`. Read-only: nothing was trained, submitted or modified.
Paths are relative to `PrimalAttention_CV/` unless stated.

Each answer says where it comes from: **[paper]** for the paper, **[code]** for this repository, **[upstream]** for
`facebookresearch/deit` (`main.py` and `README_deit.md`, read online), **[VSC docs]** for docs.vscentrum.be, and
**[proposed]** for my recommendation. Not tested on wICE: anything about the cluster comes from documentation only.

## Summary

- The ImageNet code is a fork of the DeiT repo. `main.py` trains DeiT-S. `main_combo.py` trains Primal.+DeiT-S. The two differ only in how
  the model is built, the engine they import, and the Primal flags.
- The launch flags in the code are `--low-rank 96 --rank-multi 10 --num-ksvd-layer 1 --eta 0.05`, so **s = 96** and only layer 11 is Primal.
- Because `n = min(197, 960) = 197`, `F_X` uses **all 197 tokens**, with no sampling and no randomness.
- **s = 96 > p = 64**, so every matrix built from the Primal layer has rank ≤ 64.
- The repo contains **no checkpoints, no data, no ImageNet-100 class list and no spectrum code**. `pics/heatmap.jpg` is Figure 1 itself.
- In the published Figure 1(c), the Primal.+ L[11] curve reaches about 1 within about 5 singular values. That fits a rank-limited Primal-layer matrix,
  not a softmax matrix.
- The DeiT-S ImageNet-1K accuracy in Table 7a (79.8%) equals the official DeiT-S result. This supports using the official checkpoint as the baseline.

## 3.1 Code for the image-classification experiments

### 1. Location [code]

| Role | DeiT-Small/16 | Primal.+DeiT-Small/16 |
|---|---|---|
| Entry script | `main.py` | `main_combo.py` |
| Training engine | `engine.py` | `engine_primal.py` |
| Model | timm `VisionTransformer` via `models.py:77-89` (`deit_small_patch16_224`) | `models_combo.py:75-82` (`primal_small_patch16_224`) → `ComboVisionTransformer` |
| Primal layer | n/a | `models_primal.py:29-85` (`PrimalAttention`), `:88-127` (`Block`) |
| Launch | `bashes/run_deit_small.sh`, README lines 57-61 | `bashes/run_combo_small_rank96_layer1.sh`, README lines 48-54 |

The environment is pinned to torch 1.7.1, torchvision 0.8.2, timm 0.4.12 and cudatoolkit 11.0 (README lines 12-18, `requirements.txt`). A
`diff` of `main.py` and `main_combo.py` shows the only differences are the engine import (line 21), the model module import (line 26), the
four Primal flags (`main_combo.py:34-37`), the default `--model` (`:46`), and model construction (`:266-295`). **The training protocol is
the same for both models.**

### 2. Primal-Attention in DeiT [code]

- **Which layer.** `ComboVisionTransformer.__init__` replaces `self.blocks[-(i+1)]` for `i < num_ksvd_layer` (`models_combo.py:23-34`).
  With `--num-ksvd-layer 1`, only **block 11** changes. `forward_features` runs blocks 0-10 as softmax blocks (`:48-54`). This matches the
  paper's "11+[1]" (Table 13).
- **Heads and dimensions.** 6 heads, `head_dim = 384 // 6 = 64` (`models_primal.py:34-35`), so **p = 64**. q and k come from one `nn.Linear(384, 768)`
  with bias (`:37`, `qk_bias=kwargs['qkv_bias']` = True at `models_combo.py:31`). There is no value projection.
- **Feature map.** `F.normalize(x, p=2, dim=-1)` on q and k (`:64-66`, `:74-75`), which is cosine, as in Remark 4.1.
- **Data-dependent weights.** `we`, `wr` have shape `(6, min(197, s·rank_multi), s)` and orthogonal init (`:46-47`). `gen_weights` (`:51-62`) splits the
  block's input into 6 × 64-dimensional slices.
  - When `embed_len > s·rank_multi`, it takes `linspace` indices (evenly spaced, deterministic).
  - Otherwise it uses every token.
  - Here `197 < 96·10 = 960`, so the **else branch** runs: `F_X` is all 197 tokens of the head's slice, recomputed on every forward pass.
  - The input is the **LayerNorm output**, `self.attn(self.norm1(x))` (`:124`).
- **Output map.** `concate_weight = nn.Linear(2s, head_dim)` (`:49`), applied to `[E R]` per head, then reshaped and passed through `proj` (`:78-83`).
  `attn_drop` is created but never used in `forward`.
- **Returned values.** `[escore, rscore, self.we, self.wr]` and `self.Lambda` (`:85`). Unlike the time-series code, **Φ_q and Φ_k are not
  returned**, so an analysis must recompute q and k from `self.qk` inside a hook. It can check the result against the returned `escore`.

`models_primal.py` also defines an all-Primal `PrimalFormer` (`:130-318`). `main_combo.py` does not use it.

### 3. The regularizer J [code]

`engine_primal.py:40-52`, inside `torch.cuda.amp.autocast()`:

```python
loss_escore = torch.mean((torch.einsum('...nd,...ds->...ns', score_list[i][0], Lambda_diag...)).norm(dim=-1, p=2)**2)/2
loss_rscore = (same for rscore)
loss_trace  = torch.einsum('...ps,...pd->...sd', we, wr).mean(dim=0).trace()
loss_ksvd   = loss_ksvd + (loss_escore + loss_rscore - loss_trace) ** 2
loss = loss_ce + args.eta * loss_ksvd
```

- **Squared:** `J²` enters the loss. Λ enters as `‖EΛ‖² = eᵀΛ²e`, not the paper's `eᵀΛe`.
- **Averaging.** The first two terms use a single `torch.mean` over **batch × heads × tokens**. The trace is a head mean. So J is one scalar per
  batch, squared once. This differs from the time-series code, where J was computed and squared per sample.
- **Layers.** It sums over Primal layers (one here) and divides by their number (`:51`).
- **Precision.** It is computed under autocast, so the einsums may run in float16.

### 4. Λ [code]

`nn.Parameter(nn.init.uniform_(torch.Tensor(6, s)))`, so `U[0,1)` (`models_primal.py:48`). **No positivity constraint.** Its sign does
not matter because it enters squared. At η = 0, Λ gets no gradient and stays at its initial values. The paper calls Λ a "positive diagonal
matrix" (Section 3).

### 5. Training configuration [code], [upstream], [paper]

Defaults are identical in `main.py` and `main_combo.py` (`main_combo.py` lines given).

| Setting | This repo | Upstream DeiT `main.py` | Same? |
|---|---|---|---|
| Epochs | 300 (`:41`) | 300 | yes |
| Optimizer | AdamW, eps 1e-8 (`:62-64`) | same | yes |
| lr | 5e-4 × total batch / 512 (`:77`, `:373-375`) | same formula | yes |
| Schedule | cosine, warmup 5 epochs from 1e-6, min 1e-5, cooldown 10 (`:75-95`) | same | yes |
| Weight decay | 0.05 (`:72`) | 0.05 | yes |
| Gradient clipping | none (`:68`) | none | yes |
| RandAugment | `rand-m9-mstd0.5-inc1`, color jitter 0.3, bicubic (`:102-109`) | same | yes |
| Mixup / CutMix | 0.8 / 1.0, prob 1.0, switch 0.5, batch mode (`:134-145`) | same | yes |
| Label smoothing | 0.1 (`:107`) | 0.1 | yes |
| Random erasing | 0.25, pixel, count 1 (`:124-130`) | same | yes |
| Repeated augmentation | on (`:111-113`), `RASampler` with 3 repeats (`samplers.py:16`) | on | yes |
| Stochastic depth | 0.1 (`:52`) | 0.1 | yes |
| Dropout | 0.0 (`:50`) | 0.0 | yes |
| Model EMA | on, decay 0.99996 (`:55-58`); **evaluation uses the non-EMA model** (`:471`) | on, 0.99996 | yes |
| Distillation | `none` (`:151`) | `none` | yes |
| Mixed precision | `torch.cuda.amp.autocast` + `NativeScaler` (`engine_primal.py:40`, `:377`) | `NativeScaler` in `main.py`; autocast in upstream `engine.py` not checked | likely |
| Seed | 0 + rank (`:172`, `:205-207`); `random` not seeded (`:208`) | 0 | yes |
| Eval transform | resize `224/0.875 = 256`, center crop 224 (`datasets.py:104-109`) | 0.875 | yes |
| Val batch | 1.5 × batch (`:251`) | same | yes |
| Per-process batch default | 64 (`:40`) | 64 | yes |

No difference from upstream DeiT defaults was found in the settings checked. The launch commands differ, but the total batch is the same:

| Source | GPUs × per-GPU batch | Total |
|---|---|---|
| Upstream DeiT-S README | 4 × 256 | 1024 |
| This repo's README (both models, both datasets) | **2 A100 × 512** | 1024 |
| Paper (Section 5) | 4 × V100 32 GB; Table 7a timing at batch 256 on one V100 | 1024 implied |

So lr = 5e-4 × 1024/512 = **1e-3** in all three.

### 6. s and η in the scripts [code]

`--low-rank 96 --rank-multi 10 --num-ksvd-layer 1 --eta 0.05`, the same for ImageNet-100 (`bashes/run_combo_small_rank96_layer1.sh`)
and ImageNet-1K (README line 53). The argparse defaults are different (s = 20, η = 0.1; `main_combo.py:34-37`), so the flags must be passed explicitly.

## 3.2 Availability of the two models

### 7. Trained checkpoints [code]

**None in the repo or linked from it.** The CV README has no download links, and a repo-wide search of `*.md` found links only for LRA data. Training
writes `checkpoint.pth` (overwritten every epoch) and `best_checkpoint.pth` (best on the val split), each holding the model, the EMA
model, the optimizer, the scheduler and the scaler (`main_combo.py:457-487`).

### 8. Official pretrained DeiT-S [code], [paper]

The code can use it in three ways:

- `models.py:83-88` loads the official weights when `pretrained=True`:
  `https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth`.
- `main.py` always calls `create_model(..., pretrained=False)`, but it accepts an https URL for `--resume` (`main.py:392`) and
  `--finetune` (`:270`).
- So `main.py --eval --resume <URL>` evaluates the official model with no code change.

Evidence from the paper:

- The caption says "pre-trained DeiT-Small/16".
- Section 5 says both models were trained from scratch.
- Table 7a's DeiT-S ImageNet-1K accuracy, 79.8, is exactly the official DeiT-S number. This suggests the baseline number, and possibly the
  Figure 1 model, is the official one.

This is not conclusive.

## 3.3 Availability of the data

### 9. Data in the repo [code]

**No data.** `build_dataset` (`datasets.py:56-79`) uses `torchvision.datasets.ImageFolder` on `<data-path>/train` and `<data-path>/val`. Both
need **one subfolder per class**, including `val/`, which is not ImageNet's raw flat layout. `IMNET` hard-codes 1000 classes and `IMNET-100`
100 classes. Expected layout (README lines 26-34): `/path/to/data/imagenet/{train,val}` and `imagenet-100/{train,val}`.

### 10. ImageNet-100 [code]

There is no class list in the repo. The README (line 22) points to `danielchyeh/ImageNet-100-Pytorch` to generate it; I have not inspected
that repo. `IMNET-100` simply loads whatever 100 class folders exist, so the exact 100 classes must come from that external list.

### 11. Shared ImageNet on the cluster [VSC docs]

- **No shared dataset location is documented** in the VSC pages read (wICE hardware, quick start, Slurm specifics, credits, accounting,
  advanced guide). I cannot check the cluster file system from here.
- Node-local SSDs: 960 GB on A100 and H100 nodes, and 200 GB on V100 and P100 nodes.

Questions for the HPC team or supervisor:

1. Is there a shared ImageNet-1K (and ImageNet-100) copy on wICE, and what is its path and layout?
2. If there isn't, may ImageNet-1K be stored on `VSC_SCRATCH` (about 150 GB of the 475 GiB quota), and does the group's ImageNet licence cover it?
3. Is per-job staging of the dataset to node-local SSD recommended, and what is the variable or path for that disk?
4. What is the cleanup policy on `VSC_SCRATCH`?

## 3.4 Settings the paper does not specify

No Figure 1 or spectrum code exists in the repo: a search for `svd`, `svdvals`, `cumsum` and "explained variance" found nothing in
`PrimalAttention_CV`. **Items 12-19 are therefore not settled by code.** The published figure is `pics/heatmap.jpg` (repo root, labelled
"Spectrum Analysis" in the root README). The observations below come from reading it.

| # | Question | Code | Paper or figure | Proposed |
|---|---|---|---|---|
| 12 | s for Primal.+ | **96** in every launch command | {32, 64, 96} | **96** |
| 13 | Matrix decomposed for layer 11 | not found | Figure 1(c): the "Primal.+, L[11]" curve reaches about 1 within about 5 indices | compute (a) full `K = Φ_q F_Xᵀ F_X Φ_kᵀ`, (b) `E Λ Rᵀ` and `E (Λ²/N) Rᵀ`, (c) `softmax(qkᵀ/√64)` from the layer's own q and k; use (a) for the η comparison |
| 14 | σ or σ², normalization | not found | y-axis "Normalized cumulative singular value"; caption "cumulative explained variance" | compute both; pick whichever reproduces the published DeiT-S curves (stage S1) |
| 15 | Images | not found | ImageNet-1K | 1,000 val images, the first file (sorted) of each class, eval transform |
| 16 | Aggregation | not found | "mean and standard deviation of the chosen layers" | per layer, average over images and heads; panels show mean ± std over the 4 layers; also report per-head std |
| 17 | Softmax layers | not found | x-axis 0-196 = 197 values, so the class token is included | post-softmax matrix, all 197 tokens |
| 18 | "Pre-trained" DeiT-S | official URL supported (`models.py:85`) | 79.8% equals the official number | official checkpoint |
| 19 | Seeds and checkpoint | one seed (0) per launch; `best_checkpoint.pth` is selected on val, which is also the reported test split | not stated | final epoch (`checkpoint.pth`), non-EMA weights; report best and final accuracy |

**What the figure suggests about item 13.**

- With s = 96 > p = 64, any matrix from the Primal layer has rank ≤ 64, so its cumulative curve reaches 1 by index 63 at the latest. Candidates (a) and (b) satisfy this.
- The full kernel also contains `F_X` twice. In the earlier time-series experiment that made `c(k)` reach about 1 within about 5 indices, which is
  what the published L[11] curve shows.
- Candidate (c) would normally look like a softmax layer: the DeiT L[11] curve rises until index 196. So it is unlikely to be what the paper plotted.
- Either way, **the sharp Primal L[11] curve is largely explained by the layer's rank cap and structure**, not necessarily by η.

**Item 16, open detail.** It is unclear whether panel (c)'s solid "Primal.+" curve (mean over layers 8-11) includes the Primal layer 11.
Including a curve near 1 would raise the mean noticeably, but the plotted Primal.+ and DeiT means look almost identical. The plan computes both
versions.

## Paper vs code differences

| Item | Paper | Code |
|---|---|---|
| Hardware | 4 × V100 32 GB | README: 2 × A100, batch 512 each |
| Λ in J | `eᵀΛe`, Λ positive | `‖EΛ‖²`, Λ unconstrained |
| J aggregation | `J_l` = mean over heads | one mean over batch, heads and tokens |
| `F_X` | "uniformly sampling n points" | `n = N = 197`, every token, deterministic |
| DeiT-S baseline | "pre-trained" (caption) vs "from scratch" (Section 5) | both paths possible |

## Facts checked locally

- **Parameter counts** (analytic, from the layer definitions): DeiT-S 22,050,664; Primal.+DeiT-S 22,142,696 (+92,032). In the Primal
  layer, `we` and `wr` have 113,472 parameters each, Λ 576, `concate_weight` 12,352, and `qk` replaces the 3-way `qkv`. The official DeiT-S is
  22M parameters, consistent with this.
- **Steps per epoch.** `RASampler` uses `floor(len // 256 * 256 / replicas)` samples per process (`samplers.py:35`). ImageNet-1K
  (1,281,167 images) at total batch 1024 gives 1,251 steps/epoch and **375,300 steps** for 300 epochs. ImageNet-100 (about 126,700 images,
  not verified) gives about 36,900.

## Correction

During this reconnaissance, one of my shell commands accidentally ran `pip download` for two unrelated PyPI packages, `not` and `allowed`. It
was meant to be a comment. Pip failed during metadata preparation; no files were written to the repo or `/tmp`, and nothing was installed.
