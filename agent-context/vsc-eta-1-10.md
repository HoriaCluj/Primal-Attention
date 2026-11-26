# Agent context: Waseem's (vsc39861) wICE runs, η = 1 and η = 10

For Claude, not for humans. State as of 2026-11-16. The authoritative docs are Horia's handoff
(`resources/HANDOFF.md` in this workspace), plus `RUNBOOK.md` and `REPORT.md` in
`~/Primal-Attention/analysis/imagenet_eta_ablation/` (the code repo; read them there, on branch `horia`). This file records only what
differs for Waseem's account and what is still pending. Before acting on any "status" line, verify it: ask the user
to run the check command and paste the output.

## Roles and access
- The user runs every command. Claude cannot reach wICE: `ssh` from the Mac gives "No route to host" without the
  firewall step, and login needs a browser MFA approval. Claude writes the commands, and the user pastes the output back.
- The user has run commands on the wrong machine before (wICE commands on the Mac). Always label each command
  **[Mac]** or **[wICE]**. The prompts are `waseemirshad@Waseems-MacBook-Air` and `vsc39861@tier2-p-login-N`.
- Never handle the `BEARER=` token from https://firewall.vscentrum.be. The user runs the page's `curl` lines themselves.
- Run all jobs on `gpu_h100`, spectra included. This is the user's choice; never suggest `gpu_v100`.

## Account and paths
| Item | Value |
|---|---|
| VSC ID / credit account | `vsc39861` / `intro_vsc39861` (KU Leuven). 2,000,000 at the start; 1,998,908 after the smoke test |
| `$VSC_DATA` | `/data/leuven/398/vsc39861`: code in `PrimalAttention/`, `miniconda3/`, `envs/primal-cv` |
| `$VSC_SCRATCH` | `/scratch/leuven/398/vsc39861`: `imagenet-100/`, `imagenet_smoke/`, `primal_fig1/{runs,spectra,plots}`, `tmp/` |
| Jobs folder (submit from here) | `$VSC_DATA/PrimalAttention/analysis/imagenet_eta_ablation/jobs` |
| Local clone | `~/Primal-Attention`, branch `horia` (tracks `origin/horia`, remote github.com/HoriaCluj/Primal-Attention) |
| Account override | `~/.bashrc` on wICE has `export SBATCH_ACCOUNT=intro_vsc39861`, which overrides `#SBATCH --account=intro_vsc39859` (Horia's) in every `jobs/*.slurm`. Do not edit the scripts for this |
| Temp dir | `~/.bashrc` on wICE should have `export TMPDIR=$VSC_SCRATCH/tmp`. The Miniconda installer failed with `[PYI-...] Failed to create parent directory structure` until TMPDIR was set. Confirm it is in `.bashrc` with `env \| grep TMPDIR` |

## Login [Mac]
1. Open https://firewall.vscentrum.be. The user pastes the `BEARER=`/`curl -4` lines into a Mac terminal.
2. `ssh -A vsc39861@login.hpc.kuleuven.be` and approve the MFA prompt. The certificate lasts 16 h.

## Code sync [Mac]
`--exclude primal_attention/` is new: `~/Primal-Attention/primal_attention/` is a 3.5 GB untracked leftover from the
`waseem` branch, and the earlier rsync without this exclude probably copied it to `$VSC_DATA` (75 GiB quota). If space
matters, check with `du -sh $VSC_DATA/PrimalAttention/primal_attention` on wICE. Do not delete the local folder; it is
the user's data.
```
cd ~/Primal-Attention && rsync -av --exclude primal_attention/ --exclude data/ --exclude '.venv/' --exclude 'analysis/timeseries_eta_ablation/runs/' --exclude 'PrimalAttention_TimeSeries/*/Multivariate_ts/' --exclude 'PrimalAttention_TimeSeries/*/experiments/' ./ vsc39861@login.hpc.kuleuven.be:/data/leuven/398/vsc39861/PrimalAttention/
```

## Setup status
- Done (confirmed by pasted output): login; `sam-balance`; Miniconda (after the TMPDIR fix); `make_smoke_subset.sh`;
  `env_setup.sh` (`all checks passed`).
- Done: smoke test on H100. 0.2212 s/step, 3 epochs, 24 `J_log.csv` rows, resume OK, peak memory 16,297 MiB, cost 1,092 credits.
  - `H100 support (sm_90): False` on the login node is expected, because `torch.cuda.get_arch_list()` returns `[]` without a GPU.
  - RUNBOOK says it should be True; that line in RUNBOOK is wrong.
- **Not confirmed:** the full dataset. Expect 126,689 train and 5,000 val files:
  `find $VSC_SCRATCH/imagenet-100/train -type f | wc -l; find $VSC_SCRATCH/imagenet-100/val -type f | wc -l`.
  The smoke subset was built from it, so the dataset exists, but completeness was never confirmed. It came from
  `hf download Yunzhe/ImageNet-100 --repo-type dataset` plus `convert_imagenet100.py` on wICE.

## The experiment
Same as Horia's §2b runs, changing only η: Primal.+DeiT-S/16, block 11 Primal, `--low-rank 96 --rank-multi 10
--num-ksvd-layer 1`, seed 0, 300 epochs, 4× H100 + 64 cores. Expected run time is about 1 h 45 min per run, and cost
about 260k credits per run (about 150k/h; `--time=06:00:00` is only a cap). Run dirs: `$VSC_SCRATCH/primal_fig1/runs/primal_eta{1,10}_seed0`.

**Status 2026-11-17:**
- **Jobs:** 62723079 (η = 1) COMPLETED, 300 epochs, 1 h 45 min. 62723080 (η = 10) FAILED at epoch 132 with `Loss is nan`
  (`engine_primal.py:57-59`).
- **Credits:** about 377k for both runs; balance 1,621,559.
- **Logs on the Mac:** `~/Primal-Attention/analysis/imagenet_eta_ablation/results/raw_logs/primal_eta{1,10}_seed0/`, Slurm
  `.out` files in `results/slurm_out_vsc39861/`. The summary script is in scratchpad (not kept).
- **Results:**
  - η = 1: final top-1 29.46%, best 29.92%.
  - η = 10: best 10.74% at epoch 4, then about 1–7%, then NaN.
  - Both tracked η = 0.5 exactly until the end of LR warmup. Then the KSVD term spiked (epoch-mean J² = 41 at η = 1,
    epoch 6; J² = 31 at η = 10, epoch 5) and accuracy fell to chance (1%).
  - The η = 1 run recovered slowly; the η = 10 run never did.
- **Training settings:** `train_loss_ksvd` in `log.txt` is J² without η (loss = CE + η·J²). There is no gradient
  clipping (`--clip-grad` defaults to None), and training uses AMP.
- **Spectra:** job 62729021 ran on H100, COMPLETED in 34 min, about 21k credits.
  - It produced `primal_eta1_seed0.npz` only; η = 10 was skipped because the run is unfinished.
  - Copied to the Mac: `~/Primal-Attention/analysis/imagenet_eta_ablation/results_vsc39861/{spectra,plots}`. The plots there show
    η = 1 alone; the six-model figure still needs Horia's 4 `.npz` files.
  - Finding: η = 1 is rank-collapsed in every layer. Softmax layers 0–9 have effective rank 2.8–5.9, against 10–66 for
    DeiT and η ≤ 0.5. Layer 10 is 14.2. K_full is 2.76, against 3.9 / 4.5 / 5.0 for η = 0 / 0.05 / 0.5.
  - The collapse is network-wide, so attribute it to the training collapse at epoch 6, not to η's effect on the Primal layer.
- **Combined figures: DONE 2026-11-17.**
  - Horia sent his 4 `.npz` files and `figure1_plots.py`, which is identical to the repo copy.
  - All 5 `.npz` files (DeiT, η = 0/0.05/0.5/1) are in `results_vsc39861/spectra_all/`. Figures were made on the Mac, with no
    wICE run, into `results_vsc39861/combined/{figures,results}`: `per_layer_c.png`, `per_layer_c2.png`, `ranks_by_layer.png`,
    and `training_curves.png` (which includes η = 10 from `results/raw_logs`).
  - The combined `spectra_summary.csv` rows for Horia's runs are byte-identical to his `results/spectra_summary.csv`.
- **Claude's wICE access:** Claude can reach wICE from the Mac shell (`ssh -o BatchMode=yes vsc39861@login.hpc.kuleuven.be`)
  while the user's firewall opening and MFA certificate are active (`ssh-add -l` shows `vsc-cert`). Read-only checks and
  log copies have been done this way. The user still submits jobs.

Check commands:
```
sacct -M wice -u vsc39861 --starttime 2026-11-16 --format=JobID,JobName,State,Start,End,Elapsed
wc -l $VSC_SCRATCH/primal_fig1/runs/primal_eta{1,10}_seed0/log.txt     # 300 = finished
```
If not submitted [wICE, jobs folder]:
- `sbatch --test-only --export=ALL,ETA=1,SEED=0 train_primal.slurm` should say partition `gpu_h100`.
- `ETAS="1 10" N_JOBS=1 SUBMIT=1 bash submit_chain.sh`.
- `squeue -M wice -u vsc39861 -o "%.10i %.10P %.16a %.8T %.20b %.12j"`.
- `grep -h "^ETA=" primal_in100_*.out` confirms which η each job got.

Unfinished or timed out: resubmit the same way; `train_primal.slurm` resumes, and it exits immediately if the run is finished.

**η = 10 risk:** η = 10 is 20× the largest η tried before (0.5). Check `tail -3 .../primal_eta10_seed0/log.txt` early
for `nan` or rising loss; if the run has diverged, cancel it with `scancel -M wice <id>`.

## Spectra and the figure (pending)
Goal: add η = 1 and η = 10 curves to Horia's `figures/per_layer_c.png` (and `_c2`): a 3×4 grid, one panel per layer,
normalized cumulative singular value against index, with DeiT-S plus η ∈ {0, 0.05, 0.5}.
1. After both runs log 300 epochs, run [wICE, jobs folder] `sbatch --export=ALL spectra.slurm`, on H100 (the default).
   - It writes `$VSC_SCRATCH/primal_fig1/spectra/primal_eta{1,10}_seed0.npz`.
   - It skips runs whose `.npz` exists and runs that are unfinished.
   - It draws partial plots, then copies `spectra/` and `plots/` to `$VSC_DATA/primal_fig1/`.
2. Horia's four `.npz` files are not in git. They are in his `$VSC_DATA/primal_fig1/spectra/`, which is not readable
   from vsc39861. The user will get them from Horia: `deit_seed0.npz`, `primal_eta0_seed0.npz`,
   `primal_eta0.05_seed0.npz`, `primal_eta0.5_seed0.npz`. Upload [Mac, from the folder they are in]:
   `rsync -av *.npz vsc39861@login.hpc.kuleuven.be:/scratch/leuven/398/vsc39861/primal_fig1/spectra/`
3. Replot [wICE login node, no job, no GPU]:
   ```
   source $VSC_DATA/miniconda3/etc/profile.d/conda.sh && conda activate $VSC_DATA/envs/primal-cv && cd $VSC_DATA/PrimalAttention/analysis/imagenet_eta_ablation && python figure1_plots.py --spectra $VSC_SCRATCH/primal_fig1/spectra --out $VSC_DATA/primal_fig1/plots --curve c && python figure1_plots.py --spectra $VSC_SCRATCH/primal_fig1/spectra --out $VSC_DATA/primal_fig1/plots --curve c2
   ```
   - `figure1_plots.py` colours Primal runs from `ETA_COLORS`, which has 5 entries. With 5 η values it exits with
     "more eta values than colors". 5 fits; a 6th η needs another colour. Running with 5 has not been tested.
   - Alternative: copy the `.npz` files to the Mac and run the same script locally.
4. Copy results to the Mac for analysis (logs and plots are small):
   `rsync -av vsc39861@login.hpc.kuleuven.be:/data/leuven/398/vsc39861/primal_fig1/ ~/Primal-Attention/analysis/imagenet_eta_ablation/results_vsc39861/`
   Then compare against `REPORT.md`: top-1 final/best, effective rank of the Primal layer, `J_log.csv` (J², per-head |J|, Λ collapse, KKT).

## Next jobs (prepared 2026-11-22, not yet copied to wICE or submitted)
- **W_e/W_r tracking:** added to `analysis/imagenet_eta_ablation/train_with_j_log.py` (`log_weights`). Every Primal run now also
  writes `W_log.csv` and `W_dirs.csv` next to `J_log.csv`.
  - `W_log.csv`, per head: Frobenius norms, Tr(W_eᵀW_r), cos(W_e,W_r), effective rank of W_e and W_r.
  - `W_dirs.csv`, per direction: |λ_l|, the W_e/W_r column norms, and the norms of the `concate_weight` input columns.
  - Size: about 18 MB per 300-epoch run.
  - The plot is `figure1_plots.py: w_tracking` → `figures/w_tracking.png`. It runs whenever `--logs` is given and uses the
    red-green-safe `SAFE_COLORS`.
  - Self-check: scratchpad `test_w_logging.py`, which passed (not kept).
  - Finding: `nn.init.orthogonal_` on the (H, n, s) tensor (`models_primal.py:46-47`) makes the heads orthonormal as
    flattened rows, not each head's W_e. Per-head effective rank at init is about 89, not 96.
- **ImageNet-1K smoke test:** `jobs/smoke_in1k.slurm` with `MODEL=deit` or `MODEL=primal ETA=…`.
  - It runs on the full ImageNet-1K on 4 H100s for 2 epochs, then prints s/epoch and a cost estimate for a 300-epoch run.
  - The user asked for DeiT, η = 0 and η = 0.5. Expected cost is about 35–50k credits each.
  - `common.sh` now defaults `IMAGENET1K_DIR=$VSC_SCRATCH/imagenet-1k`.
  - **Blocker:** ImageNet-1K is not on wICE yet.
    - **Source:** image-net.org was down on 2026-11-22, so the source is now Hugging Face `ILSVRC/imagenet-1k`
      (the card says ILSVRC 2012: 1,281,167 train / 50,000 val). It's gated (accept on the website); files are parquet:
      294 train + 14 validation shards, about 154 GB (28 test shards skipped). The card says the labels are indexed by
      sorted synset ids. `classes.py` is behind the gate (401), and the file-name format inside the parquet is unknown.
    - **Job:** `jobs/get_imagenet1k.slurm`, a CPU job on `batch` (8 cores, 12 h limit), using base Miniconda.
      - It runs `hf download --include data/train-* data/validation-*` into `$VSC_SCRATCH/imagenet1k_parquet`.
      - Then `convert_imagenet1k.py` writes `$VSC_SCRATCH/imagenet-1k/{train,val}/<0000..0999>/`. Folders are named by
        label index, so the ImageFolder index equals the label equals the standard sorted-wnid order.
      - The converter checks the counts, and the label order against the synsets in the train file names when present
        (it writes `synsets.txt`). The parquet copy is deleted only if the converter succeeds.
      - Self-check: scratchpad `test_convert_in1k.py`, 4 cases, passed (not kept).
    - **Unverified:** the `batch` partition name (check `sinfo -M wice -s`); whether this `hf` version accepts several
      `--include` patterns; download speed. It's a CPU job, an exception to the H100 rule.
    - **The user wants all three smoke tests** (DeiT, η = 0, η = 0.5) regardless of the ponytail pushback.
    - **DONE 2026-11-22: ImageNet-1K is on wICE** at `/scratch/leuven/398/vsc39861/imagenet-1k`.
      - Job 62841742 failed (hf 2.4 needs `--include A --include B`, and the user wasn't logged in); fixed.
      - Job 62841921 COMPLETED in 41 min: 1,281,167 train (732–1,300 per class) and 50,000 val (50 per class),
        1000 classes, `label_order_checked_against_synsets: true`. Parquet deleted.
      - Scratch used about 159 GB after. Balance 1,599,489.
    - **Remote-shell gotcha:** non-interactive `ssh` doesn't set `$VSC_DATA` etc. Use absolute paths, or `bash -lc "…"`,
      especially when submitting (the jobs need `--export=ALL` with the VSC variables).
    - **Smoke-test status: SUBMITTED 2026-11-22**, on the user's request, by Claude via `bash -lc` over ssh:
      62844692 = DeiT, 62844693 = Primal η = 0, 62844694 = Primal η = 0.5. All PENDING on gpu_h100 with intro_vsc39861.
      - Results: `grep -A12 "smoke_in1k summary" smoke_in1k_6284469*.out` in `jobs/`.
      - Outputs: `$VSC_SCRATCH/primal_fig1/smoke_in1k/<name>/`.
      - Balance before submitting: 1,599,489.
    - **RESULTS 2026-11-23: all three COMPLETED (exit 0)** on node s16g15, run one after another (6:58, 6:01, 5:58 elapsed).
      - Credits used 47,356 in total (balance 1,552,133), which is 2,500/min = 150k/h, as predicted.
      - Warm epoch (epoch 1): DeiT 137 s train + 20 s eval; η = 0 138 + 21 s; η = 0.5 139 + 20 s. That is about 0.11 s/step,
        1,251 steps/epoch. Epoch 0 was only 4–13 s slower, so the cold cache barely matters.
      - **300-epoch projection: about 13.1–13.2 h and about 1.96–1.99M credits per run**, which is more than the 1.55M left.
        All three ≈ 5.9M. Peak GPU memory 16.8 GB (DeiT) / 17.2 GB (Primal) of 80 GB.
      - Top-1 after 2 epochs: 0.56 / 0.51 / 0.47 % (chance 0.1 %). Expected, because LR warmup is 5 epochs (starting at
        1e-6) with heavy augmentation, so this is not a quality signal.
      - J_log and W_log have 18 rows each, as expected. Only FutureWarnings (amp API); no errors, no NaN.
      - GPU utilisation was not sampled (only memory), so whether the run is data-loader-bound (64 cores, ≈ 9.3k img/s) is unknown.
      - Copies on the Mac: `results_vsc39861/smoke_in1k/` (the `.out` files; the run dirs only if the rsync from scratch finished).
      - **Full runs (user decision 2026-11-23):** DeiT, Primal **η = 0.05** (replaces η = 0) and Primal η = 0.5, 300 epochs each.
      - **Full-run job (2026-11-24): `jobs/train_in1k.slurm`** (`MODEL=deit` | `MODEL=primal ETA=…`), 300 epochs, `--time=20h`,
        resumes on resubmit, outputs `$VSC_SCRATCH/primal_fig1/runs_in1k/<name>/`, logs/ckpts copied to `$VSC_DATA/primal_fig1/runs_in1k/<name>/`.
        Credit account **`lp_robustness`** (56.1M credits), passed as `sbatch --account=lp_robustness` (the .bashrc
        SBATCH_ACCOUNT would otherwise charge intro_vsc39861). `--test-only` passed for all three; md5s match the Mac.
      - **SUBMITTED 2026-11-24** by Claude on the user's request, all on lp_robustness (balance 56,119,872 before):
        62869857 = DeiT, 62869858 = Primal η = 0.05, 62869861 = Primal η = 0.5 (all job-name `train_in1k`; Slurm output
        `jobs/train_in1k_<id>.out`). Check: `sacct -M wice -j 62869857,62869858,62869861 -X`, then
        `tail -3 $VSC_DATA/primal_fig1/runs_in1k/<name>/log.txt`. Not finished -> resubmit the same command (resumes).
      - **Status 2026-11-25 09:09:** DeiT 62869857 COMPLETED in 13:51 h (166 s/epoch), final top-1 79.92 % (best 79.95; paper
        Table 7a: 79.8). Checkpoints and log are in $VSC_DATA. η = 0.05 62869858 RUNNING since 06:50, 48 epochs, 60.32 %
        (DeiT at epoch 47: 60.69), about 174 s/epoch, so about 14.4 h, ending around 21:15 on 25 Nov. Λ collapsed (74–88 of 96 |λ| near 0,
        residual ≈ 1, Jbar² 3e-5). η = 0.5 62869861 PENDING (Priority, fairshare 38); Slurm worst case 2026-12-02 03:00.
        Other users' jobs aren't visible in squeue. lp_robustness: 53,705,908 (2.41M used, matching DeiT and the η = 0.05 run so far).
      - **Status 2026-11-25 18:49:** η = 0.05 at epoch 226/300, 76.86 % (DeiT at 225: 77.27; gap about 0.4–0.7 pts since epoch 100),
        about 191 s/epoch, so it should end around 22:45. No NaN. Λ now fully collapsed (96/96 near 0, max |λ| about 3e-8). η = 0.5 PENDING, est. start
        2026-11-27 01:00. lp_robustness 52,256,488. Login gotcha: on 25 Nov login.hpc (login-genius) refused and login1/2 said
        "no available backend" for hours; it came back by 18:49. login1/2 have the same ED25519 key (use HostKeyAlias=login.hpc.kuleuven.be).
      - **Status 2026-11-25 23:18:** η = 0.05 COMPLETED 22:44 (15:54 h): **final 79.66 % top-1 (best 79.73 at ep 296), top-5 94.78**;
        DeiT 79.92 (−0.27). Λ 96/96 near zero at the end, KKT residual 1.000, eff. rank W_e 76.6 / W_r 77.5, cos 0.000.
        η = 0.5 RUNNING since 22:56, epoch 7 at 14.15 % (DeiT 12.08 at the same epoch), no spike after warmup yet, only 3–12 of 96
        λ near 0, residual 2.56; it should finish around 13:04 on 26 Nov. The logs (no .pth) are on the Mac in `results_vsc39861/runs_in1k/`. Status
        script: `bash $VSC_DATA/PrimalAttention/analysis/imagenet_eta_ablation/jobs/status_in1k.sh`. lp_robustness 51,611,788.
      - **Status 2026-11-26 12:53:** η = 0.5 RUNNING at epoch 293/300, 79.68 % top-1 (best 79.68 at ep 292), top-5 94.89; DeiT at
        the same epoch 79.91 (−0.23). No NaN, no errors. Λ has now collapsed too (96/96 near 0, max |λ| 2.9e-9, residual 1.000);
        eff. rank W_e 74.7 / W_r 75.2, cos 0.000. Finishes ~13:13. lp_robustness 49,574,827. The Mac cert expired 01:08 on 26 Nov;
        after the user's new login it took a minute to show in `ssh-add -l`.
      - **Status 2026-11-26 18:25: all three COMPLETED (exit 0).** η = 0.5 (62869861) ended 13:13 after 14:17 h: **final 79.65 %
        top-1 (best 79.84 at ep 293), top-5 94.86**; Λ 96/96 near zero, residual 1.000. Logs (no .pth) copied to the Mac; checkpoints
        in `$VSC_DATA/primal_fig1/runs_in1k/<run>/`. lp_robustness 49,336,620.
  - Check `myquota` for a file-count limit first (about 1.33M files).
- **Credits context:** measured by the smoke tests: about 2.0M per 300-epoch IN-1K run (Horia's PLAN.md said 4.55M; my
  earlier guess was 2.7–3.0M), against a balance of 1,552,133. Not even one full run fits in the intro credits.

## Gotchas specific to this account
- Scratch purges files not accessed for 30 days. Only `$VSC_DATA/primal_fig1` is safe: final checkpoints, logs, and
  spectra after the spectra job.
- `squeue`, `sbatch` and `scancel` need `-M wice` / `--clusters=wice`. The scripts set it for sbatch.
- Charged for time used. `sam-quote` shows the requested maximum (6 h = about 900k), not the expected cost.

## IN-1K regularizer analysis (2026-11-26)
- Λ collapses in both η = 0.05 and η = 0.5 on the same schedule: max|λ| 0.11 / 0.055 / 0.026 at epochs 30 / 40 / 50
  (η = 0.5: 0.11 / 0.057 / 0.029); all 576 λ < 1e-3 by epoch 80. KKT residual is exactly 1 from about epoch 20–30, which
  is what Λ = 0 gives by the logger's definition (G_e = -W_r).
- J̄² → 0 by cross-head cancellation: with Λ = 0, per-head J = -Tr(W_eᵀW_r) = -0.11, +0.15, +0.10, -0.04, -0.16, +0.06 at
  epoch 299 (η = 0.05); the loss squares the head mean (engine_primal.py:47-50).
- Likely cause: Λ isn't in the forward pass (models_primal.py:85), so CE gives it no gradient, and timm 0.4.12
  add_weight_decay decays every non-1-D parameter, Λ (6×96) included, with wd = 0.05 at lr = 1e-3. lr·wd·1251 steps ≈ e^-0.6
  per 10 epochs, which matches the observed halving between epochs 30 and 50. Positivity of Λ isn't enforced.
