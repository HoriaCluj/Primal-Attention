# Runbook: ImageNet-100 η ablation on wICE (H100)

Step-by-step commands. Background and reasons are in [PLAN.md](PLAN.md). Steps marked **(Claude)** are code changes I make; you run
everything else.

- **Laptop** = your WSL terminal, prompt `horia@...`.
- **wICE** = logged in to the cluster, prompt `vsc39859@tier2-p-login-...`.
- **Don't run `rsync` on wICE:** every `rsync` below runs on the laptop. Running one on wICE copies the wrong files, which has happened once already.

| Item | Value |
|---|---|
| Credit account | `intro_vsc39859` (about 2,000,000 credits at the start) |
| `$VSC_DATA` | `/data/leuven/398/vsc39859` (code in `PrimalAttention/`, conda in `miniconda3/`, env in `envs/primal-cv`) |
| `$VSC_SCRATCH` | dataset in `imagenet-100/`, smoke subset in `imagenet_smoke/`, runs in `primal_fig1/runs/` |
| Job scripts | `$VSC_DATA/PrimalAttention/analysis/imagenet_eta_ablation/jobs` (submit from this folder) |
| Check jobs | `squeue -M wice -u vsc39859` (PD = waiting, R = running; no line = finished) |
| Cancel a job | `scancel -M wice <jobid>` |

## Following a job (on wICE)

| Question | Command |
|---|---|
| State (`PD` = waiting, `R` = running, no line = finished) and the reason it waits | `squeue -M wice -u vsc39859` |
| Estimated start time | `squeue -M wice -u vsc39859 --start` |
| Pending / running jobs on the H100s | `squeue -M wice -p gpu_h100 -t PD -h \| wc -l` and `-t R` |
| Free H100 nodes (`idle`) | `sinfo -M wice -p gpu_h100` |
| Your job's priority | `sprio -M wice -j <jobid>` |
| After it ends: start, end, run time, state | `sacct -M wice -j <jobid> --format=JobID,State,Start,End,Elapsed` |
| Refresh automatically | `watch -n 60 squeue -M wice -u vsc39859` (Ctrl+C to stop) |
| Email at start and end (wICE email support unconfirmed) | `sbatch --export=ALL --mail-type=BEGIN,END,FAIL --mail-user=<your email> <script>` |

**Always submit with `--export=ALL`** (already inside `submit_chain.sh`). Otherwise Slurm rebuilds your login environment on the
compute node with an 8-second timeout. wICE's system login scripts take about 8.5 s, so jobs end up held with
`user env retrieval failed requeued held`.

Waiting reasons: `Priority` = others ahead; `Resources` = next in line, waiting for a free GPU; `Assoc`/`Grp`/`Billing` = an account or
credit limit (send it to Claude).

## Progress

- [ ] 1. Local test with the torch 2.4.1 environment (5 steps)
- [ ] 2. Supervisor sign-off
- [x] 3. Log in to wICE
- [x] 4. `sam-balance`, note paths
- [x] 5. Copy code and data to wICE
- [x] 6. Install Miniconda
- [x] 7. Smoke subset and environment (`env_setup.sh` fixed to use conda-forge only)
- [x] 8. First `sam-quote`: smoke 37,440, one training job 1,199,520 for the 8 h placeholder
- [x] 9. Smoke test passed: **job 62653652**, 90 s, about 0.2–0.24 s/step (earlier attempts 62653327 held, 62653597 failed; both fixed)
- [x] 10. Cost check: **charged for time used** (936 credits for 90 s); all three smoke attempts about 1,650 credits
- [x] 11. Decision: all three η fit (about 3–3.7 h and 0.45–0.56M per run); `--time` set to 6 h
- [x] 12. Second `sam-quote` (6 h requested; charged for time used)
- [x] 13. Sanity run η = 0.05: **job 62653792**, 12 Nov 06:18–07:58 (1 h 40 min), 300 epochs, final test_acc1 73.96 (paper 75.7), J_log 1,807 rows, **250,316 credits**; balance 1,748,030
- [ ] 14. Runs η = 0 and η = 0.5, plus the DeiT-S ImageNet-100 baseline (`train_deit.slurm`, `--time` 4 h): submitted 12 Nov as jobs **62663183** and **62663184** (η = 0 and 0.5, in submission order) and **62663186** (DeiT); queued, Slurm estimate 15 Nov
- [ ] 15. Spectra script (Claude)
- [ ] 16. Spectra job
- [ ] 17. Copy results to the laptop
- [ ] 18. Report (Claude)

---

## Phase A: laptop

### 1. Local test (about 12 minutes)

```bash
cd /home/horia/PrimalAttention
bash analysis/imagenet_eta_ablation/local_test.sh 2>&1 | tee data/local_test.log
```

**Success:** five `ok` lines (steps A to E), then `J_log.csv has 24 rows` for `primal_eta0.05` and 12 rows each for `primal_eta0` and
`primal_torchrun`. A `CUDA error: unknown error` is a laptop GPU hiccup: keep the laptop awake and plugged in, then rerun.

### 2. Supervisor sign-off

The scope to approve:
- ImageNet-100 instead of ImageNet-1K;
- η ∈ {0, 0.05, 0.5}, one seed each;
- H100 with torch 2.4.1 instead of the paper's V100 with torch 1.7.1;
- a cost that is known only after the smoke test.

PLAN.md, section 4.6, lists the open questions.

---

## Phase B: wICE login (done)

### 3. Log in

```bash
ssh vsc39859@login.hpc.kuleuven.be
```

### 4. Credits and paths

```bash
sam-balance
echo $VSC_DATA $VSC_SCRATCH
```

---

## Phase C: copy code and data (laptop; done)

### 5. Copy

Code (rerun after any code change; `rsync` only sends what changed):

```bash
cd /home/horia/PrimalAttention
rsync -av --exclude data/ --exclude '.venv/' --exclude 'analysis/timeseries_eta_ablation/runs/' \
    --exclude 'PrimalAttention_TimeSeries/*/Multivariate_ts/' --exclude 'PrimalAttention_TimeSeries/*/experiments/' \
    ./ vsc39859@login.hpc.kuleuven.be:/data/leuven/398/vsc39859/PrimalAttention/
```

Add `-n` (`-avn`) for a dry run that lists what would be copied without copying.

Data (done: 15,185,833,919 bytes):

```bash
rsync -av data/imagenet-100/ vsc39859@login.hpc.kuleuven.be:<VSC_SCRATCH>/imagenet-100/
```

---

## Phase D: setup on wICE (done)

### 6. Miniconda

```bash
cd $VSC_DATA
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $VSC_DATA/miniconda3
rm Miniconda3-latest-Linux-x86_64.sh
```

`conda: command not found` afterwards is normal; the scripts call conda by its full path. To use conda by hand, first run
`source $VSC_DATA/miniconda3/etc/profile.d/conda.sh`.

### 7. Smoke subset and environment

```bash
cd $VSC_DATA/PrimalAttention/analysis/imagenet_eta_ablation/jobs
bash make_smoke_subset.sh     # expect: classes: 100 train, 100 val | images: 13000 train, 1000 val
bash env_setup.sh
```

**Success:** `torch 2.4.1+cu121 | torchvision 0.19.1+cu121 | timm 0.4.12`, `H100 support (sm_90): True`, and `all checks passed`. On a login
node, `GPU visible: False` and `2. mixup skipped` are normal. To rerun only the code check:

```bash
$VSC_DATA/envs/primal-cv/bin/python ../check_env.py --data $VSC_SCRATCH/imagenet_smoke
```

If `env_setup.sh` stops with a conda "Terms of Service" error, the conda-forge fix is missing on wICE. Recopy `jobs/env_setup.sh` with the
code `rsync` in step 5, run `rm -rf $VSC_DATA/envs/primal-cv`, then rerun.

---

## Phase E: cost check and smoke test

### 8. First `sam-quote` (done)

```bash
sam-quote smoke_h100.slurm      # 37,440
sam-quote train_primal.slurm    # 1,199,520 = ONE run, for the full 8 h placeholder
```

The quote is the **requested** time × resources: 4 H100 + 64 cores ≈ 2,499 credits per minute, or about 150,000 per hour.

### 9. Smoke test (running: job 62653652)

```bash
sbatch --export=ALL smoke_h100.slurm
squeue -M wice -u vsc39859
```

The output goes to `primal_smoke_62653652.out` in the `jobs/` folder. **Success:** `epochs logged: 3 (expect 3)`, `J_log.csv rows: 24`,
and the `Epoch: [k] Total time` lines, which give the seconds per step.

### 10. What the smoke test really cost

```bash
sam-list-usagerecords      # or: sam-statement
sam-balance
```

Send Claude three things:
- the `.out` file;
- the job's actual charge;
- the balance.

**How to read the charge:**
- **Below 37,440, while the job finished in under an hour:** wICE charges for time **used**. Then `--time` only needs to be a safe upper limit.
- **About 37,440 even though the job finished early:** wICE charges for time **requested**. Then `--time` must be set tightly.

### 11. Decision and `--time` (Claude)

| Seconds per step on one H100 | One run | Action |
|---|---|---|
| **≤ 0.39 s** | ≤ 4 h, ≤ 0.6M | All three η values fit. Set `--time`, run with `N_JOBS=1` |
| **0.39–0.58 s** | 4–6 h, 0.6–0.9M | Three don't fit: two η values (0.05 and 0), or V100 |
| **> 0.58 s** | over 6 h, over 0.9M | V100, or request project credits |

Then recopy the code from the laptop (the `rsync` command in step 5).

### 12. Second `sam-quote`

```bash
sam-quote train_primal.slurm
```

Expect about 900,000: the full 6 h requested. Because wICE charges for time **used**, a run should actually cost about
450,000–555,000. If `sbatch` refuses a job for insufficient credits, submit the runs one at a time.

---

## Phase F: training runs on wICE

### 13. Sanity run, η = 0.05

```bash
cd $VSC_DATA/PrimalAttention/analysis/imagenet_eta_ablation/jobs
ETAS="0.05" N_JOBS=1 bash submit_chain.sh            # dry run
ETAS="0.05" N_JOBS=1 SUBMIT=1 bash submit_chain.sh   # submit
```

To follow it:

```bash
squeue -M wice -u vsc39859
wc -l $VSC_SCRATCH/primal_fig1/runs/primal_eta0.05_seed0/log.txt      # finished epochs, out of 300
tail -n 1 $VSC_SCRATCH/primal_fig1/runs/primal_eta0.05_seed0/log.txt  # latest epoch's stats
```

**Success:** 300 lines in `log.txt`, with `test_acc1` close to the paper's 75.7%. `log.txt`, `J_log.csv` and the checkpoints are copied to
`$VSC_DATA/primal_fig1/runs/`.

Then run `sam-balance`, and send Claude the last line of `log.txt` and the real cost. If the job hit its time limit before epoch 300, resubmit
the same command; it resumes from the last saved epoch.

### 14. Runs η = 0 and η = 0.5

Check first that `sam-balance` covers two more runs at the cost you just saw:

```bash
sam-quote train_primal.slurm
ETAS="0 0.5" N_JOBS=1 SUBMIT=1 bash submit_chain.sh
```

---

## Phase G: spectra and report

### 15. Spectra script (Claude)

Claude writes `figure1_spectra.py`, tested on the laptop's local-test checkpoints. Then recopy the code (step 5).

### 16. Spectra job (1 H100, about 37,500 credits per hour, a few hours)

```bash
sam-quote spectra.slurm
sbatch --export=ALL spectra.slurm
```

### 17. Copy results to the laptop

```bash
rsync -av vsc39859@login.hpc.kuleuven.be:/data/leuven/398/vsc39859/primal_fig1/ /home/horia/PrimalAttention/data/primal_fig1/
```

### 18. Report (Claude)

Accuracy against the paper, per-layer spectra, J and residual diagnostics, and the single-seed caveat.

---

## Optional, if credits remain

- **DeiT-S baseline on ImageNet-100**, about the cost of one Primal.+ run. Run `sam-balance`, set `--time` in `train_deit.slurm` like
  `train_primal.slurm`, then `sam-quote train_deit.slurm` and `sbatch --export=ALL train_deit.slurm`.
- **Calibration against the paper's Figure 1**, at no credit cost: download the ImageNet-1K validation set (6.4 GB) to the laptop, and
  Claude runs the comparison with the official DeiT-S.

**`sam-quote` before every new submission (steps 8, 12, 14, 16); `sam-balance` after every real run.**
