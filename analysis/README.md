# analysis/

Work on how the KSVD regularizer coefficient η affects Primal-Attention. One folder per experiment.

| Folder | Status | What it is |
|---|---|---|
| [timeseries_eta_ablation/](timeseries_eta_ablation/) | finished | η sweep on UWaveGestureLibrary with Primal.+Trans. Start with [REPORT.md](timeseries_eta_ablation/REPORT.md); method notes in [PHASE1_REPORT.md](timeseries_eta_ablation/PHASE1_REPORT.md) |
| [imagenet_eta_ablation/](imagenet_eta_ablation/) | in progress on wICE | Figure 1 of the paper on ImageNet-100 with Primal.+DeiT-Small/16. Commands to run, with progress: [RUNBOOK.md](imagenet_eta_ablation/RUNBOOK.md). Reasons: [PLAN.md](imagenet_eta_ablation/PLAN.md); evidence in [RECON_REPORT.md](imagenet_eta_ablation/RECON_REPORT.md); draft Slurm scripts in `jobs/`; dataset conversion in `convert_imagenet100.py` |

Datasets live in the git-ignored `data/` folder at the repository root.
Large outputs (`timeseries_eta_ablation/runs/`, 1.4 GB of checkpoints) are git-ignored by their own `.gitignore`.
