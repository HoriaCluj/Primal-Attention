#!/bin/bash -l
# DRAFT — not tested on wICE. Spectra of every finished Primal.+ run (final-epoch checkpoint.pth,
# non-EMA weights) on 1,000 ImageNet-100 val images (10 per class), then the figures. Evaluation only.
#
#   cd analysis/imagenet_eta_ablation/jobs && sbatch --export=ALL spectra.slurm
#
#SBATCH --clusters=wice
#SBATCH --partition=gpu_h100
#SBATCH --account=intro_vsc39859
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=187200m
#SBATCH --time=04:00:00
#SBATCH --job-name=primal_spectra
#SBATCH --output=%x_%j.out
set -euo pipefail
source "${SLURM_SUBMIT_DIR}/common.sh"
activate_env
require_data "$DATA_DIR"

# analysis/imagenet_eta_ablation/figure1_spectra.py is NOT written yet (see PLAN.md 4.2 and 4.4).
SPECTRA="$REPO_DIR/analysis/imagenet_eta_ablation/figure1_spectra.py"
[ -f "$SPECTRA" ] || { echo "$SPECTRA does not exist yet." >&2; exit 1; }

cd "$CV_DIR"
for run in "$RUNS_DIR"/*_eta*_seed* "$RUNS_DIR"/deit_seed*; do
    [ -d "$run" ] || continue
    if ! grep -q '"epoch": 299' "$run/log.txt" 2>/dev/null; then
        echo "skipping unfinished run: $run"
        continue
    fi
    model=primal_plus; case "$(basename "$run")" in deit_*) model=deit ;; esac
    python "$SPECTRA" --model "$model" --checkpoint "$run/checkpoint.pth" \
        --data-path "$DATA_DIR" --images-per-class 10 --out "$WORK_DIR/spectra"
done

mkdir -p "$SAVE_DIR"
cp -r "$WORK_DIR/spectra" "$SAVE_DIR/"
