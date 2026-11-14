#!/bin/bash -l
# Spectra of every finished run (final-epoch checkpoint.pth, non-EMA weights) on 1,000 ImageNet-100 val
# images (the first 10 files of each class), then the figures. Evaluation only, 1 H100: about 37,500 credits
# per hour, charged for time used; expected well under 1 h (about 7 min per model on a laptop CPU).
#
#   cd analysis/imagenet_eta_ablation/jobs && sbatch --export=ALL spectra.slurm
# SVD_DEVICE=cpu (sbatch --export=ALL,SVD_DEVICE=cpu ...) runs the SVDs on the CPU cores instead of the GPU.
# Runs whose .npz already exists are skipped, so a resubmitted job continues where the last one stopped.
#
#SBATCH --clusters=wice
#SBATCH --partition=gpu_h100
#SBATCH --account=intro_vsc39859
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=187200m
#SBATCH --time=02:00:00
#SBATCH --job-name=primal_spectra
#SBATCH --output=%x_%j.out
set -euo pipefail
source "${SLURM_SUBMIT_DIR}/common.sh"
activate_env
require_data "$DATA_DIR"

HERE="$REPO_DIR/analysis/imagenet_eta_ablation"
SPECTRA_OUT="$WORK_DIR/spectra"
mkdir -p "$SPECTRA_OUT"
for run in "$RUNS_DIR"/primal_eta*_seed* "$RUNS_DIR"/deit_seed*; do
    [ -d "$run" ] || continue
    name="$(basename "$run")"
    if ! grep -q '"epoch": 299' "$run/log.txt" 2>/dev/null; then
        echo "skipping unfinished run: $run"
        continue
    fi
    if [ -f "$SPECTRA_OUT/$name.npz" ]; then
        echo "already done: $name"
        continue
    fi
    model=primal_plus; case "$name" in deit_*) model=deit ;; esac
    python "$HERE/figure1_spectra.py" --model "$model" --checkpoint "$run/checkpoint.pth" --name "$name" \
        --data-path "$DATA_DIR" --images-per-class 10 --out "$SPECTRA_OUT" ${SVD_DEVICE:+--svd-device "$SVD_DEVICE"}
done

# Figures and summary table (also reproducible on the laptop from the copied .npz files).
python "$HERE/figure1_plots.py" --spectra "$SPECTRA_OUT" --out "$WORK_DIR/plots" --curve c
python "$HERE/figure1_plots.py" --spectra "$SPECTRA_OUT" --out "$WORK_DIR/plots" --curve c2

mkdir -p "$SAVE_DIR"
cp -r "$SPECTRA_OUT" "$WORK_DIR/plots" "$SAVE_DIR/"
echo "Done: $SAVE_DIR/spectra and $SAVE_DIR/plots"
