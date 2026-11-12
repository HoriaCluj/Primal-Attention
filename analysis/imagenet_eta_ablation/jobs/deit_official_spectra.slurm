#!/bin/bash -l
# DRAFT — not tested on wICE. Stage S1 on wICE. PLAN.md runs S1 on the laptop instead (no credits);
# use this only if that is not possible. Calibrates the measurement on the official DeiT-Small/16
# (trained on ImageNet-1K), so it needs the full ImageNet-1K folder: main.py builds the train set
# even with --eval. (1) Top-1 on the full ImageNet-1K val set (expect 79.8). (2) Spectra of all
# 12 layers on the 1,000 fixed images, to compare with the published Figure 1 (pics/heatmap.jpg).
# The official weights (about 88 MB) are fetched from DEIT_S_URL by torch.hub on first use.
#
#   cd analysis/imagenet_eta_ablation/jobs && sbatch --export=ALL deit_official_spectra.slurm
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
#SBATCH --job-name=deit_official_spectra
#SBATCH --output=%x_%j.out
set -euo pipefail
source "${SLURM_SUBMIT_DIR}/common.sh"
activate_env
require_data "$IMAGENET1K_DIR"

OUT="$WORK_DIR/deit_official"
mkdir -p "$OUT"
cd "$CV_DIR"

# (1) Accuracy, with the repository's own evaluation path (main.py --eval --resume <https URL>).
# torchrun with 1 process: a plain `python main.py` fails inside Slurm (see smoke_h100.slurm).
torchrun --standalone --nproc_per_node=1 main.py --eval --resume "$DEIT_S_URL" \
    --model deit_small_patch16_224 --batch-size 256 --data-path "$IMAGENET1K_DIR" --data-set IMNET --num_workers 15 \
    | tee "$OUT/eval.txt"

# (2) Spectra. analysis/imagenet_eta_ablation/figure1_spectra.py is NOT written yet (see PLAN.md 4.2 and 4.4).
SPECTRA="$REPO_DIR/analysis/imagenet_eta_ablation/figure1_spectra.py"
if [ ! -f "$SPECTRA" ]; then
    echo "Skipping spectra: $SPECTRA does not exist yet." >&2
    exit 0
fi
python "$SPECTRA" --model deit_official --checkpoint "$DEIT_S_URL" \
    --data-path "$IMAGENET1K_DIR" --images-per-class 1 --out "$WORK_DIR/spectra"
