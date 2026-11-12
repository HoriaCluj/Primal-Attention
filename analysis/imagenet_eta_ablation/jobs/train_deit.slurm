#!/bin/bash -l
# DeiT-Small/16 baseline trained on ImageNet-100 with the repository's main.py (bashes/run_deit_small.sh),
# same hardware and settings as train_primal.slurm. Its checkpoint gives the like-for-like softmax spectra
# next to the Primal.+ runs, and its accuracy shows how far our setup is from the paper's.
# About 150,000 credits per wall-hour, charged for time used. The Primal.+ eta=0.05 run took 1 h 40 min
# (job 62653792, 250,316 credits); DeiT-S is slightly cheaper per step. --time=4h is a safety limit.
#
#   cd analysis/imagenet_eta_ablation/jobs && sbatch --export=ALL,SEED=0 train_deit.slurm
#
#SBATCH --clusters=wice
#SBATCH --partition=gpu_h100
#SBATCH --account=intro_vsc39859
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --mem=748800m
#SBATCH --time=04:00:00
#SBATCH --job-name=deit_in100
#SBATCH --output=%x_%j.out
set -euo pipefail
source "${SLURM_SUBMIT_DIR}/common.sh"
SEED="${SEED:-0}"
activate_env
require_data "$DATA_DIR"

OUT="$RUNS_DIR/deit_seed${SEED}"
mkdir -p "$OUT"
CKPT="$OUT/checkpoint.pth"
if [ -f "$OUT/log.txt" ] && grep -q '"epoch": 299' "$OUT/log.txt"; then
    echo "Run $OUT already finished; nothing to do."
    exit 0
fi
RESUME=()
if [ -f "$CKPT" ] && python -c "import sys, torch; torch.load(sys.argv[1], map_location='cpu')" "$CKPT" 2>/dev/null; then
    RESUME=(--resume "$CKPT")
fi

cd "$CV_DIR"
torchrun --standalone --nproc_per_node=4 main.py \
    --model deit_small_patch16_224 --batch-size 256 \
    --data-path "$DATA_DIR" --data-set IMNET-100 \
    --seed "$SEED" --num_workers 15 --output_dir "$OUT" "${RESUME[@]}"

mkdir -p "$SAVE_DIR/runs/$(basename "$OUT")"
cp "$OUT/log.txt" "$SAVE_DIR/runs/$(basename "$OUT")/" 2>/dev/null || true
if grep -q '"epoch": 299' "$OUT/log.txt" 2>/dev/null; then
    cp "$CKPT" "$OUT/best_checkpoint.pth" "$SAVE_DIR/runs/$(basename "$OUT")/"
    echo "Finished: log and checkpoints copied to $SAVE_DIR/runs/$(basename "$OUT")"
else
    echo "Not finished (time limit?): resubmit the same sbatch command to resume."
fi
