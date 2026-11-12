#!/bin/bash -l
# DRAFT — not tested on wICE. Stages S2/S3: one Primal.+DeiT-Small/16 training run on ImageNet-100,
# on 4 H100 80 GB, batch 256 per GPU = 1024 total, as in the paper and
# upstream DeiT-S. Only ETA (and SEED) change between runs; every other flag is the repository's
# ImageNet-100 command (bashes/run_combo_small_rank96_layer1.sh), except --num_workers (speed only).
# About 150,000 credits per wall-hour (4 H100 + 64 Genoa cores), charged for time USED (smoke test 62653652:
# 936 credits for 90 s). Smoke test: about 0.2-0.24 s/step, so a run is expected to take about 3-3.7 h;
# --time=6h is a safety limit, not a cost.
#
#   cd analysis/imagenet_eta_ablation/jobs && sbatch --export=ALL,ETA=0.05,SEED=0 train_primal.slurm
#
#SBATCH --clusters=wice
#SBATCH --partition=gpu_h100
#SBATCH --account=intro_vsc39859
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --mem=748800m
#SBATCH --time=06:00:00
#SBATCH --job-name=primal_in100
#SBATCH --output=%x_%j.out
set -euo pipefail
source "${SLURM_SUBMIT_DIR}/common.sh"
: "${ETA:?pass ETA with --export=ALL,ETA=<value>}"
SEED="${SEED:-0}"
activate_env
require_data "$DATA_DIR"

OUT="$RUNS_DIR/primal_eta${ETA}_seed${SEED}"
mkdir -p "$OUT"
CKPT="$OUT/checkpoint.pth"
BACKUP="$OUT/checkpoint.backup.pth"

# Finished already? main_combo.py appends one JSON line per epoch to log.txt; the last epoch is 299.
if [ -f "$OUT/log.txt" ] && grep -q '"epoch": 299' "$OUT/log.txt"; then
    echo "Run $OUT already finished; nothing to do."
    exit 0
fi

# Resume from the newest checkpoint that loads. A job killed during torch.save can leave a broken
# checkpoint.pth, so a validated copy is kept as checkpoint.backup.pth.
RESUME=()
loads() { python -c "import sys, torch; torch.load(sys.argv[1], map_location='cpu')" "$1" 2>/dev/null; }
if [ -f "$CKPT" ] && loads "$CKPT"; then
    cp "$CKPT" "$BACKUP"
    RESUME=(--resume "$CKPT")
elif [ -f "$BACKUP" ] && loads "$BACKUP"; then
    echo "checkpoint.pth missing or unreadable; resuming from the backup copy."
    cp "$BACKUP" "$CKPT"
    RESUME=(--resume "$CKPT")
fi
echo "ETA=$ETA SEED=$SEED OUT=$OUT resume=${RESUME[*]:-none}"

# --num_workers 15 per GPU: 4 processes x (1 main + 15 loader workers) = the 64 cores wICE allows for
# 4 H100s (16 per GPU). This changes loading speed, not the model or the augmentations.
# torchrun replaces the deprecated torch.distributed.launch; it sets RANK, WORLD_SIZE and LOCAL_RANK,
# which utils.init_distributed_mode() reads (PrimalAttention_CV/utils.py:217-220).
cd "$CV_DIR"
# train_with_j_log.py runs main_combo.py unchanged and also writes $OUT/J_log.csv (per-head J, KKT residuals).
torchrun --standalone --nproc_per_node=4 "$J_LOGGER" main_combo.py \
    --model primal_small_patch16_224 --batch-size 256 \
    --data-path "$DATA_DIR" --data-set IMNET-100 \
    --low-rank 96 --rank-multi 10 --num-ksvd-layer 1 --eta "$ETA" \
    --seed "$SEED" --num_workers 15 --output_dir "$OUT" "${RESUME[@]}"

# Copy the small, important files to backed-up storage after every job.
mkdir -p "$SAVE_DIR/runs/$(basename "$OUT")"
cp "$OUT/log.txt" "$OUT/J_log.csv" "$SAVE_DIR/runs/$(basename "$OUT")/" 2>/dev/null || true
if grep -q '"epoch": 299' "$OUT/log.txt" 2>/dev/null; then
    cp "$CKPT" "$OUT/best_checkpoint.pth" "$SAVE_DIR/runs/$(basename "$OUT")/"
fi
