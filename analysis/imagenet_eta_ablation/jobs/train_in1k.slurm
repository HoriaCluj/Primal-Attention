#!/bin/bash -l
# Full 300-epoch training on ImageNet-1K (1000 classes) on 4 H100 80 GB, batch 256 per GPU = 1024, as in the paper.
# Same flags as the ImageNet-100 runs (train_deit.slurm / train_primal.slurm) except --data-set IMNET and the data path.
# Smoke tests 62844692-4 (2026-11-23): ~159 s per epoch for DeiT and Primal alike -> ~13.2 h and ~2.0M credits per run
# (150k credits per wall-hour, charged for time used). --time=20h is a safety limit, not a cost.
# Resubmitting the same command resumes from the last checkpoint; a finished run exits at once.
#
# Pass the credit account on the command line: ~/.bashrc sets SBATCH_ACCOUNT, which overrides #SBATCH --account.
#   cd analysis/imagenet_eta_ablation/jobs
#   sbatch --account=<credit account> --export=ALL,MODEL=deit train_in1k.slurm
#   sbatch --account=<credit account> --export=ALL,MODEL=primal,ETA=0.05 train_in1k.slurm
#   sbatch --account=<credit account> --export=ALL,MODEL=primal,ETA=0.5 train_in1k.slurm
#
#SBATCH --clusters=wice
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --mem=748800m
#SBATCH --time=20:00:00
#SBATCH --job-name=train_in1k
#SBATCH --output=%x_%j.out
set -euo pipefail
source "${SLURM_SUBMIT_DIR}/common.sh"
: "${MODEL:?pass MODEL=deit or MODEL=primal with --export=ALL,MODEL=...}"
SEED="${SEED:-0}"
activate_env
require_data "$IMAGENET1K_DIR"

case "$MODEL" in
    deit)   NAME=deit_seed${SEED}
            CMD=(main.py --model deit_small_patch16_224)
            LOGS=(log.txt) ;;
    primal) : "${ETA:?pass ETA for MODEL=primal}"
            NAME=primal_eta${ETA}_seed${SEED}
            # train_with_j_log.py runs main_combo.py unchanged and also writes J_log.csv, W_log.csv and W_dirs.csv.
            CMD=("$J_LOGGER" main_combo.py --model primal_small_patch16_224
                 --low-rank 96 --rank-multi 10 --num-ksvd-layer 1 --eta "$ETA")
            LOGS=(log.txt J_log.csv W_log.csv W_dirs.csv) ;;
    *)      echo "MODEL must be deit or primal" >&2; exit 2 ;;
esac
OUT="$WORK_DIR/runs_in1k/$NAME"
SAVE="$SAVE_DIR/runs_in1k/$NAME"
mkdir -p "$OUT" "$SAVE"
CKPT="$OUT/checkpoint.pth"
BACKUP="$OUT/checkpoint.backup.pth"

if [ -f "$OUT/log.txt" ] && grep -q '"epoch": 299' "$OUT/log.txt"; then
    echo "Run $OUT already finished; nothing to do."
    exit 0
fi

# Resume from the newest checkpoint that loads; a job killed during torch.save can leave a broken checkpoint.pth.
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
echo "MODEL=$MODEL ETA=${ETA:-} SEED=$SEED account=${SLURM_JOB_ACCOUNT:-?} OUT=$OUT resume=${RESUME[*]:-none}"

cd "$CV_DIR"
status=0
torchrun --standalone --nproc_per_node=4 "${CMD[@]}" \
    --batch-size 256 --data-path "$IMAGENET1K_DIR" --data-set IMNET \
    --seed "$SEED" --num_workers 15 --output_dir "$OUT" "${RESUME[@]}" || status=$?

# Scratch is purged after 30 days without access: copy the logs to $VSC_DATA after every job, the checkpoints at the end.
for f in "${LOGS[@]}"; do cp "$OUT/$f" "$SAVE/" 2>/dev/null || true; done
if grep -q '"epoch": 299' "$OUT/log.txt" 2>/dev/null; then
    cp "$CKPT" "$SAVE/"
    cp "$OUT/best_checkpoint.pth" "$SAVE/" 2>/dev/null || true
    echo "Finished: logs and checkpoints copied to $SAVE"
else
    echo "Not finished (exit $status; time limit or NaN?): check the log, then resubmit the same command to resume."
fi
exit "$status"
