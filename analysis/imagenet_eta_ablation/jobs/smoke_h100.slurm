#!/bin/bash -l
# DRAFT — not tested on wICE. Stage S0: validate the pipeline and measure speed before spending
# the main credits. One H100 on gpu_h100 for 1 hour (wICE has no H100 debug partition).
# About 37,500 credits (1 H100 + 16 Genoa cores, the documented maximum per H100).
#
#   cd analysis/imagenet_eta_ablation/jobs && sbatch --export=ALL smoke_h100.slurm
#
#SBATCH --clusters=wice
#SBATCH --partition=gpu_h100
#SBATCH --account=intro_vsc39859
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=187200m
#SBATCH --time=01:00:00
#SBATCH --job-name=primal_smoke
#SBATCH --output=%x_%j.out
set -euo pipefail
source "${SLURM_SUBMIT_DIR}/common.sh"
activate_env
require_data "$SMOKE_DATA"

OUT="$WORK_DIR/smoke/primal_eta0.05_seed0"
mkdir -p "$OUT"
cd "$CV_DIR"

# Peak GPU memory, sampled every 5 s.
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -l 5 > "$OUT/gpu_mem.txt" &
SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null || true' EXIT

# --num_workers 15: 15 loader workers + the main process use the 16 cores per H100.
COMMON=(--model primal_small_patch16_224 --batch-size 256 --data-path "$SMOKE_DATA" --data-set IMNET-100
        --low-rank 96 --rank-multi 10 --num-ksvd-layer 1 --eta 0.05 --seed 0 --num_workers 15
        --output_dir "$OUT")

# Started with torchrun (1 process), as the training jobs are. A plain `python main_combo.py` fails inside Slurm:
# utils.init_distributed_mode() sees SLURM_PROCID and starts distributed mode without MASTER_ADDR
# (PrimalAttention_CV/utils.py:221-223). torchrun sets RANK, WORLD_SIZE and MASTER_ADDR, so the first branch is used.
LAUNCH=(torchrun --standalone --nproc_per_node=1)
start=$(date +%s)
"${LAUNCH[@]}" "$J_LOGGER" main_combo.py "${COMMON[@]}" --epochs 2
mid=$(date +%s)
# Resume test: one more epoch from checkpoint.pth.
"${LAUNCH[@]}" "$J_LOGGER" main_combo.py "${COMMON[@]}" --epochs 3 --resume "$OUT/checkpoint.pth"
end=$(date +%s)

steps_per_epoch=$(python -c "n=$(find "$SMOKE_DATA/train" -type l | wc -l); print(n // 256 * 256 // 256)")
echo "steps/epoch: $steps_per_epoch"
echo "2 epochs incl. startup and eval: $((mid - start)) s; resumed 3rd epoch: $((end - mid)) s"
echo "peak GPU memory (MiB): $(sort -n "$OUT/gpu_mem.txt" | tail -1)"
echo "epochs logged: $(wc -l < "$OUT/log.txt") (expect 3)"
echo "J_log.csv rows: $(($(wc -l < "$OUT/J_log.csv") - 1)) (expect 24: 6 heads x [before training + 3 epochs])"
# Per-step time: read the 'Epoch: [k] Total time' lines printed by utils.MetricLogger in the job output,
# and compare with the paper's 2.33 s/step (2,330 s per 1K steps on one V100 at batch 256).
# The H100 is only cheaper than the V100 if it is at least about 8x faster per step: see PLAN.md, "Credits".
