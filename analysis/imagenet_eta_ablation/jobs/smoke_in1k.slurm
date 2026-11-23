#!/bin/bash -l
# Smoke test on the FULL ImageNet-1K, with the real 4-H100 setup, for EPOCHS epochs (default 2; the second
# shows the speed once the node's page cache holds the data). Checks that the 1000-class pipeline runs and
# measures the time per epoch, from which the cost of a 300-epoch run is estimated.
# About 150,000 credits per wall-hour, charged for time used; expect roughly 15-20 min, i.e. 35-50k per job.
#
#   cd analysis/imagenet_eta_ablation/jobs
#   sbatch --export=ALL,MODEL=deit smoke_in1k.slurm
#   sbatch --export=ALL,MODEL=primal,ETA=0 smoke_in1k.slurm
#   sbatch --export=ALL,MODEL=primal,ETA=0.5 smoke_in1k.slurm
#
#SBATCH --clusters=wice
#SBATCH --partition=gpu_h100
#SBATCH --account=intro_vsc39861
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --mem=748800m
#SBATCH --time=01:30:00
#SBATCH --job-name=smoke_in1k
#SBATCH --output=%x_%j.out
set -euo pipefail
source "${SLURM_SUBMIT_DIR}/common.sh"
: "${MODEL:?pass MODEL=deit or MODEL=primal with --export=ALL,MODEL=...}"
EPOCHS="${EPOCHS:-2}"
activate_env
require_data "$IMAGENET1K_DIR"

case "$MODEL" in
    deit)   NAME=deit_seed0
            CMD=(main.py --model deit_small_patch16_224) ;;
    primal) : "${ETA:?pass ETA for MODEL=primal}"
            NAME=primal_eta${ETA}_seed0
            CMD=("$J_LOGGER" main_combo.py --model primal_small_patch16_224
                 --low-rank 96 --rank-multi 10 --num-ksvd-layer 1 --eta "$ETA") ;;
    *)      echo "MODEL must be deit or primal" >&2; exit 2 ;;
esac
OUT="$WORK_DIR/smoke_in1k/$NAME"
rm -rf "$OUT" && mkdir -p "$OUT"   # a smoke test always starts from scratch
cd "$CV_DIR"

nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -l 5 > "$OUT/gpu_mem.txt" &
SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null || true' EXIT

start=$(date +%s)
torchrun --standalone --nproc_per_node=4 "${CMD[@]}" \
    --batch-size 256 --data-path "$IMAGENET1K_DIR" --data-set IMNET \
    --seed 0 --num_workers 15 --epochs "$EPOCHS" --output_dir "$OUT" 2>&1 | tee "$OUT/train_out.txt"
end=$(date +%s)

echo "=== smoke_in1k summary: $NAME ==="
echo "epochs logged: $(wc -l < "$OUT/log.txt") (expect $EPOCHS)"
grep -E "^Epoch: \[[0-9]+\] Total time|^Test: Total time" "$OUT/train_out.txt" || true
echo "wall time incl. startup: $((end - start)) s"
echo "peak GPU memory (MiB): $(sort -n "$OUT/gpu_mem.txt" | tail -1)"
[ "$MODEL" = primal ] && echo "J_log rows: $(($(wc -l < "$OUT/J_log.csv") - 1)), W_log rows: $(($(wc -l < "$OUT/W_log.csv") - 1)) (expect $((6 * (EPOCHS + 1))) each)"
# Estimate from the last epoch (warm cache): train + eval seconds x 300 epochs, at ~2,500 credits per minute.
python - "$OUT/train_out.txt" <<'EOF'
import re, sys
text = open(sys.argv[1]).read()
def secs(pattern):
    hits = re.findall(pattern + r" Total time: (\d+):(\d+):(\d+)", text)
    h, m, s = map(int, hits[-1]); return h * 3600 + m * 60 + s
train, test = secs(r"Epoch: \[\d+\]"), secs(r"Test:")
hours = 300 * (train + test) / 3600
print("last epoch: train {} s + eval {} s -> 300 epochs ~ {:.1f} h ~ {:.2f}M credits".format(train, test, hours, hours * 0.15))
EOF
