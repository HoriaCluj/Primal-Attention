#!/usr/bin/env bash
# Local pre-flight test of the ImageNet-100 training code, before any wICE credits are spent.
# Runs the repository's own scripts (PrimalAttention_CV/main_combo.py and main.py) unchanged, on a tiny
# symlinked ImageNet-100 subset, in the cluster's environment (torch 2.4.1, timm 0.4.12; see jobs/env_setup.sh).
#
#   bash analysis/imagenet_eta_ablation/local_test.sh
#
# Steps: (A) Primal.+ eta=0.05, 2 epochs  (B) resume that run for a 3rd epoch  (C) Primal.+ eta=0, 1 epoch
#        (D) DeiT-S baseline (main.py), 1 epoch  (E) Primal.+ eta=0.05, 1 epoch, launched with torchrun as on
#        wICE (distributed code path, DDP wrapper; one GPU here).  The Primal.+ steps go through
#        train_with_j_log.py, which also writes J_log.csv (per-head J and KKT residuals) next to log.txt.
# Overridable: PY (python of the pinned env), BATCH (default 8, sized for a 2 GB GPU), WORKERS, OUT.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PY:-/home/horia/envs/primal-cv-h100/bin/python}"
DATA="$ROOT/data/imagenet-100-smoke"
OUT="${OUT:-$ROOT/data/local_test_runs}"    # inside the git-ignored data/ folder
BATCH="${BATCH:-8}"
WORKERS="${WORKERS:-4}"

[ -x "$PY" ] || { echo "Python not found: $PY" >&2; exit 2; }
[ -d "$DATA/train" ] && [ -d "$DATA/val" ] || { echo "Missing $DATA (build it with jobs/make_smoke_subset.sh)" >&2; exit 2; }
rm -rf "$OUT"; mkdir -p "$OUT"
cd "$ROOT/PrimalAttention_CV"

nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -l 2 > "$OUT/gpu_mem.txt" 2>/dev/null &
SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null || true' EXIT

# The flags below are the README's ImageNet-100 command; only batch size, epochs and workers are
# reduced to fit this laptop. Single process, so init_distributed_mode() takes the non-distributed path.
export J_LOG_IMAGES=16 J_LOG_BATCH=4   # small, for the 2 GB laptop GPU (cluster default: 32 images)
PRIMAL=("$ROOT/analysis/imagenet_eta_ablation/train_with_j_log.py" main_combo.py --model primal_small_patch16_224 --data-path "$DATA" --data-set IMNET-100
        --low-rank 96 --rank-multi 10 --num-ksvd-layer 1 --batch-size "$BATCH" --num_workers "$WORKERS" --seed 0)

failed=0
run() {   # run <name> <command...>: log to $OUT/<name>.log, report time and status
    local name="$1"; shift
    local start=$SECONDS
    if "$@" > "$OUT/$name.log" 2>&1; then status=ok; else status="FAILED (exit $?)"; failed=1; fi
    echo "[$name] $status in $((SECONDS - start)) s -> $OUT/$name.log"
}

run A_primal_eta0.05_2ep "$PY" "${PRIMAL[@]}" --eta 0.05 --epochs 2 --output_dir "$OUT/primal_eta0.05"
run B_resume_3rd_epoch   "$PY" "${PRIMAL[@]}" --eta 0.05 --epochs 3 --resume "$OUT/primal_eta0.05/checkpoint.pth" --output_dir "$OUT/primal_eta0.05"
run C_primal_eta0_1ep    "$PY" "${PRIMAL[@]}" --eta 0 --epochs 1 --output_dir "$OUT/primal_eta0"
run D_deit_1ep           "$PY" main.py --model deit_small_patch16_224 --data-path "$DATA" --data-set IMNET-100 \
                              --batch-size "$BATCH" --num_workers "$WORKERS" --seed 0 --epochs 1 --output_dir "$OUT/deit"
if "$PY" -c "import torch.distributed.run" 2>/dev/null; then LAUNCH=("$PY" -m torch.distributed.run --standalone --nproc_per_node=1)
else LAUNCH=("$PY" -m torch.distributed.launch --nproc_per_node=1 --use_env); fi   # torch < 1.10
run E_primal_torchrun_1ep "${LAUNCH[@]}" "${PRIMAL[@]}" --eta 0.05 --epochs 1 --output_dir "$OUT/primal_torchrun"

echo "peak GPU memory sampled (MiB): $(sort -n "$OUT/gpu_mem.txt" | tail -1)"
for run in primal_eta0.05 primal_eta0 primal_torchrun; do
    rows=$(( $(wc -l < "$OUT/$run/J_log.csv" 2>/dev/null || echo 1) - 1 ))
    echo "$run: J_log.csv has $rows rows (expect $([ $run = primal_eta0.05 ] && echo 24 || echo 12): 6 heads x [before training + epochs])"
done
exit $failed
