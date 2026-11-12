#!/usr/bin/env bash
# Runs the eta sweep, then the analysis and plots. Safe to re-run: finished runs are skipped.
#
#   bash analysis/timeseries_eta_ablation/run_sweep.sh
#
# Optional overrides (for testing or a reduced schedule):
#   ETAS="0 0.1" SEEDS="1" EPOCHS=2 RUNS_DIR=/tmp/x OUT_DIR=/tmp/y bash analysis/timeseries_eta_ablation/run_sweep.sh
set -uo pipefail

ANALYSIS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$(dirname "$ANALYSIS")")"
PY="$ROOT/PrimalAttention_TimeSeries/Primal.+Trans./.venv/bin/python"

ETAS="${ETAS:-0 0.1 0.2 0.5}"
SEEDS="${SEEDS:-1 2 3}"
EPOCHS="${EPOCHS:-100}"
DATASET="${DATASET:-UWaveGestureLibrary}"
RUNS_DIR="${RUNS_DIR:-$ANALYSIS/runs}"
OUT_DIR="${OUT_DIR:-$ANALYSIS}"

mkdir -p "$RUNS_DIR/logs"
[ -e "$RUNS_DIR/.gitignore" ] || echo '*' > "$RUNS_DIR/.gitignore"

total=$(( $(echo $ETAS | wc -w) * $(echo $SEEDS | wc -w) ))
count=0
for seed in $SEEDS; do          # seeds on the outside, so a partial sweep stays balanced across eta
  for eta in $ETAS; do
    count=$((count + 1))
    log="$RUNS_DIR/logs/${DATASET}_eta${eta}_seed${seed}.txt"
    echo "[$(date +%H:%M:%S)] run $count/$total: eta=$eta seed=$seed (log: $log)"
    if ! "$PY" "$ANALYSIS/train_run.py" --dataset "$DATASET" --eta "$eta" --seed "$seed" \
        --epochs "$EPOCHS" --runs_dir "$RUNS_DIR" > "$log" 2>&1; then
      echo "FAILED: eta=$eta seed=$seed. Last lines of $log:"
      tail -n 15 "$log"
      echo "Fix the problem and re-run this script; finished runs are skipped."
      exit 1
    fi
    grep -E "^SUMMARY|already finished" "$log" | cut -c1-260
  done
done

echo "[$(date +%H:%M:%S)] training finished; computing spectra"
"$PY" "$ANALYSIS/run_analysis.py" --runs_dir "$RUNS_DIR" --out_dir "$OUT_DIR/results" || exit 1
echo "[$(date +%H:%M:%S)] making figures"
"$PY" "$ANALYSIS/plots.py" --results_dir "$OUT_DIR/results" --figures_dir "$OUT_DIR/figures" || exit 1
echo "[$(date +%H:%M:%S)] done. Results: $OUT_DIR/results  Figures: $OUT_DIR/figures"
