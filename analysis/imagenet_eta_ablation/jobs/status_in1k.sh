#!/bin/bash -l
# Detailed status of the ImageNet-1K runs (train_in1k.slurm). Run on a wICE login node:
#   bash $VSC_DATA/PrimalAttention/analysis/imagenet_eta_ablation/jobs/status_in1k.sh
# Optional: pass job IDs to check others, e.g.  bash status_in1k.sh 62869857 62869858
JOBS="${*:-62869857 62869858 62869861}"
RUNS="/scratch/leuven/398/vsc39861/primal_fig1/runs_in1k"
JOBS_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== $(date '+%Y-%m-%d %H:%M') ==="
echo; echo "--- Slurm jobs"
sacct -M wice -j "${JOBS// /,}" -X --format=JobID,JobName%11,Account%13,State,Elapsed,Start,End,ExitCode,NodeList
echo; echo "--- Queue (START_TIME is Slurm's worst-case estimate)"
squeue -M wice -u "$USER" --start -o "%.10i %.11j %.8T %.20S %.12r"

echo; echo "--- Runs"
python3 - "$RUNS" <<'EOF'
import csv, json, os, sys, time
runs = sys.argv[1]
def load(name):
    path = os.path.join(runs, name, "log.txt")
    return [json.loads(l) for l in open(path)] if os.path.exists(path) else None
deit = load("deit_seed0")
for name in sorted(os.listdir(runs)) if os.path.isdir(runs) else []:
    log = load(name)
    print("\n[{}]".format(name))
    if not log:
        print("  no epochs logged yet"); continue
    last = log[-1]; e = last["epoch"]
    best = max(log, key=lambda x: x["test_acc1"])
    print("  epoch {}/300   top-1 {:.2f} %  top-5 {:.2f} %   best top-1 {:.2f} % (epoch {})".format(
        e + 1, last["test_acc1"], last["test_acc5"], best["test_acc1"], best["epoch"]))
    if "train_loss_total" in last:   # main_combo.py: total = CE + eta * KSVD term
        print("  train loss {:.4f} (CE {:.4f} + KSVD term {:.2e})   test loss {:.4f}   lr {:.2e}".format(
            last["train_loss_total"], last["train_loss_ce"], last["train_loss_ksvd"], last["test_loss"], last["train_lr"]))
    else:
        print("  train loss {:.4f}   test loss {:.4f}   lr {:.2e}".format(last["train_loss"], last["test_loss"], last["train_lr"]))
    bad = [x["epoch"] for x in log if any(v != v for v in x.values() if isinstance(v, float))]
    print("  NaN in log: {}".format("epochs " + str(bad) if bad else "none"))
    if deit and name != "deit_seed0" and e < len(deit):
        print("  DeiT at the same epoch: {:.2f} %  (difference {:+.2f})".format(deit[e]["test_acc1"], last["test_acc1"] - deit[e]["test_acc1"]))
    if e < 299:
        age = time.time() - os.path.getmtime(os.path.join(runs, name, "log.txt"))
        print("  last epoch logged {:.0f} min ago".format(age / 60))
    j = os.path.join(runs, name, "J_log.csv")
    if os.path.exists(j):
        rows = list(csv.DictReader(open(j)))
        rows = [r for r in rows if r["epoch"] == rows[-1]["epoch"]]
        if rows:
            print("  J (epoch {}): mean |J| {:.3f}   Jbar^2 {:.2e}   KKT residual {:.3f}   |lambda| max {:.1e}   lambda near 0 (of 96) per head: {}".format(
                rows[0]["epoch"], sum(float(r["J_abs_mean"]) for r in rows) / len(rows), float(rows[0]["Jbar_sq"]),
                sum(float(r["residual_aggregate"]) for r in rows) / len(rows),
                max(float(r["lambda_abs_max"]) for r in rows),
                " ".join(r["lambda_near_zero"] for r in rows)))
    w = os.path.join(runs, name, "W_log.csv")
    if os.path.exists(w):
        rows = list(csv.DictReader(open(w)))
        rows = [r for r in rows if r["epoch"] == rows[-1]["epoch"]]
        m = lambda k: sum(float(r[k]) for r in rows) / len(rows)
        print("  W (epoch {}, mean over heads): |W_e| {:.2f}  |W_r| {:.2f}  cos(W_e,W_r) {:+.3f}  eff. rank W_e {:.1f}  W_r {:.1f} (of 96)".format(
            rows[0]["epoch"], m("we_fro"), m("wr_fro"), m("cos_we_wr"), m("we_eff_rank"), m("wr_eff_rank")))
EOF

echo; echo "--- Speed and finish estimate (running jobs)"
for id in $JOBS; do
    out="$JOBS_DIR/train_in1k_${id}.out"
    [ -f "$out" ] || continue
    state=$(sacct -M wice -j "$id" -X -n -o State | head -1 | tr -d ' ')
    [ "$state" = RUNNING ] || continue
    python3 - "$out" <<'EOF'
import re, sys, datetime
text = open(sys.argv[1]).read()
times = re.findall(r"Epoch: \[(\d+)\] Total time: (\d+):(\d+):(\d+)", text)
tests = re.findall(r"Test: Total time: (\d+):(\d+):(\d+)", text)
if times:
    sec = lambda h, m, s: int(h) * 3600 + int(m) * 60 + int(s)
    recent = [sec(*t[1:]) for t in times[-10:]]
    test = [sec(*t) for t in tests[-10:]] or [0]
    per = sum(recent) / len(recent) + sum(test) / len(test)
    left = 299 - int(times[-1][0])
    done = datetime.datetime.now() + datetime.timedelta(seconds=left * per)
    print("{}: {:.0f} s/epoch (train+test, last 10), {} epochs left, finishes ~{}".format(
        sys.argv[1].split("/")[-1], per, left, done.strftime("%a %d %b %H:%M")))
EOF
done

echo; echo "--- Errors in Slurm output (last 3 per job)"
for id in $JOBS; do
    out="$JOBS_DIR/train_in1k_${id}.out"
    [ -f "$out" ] && { n=$(grep -ciE "error|traceback|loss is nan|out of memory" "$out"); echo "$id: $n matching lines"; grep -iE "error|traceback|loss is nan|out of memory" "$out" | tail -3; }
done

echo; echo "--- Credits"
sam-balance | grep -E "Name|lp_robustness|intro_vsc39861"
