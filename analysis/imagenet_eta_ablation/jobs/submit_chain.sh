#!/bin/bash -l
# DRAFT — not tested on wICE. Queues, for each eta, a chain of N_JOBS train_primal.slurm jobs linked
# with --dependency=afterany, so each job resumes where the previous one stopped at the 3-day
# limit. Jobs that start after the run has finished exit immediately (and cost almost nothing).
#
# Prints the sbatch commands only, unless SUBMIT=1 is set:
#   cd analysis/imagenet_eta_ablation/jobs && ETAS="0.05" N_JOBS=2 bash submit_chain.sh            # dry run (default)
#   cd analysis/imagenet_eta_ablation/jobs && ETAS="0.05" N_JOBS=2 SUBMIT=1 bash submit_chain.sh   # submit S2
#   cd analysis/imagenet_eta_ablation/jobs && ETAS="0 0.5" N_JOBS=2 SUBMIT=1 bash submit_chain.sh  # S3, after S2 passes
#
# Choose N_JOBS from the smoke test: ceil(estimated wall hours / --time hours in train_primal.slurm) + 1 spare.
set -euo pipefail
cd "$(dirname "$0")"

ETAS="${ETAS:-0.05}"
SEED="${SEED:-0}"
N_JOBS="${N_JOBS:-2}"
SUBMIT="${SUBMIT:-0}"

for eta in $ETAS; do
    previous=""
    for i in $(seq 1 "$N_JOBS"); do
        dependency=()
        [ -n "$previous" ] && dependency=(--dependency="afterany:$previous")
        command=(sbatch --parsable "${dependency[@]}" --export="ALL,ETA=$eta,SEED=$SEED" train_primal.slurm)
        if [ "$SUBMIT" = 1 ]; then
            previous=$("${command[@]}" | cut -d';' -f1)   # --parsable prints "jobid;cluster"
            echo "eta=$eta job $i/$N_JOBS -> $previous"
        else
            echo "[dry run] ${command[*]}"
            previous="<job$i>"
        fi
    done
done
