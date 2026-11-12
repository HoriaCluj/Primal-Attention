# DRAFT — not tested on wICE. Shared paths for every job script; edit the placeholders once.
# Sourced by the scripts in this folder. Submit jobs from this folder (analysis/imagenet_eta_ablation/jobs), so that
# ${SLURM_SUBMIT_DIR}/common.sh resolves.

export REPO_DIR="${REPO_DIR:-$VSC_DATA/PrimalAttention}"          # clone of the repository
export CV_DIR="$REPO_DIR/PrimalAttention_CV"
export JOBS_DIR="$REPO_DIR/analysis/imagenet_eta_ablation/jobs"
export J_LOGGER="$REPO_DIR/analysis/imagenet_eta_ablation/train_with_j_log.py"  # per-head J log wrapper
export DATA_DIR="${DATA_DIR:-$VSC_SCRATCH/imagenet-100}"           # ImageNet-100: train/ and val/, one subfolder per class
export IMAGENET1K_DIR="${IMAGENET1K_DIR:-<IMAGENET_1K_PATH>}"      # only for S1 on wICE (deit_official_spectra.slurm)
export SMOKE_DATA="${SMOKE_DATA:-$VSC_SCRATCH/imagenet_smoke}"     # symlinked subset, see make_smoke_subset.sh
export WORK_DIR="${WORK_DIR:-$VSC_SCRATCH/primal_fig1}"
export RUNS_DIR="$WORK_DIR/runs"
export SAVE_DIR="${SAVE_DIR:-$VSC_DATA/primal_fig1}"              # backed-up copy of results
export ENV_PREFIX="${ENV_PREFIX:-$VSC_DATA/envs/primal-cv}"
export CONDA_ROOT="${CONDA_ROOT:-$VSC_DATA/miniconda3}"
export DEIT_S_URL="https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth"  # models.py:85

activate_env() {
    # shellcheck disable=SC1091
    source "$CONDA_ROOT/etc/profile.d/conda.sh"
    conda activate "$ENV_PREFIX"
}

require_data() {
    case "$1" in
        "<"*|"") echo "Set DATA_DIR in common.sh (or export it) before submitting." >&2; exit 2 ;;
    esac
    [ -d "$1/train" ] && [ -d "$1/val" ] || { echo "Expected $1/train and $1/val" >&2; exit 2; }
}
