#!/bin/bash -l
# DRAFT — not tested on wICE. One-time environment setup, run on a login node or the free
# `interactive` partition. Builds torch 2.4.1 (CUDA 12.1), torchvision 0.19.1 and timm 0.4.12 in VSC_DATA.
# The repo pins torch 1.7.1 (CUDA 11.0), which cannot run on H100 GPUs; 2.4.1 is the newest release for
# Python 3.8 and runs on H100 and V100. timm stays at the repo's 0.4.12. check_env.py verifies the stack
# against the training code. Needs internet access for conda and pip.
#
#   bash env_setup.sh
#
set -euo pipefail
source "$(dirname "$0")/common.sh"

if [ ! -x "$CONDA_ROOT/bin/conda" ]; then
    echo "Miniconda not found at $CONDA_ROOT. Install it there first (see the VSC docs or ask HPC support)." >&2
    exit 1
fi
source "$CONDA_ROOT/etc/profile.d/conda.sh"

if [ ! -d "$ENV_PREFIX" ]; then
    conda create -y -p "$ENV_PREFIX" --override-channels -c conda-forge python=3.8   # conda-forge only: no Anaconda ToS
fi
conda activate "$ENV_PREFIX"

pip install torch==2.4.1 torchvision==0.19.1
pip install timm==0.4.12 "numpy<1.24" pandas matplotlib

python - <<'EOF'
import torch, torchvision, timm, numpy
print("torch", torch.__version__, "| torchvision", torchvision.__version__, "| timm", timm.__version__, "| numpy", numpy.__version__)
print("CUDA build", torch.version.cuda, "| H100 support (sm_90):", "sm_90" in torch.cuda.get_arch_list(),
      "| GPU visible:", torch.cuda.is_available())
EOF
python "$REPO_DIR/analysis/imagenet_eta_ablation/check_env.py" --data "$SMOKE_DATA" || \
    echo "check_env.py needs the smoke subset: run make_smoke_subset.sh, then rerun this check."
