#!/bin/bash -l
# Downloads ImageNet-1K (ILSVRC 2012) from Hugging Face (ILSVRC/imagenet-1k: train and validation parquet shards,
# about 154 GB; the unlabelled test shards are skipped) and unpacks it into $IMAGENET1K_DIR with
# convert_imagenet1k.py: train/<label>/ (1,281,167 images) and val/<label>/ (50,000), 1000 classes.
# CPU only (no GPU is needed for this). Safe to resubmit: the download and the conversion skip finished files.
#
# Before submitting, once: accept the conditions on https://huggingface.co/datasets/ILSVRC/imagenet-1k
# and run `$VSC_DATA/miniconda3/bin/hf auth login` on a login node (read token).
#   cd analysis/imagenet_eta_ablation/jobs && sbatch --export=ALL get_imagenet1k.slurm
#
#SBATCH --clusters=wice
#SBATCH --partition=batch
#SBATCH --account=intro_vsc39861
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --job-name=get_imagenet1k
#SBATCH --output=%x_%j.out
set -euo pipefail
source "${SLURM_SUBMIT_DIR}/common.sh"
PY="$CONDA_ROOT/bin/python"      # base Miniconda: has huggingface_hub and pyarrow (installed for ImageNet-100)
RAW="$VSC_SCRATCH/imagenet1k_parquet"
export HF_HUB_CACHE="$VSC_SCRATCH/hf_cache"   # keep any download cache off $VSC_HOME (3 GB); the token stays in ~/.cache

"$CONDA_ROOT/bin/hf" download ILSVRC/imagenet-1k --repo-type dataset --local-dir "$RAW" \
    --include "data/train-*" --include "data/validation-*" --max-workers 8   # hf 2.x: one pattern per --include
df -h "$VSC_SCRATCH" | tail -1

"$PY" "$REPO_DIR/analysis/imagenet_eta_ablation/convert_imagenet1k.py" --parquet_dir "$RAW/data" --out "$IMAGENET1K_DIR"
# The converter exits non-zero unless every count matches, so the parquet copy is only removed after a full check.
rm -rf "$RAW"
echo "ImageNet-1K ready in $IMAGENET1K_DIR; parquet download removed."
