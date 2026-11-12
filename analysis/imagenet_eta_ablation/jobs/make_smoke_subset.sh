#!/bin/bash -l
# DRAFT — not tested on wICE. Builds a small view of ImageNet-100 for the smoke test from SYMLINKS
# (no image is copied): 130 train and 10 val images for each of the 100 classes, keeping the
# class-subfolder layout that datasets.py expects. Run once on a login node.
#
#   bash make_smoke_subset.sh
set -euo pipefail
source "$(dirname "$0")/common.sh"
require_data "$DATA_DIR"

# Overridable for a smaller local test: TRAIN_PER_CLASS=10 VAL_PER_CLASS=2 bash make_smoke_subset.sh
TRAIN_PER_CLASS="${TRAIN_PER_CLASS:-130}"
VAL_PER_CLASS="${VAL_PER_CLASS:-10}"
for split in train val; do
    per_class=$TRAIN_PER_CLASS
    [ "$split" = val ] && per_class=$VAL_PER_CLASS
    for class_dir in "$DATA_DIR/$split"/*/; do
        class=$(basename "$class_dir")
        mkdir -p "$SMOKE_DATA/$split/$class"
        # Process substitution, not a pipe into `while`: with pipefail, `head` closing early would make
        # `find` die of SIGPIPE and set -e would stop the script silently after the first class.
        mapfile -t files < <(find "$class_dir" -maxdepth 1 -type f | sort | head -n "$per_class")
        for file in "${files[@]}"; do
            ln -sf "$file" "$SMOKE_DATA/$split/$class/"
        done
    done
done
echo "classes: $(ls "$SMOKE_DATA/train" | wc -l) train, $(ls "$SMOKE_DATA/val" | wc -l) val"
echo "images:  $(find "$SMOKE_DATA/train" -type l | wc -l) train, $(find "$SMOKE_DATA/val" -type l | wc -l) val"
