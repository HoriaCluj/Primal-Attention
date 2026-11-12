"""Convert the Yunzhe/ImageNet-100 parquet files into the folder layout the training code reads.

PrimalAttention_CV/datasets.py:66-69 loads ImageNet-100 with torchvision's ImageFolder from
<data-path>/train and <data-path>/val, one subfolder per class. This script writes

    <out>/train/<wnid>/<original ImageNet file name>
    <out>/val/<wnid>/<original ImageNet file name>

using the original JPEG bytes stored in the parquet files, unchanged (no decoding, no re-encoding).
ImageFolder numbers classes by sorted folder name; the parquet labels are also sorted WordNet IDs,
so the class indices are the same.

Safe to re-run: files already written with the right size are skipped. Each file is written to a
temporary name first and renamed, so an interrupted run never leaves a half-written image.

Needs only pyarrow:
    uv run --with pyarrow python analysis/imagenet_eta_ablation/convert_imagenet100.py \
        --parquet_dir data/data --out data/imagenet-100
"""
import argparse
import collections
import glob
import json
import os
import re
import sys

import pyarrow.parquet as pq

WNID = re.compile(r"^n\d{8}$")
FILE_NAME = {
    "train": re.compile(r"^(n\d{8})_\d+\.JPEG$"),
    "val": re.compile(r"^ILSVRC2012_val_\d{8}\.JPEG$"),
}
EXPECTED = {"train": 126_689, "val": 5_000}


def label_names(parquet_file):
    info = json.loads(parquet_file.schema_arrow.metadata[b"huggingface"])
    names = info["info"]["features"]["label"]["names"]
    if len(names) != 100 or not all(WNID.match(n) for n in names) or names != sorted(names):
        raise SystemExit("Unexpected label names in {}".format(parquet_file))
    return names


def convert_split(split, parquet_dir, out_dir):
    files = sorted(glob.glob(os.path.join(parquet_dir, "{}-*.parquet".format(split))))
    if not files:
        raise SystemExit("No {}-*.parquet files in {}".format(split, parquet_dir))
    expected_files = int(re.search(r"-of-(\d+)\.parquet$", files[0]).group(1))
    if len(files) != expected_files:
        raise SystemExit("{}: found {} of {} files".format(split, len(files), expected_files))

    names = None
    written = skipped = 0
    per_class = collections.Counter()
    for number, path in enumerate(files, 1):
        parquet = pq.ParquetFile(path)
        file_names = label_names(parquet)
        if names is None:
            names = file_names
            for wnid in names:
                os.makedirs(os.path.join(out_dir, split, wnid), exist_ok=True)
        elif file_names != names:
            raise SystemExit("Label names differ in {}".format(path))

        for group in range(parquet.num_row_groups):
            table = parquet.read_row_group(group, columns=["image", "label"])
            for image, label in zip(table.column("image").to_pylist(), table.column("label").to_pylist()):
                wnid = names[label]
                # The path comes from a downloaded file: accept only plain ImageNet file names.
                file_name = image.get("path") or ""
                match = FILE_NAME[split].match(file_name)
                if not match:
                    raise SystemExit("Unexpected file name {!r} in {}".format(file_name, path))
                if split == "train" and match.group(1) != wnid:
                    raise SystemExit("{} is labelled {} in {}".format(file_name, wnid, path))

                target = os.path.join(out_dir, split, wnid, file_name)
                data = image["bytes"]
                per_class[wnid] += 1
                if os.path.exists(target) and os.path.getsize(target) == len(data):
                    skipped += 1
                    continue
                temporary = target + ".part"
                with open(temporary, "wb") as handle:
                    handle.write(data)
                os.replace(temporary, target)
                written += 1
        print("{} {}/{}: {} ({} written, {} already present)".format(
            split, number, len(files), os.path.basename(path), written, skipped), flush=True)

    total = sum(per_class.values())
    if total != EXPECTED[split] or len(per_class) != 100:
        raise SystemExit("{}: {} images in {} classes; expected {} in 100".format(
            split, total, len(per_class), EXPECTED[split]))
    on_disk = sum(len(os.listdir(os.path.join(out_dir, split, wnid))) for wnid in names)
    if on_disk != total:
        raise SystemExit("{}: {} files on disk, {} in parquet (stale or extra files?)".format(split, on_disk, total))
    return {"images": total, "classes": len(per_class), "per_class_min": min(per_class.values()),
            "per_class_max": max(per_class.values()), "written": written, "already_present": skipped}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--parquet_dir", default="data/data")
    parser.add_argument("--out", default="data/imagenet-100")
    parser.add_argument("--splits", nargs="+", default=["val", "train"], choices=["train", "val"])
    args = parser.parse_args()

    summary = {split: convert_split(split, args.parquet_dir, args.out) for split in args.splits}
    with open(os.path.join(args.out, "conversion_summary.json"), "w") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    sys.exit(main())
