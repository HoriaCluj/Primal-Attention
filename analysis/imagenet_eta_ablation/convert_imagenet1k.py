"""Convert the ILSVRC/imagenet-1k parquet files (Hugging Face) into the folder layout the training code reads.

PrimalAttention_CV/datasets.py:61-64 loads ImageNet-1K with torchvision's ImageFolder from <data-path>/train and
<data-path>/val. The dataset card says the labels are indexed by the sorted list of synset ids, and ImageFolder numbers
classes by sorted folder name, so folders named by zero-padded label index (0000 ... 0999) give the same class
indices as the usual train/<wnid>/ layout. Images are written with their original bytes (no decoding, no re-encoding).

Checks: every shard present, 1000 classes, 1,281,167 train / 50,000 val images, files on disk = rows in parquet.
When train file names carry a synset id (n01440764_...), it also checks that each label maps to one synset and that
the label order is the sorted synset order, and writes label -> synset to <out>/synsets.txt.
Safe to re-run: files already written with the right size are skipped; writes go through a temporary name.

Needs only pyarrow:
    python convert_imagenet1k.py --parquet_dir <download>/data --out <imagenet-1k>
"""
import argparse
import collections
import glob
import json
import os
import re
import sys

import pyarrow.parquet as pq

SPLITS = {"train": "train", "val": "validation"}   # output folder -> Hugging Face split name
EXPECTED = {"train": 1_281_167, "val": 50_000}
SAFE_NAME = re.compile(r"^[A-Za-z0-9_.-]+\.(JPEG|jpeg|jpg|JPG|png|PNG)$")
SYNSET = re.compile(r"^(n\d{8})_")


def convert_split(split, parquet_dir, out_dir, synsets):
    hf_split = SPLITS[split]
    files = sorted(glob.glob(os.path.join(parquet_dir, "{}-*.parquet".format(hf_split))))
    if not files:
        raise SystemExit("No {}-*.parquet files in {}".format(hf_split, parquet_dir))
    expected_files = int(re.search(r"-of-(\d+)\.parquet$", files[0]).group(1))
    if len(files) != expected_files:
        raise SystemExit("{}: found {} of {} files".format(hf_split, len(files), expected_files))

    for label in range(1000):
        os.makedirs(os.path.join(out_dir, split, "{:04d}".format(label)), exist_ok=True)
    written = skipped = row = 0
    per_class = collections.Counter()
    for number, path in enumerate(files, 1):
        parquet = pq.ParquetFile(path)
        for group in range(parquet.num_row_groups):
            table = parquet.read_row_group(group, columns=["image", "label"])
            for image, label in zip(table.column("image").to_pylist(), table.column("label").to_pylist()):
                row += 1
                if not isinstance(label, int) or not 0 <= label < 1000:
                    raise SystemExit("Label {!r} out of range in {}".format(label, path))
                # The name comes from a downloaded file: keep it only if it is a plain file name.
                name = os.path.basename(image.get("path") or "")
                if not SAFE_NAME.match(name):
                    name = "{}_{:08d}.JPEG".format(split, row)
                match = SYNSET.match(name)
                if split == "train" and match:
                    if synsets.setdefault(label, match.group(1)) != match.group(1):
                        raise SystemExit("Label {} has synsets {} and {}".format(label, synsets[label], match.group(1)))
                target = os.path.join(out_dir, split, "{:04d}".format(label), name)
                data = image["bytes"]
                per_class[label] += 1
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
    if total != EXPECTED[split] or len(per_class) != 1000:
        raise SystemExit("{}: {} images in {} classes; expected {} in 1000".format(
            split, total, len(per_class), EXPECTED[split]))
    on_disk = sum(len(os.listdir(os.path.join(out_dir, split, "{:04d}".format(c)))) for c in range(1000))
    if on_disk != total:
        raise SystemExit("{}: {} files on disk, {} in parquet (duplicate names, or stale files?)".format(
            split, on_disk, total))
    return {"images": total, "classes": len(per_class), "per_class_min": min(per_class.values()),
            "per_class_max": max(per_class.values()), "written": written, "already_present": skipped}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--parquet_dir", required=True, help="the downloaded repository's data/ folder")
    parser.add_argument("--out", required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "val"], choices=list(SPLITS))
    args = parser.parse_args()

    synsets = {}
    summary = {split: convert_split(split, args.parquet_dir, args.out, synsets) for split in args.splits}
    if synsets:
        if len(synsets) != 1000 or [synsets[c] for c in range(1000)] != sorted(synsets.values()):
            raise SystemExit("Train labels are not in sorted synset order: class indices would not match ImageNet's")
        with open(os.path.join(args.out, "synsets.txt"), "w") as handle:
            handle.write("\n".join("{:04d} {}".format(c, synsets[c]) for c in range(1000)) + "\n")
    summary["label_order_checked_against_synsets"] = len(synsets) == 1000
    with open(os.path.join(args.out, "conversion_summary.json"), "w") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    sys.exit(main())
