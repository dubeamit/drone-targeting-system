#!/usr/bin/env python3
"""Import X-AnyLabeling JSON edits back into YOLO txt labels.

Reads:
    data/anylabeling/<split>/
      images/
      labels/*.json

Writes:
    data/visdrone_yolo/labels/<split>/*.txt

Example:
    python import_anylabeling.py --split train
    python verify_yolo_annotations.py --split train --num-samples 5
"""

from __future__ import annotations

import argparse
from pathlib import Path

from anylabeling_utils import IMAGE_SUFFIXES, load_class_names, parse_anylabeling_json, resolve_image


SPLITS = ("train", "val", "test")


def import_split(
    anylabeling_dir: Path,
    yolo_dir: Path,
    split: str,
    class_names: list[str],
    *,
    dry_run: bool,
) -> tuple[int, int]:
    split_dir = anylabeling_dir / split
    image_dir = split_dir / "images"
    json_dir = split_dir / "labels"
    out_label_dir = yolo_dir / "labels" / split

    if not json_dir.is_dir():
        raise FileNotFoundError(f"Missing labels dir: {json_dir}")

    imported = 0
    skipped = 0
    for json_path in sorted(json_dir.glob("*.json")):
        image_path = image_dir / f"{json_path.stem}.jpg"
        if not image_path.exists():
            for suffix in IMAGE_SUFFIXES:
                candidate = image_dir / f"{json_path.stem}{suffix}"
                if candidate.exists():
                    image_path = candidate
                    break

        if resolve_image(image_path) is None:
            skipped += 1
            continue

        lines, _, _ = parse_anylabeling_json(json_path, class_names)
        out_path = out_label_dir / f"{json_path.stem}.txt"
        if not dry_run:
            out_label_dir.mkdir(parents=True, exist_ok=True)
            out_path.write_text("\n".join(lines) + ("\n" if lines else ""))
        imported += 1

    return imported, skipped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import X-AnyLabeling JSON back to YOLO txt")
    parser.add_argument("--anylabeling-dir", type=Path, default=Path("data/anylabeling"))
    parser.add_argument("--yolo-dir", type=Path, default=Path("data/visdrone_yolo"))
    parser.add_argument(
        "--dataset-yaml",
        type=Path,
        default=Path("data/visdrone_yolo/dataset_13cls.yaml"),
    )
    parser.add_argument("--split", choices=SPLITS, default="train")
    parser.add_argument("--splits", nargs="+", choices=SPLITS, help="Import multiple splits")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    class_names = load_class_names(args.dataset_yaml.resolve())
    splits = args.splits or [args.split]

    print(f"Importing X-AnyLabeling JSON -> YOLO txt")
    print(f"Source: {args.anylabeling_dir.resolve()}")
    print(f"Target: {args.yolo_dir.resolve() / 'labels'}")
    if args.dry_run:
        print("DRY RUN: no files will be written")

    total_imported = 0
    total_skipped = 0
    for split in splits:
        imported, skipped = import_split(
            args.anylabeling_dir.resolve(),
            args.yolo_dir.resolve(),
            split,
            class_names,
            dry_run=args.dry_run,
        )
        print(f"  {split}: imported={imported}, skipped={skipped}")
        total_imported += imported
        total_skipped += skipped

    print(f"\nDone. {total_imported} label files updated.")
    if total_skipped:
        print(f"Skipped {total_skipped} json files with missing images.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
