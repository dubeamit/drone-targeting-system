#!/usr/bin/env python3
"""Repair broken image symlinks in data/visdrone_yolo after re-extracting VisDrone DET zips.

The YOLO dataset stores symlinks to data/VisDrone2019-DET-*/images/, not copies.
If those source folders were deleted, every symlink breaks even though `ls` still
shows the filenames.

Example:
    unzip data/zip/VisDrone2019-DET-train.zip -d data/
    unzip data/zip/VisDrone2019-DET-val.zip -d data/
    unzip data/zip/VisDrone2019-DET-test-dev.zip -d data/
    python repair_yolo_images.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

SPLITS = {
    "train": ("VisDrone2019-DET-train", "train"),
    "val": ("VisDrone2019-DET-val", "val"),
    "test": ("VisDrone2019-DET-test-dev", "test"),
}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def symlink_images(src_dir: Path, dst_dir: Path) -> tuple[int, int]:
    dst_dir.mkdir(parents=True, exist_ok=True)
    linked = 0
    repaired = 0

    for img in sorted(src_dir.iterdir()):
        if img.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        link = dst_dir / img.name
        target = img.resolve()
        if link.is_symlink() or link.exists():
            try:
                if link.is_symlink() and link.resolve() == target:
                    linked += 1
                    continue
            except (FileNotFoundError, OSError):
                pass
            link.unlink(missing_ok=True)
            repaired += 1
        else:
            repaired += 1
        link.symlink_to(target)
        linked += 1

    return linked, repaired


def count_broken(image_dir: Path) -> int:
    broken = 0
    for path in image_dir.glob("*"):
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        try:
            path.resolve(strict=True)
        except (FileNotFoundError, OSError):
            broken += 1
    return broken


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Repair visdrone_yolo image symlinks")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--yolo-dir", type=Path, default=Path("data/visdrone_yolo"))
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=list(SPLITS),
        default=list(SPLITS),
    )
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    yolo_dir = args.yolo_dir.resolve()

    print(f"YOLO images dir: {yolo_dir / 'images'}")
    exit_code = 0

    for split in args.splits:
        src_name, yolo_split = SPLITS[split]
        src_dir = data_dir / src_name / "images"
        dst_dir = yolo_dir / "images" / yolo_split

        if not src_dir.is_dir():
            print(f"[{split}] MISSING source images: {src_dir}")
            print(f"       Extract: data/zip/{src_name}.zip -> {data_dir}/")
            exit_code = 1
            continue

        if args.check_only:
            broken = count_broken(dst_dir) if dst_dir.is_dir() else 0
            print(f"[{split}] broken symlinks: {broken}")
            if broken:
                exit_code = 1
            continue

        linked, repaired = symlink_images(src_dir, dst_dir)
        broken = count_broken(dst_dir)
        print(f"[{split}] linked={linked}, repaired={repaired}, broken={broken}")

    if exit_code:
        print("\nRe-extract missing DET archives, then rerun this script.")
    else:
        print("\nImage symlinks look healthy.")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
