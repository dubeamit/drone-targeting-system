#!/usr/bin/env python3
"""Export YOLO labels to X-AnyLabeling JSON for review/editing.

Creates a split layout that X-AnyLabeling can open:

    <output-dir>/<split>/
      images/   *.jpg
      labels/   *.json
      classes.txt

Prelabel integration (tree/building drafts in labels_prelabel/):

    python prelabel_tree_building.py --split train --max-images 200 --export-anylabeling
    # or manually:
    python export_anylabeling.py --split train --include-prelabel --only-prelabeled

Example:
    python export_anylabeling.py --split train --max-images 100
    # Open in X-AnyLabeling: data/anylabeling/train
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

from anylabeling_utils import (
    IMAGE_SUFFIXES,
    build_anylabeling_json,
    image_size,
    load_class_names,
    parse_merged_yolo_labels,
    resolve_image,
    write_anylabeling_json,
)

SPLITS = ("train", "val", "test")


def list_export_images(
    image_dir: Path,
    max_images: int | None,
    seed: int,
    *,
    only_prelabeled: bool,
    prelabel_dir: Path | None,
) -> list[Path]:
    candidates = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    readable = [p for p in candidates if resolve_image(p) is not None]
    broken = len(candidates) - len(readable)
    if broken:
        print(
            f"Warning: skipped {broken} broken image symlink(s). "
            "Run: python repair_yolo_images.py"
        )
    if not readable:
        raise FileNotFoundError(f"No readable images in {image_dir}")

    if only_prelabeled:
        if prelabel_dir is None or not prelabel_dir.is_dir():
            raise FileNotFoundError(
                f"--only-prelabeled requires a prelabel dir. Expected: {prelabel_dir}"
            )
        prelabel_stems = {p.stem for p in prelabel_dir.glob("*.txt")}
        readable = [p for p in readable if p.stem in prelabel_stems]
        if not readable:
            raise FileNotFoundError(
                f"No images match prelabel files in {prelabel_dir}. "
                "Run prelabel_tree_building.py first."
            )

    if max_images is None or max_images >= len(readable):
        return readable
    rng = random.Random(seed)
    return rng.sample(readable, max_images)


def link_or_copy_image(src: Path, dst: Path, copy: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy:
        shutil.copy2(src, dst)
    else:
        dst.symlink_to(src.resolve())


def export_split(
    yolo_dir: Path,
    output_dir: Path,
    split: str,
    class_names: list[str],
    *,
    max_images: int | None,
    seed: int,
    copy_images: bool,
    include_prelabel: bool,
    only_prelabeled: bool,
    prelabel_dir: Path | None,
) -> int:
    image_dir = yolo_dir / "images" / split
    label_dir = yolo_dir / "labels" / split
    split_prelabel_dir = prelabel_dir or (yolo_dir / "labels_prelabel" / split)
    out_split = output_dir / split
    out_images = out_split / "images"
    out_labels = out_split / "labels"

    images = list_export_images(
        image_dir,
        max_images,
        seed,
        only_prelabeled=only_prelabeled,
        prelabel_dir=split_prelabel_dir if include_prelabel or only_prelabeled else None,
    )
    out_split.mkdir(parents=True, exist_ok=True)
    (out_split / "classes.txt").write_text("\n".join(class_names) + "\n")

    exported = 0
    for image_link in images:
        readable = resolve_image(image_link)
        if readable is None:
            continue

        out_image = out_images / readable.name
        link_or_copy_image(readable, out_image, copy_images)

        img_width, img_height = image_size(readable)
        yolo_label = label_dir / f"{readable.stem}.txt"
        prelabel_path = split_prelabel_dir / f"{readable.stem}.txt" if include_prelabel else None
        shapes = parse_merged_yolo_labels(
            yolo_label,
            prelabel_path,
            class_names,
            img_width,
            img_height,
        )
        payload = build_anylabeling_json(shapes, readable.name, img_width, img_height)
        write_anylabeling_json(out_labels / f"{readable.stem}.json", payload)
        exported += 1

    return exported


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export YOLO dataset to X-AnyLabeling JSON")
    parser.add_argument("--yolo-dir", type=Path, default=Path("data/visdrone_yolo"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/anylabeling"))
    parser.add_argument(
        "--dataset-yaml",
        type=Path,
        default=Path("data/visdrone_yolo/dataset_13cls.yaml"),
        help="Class names source",
    )
    parser.add_argument("--split", choices=SPLITS, default="train")
    parser.add_argument("--splits", nargs="+", choices=SPLITS, help="Export multiple splits")
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--copy-images",
        action="store_true",
        help="Copy images instead of symlinking (slower, but fully self-contained)",
    )
    parser.add_argument(
        "--include-prelabel",
        action="store_true",
        help="Merge labels_prelabel/<split>/ tree+building drafts into exported JSON",
    )
    parser.add_argument(
        "--only-prelabeled",
        action="store_true",
        help="Export only images that have a labels_prelabel/<split>/*.txt file",
    )
    parser.add_argument(
        "--prelabel-dir",
        type=Path,
        default=None,
        help="Override prelabel directory (default: <yolo-dir>/labels_prelabel/<split>)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    yolo_dir = args.yolo_dir.resolve()
    output_dir = args.output_dir.resolve()
    class_names = load_class_names(args.dataset_yaml.resolve())
    splits = args.splits or [args.split]

    if args.only_prelabeled and not args.include_prelabel:
        args.include_prelabel = True

    print(f"Exporting to X-AnyLabeling format: {output_dir}")
    print(f"Classes ({len(class_names)}): {', '.join(class_names)}")
    if args.include_prelabel:
        print("Including tree/building drafts from labels_prelabel/")
    if args.only_prelabeled:
        print("Limiting export to prelabeled images only")

    total = 0
    for split in splits:
        prelabel_dir = args.prelabel_dir.resolve() if args.prelabel_dir else None
        count = export_split(
            yolo_dir,
            output_dir,
            split,
            class_names,
            max_images=args.max_images,
            seed=args.seed,
            copy_images=args.copy_images,
            include_prelabel=args.include_prelabel,
            only_prelabeled=args.only_prelabeled,
            prelabel_dir=prelabel_dir,
        )
        print(f"  {split}: {count} image/json pairs -> {output_dir / split}")
        total += count

    print("\nOpen in X-AnyLabeling:")
    print(f"  Folder: {output_dir / splits[0]}")
    print("  Set image folder to: images/")
    print("  Labels should load from: labels/*.json")
    print("\nAfter editing, import back with:")
    print("  python import_anylabeling.py --split train")
    print(f"Exported {total} files total.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
