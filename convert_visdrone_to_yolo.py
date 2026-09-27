#!/usr/bin/env python3
"""Convert VisDrone DET annotations to Ultralytics YOLO layout.

Uses VisDrone-dataset-python-toolkit's convert_to_yolo converter and writes:

    <output_dir>/
      images/{train,val,test}/   # symlinks to original images
      labels/{train,val,test}/   # YOLO txt labels
      dataset.yaml

Example:
    python convert_visdrone_to_yolo.py
    python convert_visdrone_to_yolo.py --data-dir data --output-dir data/visdrone_yolo
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from types import ModuleType

import yaml

TOOLKIT_ROOT = Path(__file__).resolve().parent / "VisDrone-dataset-python-toolkit"


def _load_toolkit_module(relative_path: str, module_name: str) -> ModuleType:
    """Load a toolkit module without importing visdrone_toolkit.__init__ (needs torch>=2.2)."""
    path = TOOLKIT_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load toolkit module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_yolo_module = _load_toolkit_module(
    "visdrone_toolkit/converters/visdrone_to_yolo.py",
    "visdrone_to_yolo",
)
convert_to_yolo = _yolo_module.convert_to_yolo
validate_yolo_format = _yolo_module.validate_yolo_format

YOLO_CLASS_NAMES = [
    "pedestrian",
    "people",
    "bicycle",
    "car",
    "van",
    "truck",
    "tricycle",
    "awning-tricycle",
    "bus",
    "motor",
    "others",
]

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
SPLITS = {
    "train": "VisDrone2019-DET-train",
    "val": "VisDrone2019-DET-val",
    "test": "VisDrone2019-DET-test-dev",
}


def symlink_images(src_dir: Path, dst_dir: Path) -> int:
    """Create per-file symlinks so Ultralytics can resolve labels via path substitution."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for img in sorted(src_dir.iterdir()):
        if img.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        link = dst_dir / img.name
        if not link.exists():
            link.symlink_to(img.resolve())
        count += 1
    return count


def convert_split(
    split_name: str,
    split_dir: Path,
    output_dir: Path,
    *,
    filter_ignored: bool,
    filter_crowd: bool,
) -> int:
    image_dir = split_dir / "images"
    ann_dir = split_dir / "annotations"
    if not image_dir.is_dir() or not ann_dir.is_dir():
        raise FileNotFoundError(f"Missing images/ or annotations/ under {split_dir}")

    labels_dir = output_dir / "labels" / split_name
    images_dir = output_dir / "images" / split_name

    n_images = symlink_images(image_dir, images_dir)
    convert_to_yolo(
        image_dir=image_dir,
        annotation_dir=ann_dir,
        output_dir=labels_dir,
        filter_ignored=filter_ignored,
        filter_crowd=filter_crowd,
        create_yaml=False,
    )
    print(f"  {split_name}: {n_images} images, labels -> {labels_dir}")
    return n_images


def write_dataset_yaml(output_dir: Path, splits: list[str]) -> Path:
    dataset = {
        "path": str(output_dir.resolve()),
        "nc": len(YOLO_CLASS_NAMES),
        "names": YOLO_CLASS_NAMES,
    }
    for split in ("train", "val", "test"):
        if split in splits:
            dataset[split] = f"images/{split}"

    yaml_path = output_dir / "dataset.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(dataset, f, default_flow_style=False, sort_keys=False)
    return yaml_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert VisDrone DET dataset to YOLO format")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory containing VisDrone2019-DET-* folders (default: data)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/visdrone_yolo"),
        help="Output YOLO dataset directory (default: data/visdrone_yolo)",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=list(SPLITS),
        default=list(SPLITS),
        help="Dataset splits to convert (default: train val test)",
    )
    parser.add_argument(
        "--keep-ignored",
        action="store_true",
        help="Keep boxes with score=0 (default: filter out)",
    )
    parser.add_argument(
        "--keep-crowd",
        action="store_true",
        help="Keep crowd/ignored regions category=0 (default: filter out)",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate YOLO label files after conversion",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Converting VisDrone DET -> YOLO")
    print(f"  source:  {data_dir}")
    print(f"  output:  {output_dir}")
    print(f"  splits:  {', '.join(args.splits)}")
    print(f"  filter ignored boxes: {not args.keep_ignored}")
    print(f"  filter crowd regions: {not args.keep_crowd}")
    print()

    converted_splits: list[str] = []
    for split in args.splits:
        split_dir = data_dir / SPLITS[split]
        if not split_dir.is_dir():
            print(f"Skipping missing split: {split_dir}")
            continue
        convert_split(
            split,
            split_dir,
            output_dir,
            filter_ignored=not args.keep_ignored,
            filter_crowd=not args.keep_crowd,
        )
        converted_splits.append(split)

    if not converted_splits:
        print("No splits converted. Check --data-dir and downloaded dataset folders.")
        return 1

    yaml_path = write_dataset_yaml(output_dir, converted_splits)
    print(f"\nDataset YAML: {yaml_path}")
    print("Use this path in ultralytics training, e.g.:")
    print(f'  model.train(data="{yaml_path}", ...)')

    if args.validate:
        print("\nValidating YOLO labels...")
        ok = True
        for split in converted_splits:
            labels_dir = output_dir / "labels" / split
            print(f"\n--- {split} ---")
            ok = validate_yolo_format(labels_dir) and ok
        if not ok:
            return 1

    print("\nConversion complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
