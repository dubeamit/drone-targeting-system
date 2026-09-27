#!/usr/bin/env python3
"""Visually verify VisDrone -> YOLO annotation conversion.

Draws original VisDrone boxes and converted YOLO boxes side by side (or overlaid)
so you can confirm the conversion is correct.

Example:
    python verify_yolo_annotations.py --split train --num-samples 5
    python verify_yolo_annotations.py --image data/VisDrone2019-DET-train/images/0000002_00005_d_0000014.jpg
"""

from __future__ import annotations

import argparse
import importlib.util
import random
from pathlib import Path
from types import ModuleType

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

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


_viz_module = _load_toolkit_module("visdrone_toolkit/visualization.py", "visdrone_visualization")
CLASS_COLORS = _viz_module.CLASS_COLORS
CLASS_NAMES = _viz_module.CLASS_NAMES

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

SPLITS = {
    "train": "VisDrone2019-DET-train",
    "val": "VisDrone2019-DET-val",
    "test": "VisDrone2019-DET-test-dev",
}

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def parse_visdrone_annotations(
    ann_path: Path,
    img_width: int,
    img_height: int,
    *,
    filter_ignored: bool,
    filter_crowd: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Parse VisDrone txt into xyxy boxes and native class ids."""
    boxes: list[list[float]] = []
    labels: list[int] = []

    if not ann_path.exists():
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int64)

    with open(ann_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 6:
                continue

            x, y, w, h = map(int, parts[:4])
            score = int(parts[4])
            category = int(parts[5])

            if filter_ignored and score == 0:
                continue
            if filter_crowd and category == 0:
                continue
            if w <= 0 or h <= 0:
                continue

            boxes.append([x, y, x + w, y + h])
            labels.append(category)

    return np.array(boxes, dtype=np.float32), np.array(labels, dtype=np.int64)


def parse_yolo_annotations(
    ann_path: Path,
    img_width: int,
    img_height: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Parse YOLO txt into xyxy boxes and YOLO class ids."""
    boxes: list[list[float]] = []
    labels: list[int] = []

    if not ann_path.exists():
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int64)

    with open(ann_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                continue

            class_id = int(parts[0])
            x_center, y_center, width, height = map(float, parts[1:])
            x1 = (x_center - width / 2) * img_width
            y1 = (y_center - height / 2) * img_height
            x2 = (x_center + width / 2) * img_width
            y2 = (y_center + height / 2) * img_height
            boxes.append([x1, y1, x2, y2])
            labels.append(class_id)

    return np.array(boxes, dtype=np.float32), np.array(labels, dtype=np.int64)


def class_name_for_visdrone(label: int) -> str:
    if 0 <= label < len(CLASS_NAMES):
        return CLASS_NAMES[label]
    return f"class_{label}"


def class_name_for_yolo(label: int) -> str:
    if 0 <= label < len(YOLO_CLASS_NAMES):
        return YOLO_CLASS_NAMES[label]
    return f"class_{label}"


def color_for_visdrone(label: int) -> np.ndarray:
    return np.array(CLASS_COLORS.get(int(label), (255, 255, 255))) / 255.0


def color_for_yolo(label: int) -> np.ndarray:
    # YOLO ids are shifted by -1 when crowd class is filtered out.
    visdrone_id = int(label) + 1
    return np.array(CLASS_COLORS.get(visdrone_id, (255, 255, 255))) / 255.0


def draw_boxes(
    ax: plt.Axes,
    image: np.ndarray,
    boxes: np.ndarray,
    labels: np.ndarray,
    *,
    label_fn,
    color_fn,
    title: str,
) -> None:
    ax.imshow(image)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.axis("off")

    for box, label in zip(boxes, labels):
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1
        color = color_fn(int(label))
        rect = patches.Rectangle(
            (x1, y1),
            width,
            height,
            linewidth=2,
            edgecolor=color,
            facecolor="none",
        )
        ax.add_patch(rect)
        ax.text(
            x1,
            max(y1 - 4, 0),
            label_fn(int(label)),
            bbox={"boxstyle": "round,pad=0.2", "facecolor": color, "alpha": 0.7},
            fontsize=7,
            color="white",
            weight="bold",
        )


def verify_image(
    image_path: Path,
    visdrone_ann_dir: Path,
    yolo_ann_dir: Path,
    *,
    filter_ignored: bool,
    filter_crowd: bool,
    mode: str,
    save_path: Path | None,
    show: bool,
) -> dict[str, int]:
    image = np.array(Image.open(image_path))
    img_height, img_width = image.shape[:2]
    stem = image_path.stem

    vis_boxes, vis_labels = parse_visdrone_annotations(
        visdrone_ann_dir / f"{stem}.txt",
        img_width,
        img_height,
        filter_ignored=filter_ignored,
        filter_crowd=filter_crowd,
    )
    yolo_boxes, yolo_labels = parse_yolo_annotations(
        yolo_ann_dir / f"{stem}.txt",
        img_width,
        img_height,
    )

    if mode == "overlay":
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.imshow(image)
        ax.set_title(
            f"{stem} | VisDrone={len(vis_boxes)} YOLO={len(yolo_boxes)}",
            fontsize=12,
            fontweight="bold",
        )
        ax.axis("off")

        for box, label in zip(vis_boxes, vis_labels):
            x1, y1, x2, y2 = box
            color = color_for_visdrone(int(label))
            rect = patches.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                linewidth=2,
                edgecolor=color,
                facecolor="none",
                linestyle="-",
            )
            ax.add_patch(rect)

        for box, label in zip(yolo_boxes, yolo_labels):
            x1, y1, x2, y2 = box
            color = color_for_yolo(int(label))
            rect = patches.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                linewidth=2,
                edgecolor=color,
                facecolor="none",
                linestyle="--",
            )
            ax.add_patch(rect)
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        draw_boxes(
            ax1,
            image,
            vis_boxes,
            vis_labels,
            label_fn=class_name_for_visdrone,
            color_fn=color_for_visdrone,
            title=f"VisDrone original ({len(vis_boxes)} boxes)",
        )
        draw_boxes(
            ax2,
            image,
            yolo_boxes,
            yolo_labels,
            label_fn=class_name_for_yolo,
            color_fn=color_for_yolo,
            title=f"YOLO converted ({len(yolo_boxes)} boxes)",
        )
        fig.suptitle(stem, fontsize=13, fontweight="bold")

    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)

    return {
        "visdrone_boxes": len(vis_boxes),
        "yolo_boxes": len(yolo_boxes),
        "match_count": int(len(vis_boxes) == len(yolo_boxes)),
    }


def resolve_paths(
    args: argparse.Namespace,
) -> tuple[Path, Path, Path, list[Path]]:
    data_dir = args.data_dir.resolve()
    yolo_dir = args.yolo_dir.resolve()

    if args.image:
        image_path = args.image.resolve()
        split_name = args.split
        visdrone_split = data_dir / SPLITS[split_name]
        return (
            visdrone_split / "annotations",
            yolo_dir / "labels" / split_name,
            visdrone_split / "images",
            [image_path],
        )

    split_name = args.split
    visdrone_split = data_dir / SPLITS[split_name]
    image_dir = visdrone_split / "images"
    all_images = sorted(
        p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES
    )
    if not all_images:
        raise FileNotFoundError(f"No images found in {image_dir}")

    if args.num_samples >= len(all_images):
        sample_images = all_images
    else:
        rng = random.Random(args.seed)
        sample_images = rng.sample(all_images, args.num_samples)

    return (
        visdrone_split / "annotations",
        yolo_dir / "labels" / split_name,
        image_dir,
        sample_images,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visually verify VisDrone YOLO conversion")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--yolo-dir", type=Path, default=Path("data/visdrone_yolo"))
    parser.add_argument("--split", choices=list(SPLITS), default="train")
    parser.add_argument("--image", type=Path, help="Verify a single image path")
    parser.add_argument("--num-samples", type=int, default=5, help="Random images to verify")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--mode",
        choices=["side-by-side", "overlay"],
        default="side-by-side",
        help="side-by-side: two panels; overlay: solid=VisDrone, dashed=YOLO",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/yolo_verify"))
    parser.add_argument("--show", action="store_true", help="Display figures interactively")
    parser.add_argument("--no-save", action="store_true", help="Do not save figures to disk")
    parser.add_argument("--keep-ignored", action="store_true")
    parser.add_argument("--keep-crowd", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    visdrone_ann_dir, yolo_ann_dir, _image_dir, images = resolve_paths(args)

    if not yolo_ann_dir.is_dir():
        print(f"YOLO labels not found: {yolo_ann_dir}")
        print("Run convert_visdrone_to_yolo.py first.")
        return 1

    print(f"Verifying {len(images)} image(s) from split '{args.split}'")
    print(f"  VisDrone annotations: {visdrone_ann_dir}")
    print(f"  YOLO labels:          {yolo_ann_dir}")

    mismatches = 0
    for image_path in images:
        save_path = None
        if not args.no_save:
            save_path = args.output_dir / args.split / f"{image_path.stem}.png"

        stats = verify_image(
            image_path,
            visdrone_ann_dir,
            yolo_ann_dir,
            filter_ignored=not args.keep_ignored,
            filter_crowd=not args.keep_crowd,
            mode=args.mode,
            save_path=save_path,
            show=args.show,
        )
        if stats["visdrone_boxes"] != stats["yolo_boxes"]:
            mismatches += 1
            print(
                f"  count mismatch: {image_path.name} "
                f"(visdrone={stats['visdrone_boxes']}, yolo={stats['yolo_boxes']})"
            )

    print()
    if mismatches:
        print(f"Done with {mismatches} box-count mismatch(es). Review saved images.")
        return 1

    print("Done. Box counts match for all checked images.")
    if not args.no_save:
        print(f"Saved figures under: {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
