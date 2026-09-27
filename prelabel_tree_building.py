#!/usr/bin/env python3
"""Pre-label tree and building boxes with YOLO-World (draft annotations for human review).

Writes ONLY class 11 (tree) and 12 (building) lines to labels_prelabel/<split>/.
These are drafts — they are NOT in the main labels/ until you review and import.

Integrated workflow with X-AnyLabeling:

    # 1) Auto-detect tree/building drafts
    python prelabel_tree_building.py --split train --max-images 200 --export-anylabeling

    # 2) Review in X-AnyLabeling: data/anylabeling/train
    #    (JSON contains existing VisDrone boxes 0-10 + draft tree/building)

    # 3) Import corrected labels back to YOLO
    python import_anylabeling.py --split train

    # 4) Train
    python train_finetune.py --model yolov9e

Alternative (merge without AnyLabeling, less recommended):

    python prelabel_tree_building.py --split train --max-images 200
    python prelabel_tree_building.py --split train --merge
"""

from __future__ import annotations

import argparse
import random
import subprocess
import sys
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from ultralytics import YOLOWorld

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
CLASS_MAP = {"tree": 11, "building": 12}
SPLITS = ("train", "val", "test")


def resolve_image_path(path: Path) -> Path | None:
    """Return a readable image path, following symlinks when valid."""
    if path.suffix.lower() not in IMAGE_SUFFIXES:
        return None
    try:
        resolved = path.resolve(strict=True)
    except (FileNotFoundError, OSError):
        return None
    return resolved if resolved.is_file() else None


def list_images(image_dir: Path, max_images: int | None, seed: int) -> list[Path]:
    candidates = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    images: list[Path] = []
    broken = 0
    for path in candidates:
        if resolve_image_path(path) is None:
            broken += 1
            continue
        images.append(path)

    if broken:
        print(
            f"Warning: skipped {broken} broken image symlink(s) in {image_dir}.\n"
            "  The YOLO layout stores symlinks to VisDrone2019-DET-*/images/.\n"
            "  Re-extract the DET zips under data/, then run: python repair_yolo_images.py"
        )
    if not images:
        raise FileNotFoundError(
            f"No readable images in {image_dir}. "
            "Extract VisDrone DET archives and run: python repair_yolo_images.py"
        )

    if max_images is None or max_images >= len(images):
        return images
    rng = random.Random(seed)
    return rng.sample(images, max_images)


def yolo_line(class_id: int, xyxy: np.ndarray, width: int, height: int) -> str:
    x1, y1, x2, y2 = xyxy
    x_center = ((x1 + x2) / 2) / width
    y_center = ((y1 + y2) / 2) / height
    box_w = (x2 - x1) / width
    box_h = (y2 - y1) / height
    return f"{class_id} {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}"


def save_preview(
    image_path: Path,
    boxes: list[tuple[int, np.ndarray]],
    class_names: dict[int, str],
    out_path: Path,
) -> None:
    image = np.array(Image.open(image_path))
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(image)
    ax.axis("off")
    ax.set_title(image_path.name)

    for class_id, xyxy in boxes:
        x1, y1, x2, y2 = xyxy
        color = "lime" if class_id == 11 else "cyan"
        rect = patches.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            linewidth=2,
            edgecolor=color,
            facecolor="none",
        )
        ax.add_patch(rect)
        ax.text(x1, max(y1 - 4, 0), class_names[class_id], color=color, fontsize=8)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pre-label tree/building with YOLO-World")
    parser.add_argument("--data-dir", type=Path, default=Path("data/visdrone_yolo"))
    parser.add_argument("--split", choices=SPLITS, default="train")
    parser.add_argument("--model", default="yolov8s-worldv2.pt", help="YOLO-World checkpoint")
    parser.add_argument("--conf", type=float, default=0.15, help="Confidence threshold")
    parser.add_argument("--max-images", type=int, default=None, help="Limit images (useful for a pilot run)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--preview-dir",
        type=Path,
        default=Path("outputs/yolo_prelabel/tree_building"),
        help="Save visual previews here",
    )
    parser.add_argument(
        "--output-label-dir",
        type=Path,
        default=None,
        help="Write draft labels here (default: <data-dir>/labels_prelabel/<split>)",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Append draft tree/building lines into labels/<split>/ (skip if using AnyLabeling)",
    )
    parser.add_argument(
        "--export-anylabeling",
        action="store_true",
        help="After prelabeling, export merged JSON for X-AnyLabeling review",
    )
    parser.add_argument(
        "--anylabeling-dir",
        type=Path,
        default=Path("data/anylabeling"),
        help="Output dir for X-AnyLabeling export (with --export-anylabeling)",
    )
    parser.add_argument("--no-preview", action="store_true")
    return parser.parse_args()


def export_for_anylabeling(args: argparse.Namespace) -> None:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve().parent / "export_anylabeling.py"),
        "--split",
        args.split,
        "--yolo-dir",
        str(args.data_dir),
        "--output-dir",
        str(args.anylabeling_dir),
        "--include-prelabel",
        "--only-prelabeled",
        "--seed",
        str(args.seed),
    ]
    if args.max_images is not None:
        cmd.extend(["--max-images", str(args.max_images)])
    print("\nExporting merged labels for X-AnyLabeling...")
    subprocess.run(cmd, check=True)


def main() -> int:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    image_dir = data_dir / "images" / args.split
    target_label_dir = data_dir / "labels" / args.split
    draft_label_dir = (
        args.output_label_dir.resolve()
        if args.output_label_dir
        else data_dir / "labels_prelabel" / args.split
    )

    if not image_dir.is_dir():
        raise FileNotFoundError(f"Missing image dir: {image_dir}")

    images = list_images(image_dir, args.max_images, args.seed)
    print(f"Pre-labeling {len(images)} images from {args.split}")

    model = YOLOWorld(args.model)
    model.set_classes(list(CLASS_MAP))

    class_names = {11: "tree", 12: "building"}
    draft_label_dir.mkdir(parents=True, exist_ok=True)

    total_boxes = 0
    for image_path in images:
        readable_path = resolve_image_path(image_path)
        if readable_path is None:
            continue

        # Pass resolved path: Ultralytics uses os.path.isfile(), which fails on broken symlinks.
        results = model.predict(
            source=str(readable_path),
            conf=args.conf,
            device=args.device,
            verbose=False,
        )[0]

        width, height = results.orig_shape[1], results.orig_shape[0]
        lines: list[str] = []
        preview_boxes: list[tuple[int, np.ndarray]] = []

        if results.boxes is not None and len(results.boxes):
            for box in results.boxes:
                name = results.names[int(box.cls)]
                if name not in CLASS_MAP:
                    continue
                class_id = CLASS_MAP[name]
                xyxy = box.xyxy[0].cpu().numpy()
                lines.append(yolo_line(class_id, xyxy, width, height))
                preview_boxes.append((class_id, xyxy))

        draft_path = draft_label_dir / f"{image_path.stem}.txt"
        draft_path.write_text("\n".join(lines))
        total_boxes += len(lines)

        if not args.no_preview and preview_boxes:
            save_preview(
                readable_path,
                preview_boxes,
                class_names,
                args.preview_dir / args.split / f"{image_path.stem}.png",
            )

    print(f"Draft labels: {draft_label_dir}")
    print(f"Boxes written: {total_boxes} (tree=11, building=12)")

    if args.merge:
        merged = 0
        for draft_path in draft_label_dir.glob("*.txt"):
            if not draft_path.read_text().strip():
                continue
            target_path = target_label_dir / draft_path.name
            existing = target_path.read_text().strip() if target_path.exists() else ""
            new_lines = draft_path.read_text().strip()
            merged_text = "\n".join(part for part in (existing, new_lines) if part)
            target_path.write_text(merged_text + "\n")
            merged += 1
        print(f"Merged tree/building lines into {target_label_dir} for {merged} files.")

    if args.export_anylabeling:
        export_for_anylabeling(args)
        print("\nNext steps:")
        print(f"  1. Open in X-AnyLabeling: {args.anylabeling_dir.resolve() / args.split}")
        print("  2. Fix tree/building boxes (and delete false positives)")
        print("  3. python import_anylabeling.py --split", args.split)
        print("  4. python train_finetune.py --model yolov9e")
    else:
        print("\nNext steps:")
        print("  1. Review previews in outputs/yolo_prelabel/tree_building/")
        print("  2. Export for AnyLabeling:")
        print(
            f"     python export_anylabeling.py --split {args.split} "
            "--include-prelabel --only-prelabeled"
        )
        print("     or rerun with: --export-anylabeling")
        print("  3. After AnyLabeling review: python import_anylabeling.py --split", args.split)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
