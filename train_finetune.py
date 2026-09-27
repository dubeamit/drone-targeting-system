#!/usr/bin/env python3
"""Fine-tune dronefreak VisDrone weights with optional extra classes (tree, building)."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def _sanitize_cuda_library_path() -> None:
    """Drop system CUDA paths that conflict with PyTorch's bundled cuDNN.

    A common cause of:
      RuntimeError: GET was unable to find an engine to execute this computation
    is LD_LIBRARY_PATH pointing at /usr/local/cuda while using pip PyTorch wheels.
    """
    ld_path = os.environ.get("LD_LIBRARY_PATH", "")
    if not ld_path:
        return
    if "/usr/local/cuda" not in ld_path and "cuda" not in ld_path.lower():
        return
    cleaned = [part for part in ld_path.split(":") if part and "/usr/local/cuda" not in part]
    if cleaned:
        os.environ["LD_LIBRARY_PATH"] = ":".join(cleaned)
    else:
        os.environ.pop("LD_LIBRARY_PATH", None)
    print("Adjusted LD_LIBRARY_PATH to avoid cuDNN conflicts with PyTorch.")


_sanitize_cuda_library_path()

from ultralytics import YOLO  # noqa: E402

DEFAULT_DATA = Path("data/merged_drone_dataset/dataset.yaml")
WEIGHTS = {
    "yolov9e": Path("weights/visdrone-yolov9e.pt"),
    "yolov11x": Path("weights/visdrone-yolov11x.pt"),
    "yolov9c": Path("weights/visdrone-yolov9c.pt"),
    "yolov8s": Path("yolov8s.pt"), # Use base COCO weights instead of visdrone weights
    "yolo26s": Path("yolo26s.pt"), # Fast model for quick testing/edge
}
DEFAULT_HPARAMS = {
    # High-VRAM GPU optimization for High-Res Drone Imagery
    "yolov9e": {"batch": 8, "lr0": 0.001, "imgsz": 1024},
    "yolov11x": {"batch": 8, "lr0": 0.005, "imgsz": 1024},
    "yolov9c": {"batch": 16, "lr0": 0.01, "imgsz": 1024},
    # Client configuration (i3, 8GB RAM, Intel UHD) - High Res for Small Objects
    "yolov8s": {"batch": 16, "lr0": 0.01, "imgsz": 1280},
    "yolo26s": {"batch": 32, "lr0": 0.01, "imgsz": 1280},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune dronefreak YOLO on local VisDrone YOLO data")
    parser.add_argument("--model", choices=list(WEIGHTS), required=True)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--lr0", type=float, default=None)
    parser.add_argument("--device", default="0")
    parser.add_argument("--project", type=Path, default=Path("runs/finetune"))
    parser.add_argument("--name", default=None, help="Run name (default: model name)")
    parser.add_argument("--freeze", type=int, default=None, help="Freeze first N layers (e.g. 10)")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable mixed precision (try this if backward pass crashes)",
    )
    parser.add_argument(
        "--fraction",
        type=float,
        default=1.0,
        help="Train on a fraction of the dataset (e.g. 0.1 for a quick test)",
    )
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    weights = WEIGHTS[args.model]
    # Removed strict exists() check to allow Ultralytics to auto-download base weights like yolov8s.pt
    # if not weights.exists():
    #     raise FileNotFoundError(
    #         f"Missing {weights}. Run: python download_weights.py --models {args.model}"
    #     )

    defaults = DEFAULT_HPARAMS[args.model]
    train_kwargs = {
        "data": str(args.data.resolve()),
        "epochs": args.epochs,
        "imgsz": args.imgsz or defaults["imgsz"],
        "batch": args.batch or defaults["batch"],
        "lr0": args.lr0 or defaults["lr0"],
        "device": args.device,
        "project": str(args.project),
        "name": args.name or args.model,
        "exist_ok": True,
        "pretrained": True,
        "patience": 50,
        "amp": not args.no_amp,
        "plots": True,
        "fraction": args.fraction,
        "workers": args.workers,
    }
    if args.freeze is not None:
        train_kwargs["freeze"] = args.freeze
    if args.resume:
        train_kwargs["resume"] = True

    print(f"Loading weights: {weights}")
    print(f"Dataset: {args.data}")
    print("Ultralytics will re-init the detection head if nc differs from the checkpoint.")
    print(
        "Note: Ultralytics may download yolo26n.pt once for AMP sanity checks — "
        "that is NOT your training model; yolov9e/yolov11x weights are still used."
    )

    model = YOLO(str(weights))
    model.train(**train_kwargs)


if __name__ == "__main__":
    main()
