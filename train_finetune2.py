#!/usr/bin/env python3
"""Fine-tune dronefreak VisDrone weights or standard YOLO weights with specific classes."""

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

DEFAULT_DATA = Path("data/visdrone_yolo/dataset_13cls.yaml")

# MODIFIED: Added yolov9c and yolov8s as safe options for laptops
WEIGHTS = {
    "yolov9e": Path("weights/visdrone-yolov9e.pt"),
    "yolov11x": Path("weights/visdrone-yolov11x.pt"),
    "yolov9c": Path("weights/visdrone-yolov9c.pt"),
    "yolov8s": Path("weights/visdrone-yolov8s.pt"),
}

# MODIFIED: Updated Batch sizes for 8GB VRAM safety & efficiency
DEFAULT_HPARAMS = {
    "yolov9e": {"batch": 2, "lr0": 0.001, "imgsz": 640},
    "yolov11x": {"batch": 2, "lr0": 0.005, "imgsz": 640},
    "yolov9c": {"batch": 8, "lr0": 0.005, "imgsz": 640},   # Safe for 8GB VRAM
    "yolov8s": {"batch": 16, "lr0": 0.002, "imgsz": 640},  # Extremely fast, highly recommended for testing
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune YOLO on local VisDrone YOLO data")
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
    parser.add_argument("--workers", type=int, default=4) # Keeps CPU RAM usage low
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    weights = WEIGHTS[args.model]
    
    # MODIFIED: Only raise error for the custom visdrone weights. 
    # Let standard weights (v9c/v8s) bypass this so Ultralytics can auto-download them.
    if not weights.exists() and "visdrone" in weights.name:
        raise FileNotFoundError(
            f"Missing {weights}. Run: python download_weights.py --models {args.model}"
        )

    defaults = DEFAULT_HPARAMS[args.model]
    
    # MODIFIED: Added specific safety & filtering kwargs
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
        "cache": False,  # ADDED: Prevents 16GB system RAM from crashing
        "classes": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    }
    
    if args.freeze is not None:
        train_kwargs["freeze"] = args.freeze
    if args.resume:
        train_kwargs["resume"] = True

    print(f"Loading weights: {weights}")
    print(f"Dataset: {args.data}")
    print("Ultralytics will re-init the detection head if nc differs from the checkpoint.")
    # print("Only targeting classes [0, 1, 3, 4, 5, 8, 9] (Humans and Vehicles).")

    model = YOLO(str(weights))
    model.train(**train_kwargs)


if __name__ == "__main__":
    main()