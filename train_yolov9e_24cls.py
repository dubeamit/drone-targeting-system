#!/usr/bin/env python3
"""
Fine-tune YOLOv9e on 24-Class Aerial Surveillance & Defense Dataset
==================================================================
Optimized for:
  - Base Checkpoint: weights/custom_yolov9e.pt (preserves converged military/VisDrone weights)
  - Class Imbalance Mitigation:
      • Boosted Classification Gain: cls=1.2 (default 0.5) to penalize rare-class misclassification
      • Synthetic Context Augmentations: mixup=0.15, copy_paste=0.10, mosaic=1.0
      • Late Convergence Purity: close_mosaic=10 (turns off cuts for final 10 epochs)
      • Cosine LR Decay: cos_lr=True with gentle lr0=0.001 fine-tuning rate
"""

import argparse
import os
import sys
from pathlib import Path

# Fix cuDNN library path conflicts if present
def _sanitize_cuda_path():
    ld = os.environ.get("LD_LIBRARY_PATH", "")
    if ld and ("/usr/local/cuda" in ld or "cuda" in ld.lower()):
        cleaned = [p for p in ld.split(":") if p and "/usr/local/cuda" not in p]
        if cleaned:
            os.environ["LD_LIBRARY_PATH"] = ":".join(cleaned)
        else:
            os.environ.pop("LD_LIBRARY_PATH", None)

_sanitize_cuda_path()

from ultralytics import YOLO

CURRENT_DIR = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = CURRENT_DIR / "weights" / "custom_yolov9e.pt"
DEFAULT_DATA = CURRENT_DIR / "data" / "merged_drone_dataset" / "dataset.yaml"


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune YOLOv9e with 24 Classes")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS, help="Starting weights checkpoint")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="Path to 24-class dataset.yaml")
    parser.add_argument("--epochs", type=int, default=40, help="Number of fine-tuning epochs (default: 40)")
    parser.add_argument("--batch", type=int, default=8, help="Batch size (default: 8)")
    parser.add_argument("--imgsz", type=int, default=1024, help="Image resolution (1024 or 1280 for small drone targets)")
    parser.add_argument("--lr0", type=float, default=0.001, help="Initial learning rate for fine-tuning")
    parser.add_argument("--lrf", type=float, default=0.01, help="Final learning rate fraction (lr0 * lrf)")
    parser.add_argument("--device", default="0", help="CUDA device index (e.g. '0')")
    parser.add_argument("--workers", type=int, default=8, help="Dataloader workers (default: 8)")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience epochs")
    parser.add_argument("--project", type=Path, default=Path("runs/finetune"), help="Output project directory")
    parser.add_argument("--name", default="yolov9e_24cls", help="Run experiment name")
    parser.add_argument("--resume", action="store_true", help="Resume from last interrupted checkpoint")
    parser.add_argument("--freeze", type=int, default=None, help="Optionally freeze first N backbone layers")
    parser.add_argument("--fraction", type=float, default=1.0, help="Fraction of dataset to train on (for quick test)")
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.weights.exists():
        print(f"Error: Starting checkpoint '{args.weights}' not found.")
        sys.exit(1)

    if not args.data.exists():
        print(f"Error: Dataset config '{args.data}' not found.")
        sys.exit(1)

    print("\n" + "=" * 75)
    print("🚀 YOLOv9e 24-CLASS FINE-TUNING PIPELINE")
    print("=" * 75)
    print(f"  • Base Checkpoint:       {args.weights.name} ({args.weights})")
    print(f"  • Dataset:               {args.data}")
    print(f"  • Target Resolution:     {args.imgsz}x{args.imgsz}")
    print(f"  • Batch Size:            {args.batch}")
    print(f"  • Epochs / Patience:     {args.epochs} / {args.patience}")
    print(f"  • Learning Rate:         {args.lr0} (Cosine decay to {args.lr0 * args.lrf})")
    print(f"  • Imbalance Mitigation:  cls_gain=1.2, mixup=0.15, copy_paste=0.10, mosaic=1.0")
    print(f"  • Acceleration Device:   CUDA (Device {args.device}) with {args.workers} workers")
    print("=" * 75 + "\n")

    train_params = {
        "data": str(args.data.resolve()),
        "epochs": args.epochs,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "device": args.device,
        "workers": args.workers,
        "project": str(args.project),
        "name": args.name,
        "exist_ok": True,
        "pretrained": True,
        "amp": True,               # Automatic Mixed Precision for speed and VRAM efficiency
        "plots": True,             # Generate PR curves, confusion matrix, and prediction batches
        "patience": args.patience,
        "fraction": args.fraction,

        # Optimization & Schedule
        "optimizer": "auto",
        "lr0": args.lr0,
        "lrf": args.lrf,
        "cos_lr": True,            # Smooth cosine annealing decay
        "warmup_epochs": 3.0,

        # Class Imbalance & Small Object Defense Strategies
        "cls": 1.2,                # Boosted classification loss weight (default 0.5)
        "box": 7.5,                # Box regression weight
        "dfl": 1.5,                # Distribution focal loss weight
        "mosaic": 1.0,             # Full mosaic augmentation
        "mixup": 0.15,             # Blends rare classes into diverse backgrounds
        "copy_paste": 0.10,        # Copies rare target instances across scenes
        "close_mosaic": 10,        # Turn off mosaic for final 10 epochs for clean convergence
    }

    if args.freeze is not None:
        train_params["freeze"] = args.freeze
    if args.resume:
        train_params["resume"] = True

    print("Initializing YOLOv9e model...")
    model = YOLO(str(args.weights))

    print("\nStarting training loop...")
    print("Note: Ultralytics will automatically expand the head from 22 to 24 classes.\n")
    model.train(**train_params)

    print("\nTraining completed! Results and weights saved to:", args.project / args.name)


if __name__ == "__main__":
    main()
