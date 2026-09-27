#!/usr/bin/env python3
"""
Drone Targeting System - Primary Model Training & Fine-Tuning Pipeline
======================================================================
Supports fine-tuning on aerial defense datasets, VisDrone benchmarks,
and custom class subsets with configurable hyper-parameters.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Fix potential CUDA / cuDNN library path conflicts
def _sanitize_cuda_library_path() -> None:
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

_sanitize_cuda_library_path()

from ultralytics import YOLO

CURRENT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA = CURRENT_DIR / "configs" / "dataset_24cls.yaml"

MODEL_PRESETS: dict[str, Path] = {
    "yolov9e": CURRENT_DIR / "weights" / "custom_yolov9e.pt",
    "yolo26s": CURRENT_DIR / "runs" / "finetune" / "yolo26s" / "weights" / "best.pt",
    "yolov8s": CURRENT_DIR / "runs" / "finetune" / "yolov8s" / "weights" / "best.pt",
    "yolo11x": Path("yolo11x.pt"),
}

DEFAULT_HPARAMS = {
    "yolov9e": {"batch": 8, "lr0": 0.001, "imgsz": 1024},
    "yolov11x": {"batch": 8, "lr0": 0.005, "imgsz": 1024},
    "yolov8s": {"batch": 16, "lr0": 0.01, "imgsz": 1024},
    "yolo26s": {"batch": 32, "lr0": 0.01, "imgsz": 1024},
}


def resolve_model_path(model_input: str) -> str:
    key = model_input.lower().strip()
    if key in MODEL_PRESETS and MODEL_PRESETS[key].exists():
        return str(MODEL_PRESETS[key])
    p = Path(model_input)
    if p.exists():
        return str(p)
    in_weights = CURRENT_DIR / "weights" / model_input
    if in_weights.exists():
        return str(in_weights)
    return model_input


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and fine-tune YOLO models for aerial drone targeting")
    parser.add_argument("--model", default="yolov9e", help="Model preset ('yolov9e', 'yolo26s', 'yolov8s') or path to .pt file")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="Path to dataset YAML config")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=None, help="Inference/training resolution (default: 1024 for high-res drone imagery)")
    parser.add_argument("--batch", type=int, default=None, help="Batch size (auto-configured per model if omitted)")
    parser.add_argument("--lr0", type=float, default=None, help="Initial learning rate")
    parser.add_argument("--device", default="0" if os.environ.get("CUDA_VISIBLE_DEVICES") else "0", help="CUDA device index or 'cpu'")
    parser.add_argument("--project", type=Path, default=Path("runs/finetune"), help="Output project directory")
    parser.add_argument("--name", default=None, help="Experiment run name")
    parser.add_argument("--freeze", type=int, default=None, help="Freeze backbone first N layers")
    parser.add_argument("--resume", action="store_true", help="Resume interrupted training")
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision training")
    parser.add_argument("--fraction", type=float, default=1.0, help="Fraction of dataset to use (e.g. 0.1 for quick test)")
    parser.add_argument("--workers", type=int, default=4, help="DataLoader workers count")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    resolved_weights = resolve_model_path(args.model)
    model_key = args.model.lower().strip()
    defaults = DEFAULT_HPARAMS.get(model_key, {"batch": 8, "lr0": 0.005, "imgsz": 1024})

    train_kwargs = {
        "data": str(args.data.resolve()) if args.data.exists() else str(args.data),
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

    print("=" * 70)
    print("🚁 DRONE TARGETING SYSTEM - MODEL TRAINING PIPELINE")
    print("=" * 70)
    print(f"  • Base Checkpoint: {resolved_weights}")
    print(f"  • Dataset Config:   {args.data}")
    print(f"  • Resolution:       {train_kwargs['imgsz']}x{train_kwargs['imgsz']}")
    print(f"  • Batch Size:       {train_kwargs['batch']}")
    print(f"  • Epochs:           {train_kwargs['epochs']}")
    print("=" * 70)

    model = YOLO(resolved_weights)
    model.train(**train_kwargs)


if __name__ == "__main__":
    main()
