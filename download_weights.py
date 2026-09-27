#!/usr/bin/env python3
"""Download dronefreak VisDrone YOLO weights into ./weights/ for easy reuse."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download

MODELS = {
    "yolov8s": "dronefreak/visdrone-yolov8s",
    "yolov9e": "dronefreak/visdrone-yolov9e",
    "yolov9c": "dronefreak/visdrone-yolov9c",
    "yolov11x": "dronefreak/visdrone-yolov11x",
}


def download_model(name: str, output_dir: Path) -> Path:
    if name not in MODELS:
        raise ValueError(f"Unknown model {name!r}. Choose from: {list(MODELS)}")

    repo_id = MODELS[name]
    cached = hf_hub_download(repo_id=repo_id, filename="best.pt")
    cached_path = Path(cached)

    output_dir.mkdir(parents=True, exist_ok=True)
    dest = output_dir / f"visdrone-{name}.pt"
    if dest.exists() or dest.is_symlink():
        dest.unlink()

    dest.symlink_to(cached_path.resolve())
    print(f"{name}: {dest} -> {cached_path}")
    return dest


def main() -> None:
    parser = argparse.ArgumentParser(description="Download dronefreak VisDrone weights")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(MODELS),
        default=list(MODELS),
        help="Which checkpoints to download (default: both)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("weights"),
        help="Directory for symlinks (default: weights/)",
    )
    args = parser.parse_args()

    print(f"HF cache root: {Path.home() / '.cache/huggingface/hub'}")
    print(f"Project weights: {args.output_dir.resolve()}\n")

    for name in args.models:
        download_model(name, args.output_dir)

    print("\nUse in training, e.g.:")
    print('  YOLO("weights/visdrone-yolov9e.pt").train(data="data/visdrone_yolo/dataset_13cls.yaml", ...)')


if __name__ == "__main__":
    main()
