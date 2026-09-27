"""Shared helpers for X-AnyLabeling JSON <-> YOLO txt conversion."""

from __future__ import annotations

import json
from pathlib import Path

import yaml
from PIL import Image

ANYLABELING_VERSION = "4.0.0-beta.10"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def load_class_names(yaml_path: Path) -> list[str]:
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    names = data.get("names", [])
    if isinstance(names, dict):
        return [names[k] for k in sorted(names, key=lambda x: int(x))]
    return list(names)


def label_to_id(class_names: list[str]) -> dict[str, int]:
    return {name: idx for idx, name in enumerate(class_names)}


def resolve_image(path: Path) -> Path | None:
    if path.suffix.lower() not in IMAGE_SUFFIXES:
        return None
    try:
        resolved = path.resolve(strict=True)
    except (FileNotFoundError, OSError):
        return None
    return resolved if resolved.is_file() else None


def yolo_to_xyxy(
    class_id: int,
    x_center: float,
    y_center: float,
    width: float,
    height: float,
    img_width: int,
    img_height: int,
) -> tuple[int, float, float, float, float]:
    x1 = (x_center - width / 2) * img_width
    y1 = (y_center - height / 2) * img_height
    x2 = (x_center + width / 2) * img_width
    y2 = (y_center + height / 2) * img_height
    return class_id, x1, y1, x2, y2


def xyxy_to_yolo(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    img_width: int,
    img_height: int,
) -> tuple[float, float, float, float]:
    x_center = ((x1 + x2) / 2) / img_width
    y_center = ((y1 + y2) / 2) / img_height
    width = abs(x2 - x1) / img_width
    height = abs(y2 - y1) / img_height
    return x_center, y_center, width, height


def make_rectangle_shape(label: str, x1: float, y1: float, x2: float, y2: float) -> dict:
    return {
        "label": label,
        "score": None,
        "points": [
            [x1, y1],
            [x2, y1],
            [x2, y2],
            [x1, y2],
        ],
        "group_id": None,
        "description": "",
        "difficult": False,
        "shape_type": "rectangle",
        "flags": {},
        "attributes": {},
        "kie_linking": [],
    }


def parse_yolo_label_file(
    label_path: Path,
    class_names: list[str],
    img_width: int,
    img_height: int,
) -> list[dict]:
    if not label_path.exists():
        return []

    shapes: list[dict] = []
    with open(label_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                continue
            class_id = int(parts[0])
            x_center, y_center, width, height = map(float, parts[1:])
            if not (0 <= class_id < len(class_names)):
                continue
            _, x1, y1, x2, y2 = yolo_to_xyxy(
                class_id, x_center, y_center, width, height, img_width, img_height
            )
            shapes.append(make_rectangle_shape(class_names[class_id], x1, y1, x2, y2))
    return shapes


def parse_merged_yolo_labels(
    label_path: Path,
    prelabel_path: Path | None,
    class_names: list[str],
    img_width: int,
    img_height: int,
) -> list[dict]:
    """Combine canonical YOLO labels with optional tree/building prelabel drafts."""
    shapes = parse_yolo_label_file(label_path, class_names, img_width, img_height)
    if prelabel_path is not None:
        shapes.extend(parse_yolo_label_file(prelabel_path, class_names, img_width, img_height))
    return shapes


def build_anylabeling_json(
    shapes: list[dict],
    image_name: str,
    img_width: int,
    img_height: int,
) -> dict:
    return {
        "version": ANYLABELING_VERSION,
        "flags": {},
        "checked": False,
        "shapes": shapes,
        "imagePath": image_name,
        "imageData": None,
        "imageHeight": img_height,
        "imageWidth": img_width,
    }


def write_anylabeling_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")


def parse_anylabeling_json(
    json_path: Path,
    class_names: list[str],
) -> tuple[list[str], int, int]:
    with open(json_path) as f:
        data = json.load(f)

    img_width = int(data["imageWidth"])
    img_height = int(data["imageHeight"])
    name_to_id = label_to_id(class_names)
    lines: list[str] = []

    for shape in data.get("shapes", []):
        if shape.get("shape_type") != "rectangle":
            continue
        label = shape.get("label")
        if label not in name_to_id:
            continue
        points = shape.get("points", [])
        if len(points) < 2:
            continue
        xs = [float(p[0]) for p in points]
        ys = [float(p[1]) for p in points]
        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)
        x_center, y_center, width, height = xyxy_to_yolo(x1, y1, x2, y2, img_width, img_height)
        class_id = name_to_id[label]
        lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

    return lines, img_width, img_height


def image_size(image_path: Path) -> tuple[int, int]:
    with Image.open(image_path) as img:
        width, height = img.size
    return width, height
