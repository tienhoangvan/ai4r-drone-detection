#!/usr/bin/env python3
"""
Convert a YOLO-format dataset into a COCO JSON annotation file.

Project structure expected:
AI4R/
├── frames/drone_02/
│                   ├── images/
│                   └── labels/
│                   └── annotations_drone02_coco.json
└── scripts/
    └── yolo_to_coco.py

Usage:
    cd ~/robotic_vision/annotation/AI4R
    python3 scripts/yolo_to_coco.py

Optional custom usage:
    python3 scripts/yolo_to_coco.py \
        --image-dir drone_02/images \
        --label-dir drone_02/labels \
        --output drone_02/annotations_drone02_coco.json \
        --class-names drone
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any

from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert YOLO labels to COCO JSON for CVAT import."
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=Path("../frames/drone_02/images"),
        help="Directory containing dataset images.",
    )
    parser.add_argument(
        "--label-dir",
        type=Path,
        default=Path("../frames/drone_02/labels"),
        help="Directory containing YOLO .txt label files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("../frames/drone_02/annotations_drone04_coco.json"),
        help="Output COCO JSON file path.",
    )
    parser.add_argument(
        "--class-names",
        nargs="+",
        default=["drone"],
        help="List of class names in YOLO class-id order. Example: --class-names drone bird",
    )
    parser.add_argument(
        "--image-exts",
        nargs="+",
        default=[".jpg", ".jpeg", ".png", ".bmp"],
        help="Allowed image file extensions.",
    )
    return parser.parse_args()


def build_categories(class_names: List[str]) -> List[Dict[str, Any]]:
    """
    Build COCO categories list.
    YOLO class 0 -> COCO category id 1
    YOLO class 1 -> COCO category id 2
    etc.
    """
    categories = []
    for idx, name in enumerate(class_names, start=1):
        categories.append(
            {
                "id": idx,
                "name": name,
                "supercategory": "object",
            }
        )
    return categories


def yolo_line_to_coco_bbox(
    line: str,
    image_width: int,
    image_height: int,
) -> tuple[int, List[float], float] | None:
    """
    Convert one YOLO annotation line:
        class_id cx cy w h
    into COCO bbox:
        [x_min, y_min, width, height]

    Returns:
        (category_id, coco_bbox, area)
    """
    parts = line.strip().split()
    if len(parts) != 5:
        return None

    try:
        class_id = int(float(parts[0]))
        cx = float(parts[1])
        cy = float(parts[2])
        bw = float(parts[3])
        bh = float(parts[4])
    except ValueError:
        return None

    # Convert normalized YOLO box to pixel coordinates
    box_w = bw * image_width
    box_h = bh * image_height
    x_min = (cx * image_width) - (box_w / 2.0)
    y_min = (cy * image_height) - (box_h / 2.0)

    # Clamp for safety
    x_min = max(0.0, x_min)
    y_min = max(0.0, y_min)
    box_w = max(0.0, min(box_w, image_width - x_min))
    box_h = max(0.0, min(box_h, image_height - y_min))

    area = box_w * box_h

    # COCO category id starts from 1
    category_id = class_id + 1

    return (
        category_id,
        [round(x_min, 2), round(y_min, 2), round(box_w, 2), round(box_h, 2)],
        round(area, 2),
    )


def main() -> None:
    args = parse_args()

    image_dir: Path = args.image_dir
    label_dir: Path = args.label_dir
    output_path: Path = args.output
    image_exts = {ext.lower() for ext in args.image_exts}
    class_names: List[str] = args.class_names

    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    if not label_dir.exists():
        raise FileNotFoundError(f"Label directory not found: {label_dir}")

    image_files = sorted(
        [p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in image_exts]
    )

    if not image_files:
        raise RuntimeError(f"No images found in: {image_dir}")

    coco: Dict[str, Any] = {
        "images": [],
        "annotations": [],
        "categories": build_categories(class_names),
    }

    annotation_id = 1
    image_id = 1
    total_missing_labels = 0
    total_invalid_lines = 0

    for img_path in image_files:
        with Image.open(img_path) as img:
            width, height = img.size

        coco["images"].append(
            {
                "id": image_id,
                "file_name": img_path.name,
                "width": width,
                "height": height,
            }
        )

        label_path = label_dir / f"{img_path.stem}.txt"

        if not label_path.exists():
            total_missing_labels += 1
            image_id += 1
            continue

        lines = label_path.read_text(encoding="utf-8").splitlines()

        for line in lines:
            if not line.strip():
                continue

            result = yolo_line_to_coco_bbox(line, width, height)
            if result is None:
                total_invalid_lines += 1
                print(f"[WARN] Invalid label line skipped: {label_path.name} -> {line}")
                continue

            category_id, bbox, area = result

            if category_id < 1 or category_id > len(class_names):
                total_invalid_lines += 1
                print(
                    f"[WARN] Class id out of range in {label_path.name}: "
                    f"category_id={category_id}"
                )
                continue

            coco["annotations"].append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": bbox,
                    "area": area,
                    "iscrowd": 0,
                }
            )
            annotation_id += 1

        image_id += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(coco, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"[OK] COCO JSON saved to: {output_path}")
    print(f"[INFO] Total images       : {len(coco['images'])}")
    print(f"[INFO] Total annotations  : {len(coco['annotations'])}")
    print(f"[INFO] Missing label files: {total_missing_labels}")
    print(f"[INFO] Invalid label lines: {total_invalid_lines}")
    print("[INFO] Categories:")
    for cat in coco["categories"]:
        print(f"       - id={cat['id']}, name={cat['name']}")


if __name__ == "__main__":
    main()