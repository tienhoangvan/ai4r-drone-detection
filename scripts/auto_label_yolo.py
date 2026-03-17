#!/usr/bin/env python3
"""
auto_label_yolo.py

Offline auto-labeling script for generating YOLO-format annotations
from a trained Ultralytics YOLO model (YOLOv8/YOLOv11, etc.).

Main features:
- GPU-optimized (uses CUDA if available, otherwise falls back to CPU)
- Batch inference for higher throughput
- Generates YOLO .txt labels next to each image
- Logs statistics: total images, total bounding boxes, average boxes per image,
  and a simple distribution histogram
- Designed for the **first auto-labeling round** before cleaning labels in CVAT

Usage example:
--------------
python auto_label_yolo.py \
    --model models/yolo26n.pt \
    --image-dirs frames/case01 frames/case02 \
    --imgsz 768 \
    --conf 0.3 \
    --iou 0.45 \
    --batch 16
"""

import argparse
from pathlib import Path
from collections import Counter, defaultdict
from typing import List, Tuple

import torch
from ultralytics import YOLO


# -------- Default config values (you can override via CLI) -------- #
DEFAULT_MODEL_PATH = "models/yolo26n.pt"
DEFAULT_IMAGE_DIRS = ["frames/case01"]
DEFAULT_IMG_SUFFIXES = (".jpg", ".jpeg", ".png")
DEFAULT_IMGSZ = 768
DEFAULT_CONF = 0.30
DEFAULT_IOU = 0.45
DEFAULT_BATCH_SIZE = 16


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Batch auto-label images with a YOLO model and save YOLO-format labels."
    )

    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help=f"Path to YOLO model (.pt). Default: {DEFAULT_MODEL_PATH}",
    )
    parser.add_argument(
        "--image-dirs",
        type=str,
        nargs="+",
        default=DEFAULT_IMAGE_DIRS,
        help=f"List of image directories to process. Default: {DEFAULT_IMAGE_DIRS}",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=DEFAULT_IMGSZ,
        help=f"Inference image size (square). Default: {DEFAULT_IMGSZ}",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=DEFAULT_CONF,
        help=f"Confidence threshold. Default: {DEFAULT_CONF}",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=DEFAULT_IOU,
        help=f"IOU threshold for NMS. Default: {DEFAULT_IOU}",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Batch size for inference. Default: {DEFAULT_BATCH_SIZE}",
    )
    parser.add_argument(
        "--suffixes",
        type=str,
        nargs="+",
        default=list(DEFAULT_IMG_SUFFIXES),
        help=f"Image file suffixes. Default: {DEFAULT_IMG_SUFFIXES}",
    )
    parser.add_argument(
        "--keep-empty",
        action="store_true",
        help="If set, create empty .txt files for images with no detections.",
    )

    return parser.parse_args()


def collect_images(image_dirs: List[str], suffixes: Tuple[str, ...]) -> List[Path]:
    """
    Scan provided directories and collect all image paths with given suffixes.

    Args:
        image_dirs: List of directories containing images.
        suffixes: Allowed image file extensions.

    Returns:
        List of Path objects for all found images.
    """
    all_images = []
    for d in image_dirs:
        p = Path(d)
        if not p.exists():
            print(f"[WARN] Image directory does not exist, skipping: {p}")
            continue
        if not p.is_dir():
            print(f"[WARN] Not a directory, skipping: {p}")
            continue

        images = sorted([f for f in p.rglob("*") if f.suffix.lower() in suffixes])
        print(f"[INFO] Found {len(images)} images in directory: {p}")
        all_images.extend(images)

    print(f"[INFO] Total images collected from all dirs: {len(all_images)}")
    return all_images


def select_device() -> str:
    """
    Choose the best device for inference:
    - If CUDA is available: use GPU 0
    - Otherwise: use CPU

    Returns:
        Device string: 'cpu' or 'cuda:0'
    """
    if torch.cuda.is_available():
        device = "cuda:0"
        print(f"[INFO] Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        print("[INFO] CUDA not available. Falling back to CPU.")
    return device


def load_model(model_path: str, device: str) -> YOLO:
    """
    Load the YOLO model and move it to the selected device.

    Note:
    - We use model.to(device) once here and do NOT pass `device=...`
      to the model call later. This avoids internal device re-selection
      issues inside Ultralytics and keeps everything stable.

    Args:
        model_path: Path to the .pt model file.
        device: 'cpu' or 'cuda:0'.

    Returns:
        Loaded YOLO model.
    """
    model = YOLO(model_path)
    print(f"[INFO] Loaded model from: {model_path}")

    # Move model to selected device
    if device != "cpu":
        model.to(device)
        print(f"[INFO] Model moved to device: {device}")
    else:
        print("[INFO] Model running on CPU.")

    return model


def run_batch_inference(
    model: YOLO,
    batch_paths: List[Path],
    imgsz: int,
    conf: float,
    iou: float,
):
    """
    Run YOLO inference on a batch of image paths.

    Args:
        model: YOLO model instance.
        batch_paths: List of image Paths for this batch.
        imgsz: Image size for inference.
        conf: Confidence threshold.
        iou: IOU threshold for NMS.

    Returns:
        List of Results objects (one per image).
    """
    # We pass a list of string paths to the model directly.
    # This is more stable than using `model.predict()` in complex environments.
    sources = [str(p) for p in batch_paths]

    results = model(
        sources,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        verbose=False,
    )

    return results


def save_yolo_labels_for_image(
    img_path: Path,
    result,
    keep_empty: bool = False,
) -> int:
    """
    Save YOLO-format labels for a single image based on YOLO result.

    Args:
        img_path: Path to the input image.
        result: Single YOLO result object for this image.
        keep_empty: If True, create an empty .txt file even if no boxes.

    Returns:
        Number of bounding boxes written for this image.
    """
    # YOLO output boxes: result.boxes
    boxes = result.boxes

    # Original image shape (height, width)
    h, w = result.orig_shape  # (H, W)
    if h <= 0 or w <= 0:
        print(f"[WARN] Invalid image shape for: {img_path}")
        return 0

    # YOLO label file path: same name, .txt extension
    label_path = img_path.with_suffix(".txt")

    lines = []
    for box in boxes:
        # class index (int)
        cls_id = int(box.cls.item())

        # xywhn: normalized [x_center, y_center, width, height] (all in 0–1)
        cx, cy, bw, bh = box.xywhn[0].tolist()

        line = f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"
        lines.append(line)

    if not lines and not keep_empty:
        # No detections and we don't want empty label files
        return 0

    with open(label_path, "w") as f:
        if lines:
            f.write("\n".join(lines))
        else:
            # If keep_empty is True, we write an empty file
            pass

    return len(lines)


def auto_label(
    model: YOLO,
    images: List[Path],
    imgsz: int,
    conf: float,
    iou: float,
    batch_size: int,
    keep_empty: bool = False,
):
    """
    Main loop for batch auto-labeling.

    - Splits the images into batches
    - Runs inference
    - Writes YOLO label files
    - Collects and prints statistics

    Args:
        model: YOLO model.
        images: List of image paths to process.
        imgsz: Inference image size.
        conf: Confidence threshold.
        iou: IOU threshold for NMS.
        batch_size: Number of images per batch.
        keep_empty: Whether to create empty .txt files for no-detection images.
    """
    total_images = len(images)
    if total_images == 0:
        print("[WARN] No images to process. Exiting.")
        return

    print(
        f"[INFO] Starting auto-labeling for {total_images} images "
        f"(batch_size={batch_size}, conf={conf}, iou={iou}, imgsz={imgsz})"
    )

    # Stats
    per_image_box_count = []  # list of integers: number of boxes per image
    class_counter = Counter()  # counts of each class id
    errors = []

    # Process in batches
    for start in range(0, total_images, batch_size):
        end = min(start + batch_size, total_images)
        batch_paths = images[start:end]

        try:
            results = run_batch_inference(
                model=model,
                batch_paths=batch_paths,
                imgsz=imgsz,
                conf=conf,
                iou=iou,
            )
        except Exception as e:
            print(f"[ERROR] Inference failed for batch {start}-{end}: {e}")
            errors.append((start, end, str(e)))
            continue

        # results is an iterable with one result per input image
        for img_path, res in zip(batch_paths, results):
            num_boxes = len(res.boxes)
            per_image_box_count.append(num_boxes)

            # Update class distribution
            for box in res.boxes:
                cls_id = int(box.cls.item())
                class_counter[cls_id] += 1

            # Save YOLO labels
            n_written = save_yolo_labels_for_image(
                img_path=img_path,
                result=res,
                keep_empty=keep_empty,
            )

            print(
                f"[OK] {img_path.name}: "
                f"{num_boxes} detections, {n_written} boxes saved."
            )

    # Print summary statistics
    print("\n========== Auto-labeling Summary ==========")
    print(f"Total images processed : {total_images}")
    print(f"Images with labels     : {sum(1 for c in per_image_box_count if c > 0)}")
    print(f"Images with no labels  : {sum(1 for c in per_image_box_count if c == 0)}")
    print(f"Total boxes            : {sum(per_image_box_count)}")

    if per_image_box_count:
        avg_boxes = sum(per_image_box_count) / len(per_image_box_count)
        print(f"Average boxes / image  : {avg_boxes:.2f}")

        # Simple histogram: how many images with 0,1,2,3,... boxes
        dist = Counter(per_image_box_count)
        print("\n[Histogram] Box count per image:")
        for k in sorted(dist.keys()):
            print(f"  {k} boxes: {dist[k]} images")

    if class_counter:
        print("\n[Class distribution] (class_id: count):")
        for cls_id, count in sorted(class_counter.items()):
            print(f"  Class {cls_id}: {count} boxes")

    if errors:
        print("\n[WARN] Some batches failed during inference:")
        for (s, e, msg) in errors:
            print(f"  Batch {s}-{e}: {msg}")

    print("===========================================\n")
    print("[INFO] Auto-labeling completed.")


def main():
    args = parse_args()

    # 1. Collect all images from given directories
    images = collect_images(args.image_dirs, tuple(s.lower() for s in args.suffixes))

    # 2. Select device (GPU if available, else CPU)
    device = select_device()

    # 3. Load YOLO model to the selected device
    model = load_model(args.model, device=device)

    # 4. Run auto-labeling
    auto_label(
        model=model,
        images=images,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        batch_size=args.batch,
        keep_empty=args.keep_empty,
    )


if __name__ == "__main__":
    main()
