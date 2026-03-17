from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path
from typing import List, Tuple

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def find_images(images_dir: Path) -> List[Path]:
    files = [p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    return sorted(files)


def corresponding_label(image_path: Path, labels_dir: Path) -> Path:
    return labels_dir / f"{image_path.stem}.txt"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def split_list(items: List[Path], train_ratio: float, val_ratio: float, seed: int) -> Tuple[List[Path], List[Path], List[Path]]:
    items = items[:]
    random.Random(seed).shuffle(items)

    n = len(items)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_items = items[:n_train]
    val_items = items[n_train:n_train + n_val]
    test_items = items[n_train + n_val:]

    return train_items, val_items, test_items


def copy_split(items: List[Path], labels_dir: Path, out_root: Path, split_name: str) -> None:
    out_img_dir = out_root / "images" / split_name
    out_lbl_dir = out_root / "labels" / split_name
    ensure_dir(out_img_dir)
    ensure_dir(out_lbl_dir)

    for img_path in items:
        lbl_path = corresponding_label(img_path, labels_dir)

        shutil.copy2(img_path, out_img_dir / img_path.name)

        # nếu không có label thì tạo file rỗng
        out_lbl_path = out_lbl_dir / f"{img_path.stem}.txt"
        if lbl_path.exists():
            shutil.copy2(lbl_path, out_lbl_path)
        else:
            out_lbl_path.write_text("", encoding="utf-8")


def validate_pairs(images: List[Path], labels_dir: Path) -> Tuple[List[Path], List[Path]]:
    matched = []
    missing_labels = []

    for img_path in images:
        lbl_path = corresponding_label(img_path, labels_dir)
        if lbl_path.exists():
            matched.append(img_path)
        else:
            missing_labels.append(img_path)

    return matched, missing_labels


def write_data_yaml(out_root: Path, class_names: List[str]) -> Path:
    yaml_path = out_root / "data.yaml"

    lines = [
        f"path: {out_root.resolve().as_posix()}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "names:",
    ]
    for idx, name in enumerate(class_names):
        lines.append(f"  {idx}: {name}")

    yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return yaml_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build YOLO26 dataset from round1/images and round1/labels")
    parser.add_argument(
        "--images-dir",
        type=str,
        default="../prepare_datasets/round1/images",
        help="Path to source images folder",
    )
    parser.add_argument(
        "--labels-dir",
        type=str,
        default="../prepare_datasets/round1/labels",
        help="Path to source labels folder",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="../dataset/drone_round1",
        help="Path to output YOLO dataset folder",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        default=["drone"],
        help="Class names in order, e.g. --classes drone",
    )
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train split ratio")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--include-unlabeled",
        action="store_true",
        help="Include images without labels (empty txt will be created)",
    )
    args = parser.parse_args()

    images_dir = Path(args.images_dir).resolve()
    labels_dir = Path(args.labels_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    if not images_dir.exists():
        raise FileNotFoundError(f"Images folder not found: {images_dir}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"Labels folder not found: {labels_dir}")

    if args.train_ratio <= 0 or args.val_ratio < 0 or (args.train_ratio + args.val_ratio) >= 1.0:
        raise ValueError("Require: train_ratio > 0, val_ratio >= 0, and train_ratio + val_ratio < 1.0")

    all_images = find_images(images_dir)
    if not all_images:
        raise RuntimeError(f"No images found in: {images_dir}")

    matched_images, missing_labels = validate_pairs(all_images, labels_dir)

    print("=== SOURCE SUMMARY ===")
    print(f"Images found       : {len(all_images)}")
    print(f"Matched labels     : {len(matched_images)}")
    print(f"Missing labels     : {len(missing_labels)}")

    if missing_labels and not args.include_unlabeled:
        print("\n[WARN] Some images do not have corresponding label files.")
        print("They will be skipped. Use --include-unlabeled if you want to include them.")
        for p in missing_labels[:10]:
            print(f"  - {p.name}")
        if len(missing_labels) > 10:
            print(f"  ... and {len(missing_labels) - 10} more")

    selected_images = all_images if args.include_unlabeled else matched_images

    if not selected_images:
        raise RuntimeError("No usable images found after filtering.")

    train_items, val_items, test_items = split_list(
        selected_images,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    print("\n=== SPLIT SUMMARY ===")
    print(f"Train: {len(train_items)}")
    print(f"Val  : {len(val_items)}")
    print(f"Test : {len(test_items)}")

    # Tạo cấu trúc output
    ensure_dir(output_dir / "images" / "train")
    ensure_dir(output_dir / "images" / "val")
    ensure_dir(output_dir / "images" / "test")
    ensure_dir(output_dir / "labels" / "train")
    ensure_dir(output_dir / "labels" / "val")
    ensure_dir(output_dir / "labels" / "test")

    copy_split(train_items, labels_dir, output_dir, "train")
    copy_split(val_items, labels_dir, output_dir, "val")
    copy_split(test_items, labels_dir, output_dir, "test")

    yaml_path = write_data_yaml(output_dir, args.classes)

    print("\n=== DONE ===")
    print(f"Dataset created at : {output_dir}")
    print(f"data.yaml          : {yaml_path}")
    print("\nTrain YOLO26 with:")
    print(f"yolo detect train model=../models/yolo26n.pt data={yaml_path} epochs=50 imgsz=640")


if __name__ == "__main__":
    main()