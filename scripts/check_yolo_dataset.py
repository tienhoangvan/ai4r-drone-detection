from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import yaml

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
EPS = 1e-6


def is_image_file(p: Path) -> bool:
    # Step: basic image file filter (extension + regular file)
    return p.is_file() and p.suffix.lower() in IMAGE_EXTS


def read_yaml(yaml_path: Path) -> Dict:
    # Step: load YOLO data.yaml (Ultralytics format)
    with yaml_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_dataset_paths(data_yaml_path: Path, data: Dict) -> Dict[str, Path]:
    # Step: resolve base dataset path and split directories to absolute paths
    base = Path(data.get("path", data_yaml_path.parent)).expanduser()
    if not base.is_absolute():
        base = (data_yaml_path.parent / base).resolve()

    out = {"base": base}
    for split in ("train", "val", "test"):
        rel = data.get(split, None)
        if rel:
            p = Path(rel)
            if not p.is_absolute():
                p = (base / p).resolve()
            out[split] = p
    return out


def image_label_pairs(images_dir: Path, labels_dir: Path) -> Tuple[List[Tuple[Path, Path]], List[Path], List[Path]]:
    # Step: find (image, label) pairs + missing labels + orphan labels
    images = sorted([p for p in images_dir.iterdir() if is_image_file(p)]) if images_dir.exists() else []
    pairs = []
    missing_labels = []

    for img in images:
        lbl = labels_dir / f"{img.stem}.txt"
        if lbl.exists():
            pairs.append((img, lbl))
        else:
            missing_labels.append(img)

    label_files = sorted([p for p in labels_dir.iterdir() if p.is_file() and p.suffix.lower() == ".txt"]) if labels_dir.exists() else []
    image_stems = {img.stem for img in images}
    orphan_labels = [lbl for lbl in label_files if lbl.stem not in image_stems]

    return pairs, missing_labels, orphan_labels


def validate_label_line(line: str, num_classes: int, line_no: int, file_path: Path) -> List[str]:
    # Step: validate one YOLO label line: "cls x_center y_center width height" (all normalized)
    errors = []
    parts = line.strip().split()

    if len(parts) != 5:
        errors.append(f"{file_path}:{line_no} -> expected 5 values, got {len(parts)}")
        return errors

    cls_s, x_s, y_s, w_s, h_s = parts

    # Step: validate class id (must be an integer in [0, num_classes-1])
    try:
        cls_f = float(cls_s)
        cls_i = int(cls_f)
        if abs(cls_f - cls_i) > EPS:
            errors.append(f"{file_path}:{line_no} -> class id must be integer, got {cls_s}")
        if cls_i < 0:
            errors.append(f"{file_path}:{line_no} -> class id must be >= 0, got {cls_i}")
        if cls_i >= num_classes:
            errors.append(f"{file_path}:{line_no} -> class id {cls_i} out of range [0, {num_classes - 1}]")
    except ValueError:
        errors.append(f"{file_path}:{line_no} -> invalid class id: {cls_s}")

    # Step: validate bbox fields as floats in [0,1] (with tolerance)
    vals = []
    for name, s in [("x_center", x_s), ("y_center", y_s), ("width", w_s), ("height", h_s)]:
        try:
            v = float(s)
            vals.append((name, v))
        except ValueError:
            errors.append(f"{file_path}:{line_no} -> invalid float for {name}: {s}")

    if len(vals) == 4:
        d = dict(vals)
        x, y, w, h = d["x_center"], d["y_center"], d["width"], d["height"]

        for name, v in vals:
            if v < -EPS or v > 1 + EPS:
                errors.append(f"{file_path}:{line_no} -> {name}={v} out of [0,1] with tolerance {EPS}")

        if w <= EPS:
            errors.append(f"{file_path}:{line_no} -> width must be > 0, got {w}")
        if h <= EPS:
            errors.append(f"{file_path}:{line_no} -> height must be > 0, got {h}")

        # Step: boundary check (box corners must stay within [0,1] up to EPS)
        x_min = x - w / 2
        x_max = x + w / 2
        y_min = y - h / 2
        y_max = y + h / 2

        if x_min < -EPS or x_max > 1 + EPS or y_min < -EPS or y_max > 1 + EPS:
            errors.append(
                f"{file_path}:{line_no} -> box exceeds image bounds beyond tolerance {EPS}: "
                f"(x_min={x_min:.8f}, y_min={y_min:.8f}, x_max={x_max:.8f}, y_max={y_max:.8f})"
            )

    return errors


def validate_label_file(label_path: Path, num_classes: int) -> Tuple[List[str], List[str]]:
    # Step: validate one label file (hard errors + soft warnings)
    errors = []
    warnings = []

    text = label_path.read_text(encoding="utf-8").strip()
    if not text:
        # Empty label files are allowed but reported as a warning
        warnings.append(f"{label_path} -> empty label file")
        return errors, warnings

    lines = text.splitlines()
    for i, line in enumerate(lines, start=1):
        line = line.strip()
        if not line:
            continue

        line_errors = validate_label_line(line, num_classes, i, label_path)
        errors.extend(line_errors)

        # Step: soft warnings for unusually small/large boxes
        parts = line.split()
        if len(parts) == 5:
            _, _, _, w_s, h_s = parts
            try:
                w = float(w_s)
                h = float(h_s)
                if w < 0.005 or h < 0.005:
                    warnings.append(f"{label_path}:{i} -> very small box (w={w:.6f}, h={h:.6f})")
                if w > 0.9 or h > 0.9:
                    warnings.append(f"{label_path}:{i} -> very large box (w={w:.6f}, h={h:.6f})")
            except ValueError:
                pass

    return errors, warnings


def infer_labels_dir(images_dir: Path) -> Path:
    # Step: infer labels directory from images directory (images/<split> -> labels/<split>)
    parts = list(images_dir.parts)
    if "images" in parts:
        idx = parts.index("images")
        new_parts = parts[:]
        new_parts[idx] = "labels"
        return Path(*new_parts)
    return images_dir.parent.parent / "labels" / images_dir.name


def main() -> None:
    # Step 1: Parse arguments (data.yaml path, print limits)
    parser = argparse.ArgumentParser(description="Check YOLO dataset integrity before training (v2 with tolerance)")
    parser.add_argument(
        "--data",
        type=str,
        default="../dataset/drone_round1/data.yaml",
        help="Path to data.yaml",
    )
    parser.add_argument(
        "--max-print",
        type=int,
        default=50,
        help="Maximum number of errors/warnings to print per category",
    )
    args = parser.parse_args()

    # Step 2: Load and validate the data.yaml
    data_yaml = Path(args.data).resolve()
    if not data_yaml.exists():
        raise FileNotFoundError(f"data.yaml not found: {data_yaml}")

    data = read_yaml(data_yaml)

    # Step 3: Extract class names and determine num_classes
    names = data.get("names", None)
    if names is None:
        raise ValueError("Missing 'names' in data.yaml")

    if isinstance(names, dict):
        class_names = [names[k] for k in sorted(names.keys(), key=lambda x: int(x))]
    elif isinstance(names, list):
        class_names = names
    else:
        raise ValueError("'names' in data.yaml must be a dict or list")

    num_classes = len(class_names)
    if num_classes == 0:
        raise ValueError("No classes found in data.yaml")

    # Step 4: Resolve dataset split directories from data.yaml
    paths = resolve_dataset_paths(data_yaml, data)

    print("=== DATASET CHECK V2 ===")
    print(f"data.yaml    : {data_yaml}")
    print(f"dataset path : {paths['base']}")
    print(f"classes      : {class_names}")
    print(f"tolerance    : {EPS}")

    total_images = 0
    total_pairs = 0
    total_missing_labels = 0
    total_orphan_labels = 0
    total_errors = []
    total_warnings = []

    # Step 5: Iterate through each split and collect validation issues
    for split in ("train", "val", "test"):
        if split not in paths:
            continue

        images_dir = paths[split]
        labels_dir = infer_labels_dir(images_dir)

        print(f"\n--- Split: {split} ---")
        print(f"images dir : {images_dir}")
        print(f"labels dir : {labels_dir}")

        if not images_dir.exists():
            print(f"[ERROR] Missing images directory: {images_dir}")
            total_errors.append(f"Missing images directory: {images_dir}")
            continue

        if not labels_dir.exists():
            print(f"[ERROR] Missing labels directory: {labels_dir}")
            total_errors.append(f"Missing labels directory: {labels_dir}")
            continue

        # Step 5.1: Build pair lists and basic stats
        pairs, missing_labels, orphan_labels = image_label_pairs(images_dir, labels_dir)

        num_images = len([p for p in images_dir.iterdir() if is_image_file(p)])
        total_images += num_images
        total_pairs += len(pairs)
        total_missing_labels += len(missing_labels)
        total_orphan_labels += len(orphan_labels)

        print(f"images         : {num_images}")
        print(f"matched labels : {len(pairs)}")
        print(f"missing labels : {len(missing_labels)}")
        print(f"orphan labels  : {len(orphan_labels)}")

        if missing_labels:
            total_warnings.extend([f"Missing label for image: {p}" for p in missing_labels])

        if orphan_labels:
            total_warnings.extend([f"Orphan label without image: {p}" for p in orphan_labels])

        # Step 5.2: Validate each label file for this split
        for _, lbl in pairs:
            errors, warnings = validate_label_file(lbl, num_classes)
            total_errors.extend(errors)
            total_warnings.extend(warnings)

    # Step 6: Print summary and (optionally) a capped list of issues
    print("\n=== SUMMARY ===")
    print(f"Total images         : {total_images}")
    print(f"Matched image-label  : {total_pairs}")
    print(f"Missing labels       : {total_missing_labels}")
    print(f"Orphan labels        : {total_orphan_labels}")
    print(f"Errors               : {len(total_errors)}")
    print(f"Warnings             : {len(total_warnings)}")

    if total_errors:
        print("\n=== ERRORS ===")
        for msg in total_errors[:args.max_print]:
            print(msg)
        if len(total_errors) > args.max_print:
            print(f"... and {len(total_errors) - args.max_print} more errors")

    if total_warnings:
        print("\n=== WARNINGS ===")
        for msg in total_warnings[:args.max_print]:
            print(msg)
        if len(total_warnings) > args.max_print:
            print(f"... and {len(total_warnings) - args.max_print} more warnings")

    if not total_errors:
        print("\n[OK] Dataset passed hard checks and is ready for YOLO training.")
    else:
        print("\n[FAIL] Fix dataset errors before training.")


if __name__ == "__main__":
    main()