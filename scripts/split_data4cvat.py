#!/usr/bin/env python3
"""
split_data4cvat.py

Split a mixed folder (images + YOLO labels) into two folders for CVAT import:
- images/ : image files only
- labels/ : label files only (.txt), matched by stem

Default behavior is SAFE: copy files (non-destructive).
Use --move to move files instead of copying.

Example:
  python3 scripts/split_data4cvat.py --src frames/drone_02
  python3 scripts/split_data4cvat.py --src frames/drone_02 --move
  python3 scripts/split_data4cvat.py --src frames/drone_02 --dst frames/drone_02_cvat
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Iterable, List, Set, Tuple

IMAGE_EXTS: Set[str] = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
LABEL_EXT = ".txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split a mixed image+label folder into images/ and labels/ subfolders (for CVAT)."
    )
    parser.add_argument(
        "--src",
        type=str,
        default="frames/drone_02",
        help="Source folder containing mixed images and labels",
    )
    parser.add_argument(
        "--dst",
        type=str,
        default="",
        help="Destination root folder. If omitted, uses --src (creates subfolders inside it).",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="If set, search recursively under --src (rglob). Default: only direct children.",
    )
    parser.add_argument(
        "--move",
        action="store_true",
        help="Move files instead of copying (destructive). Default: copy.",
    )
    parser.add_argument(
        "--include-orphan-labels",
        action="store_true",
        help="If set, also copy/move label files that have no matching image.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print actions without copying/moving any files.",
    )
    return parser.parse_args()


def is_image_file(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMAGE_EXTS


def is_label_file(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() == LABEL_EXT


def iter_files(src_dir: Path, recursive: bool) -> Iterable[Path]:
    if recursive:
        return (p for p in src_dir.rglob("*") if p.is_file())
    return (p for p in src_dir.iterdir() if p.is_file())


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def collect_images_and_labels(src_dir: Path, recursive: bool) -> Tuple[List[Path], List[Path]]:
    files = list(iter_files(src_dir, recursive=recursive))
    images = sorted([p for p in files if is_image_file(p)])
    labels = sorted([p for p in files if is_label_file(p)])
    return images, labels


def copy_or_move(src: Path, dst: Path, move: bool, dry_run: bool) -> None:
    if dry_run:
        print(f"[DRY] {'MOVE' if move else 'COPY'} {src} -> {dst}")
        return

    ensure_dir(dst.parent)
    if move:
        shutil.move(str(src), str(dst))
    else:
        shutil.copy2(src, dst)


def main() -> None:
    args = parse_args()

    # Step 1: Resolve and validate the source directory
    src_dir = Path(args.src).resolve()
    if not src_dir.exists() or not src_dir.is_dir():
        raise FileNotFoundError(f"Source folder not found or not a directory: {src_dir}")

    # Step 2: Decide destination root and create images/labels subfolders
    dst_root = Path(args.dst).resolve() if args.dst else src_dir
    images_out = dst_root / "images"
    labels_out = dst_root / "labels"

    if not args.dry_run:
        ensure_dir(images_out)
        ensure_dir(labels_out)

    # Step 3: Collect images and labels
    images, labels = collect_images_and_labels(src_dir, recursive=args.recursive)
    label_by_stem = {p.stem: p for p in labels}
    image_stems = {p.stem for p in images}

    # Step 4: For each image, copy/move the image and its matched label (if present)
    n_img = 0
    n_lbl = 0
    n_missing_lbl = 0

    for img_path in images:
        img_dst = images_out / img_path.name
        copy_or_move(img_path, img_dst, move=args.move, dry_run=args.dry_run)
        n_img += 1

        lbl_path = label_by_stem.get(img_path.stem, None)
        if lbl_path is None:
            n_missing_lbl += 1
            continue

        lbl_dst = labels_out / lbl_path.name
        copy_or_move(lbl_path, lbl_dst, move=args.move, dry_run=args.dry_run)
        n_lbl += 1

    # Step 5: Optionally handle orphan labels (labels without any matching image)
    n_orphan_lbl = 0
    if args.include_orphan_labels:
        for lbl_path in labels:
            if lbl_path.stem in image_stems:
                continue
            lbl_dst = labels_out / lbl_path.name
            copy_or_move(lbl_path, lbl_dst, move=args.move, dry_run=args.dry_run)
            n_orphan_lbl += 1

    # Step 6: Report summary
    print("\n=== SPLIT FOR CVAT ===")
    print(f"src           : {src_dir}")
    print(f"dst_root      : {dst_root}")
    print(f"images_out    : {images_out}")
    print(f"labels_out    : {labels_out}")
    print(f"mode          : {'MOVE' if args.move else 'COPY'}{' (dry-run)' if args.dry_run else ''}")
    print(f"recursive     : {args.recursive}")
    print(f"images_found  : {len(images)}")
    print(f"labels_found  : {len(labels)}")
    print(f"images_written: {n_img}")
    print(f"labels_written: {n_lbl}")
    print(f"missing_labels_for_images: {n_missing_lbl}")
    if args.include_orphan_labels:
        print(f"orphan_labels_written    : {n_orphan_lbl}")


if __name__ == "__main__":
    main()

