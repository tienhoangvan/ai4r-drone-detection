"""
merge_datasets_multi_round.py

Merge multiple YOLO dataset rounds into a single YOLO-style dataset:
- Copies images/labels from each round into one output folder
- Prefixes filenames with the round folder name to avoid collisions
- Generates a minimal `data.yaml` for Ultralytics training

Run (one-line example):
  python3 merge_datasets_multi_round.py --rounds ../dataset/drone_round1 ../dataset/drone_round2 --output ../dataset/drone_merged
"""

import shutil
from pathlib import Path
import argparse
import yaml
from typing import Dict, Iterable, List, Optional, Set, Tuple
import re

# ---------------------------
# CONFIG
# ---------------------------

# Step 0: Define the expected YOLO split names (must match folder layout).
SPLITS = ["train", "val", "test"]
IMAGE_EXTS: Set[str] = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
LABEL_EXT = ".txt"

# ---------------------------
# UTIL
# ---------------------------

def safe_copy(src, dst):
    # Step: Ensure destination folder exists and copy while preserving metadata.
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def add_prefix(filename, prefix):
    # Step: Create a collision-free filename by adding a round-specific prefix.
    return f"{prefix}_{filename}"


def normalize_round_prefix(round_folder_name: str) -> str:
    """
    Convert round folder name into a short prefix.

    Example:
      "drone_round1" -> "r1"
      "round2"        -> "r2"

    If no known pattern is found, fall back to the original folder name.
    """
    # Prefer patterns like "<anything>_round<digits>" or "round<digits>"
    m = re.search(r"(?:^|[_-])round(\d+)(?:$|[_-])", round_folder_name)
    if m:
        # Strip leading zeros by converting to int.
        return f"r{int(m.group(1))}"

    # Also accept already-short names like "r1"
    m2 = re.fullmatch(r"r(\d+)", round_folder_name.strip(), flags=0)
    if m2:
        return f"r{int(m2.group(1))}"

    return round_folder_name

def is_image_file(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMAGE_EXTS


def is_label_file(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() == LABEL_EXT


def iter_images(dir_path: Path) -> Iterable[Path]:
    # Step: Iterate only over valid image files (prevents copying labels/other artifacts by mistake).
    return (p for p in dir_path.iterdir() if is_image_file(p))


def iter_labels(dir_path: Path) -> Iterable[Path]:
    return (p for p in dir_path.iterdir() if is_label_file(p))


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent


def _candidate_roots() -> List[Path]:
    # Step: Search roots for resolving relative paths (cwd first, then script-relative, then repo root).
    return [Path.cwd(), SCRIPT_DIR, REPO_ROOT]


def resolve_path(p: str) -> Path:
    """
    Resolve a path string to an absolute Path.

    Rules:
    - Absolute paths are returned as-is (expanded + resolved).
    - Relative paths are resolved against the current working directory.
      If the resulting path does NOT exist, we try resolving against:
        - the scripts/ directory (SCRIPT_DIR)
        - the repository root (REPO_ROOT)
    This makes CLI usage robust when users run from different working directories.
    """
    raw = Path(p).expanduser()
    if raw.is_absolute():
        return raw.resolve()

    # First attempt: relative to current working directory (default Python behavior).
    primary = (Path.cwd() / raw).resolve()
    if primary.exists():
        return primary

    # Fallbacks: script-relative and repo-root-relative.
    for root in _candidate_roots()[1:]:
        cand = (root / raw).resolve()
        if cand.exists():
            return cand

    # Nothing exists yet; return the primary resolution (useful for output paths we will create).
    return primary


def resolve_existing_dir(p: str, kind: str) -> Optional[Path]:
    """
    Resolve a path that MUST point to an existing directory.
    Returns None if no candidate exists.
    """
    raw = Path(p).expanduser()
    if raw.is_absolute():
        cand = raw.resolve()
        return cand if cand.is_dir() else None

    for root in _candidate_roots():
        cand = (root / raw).resolve()
        if cand.is_dir():
            return cand
    return None


def detect_round_layout(round_path: Path) -> Tuple[Dict[str, Path], Dict[str, Path]]:
    """
    Detect where images and labels live for each split inside a round.

    Supported layouts:
      A) YOLO split layout (preferred):
         round/images/train, round/images/val, round/images/test
         round/labels/train, round/labels/val, round/labels/test
      B) Flat layout under images/ and labels/ (treated as train):
         round/images/*.{jpg,png,...}
         round/labels/*.txt
      C) Fully mixed folder (treated as train):
         round/*.{jpg,png,...} and round/*.txt
    """
    img_dirs: Dict[str, Path] = {}
    lbl_dirs: Dict[str, Path] = {}

    # Layout A: split folders
    has_split_images = any((round_path / "images" / s).is_dir() for s in SPLITS)
    has_split_labels = any((round_path / "labels" / s).is_dir() for s in SPLITS)
    if has_split_images and has_split_labels:
        for s in SPLITS:
            img_dirs[s] = round_path / "images" / s
            lbl_dirs[s] = round_path / "labels" / s
        return img_dirs, lbl_dirs

    # Layout B: flat images/ and labels/ (map to train)
    flat_images = round_path / "images"
    flat_labels = round_path / "labels"
    if flat_images.is_dir() and flat_labels.is_dir():
        if any(iter_images(flat_images)) or any(iter_labels(flat_labels)):
            img_dirs["train"] = flat_images
            lbl_dirs["train"] = flat_labels
            return img_dirs, lbl_dirs

    # Layout C: mixed directly in round folder (map to train)
    if any(is_image_file(p) for p in round_path.iterdir() if p.is_file()) or any(
        is_label_file(p) for p in round_path.iterdir() if p.is_file()
    ):
        img_dirs["train"] = round_path
        lbl_dirs["train"] = round_path
        return img_dirs, lbl_dirs

    return img_dirs, lbl_dirs


# ---------------------------
# MERGE LOGIC
# ---------------------------

def merge_rounds(round_paths, output_path):
    # Step 1: Normalize output path and create the output folder structure.
    output_path = resolve_path(str(output_path))
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Output dataset root: {output_path}")

    for split in SPLITS:
        # Step 1.1: Create `images/<split>` and `labels/<split>` folders.
        (output_path / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_path / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Step 2: Iterate through each dataset round and copy files into the merged output.
    total_images_copied = 0
    total_labels_copied = 0
    total_images_missing_labels = 0
    processed_rounds = 0

    for round_path in round_paths:
        resolved_round = resolve_existing_dir(str(round_path), kind="round")
        if resolved_round is None:
            # Give the user a helpful hint about how the path was interpreted.
            primary = (Path.cwd() / Path(str(round_path)).expanduser()).resolve()
            script_rel = (SCRIPT_DIR / Path(str(round_path)).expanduser()).resolve()
            repo_rel = (REPO_ROOT / Path(str(round_path)).expanduser()).resolve()
            print(f"[WARN] Round folder not found (skipping): {round_path}")
            print("       Tried:")
            print(f"         - cwd      : {primary}")
            print(f"         - scripts/  : {script_rel}")
            print(f"         - repo root : {repo_rel}")
            continue
        round_path = resolved_round
        round_name = round_path.name  # e.g. drone_round1
        round_prefix = normalize_round_prefix(round_name)  # e.g. r1

        print(f"\n[INFO] Processing {round_name} (prefix: {round_prefix})")
        processed_rounds += 1

        # Step 2.1: Detect where split folders are for this round (supports multiple layouts).
        img_dirs, lbl_dirs = detect_round_layout(round_path)
        if not img_dirs:
            print(f"[WARN] No supported dataset layout detected under: {round_path} (skipping)")
            continue

        used_splits = sorted(img_dirs.keys())
        print(f"[INFO] Detected splits for {round_name}: {used_splits}")

        for split, img_dir in img_dirs.items():
            # Step 2.2: Locate the label directory for this split if available.
            lbl_dir = lbl_dirs.get(split, None)
            if not img_dir.exists():
                continue
            if lbl_dir is None or not lbl_dir.exists():
                print(f"[WARN] Missing labels dir for {round_name}/{split}: {lbl_dir}")

            # Step 2.3: For each image, copy it with a prefixed name to avoid collisions.
            images_in_split = list(iter_images(img_dir))
            if not images_in_split:
                continue

            # If layout B/C mapped everything to "train", we still write into the corresponding output split.
            out_split = split if split in SPLITS else "train"

            copied_images_this_split = 0
            copied_labels_this_split = 0

            for img_file in sorted(images_in_split):
                new_name = add_prefix(img_file.name, round_prefix)

                dst_img = output_path / "images" / out_split / new_name
                safe_copy(img_file, dst_img)
                total_images_copied += 1
                copied_images_this_split += 1

                # Step 2.4: Copy the corresponding YOLO label file if it exists (<stem>.txt).
                if lbl_dir is not None and lbl_dir.exists():
                    label_file = lbl_dir / (img_file.stem + ".txt")
                    if label_file.exists():
                        # Note: We also prefix the label filename so it matches the renamed image stem.
                        dst_lbl = output_path / "labels" / out_split / (Path(new_name).stem + ".txt")
                        safe_copy(label_file, dst_lbl)
                        total_labels_copied += 1
                        copied_labels_this_split += 1
                    else:
                        total_images_missing_labels += 1
                else:
                    total_images_missing_labels += 1

            print(
                f"[INFO] {round_name}/{split}: copied {copied_images_this_split} images, "
                f"{copied_labels_this_split} labels -> output split '{out_split}'"
            )

    print("\n[INFO] Merge completed!")
    print(f"[INFO] Rounds processed             : {processed_rounds}/{len(round_paths)}")
    print(f"[INFO] Total images copied          : {total_images_copied}")
    print(f"[INFO] Total labels copied          : {total_labels_copied}")
    print(f"[INFO] Images without matching label: {total_images_missing_labels}")


# ---------------------------
# DATA.YAML
# ---------------------------

def create_data_yaml(output_path, class_names):
    # Step 3: Create a minimal Ultralytics/YOLO `data.yaml` for the merged dataset.
    data_yaml = {
        "path": str(resolve_path(str(output_path))),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": class_names
    }

    yaml_path = resolve_path(str(output_path)) / "data.yaml"

    # Step 3.1: Write data.yaml to the merged dataset root.
    with open(yaml_path, "w") as f:
        yaml.safe_dump(data_yaml, f, sort_keys=False)

    print(f"[INFO] data.yaml created at {yaml_path}")


# ---------------------------
# MAIN
# ---------------------------

def main():
    # Step 4: Parse CLI arguments (round list, output path, class names).
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rounds",
        nargs="+",
        required=True,
        help="List of dataset round folders"
    )
    parser.add_argument(
        "--output",
        default="../dataset/drone_merged",
        help="Output merged dataset"
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        default=["drone"],
        help="Class names"
    )

    args = parser.parse_args()

    # Step 5: Merge all provided rounds and generate `data.yaml`.
    merge_rounds(args.rounds, args.output)
    create_data_yaml(args.output, args.classes)


if __name__ == "__main__":
    main()