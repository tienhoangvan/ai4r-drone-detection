# 🚀 AI4R -- YOLO Drone Detection Pipeline (Production-Ready README)

## 🧠 Overview

This repository provides a **complete, real-world, data-centric
pipeline** for drone detection using YOLO (YOLO26-style workflow).

It is designed for: - 🎓 Teaching (AI4R course) - 🧪 Mini-project /
lab - 🔬 Research prototyping

------------------------------------------------------------------------

## 🔁 Core Idea

``` text
Manual Label → Train → Auto Label → Refine → Merge → Retrain → Repeat
```

👉 This is a **Human-in-the-loop + Iterative Learning System**

------------------------------------------------------------------------

## 📁 Project Structure

    AI4R/
    ├── dataset/
    │   ├── drone_round1/
    │   ├── drone_round2/
    │   └── drone_merged/
    │
    ├── frames/
    │
    ├── models/
    │   ├── yolo26n.pt
    │   ├── yolo26n_drone_r1.pt
    │   └── yolo26n_drone_r2.pt
    │
    ├── prepare_datasets/
    │   ├── round1/
    │   └── round2/
    │
    ├── runs/detect/
    │
    ├── scripts/
    │   ├── auto_label_yolo.py
    │   ├── build_yolo26_dataset.py
    │   ├── check_yolo_dataset.py
    │   ├── detection_image.py
    │   ├── detection_video.py
    │   ├── fix_yolo_label_bounds.py
    │   ├── merge_datasets_multi_round.py
    │   ├── split_data4cvat.py
    │   ├── train_simple.py
    │   ├── train_optimized.py
    │   └── yolo_to_coco.py
    │
    ├── store_CVAT/
    ├── videos/
    ├── Readme.md
    └── requirements.txt

------------------------------------------------------------------------

# ⚙️ Installation

``` bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

------------------------------------------------------------------------

# 🟢 STEP-BY-STEP PIPELINE

------------------------------------------------------------------------

## 🟢 STEP 1 --- Collect Video Data

``` bash
mkdir -p videos
# Copy your drone videos here
```

------------------------------------------------------------------------

## 🟢 STEP 2 --- Extract Frames

``` bash
ffmpeg -i videos/drone_01.mp4 -vf "fps=3" frames/drone_01/frame_%06d.jpg
```

------------------------------------------------------------------------

# 🔵 FIRST ITERATION (MANUAL LABELING)

------------------------------------------------------------------------

## 🟢 STEP 3 --- Manual Labeling (CVAT)

-   Upload `frames/drone_01/`
-   Create label: `drone`
-   Annotate bounding boxes

------------------------------------------------------------------------

## 🟢 STEP 4 --- Export Dataset

-   Format: **YOLO 1.1**
-   Save to:

``` bash
store_CVAT/round1.zip
```

------------------------------------------------------------------------

## 🟢 STEP 5 --- Prepare Dataset (Round 1)

``` bash
mkdir -p prepare_datasets/round1/images
mkdir -p prepare_datasets/round1/labels

unzip store_CVAT/round1.zip -d temp_r1
cp -r temp_r1/images/* prepare_datasets/round1/images/
cp -r temp_r1/labels/* prepare_datasets/round1/labels/
```

------------------------------------------------------------------------

## 🟢 STEP 6 --- Build YOLO Dataset

``` bash
python scripts/build_yolo26_dataset.py \
  --input prepare_datasets/round1 \
  --output dataset/drone_round1
```

------------------------------------------------------------------------

## 🟢 STEP 7 --- Validate Dataset

``` bash
python scripts/check_yolo_dataset.py \
  --data dataset/drone_round1/data.yaml
```

Fix if needed:

``` bash
python scripts/fix_yolo_label_bounds.py
```

------------------------------------------------------------------------

## 🟢 STEP 8 --- Train Model (Round 1)

``` bash
python scripts/train_simple.py \
  --model models/yolo26n.pt \
  --data dataset/drone_round1/data.yaml \
  --epochs 100 \
  --batch 4
```

------------------------------------------------------------------------

## 🟢 STEP 9 --- Evaluate

``` bash
python scripts/detection_image.py
python scripts/detection_video.py
```

------------------------------------------------------------------------

# 🟡 SECOND ITERATION (AUTO LABEL LOOP)

------------------------------------------------------------------------

## 🟡 STEP 10 --- Extract New Frames

``` bash
ffmpeg -i videos/drone_02.mp4 -vf "fps=3" frames/drone_02/frame_%06d.jpg
```

------------------------------------------------------------------------

## 🟡 STEP 11 --- Auto Label

``` bash
python scripts/auto_label_yolo.py \
  --weights runs/detect/train-r1/weights/best.pt \
  --source frames/drone_02 \
  --output prepare_datasets/round2
```

------------------------------------------------------------------------

## 🟡 STEP 12 --- Refine in CVAT

-   Upload images + labels
-   Fix errors manually

------------------------------------------------------------------------

## 🟡 STEP 13 --- Export Round 2

``` bash
store_CVAT/round2.zip
```

------------------------------------------------------------------------

## 🟡 STEP 14 --- Prepare Round 2

``` bash
mkdir -p prepare_datasets/round2/images
mkdir -p prepare_datasets/round2/labels

unzip store_CVAT/round2.zip -d temp_r2
cp -r temp_r2/images/* prepare_datasets/round2/images/
cp -r temp_r2/labels/* prepare_datasets/round2/labels/
```

------------------------------------------------------------------------

## 🟡 STEP 15 --- Build Dataset Round 2

``` bash
python scripts/build_yolo26_dataset.py \
  --input prepare_datasets/round2 \
  --output dataset/drone_round2
```

------------------------------------------------------------------------

## 🟡 STEP 16 --- Merge Datasets

``` bash
python scripts/merge_datasets_multi_round.py \
  --inputs dataset/drone_round1 dataset/drone_round2 \
  --output dataset/drone_merged
```

------------------------------------------------------------------------

## 🟡 STEP 17 --- Retrain

``` bash
python scripts/train_simple.py \
  --model runs/detect/train-r1/weights/best.pt \
  --data dataset/drone_merged/data.yaml \
  --epochs 100 \
  --batch 4
```

------------------------------------------------------------------------

# 🔁 ITERATION LOOP

``` text
Frames → Auto Label → CVAT Fix → Build → Merge → Train → Evaluate → Repeat
```

------------------------------------------------------------------------

# 🧠 Script Mapping

  Script                          Purpose
  ------------------------------- --------------------
  auto_label_yolo.py              Auto annotation
  build_yolo26_dataset.py         Build YOLO dataset
  check_yolo_dataset.py           Validate labels
  fix_yolo_label_bounds.py        Fix bbox errors
  merge_datasets_multi_round.py   Merge datasets
  train_simple.py                 Training
  detection_image.py              Test image
  detection_video.py              Test video

------------------------------------------------------------------------

# 📊 Best Practices

-   Always validate dataset before training
-   Use merged dataset (NOT only new data)
-   Fine-tune from previous weights

------------------------------------------------------------------------

# 📌 Notes

-   Do NOT push large datasets or weights
-   Use `.gitignore`

------------------------------------------------------------------------

# 📎 License

Educational / Research use only
