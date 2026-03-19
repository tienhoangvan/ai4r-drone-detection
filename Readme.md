🚀 AI4R – YOLO Drone Detection Pipeline (Full Iterative Workflow)
🧠 Overview

This project implements a complete real-world pipeline for drone detection using YOLO26, following a data-centric + iterative learning approach.

Core idea:

Manual Label → Train → Auto Label → Refine → Retrain → Repeat

👉 This is a Human-in-the-loop + Iterative AI system

📁 Project Structure
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
│   │   ├── images/
│   │   └── labels/
│   └── round2/
│       ├── images/
│       └── labels/
│
├── runs/detect/
│   ├── train/
│   ├── train-r1/
│   └── train-r2/
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
│   ├── train_optimized.py
│   ├── train_simple.py
│   └── yolo_to_coco.py
│
├── store_CVAT/
│   ├── round1_*.zip
│   └── round2_*.zip
│
├── videos/
│
├── .gitignore
├── Readme.md
└── requirements.txt
🔁 FULL PIPELINE
🟢 STEP 1 — Collect Video Data

Put raw videos into:

videos/
🟢 STEP 2 — Extract Frames
ffmpeg -i videos/drone_01.mp4 -vf "fps=3" frames/drone_01/frame_%06d.jpg
🔵 FIRST ITERATION (MANUAL LABELING)
🟢 STEP 3 — Upload to CVAT (Manual Labeling)

Create task (Object Detection)

Upload images from:

frames/drone_01/

Create label:

drone

Annotate manually

👉 Đây là bước quan trọng nhất

🟢 STEP 4 — Export from CVAT

Format:

YOLO 1.1

Save to:

store_CVAT/round1_*.zip
🟢 STEP 5 — Prepare Intermediate Dataset
mkdir -p prepare_datasets/round1/images
mkdir -p prepare_datasets/round1/labels

unzip store_CVAT/round1_*.zip -d temp_round1
cp -r temp_round1/images/* prepare_datasets/round1/images/
cp -r temp_round1/labels/* prepare_datasets/round1/labels/
🟢 STEP 6 — Build YOLO Training Dataset
cd scripts

python3 build_yolo26_dataset.py \
    --input ../prepare_datasets/round1 \
    --output ../dataset/drone_round1
🟢 STEP 7 — Validate Dataset
python3 check_yolo_dataset.py \
    --data ../dataset/drone_round1/data.yaml

Fix nếu cần:

python3 fix_yolo_label_bounds.py
🟢 STEP 8 — Train Model (Round 1)
python3 train_simple.py \
    --model ../models/yolo26n.pt \
    --data ../dataset/drone_round1/data.yaml \
    --epochs 100 \
    --batch 4

Output:

runs/detect/train-r1/
🟢 STEP 9 — Evaluate Model
python3 detection_image.py
python3 detection_video.py
🔁 NEXT ITERATIONS (AUTO LABEL LOOP)
🟡 STEP 10 — Extract New Frames
ffmpeg -i videos/drone_02.mp4 -vf "fps=3" frames/drone_02/frame_%06d.jpg
🟡 STEP 11 — Auto Label
python3 auto_label_yolo.py \
    --weights ../runs/detect/train-r1/weights/best.pt \
    --source ../frames/drone_02 \
    --output ../prepare_datasets/round2
🟡 STEP 12 — Refine in CVAT

Upload images + labels

Fix:

missing objects

wrong bbox

false positives

🟡 STEP 13 — Export Round 2
store_CVAT/round2_*.zip
🟡 STEP 14 — Prepare Round 2 Dataset
mkdir -p prepare_datasets/round2/images
mkdir -p prepare_datasets/round2/labels

unzip store_CVAT/round2_*.zip -d temp_round2
cp -r temp_round2/images/* prepare_datasets/round2/images/
cp -r temp_round2/labels/* prepare_datasets/round2/labels/
🟡 STEP 15 — Build Dataset Round 2
python3 build_yolo26_dataset.py \
    --input ../prepare_datasets/round2 \
    --output ../dataset/drone_round2
🟡 STEP 16 — Merge Multi-Round Dataset
python3 merge_datasets_multi_round.py \
    --inputs ../dataset/drone_round1 ../dataset/drone_round2 \
    --output ../dataset/drone_merged
🟡 STEP 17 — Retrain (Round 2)
python3 train_simple.py \
    --model ../runs/detect/train-r1/weights/best.pt \
    --data ../dataset/drone_merged/data.yaml \
    --epochs 100 \
    --batch 4
🔁 ITERATION LOOP
Video → Frames → Auto Label → CVAT Refine
      → Build Dataset → Merge → Train → Evaluate → Repeat
⚙️ Installation
python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
📊 Training Tips

imgsz=640 → fast

imgsz=960 → better for small drone

batch nhỏ nếu GPU yếu

luôn validate dataset trước khi train

🧠 Key Concepts

Transfer Learning

Data-Centric AI

Human-in-the-loop

Iterative Training

Semi-auto labeling

👨‍🏫 For Students (VERY IMPORTANT)

Follow strictly:

1. Manual labeling FIRST
2. Train initial model
3. Auto-label new data
4. Fix labels in CVAT
5. Merge datasets
6. Retrain
7. Repeat

👉 Đây chính là cách build AI system ngoài thực tế

📌 Notes

Không push dataset lớn lên GitHub

Không push file .pt lớn (>100MB)

Sử dụng .gitignore

🔥 Extensions

Tracking (ByteTrack)

ROS2 integration

Real-time detection

Multi-class detection

📎 License
Educational & research use only.