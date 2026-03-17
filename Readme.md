# AI4R – YOLO Drone Detection Pipeline

## 🚀 Overview

This project implements a complete **end-to-end pipeline for drone detection using YOLO26**, following a real-world computer vision workflow.

Core idea:

* Start with **manual labeling**
* Train initial model using **transfer learning**
* Use trained model to **auto-label new data**
* Refine labels in CVAT
* Retrain model
* Repeat until performance is satisfactory

👉 This is a **Human-in-the-loop + Iterative Learning pipeline**

---

## 📁 Project Structure

```
AI4R/
├── frames/                  # Extracted frames from videos
├── models/                  # YOLO pretrained & trained weights
├── prepare_datasets/        # Intermediate dataset (raw from CVAT)
├── dataset/                 # Final dataset for training YOLO
├── runs/                    # Training outputs
├── scripts/                 # All pipeline scripts
├── store_CVAT/              # CVAT export/import
├── videos/                  # Raw videos
├── Readme.md
├── requirements.txt
```

---

# 🔁 FULL PIPELINE

---

## 🟢 STEP 1 — Collect Video Data

Place videos into:

```bash
videos/
```

---

## 🟢 STEP 2 — Extract Frames from Video

```bash
cd scripts

ffmpeg -i ../videos/drone_01.mp4 -vf "fps=3" ../frames/drone_01/frame_%06d.jpg
```

---

# 🔵 FIRST ITERATION (MANUAL LABELING)

---

## 🟢 STEP 3 — Upload Images to CVAT (Manual Labeling)

1. Open CVAT
2. Create a new task → **Object Detection**
3. Upload images from:

```bash
frames/drone_01/
```

4. Create label:

```
drone
```

5. Perform **manual annotation**

👉 This step is critical to create the initial dataset.

---

## 🟢 STEP 4 — Export Dataset from CVAT

Export format:

```
YOLO 1.1
```

Save to:

```bash
store_CVAT/round1/
```

---

## 🟢 STEP 5 — Build YOLO Training Dataset

### 📌 5.1 Organize CVAT Export

CVAT output is **raw YOLO 1.1 format**.

Reorganize into intermediate dataset:

```bash
prepare_datasets/round1/
 ├── images/
 └── labels/
```

Copy data:

```bash
cp -r store_CVAT/round1/images/* prepare_datasets/round1/images/
cp -r store_CVAT/round1/labels/* prepare_datasets/round1/labels/
```

---

### 📌 5.2 Convert to Training Dataset

```bash
cd scripts
python3 build_yolo26_dataset.py
```

---

### 📌 5.3 Output Dataset

```
dataset/drone_round1/
 ├── images/
 │   ├── train/
 │   ├── val/
 │   └── test/
 ├── labels/
 │   ├── train/
 │   ├── val/
 │   └── test/
 └── data.yaml
```

---

### 📌 Key Pipeline

```
CVAT (YOLO 1.1)
        ↓
prepare_datasets/round1/   (raw intermediate)
        ↓
build_yolo26_dataset.py
        ↓
dataset/drone_round1/      (final training dataset)
```

👉 Only `dataset/drone_round1/` is used for training.

---

## 🟢 STEP 6 — Validate Dataset

```bash
python3 check_yolo_dataset.py
```

Fix errors if needed:

```bash
python3 fix_yolo_label_bounds.py
```

---

## 🟢 STEP 7 — Train YOLO (Transfer Learning)

```bash
python3 train_simple.py
```

Output:

```
runs/detect/train-*/weights/best.pt
```

---

## 🟢 STEP 8 — Evaluate Model

```bash
python3 detection_image.py
python3 detection_video.py
```

---

# 🔁 NEXT ITERATIONS (AUTO-LABEL LOOP)

---

## 🟡 STEP 9 — Extract New Frames

```bash
ffmpeg -i ../videos/drone_02.mp4 -vf "fps=3" ../frames/drone_02/frame_%06d.jpg
```

---

## 🟡 STEP 10 — Auto Label Using YOLO

```bash
python3 auto_label_yolo.py
```

👉 Generate labels automatically using trained model

---

## 🟡 STEP 11 — Upload to CVAT (Refinement)

1. Upload images + auto labels
2. Review annotations
3. Fix:

   * missing drones
   * incorrect bounding boxes
   * false positives

👉 Human correction is mandatory

---

## 🟡 STEP 12 — Export Updated Dataset

Export again:

```
YOLO 1.1
```

Save to:

```bash
store_CVAT/round2/
```

---

## 🟡 STEP 13 — Rebuild Dataset

```bash
python3 build_yolo26_dataset.py
```

---

## 🟡 STEP 14 — Validate Dataset

```bash
python3 check_yolo_dataset.py
```

---

## 🟡 STEP 15 — Retrain Model

```bash
python3 train_simple.py
```

---

# 🔁 ITERATION LOOP

```
Video → Frames → (Auto Label) → CVAT Refine
      → Export → Build Dataset → Validate
      → Train → Evaluate → Repeat
```

---

## ⚙️ Installation

```bash
python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

---

## 📊 Training Tips

* Use `imgsz=640` for low GPU
* Use `imgsz=960` for better small-object detection
* Reduce `batch` if CUDA OOM occurs
* Always validate dataset before training

---

## 🧠 Key Concepts

* Transfer Learning
* Human-in-the-loop annotation
* Iterative dataset improvement
* Semi-automatic labeling

---

## 👨‍🏫 For Students

Follow strictly:

1. Manual labeling FIRST
2. Train model
3. Auto-label new data
4. Refine in CVAT
5. Retrain

👉 This is how real-world AI systems are built.

---

## 📌 Notes

* Dataset is not included
* Use your own videos
* Large model files should not be pushed to GitHub

---

## 🔥 Future Work

* Multi-class detection
* Tracking (ByteTrack)
* Real-time deployment
* ROS2 integration

---

## 📎 License

Educational & research use only.
