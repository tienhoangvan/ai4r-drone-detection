from ultralytics import YOLO

# Step 1: Load a pretrained YOLO model (base weights)
model = YOLO("../models/yolo26n.pt")

# Step 2: Launch training with minimal configuration
# - data: dataset definition (YOLO data.yaml)
# - epochs/imgsz/batch: core training knobs
results = model.train(
    data="../dataset/drone_round1/data.yaml",
    epochs=100,
    imgsz=640,
    batch=4   # adjust if needed: 2 or 8
)

# Step 3: Print a completion message (training logs are written under runs/)
print("Training completed!")