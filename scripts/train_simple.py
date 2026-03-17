from ultralytics import YOLO

# Load model (pretrained)
model = YOLO("../models/yolo26n.pt")

# Train with minimal parameters + batch
results = model.train(
    data="../dataset/drone_round1/data.yaml",
    epochs=100,
    imgsz=640,
    batch=4   # adjust if needed: 2 or 8
)

print("Training completed!")