import cv2
from ultralytics import YOLO

# Step 1: Load the trained YOLO model (weights on disk)
model = YOLO("../models/yolo26n_drone_r1_1.pt")
# model = YOLO("../models/yolo26n.pt")

# Step 2: Read an input image from the dataset (BGR via OpenCV)
image_path = "../dataset/drone_round1/images/test/frame_000389.jpg"
image = cv2.imread(image_path)

# Step 3: Fail fast if the image cannot be loaded (wrong path / missing file)
if image is None:
    raise FileNotFoundError(f"Cannot read image: {image_path}")

# Step 4: Run inference on the image (returns a list of results)
results = model(image)

# Step 5: Render predicted bounding boxes/labels onto the image
annotated_image = results[0].plot()

# Step 6: Resize for display (optional)
scale = 0.8
resized_image = cv2.resize(annotated_image, None, fx=scale, fy=scale)

# Step 7: Display and wait for a key press to exit
cv2.imshow("YOLO Detection on Image", resized_image)

cv2.waitKey(0)
cv2.destroyAllWindows()