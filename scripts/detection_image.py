import cv2
from ultralytics import YOLO

# Load the YOLO model
model = YOLO("../models/yolo26n_drone_r1_1.pt")
# model = YOLO("../models/yolo26n.pt")

# Image path
image_path = "../dataset/drone_round1/images/test/frame_000389.jpg"
image = cv2.imread(image_path)

if image is None:
    raise FileNotFoundError(f"Cannot read image: {image_path}")

# Run detection on the image
results = model(image)

# Visualize the results on the image
annotated_image = results[0].plot()

scale = 0.8
resized_image = cv2.resize(annotated_image, None, fx=scale, fy=scale)

# Display the annotated image
cv2.imshow("YOLO Detection on Image", resized_image)

# Wait until a key is pressed, then close
cv2.waitKey(0)
cv2.destroyAllWindows()