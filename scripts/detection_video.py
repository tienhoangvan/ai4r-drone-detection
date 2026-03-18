import cv2

from ultralytics import YOLO

# Step 1: Load the trained YOLO model (weights on disk)
model = YOLO("../models/yolo26n_drone_r1_1.pt")

# Step 2: Open an input video stream (file path)
# video_path = "video_test/phantom_4_large.mp4"
video_path = "../videos/drone_02.mp4"
cap = cv2.VideoCapture(video_path)


# Step 3: Loop through frames until the stream ends or the user quits
while cap.isOpened():
    # Step 3.1: Read the next frame
    success, frame = cap.read()

    if success:
        # Step 3.2: Run inference on the current frame
        results = model(frame)
        
        # Step 3.3: Render predicted bounding boxes/labels on the frame
        annotated_frame = results[0].plot()

        # Step 3.4: Resize for display (optional)
        scale = 0.6
        resized_frame = cv2.resize(annotated_frame, None, fx=scale, fy=scale)

        # Step 3.5: Display the annotated frame
        cv2.imshow("YOLO Detection", resized_frame)

        # Step 3.6: Allow quitting with 'q'
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Step 3.7: End of stream (no more frames)
        break

# Step 4: Release resources and close windows
cap.release()
cv2.destroyAllWindows()


