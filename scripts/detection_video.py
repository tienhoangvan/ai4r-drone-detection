import cv2

from ultralytics import YOLO

# Load the YOLO11 model
model = YOLO("../models/yolo26n_drone_r1_1.pt")

# Open the video file
#video_path = "video_test/phantom_4_large.mp4"
video_path = "../videos/drone_02.mp4"
cap = cv2.VideoCapture(video_path)


# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLO tracking on the frame, persisting tracks between frames

        results = model(frame)
        
        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        scale = 0.6
        resized_frame = cv2.resize(annotated_frame, None, fx=scale, fy=scale)

        # Display the annotated frame
        cv2.imshow("YOLO Detection", resized_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()


