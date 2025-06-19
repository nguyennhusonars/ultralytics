import cv2
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO("/home/sonnn/Documents/ultralytics/runs/detect/yolo11custom_scratch_vehicleonly_480/train3/weights/best.pt")

# Open RTSP stream
rtsp_url = "rtsp://192.169.1.251/stream1"
cap = cv2.VideoCapture(rtsp_url)

if not cap.isOpened():
    print("Error: Unable to open RTSP stream.")
    exit()

# Tracking settings
tracker_config = {
    'tracker_type': 'bytetrack'  # default is bytetrack; no need to define unless using others
}

while True:
    # Read frame from RTSP stream
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to grab frame.")
        break
    
    # Perform tracking
    results = model.track(source=frame, persist=True, conf=0.05, imgsz=480)

    # Annotate frame with results (boxes, labels, tracking IDs)
    frame_with_annotations = results[0].plot()

    # Resize the frame to 720p
    frame_with_annotations = cv2.resize(frame_with_annotations, (1280, 720))

    # Display the frame
    cv2.imshow("RTSP Stream with YOLOv8 Tracking", frame_with_annotations)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
