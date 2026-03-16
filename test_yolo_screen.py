import cv2
import numpy as np
from ultralytics import YOLO
import mss

# Load the YOLOv8 model (replace with your model path)
model = YOLO("/home/sonnn/Documents/ultralytics_sonnn/best_swalk_5.3GF.pt")

# Define the monitor for "Display 1" (change this based on your system setup)
# You can list all displays using mss and selecting the correct one
with mss.mss() as sct:
    monitor = sct.monitors[2]  # Typically 1 is the primary monitor, change if needed

    while True:
        # Capture the screen content of the specified monitor (Display 1)
        screenshot = sct.grab(monitor)
        
        # Convert the screenshot to a numpy array (BGR format for OpenCV)
        frame = np.array(screenshot)
        
        # Convert from RGBA (mss) to BGR (OpenCV)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
        
        # Run YOLO object detection on the captured frame
        results = model(frame)
        
        # Render the results (boxes, labels, tracking IDs)
        frame_with_annotations = results[0].plot(line_width=1, font_size=0.5)

        frame_with_annotations = cv2.resize(frame_with_annotations, (1920, 1080))
    
        # Display the frame with YOLO annotations
        cv2.imshow("YOLO Prediction on Display 1", frame_with_annotations)

        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Cleanup
cv2.destroyAllWindows()

