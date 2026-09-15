import os
from ultralytics import YOLO
import cv2

# Load the pre-trained model
model = YOLO('/home/sonnn/Documents/ultralytics/vehicle-color-cls/yolov8n-color25/weights/best.pt')  # Path to your trained YOLOv8 model

# Path to the dataset folder (all images are in subfolders named by the label)
dataset_folder = '/home/sonnn/Downloads/data_color_current/val'

# First, rename all files in each label folder to ordinal names
for label_folder in os.listdir(dataset_folder):
    label_folder_path = os.path.join(dataset_folder, label_folder)
    if os.path.isdir(label_folder_path):
        images = [img for img in os.listdir(label_folder_path) if img.lower().endswith(('.jpg', '.jpeg', '.png'))]
        images.sort()  # Optional: sort to have consistent renaming
        for idx, img_name in enumerate(images, start=1):
            new_name = f"{idx:05d}.jpg"
            src_path = os.path.join(label_folder_path, img_name)
            dst_path = os.path.join(label_folder_path, new_name)
            if src_path != dst_path:
                os.rename(src_path, dst_path)

# Now, perform inference and rename as before
for label_folder in os.listdir(dataset_folder):
    label_folder_path = os.path.join(dataset_folder, label_folder)

    if os.path.isdir(label_folder_path):  # Only process if it's a folder
        # Get all image files in the current folder
        for image_name in os.listdir(label_folder_path):
            image_path = os.path.join(label_folder_path, image_name)
            
            # Process only image files (you can add more formats if needed)
            if image_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                # Perform inference on the image
                results = model(image_path)  # Get predictions

                # Access the classification result directly
                predictions = results[0].probs  # Class probabilities (confidence scores)
                labels = results[0].names  # Class names (labels)

                # Check if predictions exist
                if predictions is not None and len(predictions.data) > 0:
                    # Get the top predicted class index (top1) and its confidence score
                    best_class_index = predictions.top1  # Class index with the highest probability
                    best_class_label = labels[best_class_index]  # Class label
                    confidence_score = predictions.top1conf.item()  # Confidence score for the top class

                    # Build the new file name with the format <result_label>_<score>_<original_name>.jpg
                    new_image_name = f"{best_class_label}_{confidence_score:.2f}_{image_name}"
                    new_image_path = os.path.join(label_folder_path, new_image_name)

                    # Rename the image (Replace original image)
                    os.rename(image_path, new_image_path)

                    print(f"Renamed '{image_name}' to '{new_image_name}'")

print("Renaming and inference complete!")
