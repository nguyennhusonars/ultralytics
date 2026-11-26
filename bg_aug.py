import os
import random

import cv2
import numpy as np
import torch

from ultralytics import YOLO

# Load segmentation model and move to GPU
seg_model = YOLO("yolov8m-seg.pt")
seg_model.to("cuda")  # Ensure model is on GPU

# Check GPU availability
print(torch.cuda.is_available())
print(torch.cuda.current_device())
print(torch.cuda.get_device_name(0))


def change_background(image, mask, bg_img):
    # Composite: where mask==1, use image; else use bg_img
    result = np.where(mask[..., None], image, bg_img)
    return result


# Path to your backgrounds folder
backgrounds_folder = "/home/sonnn/Pictures/BingWallpaper"
background_files = [
    os.path.join(backgrounds_folder, f)
    for f in os.listdir(backgrounds_folder)
    if f.lower().endswith((".jpg", ".png", ".jpeg"))
]

# Dataset paths
input_root = "/home/sonnn/Downloads/data_color_current/val"
output_root = "/home/sonnn/Downloads/data_color_current/val_aug"

for label in os.listdir(input_root):
    label_dir = os.path.join(input_root, label)
    if not os.path.isdir(label_dir):
        continue
    output_label_dir = os.path.join(output_root, label)
    os.makedirs(output_label_dir, exist_ok=True)
    for img_name in os.listdir(label_dir):
        if not img_name.lower().endswith((".jpg", ".png", ".jpeg")):
            continue
        img_path = os.path.join(label_dir, img_name)
        image = cv2.imread(img_path)
        results = seg_model(image, device="cuda")
        if not results or not hasattr(results[0], "masks") or results[0].masks is None:
            continue  # Skip if no mask found
        # Find the biggest mask by area
        masks = results[0].masks.data.cpu().numpy().astype(np.uint8)  # shape: (N, H, W)
        if masks.ndim == 2:
            mask = masks  # Only one mask
            area = np.sum(mask)
        else:
            # Compute area for each mask
            areas = [np.sum(m) for m in masks]
            if len(areas) == 0:
                continue
            biggest_idx = np.argmax(areas)
            mask = masks[biggest_idx]
            area = areas[biggest_idx]
        # Skip if the biggest vehicle is too small
        if area < 100000:
            continue
        for i in range(1):  # Generate 2 augmentations per image
            bg_path = random.choice(background_files)
            bg_img = cv2.imread(bg_path)
            if image is None or bg_img is None:
                continue  # Skip if image or background failed to load
            # Resize background to match image size
            bg_img = cv2.resize(bg_img, (image.shape[1], image.shape[0]))
            # Ensure mask shape matches image
            if mask.shape != image.shape[:2]:
                mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
            else:
                mask_resized = mask
            augmented = change_background(image, mask_resized, bg_img)
            base, ext = os.path.splitext(img_name)
            out_name = f"{base}_aug{i + 1}_{area}{ext}"
            out_path = os.path.join(output_label_dir, out_name)
            cv2.imwrite(out_path, augmented)
