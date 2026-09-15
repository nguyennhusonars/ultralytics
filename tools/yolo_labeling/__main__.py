"""Entry point: `python -m yolo_labeling [-i ...] [-l ...] [-y ...] [-m ...]`."""

import argparse
import tkinter as tk

from yolo_labeling.app import YOLOLabelingTool


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='YOLO Labeling Tool')
    parser.add_argument('--image-folder', '-i', help='Path to image folder')
    parser.add_argument('--label-folder', '-l', help='Path to label folder')
    parser.add_argument('--model-path', '-m', help='Path to YOLO model (.pt file)')
    parser.add_argument('--yaml-path', '-y', help='Path to YOLO data.yaml file (for class names and paths)')
    args = parser.parse_args()
    
    root = tk.Tk()
    YOLOLabelingTool(root, 
                          image_folder=args.image_folder, 
                          label_folder=args.label_folder, 
                          model_path=args.model_path,
                          yaml_path=args.yaml_path)
    root.mainloop()

if __name__ == "__main__":
    main() 
