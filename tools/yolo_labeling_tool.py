import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
import cv2
from PIL import Image, ImageTk
import numpy as np
from ultralytics import YOLO
import os
from pathlib import Path
import json
import random
import argparse
import time
import yaml
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import seaborn as sns

class YOLOLabelingTool:
    def __init__(self, root, image_folder=None, label_folder=None, model_path=None, yaml_path=None):
        self.root = root
        self.root.title("YOLO Labeling Tool")
        
        # Variables
        self.current_image_path = None
        self.current_image = None
        self.current_image_index = -1
        self.image_paths = []
        self.filtered_image_paths = []  # Store filtered image paths
        self.image_folder = image_folder
        self.label_folder = label_folder
        self.pretrained_model = None
        self.current_class = 0
        self.drawing = False
        self.start_x = None
        self.start_y = None
        self.current_rect = None
        self.user_boxes = []  # [(class_id, x1, y1, x2, y2), ...]
        self.label_boxes = []  # [(class_id, x1, y1, x2, y2), ...]
        self.model_boxes = []  # [(class_id, x1, y1, x2, y2), ...]
        self.selected_prediction = None  # Index of currently selected prediction
        self.selected_prediction_rect = None  # Rectangle for selected prediction
        self.selected_box_source = None  # Source of selected box ('user' or 'label')
        self.class_buttons = {}  # Store class buttons for updating
        self.class_labels = {}  # Store class label widgets
        
        # Filter variables
        self.filter_by_class_enabled = False
        self.filter_class_id = 0
        self.filter_by_size_enabled = False
        self.filter_min_size = 0  # In pixels (area)
        self.filter_max_size = 100000  # In pixels (area)
        self.filter_by_iou_enabled = False
        self.filter_iou_threshold = 0.5
        self.filter_by_name_enabled = False
        self.filter_name_text = ""
        self.filter_name_case_sensitive = False
        self.filter_by_overlap_enabled = False
        self.filter_overlap_threshold = 0.5  # Minimum IOU for considering boxes overlapping
        
        # Multi-class presence/absence filter
        self.filter_by_class_presence_enabled = False
        self.filter_must_have_classes = set()  # Set of class IDs that must be present
        self.filter_must_not_have_classes = set()  # Set of class IDs that must NOT be present
        
        # Crosshair cursor state
        self.crosshair_h = None  # Horizontal line ID
        self.crosshair_v = None  # Vertical line ID
        
        # Cache for faster filtering
        self.image_dimensions_cache = {}  # {img_path: (width, height)}
        self.labels_cache = {}  # {img_path: [(cls_id, x1, y1, x2, y2), ...]}
        self.model_predictions_cache = {}  # {img_path: [(cls_id, x1, y1, x2, y2), ...]}
        
        # Batch view variables
        self.batch_mode = False
        self.batch_images = []  # List of images in current batch
        self.batch_canvases = []  # List of canvases for batch view
        self.batch_frame = None  # Frame to hold batch view
        self.editing_batch_image = None  # Index of image being edited in batch mode
        self.batch_size = (1, 1)  # Current batch size (rows, cols)
        
        # Zoom related variables
        self.zoom_scale = 1.0
        self.min_zoom = 0.1
        self.max_zoom = 5.0
        self.zoom_factor = 1.1  # Zoom in/out factor per scroll
        self.image_tk = None  # Store the zoomed PhotoImage
        
        # Class visibility controls
        self.class_visibility = {}  # Dictionary to store visibility state for each class
        self.class_checkboxes = {}  # Store checkbox variables
        
        # Box visibility controls
        self.show_model_predictions = True  # Toggle for model predictions
        self.show_label_boxes = True  # Toggle for label file boxes
        self.show_user_boxes = True  # Toggle for user-drawn boxes
        
        # Copy/Paste clipboard for labels
        self.label_clipboard = []  # Store copied labels: [(cls_id, x1, y1, x2, y2), ...]
        
        # Overlap highlighting
        self.highlight_overlaps = False  # Toggle to highlight overlapping boxes
        self.overlapping_boxes = []  # Store indices of overlapping boxes for highlighting
        
        # Colors for different sources
        self.model_color = (0, 255, 0)  # Green for model predictions (thin)
        self.label_color = (255, 0, 0)  # Red for label file (thick)
        self.user_color = (0, 0, 255)   # Blue for user drawings (thick)
        
        # Class colors and names (you can customize this)
        # Store colors in BGR format for OpenCV and convert when needed for Tkinter
        self.class_info = {
            0: {"name": "Motorbike", "color": (0, 0, 255)},      # Red in BGR
            1: {"name": "Billboard", "color": (0, 255, 0)},      # Green in BGR
            2: {"name": "Car", "color": (255, 0, 0)},      # Blue in BGR
            3: {"name": "Chair", "color": (0, 255, 255)},    # Yellow in BGR
            4: {"name": "Plastic bag", "color": (255, 0, 255)},    # Magenta in BGR
            5: {"name": "Human", "color": (255, 255, 0)},    # Cyan in BGR
            6: {"name": "Mobile stall", "color": (0, 0, 128)},      # Dark Red in BGR
            7: {"name": "Table", "color": (0, 128, 0)},      # Dark Green in BGR
            8: {"name": "Others", "color": (128, 0, 0)},      # Dark Blue in BGR
            9: {"name": "Umbrella", "color": (0, 128, 128)}     # Dark Yellow in BGR
        }
        
        # Add variables for box editing and model visibility
        self.selected_box_index = None
        self.selected_box_rect = None
        self.editing_box = False  # Flag for editing mode
        
        # Load class configuration
        self.config_file = "class_config.json"
        # Store yaml_path for later loading after GUI is set up
        self.yaml_path_to_load = yaml_path
        
        # Load from JSON config first (will be overridden by YAML if provided)
        if not yaml_path:
            self.load_class_config()
        else:
            # Initialize with default class for now, will load from YAML after GUI setup
            self.class_info = {
                0: {"name": "Default", "color": [0, 0, 255]}
            }
        
        self.setup_gui()
        self.setup_bindings()
        
        # Now load YAML after GUI is set up
        if self.yaml_path_to_load:
            self.load_classes_from_yaml(self.yaml_path_to_load)
            # Refresh class buttons after loading YAML
            self.setup_class_buttons()
            # Update class filter dropdown
            if hasattr(self, 'update_class_filter_dropdown'):
                self.update_class_filter_dropdown()
        
        # Initialize with provided paths if available
        if self.image_folder and not self.yaml_path_to_load:
            # Only load if not already loaded by YAML
            self.load_images_from_folder(self.image_folder)
            
        if model_path:
            self.load_model(model_path)
            
        # Load labels if both image folder and label folder are provided
        if self.current_image_path and self.label_folder:
            self.load_labels()
            self.display_image()
        
        if self.image_folder and self.pretrained_model:
            self.precompute_model_predictions()
        
    def bgr_to_rgb_hex(self, bgr_color):
        """Convert BGR color tuple to RGB hex string for Tkinter"""
        b, g, r = bgr_color
        return f'#{r:02x}{g:02x}{b:02x}'
        
    def setup_gui(self):
        # Main frame
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Left panel for controls
        self.control_panel = ttk.Frame(self.main_frame)
        self.control_panel.pack(side=tk.LEFT, fill=tk.Y, padx=5)
        
        # Create a notebook for tabs in the control panel
        self.control_tabs = ttk.Notebook(self.control_panel)
        self.control_tabs.pack(fill=tk.BOTH, expand=True)
        
        # Create tabs for different controls
        self.setup_tab = ttk.Frame(self.control_tabs)
        self.filters_tab = ttk.Frame(self.control_tabs)
        self.classes_tab = ttk.Frame(self.control_tabs)
        self.display_tab = ttk.Frame(self.control_tabs)
        self.analyze_tab = ttk.Frame(self.control_tabs)
        
        self.control_tabs.add(self.setup_tab, text="Setup")
        self.control_tabs.add(self.filters_tab, text="Filters")
        self.control_tabs.add(self.classes_tab, text="Classes")
        self.control_tabs.add(self.display_tab, text="Display")
        self.control_tabs.add(self.analyze_tab, text="Analyze")
        
        # Setup tab - Folder selection in a more compact layout
        path_frame = ttk.LabelFrame(self.setup_tab, text="Data Paths")
        path_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Image folder row
        img_frame = ttk.Frame(path_frame)
        img_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(img_frame, text="Images:").pack(side=tk.LEFT)
        ttk.Button(img_frame, text="Browse", width=8, command=self.select_image_folder).pack(side=tk.RIGHT)
        
        # Label folder row
        lbl_frame = ttk.Frame(path_frame)
        lbl_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(lbl_frame, text="Labels:").pack(side=tk.LEFT)
        ttk.Button(lbl_frame, text="Browse", width=8, command=self.select_label_folder).pack(side=tk.RIGHT)
        
        # Model path row
        mdl_frame = ttk.Frame(path_frame)
        mdl_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(mdl_frame, text="Model:").pack(side=tk.LEFT)
        ttk.Button(mdl_frame, text="Browse", width=8, command=self.select_model).pack(side=tk.RIGHT)
        
        # YAML path row
        yaml_frame = ttk.Frame(path_frame)
        yaml_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(yaml_frame, text="YAML:").pack(side=tk.LEFT)
        ttk.Button(yaml_frame, text="Browse", width=8, command=self.select_yaml).pack(side=tk.RIGHT)
        
        # Navigation frame
        nav_frame = ttk.LabelFrame(self.setup_tab, text="Navigation")
        nav_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Add navigation buttons
        self.nav_frame = ttk.Frame(nav_frame)
        self.nav_frame.pack(pady=5)
        ttk.Button(self.nav_frame, text="Previous", command=self.prev_image).pack(side=tk.LEFT, padx=5)
        
        # Add jump to image number functionality
        self.jump_frame = ttk.Frame(self.nav_frame)
        self.jump_frame.pack(side=tk.LEFT, padx=5)
        self.jump_var = tk.StringVar()
        self.jump_entry = ttk.Entry(self.jump_frame, textvariable=self.jump_var, width=5)
        self.jump_entry.pack(side=tk.LEFT)
        ttk.Button(self.jump_frame, text="Go", command=self.jump_to_image).pack(side=tk.LEFT, padx=2)
        
        ttk.Button(self.nav_frame, text="Next", command=self.next_image).pack(side=tk.LEFT, padx=5)
        
        # Image counter display
        self.counter_var = tk.StringVar(value="Image: 0/0")
        ttk.Label(nav_frame, textvariable=self.counter_var).pack(pady=5)
        
        # Add Extract and Accept All buttons to setup tab
        action_frame = ttk.Frame(self.setup_tab)
        action_frame.pack(pady=10, fill=tk.X)
        
        ttk.Button(action_frame, text="Extract Labels", command=self.save_labels).pack(side=tk.LEFT, padx=5)
        ttk.Button(action_frame, text="Accept All", command=self.accept_all_predictions).pack(side=tk.LEFT, padx=5)
        
        # Add clipboard management buttons
        ttk.Button(action_frame, text="📋 View Clipboard", command=self.show_clipboard_viewer).pack(side=tk.LEFT, padx=5)
        ttk.Button(action_frame, text="📄 Paste Labels", command=self.paste_labels).pack(side=tk.LEFT, padx=5)
        
        # Add delete image button to setup tab
        ttk.Button(self.setup_tab, text="Delete Current Image", 
                  command=self.delete_current_image,
                  style="Delete.TButton").pack(pady=5)
        
        # Create a style for the delete button
        delete_style = ttk.Style()
        delete_style.configure("Delete.TButton", foreground="red")
        
        # Add Filter tab controls
        filter_class_frame = ttk.LabelFrame(self.filters_tab, text="Filter by Class")
        filter_class_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Class filter variables
        self.filter_by_class_enabled = False
        self.filter_class_id = 0
        self.filter_class_var = tk.BooleanVar(value=False)
        
        # Size filter variables
        self.filter_by_size_enabled = False
        self.filter_min_size = 0
        self.filter_max_size = 100000
        
        # Enable/disable class filter
        ttk.Checkbutton(filter_class_frame, text="Filter by Class", 
                       variable=self.filter_class_var,
                       command=self.toggle_class_filter).pack(anchor=tk.W, padx=5, pady=2)
        
        # Class selection for filter
        filter_class_select = ttk.Frame(filter_class_frame)
        filter_class_select.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(filter_class_select, text="Class:").pack(side=tk.LEFT)
        self.filter_class_combo = ttk.Combobox(filter_class_select, width=15, state="disabled")
        self.filter_class_combo.pack(side=tk.LEFT, padx=5)
        self.filter_class_combo.bind("<<ComboboxSelected>>", self.update_filter)
        
        # Size filter
        filter_size_frame = ttk.LabelFrame(self.filters_tab, text="Filter by Size")
        filter_size_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Enable/disable size filter
        self.filter_size_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(filter_size_frame, text="Filter by Object Size", 
                       variable=self.filter_size_var,
                       command=self.toggle_size_filter).pack(anchor=tk.W, padx=5, pady=2)
        
        # Min size
        min_size_frame = ttk.Frame(filter_size_frame)
        min_size_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(min_size_frame, text="Min Area (px²):").pack(side=tk.LEFT)
        self.min_size_var = tk.StringVar(value="0")
        min_size_entry = ttk.Entry(min_size_frame, textvariable=self.min_size_var, width=8)
        min_size_entry.pack(side=tk.LEFT, padx=5)
        min_size_entry.bind("<Return>", self.update_filter)
        min_size_entry.bind("<FocusOut>", self.update_filter)
        
        # Max size
        max_size_frame = ttk.Frame(filter_size_frame)
        max_size_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(max_size_frame, text="Max Area (px²):").pack(side=tk.LEFT)
        self.max_size_var = tk.StringVar(value="100000")
        max_size_entry = ttk.Entry(max_size_frame, textvariable=self.max_size_var, width=8)
        max_size_entry.pack(side=tk.LEFT, padx=5)
        max_size_entry.bind("<Return>", self.update_filter)
        max_size_entry.bind("<FocusOut>", self.update_filter)
        
        # Overlapping boxes filter (quality control)
        filter_overlap_frame = ttk.LabelFrame(self.filters_tab, text="Filter by Overlapping Labels")
        filter_overlap_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Enable/disable overlap filter
        self.filter_overlap_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(filter_overlap_frame, text="Show images with overlapping boxes (different classes)", 
                       variable=self.filter_overlap_var,
                       command=self.toggle_overlap_filter).pack(anchor=tk.W, padx=5, pady=2)
        
        # Overlap threshold
        overlap_threshold_frame = ttk.Frame(filter_overlap_frame)
        overlap_threshold_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(overlap_threshold_frame, text="Min IOU Overlap:").pack(side=tk.LEFT)
        self.filter_overlap_threshold_var = tk.StringVar(value="0.5")
        overlap_entry = ttk.Entry(overlap_threshold_frame, textvariable=self.filter_overlap_threshold_var, width=8)
        overlap_entry.pack(side=tk.LEFT, padx=5)
        overlap_entry.bind("<Return>", self.update_filter)
        overlap_entry.bind("<FocusOut>", self.update_filter)
        ttk.Label(overlap_threshold_frame, text="(0.0-1.0, higher = more overlap)").pack(side=tk.LEFT, padx=5)
        
        # Highlight overlapping boxes option
        self.highlight_overlaps_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(filter_overlap_frame, text="Highlight overlapping boxes in red", 
                       variable=self.highlight_overlaps_var,
                       command=self.toggle_overlap_highlight).pack(anchor=tk.W, padx=5, pady=2)
        
        # Filename filter
        filter_name_frame = ttk.LabelFrame(self.filters_tab, text="Filter by Filename")
        filter_name_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Enable/disable filename filter
        self.filter_name_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(filter_name_frame, text="Filter by Filename (contains)", 
                       variable=self.filter_name_var,
                       command=self.toggle_name_filter).pack(anchor=tk.W, padx=5, pady=2)
        
        # Filename input
        name_input_frame = ttk.Frame(filter_name_frame)
        name_input_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(name_input_frame, text="Contains:").pack(side=tk.LEFT)
        self.filter_name_text_var = tk.StringVar(value="")
        name_entry = ttk.Entry(name_input_frame, textvariable=self.filter_name_text_var, width=20)
        name_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        name_entry.bind("<Return>", self.update_filter)
        name_entry.bind("<FocusOut>", self.update_filter)
        
        # Case sensitive checkbox
        self.filter_name_case_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(filter_name_frame, text="Case sensitive", 
                       variable=self.filter_name_case_var,
                       command=self.update_filter).pack(anchor=tk.W, padx=5, pady=2)
        
        # Class Presence/Absence Filter
        filter_presence_frame = ttk.LabelFrame(self.filters_tab, text="Filter by Class Presence/Absence")
        filter_presence_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Enable/disable presence filter
        self.filter_presence_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(filter_presence_frame, text="Filter by class presence", 
                       variable=self.filter_presence_var,
                       command=self.toggle_presence_filter).pack(anchor=tk.W, padx=5, pady=2)
        
        # Must have classes (multi-select)
        must_have_frame = ttk.Frame(filter_presence_frame)
        must_have_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(must_have_frame, text="Must have classes:").pack(anchor=tk.W)
        
        # Scrollable frame for must-have checkboxes
        must_have_scroll_frame = ttk.Frame(must_have_frame)
        must_have_scroll_frame.pack(fill=tk.BOTH, expand=True)
        self.must_have_canvas = tk.Canvas(must_have_scroll_frame, height=80)
        must_have_scrollbar = ttk.Scrollbar(must_have_scroll_frame, orient="vertical", command=self.must_have_canvas.yview)
        self.must_have_checkboxes_frame = ttk.Frame(self.must_have_canvas)
        
        self.must_have_canvas.create_window((0, 0), window=self.must_have_checkboxes_frame, anchor="nw")
        self.must_have_canvas.configure(yscrollcommand=must_have_scrollbar.set)
        
        self.must_have_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        must_have_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Must NOT have classes (multi-select)
        must_not_have_frame = ttk.Frame(filter_presence_frame)
        must_not_have_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(must_not_have_frame, text="Must NOT have classes:").pack(anchor=tk.W)
        
        # Scrollable frame for must-not-have checkboxes
        must_not_have_scroll_frame = ttk.Frame(must_not_have_frame)
        must_not_have_scroll_frame.pack(fill=tk.BOTH, expand=True)
        self.must_not_have_canvas = tk.Canvas(must_not_have_scroll_frame, height=80)
        must_not_have_scrollbar = ttk.Scrollbar(must_not_have_scroll_frame, orient="vertical", command=self.must_not_have_canvas.yview)
        self.must_not_have_checkboxes_frame = ttk.Frame(self.must_not_have_canvas)
        
        self.must_not_have_canvas.create_window((0, 0), window=self.must_not_have_checkboxes_frame, anchor="nw")
        self.must_not_have_canvas.configure(yscrollcommand=must_not_have_scrollbar.set)
        
        self.must_not_have_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        must_not_have_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Store checkbox variables
        self.must_have_class_vars = {}  # {class_id: BooleanVar}
        self.must_not_have_class_vars = {}  # {class_id: BooleanVar}
        
        # Bind canvas resize
        self.must_have_checkboxes_frame.bind("<Configure>", lambda e: self.must_have_canvas.configure(scrollregion=self.must_have_canvas.bbox("all")))
        self.must_not_have_checkboxes_frame.bind("<Configure>", lambda e: self.must_not_have_canvas.configure(scrollregion=self.must_not_have_canvas.bbox("all")))
        
        # Apply filter button
        ttk.Button(self.filters_tab, text="Apply Filters", 
                  command=self.apply_filters).pack(pady=(10, 5))
        
        # Progress bar for filter operations
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(self.filters_tab, variable=self.progress_var, 
                                          mode='determinate', length=200)
        self.progress_bar.pack(pady=(0, 10), fill=tk.X, padx=5)
        
        # Result counter label
        self.filter_result_var = tk.StringVar(value="")
        ttk.Label(self.filters_tab, textvariable=self.filter_result_var).pack(pady=5)
        
        # Cache builder button
        ttk.Button(self.filters_tab, text="Build Cache (for faster filtering)", 
                  command=self.build_label_cache).pack(pady=(10, 5))
        
        # Classes tab
        ttk.Label(self.classes_tab, text="Classes:").pack(pady=(5, 5))
        self.class_frame = ttk.Frame(self.classes_tab)
        self.class_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Add class button
        ttk.Button(self.classes_tab, text="Add New Class", 
                  command=self.add_new_class).pack(pady=5)
        
        # Selected class indicator
        self.selected_class_var = tk.StringVar(value="Selected Class: 0")
        ttk.Label(self.classes_tab, textvariable=self.selected_class_var).pack(pady=5)
        
        # Display tab - Add batch view and visibility controls
        # Add batch view controls
        batch_frame = ttk.LabelFrame(self.display_tab, text="Batch View")
        batch_frame.pack(pady=5, fill=tk.X, padx=5)
        
        # Batch size selection
        self.batch_size_var = tk.StringVar(value="1x1")
        batch_sizes = ["1x1", "2x2", "3x3", "4x4", "5x5"]
        for size in batch_sizes:
            ttk.Radiobutton(batch_frame, text=size, value=size, 
                           variable=self.batch_size_var,
                           command=self.toggle_batch_view).pack(anchor=tk.W, padx=5, pady=2)
        
        # Add visibility controls frame
        visibility_frame = ttk.LabelFrame(self.display_tab, text="Display Options")
        visibility_frame.pack(pady=5, fill=tk.X, padx=5)
        
        # Add checkboxes for visibility controls
        self.show_predictions_var = tk.BooleanVar(value=self.show_model_predictions)
        ttk.Checkbutton(visibility_frame, text="Show Predictions", 
                       variable=self.show_predictions_var,
                       command=self.toggle_predictions).pack(anchor=tk.W, padx=5, pady=2)
        
        self.show_labels_var = tk.BooleanVar(value=self.show_label_boxes)
        ttk.Checkbutton(visibility_frame, text="Show Label Boxes", 
                       variable=self.show_labels_var,
                       command=self.toggle_label_boxes).pack(anchor=tk.W, padx=5, pady=2)
        
        self.show_drawings_var = tk.BooleanVar(value=self.show_user_boxes)
        ttk.Checkbutton(visibility_frame, text="Show User Drawings", 
                       variable=self.show_drawings_var,
                       command=self.toggle_user_boxes).pack(anchor=tk.W, padx=5, pady=2)
        
        # Add instructions label
        instructions = (
            "Instructions:\n"
            "- Left click + drag: Draw box\n"
            "- Right click: Select box & show menu\n"
            "- Double click: Edit selected box\n"
            "- S key: Save labels\n"
            "- Left/Right arrows: Navigate images\n"
            "- A key: Accept all predictions\n"
            "- Right-click box → Add to Clipboard\n"
            "- Ctrl+V: Paste labels from clipboard"
        )
        ttk.Label(self.display_tab, text=instructions, justify=tk.LEFT).pack(pady=10)
        
        # Analyze tab - Dataset insights and visualizations
        ttk.Label(self.analyze_tab, text="Dataset Analysis", font=('Arial', 12, 'bold')).pack(pady=(5, 10))
        
        # Analysis buttons
        analyze_btn_frame = ttk.Frame(self.analyze_tab)
        analyze_btn_frame.pack(pady=5, fill=tk.X, padx=5)
        
        ttk.Button(analyze_btn_frame, text="📊 Analyze Dataset", 
                  command=self.analyze_dataset).pack(fill=tk.X, pady=2)
        ttk.Button(analyze_btn_frame, text="🔄 Refresh Analysis", 
                  command=self.refresh_analysis).pack(fill=tk.X, pady=2)
        
        # Analysis options
        options_frame = ttk.LabelFrame(self.analyze_tab, text="Analysis Options")
        options_frame.pack(pady=5, fill=tk.X, padx=5)
        
        self.analyze_all_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(options_frame, text="Analyze all images (slow)", 
                       variable=self.analyze_all_var).pack(anchor=tk.W, padx=5, pady=2)
        
        self.sample_size_var = tk.StringVar(value="1000")
        sample_frame = ttk.Frame(options_frame)
        sample_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(sample_frame, text="Sample size:").pack(side=tk.LEFT)
        ttk.Entry(sample_frame, textvariable=self.sample_size_var, width=10).pack(side=tk.LEFT, padx=5)
        ttk.Label(sample_frame, text="images (if not analyzing all)").pack(side=tk.LEFT)
        
        # Progress bar for analysis
        self.analyze_progress_var = tk.DoubleVar()
        self.analyze_progress_bar = ttk.Progressbar(self.analyze_tab, variable=self.analyze_progress_var, 
                                                    mode='determinate', length=200)
        self.analyze_progress_bar.pack(pady=(5, 0), fill=tk.X, padx=5)
        
        # Stats display
        self.stats_frame = ttk.LabelFrame(self.analyze_tab, text="Dataset Statistics")
        self.stats_frame.pack(pady=5, fill=tk.BOTH, expand=True, padx=5)
        
        self.stats_text = tk.Text(self.stats_frame, height=10, wrap=tk.WORD, font=('Courier', 9))
        self.stats_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        stats_scrollbar = ttk.Scrollbar(self.stats_text, command=self.stats_text.yview)
        stats_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.stats_text.config(yscrollcommand=stats_scrollbar.set)
        
        # Canvas for image display with scrollbars
        self.canvas_frame = ttk.Frame(self.main_frame)
        self.canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Create scrollbars for canvas
        self.canvas_vscroll = ttk.Scrollbar(self.canvas_frame, orient="vertical")
        self.canvas_hscroll = ttk.Scrollbar(self.canvas_frame, orient="horizontal")
        
        self.canvas = tk.Canvas(self.canvas_frame, bg='gray', cursor="crosshair",
                               xscrollcommand=self.canvas_hscroll.set,
                               yscrollcommand=self.canvas_vscroll.set)
        
        # Configure scrollbars
        self.canvas_vscroll.config(command=self.canvas.yview)
        self.canvas_hscroll.config(command=self.canvas.xview)
        
        # Pack scrollbars and canvas
        self.canvas_vscroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas_hscroll.pack(side=tk.BOTTOM, fill=tk.X)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Setup class buttons
        self.setup_class_buttons()
        
        # Create context menus
        self.create_context_menus()
        
        # In Filters tab (after size filter controls)
        filter_iou_frame = ttk.LabelFrame(self.filters_tab, text="Filter by Label/Prediction IOU")
        filter_iou_frame.pack(fill=tk.X, padx=5, pady=5)

        self.filter_iou_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(filter_iou_frame, text="Filter by Label/Prediction IOU", 
                       variable=self.filter_iou_var,
                       command=self.toggle_iou_filter).pack(anchor=tk.W, padx=5, pady=2)

        # IOU threshold input
        thresh_frame = ttk.Frame(filter_iou_frame)
        thresh_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(thresh_frame, text="IOU Threshold:").pack(side=tk.LEFT)
        self.iou_thresh_var = tk.StringVar(value="0.5")
        iou_thresh_entry = ttk.Entry(thresh_frame, textvariable=self.iou_thresh_var, width=8)
        iou_thresh_entry.pack(side=tk.LEFT, padx=5)
        iou_thresh_entry.bind("<Return>", self.update_filter)
        iou_thresh_entry.bind("<FocusOut>", self.update_filter)
        
        if not self.pretrained_model:
            self.filter_iou_var.set(False)
            iou_thresh_entry.config(state='disabled')
        else:
            iou_thresh_entry.config(state='normal')
        
    def create_context_menus(self):
        """Create context menus for right-click actions"""
        # Context menu for prediction boxes
        self.prediction_menu = tk.Menu(self.root, tearoff=0)
        self.prediction_menu.add_command(label="Accept", command=self.accept_selected_prediction)
        self.prediction_menu.add_command(label="Cancel", command=self.clear_selections)
        
        # Context menu for user/label boxes
        self.box_menu = tk.Menu(self.root, tearoff=0)
        self.box_menu.add_command(label="Add to Clipboard", command=self.add_box_to_clipboard)
        self.box_menu.add_command(label="Change Class", command=self.show_class_reassignment_menu)
        self.box_menu.add_command(label="Delete", command=self.delete_selected_box)
        self.box_menu.add_command(label="Cancel", command=self.clear_selections)
        
        # Create class reassignment submenu (will be populated dynamically)
        self.class_reassign_menu = tk.Menu(self.box_menu, tearoff=0)
        
    def setup_bindings(self):
        self.canvas.bind("<ButtonPress-1>", self.start_draw)
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<ButtonRelease-1>", self.end_draw)
        self.canvas.bind("<Motion>", self.draw_crosshair)  # Track mouse movement for crosshair
        self.root.bind("<s>", self.save_labels)
        self.root.bind("<Left>", lambda e: self.prev_image())
        self.root.bind("<Right>", lambda e: self.next_image())
        self.canvas.bind("<ButtonPress-3>", self.select_box_or_prediction)  # Right click to select box or prediction
        self.root.bind("<Delete>", self.delete_selected_box)  # Delete key to remove selected box
        self.canvas.bind("<Double-Button-1>", self.start_edit_box)  # Double click to edit box
        self.canvas.bind("<MouseWheel>", self.handle_zoom)  # Windows and MacOS
        self.canvas.bind("<Button-4>", self.handle_zoom)  # Linux scroll up
        self.canvas.bind("<Button-5>", self.handle_zoom)  # Linux scroll down
        self.root.bind("<Return>", lambda e: self.accept_selected_prediction())  # Enter key to accept selected prediction
        self.root.bind("<a>", lambda e: self.accept_all_predictions())  # 'a' key to accept all predictions
        self.root.bind("<Control-v>", lambda e: self.paste_labels())  # Ctrl+V to paste labels
        self.canvas.bind("<ButtonPress-1>", self.on_canvas_click, add="+")  # Handle clicks to cancel selection
        
    def on_canvas_click(self, event):
        """Handle clicks on canvas to cancel selection"""
        # Clear selections when clicking elsewhere on canvas
        if self.selected_box_index is not None or self.selected_prediction is not None:
            self.clear_selections()
    
    def draw_crosshair(self, event):
        """Draw crosshair cursor lines following the mouse"""
        # Only show crosshair when an image is loaded and not in batch mode
        if not hasattr(self, 'current_image') or self.current_image is None or self.batch_mode:
            return
        
        # Get canvas coordinates (accounts for scrolling)
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)
        
        # Get canvas dimensions and scroll region
        scroll_region = self.canvas.cget("scrollregion")
        if scroll_region:
            # Parse scrollregion: "x1 y1 x2 y2"
            x1, y1, x2, y2 = map(float, scroll_region.split())
            canvas_width = x2
            canvas_height = y2
        else:
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
        
        # Remove old crosshair lines
        if self.crosshair_h:
            self.canvas.delete(self.crosshair_h)
        if self.crosshair_v:
            self.canvas.delete(self.crosshair_v)
        
        # Get color of currently selected class
        current_class_color = self.class_info[self.current_class]["color"]
        crosshair_color = self.bgr_to_rgb_hex(current_class_color)
        
        # Draw new crosshair lines at cursor position (in canvas coordinates)
        # Horizontal line (full width of scrollregion)
        self.crosshair_h = self.canvas.create_line(
            0, canvas_y, canvas_width, canvas_y,
            fill=crosshair_color, width=1, dash=(4, 4), tags='crosshair'
        )
        # Vertical line (full height of scrollregion)
        self.crosshair_v = self.canvas.create_line(
            canvas_x, 0, canvas_x, canvas_height,
            fill=crosshair_color, width=1, dash=(4, 4), tags='crosshair'
        )
            
    def select_image_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            self.load_images_from_folder(folder)
            
    def load_images_from_folder(self, folder):
        """Load images from the specified folder"""
        self.image_folder = folder
        self.image_paths = []
        for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            self.image_paths.extend(list(Path(folder).glob(f'*{ext}')))
        self.image_paths.sort()
        
        # Initialize filtered paths to all paths
        self.filtered_image_paths = self.image_paths.copy()
        
        # Clear cache when loading new folder
        self.image_dimensions_cache = {}
        self.labels_cache = {}
        
        if self.image_paths:
            self.current_image_index = 0
            self.load_image()
        
        # Update class filter dropdown if the method exists
        if hasattr(self, 'update_class_filter_dropdown'):
            self.update_class_filter_dropdown()
            
        self.update_status()
        self.update_counter()

    def add_images_from_folder(self, folder):
        """Add images from a folder to the current image list without clearing existing ones."""
        folder_path = Path(folder)
        if not folder_path.exists():
            return
        added = []
        for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            for p in folder_path.glob(f'*{ext}'):
                added.append(p)
        # Avoid duplicates
        existing = set(str(p) for p in self.image_paths)
        for p in sorted(added):
            if str(p) not in existing:
                self.image_paths.append(p)
        # Sort paths
        self.image_paths = sorted(self.image_paths, key=lambda x: str(x))
        # Update filtered paths to include new images
        self.filtered_image_paths = self.image_paths.copy()
        
    def select_label_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            self.label_folder = folder
            if self.current_image_path:
                self.load_labels()
                if self.highlight_overlaps:
                    self.detect_current_overlaps()
                self.display_image()
            self.update_status()
            
    def select_model(self):
        model_path = filedialog.askopenfilename(filetypes=[("PT files", "*.pt")])
        if model_path:
            self.load_model(model_path)
    
    def select_yaml(self):
        """Select and load YAML configuration file"""
        yaml_path = filedialog.askopenfilename(filetypes=[("YAML files", "*.yaml *.yml")])
        if yaml_path:
            self.load_classes_from_yaml(yaml_path)
            # Refresh class buttons to show new classes
            self.setup_class_buttons()
            # Update class filter dropdown if it exists
            if hasattr(self, 'update_class_filter_dropdown'):
                self.update_class_filter_dropdown()
            # Redisplay current image with new classes
            if self.current_image:
                self.display_image()
                
    def load_model(self, model_path):
        """Load YOLO model from the specified path"""
        try:
            self.pretrained_model = YOLO(model_path)
            
            # Update class_info with model's class names if available
            if hasattr(self.pretrained_model, 'names') and self.pretrained_model.names:
                model_names = self.pretrained_model.names
                print(f"Model has {len(model_names)} classes: {model_names}")
                
                # Generate random colors for new classes
                import random
                existing_classes = set(self.class_info.keys())
                
                for cls_id, cls_name in model_names.items():
                    if cls_id not in existing_classes:
                        # Generate a random color in BGR format
                        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                        self.class_info[cls_id] = {"name": cls_name, "color": color}
                        print(f"Added class {cls_id}: {cls_name}")
                    else:
                        # Update name if different
                        if self.class_info[cls_id]["name"] != cls_name:
                            print(f"Updating class {cls_id} name from '{self.class_info[cls_id]['name']}' to '{cls_name}'")
                            self.class_info[cls_id]["name"] = cls_name
                
                # Refresh UI components after updating class_info
                if hasattr(self, 'setup_class_buttons'):
                    self.setup_class_buttons()
                if hasattr(self, 'update_class_filter_dropdown'):
                    self.update_class_filter_dropdown()
            
            if self.current_image_path:
                self.load_model_predictions()
                self.display_image()
            self.update_status()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {str(e)}")
                
    def load_image(self):
        if 0 <= self.current_image_index < len(self.filtered_image_paths):
            self.current_image_path = self.filtered_image_paths[self.current_image_index]
            
            # Load image
            self.current_image = cv2.imread(str(self.current_image_path))
            
            # Update cache for this image if not already cached
            if str(self.current_image_path) not in self.image_dimensions_cache and self.current_image is not None:
                img_height, img_width = self.current_image.shape[:2]
                self.image_dimensions_cache[str(self.current_image_path)] = (img_width, img_height)
            
            # Load labels and predictions first
            self.load_labels()
            self.load_model_predictions()
            
            # Detect overlaps if highlight mode is on (must be after load_labels)
            if self.highlight_overlaps:
                self.detect_current_overlaps()
            
            # Display image after all data is loaded
            self.display_image()
            
            self.update_status()
            self.update_counter()
            
    def display_image(self):
        if self.current_image is None:
            return
        
        # Clear canvas to prevent memory leaks
        self.canvas.delete("all")
        
        # Get original dimensions
        height, width = self.current_image.shape[:2]
        
        # Calculate new dimensions
        new_width = int(width * self.zoom_scale)
        new_height = int(height * self.zoom_scale)
        
        # Limit maximum dimensions to prevent excessive memory usage (50 megapixels)
        max_pixels = 50_000_000
        if new_width * new_height > max_pixels:
            scale_factor = (max_pixels / (new_width * new_height)) ** 0.5
            new_width = int(new_width * scale_factor)
            new_height = int(new_height * scale_factor)
        
        # Use INTER_NEAREST for faster zooming at high scales (>2x), INTER_LINEAR for better quality at lower scales
        interpolation = cv2.INTER_NEAREST if self.zoom_scale > 2.0 else cv2.INTER_LINEAR
        
        # Resize image - convert BGR to RGB in one step to avoid extra copy
        image_resized = cv2.cvtColor(
            cv2.resize(self.current_image, (new_width, new_height), interpolation=interpolation),
            cv2.COLOR_BGR2RGB
        )
        
        # Optimize: Only draw boxes if zoom is reasonable (avoid drawing tiny boxes at very low zoom)
        should_draw_boxes = self.zoom_scale >= 0.1
        
        # Draw model predictions (thin lines) if enabled
        if should_draw_boxes and self.show_model_predictions:
            for i, (cls_id, x1, y1, x2, y2) in enumerate(self.model_boxes):
                # Convert numpy values to native Python types if needed
                if hasattr(cls_id, 'item'):
                    cls_id = int(cls_id.item())
                
                # Skip if class not in class_info
                if cls_id not in self.class_info:
                    continue
                    
                if not self.class_visibility.get(cls_id, True):
                    continue
                # Convert coordinates to zoomed space
                zx1, zy1 = self.image_to_canvas_coords(x1, y1)
                zx2, zy2 = self.image_to_canvas_coords(x2, y2)
                color = self.class_info[cls_id]["color"]
                
                # Draw box with thinner lines for predictions
                cv2.rectangle(image_resized, (int(zx1), int(zy1)), (int(zx2), int(zy2)), 
                            color[::-1], 1)  # Reverse color for RGB image
                
                # Add small label showing prediction number and class name
                class_name = self.class_info[cls_id]["name"]
                label_text = f"{class_name}"
                font_scale = 0.5 * self.zoom_scale  # Scale font size with zoom
                thickness = max(1, int(1 * self.zoom_scale))  # Scale thickness with zoom
                cv2.putText(image_resized, label_text, 
                          (int(zx1), int(zy1)-5),
                          cv2.FONT_HERSHEY_SIMPLEX, font_scale, 
                          color[::-1], thickness)
            
        # Draw label file boxes (thick lines)
        if should_draw_boxes and self.show_label_boxes:
            for i, (cls_id, x1, y1, x2, y2) in enumerate(self.label_boxes):
                # Convert numpy values to native Python types if needed
                if hasattr(cls_id, 'item'):
                    cls_id = int(cls_id.item())
                    
                if not self.class_visibility.get(cls_id, True):
                    continue
                # Convert coordinates to zoomed space
                zx1, zy1 = self.image_to_canvas_coords(x1, y1)
                zx2, zy2 = self.image_to_canvas_coords(x2, y2)
                
                # Check if this box is overlapping and highlight mode is on
                if self.highlight_overlaps and i in self.overlapping_boxes:
                    color = (0, 0, 255)  # Red for overlapping boxes (BGR)
                    thickness = max(4, int(4 * self.zoom_scale))  # Thicker for overlap
                else:
                    color = self.class_info[cls_id]["color"]
                    thickness = max(2, int(2 * self.zoom_scale))  # Normal thickness
                
                cv2.rectangle(image_resized, (int(zx1), int(zy1)), (int(zx2), int(zy2)), 
                            color[::-1], thickness)  # Reverse color for RGB image
                
                # Add label showing box source and number
                if self.highlight_overlaps and i in self.overlapping_boxes:
                    label_text = f"L{i+1} ⚠"  # Warning icon for overlapping
                else:
                    label_text = f"L{i+1}"
                font_scale = 0.5 * self.zoom_scale  # Scale font size with zoom
                thickness_text = max(1, int(1 * self.zoom_scale))  # Scale thickness with zoom
                cv2.putText(image_resized, label_text, 
                          (int(zx1), int(zy1)-5),
                          cv2.FONT_HERSHEY_SIMPLEX, font_scale, 
                          color[::-1], thickness_text)
            
        # Draw user boxes (thick lines)
        if should_draw_boxes and self.show_user_boxes:
            for i, (cls_id, x1, y1, x2, y2) in enumerate(self.user_boxes):
                # Convert numpy values to native Python types if needed
                if hasattr(cls_id, 'item'):
                    cls_id = int(cls_id.item())
                    
                if not self.class_visibility.get(cls_id, True):
                    continue
                # Convert coordinates to zoomed space
                zx1, zy1 = self.image_to_canvas_coords(x1, y1)
                zx2, zy2 = self.image_to_canvas_coords(x2, y2)
                color = self.class_info[cls_id]["color"]
                thickness = max(2, int(2 * self.zoom_scale))  # Scale thickness with zoom
                cv2.rectangle(image_resized, (int(zx1), int(zy1)), (int(zx2), int(zy2)), 
                            color[::-1], thickness)  # Reverse color for RGB image
                
                # Add label showing box source and number
                label_text = f"U{i+1}"
                font_scale = 0.5 * self.zoom_scale  # Scale font size with zoom
                thickness = max(1, int(1 * self.zoom_scale))  # Scale thickness with zoom
                cv2.putText(image_resized, label_text, 
                          (int(zx1), int(zy1)-5),
                          cv2.FONT_HERSHEY_SIMPLEX, font_scale, 
                          color[::-1], thickness)
            
        # Convert to PhotoImage
        image_pil = Image.fromarray(image_resized)
        self.photo = ImageTk.PhotoImage(image=image_pil)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
        
        # Add filename overlay at the top
        if self.current_image_path:
            filename = self.current_image_path.name
            # Add background rectangle for better readability
            font_size = max(12, int(14 * self.zoom_scale))
            text_bg_color = '#000000'
            text_fg_color = '#FFFFFF'
            
            self.canvas.create_rectangle(0, 0, new_width, font_size + 10, 
                                        fill=text_bg_color, stipple='gray75', 
                                        outline='', tags='filename_overlay')
            self.canvas.create_text(10, 5, anchor=tk.NW, 
                                   text=f"File: {filename}", 
                                   fill=text_fg_color, 
                                   font=('Arial', font_size, 'bold'),
                                   tags='filename_overlay')
        
        # Update canvas scroll region to match the zoomed image size
        self.canvas.configure(scrollregion=(0, 0, new_width, new_height))
        
        # Re-highlight selected box if any
        self.highlight_selected_box()
        
        # Re-highlight selected prediction if any
        if self.selected_prediction is not None:
            self.highlight_selected_prediction()
        
    def load_labels(self):
        self.label_boxes = []
        # Determine label file location. Prefer explicit self.label_folder, otherwise
        # try per-image-folder mappings populated when loading YAML with multiple paths.
        label_dir = None
        if self.label_folder:
            label_dir = Path(self.label_folder)
        else:
            # Try to find a matching image-folder key
            img_path_str = str(self.current_image_path)
            for img_folder, lbl_folder in getattr(self, 'per_image_label_folders', {}).items():
                if img_path_str.startswith(img_folder):
                    label_dir = Path(lbl_folder)
                    break
        # Fallback: try conventional siblings
        if label_dir is None:
            p = Path(self.current_image_path)
            candidates = [p.parent / 'labels', p.parent.parent / 'labels']
            for c in candidates:
                if c.exists():
                    label_dir = c
                    break

        if label_dir:
            label_path = label_dir / f"{self.current_image_path.stem}.txt"
            if label_path.exists():
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                    img_height, img_width = self.current_image.shape[:2]

                    # Update cache for these labels
                    cached_labels = []

                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) < 5:
                            continue
                        cls_id, x_center, y_center, w, h = map(float, parts[:5])
                        x1 = int((x_center - w/2) * img_width)
                        y1 = int((y_center - h/2) * img_height)
                        x2 = int((x_center + w/2) * img_width)
                        y2 = int((y_center + h/2) * img_height)
                        self.label_boxes.append((cls_id, x1, y1, x2, y2))

                        # Add to cache with box area
                        box_area = (x2 - x1) * (y2 - y1)
                        cached_labels.append((cls_id, x1, y1, x2, y2, box_area))

                    # Update labels cache
                    self.labels_cache[str(self.current_image_path)] = cached_labels
        
    def load_model_predictions(self):
        self.model_boxes = []
        if self.current_image_path:
            preds = self.model_predictions_cache.get(str(self.current_image_path), [])
            self.model_boxes = preds
        
    def start_draw(self, event):
        self.drawing = True
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)
        self.start_x, self.start_y = self.canvas_to_image_coords(canvas_x, canvas_y)
        
        # Create initial rectangle with RGB hex color
        color = self.class_info[self.current_class]["color"]
        rgb_color = self.bgr_to_rgb_hex(color)
        
        # Convert to canvas coordinates for display
        canvas_start_x, canvas_start_y = self.image_to_canvas_coords(self.start_x, self.start_y)
        self.current_rect = self.canvas.create_rectangle(
            canvas_start_x, canvas_start_y, canvas_start_x, canvas_start_y,
            outline=rgb_color, width=2
        )
        
    def draw(self, event):
        if not self.drawing and not self.editing_box:
            return
            
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)
        cur_x, cur_y = self.canvas_to_image_coords(canvas_x, canvas_y)
        
        # Get image dimensions
        img_height, img_width = self.current_image.shape[:2]
        
        # Constrain coordinates to image boundaries
        cur_x = max(0, min(cur_x, img_width))
        cur_y = max(0, min(cur_y, img_height))
        
        if self.editing_box:
            # Get the current box
            cls_id, x1, y1, x2, y2 = self.user_boxes[self.selected_box_index]
            
            # Determine which corner to move based on the start point
            if abs(self.start_x - x1) < 5 and abs(self.start_y - y1) < 5:  # Top-left
                x1, y1 = cur_x, cur_y
            elif abs(self.start_x - x2) < 5 and abs(self.start_y - y1) < 5:  # Top-right
                x2, y1 = cur_x, cur_y
            elif abs(self.start_x - x1) < 5 and abs(self.start_y - y2) < 5:  # Bottom-left
                x1, y2 = cur_x, cur_y
            elif abs(self.start_x - x2) < 5 and abs(self.start_y - y2) < 5:  # Bottom-right
                x2, y2 = cur_x, cur_y
                
            # Ensure coordinates stay within image boundaries
            x1 = max(0, min(x1, img_width))
            y1 = max(0, min(y1, img_height))
            x2 = max(0, min(x2, img_width))
            y2 = max(0, min(y2, img_height))
                
            # Convert to canvas coordinates for display
            zx1, zy1 = self.image_to_canvas_coords(x1, y1)
            zx2, zy2 = self.image_to_canvas_coords(x2, y2)
            self.canvas.coords(self.current_rect, zx1, zy1, zx2, zy2)
        else:
            # Normal drawing mode
            canvas_start_x, canvas_start_y = self.image_to_canvas_coords(self.start_x, self.start_y)
            self.canvas.coords(self.current_rect, canvas_start_x, canvas_start_y, canvas_x, canvas_y)
            
        # Show dimensions in image coordinates
        width = abs(cur_x - self.start_x)
        height = abs(cur_y - self.start_y)
        self.update_status(f"Box size: {int(width)}x{int(height)} pixels")
        
    def end_draw(self, event):
        if not self.drawing and not self.editing_box:
            return
            
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)
        end_x, end_y = self.canvas_to_image_coords(canvas_x, canvas_y)
        
        # Get image dimensions
        img_height, img_width = self.current_image.shape[:2]
        
        # Constrain coordinates to image boundaries
        end_x = max(0, min(end_x, img_width))
        end_y = max(0, min(end_y, img_height))
        
        # Delete the preview rectangle
        if self.current_rect:
            self.canvas.delete(self.current_rect)
            
        # Ensure coordinates are in correct order
        x1, x2 = min(self.start_x, end_x), max(self.start_x, end_x)
        y1, y2 = min(self.start_y, end_y), max(self.start_y, end_y)
        
        # Ensure coordinates stay within image boundaries
        x1 = max(0, min(x1, img_width))
        y1 = max(0, min(y1, img_height))
        x2 = max(0, min(x2, img_width))
        y2 = max(0, min(y2, img_height))
        
        # Calculate box dimensions
        width = x2 - x1
        height = y2 - y1
        
        # Check minimum size (10x10 pixels)
        if width < 10 or height < 10:
            self.update_status("Box too small! Minimum size is 10x10 pixels")
            self.drawing = False
            self.editing_box = False
            return
        
        if self.editing_box:
            # Update existing box
            self.user_boxes[self.selected_box_index] = (
                self.user_boxes[self.selected_box_index][0],  # Keep the same class
                x1, y1, x2, y2
            )
            self.editing_box = False
        else:
            # Add new box
            self.user_boxes.append((self.current_class, x1, y1, x2, y2))
            
        self.drawing = False
        
        # Automatically save labels after adding/editing box
        self.save_labels()
        
        # Re-detect overlaps if highlight mode is on (new/edited box might create overlaps)
        if self.highlight_overlaps:
            self.detect_current_overlaps()
        
        self.display_image()
        self.update_status(f"Box size: {int(width)}x{int(height)} pixels")
        
    def save_labels(self, event=None):
        """Save labels to file"""
        if not self.current_image_path:
            messagebox.showerror("Error", "No image loaded")
            return
        
        # Determine label directory - use label_folder if set, otherwise find from per_image mapping
        label_dir = None
        if self.label_folder:
            label_dir = Path(self.label_folder)
        else:
            # Try to find from per-image mapping
            img_path_str = str(self.current_image_path)
            for img_folder, lbl_folder in getattr(self, 'per_image_label_folders', {}).items():
                if img_path_str.startswith(img_folder):
                    label_dir = Path(lbl_folder)
                    break
        
        # If still no label directory, try to create one next to images
        if label_dir is None:
            img_path = Path(self.current_image_path)
            # Try images/../labels pattern
            if img_path.parent.name == 'images':
                label_dir = img_path.parent.parent / 'labels'
            else:
                label_dir = img_path.parent / 'labels'
            
            # Create the directory if it doesn't exist
            try:
                label_dir.mkdir(parents=True, exist_ok=True)
                # Remember this mapping for future saves
                if not hasattr(self, 'per_image_label_folders'):
                    self.per_image_label_folders = {}
                self.per_image_label_folders[str(img_path.parent)] = str(label_dir)
            except Exception as e:
                messagebox.showerror("Error", f"Cannot create label directory: {str(e)}")
                return
            
        # Combine label file boxes and user boxes
        all_boxes = self.label_boxes + self.user_boxes
        
        if not all_boxes:
            # If there are no boxes, delete the label file if it exists
            label_path = label_dir / f"{self.current_image_path.stem}.txt"
            if label_path.exists():
                try:
                    os.remove(label_path)
                    self.update_status("Empty label file deleted")
                    
                    # Remove from cache
                    if str(self.current_image_path) in self.labels_cache:
                        del self.labels_cache[str(self.current_image_path)]
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to delete empty label file: {str(e)}")
            return
            
        # Convert to YOLO format
        img_height, img_width = self.current_image.shape[:2]
        yolo_lines = []
        
        # Also create cached labels with additional box area
        cached_labels = []
        
        for cls_id, x1, y1, x2, y2 in all_boxes:
            # Convert to YOLO format (x_center, y_center, width, height)
            x_center = (x1 + x2) / (2 * img_width)
            y_center = (y1 + y2) / (2 * img_height)
            width = (x2 - x1) / img_width
            height = (y2 - y1) / img_height
            
            yolo_lines.append(f"{int(cls_id)} {x_center} {y_center} {width} {height}")
            
            # Add to cache
            box_area = (x2 - x1) * (y2 - y1)
            cached_labels.append((cls_id, x1, y1, x2, y2, box_area))
            
        # Update the cache
        self.labels_cache[str(self.current_image_path)] = cached_labels
            
        # Save to file
        label_path = label_dir / f"{self.current_image_path.stem}.txt"
        try:
            with open(label_path, 'w') as f:
                f.write('\n'.join(yolo_lines))
            self.update_status(f"Labels saved to {label_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save labels: {str(e)}")
    
    def add_box_to_clipboard(self):
        """Add selected box to clipboard"""
        if self.selected_box_index is None:
            return
        
        # Get the selected box based on source
        if self.selected_box_source == 'user':
            if 0 <= self.selected_box_index < len(self.user_boxes):
                box = self.user_boxes[self.selected_box_index]
        elif self.selected_box_source == 'label':
            if 0 <= self.selected_box_index < len(self.label_boxes):
                box = self.label_boxes[self.selected_box_index]
        else:
            return
        
        # Add to clipboard (avoid duplicates)
        box_tuple = tuple(box)
        if box_tuple not in self.label_clipboard:
            self.label_clipboard.append(box_tuple)
            cls_id = box[0]
            cls_name = self.class_info[cls_id]['name']
            self.update_status(f"📋 Added {cls_name} to clipboard ({len(self.label_clipboard)} total)")
            messagebox.showinfo("Added to Clipboard", f"Added {cls_name} box to clipboard\n\nTotal: {len(self.label_clipboard)} labels")
        else:
            messagebox.showinfo("Already in Clipboard", "This box is already in the clipboard")
    
    def show_clipboard_viewer(self):
        """Show a window to view and manage clipboard contents"""
        if not self.label_clipboard:
            messagebox.showinfo("Empty Clipboard", "Clipboard is empty.\n\nRight-click on boxes and select 'Add to Clipboard' to add labels.")
            return
        
        # Create clipboard viewer window
        viewer = tk.Toplevel(self.root)
        viewer.title("Label Clipboard")
        viewer.geometry("400x500")
        
        # Title
        ttk.Label(viewer, text=f"Clipboard ({len(self.label_clipboard)} labels)", 
                 font=('Arial', 12, 'bold')).pack(pady=10)
        
        # Clear all button
        btn_frame = ttk.Frame(viewer)
        btn_frame.pack(pady=5)
        ttk.Button(btn_frame, text="Clear All", 
                  command=lambda: self.clear_clipboard(viewer)).pack(side=tk.LEFT, padx=5)
        
        # Scrollable frame for clipboard items
        canvas = tk.Canvas(viewer)
        scrollbar = ttk.Scrollbar(viewer, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Add items to clipboard viewer
        for idx, (cls_id, x1, y1, x2, y2) in enumerate(self.label_clipboard):
            item_frame = ttk.Frame(scrollable_frame, relief=tk.RAISED, borderwidth=1)
            item_frame.pack(fill=tk.X, padx=10, pady=5)
            
            cls_name = self.class_info[cls_id]['name']
            color = self.class_info[cls_id]['color']
            
            # Color indicator
            color_hex = '#%02x%02x%02x' % (color[2], color[1], color[0])  # BGR to RGB
            color_label = tk.Label(item_frame, text="  ", bg=color_hex, width=2)
            color_label.pack(side=tk.LEFT, padx=5)
            
            # Label info
            info_text = f"{idx+1}. {cls_name}\n   Box: ({int(x1)}, {int(y1)}) → ({int(x2)}, {int(y2)})"
            ttk.Label(item_frame, text=info_text).pack(side=tk.LEFT, padx=5)
            
            # Remove button
            ttk.Button(item_frame, text="×", width=2,
                      command=lambda i=idx: self.remove_from_clipboard(i, viewer)).pack(side=tk.RIGHT, padx=5)
        
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    def remove_from_clipboard(self, index, viewer_window):
        """Remove a specific item from clipboard"""
        if 0 <= index < len(self.label_clipboard):
            removed = self.label_clipboard.pop(index)
            cls_name = self.class_info[removed[0]]['name']
            self.update_status(f"Removed {cls_name} from clipboard")
            
            # Refresh viewer
            viewer_window.destroy()
            if self.label_clipboard:  # Only show if clipboard still has items
                self.show_clipboard_viewer()
    
    def clear_clipboard(self, viewer_window):
        """Clear all items from clipboard"""
        result = messagebox.askyesno("Clear Clipboard", 
                                    f"Remove all {len(self.label_clipboard)} labels from clipboard?")
        if result:
            self.label_clipboard.clear()
            self.update_status("Clipboard cleared")
            viewer_window.destroy()
    
    def paste_labels(self):
        """Paste labels from clipboard to current image"""
        if not self.current_image_path:
            messagebox.showwarning("No Image", "Please load an image first")
            return
        
        if not self.label_clipboard:
            messagebox.showinfo("Empty Clipboard", "No labels in clipboard. Use Copy Labels first.")
            return
        
        # Add clipboard labels to user boxes
        pasted_count = 0
        for box in self.label_clipboard:
            cls_id, x1, y1, x2, y2 = box
            
            # Validate that class exists in current class_info
            if cls_id not in self.class_info:
                messagebox.showwarning(
                    "Unknown Class",
                    f"Class ID {cls_id} not found in current configuration. Skipping some labels."
                )
                continue
            
            # Add to user boxes (avoiding duplicates)
            box_tuple = (cls_id, x1, y1, x2, y2)
            if box_tuple not in self.user_boxes:
                self.user_boxes.append(box_tuple)
                pasted_count += 1
        
        # Re-detect overlaps if highlight mode is on (boxes added)
        if self.highlight_overlaps:
            self.detect_current_overlaps()
        
        # Save and refresh display
        self.save_labels()
        self.display_image()
        
        self.update_status(f"📄 Pasted {pasted_count} labels")
        
    def select_box_or_prediction(self, event):
        """Handle right-click to select/deselect boxes or predictions and show context menu"""
        if not (self.user_boxes or self.label_boxes or (self.model_boxes and self.show_model_predictions)):
            return
            
        # Get click coordinates in image space
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)
        x, y = self.canvas_to_image_coords(canvas_x, canvas_y)
        
        # Clear all selections first
        self.clear_selections()
        
        # Find all boxes that contain the click point and select the smallest one
        candidates = []
        
        # Check user boxes
        for i, (cls_id, x1, y1, x2, y2) in enumerate(self.user_boxes):
            if x1 <= x <= x2 and y1 <= y <= y2:
                area = (x2 - x1) * (y2 - y1)
                candidates.append(('user', i, area, cls_id))
        
        # Check label boxes
        for i, (cls_id, x1, y1, x2, y2) in enumerate(self.label_boxes):
            if x1 <= x <= x2 and y1 <= y <= y2:
                area = (x2 - x1) * (y2 - y1)
                candidates.append(('label', i, area, cls_id))
        
        # Check model predictions if visible
        if self.show_model_predictions:
            for i, (cls_id, x1, y1, x2, y2) in enumerate(self.model_boxes):
                if x1 <= x <= x2 and y1 <= y <= y2:
                    area = (x2 - x1) * (y2 - y1)
                    candidates.append(('prediction', i, area, int(cls_id)))
        
        # If we found any candidates, select the one with smallest area
        if candidates:
            # Sort by area (smallest first)
            candidates.sort(key=lambda c: c[2])
            source, index, area, cls_id = candidates[0]
            
            if source == 'prediction':
                self.selected_prediction = index
                self.highlight_selected_prediction()
                self.update_status(f"Selected prediction {index+1} (Class: {self.class_info[cls_id]['name']})")
                self.prediction_menu.post(event.x_root, event.y_root)
            else:
                self.selected_box_index = index
                self.selected_box_source = source
                self.highlight_selected_box()
                self.update_status(f"Selected {source} box {index+1} (Class: {self.class_info[cls_id]['name']})")
                self.box_menu.post(event.x_root, event.y_root)

    def delete_selected_box(self, event=None):
        """Delete the currently selected box"""
        if self.selected_box_index is not None:
            if self.selected_box_source == 'user':
                # Remove the box from user_boxes
                del self.user_boxes[self.selected_box_index]
                self.update_status("User box deleted")
            elif self.selected_box_source == 'label':
                # Remove the box from label_boxes
                del self.label_boxes[self.selected_box_index]
                self.update_status("Label box deleted")
            
            # Clear selection
            if self.selected_box_rect:
                self.canvas.delete(self.selected_box_rect)
            self.selected_box_index = None
            self.selected_box_source = None
            self.selected_box_rect = None
            
            # Re-detect overlaps if highlight mode is on (indices have changed!)
            if self.highlight_overlaps:
                self.detect_current_overlaps()
            
            # Automatically save labels after deletion
            self.save_labels()
            
            # Redraw
            self.display_image()

    def clear_selections(self):
        """Clear all selections (boxes and predictions)"""
        if self.selected_box_index is not None:
            if self.selected_box_rect:
                self.canvas.delete(self.selected_box_rect)
            self.selected_box_index = None
            self.selected_box_source = None
            self.selected_box_rect = None
            
        if self.selected_prediction is not None:
            if self.selected_prediction_rect:
                self.canvas.delete(self.selected_prediction_rect)
            # Also delete any prediction labels
            self.canvas.delete('prediction_label')
            self.selected_prediction = None
            self.selected_prediction_rect = None

    def highlight_selected_box(self):
        """Draw highlight around selected box"""
        if self.selected_box_index is not None:
            # Get the box based on its source
            if self.selected_box_source == 'user':
                boxes = self.user_boxes
            else:  # label
                boxes = self.label_boxes
                
            cls_id, x1, y1, x2, y2 = boxes[self.selected_box_index]
            
            # Convert numpy values to native Python types if needed
            if hasattr(cls_id, 'item'):  # Check if it's a numpy type
                cls_id = int(cls_id.item())
                
            color = self.class_info[cls_id]["color"]
            rgb_color = self.bgr_to_rgb_hex(color)
            
            # Delete previous highlight if exists
            if self.selected_box_rect:
                self.canvas.delete(self.selected_box_rect)
                
            # Convert to canvas coordinates
            zx1, zy1 = self.image_to_canvas_coords(x1, y1)
            zx2, zy2 = self.image_to_canvas_coords(x2, y2)
            
            # Create new highlight with fill and thicker dashed outline
            # Use a lighter version of the color for fill
            r = int(color[2])  # BGR to RGB
            g = int(color[1])
            b = int(color[0])
            # Create lighter version by mixing with white
            fill_color = f'#{r:02x}{g:02x}{b:02x}'
            
            # Create new highlight with fill and thicker dashed outline
            self.selected_box_rect = self.canvas.create_rectangle(
                zx1, zy1, zx2, zy2,
                outline=rgb_color,
                fill=fill_color,
                stipple='gray50',  # Use stipple pattern for transparency effect
                width=3,  # Thicker outline
                dash=(10, 5)  # Longer dashes for better visibility
            )
            
            # Bring the highlight to the front
            self.canvas.tag_raise(self.selected_box_rect)

    def prev_image(self):
        """Move to previous image or batch"""
        if self.batch_mode:
            # Move to previous batch
            batch_size = self.batch_size[0] * self.batch_size[1]
            if self.current_image_index >= batch_size:
                self.current_image_index -= batch_size
                self.load_batch_images()
                self.update_counter()
        else:
            # Normal single image navigation
            if self.current_image_index > 0:
                self.current_image_index -= 1
                self.user_boxes = []
                self.selected_box_index = None
                self.selected_box_rect = None
                self.load_image()
            
    def next_image(self):
        """Move to next image or batch"""
        if self.batch_mode:
            # Move to next batch
            batch_size = self.batch_size[0] * self.batch_size[1]
            if self.current_image_index + batch_size < len(self.filtered_image_paths):
                self.current_image_index += batch_size
                self.load_batch_images()
                self.update_counter()
        else:
            # Normal single image navigation
            if self.current_image_index < len(self.filtered_image_paths) - 1:
                self.current_image_index += 1
                self.user_boxes = []
                self.selected_box_index = None
                self.selected_box_rect = None
                self.load_image()
                
    def update_status(self, message=None):
        if message:
            self.status_var.set(message)
        else:
            status = []
            if self.image_folder:
                status.append(f"Images: {len(self.image_paths)}")
            if self.current_image_index >= 0:
                status.append(f"Current: {self.current_image_index + 1}")
            if self.label_folder:
                status.append(f"Label folder: {self.label_folder}")
            if self.pretrained_model:
                status.append("Model loaded")
            self.status_var.set(" | ".join(status))
        
    def toggle_predictions(self):
        """Toggle visibility of model predictions"""
        self.show_model_predictions = self.show_predictions_var.get()
        self.display_image()
        
    def toggle_label_boxes(self):
        """Toggle visibility of label boxes"""
        self.show_label_boxes = self.show_labels_var.get()
        self.display_image()
        
    def toggle_user_boxes(self):
        """Toggle visibility of user-drawn boxes"""
        self.show_user_boxes = self.show_drawings_var.get()
        self.display_image()
        
    def start_edit_box(self, event):
        """Handle double-click to start editing a box"""
        if not self.user_boxes:
            return
            
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)
        
        # Check if click is inside any user-drawn box
        for i, (cls_id, x1, y1, x2, y2) in enumerate(self.user_boxes):
            if x1 <= x <= x2 and y1 <= y <= y2:
                self.selected_box_index = i
                self.editing_box = True
                # Start dragging from the nearest corner
                corners = [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]
                distances = [((cx-x)**2 + (cy-y)**2)**0.5 for cx, cy in corners]
                nearest_corner = corners[distances.index(min(distances))]
                self.start_x, self.start_y = nearest_corner
                
                # Create edit rectangle with RGB hex color
                color = self.class_info[cls_id]["color"]
                rgb_color = self.bgr_to_rgb_hex(color)
                self.current_rect = self.canvas.create_rectangle(
                    x1, y1, x2, y2,
                    outline=rgb_color,
                    width=2
                )
                break
        
    def handle_zoom(self, event):
        """Handle zoom with mouse wheel, centered on cursor position"""
        # Get the current mouse position in canvas coordinates
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)
        
        # Get current scroll position as fractions
        old_x_fraction = canvas_x / (self.current_image.shape[1] * self.zoom_scale) if self.current_image is not None else 0
        old_y_fraction = canvas_y / (self.current_image.shape[0] * self.zoom_scale) if self.current_image is not None else 0
        
        # Store old zoom scale
        old_zoom = self.zoom_scale
        
        # Determine zoom direction
        if event.num == 5 or event.delta < 0:  # Scroll down or negative delta
            self.zoom_scale = max(self.min_zoom, self.zoom_scale / self.zoom_factor)
        elif event.num == 4 or event.delta > 0:  # Scroll up or positive delta
            self.zoom_scale = min(self.max_zoom, self.zoom_scale * self.zoom_factor)
        
        # Redraw image with new zoom level
        self.display_image()
        
        # Adjust scroll position to keep the same point under the cursor
        if self.current_image is not None:
            # Calculate new dimensions
            new_width = self.current_image.shape[1] * self.zoom_scale
            new_height = self.current_image.shape[0] * self.zoom_scale
            
            # Calculate where the cursor point should be after zoom
            new_canvas_x = old_x_fraction * new_width
            new_canvas_y = old_y_fraction * new_height
            
            # Calculate the offset needed to keep cursor at same position
            # We want the new canvas position to align with the window position
            offset_x = new_canvas_x - event.x
            offset_y = new_canvas_y - event.y
            
            # Scroll to the new position
            # Convert to fractions for xview/yview
            if new_width > 0:
                self.canvas.xview_moveto(offset_x / new_width)
            if new_height > 0:
                self.canvas.yview_moveto(offset_y / new_height)
        
        self.update_status(f"Zoom: {self.zoom_scale:.2f}x")

    def canvas_to_image_coords(self, x, y):
        """Convert canvas coordinates to image coordinates"""
        return int(x / self.zoom_scale), int(y / self.zoom_scale)

    def image_to_canvas_coords(self, x, y):
        """Convert image coordinates to canvas coordinates"""
        return x * self.zoom_scale, y * self.zoom_scale

    def toggle_class_filter(self):
        """Toggle class filter on/off"""
        self.filter_by_class_enabled = self.filter_class_var.get()
        if self.filter_by_class_enabled:
            self.filter_class_combo.config(state='readonly')
        else:
            self.filter_class_combo.config(state='disabled')

    def toggle_size_filter(self):
        """Toggle size filter on/off"""
        self.filter_by_size_enabled = self.filter_size_var.get()

    def toggle_name_filter(self):
        """Toggle filename filter on/off"""
        self.filter_by_name_enabled = self.filter_name_var.get()
    
    def toggle_overlap_filter(self):
        """Toggle overlapping boxes filter on/off"""
        self.filter_by_overlap_enabled = self.filter_overlap_var.get()
    
    def toggle_overlap_highlight(self):
        """Toggle highlighting of overlapping boxes"""
        self.highlight_overlaps = self.highlight_overlaps_var.get()
        # Detect overlaps for current image
        if self.highlight_overlaps:
            self.detect_current_overlaps()
        else:
            self.overlapping_boxes = []
        self.display_image()
    
    def detect_current_overlaps(self):
        """Detect and store indices of all overlapping boxes in current image"""
        self.overlapping_boxes = []
        
        if not self.label_boxes:
            return
        
        # Check all pairs of boxes
        for i, box1 in enumerate(self.label_boxes):
            if len(box1) < 5:  # Safety check
                continue
            cls1, x1_1, y1_1, x2_1, y2_1 = box1[:5]
            for j, box2 in enumerate(self.label_boxes):
                if i >= j:  # Skip same box and already-checked pairs
                    continue
                if len(box2) < 5:  # Safety check
                    continue
                cls2, x1_2, y1_2, x2_2, y2_2 = box2[:5]
                
                # Only check boxes with DIFFERENT classes
                if cls1 != cls2:
                    try:
                        iou = self.calculate_iou((x1_1, y1_1, x2_1, y2_1), (x1_2, y1_2, x2_2, y2_2))
                        if iou >= self.filter_overlap_threshold:
                            # Mark both boxes as overlapping
                            if i not in self.overlapping_boxes:
                                self.overlapping_boxes.append(i)
                            if j not in self.overlapping_boxes:
                                self.overlapping_boxes.append(j)
                    except Exception as e:
                        # Skip boxes that cause errors in IOU calculation
                        continue
    
    def toggle_presence_filter(self):
        """Toggle class presence/absence filter on/off"""
        self.filter_by_class_presence_enabled = self.filter_presence_var.get()
    
    def populate_presence_filter_checkboxes(self):
        """Populate checkboxes for class presence/absence filter"""
        # Clear existing checkboxes
        for widget in self.must_have_checkboxes_frame.winfo_children():
            widget.destroy()
        for widget in self.must_not_have_checkboxes_frame.winfo_children():
            widget.destroy()
        
        self.must_have_class_vars.clear()
        self.must_not_have_class_vars.clear()
        
        # Create checkboxes for each class
        classes = sorted(self.class_info.keys())
        for class_id in classes:
            class_name = self.class_info[class_id]['name']
            
            # Must have checkbox
            var_have = tk.BooleanVar(value=False)
            self.must_have_class_vars[class_id] = var_have
            cb_have = ttk.Checkbutton(
                self.must_have_checkboxes_frame,
                text=f"{class_id}: {class_name}",
                variable=var_have,
                command=self.update_presence_filter
            )
            cb_have.pack(anchor=tk.W, padx=5, pady=1)
            
            # Must NOT have checkbox
            var_not_have = tk.BooleanVar(value=False)
            self.must_not_have_class_vars[class_id] = var_not_have
            cb_not_have = ttk.Checkbutton(
                self.must_not_have_checkboxes_frame,
                text=f"{class_id}: {class_name}",
                variable=var_not_have,
                command=self.update_presence_filter
            )
            cb_not_have.pack(anchor=tk.W, padx=5, pady=1)
        
        # Update scroll regions
        self.must_have_checkboxes_frame.update_idletasks()
        self.must_not_have_checkboxes_frame.update_idletasks()
        self.must_have_canvas.configure(scrollregion=self.must_have_canvas.bbox("all"))
        self.must_not_have_canvas.configure(scrollregion=self.must_not_have_canvas.bbox("all"))
    
    def update_presence_filter(self):
        """Update the sets of must-have and must-not-have classes"""
        self.filter_must_have_classes.clear()
        self.filter_must_not_have_classes.clear()
        
        for class_id, var in self.must_have_class_vars.items():
            if var.get():
                self.filter_must_have_classes.add(class_id)
        
        for class_id, var in self.must_not_have_class_vars.items():
            if var.get():
                self.filter_must_not_have_classes.add(class_id)

    def update_class_filter_dropdown(self):
        """Update the class filter dropdown with available classes"""
        classes = list(self.class_info.keys())
        class_names = [f"{cid}: {self.class_info[cid]['name']}" for cid in classes]
        self.filter_class_combo['values'] = class_names
        if class_names:
            self.filter_class_combo.current(0)

    def update_filter(self, event=None):
        """Update filter parameters"""
        # Update class filter
        if self.filter_by_class_enabled:
            selected = self.filter_class_combo.get()
            if selected:
                try:
                    # Extract class ID from the format "ID: Name"
                    cls_id = int(selected.split(':', 1)[0].strip())
                    self.filter_class_id = cls_id
                except (ValueError, IndexError):
                    # Handle invalid format or empty string
                    pass
        
        # Update size filter
        try:
            self.filter_min_size = int(self.min_size_var.get())
        except ValueError:
            self.filter_min_size = 0
            self.min_size_var.set("0")
            
        try:
            self.filter_max_size = int(self.max_size_var.get())
        except ValueError:
            self.filter_max_size = 100000
            self.max_size_var.set("100000")

        # Add IOU filter update
        try:
            self.filter_iou_threshold = float(self.iou_thresh_var.get())
        except ValueError:
            self.filter_iou_threshold = 0.5
            self.iou_thresh_var.set("0.5")
        
        # Update overlap filter threshold
        try:
            self.filter_overlap_threshold = float(self.filter_overlap_threshold_var.get())
            # Clamp to 0.0-1.0 range
            if self.filter_overlap_threshold < 0.0:
                self.filter_overlap_threshold = 0.0
                self.filter_overlap_threshold_var.set("0.0")
            elif self.filter_overlap_threshold > 1.0:
                self.filter_overlap_threshold = 1.0
                self.filter_overlap_threshold_var.set("1.0")
            
            # Re-detect overlaps if highlight mode is on
            if self.highlight_overlaps:
                self.detect_current_overlaps()
                self.display_image()
        except ValueError:
            self.filter_overlap_threshold = 0.5
            self.filter_overlap_threshold_var.set("0.5")
        
        # Update filename filter
        self.filter_name_text = self.filter_name_text_var.get()
        self.filter_name_case_sensitive = self.filter_name_case_var.get()

    def apply_filters(self):
        """Apply all active filters to image paths, with progress and ETA."""
        if not self.image_paths:
            self.filter_result_var.set("No images loaded")
            return
        
        # Check if any filter requiring labels is enabled
        requires_labels = (self.filter_by_class_enabled or self.filter_by_size_enabled or 
                          self.filter_by_iou_enabled or self.filter_by_class_presence_enabled or
                          self.filter_by_overlap_enabled)
        
        if requires_labels and not self.label_folder and not hasattr(self, 'per_image_label_folders'):
            self.filter_result_var.set("Label folder not set (required for class/size/IOU/presence/overlap filters)")
            return
            
        self.progress_var.set(0)
        self.filter_result_var.set("Filtering...")
        self.root.update_idletasks()
        self.filtered_image_paths = []
        total_with_labels = 0
        total_images = len(self.image_paths)
        import time
        start_time = time.time()
        
        # Pre-compute label folder mapping for faster lookup
        label_folder_map = {}
        if hasattr(self, 'per_image_label_folders'):
            for img_folder, lbl_folder in self.per_image_label_folders.items():
                label_folder_map[img_folder] = lbl_folder
        
        # Pre-build a set of image paths that have labels (if cache is empty)
        # This avoids calling .exists() repeatedly in the loop
        has_label_set = set()
        if requires_labels and not self.labels_cache:
            self.filter_result_var.set("Checking which images have labels...")
            self.root.update_idletasks()
            
            # Build set of images with labels (single pass)
            for img_path in self.image_paths:
                img_path_str = str(img_path)
                label_path = None
                
                if self.label_folder:
                    label_path = Path(self.label_folder) / f"{img_path.stem}.txt"
                else:
                    for img_folder, lbl_folder in label_folder_map.items():
                        if img_path_str.startswith(img_folder):
                            label_path = Path(lbl_folder) / f"{img_path.stem}.txt"
                            break
                
                if label_path and label_path.exists():
                    has_label_set.add(img_path_str)
        
        # Process in batches with less frequent UI updates
        batch_size = 100
        ui_update_interval = 200  # Update UI every N images instead of every 10
        
        for batch_idx in range(0, total_images, batch_size):
            batch_end = min(batch_idx + batch_size, total_images)
            batch = self.image_paths[batch_idx:batch_end]
            
            for i, img_path in enumerate(batch):
                img_path_str = str(img_path)
                
                # Apply filename filter first (doesn't require label file)
                if self.filter_by_name_enabled and self.filter_name_text:
                    filename = img_path.name
                    search_text = self.filter_name_text
                    if not self.filter_name_case_sensitive:
                        filename = filename.lower()
                        search_text = search_text.lower()
                    if search_text not in filename:
                        continue
                
                # If no label-based filters, add the image
                if not requires_labels:
                    self.filtered_image_paths.append(img_path)
                else:
                    # Find label path - optimized lookup
                    label_path = None
                    if self.label_folder:
                        label_path = Path(self.label_folder) / f"{img_path.stem}.txt"
                    else:
                        # Fast lookup using pre-computed mapping
                        for img_folder, lbl_folder in label_folder_map.items():
                            if img_path_str.startswith(img_folder):
                                label_path = Path(lbl_folder) / f"{img_path.stem}.txt"
                                break
                    
                    # Check if label exists - use cache or pre-built set
                    if label_path:
                        has_label = (img_path_str in self.labels_cache or 
                                   img_path_str in has_label_set or
                                   (not has_label_set and label_path.exists()))
                        
                        if has_label:
                            total_with_labels += 1
                            if self.image_passes_filters(img_path, label_path):
                                self.filtered_image_paths.append(img_path)
                
                # Progress feedback - less frequent updates
                current_idx = batch_idx + i + 1
                if current_idx % ui_update_interval == 0 or current_idx == total_images:
                    elapsed = time.time() - start_time
                    progress = (current_idx / total_images) * 100
                    if current_idx > 0:
                        rate = elapsed / current_idx
                        remaining = total_images - current_idx
                        eta = rate * remaining
                        eta_str = time.strftime('%H:%M:%S', time.gmtime(eta))
                    else:
                        eta_str = '--:--:--'
                    elapsed_str = time.strftime('%H:%M:%S', time.gmtime(elapsed))
                    self.progress_var.set(progress)
                    self.filter_result_var.set(f"Filtering: {current_idx}/{total_images} | Elapsed: {elapsed_str} | ETA: {eta_str}")
                    self.root.update_idletasks()
        
        self.current_image_index = 0 if self.filtered_image_paths else -1
        self.progress_var.set(100)
        if self.current_image_index >= 0:
            self.load_image()
        else:
            self.current_image = None
            self.canvas.delete("all")
            self.canvas.create_text(
                self.canvas.winfo_width() // 2, 
                self.canvas.winfo_height() // 2,
                text="No images match the current filters",
                fill="white", font=("Arial", 14)
            )
        self.update_counter()
        count = len(self.filtered_image_paths)
        total = len(self.image_paths)
        if self.filter_by_class_enabled or self.filter_by_size_enabled or self.filter_by_iou_enabled:
            self.filter_result_var.set(f"Found: {count}/{total_with_labels} labeled images")
        elif self.filter_by_name_enabled:
            self.filter_result_var.set(f"Found: {count}/{total} images matching '{self.filter_name_text}'")
        else:
            self.filter_result_var.set(f"Showing all {count} images")
        self.update_status(f"Filter applied: {count}/{total} images match")

    def update_counter(self):
        """Update the image counter display"""
        if self.filtered_image_paths:
            current = self.current_image_index + 1
            total = len(self.filtered_image_paths)
            self.counter_var.set(f"Image: {current}/{total}")
        else:
            self.counter_var.set("Image: 0/0")

    def load_batch_images(self):
        """Load images for batch view with optimized performance"""
        if not self.filtered_image_paths:
            return
            
        # Calculate how many images to load
        batch_size = self.batch_size[0] * self.batch_size[1]
        start_idx = self.current_image_index
        
        # Ensure current_index is valid
        if start_idx < 0:
            start_idx = 0
        elif start_idx >= len(self.filtered_image_paths):
            start_idx = max(0, len(self.filtered_image_paths) - batch_size)
            
        # Update current_index if needed
        if self.current_image_index != start_idx:
            self.current_image_index = start_idx
        
        # Show loading message (no update_idletasks here to avoid lag)
        self.update_status("Loading batch images...")
        
        # Load all images first, then update canvases in one pass
        # This reduces UI update frequency
        for i, canvas in enumerate(self.batch_canvases):
            img_idx = start_idx + i
            if img_idx < len(self.filtered_image_paths):
                # Load the actual image (optimized - no intermediate updates)
                self.load_batch_image(img_idx, canvas)
            else:
                # Clear canvas if no image available
                canvas.delete("all")
                canvas.create_text(canvas.winfo_width()//2, canvas.winfo_height()//2,
                                 text="No Image", fill="white", anchor=tk.CENTER)
        
        # Single UI update at the end
        self.update_counter()
        self.update_status("Batch loaded")
        self.root.update_idletasks()

    def load_batch_image(self, img_idx, canvas):
        """Load and display a single image in batch view - optimized"""
        img_path = self.filtered_image_paths[img_idx]
        img_path_str = str(img_path)
        
        # Read image
        img = cv2.imread(img_path_str)
        if img is None:
            return
            
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Get canvas size
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()
        
        # Calculate scaling to fit while maintaining aspect ratio
        img_height, img_width = img_rgb.shape[:2]
        scale = min(canvas_width/img_width, canvas_height/img_height)
        new_width = int(img_width * scale)
        new_height = int(img_height * scale)
        
        # Resize image
        img_resized = cv2.resize(img_rgb, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        # Find label path - support both single and multi-folder
        label_path = None
        if self.label_folder:
            label_path = Path(self.label_folder) / f"{img_path.stem}.txt"
        else:
            # Multi-folder lookup
            for img_folder, lbl_folder in getattr(self, 'per_image_label_folders', {}).items():
                if img_path_str.startswith(img_folder):
                    label_path = Path(lbl_folder) / f"{img_path.stem}.txt"
                    break
        
        # Get labels from cache if available, otherwise read from file
        label_boxes = []
        if label_path and label_path.exists():
            label_boxes = self.get_label_boxes(img_path_str, label_path)
        
        # Draw labels on image
        label_count = len(label_boxes) if label_boxes else 0
        if label_boxes:
            for cls_id, x1, y1, x2, y2 in label_boxes:
                # Scale coordinates to fit resized image
                x1_scaled = int(x1 * scale)
                y1_scaled = int(y1 * scale)
                x2_scaled = int(x2 * scale)
                y2_scaled = int(y2 * scale)
                
                # Get class color
                if cls_id in self.class_info:
                    color_raw = self.class_info[cls_id]["color"]
                    
                    # Convert color to RGB tuple for drawing on RGB image
                    # (image_resized is already in RGB format)
                    if isinstance(color_raw, (list, tuple)) and len(color_raw) >= 3:
                        # Assuming stored as BGR - reverse to RGB
                        color_rgb = tuple(int(c) for c in reversed(color_raw[:3]))
                    elif isinstance(color_raw, str) and color_raw.startswith('#'):
                        # Hex color - convert to RGB
                        r = int(color_raw[1:3], 16)
                        g = int(color_raw[3:5], 16)
                        b = int(color_raw[5:7], 16)
                        color_rgb = (r, g, b)  # RGB format
                    else:
                        # Fallback - white
                        color_rgb = (255, 255, 255)
                    
                    # Draw rectangle
                    cv2.rectangle(img_resized, (x1_scaled, y1_scaled), (x2_scaled, y2_scaled), 
                                color_rgb, 2)  # Thicker for batch view
                    
                    # Add class name
                    class_name = self.class_info[cls_id]["name"]
                    font_scale = 0.4  # Smaller font for batch view
                    thickness = 1
                    cv2.putText(img_resized, class_name, 
                              (x1_scaled, y1_scaled-5),
                              cv2.FONT_HERSHEY_SIMPLEX, font_scale, 
                              color_rgb, thickness)
        
        # Convert to PhotoImage
        img_pil = Image.fromarray(img_resized)
        photo = ImageTk.PhotoImage(image=img_pil)
        
        # Store reference to prevent garbage collection
        canvas.photo = photo
        
        # Clear canvas and display image
        canvas.delete("all")
        canvas.create_image(canvas_width//2, canvas_height//2, 
                          image=photo, anchor=tk.CENTER)
        
        # Add black background for text overlays
        canvas.create_rectangle(0, 0, canvas_width, 70, 
                              fill='#000000', stipple='gray75', 
                              outline='', tags='overlay')
        
        # Add image number label
        canvas.create_text(10, 10, text=f"#{img_idx + 1}/{len(self.filtered_image_paths)}", 
                         fill="white", anchor=tk.NW, font=('Arial', 10, 'bold'),
                         tags='overlay')
        
        # Add filename
        filename = img_path.name
        # Truncate if too long
        if len(filename) > 25:
            filename = filename[:22] + '...'
        canvas.create_text(10, 30, text=f"{filename}", 
                         fill="yellow", anchor=tk.NW, font=('Arial', 9),
                         tags='overlay')
                         
        # Add label count
        canvas.create_text(10, 50, text=f"Boxes: {label_count}", 
                         fill="lime", anchor=tk.NW, font=('Arial', 9),
                         tags='overlay')
                             
    def select_batch_image(self, event, canvas_idx):
        """Handle click on batch image to enter edit mode"""
        if not self.batch_mode:
            return
            
        # Calculate image index
        img_idx = self.current_image_index + canvas_idx
        
        if img_idx >= len(self.filtered_image_paths):
            return
            
        # Switch to single image view for editing
        self.batch_mode = False
        self.editing_batch_image = img_idx
        self.current_image_index = img_idx
        
        # Show single image view
        self.batch_frame.pack_forget()
        self.canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Load the selected image
        self.load_image()
        
        # Add "Back to Batch" button
        if not hasattr(self, 'back_to_batch_btn'):
            self.back_to_batch_btn = ttk.Button(self.control_panel, 
                                              text="Back to Batch View",
                                              command=self.back_to_batch_view)
            self.back_to_batch_btn.pack(pady=5)
            
    def back_to_batch_view(self):
        """Return to batch view from edit mode"""
        if self.editing_batch_image is not None:
            # Save any changes made in edit mode
            self.save_labels()
            
            # Remove back button
            if hasattr(self, 'back_to_batch_btn'):
                self.back_to_batch_btn.destroy()
                delattr(self, 'back_to_batch_btn')
            
            # Calculate which batch to return to based on edited image index
            batch_size = self.batch_size[0] * self.batch_size[1]
            # Set current_image_index to the start of the batch containing the edited image
            self.current_image_index = (self.editing_batch_image // batch_size) * batch_size
            
            # Switch back to batch view
            self.batch_mode = True
            self.editing_batch_image = None
            self.canvas_frame.pack_forget()
            
            # Show existing batch frame (don't recreate)
            if self.batch_frame and self.batch_frame.winfo_exists():
                self.batch_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
                # Just reload images without recreating canvases
                self.root.update_idletasks()
                self.load_batch_images()
            else:
                # If batch frame doesn't exist, create it
                self.setup_batch_view()
                self.root.update_idletasks()
                self.load_batch_images()
            
    def toggle_batch_view(self):
        """Toggle between single image and batch view"""
        batch_size = self.batch_size_var.get()
        rows, cols = map(int, batch_size.split('x'))
        self.batch_size = (rows, cols)
        
        # Clear existing view
        if self.batch_frame:
            self.batch_frame.destroy()
            self.batch_frame = None
        self.canvas_frame.pack_forget()
        
        if batch_size == "1x1":
            # Switch to single image view
            self.batch_mode = False
            self.canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            self.load_image()  # Reload current image
        else:
            # Switch to batch view
            self.batch_mode = True
            self.setup_batch_view()
            
            # Force layout update
            self.root.update_idletasks()
            
            # Reload batch images with current index
            self.load_batch_images()
            
    def setup_batch_view(self):
        """Setup the batch view grid"""
        self.batch_frame = ttk.Frame(self.main_frame)
        self.batch_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Create grid of canvases
        self.batch_canvases = []
        for row in range(self.batch_size[0]):
            for col in range(self.batch_size[1]):
                canvas = tk.Canvas(self.batch_frame, bg='gray', cursor="arrow")
                canvas.grid(row=row, column=col, sticky="nsew")
                canvas.bind("<Button-1>", lambda e, idx=len(self.batch_canvases): self.select_batch_image(e, idx))
                self.batch_canvases.append(canvas)
        
        # Configure grid weights
        for i in range(self.batch_size[0]):
            self.batch_frame.grid_rowconfigure(i, weight=1)
        for i in range(self.batch_size[1]):
            self.batch_frame.grid_columnconfigure(i, weight=1)
            
        # Force layout update
        self.batch_frame.update_idletasks()
        
    def add_new_class(self):
        """Add a new class with a dialog for name input"""
        name = simpledialog.askstring("New Class", "Enter class name:")
        if name:
            # Find the next available class ID
            next_id = max(self.class_info.keys()) + 1 if self.class_info else 0
            
            # Generate a distinct color for the new class
            color = self.generate_distinct_color()
            
            # Add new class
            self.class_info[next_id] = {"name": name, "color": color}
            
            # Save configuration
            self.save_class_config()
            
            # Refresh class buttons
            self.setup_class_buttons()
            
            # Select the new class
            self.select_class(next_id)

    def edit_class_name(self, class_id):
        """Edit the name of a class"""
        current_name = self.class_info[class_id]["name"]
        new_name = simpledialog.askstring("Edit Class", "Enter new name:",
                                        initialvalue=current_name)
        if new_name:
            self.class_info[class_id]["name"] = new_name
            self.save_class_config()
            self.class_labels[class_id].configure(text=new_name)

    def delete_class(self, class_id):
        """Delete a class if it's not the last one"""
        if len(self.class_info) <= 1:
            messagebox.showwarning("Warning", "Cannot delete the last class")
            return
            
        if messagebox.askyesno("Confirm Delete", 
                              f"Delete class '{self.class_info[class_id]['name']}'?"):
            # Remove class from configuration
            del self.class_info[class_id]
            
            # Update any boxes using this class to use the first available class
            first_class = min(self.class_info.keys())
            for boxes in [self.user_boxes, self.label_boxes]:
                for i, box in enumerate(boxes):
                    if box[0] == class_id:
                        boxes[i] = (first_class, *box[1:])
            
            # Save configuration
            self.save_class_config()
            
            # Refresh class buttons
            self.setup_class_buttons()
            
            # If current class was deleted, select the first available class
            if self.current_class == class_id:
                self.select_class(first_class)
            
            # Redraw to update any boxes
            self.display_image()

    def edit_selected_box(self):
        """Start editing the selected box"""
        if self.selected_box_index is not None:
            # Get the box based on its source
            if self.selected_box_source == 'user':
                boxes = self.user_boxes
            else:  # label
                boxes = self.label_boxes
                
            cls_id, x1, y1, x2, y2 = boxes[self.selected_box_index]
            
            # Start editing mode
            self.editing_box = True
            self.start_x, self.start_y = x1, y1  # Start from top-left corner
            
            # Create edit rectangle with RGB hex color
            color = self.class_info[cls_id]["color"]
            rgb_color = self.bgr_to_rgb_hex(color)
            
            # Convert to canvas coordinates
            zx1, zy1 = self.image_to_canvas_coords(x1, y1)
            zx2, zy2 = self.image_to_canvas_coords(x2, y2)
            
            # Create rectangle for editing
            self.current_rect = self.canvas.create_rectangle(
                zx1, zy1, zx2, zy2,
                outline=rgb_color,
                width=2
            )
            
    def select_class(self, class_id):
        self.current_class = class_id
        self.selected_class_var.set(f"Selected Class: {class_id} ({self.class_info[class_id]['name']})")

    def delete_current_image(self):
        """Delete the current image and its corresponding label file"""
        if not self.current_image_path:
            messagebox.showinfo("Info", "No image loaded to delete")
            return
            
        # Ask for confirmation
        if not messagebox.askyesno("Confirm Delete", 
                                 f"Are you sure you want to delete:\n{self.current_image_path.name}\n\nThis action cannot be undone!"):
            return
            
        # Store the current index
        current_idx = self.current_image_index
        
        # Delete the image file
        try:
            os.remove(self.current_image_path)
            self.update_status(f"Deleted image: {self.current_image_path.name}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to delete image: {str(e)}")
            return
            
        # Delete the label file if it exists
        if self.label_folder:
            label_path = Path(self.label_folder) / f"{self.current_image_path.stem}.txt"
            if label_path.exists():
                try:
                    os.remove(label_path)
                    self.update_status(f"Deleted label: {label_path.name}")
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to delete label: {str(e)}")
        
        # Get the deleted path
        deleted_path = self.filtered_image_paths[current_idx]
        deleted_path_str = str(deleted_path)
        
        # Remove from cache
        if deleted_path_str in self.image_dimensions_cache:
            del self.image_dimensions_cache[deleted_path_str]
        if deleted_path_str in self.labels_cache:
            del self.labels_cache[deleted_path_str]
        
        # Remove the item from both image_paths and filtered_image_paths
        if deleted_path in self.image_paths:
            self.image_paths.remove(deleted_path)
        self.filtered_image_paths.remove(deleted_path)
        
        # Load the next image or the previous one if this was the last
        if not self.filtered_image_paths:
            # No more images
            self.current_image_path = None
            self.current_image = None
            self.current_image_index = -1
            self.canvas.delete("all")
            self.update_status("No images left in folder")
            self.update_counter()
        else:
            # Adjust index if needed
            if current_idx >= len(self.filtered_image_paths):
                self.current_image_index = len(self.filtered_image_paths) - 1
            else:
                self.current_image_index = current_idx
                
            # Load the image
            self.load_image()

    def show_class_reassignment_menu(self):
        """Show submenu with available classes for reassignment"""
        if self.selected_box_index is None:
            return
            
        # Clear existing menu items
        self.class_reassign_menu.delete(0, 'end')
        
        # Get current box class
        if self.selected_box_source == 'user':
            current_class = self.user_boxes[self.selected_box_index][0]
        else:  # label
            current_class = self.label_boxes[self.selected_box_index][0]
        
        # Add class options to menu
        for class_id in sorted(self.class_info.keys()):
            # Convert color to hex for display
            color = self.class_info[class_id]["color"]
            color_hex = self.bgr_to_rgb_hex(color)
            
            # Create a checkmark for the current class
            is_current = "✓ " if class_id == current_class else "  "
            
            # Add menu item with color indicator
            label = f"{is_current}{class_id}: {self.class_info[class_id]['name']}"
            self.class_reassign_menu.add_command(
                label=label,
                command=lambda id=class_id: self.reassign_box_class(id),
                background=color_hex
            )
            
        # Show the popup menu at the current mouse position
        try:
            self.class_reassign_menu.tk_popup(
                self.box_menu.winfo_pointerx(),
                self.box_menu.winfo_pointery()
            )
        finally:
            # Make sure to release the grab
            self.class_reassign_menu.grab_release()
            
    def reassign_box_class(self, new_class_id):
        """Reassign the class of the selected box"""
        if self.selected_box_index is None:
            return
            
        # Update the class ID based on the box source
        if self.selected_box_source == 'user':
            # Get current box coordinates
            _, x1, y1, x2, y2 = self.user_boxes[self.selected_box_index]
            # Update with new class
            self.user_boxes[self.selected_box_index] = (new_class_id, x1, y1, x2, y2)
        else:  # label
            # Get current box coordinates
            _, x1, y1, x2, y2 = self.label_boxes[self.selected_box_index]
            # Update with new class
            self.label_boxes[self.selected_box_index] = (new_class_id, x1, y1, x2, y2)
            
        # Clear selection
        self.clear_selections()
        
        # Re-detect overlaps if highlight mode is on (class change can affect overlaps!)
        if self.highlight_overlaps:
            self.detect_current_overlaps()
        
        # Save labels to update the file
        self.save_labels()
        
        # Redraw to show updated colors
        self.display_image()
        
        self.update_status(f"Changed class to: {self.class_info[new_class_id]['name']}")

    def jump_to_image(self):
        """Jump to a specific image number"""
        try:
            # Get the target image number (1-based)
            target_num = int(self.jump_var.get())
            
            # Convert to 0-based index
            target_idx = target_num - 1
            
            # Validate the index
            if 0 <= target_idx < len(self.filtered_image_paths):
                self.current_image_index = target_idx
                self.user_boxes = []  # Clear user boxes
                self.selected_box_index = None  # Clear selection
                self.selected_box_rect = None
                self.load_image()
            else:
                messagebox.showwarning("Warning", f"Image number must be between 1 and {len(self.filtered_image_paths)}")
        except ValueError:
            messagebox.showwarning("Warning", "Please enter a valid number")

    def toggle_batch_view(self):
        """Toggle between single image and batch view"""
        batch_size = self.batch_size_var.get()
        rows, cols = map(int, batch_size.split('x'))
        self.batch_size = (rows, cols)
        
        # Clear existing view
        if self.batch_frame:
            self.batch_frame.destroy()
            self.batch_frame = None
        self.canvas_frame.pack_forget()
        
        if batch_size == "1x1":
            # Switch to single image view
            self.batch_mode = False
            self.canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            self.load_image()  # Reload current image
        else:
            # Switch to batch view
            self.batch_mode = True
            self.setup_batch_view()
            
            # Force layout update
            self.root.update_idletasks()
            
            # Reload batch images with current index
            self.load_batch_images()
            
    def setup_batch_view(self):
        """Setup the batch view grid"""
        self.batch_frame = ttk.Frame(self.main_frame)
        self.batch_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Create grid of canvases
        self.batch_canvases = []
        for row in range(self.batch_size[0]):
            for col in range(self.batch_size[1]):
                canvas = tk.Canvas(self.batch_frame, bg='gray', cursor="arrow")
                canvas.grid(row=row, column=col, sticky="nsew")
                canvas.bind("<Button-1>", lambda e, idx=len(self.batch_canvases): self.select_batch_image(e, idx))
                self.batch_canvases.append(canvas)
        
        # Configure grid weights
        for i in range(self.batch_size[0]):
            self.batch_frame.grid_rowconfigure(i, weight=1)
        for i in range(self.batch_size[1]):
            self.batch_frame.grid_columnconfigure(i, weight=1)
            
        # Force layout update
        self.batch_frame.update_idletasks()
        

    def select_batch_image(self, event, canvas_idx):
        """Handle click on batch image to enter edit mode"""
        if not self.batch_mode:
            return
            
        # Calculate image index
        img_idx = self.current_image_index + canvas_idx
        
        if img_idx >= len(self.filtered_image_paths):
            return
            
        # Switch to single image view for editing
        self.batch_mode = False
        self.editing_batch_image = img_idx
        self.current_image_index = img_idx
        
        # Show single image view
        self.batch_frame.pack_forget()
        self.canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Load the selected image
        self.load_image()
        
        # Add "Back to Batch" button
        if not hasattr(self, 'back_to_batch_btn'):
            self.back_to_batch_btn = ttk.Button(self.control_panel, 
                                              text="Back to Batch View",
                                              command=self.back_to_batch_view)
            self.back_to_batch_btn.pack(pady=5)
            
    def generate_distinct_color(self):
        """Generate a random color that is distinct from existing colors"""
        existing_colors = [tuple(info['color']) for info in self.class_info.values()]
        while True:
            # Generate bright colors by ensuring at least one component is > 128
            color = [random.randint(0, 255) for _ in range(3)]
            if max(color) < 128:
                continue
            if tuple(color) not in existing_colors:
                return color

    def setup_class_buttons(self):
        """Setup class selection buttons with edit/delete options"""
        # Clear existing buttons
        for widget in self.class_frame.winfo_children():
            widget.destroy()
        self.class_buttons.clear()
        self.class_labels.clear()
        
        # Create scrollable frame for classes
        canvas = tk.Canvas(self.class_frame)
        scrollbar = ttk.Scrollbar(self.class_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Create class buttons
        for class_id in sorted(self.class_info.keys()):
            frame = ttk.Frame(scrollable_frame)
            frame.pack(fill=tk.X, pady=2)
            
            # Checkbox for visibility
            var = tk.BooleanVar(value=True)
            self.class_checkboxes[class_id] = var
            self.class_visibility[class_id] = True
            
            checkbox = ttk.Checkbutton(frame, variable=var,
                                     command=lambda id=class_id: self.toggle_class_visibility(id))
            checkbox.pack(side=tk.LEFT)
            
            # Color button
            color = self.class_info[class_id]["color"]
            rgb_color = self.bgr_to_rgb_hex(color)
            btn = tk.Button(frame, width=2, bg=rgb_color, relief=tk.RAISED)
            btn.configure(command=lambda id=class_id: self.select_class(id))
            btn.pack(side=tk.LEFT, padx=2)
            self.class_buttons[class_id] = btn
            
            # Class name label (clickable for editing)
            label = ttk.Label(frame, text=self.class_info[class_id]["name"])
            label.pack(side=tk.LEFT, padx=5)
            label.bind('<Double-Button-1>', lambda e, id=class_id: self.edit_class_name(id))
            self.class_labels[class_id] = label
            
            # Delete button
            if len(self.class_info) > 1:  # Only show delete button if more than one class
                delete_btn = ttk.Button(frame, text="×", width=2,
                                      command=lambda id=class_id: self.delete_class(id))
                delete_btn.pack(side=tk.RIGHT, padx=2)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    def toggle_class_visibility(self, class_id):
        """Toggle visibility of a class"""
        self.class_visibility[class_id] = self.class_checkboxes[class_id].get()
        self.display_image()

    def highlight_selected_prediction(self):
        """Draw highlight around selected prediction"""
        if self.selected_prediction is not None:
            cls_id, x1, y1, x2, y2 = self.model_boxes[self.selected_prediction]
            color = self.class_info[int(cls_id)]["color"]
            rgb_color = self.bgr_to_rgb_hex(color)
            
            # Delete previous highlight if exists
            if self.selected_prediction_rect:
                self.canvas.delete(self.selected_prediction_rect)
                
            # Convert to canvas coordinates
            zx1, zy1 = self.image_to_canvas_coords(x1, y1)
            zx2, zy2 = self.image_to_canvas_coords(x2, y2)
            
            # Create a distinct fill color for predictions
            r = int(color[2])  # BGR to RGB
            g = int(color[1])
            b = int(color[0])
            fill_color = f'#{r:02x}{g:02x}{b:02x}'
            
            # Create new highlight with dotted outline and different stipple pattern
            self.selected_prediction_rect = self.canvas.create_rectangle(
                zx1, zy1, zx2, zy2,
                outline=rgb_color,
                fill=fill_color,
                stipple='gray25',  # Different stipple pattern from label boxes
                width=2,  # Thinner outline for predictions
                dash=(5, 5)  # Shorter dashes for predictions
            )
            
            # Add a "P" label to indicate it's a prediction
            label_x = zx1 + 5
            label_y = zy1 - 15
            prediction_num = self.selected_prediction + 1
            self.canvas.create_text(
                label_x, label_y,
                text=f"P{prediction_num}",
                fill=rgb_color,
                font=('TkDefaultFont', 10, 'bold'),
                tags=('prediction_label',)
            )
            
            # Bring the highlight to the front
            self.canvas.tag_raise(self.selected_prediction_rect)

    def accept_selected_prediction(self, event=None):
        """Accept the currently selected prediction and add it to user boxes"""
        if self.selected_prediction is not None:
            # Get the selected prediction
            cls_id, x1, y1, x2, y2 = self.model_boxes[self.selected_prediction]
            
            # Convert numpy values to native Python types to avoid issues
            if hasattr(cls_id, 'item'):  # Check if it's a numpy type
                cls_id = int(cls_id.item())
            
            # Add the selected prediction to user boxes with converted class_id
            self.user_boxes.append((cls_id, x1, y1, x2, y2))
            
            # Clear the selection
            if self.selected_prediction_rect:
                self.canvas.delete(self.selected_prediction_rect)
            self.canvas.delete('prediction_label')
            self.selected_prediction = None
            self.selected_prediction_rect = None
            
            # Re-detect overlaps if highlight mode is on (new box added)
            if self.highlight_overlaps:
                self.detect_current_overlaps()
            
            # Automatically save labels after accepting prediction
            self.save_labels()
            
            # Redraw
            self.display_image()
            self.update_status("Prediction accepted")

    def accept_all_predictions(self, event=None):
        """Accept all model predictions and add them to user boxes"""
        if self.model_boxes:
            # Add all predictions to user boxes, converting numpy values to native Python types
            for cls_id, x1, y1, x2, y2 in self.model_boxes:
                # Convert numpy values to native Python types
                if hasattr(cls_id, 'item'):  # Check if it's a numpy type
                    cls_id = int(cls_id.item())
                self.user_boxes.append((cls_id, x1, y1, x2, y2))
            
            # Clear any selection
            self.clear_selections()
            
            # Re-detect overlaps if highlight mode is on (multiple boxes added)
            if self.highlight_overlaps:
                self.detect_current_overlaps()
            
            # Automatically save labels after accepting predictions
            self.save_labels()
            
            # Redraw
            self.display_image()
            self.update_status(f"Accepted {len(self.model_boxes)} predictions")

    def load_class_config(self):
        """Load class configuration from JSON file"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                    self.class_info = {int(k): v for k, v in config['classes'].items()}
            else:
                # Create default config if file doesn't exist
                self.class_info = {
                    0: {"name": "Default", "color": [0, 0, 255]}
                }
                self.save_class_config()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load class configuration: {str(e)}")
            self.class_info = {
                0: {"name": "Default", "color": [0, 0, 255]}
            }

    def load_classes_from_yaml(self, yaml_path):
        """Load class names and dataset paths from YOLO data.yaml file"""
        try:
            yaml_file = Path(yaml_path)
            with open(yaml_file, 'r') as f:
                data = yaml.safe_load(f)
                
            # Extract class names from the YAML file
            # YOLO data.yaml can have 'names' as a list or dict
            if 'names' not in data:
                raise ValueError("No 'names' field found in YAML file")
            
            names = data['names']
            
            # Generate distinct colors for each class
            self.class_info = {}
            
            if isinstance(names, list):
                # names is a list: ['class0', 'class1', ...]
                for idx, name in enumerate(names):
                    color = self.generate_distinct_color_by_index(idx, len(names))
                    self.class_info[idx] = {"name": name, "color": color}
            elif isinstance(names, dict):
                # names is a dict: {0: 'class0', 1: 'class1', ...}
                for idx, name in names.items():
                    idx = int(idx)
                    color = self.generate_distinct_color_by_index(idx, len(names))
                    self.class_info[idx] = {"name": name, "color": color}
            else:
                raise ValueError("Invalid 'names' format in YAML file")
            
            # Load image and label paths from YAML. Support strings or lists for train/val/test.
            loaded_images = False
            loaded_labels = False

            # Get base path - can be absolute or relative to YAML file
            if 'path' in data and data['path'] is not None:
                base_path = Path(data['path'])
                # If relative path, make it relative to YAML file location
                if not base_path.is_absolute():
                    base_path = yaml_file.parent / base_path
                base_path = base_path.resolve()
            else:
                # If no path specified, use YAML file directory
                base_path = yaml_file.parent.resolve()

            # Prepare per-image-folder label mapping and statistics
            self.per_image_label_folders = {}
            folder_stats = []  # Track (folder_path, image_count, label_path)
            
            def resolve_and_add_image_folder(entry):
                # entry can be a string path pointing directly to images or to a train folder
                p = Path(entry)
                if not p.is_absolute():
                    p = base_path / p
                p = p.resolve()

                # If the entry is the images folder already
                candidates = [p, p / 'images', p.parent / 'images']
                found = None
                for c in candidates:
                    if c.exists() and any(c.glob('*.[jp][pn]g')) or c.exists() and any(c.glob('*.bmp')):
                        found = c
                        break
                if not found:
                    # Accept folder even if it has other image file extensions
                    for c in candidates:
                        if c.exists() and any(c.iterdir()):
                            found = c
                            break
                if found:
                    # Count images before adding
                    before_count = len(self.image_paths)
                    
                    # Add images from this folder
                    self.add_images_from_folder(str(found))
                    
                    # Count images added
                    after_count = len(self.image_paths)
                    images_added = after_count - before_count
                    
                    # Try to find a nearby labels folder
                    label_folder_found = None
                    possible_label_paths = [
                        found.parent / 'labels',
                        found.parent.parent / 'labels',
                        base_path / 'labels' / found.name,
                        base_path / 'labels',
                    ]
                    for lbl in possible_label_paths:
                        if lbl.exists():
                            self.per_image_label_folders[str(found)] = str(lbl)
                            label_folder_found = str(lbl)
                            break
                    
                    # Record statistics
                    folder_stats.append((str(found), images_added, label_folder_found))
                    
                    return True
                return False

            # Handle train entries (list or single)
            if 'train' in data and data['train']:
                train_entries = data['train']
                if isinstance(train_entries, (list, tuple)):
                    for entry in train_entries:
                        try:
                            if resolve_and_add_image_folder(entry):
                                loaded_images = True
                        except Exception:
                            continue
                else:
                    try:
                        if resolve_and_add_image_folder(train_entries):
                            loaded_images = True
                    except Exception:
                        pass

            # If no train images found, try val/test or common folders
            if not loaded_images:
                for key in ('val', 'test'):
                    if key in data and data[key]:
                        entries = data[key]
                        if isinstance(entries, (list, tuple)):
                            for entry in entries:
                                try:
                                    if resolve_and_add_image_folder(entry):
                                        loaded_images = True
                                except Exception:
                                    continue
                        else:
                            try:
                                if resolve_and_add_image_folder(entries):
                                    loaded_images = True
                            except Exception:
                                pass

            # As a final fallback, try common image directories under base_path
            if not loaded_images:
                fallback_candidates = [
                    base_path / 'train' / 'images',
                    base_path / 'images' / 'train',
                    base_path / 'images',
                    base_path / 'train',
                ]
                for cand in fallback_candidates:
                    if cand.exists() and any(cand.iterdir()):
                        self.add_images_from_folder(str(cand))
                        loaded_images = True
                        # find labels near this candidate
                        lbl = cand.parent / 'labels'
                        if lbl.exists():
                            self.per_image_label_folders[str(cand)] = str(lbl)
                            loaded_labels = True
                        break
            
            # Build success message with detailed statistics
            msg_parts = [f"Loaded {len(self.class_info)} classes"]
            if loaded_images:
                msg_parts.append(f"{len(self.image_paths)} total images from {len(folder_stats)} folders")
            if loaded_labels:
                msg_parts.append("labels folder")
            
            # Build detailed log
            detail_log = f"Successfully loaded from: {yaml_path}\n\n"
            detail_log += f"Classes: {len(self.class_info)}\n"
            detail_log += f"Total Images: {len(self.image_paths)}\n"
            detail_log += f"Total Folders: {len(folder_stats)}\n\n"
            
            if folder_stats:
                detail_log += "Folder Details:\n"
                detail_log += "-" * 80 + "\n"
                for idx, (folder, count, label_folder) in enumerate(folder_stats, 1):
                    # Shorten folder path for display
                    short_folder = folder
                    if len(folder) > 60:
                        parts = folder.split('/')
                        short_folder = '/'.join(['...'] + parts[-3:])
                    
                    detail_log += f"{idx}. {short_folder}\n"
                    detail_log += f"   Images: {count}\n"
                    if label_folder:
                        short_label = label_folder
                        if len(label_folder) > 60:
                            parts = label_folder.split('/')
                            short_label = '/'.join(['...'] + parts[-3:])
                        detail_log += f"   Labels: {short_label}\n"
                    else:
                        detail_log += f"   Labels: Not found\n"
                    detail_log += "\n"
            
            # Print to console for logging
            print("\n" + "=" * 80)
            print("YAML LOADING SUMMARY")
            print("=" * 80)
            print(detail_log)
            print("=" * 80 + "\n")
            
            # Show simplified message in GUI
            messagebox.showinfo("YAML Loaded", 
                              f"Successfully loaded:\n" + "\n".join(f"  • {part}" for part in msg_parts) +
                              f"\n\nCheck console for detailed folder statistics.")
            
            # Update UI components that depend on classes
            if hasattr(self, 'update_class_filter_dropdown'):
                self.update_class_filter_dropdown()
            if hasattr(self, 'populate_presence_filter_checkboxes'):
                self.populate_presence_filter_checkboxes()
            
            # Load current image if available
            if self.image_paths and self.current_image_index < 0:
                self.current_image_index = 0
                self.load_image()
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load YAML file: {str(e)}")
            # Fall back to loading from JSON config
            self.load_class_config()

    def generate_distinct_color_by_index(self, idx, total):
        """Generate a distinct color based on index using HSV color space"""
        import colorsys
        
        # Use golden ratio for better color distribution
        golden_ratio = 0.618033988749895
        hue = (idx * golden_ratio) % 1.0
        
        # Use high saturation and value for vibrant colors
        saturation = 0.8 + (idx % 3) * 0.1  # Vary between 0.8-1.0
        value = 0.8 + (idx % 2) * 0.2       # Vary between 0.8-1.0
        
        # Convert HSV to RGB
        r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
        
        # Convert to 0-255 range and return in BGR format for OpenCV
        return (int(b * 255), int(g * 255), int(r * 255))

    def save_class_config(self):
        """Save class configuration to JSON file"""
        try:
            config = {'classes': {str(k): v for k, v in self.class_info.items()}}
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=4)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save class configuration: {str(e)}")

    def build_label_cache(self):
        """Build a cache of label data and image dimensions to speed up filtering"""
        if not self.image_paths:
            messagebox.showinfo("Info", "No images loaded")
            return
        
        # Check if we have label folders (either single or multi-folder)
        has_label_folders = self.label_folder or hasattr(self, 'per_image_label_folders')
        if not has_label_folders:
            messagebox.showinfo("Info", "No label folder set")
            return
        
        # Import PIL once at the start
        try:
            from PIL import Image
            use_pil = True
        except ImportError:
            use_pil = False
            
        # Reset progress bar
        self.progress_var.set(0)
        self.filter_result_var.set("Building cache...")
        self.root.update_idletasks()
        
        # Clear existing cache
        self.image_dimensions_cache = {}
        self.labels_cache = {}
        
        # Pre-compute label folder mapping for faster lookup
        label_folder_map = {}
        if hasattr(self, 'per_image_label_folders'):
            for img_folder, lbl_folder in self.per_image_label_folders.items():
                label_folder_map[img_folder] = lbl_folder
        
        total_images = len(self.image_paths)
        cached_count = 0
        import time
        import gc
        import os
        import sys
        start_time = time.time()
        last_update_time = start_time
        last_console_time = start_time
        current_folder = None
        folder_start_time = start_time
        folder_count = 0
        
        for i, img_path in enumerate(self.image_paths):
            # Track folder changes to detect slow folders
            img_folder_path = str(img_path.parent)
            if img_folder_path != current_folder:
                if current_folder and folder_count > 0:
                    folder_time = time.time() - folder_start_time
                    folder_speed = folder_count / folder_time if folder_time > 0 else 0
                    if folder_speed < 1000:  # Alert if folder is slow
                        print(f"  [SLOW FOLDER] {os.path.basename(current_folder)}: {folder_count} imgs in {folder_time:.1f}s ({folder_speed:.0f} img/s)")
                current_folder = img_folder_path
                folder_start_time = time.time()
                folder_count = 0
            folder_count += 1
            
            # Update UI only every 2 seconds (very infrequent to avoid slowdown)
            # Print to console more frequently for progress tracking
            current_time = time.time()
            should_update_ui = (current_time - last_update_time >= 2.0) or (i == total_images - 1)
            should_print = (current_time - last_console_time >= 1.0) or (i == total_images - 1)
            
            # Console output (doesn't slow down like UI)
            if should_print:
                elapsed = current_time - start_time
                progress = ((i + 1) / total_images) * 100
                speed = (i + 1) / elapsed if elapsed > 0 else 0
                
                if i > 0:
                    rate = elapsed / (i + 1)
                    remaining = total_images - (i + 1)
                    eta = rate * remaining
                    eta_str = time.strftime('%H:%M:%S', time.gmtime(eta))
                else:
                    eta_str = '--:--:--'
                
                # Print to console (fast, doesn't affect performance)
                print(f"Cache progress: {i+1}/{total_images} ({progress:.1f}%) | {speed:.0f} img/s | Cached: {cached_count} | ETA: {eta_str}")
                last_console_time = current_time
            
            # UI update (expensive, do rarely)
            if should_update_ui:
                elapsed = current_time - start_time
                progress = ((i + 1) / total_images) * 100
                speed = (i + 1) / elapsed if elapsed > 0 else 0
                
                if i > 0:
                    rate = elapsed / (i + 1)
                    remaining = total_images - (i + 1)
                    eta = rate * remaining
                    eta_str = time.strftime('%H:%M:%S', time.gmtime(eta))
                else:
                    eta_str = '--:--:--'
                
                self.progress_var.set(progress)
                self.filter_result_var.set(f"Cache: {i+1}/{total_images} ({speed:.0f} img/s) | ETA: {eta_str}")
                self.root.update_idletasks()
                last_update_time = current_time
            
            # VERY aggressive garbage collection to prevent cumulative slowdown
            # After 20k images, collect every 100 images
            # After 10k images, collect every 200 images
            # Before 10k, collect every 500 images
            if i > 20000:
                gc_interval = 100
            elif i > 10000:
                gc_interval = 200
            else:
                gc_interval = 500
            
            if (i + 1) % gc_interval == 0:
                gc.collect(2)  # Full collection including generation 2
            
            # Find label path - support both single and multi-folder
            img_path_str = str(img_path)
            # Intern frequently repeated strings to reduce memory fragmentation
            if i > 15000:
                img_path_str = sys.intern(img_path_str)
            
            label_path = None
            
            if self.label_folder:
                label_path = Path(self.label_folder) / f"{img_path.stem}.txt"
            else:
                # Multi-folder lookup
                for img_folder, lbl_folder in label_folder_map.items():
                    if img_path_str.startswith(img_folder):
                        label_path = Path(lbl_folder) / f"{img_path.stem}.txt"
                        break
            
            if not label_path:
                continue
            
            # Use try-except instead of exists() check - faster when file exists
            try:
                # Get image dimensions using direct file reading (NO PIL/cv2 memory issues!)
                img_width, img_height = None, None
                
                try:
                    # Try to get dimensions from image file header directly
                    import struct
                    import imghdr
                    
                    with open(img_path_str, 'rb') as f:
                        head = f.read(24)
                        img_type = imghdr.what(None, head)
                        
                        if img_type == 'png':
                            # PNG: width and height are at bytes 16-20 and 20-24
                            w, h = struct.unpack('>LL', head[16:24])
                            img_width, img_height = int(w), int(h)
                        elif img_type == 'jpeg':
                            # JPEG: need to parse segments
                            f.seek(0)
                            size = 2
                            ftype = 0
                            while not 0xc0 <= ftype <= 0xcf or ftype in (0xc4, 0xc8, 0xcc):
                                f.seek(size, 1)
                                byte = f.read(1)
                                while ord(byte) == 0xff:
                                    byte = f.read(1)
                                ftype = ord(byte)
                                size = struct.unpack('>H', f.read(2))[0] - 2
                            f.seek(1, 1)
                            h, w = struct.unpack('>HH', f.read(4))
                            img_width, img_height = int(w), int(h)
                        elif img_type == 'gif':
                            # GIF: width and height at bytes 6-8 and 8-10
                            w, h = struct.unpack('<HH', head[6:10])
                            img_width, img_height = int(w), int(h)
                        elif img_type == 'bmp':
                            # BMP: width and height at bytes 18-22 and 22-26
                            w, h = struct.unpack('<LL', head[18:26])
                            img_width, img_height = int(w), int(h)
                except Exception:
                    # Fallback: try cv2 (slower but reliable)
                    pass
                
                # If header reading failed, use cv2 as fallback
                if img_width is None or img_height is None:
                    img = cv2.imread(img_path_str, cv2.IMREAD_UNCHANGED)
                    if img is None:
                        continue
                    img_height, img_width = img.shape[:2]
                    del img
                    gc.collect()  # Force cleanup when using cv2
                
                self.image_dimensions_cache[img_path_str] = (img_width, img_height)
                
                # Read and parse label file (will raise FileNotFoundError if doesn't exist)
                labels = []
                label_file = None
                try:
                    label_file = open(label_path, 'r')
                    for line in label_file:
                        parts = line.strip().split()
                        if len(parts) < 5:
                            continue
                            
                        try:
                            cls_id = int(float(parts[0]))
                            x_center, y_center, w, h = map(float, parts[1:5])
                            
                            # Convert YOLO format to pixel coordinates
                            x1 = int((x_center - w/2) * img_width)
                            y1 = int((y_center - h/2) * img_height)
                            x2 = int((x_center + w/2) * img_width)
                            y2 = int((y_center + h/2) * img_height)
                            
                            box_area = (x2 - x1) * (y2 - y1)
                            labels.append((cls_id, x1, y1, x2, y2, box_area))
                        except (ValueError, IndexError):
                            continue
                finally:
                    if label_file:
                        label_file.close()
                        del label_file
                            
                self.labels_cache[img_path_str] = labels
                cached_count += 1
            except (FileNotFoundError, IOError):
                # Label file doesn't exist - skip
                continue
            except Exception:
                continue
            finally:
                # Explicit cleanup to prevent cumulative slowdown
                # Clear temporary objects after each iteration
                if i > 15000:  # Only after 15k images to avoid overhead
                    img_path_str = None
                    label_path = None
                    labels = None
        
        # Final garbage collection
        gc.collect()
        
        # Set progress to 100% when done
        self.progress_var.set(100)
        dimensions_cached = len(self.image_dimensions_cache)
        labels_cached = len(self.labels_cache)
        total_time = time.time() - start_time
        time_str = time.strftime('%H:%M:%S', time.gmtime(total_time))
        
        self.filter_result_var.set(f"Cache complete: {labels_cached} labels, {dimensions_cached} dimensions in {time_str}")
        messagebox.showinfo("Cache Built", 
                          f"Successfully cached:\n"
                          f"• {labels_cached} images with labels\n"
                          f"• {dimensions_cached} image dimensions\n"
                          f"• Total images scanned: {total_images}\n"
                          f"• Time taken: {time_str}\n\n"
                          f"Filtering will now be much faster!")
        
    def toggle_iou_filter(self):
        self.filter_by_iou_enabled = self.filter_iou_var.get()
        if self.filter_by_iou_enabled:
            # Enable threshold entry
            pass  # (Entry is always enabled for now)

    @staticmethod
    def calculate_iou(boxA, boxB):
        # box: (x1, y1, x2, y2)
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interW = max(0, xB - xA)
        interH = max(0, yB - yA)
        interArea = interW * interH
        boxAArea = max(0, (boxA[2] - boxA[0])) * max(0, (boxA[3] - boxA[1]))
        boxBArea = max(0, (boxB[2] - boxB[0])) * max(0, (boxB[3] - boxB[1]))
        unionArea = boxAArea + boxBArea - interArea
        if unionArea == 0:
            return 0.0
        return interArea / unionArea

    def track_inference_progress(self, current, total, start_time):
        """Update progress bar, status, and show ETA/elapsed time during model inference."""
        elapsed = time.time() - start_time
        if current > 0:
            rate = elapsed / current
            remaining = total - current
            eta = rate * remaining
            eta_str = time.strftime('%H:%M:%S', time.gmtime(eta))
        else:
            eta_str = '--:--:--'
        elapsed_str = time.strftime('%H:%M:%S', time.gmtime(elapsed))
        self.progress_var.set((current)/total*100)
        self.filter_result_var.set(f"Model predictions: {current}/{total} | Elapsed: {elapsed_str} | ETA: {eta_str}")
        self.root.update_idletasks()

    def precompute_model_predictions(self):
        """Run model on all images and cache predictions. Show progress bar and ETA."""
        if not self.image_paths or not self.pretrained_model:
            print("Cannot precompute: image_paths or pretrained_model missing")
            return
        
        print(f"Starting model predictions on {len(self.image_paths)} images...")
        self.progress_var.set(0)
        self.filter_result_var.set("Running model on all images...")
        self.root.update_idletasks()
        total = len(self.image_paths)
        start_time = time.time()
        total_predictions = 0
        
        for i, img_path in enumerate(self.image_paths):
            try:
                img = cv2.imread(str(img_path))
                preds = []
                if img is not None:
                    results = self.pretrained_model(img, verbose=False)
                    for r in results:
                        boxes = r.boxes
                        for box in boxes:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            cls_id = int(box.cls[0].cpu().numpy())
                            preds.append((cls_id, x1, y1, x2, y2))
                            total_predictions += 1
                self.model_predictions_cache[str(img_path)] = preds
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                self.model_predictions_cache[str(img_path)] = []
            if i % 10 == 0 or i == total - 1:
                self.track_inference_progress(i+1, total, start_time)
        
        self.progress_var.set(100)
        elapsed = time.time() - start_time
        elapsed_str = time.strftime('%H:%M:%S', time.gmtime(elapsed))
        print(f"Model predictions complete: {total_predictions} total predictions across {total} images")
        self.filter_result_var.set(f"Model predictions cached for {total} images ({total_predictions} boxes) | Total time: {elapsed_str}")

    # --- Utility Methods ---
    def get_label_boxes(self, img_path_str, label_path):
        """Return label boxes for an image as (cls_id, x1, y1, x2, y2) tuples."""
        # Check cache first
        if img_path_str in self.labels_cache:
            return [(l[0], l[1], l[2], l[3], l[4]) for l in self.labels_cache[img_path_str]]
        
        boxes = []
        
        # Get image dimensions (use cache or read header only - FAST!)
        if img_path_str in self.image_dimensions_cache:
            img_width, img_height = self.image_dimensions_cache[img_path_str]
        else:
            # Read only image header to get dimensions (much faster than cv2.imread)
            try:
                from PIL import Image
                with Image.open(img_path_str) as img:
                    img_width, img_height = img.size
                    self.image_dimensions_cache[img_path_str] = (img_width, img_height)
            except Exception:
                # Fallback to cv2 if PIL fails
                try:
                    img = cv2.imread(img_path_str, cv2.IMREAD_UNCHANGED)
                    if img is not None:
                        img_height, img_width = img.shape[:2]
                        self.image_dimensions_cache[img_path_str] = (img_width, img_height)
                    else:
                        return boxes  # Return empty if can't read image
                except Exception:
                    return boxes
        
        # Parse label file (skip existence check - already done in apply_filters)
        try:
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls_id, x_center, y_center, w, h = map(float, parts[:5])
                        x1 = int((x_center - w/2) * img_width)
                        y1 = int((y_center - h/2) * img_height)
                        x2 = int((x_center + w/2) * img_width)
                        y2 = int((y_center + h/2) * img_height)
                        boxes.append((int(cls_id), x1, y1, x2, y2))
        except Exception:
            pass  # Return empty boxes on error
        
        return boxes

    def get_pred_boxes(self, img_path_str):
        """Return prediction boxes for an image as (cls_id, x1, y1, x2, y2) tuples."""
        return self.model_predictions_cache.get(img_path_str, [])

    # --- Centralized Filter Logic ---
    def image_passes_filters(self, img_path, label_path):
        img_path_str = str(img_path)
        
        # Early exit if no label-based filters are enabled
        if not (self.filter_by_class_enabled or self.filter_by_size_enabled or 
                self.filter_by_iou_enabled or self.filter_by_class_presence_enabled or
                self.filter_by_overlap_enabled):
            return True
        
        # Label boxes (uses cache if available)
        label_boxes = self.get_label_boxes(img_path_str, label_path)
        
        # Early exit if no boxes and filters require boxes
        if not label_boxes:
            if (self.filter_by_class_enabled or self.filter_by_size_enabled or 
                self.filter_by_class_presence_enabled or self.filter_by_overlap_enabled):
                return False
        
        # Class presence/absence filter (check first - most specific and fast)
        if self.filter_by_class_presence_enabled:
            # Get all class IDs present in this image (single pass)
            present_classes = set(box[0] for box in label_boxes)
            
            # Check must-have classes: all must be present
            if self.filter_must_have_classes:
                if not self.filter_must_have_classes.issubset(present_classes):
                    return False
            
            # Check must-not-have classes: none should be present
            if self.filter_must_not_have_classes:
                if self.filter_must_not_have_classes.intersection(present_classes):
                    return False
        
        # Class/size filter (needs to check individual boxes)
        if self.filter_by_class_enabled or self.filter_by_size_enabled:
            match_found = False
            for box in label_boxes:
                cls_id, x1, y1, x2, y2 = box
                
                # Only compute area if size filter is enabled
                if self.filter_by_size_enabled:
                    box_area = (x2 - x1) * (y2 - y1)
                    size_match = self.filter_min_size <= box_area <= self.filter_max_size
                else:
                    size_match = True
                
                class_match = (not self.filter_by_class_enabled) or (cls_id == self.filter_class_id)
                
                if class_match and size_match:
                    match_found = True
                    break
            
            if not match_found:
                return False
        
        # IOU filter (most expensive - check last)
        if self.filter_by_iou_enabled:
            preds = self.get_pred_boxes(img_path_str)
            if not preds:
                return False
            
            found_diff = False
            # Check labels against predictions
            for lbox in label_boxes:
                l_cls, lx1, ly1, lx2, ly2 = lbox
                l_box = (lx1, ly1, lx2, ly2)
                best_iou = 0.0
                for mbox in preds:
                    m_cls, mx1, my1, mx2, my2 = mbox
                    m_box = (mx1, my1, mx2, my2)
                    iou = self.calculate_iou(l_box, m_box)
                    if iou > best_iou:
                        best_iou = iou
                        if best_iou >= self.filter_iou_threshold:
                            break  # Early exit if threshold met
                if best_iou < self.filter_iou_threshold:
                    found_diff = True
                    break
            
            # Check predictions against labels
            if not found_diff:
                for mbox in preds:
                    m_cls, mx1, my1, mx2, my2 = mbox
                    m_box = (mx1, my1, mx2, my2)
                    best_iou = 0.0
                    for lbox in label_boxes:
                        l_cls, lx1, ly1, lx2, ly2 = lbox
                        l_box = (lx1, ly1, lx2, ly2)
                        iou = self.calculate_iou(m_box, l_box)
                        if iou > best_iou:
                            best_iou = iou
                            if best_iou >= self.filter_iou_threshold:
                                break  # Early exit if threshold met
                    if best_iou < self.filter_iou_threshold:
                        found_diff = True
                        break
            
            if not found_diff:
                return False
        
        # Overlapping boxes filter (quality control - check for same object labeled as different classes)
        if self.filter_by_overlap_enabled:
            has_overlap = False
            # Check all pairs of boxes
            for i, box1 in enumerate(label_boxes):
                cls1, x1_1, y1_1, x2_1, y2_1 = box1
                for j, box2 in enumerate(label_boxes):
                    if i >= j:  # Skip same box and already-checked pairs
                        continue
                    cls2, x1_2, y1_2, x2_2, y2_2 = box2
                    
                    # Only check boxes with DIFFERENT classes
                    if cls1 != cls2:
                        iou = self.calculate_iou((x1_1, y1_1, x2_1, y2_1), (x1_2, y1_2, x2_2, y2_2))
                        if iou >= self.filter_overlap_threshold:
                            has_overlap = True
                            break
                if has_overlap:
                    break
            
            if not has_overlap:
                return False
        
        return True
    
    def analyze_dataset(self):
        """Analyze dataset and show visualizations"""
        if not self.image_paths:
            messagebox.showwarning("No Data", "Please load a dataset first")
            return
        
        if not self.label_folder and not hasattr(self, 'per_image_label_folders'):
            messagebox.showwarning("No Labels", "Label folder not set. Cannot analyze.")
            return
        
        # Determine sample size
        analyze_all = self.analyze_all_var.get()
        try:
            sample_size = int(self.sample_size_var.get())
        except ValueError:
            sample_size = 1000
            self.sample_size_var.set("1000")
        
        total_images = len(self.image_paths)
        if analyze_all or total_images <= sample_size:
            images_to_analyze = self.image_paths
        else:
            # Random sample
            import random
            images_to_analyze = random.sample(self.image_paths, min(sample_size, total_images))
        
        # Show progress
        self.analyze_progress_var.set(0)
        self.stats_text.delete(1.0, tk.END)
        self.stats_text.insert(tk.END, f"Analyzing {len(images_to_analyze)} images...\n")
        self.root.update_idletasks()
        
        # Collect statistics
        stats = self._collect_statistics(images_to_analyze)
        
        # Complete progress
        self.analyze_progress_var.set(100)
        
        # Display statistics
        self._display_statistics(stats)
        
        # Show visualizations in new window
        self._show_visualizations(stats)
    
    def refresh_analysis(self):
        """Refresh the current analysis"""
        self.analyze_dataset()
    
    def _collect_statistics(self, images_to_analyze):
        """Collect statistics from images with progress updates"""
        stats = {
            'total_images': len(images_to_analyze),
            'images_with_labels': 0,
            'total_boxes': 0,
            'class_counts': {},
            'class_areas': {},
            'class_positions_x': {},  # Normalized x centers
            'class_positions_y': {},  # Normalized y centers
            'class_widths': {},  # Normalized widths
            'class_heights': {},  # Normalized heights
            'boxes_per_image': [],
            'image_dimensions': [],
        }
        
        # Initialize for all classes
        for cls_id, cls_info in self.class_info.items():
            stats['class_counts'][cls_id] = 0
            stats['class_areas'][cls_id] = []
            stats['class_positions_x'][cls_id] = []
            stats['class_positions_y'][cls_id] = []
            stats['class_widths'][cls_id] = []
            stats['class_heights'][cls_id] = []
        
        # Progress tracking
        import time
        total = len(images_to_analyze)
        last_update_time = time.time()
        update_interval = 0.5  # Update UI every 0.5 seconds
        
        # Pre-build label folder map for faster lookup
        label_folder_map = {}
        if hasattr(self, 'per_image_label_folders'):
            for img_folder, lbl_folder in self.per_image_label_folders.items():
                label_folder_map[img_folder] = lbl_folder
        
        # Analyze images with progress updates
        for i, img_path in enumerate(images_to_analyze):
            img_path_str = str(img_path)
            
            # Update progress periodically
            current_time = time.time()
            if current_time - last_update_time >= update_interval or i == total - 1:
                progress_pct = ((i + 1) / total) * 100
                self.analyze_progress_var.set(progress_pct)
                self.stats_text.delete(1.0, tk.END)
                self.stats_text.insert(tk.END, f"Analyzing: {i+1}/{total} ({progress_pct:.1f}%)\n")
                self.stats_text.insert(tk.END, f"Images with labels: {stats['images_with_labels']}\n")
                self.stats_text.insert(tk.END, f"Total boxes found: {stats['total_boxes']}\n")
                self.root.update_idletasks()
                last_update_time = current_time
            
            # Get image dimensions (use cache or fast header reading)
            if img_path_str in self.image_dimensions_cache:
                img_width, img_height = self.image_dimensions_cache[img_path_str]
            else:
                # Use fast header reading instead of cv2
                img_width, img_height = None, None
                try:
                    import imghdr
                    import struct
                    with open(img_path_str, 'rb') as f:
                        head = f.read(24)
                        img_type = imghdr.what(None, head)
                        
                        if img_type == 'png':
                            w, h = struct.unpack('>LL', head[16:24])
                            img_width, img_height = int(w), int(h)
                        elif img_type == 'jpeg':
                            f.seek(0)
                            size = 2
                            ftype = 0
                            while not 0xc0 <= ftype <= 0xcf or ftype in (0xc4, 0xc8, 0xcc):
                                f.seek(size, 1)
                                byte = f.read(1)
                                while ord(byte) == 0xff:
                                    byte = f.read(1)
                                ftype = ord(byte)
                                size = struct.unpack('>H', f.read(2))[0] - 2
                            f.seek(1, 1)
                            h, w = struct.unpack('>HH', f.read(4))
                            img_width, img_height = int(w), int(h)
                except Exception:
                    # Fallback to cv2
                    img = cv2.imread(img_path_str)
                    if img is not None:
                        img_height, img_width = img.shape[:2]
                
                if img_width and img_height:
                    stats['image_dimensions'].append((img_width, img_height))
                    self.image_dimensions_cache[img_path_str] = (img_width, img_height)
                else:
                    continue
            
            # Find label path - optimized lookup
            label_path = None
            if self.label_folder:
                label_path = Path(self.label_folder) / f"{img_path.stem}.txt"
            else:
                # Multi-folder lookup using pre-built map
                for img_folder, lbl_folder in label_folder_map.items():
                    if img_path_str.startswith(img_folder):
                        label_path = Path(lbl_folder) / f"{img_path.stem}.txt"
                        break
            
            if not label_path or not label_path.exists():
                stats['boxes_per_image'].append(0)
                continue
            
            # Read labels
            boxes = self.get_label_boxes(img_path_str, label_path)
            if boxes:
                stats['images_with_labels'] += 1
                stats['boxes_per_image'].append(len(boxes))
                stats['total_boxes'] += len(boxes)
                
                for cls_id, x1, y1, x2, y2 in boxes:
                    if cls_id in stats['class_counts']:
                        stats['class_counts'][cls_id] += 1
                        
                        # Box area (pixels)
                        area = (x2 - x1) * (y2 - y1)
                        stats['class_areas'][cls_id].append(area)
                        
                        # Normalized positions and sizes
                        x_center = ((x1 + x2) / 2) / img_width
                        y_center = ((y1 + y2) / 2) / img_height
                        width = (x2 - x1) / img_width
                        height = (y2 - y1) / img_height
                        
                        stats['class_positions_x'][cls_id].append(x_center)
                        stats['class_positions_y'][cls_id].append(y_center)
                        stats['class_widths'][cls_id].append(width)
                        stats['class_heights'][cls_id].append(height)
            else:
                stats['boxes_per_image'].append(0)
        
        return stats
    
    def _display_statistics(self, stats):
        """Display statistics in text widget"""
        self.stats_text.delete(1.0, tk.END)
        
        text = f"{'='*60}\n"
        text += f"DATASET STATISTICS\n"
        text += f"{'='*60}\n\n"
        
        text += f"Total Images Analyzed: {stats['total_images']}\n"
        text += f"Images with Labels: {stats['images_with_labels']}\n"
        text += f"Total Bounding Boxes: {stats['total_boxes']}\n"
        
        if stats['boxes_per_image']:
            avg_boxes = np.mean(stats['boxes_per_image'])
            text += f"Average Boxes per Image: {avg_boxes:.2f}\n"
        
        text += f"\n{'='*60}\n"
        text += f"CLASS DISTRIBUTION\n"
        text += f"{'='*60}\n\n"
        
        text += f"{'Class':<20} {'Count':<10} {'Avg Area (px²)':<15} {'Avg Width':<12} {'Avg Height'}\n"
        text += f"{'-'*60}\n"
        
        for cls_id in sorted(stats['class_counts'].keys()):
            cls_name = self.class_info[cls_id]['name']
            count = stats['class_counts'][cls_id]
            
            if count > 0:
                avg_area = np.mean(stats['class_areas'][cls_id])
                avg_width = np.mean(stats['class_widths'][cls_id])
                avg_height = np.mean(stats['class_heights'][cls_id])
                
                text += f"{cls_name:<20} {count:<10} {avg_area:<15.0f} {avg_width:<12.3f} {avg_height:.3f}\n"
            else:
                text += f"{cls_name:<20} {count:<10} {'N/A':<15} {'N/A':<12} N/A\n"
        
        if stats['image_dimensions']:
            text += f"\n{'='*60}\n"
            text += f"IMAGE DIMENSIONS\n"
            text += f"{'='*60}\n\n"
            widths = [w for w, h in stats['image_dimensions']]
            heights = [h for w, h in stats['image_dimensions']]
            text += f"Average Width: {np.mean(widths):.0f}px\n"
            text += f"Average Height: {np.mean(heights):.0f}px\n"
            text += f"Min Width: {min(widths)}px, Max Width: {max(widths)}px\n"
            text += f"Min Height: {min(heights)}px, Max Height: {max(heights)}px\n"
        
        self.stats_text.insert(tk.END, text)
    
    def _show_visualizations(self, stats):
        """Show visualizations in a new window with matplotlib charts"""
        try:
            viz_window = tk.Toplevel(self.root)
            viz_window.title("Dataset Visualizations")
            viz_window.geometry("1200x800")
            
            # Create matplotlib figure with subplots
            fig = Figure(figsize=(12, 8), dpi=100)
            
            # 1. Class distribution bar chart
            ax1 = fig.add_subplot(2, 3, 1)
            classes = []
            counts = []
            colors_for_plot = []
            
            for cls_id in sorted(stats['class_counts'].keys()):
                if stats['class_counts'][cls_id] > 0:
                    classes.append(self.class_info[cls_id]['name'])
                    counts.append(stats['class_counts'][cls_id])
                    # Get color and convert to RGB
                    color = self.class_info[cls_id]['color']
                    if isinstance(color, str) and color.startswith('#'):
                        # Hex color
                        colors_for_plot.append(tuple(int(color[i:i+2], 16)/255 for i in (1, 3, 5)))
                    elif isinstance(color, (tuple, list)):
                        # RGB tuple - normalize if needed
                        if all(c <= 1 for c in color):
                            colors_for_plot.append(color)
                        else:
                            colors_for_plot.append(tuple(c/255 for c in color))
                    else:
                        colors_for_plot.append((0.5, 0.5, 0.5))  # Gray fallback
            
            if classes:
                ax1.bar(classes, counts, color=colors_for_plot)
                ax1.set_title('Class Distribution', fontweight='bold')
                ax1.set_xlabel('Class')
                ax1.set_ylabel('Count')
                ax1.tick_params(axis='x', rotation=45)
                for tick in ax1.get_xticklabels():
                    tick.set_rotation(45)
                    tick.set_ha('right')
            
            # 2. Box size distribution
            ax2 = fig.add_subplot(2, 3, 2)
            all_areas = []
            area_labels = []
            for cls_id in sorted(stats['class_counts'].keys()):
                if stats['class_areas'][cls_id]:
                    all_areas.extend(stats['class_areas'][cls_id])
                    area_labels.extend([self.class_info[cls_id]['name']] * len(stats['class_areas'][cls_id]))
            
            if all_areas:
                ax2.hist(all_areas, bins=30, edgecolor='black', alpha=0.7)
                ax2.set_title('Box Area Distribution', fontweight='bold')
                ax2.set_xlabel('Area (pixels²)')
                ax2.set_ylabel('Frequency')
                ax2.set_yscale('log')
            
            # 3. Boxes per image distribution
            ax3 = fig.add_subplot(2, 3, 3)
            if stats['boxes_per_image']:
                ax3.hist(stats['boxes_per_image'], bins=20, edgecolor='black', alpha=0.7, color='skyblue')
                ax3.set_title('Boxes per Image', fontweight='bold')
                ax3.set_xlabel('Number of Boxes')
                ax3.set_ylabel('Number of Images')
            
            # 4. Spatial distribution (heatmap of box centers)
            ax4 = fig.add_subplot(2, 3, 4)
            all_x = []
            all_y = []
            for cls_id in sorted(stats['class_counts'].keys()):
                all_x.extend(stats['class_positions_x'][cls_id])
                all_y.extend(stats['class_positions_y'][cls_id])
            
            if all_x and all_y:
                # Create 2D histogram (heatmap)
                h, xedges, yedges = np.histogram2d(all_x, all_y, bins=20, range=[[0, 1], [0, 1]])
                extent = [0, 1, 1, 0]  # Invert y-axis to match image coordinates
                im = ax4.imshow(h.T, extent=extent, origin='upper', cmap='hot', aspect='auto', interpolation='bilinear')
                ax4.set_title('Spatial Distribution (Box Centers)', fontweight='bold')
                ax4.set_xlabel('Normalized X')
                ax4.set_ylabel('Normalized Y')
                fig.colorbar(im, ax=ax4, label='Density')
            
            # 5. Average box size per class
            ax5 = fig.add_subplot(2, 3, 5)
            class_names = []
            avg_widths = []
            avg_heights = []
            for cls_id in sorted(stats['class_counts'].keys()):
                if stats['class_widths'][cls_id]:
                    class_names.append(self.class_info[cls_id]['name'])
                    avg_widths.append(np.mean(stats['class_widths'][cls_id]))
                    avg_heights.append(np.mean(stats['class_heights'][cls_id]))
            
            if class_names:
                x = np.arange(len(class_names))
                width_bar = 0.35
                ax5.bar(x - width_bar/2, avg_widths, width_bar, label='Width', alpha=0.8)
                ax5.bar(x + width_bar/2, avg_heights, width_bar, label='Height', alpha=0.8)
                ax5.set_title('Average Box Size per Class', fontweight='bold')
                ax5.set_xlabel('Class')
                ax5.set_ylabel('Normalized Size')
                ax5.set_xticks(x)
                ax5.set_xticklabels(class_names, rotation=45, ha='right')
                ax5.legend()
            
            # 6. Class distribution pie chart
            ax6 = fig.add_subplot(2, 3, 6)
            if classes and counts:
                ax6.pie(counts, labels=classes, autopct='%1.1f%%', colors=colors_for_plot, startangle=90)
                ax6.set_title('Class Proportion', fontweight='bold')
            
            fig.tight_layout()
            
            # Embed in tkinter window
            canvas = FigureCanvasTkAgg(fig, master=viz_window)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # Add export button
            export_frame = ttk.Frame(viz_window)
            export_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=5)
            
            def export_plot():
                file_path = filedialog.asksaveasfilename(
                    defaultextension='.png',
                    filetypes=[('PNG files', '*.png'), ('PDF files', '*.pdf'), ('All files', '*.*')]
                )
                if file_path:
                    fig.savefig(file_path, dpi=300, bbox_inches='tight')
                    messagebox.showinfo("Success", f"Visualization saved to {file_path}")
            
            ttk.Button(export_frame, text="💾 Export Visualization", command=export_plot).pack()
                
        except Exception as e:
            messagebox.showerror("Visualization Error", f"Failed to create visualizations: {str(e)}\n\nPlease check the console for details.")
            import traceback
            traceback.print_exc()

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='YOLO Labeling Tool')
    parser.add_argument('--image-folder', '-i', help='Path to image folder')
    parser.add_argument('--label-folder', '-l', help='Path to label folder')
    parser.add_argument('--model-path', '-m', help='Path to YOLO model (.pt file)')
    parser.add_argument('--yaml-path', '-y', help='Path to YOLO data.yaml file (for class names and paths)')
    args = parser.parse_args()
    
    root = tk.Tk()
    app = YOLOLabelingTool(root, 
                          image_folder=args.image_folder, 
                          label_folder=args.label_folder, 
                          model_path=args.model_path,
                          yaml_path=args.yaml_path)
    root.mainloop()

if __name__ == "__main__":
    main() 
