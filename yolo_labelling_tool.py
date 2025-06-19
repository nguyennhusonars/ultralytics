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

class YOLOLabelingTool:
    def __init__(self, root, image_folder=None, label_folder=None, model_path=None):
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
        
        # Cache for faster filtering
        self.image_dimensions_cache = {}  # {img_path: (width, height)}
        self.labels_cache = {}  # {img_path: [(cls_id, x1, y1, x2, y2), ...]}
        
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
        self.load_class_config()
        
        self.setup_gui()
        self.setup_bindings()
        
        # Initialize with provided paths if available
        if self.image_folder:
            self.load_images_from_folder(self.image_folder)
            
        if model_path:
            self.load_model(model_path)
            
        # Load labels if both image folder and label folder are provided
        if self.current_image_path and self.label_folder:
            self.load_labels()
        
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
        
        self.control_tabs.add(self.setup_tab, text="Setup")
        self.control_tabs.add(self.filters_tab, text="Filters")
        self.control_tabs.add(self.classes_tab, text="Classes")
        self.control_tabs.add(self.display_tab, text="Display")
        
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
            "- A key: Accept all predictions"
        )
        ttk.Label(self.display_tab, text=instructions, justify=tk.LEFT).pack(pady=10)
        
        # Canvas for image display
        self.canvas_frame = ttk.Frame(self.main_frame)
        self.canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.canvas = tk.Canvas(self.canvas_frame, bg='gray', cursor="crosshair")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Setup class buttons
        self.setup_class_buttons()
        
        # Create context menus
        self.create_context_menus()
        
    def create_context_menus(self):
        """Create context menus for right-click actions"""
        # Context menu for prediction boxes
        self.prediction_menu = tk.Menu(self.root, tearoff=0)
        self.prediction_menu.add_command(label="Accept", command=self.accept_selected_prediction)
        self.prediction_menu.add_command(label="Cancel", command=self.clear_selections)
        
        # Context menu for user/label boxes
        self.box_menu = tk.Menu(self.root, tearoff=0)
        self.box_menu.add_command(label="Change Class", command=self.show_class_reassignment_menu)
        self.box_menu.add_command(label="Delete", command=self.delete_selected_box)
        self.box_menu.add_command(label="Cancel", command=self.clear_selections)
        
        # Create class reassignment submenu (will be populated dynamically)
        self.class_reassign_menu = tk.Menu(self.box_menu, tearoff=0)
        
    def setup_bindings(self):
        self.canvas.bind("<ButtonPress-1>", self.start_draw)
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<ButtonRelease-1>", self.end_draw)
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
        self.canvas.bind("<ButtonPress-1>", self.on_canvas_click, add="+")  # Handle clicks to cancel selection
        
    def on_canvas_click(self, event):
        """Handle clicks on canvas to cancel selection"""
        # Clear selections when clicking elsewhere on canvas
        if self.selected_box_index is not None or self.selected_prediction is not None:
            self.clear_selections()
            
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
        
    def select_label_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            self.label_folder = folder
            if self.current_image_path:
                self.load_labels()
            self.update_status()
            
    def select_model(self):
        model_path = filedialog.askopenfilename(filetypes=[("PT files", "*.pt")])
        if model_path:
            self.load_model(model_path)
                
    def load_model(self, model_path):
        """Load YOLO model from the specified path"""
        try:
            self.pretrained_model = YOLO(model_path)
            if self.current_image_path:
                self.load_model_predictions()
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
            
            self.display_image()
            self.load_labels()
            self.load_model_predictions()
            self.update_status()
            self.update_counter()
            
    def display_image(self):
        if self.current_image is None:
            return
            
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(self.current_image.copy(), cv2.COLOR_BGR2RGB)
        
        # Get original dimensions
        height, width = image_rgb.shape[:2]
        
        # Calculate new dimensions
        new_width = int(width * self.zoom_scale)
        new_height = int(height * self.zoom_scale)
        
        # Resize image
        image_resized = cv2.resize(image_rgb, (new_width, new_height))
        
        # Draw model predictions (thin lines) if enabled
        if self.show_model_predictions:
            for i, (cls_id, x1, y1, x2, y2) in enumerate(self.model_boxes):
                # Convert numpy values to native Python types if needed
                if hasattr(cls_id, 'item'):
                    cls_id = int(cls_id.item())
                    
                if not self.class_visibility.get(cls_id, True):
                    continue
                # Convert coordinates to zoomed space
                zx1, zy1 = self.image_to_canvas_coords(x1, y1)
                zx2, zy2 = self.image_to_canvas_coords(x2, y2)
                color = self.class_info[cls_id]["color"]
                
                # Draw box with thinner lines for predictions
                cv2.rectangle(image_resized, (int(zx1), int(zy1)), (int(zx2), int(zy2)), 
                            color[::-1], 1)  # Reverse color for RGB image
                
                # Add small label showing prediction number
                label_text = f"P{i+1}"
                font_scale = 0.5 * self.zoom_scale  # Scale font size with zoom
                thickness = max(1, int(1 * self.zoom_scale))  # Scale thickness with zoom
                cv2.putText(image_resized, label_text, 
                          (int(zx1), int(zy1)-5),
                          cv2.FONT_HERSHEY_SIMPLEX, font_scale, 
                          color[::-1], thickness)
            
        # Draw label file boxes (thick lines)
        if self.show_label_boxes:
            for i, (cls_id, x1, y1, x2, y2) in enumerate(self.label_boxes):
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
                label_text = f"L{i+1}"
                font_scale = 0.5 * self.zoom_scale  # Scale font size with zoom
                thickness = max(1, int(1 * self.zoom_scale))  # Scale thickness with zoom
                cv2.putText(image_resized, label_text, 
                          (int(zx1), int(zy1)-5),
                          cv2.FONT_HERSHEY_SIMPLEX, font_scale, 
                          color[::-1], thickness)
            
        # Draw user boxes (thick lines)
        if self.show_user_boxes:
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
        
        # Re-highlight selected box if any
        self.highlight_selected_box()
        
        # Re-highlight selected prediction if any
        if self.selected_prediction is not None:
            self.highlight_selected_prediction()
        
    def load_labels(self):
        self.label_boxes = []
        if self.label_folder:
            label_path = Path(self.label_folder) / f"{self.current_image_path.stem}.txt"
            if label_path.exists():
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                    img_height, img_width = self.current_image.shape[:2]
                    
                    # Update cache for these labels
                    cached_labels = []
                    
                    for line in lines:
                        cls_id, x_center, y_center, w, h = map(float, line.strip().split())
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
                    
        self.display_image()
        
    def load_model_predictions(self):
        self.model_boxes = []
        if self.pretrained_model and self.current_image_path:
            try:
                results = self.pretrained_model(self.current_image)
                for r in results:
                    boxes = r.boxes
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        cls_id = box.cls[0].cpu().numpy()
                        self.model_boxes.append((cls_id, x1, y1, x2, y2))
            except Exception as e:
                messagebox.showerror("Error", f"Failed to get model predictions: {str(e)}")
        self.display_image()
        
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
        
        self.display_image()
        self.update_status(f"Box size: {int(width)}x{int(height)} pixels")
        
    def save_labels(self, event=None):
        """Save labels to file"""
        if not self.label_folder or not self.current_image_path:
            messagebox.showerror("Error", "No label folder selected or no image loaded")
            return
            
        # Combine label file boxes and user boxes
        all_boxes = self.label_boxes + self.user_boxes
        
        if not all_boxes:
            # If there are no boxes, delete the label file if it exists
            label_path = Path(self.label_folder) / f"{self.current_image_path.stem}.txt"
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
        label_path = Path(self.label_folder) / f"{self.current_image_path.stem}.txt"
        try:
            with open(label_path, 'w') as f:
                f.write('\n'.join(yolo_lines))
            self.update_status("Labels saved")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save labels: {str(e)}")
        
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
        
        # Check user boxes first
        for i, (cls_id, x1, y1, x2, y2) in enumerate(self.user_boxes):
            if x1 <= x <= x2 and y1 <= y <= y2:
                self.selected_box_index = i
                self.selected_box_source = 'user'
                self.highlight_selected_box()
                self.update_status(f"Selected user box {i+1} (Class: {self.class_info[cls_id]['name']})")
                # Show context menu for user box
                self.box_menu.post(event.x_root, event.y_root)
                return
                
        # Then check label boxes
        for i, (cls_id, x1, y1, x2, y2) in enumerate(self.label_boxes):
            if x1 <= x <= x2 and y1 <= y <= y2:
                self.selected_box_index = i
                self.selected_box_source = 'label'
                self.highlight_selected_box()
                self.update_status(f"Selected label box {i+1} (Class: {self.class_info[cls_id]['name']})")
                # Show context menu for label box
                self.box_menu.post(event.x_root, event.y_root)
                return
                
        # Finally check model predictions if visible
        if self.show_model_predictions:
            for i, (cls_id, x1, y1, x2, y2) in enumerate(self.model_boxes):
                if x1 <= x <= x2 and y1 <= y <= y2:
                    self.selected_prediction = i
                    self.highlight_selected_prediction()
                    self.update_status(f"Selected prediction {i+1} (Class: {self.class_info[int(cls_id)]['name']})")
                    # Show context menu for prediction
                    self.prediction_menu.post(event.x_root, event.y_root)
                    return

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
        """Handle zoom with mouse wheel"""
        # Get the current mouse position
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)
        
        # Determine zoom direction
        if event.num == 5 or event.delta < 0:  # Scroll down or negative delta
            self.zoom_scale = max(self.min_zoom, self.zoom_scale / self.zoom_factor)
        elif event.num == 4 or event.delta > 0:  # Scroll up or positive delta
            self.zoom_scale = min(self.max_zoom, self.zoom_scale * self.zoom_factor)
            
        self.display_image()
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

    def apply_filters(self):
        """Apply filters to image paths"""
        if not self.image_paths:
            self.filter_result_var.set("No images loaded")
            return
            
        if not self.label_folder:
            self.filter_result_var.set("Label folder not set")
            return
            
        # Reset progress bar
        self.progress_var.set(0)
        self.filter_result_var.set("Filtering...")
        self.root.update_idletasks()
        
        if not self.filter_by_class_enabled and not self.filter_by_size_enabled:
            # No filters active, show all images
            self.filtered_image_paths = self.image_paths.copy()
            self.progress_var.set(100)
            self.filter_result_var.set(f"Showing all {len(self.filtered_image_paths)} images")
        else:
            # Apply filters
            self.filtered_image_paths = []
            total_with_labels = 0
            total_images = len(self.image_paths)
            
            # Process images in batches of 50 to avoid UI freezes
            batch_size = 50
            for batch_idx in range(0, total_images, batch_size):
                # Process a batch of images
                batch_end = min(batch_idx + batch_size, total_images)
                batch = self.image_paths[batch_idx:batch_end]
                
                for i, img_path in enumerate(batch):
                    # Update progress
                    progress = ((batch_idx + i) / total_images) * 100
                    if (batch_idx + i) % 10 == 0:
                        self.progress_var.set(progress)
                        self.filter_result_var.set(f"Filtering: {batch_idx + i}/{total_images} ({int(progress)}%)")
                        self.root.update_idletasks()
                    
                    img_path_str = str(img_path)
                    label_path = Path(self.label_folder) / f"{img_path.stem}.txt"
                    if not label_path.exists():
                        continue
                        
                    total_with_labels += 1
                    
                    # Check if we have cached label data
                    if img_path_str in self.labels_cache:
                        labels = self.labels_cache[img_path_str]
                        match_found = False
                        
                        for cls_id, _, _, _, _, box_area in labels:
                            # Check class filter
                            class_match = (not self.filter_by_class_enabled) or (cls_id == self.filter_class_id)
                            
                            # Check size filter
                            size_match = (not self.filter_by_size_enabled) or (self.filter_min_size <= box_area <= self.filter_max_size)
                            
                            if class_match and size_match:
                                match_found = True
                                break
                                
                        if match_found:
                            self.filtered_image_paths.append(img_path)
                            
                    else:
                        # No cached data available, process the old way
                        try:
                            with open(label_path, 'r') as f:
                                lines = f.readlines()
                                
                            if not lines:
                                continue
                                
                            # Get image dimensions
                            if img_path_str in self.image_dimensions_cache:
                                img_width, img_height = self.image_dimensions_cache[img_path_str]
                            else:
                                try:
                                    img = cv2.imread(img_path_str, cv2.IMREAD_UNCHANGED)
                                    if img is None:
                                        continue
                                    img_height, img_width = img.shape[:2]
                                    
                                    # Cache the dimensions for future use
                                    self.image_dimensions_cache[img_path_str] = (img_width, img_height)
                                except Exception:
                                    continue
                                    
                            match_found = False
                            
                            for line in lines:
                                parts = line.strip().split()
                                if not parts:
                                    continue
                                
                                try:
                                    cls_id = int(float(parts[0]))
                                    x_center, y_center, w, h = map(float, parts[1:5])
                                    
                                    # Convert YOLO format to pixel coordinates
                                    box_width = w * img_width
                                    box_height = h * img_height
                                    box_area = box_width * box_height
                                    
                                    # Check class filter
                                    class_match = (not self.filter_by_class_enabled) or (cls_id == self.filter_class_id)
                                    
                                    # Check size filter
                                    size_match = (not self.filter_by_size_enabled) or (self.filter_min_size <= box_area <= self.filter_max_size)
                                    
                                    if class_match and size_match:
                                        match_found = True
                                        break
                                except (ValueError, IndexError):
                                    continue
                            
                            if match_found:
                                self.filtered_image_paths.append(img_path)
                        except Exception:
                            continue
                
                # Give UI a chance to update
                self.root.update_idletasks()
        
        # Update display
        self.current_image_index = 0 if self.filtered_image_paths else -1
        
        # Complete progress bar
        self.progress_var.set(100)
        
        if self.current_image_index >= 0:
            self.load_image()
        else:
            # No images match filter
            self.current_image = None
            self.canvas.delete("all")
            self.canvas.create_text(
                self.canvas.winfo_width() // 2, 
                self.canvas.winfo_height() // 2,
                text="No images match the current filters",
                fill="white", font=("Arial", 14)
            )
            
        # Update counter and status
        self.update_counter()
        count = len(self.filtered_image_paths)
        total = len(self.image_paths)
        if self.filter_by_class_enabled or self.filter_by_size_enabled:
            self.filter_result_var.set(f"Found: {count}/{total_with_labels} labeled images")
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
        """Load images for batch view"""
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
        
        # Load images for each cell
        for i, canvas in enumerate(self.batch_canvases):
            img_idx = start_idx + i
            if img_idx < len(self.filtered_image_paths):
                self.load_batch_image(img_idx, canvas)
            else:
                # Clear canvas if no image available
                canvas.delete("all")
                canvas.create_text(canvas.winfo_width()//2, canvas.winfo_height()//2,
                                 text="No Image", fill="white", anchor=tk.CENTER)
                                 
        # Update counter
        self.update_counter()

    def load_batch_image(self, img_idx, canvas):
        """Load and display a single image in batch view"""
        img_path = self.filtered_image_paths[img_idx]
        img = cv2.imread(str(img_path))
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
        img_resized = cv2.resize(img_rgb, (new_width, new_height))
        
        # Load and draw labels if available
        if self.label_folder:
            label_path = Path(self.label_folder) / f"{img_path.stem}.txt"
            if label_path.exists():
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        cls_id, x_center, y_center, w, h = map(float, line.strip().split())
                        # Convert YOLO format to pixel coordinates
                        x1 = int((x_center - w/2) * img_width)
                        y1 = int((y_center - h/2) * img_height)
                        x2 = int((x_center + w/2) * img_width)
                        y2 = int((y_center + h/2) * img_height)
                        
                        # Scale coordinates to fit resized image
                        x1 = int(x1 * scale)
                        y1 = int(y1 * scale)
                        x2 = int(x2 * scale)
                        y2 = int(y2 * scale)
                        
                        # Get class color
                        cls_id = int(cls_id)
                        if cls_id in self.class_info:
                            color = self.class_info[cls_id]["color"]
                            # Draw rectangle
                            cv2.rectangle(img_resized, (x1, y1), (x2, y2), 
                                        color[::-1], 1)  # Reverse color for RGB
                            
                            # Add class name
                            class_name = self.class_info[cls_id]["name"]
                            font_scale = 0.4  # Smaller font for batch view
                            thickness = 1
                            cv2.putText(img_resized, class_name, 
                                      (x1, y1-5),
                                      cv2.FONT_HERSHEY_SIMPLEX, font_scale, 
                                      color[::-1], thickness)
        
        # Convert to PhotoImage
        img_pil = Image.fromarray(img_resized)
        photo = ImageTk.PhotoImage(image=img_pil)
        
        # Store reference to prevent garbage collection
        canvas.photo = photo
        
        # Clear canvas and display image
        canvas.delete("all")
        canvas.create_image(canvas_width//2, canvas_height//2, 
                          image=photo, anchor=tk.CENTER)
        
        # Add image number label
        canvas.create_text(10, 10, text=f"#{img_idx + 1}/{len(self.filtered_image_paths)}", 
                         fill="white", anchor=tk.NW)
                         
        # Add label count if labels exist
        if self.label_folder and label_path.exists():
            label_count = len(lines)
            canvas.create_text(10, 30, text=f"Labels: {label_count}", 
                             fill="white", anchor=tk.NW)
                             
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
            
            # Switch back to batch view
            self.batch_mode = True
            self.editing_batch_image = None
            self.canvas_frame.pack_forget()
            self.batch_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            
            # Setup batch view and reload images
            self.setup_batch_view()
            
            # Ensure images are loaded after setup is complete
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
        
    def load_batch_images(self):
        """Load images for batch view"""
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
        
        # Load images for each cell
        for i, canvas in enumerate(self.batch_canvases):
            img_idx = start_idx + i
            if img_idx < len(self.filtered_image_paths):
                self.load_batch_image(img_idx, canvas)
            else:
                # Clear canvas if no image available
                canvas.delete("all")
                canvas.create_text(canvas.winfo_width()//2, canvas.winfo_height()//2,
                                 text="No Image", fill="white", anchor=tk.CENTER)
                                 
        # Update counter
        self.update_counter()

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
            
            # Switch back to batch view
            self.batch_mode = True
            self.editing_batch_image = None
            self.canvas_frame.pack_forget()
            self.batch_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            
            # Setup batch view and reload images
            self.setup_batch_view()
            
            # Ensure images are loaded after setup is complete
            self.root.update_idletasks()
            self.load_batch_images()
            
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
        if not self.image_paths or not self.label_folder:
            messagebox.showinfo("Info", "No images or label folder not set")
            return
            
        # Reset progress bar
        self.progress_var.set(0)
        self.filter_result_var.set("Building cache...")
        self.root.update_idletasks()
        
        # Clear existing cache
        self.image_dimensions_cache = {}
        self.labels_cache = {}
        
        total_images = len(self.image_paths)
        for i, img_path in enumerate(self.image_paths):
            # Update progress every 10 images to avoid UI freezing
            if i % 10 == 0:
                progress = (i / total_images) * 100
                self.progress_var.set(progress)
                self.filter_result_var.set(f"Building cache: {i}/{total_images} ({int(progress)}%)")
                self.root.update_idletasks()
            
            # Get label data
            label_path = Path(self.label_folder) / f"{img_path.stem}.txt"
            if not label_path.exists():
                continue
                
            try:
                # Get image dimensions without loading full image
                img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
                if img is None:
                    continue
                    
                img_height, img_width = img.shape[:2]
                self.image_dimensions_cache[str(img_path)] = (img_width, img_height)
                
                # Read and parse label file
                labels = []
                with open(label_path, 'r') as f:
                    for line in f.readlines():
                        parts = line.strip().split()
                        if not parts:
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
                            
                self.labels_cache[str(img_path)] = labels
            except Exception:
                continue
                
        # Set progress to 100% when done
        self.progress_var.set(100)
        self.filter_result_var.set(f"Cache built for {len(self.labels_cache)}/{total_images} images")
        
def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='YOLO Labeling Tool')
    parser.add_argument('--image-folder', '-i', help='Path to image folder')
    parser.add_argument('--label-folder', '-l', help='Path to label folder')
    parser.add_argument('--model-path', '-m', help='Path to YOLO model (.pt file)')
    args = parser.parse_args()
    
    root = tk.Tk()
    app = YOLOLabelingTool(root, 
                          image_folder=args.image_folder, 
                          label_folder=args.label_folder, 
                          model_path=args.model_path)
    root.mainloop()

if __name__ == "__main__":
    main() 