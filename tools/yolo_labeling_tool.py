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

        # Edge-drag (resize-by-edge) state
        self._resize_active = False
        self._resize_source = None        # 'user' or 'label'
        self._resize_index = None
        self._resize_handle = None        # 'tl','tr','bl','br','top','bottom','left','right'
        self._last_cursor = 'crosshair'
        self._edge_hover_pixels = 7       # tolerance in canvas pixels

        # Click-to-select state: a press inside a box stays in this "tentative"
        # state until the release (→ select) or significant motion (→ switch to
        # drawing a new box from the press location).
        self._click_select_candidate = None  # (source, index) or None

        # User-editable mapping of digit-key → class id. Persisted in JSON
        # config alongside the class list. Defaults are filled in by
        # _init_default_hotkeys() once class_info is known.
        self.class_hotkeys = {}            # {'1': 0, '2': 3, ...}

        # Crop-review window state (lazy; set when window is opened)
        self._crop_review_window = None
        self._crop_review_entries = []          # [{img_path, line_idx, cls_id, bbox}, ...]
        self._crop_review_target = None         # class id under review
        self._crop_review_page = 0
        self._crop_review_selected = None       # entry index, or None
        self._crop_review_canvases = []         # [(cell_frame, canvas, caption_lbl, entry), ...]
        self._crop_review_dirty_paths = set()   # which image paths got modified
        # Persistent caches so re-renders are near-instant:
        self._crop_review_thumb_cache = {}      # id(entry) -> PhotoImage
        self._crop_review_img_cache = {}        # path-str -> cv2 image (insertion-order LRU)
        self._crop_review_img_cache_max = 32

        # Undo / redo (per-image)
        self.history_undo = []
        self.history_redo = []
        self.max_history = 100

        # Auto-fit on new image load
        self.auto_fit_on_load = True
        self._initial_fit_pending = False

        # Scroll-triggered re-render id (for debounced redraw via after_idle)
        self._scroll_redraw_id = None

        # Path-display variables (populated in setup_gui)
        self.image_path_display_var = None
        self.label_path_display_var = None
        self.model_path_display_var = None
        self.yaml_path_display_var = None

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
            if self.yaml_path_display_var is not None:
                self.yaml_path_display_var.set(self._shorten_path(self.yaml_path_to_load))
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

    def _shorten_path(self, path, max_len=42):
        """Return a path string trimmed for display (keeps tail)."""
        if not path:
            return "(not set)"
        s = str(path)
        if len(s) <= max_len:
            return s
        return "…" + s[-(max_len - 1):]

    def _update_path_displays(self):
        """Refresh all path-display labels from current state."""
        if self.image_path_display_var is not None:
            self.image_path_display_var.set(self._shorten_path(self.image_folder) if self.image_folder else "(not set)")
        if self.label_path_display_var is not None:
            self.label_path_display_var.set(self._shorten_path(self.label_folder) if self.label_folder else "(not set)")

    def _is_entry_focused(self):
        """True when keyboard focus is on a text entry widget."""
        try:
            w = self.root.focus_get()
        except KeyError:
            return False
        if w is None:
            return False
        # Match common text-input widget classes
        return isinstance(w, (tk.Entry, ttk.Entry, ttk.Combobox, tk.Text, tk.Spinbox))

    # --- Fit-to-window / zoom helpers ---
    def fit_to_window(self, event=None):
        """Scale the current image so it fits inside the canvas viewport."""
        if self.current_image is None or self.batch_mode:
            return
        h, w = self.current_image.shape[:2]
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw < 10 or ch < 10:
            # Canvas isn't drawn yet — try again shortly
            self.root.after(80, self.fit_to_window)
            return
        scale = min(cw / w, ch / h)
        self.zoom_scale = max(self.min_zoom, min(scale, self.max_zoom))
        self.display_image()
        self.canvas.xview_moveto(0)
        self.canvas.yview_moveto(0)
        self.update_status(f"Zoom: {self.zoom_scale:.2f}x (fit)")

    def _zoom_at_viewport_center(self, zoom_in=True):
        """Zoom in or out keeping the viewport center stable. Used by +/- keys."""
        if self.current_image is None or self.batch_mode:
            return
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        # Image-space anchor at viewport center
        cx_canvas = self.canvas.canvasx(cw // 2)
        cy_canvas = self.canvas.canvasy(ch // 2)
        anchor_img_x = cx_canvas / self.zoom_scale
        anchor_img_y = cy_canvas / self.zoom_scale

        if zoom_in:
            self.zoom_scale = min(self.max_zoom, self.zoom_scale * self.zoom_factor)
        else:
            self.zoom_scale = max(self.min_zoom, self.zoom_scale / self.zoom_factor)

        new_w = max(1, int(self.current_image.shape[1] * self.zoom_scale))
        new_h = max(1, int(self.current_image.shape[0] * self.zoom_scale))
        self.canvas.configure(scrollregion=(0, 0, new_w, new_h))

        new_cx = anchor_img_x * self.zoom_scale
        new_cy = anchor_img_y * self.zoom_scale
        if new_w > 0:
            self.canvas.xview_moveto(max(0.0, min(1.0, (new_cx - cw // 2) / new_w)))
        if new_h > 0:
            self.canvas.yview_moveto(max(0.0, min(1.0, (new_cy - ch // 2) / new_h)))

        self.display_image()
        self.update_status(f"Zoom: {self.zoom_scale:.2f}x")

    # --- Scroll-triggered re-render (needed because display_image only renders
    # the visible crop) ---
    def _on_canvas_xview(self, *args):
        self.canvas.xview(*args)
        self._schedule_scroll_redraw()

    def _on_canvas_yview(self, *args):
        self.canvas.yview(*args)
        self._schedule_scroll_redraw()

    def _schedule_scroll_redraw(self):
        """Coalesce multiple scroll events into a single redraw."""
        if self._scroll_redraw_id is not None:
            try:
                self.root.after_cancel(self._scroll_redraw_id)
            except Exception:
                pass
        # after_idle runs as soon as no more events are pending — gives smooth
        # rendering at the end of a scrollbar drag instead of one redraw per tick.
        self._scroll_redraw_id = self.root.after_idle(self._do_scroll_redraw)

    def _do_scroll_redraw(self):
        self._scroll_redraw_id = None
        if self.current_image is None or self.batch_mode:
            return
        self.display_image()

    # --- Undo / redo helpers ---
    def _snapshot_boxes(self):
        """Return a deep-ish snapshot of user_boxes and label_boxes for history."""
        return (
            [tuple(b) for b in self.user_boxes],
            [tuple(b) for b in self.label_boxes],
        )

    def _push_history(self):
        """Save current box state to undo stack and clear redo stack."""
        self.history_undo.append(self._snapshot_boxes())
        if len(self.history_undo) > self.max_history:
            self.history_undo.pop(0)
        self.history_redo.clear()

    def _restore_snapshot(self, snap):
        user_boxes, label_boxes = snap
        self.user_boxes = list(user_boxes)
        self.label_boxes = list(label_boxes)
        self.clear_selections()
        if self.highlight_overlaps:
            self.detect_current_overlaps()
        self.save_labels()
        self.display_image()

    def undo(self, event=None):
        if self._is_entry_focused():
            return
        if not self.history_undo:
            self.update_status("Nothing to undo")
            return
        self.history_redo.append(self._snapshot_boxes())
        self._restore_snapshot(self.history_undo.pop())
        self.update_status(f"Undo ({len(self.history_undo)} left)")

    def redo(self, event=None):
        if self._is_entry_focused():
            return
        if not self.history_redo:
            self.update_status("Nothing to redo")
            return
        self.history_undo.append(self._snapshot_boxes())
        self._restore_snapshot(self.history_redo.pop())
        self.update_status(f"Redo ({len(self.history_redo)} left)")

    # --- Edge-drag (hover-to-resize) helpers ---
    _HANDLE_CURSORS = {
        'tl': 'top_left_corner', 'br': 'bottom_right_corner',
        'tr': 'top_right_corner', 'bl': 'bottom_left_corner',
        'top': 'sb_v_double_arrow', 'bottom': 'sb_v_double_arrow',
        'left': 'sb_h_double_arrow', 'right': 'sb_h_double_arrow',
    }

    def _detect_edge_hover(self, img_x, img_y):
        """Return (source, index, handle) if (img_x, img_y) is near a box edge/corner, else None.
        Searches visible user + label boxes, prefers the box with the smallest area."""
        if self.zoom_scale <= 0:
            return None
        # Convert canvas-space tolerance to image-space pixels
        tol = max(2.0, self._edge_hover_pixels / self.zoom_scale)
        candidates = []
        sources = []
        if self.show_user_boxes:
            sources.append(('user', self.user_boxes))
        if self.show_label_boxes:
            sources.append(('label', self.label_boxes))
        for source, boxes in sources:
            for i, box in enumerate(boxes):
                if len(box) < 5:
                    continue
                cls_id, x1, y1, x2, y2 = box[:5]
                cls_int = int(cls_id.item()) if hasattr(cls_id, 'item') else int(cls_id)
                if not self.class_visibility.get(cls_int, True):
                    continue
                # Bounding probe with tolerance — skip far-away boxes fast
                if img_x < min(x1, x2) - tol or img_x > max(x1, x2) + tol:
                    continue
                if img_y < min(y1, y2) - tol or img_y > max(y1, y2) + tol:
                    continue
                near_left = abs(img_x - x1) <= tol and min(y1, y2) - tol <= img_y <= max(y1, y2) + tol
                near_right = abs(img_x - x2) <= tol and min(y1, y2) - tol <= img_y <= max(y1, y2) + tol
                near_top = abs(img_y - y1) <= tol and min(x1, x2) - tol <= img_x <= max(x1, x2) + tol
                near_bottom = abs(img_y - y2) <= tol and min(x1, x2) - tol <= img_x <= max(x1, x2) + tol
                handle = None
                if near_top and near_left:
                    handle = 'tl'
                elif near_top and near_right:
                    handle = 'tr'
                elif near_bottom and near_left:
                    handle = 'bl'
                elif near_bottom and near_right:
                    handle = 'br'
                elif near_top:
                    handle = 'top'
                elif near_bottom:
                    handle = 'bottom'
                elif near_left:
                    handle = 'left'
                elif near_right:
                    handle = 'right'
                if handle is not None:
                    area = max(1, abs((x2 - x1) * (y2 - y1)))
                    candidates.append((source, i, handle, area))
        if not candidates:
            return None
        candidates.sort(key=lambda c: c[3])
        s, idx, h, _ = candidates[0]
        return s, idx, h

    def _detect_box_at(self, img_x, img_y):
        """Return (source, index) for the smallest visible box containing
        (img_x, img_y), or None. Used by left-click selection."""
        candidates = []
        sources = []
        if self.show_user_boxes:
            sources.append(('user', self.user_boxes))
        if self.show_label_boxes:
            sources.append(('label', self.label_boxes))
        for source, boxes in sources:
            for i, box in enumerate(boxes):
                if len(box) < 5:
                    continue
                cls_id, x1, y1, x2, y2 = box[:5]
                cls_int = int(cls_id.item()) if hasattr(cls_id, 'item') else int(cls_id)
                if not self.class_visibility.get(cls_int, True):
                    continue
                if min(x1, x2) <= img_x <= max(x1, x2) and min(y1, y2) <= img_y <= max(y1, y2):
                    area = max(1, abs((x2 - x1) * (y2 - y1)))
                    candidates.append((source, i, area))
        if not candidates:
            return None
        candidates.sort(key=lambda c: c[2])
        return (candidates[0][0], candidates[0][1])

    def _set_canvas_cursor(self, cursor):
        if cursor != self._last_cursor:
            try:
                self.canvas.configure(cursor=cursor)
            except tk.TclError:
                # Unknown cursor name on this platform — fall back silently
                self.canvas.configure(cursor='crosshair')
                cursor = 'crosshair'
            self._last_cursor = cursor

    def _start_edge_resize(self, source, index, handle):
        """Begin resizing the given box by the given handle. Pushes history."""
        boxes = self.user_boxes if source == 'user' else self.label_boxes
        if not (0 <= index < len(boxes)):
            return
        self._push_history()
        cls_id, x1, y1, x2, y2 = boxes[index][:5]
        # Normalize so x1 <= x2 and y1 <= y2
        nx1, nx2 = min(x1, x2), max(x1, x2)
        ny1, ny2 = min(y1, y2), max(y1, y2)
        boxes[index] = (cls_id, nx1, ny1, nx2, ny2)
        self._resize_active = True
        self._resize_source = source
        self._resize_index = index
        self._resize_handle = handle
        # Create the live preview rectangle
        color = self.class_info[int(cls_id.item()) if hasattr(cls_id, 'item') else int(cls_id)]["color"]
        rgb_color = self.bgr_to_rgb_hex(color)
        zx1, zy1 = self.image_to_canvas_coords(nx1, ny1)
        zx2, zy2 = self.image_to_canvas_coords(nx2, ny2)
        if self.current_rect:
            self.canvas.delete(self.current_rect)
        self.current_rect = self.canvas.create_rectangle(
            zx1, zy1, zx2, zy2, outline=rgb_color, width=2, dash=(6, 4)
        )

    def _update_edge_resize(self, img_x, img_y):
        """Update the box being resized based on the current pointer position."""
        if not self._resize_active or self._resize_index is None:
            return
        boxes = self.user_boxes if self._resize_source == 'user' else self.label_boxes
        if not (0 <= self._resize_index < len(boxes)):
            self._resize_active = False
            return
        cls_id, x1, y1, x2, y2 = boxes[self._resize_index][:5]
        img_height, img_width = self.current_image.shape[:2]
        img_x = max(0, min(img_x, img_width))
        img_y = max(0, min(img_y, img_height))
        h = self._resize_handle
        if h in ('tl', 'left', 'bl'):
            x1 = img_x
        if h in ('tr', 'right', 'br'):
            x2 = img_x
        if h in ('tl', 'top', 'tr'):
            y1 = img_y
        if h in ('bl', 'bottom', 'br'):
            y2 = img_y
        boxes[self._resize_index] = (cls_id, x1, y1, x2, y2)
        # Update preview rectangle
        nx1, nx2 = min(x1, x2), max(x1, x2)
        ny1, ny2 = min(y1, y2), max(y1, y2)
        if self.current_rect:
            zx1, zy1 = self.image_to_canvas_coords(nx1, ny1)
            zx2, zy2 = self.image_to_canvas_coords(nx2, ny2)
            self.canvas.coords(self.current_rect, zx1, zy1, zx2, zy2)
        self.update_status(f"Resize: {int(nx2-nx1)}x{int(ny2-ny1)} px")

    def _finish_edge_resize(self):
        """Finalize the in-progress edge resize."""
        if not self._resize_active or self._resize_index is None:
            self._resize_active = False
            return
        boxes = self.user_boxes if self._resize_source == 'user' else self.label_boxes
        if 0 <= self._resize_index < len(boxes):
            cls_id, x1, y1, x2, y2 = boxes[self._resize_index][:5]
            # Normalize ordering
            nx1, nx2 = min(x1, x2), max(x1, x2)
            ny1, ny2 = min(y1, y2), max(y1, y2)
            img_height, img_width = self.current_image.shape[:2]
            nx1 = max(0, min(nx1, img_width))
            nx2 = max(0, min(nx2, img_width))
            ny1 = max(0, min(ny1, img_height))
            ny2 = max(0, min(ny2, img_height))
            # Enforce 10px minimum — revert if too tiny
            if (nx2 - nx1) < 10 or (ny2 - ny1) < 10:
                if self.history_undo:
                    snap = self.history_undo.pop()
                    self.user_boxes, self.label_boxes = list(snap[0]), list(snap[1])
                self.update_status("Resize too small — reverted")
            else:
                boxes[self._resize_index] = (cls_id, nx1, ny1, nx2, ny2)
        if self.current_rect:
            self.canvas.delete(self.current_rect)
            self.current_rect = None
        self._resize_active = False
        self._resize_source = None
        self._resize_index = None
        self._resize_handle = None
        if self.highlight_overlaps:
            self.detect_current_overlaps()
        self.save_labels()
        self.display_image()
        
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

        self.image_path_display_var = tk.StringVar(value="(not set)")
        self.label_path_display_var = tk.StringVar(value="(not set)")
        self.model_path_display_var = tk.StringVar(value="(not set)")
        self.yaml_path_display_var = tk.StringVar(value="(not set)")

        def _path_row(parent, label_text, browse_cmd, var):
            row = ttk.Frame(parent)
            row.pack(fill=tk.X, padx=5, pady=2)
            top = ttk.Frame(row)
            top.pack(fill=tk.X)
            ttk.Label(top, text=label_text).pack(side=tk.LEFT)
            ttk.Button(top, text="Browse", width=8, command=browse_cmd).pack(side=tk.RIGHT)
            ttk.Label(row, textvariable=var, foreground="#555",
                      wraplength=240, justify=tk.LEFT, font=('TkDefaultFont', 8)
                      ).pack(fill=tk.X, padx=(8, 0))

        _path_row(path_frame, "Images:", self.select_image_folder, self.image_path_display_var)
        _path_row(path_frame, "Labels:", self.select_label_folder, self.label_path_display_var)
        _path_row(path_frame, "Model:",  self.select_model,         self.model_path_display_var)
        _path_row(path_frame, "YAML:",   self.select_yaml,          self.yaml_path_display_var)

        # Update displays for paths supplied via CLI / constructor (set before setup_gui ran)
        if self.image_folder:
            self.image_path_display_var.set(self._shorten_path(self.image_folder))
        if self.label_folder:
            self.label_path_display_var.set(self._shorten_path(self.label_folder))
        
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

        # Open separate Crop Review window for this class across the filtered set
        ttk.Button(filter_class_frame, text="🔍 Review crops in separate window…",
                   command=self.open_class_crop_review
                   ).pack(fill=tk.X, padx=5, pady=(4, 6))
        
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
                  command=self.add_new_class).pack(pady=(5, 2), fill=tk.X, padx=10)

        # Bulk-reassign: rewrite every label of class A as class B (across current
        # image / filtered set / whole dataset).
        ttk.Button(self.classes_tab, text="🔀 Reassign Class…",
                  command=self.show_reassign_class_dialog).pack(pady=2, fill=tk.X, padx=10)

        # Class groups: edit equivalence sets for "Find Bad Annotations" so
        # that e.g. 'sedan'/'suv'/'hatchback' all count as 'car'.
        ttk.Button(self.classes_tab, text="🧩 Edit Class Groups…",
                  command=self.edit_class_groups).pack(pady=2, fill=tk.X, padx=10)

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
        
        # Auto-fit toggle
        self.auto_fit_var = tk.BooleanVar(value=self.auto_fit_on_load)
        ttk.Checkbutton(visibility_frame, text="Auto-fit on image load",
                       variable=self.auto_fit_var,
                       command=self._toggle_auto_fit).pack(anchor=tk.W, padx=5, pady=2)

        # Help button + short hint
        ttk.Button(self.display_tab, text="⌨ Show All Shortcuts (?)",
                  command=self.show_shortcuts_help).pack(pady=(10, 5), fill=tk.X, padx=5)

        instructions = (
            "Quick reference:\n"
            "  Draw box: drag on empty area\n"
            "  Select box: left-click inside it\n"
            "  Change class of selected box: keys 1-9, 0\n"
            "  (Click [n] next to a class to rebind its hotkey)\n"
            "  Resize box: drag any edge or corner\n"
            "  Right-click box: open menu\n"
            "  Switch draw class (no selection): keys 1-9, 0\n"
            "  Undo / Redo: Ctrl+Z / Ctrl+Y\n"
            "  Zoom: mouse wheel or +/-\n"
            "  Fit to window: F\n"
            "  Navigate: ← →\n"
            "  Accept all predictions: A\n"
            "  Toggle layers: Shift+P / L / U\n"
            "  Delete image: D     |  Help: ?"
        )
        ttk.Label(self.display_tab, text=instructions, justify=tk.LEFT,
                  font=('TkDefaultFont', 9)).pack(pady=5, padx=5, anchor=tk.W)

        # Analyze tab - Dataset insights and visualizations
        ttk.Label(self.analyze_tab, text="Dataset Analysis", font=('Arial', 12, 'bold')).pack(pady=(5, 10))
        
        # Analysis buttons
        analyze_btn_frame = ttk.Frame(self.analyze_tab)
        analyze_btn_frame.pack(pady=5, fill=tk.X, padx=5)
        
        ttk.Button(analyze_btn_frame, text="📊 Analyze Dataset",
                  command=self.analyze_dataset).pack(fill=tk.X, pady=2)
        ttk.Button(analyze_btn_frame, text="🔄 Refresh Analysis",
                  command=self.refresh_analysis).pack(fill=tk.X, pady=2)
        # Run the loaded model against every GT box and surface mismatches /
        # missing predictions for hand-review (uses class_groups if set).
        ttk.Button(analyze_btn_frame, text="🔍 Find Bad Annotations…",
                  command=self.show_find_bad_dialog).pack(fill=tk.X, pady=2)
        
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
        
        # Configure scrollbars — wrap so a scroll triggers a re-render
        # (display_image only renders the visible crop, so the viewport must
        # be re-rendered when scrolled to reveal a different region).
        self.canvas_vscroll.config(command=self._on_canvas_yview)
        self.canvas_hscroll.config(command=self._on_canvas_xview)
        
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

    def _toggle_auto_fit(self):
        self.auto_fit_on_load = self.auto_fit_var.get()

    def show_shortcuts_help(self, event=None):
        """Show a popup window listing every shortcut."""
        win = tk.Toplevel(self.root)
        win.title("Keyboard Shortcuts")
        win.geometry("520x560")
        win.transient(self.root)

        ttk.Label(win, text="Keyboard & Mouse Shortcuts",
                  font=('Arial', 13, 'bold')).pack(pady=(12, 6))

        rows = [
            ("Mouse", ""),
            ("Click inside box",          "Select that box (then 1-9/0 changes its class)"),
            ("Drag on empty area",        "Draw new box (current class)"),
            ("Drag a box edge / corner",  "Resize that side/corner"),
            ("Double-click inside box",   "Edit user box from corner"),
            ("Right-click",               "Select box → context menu"),
            ("Wheel up / down",           "Zoom in / out at cursor"),
            ("", ""),
            ("Navigation", ""),
            ("← / →",                     "Previous / Next image"),
            ("D",                         "Delete current image (confirm)"),
            ("", ""),
            ("Labels", ""),
            ("1-9, 0",                    "Reassign selected box (mapping is editable in Classes tab)"),
            ("",                          "or set current draw class when nothing is selected"),
            ("S",                         "Save labels"),
            ("A",                         "Accept all model predictions"),
            ("Enter",                     "Accept selected prediction"),
            ("Delete",                    "Delete selected box"),
            ("Ctrl+V",                    "Paste labels from clipboard"),
            ("Ctrl+Z / Ctrl+Y",           "Undo / Redo box changes"),
            ("Esc",                       "Cancel selection"),
            ("", ""),
            ("View", ""),
            ("F",                         "Fit image to window"),
            ("+ / -",                     "Zoom in / out (viewport center)"),
            ("Shift+P / L / U",           "Toggle Predictions / Labels / User layers"),
            ("? or H",                    "Show this help"),
        ]

        body = ttk.Frame(win)
        body.pack(fill=tk.BOTH, expand=True, padx=12, pady=6)

        for left, right in rows:
            row = ttk.Frame(body)
            row.pack(fill=tk.X, pady=1)
            if not left and not right:
                ttk.Separator(body, orient='horizontal').pack(fill=tk.X, pady=4)
                continue
            if not right:
                ttk.Label(row, text=left, font=('TkDefaultFont', 10, 'bold'),
                          foreground='#0066cc').pack(anchor=tk.W)
            else:
                ttk.Label(row, text=left, width=22, anchor=tk.W,
                          font=('Courier', 10)).pack(side=tk.LEFT)
                ttk.Label(row, text=right, anchor=tk.W).pack(side=tk.LEFT)

        ttk.Button(win, text="Close", command=win.destroy).pack(pady=10)
        win.bind("<Escape>", lambda e: win.destroy())

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
        self.canvas.bind("<ButtonPress-3>", self.select_box_or_prediction)
        self.canvas.bind("<Double-Button-1>", self.start_edit_box)
        self.canvas.bind("<MouseWheel>", self.handle_zoom)
        self.canvas.bind("<Button-4>", self.handle_zoom)
        self.canvas.bind("<Button-5>", self.handle_zoom)
        self.canvas.bind("<ButtonPress-1>", self.on_canvas_click, add="+")

        # Helper that suppresses key handlers while the user is typing in an entry
        def kb(handler):
            def wrapped(event):
                if self._is_entry_focused():
                    return
                return handler(event)
            return wrapped

        self.root.bind("<s>",            kb(lambda e: self.save_labels()))
        self.root.bind("<Left>",         kb(lambda e: self.prev_image()))
        self.root.bind("<Right>",        kb(lambda e: self.next_image()))
        self.root.bind("<Delete>",       kb(self.delete_selected_box))
        self.root.bind("<Return>",       kb(lambda e: self.accept_selected_prediction()))
        self.root.bind("<a>",            kb(lambda e: self.accept_all_predictions()))
        self.root.bind("<Control-v>",    kb(lambda e: self.paste_labels()))

        # Undo / redo
        self.root.bind("<Control-z>",        self.undo)
        self.root.bind("<Control-Z>",        self.undo)
        self.root.bind("<Control-y>",        self.redo)
        self.root.bind("<Control-Y>",        self.redo)
        self.root.bind("<Control-Shift-z>",  self.redo)
        self.root.bind("<Control-Shift-Z>",  self.redo)

        # View shortcuts
        self.root.bind("<f>",            kb(lambda e: self.fit_to_window()))
        self.root.bind("<plus>",         kb(lambda e: self._zoom_at_viewport_center(True)))
        self.root.bind("<equal>",        kb(lambda e: self._zoom_at_viewport_center(True)))
        self.root.bind("<KP_Add>",       kb(lambda e: self._zoom_at_viewport_center(True)))
        self.root.bind("<minus>",        kb(lambda e: self._zoom_at_viewport_center(False)))
        self.root.bind("<KP_Subtract>",  kb(lambda e: self._zoom_at_viewport_center(False)))

        # Layer toggles (Shift+P/L/U) — synchronises both the BooleanVar and visibility state
        self.root.bind("<P>", kb(lambda e: self._toggle_layer('predictions')))
        self.root.bind("<L>", kb(lambda e: self._toggle_layer('labels')))
        self.root.bind("<U>", kb(lambda e: self._toggle_layer('user')))

        # Selection / image actions
        self.root.bind("<Escape>",       kb(lambda e: self.clear_selections()))
        self.root.bind("<d>",            kb(lambda e: self.delete_current_image()))

        # Help overlay
        self.root.bind("<question>",     kb(lambda e: self.show_shortcuts_help()))
        self.root.bind("<h>",            kb(lambda e: self.show_shortcuts_help()))

        # Class selection via number keys — both top-row digits and the
        # num-pad equivalents (KP_0..KP_9) when NumLock is on.
        for digit in '0123456789':
            self.root.bind(f"<Key-{digit}>",
                           kb(lambda e, d=digit: self._kb_select_class(d)))
            self.root.bind(f"<KP_{digit}>",
                           kb(lambda e, d=digit: self._kb_select_class(d)))

    def _toggle_layer(self, which):
        """Flip a Boolean layer toggle (predictions/labels/user) and refresh."""
        if which == 'predictions':
            self.show_predictions_var.set(not self.show_predictions_var.get())
            self.toggle_predictions()
            state = "ON" if self.show_model_predictions else "OFF"
            self.update_status(f"Predictions: {state}")
        elif which == 'labels':
            self.show_labels_var.set(not self.show_labels_var.get())
            self.toggle_label_boxes()
            state = "ON" if self.show_label_boxes else "OFF"
            self.update_status(f"Label boxes: {state}")
        elif which == 'user':
            self.show_drawings_var.set(not self.show_drawings_var.get())
            self.toggle_user_boxes()
            state = "ON" if self.show_user_boxes else "OFF"
            self.update_status(f"User boxes: {state}")

    def _kb_select_class(self, digit):
        """Number-key handler.

        The digit→class mapping is user-editable via the Classes tab (each
        class has a clickable hotkey button). Defaults follow the convention
        '1'..'9' → classes 0..8 and '0' → class 9, but can be remapped.

        - If a user/label box is currently selected → reassign its class to
          the mapped target. Selection persists so a follow-up keystroke can
          reassign again.
        - Otherwise → set the current draw class.
        """
        target = self.class_hotkeys.get(digit)
        if target is None or target not in self.class_info:
            self.update_status(f"No class bound to key '{digit}' (set one in the Classes tab)")
            return

        if self.selected_box_index is not None and self.selected_box_source in ('user', 'label'):
            self.reassign_box_class(target, keep_selection=True)
            self.select_class(target)
            self.update_status(f"Box class → {target}: {self.class_info[target]['name']}")
        else:
            self.select_class(target)
            self.update_status(f"Class → {target}: {self.class_info[target]['name']}")
        
    def on_canvas_click(self, event):
        """Handle clicks on canvas to cancel selection"""
        # If a click-select on a box is pending, let end_draw handle the
        # selection swap — clearing here would only cause a brief flicker.
        if self._click_select_candidate is not None:
            return
        # Clear selections when clicking elsewhere on canvas
        if self.selected_box_index is not None or self.selected_prediction is not None:
            self.clear_selections()
    
    def draw_crosshair(self, event):
        """Draw crosshair cursor lines following the mouse and update cursor on box edges."""
        # Only show crosshair when an image is loaded and not in batch mode
        if not hasattr(self, 'current_image') or self.current_image is None or self.batch_mode:
            return

        # Get canvas coordinates (accounts for scrolling)
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)

        # Update cursor when hovering near a box edge (only when idle — not drawing/editing/resizing)
        if not self.drawing and not self.editing_box and not self._resize_active:
            img_x, img_y = self.canvas_to_image_coords(canvas_x, canvas_y)
            hover = self._detect_edge_hover(img_x, img_y)
            if hover is not None:
                self._set_canvas_cursor(self._HANDLE_CURSORS[hover[2]])
            else:
                self._set_canvas_cursor('crosshair')

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
        self.crosshair_h = self.canvas.create_line(
            0, canvas_y, canvas_width, canvas_y,
            fill=crosshair_color, width=1, dash=(4, 4), tags='crosshair'
        )
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
        if self.image_path_display_var is not None:
            self.image_path_display_var.set(self._shorten_path(folder))
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
            if self.label_path_display_var is not None:
                self.label_path_display_var.set(self._shorten_path(folder))
            if self.current_image_path:
                self.load_labels()
                if self.highlight_overlaps:
                    self.detect_current_overlaps()
                self.display_image()
            self.update_status()
            
    def select_model(self):
        model_path = filedialog.askopenfilename(filetypes=[("PT files", "*.pt")])
        if model_path:
            if self.model_path_display_var is not None:
                self.model_path_display_var.set(self._shorten_path(model_path))
            self.load_model(model_path)

    def select_yaml(self):
        """Select and load YAML configuration file"""
        yaml_path = filedialog.askopenfilename(filetypes=[("YAML files", "*.yaml *.yml")])
        if yaml_path:
            if self.yaml_path_display_var is not None:
                self.yaml_path_display_var.set(self._shorten_path(yaml_path))
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
            if self.model_path_display_var is not None:
                self.model_path_display_var.set(self._shorten_path(model_path))
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
                
                # Re-validate hotkey map for the updated class list
                self._cleanup_hotkeys()
                self._init_default_hotkeys()
                self.save_class_config()

                # Refresh UI components after updating class_info
                if hasattr(self, 'setup_class_buttons'):
                    self.setup_class_buttons()
                if hasattr(self, 'update_class_filter_dropdown'):
                    self.update_class_filter_dropdown()
                # Sync any open crop-review window
                self._crop_review_refresh_class_info()

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

            # Reset per-image history — undo never crosses image boundaries
            self.history_undo.clear()
            self.history_redo.clear()

            # Load labels and predictions first
            self.load_labels()
            self.load_model_predictions()

            # Detect overlaps if highlight mode is on (must be after load_labels)
            if self.highlight_overlaps:
                self.detect_current_overlaps()

            # Auto-fit to viewport so we always see the full image first
            if self.auto_fit_on_load and self.current_image is not None and not self.batch_mode:
                self.fit_to_window()
            else:
                self.display_image()

            self.update_status()
            self.update_counter()
            
    def display_image(self):
        """Render the image into the canvas.

        Performance: instead of resizing the whole image to the zoomed size and
        creating one giant PhotoImage every frame, we crop the visible viewport
        (plus a padding ring) out of the original image and only resize that
        slice. The rendering cost becomes bounded by canvas pixels, not by
        zoom², which makes high-zoom interactions fast.
        """
        if self.current_image is None:
            return

        self.canvas.delete("all")

        img_h, img_w = self.current_image.shape[:2]
        zoom = self.zoom_scale

        # Full size of the zoomed scrollregion (capped at 50M pixels)
        full_w = int(img_w * zoom)
        full_h = int(img_h * zoom)
        max_pixels = 50_000_000
        if full_w * full_h > max_pixels:
            scale_factor = (max_pixels / (full_w * full_h)) ** 0.5
            full_w = max(1, int(full_w * scale_factor))
            full_h = max(1, int(full_h * scale_factor))
            effective_zoom = full_w / img_w
        else:
            effective_zoom = zoom

        cw = max(1, self.canvas.winfo_width())
        ch = max(1, self.canvas.winfo_height())
        view_x = self.canvas.canvasx(0)
        view_y = self.canvas.canvasy(0)

        # When the zoomed image is significantly larger than the viewport,
        # crop-then-resize only the visible region (+ padding ring).
        use_crop = full_w > cw * 1.3 or full_h > ch * 1.3

        if use_crop:
            pad_canvas = 120  # canvas-pixel padding so small scrolls don't expose gray
            ix1 = max(0, int(view_x / effective_zoom - pad_canvas / effective_zoom))
            iy1 = max(0, int(view_y / effective_zoom - pad_canvas / effective_zoom))
            ix2 = min(img_w, int((view_x + cw) / effective_zoom + pad_canvas / effective_zoom) + 1)
            iy2 = min(img_h, int((view_y + ch) / effective_zoom + pad_canvas / effective_zoom) + 1)
            if ix2 <= ix1 or iy2 <= iy1:
                ix1, iy1 = 0, 0
                ix2, iy2 = max(1, img_w), max(1, img_h)
            crop = self.current_image[iy1:iy2, ix1:ix2]
            buf_w = max(1, int(round((ix2 - ix1) * effective_zoom)))
            buf_h = max(1, int(round((iy2 - iy1) * effective_zoom)))
            interp = cv2.INTER_NEAREST if effective_zoom > 2.0 else cv2.INTER_LINEAR
            image_resized = cv2.cvtColor(
                cv2.resize(crop, (buf_w, buf_h), interpolation=interp),
                cv2.COLOR_BGR2RGB
            )
            crop_canvas_x = ix1 * effective_zoom
            crop_canvas_y = iy1 * effective_zoom
        else:
            # Image fits in viewport — render once at full size
            if abs(effective_zoom - 1.0) > 1e-6:
                interp = cv2.INTER_NEAREST if effective_zoom > 2.0 else cv2.INTER_LINEAR
                image_resized = cv2.cvtColor(
                    cv2.resize(self.current_image, (full_w, full_h), interpolation=interp),
                    cv2.COLOR_BGR2RGB
                )
            else:
                image_resized = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)
            ix1 = iy1 = 0
            ix2, iy2 = img_w, img_h
            crop_canvas_x = 0
            crop_canvas_y = 0

        # Convert image-space coords → buffer-space pixel coords
        def to_buf(x, y):
            return int(round((x - ix1) * effective_zoom)), int(round((y - iy1) * effective_zoom))

        # Skip boxes entirely outside the rendered crop
        def outside(x1, y1, x2, y2):
            return x2 < ix1 or x1 > ix2 or y2 < iy1 or y1 > iy2

        should_draw_boxes = effective_zoom >= 0.1
        # Cap font/line scaling so it doesn't explode at high zoom
        text_zoom = min(effective_zoom, 1.5)

        # Draw model predictions (thin lines)
        if should_draw_boxes and self.show_model_predictions:
            for i, (cls_id, x1, y1, x2, y2) in enumerate(self.model_boxes):
                if hasattr(cls_id, 'item'):
                    cls_id = int(cls_id.item())
                else:
                    cls_id = int(cls_id)
                if cls_id not in self.class_info or not self.class_visibility.get(cls_id, True):
                    continue
                if outside(x1, y1, x2, y2):
                    continue
                zx1, zy1 = to_buf(x1, y1)
                zx2, zy2 = to_buf(x2, y2)
                color = self.class_info[cls_id]["color"]
                cv2.rectangle(image_resized, (zx1, zy1), (zx2, zy2), color[::-1], 1)
                cv2.putText(image_resized, self.class_info[cls_id]["name"],
                            (zx1, zy1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5 * text_zoom,
                            color[::-1], max(1, int(text_zoom)))

        # Draw label file boxes (thick lines)
        if should_draw_boxes and self.show_label_boxes:
            for i, (cls_id, x1, y1, x2, y2) in enumerate(self.label_boxes):
                if hasattr(cls_id, 'item'):
                    cls_id = int(cls_id.item())
                else:
                    cls_id = int(cls_id)
                if cls_id not in self.class_info or not self.class_visibility.get(cls_id, True):
                    continue
                if outside(x1, y1, x2, y2):
                    continue
                zx1, zy1 = to_buf(x1, y1)
                zx2, zy2 = to_buf(x2, y2)
                if self.highlight_overlaps and i in self.overlapping_boxes:
                    color = (0, 0, 255)
                    thickness = max(4, int(4 * text_zoom))
                    label_text = f"L {self.class_info[cls_id]['name']} ⚠"
                else:
                    color = self.class_info[cls_id]["color"]
                    thickness = max(2, int(2 * text_zoom))
                    label_text = f"L {self.class_info[cls_id]['name']}"
                cv2.rectangle(image_resized, (zx1, zy1), (zx2, zy2), color[::-1], thickness)
                cv2.putText(image_resized, label_text,
                            (zx1, zy1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5 * text_zoom,
                            color[::-1], max(1, int(text_zoom)))

        # Draw user boxes (thick lines)
        if should_draw_boxes and self.show_user_boxes:
            for i, (cls_id, x1, y1, x2, y2) in enumerate(self.user_boxes):
                if hasattr(cls_id, 'item'):
                    cls_id = int(cls_id.item())
                else:
                    cls_id = int(cls_id)
                if cls_id not in self.class_info or not self.class_visibility.get(cls_id, True):
                    continue
                if outside(x1, y1, x2, y2):
                    continue
                zx1, zy1 = to_buf(x1, y1)
                zx2, zy2 = to_buf(x2, y2)
                color = self.class_info[cls_id]["color"]
                cv2.rectangle(image_resized, (zx1, zy1), (zx2, zy2),
                              color[::-1], max(2, int(2 * text_zoom)))
                cv2.putText(image_resized, f"U {self.class_info[cls_id]['name']}",
                            (zx1, zy1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5 * text_zoom,
                            color[::-1], max(1, int(text_zoom)))

        # Place the rendered buffer at its crop offset in canvas coords
        image_pil = Image.fromarray(image_resized)
        self.photo = ImageTk.PhotoImage(image=image_pil)
        self.canvas.create_image(crop_canvas_x, crop_canvas_y, anchor=tk.NW, image=self.photo)

        # Filename + box-count overlay — stick it to the viewport top-left
        if self.current_image_path:
            filename = self.current_image_path.name
            font_size = 14
            bar_height = font_size + 10
            self.canvas.create_rectangle(view_x, view_y, view_x + cw, view_y + bar_height,
                                         fill='#000000', stipple='gray75',
                                         outline='', tags='filename_overlay')
            self.canvas.create_text(view_x + 10, view_y + 5, anchor=tk.NW,
                                    text=f"File: {filename}",
                                    fill='#FFFFFF',
                                    font=('Arial', font_size, 'bold'),
                                    tags='filename_overlay')
            counts_text = (
                f"U:{len(self.user_boxes)}  "
                f"L:{len(self.label_boxes)}  "
                f"P:{len(self.model_boxes)}"
            )
            self.canvas.create_text(view_x + cw - 10, view_y + 5, anchor=tk.NE,
                                    text=counts_text,
                                    fill='#FFD166',
                                    font=('Arial', font_size, 'bold'),
                                    tags='filename_overlay')

        # Scrollregion must reflect the full zoomed size so scrollbars are accurate
        self.canvas.configure(scrollregion=(0, 0, full_w, full_h))

        # Re-highlight any active selection
        self.highlight_selected_box()
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
        if self.current_image is None or self.batch_mode:
            return
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)
        img_x, img_y = self.canvas_to_image_coords(canvas_x, canvas_y)

        # 1) Pointer on a box edge → start an edge-resize.
        hover = self._detect_edge_hover(img_x, img_y)
        if hover is not None and not self.editing_box:
            source, idx, handle = hover
            self._start_edge_resize(source, idx, handle)
            return

        # 2) Pointer inside an existing box → tentative click-to-select.
        # We commit to the selection on release; if the user actually drags,
        # we'll switch over to drawing a new box from the press point.
        inside = self._detect_box_at(img_x, img_y)
        if inside is not None and not self.editing_box:
            self._click_select_candidate = inside
            self.start_x, self.start_y = img_x, img_y
            return

        # 3) Empty area → start drawing a new box.
        self.drawing = True
        self.start_x, self.start_y = img_x, img_y

        color = self.class_info[self.current_class]["color"]
        rgb_color = self.bgr_to_rgb_hex(color)
        canvas_start_x, canvas_start_y = self.image_to_canvas_coords(self.start_x, self.start_y)
        self.current_rect = self.canvas.create_rectangle(
            canvas_start_x, canvas_start_y, canvas_start_x, canvas_start_y,
            outline=rgb_color, width=2
        )
        
    def draw(self, event):
        # Edge-resize in progress?
        if self._resize_active:
            canvas_x = self.canvas.canvasx(event.x)
            canvas_y = self.canvas.canvasy(event.y)
            img_x, img_y = self.canvas_to_image_coords(canvas_x, canvas_y)
            self._update_edge_resize(img_x, img_y)
            return

        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)
        cur_x, cur_y = self.canvas_to_image_coords(canvas_x, canvas_y)

        # Pending click-select on a box: if motion exceeds ~4 canvas pixels,
        # treat as a drag and switch into drawing-a-new-box mode from the press
        # point. Otherwise stay pending until release.
        if self._click_select_candidate is not None and not self.drawing:
            threshold_img = max(1.0, 4.0 / max(self.zoom_scale, 0.01))
            if abs(cur_x - self.start_x) > threshold_img or abs(cur_y - self.start_y) > threshold_img:
                self._click_select_candidate = None
                self.drawing = True
                color = self.class_info[self.current_class]["color"]
                rgb_color = self.bgr_to_rgb_hex(color)
                canvas_start_x, canvas_start_y = self.image_to_canvas_coords(self.start_x, self.start_y)
                self.current_rect = self.canvas.create_rectangle(
                    canvas_start_x, canvas_start_y, canvas_start_x, canvas_start_y,
                    outline=rgb_color, width=2
                )
            else:
                return

        if not self.drawing and not self.editing_box:
            return
        
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
        # Finalize edge-resize if active
        if self._resize_active:
            self._finish_edge_resize()
            return

        # Click-without-drag inside an existing box → commit selection.
        if self._click_select_candidate is not None:
            source, idx = self._click_select_candidate
            self._click_select_candidate = None
            boxes = self.user_boxes if source == 'user' else self.label_boxes
            if 0 <= idx < len(boxes):
                # Clear any prior selection / prediction highlight before applying ours
                if self.selected_box_index is not None or self.selected_prediction is not None:
                    self.clear_selections()
                self.selected_box_index = idx
                self.selected_box_source = source
                self.highlight_selected_box()
                cls_id = boxes[idx][0]
                if hasattr(cls_id, 'item'):
                    cls_id = int(cls_id.item())
                else:
                    cls_id = int(cls_id)
                cls_name = self.class_info.get(cls_id, {}).get('name', '?')
                self.update_status(
                    f"Selected {source} box {idx+1} ({cls_name}) — press 1-9/0 to change class, Esc to cancel"
                )
            return

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
            # Snapshot state before editing
            self._push_history()
            # Update existing box
            self.user_boxes[self.selected_box_index] = (
                self.user_boxes[self.selected_box_index][0],  # Keep the same class
                x1, y1, x2, y2
            )
            self.editing_box = False
        else:
            # Snapshot state before adding new box
            self._push_history()
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

        # Snapshot state before pasting
        self._push_history()

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
            # Snapshot state before deletion
            self._push_history()
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

            # Defensive: index may be stale across image switches
            if not (0 <= self.selected_box_index < len(boxes)):
                self.selected_box_index = None
                self.selected_box_source = None
                return

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
        # If an edge resize is happening (or just happened), don't take over.
        if self._resize_active:
            return

        # Double-click takes priority over a pending click-select that was
        # set by the first half of the double-click sequence.
        self._click_select_candidate = None

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
        """Zoom with the mouse wheel, anchored on the cursor position."""
        if self.current_image is None:
            return

        # Image-space point under the cursor — keep this stable across the zoom.
        old_canvas_x = self.canvas.canvasx(event.x)
        old_canvas_y = self.canvas.canvasy(event.y)
        anchor_img_x = old_canvas_x / self.zoom_scale
        anchor_img_y = old_canvas_y / self.zoom_scale

        if event.num == 5 or event.delta < 0:
            self.zoom_scale = max(self.min_zoom, self.zoom_scale / self.zoom_factor)
        elif event.num == 4 or event.delta > 0:
            self.zoom_scale = min(self.max_zoom, self.zoom_scale * self.zoom_factor)

        # Update scrollregion first so xview_moveto fractions are based on the
        # new size, THEN scroll the viewport, THEN render (so display_image
        # reads the post-scroll viewport and renders the right crop).
        new_w = max(1, int(self.current_image.shape[1] * self.zoom_scale))
        new_h = max(1, int(self.current_image.shape[0] * self.zoom_scale))
        self.canvas.configure(scrollregion=(0, 0, new_w, new_h))

        new_canvas_x = anchor_img_x * self.zoom_scale
        new_canvas_y = anchor_img_y * self.zoom_scale
        offset_x = new_canvas_x - event.x
        offset_y = new_canvas_y - event.y
        if new_w > 0:
            self.canvas.xview_moveto(max(0.0, min(1.0, offset_x / new_w)))
        if new_h > 0:
            self.canvas.yview_moveto(max(0.0, min(1.0, offset_y / new_h)))

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

            # Auto-bind a default hotkey if one is still free
            self._init_default_hotkeys()

            # Save configuration
            self.save_class_config()

            # Refresh class buttons
            self.setup_class_buttons()

            # Select the new class
            self.select_class(next_id)

            # Sync the crop-review window (if open) so its dropdown lists
            # the new class.
            self._crop_review_refresh_class_info()

    def edit_class_name(self, class_id):
        """Edit the name of a class"""
        current_name = self.class_info[class_id]["name"]
        new_name = simpledialog.askstring("Edit Class", "Enter new name:",
                                        initialvalue=current_name)
        if new_name:
            self.class_info[class_id]["name"] = new_name
            self.save_class_config()
            self.class_labels[class_id].configure(text=new_name)
            # Sync the crop-review window (if open) so the dropdown and
            # captions pick up the new name.
            self._crop_review_refresh_class_info()

    def delete_class(self, class_id):
        """Delete a class if it's not the last one"""
        if len(self.class_info) <= 1:
            messagebox.showwarning("Warning", "Cannot delete the last class")
            return
            
        if messagebox.askyesno("Confirm Delete",
                              f"Delete class '{self.class_info[class_id]['name']}'?"):
            # Remove class from configuration
            del self.class_info[class_id]

            # Drop any hotkey that pointed to this class
            self._cleanup_hotkeys()

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

            # Sync the crop-review window — closes itself if the reviewed
            # class was the one deleted, otherwise just refreshes dropdown.
            self._crop_review_refresh_class_info()

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
        self._highlight_selected_class()

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
            
    def reassign_box_class(self, new_class_id, keep_selection=False):
        """Reassign the class of the selected box.

        keep_selection=True is used by the hotkey path so the user can press
        another number to reassign again without re-clicking.
        """
        if self.selected_box_index is None:
            return

        # Snapshot state before class change
        self._push_history()

        # Update the class ID based on the box source
        if self.selected_box_source == 'user':
            _, x1, y1, x2, y2 = self.user_boxes[self.selected_box_index]
            self.user_boxes[self.selected_box_index] = (new_class_id, x1, y1, x2, y2)
        else:  # label
            _, x1, y1, x2, y2 = self.label_boxes[self.selected_box_index]
            self.label_boxes[self.selected_box_index] = (new_class_id, x1, y1, x2, y2)

        if not keep_selection:
            self.clear_selections()

        # Class change can change which boxes overlap
        if self.highlight_overlaps:
            self.detect_current_overlaps()

        self.save_labels()
        self.display_image()  # also re-applies highlight_selected_box if we kept the selection
        self.update_status(f"Changed class to: {self.class_info[new_class_id]['name']}")

    # --- Bulk class reassignment ---
    def _find_label_file_for(self, img_path):
        """Return the .txt label-file path for an image, or None if not locatable."""
        if img_path is None:
            return None
        if self.label_folder:
            return Path(self.label_folder) / f"{img_path.stem}.txt"
        img_path_str = str(img_path)
        for img_folder, lbl_folder in getattr(self, 'per_image_label_folders', {}).items():
            if img_path_str.startswith(img_folder):
                return Path(lbl_folder) / f"{img_path.stem}.txt"
        # Fallback: conventional siblings
        p = Path(img_path)
        for cand in [p.parent / 'labels', p.parent.parent / 'labels']:
            if cand.exists():
                return cand / f"{img_path.stem}.txt"
        return None

    def _scope_paths(self, scope):
        """Return the list of image paths for a given scope string."""
        if scope == 'current':
            return [self.current_image_path] if self.current_image_path else []
        if scope == 'filtered':
            return list(self.filtered_image_paths)
        return list(self.image_paths)

    def _count_class_labels(self, class_id, scope):
        """Count labels matching class_id within scope. Returns (label_count, image_count)."""
        paths = self._scope_paths(scope)
        label_count = 0
        image_count = 0
        for p in paths:
            if p is None:
                continue
            count_in_img = 0
            # Prefer cache when available
            cached = self.labels_cache.get(str(p))
            if cached is not None:
                count_in_img = sum(1 for lbl in cached if int(lbl[0]) == class_id)
            else:
                lf = self._find_label_file_for(p)
                if lf is None or not lf.exists():
                    continue
                try:
                    with open(lf) as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) < 5:
                                continue
                            try:
                                if int(float(parts[0])) == class_id:
                                    count_in_img += 1
                            except ValueError:
                                continue
                except (IOError, OSError):
                    continue
            if count_in_img > 0:
                label_count += count_in_img
                image_count += 1
        return label_count, image_count

    def _apply_bulk_reassign(self, from_class, to_class, scope):
        """Rewrite label files in scope so that class==from_class becomes to_class.

        Touches files directly so the operation works across the whole dataset
        without keeping every image in memory. The current image's in-memory
        state is saved before, and reloaded after.
        """
        paths = self._scope_paths(scope)
        if not paths:
            messagebox.showinfo("Reassign", "No images in scope.")
            return

        # If the current image is in scope, persist user_boxes first so we
        # don't lose unsaved drawings.
        current_in_scope = (self.current_image_path is not None
                            and self.current_image_path in paths)
        if current_in_scope:
            self.save_labels()

        self.progress_var.set(0)
        self.filter_result_var.set("Reassigning…")
        self.root.update_idletasks()

        total = len(paths)
        modified_files = 0
        modified_labels = 0
        last_ui = time.time()

        for i, p in enumerate(paths):
            lf = self._find_label_file_for(p)
            if lf is None or not lf.exists():
                continue
            try:
                with open(lf) as f:
                    lines = f.readlines()
            except (IOError, OSError):
                continue

            changed_in_file = 0
            new_lines = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) < 5:
                    new_lines.append(line)
                    continue
                try:
                    cls_id = int(float(parts[0]))
                except ValueError:
                    new_lines.append(line)
                    continue
                if cls_id == from_class:
                    parts[0] = str(to_class)
                    new_lines.append(' '.join(parts) + '\n')
                    changed_in_file += 1
                else:
                    new_lines.append(line)

            if changed_in_file:
                try:
                    with open(lf, 'w') as f:
                        f.writelines(new_lines)
                    modified_files += 1
                    modified_labels += changed_in_file
                    # Invalidate cached labels so subsequent filters see the new state
                    self.labels_cache.pop(str(p), None)
                except (IOError, OSError) as e:
                    print(f"Failed to write {lf}: {e}")
                    continue

            now = time.time()
            if now - last_ui >= 0.3 or i == total - 1:
                self.progress_var.set(((i + 1) / total) * 100)
                self.filter_result_var.set(
                    f"Reassigning: {i+1}/{total} | {modified_labels} labels in {modified_files} files"
                )
                self.root.update_idletasks()
                last_ui = now

        self.progress_var.set(100)

        # Reload the current image's labels so the canvas reflects the change.
        if current_in_scope:
            self.user_boxes = []  # those were saved into the file already
            self.load_labels()
            if self.highlight_overlaps:
                self.detect_current_overlaps()
            self.display_image()

        from_name = self.class_info.get(from_class, {}).get('name', f'?{from_class}')
        to_name = self.class_info.get(to_class, {}).get('name', f'?{to_class}')
        summary = f"Reassigned {modified_labels} labels ({from_name} → {to_name}) in {modified_files} files"
        self.filter_result_var.set(summary)
        self.update_status(summary)
        messagebox.showinfo("Reassign complete", summary)

    def _collect_class_crops(self, target_class, paths):
        """Walk `paths`, return list of {img_path, line_idx, cls_id, bbox_norm}
        for every label line matching target_class.

        bbox_norm is the YOLO-normalized (cx, cy, w, h) tuple — kept normalized
        so the pixel rectangle is resolved at *render* time against the
        actual loaded image's dimensions. This avoids subtle wrong-crop bugs
        if the cached header dimensions disagree with what cv2 actually reads.
        """
        crops = []
        for img_path in paths:
            if img_path is None:
                continue
            lf = self._find_label_file_for(img_path)
            if lf is None or not lf.exists():
                continue
            try:
                with open(lf) as f:
                    lines = f.readlines()
            except (IOError, OSError):
                continue
            for li, line in enumerate(lines):
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                try:
                    cls_id = int(float(parts[0]))
                    if cls_id != target_class:
                        continue
                    cx, cy, w, h = map(float, parts[1:5])
                except ValueError:
                    continue
                crops.append({
                    'img_path': img_path,
                    'line_idx': li,
                    'cls_id': cls_id,
                    'bbox_norm': (cx, cy, w, h),
                })
        return crops

    @staticmethod
    def _iou_xyxy(a, b):
        """Compute IoU of two boxes in (x1, y1, x2, y2) pixel coords."""
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
        iw = max(0.0, ix2 - ix1); ih = max(0.0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0.0:
            return 0.0
        a_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        b_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        union = a_area + b_area - inter
        return inter / union if union > 0 else 0.0

    def _bad_scan_worker(self, config):
        """Background-thread scanner. For every GT label in every image of
        the scope: run the model, find the best-IoU prediction, and decide
        whether the GT is suspicious. Pushes ('entry', dict) / ('progress',
        i, total) / ('done', reason) / ('error', msg) messages onto
        ``self._bad_scan_queue``.

        Must not touch any Tk widget — the main thread reads the queue and
        renders results."""
        import queue as _queue  # local alias keeps scope tidy
        try:
            paths = list(config['paths'])
            conf_thr = float(config['conf'])
            iou_match = float(config['iou_match'])
            detect_mismatch = bool(config['mismatch'])
            detect_missing = bool(config['missing'])
            for i, path in enumerate(paths):
                if self._bad_scan_cancelled.is_set():
                    self._bad_scan_queue.put(("done", "cancelled"))
                    return
                try:
                    img = cv2.imread(str(path))
                except Exception:
                    img = None
                if img is None:
                    self._bad_scan_queue.put(("progress", i + 1, len(paths)))
                    continue
                ih, iw = img.shape[:2]
                # Run model
                try:
                    results = self.pretrained_model(img, verbose=False, conf=conf_thr)
                except Exception as e:
                    self._bad_scan_queue.put(("error", f"model error on {path.name}: {e}"))
                    self._bad_scan_queue.put(("progress", i + 1, len(paths)))
                    continue
                preds = []
                for r in results:
                    boxes = getattr(r, 'boxes', None)
                    if boxes is None:
                        continue
                    for box in boxes:
                        try:
                            xyxy = box.xyxy[0].cpu().numpy()
                            pcls = int(box.cls[0].cpu().numpy())
                            pconf = float(box.conf[0].cpu().numpy())
                        except Exception:
                            continue
                        preds.append((
                            pcls, pconf,
                            float(xyxy[0]), float(xyxy[1]),
                            float(xyxy[2]), float(xyxy[3]),
                        ))
                # Read GT labels (raw lines so line_idx aligns with file)
                lf = self._find_label_file_for(path)
                if lf is None or not lf.exists():
                    self._bad_scan_queue.put(("progress", i + 1, len(paths)))
                    continue
                try:
                    lines = lf.read_text().splitlines()
                except (IOError, OSError):
                    self._bad_scan_queue.put(("progress", i + 1, len(paths)))
                    continue
                for li, line in enumerate(lines):
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    try:
                        gt_cls = int(float(parts[0]))
                        cx, cy, bw, bh = map(float, parts[1:5])
                    except ValueError:
                        continue
                    gx1 = (cx - bw / 2.0) * iw
                    gy1 = (cy - bh / 2.0) * ih
                    gx2 = (cx + bw / 2.0) * iw
                    gy2 = (cy + bh / 2.0) * ih
                    gt_box = (gx1, gy1, gx2, gy2)
                    best_iou = 0.0
                    best_pred = None  # (cls, conf)
                    for pcls, pconf, px1, py1, px2, py2 in preds:
                        iou = self._iou_xyxy(gt_box, (px1, py1, px2, py2))
                        if iou > best_iou:
                            best_iou = iou
                            best_pred = (pcls, pconf)
                    entry = None
                    if best_pred is None or best_iou < iou_match:
                        if detect_missing:
                            entry = {
                                'img_path': path,
                                'line_idx': li,
                                'cls_id': gt_cls,
                                'bbox_norm': (cx, cy, bw, bh),
                                'pred_cls_id': None,
                                'pred_conf': 0.0,
                                'iou': best_iou,
                                'score': 1.0 - best_iou,
                                'kind': 'missing',
                            }
                    else:
                        pcls, pconf = best_pred
                        if not self._classes_equivalent(gt_cls, pcls):
                            if detect_mismatch:
                                entry = {
                                    'img_path': path,
                                    'line_idx': li,
                                    'cls_id': gt_cls,
                                    'bbox_norm': (cx, cy, bw, bh),
                                    'pred_cls_id': pcls,
                                    'pred_conf': pconf,
                                    'iou': best_iou,
                                    'score': pconf,
                                    'kind': 'class_mismatch',
                                }
                    if entry is not None:
                        self._bad_scan_queue.put(("entry", entry))
                self._bad_scan_queue.put(("progress", i + 1, len(paths)))
            self._bad_scan_queue.put(("done", "complete"))
        except Exception as e:
            try:
                self._bad_scan_queue.put(("error", f"scan worker crashed: {e}"))
                self._bad_scan_queue.put(("done", "error"))
            except Exception:
                pass

    def _bad_scan_drain(self):
        """Main thread: drain ``_bad_scan_queue`` and stream entries into
        the open crop review window. Schedules itself until the scan is
        done; safe no-op if the review window was closed mid-scan."""
        if not hasattr(self, '_bad_scan_queue'):
            return
        if self._crop_review_window is None:
            # Window closed mid-scan — abort the scan cleanly.
            self._bad_scan_cancelled.set()
            return
        import queue as _queue
        new_entries = []
        progress = None
        done_msg = None
        last_error = None
        while True:
            try:
                msg = self._bad_scan_queue.get_nowait()
            except _queue.Empty:
                break
            kind = msg[0]
            if kind == 'entry':
                new_entries.append(msg[1])
            elif kind == 'progress':
                progress = (msg[1], msg[2])
            elif kind == 'done':
                done_msg = msg[1]
            elif kind == 'error':
                last_error = msg[1]
        if new_entries:
            self._crop_review_entries.extend(new_entries)
            # Re-sort by descending score so the worst rises to the top.
            self._crop_review_entries.sort(key=lambda e: -e.get('score', 0.0))
            # Coalesce render (re-use the after_idle scheme used by reassign).
            prev = getattr(self, '_crop_review_render_after', None)
            if prev is not None:
                try:
                    self.root.after_cancel(prev)
                except tk.TclError:
                    pass
            def _do_render():
                self._crop_review_render_after = None
                if self._crop_review_window is not None:
                    self._crop_review_render()
            self._crop_review_render_after = self.root.after_idle(_do_render)
        if progress is not None or done_msg is not None:
            self._bad_scan_update_status(progress, done_msg, last_error)
        if done_msg is None:
            # keep polling
            self._bad_scan_drain_after = self.root.after(150, self._bad_scan_drain)
        else:
            self._bad_scan_drain_after = None

    def _bad_scan_update_status(self, progress, done_msg, last_error):
        """Format the bad-scan progress line for the review window's status bar."""
        if self._crop_review_window is None:
            return
        n_bad = len(self._crop_review_entries)
        parts = []
        if progress is not None:
            i, total = progress
            parts.append(f"Scanned {i}/{total}")
        parts.append(f"{n_bad} bad found")
        if done_msg == 'complete':
            parts.append("✓ done")
        elif done_msg == 'cancelled':
            parts.append("⛔ cancelled")
        elif done_msg == 'error':
            parts.append("❌ error")
        if last_error:
            parts.append(last_error[:60])
        self._crop_review_status_var.set("  ·  ".join(parts))

    def _rewrite_label_line(self, img_path, line_idx, new_class_id):
        """Rewrite a single line's class id in the label file. Returns True on success."""
        lf = self._find_label_file_for(img_path)
        if lf is None or not lf.exists():
            return False
        try:
            with open(lf) as f:
                lines = f.readlines()
        except (IOError, OSError):
            return False
        if not (0 <= line_idx < len(lines)):
            return False
        parts = lines[line_idx].strip().split()
        if len(parts) < 5:
            return False
        parts[0] = str(new_class_id)
        lines[line_idx] = ' '.join(parts) + '\n'
        try:
            with open(lf, 'w') as f:
                f.writelines(lines)
        except (IOError, OSError):
            return False
        # Invalidate the cache so subsequent filters see the new state
        self.labels_cache.pop(str(img_path), None)
        return True

    def _delete_label_line(self, img_path, line_idx):
        """Drop a single line from the label file. Returns True on success.

        If removing the line leaves the file empty, the file is removed too,
        matching the convention used by save_labels.
        """
        lf = self._find_label_file_for(img_path)
        if lf is None or not lf.exists():
            return False
        try:
            with open(lf) as f:
                lines = f.readlines()
        except (IOError, OSError):
            return False
        if not (0 <= line_idx < len(lines)):
            return False
        del lines[line_idx]
        try:
            if lines:
                with open(lf, 'w') as f:
                    f.writelines(lines)
            else:
                os.remove(lf)
        except (IOError, OSError):
            return False
        self.labels_cache.pop(str(img_path), None)
        return True

    def edit_class_groups(self):
        """Open a small editor where the user can define equivalence groups
        like ``bus: bus_m, bus_l, bus_xl`` (one group per line). Saved into
        class_config.json under the ``class_groups`` key."""
        win = tk.Toplevel(self.root)
        win.title("Edit Class Groups")
        win.geometry("600x440")
        win.transient(self.root)
        ttk.Label(
            win,
            text=("One group per line, format:  group_name: member1, member2, ...\n"
                  "Members are class names from this project.\n"
                  "Used by 'Find Bad Annotations' to treat all members as the "
                  "same class.\nLeave blank to disable grouping."),
            justify=tk.LEFT,
            foreground="#555",
        ).pack(anchor=tk.W, padx=10, pady=(10, 4))

        # Pre-populate from the current class_groups
        groups = getattr(self, 'class_groups', {}) or {}
        text_lines = []
        for k, v in groups.items():
            members = v if isinstance(v, (list, tuple)) else [v]
            text_lines.append(f"{k}: {', '.join(str(m) for m in members)}")
        txt = tk.Text(win, wrap=tk.WORD, height=18, font=('TkDefaultFont', 10))
        txt.pack(fill=tk.BOTH, expand=True, padx=10, pady=4)
        if text_lines:
            txt.insert("1.0", "\n".join(text_lines))

        # Show the available class names as a sticky hint above the buttons
        all_names = sorted(info['name'] for info in self.class_info.values())
        ttk.Label(
            win,
            text="Available classes: " + ", ".join(all_names),
            foreground="#888",
            wraplength=560,
            justify=tk.LEFT,
        ).pack(anchor=tk.W, padx=10, pady=(2, 0))

        status_var = tk.StringVar(value="")
        ttk.Label(win, textvariable=status_var, foreground="#cc2222").pack(
            anchor=tk.W, padx=10, pady=2
        )
        btn_row = ttk.Frame(win)
        btn_row.pack(fill=tk.X, padx=10, pady=(4, 10))

        def _parse_and_save():
            raw = txt.get("1.0", "end").strip()
            known = {info['name'] for info in self.class_info.values()}
            new_groups = {}
            unknown_members = []
            for line in raw.splitlines():
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if ':' not in line:
                    status_var.set(f"Missing ':' on line: {line!r}")
                    return
                group_key, members_str = line.split(':', 1)
                group_key = group_key.strip()
                members = [m.strip() for m in members_str.split(',') if m.strip()]
                if not group_key or not members:
                    status_var.set(f"Empty group or members in: {line!r}")
                    return
                # Validate every member exists as a class name
                for m in members:
                    if m not in known:
                        unknown_members.append(m)
                new_groups[group_key] = members
            if unknown_members:
                status_var.set(
                    f"Unknown class name(s): {', '.join(sorted(set(unknown_members)))}"
                )
                return
            self.class_groups = new_groups
            self._rebuild_class_group_lookup()
            self.save_class_config()
            win.destroy()

        ttk.Button(btn_row, text="Save", command=_parse_and_save).pack(side=tk.RIGHT)
        ttk.Button(btn_row, text="Cancel", command=win.destroy).pack(
            side=tk.RIGHT, padx=4
        )

    def show_find_bad_dialog(self):
        """Config dialog for 'Find Bad Annotations'. Reads scope/threshold
        choices, then opens the bad-annotation review window."""
        if not self.pretrained_model:
            messagebox.showwarning(
                "No model loaded",
                "Load a YOLO model first (Setup tab → Pretrained Model)."
            )
            return
        if not self.filtered_image_paths:
            messagebox.showinfo("No images", "Load an image folder first.")
            return

        win = tk.Toplevel(self.root)
        win.title("Find Bad Annotations")
        win.geometry("440x280")
        win.transient(self.root)

        # Confidence threshold for predictions
        conf_var = tk.DoubleVar(value=0.40)
        iou_var = tk.DoubleVar(value=0.50)
        mismatch_var = tk.BooleanVar(value=True)
        missing_var = tk.BooleanVar(value=True)

        frm = ttk.Frame(win); frm.pack(fill=tk.BOTH, expand=True, padx=12, pady=10)

        def _row(parent, label_text, var, width=8):
            row = ttk.Frame(parent); row.pack(fill=tk.X, pady=3)
            ttk.Label(row, text=label_text, width=28, anchor='w').pack(side=tk.LEFT)
            ttk.Entry(row, textvariable=var, width=width).pack(side=tk.LEFT)

        _row(frm, "Prediction confidence (≥):", conf_var)
        _row(frm, "IoU threshold for match (≥):", iou_var)

        ttk.Checkbutton(frm, text="Class mismatch (GT overlaps confident pred of different class)",
                        variable=mismatch_var).pack(anchor=tk.W, pady=(8, 2))
        ttk.Checkbutton(frm, text="Missing prediction (no overlap; model sees nothing)",
                        variable=missing_var).pack(anchor=tk.W, pady=2)

        # Show whether class_groups will be used
        n_groups = len(getattr(self, 'class_groups', {}) or {})
        ttk.Label(
            frm,
            text=(f"Class groups: {n_groups} defined (will be used to relax "
                  "class-match checks)") if n_groups else
                 ("Class groups: none defined — exact class match. "
                  "Define via Setup → 'Edit Class Groups…' if needed."),
            foreground="#666",
            wraplength=400,
            justify=tk.LEFT,
        ).pack(anchor=tk.W, pady=(8, 4))

        # Scope: current scope is "currently filtered images" (the user already
        # narrowed the working set via Filter tab). Keep it simple — just scan
        # whatever is in filtered_image_paths.
        ttk.Label(
            frm,
            text=f"Scope: {len(self.filtered_image_paths)} currently filtered images.",
            foreground="#666",
        ).pack(anchor=tk.W, pady=(0, 4))

        status_var = tk.StringVar()
        ttk.Label(frm, textvariable=status_var, foreground="#cc2222").pack(
            anchor=tk.W, pady=2
        )

        btn_row = ttk.Frame(frm); btn_row.pack(fill=tk.X, pady=(10, 0))

        def _start():
            if not (mismatch_var.get() or missing_var.get()):
                status_var.set("Pick at least one detection signal.")
                return
            try:
                conf = float(conf_var.get())
                iou_match = float(iou_var.get())
            except (tk.TclError, ValueError):
                status_var.set("Confidence and IoU must be numbers.")
                return
            if not (0 < conf <= 1.0):
                status_var.set("Confidence must be in (0, 1].")
                return
            if not (0 < iou_match <= 1.0):
                status_var.set("IoU must be in (0, 1].")
                return
            config = {
                'paths': list(self.filtered_image_paths),
                'conf': conf,
                'iou_match': iou_match,
                'mismatch': mismatch_var.get(),
                'missing': missing_var.get(),
            }
            win.destroy()
            self.open_bad_label_review(config)

        ttk.Button(btn_row, text="Start scan", command=_start).pack(side=tk.RIGHT)
        ttk.Button(btn_row, text="Cancel", command=win.destroy).pack(
            side=tk.RIGHT, padx=4
        )

    def show_reassign_class_dialog(self):
        """Open a dialog to bulk-reassign every label of one class to another."""
        if len(self.class_info) < 2:
            messagebox.showinfo("Reassign", "Need at least 2 classes to reassign.")
            return
        if not self.image_paths:
            messagebox.showinfo("Reassign", "No images loaded.")
            return

        win = tk.Toplevel(self.root)
        win.title("Reassign Class Labels")
        win.transient(self.root)
        win.resizable(False, False)

        ttk.Label(win, text="Change all labels:",
                  font=('TkDefaultFont', 11, 'bold')).pack(padx=20, pady=(15, 8))

        class_options = [f"{cid}: {info['name']}"
                         for cid, info in sorted(self.class_info.items())]

        from_frame = ttk.Frame(win)
        from_frame.pack(fill=tk.X, padx=20, pady=2)
        ttk.Label(from_frame, text="From class:", width=12).pack(side=tk.LEFT)
        from_var = tk.StringVar(value=class_options[0])
        from_cb = ttk.Combobox(from_frame, textvariable=from_var, values=class_options,
                               state='readonly', width=28)
        from_cb.pack(side=tk.LEFT)

        to_frame = ttk.Frame(win)
        to_frame.pack(fill=tk.X, padx=20, pady=2)
        ttk.Label(to_frame, text="To class:", width=12).pack(side=tk.LEFT)
        to_var = tk.StringVar(value=class_options[1])
        to_cb = ttk.Combobox(to_frame, textvariable=to_var, values=class_options,
                             state='readonly', width=28)
        to_cb.pack(side=tk.LEFT)

        ttk.Label(win, text="Scope:", font=('TkDefaultFont', 10, 'bold')
                  ).pack(anchor=tk.W, padx=20, pady=(15, 5))
        scope_var = tk.StringVar(value='dataset')
        scope_frame = ttk.Frame(win)
        scope_frame.pack(fill=tk.X, padx=30)
        ttk.Radiobutton(scope_frame, text="Current image only",
                        variable=scope_var, value='current').pack(anchor=tk.W)
        ttk.Radiobutton(scope_frame,
                        text=f"Filtered images ({len(self.filtered_image_paths)})",
                        variable=scope_var, value='filtered').pack(anchor=tk.W)
        ttk.Radiobutton(scope_frame,
                        text=f"Entire dataset ({len(self.image_paths)})",
                        variable=scope_var, value='dataset').pack(anchor=tk.W)

        preview_var = tk.StringVar(value="Computing preview…")
        ttk.Label(win, textvariable=preview_var, foreground='#0066cc'
                  ).pack(padx=20, pady=(15, 5))
        ttk.Label(win, text="⚠ Writes label files directly — cannot be undone.",
                  foreground='#cc6600').pack(padx=20, pady=(0, 8))

        def parse_class_id(s):
            try:
                return int(s.split(':', 1)[0].strip())
            except (ValueError, IndexError):
                return None

        def update_preview(*_):
            from_id = parse_class_id(from_var.get())
            to_id = parse_class_id(to_var.get())
            if from_id is None or to_id is None:
                preview_var.set("(invalid selection)")
                return
            if from_id == to_id:
                preview_var.set("From and To are the same — no change would be made.")
                return
            preview_var.set("Counting matching labels…")
            win.update_idletasks()
            count, img_count = self._count_class_labels(from_id, scope_var.get())
            preview_var.set(f"Will change {count} labels across {img_count} images.")

        from_cb.bind('<<ComboboxSelected>>', update_preview)
        to_cb.bind('<<ComboboxSelected>>', update_preview)
        for w in scope_frame.winfo_children():
            if isinstance(w, ttk.Radiobutton):
                w.config(command=update_preview)
        update_preview()

        btn_frame = ttk.Frame(win)
        btn_frame.pack(pady=12)

        def on_apply():
            from_id = parse_class_id(from_var.get())
            to_id = parse_class_id(to_var.get())
            if from_id is None or to_id is None:
                return
            if from_id == to_id:
                messagebox.showinfo("Reassign", "From and To are the same.")
                return
            from_name = self.class_info[from_id]['name']
            to_name = self.class_info[to_id]['name']
            if not messagebox.askyesno(
                "Confirm reassign",
                f"Change all '{from_name}' labels to '{to_name}'?\n\n"
                f"Scope: {scope_var.get()}\n"
                f"This rewrites label files and cannot be undone."
            ):
                return
            win.destroy()
            self._apply_bulk_reassign(from_id, to_id, scope_var.get())

        ttk.Button(btn_frame, text="Cancel", command=win.destroy).pack(side=tk.LEFT, padx=8)
        ttk.Button(btn_frame, text="Apply", command=on_apply).pack(side=tk.LEFT, padx=8)

        # Center on parent
        win.update_idletasks()
        try:
            pw, ph = self.root.winfo_width(), self.root.winfo_height()
            px, py = self.root.winfo_rootx(), self.root.winfo_rooty()
            ww, wh = win.winfo_width(), win.winfo_height()
            win.geometry(f"+{px + (pw - ww) // 2}+{py + (ph - wh) // 2}")
        except Exception:
            pass

    # --- Crop-review window: audit & relabel one class at a time ---
    def open_class_crop_review(self):
        """Entry point: open the per-class crop review for the current
        Filter-by-Class selection across the currently-filtered images."""
        if not self.image_paths:
            messagebox.showinfo("Crop Review", "Load images first.")
            return
        # Resolve which class to review
        target_class = None
        if self.filter_by_class_enabled and self.filter_class_id in self.class_info:
            target_class = self.filter_class_id
        else:
            # No Filter-by-Class active: ask the user
            target_class = self._ask_class("Review crops of which class?")
            if target_class is None:
                return

        self.update_status("Collecting crops…")
        self.root.update_idletasks()
        entries = self._collect_class_crops(target_class, self.filtered_image_paths)
        if not entries:
            messagebox.showinfo(
                "Crop Review",
                f"No labels of class '{self.class_info[target_class]['name']}' in the filtered set."
            )
            return
        self._open_crop_review_window(entries, target_class)

    def _ask_class(self, prompt):
        """Tiny modal to pick a class id. Returns class_id or None."""
        if not self.class_info:
            return None
        dlg = tk.Toplevel(self.root)
        dlg.title("Pick a class")
        dlg.transient(self.root)
        dlg.resizable(False, False)
        ttk.Label(dlg, text=prompt, font=('TkDefaultFont', 10, 'bold')
                  ).pack(padx=20, pady=(15, 8))
        opts = [f"{cid}: {info['name']}" for cid, info in sorted(self.class_info.items())]
        var = tk.StringVar(value=opts[0])
        ttk.Combobox(dlg, textvariable=var, values=opts, state='readonly', width=28
                     ).pack(padx=20, pady=4)
        chosen = {'value': None}
        def on_ok():
            try:
                chosen['value'] = int(var.get().split(':', 1)[0])
            except Exception:
                pass
            dlg.destroy()
        ttk.Frame(dlg).pack(pady=4)
        bf = ttk.Frame(dlg)
        bf.pack(pady=10)
        ttk.Button(bf, text="Cancel", command=dlg.destroy).pack(side=tk.LEFT, padx=6)
        ttk.Button(bf, text="OK", command=on_ok).pack(side=tk.LEFT, padx=6)
        dlg.grab_set()
        self.root.wait_window(dlg)
        return chosen['value']

    def _open_crop_review_window(self, entries, target_class, mode='filter'):
        """Open the crop review window.

        Args:
            entries: initial list of crop dicts. In 'bad' mode this is
                typically empty and gets filled by the background scanner.
            target_class: GT class under review (filter mode), or None in
                'bad' mode where each entry can have any GT class.
            mode: 'filter' (one-class crop review) or 'bad' (mismatch /
                missing annotations).
        """
        if self._crop_review_window is not None and self._crop_review_window.winfo_exists():
            self._crop_review_window.destroy()

        self._crop_review_mode = mode
        self._crop_review_entries = entries
        self._crop_review_target = target_class
        self._crop_review_page = 0
        self._crop_review_selected = None
        self._crop_review_dirty_paths = set()
        # Reset persistent caches for a fresh session
        self._crop_review_thumb_cache = {}
        self._crop_review_img_cache = {}

        win = tk.Toplevel(self.root)
        if mode == 'bad':
            win.title("Bad Annotation Review")
        else:
            cls_name = self.class_info[target_class]['name']
            win.title(f"Crop Review — {cls_name}")
        win.geometry("1100x780")
        self._crop_review_window = win

        # Top toolbar: title + page nav + grid selector
        top = ttk.Frame(win)
        top.pack(fill=tk.X, padx=8, pady=6)
        self._crop_review_title_var = tk.StringVar()
        ttk.Label(top, textvariable=self._crop_review_title_var,
                  font=('TkDefaultFont', 11, 'bold')).pack(side=tk.LEFT)
        ttk.Button(top, text="◀ Prev", command=self._crop_review_prev_page).pack(side=tk.LEFT, padx=(20, 4))
        ttk.Button(top, text="Next ▶", command=self._crop_review_next_page).pack(side=tk.LEFT)
        ttk.Label(top, text="   Go to:").pack(side=tk.LEFT, padx=(20, 4))
        self._crop_review_goto_var = tk.StringVar()
        self._crop_review_goto_entry = ttk.Entry(
            top, textvariable=self._crop_review_goto_var, width=5
        )
        self._crop_review_goto_entry.pack(side=tk.LEFT)
        self._crop_review_goto_entry.bind(
            '<Return>', lambda e: self._crop_review_goto_page()
        )
        ttk.Button(top, text="Go",
                   command=self._crop_review_goto_page).pack(side=tk.LEFT, padx=4)
        ttk.Label(top, text="   Grid:").pack(side=tk.LEFT, padx=(20, 4))
        self._crop_review_grid_var = tk.StringVar(value='4x6')
        gcb = ttk.Combobox(top, textvariable=self._crop_review_grid_var,
                           values=['3x4', '4x6', '5x8', '6x10'],
                           width=6, state='readonly')
        gcb.pack(side=tk.LEFT)
        gcb.bind('<<ComboboxSelected>>', lambda e: self._crop_review_render())

        # Action bar (selection + reassign target)
        action = ttk.Frame(win)
        action.pack(fill=tk.X, padx=8, pady=4)
        if mode == 'bad':
            default_help = ("Worst annotations first. 1-9/0 reassigns, "
                            "Delete drops, K marks reviewed (no change). "
                            "Click a crop to jump to the main window.")
        else:
            default_help = ("Click a crop, then press 1-9/0 to reassign "
                            "(selection auto-advances). Delete removes. "
                            "Page nav drops handled.")
        self._crop_review_sel_var = tk.StringVar(value=default_help)
        ttk.Label(action, textvariable=self._crop_review_sel_var,
                  foreground='#0066cc').pack(side=tk.LEFT)
        ttk.Label(action, text="   Reassign to: ").pack(side=tk.LEFT)
        class_options = [f"{cid}: {info['name']}"
                         for cid, info in sorted(self.class_info.items())]
        self._crop_review_target_var = tk.StringVar()
        self._crop_review_target_cb = ttk.Combobox(
            action, textvariable=self._crop_review_target_var,
            values=class_options, state='readonly', width=22
        )
        self._crop_review_target_cb.pack(side=tk.LEFT)
        ttk.Button(action, text="Apply",
                   command=self._crop_review_apply_dropdown).pack(side=tk.LEFT, padx=4)
        # Delete the selected crop's label outright (Delete / Backspace also work)
        delete_btn = tk.Button(action, text="🗑 Delete", fg='#cc2222',
                               command=self._crop_review_delete_selected)
        delete_btn.pack(side=tk.LEFT, padx=4)
        # In bad-annotation mode, give the user a way to stop the background
        # scan and a "mark reviewed" action that just drops the entry without
        # touching the label file.
        if mode == 'bad':
            ttk.Button(action, text="✓ Mark reviewed (K)",
                       command=self._crop_review_mark_reviewed).pack(
                           side=tk.LEFT, padx=4)
            self._bad_scan_cancel_btn = ttk.Button(
                action, text="⛔ Cancel scan",
                command=self._bad_scan_request_cancel,
            )
            self._bad_scan_cancel_btn.pack(side=tk.LEFT, padx=4)

        # Crop grid host
        self._crop_review_grid_frame = ttk.Frame(win)
        self._crop_review_grid_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)
        # Debounced re-render when the grid frame is resized, so cells
        # adaptively fill the window and thumbnails re-fit to the new cell size.
        self._crop_review_resize_after = None
        self._crop_review_last_cell_size = (0, 0)
        self._crop_review_grid_frame.bind('<Configure>',
                                          self._crop_review_on_grid_resize)

        # Status bar
        self._crop_review_status_var = tk.StringVar(value="")
        ttk.Label(win, textvariable=self._crop_review_status_var,
                  relief=tk.SUNKEN, anchor='w').pack(side=tk.BOTTOM, fill=tk.X)

        # Hotkeys: digits (both top-row and num-pad) = reassign selection;
        # Delete/Backspace = remove; arrows = paginate; Esc = close.
        # Suppress when focus is on the page-goto entry so typing a page works.
        def _hk(fn):
            def wrapped(e):
                try:
                    if win.focus_get() is self._crop_review_goto_entry:
                        return None
                except Exception:
                    pass
                return fn(e)
            return wrapped
        for digit in '0123456789':
            win.bind(f"<Key-{digit}>",
                     _hk(lambda e, d=digit: self._crop_review_hotkey(d)))
            win.bind(f"<KP_{digit}>",
                     _hk(lambda e, d=digit: self._crop_review_hotkey(d)))
        win.bind("<Delete>",    _hk(lambda e: self._crop_review_delete_selected()))
        win.bind("<BackSpace>", _hk(lambda e: self._crop_review_delete_selected()))
        win.bind("<Left>",      _hk(lambda e: self._crop_review_prev_page()))
        win.bind("<Right>",     _hk(lambda e: self._crop_review_next_page()))
        # K = mark current bad-annotation entry as reviewed (drop from list,
        # don't touch label file). Only meaningful in bad-annotation mode.
        if mode == 'bad':
            win.bind("<k>", _hk(lambda e: self._crop_review_mark_reviewed()))
            win.bind("<K>", _hk(lambda e: self._crop_review_mark_reviewed()))
        win.bind("<Escape>", lambda e: self._crop_review_close())
        win.protocol("WM_DELETE_WINDOW", self._crop_review_close)

        self._crop_review_render()

    def _crop_review_cancel_pending(self):
        """Cancel every after-callback scheduled by the review window. Used
        on close (and as a defensive sweep before potentially-recursive ops)
        so callbacks can't fire after teardown and touch dead widgets."""
        for attr in ('_crop_review_resize_after',
                     '_crop_review_render_after',
                     '_crop_review_main_after',
                     '_crop_review_jump_after',
                     '_bad_scan_drain_after'):
            pending = getattr(self, attr, None)
            if pending is not None:
                try:
                    self.root.after_cancel(pending)
                except tk.TclError:
                    pass
                setattr(self, attr, None)

    def _crop_review_close(self):
        # Guard against re-entry — Tk can fire WM_DELETE_WINDOW twice in
        # rapid succession if the user clicks the close box while a render
        # is mid-flight, and we don't want the cleanup logic to run twice.
        if getattr(self, '_crop_review_closing', False):
            return
        if self._crop_review_window is None:
            return
        self._crop_review_closing = True
        try:
            win = self._crop_review_window
            self._crop_review_window = None
            # If a background bad-annotation scan is in flight, tell it to
            # stop. The worker thread is daemonized so it won't block exit
            # even if the user closes Python immediately after.
            cancel_evt = getattr(self, '_bad_scan_cancelled', None)
            if cancel_evt is not None:
                cancel_evt.set()
            self._crop_review_cancel_pending()
            dirty = self._crop_review_dirty_paths
            self._crop_review_entries = []
            self._crop_review_canvases = []
            self._crop_review_thumb_cache = {}
            self._crop_review_img_cache = {}
            try:
                win.destroy()
            except Exception:
                pass
            # If the main current image was modified, reload its labels &
            # redraw. Defer one tick so the close finishes first — this
            # keeps the UI responsive when the user mashes Esc.
            if self.current_image_path and str(self.current_image_path) in dirty:
                def _post_close_refresh():
                    self.load_labels()
                    if self.highlight_overlaps:
                        self.detect_current_overlaps()
                    self.display_image()
                self.root.after_idle(_post_close_refresh)
        finally:
            self._crop_review_closing = False

    def _crop_review_get_grid(self):
        try:
            r, c = self._crop_review_grid_var.get().split('x')
            return max(1, int(r)), max(1, int(c))
        except Exception:
            return 4, 6

    def _crop_review_prev_page(self):
        # Flush handled entries lazily — only at page-change time, so the
        # current page stays stable while the user is still working on it.
        self._crop_review_flush_handled()
        if self._crop_review_page > 0:
            self._crop_review_page -= 1
            self._crop_review_render()

    def _crop_review_next_page(self):
        self._crop_review_flush_handled()
        rows, cols = self._crop_review_get_grid()
        per_page = rows * cols
        max_page = max(0, (len(self._crop_review_entries) - 1) // per_page)
        if self._crop_review_page < max_page:
            self._crop_review_page += 1
            self._crop_review_render()
        else:
            # Already on the last page; user pressed Next so flush handled and
            # re-render the same page to compact the view.
            self._crop_review_render()

    def _crop_review_goto_page(self):
        raw = self._crop_review_goto_var.get().strip()
        if not raw:
            return
        try:
            target = int(raw)
        except ValueError:
            self._crop_review_status_var.set(f"Invalid page: {raw!r}")
            return
        rows, cols = self._crop_review_get_grid()
        per_page = rows * cols
        total = len(self._crop_review_entries)
        max_page = max(0, (total - 1) // per_page) if total else 0
        # User-facing pages are 1-indexed; clamp to valid range.
        target_idx = max(0, min(max_page, target - 1))
        self._crop_review_flush_handled()
        self._crop_review_page = target_idx
        self._crop_review_goto_var.set('')
        # Drop focus so subsequent digit hotkeys work for reassign.
        try:
            self._crop_review_window.focus_set()
        except Exception:
            pass
        self._crop_review_render()

    def _crop_review_refresh_class_info(self):
        """Sync the open crop-review window with the current class_info.

        Called after any change to class_info (add / delete / rename / reload).
        Safe no-op if the window isn't open.
        """
        win = self._crop_review_window
        if win is None:
            return
        try:
            if not win.winfo_exists():
                return
        except tk.TclError:
            return

        # In bad-annotation mode there's no single "target" class; entries
        # span every GT class, so skip the target-class-deleted guard below.
        if getattr(self, '_crop_review_mode', 'filter') != 'bad':
            # If the class under review was deleted, the review can't continue.
            if self._crop_review_target not in self.class_info:
                try:
                    self._crop_review_status_var.set(
                        f"Class {self._crop_review_target} was removed — closing review."
                    )
                except Exception:
                    pass
                self._crop_review_close()
                return

        # Rebuild the "Reassign to" dropdown.
        # Setting values via configure() is more reliable across Tk versions
        # than the dict-style ['values'] = ... assignment.
        class_options = [f"{cid}: {info['name']}"
                         for cid, info in sorted(self.class_info.items())]
        try:
            self._crop_review_target_cb.configure(values=class_options)
        except tk.TclError:
            pass

        # The displayed text in the combobox entry is held by the StringVar,
        # which may still contain the OLD "ID: OldName". Reconcile it:
        #   - If the previously-shown class still exists, refresh its name.
        #   - Else fall back to the first option (or empty).
        try:
            current_text = self._crop_review_target_var.get()
        except tk.TclError:
            current_text = ''
        cur_cls_id = None
        if current_text:
            try:
                cur_cls_id = int(current_text.split(':', 1)[0].strip())
            except (ValueError, IndexError):
                cur_cls_id = None
        if cur_cls_id is not None and cur_cls_id in self.class_info:
            new_display = f"{cur_cls_id}: {self.class_info[cur_cls_id]['name']}"
            if new_display != current_text:
                self._crop_review_target_var.set(new_display)
        elif class_options:
            self._crop_review_target_var.set(class_options[0])
        else:
            self._crop_review_target_var.set('')

        # Drop entries whose class no longer exists (rare but possible after delete)
        stale = [e for e in self._crop_review_entries
                 if e['cls_id'] not in self.class_info]
        if stale:
            for e in stale:
                self._crop_review_thumb_cache.pop(id(e), None)
            self._crop_review_entries = [
                e for e in self._crop_review_entries if e['cls_id'] in self.class_info
            ]
            if (self._crop_review_selected is not None
                and self._crop_review_selected >= len(self._crop_review_entries)):
                self._crop_review_selected = None

        # Cheap re-render — image+thumb caches mean only widgets are rebuilt.
        self._crop_review_render()

        # Force the Combobox to redraw so users see the new options immediately
        # rather than waiting for the next idle cycle.
        try:
            self._crop_review_target_cb.update_idletasks()
        except tk.TclError:
            pass

    def _crop_review_flush_handled(self):
        """Remove entries whose current class no longer matches the review
        target. Called only when the user navigates pages — keeps the visible
        page stable during rapid in-place reassigns.

        In bad-annotation mode this is a no-op: entries don't share a single
        target class, so there's nothing to flush by class identity. Bad-mode
        entries are removed individually by reassign / delete / mark-reviewed
        actions instead."""
        if getattr(self, '_crop_review_mode', 'filter') == 'bad':
            return
        target = self._crop_review_target
        before = len(self._crop_review_entries)
        # Free thumb-cache slots for entries being dropped
        keep = []
        for e in self._crop_review_entries:
            if e['cls_id'] == target:
                keep.append(e)
            else:
                self._crop_review_thumb_cache.pop(id(e), None)
        if len(keep) != before:
            self._crop_review_entries = keep

    def _crop_review_load_image(self, ip_str):
        """Persistent LRU cache of cv2.imread results, so paging back and forth
        doesn't trigger fresh disk reads."""
        img = self._crop_review_img_cache.get(ip_str)
        if img is not None:
            # Bump to most-recent by re-inserting
            del self._crop_review_img_cache[ip_str]
            self._crop_review_img_cache[ip_str] = img
            return img
        img = cv2.imread(ip_str)
        if img is None:
            return None
        self._crop_review_img_cache[ip_str] = img
        # LRU eviction (dict preserves insertion order)
        if len(self._crop_review_img_cache) > self._crop_review_img_cache_max:
            oldest = next(iter(self._crop_review_img_cache))
            self._crop_review_img_cache.pop(oldest, None)
        return img

    def _crop_review_build_thumbnail(self, entry, cell_w, cell_h):
        """Return a PhotoImage for this entry, computing once and caching."""
        eid = id(entry)
        cached = self._crop_review_thumb_cache.get(eid)
        if cached is not None:
            return cached
        img = self._crop_review_load_image(str(entry['img_path']))
        if img is None:
            return None
        ih, iw = img.shape[:2]
        cx, cy, bw, bh = entry['bbox_norm']
        x1 = int(round((cx - bw / 2) * iw))
        y1 = int(round((cy - bh / 2) * ih))
        x2 = int(round((cx + bw / 2) * iw))
        y2 = int(round((cy + bh / 2) * ih))
        x1c, y1c = max(0, min(x1, iw)), max(0, min(y1, ih))
        x2c, y2c = max(0, min(x2, iw)), max(0, min(y2, ih))
        if x2c <= x1c or y2c <= y1c:
            return None
        crop = img[y1c:y2c, x1c:x2c]
        ch, cw = crop.shape[:2]
        scale = min(cell_w / cw, cell_h / ch)
        nw, nh = max(1, int(cw * scale)), max(1, int(ch * scale))
        crop_rgb = cv2.cvtColor(
            cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA),
            cv2.COLOR_BGR2RGB
        )
        photo = ImageTk.PhotoImage(image=Image.fromarray(crop_rgb))
        self._crop_review_thumb_cache[eid] = photo
        return photo

    def _crop_review_caption_text(self, entry):
        """Build the per-cell caption.

        In standard (class-filter) mode this shows the file + line, with an
        ``→ NewClass`` suffix once the entry has been reassigned.

        In bad-annotation mode (entry carries ``kind`` / ``pred_cls_id``) it
        shows the GT-vs-prediction diagnosis the user needs to act on, e.g.
        ``img42.jpg #3   GT: sedan → Pred: truck (0.78)`` or
        ``img42.jpg #3   GT: human   (no pred, IoU 0.00)``."""
        base = f"{entry['img_path'].name}  #{entry['line_idx'] + 1}"
        cls_id = entry['cls_id']
        cls_name = self.class_info.get(cls_id, {}).get('name', f'?{cls_id}')

        kind = entry.get('kind')
        if kind in ('class_mismatch', 'missing'):
            if kind == 'class_mismatch':
                pcid = entry.get('pred_cls_id')
                pname = self.class_info.get(pcid, {}).get('name', f'?{pcid}')
                pconf = entry.get('pred_conf', 0.0)
                tail = f"GT: {cls_name} → Pred: {pname} ({pconf:.2f})"
            else:
                iou = entry.get('iou', 0.0)
                tail = f"GT: {cls_name}   (no pred, IoU {iou:.2f})"
            # When the user has reassigned the entry to a different class,
            # also surface the new class so the action bar matches.
            target = self._crop_review_target
            if target is not None and cls_id != target:
                tgt_name = self.class_info.get(target, {}).get('name', f'?{target}')
                tail = f"{tgt_name} ← " + tail
            return f"{base}   {tail}"

        # Default (class-filter) caption path.
        if cls_id != self._crop_review_target:
            text = base + f"   → {cls_name}"
        else:
            text = base
        return text

    def _crop_review_mark_reviewed(self):
        """Bad-annotation mode: drop the currently selected entry from the
        review list WITHOUT changing the label file (the user has decided
        the GT is actually fine and the model is wrong)."""
        if getattr(self, '_crop_review_mode', 'filter') != 'bad':
            return
        sel = self._crop_review_selected
        if sel is None or not (0 <= sel < len(self._crop_review_entries)):
            return
        entry = self._crop_review_entries[sel]
        self._crop_review_status_var.set(
            f"Marked reviewed: {entry['img_path'].name} #{entry['line_idx'] + 1}"
        )
        self._crop_review_thumb_cache.pop(id(entry), None)
        del self._crop_review_entries[sel]
        if sel < len(self._crop_review_entries):
            self._crop_review_selected = sel
        elif self._crop_review_entries:
            self._crop_review_selected = len(self._crop_review_entries) - 1
        else:
            self._crop_review_selected = None
        # Coalesced re-render (same pattern reassign uses)
        prev = getattr(self, '_crop_review_render_after', None)
        if prev is not None:
            try:
                self.root.after_cancel(prev)
            except tk.TclError:
                pass
        def _do_render():
            self._crop_review_render_after = None
            if self._crop_review_window is not None:
                self._crop_review_render()
        self._crop_review_render_after = self.root.after_idle(_do_render)

    def _bad_scan_request_cancel(self):
        """User clicked the Cancel-scan button. Tells the worker to stop on
        its next iteration; entries already in the queue still get drained."""
        if getattr(self, '_bad_scan_cancelled', None) is not None:
            self._bad_scan_cancelled.set()
            try:
                self._crop_review_status_var.set(
                    "Cancel requested — finishing in-flight work…"
                )
            except Exception:
                pass

    def open_bad_label_review(self, config):
        """Open the crop review window in 'bad annotation' mode and kick
        off a background scanner using ``config`` (a dict with keys: paths,
        conf, iou_match, mismatch, missing)."""
        if not self.pretrained_model:
            messagebox.showwarning(
                "No model loaded",
                "Load a YOLO model first (Setup tab → Pretrained Model)."
            )
            return
        # Make sure the group lookup matches whatever class_groups currently is
        self._rebuild_class_group_lookup()
        # Open the window first with an empty entry list — the worker streams
        # results in via the after-loop drainer.
        self._open_crop_review_window(entries=[], target_class=None, mode='bad')
        # Spawn the scanner
        import threading, queue
        self._bad_scan_cancelled = threading.Event()
        self._bad_scan_queue = queue.Queue()
        self._bad_scan_thread = threading.Thread(
            target=self._bad_scan_worker, args=(config,), daemon=True,
        )
        self._bad_scan_thread.start()
        self._crop_review_status_var.set(
            f"Scanning {len(config['paths'])} images — entries appear as they're found…"
        )
        # Start the main-thread drain loop
        self._bad_scan_drain_after = self.root.after(150, self._bad_scan_drain)

    def _crop_review_apply_cell_style(self, cell, entry):
        """Set a cell's background to reflect its current state.
        Priority: selection > handled > normal."""
        target = self._crop_review_target
        sel = self._crop_review_selected
        is_selected = (sel is not None
                       and 0 <= sel < len(self._crop_review_entries)
                       and self._crop_review_entries[sel] is entry)
        # "Handled" only meaningful in filter mode (where there's a single
        # target class). In bad-annotation mode every visible entry is by
        # definition unhandled — handled entries are removed on action.
        if getattr(self, '_crop_review_mode', 'filter') == 'bad':
            is_handled = False
        else:
            is_handled = (entry['cls_id'] != target)
        try:
            if is_selected:
                cell.configure(bg='#FFD166', borderwidth=3)
            elif is_handled:
                cell.configure(bg='#5a3a1a', borderwidth=1)
            else:
                cell.configure(bg='#222', borderwidth=1)
        except tk.TclError:
            pass

    def _crop_review_compute_cell_size(self, rows, cols):
        """Return (cell_w, cell_h) sized so the grid fills the host frame.
        Falls back to a sensible default before the frame is realized."""
        try:
            fw = self._crop_review_grid_frame.winfo_width()
            fh = self._crop_review_grid_frame.winfo_height()
        except tk.TclError:
            fw, fh = 0, 0
        # Account for the 2px grid-cell padding on both sides (padx/pady=2).
        gap = 4
        avail_w = max(0, fw - gap * cols - 4)
        avail_h = max(0, fh - gap * rows - 4)
        cell_w = max(60, avail_w // cols) if cols and avail_w > 60 * cols else 200
        cell_h = max(40, avail_h // rows) if rows and avail_h > 40 * rows else 150
        return cell_w, cell_h

    def _crop_review_on_grid_resize(self, _event=None):
        """Debounced reaction to grid-frame resizing — re-render so cells
        and thumbnails adapt to the new window size."""
        if self._crop_review_window is None:
            return
        rows, cols = self._crop_review_get_grid()
        new_w, new_h = self._crop_review_compute_cell_size(rows, cols)
        # Skip if size didn't actually change (Configure fires on every move too)
        if (new_w, new_h) == self._crop_review_last_cell_size:
            return
        if self._crop_review_resize_after is not None:
            try:
                self.root.after_cancel(self._crop_review_resize_after)
            except tk.TclError:
                pass
        def _do_resize():
            self._crop_review_resize_after = None
            if self._crop_review_window is None:
                return
            # Cell size changed → thumbnails computed at the old size are stale.
            self._crop_review_thumb_cache.clear()
            self._crop_review_render()
        self._crop_review_resize_after = self.root.after(80, _do_resize)

    def _crop_review_render(self):
        # Tear down old cells (we still rebuild widgets, but image data is cached)
        for w in self._crop_review_grid_frame.winfo_children():
            w.destroy()
        self._crop_review_canvases = []

        rows, cols = self._crop_review_get_grid()
        per_page = rows * cols
        total = len(self._crop_review_entries)
        total_pages = max(1, (total + per_page - 1) // per_page) if total else 1
        if self._crop_review_page >= total_pages:
            self._crop_review_page = max(0, total_pages - 1)

        page = self._crop_review_page
        start = page * per_page
        end = min(start + per_page, total)
        if getattr(self, '_crop_review_mode', 'filter') == 'bad':
            head = "Bad annotations (worst first)"
        else:
            cls_name = self.class_info.get(self._crop_review_target, {}).get('name', '?')
            head = f"Class: {cls_name}"
        self._crop_review_title_var.set(
            f"{head}   |   page {page + 1}/{total_pages}   "
            f"|   showing {start + 1 if total else 0}–{end} of {total}"
        )

        # Compute per-cell size from the actual grid frame size so cells
        # adaptively fill the window. All cells share the same size via the
        # 'cells' uniform group + weight=1.
        cell_w, cell_h = self._crop_review_compute_cell_size(rows, cols)
        self._crop_review_last_cell_size = (cell_w, cell_h)
        for r in range(rows):
            self._crop_review_grid_frame.grid_rowconfigure(
                r, weight=1, uniform='cells')
        for c in range(cols):
            self._crop_review_grid_frame.grid_columnconfigure(
                c, weight=1, uniform='cells')

        for r in range(rows):
            for c in range(cols):
                idx = start + r * cols + c
                if idx >= total:
                    break
                entry = self._crop_review_entries[idx]
                cell = tk.Frame(self._crop_review_grid_frame, bg='#222',
                                relief=tk.SUNKEN, borderwidth=1, padx=2, pady=2)
                cell.grid(row=r, column=c, padx=2, pady=2, sticky='nsew')

                photo = self._crop_review_build_thumbnail(entry, cell_w, cell_h)
                cnv = tk.Canvas(cell, width=cell_w, height=cell_h, bg='#111',
                                highlightthickness=0)
                cnv.pack(expand=True)
                if photo is not None:
                    cnv.create_image(cell_w // 2, cell_h // 2,
                                     image=photo, anchor='center')
                    cnv.photo = photo  # keep extra ref alive on the canvas too
                else:
                    cnv.create_text(cell_w // 2, cell_h // 2,
                                    text="(failed to read)", fill='#888')

                caption_lbl = ttk.Label(cell, text=self._crop_review_caption_text(entry),
                                        font=('TkDefaultFont', 8))
                caption_lbl.pack()

                def on_click(_e, ent=entry):
                    try:
                        self._crop_review_selected = self._crop_review_entries.index(ent)
                    except ValueError:
                        return
                    self._crop_review_refresh_action_bar()
                    # Cheap per-cell style update — no widget destruction.
                    for c2, _cnv2, _lbl2, e2 in self._crop_review_canvases:
                        self._crop_review_apply_cell_style(c2, e2)
                    # Jump the main window to this crop's source image
                    self._crop_review_jump_to_entry(ent)
                for widget in (cnv, cell, caption_lbl):
                    widget.bind('<Button-1>', on_click)

                self._crop_review_canvases.append((cell, cnv, caption_lbl, entry))
                self._crop_review_apply_cell_style(cell, entry)

        self._crop_review_refresh_action_bar()

    def _crop_review_refresh_action_bar(self):
        """Update only the status line + dropdown that describe the current
        selection. Called instead of repainting every cell."""
        sel = self._crop_review_selected
        if sel is not None and 0 <= sel < len(self._crop_review_entries):
            entry = self._crop_review_entries[sel]
            cur_name = self.class_info.get(entry['cls_id'], {}).get('name', '?')
            self._crop_review_sel_var.set(
                f"Selected: {entry['img_path'].name} #{entry['line_idx'] + 1} "
                f"({cur_name}) — press 1-9/0 to reassign"
            )
            opt = f"{entry['cls_id']}: {cur_name}"
            if opt in self._crop_review_target_cb['values']:
                self._crop_review_target_var.set(opt)
        else:
            self._crop_review_sel_var.set(
                "Click a crop, then press 1-9/0 to reassign (selection auto-advances). Delete removes. Page nav drops handled."
            )

    # Back-compat alias for the previous name
    def _crop_review_update_selection(self):
        for cell, _cnv, _lbl, entry in self._crop_review_canvases:
            self._crop_review_apply_cell_style(cell, entry)
        self._crop_review_refresh_action_bar()

    def _crop_review_find_cell_for(self, entry):
        for cell, cnv, caption_lbl, ent in self._crop_review_canvases:
            if ent is entry:
                return cell, cnv, caption_lbl
        return None, None, None

    def _crop_review_jump_to_entry(self, entry):
        """Navigate the main window to the entry's source image and select
        its label box. Keeps focus on the review window so digit/arrow
        hotkeys keep working there."""
        if not self.filtered_image_paths:
            return
        target_path = entry.get('img_path')
        if target_path is None:
            return
        # filtered_image_paths holds Path objects — compare by string for safety
        target_str = str(target_path)
        target_idx = None
        for i, p in enumerate(self.filtered_image_paths):
            if str(p) == target_str:
                target_idx = i
                break
        if target_idx is None:
            self._crop_review_status_var.set(
                f"Image not in current filter: {target_path.name}"
            )
            return

        # Defer the (potentially slow) load_image + highlight to after_idle so
        # the click handler returns immediately. This keeps the review window
        # responsive while the main window catches up; rapid clicks coalesce
        # because each new click cancels the previous pending load.
        prev = getattr(self, '_crop_review_jump_after', None)
        if prev is not None:
            try:
                self.root.after_cancel(prev)
            except tk.TclError:
                pass

        def _do_jump(target_idx=target_idx, entry=entry):
            self._crop_review_jump_after = None
            if target_idx != self.current_image_index or self.current_image is None:
                # Clear stale selection before load_image() — display_image()
                # runs inside it and would use a now-invalid box index.
                self.selected_box_index = None
                self.selected_box_source = None
                if self.selected_box_rect:
                    try:
                        self.canvas.delete(self.selected_box_rect)
                    except tk.TclError:
                        pass
                    self.selected_box_rect = None
                self.current_image_index = target_idx
                self.load_image()

            sel_idx = None
            li = entry.get('line_idx')
            if li is not None and 0 <= li < len(self.label_boxes):
                sel_idx = li
            if sel_idx is None and self.current_image is not None:
                cx, cy, bw, bh = entry.get('bbox_norm', (0, 0, 0, 0))
                img_h, img_w = self.current_image.shape[:2]
                tx1 = int((cx - bw / 2) * img_w)
                ty1 = int((cy - bh / 2) * img_h)
                tx2 = int((cx + bw / 2) * img_w)
                ty2 = int((cy + bh / 2) * img_h)
                best_i, best_d = None, None
                for i, (_c, x1, y1, x2, y2) in enumerate(self.label_boxes):
                    d = (abs(x1 - tx1) + abs(y1 - ty1)
                         + abs(x2 - tx2) + abs(y2 - ty2))
                    if best_d is None or d < best_d:
                        best_d = d
                        best_i = i
                sel_idx = best_i

            if sel_idx is not None:
                self.selected_box_index = sel_idx
                self.selected_box_source = 'label'
                self.highlight_selected_box()

        self._crop_review_jump_after = self.root.after_idle(_do_jump)

        # Lift the main window so the user can see the change. We do NOT
        # also lift+focus the review window: forcing focus back triggers a
        # ping-pong with some Linux WMs (Wayland/i3/etc.) and can wedge the
        # event loop. The review window keeps focus because we don't ask
        # for a change — the WM leaves it where it was.
        try:
            self.root.lift()
        except tk.TclError:
            pass

    def _crop_review_advance_selection(self):
        """After a reassign, jump selection to the next on-page cell whose
        entry is still unhandled (cls_id == target). Wraps around the page."""
        if not self._crop_review_canvases:
            self._crop_review_selected = None
            self._crop_review_refresh_action_bar()
            return
        sel = self._crop_review_selected
        cur_entry = (self._crop_review_entries[sel]
                     if sel is not None and 0 <= sel < len(self._crop_review_entries)
                     else None)
        # Find the index of cur_entry within the visible-cells list
        cur_pos = -1
        for i, (_, _, _, ent) in enumerate(self._crop_review_canvases):
            if ent is cur_entry:
                cur_pos = i
                break
        target = self._crop_review_target
        n = len(self._crop_review_canvases)
        for offset in range(1, n + 1):
            pos = (cur_pos + offset) % n
            _, _, _, ent = self._crop_review_canvases[pos]
            if ent['cls_id'] == target:  # not yet handled
                try:
                    self._crop_review_selected = self._crop_review_entries.index(ent)
                except ValueError:
                    continue
                self._crop_review_refresh_action_bar()
                for cell, _cnv, _lbl, e in self._crop_review_canvases:
                    self._crop_review_apply_cell_style(cell, e)
                return
        # Every cell on this page has been reassigned — leave selection and
        # tell the user to advance.
        self._crop_review_status_var.set(
            "Page handled — press → for next page (handled crops will drop)."
        )

    def _crop_review_hotkey(self, digit):
        target = self.class_hotkeys.get(digit)
        if target is None or target not in self.class_info:
            self._crop_review_status_var.set(f"No class bound to '{digit}'")
            return
        if self._crop_review_selected is None:
            self._crop_review_status_var.set("Pick a crop first.")
            return
        self._crop_review_reassign(self._crop_review_selected, target)

    def _crop_review_apply_dropdown(self):
        if self._crop_review_selected is None:
            return
        s = self._crop_review_target_var.get()
        try:
            target = int(s.split(':', 1)[0].strip())
        except (ValueError, IndexError):
            return
        if target in self.class_info:
            self._crop_review_reassign(self._crop_review_selected, target)

    def _crop_review_delete_selected(self):
        """Remove the selected crop's label line from disk AND from the main
        window if it's currently showing the same image."""
        sel = self._crop_review_selected
        if sel is None or not (0 <= sel < len(self._crop_review_entries)):
            self._crop_review_status_var.set("Pick a crop first.")
            return
        entry = self._crop_review_entries[sel]
        deleted_path = entry['img_path']
        deleted_idx = entry['line_idx']
        if not self._delete_label_line(deleted_path, deleted_idx):
            self._crop_review_status_var.set(
                f"FAILED to delete from {deleted_path.name}"
            )
            return
        self._crop_review_dirty_paths.add(str(deleted_path))
        cls_name = self.class_info.get(entry['cls_id'], {}).get('name', '?')
        self._crop_review_status_var.set(
            f"Deleted {deleted_path.name} #{deleted_idx + 1} ({cls_name})"
        )

        # Propagate the delete to the main window if it shows this image.
        self._crop_review_apply_delete_to_main(deleted_path, deleted_idx)

        # Fix up line indices: every remaining entry from the *same* file with
        # a higher line index must shift up by 1 because we deleted a line above.
        for e in self._crop_review_entries:
            if e is entry:
                continue
            if e['img_path'] == deleted_path and e['line_idx'] > deleted_idx:
                e['line_idx'] -= 1
        # Drop the thumbnail from cache before the entry vanishes
        self._crop_review_thumb_cache.pop(id(entry), None)
        self._crop_review_entries.pop(sel)
        # Keep selection anchored to the same on-screen slot when possible.
        if sel < len(self._crop_review_entries):
            self._crop_review_selected = sel
        elif self._crop_review_entries:
            self._crop_review_selected = len(self._crop_review_entries) - 1
        else:
            self._crop_review_selected = None
        # Coalesce render via after_idle so mashing Delete doesn't pile up
        # render passes (same pattern used by reassign).
        prev = getattr(self, '_crop_review_render_after', None)
        if prev is not None:
            try:
                self.root.after_cancel(prev)
            except tk.TclError:
                pass
        def _do_render():
            self._crop_review_render_after = None
            if self._crop_review_window is not None:
                self._crop_review_render()
        self._crop_review_render_after = self.root.after_idle(_do_render)

    def _crop_review_apply_delete_to_main(self, deleted_path, deleted_idx):
        """If the main window currently shows ``deleted_path``, remove the box
        at ``deleted_idx`` from ``label_boxes``, fix up the selection index,
        invalidate the labels cache, and schedule a coalesced redraw."""
        if not self.current_image_path:
            return
        if str(self.current_image_path) != str(deleted_path):
            return
        if not (0 <= deleted_idx < len(self.label_boxes)):
            return
        # Drop the entry; subsequent boxes shift down by one index.
        del self.label_boxes[deleted_idx]
        # Fix up selection on the main canvas.
        if self.selected_box_source == 'label':
            if self.selected_box_index == deleted_idx:
                # The selected box was the one we just removed.
                self.selected_box_index = None
                self.selected_box_source = None
                if self.selected_box_rect:
                    try:
                        self.canvas.delete(self.selected_box_rect)
                    except tk.TclError:
                        pass
                    self.selected_box_rect = None
            elif (self.selected_box_index is not None
                  and self.selected_box_index > deleted_idx):
                # Selected box was below the deleted one; it shifted up by 1.
                self.selected_box_index -= 1
        # Drop any stale labels cache entry for this image.
        try:
            if hasattr(self, 'labels_cache'):
                self.labels_cache.pop(str(self.current_image_path), None)
        except Exception:
            pass
        # Coalesce the main-window redraw (same scheme as reassign).
        prev = getattr(self, '_crop_review_main_after', None)
        if prev is not None:
            try:
                self.root.after_cancel(prev)
            except tk.TclError:
                pass
        def _do_main_redraw():
            self._crop_review_main_after = None
            self.display_image()
            if (self.selected_box_source == 'label'
                    and self.selected_box_index is not None):
                self.highlight_selected_box()
        self._crop_review_main_after = self.root.after_idle(_do_main_redraw)

    def _crop_review_reassign(self, entry_idx, new_class_id):
        """Reassign one crop's class. Updates the label file, removes the
        crop from the review grid immediately, and — if the main window is
        showing this entry's source image — applies the change there too."""
        if not (0 <= entry_idx < len(self._crop_review_entries)):
            return
        entry = self._crop_review_entries[entry_idx]
        old_cls = entry['cls_id']
        if old_cls == new_class_id:
            self._crop_review_status_var.set("Already this class.")
            return
        if not self._rewrite_label_line(entry['img_path'], entry['line_idx'], new_class_id):
            self._crop_review_status_var.set(
                f"FAILED to write label file for {entry['img_path'].name}"
            )
            return
        self._crop_review_dirty_paths.add(str(entry['img_path']))
        entry['cls_id'] = new_class_id
        old_name = self.class_info.get(old_cls, {}).get('name', '?')
        new_name = self.class_info[new_class_id]['name']
        self._crop_review_status_var.set(
            f"Reassigned {entry['img_path'].name} #{entry['line_idx'] + 1}: "
            f"{old_name} → {new_name}"
        )

        # Apply the in-memory change to main-window state synchronously (cheap
        # — just a tuple replacement). The actual canvas redraw is deferred so
        # rapid hotkey tapping doesn't queue multiple display_image calls.
        self._crop_review_apply_change_to_main(entry, new_class_id)

        # Drop the entry from the review list. The next entry will shift up
        # to take this slot; the user keeps tapping without losing position.
        self._crop_review_thumb_cache.pop(id(entry), None)
        del self._crop_review_entries[entry_idx]
        # Keep selection anchored to the same on-screen slot when possible.
        if entry_idx < len(self._crop_review_entries):
            self._crop_review_selected = entry_idx
        elif self._crop_review_entries:
            self._crop_review_selected = len(self._crop_review_entries) - 1
        else:
            self._crop_review_selected = None
        # Coalesce render via after_idle: if the user mashes hotkeys faster
        # than we can draw, only the most-recent state is rendered. Cancel
        # any prior pending render so we don't pile up.
        prev = getattr(self, '_crop_review_render_after', None)
        if prev is not None:
            try:
                self.root.after_cancel(prev)
            except tk.TclError:
                pass
        def _do_render():
            self._crop_review_render_after = None
            if self._crop_review_window is not None:
                self._crop_review_render()
        self._crop_review_render_after = self.root.after_idle(_do_render)

    def _crop_review_apply_change_to_main(self, entry, new_class_id):
        """If the main window currently shows the same image as the reassigned
        crop, mutate its in-memory ``label_boxes``. The canvas redraw is
        deferred via ``after_idle`` so rapid reassigns coalesce into a single
        repaint instead of one per keystroke."""
        if not self.current_image_path:
            return
        if str(self.current_image_path) != str(entry['img_path']):
            return
        li = entry['line_idx']
        if not (0 <= li < len(self.label_boxes)):
            return
        # label_boxes entries are (cls_id, x1, y1, x2, y2); preserve geometry.
        b = self.label_boxes[li]
        self.label_boxes[li] = (new_class_id,) + tuple(b[1:])
        try:
            if hasattr(self, 'labels_cache'):
                self.labels_cache.pop(str(self.current_image_path), None)
        except Exception:
            pass
        # Coalesce the main-window redraw — multiple in-flight reassigns
        # collapse to a single repaint on the next idle.
        prev = getattr(self, '_crop_review_main_after', None)
        if prev is not None:
            try:
                self.root.after_cancel(prev)
            except tk.TclError:
                pass
        def _do_main_redraw(li=li):
            self._crop_review_main_after = None
            self.display_image()
            if (self.selected_box_source == 'label'
                    and self.selected_box_index == li):
                self.highlight_selected_box()
        self._crop_review_main_after = self.root.after_idle(_do_main_redraw)

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
            btn = tk.Button(frame, width=2, bg=rgb_color, relief=tk.RAISED, borderwidth=2)
            btn.configure(command=lambda id=class_id: self.select_class(id))
            btn.pack(side=tk.LEFT, padx=2)
            self.class_buttons[class_id] = btn

            # Clickable hotkey button — shows the current digit (or "—" if unbound).
            # Clicking opens a capture-key dialog so the user can rebind it.
            hk = self._class_to_hotkey(class_id) or "—"
            hk_btn = tk.Button(frame, text=f"[{hk}]", width=3, relief=tk.FLAT,
                               foreground="#444" if hk != "—" else "#999",
                               font=('TkDefaultFont', 8, 'bold'),
                               cursor='hand2',
                               command=lambda cid=class_id: self.assign_hotkey_for_class(cid))
            hk_btn.pack(side=tk.LEFT, padx=(2, 0))

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
        # Apply selection highlight (sunken relief on the active class button)
        self._highlight_selected_class()

    def _highlight_selected_class(self):
        """Visually mark the currently-selected class button."""
        for cls_id, btn in self.class_buttons.items():
            try:
                if cls_id == self.current_class:
                    btn.configure(relief=tk.SUNKEN, borderwidth=4)
                else:
                    btn.configure(relief=tk.RAISED, borderwidth=2)
            except tk.TclError:
                # Button may have been destroyed mid-rebuild — skip silently
                pass

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
            # Snapshot state before accepting prediction
            self._push_history()
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
            # Snapshot state before accepting all
            self._push_history()
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
                    # Optional 'hotkeys' map: {"1": 0, "2": 1, ...}
                    raw_hotkeys = config.get('hotkeys', {})
                    self.class_hotkeys = {
                        str(k): int(v) for k, v in raw_hotkeys.items()
                        if str(k) in '0123456789'
                    }
                    # Optional 'class_groups' map: {"bus": ["bus_m", "bus_l", ...], ...}
                    # Used by Find Bad Annotations to treat all classes in a
                    # group as one when checking class match.
                    self.class_groups = config.get('class_groups', {}) or {}
            else:
                # Create default config if file doesn't exist
                self.class_info = {
                    0: {"name": "Default", "color": [0, 0, 255]}
                }
                self.class_groups = {}
                self.save_class_config()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load class configuration: {str(e)}")
            self.class_info = {
                0: {"name": "Default", "color": [0, 0, 255]}
            }
            self.class_groups = {}
        # Drop any saved bindings for classes that no longer exist, then fill
        # in defaults so 0-9 always have something sensible mapped.
        self._cleanup_hotkeys()
        self._init_default_hotkeys()
        # Re-derive the class-id -> group-key lookup from the (possibly fresh)
        # class_info / class_groups state.
        self._rebuild_class_group_lookup()

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

            # Re-validate hotkey map for the new class list, then fill in defaults
            # for any digits that no longer point anywhere.
            self._cleanup_hotkeys()
            self._init_default_hotkeys()

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
                nonlocal loaded_labels
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
                            loaded_labels = True
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

            # Also load validation/test entries if present, even when train images were found
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
                              f"Successfully loaded:\n" + "\n".join(f"  {part}" for part in msg_parts) +
                              f"\n\nCheck console for detailed folder statistics.")
            
            # Update UI components that depend on classes
            if hasattr(self, 'update_class_filter_dropdown'):
                self.update_class_filter_dropdown()
            if hasattr(self, 'populate_presence_filter_checkboxes'):
                self.populate_presence_filter_checkboxes()
            # Sync any open crop-review window
            self._crop_review_refresh_class_info()

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

    # --- Hotkey management ---
    def _init_default_hotkeys(self):
        """Fill in default key→class bindings for digits not yet assigned.

        Defaults follow the common convention: 1..9 → class 0..8, 0 → class 9.
        Pre-existing user assignments (already in self.class_hotkeys) are
        preserved; we only fill *gaps*.
        """
        used_classes = set(self.class_hotkeys.values())
        defaults = {str(i + 1): i for i in range(9)}
        defaults['0'] = 9
        for digit, cls_id in defaults.items():
            if digit in self.class_hotkeys:
                continue
            if cls_id in used_classes:
                continue
            if cls_id not in self.class_info:
                continue
            self.class_hotkeys[digit] = cls_id
            used_classes.add(cls_id)

    def _cleanup_hotkeys(self):
        """Drop bindings for class ids that no longer exist."""
        for digit in list(self.class_hotkeys.keys()):
            if self.class_hotkeys[digit] not in self.class_info:
                del self.class_hotkeys[digit]

    def _class_to_hotkey(self, class_id):
        """Return the digit currently bound to class_id, or None."""
        for digit, cid in self.class_hotkeys.items():
            if cid == class_id:
                return digit
        return None

    def assign_hotkey_for_class(self, class_id):
        """Modal dialog: capture a 0-9 keypress and bind it to class_id."""
        if class_id not in self.class_info:
            return
        cls_name = self.class_info[class_id]['name']
        current = self._class_to_hotkey(class_id)

        win = tk.Toplevel(self.root)
        win.title("Assign hotkey")
        win.transient(self.root)
        win.resizable(False, False)

        ttk.Label(win, text=f"Hotkey for class:", font=('TkDefaultFont', 10)
                  ).pack(padx=20, pady=(15, 0))
        ttk.Label(win, text=cls_name, font=('TkDefaultFont', 13, 'bold')
                  ).pack(padx=20, pady=(0, 8))
        current_text = f"Current: {current}" if current else "Current: (none)"
        ttk.Label(win, text=current_text, foreground='#0066cc').pack(padx=20, pady=2)
        ttk.Label(win, text="Press 0-9 (top row or num-pad) to bind   |   Backspace to clear   |   Esc to cancel",
                  foreground='#666').pack(padx=20, pady=(10, 15))

        def on_key(event):
            ch = event.keysym
            if ch == 'Escape':
                win.destroy()
                return
            if ch in ('BackSpace', 'Delete'):
                if current and current in self.class_hotkeys:
                    del self.class_hotkeys[current]
                    self.update_status(f"Cleared hotkey for {cls_name}")
                self.save_class_config()
                self.setup_class_buttons()
                win.destroy()
                return
            # Accept both top-row digits ('0'..'9') and numpad ('KP_0'..'KP_9').
            # Normalize to the bare digit character for storage so the main
            # hotkey lookup works the same regardless of input source.
            digit = None
            if len(ch) == 1 and ch in '0123456789':
                digit = ch
            elif ch.startswith('KP_') and len(ch) == 4 and ch[3] in '0123456789':
                digit = ch[3]
            if digit is not None:
                prev_class = self.class_hotkeys.get(digit)
                # Drop any prior binding for THIS class first
                if current and current in self.class_hotkeys:
                    del self.class_hotkeys[current]
                self.class_hotkeys[digit] = class_id
                self.save_class_config()
                self.setup_class_buttons()
                if prev_class is not None and prev_class != class_id:
                    prev_name = self.class_info.get(prev_class, {}).get('name', '?')
                    self.update_status(f"Key '{digit}' → {cls_name} (unbound from {prev_name})")
                else:
                    self.update_status(f"Key '{digit}' → {cls_name}")
                win.destroy()

        win.bind("<Key>", on_key)
        # Capture all keypresses by grabbing focus and modally
        win.update_idletasks()
        # Center on parent
        try:
            pw, ph = self.root.winfo_width(), self.root.winfo_height()
            px, py = self.root.winfo_rootx(), self.root.winfo_rooty()
            ww, wh = win.winfo_width(), win.winfo_height()
            win.geometry(f"+{px + (pw - ww) // 2}+{py + (ph - wh) // 2}")
        except Exception:
            pass
        win.focus_force()
        win.grab_set()

    def save_class_config(self):
        """Save class configuration to JSON file"""
        try:
            config = {
                'classes': {str(k): v for k, v in self.class_info.items()},
                'hotkeys': dict(self.class_hotkeys),
                'class_groups': getattr(self, 'class_groups', {}) or {},
            }
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=4)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save class configuration: {str(e)}")

    def _rebuild_class_group_lookup(self):
        """Derive ``self._class_to_group`` from ``self.class_groups``.

        ``self._class_to_group`` is a {class_id: group_key} dict used at
        comparison time. Classes that don't belong to any group are absent
        from the dict; comparisons fall back to exact-class match for them."""
        name_to_id = {info['name']: cid for cid, info in self.class_info.items()}
        self._class_to_group = {}
        groups = getattr(self, 'class_groups', {}) or {}
        for group_key, members in groups.items():
            if not isinstance(members, (list, tuple)):
                continue
            for m in members:
                # Each member may be a class name (str) or an int id
                if isinstance(m, str):
                    cid = name_to_id.get(m)
                else:
                    try:
                        cid = int(m)
                    except (TypeError, ValueError):
                        cid = None
                if cid is None:
                    continue
                self._class_to_group[cid] = group_key

    def _classes_equivalent(self, cls_a, cls_b):
        """Return True when two class ids should be treated as the same class
        for bad-annotation analysis. Same id always matches; otherwise both
        must belong to the same group key in self.class_groups."""
        if cls_a == cls_b:
            return True
        gmap = getattr(self, '_class_to_group', None) or {}
        ga = gmap.get(int(cls_a))
        gb = gmap.get(int(cls_b))
        return ga is not None and ga == gb

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
                          f" {labels_cached} images with labels\n"
                          f" {dimensions_cached} image dimensions\n"
                          f" Total images scanned: {total_images}\n"
                          f" Time taken: {time_str}\n\n"
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
