import queue
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
from PIL import Image, ImageTk
import numpy as np
import os
from pathlib import Path
import random
import time

from yolo_labeling.ui.crop_review import CropReviewMixin
from yolo_labeling.ui.dialogs import DialogsMixin
from yolo_labeling.ui.session import SessionStatsMixin
from yolo_labeling.ui.stats import StatisticsMixin
from yolo_labeling.ui.batch import BatchViewMixin
from yolo_labeling.ui.filters import FiltersMixin
from yolo_labeling.ui.classes import ClassManagementMixin
from yolo_labeling.core import draw as _draw
from yolo_labeling.core import geometry as _geom
from yolo_labeling.core import labelio as _labelio
from yolo_labeling.core import predcache as _predcache
from yolo_labeling.core import reviewmarks as _reviewmarks
from yolo_labeling.core.classmap import PredictionClassMap as _PredictionClassMap
from yolo_labeling.core.paths import LabelPathResolver as _LabelPathResolver


class YOLOLabelingTool(
    CropReviewMixin,
    DialogsMixin,
    SessionStatsMixin,
    StatisticsMixin,
    BatchViewMixin,
    FiltersMixin,
    ClassManagementMixin,
):
    """The labeling application. Behaviour lives in the mixins above."""

    _HANDLE_CURSORS = {
        'tl': 'top_left_corner', 'br': 'bottom_right_corner',
        'tr': 'top_right_corner', 'bl': 'bottom_left_corner',
        'top': 'sb_v_double_arrow', 'bottom': 'sb_v_double_arrow',
        'left': 'sb_h_double_arrow', 'right': 'sb_h_double_arrow',
    }

    IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.bmp')

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
        # Which class pairs count as an overlap: 'different' (the classic
        # confusion error), 'same' (a duplicate box on one object), or 'any'.
        self.filter_overlap_scope = 'different'
        
        # Multi-class presence/absence filter
        self.filter_by_class_presence_enabled = False
        self.filter_must_have_classes = set()  # Set of class IDs that must be present
        self.filter_must_not_have_classes = set()  # Set of class IDs that must NOT be present

        # Relative-size filter: width/height as a percentage of the image.
        # Distinct from the absolute-area filter above -- area alone cannot
        # separate a full-frame box from a thin wide sliver, and those are
        # different annotation mistakes.
        self.filter_by_relsize_enabled = False
        self.filter_min_w_pct = 0.0
        self.filter_max_w_pct = 100.0
        self.filter_min_h_pct = 0.0
        self.filter_max_h_pct = 100.0

        # Review-state filter. The checkbox is the on/off; the mode picks which
        # side to show. Deliberately NOT a label-based filter -- see
        # apply_filters(), where listing it would drop unlabelled images.
        self.filter_by_review_enabled = False
        self.filter_review_state = 'unreviewed'

        # Reviewed-image marks, opened once the dataset root is known.
        self.dataset_root = None
        self.review_marks = None
        self._review_flush_after = None

        # Crosshair cursor state
        self.crosshair_h = None  # Horizontal line ID
        self.crosshair_v = None  # Vertical line ID
        
        # Cache for faster filtering
        self.image_dimensions_cache = {}  # {img_path: (width, height)}
        self.labels_cache = {}  # {img_path: [(cls_id, x1, y1, x2, y2), ...]}
        # Batch-view thumbnails, keyed by (path, canvas_w, canvas_h). Paging back
        # to a previous batch used to re-decode every tile from disk.
        self._batch_thumb_cache = {}
        # Model-class -> dataset-class translation (see core/classmap.py).
        self.model_names = {}
        self.model_path = None
        self.prediction_class_map = _PredictionClassMap()
        # Confidence floor for assistive predictions. Ultralytics' own default
        # is 0.25; making it ours means it can be tuned and persisted.
        self.prediction_conf = 0.25
        # Ultralytics/torch models are NOT safe to call from two threads at
        # once. "Find Bad Annotations" runs inference on a worker thread while
        # the GUI thread now infers lazily on every image change, so both must
        # take this lock. Waiting costs one inference (~30ms); racing corrupts
        # results or segfaults.
        self._model_lock = threading.Lock()
        self._predict_retry_after = None   # pending lazy-inference retry
        # Durable prediction store, opened when a model is loaded. The
        # in-memory dict above is its hot layer; both hold boxes at
        # predcache.CONF_FLOOR and are narrowed at read time.
        self.pred_cache = None
        self._inference_running = False
        self._inference_drain_after = None
        self._batch_thumb_cache_max = 120
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
        # Captions are opaque plates, so on a small object they can cover the
        # thing you are trying to look at. On by default; Shift+T hides them.
        self.show_box_labels = True
        
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
        self._crop_review_thumb_cache = {}      # id(entry) -> PhotoImage (LRU)
        self._crop_review_thumb_cache_max = 256  # bounded so long sessions don't bloat
        self._crop_review_img_cache = {}        # path-str -> cv2 image (insertion-order LRU)
        self._crop_review_img_cache_max = 32    # raw frames are big (~10–50MB each); keep modest

        # Undo / redo (per-image)
        self.history_undo = []
        self.history_redo = []
        self.max_history = 100

        # Auto-fit on new image load
        self.auto_fit_on_load = True
        self._initial_fit_pending = False

        # Session activity tracking (this-run-only). Populated by the
        # _session_record_* hooks; surfaced in the status bar and the
        # session-stats dialog.
        self._session_start_time = time.time()
        self._session_counts = {
            'created':           {'total': 0, 'by_class': {}},
            'deleted':           {'total': 0, 'by_class': {}},
            'class_reassigned':  {'total': 0, 'by_class': {}},  # keyed by new class
            'geometry_modified': {'total': 0, 'by_class': {}},
        }
        self._session_images_touched = set()
        # Timeline + per-image counters drive pace, sparkline, and hot-list
        # insights. Cap the action log so long sessions don't bloat memory.
        from collections import deque
        self._session_action_log = deque(maxlen=20000)  # (ts, kind, cls_id)
        self._session_per_image = {}                     # {img_path_str: count}

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
                self._session_record_geometry_modified(cls_id, self.current_image_path)
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

        # The Filters tab outgrew its page: eight filter frames plus the Apply
        # footer no longer fit an 800px window, and the IoU frame was already
        # falling off the bottom. Wrap it in a scrolling canvas and REBIND THE
        # NAME, so every later `self.filters_tab` widget packs into the
        # scrollable inner frame with no other line changed. Same canvas +
        # inner-frame + <Configure> idiom as the presence checkboxes below.
        _filters_canvas = tk.Canvas(self.filters_tab, highlightthickness=0, width=250)
        _filters_scroll = ttk.Scrollbar(self.filters_tab, orient="vertical",
                                        command=_filters_canvas.yview)
        _filters_inner = ttk.Frame(_filters_canvas)
        _filters_inner.bind(
            "<Configure>",
            lambda e: _filters_canvas.configure(scrollregion=_filters_canvas.bbox("all")))
        _filters_canvas.create_window((0, 0), window=_filters_inner, anchor="nw")
        _filters_canvas.configure(yscrollcommand=_filters_scroll.set)
        _filters_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        _filters_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.filters_tab = _filters_inner
        
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
        ttk.Button(path_frame, text="Map prediction classes…",
                   command=self.edit_prediction_class_map).pack(fill=tk.X, pady=(2, 2))
        conf_row = ttk.Frame(path_frame)
        conf_row.pack(fill=tk.X, pady=(0, 6))
        ttk.Label(conf_row, text="Prediction conf ≥").pack(side=tk.LEFT)
        self.prediction_conf_var = tk.StringVar(value=f"{self.prediction_conf:.2f}")
        ttk.Spinbox(conf_row, from_=0.01, to=0.95, increment=0.05, width=6,
                    textvariable=self.prediction_conf_var,
                    command=self._on_prediction_conf_change).pack(side=tk.LEFT, padx=6)
        ttk.Button(conf_row, text="Apply",
                   command=self._on_prediction_conf_change).pack(side=tk.LEFT)

        # Inference coverage. Lazy inference means the dataset is only ever
        # partly inferred, and until now nothing said how much -- so there was
        # no way to know whether the next scan would take a minute or an hour.
        infer_frame = ttk.LabelFrame(path_frame, text="Dataset inference")
        infer_frame.pack(fill=tk.X, pady=(4, 6))
        self.inference_coverage_var = tk.StringVar(value="No model loaded")
        ttk.Label(infer_frame, textvariable=self.inference_coverage_var,
                  foreground="#555", font=('TkDefaultFont', 8),
                  wraplength=240, justify=tk.LEFT).pack(fill=tk.X, padx=6, pady=(4, 2))
        self.inference_progress = ttk.Progressbar(
            infer_frame, mode='determinate', maximum=100.0)
        self.inference_progress.pack(fill=tk.X, padx=6, pady=(0, 4))
        infer_btns = ttk.Frame(infer_frame)
        infer_btns.pack(fill=tk.X, padx=6, pady=(0, 5))
        self.inference_run_btn = ttk.Button(
            infer_btns, text="⚡ Infer all", width=12,
            command=self.run_dataset_inference)
        self.inference_run_btn.pack(side=tk.LEFT)
        self.inference_cancel_btn = ttk.Button(
            infer_btns, text="⛔ Stop", width=8, state=tk.DISABLED,
            command=self.cancel_dataset_inference)
        self.inference_cancel_btn.pack(side=tk.LEFT, padx=4)
        ttk.Button(infer_btns, text="Clear", width=7,
                   command=self.clear_prediction_cache).pack(side=tk.RIGHT)

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
        
        # Review state. Packed first because "show me what I haven't gone
        # through yet" is the daily reach.
        filter_review_frame = ttk.LabelFrame(self.filters_tab, text="Filter by Review State")
        filter_review_frame.pack(fill=tk.X, padx=5, pady=5)

        self.filter_review_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(filter_review_frame, text="Filter by review state",
                        variable=self.filter_review_var,
                        command=self.toggle_review_filter).pack(anchor=tk.W, padx=5, pady=2)

        self.filter_review_state_var = tk.StringVar(value='unreviewed')
        review_modes = ttk.Frame(filter_review_frame)
        review_modes.pack(fill=tk.X, padx=5, pady=1)
        for text, value in (("Not reviewed", 'unreviewed'), ("Reviewed", 'reviewed')):
            ttk.Radiobutton(review_modes, text=text, value=value,
                            variable=self.filter_review_state_var,
                            command=self.update_filter).pack(side=tk.LEFT, padx=(0, 8))

        self.review_coverage_var = tk.StringVar(value="No dataset loaded")
        ttk.Label(filter_review_frame, textvariable=self.review_coverage_var,
                  foreground="#555", font=('TkDefaultFont', 8),
                  wraplength=230, justify=tk.LEFT).pack(anchor=tk.W, padx=5, pady=(3, 1))
        ttk.Button(filter_review_frame, text="Clear all marks",
                   command=self.clear_review_marks).pack(anchor=tk.W, padx=5, pady=(0, 4))

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

        # Relative size. Shares this frame with absolute area rather than
        # claiming a seventh LabelFrame -- they are siblings, and area alone
        # cannot tell a full-frame box from a thin wide sliver.
        ttk.Separator(filter_size_frame, orient='horizontal').pack(
            fill=tk.X, padx=5, pady=(6, 2))
        self.filter_relsize_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(filter_size_frame, text="By relative size (% of image)",
                        variable=self.filter_relsize_var,
                        command=self.toggle_relsize_filter).pack(anchor=tk.W, padx=5, pady=2)

        self.min_w_pct_var = tk.StringVar(value="0")
        self.max_w_pct_var = tk.StringVar(value="100")
        self.min_h_pct_var = tk.StringVar(value="0")
        self.max_h_pct_var = tk.StringVar(value="100")

        def _pct_row(label_text, min_var, max_var):
            row = ttk.Frame(filter_size_frame)
            row.pack(fill=tk.X, padx=5, pady=1)
            ttk.Label(row, text=label_text, width=7).pack(side=tk.LEFT)
            for var in (min_var, max_var):
                entry = ttk.Entry(row, textvariable=var, width=5)
                entry.pack(side=tk.LEFT, padx=2)
                entry.bind("<Return>", self.update_filter)
                entry.bind("<FocusOut>", self.update_filter)
            ttk.Label(row, text="%").pack(side=tk.LEFT)

        _pct_row("Width:", self.min_w_pct_var, self.max_w_pct_var)
        _pct_row("Height:", self.min_h_pct_var, self.max_h_pct_var)

        # Overlapping boxes filter (quality control)
        filter_overlap_frame = ttk.LabelFrame(self.filters_tab, text="Filter by Overlapping Labels")
        filter_overlap_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Enable/disable overlap filter
        self.filter_overlap_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(filter_overlap_frame, text="Show images with overlapping boxes",
                       variable=self.filter_overlap_var,
                       command=self.toggle_overlap_filter).pack(anchor=tk.W, padx=5, pady=2)

        # Which pairs count. Same-class overlap is the duplicate-box error
        # (one object labelled twice); different-class is the confusion error.
        # They are worth isolating separately, so this is a scope rather than
        # a single "include same class" checkbox.
        self.filter_overlap_scope_var = tk.StringVar(value='different')
        scope_row = ttk.Frame(filter_overlap_frame)
        scope_row.pack(fill=tk.X, padx=5, pady=1)
        for text, value in (("Diff. class", 'different'),
                            ("Same class", 'same'),
                            ("Any", 'any')):
            ttk.Radiobutton(scope_row, text=text, value=value,
                            variable=self.filter_overlap_scope_var,
                            command=self._on_overlap_scope_change).pack(side=tk.LEFT, padx=(0, 6))

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
        
        self.show_box_text_var = tk.BooleanVar(value=self.show_box_labels)
        ttk.Checkbutton(visibility_frame, text="Show Box Text (Shift+T)",
                       variable=self.show_box_text_var,
                       command=self.toggle_box_text).pack(anchor=tk.W, padx=5, pady=2)

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
            "  Toggle layers: Shift+P / L / U   ·   box text: Shift+T\n"
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
        # Export every label box of the currently filtered images as crops
        # in a folder structure that a Qwen-VL verifier can consume.
        ttk.Button(analyze_btn_frame, text="📤 Export Crops for Qwen…",
                  command=self.show_export_crops_dialog).pack(fill=tk.X, pady=2)
        # Ingest Qwen's verification.json and open a review window showing
        # disagreements between GT and the model's verdict.
        ttk.Button(analyze_btn_frame, text="🤖 Import Qwen Verification…",
                  command=self.show_import_qwen_dialog).pack(fill=tk.X, pady=2)
        
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
        
        # Status bar — split into a left "action message" label and a right
        # "session" mini-summary that stays visible regardless of transient
        # messages. Clicking the session label opens the full stats dialog.
        status_frame = ttk.Frame(self.root, relief=tk.SUNKEN)
        status_frame.pack(side=tk.BOTTOM, fill=tk.X)
        self.session_var = tk.StringVar(value="")
        self.session_label = ttk.Label(status_frame, textvariable=self.session_var,
                                       cursor='hand2', padding=(8, 0))
        self.session_label.pack(side=tk.RIGHT)
        self.session_label.bind('<Button-1>', lambda e: self._session_show_dialog())
        self.status_var = tk.StringVar()
        self.status_bar = ttk.Label(status_frame, textvariable=self.status_var, anchor='w')
        self.status_bar.pack(side=tk.LEFT, fill=tk.X, expand=True)
        # Kick off the session label render + periodic elapsed-time refresh.
        self._session_refresh_label()
        
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
            ("K",                         "Toggle reviewed (whole page in grid view)"),
            ("", ""),
            ("Labels", ""),
            ("1-9, 0",                    "Reassign selected box (mapping is editable in Classes tab)"),
            ("",                          "or set current draw class when nothing is selected"),
            ("S",                         "Save labels"),
            ("A",                         "Accept all VISIBLE model predictions"),
            ("Shift+A",                   "Accept predictions of the current class only"),
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
            ("Shift+T",                   "Toggle box text (hide captions on small objects)"),
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
        # In the grid, `a` accepts across the whole page (with confirmation);
        # in single-image view it accepts on the current image only.
        self.root.bind("<a>", kb(lambda e: (self.accept_batch_predictions()
                                            if self.batch_mode
                                            else self.accept_all_predictions())))
        # Shift+A: only the class currently selected in the Classes panel.
        self.root.bind("<A>",            kb(lambda e: self.accept_predictions_for_current_class()))
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

        # Layer toggles (Shift+P/L/U/T) — synchronises both the BooleanVar and visibility state
        self.root.bind("<P>", kb(lambda e: self._toggle_layer('predictions')))
        self.root.bind("<L>", kb(lambda e: self._toggle_layer('labels')))
        self.root.bind("<U>", kb(lambda e: self._toggle_layer('user')))
        self.root.bind("<T>", kb(lambda e: self._toggle_layer('text')))

        # Selection / image actions
        self.root.bind("<Escape>",       kb(lambda e: self.clear_selections()))
        self.root.bind("<d>",            kb(lambda e: self.delete_current_image()))
        # Reviewed toggle. Context-aware like `a`: whole page in grid view.
        self.root.bind("<k>",            kb(self.toggle_review_current))
        self.root.bind("<K>",            kb(self.toggle_review_current))

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
        elif which == 'text':
            self.show_box_text_var.set(not self.show_box_text_var.get())
            self.toggle_box_text()
            state = "ON" if self.show_box_labels else "OFF"
            self.update_status(f"Box text: {state}")

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

    @classmethod
    def _iter_images(cls, folder):
        """Yield image files in `folder`, case-insensitively.

        The previous lowercase-only globs made `.JPG` files invisible on Linux,
        so a dataset-scope bulk reassign silently skipped them and left the
        dataset with mixed class ids and no warning.
        """
        exts = {e.lower() for e in cls.IMAGE_EXTS}
        for entry in Path(folder).iterdir():
            if entry.is_file() and entry.suffix.lower() in exts:
                yield entry

    def load_images_from_folder(self, folder):
        """Load images from the specified folder"""
        self.image_folder = folder
        if self.image_path_display_var is not None:
            self.image_path_display_var.set(self._shorten_path(folder))
        self.image_paths = sorted(self._iter_images(folder))
        
        # Initialize filtered paths to all paths
        self.filtered_image_paths = self.image_paths.copy()
        
        # Clear caches when loading a new folder. model_predictions_cache was
        # previously left behind, so predictions from the PREVIOUS dataset were
        # shown for any same-named file in the new one.
        self.image_dimensions_cache = {}
        self.labels_cache = {}
        self.model_predictions_cache = {}
        
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
        added = list(self._iter_images(folder_path))
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
            # Imported here rather than at module scope: `from ultralytics
            # import YOLO` drags in torch and costs ~1.5s, which every launch
            # and every test run paid even when no model was ever loaded.
            from ultralytics import YOLO

            self.pretrained_model = YOLO(model_path)
            self.model_path = model_path

            # The model's class list is NOT merged into class_info. A model
            # rarely shares the dataset's taxonomy, and merging it used to
            # RENAME the dataset's own classes (class 2 "bus_m" silently became
            # "car") and persist that to class_config.json -- corrupting the
            # meaning of every label file already on disk.
            #
            # Instead the model keeps its own namespace and a translation table
            # maps its ids onto dataset ids at the point predictions enter.
            self.model_names = dict(getattr(self.pretrained_model, "names", {}) or {})
            self._load_prediction_class_map(model_path)
            # Keyed on this weights file's size+mtime, so a retrained checkpoint
            # at the same path does not serve the old model's predictions.
            self._open_pred_cache(model_path)
            if self.pred_cache is not None:
                n = self.pred_cache.count()
                if n:
                    self.update_status(f"Prediction cache: {n} images already inferred")
            self.update_inference_coverage()

            if self.model_names:
                self.update_status(
                    self.prediction_class_map.summary(self.model_names, self.class_info))

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

            # Reset per-image state — none of it may cross an image boundary.
            # `user_boxes` MUST be cleared here rather than at call sites: this
            # method is the single funnel every navigation path goes through,
            # and only 3 of its 12 callers used to clear it. The other 9 carried
            # the previous image's boxes forward, and the next autosave wrote
            # them into the new image's label file at the old coordinates.
            self.user_boxes = []
            self.selected_box_index = None
            self.selected_box_source = None
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

            # Opening an image counts as going through it. Marked here, in the
            # single navigation funnel, so every one of the twelve callers gets
            # it -- and after the decode, so an unreadable image is never
            # silently marked as reviewed.
            if self.current_image is not None:
                self._mark_reviewed([self.current_image_path])

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
            for i, (cls_id, x1, y1, x2, y2, *_rest) in enumerate(self.model_boxes):
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
                # Show the score: with an unstable model it is the difference
                # between "the box moved" and "the box dropped below cutoff".
                _conf = _rest[0] if _rest else None
                _plabel = self.class_info[cls_id]["name"]
                if _conf is not None:
                    _plabel = f"{_plabel} {float(_conf):.2f}"
                # Above the box. The label layer takes the inside-top slot,
                # so a prediction and the GT box it matches read as two rows
                # rather than one smear.
                if self.show_box_labels:
                    _draw.draw_tag(image_resized, _plabel, zx1, zy1, color[::-1],
                                   0.5 * text_zoom, max(1, int(text_zoom)),
                                   anchor='above')

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
                if self.show_box_labels:
                    _draw.draw_tag(image_resized, label_text, zx1, zy1 + thickness,
                                   color[::-1], 0.5 * text_zoom,
                                   max(1, int(text_zoom)), anchor='below')

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
                thickness = max(2, int(2 * text_zoom))
                cv2.rectangle(image_resized, (zx1, zy1), (zx2, zy2),
                              color[::-1], thickness)
                # Bottom edge: the third distinct slot. A user box drawn over
                # an existing label still shows both captions.
                if self.show_box_labels:
                    _draw.draw_tag(image_resized,
                                   f"U {self.class_info[cls_id]['name']}",
                                   zx1, zy2 - thickness, color[::-1],
                                   0.5 * text_zoom, max(1, int(text_zoom)),
                                   anchor='above')

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

    def _resolver(self):
        """Single source of truth for image -> label-file mapping.

        Replaces the six divergent resolvers that previously let `load_labels`
        and `save_labels` disagree about where a dataset's labels live.
        """
        return _LabelPathResolver(
            label_folder=self.label_folder,
            per_image=getattr(self, 'per_image_label_folders', {}),
        )

    def load_labels(self):
        self.label_boxes = []
        # Source lines + a snapshot of what we parsed, so `save_labels` can tell
        # whether anything actually changed and skip rewriting untouched files.
        self._label_lines = []
        self._label_src_text = None
        self._loaded_boxes = []

        if self.current_image_path is None:
            return
        # An unreadable image gives us no geometry to denormalise against. Bail
        # out rather than proceeding with an empty box list, which used to make
        # save_labels delete a perfectly good label file.
        if self.current_image is None:
            self.update_status(f"Cannot read image: {self.current_image_path.name}")
            return

        label_path = self._resolver().resolve(self.current_image_path)
        if not label_path.is_file():
            return

        # Never raises: malformed lines are retained verbatim, not fatal.
        self._label_lines = _labelio.read_label_file(label_path)
        self._label_src_text = label_path.read_bytes().decode('utf-8', errors='replace')

        img_height, img_width = self.current_image.shape[:2]
        cached_labels = []
        for ln in self._label_lines:
            if not ln.parsed:
                continue
            x_center, y_center, w, h = ln.bbox
            # round() rather than int(): truncation floored both edges, shrinking
            # every box by up to a pixel on each load/save cycle. Clipping keeps
            # out-of-range labels inside the image.
            x1, y1, x2, y2 = _geom.yolo_to_pixel(
                x_center, y_center, w, h, img_width, img_height)
            self.label_boxes.append((float(ln.cls_id), x1, y1, x2, y2))
            cached_labels.append((float(ln.cls_id), x1, y1, x2, y2, (x2 - x1) * (y2 - y1)))

        self._loaded_boxes = list(self.label_boxes)
        self.labels_cache[str(self.current_image_path)] = cached_labels

    def _on_prediction_conf_change(self, *args):
        """Apply a new confidence threshold.

        Predictions are cached at `predcache.CONF_FLOOR`, well below anything
        the slider offers, so this is a re-filter of data already in hand. It
        used to empty the cache and re-infer, which on a 46k-image dataset made
        the threshold effectively un-tunable.
        """
        try:
            value = float(self.prediction_conf_var.get())
        except (TypeError, ValueError):
            self.update_status("Prediction conf must be a number between 0.01 and 0.95")
            return
        value = min(max(value, 0.01), 0.95)
        self.prediction_conf_var.set(f"{value:.2f}")
        if value == self.prediction_conf:
            return
        self.prediction_conf = value
        if self.current_image_path:
            self.load_model_predictions()
            self.display_image()
        self.update_status(f"Prediction confidence threshold: {value:.2f}")

    def _predict_image(self, img, blocking=True):
        """Run the loaded model on one decoded image -> [(cls, x1, y1, x2, y2)].

        Args:
            blocking: When False, return ``None`` immediately if another thread
                (the bad-annotation scan) owns the model rather than waiting for
                it. Callers on the GUI thread MUST pass False: Tk's event loop
                lives on that thread, so waiting here freezes the window --
                including the scan's own progress bar, which is driven by an
                `after()` callback on this very loop. Locks are not fair
                either, so a blocked GUI thread loses the race repeatedly and
                the freeze outlasts a single inference by a wide margin.
        """
        preds = []
        if self.pretrained_model is None or img is None:
            return preds
        if not self._model_lock.acquire(blocking=blocking):
            return None
        try:
            # Without an explicit conf, ultralytics applies its own default of
            # 0.25. Making it explicit means the user controls which assistive
            # predictions appear -- important when a model's confidence is
            # unstable, since a fixed hidden cutoff makes boxes flicker in and
            # out between near-identical frames.
            # Infer at the cache floor, NOT at the display threshold: one
            # result then serves every consumer, and moving the slider
            # re-filters instead of re-inferring.
            results = self.pretrained_model(img, verbose=False,
                                            conf=_predcache.CONF_FLOOR)
        except Exception as e:
            self.update_status(f"Prediction failed: {e}")
            return []
        finally:
            self._model_lock.release()
        try:
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].cpu().numpy()) if box.conf is not None else 0.0
                    preds.append((int(box.cls[0].cpu().numpy()), x1, y1, x2, y2, conf))
        except Exception as e:
            self.update_status(f"Prediction failed: {e}")
            return []
        # Translate model class ids to dataset class ids HERE, once. Everything
        # downstream -- colours, class visibility, filters, accept, statistics --
        # then only ever deals in dataset ids.
        # Deliberately NOT translated here. The cache stores what the model
        # said; class mapping is a read-side view, exactly like the confidence
        # threshold. That way editing the map re-derives from cache instead of
        # re-inferring -- and the bad-annotation scan can still tell an UNMAPPED
        # prediction (inconclusive) from no prediction at all, a distinction
        # `PredictionClassMap.apply` erases by dropping unmapped boxes.
        return _predcache.clip_boxes(preds)

    def load_model_predictions(self):
        """Predictions for the current image, computed on demand.

        This used to rely on a full-dataset inference pass run from __init__
        BEFORE mainloop() started -- a dead, uncancellable window of 14 minutes
        to 3.5 hours on 50k images. Inferring one image when it is actually
        displayed costs ~30ms and caches the result.
        """
        self.model_boxes = []
        if not self.current_image_path:
            return
        key = str(self.current_image_path)
        preds = self._cached_predictions(self.current_image_path,
                                         self.current_image, blocking=False)
        if preds is None:
            # A background scan owns the model. Come back for it instead of
            # parking the event loop.
            if self.pretrained_model is not None and self.current_image is not None:
                self._schedule_prediction_retry(key)
                self.update_status("Predictions deferred — scan is using the model")
            return
        self.model_boxes = self._display_predictions(self.current_image_path)

    def _peek_cached_predictions(self, img_path):
        """Raw cached predictions for `img_path`, or None if we do not have them.

        Deliberately cheap: no decode, no inference, no model lock. Callers that
        would otherwise decode an image just to discover they already had its
        predictions ask here first.

        That matters more than it sounds. Measured on pvn-vehicle frames, a
        decode is 3.4ms against 0.018ms for the SQLite lookup -- so decoding
        every frame to reach data already in the cache costs ~6 minutes across
        46k images, which from the outside is indistinguishable from
        re-running the model.
        """
        key = str(img_path)
        hit = self.model_predictions_cache.get(key)
        if hit is not None:
            return hit
        if self.pred_cache is not None:
            hit = self.pred_cache.get(img_path)
            if hit is not None:
                self.model_predictions_cache[key] = hit    # promote to the hot layer
                return hit
        return None

    def _cached_predictions(self, img_path, img=None, blocking=True):
        """Predictions for one image AT THE CACHE FLOOR, from the cheapest source.

        Memory, then the durable store, then the model. Callers narrow the
        result to their own threshold with `predcache.filter_conf` -- the scan
        at 0.40 and the canvas at 0.25 share one entry this way, so a second
        scan costs no inference at all.

        Returns None only when `blocking=False` and the model is busy.
        """
        hit = self._peek_cached_predictions(img_path)
        if hit is not None:
            return hit

        key = str(img_path)
        if self.pretrained_model is None:
            return []
        if img is None:
            img = cv2.imread(key)
        if img is None:
            return []
        preds = self._predict_image(img, blocking=blocking)
        if preds is None:
            return None
        self.model_predictions_cache[key] = preds
        if self.pred_cache is not None:
            try:
                self.pred_cache.put(img_path, preds, commit=False)
            except Exception:
                pass          # a cache failure must never break labelling
        return preds

    def _display_predictions(self, img_path, conf=None):
        """Cached predictions as DATASET ids, narrowed to `conf`.

        The single read-side view over the raw cache: translate, then threshold.
        Every consumer that shows or accepts predictions goes through here, so
        none of them ever sees a model id.
        """
        raw = self.model_predictions_cache.get(str(img_path), [])
        if conf is None:
            conf = getattr(self, 'prediction_conf', None)
        return self.prediction_class_map.apply(_predcache.filter_conf(raw, conf))

    def _open_pred_cache(self, model_path):
        """(Re)open the durable store for `model_path`."""
        if self.pred_cache is not None:
            self.pred_cache.close()
            self.pred_cache = None
        self.model_predictions_cache = {}
        if not model_path:
            return
        try:
            self.pred_cache = _predcache.PredictionCache(
                Path(".yolo_labeling_cache") / "predictions.sqlite", model_path)
        except Exception as e:
            self.pred_cache = None
            self.update_status(f"Prediction cache unavailable: {e}")

    # ------------------------------------------------------------------
    # reviewed-image marks
    # ------------------------------------------------------------------

    def _review_root(self):
        """Where review marks live. Falls back until something is usable.

        The dataset root is the point of the whole storage choice, so prefer it;
        an image folder still gives marks that travel with the images; CWD is
        the last resort and behaves like `class_config.json`.
        """
        if self.dataset_root:
            return Path(self.dataset_root)
        if getattr(self, 'image_folder', None):
            return Path(self.image_folder).resolve().parent
        return Path.cwd()

    def _open_review_marks(self):
        """(Re)open the review store for the current dataset root."""
        self.flush_review_marks()
        try:
            root = self._review_root()
            self.review_marks = _reviewmarks.ReviewMarks(
                root / _reviewmarks.FILENAME, root)
        except Exception as e:
            self.review_marks = None
            self.update_status(f"Review marks unavailable: {e}")
            return
        self.update_review_coverage()

    # -- marking ---------------------------------------------------------

    def _mark_reviewed(self, paths):
        """Mark `paths` reviewed and refresh anything that shows the state."""
        if self.review_marks is None or not paths:
            return 0
        n = self.review_marks.mark_many(paths)
        if n:
            self._schedule_review_flush()
            self.update_review_coverage()
            self.update_status()
        return n

    def toggle_review_current(self, event=None):
        """`k`: flip reviewed state. Whole page in grid view, else this image.

        Context-aware in the same way `a` is, so the key means the same thing
        in both views: "act on what I am looking at".
        """
        if self.review_marks is None:
            self.update_status("No dataset loaded — nothing to mark")
            return
        if getattr(self, 'batch_mode', False):
            paths = self._batch_visible_paths()
            if not paths:
                return
            # Mark the page unless it is already fully marked, then clear it.
            # Showing a page marks it, so in practice this key is mostly used
            # to take a page back off the reviewed list.
            clearing = all(self.review_marks.is_marked(p) for p in paths)
            for p in paths:
                self.review_marks.unmark(p) if clearing else self.review_marks.mark(p)
            self.update_status(
                f"Page marked {'unreviewed' if clearing else 'reviewed'} "
                f"({len(paths)} images)")
            self._schedule_review_flush()
            self.update_review_coverage()
            # Repaint the ticks WITHOUT re-marking: a plain repaint would
            # instantly undo the clear we just did.
            self.load_batch_images(mark_reviewed=not clearing)
            return

        if not self.current_image_path:
            return
        now = self.review_marks.toggle(self.current_image_path)
        self._schedule_review_flush()
        self.update_review_coverage()
        self.update_status(f"Marked {'reviewed' if now else 'unreviewed'}")

    def is_reviewed(self, img_path):
        return bool(self.review_marks and self.review_marks.is_marked(img_path))

    # -- persistence -----------------------------------------------------

    def _schedule_review_flush(self, delay=2000):
        """Debounce the write. One pass over 46k images is not 46k writes."""
        root = getattr(self, 'root', None)
        if root is None:
            return
        if self._review_flush_after is not None:
            try:
                root.after_cancel(self._review_flush_after)
            except tk.TclError:
                pass
        self._review_flush_after = root.after(delay, self.flush_review_marks)

    def flush_review_marks(self):
        """Write pending marks now."""
        self._review_flush_after = None
        if self.review_marks is None:
            return
        try:
            self.review_marks.save()
        except Exception as e:
            self.update_status(f"Could not save review marks: {e}")

    def clear_review_marks(self):
        """Discard every mark for this dataset."""
        if self.review_marks is None:
            return
        n = self.review_marks.count()
        if not n:
            self.update_status("No review marks to clear")
            return
        if not messagebox.askyesno(
                "Clear review marks",
                f"Forget that {n:,} image(s) were reviewed?\n\n"
                "This cannot be undone."):
            return
        self.review_marks.clear()
        self.flush_review_marks()
        self.update_review_coverage()
        self.update_status(f"Cleared {n:,} review marks")

    def update_review_coverage(self):
        """Refresh the 'N of M reviewed' readout on the Filters tab."""
        if not hasattr(self, 'review_coverage_var'):
            return
        total = len(self.image_paths or [])
        if self.review_marks is None:
            self.review_coverage_var.set("No dataset loaded")
            return
        done = self.review_marks.count()
        pct = (100.0 * min(done, total) / total) if total else 0.0
        self.review_coverage_var.set(
            f"{done:,} of {total:,} images reviewed ({pct:.0f}%)")

    def _schedule_prediction_retry(self, key, delay=400):
        """Re-attempt lazy inference for `key` once the model is free.

        Only one retry is ever pending: navigating away cancels it, so flicking
        through images during a scan cannot pile up a backlog of stale work.
        """
        prev = getattr(self, '_predict_retry_after', None)
        if prev is not None:
            try:
                self.root.after_cancel(prev)
            except tk.TclError:
                pass

        def _retry():
            self._predict_retry_after = None
            if str(self.current_image_path) != key:
                return                       # user moved on; nothing to do
            if key in self.model_predictions_cache:
                return
            preds = self._cached_predictions(self.current_image_path,
                                             self.current_image, blocking=False)
            if preds is None:
                self._schedule_prediction_retry(key, delay)
                return
            self.model_boxes = self._display_predictions(self.current_image_path)
            self.display_image()

        self._predict_retry_after = self.root.after(delay, _retry)

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
            _kept_cls = self.user_boxes[self.selected_box_index][0]
            self.user_boxes[self.selected_box_index] = (
                _kept_cls,  # Keep the same class
                x1, y1, x2, y2
            )
            self.editing_box = False
            self._session_record_geometry_modified(_kept_cls, self.current_image_path)
        else:
            # Snapshot state before adding new box
            self._push_history()
            # Add new box
            self.user_boxes.append((self.current_class, x1, y1, x2, y2))
            self._session_record_created(self.current_class, self.current_image_path)

        self.drawing = False

        # Automatically save labels after adding/editing box
        self.save_labels()
        
        # Re-detect overlaps if highlight mode is on (new/edited box might create overlaps)
        if self.highlight_overlaps:
            self.detect_current_overlaps()
        
        self.display_image()
        self.update_status(f"Box size: {int(width)}x{int(height)} pixels")

    def save_labels(self, event=None):
        """Save labels to file. Atomic, and a no-op when nothing changed."""
        if not self.current_image_path:
            messagebox.showerror("Error", "No image loaded")
            return
        if self.current_image is None:
            # No geometry to normalise against -- refuse rather than destroy.
            self.update_status("Cannot save: image could not be read")
            return

        label_path = self._resolver().resolve(self.current_image_path)
        all_boxes = self.label_boxes + self.user_boxes

        # Nothing was edited: leave the file exactly as it is. This is what
        # preserves segmentation polygons, confidence columns, CRLF terminators
        # and float precision when the user merely opens an image and hits `s`.
        if not self.user_boxes and all_boxes == getattr(self, '_loaded_boxes', None):
            self.update_status("No changes to save")
            return

        if not all_boxes:
            # A zero-byte .txt is a valid negative sample; deleting it is not
            # recoverable, so write an empty file instead.
            try:
                _labelio.write_label_file(label_path, "")
                self.labels_cache[str(self.current_image_path)] = []
                self._loaded_boxes = []
                self._mark_reviewed([self.current_image_path])
                self.update_status("All boxes removed; label file is now empty")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to write label file: {str(e)}")
            return

        img_height, img_width = self.current_image.shape[:2]
        out_lines = []
        cached_labels = []
        for cls_id, x1, y1, x2, y2 in all_boxes:
            # Clamp to [0, 1]: a paste across differently-sized images could
            # otherwise write values like 2.73 straight into the label file.
            x_center, y_center, width, height = _geom.pixel_to_yolo(
                x1, y1, x2, y2, img_width, img_height)
            out_lines.append(_labelio.LabelLine(
                int(cls_id), (x_center, y_center, width, height), "", dirty=True))
            cached_labels.append((cls_id, x1, y1, x2, y2, (x2 - x1) * (y2 - y1)))

        try:
            _labelio.write_label_file(label_path, out_lines)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save labels: {str(e)}")
            return
        # Only trust the cache once the bytes are actually on disk.
        self.labels_cache[str(self.current_image_path)] = cached_labels
        self._loaded_boxes = list(all_boxes)
        # Editing an image is reviewing it, so skip the dwell entirely.
        self._mark_reviewed([self.current_image_path])
        self.update_status(f"Labels saved to {label_path}")

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
            for i, (cls_id, x1, y1, x2, y2, *_rest) in enumerate(self.model_boxes):
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
                _cls = self.user_boxes[self.selected_box_index][0]
                del self.user_boxes[self.selected_box_index]
                self._session_record_deleted(_cls, self.current_image_path)
                self.update_status("User box deleted")
            elif self.selected_box_source == 'label':
                _cls = self.label_boxes[self.selected_box_index][0]
                del self.label_boxes[self.selected_box_index]
                self._session_record_deleted(_cls, self.current_image_path)
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
            if self.current_image_path is not None and self.review_marks is not None:
                status.append("✓ reviewed" if self.is_reviewed(self.current_image_path)
                              else "unreviewed")
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

    def toggle_box_text(self):
        """Toggle the class-name captions on every box layer.

        Unlike the other layer toggles this refreshes the batch grid too: tiles
        are heavily downscaled, so that is exactly where a caption is most
        likely to be bigger than the object it names.
        """
        self.show_box_labels = self.show_box_text_var.get()
        if getattr(self, 'batch_mode', False):
            self.load_batch_images()
        else:
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

        # Coalesce the expensive redraw the same way scrolling already does.
        # The zoom maths above stays synchronous so the cursor anchor and the
        # scrollbars track exactly; only display_image() (35-90ms: cv2.resize +
        # cvtColor + PhotoImage + Tk paint) is deferred. A 15-notch wheel flick
        # used to queue 15 full redraws and kept zooming for ~1s after the user
        # stopped; now it renders once, at the end.
        self._schedule_scroll_redraw()
        self.update_status(f"Zoom: {self.zoom_scale:.2f}x")

    def canvas_to_image_coords(self, x, y):
        """Convert canvas coordinates to image coordinates"""
        return int(x / self.zoom_scale), int(y / self.zoom_scale)

    def image_to_canvas_coords(self, x, y):
        """Convert image coordinates to canvas coordinates"""
        return x * self.zoom_scale, y * self.zoom_scale

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
        """Indices of boxes that overlap another box, per `filter_overlap_scope`.

        Uses a set for membership: the previous version appended to a list and
        tested `if i not in self.overlapping_boxes` inside the inner loop, making
        it O(b^3). With 100 mutually-overlapping boxes that was ~1M operations
        on every image load, box edit, paste and undo.
        """
        self.overlapping_boxes = []
        if not self.label_boxes:
            return

        boxes = [b[:5] for b in self.label_boxes if len(b) >= 5]
        threshold = self.filter_overlap_threshold
        hits = set()
        for i in range(len(boxes)):
            cls1, ax1, ay1, ax2, ay2 = boxes[i]
            for j in range(i + 1, len(boxes)):
                cls2, bx1, by1, bx2, by2 = boxes[j]
                if not self.overlap_pair_counts(cls1, cls2):
                    continue
                if _geom.iou_xyxy((ax1, ay1, ax2, ay2), (bx1, by1, bx2, by2)) >= threshold:
                    hits.add(i)
                    hits.add(j)
        self.overlapping_boxes = sorted(hits)

    def update_counter(self):
        """Update the image counter display"""
        if self.filtered_image_paths:
            current = self.current_image_index + 1
            total = len(self.filtered_image_paths)
            self.counter_var.set(f"Image: {current}/{total}")
        else:
            self.counter_var.set("Image: 0/0")

    def _read_reduced(self, path_str, target_w, target_h):
        """Decode an image no larger than needed, using libjpeg's scaled decode."""
        size = self._image_size(path_str)
        flag = cv2.IMREAD_COLOR
        if size:
            w, h = size
            ratio = min(w / max(target_w, 1), h / max(target_h, 1))
            if ratio >= 8:
                flag = cv2.IMREAD_REDUCED_COLOR_8
            elif ratio >= 4:
                flag = cv2.IMREAD_REDUCED_COLOR_4
            elif ratio >= 2:
                flag = cv2.IMREAD_REDUCED_COLOR_2
        return cv2.imread(path_str, flag)

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
        
        # Drop its review mark too, or the key accumulates in the store forever.
        if self.review_marks is not None and self.review_marks.unmark(deleted_path):
            self._schedule_review_flush()

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
        import threading
        import queue
        self._bad_scan_cancelled = threading.Event()
        self._bad_scan_queue = queue.Queue()
        self._bad_scan_started_at = time.time()      # drives the ETA readout
        self._bad_scan_thread = threading.Thread(
            target=self._bad_scan_worker, args=(config,), daemon=True,
        )
        self._bad_scan_thread.start()
        self._crop_review_status_var.set(
            f"Scanning {len(config['paths'])} images — entries appear as they're found…"
        )
        # Start the main-thread drain loop
        self._bad_scan_drain_after = self.root.after(150, self._bad_scan_drain)

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

    def highlight_selected_prediction(self):
        """Draw highlight around selected prediction"""
        if self.selected_prediction is not None:
            cls_id, x1, y1, x2, y2 = self.model_boxes[self.selected_prediction][:5]
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
            cls_id, x1, y1, x2, y2 = self.model_boxes[self.selected_prediction][:5]
            
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

    def accept_predictions_for_current_class(self, event=None):
        """Accept only the predictions matching the currently selected class."""
        self.accept_all_predictions(cls_id=self.current_class)

    def accept_all_predictions(self, event=None, cls_id=None):
        """Accept model predictions in bulk.

        Args:
            cls_id: When given, accept only predictions of that dataset class.
                When None, accept every prediction whose class is currently
                VISIBLE in the Classes panel.

        Visibility is honoured because predictions of hidden classes are not
        drawn (see display_image). Accepting boxes the user cannot see -- which
        is what this did before -- writes labels they never had a chance to
        review. Hiding a class is therefore also how you exclude it here.
        """
        if not self.model_boxes:
            return

        def wanted(c):
            if c not in self.class_info:
                return False
            if cls_id is not None:
                return c == int(cls_id)
            return self.class_visibility.get(c, True)

        chosen = []
        for box in self.model_boxes:
            c = box[0]
            c = int(c.item()) if hasattr(c, 'item') else int(c)
            if wanted(c):
                chosen.append((c, *box[1:5]))

        if not chosen:
            what = (f"class '{self.class_info.get(int(cls_id), {}).get('name', cls_id)}'"
                    if cls_id is not None else "any visible class")
            self.update_status(f"No predictions to accept for {what}")
            return

        self._push_history()
        self.user_boxes.extend(chosen)
        skipped = len(self.model_boxes) - len(chosen)

        self.clear_selections()

        # Re-detect overlaps if highlight mode is on (multiple boxes added)
        if self.highlight_overlaps:
            self.detect_current_overlaps()

        # Automatically save labels after accepting predictions
        self.save_labels()
        self.display_image()

        note = f" ({skipped} skipped: hidden or filtered out)" if skipped else ""
        self.update_status(f"Accepted {len(chosen)} predictions{note}")

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
            
            # NOTE: a periodic gc.collect(2) used to run here, "to prevent
            # cumulative slowdown". It caused the slowdown it claimed to fix:
            # ~370 full gen-2 collections over 50k images, each walking a heap
            # that the cache itself keeps growing. Removed.

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

    @staticmethod
    def calculate_iou(boxA, boxB):
        """Delegates to core.geometry.iou_xyxy."""
        return _geom.iou_xyxy(boxA, boxB)

    def clear_prediction_cache(self):
        """Discard every cached prediction for the loaded model."""
        if self.pred_cache is None:
            self.update_status("No prediction cache to clear")
            return
        n = self.pred_cache.count()
        if not n:
            self.update_status("Prediction cache is already empty")
            return
        if not messagebox.askyesno(
                "Clear prediction cache",
                f"Discard {n:,} cached prediction(s) for this model?\n\n"
                "They will be re-inferred on demand."):
            return
        self.pred_cache.clear()
        self.model_predictions_cache = {}
        self.update_status(f"Cleared {n:,} cached predictions")
        self.update_inference_coverage()

    # ------------------------------------------------------------------
    # whole-dataset inference
    # ------------------------------------------------------------------

    def _inference_scope(self):
        """Images a full pass would cover: the filtered set, else everything."""
        return list(self.filtered_image_paths or self.image_paths or [])

    def update_inference_coverage(self, note=""):
        """Refresh the 'N of M inferred' readout.

        Counts rows rather than stat-ing every file: verifying 46k images
        against their mtimes would cost more than the readout is worth. The
        number is therefore 'cached for this model', which can exceed the
        current filter -- worded accordingly.
        """
        if not hasattr(self, 'inference_coverage_var'):
            return
        total = len(self._inference_scope())
        if self.pretrained_model is None or self.pred_cache is None:
            self.inference_coverage_var.set(
                "No model loaded" if self.pretrained_model is None
                else f"0 of {total} images inferred (cache unavailable)")
            return
        try:
            done = self.pred_cache.count()
        except Exception:
            done = 0
        pct = (100.0 * min(done, total) / total) if total else 0.0
        self.inference_coverage_var.set(
            f"{done:,} of {total:,} images inferred for this model ({pct:.0f}%)"
            + (f"  ·  {note}" if note else ""))
        if not getattr(self, '_inference_running', False):
            try:
                self.inference_progress['value'] = pct
            except tk.TclError:
                pass

    def run_dataset_inference(self):
        """Infer every image in scope, skipping anything already cached.

        Runs on a worker thread. The old version of this looped on the GUI
        thread with `update_idletasks()`, which animates a progress bar on a
        window that cannot accept input -- for hours.
        """
        if self.pretrained_model is None:
            messagebox.showwarning("No model",
                                   "Load a YOLO model first (Setup tab → Model).")
            return
        if getattr(self, '_inference_running', False):
            self.update_status("Inference already running")
            return
        paths = self._inference_scope()
        if not paths:
            messagebox.showinfo("No images", "Load images first.")
            return
        if not messagebox.askyesno(
                "Infer whole dataset",
                f"Run the model over {len(paths):,} image(s)?\n\n"
                "Already-cached images are skipped. You can stop at any time "
                "and keep the work done so far."):
            return

        self._inference_running = True
        self._inference_cancel = threading.Event()
        self._inference_queue = queue.Queue()
        self.inference_run_btn.configure(state=tk.DISABLED)
        self.inference_cancel_btn.configure(state=tk.NORMAL)
        self._inference_started_at = time.time()
        threading.Thread(target=self._inference_worker, args=(paths,),
                         daemon=True).start()
        self._inference_drain_after = self.root.after(200, self._inference_drain)

    def cancel_dataset_inference(self):
        evt = getattr(self, '_inference_cancel', None)
        if evt is not None:
            evt.set()
            self.update_status("Stopping inference — finishing in-flight work…")

    def _inference_worker(self, paths):
        """Worker thread: fill the cache. Touches no Tk widget."""
        done = skipped = failed = 0
        try:
            for i, path in enumerate(paths):
                if self._inference_cancel.is_set():
                    break
                try:
                    if self.pred_cache is not None and self.pred_cache.has(path):
                        skipped += 1
                    else:
                        img = cv2.imread(str(path))
                        if img is None:
                            failed += 1
                        else:
                            self._cached_predictions(path, img, blocking=True)
                            done += 1
                except Exception:
                    failed += 1
                if (i + 1) % 50 == 0 or i == len(paths) - 1:
                    if self.pred_cache is not None:
                        try:
                            self.pred_cache.commit()
                        except Exception:
                            pass
                    self._inference_queue.put(
                        ("progress", i + 1, len(paths), done, skipped, failed))
        finally:
            if self.pred_cache is not None:
                try:
                    self.pred_cache.commit()
                except Exception:
                    pass
            reason = "cancelled" if self._inference_cancel.is_set() else "complete"
            self._inference_queue.put(("done", reason, done, skipped, failed))

    def _inference_drain(self):
        """Main thread: render worker progress."""
        last = None
        finished = None
        while True:
            try:
                msg = self._inference_queue.get_nowait()
            except queue.Empty:
                break
            if msg[0] == 'progress':
                last = msg
            else:
                finished = msg

        if last is not None:
            _, i, total, done, skipped, failed = last
            pct = 100.0 * i / total if total else 0.0
            try:
                self.inference_progress['value'] = pct
            except tk.TclError:
                pass
            elapsed = time.time() - self._inference_started_at
            eta = ""
            if done and elapsed > 1.0:
                rate = done / elapsed
                eta = f"  ETA {self._fmt_eta((total - i) / rate)}" if rate else ""
            self.inference_coverage_var.set(
                f"{i:,}/{total:,}  ·  new {done:,}  cached {skipped:,}"
                f"{f'  failed {failed:,}' if failed else ''}{eta}")

        if finished is not None:
            _, reason, done, skipped, failed = finished
            self._inference_running = False
            self._inference_drain_after = None
            try:
                self.inference_run_btn.configure(state=tk.NORMAL)
                self.inference_cancel_btn.configure(state=tk.DISABLED)
            except tk.TclError:
                pass
            mark = "✓" if reason == "complete" else "⛔"
            self.update_status(
                f"{mark} Inference {reason}: {done:,} new, {skipped:,} already cached"
                + (f", {failed:,} failed" if failed else ""))
            self.update_inference_coverage(
                note=f"{failed:,} unreadable" if failed else "")
            return

        self._inference_drain_after = self.root.after(200, self._inference_drain)

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

    def refresh_analysis(self):
        """Refresh the current analysis"""
        self.analyze_dataset()

    def _display_statistics(self, stats):
        """Display statistics in text widget"""
        self.stats_text.delete(1.0, tk.END)
        
        text = f"{'='*60}\n"
        text += "DATASET STATISTICS\n"
        text += f"{'='*60}\n\n"
        
        text += f"Total Images Analyzed: {stats['total_images']}\n"
        text += f"Images with Labels: {stats['images_with_labels']}\n"
        text += f"Total Bounding Boxes: {stats['total_boxes']}\n"
        
        if stats['boxes_per_image']:
            avg_boxes = np.mean(stats['boxes_per_image'])
            text += f"Average Boxes per Image: {avg_boxes:.2f}\n"
        
        text += f"\n{'='*60}\n"
        text += "CLASS DISTRIBUTION\n"
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
            text += "IMAGE DIMENSIONS\n"
            text += f"{'='*60}\n\n"
            widths = [w for w, h in stats['image_dimensions']]
            heights = [h for w, h in stats['image_dimensions']]
            text += f"Average Width: {np.mean(widths):.0f}px\n"
            text += f"Average Height: {np.mean(heights):.0f}px\n"
            text += f"Min Width: {min(widths)}px, Max Width: {max(widths)}px\n"
            text += f"Min Height: {min(heights)}px, Max Height: {max(heights)}px\n"
        
        self.stats_text.insert(tk.END, text)

