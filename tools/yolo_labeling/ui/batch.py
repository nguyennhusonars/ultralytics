"""Grid/batch browsing of many images at once.

Extracted from the original single-file `yolo_labeling_tool.py`. Implemented as a
mixin so every `self.*` reference keeps working unchanged through the MRO.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

import threading
from tkinter import messagebox
from yolo_labeling.core import draw as _draw
from yolo_labeling.core import geometry as _geom
from yolo_labeling.core import labelio as _labelio
import cv2
from PIL import Image, ImageTk



class BatchViewMixin:
    def load_batch_images(self, mark_reviewed=True):
        """Load images for batch view with optimized performance.

        Args:
            mark_reviewed: Whether showing this page counts as reviewing it.
                False when repainting a page the user just un-marked -- the
                repaint would otherwise re-mark it instantly and un-marking a
                page in grid view would be impossible.
        """
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
        # Fill in any tiles that have no cached predictions, off the GUI thread.
        self._batch_predict_async()
        # Showing a page counts as going through its images. Marked here
        # rather than in load_image() because the grid bypasses that funnel
        # entirely.
        if mark_reviewed:
            self._mark_reviewed(self._batch_visible_paths())
        self.update_status("Batch loaded")
        self.root.update_idletasks()

    def _batch_class_color(self, cls_id):
        """Class colour as an RGB tuple (tiles are drawn in RGB, config is BGR)."""
        raw = self.class_info.get(cls_id, {}).get("color")
        if isinstance(raw, (list, tuple)) and len(raw) >= 3:
            return tuple(int(c) for c in reversed(raw[:3]))
        if isinstance(raw, str) and raw.startswith('#'):
            return (int(raw[1:3], 16), int(raw[3:5], 16), int(raw[5:7], 16))
        return (255, 255, 255)

    def load_batch_image(self, img_idx, canvas):
        """Load and display a single image in batch view - optimized"""
        img_path = self.filtered_image_paths[img_idx]
        img_path_str = str(img_path)
        
        canvas_width = max(1, canvas.winfo_width())
        canvas_height = max(1, canvas.winfo_height())

        full_size = self._image_size(img_path_str)
        if full_size is None:
            return
        full_w, full_h = full_size

        cache_key = (img_path_str, canvas_width, canvas_height)
        cached = self._batch_thumb_cache.get(cache_key)
        if cached is not None:
            img_resized = cached.copy()      # copy: boxes get drawn into it
            new_height, new_width = img_resized.shape[:2]
        else:
            # Decode at reduced resolution -- a ~300x200 tile needs nothing like
            # a 12MP frame. This is the difference between ~45ms and ~6ms a tile,
            # i.e. ~1s vs ~0.15s for a 5x5 page.
            img = self._read_reduced(img_path_str, canvas_width, canvas_height)
            if img is None:
                return
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            dec_h, dec_w = img_rgb.shape[:2]
            fit = min(canvas_width / dec_w, canvas_height / dec_h)
            new_width, new_height = max(1, int(dec_w * fit)), max(1, int(dec_h * fit))
            img_resized = cv2.resize(img_rgb, (new_width, new_height),
                                     interpolation=cv2.INTER_AREA)
            self._batch_thumb_cache[cache_key] = img_resized.copy()
            while len(self._batch_thumb_cache) > self._batch_thumb_cache_max:
                self._batch_thumb_cache.pop(next(iter(self._batch_thumb_cache)))

        # Label boxes are in FULL-resolution pixels, so their scale must come
        # from the full image size -- not from whatever resolution we decoded at.
        scale = new_width / full_w if full_w else 1.0

        label_path = self._resolver().resolve(img_path)
        label_boxes = []
        if label_path.exists():
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
                    
                    # Add class name, inside the top edge -- predictions
                    # take the slot above the box (see below), so the two
                    # captions on one object do not collide.
                    if getattr(self, 'show_box_labels', True):
                        class_name = self.class_info[cls_id]["name"]
                        _draw.draw_tag(img_resized, class_name,
                                       x1_scaled, y1_scaled + 2, color_rgb,
                                       scale=0.4, thickness=1, anchor='below')
        
        # Draw model predictions as THIN rectangles so they read as suggestions
        # rather than committed labels -- same visual language as single view.
        # Only cached predictions are drawn here; uncached tiles are filled in
        # by the background pass (see _batch_predict_async).
        pred_count = 0
        if getattr(self, 'show_model_predictions', True):
            for pbox in self._display_predictions(img_path_str):
                cls_id = int(pbox[0])
                if cls_id not in self.class_info:
                    continue
                if not self.class_visibility.get(cls_id, True):
                    continue
                pred_count += 1
                color_rgb = self._batch_class_color(cls_id)
                px1, py1, px2, py2 = (int(v * scale) for v in pbox[1:5])
                cv2.rectangle(img_resized, (px1, py1), (px2, py2), color_rgb, 1)
                if getattr(self, 'show_box_labels', True):
                    tag = self.class_info[cls_id]["name"]
                    if len(pbox) > 5:
                        tag = f"{tag} {float(pbox[5]):.2f}"
                    _draw.draw_tag(img_resized, tag, px1, py1, color_rgb,
                                   scale=0.35, thickness=1, anchor='above')

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

        # Reviewed tick. A canvas item, not a `draw_tag` plate baked into the
        # pixels: that keeps it out of `_batch_thumb_cache`, so toggling review
        # state needs no thumbnail invalidation.
        if self.is_reviewed(img_path):
            canvas.create_text(canvas_width - 10, 10, text="✓",
                               fill="#00ff88", anchor=tk.NE,
                               font=('Arial', 14, 'bold'), tags='overlay')

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
        # The tiles use grid, so this must too -- Tk forbids mixing geometry
        # managers within one container. Sits on its own row beneath the grid.
        ttk.Button(self.batch_frame, text="Accept predictions on this page",
                   command=self.accept_batch_predictions).grid(
            row=self.batch_size[0], column=0,
            columnspan=max(1, self.batch_size[1]), sticky="ew")


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


    # ------------------------------------------------------------------
    # predictions in batch view
    # ------------------------------------------------------------------

    def _batch_visible_paths(self):
        """Paths of the images currently shown in the grid."""
        if not self.batch_mode or not self.filtered_image_paths:
            return []
        start = max(0, self.current_image_index)
        n = len(self.batch_canvases) or (self.batch_size[0] * self.batch_size[1])
        return list(self.filtered_image_paths[start:start + n])

    def _batch_predict_async(self):
        """Infer predictions for grid tiles that have none cached yet.

        Runs on a worker thread and takes `_model_lock`, so it cannot race the
        crop-review scanner or the GUI thread's own lazy inference. Results are
        posted back with `after(0, ...)`; only then are the tiles redrawn.
        """
        if self.pretrained_model is None or not self.batch_mode:
            return
        # `_peek_cached_predictions` also consults the DURABLE store, so a page
        # revisited after a restart is served from disk instead of re-inferred.
        todo = [p for p in self._batch_visible_paths()
                if self._peek_cached_predictions(p) is None]
        if not todo:
            return

        prev = getattr(self, '_batch_predict_thread', None)
        if prev is not None and prev.is_alive():
            return                      # a pass is already in flight for this page

        def worker(paths):
            done = {}
            for path in paths:
                if not self.batch_mode:
                    break               # user left the grid; stop wasting inference
                try:
                    preds = self._cached_predictions(path, blocking=True)
                except Exception:
                    preds = []
                done[str(path)] = preds if preds is not None else []
            self.root.after(0, lambda: self._batch_predictions_ready(done))

        self._batch_predict_thread = threading.Thread(
            target=worker, args=(todo,), daemon=True)
        self._batch_predict_thread.start()
        self.update_status(f"Predicting {len(todo)} tile(s)…")

    def _batch_predictions_ready(self, results):
        """Main thread: fold worker results into the cache and repaint tiles."""
        if not results:
            return
        self.model_predictions_cache.update(results)
        if not self.batch_mode:
            return
        # Thumbnails have the old (prediction-free) overlay baked in.
        for key in list(self._batch_thumb_cache):
            if key[0] in results:
                self._batch_thumb_cache.pop(key, None)
        self.load_batch_images()
        n = sum(len(v) for v in results.values())
        self.update_status(f"Predicted {n} box(es) across {len(results)} tile(s)")

    def accept_batch_predictions(self, event=None):
        """Accept visible predictions across every tile on the current page.

        Writes to many label files in one action, so it always confirms first
        and reports exactly what it is about to do.
        """
        if not self.batch_mode:
            return
        paths = self._batch_visible_paths()
        plan = {}
        for path in paths:
            preds = self._display_predictions(path)
            keep = [p for p in preds
                    if int(p[0]) in self.class_info
                    and self.class_visibility.get(int(p[0]), True)]
            if keep:
                plan[path] = keep
        if not plan:
            self.update_status("No visible predictions to accept on this page")
            return

        n_boxes = sum(len(v) for v in plan.values())
        if not messagebox.askyesno(
                "Accept predictions",
                f"Add {n_boxes} predicted box(es) to {len(plan)} label file(s)?\n\n"
                "This appends to the existing labels on disk and cannot be undone "
                "from the grid."):
            return

        written, failed = 0, []
        for path, boxes in plan.items():
            try:
                self._append_boxes_to_label_file(path, boxes)
                written += 1
            except Exception as e:
                failed.append(f"{path.name}: {e}")

        # Both caches now describe stale content.
        for path in plan:
            self.labels_cache.pop(str(path), None)
        for key in list(self._batch_thumb_cache):
            if key[0] in {str(p) for p in plan}:
                self._batch_thumb_cache.pop(key, None)
        self.load_batch_images()

        if failed:
            messagebox.showerror(
                "Some files were not written",
                f"{written} succeeded, {len(failed)} failed:\n" + "\n".join(failed[:8]))
        else:
            self.update_status(f"Accepted {n_boxes} box(es) into {written} file(s)")

    def _append_boxes_to_label_file(self, img_path, boxes):
        """Append pixel boxes to `img_path`'s label file, atomically.

        `save_labels` only ever works on the current image, so batch acceptance
        needs its own writer. It reuses the same core pieces, which is why the
        atomicity and polygon-preservation guarantees still hold here.
        """
        size = self._image_size(str(img_path))
        if size is None:
            raise ValueError("could not read image dimensions")
        img_w, img_h = size

        label_path = self._resolver().resolve(img_path)
        lines = _labelio.read_label_file(label_path)   # existing lines kept verbatim
        for box in boxes:
            cls_id = int(box[0])
            cx, cy, w, h = _geom.pixel_to_yolo(*box[1:5], img_w, img_h)
            lines.append(_labelio.LabelLine(cls_id, (cx, cy, w, h), "", dirty=True))
        _labelio.write_label_file(label_path, lines)
