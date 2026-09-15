"""Dataset statistics and matplotlib reporting.

Extracted from the original single-file `yolo_labeling_tool.py`. Implemented as a
mixin so every `self.*` reference keeps working unchanged through the MRO.
"""

from __future__ import annotations

import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np



class StatisticsMixin:
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

    def _show_visualizations(self, stats):
        """Show visualizations in a new window with matplotlib charts"""
        # Deferred: matplotlib + the TkAgg backend cost ~0.6s to import and are
        # only needed when the user actually opens the analytics window.
        import matplotlib
        matplotlib.use('TkAgg')
        import matplotlib.pyplot as plt  # noqa: F401
        import seaborn as sns  # noqa: F401
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.figure import Figure

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

