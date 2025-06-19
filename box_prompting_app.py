import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw
import cv2
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity
import json

class SmartBoxPromptingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Smart Box Prompting App - Roboflow Style")
        self.root.geometry("1400x900")
        
        # Initialize variables
        self.current_image = None
        self.current_image_cv = None
        self.display_image = None
        self.photo = None
        self.canvas_image = None
        self.scale_factor = 1.0
        
        # Box drawing variables
        self.drawing = False
        self.start_x = None
        self.start_y = None
        self.current_box = None
        
        # Object detection and classification
        self.classes = {}  # {class_name: {'boxes': [], 'template': template_features, 'color': color}}
        self.class_counter = 0
        self.current_class = None
        
        # Feature extraction
        self.sift = cv2.SIFT_create()
        self.orb = cv2.ORB_create()
        
        # Colors for different classes
        self.colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'cyan', 'pink', 
                      'brown', 'gray', 'lime', 'navy', 'maroon', 'olive', 'teal']
        
        # Detection parameters
        self.similarity_threshold = 0.7
        self.auto_detect_enabled = True
        
        self.setup_ui()
        
    def setup_ui(self):
        """Set up the enhanced user interface"""
        
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Top toolbar
        toolbar = ttk.Frame(main_frame)
        toolbar.pack(fill=tk.X, pady=(0, 10))
        
        # File operations
        file_frame = ttk.LabelFrame(toolbar, text="File Operations", padding=5)
        file_frame.pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(file_frame, text="Load Image", command=self.load_image).pack(side=tk.LEFT, padx=2)
        ttk.Button(file_frame, text="Export Annotations", command=self.export_annotations).pack(side=tk.LEFT, padx=2)
        
        # Detection controls
        detection_frame = ttk.LabelFrame(toolbar, text="Detection Controls", padding=5)
        detection_frame.pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(detection_frame, text="Auto-Detect Similar", command=self.auto_detect_similar).pack(side=tk.LEFT, padx=2)
        ttk.Button(detection_frame, text="Clear Current Class", command=self.clear_current_class).pack(side=tk.LEFT, padx=2)
        ttk.Button(detection_frame, text="Clear All", command=self.clear_all_classes).pack(side=tk.LEFT, padx=2)
        
        # Settings
        settings_frame = ttk.LabelFrame(toolbar, text="Settings", padding=5)
        settings_frame.pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Label(settings_frame, text="Similarity:").pack(side=tk.LEFT)
        self.similarity_var = tk.DoubleVar(value=0.7)
        similarity_scale = ttk.Scale(settings_frame, from_=0.3, to=0.9, 
                                   variable=self.similarity_var, length=100)
        similarity_scale.pack(side=tk.LEFT, padx=5)
        
        self.auto_detect_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(settings_frame, text="Auto-detect", 
                       variable=self.auto_detect_var).pack(side=tk.LEFT, padx=5)
        
        # Status
        self.status_label = ttk.Label(toolbar, text="Load an image to start")
        self.status_label.pack(side=tk.RIGHT)
        
        # Content frame
        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left panel - Canvas for image
        canvas_frame = ttk.LabelFrame(content_frame, text="Image Canvas", padding=10)
        canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # Canvas with scrollbars
        canvas_container = ttk.Frame(canvas_frame)
        canvas_container.pack(fill=tk.BOTH, expand=True)
        
        self.canvas = tk.Canvas(canvas_container, bg='white', cursor='crosshair')
        
        # Scrollbars
        v_scrollbar = ttk.Scrollbar(canvas_container, orient=tk.VERTICAL, command=self.canvas.yview)
        h_scrollbar = ttk.Scrollbar(canvas_container, orient=tk.HORIZONTAL, command=self.canvas.xview)
        
        self.canvas.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        
        # Pack scrollbars and canvas
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Bind canvas events
        self.canvas.bind("<Button-1>", self.start_draw)
        self.canvas.bind("<B1-Motion>", self.draw_box)
        self.canvas.bind("<ButtonRelease-1>", self.end_draw)
        
        # Right panel - Classes and controls
        right_panel = ttk.Frame(content_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Class management
        class_frame = ttk.LabelFrame(right_panel, text="Object Classes", padding=10)
        class_frame.pack(fill=tk.BOTH, expand=True, padx=(0, 0))
        
        # Class list
        ttk.Label(class_frame, text="Classes:").pack(anchor=tk.W)
        
        class_list_frame = ttk.Frame(class_frame)
        class_list_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.class_listbox = tk.Listbox(class_list_frame, width=35, height=8)
        class_scrollbar = ttk.Scrollbar(class_list_frame, orient=tk.VERTICAL, command=self.class_listbox.yview)
        self.class_listbox.configure(yscrollcommand=class_scrollbar.set)
        self.class_listbox.bind('<<ListboxSelect>>', self.on_class_select)
        
        self.class_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        class_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Class controls
        class_controls = ttk.Frame(class_frame)
        class_controls.pack(fill=tk.X, pady=5)
        
        ttk.Button(class_controls, text="Rename Class", command=self.rename_class).pack(fill=tk.X, pady=1)
        ttk.Button(class_controls, text="Delete Class", command=self.delete_class).pack(fill=tk.X, pady=1)
        
        # Current class info
        self.current_class_frame = ttk.LabelFrame(right_panel, text="Current Class", padding=10)
        self.current_class_frame.pack(fill=tk.X, pady=10)
        
        self.current_class_label = ttk.Label(self.current_class_frame, text="No class selected")
        self.current_class_label.pack()
        
        # Detection info
        detection_info_frame = ttk.LabelFrame(right_panel, text="Detection Info", padding=10)
        detection_info_frame.pack(fill=tk.X, pady=10)
        
        self.detection_info_label = ttk.Label(detection_info_frame, text="Draw a box to start detection")
        self.detection_info_label.pack()
        
        # Instructions
        instructions = ttk.LabelFrame(right_panel, text="Instructions", padding=5)
        instructions.pack(fill=tk.X, pady=10)
        
        instruction_text = """
1. Load an image
2. Draw a bounding box around an object
3. App will auto-detect similar objects
4. Each new object type creates a new class
5. Adjust similarity threshold as needed
6. Export annotations when done
        """
        
        ttk.Label(instructions, text=instruction_text.strip(), justify=tk.LEFT, 
                 font=('TkDefaultFont', 8)).pack()
        
        # Image info
        self.info_frame = ttk.LabelFrame(right_panel, text="Image Info", padding=5)
        self.info_frame.pack(fill=tk.X, pady=10)
        
        self.info_label = ttk.Label(self.info_frame, text="No image loaded")
        self.info_label.pack()
        
    def load_image(self):
        """Load an image file"""
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.gif *.bmp *.tiff"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            try:
                # Load image
                self.current_image = Image.open(file_path)
                self.current_image_cv = cv2.imread(file_path)
                self.current_image_cv = cv2.cvtColor(self.current_image_cv, cv2.COLOR_BGR2RGB)
                
                self.display_image_on_canvas()
                
                # Update info
                width, height = self.current_image.size
                self.info_label.config(text=f"Size: {width}x{height}\nFile: {os.path.basename(file_path)}")
                
                # Clear existing data
                self.clear_all_classes()
                
                self.status_label.config(text="Image loaded. Draw boxes to detect similar objects.")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image: {str(e)}")
    
    def display_image_on_canvas(self):
        """Display the image on canvas with proper scaling"""
        if self.current_image is None:
            return
            
        # Calculate scale to fit image in canvas
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width <= 1 or canvas_height <= 1:
            # Canvas not ready, try again later
            self.root.after(100, self.display_image_on_canvas)
            return
        
        img_width, img_height = self.current_image.size
        
        # Calculate scale factor
        scale_x = (canvas_width - 20) / img_width
        scale_y = (canvas_height - 20) / img_height
        self.scale_factor = min(scale_x, scale_y, 1.0)  # Don't upscale
        
        # Resize image
        new_width = int(img_width * self.scale_factor)
        new_height = int(img_height * self.scale_factor)
        
        self.display_image = self.current_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(self.display_image)
        
        # Clear canvas and display image
        self.canvas.delete("all")
        self.canvas_image = self.canvas.create_image(10, 10, anchor=tk.NW, image=self.photo)
        
        # Update canvas scroll region
        self.canvas.configure(scrollregion=(0, 0, new_width + 20, new_height + 20))
        
        # Redraw all existing boxes
        self.redraw_all_boxes()
        
    def extract_features(self, image_region):
        """Extract features from an image region using multiple methods"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image_region, cv2.COLOR_RGB2GRAY)
            
            # Method 1: Color histogram
            color_hist = []
            for i in range(3):  # RGB channels
                hist = cv2.calcHist([image_region], [i], None, [32], [0, 256])
                color_hist.extend(hist.flatten())
            
            # Method 2: Texture features using LBP-like approach
            texture_features = []
            if gray.shape[0] > 10 and gray.shape[1] > 10:
                # Simple texture measure
                sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
                sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
                texture_features.extend([np.mean(sobel_x), np.std(sobel_x), 
                                       np.mean(sobel_y), np.std(sobel_y)])
            else:
                texture_features.extend([0, 0, 0, 0])
            
            # Method 3: Shape features
            contours, _ = cv2.findContours(cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)[1], 
                                         cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            shape_features = []
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest_contour)
                perimeter = cv2.arcLength(largest_contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                else:
                    circularity = 0
                shape_features.extend([area / (gray.shape[0] * gray.shape[1]), circularity])
            else:
                shape_features.extend([0, 0])
            
            # Combine all features
            all_features = np.array(color_hist + texture_features + shape_features)
            
            # Normalize features
            if np.linalg.norm(all_features) > 0:
                all_features = all_features / np.linalg.norm(all_features)
            
            return all_features
            
        except Exception as e:
            print(f"Feature extraction error: {e}")
            return np.zeros(100)  # Return zero vector if extraction fails
    
    def find_similar_objects(self, template_features, template_box):
        """Find similar objects in the image using sliding window and feature matching"""
        if self.current_image_cv is None:
            return []
        
        similar_boxes = []
        img_height, img_width = self.current_image_cv.shape[:2]
        
        # Template dimensions
        template_w = template_box[2] - template_box[0]
        template_h = template_box[3] - template_box[1]
        
        # Define search parameters
        scale_factors = [0.8, 1.0, 1.2]  # Different scales to search
        step_size = max(10, min(template_w, template_h) // 4)  # Adaptive step size
        
        try:
            for scale in scale_factors:
                search_w = int(template_w * scale)
                search_h = int(template_h * scale)
                
                if search_w < 10 or search_h < 10:
                    continue
                
                # Slide window across image
                for y in range(0, img_height - search_h, step_size):
                    for x in range(0, img_width - search_w, step_size):
                        # Extract region
                        region = self.current_image_cv[y:y+search_h, x:x+search_w]
                        
                        if region.size == 0:
                            continue
                        
                        # Extract features
                        region_features = self.extract_features(region)
                        
                        # Calculate similarity
                        if len(region_features) > 0 and len(template_features) > 0:
                            similarity = cosine_similarity([template_features], [region_features])[0][0]
                            
                            if similarity > self.similarity_var.get():
                                # Check if this box overlaps significantly with template or existing boxes
                                new_box = (x, y, x + search_w, y + search_h)
                                if not self.is_overlapping(new_box, template_box, 0.3) and \
                                   not any(self.is_overlapping(new_box, box, 0.3) for box in similar_boxes):
                                    similar_boxes.append(new_box)
                                    
                                    # Limit number of detections to prevent performance issues
                                    if len(similar_boxes) >= 20:
                                        return similar_boxes
            
        except Exception as e:
            print(f"Error in finding similar objects: {e}")
        
        return similar_boxes
    
    def is_overlapping(self, box1, box2, threshold=0.3):
        """Check if two boxes overlap significantly"""
        x1_inter = max(box1[0], box2[0])
        y1_inter = max(box1[1], box2[1])
        x2_inter = min(box1[2], box2[2])
        y2_inter = min(box1[3], box2[3])
        
        if x1_inter >= x2_inter or y1_inter >= y2_inter:
            return False
        
        inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        union_area = box1_area + box2_area - inter_area
        
        if union_area == 0:
            return False
        
        iou = inter_area / union_area
        return iou > threshold
    
    def start_draw(self, event):
        """Start drawing a bounding box"""
        if self.current_image is None:
            return
            
        # Get canvas coordinates
        self.start_x = self.canvas.canvasx(event.x)
        self.start_y = self.canvas.canvasy(event.y)
        self.drawing = True
        
    def draw_box(self, event):
        """Update the bounding box while dragging"""
        if not self.drawing or self.current_image is None:
            return
            
        # Get current coordinates
        current_x = self.canvas.canvasx(event.x)
        current_y = self.canvas.canvasy(event.y)
        
        # Remove previous temporary box
        if self.current_box:
            self.canvas.delete(self.current_box)
            
        # Draw new temporary box
        self.current_box = self.canvas.create_rectangle(
            self.start_x, self.start_y, current_x, current_y,
            outline='red', width=2, tags="temp_box"
        )
        
    def end_draw(self, event):
        """Finish drawing the bounding box and trigger auto-detection"""
        if not self.drawing or self.current_image is None:
            return
            
        # Get final coordinates
        end_x = self.canvas.canvasx(event.x)
        end_y = self.canvas.canvasy(event.y)
        
        # Calculate box coordinates
        canvas_x1 = min(self.start_x, end_x)
        canvas_y1 = min(self.start_y, end_y)
        canvas_x2 = max(self.start_x, end_x)
        canvas_y2 = max(self.start_y, end_y)
        
        # Remove temporary box
        if self.current_box:
            self.canvas.delete(self.current_box)
        
        # Only create box if it has reasonable size
        if abs(canvas_x2 - canvas_x1) > 20 and abs(canvas_y2 - canvas_y1) > 20:
            # Convert to original image coordinates
            orig_x1 = int((canvas_x1 - 10) / self.scale_factor)
            orig_y1 = int((canvas_y1 - 10) / self.scale_factor)
            orig_x2 = int((canvas_x2 - 10) / self.scale_factor)
            orig_y2 = int((canvas_y2 - 10) / self.scale_factor)
            
            # Ensure coordinates are within image bounds
            orig_x1 = max(0, orig_x1)
            orig_y1 = max(0, orig_y1)
            orig_x2 = min(self.current_image_cv.shape[1], orig_x2)
            orig_y2 = min(self.current_image_cv.shape[0], orig_y2)
            
            template_box = (orig_x1, orig_y1, orig_x2, orig_y2)
            
            # Extract template region and features
            try:
                template_region = self.current_image_cv[orig_y1:orig_y2, orig_x1:orig_x2]
                if template_region.size > 0:
                    template_features = self.extract_features(template_region)
                    
                    # Create new class or add to existing one
                    class_name = f"Class_{self.class_counter}"
                    self.class_counter += 1
                    
                    color = self.colors[len(self.classes) % len(self.colors)]
                    
                    # Find similar objects
                    self.status_label.config(text="Detecting similar objects...")
                    self.root.update()
                    
                    similar_boxes = []
                    if self.auto_detect_var.get():
                        similar_boxes = self.find_similar_objects(template_features, template_box)
                    
                    # Create class data
                    all_boxes = [template_box] + similar_boxes
                    
                    self.classes[class_name] = {
                        'boxes': all_boxes,
                        'template_features': template_features,
                        'color': color,
                        'template_box': template_box
                    }
                    
                    # Update UI
                    self.update_class_list()
                    self.draw_class_boxes(class_name)
                    
                    # Select the new class
                    self.current_class = class_name
                    self.current_class_label.config(text=f"Selected: {class_name}")
                    
                    detection_info = f"Detected {len(all_boxes)} objects\n"
                    detection_info += f"Similarity threshold: {self.similarity_var.get():.2f}"
                    self.detection_info_label.config(text=detection_info)
                    
                    self.status_label.config(text=f"Created {class_name} with {len(all_boxes)} objects")
                    
            except Exception as e:
                messagebox.showerror("Error", f"Failed to process selection: {str(e)}")
        
        # Reset drawing state
        self.drawing = False
        self.current_box = None
    
    def draw_class_boxes(self, class_name):
        """Draw all boxes for a specific class"""
        if class_name not in self.classes:
            return
            
        class_data = self.classes[class_name]
        color = class_data['color']
        
        # Remove existing boxes for this class
        self.canvas.delete(f"class_{class_name}")
        
        # Draw all boxes
        for i, box in enumerate(class_data['boxes']):
            # Convert to canvas coordinates
            canvas_x1 = box[0] * self.scale_factor + 10
            canvas_y1 = box[1] * self.scale_factor + 10
            canvas_x2 = box[2] * self.scale_factor + 10
            canvas_y2 = box[3] * self.scale_factor + 10
            
            # Draw box
            self.canvas.create_rectangle(
                canvas_x1, canvas_y1, canvas_x2, canvas_y2,
                outline=color, width=2, tags=f"class_{class_name}"
            )
            
            # Add class label
            self.canvas.create_text(
                canvas_x1, canvas_y1 - 5,
                text=f"{class_name}",
                anchor=tk.SW, fill=color, font=('Arial', 8, 'bold'),
                tags=f"class_{class_name}"
            )
    
    def redraw_all_boxes(self):
        """Redraw all boxes after image scaling"""
        for class_name in self.classes:
            self.draw_class_boxes(class_name)
    
    def update_class_list(self):
        """Update the class listbox"""
        self.class_listbox.delete(0, tk.END)
        for class_name, class_data in self.classes.items():
            count = len(class_data['boxes'])
            self.class_listbox.insert(tk.END, f"{class_name} ({count} objects)")
    
    def on_class_select(self, event):
        """Handle class selection"""
        selection = self.class_listbox.curselection()
        if selection:
            class_text = self.class_listbox.get(selection[0])
            class_name = class_text.split(' (')[0]
            self.current_class = class_name
            self.current_class_label.config(text=f"Selected: {class_name}")
            
            # Highlight selected class boxes
            self.highlight_class(class_name)
    
    def highlight_class(self, class_name):
        """Highlight boxes of selected class"""
        # Remove previous highlights
        self.canvas.delete("highlight")
        
        if class_name in self.classes:
            class_data = self.classes[class_name]
            for box in class_data['boxes']:
                # Convert to canvas coordinates
                canvas_x1 = box[0] * self.scale_factor + 10
                canvas_y1 = box[1] * self.scale_factor + 10
                canvas_x2 = box[2] * self.scale_factor + 10
                canvas_y2 = box[3] * self.scale_factor + 10
                
                # Create highlight
                self.canvas.create_rectangle(
                    canvas_x1-2, canvas_y1-2, canvas_x2+2, canvas_y2+2,
                    outline='white', width=3, tags="highlight"
                )
    
    def auto_detect_similar(self):
        """Re-run auto-detection for current class"""
        if not self.current_class or self.current_class not in self.classes:
            messagebox.showinfo("Info", "Please select a class first")
            return
        
        class_data = self.classes[self.current_class]
        template_features = class_data['template_features']
        template_box = class_data['template_box']
        
        self.status_label.config(text="Re-detecting similar objects...")
        self.root.update()
        
        # Find similar objects
        similar_boxes = self.find_similar_objects(template_features, template_box)
        
        # Update class data
        all_boxes = [template_box] + similar_boxes
        self.classes[self.current_class]['boxes'] = all_boxes
        
        # Update UI
        self.update_class_list()
        self.draw_class_boxes(self.current_class)
        
        detection_info = f"Re-detected {len(all_boxes)} objects\n"
        detection_info += f"Similarity threshold: {self.similarity_var.get():.2f}"
        self.detection_info_label.config(text=detection_info)
        
        self.status_label.config(text=f"Re-detected {len(all_boxes)} objects for {self.current_class}")
    
    def rename_class(self):
        """Rename the selected class"""
        if not self.current_class or self.current_class not in self.classes:
            messagebox.showinfo("Info", "Please select a class first")
            return
        
        new_name = tk.simpledialog.askstring("Rename Class", 
                                           f"Enter new name for {self.current_class}:",
                                           initialvalue=self.current_class)
        
        if new_name and new_name != self.current_class and new_name not in self.classes:
            # Update class data
            self.classes[new_name] = self.classes.pop(self.current_class)
            
            # Update UI
            self.canvas.delete(f"class_{self.current_class}")
            self.current_class = new_name
            self.update_class_list()
            self.draw_class_boxes(new_name)
            self.current_class_label.config(text=f"Selected: {new_name}")
    
    def delete_class(self):
        """Delete the selected class"""
        if not self.current_class or self.current_class not in self.classes:
            messagebox.showinfo("Info", "Please select a class first")
            return
        
        if messagebox.askyesno("Confirm", f"Delete class {self.current_class}?"):
            # Remove from canvas
            self.canvas.delete(f"class_{self.current_class}")
            
            # Remove from data
            del self.classes[self.current_class]
            
            # Update UI
            self.current_class = None
            self.current_class_label.config(text="No class selected")
            self.update_class_list()
            self.detection_info_label.config(text="Class deleted")
    
    def clear_current_class(self):
        """Clear the current class"""
        if self.current_class:
            self.delete_class()
    
    def clear_all_classes(self):
        """Clear all classes"""
        if self.classes and messagebox.askyesno("Confirm", "Clear all classes?"):
            # Clear canvas
            for class_name in self.classes:
                self.canvas.delete(f"class_{class_name}")
            self.canvas.delete("highlight")
            
            # Clear data
            self.classes.clear()
            self.current_class = None
            self.class_counter = 0
            
            # Update UI
            self.update_class_list()
            self.current_class_label.config(text="No class selected")
            self.detection_info_label.config(text="All classes cleared")
            self.status_label.config(text="All classes cleared")
    
    def export_annotations(self):
        """Export annotations in multiple formats"""
        if not self.classes:
            messagebox.showinfo("Info", "No annotations to export")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Export Annotations",
            defaultextension=".json",
            filetypes=[
                ("JSON files", "*.json"),
                ("YOLO format", "*.txt"),
                ("CSV files", "*.csv"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            try:
                if file_path.endswith('.json'):
                    self.export_json(file_path)
                elif file_path.endswith('.txt'):
                    self.export_yolo(file_path)
                elif file_path.endswith('.csv'):
                    self.export_csv(file_path)
                else:
                    self.export_json(file_path)
                    
                messagebox.showinfo("Success", f"Annotations exported to {file_path}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export annotations: {str(e)}")
    
    def export_json(self, file_path):
        """Export annotations in JSON format"""
        annotations = {
            "image_info": {
                "width": self.current_image.width if self.current_image else 0,
                "height": self.current_image.height if self.current_image else 0
            },
            "classes": {}
        }
        
        for class_name, class_data in self.classes.items():
            annotations["classes"][class_name] = {
                "boxes": class_data['boxes'],
                "count": len(class_data['boxes']),
                "color": class_data['color']
            }
        
        with open(file_path, 'w') as f:
            json.dump(annotations, f, indent=2)
    
    def export_yolo(self, file_path):
        """Export annotations in YOLO format"""
        if not self.current_image:
            return
            
        img_width = self.current_image.width
        img_height = self.current_image.height
        
        with open(file_path, 'w') as f:
            class_id = 0
            for class_name, class_data in self.classes.items():
                for box in class_data['boxes']:
                    # Convert to YOLO format (normalized center coordinates and dimensions)
                    x_center = (box[0] + box[2]) / 2.0 / img_width
                    y_center = (box[1] + box[3]) / 2.0 / img_height
                    width = (box[2] - box[0]) / img_width
                    height = (box[3] - box[1]) / img_height
                    
                    f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                class_id += 1
        
        # Also create classes.txt file
        classes_file = file_path.replace('.txt', '_classes.txt')
        with open(classes_file, 'w') as f:
            for class_name in self.classes.keys():
                f.write(f"{class_name}\n")
    
    def export_csv(self, file_path):
        """Export annotations in CSV format"""
        with open(file_path, 'w') as f:
            f.write("class_name,x1,y1,x2,y2,width,height,area\n")
            
            for class_name, class_data in self.classes.items():
                for box in class_data['boxes']:
                    width = box[2] - box[0]
                    height = box[3] - box[1]
                    area = width * height
                    f.write(f"{class_name},{box[0]},{box[1]},{box[2]},{box[3]},{width},{height},{area}\n")

def main():
    # Import additional modules that might not be available
    try:
        import tkinter.simpledialog
        global tk
        tk.simpledialog = tkinter.simpledialog
    except ImportError:
        print("Warning: Some features may not work without tkinter.simpledialog")
    
    try:
        from sklearn.metrics.pairwise import cosine_similarity
    except ImportError:
        print("Error: scikit-learn is required for similarity matching")
        print("Install with: pip install scikit-learn")
        return
    
    root = tk.Tk()
    app = SmartBoxPromptingApp(root)
    
    # Set window icon and properties
    root.resizable(True, True)
    root.minsize(1000, 700)
    
    # Center window on screen
    root.update_idletasks()
    x = (root.winfo_screenwidth() // 2) - (root.winfo_width() // 2)
    y = (root.winfo_screenheight() // 2) - (root.winfo_height() // 2)
    root.geometry(f"+{x}+{y}")
    
    try:
        root.mainloop()
    except KeyboardInterrupt:
        print("Application closed by user")
    except Exception as e:
        print(f"Application error: {e}")

if __name__ == "__main__":
    main()
