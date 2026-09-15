"""Class registry, hotkeys, colours and groups.

Extracted from the original single-file `yolo_labeling_tool.py`. Implemented as a
mixin so every `self.*` reference keeps working unchanged through the MRO.
"""

from __future__ import annotations

import json
import os
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, simpledialog, ttk

import yaml

from yolo_labeling.core.classmap import PredictionClassMap as _PredictionClassMap
from yolo_labeling.core.classmap import auto_map_by_name as _auto_map_by_name
from yolo_labeling.core.classmap import normalize as _normalize



class ClassManagementMixin:
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

    def select_class(self, class_id):
        self.current_class = class_id
        self.selected_class_var.set(f"Selected Class: {class_id} ({self.class_info[class_id]['name']})")
        self._highlight_selected_class()

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
        self._session_record_reassigned(new_class_id, self.current_image_path)

        if not keep_selection:
            self.clear_selections()

        # Class change can change which boxes overlap
        if self.highlight_overlaps:
            self.detect_current_overlaps()

        self.save_labels()
        self.display_image()  # also re-applies highlight_selected_box if we kept the selection
        self.update_status(f"Changed class to: {self.class_info[new_class_id]['name']}")

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
                    self.prediction_maps = config.get('prediction_maps', {}) or {}
            else:
                # Create default config if file doesn't exist
                self.class_info = {
                    0: {"name": "Default", "color": [0, 0, 255]}
                }
                self.class_groups = {}
                self.prediction_maps = {}
                self.save_class_config()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load class configuration: {str(e)}")
            self.class_info = {
                0: {"name": "Default", "color": [0, 0, 255]}
            }
            self.class_groups = {}
            self.prediction_maps = {}
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

            # Adopt the dataset's own `groups:` block, if it has one and the
            # user has not defined groups by hand. A dataset trained with the
            # `groups` feature collapses its fine classes into coarse ones, so
            # this block is exactly the equivalence relation Find Bad
            # Annotations needs -- and it is already written down. Re-typing it
            # in the GUI is busywork and a chance to get it wrong.
            #
            # Hand-edited groups win: an empty class_groups is the fresh state,
            # so a non-empty one means the user made a deliberate choice.
            yaml_groups = data.get('groups') or {}
            if yaml_groups and not (getattr(self, 'class_groups', {}) or {}):
                self.class_groups = {
                    k: list(v) for k, v in yaml_groups.items()
                    if isinstance(v, (list, tuple))
                }
                self.save_class_config()
            self._rebuild_class_group_lookup()

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

            # Publish the dataset root. It was previously a local, but review
            # marks are keyed relative to it so that days of review progress
            # survive the dataset being copied or opened from another
            # directory -- which an absolute key could not do.
            self.dataset_root = base_path
            self._open_review_marks()

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
                        detail_log += "   Labels: Not found\n"
                    detail_log += "\n"
            
            # Print to console for logging
            print("\n" + "=" * 80)
            print("YAML LOADING SUMMARY")
            print("=" * 80)
            print(detail_log)
            print("=" * 80 + "\n")
            
            # Show simplified message in GUI
            messagebox.showinfo("YAML Loaded", 
                              "Successfully loaded:\n" + "\n".join(f"  {part}" for part in msg_parts) +
                              "\n\nCheck console for detailed folder statistics.")
            
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

        ttk.Label(win, text="Hotkey for class:", font=('TkDefaultFont', 10)
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
                # Per-model prediction class maps, keyed by model filename.
                'prediction_maps': getattr(self, 'prediction_maps', {}) or {},
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


    # ------------------------------------------------------------------
    # prediction class mapping
    # ------------------------------------------------------------------

    def _model_key(self, model_path):
        """Config key for a model: its filename, so the map follows the weights."""
        return Path(model_path).name if model_path else "<none>"

    def _load_prediction_class_map(self, model_path):
        """Load this model's saved map, or seed a fresh one by name matching.

        Seeding by name is what makes the common case free: a model fine-tuned
        on a similar taxonomy maps itself, and only the leftovers need attention.
        """
        if not hasattr(self, 'prediction_maps'):
            self.prediction_maps = {}
        key = self._model_key(model_path)
        saved = self.prediction_maps.get(key)
        if saved:
            self.prediction_class_map = _PredictionClassMap.from_json(saved)
            return
        auto = _auto_map_by_name(self.model_names, self.class_info)
        self.prediction_class_map = _PredictionClassMap(mapping=auto)
        self.prediction_maps[key] = self.prediction_class_map.to_json()
        self.save_class_config()

    def save_prediction_class_map(self):
        """Persist the current map against the loaded model."""
        if not hasattr(self, 'prediction_maps'):
            self.prediction_maps = {}
        self.prediction_maps[self._model_key(getattr(self, 'model_path', None))] = \
            self.prediction_class_map.to_json()
        self.save_class_config()
        # Predictions were cached under the old mapping.
        self.model_predictions_cache = {}
        if getattr(self, 'current_image_path', None):
            self.load_model_predictions()
            self.display_image()

    def edit_prediction_class_map(self):
        """Editor for the model-class -> dataset-class translation table.

        Text-based, one mapping per line, matching the `edit_class_groups`
        convention. Names are used rather than ids because the entire problem
        this solves is that the two id spaces disagree.
        """
        if not getattr(self, 'model_names', None):
            messagebox.showinfo(
                "No model loaded",
                "Load a model first (Setup tab -> Model). Its class list is\n"
                "then mapped onto this project's classes.")
            return

        cmap = self.prediction_class_map
        model_key = self._model_key(getattr(self, 'model_path', None))

        win = tk.Toplevel(self.root)
        win.title(f"Prediction Class Mapping — {model_key}")
        win.geometry("640x560")
        win.transient(self.root)

        ttk.Label(
            win,
            text=("One mapping per line:   model_class: dataset_class\n"
                  "Model classes you leave out are DROPPED — nothing is written "
                  "to your labels.\nSeveral model classes may point at the same "
                  "dataset class (e.g. car, bus,\ntruck -> vehicle). Names or "
                  "numeric ids both work."),
            justify=tk.LEFT, foreground="#555",
        ).pack(anchor=tk.W, padx=10, pady=(10, 4))

        text = tk.Text(win, height=18, wrap=tk.NONE)
        text.pack(fill=tk.BOTH, expand=True, padx=10)

        ds_name = {int(i): info.get('name', str(i)) for i, info in self.class_info.items()}

        def render(mapping):
            text.delete("1.0", tk.END)
            lines = []
            for m_id, m_name in sorted(self.model_names.items()):
                if int(m_id) in mapping:
                    tgt = ds_name.get(int(mapping[int(m_id)]), mapping[int(m_id)])
                    lines.append(f"{m_name}: {tgt}")
                else:
                    lines.append(f"# {m_name}: ")     # commented out == dropped
            text.insert("1.0", "\n".join(lines))

        render(cmap.mapping)

        opts = ttk.Frame(win)
        opts.pack(fill=tk.X, padx=10, pady=(6, 0))
        drop_var = tk.BooleanVar(value=cmap.drop_unmapped)
        on_var = tk.BooleanVar(value=cmap.enabled)
        ttk.Checkbutton(opts, text="Drop unmapped predictions (recommended)",
                        variable=drop_var).pack(anchor=tk.W)
        ttk.Checkbutton(opts, text="Mapping enabled",
                        variable=on_var).pack(anchor=tk.W)

        status = ttk.Label(win, text="", foreground="#a00")
        status.pack(anchor=tk.W, padx=10)

        def resolve_model(tok):
            tok = tok.strip()
            for i, n in self.model_names.items():
                if _normalize(n) == _normalize(tok):
                    return int(i)
            return int(tok) if tok.lstrip("-").isdigit() else None

        def resolve_dataset(tok):
            tok = tok.strip()
            for i, n in ds_name.items():
                if _normalize(n) == _normalize(tok):
                    return int(i)
            if tok.lstrip("-").isdigit() and int(tok) in ds_name:
                return int(tok)
            return None

        def parse():
            mapping, errors = {}, []
            for lineno, raw in enumerate(text.get("1.0", tk.END).splitlines(), 1):
                line = raw.strip()
                if not line or line.startswith("#") or ":" not in line:
                    continue
                left, right = line.split(":", 1)
                if not right.strip():
                    continue                       # written but left blank == dropped
                m_id, ds_id = resolve_model(left), resolve_dataset(right)
                if m_id is None:
                    errors.append(f"line {lineno}: unknown model class {left.strip()!r}")
                elif ds_id is None:
                    errors.append(f"line {lineno}: unknown dataset class {right.strip()!r}")
                else:
                    mapping[m_id] = ds_id
            return mapping, errors

        def on_auto():
            render(_auto_map_by_name(self.model_names, self.class_info))
            status.config(text="Re-seeded by name match. Review, then Save.",
                          foreground="#060")

        def on_save():
            mapping, errors = parse()
            if errors:
                status.config(text=" | ".join(errors[:3]), foreground="#a00")
                return
            cmap.mapping = mapping
            cmap.drop_unmapped = bool(drop_var.get())
            cmap.enabled = bool(on_var.get())
            self.save_prediction_class_map()
            self.update_status(cmap.summary(self.model_names, self.class_info))
            win.destroy()

        btns = ttk.Frame(win)
        btns.pack(fill=tk.X, padx=10, pady=8)
        ttk.Button(btns, text="Auto-map by name", command=on_auto).pack(side=tk.LEFT)
        ttk.Button(btns, text="Save", command=on_save).pack(side=tk.RIGHT)
        ttk.Button(btns, text="Cancel", command=win.destroy).pack(side=tk.RIGHT, padx=6)
