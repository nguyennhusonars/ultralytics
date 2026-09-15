"""Modal dialogs: crop export, Qwen import, bad-annotation scan, bulk reassign.

Extracted from the original single-file `yolo_labeling_tool.py`. Implemented as a
mixin so every `self.*` reference keeps working unchanged through the MRO.
"""

from __future__ import annotations

import json
import re
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2

from yolo_labeling.core import predcache as _predcache
from yolo_labeling.ui.crop_review import _entry_rank



class DialogsMixin:
    def show_export_crops_dialog(self):
        """Export every GT box from the currently filtered images as a crop
        file under ``<output>/<class_name>/<...>.jpg``. Also writes a sidecar
        ``manifest.json`` that maps each exported filename back to its origin
        (img_path, line_idx, cls_id, bbox_norm). The Qwen verifier reads the
        class folders; the labeling tool's importer reads the manifest."""
        if not self.filtered_image_paths:
            messagebox.showinfo("No images", "Load an image folder first.")
            return
        out_dir = filedialog.askdirectory(
            title="Choose output folder for crops + manifest"
        )
        if not out_dir:
            return
        out_dir = Path(out_dir)
        win = tk.Toplevel(self.root)
        win.title("Export Crops")
        win.geometry("400x200")
        win.transient(self.root)
        msg = tk.StringVar(value="Preparing…")
        ttk.Label(win, textvariable=msg, wraplength=380, justify=tk.LEFT).pack(
            padx=12, pady=8
        )
        prog = ttk.Progressbar(win, mode='determinate', length=360)
        prog.pack(padx=12, pady=4)
        done_btn = ttk.Button(win, text="Close", command=win.destroy,
                              state=tk.DISABLED)
        done_btn.pack(pady=8)
        win.update_idletasks()

        manifest = {
            "exported_from": str(self.image_folder) if hasattr(self, 'image_folder') else "",
            "class_names": {str(cid): info['name']
                            for cid, info in self.class_info.items()},
            "entries": [],
        }
        n_exported = 0
        n_skipped = 0
        total = len(self.filtered_image_paths)
        prog['maximum'] = total

        def _safe(name):
            return re.sub(r'[^A-Za-z0-9._-]', '_', str(name))

        for i, img_path in enumerate(self.filtered_image_paths):
            try:
                img = cv2.imread(str(img_path))
            except Exception:
                img = None
            if img is None:
                n_skipped += 1
                prog['value'] = i + 1
                if i % 5 == 0:
                    msg.set(f"{i + 1}/{total} images — exported {n_exported}, "
                            f"skipped {n_skipped}")
                    win.update_idletasks()
                continue
            ih, iw = img.shape[:2]
            lf = self._find_label_file_for(img_path)
            if lf is None or not lf.exists():
                prog['value'] = i + 1
                continue
            try:
                lines = lf.read_text().splitlines()
            except (IOError, OSError):
                continue
            for li, line in enumerate(lines):
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                try:
                    cls_id = int(float(parts[0]))
                    cx, cy, bw, bh = map(float, parts[1:5])
                except ValueError:
                    continue
                cls_name = self.class_info.get(cls_id, {}).get('name')
                if not cls_name:
                    n_skipped += 1
                    continue
                # Crop in pixel space, clipped to image
                x1 = max(0, int((cx - bw / 2) * iw))
                y1 = max(0, int((cy - bh / 2) * ih))
                x2 = min(iw, int((cx + bw / 2) * iw))
                y2 = min(ih, int((cy + bh / 2) * ih))
                if x2 <= x1 or y2 <= y1:
                    n_skipped += 1
                    continue
                crop = img[y1:y2, x1:x2]
                cls_dir = out_dir / _safe(cls_name)
                cls_dir.mkdir(parents=True, exist_ok=True)
                # Use a deterministic, collision-resistant crop filename.
                stem = _safe(Path(img_path).stem)
                crop_name = f"{stem}__line{li:04d}.jpg"
                crop_path = cls_dir / crop_name
                try:
                    cv2.imwrite(str(crop_path), crop)
                except Exception:
                    n_skipped += 1
                    continue
                manifest["entries"].append({
                    "filename": crop_path.relative_to(out_dir).as_posix(),
                    "img_path": str(img_path),
                    "line_idx": li,
                    "cls_id": cls_id,
                    "bbox_norm": [cx, cy, bw, bh],
                })
                n_exported += 1
            prog['value'] = i + 1
            if i % 5 == 0:
                msg.set(f"{i + 1}/{total} images — exported {n_exported}, "
                        f"skipped {n_skipped}")
                win.update_idletasks()

        try:
            (out_dir / "manifest.json").write_text(
                json.dumps(manifest, indent=2)
            )
        except IOError as e:
            messagebox.showerror("Export failed",
                                 f"Could not write manifest.json: {e}")
            win.destroy()
            return
        msg.set(
            f"Done. Exported {n_exported} crops, skipped {n_skipped}. "
            f"Manifest at {out_dir / 'manifest.json'}.\n\n"
            f"Next: point qwen3_class_verify.py's IMAGE_FOLDER at "
            f"{out_dir}, run it, then click "
            f"'Import Qwen Verification' here with the resulting "
            f"verification.json."
        )
        done_btn.config(state=tk.NORMAL)

    def show_import_qwen_dialog(self):
        """Read a Qwen verification.json + its sibling manifest.json (or a
        user-supplied manifest) and open the bad-annotation review window
        populated with every entry where Qwen disagrees with the GT class."""
        v_path = filedialog.askopenfilename(
            title="Select Qwen verification.json",
            filetypes=[("JSON", "*.json")],
        )
        if not v_path:
            return
        v_path = Path(v_path)
        # The manifest typically sits next to the exported crops folder. Try
        # to find it automatically; otherwise ask the user.
        manifest_path = None
        for guess in [v_path.parent.parent / "manifest.json",
                      v_path.parent / "manifest.json"]:
            if guess.exists():
                manifest_path = guess
                break
        if manifest_path is None:
            manifest_path = Path(filedialog.askopenfilename(
                title="Select the matching manifest.json",
                filetypes=[("JSON", "*.json")],
            ) or "")
        if not manifest_path or not manifest_path.exists():
            messagebox.showerror("Missing manifest",
                                 "Cannot import without the matching manifest.json.")
            return

        try:
            ver = json.loads(v_path.read_text())
            man = json.loads(manifest_path.read_text())
        except (IOError, ValueError) as e:
            messagebox.showerror("Read error", f"Failed to parse JSON: {e}")
            return

        # Build manifest lookup: filename -> origin record
        man_lookup = {}
        for rec in man.get("entries", []):
            if rec.get("filename"):
                man_lookup[rec["filename"]] = rec
        # Name -> id map for current class_info (qwen uses names)
        name_to_id = {info['name']: cid for cid, info in self.class_info.items()}

        results = ver.get("results") if isinstance(ver, dict) else ver
        if not isinstance(results, list):
            messagebox.showerror("Bad file",
                                 "verification.json must contain a 'results' list.")
            return

        entries = []
        n_unknown = n_match = n_parse_err = n_unmapped = 0
        # Confidence weighting for sort order
        conf_w = {"high": 1.0, "medium": 0.7, "low": 0.4}
        for r in results:
            fname = r.get("filename")
            pred_name = (r.get("pred_class") or "").strip()
            if pred_name == "unknown" or not pred_name:
                n_unknown += 1
                continue
            if pred_name == "parse_error":
                n_parse_err += 1
                continue
            origin = man_lookup.get(fname)
            if origin is None:
                n_unmapped += 1
                continue
            gt_cls_id = int(origin.get("cls_id", -1))
            pred_cls_id = name_to_id.get(pred_name)
            if pred_cls_id is None:
                n_unmapped += 1
                continue
            # Skip agreements (the user only cares about disagreements). If
            # class_groups is set, also skip when within the same group.
            if self._classes_equivalent(gt_cls_id, pred_cls_id):
                n_match += 1
                continue
            cx, cy, bw, bh = origin.get("bbox_norm", [0.5, 0.5, 0.1, 0.1])
            conf_str = (r.get("confidence") or "low").lower()
            score = conf_w.get(conf_str, 0.4)
            entries.append({
                "img_path": Path(origin["img_path"]),
                "line_idx": int(origin["line_idx"]),
                "cls_id": gt_cls_id,
                "bbox_norm": (float(cx), float(cy), float(bw), float(bh)),
                "pred_cls_id": pred_cls_id,
                "pred_conf": score,
                "iou": 1.0,
                "score": score,
                "kind": "class_mismatch",  # reuse caption rendering
                "qwen_notes": r.get("notes", ""),
                "qwen_confidence": conf_str,
            })

        if not entries:
            messagebox.showinfo(
                "No disagreements",
                f"Qwen agreed with GT on all crops.\n\n"
                f"Skipped: {n_match} matches, {n_unknown} 'unknown', "
                f"{n_parse_err} parse errors, {n_unmapped} unmapped."
            )
            return
        # Same ordering the bad-label review uses.
        entries.sort(key=_entry_rank)
        self._open_crop_review_window(entries=entries, target_class=None,
                                      mode='bad')
        self._crop_review_status_var.set(
            f"Loaded {len(entries)} qwen disagreements  ·  "
            f"matches:{n_match}  unknown:{n_unknown}  "
            f"parse_err:{n_parse_err}  unmapped:{n_unmapped}"
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
        win.transient(self.root)
        win.resizable(False, True)

        conf_var = tk.StringVar(value="0.40")
        iou_var = tk.StringVar(value="0.50")
        mismatch_var = tk.BooleanVar(value=True)
        missing_var = tk.BooleanVar(value=True)
        scope_var = tk.StringVar(value='filtered')

        frm = ttk.Frame(win)
        frm.pack(fill=tk.BOTH, expand=True, padx=12, pady=10)

        def _hint(parent, text):
            ttk.Label(parent, text=text, foreground="#666", wraplength=430,
                      font=('TkDefaultFont', 8), justify=tk.LEFT).pack(
                          anchor=tk.W, padx=(4, 0), pady=(0, 4))

        # -- what the scan reports ------------------------------------------
        # Named for what the user should DO about each, not for the internal
        # mechanism. "Missing prediction" in particular read as "missing
        # annotation", which is the opposite of what it finds.
        what = ttk.LabelFrame(frm, text="Report")
        what.pack(fill=tk.X, pady=(0, 8))
        ttk.Checkbutton(what, text="Wrong class",
                        variable=mismatch_var).pack(anchor=tk.W, padx=5, pady=(4, 0))
        _hint(what, "A labelled box where the model confidently sees a different "
                    "class. Fix by reassigning the class.")
        ttk.Checkbutton(what, text="Probably not an object",
                        variable=missing_var).pack(anchor=tk.W, padx=5)
        _hint(what, "A labelled box the model sees nothing at. Often a stray or "
                    "duplicate box. Fix by deleting it.\n"
                    "Note: this does NOT find objects you forgot to label — the "
                    "scan only ever examines boxes that already exist.")

        # -- what counts as agreement ---------------------------------------
        match = ttk.LabelFrame(frm, text="Matching")
        match.pack(fill=tk.X, pady=(0, 8))

        def _num_row(parent, label_text, var):
            row = ttk.Frame(parent)
            row.pack(fill=tk.X, padx=5, pady=(4, 0))
            ttk.Label(row, text=label_text, width=26, anchor='w').pack(side=tk.LEFT)
            ttk.Entry(row, textvariable=var, width=8).pack(side=tk.LEFT)

        _num_row(match, "Ignore predictions below:", conf_var)
        _hint(match, "The model's own confidence. Predictions weaker than this "
                     "are not considered at all, in either direction.")
        _num_row(match, "Same object when IoU ≥:", iou_var)
        _hint(match, "How much a labelled box and a prediction must overlap "
                     "before they are treated as the same object. Below this, "
                     "the model is considered to have seen nothing there.")

        # -- optional confusion pair ----------------------------------------
        # The predicted side lists only classes THIS model can actually emit --
        # offering all 13 dataset classes when the model has 7 invites a query
        # that can only ever return nothing.
        ANY = "(any)"
        gt_var = tk.StringVar(value=ANY)
        pred_var = tk.StringVar(value=ANY)
        all_opts = [ANY] + [f"{cid}: {info['name']}"
                            for cid, info in sorted(self.class_info.items())]
        cmap = getattr(self, 'prediction_class_map', None)
        emittable = sorted(set((cmap.mapping or {}).values())) if cmap else []
        pred_opts = [ANY] + [f"{cid}: {self.class_info[cid]['name']}"
                             for cid in emittable if cid in self.class_info]
        if len(pred_opts) == 1:                      # no map -> model ids raw
            pred_opts = all_opts

        pair = ttk.LabelFrame(frm, text="Restrict to one confusion (optional)")
        pair.pack(fill=tk.X, pady=(0, 8))

        def _combo_row(label_text, var, options):
            row = ttk.Frame(pair)
            row.pack(fill=tk.X, padx=5, pady=(4, 0))
            ttk.Label(row, text=label_text, width=26, anchor='w').pack(side=tk.LEFT)
            ttk.Combobox(row, textvariable=var, values=options,
                         state='readonly', width=22).pack(side=tk.LEFT)

        _combo_row("Labelled as:", gt_var, all_opts)
        _combo_row("…but model says:", pred_var, pred_opts)
        n_groups = len(getattr(self, 'class_groups', {}) or {})
        _hint(pair, (f"Class groups: {n_groups} defined — classes inside a group "
                     "count as agreeing, so a coarse model is not asked to tell "
                     "apart sub-types it cannot see. Naming both sides above "
                     "overrides that."
                     if n_groups else
                     "Class groups: none — classes must match exactly. Define "
                     "them via Classes → 'Edit Class Groups…', or let them load "
                     "from your data.yaml `groups:` block."))

        def _cid(var):
            v = var.get()
            return None if v == ANY else int(v.split(':', 1)[0])

        # -- scope -----------------------------------------------------------
        n_filtered = len(self.filtered_image_paths)
        n_all = len(self.image_paths or [])
        scope = ttk.LabelFrame(frm, text="Scope")
        scope.pack(fill=tk.X, pady=(0, 8))
        ttk.Radiobutton(scope, text=f"Filtered images ({n_filtered:,})",
                        value='filtered', variable=scope_var).pack(anchor=tk.W, padx=5, pady=(4, 0))
        ttk.Radiobutton(scope, text=f"All loaded images ({n_all:,})",
                        value='all', variable=scope_var).pack(anchor=tk.W, padx=5)
        cached = 0
        if getattr(self, 'pred_cache', None) is not None:
            try:
                cached = self.pred_cache.count()
            except Exception:
                cached = 0
        _hint(scope, f"{cached:,} image(s) already inferred for this model and "
                     "will be read from cache; the rest run the model. Setup → "
                     "'Infer all' primes the cache up front.")

        status_var = tk.StringVar()
        ttk.Label(frm, textvariable=status_var, foreground="#cc2222",
                  wraplength=430, justify=tk.LEFT).pack(anchor=tk.W, pady=2)

        btn_row = ttk.Frame(frm)
        btn_row.pack(fill=tk.X, pady=(10, 0))

        def _start():
            if not (mismatch_var.get() or missing_var.get()):
                status_var.set("Pick at least one thing to report.")
                return
            if _cid(pred_var) is not None and not mismatch_var.get():
                status_var.set("A 'model says' restriction only applies to "
                               "'Wrong class' — switch it on, or clear the "
                               "restriction.")
                return
            try:
                conf = float(conf_var.get())
                iou_match = float(iou_var.get())
            except (tk.TclError, ValueError):
                status_var.set("Confidence and IoU must be numbers.")
                return
            if not (0 < conf <= 1.0):
                status_var.set("Confidence must be between 0 and 1.")
                return
            if not (0 < iou_match <= 1.0):
                status_var.set("IoU must be between 0 and 1.")
                return

            paths = (list(self.image_paths) if scope_var.get() == 'all'
                     else list(self.filtered_image_paths))
            if not paths:
                status_var.set("That scope contains no images.")
                return

            # Below the cache floor nothing can be served from the store, so
            # the scan silently re-runs the model over the whole scope. Say so
            # rather than letting it look like the cache stopped working.
            if conf < _predcache.CONF_FLOOR and not messagebox.askyesno(
                    "Below the cache floor",
                    f"Predictions are cached at confidence "
                    f"{_predcache.CONF_FLOOR:.2f} and above.\n\n"
                    f"Scanning at {conf:.2f} cannot use the cache, so the model "
                    f"will re-run on all {len(paths):,} image(s).\n\nContinue?"):
                return

            config = {
                'paths': paths,
                'conf': conf,
                'iou_match': iou_match,
                'mismatch': mismatch_var.get(),
                'missing': missing_var.get(),
                'gt_class': _cid(gt_var),
                'pred_class': _cid(pred_var),
            }
            win.destroy()
            self.open_bad_label_review(config)

        ttk.Button(btn_row, text="Start scan", command=_start).pack(side=tk.RIGHT)
        ttk.Button(btn_row, text="Cancel", command=win.destroy).pack(
            side=tk.RIGHT, padx=4)

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

