"""Crop-review window and background bad-annotation scanning.

Extracted from the original single-file `yolo_labeling_tool.py`. Implemented as a
mixin so every `self.*` reference keeps working unchanged through the MRO.
"""

from __future__ import annotations

import threading
import time as _time
import tkinter as tk
from tkinter import messagebox, ttk

import cv2
from PIL import Image, ImageTk

from yolo_labeling.core import geometry as _geom
from yolo_labeling.core import predcache as _predcache


#: Report order for bad-annotation entries.
#:
#: `score` is only comparable WITHIN a kind: a wrong-class entry scores by how
#: confident the model is that you are wrong (0-1), while a sees-nothing entry
#: scores by how little the model saw (also 0-1, but not the same quantity).
#: Sorting the mixed list on `score` alone put two different scales into one
#: ordering, which is why "worst first" did not mean anything. Group by kind,
#: then sort within it.
KIND_ORDER = {'class_mismatch': 0, 'missing': 1}


def _entry_rank(entry):
    """Sort key: kind group first, then most-severe within the group."""
    return (KIND_ORDER.get(entry.get('kind'), 9), -entry.get('score', 0.0))


class CropReviewMixin:
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
        """Delegates to core.geometry.iou_xyxy."""
        return _geom.iou_xyxy(a, b)

    def _bad_scan_worker(self, config):
        """Background-thread scanner. For every GT label in every image of
        the scope: run the model, find the best-IoU prediction, and decide
        whether the GT is suspicious. Pushes ('entry', dict) / ('progress',
        i, total) / ('done', reason) / ('error', msg) messages onto
        ``self._bad_scan_queue``.

        Must not touch any Tk widget — the main thread reads the queue and
        renders results."""
        try:
            paths = list(config['paths'])
            conf_thr = float(config['conf'])
            iou_match = float(config['iou_match'])
            detect_mismatch = bool(config['mismatch'])
            detect_missing = bool(config['missing'])
            # Snapshot the map once: the user can edit it mid-scan, and a map
            # that changes underneath a running scan would make the results
            # incoherent between the first and last image.
            n_since_commit = 0
            cmap = getattr(self, 'prediction_class_map', None)
            # Optional confusion-pair filter: "show me GT class A that the
            # model calls class B". Either side may be None for "any".
            gt_filter = config.get('gt_class')
            pred_filter = config.get('pred_class')
            # Naming BOTH sides is an explicit request for that raw pair, so it
            # overrides the group relaxation. Without this, asking for
            # bus_l-called-bus_m returns nothing -- the group says they are the
            # same class -- which is exactly the sub-type confusion a user would
            # be reaching for this filter to investigate.
            pair_mode = gt_filter is not None and pred_filter is not None
            for i, path in enumerate(paths):
                if self._bad_scan_cancelled.is_set():
                    if self.pred_cache is not None:
                        self.pred_cache.commit()
                    self._bad_scan_queue.put(("done", "cancelled"))
                    return
                # Dimensions from the JPEG header, not a full decode. GT boxes
                # are normalised, so all this loop needs is (w, h). Measured on
                # pvn-vehicle frames: 0.03ms here against 3.4ms to decode, i.e.
                # ~6 minutes of pure decode across 46k images. That, not
                # inference, is what made a fully-cached scan feel slow.
                size = self._image_size(str(path))
                if size is None:
                    self._bad_scan_queue.put(("progress", i + 1, len(paths)))
                    continue
                iw, ih = size

                # Cache first, and cheaply. Only a miss pays for a decode.
                cached = self._peek_cached_predictions(path)
                if cached is None:
                    try:
                        img = cv2.imread(str(path))
                    except Exception:
                        img = None
                    if img is None:
                        # Unreadable: skip rather than report every GT box as
                        # 'missing', which is what "no predictions" would mean.
                        self._bad_scan_queue.put(("progress", i + 1, len(paths)))
                        continue
                    try:
                        cached = self._cached_predictions(path, img, blocking=True)
                    except Exception as e:
                        self._bad_scan_queue.put(("error", f"model error on {path.name}: {e}"))
                        self._bad_scan_queue.put(("progress", i + 1, len(paths)))
                        continue
                    if cached is None:
                        self._bad_scan_queue.put(("progress", i + 1, len(paths)))
                        continue
                    n_since_commit += 1
                    if n_since_commit >= 200 and self.pred_cache is not None:
                        # Commit periodically so cancelling -- or a crash --
                        # keeps the work already done.
                        self.pred_cache.commit()
                        n_since_commit = 0
                preds = []
                for box in _predcache.filter_conf(cached or [], conf_thr):
                    # Translate HERE rather than reusing `_display_predictions`:
                    # that helper drops unmapped classes, and this scan needs to
                    # keep them as None. An unmapped box still proves the model
                    # saw SOMETHING here, so the GT must not be reported as
                    # 'missing'; it merely cannot support a class-mismatch claim.
                    ds_cls = cmap.translate(box[0]) if cmap is not None else box[0]
                    preds.append((
                        ds_cls, float(box[5]) if len(box) > 5 else 0.0,
                        float(box[1]), float(box[2]),
                        float(box[3]), float(box[4]),
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
                    if gt_filter is not None and gt_cls != int(gt_filter):
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
                        # A 'missing' entry has no prediction, so it can never
                        # satisfy a "predicted as B" filter.
                        if detect_missing and pred_filter is None:
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
                        # pcls is None when the model's class has no entry in
                        # the prediction map — inconclusive, never a mismatch.
                        differs = (pcls is not None and (
                            pcls != gt_cls if pair_mode
                            else not self._classes_equivalent(gt_cls, pcls)))
                        if differs and (pred_filter is None
                                        or pcls == int(pred_filter)):
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
            if self.pred_cache is not None:
                self.pred_cache.commit()
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
            self._crop_review_entries.sort(key=_entry_rank)
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

    @staticmethod
    def _fmt_eta(seconds):
        """Compact duration: '45s', '7m 20s', '1h 12m'."""
        seconds = max(0, int(seconds))
        if seconds < 60:
            return f"{seconds}s"
        if seconds < 3600:
            return f"{seconds // 60}m {seconds % 60:02d}s"
        return f"{seconds // 3600}h {(seconds % 3600) // 60:02d}m"

    def _bad_scan_update_status(self, progress, done_msg, last_error):
        """Format the bad-scan progress line for the review window's status bar."""
        if self._crop_review_window is None:
            return
        n_bad = len(self._crop_review_entries)
        parts = []
        if progress is not None:
            i, total = progress
            parts.append(f"Scanned {i}/{total}")
            bar = getattr(self, '_bad_scan_progress', None)
            if bar is not None and total:
                try:
                    bar['value'] = 100.0 * i / total
                except tk.TclError:
                    pass                       # window closed mid-update
            # Rate and ETA turn "it is still going" into a decision about
            # whether to wait or cancel.
            started = getattr(self, '_bad_scan_started_at', None)
            if started is not None and done_msg is None:
                elapsed = _time.time() - started
                if elapsed > 1.0 and i:
                    rate = i / elapsed
                    parts.append(f"{rate:.1f} img/s")
                    parts.append(f"ETA {self._fmt_eta((total - i) / rate)}")
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
            default_help = ("Wrong-class first, then probably-not-an-object. "
                            "1-9/0 reassigns, Delete drops, K marks reviewed "
                            "(no change). Click a crop to jump to the main window.")
        else:
            default_help = ("Click a crop, then press 1-9/0 to reassign "
                            "(selection auto-advances). Delete removes. "
                            "Page nav drops handled.")
        # Scan controls are pinned RIGHT and packed FIRST. Everything here used
        # to be side=LEFT behind a full sentence of help text, so on a 1100px
        # window the progress bar and Cancel button were pushed clean off the
        # edge -- present, updating, and invisible.
        if mode == 'bad':
            self._bad_scan_cancel_btn = ttk.Button(
                action, text="⛔ Cancel scan",
                command=self._bad_scan_request_cancel,
            )
            self._bad_scan_cancel_btn.pack(side=tk.RIGHT, padx=4)
            # A determinate bar, because scanning tens of thousands of frames
            # at a few per second is a coffee-break operation and a bare
            # "Scanned 812/46049" does not convey that.
            self._bad_scan_progress = ttk.Progressbar(
                action, mode='determinate', length=160, maximum=100.0)
            self._bad_scan_progress.pack(side=tk.RIGHT, padx=(8, 4))

        ttk.Label(action, text="Reassign to: ").pack(side=tk.LEFT)
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
        if mode == 'bad':
            ttk.Button(action, text="✓ Mark reviewed (K)",
                       command=self._crop_review_mark_reviewed).pack(
                           side=tk.LEFT, padx=4)

        # Help / selection text gets its own row. It is the one piece here whose
        # length is unbounded -- a selection caption names a file, two classes
        # and a score -- so sharing a row with the controls means the controls
        # lose.
        self._crop_review_sel_var = tk.StringVar(value=default_help)
        ttk.Label(win, textvariable=self._crop_review_sel_var,
                  foreground='#0066cc', anchor='w').pack(
                      fill=tk.X, padx=8, pady=(0, 4))

        # Status bar BEFORE the grid: pack() hands each widget a slab of the
        # remaining cavity in call order, so anything packed after a
        # fill=BOTH/expand=True sibling is allotted zero height and never
        # appears. That is why the scan's progress line was invisible.
        self._crop_review_status_var = tk.StringVar(value="")
        ttk.Label(win, textvariable=self._crop_review_status_var,
                  relief=tk.SUNKEN, anchor='w').pack(side=tk.BOTTOM, fill=tk.X)

        # Crop grid host
        self._crop_review_grid_frame = ttk.Frame(win)
        self._crop_review_grid_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)
        # Debounced re-render when the grid frame is resized, so cells
        # adaptively fill the window and thumbnails re-fit to the new cell size.
        self._crop_review_resize_after = None
        self._crop_review_last_cell_size = (0, 0)
        self._crop_review_grid_frame.bind('<Configure>',
                                          self._crop_review_on_grid_resize)

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
                     '_crop_review_prefetch_after',
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
            prefetch_cancel = getattr(self, '_crop_review_prefetch_cancel', None)
            if prefetch_cancel is not None:
                prefetch_cancel.set()
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
            # sync_visible reuses the existing cell widgets (~3.4× faster
            # than full render) and falls back to render() if the layout
            # can't be reused.
            self._crop_review_sync_visible()

    def _crop_review_next_page(self):
        self._crop_review_flush_handled()
        rows, cols = self._crop_review_get_grid()
        per_page = rows * cols
        max_page = max(0, (len(self._crop_review_entries) - 1) // per_page)
        if self._crop_review_page < max_page:
            self._crop_review_page += 1
        # Either advanced, or already on last page — sync_visible refreshes
        # in-place using the existing cell widgets.
        self._crop_review_sync_visible()

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
        # Reuse existing cells (sync_visible falls back to render() if it
        # can't, e.g. on a structural change).
        self._crop_review_sync_visible()

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

    def _crop_review_ensure_cache_size(self, per_page):
        """Size the THUMBNAIL LRU to hold ~5 pages so paging back and forth
        always hits cache. We deliberately do NOT grow the RAW IMAGE cache
        with per_page — each raw cv2 frame can be 10–50MB, and storing
        per_page*3 of them at 6x10 would mean gigabytes resident."""
        needed = max(self._crop_review_thumb_cache_max, per_page * 5)
        if self._crop_review_thumb_cache_max < needed:
            self._crop_review_thumb_cache_max = needed

    def _crop_review_thumb_cache_put(self, eid, photo):
        """LRU insert into the thumbnail cache. If `eid` is already there,
        move it to the most-recent slot. Evict the oldest when over cap."""
        if eid in self._crop_review_thumb_cache:
            del self._crop_review_thumb_cache[eid]
        self._crop_review_thumb_cache[eid] = photo
        # Bound the cache — long sessions with thousands of entries would
        # otherwise accumulate hundreds of MB of PhotoImage objects.
        while len(self._crop_review_thumb_cache) > self._crop_review_thumb_cache_max:
            oldest = next(iter(self._crop_review_thumb_cache))
            self._crop_review_thumb_cache.pop(oldest, None)

    def _crop_review_thumb_cache_get(self, eid):
        """LRU access — moves the entry to the most-recent slot on hit."""
        photo = self._crop_review_thumb_cache.get(eid)
        if photo is None:
            return None
        del self._crop_review_thumb_cache[eid]
        self._crop_review_thumb_cache[eid] = photo
        return photo

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

    def _crop_review_schedule_prefetch(self):
        """Queue a background prefetch of adjacent-page thumbnails. Called
        at the end of every render / sync_visible so the user's *next*
        Prev/Next click finds thumbs already cached.

        We schedule on after_idle so it never delays the current paint."""
        if self._crop_review_window is None:
            return
        prev = getattr(self, '_crop_review_prefetch_after', None)
        if prev is not None:
            try:
                self.root.after_cancel(prev)
            except tk.TclError:
                pass
            self._crop_review_prefetch_after = None

        def _kick():
            self._crop_review_prefetch_after = None
            if self._crop_review_window is not None:
                self._crop_review_prefetch_adjacent()
        self._crop_review_prefetch_after = self.root.after_idle(_kick)

    def _crop_review_prefetch_adjacent(self):
        """Decode thumbnails for the next and previous pages in a worker
        thread, so the user's next page-nav paints from cache instead of
        blocking the UI on dozens of cv2.imread calls.

        Self-cancels any in-flight prefetch via the shared cancel event so
        rapid paging doesn't pile up worker threads."""
        if self._crop_review_window is None:
            return
        rows, cols = self._crop_review_get_grid()
        per_page = rows * cols
        total = len(self._crop_review_entries)
        if total == 0:
            return
        cell_w, cell_h = self._crop_review_last_cell_size
        if cell_w <= 0 or cell_h <= 0:
            return
        page = self._crop_review_page

        # Collect entries from upcoming and previous pages in priority order:
        # next page first (highest hit-rate on Next click), then +2 (covers
        # rapid Next-Next), then -1 / -2 for back-paging. Skip already-cached.
        targets = []
        for offset in (1, 2, -1, -2):
            p = page + offset
            if p < 0:
                continue
            s = p * per_page
            if s >= total:
                continue
            e_ = min(s + per_page, total)
            for entry in self._crop_review_entries[s:e_]:
                if id(entry) not in self._crop_review_thumb_cache:
                    targets.append(entry)
        if not targets:
            return

        # Cancel any prior prefetch so it stops touching the cache.
        prev_cancel = getattr(self, '_crop_review_prefetch_cancel', None)
        if prev_cancel is not None:
            prev_cancel.set()

        cancel = threading.Event()
        self._crop_review_prefetch_cancel = cancel

        # Snapshot dimensions so a later grid-size change in the main
        # thread doesn't corrupt in-flight work.
        snap_w, snap_h = cell_w, cell_h

        def worker():
            local_imgs = {}
            for entry in targets:
                if cancel.is_set():
                    return
                ip = str(entry['img_path'])
                img = local_imgs.get(ip)
                if img is None:
                    # dict.get / dict[__setitem__] are atomic under CPython's
                    # GIL, so reading the main-thread LRU here is safe.
                    img = self._crop_review_img_cache.get(ip)
                    if img is None:
                        img = cv2.imread(ip)
                        if img is None:
                            continue
                    local_imgs[ip] = img
                ih, iw = img.shape[:2]
                cx, cy, bw, bh = entry['bbox_norm']
                x1 = int(round((cx - bw / 2) * iw))
                y1 = int(round((cy - bh / 2) * ih))
                x2 = int(round((cx + bw / 2) * iw))
                y2 = int(round((cy + bh / 2) * ih))
                x1c, y1c = max(0, min(x1, iw)), max(0, min(y1, ih))
                x2c, y2c = max(0, min(x2, iw)), max(0, min(y2, ih))
                if x2c <= x1c or y2c <= y1c:
                    continue
                crop = img[y1c:y2c, x1c:x2c]
                ch, cw = crop.shape[:2]
                scale = min(snap_w / cw, snap_h / ch)
                nw, nh = max(1, int(cw * scale)), max(1, int(ch * scale))
                crop_rgb = cv2.cvtColor(
                    cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA),
                    cv2.COLOR_BGR2RGB
                )
                pil = Image.fromarray(crop_rgb)

                def _commit(entry=entry, pil=pil, ip=ip, img=img):
                    # We DELIBERATELY do not check ``cancel.is_set()`` here.
                    # The PIL image is already decoded — discarding it would
                    # force a synchronous re-decode on the next paint, which
                    # is the exact "fast then slow" symptom we're fixing.
                    if self._crop_review_window is None:
                        return
                    # Guard against id(entry) reuse: the entry may have been
                    # reassigned/deleted/flushed while the worker was decoding.
                    # If it's no longer in the live list, drop the result —
                    # otherwise we'd cache a thumbnail at an id() that Python
                    # may later reuse for an unrelated entry.
                    found = False
                    for e in self._crop_review_entries:
                        if e is entry:
                            found = True
                            break
                    if not found:
                        return
                    if id(entry) in self._crop_review_thumb_cache:
                        # Bump LRU but don't redo the PhotoImage construction.
                        self._crop_review_thumb_cache_get(id(entry))
                        return
                    try:
                        photo = ImageTk.PhotoImage(image=pil)
                    except tk.TclError:
                        return
                    self._crop_review_thumb_cache_put(id(entry), photo)
                    if ip not in self._crop_review_img_cache:
                        self._crop_review_img_cache[ip] = img
                        if len(self._crop_review_img_cache) > self._crop_review_img_cache_max:
                            oldest = next(iter(self._crop_review_img_cache))
                            self._crop_review_img_cache.pop(oldest, None)
                try:
                    self.root.after(0, _commit)
                except (tk.TclError, RuntimeError):
                    return

        threading.Thread(target=worker, daemon=True).start()

    def _crop_review_build_thumbnail(self, entry, cell_w, cell_h):
        """Return a PhotoImage for this entry, computing once and caching."""
        eid = id(entry)
        cached = self._crop_review_thumb_cache_get(eid)
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
        self._crop_review_thumb_cache_put(eid, photo)
        return photo

    def _crop_review_caption_text(self, entry):
        """Build the per-cell caption.

        In standard (class-filter) mode this shows the file + line, with an
        ``→ NewClass`` suffix once the entry has been reassigned.

        In bad-annotation mode (entry carries ``kind`` / ``pred_cls_id``) it
        states the diagnosis in the same words the scan dialog used, e.g.
        ``img42.jpg #3   labelled sedan, model says truck (0.78)`` or
        ``img42.jpg #3   labelled human, model sees nothing (best IoU 0.00)``."""
        base = f"{entry['img_path'].name}  #{entry['line_idx'] + 1}"
        cls_id = entry['cls_id']
        cls_name = self.class_info.get(cls_id, {}).get('name', f'?{cls_id}')

        kind = entry.get('kind')
        if kind in ('class_mismatch', 'missing'):
            if kind == 'class_mismatch':
                pcid = entry.get('pred_cls_id')
                pname = self.class_info.get(pcid, {}).get('name', f'?{pcid}')
                pconf = entry.get('pred_conf', 0.0)
                tail = f"labelled {cls_name}, model says {pname} ({pconf:.2f})"
            else:
                iou = entry.get('iou', 0.0)
                tail = f"labelled {cls_name}, model sees nothing (best IoU {iou:.2f})"
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
        # Coalesced in-place sync (fast path: reuses cell widgets).
        self._crop_review_schedule_sync()

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

    def _crop_review_schedule_sync(self):
        """Coalesce in-place visible-cells refresh via ``after_idle``. Used
        by reassign / delete / mark-reviewed: mashing a hotkey collapses
        N updates into one render on the next idle tick."""
        prev = getattr(self, '_crop_review_render_after', None)
        if prev is not None:
            try:
                self.root.after_cancel(prev)
            except tk.TclError:
                pass
        def _do_sync():
            self._crop_review_render_after = None
            if self._crop_review_window is not None:
                self._crop_review_sync_visible()
        self._crop_review_render_after = self.root.after_idle(_do_sync)

    def _crop_review_install_entry(self, cell, cnv, caption_lbl, entry,
                                   cell_w, cell_h):
        """Install ``entry`` into an EXISTING cell. Updates the canvas image,
        caption text, click binding, and style. Does not create or destroy
        any widgets — used both by the full render path and by the in-place
        sync used after reassign/delete/mark-reviewed for snappy feedback."""
        photo = self._crop_review_build_thumbnail(entry, cell_w, cell_h)
        cnv.delete("all")
        if photo is not None:
            cnv.create_image(cell_w // 2, cell_h // 2,
                             image=photo, anchor='center')
            cnv.photo = photo  # keep extra ref alive on the canvas too
        else:
            cnv.create_text(cell_w // 2, cell_h // 2,
                            text="(failed to read)", fill='#888')
        caption_lbl.config(text=self._crop_review_caption_text(entry))

        def on_click(_e, ent=entry):
            try:
                self._crop_review_selected = self._crop_review_entries.index(ent)
            except ValueError:
                return
            self._crop_review_refresh_action_bar()
            for c2, _cnv2, _lbl2, e2 in self._crop_review_canvases:
                self._crop_review_apply_cell_style(c2, e2)
            self._crop_review_jump_to_entry(ent)
        for w in (cnv, cell, caption_lbl):
            w.bind('<Button-1>', on_click)
        self._crop_review_apply_cell_style(cell, entry)

    def _crop_review_sync_visible(self):
        """Fast in-place refresh of the visible cells: reuse the cell widgets
        and just rebind them to the entries that should now be visible.

        Used by reassign / delete / mark-reviewed where the grid layout is
        unchanged. Falls back to a full ``_crop_review_render`` when the
        layout can't be reused (e.g. page needs more cells than exist).

        Cuts post-action latency from ~150 ms (destroy + recreate 24
        widgets) down to ~10 ms (just data updates on existing widgets)."""
        if self._crop_review_window is None:
            return
        rows, cols = self._crop_review_get_grid()
        per_page = rows * cols
        self._crop_review_ensure_cache_size(per_page)
        total = len(self._crop_review_entries)
        total_pages = max(1, (total + per_page - 1) // per_page) if total else 1
        if self._crop_review_page >= total_pages:
            self._crop_review_page = max(0, total_pages - 1)
        page = self._crop_review_page
        start = page * per_page
        end = min(start + per_page, total)
        page_entries = self._crop_review_entries[start:end]

        cells = list(self._crop_review_canvases)
        # If we need more cells than we currently have, only the full render
        # path knows how to grow the grid — fall back to it.
        if len(page_entries) > len(cells):
            self._crop_review_render()
            return

        cell_w, cell_h = self._crop_review_last_cell_size
        if cell_w <= 0 or cell_h <= 0:
            self._crop_review_render()
            return

        # Title update is trivially cheap and keeps the page indicator honest
        if getattr(self, '_crop_review_mode', 'filter') == 'bad':
            head = "Bad annotations (worst first)"
        else:
            cls_name = self.class_info.get(self._crop_review_target, {}).get('name', '?')
            head = f"Class: {cls_name}"
        self._crop_review_title_var.set(
            f"{head}   |   page {page + 1}/{total_pages}   "
            f"|   showing {start + 1 if total else 0}–{end} of {total}"
        )

        new_canvases = []
        for i, entry in enumerate(page_entries):
            cell, cnv, caption_lbl, _old = cells[i]
            self._crop_review_install_entry(
                cell, cnv, caption_lbl, entry, cell_w, cell_h
            )
            new_canvases.append((cell, cnv, caption_lbl, entry))

        # If the visible-page count shrank (e.g. last entry on last page),
        # destroy the now-unused trailing cells. This is at most a few
        # widgets so it's still much cheaper than rebuilding all of them.
        for cell, _cnv, _lbl, _entry in cells[len(page_entries):]:
            try:
                cell.destroy()
            except tk.TclError:
                pass

        self._crop_review_canvases = new_canvases
        self._crop_review_refresh_action_bar()
        self._crop_review_schedule_prefetch()

    def _crop_review_render(self):
        # Tear down old cells (we still rebuild widgets, but image data is cached)
        for w in self._crop_review_grid_frame.winfo_children():
            w.destroy()
        self._crop_review_canvases = []

        rows, cols = self._crop_review_get_grid()
        per_page = rows * cols
        self._crop_review_ensure_cache_size(per_page)
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
                cnv = tk.Canvas(cell, width=cell_w, height=cell_h, bg='#111',
                                highlightthickness=0)
                cnv.pack(expand=True)
                caption_lbl = ttk.Label(cell, text="",
                                        font=('TkDefaultFont', 8))
                caption_lbl.pack()
                # Hand off content installation to the shared helper so the
                # rapid-action path (sync_visible) and the full render path
                # build the same cell state.
                self._crop_review_install_entry(
                    cell, cnv, caption_lbl, entry, cell_w, cell_h
                )
                self._crop_review_canvases.append(
                    (cell, cnv, caption_lbl, entry)
                )

        self._crop_review_refresh_action_bar()
        self._crop_review_schedule_prefetch()

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
        elif getattr(self, '_crop_review_mode', 'filter') == 'bad':
            self._crop_review_sel_var.set(
                "Wrong-class first, then probably-not-an-object. 1-9/0 "
                "reassigns, Delete drops, K marks reviewed (no change). "
                "Click a crop to jump to the main window."
            )
        else:
            self._crop_review_sel_var.set(
                "Click a crop, then press 1-9/0 to reassign (selection auto-advances). Delete removes. Page nav drops handled."
            )

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
        self._session_record_deleted(entry['cls_id'], deleted_path)
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
        # Coalesced in-place sync — mashing Delete collapses to one repaint.
        self._crop_review_schedule_sync()

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
        self._session_record_reassigned(new_class_id, entry['img_path'])
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
        # Coalesced in-place sync — if the user mashes hotkeys faster than
        # we can draw, only the most-recent state ends up rendered, and the
        # render itself is just a content update on existing widgets.
        self._crop_review_schedule_sync()

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

