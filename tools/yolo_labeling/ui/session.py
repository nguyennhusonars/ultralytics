"""Per-session labeling telemetry.

Extracted from the original single-file `yolo_labeling_tool.py`. Implemented as a
mixin so every `self.*` reference keeps working unchanged through the MRO.
"""

from __future__ import annotations

import time
import tkinter as tk
from pathlib import Path
from tkinter import ttk




class SessionStatsMixin:
    def _session_record(self, kind, cls_id, img_path):
        try:
            bucket = self._session_counts.get(kind)
            if bucket is None:
                return
            bucket['total'] += 1
            cls_id = int(cls_id) if cls_id is not None else -1
            bucket['by_class'][cls_id] = bucket['by_class'].get(cls_id, 0) + 1
            self._session_action_log.append((time.time(), kind, cls_id))
            if img_path is not None:
                ip = str(img_path)
                self._session_images_touched.add(ip)
                self._session_per_image[ip] = self._session_per_image.get(ip, 0) + 1
        except Exception:
            pass
        # Coalesce frequent updates — the status label only needs the new
        # numbers once Tk returns to idle.
        try:
            self.root.after_idle(self._session_refresh_label)
        except Exception:
            pass

    def _session_record_created(self, cls_id, img_path):
        self._session_record('created', cls_id, img_path)

    def _session_record_deleted(self, cls_id, img_path):
        self._session_record('deleted', cls_id, img_path)

    def _session_record_reassigned(self, new_cls_id, img_path):
        # We index by destination class — that's what the user typically
        # wants to see ("how many things did I correct INTO class X").
        self._session_record('class_reassigned', new_cls_id, img_path)

    def _session_record_geometry_modified(self, cls_id, img_path):
        self._session_record('geometry_modified', cls_id, img_path)

    def _session_format_elapsed(self):
        secs = max(0, int(time.time() - self._session_start_time))
        h, rem = divmod(secs, 3600)
        m, s = divmod(rem, 60)
        if h:
            return f"{h:d}h{m:02d}m"
        if m:
            return f"{m:d}m{s:02d}s"
        return f"{s:d}s"

    @staticmethod
    def _session_format_duration(secs):
        secs = max(0, int(secs))
        h, rem = divmod(secs, 3600)
        m, s = divmod(rem, 60)
        if h:
            return f"{h:d}h{m:02d}m"
        if m:
            return f"{m:d}m{s:02d}s"
        return f"{s:d}s"

    def _session_pace_stats(self):
        """Derive throughput insights from the action log. Returns a dict
        with overall, recent-window, and idle-aware-active rates so the
        dialog can show whether the user has slowed down lately."""
        now = time.time()
        elapsed = max(1.0, now - self._session_start_time)
        log = list(self._session_action_log)
        n = len(log)

        # Recent rate (last 5 minutes) — normalized to per-minute even if
        # the session is shorter than the window.
        window = 300.0
        cutoff = now - window
        recent_n = sum(1 for t, _, _ in log if t >= cutoff)
        recent_window = min(window, elapsed)
        recent_apm = recent_n * 60.0 / max(1.0, recent_window)

        # Idle-aware rate — gaps > 60s count as idle. This filters out
        # AFK time so the "active" pace reflects head-down labeling.
        idle_threshold = 60.0
        prev_t = self._session_start_time
        idle = 0.0
        for t, _, _ in log:
            gap = t - prev_t
            if gap > idle_threshold:
                idle += gap - idle_threshold
            prev_t = t
        # Trailing idle (from last action to now)
        trailing_gap = now - prev_t
        if trailing_gap > idle_threshold:
            idle += trailing_gap - idle_threshold
        active = max(1.0, elapsed - idle)

        return {
            'n': n,
            'elapsed': elapsed,
            'idle': idle,
            'active': active,
            'apm_overall': n * 60.0 / elapsed,
            'apm_recent':  recent_apm,
            'apm_active':  n * 60.0 / active,
            'idle_pct':    100.0 * idle / elapsed,
            'avg_secs':    elapsed / max(1, n),
        }

    def _session_timeline_sparkline(self, n_buckets=12, bucket_secs=300):
        """Compact unicode sparkline of action counts in `n_buckets`
        rolling windows ending now. Default = last 60 min in 5-min steps.
        Returns (sparkline_str, max_per_bucket)."""
        now = time.time()
        buckets = [0] * n_buckets
        for t, _, _ in self._session_action_log:
            age = now - t
            if age < 0:
                continue
            idx = int(age // bucket_secs)
            if 0 <= idx < n_buckets:
                buckets[n_buckets - 1 - idx] += 1   # most-recent on the right
        peak = max(buckets) if buckets else 0
        if peak == 0:
            return ("·" * n_buckets, 0)
        bars = "▁▂▃▄▅▆▇█"
        out = []
        for v in buckets:
            if v == 0:
                out.append(" ")
            else:
                step = int(round((v / peak) * (len(bars) - 1)))
                out.append(bars[max(0, min(len(bars) - 1, step))])
        return ("".join(out), peak)

    def _session_top_images(self, n=5):
        items = sorted(self._session_per_image.items(),
                       key=lambda kv: -kv[1])
        return items[:n]

    def _session_action_mix(self):
        """Returns list of (label, count, pct) for the four kinds."""
        c = self._session_counts
        order = (('created',           'Created'),
                 ('deleted',           'Deleted'),
                 ('class_reassigned',  'Reassigned'),
                 ('geometry_modified', 'Moved/Resized'))
        total = sum(c[k]['total'] for k, _ in order)
        out = []
        for k, label in order:
            n = c[k]['total']
            pct = 100.0 * n / total if total else 0.0
            out.append((label, n, pct))
        return out

    def _session_refresh_label(self):
        """Render the right-side status mini and re-schedule itself so the
        elapsed time keeps ticking even when no action is taken."""
        if not hasattr(self, 'session_var'):
            return
        c = self._session_counts
        pace = self._session_pace_stats()
        self.session_var.set(
            f"⏱ {self._session_format_elapsed()}  "
            f"· {pace['apm_recent']:.1f}/min  "
            f"│ +{c['created']['total']} "
            f"−{c['deleted']['total']} "
            f"⟳{c['class_reassigned']['total'] + c['geometry_modified']['total']}  "
            f"│ click for details"
        )
        # Re-arm only if the root is still alive. Use 10s — coarser than 1s
        # so we don't burn cycles, finer than a minute so elapsed feels live.
        try:
            existing = getattr(self, '_session_refresh_after', None)
            if existing is not None:
                self.root.after_cancel(existing)
        except tk.TclError:
            pass
        try:
            self._session_refresh_after = self.root.after(10_000, self._session_refresh_label)
        except tk.TclError:
            pass

    def _session_show_dialog(self):
        """Pop a dialog with the full session breakdown: totals, pace,
        per-class, hot images, and a recent-activity sparkline."""
        win = tk.Toplevel(self.root)
        win.title("Session stats")
        win.geometry("620x720")

        top = ttk.Frame(win, padding=8)
        top.pack(fill=tk.X)
        start = time.strftime('%Y-%m-%d %H:%M:%S',
                              time.localtime(self._session_start_time))
        ttk.Label(top, text=f"Started: {start}",
                  font=('TkDefaultFont', 10)).pack(anchor='w')
        ttk.Label(top, text=f"Elapsed: {self._session_format_elapsed()}",
                  font=('TkDefaultFont', 10)).pack(anchor='w')
        ttk.Label(top, text=f"Images touched: {len(self._session_images_touched)}",
                  font=('TkDefaultFont', 10)).pack(anchor='w')

        ttk.Separator(win, orient='horizontal').pack(fill=tk.X, padx=8, pady=4)

        # ── Pace section ────────────────────────────────────────────────
        pace = self._session_pace_stats()
        psec = ttk.Frame(win, padding=8)
        psec.pack(fill=tk.X)
        ttk.Label(psec, text="Pace",
                  font=('TkDefaultFont', 10, 'bold')).pack(anchor='w')
        ttk.Label(psec, text=(
            f"  Overall: {pace['apm_overall']:.2f} actions/min   "
            f"({pace['n']} actions in {self._session_format_duration(pace['elapsed'])})"
        )).pack(anchor='w')
        ttk.Label(psec, text=(
            f"  Active rate (idle ≥60s removed): {pace['apm_active']:.2f}/min   "
            f"— active {self._session_format_duration(pace['active'])}, "
            f"idle {self._session_format_duration(pace['idle'])} "
            f"({pace['idle_pct']:.0f}%)"
        )).pack(anchor='w')
        ttk.Label(psec, text=(
            f"  Last 5 min: {pace['apm_recent']:.2f}/min   "
            f"Avg seconds per action: {pace['avg_secs']:.1f}s"
        )).pack(anchor='w')

        # Sparkline — last 60 min, 5-min buckets
        spark, peak = self._session_timeline_sparkline(n_buckets=12,
                                                       bucket_secs=300)
        ttk.Label(psec, text=(
            f"  Last hour ({len(spark)}×5min): [{spark}]   peak={peak}"
        ), font=('TkFixedFont', 10)).pack(anchor='w', pady=(4, 0))

        ttk.Separator(win, orient='horizontal').pack(fill=tk.X, padx=8, pady=4)

        # ── Totals + action mix ─────────────────────────────────────────
        c = self._session_counts
        totals = ttk.Frame(win, padding=8)
        totals.pack(fill=tk.X)
        ttk.Label(totals, text="Totals",
                  font=('TkDefaultFont', 10, 'bold')).pack(anchor='w')
        mix = self._session_action_mix()
        for label, n, pct in mix:
            # Tiny inline bar — width proportional to share of total.
            bar = '█' * int(round(pct / 5)) if pct > 0 else ''
            ttk.Label(
                totals,
                text=f"  {label:<14}: {n:>5}   {pct:5.1f}%  {bar}",
                font=('TkFixedFont', 10),
            ).pack(anchor='w')

        ttk.Separator(win, orient='horizontal').pack(fill=tk.X, padx=8, pady=4)

        # Per-class breakdown table
        body = ttk.Frame(win, padding=(8, 0, 8, 8))
        body.pack(fill=tk.BOTH, expand=True)
        ttk.Label(body, text="Per class",
                  font=('TkDefaultFont', 10, 'bold')).pack(anchor='w')

        cols = ('class', 'created', 'deleted', 'reassigned_to', 'moved')
        tree = ttk.Treeview(body, columns=cols, show='headings', height=14)
        tree.heading('class', text='Class')
        tree.heading('created', text='Created')
        tree.heading('deleted', text='Deleted')
        tree.heading('reassigned_to', text='→ Class')
        tree.heading('moved', text='Moved/Resized')
        tree.column('class', width=160, anchor='w')
        tree.column('created', width=80, anchor='e')
        tree.column('deleted', width=80, anchor='e')
        tree.column('reassigned_to', width=80, anchor='e')
        tree.column('moved', width=110, anchor='e')
        vsb = ttk.Scrollbar(body, orient='vertical', command=tree.yview)
        tree.configure(yscrollcommand=vsb.set)
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)

        # Build rows: union of class ids that show up in any bucket
        seen = set()
        for kind in c:
            seen.update(c[kind]['by_class'].keys())
        if not seen:
            tree.insert('', 'end', values=('(no activity yet)', '', '', '', ''))
        else:
            rows = []
            for cls_id in sorted(seen, key=lambda x: (x < 0, x)):
                name = (self.class_info.get(cls_id, {}).get('name', f'?{cls_id}')
                        if cls_id >= 0 else 'unknown')
                rows.append((
                    cls_id,
                    f"{cls_id}: {name}" if cls_id >= 0 else name,
                    c['created']['by_class'].get(cls_id, 0),
                    c['deleted']['by_class'].get(cls_id, 0),
                    c['class_reassigned']['by_class'].get(cls_id, 0),
                    c['geometry_modified']['by_class'].get(cls_id, 0),
                ))
            # Sort by total activity (desc) so the busiest classes float up
            rows.sort(key=lambda r: -(r[2] + r[3] + r[4] + r[5]))
            for _id, name, cr, de, re_, mo in rows:
                tree.insert('', 'end', values=(name, cr, de, re_, mo))

        # ── Hot images ──────────────────────────────────────────────────
        ttk.Separator(win, orient='horizontal').pack(fill=tk.X, padx=8, pady=4)
        hot = ttk.Frame(win, padding=8)
        hot.pack(fill=tk.X)
        ttk.Label(hot, text="Most touched images",
                  font=('TkDefaultFont', 10, 'bold')).pack(anchor='w')
        top_imgs = self._session_top_images(n=5)
        if not top_imgs:
            ttk.Label(hot, text="  (none yet)",
                      font=('TkFixedFont', 10)).pack(anchor='w')
        else:
            for ip, n in top_imgs:
                name = Path(ip).name
                ttk.Label(hot, text=f"  {n:>3}×  {name}",
                          font=('TkFixedFont', 10)).pack(anchor='w')

        # Footer
        foot = ttk.Frame(win, padding=8)
        foot.pack(fill=tk.X)
        ttk.Button(foot, text="Close", command=win.destroy).pack(side=tk.RIGHT)
        ttk.Button(foot, text="Reset counters",
                   command=lambda: (self._session_reset(), win.destroy(),
                                    self._session_show_dialog())
                   ).pack(side=tk.RIGHT, padx=(0, 6))
        win.bind('<Escape>', lambda e: win.destroy())

    def _session_reset(self):
        """Zero all counters and reset the start time. Confirmation-free
        because it's bound to an explicit button click."""
        self._session_start_time = time.time()
        for k in self._session_counts:
            self._session_counts[k] = {'total': 0, 'by_class': {}}
        self._session_images_touched = set()
        self._session_action_log.clear()
        self._session_per_image = {}
        self._session_refresh_label()

