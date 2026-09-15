"""Durable prediction cache.

Running the detector over a 46k-image dataset is a coffee-break operation, and
until now every consumer threw the result away: the bad-annotation scan inferred
every image and kept nothing, nudging the confidence slider wiped the in-memory
cache wholesale, and restarting the tool started from zero.

Two ideas make one cache serve everybody:

**Store at a floor, filter at read.** Predictions are cached at a low confidence
floor and each consumer filters to its own threshold. The scan at 0.40 and the
canvas at 0.25 then share one entry, and moving the slider becomes a filter over
data already in hand rather than a reason to re-infer.

**Key on identity, not path.** An entry is valid only while the image's size and
mtime match what was inferred, and only for the model file that produced it, so
a re-exported frame or a retrained checkpoint invalidates itself.

SQLite rather than a JSON blob because the scan needs to commit incrementally --
cancelling half way through, or a crash, must not cost the work already done.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path

#: Confidence floor everything is cached at. Any consumer wanting a threshold
#: at or above this is served from cache; below it, we must re-infer.
CONF_FLOOR = 0.10

#: Per-image cap, highest confidence first. A low floor on a busy traffic frame
#: can return the detector's NMS maximum; keeping the strongest 100 bounds the
#: cache without affecting any realistic threshold.
MAX_BOXES = 100


def model_key(model_path) -> str:
    """Identity of a weights file: name plus size and mtime.

    Retraining in place keeps the filename, so the name alone would serve
    predictions from the previous checkpoint.
    """
    if not model_path:
        return "<none>"
    p = Path(model_path)
    try:
        st = p.stat()
        return f"{p.name}:{st.st_size}:{int(st.st_mtime)}"
    except OSError:
        return p.name


def _image_stamp(img_path):
    """(size, mtime) for an image, or None if it cannot be read."""
    try:
        st = Path(img_path).stat()
        return st.st_size, int(st.st_mtime)
    except OSError:
        return None


def clip_boxes(boxes):
    """Drop below-floor boxes and keep the strongest `MAX_BOXES`."""
    kept = [b for b in boxes if len(b) < 6 or float(b[5]) >= CONF_FLOOR]
    if len(kept) > MAX_BOXES:
        kept = sorted(kept, key=lambda b: -float(b[5]) if len(b) > 5 else 0.0)[:MAX_BOXES]
    return kept


def filter_conf(boxes, conf):
    """The read-side half of 'store at a floor': narrow to a threshold."""
    if conf is None or conf <= CONF_FLOOR:
        return list(boxes)
    return [b for b in boxes if len(b) < 6 or float(b[5]) >= conf]


class PredictionCache:
    """SQLite-backed store of per-image predictions for one model.

    Safe to call from the scan worker thread and the GUI thread: SQLite is
    opened with `check_same_thread=False` and every statement is serialised on
    an instance lock.
    """

    def __init__(self, db_path, model_path):
        self.path = Path(db_path)
        self.model_key = model_key(model_path)
        self._lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(self.path), check_same_thread=False)
        # WAL keeps the scan's incremental commits cheap and lets a reader run
        # while the worker writes.
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=NORMAL")
        self._db.execute("""
            CREATE TABLE IF NOT EXISTS preds (
                model  TEXT NOT NULL,
                img    TEXT NOT NULL,
                size   INTEGER NOT NULL,
                mtime  INTEGER NOT NULL,
                floor  REAL NOT NULL,
                boxes  TEXT NOT NULL,
                PRIMARY KEY (model, img)
            )
        """)
        self._db.commit()

    # -- reads ---------------------------------------------------------------
    def get(self, img_path, conf=None):
        """Cached boxes for `img_path` at `conf`, or None on a miss.

        A miss covers all of: never inferred, inferred for a different model,
        the image changed on disk, or `conf` is below what we cached at.
        """
        stamp = _image_stamp(img_path)
        if stamp is None:
            return None
        size, mtime = stamp
        with self._lock:
            row = self._db.execute(
                "SELECT size, mtime, floor, boxes FROM preds WHERE model=? AND img=?",
                (self.model_key, str(img_path)),
            ).fetchone()
        if row is None:
            return None
        r_size, r_mtime, r_floor, raw = row
        if r_size != size or r_mtime != mtime:
            return None                       # image changed since inference
        if conf is not None and conf < r_floor:
            return None                       # cached data cannot answer this
        try:
            boxes = [tuple(b) for b in json.loads(raw)]
        except (ValueError, TypeError):
            return None                       # tolerate a corrupted row
        return filter_conf(boxes, conf)

    def has(self, img_path):
        return self.get(img_path) is not None

    # -- writes --------------------------------------------------------------
    def put(self, img_path, boxes, commit=True):
        """Store `boxes` (already at CONF_FLOOR) for `img_path`."""
        stamp = _image_stamp(img_path)
        if stamp is None:
            return
        size, mtime = stamp
        payload = json.dumps([
            [int(b[0])] + [round(float(v), 1) for v in b[1:5]]
            + ([round(float(b[5]), 3)] if len(b) > 5 else [])
            for b in clip_boxes(boxes)
        ], separators=(',', ':'))
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO preds (model, img, size, mtime, floor, boxes)"
                " VALUES (?,?,?,?,?,?)",
                (self.model_key, str(img_path), size, mtime, CONF_FLOOR, payload),
            )
            if commit:
                self._db.commit()

    def commit(self):
        with self._lock:
            self._db.commit()

    # -- housekeeping --------------------------------------------------------
    def count(self):
        with self._lock:
            return self._db.execute(
                "SELECT COUNT(*) FROM preds WHERE model=?", (self.model_key,)
            ).fetchone()[0]

    def clear(self, all_models=False):
        with self._lock:
            if all_models:
                self._db.execute("DELETE FROM preds")
            else:
                self._db.execute("DELETE FROM preds WHERE model=?", (self.model_key,))
            self._db.commit()

    def close(self):
        with self._lock:
            try:
                self._db.commit()
                self._db.close()
            except sqlite3.Error:
                pass
