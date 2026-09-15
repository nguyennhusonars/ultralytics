"""Which images the user has already been through.

Reviewing a 46k-image dataset by hand is days of work, and until now the tool
kept no record of it. Opening an image, judging it correct and leaving it
untouched left no trace at all -- so there was no way to ask "what haven't I
looked at yet", and restarting lost any sense of position.

Two choices shape this module:

**Keys are dataset-relative.** Every other store here keys on the absolute path
string (`predcache`, the in-memory caches, session telemetry). That is right for
a machine-local cache and wrong for this: review progress is worth days, so it
has to survive the dataset being copied to another disk or another machine.
Keys are POSIX-style relative paths under the dataset root, with an absolute
fallback for images that live outside it.

**A timestamp, not a bare set.** Storing *when* costs nothing over a list of
names and answers "when did I last go through this", which a plain set cannot.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path

from yolo_labeling.core import labelio as _labelio
from yolo_labeling.core.paths import is_within

#: Bumped only if the on-disk shape changes incompatibly.
SCHEMA_VERSION = 1

#: Written beside the dataset's data.yaml.
FILENAME = "review_state.json"


class ReviewMarks:
    """A persistent set of reviewed images, keyed relative to `root`.

    Single-threaded by design: unlike `predcache.PredictionCache` nothing here
    is touched from a worker thread, so there is no lock.

    Mutations only set an in-memory flag; `save()` is explicit and the caller
    debounces it. Marking on every navigation would otherwise mean 46k writes
    for one pass over the dataset.
    """

    def __init__(self, path, root=None):
        self.path = Path(path)
        self.root = Path(root) if root else None
        self.marks: dict[str, int] = {}
        self.dirty = False
        self.load()

    # -- keys ----------------------------------------------------------------

    def key(self, img_path) -> str:
        """Storage key for an image: relative to the root where possible.

        `as_posix()` rather than the native separator, so marks made on Linux
        are still readable if the dataset is opened on Windows.

        The plain `relative_to` is tried before anything that touches the
        filesystem: this runs once per image inside the filter loop, i.e. up to
        46k times per "Apply Filters" click, and `Path.resolve()` is a syscall.
        Paths from the yaml loader are already resolved, so the cheap branch is
        the one that normally hits.
        """
        if self.root is None:
            return str(img_path)
        p = Path(img_path)
        try:
            return p.relative_to(self.root).as_posix()
        except ValueError:
            pass
        if is_within(p, self.root):
            try:
                return p.resolve().relative_to(self.root.resolve()).as_posix()
            except (ValueError, OSError):
                pass
        return str(p)

    # -- reads ---------------------------------------------------------------

    def is_marked(self, img_path) -> bool:
        return self.key(img_path) in self.marks

    def marked_at(self, img_path) -> int | None:
        """Unix timestamp of the mark, or None if unmarked."""
        return self.marks.get(self.key(img_path))

    def count(self) -> int:
        return len(self.marks)

    # -- writes --------------------------------------------------------------

    def mark(self, img_path, when=None) -> bool:
        """Mark one image reviewed. Returns True if this changed anything.

        Re-marking an already-marked image is a no-op rather than a timestamp
        refresh, so idling on an image the dwell timer already caught does not
        dirty the store.
        """
        k = self.key(img_path)
        if k in self.marks:
            return False
        self.marks[k] = int(when if when is not None else time.time())
        self.dirty = True
        return True

    def mark_many(self, img_paths) -> int:
        """Mark several at once (a grid page). Returns how many changed."""
        return sum(1 for p in img_paths if self.mark(p))

    def unmark(self, img_path) -> bool:
        k = self.key(img_path)
        if k not in self.marks:
            return False
        del self.marks[k]
        self.dirty = True
        return True

    def toggle(self, img_path) -> bool:
        """Flip one image. Returns its new state."""
        if self.is_marked(img_path):
            self.unmark(img_path)
            return False
        self.mark(img_path)
        return True

    def clear(self) -> int:
        """Drop every mark. Returns how many were dropped."""
        n = len(self.marks)
        if n:
            self.marks = {}
            self.dirty = True
        return n

    # -- persistence ---------------------------------------------------------

    def load(self) -> None:
        """Read the store. A missing or damaged file loads as empty.

        Review progress is an aid, never the source of truth for anything on
        disk, so a corrupt file must degrade to "nothing reviewed yet" rather
        than block the user from opening their dataset.
        """
        self.marks = {}
        self.dirty = False
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        if not isinstance(raw, dict):
            return
        marks = raw.get("marks")
        if not isinstance(marks, dict):
            return
        for k, v in marks.items():
            try:
                self.marks[str(k)] = int(v)
            except (TypeError, ValueError):
                continue        # tolerate a hand-edited file

    def save(self, force=False) -> bool:
        """Write the store atomically. Returns True if it wrote."""
        if not (self.dirty or force):
            return False
        payload = json.dumps(
            {"version": SCHEMA_VERSION, "marks": self.marks},
            indent=1, sort_keys=True,
        )
        # `--dry-run` must no-op EVERY write, not just label writes.
        if _labelio.DRY_RUN:
            _labelio.DRY_RUN_LOG.append(("write", str(self.path), payload))
            self.dirty = False
            return False

        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(self.path.parent),
                                   prefix=f".{self.path.name}.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(payload)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, self.path)      # atomic on POSIX
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
        self.dirty = False
        return True
