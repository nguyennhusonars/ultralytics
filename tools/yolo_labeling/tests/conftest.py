"""Shared fixtures and safety rails.

Two autouse fixtures matter a great deal here:

* `_chdir_tmp` -- the tool writes `class_config.json` relative to the CWD
  (monolith line 212). Without this, running the suite scribbles config files
  into the repo.
* `dialogs` -- `messagebox.*` and `filedialog.*` block forever without a
  display, and ~40 call sites use them. We swap in recorders and, at teardown,
  assert no error dialog was raised. A silent `showerror` is precisely how a
  failed save hides today, so this turns that into a test failure.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from fixtures import build_dataset  # noqa: E402

MONOLITH = Path(__file__).resolve().parents[2] / "yolo_labeling_tool.py"


# --------------------------------------------------------------------------
# safety rails
# --------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _chdir_tmp(tmp_path, monkeypatch):
    """Never let the tool write config into the repo."""
    monkeypatch.chdir(tmp_path)


class DialogRecorder:
    """Stand-in for `tkinter.messagebox` / `filedialog`.

    Records every call instead of opening a window. `answer` / `path` control
    what interactive prompts return.
    """

    def __init__(self):
        self.errors: list[tuple[str, str]] = []
        self.warnings: list[tuple[str, str]] = []
        self.infos: list[tuple[str, str]] = []
        self.asks: list[tuple[str, str]] = []
        self.answer = True          # what askyesno/askokcancel return
        self.path = ""              # what the file dialogs return

    # -- messagebox surface --
    def showerror(self, title="", message="", **kw):
        self.errors.append((title, message))

    def showwarning(self, title="", message="", **kw):
        self.warnings.append((title, message))

    def showinfo(self, title="", message="", **kw):
        self.infos.append((title, message))

    def askyesno(self, title="", message="", **kw):
        self.asks.append((title, message))
        return self.answer

    askokcancel = askyesno
    askretrycancel = askyesno

    def askquestion(self, title="", message="", **kw):
        self.asks.append((title, message))
        return "yes" if self.answer else "no"

    # -- filedialog surface --
    def askopenfilename(self, **kw):
        return self.path

    def askdirectory(self, **kw):
        return self.path

    def asksaveasfilename(self, **kw):
        return self.path


@pytest.fixture
def dialogs():
    """A DialogRecorder. Attach it to a module with `patch_dialogs`."""
    return DialogRecorder()


@pytest.fixture
def patch_dialogs(dialogs, monkeypatch):
    """Return a callable that redirects a module's dialog calls to the recorder."""

    def _patch(module):
        # Dialogs are raised from the mixin modules too, not just wherever
        # YOLOLabelingTool happens to be defined -- patch every one of them.
        targets = [module]
        pkg = sys.modules.get("yolo_labeling.ui")
        if pkg is not None:
            targets += [m for name, m in sys.modules.items()
                        if name.startswith("yolo_labeling.ui.")]
        for target in targets:
            monkeypatch.setattr(target, "messagebox", dialogs, raising=False)
            monkeypatch.setattr(target, "filedialog", dialogs, raising=False)
        return dialogs

    return _patch


# --------------------------------------------------------------------------
# dataset
# --------------------------------------------------------------------------


@pytest.fixture
def ds(tmp_path):
    """A freshly built edge-case dataset under `tmp_path/ds`."""
    return build_dataset(tmp_path / "ds")


# --------------------------------------------------------------------------
# the monolith, for characterization tests
# --------------------------------------------------------------------------


@pytest.fixture(scope="session")
def monolith():
    """The module exposing `YOLOLabelingTool`, session-scoped (import costs ~2.5s).

    `YLT_IMPL=package` runs the whole suite against the split package instead of
    the original single file. Running it both ways is what proves the split is
    behaviour-preserving; once the monolith is deleted, `package` is the only
    option and this fixture quietly becomes a no-op indirection.
    """
    if os.environ.get("YLT_IMPL", "monolith") == "package" or not MONOLITH.exists():
        from yolo_labeling import app as mod
        return mod
    spec = importlib.util.spec_from_file_location("yolo_labeling_tool", MONOLITH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["yolo_labeling_tool"] = mod
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------
# LegacyProbe -- drive the monolith's data paths without a display
# --------------------------------------------------------------------------


class LegacyProbe:
    """A bare `YOLOLabelingTool` carrying only the attributes the data paths read.

    `load_labels` and `save_labels` touch a small, known set of attributes and
    one helper (`update_status`). Building the instance with `object.__new__`
    skips `setup_gui()` entirely, so characterization runs in the fast suite
    with no Tk, no display and no 2.5 s widget construction.

    `open()` mirrors the data half of `load_image()`: set the path, decode the
    image, then parse labels. It deliberately does NOT clear `user_boxes` --
    that omission is the bug under test.
    """

    def __init__(self, mod, label_folder=None, per_image=None):
        app = object.__new__(mod.YOLOLabelingTool)
        app.label_folder = str(label_folder) if label_folder else None
        app.per_image_label_folders = dict(per_image or {})
        app.label_boxes = []
        app.user_boxes = []
        app.model_boxes = []
        app.pretrained_model = None
        app.pred_cache = None
        app.prediction_conf = 0.25
        # Review marks: a REAL rootless store, not None, so a headless test can
        # assert "browsing marked nothing" against the actual object rather
        # than a hole. A rootless store never touches disk.
        from yolo_labeling.core.reviewmarks import ReviewMarks
        app.dataset_root = None
        app.review_marks = ReviewMarks(Path(os.devnull), None)
        app._review_flush_after = None
        app.filter_by_review_enabled = False
        app.filter_review_state = 'unreviewed'
        app.filter_by_relsize_enabled = False
        from yolo_labeling.core.classmap import PredictionClassMap
        app.prediction_class_map = PredictionClassMap()
        app.labels_cache = {}
        app.model_predictions_cache = {}
        app.image_dimensions_cache = {}
        app.current_image = None
        app.current_image_path = None
        app.status = []
        app.selected_box_index = None
        app.selected_box_source = None
        app.filtered_image_paths = []
        app.current_image_index = -1
        app.history_undo = []
        app.history_redo = []
        app.highlight_overlaps = False
        app.auto_fit_on_load = False
        app.batch_mode = False
        # Headless stubs for the view half of load_image().
        app.update_status = lambda *a, **k: app.status.append(a[0] if a else "")
        app.update_counter = lambda *a, **k: None
        app.display_image = lambda *a, **k: None
        app.fit_to_window = lambda *a, **k: None
        self.app = app

    # -- navigation -------------------------------------------------------
    def open(self, path):
        """Navigate to `path` through the app's REAL `load_image()`.

        Driving the actual method (rather than re-implementing its data half)
        is what makes the navigation tests honest: if `load_image` forgets to
        reset per-image state, these tests see it.
        """
        path = Path(path)
        self.app.filtered_image_paths = [path]
        self.app.current_image_index = 0
        self.app.load_image()
        return self

    def save(self):
        self.app.save_labels()
        return self

    # -- convenience ------------------------------------------------------
    @property
    def label_boxes(self):
        return self.app.label_boxes

    @property
    def user_boxes(self):
        return self.app.user_boxes

    def add_user_box(self, cls_id, x1, y1, x2, y2):
        self.app.user_boxes.append((cls_id, x1, y1, x2, y2))
        return self


@pytest.fixture
def legacy(monolith, patch_dialogs):
    """Factory for `LegacyProbe`s, with dialogs already redirected to a recorder."""
    recorder = patch_dialogs(monolith)

    def _make(label_folder=None, per_image=None):
        probe = LegacyProbe(monolith, label_folder=label_folder, per_image=per_image)
        probe.dialogs = recorder
        return probe

    _make.dialogs = recorder
    return _make


# --------------------------------------------------------------------------
# real Tk application, for the handful of tests that need a display
# --------------------------------------------------------------------------


@pytest.fixture
def gui_app(monolith, patch_dialogs, ds):
    """A fully constructed `YOLOLabelingTool` on a real (Xvfb) display.

    Dialogs are redirected before construction because `load_classes_from_yaml`
    pops a modal `showinfo` mid-startup, which would block forever headlessly.
    No model is passed, so no inference runs.
    """
    tk = pytest.importorskip("tkinter")
    recorder = patch_dialogs(monolith)
    try:
        root = tk.Tk()
    except tk.TclError as exc:                      # pragma: no cover
        pytest.skip(f"no display available: {exc}")
    root.geometry("1200x800")
    app = monolith.YOLOLabelingTool(root, yaml_path=str(ds.yaml))
    root.update_idletasks()
    app.dialogs = recorder
    try:
        yield app
    finally:
        try:
            root.destroy()
        except Exception:
            pass
