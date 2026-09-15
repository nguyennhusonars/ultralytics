"""Accepting model predictions, selectively.

`a` used to accept every prediction on the image -- including classes the user
had hidden and therefore could not see or review. Acceptance now follows what
is actually on screen, and `Shift+A` narrows it to one class.
"""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.gui, pytest.mark.legacy]

# dataset fixture classes: 0 person, 1 car, 2 bus_m, 3 bus_l
PREDS = [
    (0, 1, 1, 10, 10, 0.9),
    (1, 2, 2, 12, 12, 0.8),
    (1, 3, 3, 13, 13, 0.7),
    (2, 4, 4, 14, 14, 0.6),
]


@pytest.fixture
def app(gui_app):
    import numpy as np

    gui_app.current_image = np.zeros((48, 64, 3), np.uint8)
    gui_app.model_boxes = list(PREDS)
    gui_app.user_boxes = []
    gui_app.label_boxes = []
    gui_app.class_visibility = {c: True for c in gui_app.class_info}
    gui_app.save_labels = lambda *a, **k: None      # keep the test off disk
    return gui_app


def _classes(app):
    return sorted(b[0] for b in app.user_boxes)


def test_accept_all_takes_every_visible_prediction(app):
    app.accept_all_predictions()
    assert _classes(app) == [0, 1, 1, 2]


def test_hidden_classes_are_not_accepted(app):
    """The bug: `a` used to write boxes the user could not see."""
    app.class_visibility[1] = False
    app.accept_all_predictions()
    assert _classes(app) == [0, 2], "the two hidden 'car' predictions were skipped"


def test_accepting_one_class_only(app):
    app.accept_all_predictions(cls_id=1)
    assert _classes(app) == [1, 1]


def test_shift_a_accepts_the_currently_selected_class(app):
    app.current_class = 2
    app.accept_predictions_for_current_class()
    assert _classes(app) == [2]


def test_explicit_class_overrides_visibility(app):
    """Naming a class is a deliberate act; hiding it should not veto that."""
    app.class_visibility[1] = False
    app.accept_all_predictions(cls_id=1)
    assert _classes(app) == [1, 1]


def test_classes_absent_from_the_registry_are_never_accepted(app):
    app.model_boxes = list(PREDS) + [(99, 5, 5, 15, 15, 0.5)]
    app.accept_all_predictions()
    assert 99 not in _classes(app)


def test_confidence_is_not_written_into_labels(app):
    app.accept_all_predictions()
    assert all(len(b) == 5 for b in app.user_boxes)


def test_nothing_to_accept_reports_and_does_not_touch_history(app):
    app.class_visibility = {c: False for c in app.class_info}
    depth = len(app.history_undo)
    app.accept_all_predictions()
    assert app.user_boxes == []
    assert len(app.history_undo) == depth, "an empty accept must not be undoable"


def test_status_reports_how_many_were_skipped(app):
    app.class_visibility[1] = False
    msgs = []
    app.update_status = msgs.append
    app.accept_all_predictions()
    assert any("2 skipped" in str(m) for m in msgs), msgs


def test_accept_is_undoable(app):
    app.accept_all_predictions()
    assert app.user_boxes
    app.undo()
    assert app.user_boxes == []
