"""The box-text toggle.

Captions sit on an opaque plate, which is what makes them readable -- and also
what makes them a problem on a small object, where the plate can be larger than
the box it names. Shift+T hides them; the boxes themselves stay.
"""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.gui, pytest.mark.legacy]


@pytest.fixture
def loaded(gui_app, ds):
    gui_app.filtered_image_paths = [ds.image("camA/0001.jpg")]
    gui_app.current_image_index = 0
    gui_app.load_image()
    gui_app.root.update_idletasks()
    return gui_app


def test_defaults_to_on(gui_app):
    assert gui_app.show_box_labels is True
    assert gui_app.show_box_text_var.get() is True


def test_toggle_flips_state_and_var_together(loaded):
    loaded._toggle_layer('text')
    assert loaded.show_box_labels is False
    assert loaded.show_box_text_var.get() is False

    loaded._toggle_layer('text')
    assert loaded.show_box_labels is True
    assert loaded.show_box_text_var.get() is True


def test_hiding_text_keeps_the_boxes(loaded):
    """The point of the toggle: lose the caption, keep the annotation."""
    n_boxes = len(loaded.label_boxes)
    assert n_boxes > 0, "fixture image must have labels for this to mean anything"

    loaded._toggle_layer('text')
    loaded.root.update_idletasks()

    assert loaded.show_label_boxes is True
    assert len(loaded.label_boxes) == n_boxes


def test_render_actually_changes(loaded):
    """Drive the real render both ways and count the captions it emits."""
    from yolo_labeling.core import draw as _draw

    seen = []
    real = _draw.draw_tag

    def spy(*a, **k):
        seen.append(a[1])
        return real(*a, **k)

    _draw.draw_tag = spy
    try:
        loaded.display_image()
        with_text = list(seen)
        seen.clear()

        loaded._toggle_layer('text')
        loaded.display_image()
        without_text = list(seen)
    finally:
        _draw.draw_tag = real

    assert with_text, "captions are drawn when the toggle is on"
    assert without_text == [], "no captions are drawn when it is off"


def test_toggle_refreshes_the_batch_grid(gui_app):
    """Tiles are downscaled hardest, so this is where captions hurt most."""
    gui_app.batch_size_var.set("2x2")
    gui_app.toggle_batch_view()
    gui_app.root.update_idletasks()
    assert gui_app.batch_mode is True

    calls = []
    real = gui_app.load_batch_images
    gui_app.load_batch_images = lambda *a, **k: (calls.append(1), real(*a, **k))[1]
    try:
        gui_app._toggle_layer('text')
    finally:
        gui_app.load_batch_images = real

    assert calls, "the grid must re-render, not wait for the next page turn"
    assert gui_app.show_box_labels is False


def test_shortcut_is_bound(gui_app):
    assert gui_app.root.bind("<T>"), "Shift+T must be bound"
