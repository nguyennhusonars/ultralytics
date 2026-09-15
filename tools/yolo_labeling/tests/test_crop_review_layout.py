"""The review window's scan controls must actually be on screen.

A StringVar holds its text and a Progressbar holds its value whether or not the
widget is mapped, so state assertions pass happily against an invisible widget.
These check geometry instead: mapped, non-zero size, and inside the window.
"""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.gui, pytest.mark.legacy]


@pytest.fixture
def review(gui_app, ds):
    """A bad-annotation review window with a long selection caption set."""
    entry = {
        'img_path': ds.image("camA/0001.jpg"), 'line_idx': 0, 'cls_id': 0,
        'bbox_norm': (0.5, 0.5, 0.2, 0.3), 'pred_cls_id': 1, 'pred_conf': 0.8,
        'iou': 0.7, 'score': 0.8, 'kind': 'class_mismatch',
    }
    gui_app._open_crop_review_window(entries=[entry], target_class=None, mode='bad')
    gui_app.root.update_idletasks()
    gui_app._crop_review_window.update_idletasks()
    return gui_app


def _visible_in(win, widget):
    """True when `widget` is mapped and lies within `win`'s visible area."""
    if not widget.winfo_ismapped():
        return False
    if widget.winfo_width() <= 1 or widget.winfo_height() <= 1:
        return False
    x = widget.winfo_rootx() - win.winfo_rootx()
    y = widget.winfo_rooty() - win.winfo_rooty()
    return (x >= 0 and y >= 0
            and x + widget.winfo_width() <= win.winfo_width()
            and y + widget.winfo_height() <= win.winfo_height())


def test_progress_bar_is_on_screen(review):
    """It was pushed off the right edge behind a sentence of help text."""
    win = review._crop_review_window
    bar = review._bad_scan_progress
    assert _visible_in(win, bar), (
        f"progress bar at x={bar.winfo_rootx() - win.winfo_rootx()} "
        f"w={bar.winfo_width()} in a {win.winfo_width()}px window")


def test_cancel_button_is_on_screen(review):
    win = review._crop_review_window
    assert _visible_in(win, review._bad_scan_cancel_btn)


def test_status_bar_has_height(review):
    """Packed after a fill=BOTH/expand=True sibling, it got zero height."""
    win = review._crop_review_window
    status = [w for w in win.winfo_children()
              if isinstance(w, __import__('tkinter').ttk.Label)
              and str(w.cget('textvariable')) == str(review._crop_review_status_var)]
    assert status, "the status label exists"
    assert status[0].winfo_ismapped()
    assert status[0].winfo_height() > 1, "status bar was allotted no height"


def test_action_row_fits_the_window(review):
    """The reported symptom, measured directly.

    Tk does not push an over-full pack row off screen -- it compresses whatever
    can shrink (labels) and clips whatever cannot (buttons). So "is this widget
    at a visible coordinate" is the wrong question; the right one is whether the
    row's REQUESTED width fits the window at all. With the help sentence sharing
    the row this asked for 1422px inside a 1100px window, which is why the
    screenshot cut off mid-way through "Mark reviewed".
    """
    win = review._crop_review_window
    action = review._bad_scan_progress.master
    win.update_idletasks()

    need = action.winfo_reqwidth()
    have = win.winfo_width()
    assert need <= have, (
        f"action row wants {need}px in a {have}px window -- "
        f"{need - have}px of controls get clipped")


def test_scan_controls_survive_a_long_selection_caption(review):
    """Selecting a crop writes a long caption -- the controls must not move."""
    win = review._crop_review_window
    before = review._bad_scan_progress.winfo_rootx()

    review._crop_review_selected = 0
    review._crop_review_refresh_action_bar()
    win.update_idletasks()

    assert _visible_in(win, review._bad_scan_progress)
    assert review._bad_scan_progress.winfo_rootx() == before, \
        "the caption is on its own row, so it cannot shove the controls"


def test_bad_mode_hint_does_not_revert_to_filter_wording(review):
    """Deselecting used to restore the crop-filter help text in bad mode."""
    review._crop_review_selected = 0
    review._crop_review_refresh_action_bar()
    review._crop_review_selected = None
    review._crop_review_refresh_action_bar()

    hint = review._crop_review_sel_var.get()
    # Assert the marker that distinguishes the two modes, not the exact prose:
    # `K` exists only in bad-annotation mode, `auto-advances` only in filter mode.
    assert "K marks reviewed" in hint, hint
    assert "auto-advances" not in hint


def test_filter_mode_still_gets_its_own_hint(gui_app, ds):
    gui_app._open_crop_review_window(entries=[], target_class=0, mode='filter')
    gui_app.root.update_idletasks()
    gui_app._crop_review_selected = None
    gui_app._crop_review_refresh_action_bar()
    assert "auto-advances" in gui_app._crop_review_sel_var.get()
