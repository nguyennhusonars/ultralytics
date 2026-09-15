"""Marking images as reviewed.

Opening an image marks it, immediately. There is no dwell and no cancellation
window, so the properties worth pinning are the boundaries: an unreadable image
is not marked, marking never touches a label file, and the manual toggle wins
over the automatic mark rather than fighting it.
"""

from __future__ import annotations

import pytest

from fixtures import assert_only_changed, snapshot
from yolo_labeling.core.reviewmarks import FILENAME, ReviewMarks

pytestmark = [pytest.mark.gui, pytest.mark.legacy]


@pytest.fixture
def app(gui_app, ds):
    gui_app.dataset_root = ds.root
    gui_app.review_marks = ReviewMarks(ds.root / FILENAME, ds.root)
    gui_app.filtered_image_paths = [
        ds.image("camA/0001.jpg"), ds.image("camA/0002.jpg"), ds.image("camB/0001.jpg"),
    ]
    gui_app.current_image_index = 0
    return gui_app


# -- automatic marking -----------------------------------------------------


def test_opening_an_image_marks_it(app):
    app.load_image()
    assert app.review_marks.is_marked(app.current_image_path)


def test_walking_the_set_marks_every_image(app):
    app.load_image()
    app.next_image()
    app.next_image()
    assert app.review_marks.count() == 3


def test_an_unreadable_image_is_not_marked(app, ds):
    """`load_image` marks after the decode, so a corrupt frame is never
    silently recorded as gone through."""
    app.filtered_image_paths = [ds.image("bad/corrupt.jpg")]
    app.current_image_index = 0
    app.load_image()
    assert app.review_marks.count() == 0


def test_revisiting_does_not_re_dirty_the_store(app):
    app.load_image()
    app.flush_review_marks()
    assert app.review_marks.dirty is False

    app.load_image()
    assert app.review_marks.dirty is False, "a redundant mark scheduled a write"


# -- the manual toggle -----------------------------------------------------


def test_k_unmarks_an_image_that_opening_just_marked(app):
    """The toggle has to win: opening marks, so `k` is mainly for undoing that."""
    app.load_image()
    assert app.review_marks.is_marked(app.current_image_path)

    app.toggle_review_current()
    assert not app.review_marks.is_marked(app.current_image_path)

    app.toggle_review_current()
    assert app.review_marks.is_marked(app.current_image_path)


def test_unmarking_survives_staying_on_the_image(app):
    """Nothing pending can silently re-mark it afterwards."""
    app.load_image()
    app.toggle_review_current()
    app.root.update()
    app.root.update_idletasks()
    assert not app.review_marks.is_marked(app.current_image_path)


# -- other paths that mark -------------------------------------------------


def test_saving_labels_marks_the_image(app):
    app.load_image()
    app.toggle_review_current()          # start from unmarked
    app.user_boxes.append((0, 5, 5, 20, 20))
    app.save_labels()
    assert app.review_marks.is_marked(app.current_image_path)


def test_deleting_an_image_drops_its_mark(app):
    """Otherwise the key accumulates in the store forever."""
    app.load_image()
    target = app.current_image_path
    assert app.review_marks.is_marked(target)

    app.dialogs.answer = True
    app.delete_current_image()
    assert not app.review_marks.is_marked(target)


# -- grid view -------------------------------------------------------------


def test_showing_a_grid_page_marks_all_its_tiles(app):
    app.batch_size_var.set("2x2")
    app.toggle_batch_view()
    app.root.update_idletasks()

    paths = app._batch_visible_paths()
    assert paths
    assert all(app.review_marks.is_marked(p) for p in paths)


def test_k_toggles_a_whole_grid_page(app):
    app.batch_size_var.set("2x2")
    app.toggle_batch_view()
    app.root.update_idletasks()
    paths = list(app._batch_visible_paths())

    app.toggle_review_current()          # page is fully marked -> clears it
    assert not any(app.review_marks.is_marked(p) for p in paths)

    app.toggle_review_current()
    assert all(app.review_marks.is_marked(p) for p in paths)


# -- the dataset is never touched -----------------------------------------


def test_marking_never_writes_a_label_file(app, ds):
    before = snapshot(ds.root)
    app.load_image()
    app.next_image()
    app.flush_review_marks()
    assert_only_changed(before, snapshot(ds.root), set())


def test_marks_survive_a_reopen(app, ds):
    app.load_image()
    app.flush_review_marks()

    reopened = ReviewMarks(ds.root / FILENAME, ds.root)
    assert reopened.is_marked(app.current_image_path)
