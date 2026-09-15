"""The review-state filter and the relative width/height filter.

The two filters are tested together because the interesting property is what
separates them: one needs label files and one must NOT, and getting that
backwards silently deletes images from the working set.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from fixtures import assert_only_changed, snapshot
from yolo_labeling.core.reviewmarks import FILENAME, ReviewMarks
from yolo_labeling.ui import filters as _filters

pytestmark = [pytest.mark.gui, pytest.mark.legacy]


@pytest.fixture
def app(gui_app, ds):
    """A GUI app with a real, dataset-rooted review store."""
    gui_app.dataset_root = ds.root
    gui_app.review_marks = ReviewMarks(ds.root / FILENAME, ds.root)
    return gui_app


def _probe(ds):
    """A bare FiltersMixin carrying only what `image_passes_filters` reads."""
    p = type("P", (_filters.FiltersMixin,), {})()
    p.labels_cache = {}
    p.image_dimensions_cache = {}
    p.model_predictions_cache = {}
    p.filter_by_class_enabled = False
    p.filter_by_size_enabled = False
    p.filter_by_iou_enabled = False
    p.filter_by_class_presence_enabled = False
    p.filter_by_overlap_enabled = False
    p.filter_by_relsize_enabled = True
    p.filter_class_id = 0
    p.filter_min_size, p.filter_max_size = 0, 100000
    p.filter_min_w_pct, p.filter_max_w_pct = 0.0, 100.0
    p.filter_min_h_pct, p.filter_max_h_pct = 0.0, 100.0
    return p


def _passes(p, ds, rel):
    return p.image_passes_filters(ds.image(f"{rel}.jpg"), ds.label(rel))


# -- relative size ---------------------------------------------------------
#
# camA/0001 is 64x48 with two boxes: 20% x 30% and 10% x 10%.
# camB/wide  is 100x10 with one full-frame box: 100% x 100%.


def test_a_full_frame_box_is_found_by_a_high_width_floor(ds):
    p = _probe(ds)
    p.filter_min_w_pct = 90.0
    assert _passes(p, ds, "camB/wide")
    assert not _passes(p, ds, "camA/0001"), "widest box there is 20%"


def test_a_low_ceiling_finds_only_the_small_boxes(ds):
    p = _probe(ds)
    p.filter_max_w_pct = 15.0
    p.filter_max_h_pct = 15.0
    assert _passes(p, ds, "camA/0001"), "the 10%x10% box qualifies"
    assert not _passes(p, ds, "camB/wide")


def test_width_and_height_constrain_independently(ds):
    """A 100x10 image's full-frame box is 100% on both axes."""
    p = _probe(ds)
    p.filter_min_w_pct = 90.0
    p.filter_max_h_pct = 50.0
    assert not _passes(p, ds, "camB/wide"), "height is 100%, above the ceiling"


def test_every_condition_must_hold_on_the_same_box(ds):
    """The property most likely to be implemented wrong.

    camA/0001 has a 20%x30% box and a 10%x10% box. `min_w=15` is satisfied only
    by the first, `max_h=15` only by the second. Neither box satisfies both, so
    the image must NOT pass -- matching how class+area already behave.
    """
    p = _probe(ds)
    p.filter_min_w_pct = 15.0
    p.filter_max_h_pct = 15.0
    assert not _passes(p, ds, "camA/0001")


def test_bounds_are_inclusive(ds):
    p = _probe(ds)
    p.filter_min_w_pct = 100.0
    p.filter_min_h_pct = 100.0
    assert _passes(p, ds, "camB/wide"), "exactly 100% must pass a 100% floor"


def test_an_image_with_no_boxes_cannot_match(ds):
    p = _probe(ds)
    assert not _passes(p, ds, "camA/0002"), "empty label file"


def test_an_unreadable_image_is_rejected_not_fatal(ds):
    """No dimensions means relative size is unknowable."""
    p = _probe(ds)
    assert not _passes(p, ds, "bad/corrupt")


# -- the requires_labels trap ---------------------------------------------


def test_review_filter_keeps_images_that_have_no_label_file(app, ds):
    """The headline hazard.

    `camA/0003.jpg` has no .txt at all. Listing the review filter in
    `requires_labels` would route it down the label branch and drop it -- and an
    unlabelled image is exactly what "not yet reviewed" should surface.
    """
    app.filter_by_review_enabled = True
    app.filter_review_state = 'unreviewed'
    app.apply_filters()
    assert ds.image("camA/0003.jpg") in app.filtered_image_paths


def test_relative_size_filter_does_drop_unlabelled_images(app, ds):
    """The mirror image: this one IS label-based, so the same file goes."""
    app.filter_by_relsize_enabled = True
    app.apply_filters()
    assert ds.image("camA/0003.jpg") not in app.filtered_image_paths


# -- review state selection ------------------------------------------------


def test_unreviewed_hides_what_was_marked(app, ds):
    marked = ds.image("camA/0001.jpg")
    app.review_marks.mark(marked)

    app.filter_by_review_enabled = True
    app.filter_review_state = 'unreviewed'
    app.apply_filters()
    assert marked not in app.filtered_image_paths
    assert ds.image("camB/0001.jpg") in app.filtered_image_paths


def test_reviewed_shows_only_what_was_marked(app, ds):
    marked = ds.image("camA/0001.jpg")
    app.review_marks.mark(marked)

    app.filter_by_review_enabled = True
    app.filter_review_state = 'reviewed'
    app.apply_filters()
    assert app.filtered_image_paths == [marked]


def test_the_filter_is_inert_while_its_checkbox_is_off(app, ds):
    app.review_marks.mark(ds.image("camA/0001.jpg"))
    app.filter_by_review_enabled = False
    app.filter_review_state = 'reviewed'
    app.apply_filters()
    assert len(app.filtered_image_paths) == len(app.image_paths)


def test_review_filter_combines_with_the_filename_filter(app, ds):
    app.review_marks.mark(ds.image("camA/0001.jpg"))
    app.review_marks.mark(ds.image("camB/0001.jpg"))

    app.filter_by_review_enabled = True
    app.filter_review_state = 'reviewed'
    app.filter_by_name_enabled = True
    app.filter_name_text = "0001"
    app.apply_filters()

    names = {p.name for p in app.filtered_image_paths}
    assert names == {"0001.jpg"}
    assert len(app.filtered_image_paths) == 2, "one per camera folder"


def test_filtering_writes_no_label_files(app, ds):
    """A review mark must never reach the dataset's labels."""
    before = snapshot(ds.root)
    app.review_marks.mark(ds.image("camA/0001.jpg"))
    app.filter_by_review_enabled = True
    app.apply_filters()
    app.flush_review_marks()
    assert_only_changed(before, snapshot(ds.root), set())
    assert (ds.root / FILENAME).exists(), "but the store itself was written"


def test_the_store_lands_beside_data_yaml(app, ds):
    app.review_marks.mark(ds.image("camA/0001.jpg"))
    app.flush_review_marks()
    assert (ds.root / FILENAME).parent == Path(ds.yaml).parent
