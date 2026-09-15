"""Pin CURRENTLY-CORRECT behaviour of the monolith.

These are the invariants the refactor must preserve. They are not aspirational
-- every assertion here passes against the pre-refactor code today. Anything
that is broken belongs in `test_bugs.py` instead.

Deleted in Phase 7 once the equivalent unit tests exist against the package.
"""

from __future__ import annotations

import pytest

from fixtures import assert_only_changed, read_exact, snapshot

pytestmark = pytest.mark.legacy


# -- label parsing ---------------------------------------------------------


def test_wellformed_labels_parse_to_pixel_boxes(ds, legacy):
    """64x48 image; `0 0.5 0.5 0.2 0.3` -> x 0.4..0.6, y 0.35..0.65.

    BLESSED DIFF (Phase 3c): these were `int()` truncations before -- x1 was 25
    (floor of 25.6) rather than 26. Truncation floored both edges, so every box
    shrank by up to a pixel. Values here are now `round()`.
    """
    p = legacy().open(ds.image("camA/0001.jpg"))
    assert len(p.label_boxes) == 2
    cls, x1, y1, x2, y2 = p.label_boxes[0]
    assert int(cls) == 0
    assert (x1, x2) == (round(0.4 * 64), round(0.6 * 64))    # was int(): (25, 38)
    assert (y1, y2) == (round(0.35 * 48), round(0.65 * 48))


def test_empty_label_file_yields_no_boxes(ds, legacy):
    p = legacy().open(ds.image("camA/0002.jpg"))
    assert p.label_boxes == []


def test_missing_label_file_yields_no_boxes(ds, legacy):
    p = legacy().open(ds.image("camA/0003.jpg"))
    assert p.label_boxes == []


def test_loading_populates_labels_cache(ds, legacy):
    p = legacy().open(ds.image("camA/0001.jpg"))
    key = str(ds.image("camA/0001.jpg"))
    assert key in p.app.labels_cache
    assert len(p.app.labels_cache[key]) == 2
    assert len(p.app.labels_cache[key][0]) == 6, "cache rows carry box area"


# -- label-directory resolution (current behaviour, divergence and all) -----


def test_explicit_label_folder_wins(ds, legacy):
    p = legacy(label_folder=ds.root / "camB" / "labels")
    p.open(ds.image("camA/0001.jpg"))          # camA image, camB labels
    assert len(p.label_boxes) == 1              # camB/0001.txt has one box


def test_sibling_labels_dir_is_found(ds, legacy):
    """`<parent>/images/x.jpg` resolves to `<parent>/labels/x.txt`."""
    p = legacy().open(ds.image("camA/0001.jpg"))
    assert len(p.label_boxes) == 2


def test_per_image_mapping_is_consulted(ds, legacy):
    mapping = {str(ds.root / "camA" / "images"): str(ds.root / "camB" / "labels")}
    p = legacy(per_image=mapping)
    p.open(ds.image("camA/0001.jpg"))
    assert len(p.label_boxes) == 1


# -- saving ----------------------------------------------------------------


def test_save_is_idempotent_from_the_second_pass(ds, legacy):
    """int()-truncation settles after two round trips, then never moves again.

    Verified numerically: a 14.4px box becomes 15px then 14px and stops. The
    loss is bounded and under one pixel -- but it is NOT a single pass, so the
    baseline for this assertion has to be pass 2.
    """
    p = legacy()
    for _ in range(2):                       # converges after TWO passes, not one
        p.open(ds.image("camA/0001.jpg"))
        p.save()
    settled = read_exact(ds.label("camA/0001"))
    for _ in range(15):
        p.open(ds.image("camA/0001.jpg"))
        p.save()
    assert read_exact(ds.label("camA/0001")) == settled, "no unbounded drift"


def test_save_combines_label_and_user_boxes(ds, legacy):
    p = legacy()
    p.open(ds.image("camA/0001.jpg"))
    p.add_user_box(2, 10, 10, 20, 20)
    p.save()
    assert len(read_exact(ds.label("camA/0001")).strip().splitlines()) == 3


def test_save_creates_a_labels_dir_when_none_exists(ds, legacy):
    """`frames/` has no labels dir; saving creates `frames/labels/`."""
    p = legacy()
    p.open(ds.root / "frames" / "x0001.jpg")
    p.add_user_box(0, 8, 8, 24, 24)
    p.save()
    assert (ds.root / "frames" / "labels" / "x0001.txt").exists()


def test_save_updates_the_cache(ds, legacy):
    p = legacy()
    p.open(ds.image("camA/0001.jpg"))
    p.add_user_box(3, 1, 1, 9, 9)
    p.save()
    assert len(p.app.labels_cache[str(ds.image("camA/0001.jpg"))]) == 3


def test_save_without_a_current_image_reports_an_error(ds, legacy):
    p = legacy()
    p.save()
    assert p.dialogs.errors, "an error dialog is raised rather than silence"


# -- read-only operations must never write ---------------------------------


def test_merely_opening_images_writes_nothing(ds, legacy):
    before = snapshot(ds.root)
    p = legacy()
    for rel in ("camA/0001.jpg", "camB/0001.jpg", "seg/poly1.jpg", "camA/0002.jpg"):
        p.open(ds.image(rel))
    assert_only_changed(before, snapshot(ds.root), set())


def test_saving_touches_only_the_current_images_label_file(ds, legacy):
    before = snapshot(ds.root)
    p = legacy()
    p.open(ds.image("camA/0001.jpg"))
    p.add_user_box(1, 2, 2, 12, 12)
    p.save()
    assert_only_changed(before, snapshot(ds.root), {"camA/labels/0001.txt"})
