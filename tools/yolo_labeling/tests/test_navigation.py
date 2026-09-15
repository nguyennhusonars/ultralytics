"""Navigation invariants -- principally that per-image state never leaks.

The `user_boxes` leak was not a missing line at one call site; it was a
misplaced responsibility. Clearing lived at 3 of `load_image()`'s 12 callers, so
every new navigation path added over time was leaky by default. These tests pin
the fix at the funnel, not at the callers.
"""

from __future__ import annotations

import inspect

import pytest

from fixtures import CAMA_0001, CAMB_0001, assert_only_changed, read_exact, snapshot

pytestmark = pytest.mark.legacy


# -- the funnel ------------------------------------------------------------


def test_load_image_clears_user_boxes_whoever_calls_it(ds, legacy):
    """Called directly, with no call-site cooperation at all."""
    p = legacy()
    p.open(ds.image("camA/0001.jpg"))
    p.add_user_box(0, 5, 5, 20, 20)
    assert p.user_boxes

    p.app.filtered_image_paths = [ds.image("camB/0001.jpg")]
    p.app.current_image_index = 0
    p.app.load_image()                       # the funnel, invoked bare
    assert p.user_boxes == []


def test_clearing_lives_inside_load_image_not_at_call_sites(monolith):
    """Structural guard: keep the reset where every path must pass through it.

    If someone moves this back out to the callers, the next navigation feature
    silently reintroduces the leak.
    """
    src = inspect.getsource(monolith.YOLOLabelingTool.load_image)
    assert "self.user_boxes = []" in src


def test_per_image_history_is_reset_too(ds, legacy):
    p = legacy()
    p.open(ds.image("camA/0001.jpg"))
    p.app.history_undo.append(("fake", []))
    p.open(ds.image("camB/0001.jpg"))
    assert p.app.history_undo == []


def test_selection_state_is_reset(ds, legacy):
    p = legacy()
    p.open(ds.image("camA/0001.jpg"))
    p.app.selected_box_index = 0
    p.open(ds.image("camB/0001.jpg"))
    assert p.app.selected_box_index is None


# -- the practical failure mode --------------------------------------------


def test_same_stem_in_two_folders_never_cross_writes(ds, legacy):
    """camA/0001 and camB/0001 share a stem -- how this bug destroyed data."""
    before = snapshot(ds.root)
    p = legacy()
    p.open(ds.image("camA/0001.jpg"))
    p.add_user_box(0, 5, 5, 20, 20)
    p.save()
    p.open(ds.image("camB/0001.jpg"))
    p.save()
    assert read_exact(ds.label("camB/0001")) == CAMB_0001
    assert_only_changed(before, snapshot(ds.root), {"camA/labels/0001.txt"})


def test_browsing_without_editing_writes_nothing_at_all(ds, legacy):
    """Pure navigation must leave the dataset byte-identical."""
    before = snapshot(ds.root)
    p = legacy()
    for rel in ("camA/0001.jpg", "camB/0001.jpg", "seg/poly1.jpg",
                "camA/0002.jpg", "camB/wide.jpg", "seg/mixed.jpg"):
        p.open(ds.image(rel))
        p.save()                              # even an explicit save
    assert_only_changed(before, snapshot(ds.root), set())


def test_an_edit_then_navigating_away_leaves_the_edit_in_the_right_file(ds, legacy):
    p = legacy()
    p.open(ds.image("camA/0001.jpg"))
    p.add_user_box(3, 1, 1, 30, 30)
    p.save()
    edited = read_exact(ds.label("camA/0001"))
    assert len(edited.strip().splitlines()) == 3

    p.open(ds.image("camB/0001.jpg"))
    p.save()
    assert read_exact(ds.label("camA/0001")) == edited      # still there
    assert read_exact(ds.label("camB/0001")) == CAMB_0001   # untouched


def test_returning_to_an_image_reloads_from_disk(ds, legacy):
    p = legacy()
    p.open(ds.image("camA/0001.jpg"))
    n = len(p.label_boxes)
    p.open(ds.image("camB/0001.jpg"))
    p.open(ds.image("camA/0001.jpg"))
    assert len(p.label_boxes) == n
    assert read_exact(ds.label("camA/0001")) == CAMA_0001


def test_navigating_onto_an_unreadable_image_does_not_poison_the_next_save(ds, legacy):
    """The two data-loss bugs in combination: bad image, then a real one."""
    before = snapshot(ds.root)
    p = legacy()
    p.open(ds.image("camA/0001.jpg"))
    p.open(ds.image("bad/corrupt.jpg"))       # unreadable
    p.save()                                  # must refuse
    p.open(ds.image("camB/0001.jpg"))
    p.save()
    assert_only_changed(before, snapshot(ds.root), set())
