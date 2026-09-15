"""Which class pairs count as an overlap.

Different-class overlap is one object labelled as two things. Same-class
overlap is one object labelled twice -- the classic duplicate-box error, which
the filter could not find at all until the scope existed.

The filter and the red highlighter each used to hardcode `cls1 != cls2`
independently, so these also pin that the two agree.
"""

from __future__ import annotations

import pytest

from yolo_labeling.core import geometry as _geom
from yolo_labeling.ui import filters as _filters

#: Two boxes on top of each other -- IoU 1.0 whatever the classes are.
BOX = (10.0, 10.0, 50.0, 50.0)
FAR = (200.0, 200.0, 240.0, 240.0)


def _probe(scope, boxes):
    """A bare FiltersMixin carrying only what the overlap branch reads."""
    p = type("P", (_filters.FiltersMixin,), {})()
    # `calculate_iou` is defined on the app class, not the mixin; it delegates
    # straight to core.geometry, so the probe can borrow the same function.
    p.calculate_iou = staticmethod(_geom.iou_xyxy)
    p.labels_cache = {"img": [(c, *b, 0) for c, b in boxes]}
    p.image_dimensions_cache = {"img": (640, 480)}
    p.filter_overlap_scope = scope
    p.filter_overlap_threshold = 0.5
    p.filter_by_overlap_enabled = True
    for flag in ("filter_by_class_enabled", "filter_by_size_enabled",
                 "filter_by_iou_enabled", "filter_by_class_presence_enabled",
                 "filter_by_relsize_enabled"):
        setattr(p, flag, False)
    return p


def _passes(scope, boxes):
    return _probe(scope, boxes).image_passes_filters("img", "unused.txt")


DIFFERENT = [(0, BOX), (1, BOX)]      # one object, two classes
SAME = [(0, BOX), (0, BOX)]           # one object, labelled twice
APART = [(0, BOX), (1, FAR)]          # no overlap at all


# -- the scopes ------------------------------------------------------------


@pytest.mark.parametrize("scope,boxes,expected", [
    ('different', DIFFERENT, True),
    ('different', SAME,      False),   # the old behaviour: duplicates invisible
    ('same',      SAME,      True),    # what this change unlocks
    ('same',      DIFFERENT, False),
    ('any',       DIFFERENT, True),
    ('any',       SAME,      True),
])
def test_scope_selects_the_right_pairs(scope, boxes, expected):
    assert _passes(scope, boxes) is expected


@pytest.mark.parametrize("scope", ['different', 'same', 'any'])
def test_boxes_that_do_not_touch_never_match(scope):
    assert _passes(scope, APART) is False


def test_the_default_scope_is_the_old_behaviour():
    """Existing setups must not change meaning on upgrade."""
    p = _probe('different', SAME)
    del p.filter_overlap_scope
    assert p.overlap_pair_counts(0, 1) is True
    assert p.overlap_pair_counts(0, 0) is False


def test_threshold_still_applies():
    """Scope decides which pairs are considered, not whether IoU matters."""
    p = _probe('same', SAME)
    p.filter_overlap_threshold = 1.01      # unreachable
    assert p.image_passes_filters("img", "unused.txt") is False


# -- the filter and the highlighter must agree -----------------------------


@pytest.mark.gui
@pytest.mark.legacy
@pytest.mark.parametrize("scope,expect_hits", [
    ('different', False),
    ('same', True),
    ('any', True),
])
def test_the_red_highlighter_follows_the_same_scope(gui_app, scope, expect_hits):
    """Selecting an image the highlighter then draws nothing on is the bug
    that having two independent copies of this rule invited."""
    gui_app.filter_overlap_scope = scope
    gui_app.filter_overlap_threshold = 0.5
    gui_app.label_boxes = [(0, *BOX), (0, *BOX)]      # duplicate, same class
    gui_app.detect_current_overlaps()
    assert bool(gui_app.overlapping_boxes) is expect_hits


@pytest.mark.gui
@pytest.mark.legacy
def test_changing_the_scope_refreshes_the_highlighting(gui_app):
    gui_app.highlight_overlaps = True
    gui_app.filter_overlap_threshold = 0.5
    gui_app.label_boxes = [(0, *BOX), (0, *BOX)]

    gui_app.filter_overlap_scope_var.set('same')
    gui_app._on_overlap_scope_change()
    assert gui_app.filter_overlap_scope == 'same'
    assert gui_app.overlapping_boxes == [0, 1]

    gui_app.filter_overlap_scope_var.set('different')
    gui_app._on_overlap_scope_change()
    assert gui_app.overlapping_boxes == []
