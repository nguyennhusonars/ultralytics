"""Predictions and bulk acceptance in the grid view.

Batch acceptance writes to MANY label files in one action, so these tests lean
hard on snapshot diffing: the set of files that changed must be exactly the set
we intended to change.
"""

from __future__ import annotations

import pytest

from fixtures import assert_only_changed, read_exact, snapshot

pytestmark = [pytest.mark.gui, pytest.mark.legacy]


@pytest.fixture
def grid(gui_app, ds):
    gui_app.batch_size_var.set("2x2")
    gui_app.toggle_batch_view()
    gui_app.root.update_idletasks()
    gui_app.pretrained_model = None          # keep inference out of these tests
    return gui_app


def _seed(app, path, boxes):
    """Seed the raw prediction cache.

    The cache holds what the MODEL said, untranslated -- class mapping is
    applied on read. With no model loaded the app's map is empty and would drop
    everything, so switch translation off to make these ids pass through.
    """
    app.prediction_class_map.enabled = False
    app.model_predictions_cache[str(path)] = boxes


def test_visible_paths_track_the_page(grid):
    paths = grid._batch_visible_paths()
    assert paths, "the grid reports which images it is showing"
    assert len(paths) <= len(grid.batch_canvases)


def test_accept_writes_only_the_intended_files(grid, ds):
    before = snapshot(ds.root)
    target = grid._batch_visible_paths()[0]
    _seed(grid, target, [(0, 5, 5, 20, 20, 0.9)])
    grid.dialogs.answer = True
    grid.accept_batch_predictions()

    rel = str(grid._resolver().resolve(target).relative_to(ds.root))
    assert_only_changed(before, snapshot(ds.root), {rel})


def test_accept_appends_rather_than_replacing(grid, ds):
    target = grid._batch_visible_paths()[0]
    label = grid._resolver().resolve(target)
    original = read_exact(label) if label.exists() else ""
    n_before = len([ln for ln in original.splitlines() if ln.strip()])

    _seed(grid, target, [(0, 5, 5, 20, 20, 0.9), (1, 6, 6, 21, 21, 0.8)])
    grid.dialogs.answer = True
    grid.accept_batch_predictions()

    after = read_exact(label)
    assert len(after.strip().splitlines()) == n_before + 2
    for line in original.strip().splitlines():
        assert line in after, "pre-existing labels are preserved verbatim"


def test_declining_the_confirmation_writes_nothing(grid, ds):
    before = snapshot(ds.root)
    _seed(grid, grid._batch_visible_paths()[0], [(0, 5, 5, 20, 20, 0.9)])
    grid.dialogs.answer = False
    grid.accept_batch_predictions()
    assert_only_changed(before, snapshot(ds.root), set())


def test_hidden_classes_are_excluded(grid, ds):
    before = snapshot(ds.root)
    _seed(grid, grid._batch_visible_paths()[0], [(0, 5, 5, 20, 20, 0.9)])
    grid.class_visibility[0] = False
    grid.dialogs.answer = True
    grid.accept_batch_predictions()
    assert_only_changed(before, snapshot(ds.root), set())


def test_nothing_cached_means_nothing_written(grid, ds):
    before = snapshot(ds.root)
    grid.model_predictions_cache = {}
    grid.dialogs.answer = True
    grid.accept_batch_predictions()
    assert_only_changed(before, snapshot(ds.root), set())
    assert not grid.dialogs.asks, "no pointless confirmation prompt"


def test_confirmation_states_the_blast_radius(grid):
    for p in grid._batch_visible_paths()[:2]:
        _seed(grid, p, [(0, 5, 5, 20, 20, 0.9)])
    grid.dialogs.answer = False
    grid.accept_batch_predictions()
    assert grid.dialogs.asks, "the user is asked before many files are written"
    body = grid.dialogs.asks[-1][1]
    assert "2" in body, f"box/file counts are stated: {body!r}"


def test_written_coordinates_are_normalised(grid, ds):
    target = grid._batch_visible_paths()[0]
    _seed(grid, target, [(0, 5, 5, 20, 20, 0.9)])
    grid.dialogs.answer = True
    grid.accept_batch_predictions()
    for line in read_exact(grid._resolver().resolve(target)).strip().splitlines():
        for v in map(float, line.split()[1:5]):
            assert 0.0 <= v <= 1.0


def test_confidence_never_reaches_the_label_file(grid, ds):
    target = grid._batch_visible_paths()[0]
    _seed(grid, target, [(0, 5, 5, 20, 20, 0.9)])
    grid.dialogs.answer = True
    grid.accept_batch_predictions()
    for line in read_exact(grid._resolver().resolve(target)).strip().splitlines():
        assert len(line.split()) == 5


def test_accept_outside_batch_mode_is_a_noop(gui_app, ds):
    before = snapshot(ds.root)
    assert gui_app.batch_mode is False
    gui_app.accept_batch_predictions()
    assert_only_changed(before, snapshot(ds.root), set())
