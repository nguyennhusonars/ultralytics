"""Machine-enforced bug ledger.

Every test here starts life as a confirmed defect marked `xfail(strict=True)`.
Strict means an xfail that starts PASSING is a suite failure -- so the moment a
bug is fixed the suite forces you to come here and flip the marker, and the test
becomes a permanent regression guard. The ledger shrinks to zero as the fixes
land; tests with no marker are already fixed and must stay passing.

Tests run against the monolith through `LegacyProbe`, which faithfully models
the leaky code paths (see its docstring).
"""

from __future__ import annotations

import pytest

from fixtures import BAD_CORRUPT, CAMA_0001, CAMB_0001, POLY1, read_exact

pytestmark = pytest.mark.legacy


# ==========================================================================
# BUG 1 -- user_boxes leak across images
# ==========================================================================


def test_user_boxes_do_not_survive_an_image_change(ds, legacy):
    """9 of 12 `load_image()` call sites never clear `user_boxes`."""
    p = legacy()
    p.open(ds.image("camA/0001.jpg"))
    p.add_user_box(0, 5, 5, 20, 20)

    p.open(ds.image("camB/0001.jpg"))       # a leaky navigation path
    assert p.user_boxes == []


def test_leaked_box_is_not_written_into_the_next_images_label_file(ds, legacy):
    """The consequence: camA's box lands in camB's label file."""
    p = legacy()
    p.open(ds.image("camA/0001.jpg"))
    p.add_user_box(0, 5, 5, 20, 20)

    p.open(ds.image("camB/0001.jpg"))
    p.save()
    assert read_exact(ds.label("camB/0001")) == CAMB_0001


# ==========================================================================
# BUG 2 -- unreadable image causes a valid label file to be deleted
# ==========================================================================


def test_unreadable_image_never_deletes_its_label_file(ds, legacy):
    p = legacy()
    try:
        p.open(ds.image("bad/corrupt.jpg"))   # cv2 returns None -> AttributeError
    except AttributeError:
        pass
    p.save()
    assert ds.label("bad/corrupt").exists(), "a valid label file was deleted"
    assert read_exact(ds.label("bad/corrupt")) == BAD_CORRUPT


def test_zero_byte_image_never_deletes_its_label_file(ds, legacy):
    p = legacy()
    try:
        p.open(ds.image("bad/empty.jpg"))
    except AttributeError:
        pass
    p.save()
    assert ds.label("bad/empty").exists()


def test_unreadable_image_reports_instead_of_raising(ds, legacy):
    p = legacy()
    p.open(ds.image("bad/corrupt.jpg"))       # should not raise
    assert p.app.current_image is None


# ==========================================================================
# BUG 4 -- segmentation polygons destroyed on save
# ==========================================================================


def test_segmentation_polygons_survive_a_save(ds, legacy):
    """Opening a seg dataset and pressing `s` currently destroys all polygons."""
    p = legacy()
    p.open(ds.image("seg/poly1.jpg"))
    p.save()
    assert read_exact(ds.label("seg/poly1")) == POLY1


def test_extra_columns_survive_a_save(ds, legacy):
    """A 6th confidence column is dropped on the first save."""
    lbl = ds.label("camA/0001")
    lbl.write_bytes(b"0 0.500000 0.500000 0.200000 0.300000 0.987\n")
    p = legacy()
    p.open(ds.image("camA/0001.jpg"))
    p.save()
    assert "0.987" in read_exact(lbl)


# ==========================================================================
# BUG 10 -- deleting every box removes the file rather than emptying it
# ==========================================================================


def test_removing_all_boxes_writes_an_empty_file_not_a_deletion(ds, legacy):
    p = legacy()
    p.open(ds.image("camA/0001.jpg"))
    p.app.label_boxes = []                    # user deleted every box
    p.save()
    assert ds.label("camA/0001").exists(), "a 0-byte .txt is a valid negative sample"


# ==========================================================================
# BUG 6 -- startswith prefix collision in the per-folder label mapping
# ==========================================================================


def test_prefix_collision_resolves_to_the_right_label_folder(ds, legacy):
    # Bare directory keys -- what `data.yaml` yields when it lists a folder
    # directly. "<root>/ds_set1" IS a string prefix of "<root>/ds_set10/...",
    # so `startswith` matches the wrong entry and dict order decides.
    mapping = {
        str(ds.root / "ds_set1"): str(ds.root / "ds_set1" / "labels"),
        str(ds.root / "ds_set10"): str(ds.root / "ds_set10" / "labels"),
    }
    assert str(ds.image("ds_set10/a.jpg")).startswith(str(ds.root / "ds_set1"))
    p = legacy(per_image=mapping)
    p.open(ds.image("ds_set10/a.jpg"))
    # ds_set10/a.txt holds class 1; ds_set1/a.txt holds class 0
    assert p.label_boxes and int(p.label_boxes[0][0]) == 1


# ==========================================================================
# BUG 12 / 14 / 15 -- serialization and coordinate defects
# ==========================================================================


def test_saved_file_ends_with_a_newline(ds, legacy):
    p = legacy()
    p.open(ds.image("camA/0001.jpg"))
    p.save()
    assert read_exact(ds.label("camA/0001")).endswith("\n")


def test_saved_coordinates_are_clamped_to_unit_interval(ds, legacy):
    p = legacy()
    p.open(ds.image("camA/0001.jpg"))
    p.app.label_boxes = []
    p.add_user_box(0, -50, -50, 5000, 5000)   # wildly out of bounds
    p.save()
    for line in read_exact(ds.label("camA/0001")).strip().splitlines():
        for v in map(float, line.split()[1:5]):
            assert 0.0 <= v <= 1.0, f"{v} outside [0,1] in {line!r}"


def test_load_save_round_trip_is_lossless(ds, legacy):
    """`load` truncates with int(); `save` divides back out, shrinking boxes."""
    p = legacy()
    p.open(ds.image("camA/0001.jpg"))
    p.save()
    assert read_exact(ds.label("camA/0001")) == CAMA_0001


# ==========================================================================
# BUG 13 -- one malformed line aborts parsing and truncates the file on save
# ==========================================================================


def test_malformed_line_does_not_abort_label_parsing(ds, legacy):
    """`map(float, parts[:5])` is unguarded, so `abc 0.5 ...` raises."""
    p = legacy()
    p.open(ds.image("bad/ok.jpg"))            # must not raise


def test_malformed_line_does_not_cause_the_file_to_be_truncated(ds, legacy):
    """The dangerous consequence.

    `load_labels` clears `label_boxes`, appends the 6 lines before the bad one,
    then raises. The 6 parsed boxes stay in memory, so the next save writes only
    those and silently discards everything after the malformed line.
    """
    p = legacy()
    before = read_exact(ds.label("bad/ok"))
    try:
        p.open(ds.image("bad/ok.jpg"))
    except ValueError:
        pass
    p.save()
    after = read_exact(ds.label("bad/ok"))
    assert len(after.strip().splitlines()) >= len(
        [ln for ln in before.strip().splitlines() if ln.strip()]
    ), "lines after the malformed one were destroyed"
