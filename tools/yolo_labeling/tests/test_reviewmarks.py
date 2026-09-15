"""The reviewed-image store.

Review progress represents days of human work, so the properties that matter
are the durability ones: it survives a copy of the dataset, it survives a
corrupt file, and a crash mid-write never destroys what was already there.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from yolo_labeling.core import labelio as _labelio
from yolo_labeling.core.reviewmarks import FILENAME, SCHEMA_VERSION, ReviewMarks


@pytest.fixture
def root(tmp_path):
    """A dataset root with a couple of images in the usual layout."""
    d = tmp_path / "ds"
    (d / "camA" / "images").mkdir(parents=True)
    (d / "camB" / "images").mkdir(parents=True)
    for p in [d / "camA/images/0001.jpg", d / "camA/images/0002.jpg",
              d / "camB/images/0001.jpg"]:
        p.write_bytes(b"x")
    return d


@pytest.fixture
def marks(root):
    return ReviewMarks(root / FILENAME, root)


# -- basics ----------------------------------------------------------------


def test_starts_empty(marks, root):
    assert marks.count() == 0
    assert not marks.is_marked(root / "camA/images/0001.jpg")


def test_mark_and_unmark(marks, root):
    img = root / "camA/images/0001.jpg"
    assert marks.mark(img) is True
    assert marks.is_marked(img)
    assert marks.count() == 1

    assert marks.unmark(img) is True
    assert not marks.is_marked(img)


def test_re_marking_is_a_no_op(marks, root):
    """Idling on an image the dwell timer already caught must not dirty the store."""
    img = root / "camA/images/0001.jpg"
    marks.mark(img)
    marks.save()
    assert marks.dirty is False

    assert marks.mark(img) is False
    assert marks.dirty is False, "a redundant mark scheduled a pointless write"


def test_toggle_reports_the_new_state(marks, root):
    img = root / "camA/images/0001.jpg"
    assert marks.toggle(img) is True
    assert marks.toggle(img) is False


def test_mark_many_counts_only_changes(marks, root):
    a, b = root / "camA/images/0001.jpg", root / "camA/images/0002.jpg"
    marks.mark(a)
    assert marks.mark_many([a, b]) == 1, "a was already marked"
    assert marks.count() == 2


def test_clear(marks, root):
    marks.mark_many([root / "camA/images/0001.jpg", root / "camB/images/0001.jpg"])
    assert marks.clear() == 2
    assert marks.count() == 0


def test_unmark_of_an_unmarked_image_is_harmless(marks, root):
    assert marks.unmark(root / "camA/images/0001.jpg") is False
    assert marks.dirty is False


# -- keys travel with the dataset ------------------------------------------


def test_keys_are_dataset_relative(marks, root):
    """The whole point of the storage choice: absolute keys would not survive a copy."""
    marks.mark(root / "camA/images/0001.jpg")
    assert list(marks.marks) == ["camA/images/0001.jpg"]
    assert str(root) not in json.dumps(marks.marks)


def test_marks_survive_the_dataset_being_moved(root, tmp_path):
    """Copy the dataset elsewhere and the marks still apply."""
    src = ReviewMarks(root / FILENAME, root)
    src.mark(root / "camA/images/0001.jpg")
    src.save()

    moved = tmp_path / "elsewhere" / "ds"
    shutil.copytree(root, moved)

    dst = ReviewMarks(moved / FILENAME, moved)
    assert dst.is_marked(moved / "camA/images/0001.jpg")
    assert dst.count() == 1


def test_same_stem_in_two_folders_stays_distinct(marks, root):
    """camA/0001 and camB/0001 share a stem -- the cross-contamination trap."""
    marks.mark(root / "camA/images/0001.jpg")
    assert not marks.is_marked(root / "camB/images/0001.jpg")


def test_an_image_outside_the_root_falls_back_to_absolute(tmp_path, root):
    outside = tmp_path / "stray.jpg"
    outside.write_bytes(b"x")
    m = ReviewMarks(root / FILENAME, root)
    m.mark(outside)
    assert list(m.marks) == [str(outside)]
    assert m.is_marked(outside)


def test_no_root_means_absolute_keys(root):
    m = ReviewMarks(root / FILENAME, None)
    img = root / "camA/images/0001.jpg"
    m.mark(img)
    assert list(m.marks) == [str(img)]


# -- persistence -----------------------------------------------------------


def test_round_trips_through_disk(root):
    a = ReviewMarks(root / FILENAME, root)
    a.mark(root / "camA/images/0001.jpg")
    a.mark(root / "camB/images/0001.jpg")
    assert a.save() is True

    b = ReviewMarks(root / FILENAME, root)
    assert b.count() == 2
    assert b.is_marked(root / "camB/images/0001.jpg")


def test_save_records_a_timestamp(marks, root):
    marks.mark(root / "camA/images/0001.jpg", when=1757400000)
    marks.save()
    raw = json.loads((root / FILENAME).read_text())
    assert raw["version"] == SCHEMA_VERSION
    assert raw["marks"]["camA/images/0001.jpg"] == 1757400000


def test_save_is_skipped_when_nothing_changed(marks, root):
    assert marks.save() is False, "a clean store must not rewrite the file"
    marks.mark(root / "camA/images/0001.jpg")
    assert marks.save() is True
    assert marks.save() is False


def test_a_missing_file_loads_empty(root):
    m = ReviewMarks(root / "does_not_exist.json", root)
    assert m.count() == 0


def test_a_corrupt_file_loads_empty_rather_than_raising(root):
    """A damaged progress file must never block opening the dataset."""
    (root / FILENAME).write_text("{not json at all")
    m = ReviewMarks(root / FILENAME, root)
    assert m.count() == 0


@pytest.mark.parametrize("payload", [
    '[]',                                    # wrong top-level type
    '{"version": 1}',                        # no marks key
    '{"version": 1, "marks": "nope"}',       # marks is not a dict
])
def test_malformed_shapes_load_empty(root, payload):
    (root / FILENAME).write_text(payload)
    assert ReviewMarks(root / FILENAME, root).count() == 0


def test_a_bad_timestamp_is_skipped_not_fatal(root):
    """Tolerate a hand-edited file: drop the bad row, keep the good ones."""
    (root / FILENAME).write_text(json.dumps({
        "version": 1,
        "marks": {"camA/images/0001.jpg": "yesterday",
                  "camB/images/0001.jpg": 1757400000},
    }))
    m = ReviewMarks(root / FILENAME, root)
    assert m.count() == 1
    assert m.is_marked(root / "camB/images/0001.jpg")


def test_write_leaves_no_temp_file_behind(marks, root):
    marks.mark(root / "camA/images/0001.jpg")
    marks.save()
    leftovers = [p.name for p in root.iterdir() if p.name.startswith(".")]
    assert leftovers == [], f"temp file survived: {leftovers}"


def test_the_store_is_created_under_a_missing_directory(tmp_path, root):
    nested = tmp_path / "a" / "b" / FILENAME
    m = ReviewMarks(nested, root)
    m.mark(root / "camA/images/0001.jpg")
    m.save()
    assert nested.exists()


def test_dry_run_writes_nothing(marks, root, monkeypatch):
    """`--dry-run` must no-op every write, not only label writes."""
    monkeypatch.setattr(_labelio, "DRY_RUN", True)
    monkeypatch.setattr(_labelio, "DRY_RUN_LOG", [])

    marks.mark(root / "camA/images/0001.jpg")
    assert marks.save() is False
    assert not (root / FILENAME).exists()
    assert _labelio.DRY_RUN_LOG and _labelio.DRY_RUN_LOG[0][0] == "write"


def test_marks_never_touch_label_files(marks, root):
    """The store lives beside data.yaml; it must not write into labels/."""
    marks.mark(root / "camA/images/0001.jpg")
    marks.save()
    assert list(Path(root).rglob("*.txt")) == []
