"""Meta-tests: prove the harness itself is sound before anything is built on it."""

from __future__ import annotations

import cv2
import pytest
from PIL import Image

from fixtures import (
    BAD_CORRUPT,
    CAMA_0001,
    CAMB_0001,
    GAUNTLET,
    MIXED,
    POLY1,
    assert_only_changed,
    diff_snapshots,
    read_exact,
    snapshot,
)


def test_dataset_builds(ds):
    assert ds.yaml.exists() and ds.config.exists()
    assert ds.image("camA/0001.jpg").exists()
    assert ds.label("camA/0001").read_text() == CAMA_0001


def test_same_stem_across_folders_are_distinct_files(ds):
    """camA/0001 and camB/0001 share a stem -- the cross-contamination trap."""
    a, b = ds.label("camA/0001"), ds.label("camB/0001")
    assert a != b
    assert a.read_text() == CAMA_0001
    assert b.read_text() == CAMB_0001


def test_corrupt_image_is_unreadable_but_its_labels_are_valid(ds):
    """The headline data-loss trap: cv2 fails, yet the label file is good."""
    assert cv2.imread(str(ds.image("bad/corrupt.jpg"))) is None
    assert cv2.imread(str(ds.image("bad/empty.jpg"))) is None
    assert ds.label("bad/corrupt").read_text() == BAD_CORRUPT
    assert len(BAD_CORRUPT.strip().splitlines()) == 2


def test_segmentation_labels_have_polygon_arity(ds):
    lines = POLY1.strip().splitlines()
    assert len(lines[0].split()) == 9    # class + 4 points
    assert len(lines[1].split()) == 11   # class + 5 points
    assert ds.label("seg/poly1").read_text() == POLY1
    assert ds.label("seg/mixed").read_text() == MIXED


def test_gauntlet_has_no_trailing_newline(ds):
    """A missing trailing newline is itself an edge case we must round-trip."""
    text = read_exact(ds.label("bad/ok"))
    assert text == GAUNTLET
    assert not text.endswith("\n")
    assert "\r\n" in text


def test_read_text_would_corrupt_crlf_comparisons(ds):
    """Guard rail: `read_text()` normalises CRLF, `read_exact()` does not.

    Without this distinction a byte-identity assertion could pass against a
    file the tool had actually rewritten.
    """
    p = ds.label("bad/ok")
    assert "\r\n" in read_exact(p)
    assert "\r\n" not in p.read_text()


def test_uppercase_extension_exists_but_lowercase_glob_misses_it(ds):
    """Pins the case-sensitivity bug: the file is there, the glob can't see it."""
    up = ds.image("camA/0004.JPG")
    assert up.exists()
    found = {p.name for p in (ds.root / "camA/images").glob("*.jpg")}
    assert "0004.JPG" not in found


def test_prefix_collision_pair_is_distinct(ds):
    """`ds_set1` is a strict prefix of `ds_set10` -- breaks startswith matching."""
    assert str(ds.root / "ds_set10").startswith(str(ds.root / "ds_set1"))
    assert ds.label("ds_set1/a").read_text() != ds.label("ds_set10/a").read_text()


def test_exif_rotated_image_disagrees_between_readers(ds):
    """cv2.imread APPLIES EXIF orientation; PIL's raw `.size` does not.

    The tool reads dimensions from both (`load_labels` uses cv2, the cache
    builder and `get_label_boxes` use PIL/header parsing), so for this image
    they disagree by a transpose -- which silently corrupts pixel boxes.
    """
    p = ds.image("camB/rot.jpg")
    with Image.open(p) as im:
        assert im.getexif().get(274) == 6
        pil_wh = im.size
    h, w = cv2.imread(str(p)).shape[:2]
    cv_wh = (w, h)

    assert pil_wh == (64, 48), "PIL reports the stored, unrotated geometry"
    assert cv_wh == (48, 64), "cv2 reports the rotation-applied geometry"
    assert pil_wh != cv_wh, "the readers must disagree -- that IS the bug"


def test_unicode_and_space_in_stem(ds):
    assert ds.image("weird/имя пробел.jpg").exists()
    assert ds.label("weird/имя пробел").exists()


@pytest.mark.parametrize("rel", ["camA/0002"])
def test_empty_label_file_is_valid_negative_sample(ds, rel):
    p = ds.label(rel)
    assert p.exists() and p.read_text() == ""


# -- change detection ------------------------------------------------------


def test_snapshot_detects_modify_add_remove(ds):
    before = snapshot(ds.root)
    ds.label("camA/0001").write_text("0 0.1 0.1 0.1 0.1\n")   # modified
    ds.label("camA/0003").write_text("0 0.2 0.2 0.2 0.2\n")   # added
    ds.label("camB/0001").unlink()                            # removed
    d = diff_snapshots(before, snapshot(ds.root))
    assert d["camA/labels/0001.txt"] == "modified"
    assert d["camA/labels/0003.txt"] == "added"
    assert d["camB/labels/0001.txt"] == "removed"


def test_assert_only_changed_passes_when_nothing_moves(ds):
    before = snapshot(ds.root)
    assert_only_changed(before, snapshot(ds.root), set())


def test_assert_only_changed_flags_an_unexpected_write(ds):
    before = snapshot(ds.root)
    ds.label("camB/0001").write_text("9 0.1 0.1 0.1 0.1\n")
    with pytest.raises(AssertionError, match="unexpectedly changed"):
        assert_only_changed(before, snapshot(ds.root), set())


def test_assert_only_changed_flags_a_missing_expected_write(ds):
    before = snapshot(ds.root)
    with pytest.raises(AssertionError, match="expected to change but did not"):
        assert_only_changed(before, snapshot(ds.root), {"camA/labels/0001.txt"})


# -- safety rails ----------------------------------------------------------


def test_cwd_is_tmp_not_the_repo(tmp_path):
    """The autouse chdir guard keeps class_config.json out of the repo."""
    from pathlib import Path

    assert Path.cwd() == tmp_path


def test_dialog_recorder_captures_instead_of_blocking(dialogs):
    dialogs.showerror("t", "boom")
    dialogs.answer = False
    assert dialogs.errors == [("t", "boom")]
    assert dialogs.askyesno("q", "sure?") is False
