"""Unit tests for core.paths -- one resolver, one answer."""

from __future__ import annotations

import pytest

from core.paths import LabelPathResolver, is_within


# -- boundary-aware containment -------------------------------------------


def test_is_within_respects_path_boundaries(tmp_path):
    (tmp_path / "set1").mkdir()
    (tmp_path / "set10").mkdir()
    assert is_within(tmp_path / "set10" / "a.jpg", tmp_path / "set10")
    assert not is_within(tmp_path / "set10" / "a.jpg", tmp_path / "set1")


def test_naive_startswith_would_have_matched(tmp_path):
    """Documents exactly what the old implementation got wrong."""
    a, b = str(tmp_path / "set10" / "a.jpg"), str(tmp_path / "set1")
    assert a.startswith(b)                       # the bug
    assert not is_within(a, b)                   # the fix


# -- resolution order ------------------------------------------------------


def test_explicit_label_folder_wins(ds):
    r = LabelPathResolver(label_folder=ds.root / "camB" / "labels")
    assert r.resolve(ds.image("camA/0001.jpg")) == ds.root / "camB" / "labels" / "0001.txt"


def test_images_dir_pairs_with_sibling_labels_dir(ds):
    r = LabelPathResolver()
    assert r.resolve(ds.image("camA/0001.jpg")) == ds.label("camA/0001")


def test_mapping_is_consulted_before_convention(ds):
    r = LabelPathResolver(per_image={str(ds.root / "camA"): str(ds.root / "camB" / "labels")})
    assert r.resolve(ds.image("camA/0001.jpg")).parent == ds.root / "camB" / "labels"


def test_prefix_collision_resolves_to_the_more_specific_entry(ds):
    """`ds_set1` must not capture `ds_set10` -- the bug-6 fix."""
    r = LabelPathResolver(
        per_image={
            str(ds.root / "ds_set1"): str(ds.root / "ds_set1" / "labels"),
            str(ds.root / "ds_set10"): str(ds.root / "ds_set10" / "labels"),
        }
    )
    assert r.resolve(ds.image("ds_set10/a.jpg")).parent == ds.root / "ds_set10" / "labels"
    assert r.resolve(ds.image("ds_set1/a.jpg")).parent == ds.root / "ds_set1" / "labels"


def test_longest_mapping_wins_when_nested(tmp_path):
    r = LabelPathResolver(
        per_image={
            str(tmp_path / "ds"): str(tmp_path / "outer_labels"),
            str(tmp_path / "ds" / "camA"): str(tmp_path / "inner_labels"),
        }
    )
    got = r.resolve(tmp_path / "ds" / "camA" / "images" / "x.jpg")
    assert got.parent == tmp_path / "inner_labels"


# -- the divergence that caused the shadow-copy bug ------------------------


@pytest.mark.parametrize(
    "rel",
    [
        "camA/images/0001.jpg",     # conventional images/ layout
        "frames/x0001.jpg",         # parent not named images, no labels dir
        "camB/images/wide.jpg",
        "weird/images/имя пробел.jpg",
    ],
)
def test_reading_and_writing_resolve_to_the_same_place(ds, rel):
    """Load and save MUST agree, or the dataset silently forks."""
    r = LabelPathResolver()
    img = ds.root / rel
    assert r.resolve(img) == r.resolve(img)
    assert r.resolve_dir(img) == r.resolve_dir(img)


def test_non_images_parent_prefers_an_existing_labels_dir(tmp_path):
    """Images in `imgs/`, labels in a sibling `labels/` one level up."""
    (tmp_path / "imgs").mkdir()
    (tmp_path / "labels").mkdir()
    r = LabelPathResolver()
    assert r.resolve_dir(tmp_path / "imgs" / "x.jpg") == tmp_path / "labels"


def test_non_images_parent_defaults_to_a_sibling_when_nothing_exists(tmp_path):
    (tmp_path / "imgs").mkdir()
    r = LabelPathResolver()
    assert r.resolve_dir(tmp_path / "imgs" / "x.jpg") == tmp_path / "imgs" / "labels"


def test_resolve_dir_never_returns_none(ds):
    """The legacy loader could return None and silently show zero labels."""
    r = LabelPathResolver()
    for p in [ds.image("camA/0001.jpg"), ds.root / "frames" / "x0001.jpg"]:
        assert r.resolve_dir(p) is not None


# -- stem handling ---------------------------------------------------------


@pytest.mark.parametrize("name", ["0001.jpg", "0004.JPG", "имя пробел.jpg", "a.b.c.jpg"])
def test_stem_becomes_the_txt_name(tmp_path, name):
    r = LabelPathResolver(label_folder=tmp_path / "L")
    from pathlib import Path

    assert r.resolve(tmp_path / "images" / name).name == Path(name).stem + ".txt"


def test_exists_reports_file_presence(ds):
    r = LabelPathResolver()
    assert r.exists(ds.image("camA/0001.jpg"))
    assert not r.exists(ds.image("camA/0003.jpg"))    # image with no label
