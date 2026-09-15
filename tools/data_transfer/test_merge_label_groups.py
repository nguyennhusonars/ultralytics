"""Tests for merge_label_groups.py.

This script rewrites every label file in the dataset, so the properties that
matter are the conservative ones: it changes field 0 and nothing else, and it
refuses rather than guesses whenever the mapping is ambiguous.

Run with:  pytest tools/data_transfer/test_merge_label_groups.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from merge_label_groups import (  # noqa: E402
    build_mapping,
    dataset_root,
    label_dirs,
    main,
    remap_text,
)

NAMES = ["BSD", "BSV", "bus_m", "bus_l", "bus_xl", "car", "human",
         "twowheels_standalone", "bike_with_driver", "motorbike_with_driver",
         "truck_m", "truck_l", "truck_xl"]
GROUPS = {
    "BSD": ["BSD"], "BSV": ["BSV"],
    "bus": ["bus_m", "bus_l", "bus_xl"],
    "car": ["car"], "human": ["human"],
    "motorbike": ["motorbike_with_driver", "bike_with_driver", "twowheels_standalone"],
    "truck": ["truck_m", "truck_l", "truck_xl"],
}
#: The real pvn-vehicle mapping, stated independently of the code under test.
EXPECTED = {0: 0, 1: 1, 2: 2, 3: 2, 4: 2, 5: 3, 6: 4,
            7: 5, 8: 5, 9: 5, 10: 6, 11: 6, 12: 6}


def _data(**over):
    d = {"names": list(NAMES), "nc": len(NAMES), "groups": dict(GROUPS),
         "train": ["a/images"], "val": ["b/images"]}
    d.update(over)
    return d


# -- the mapping -----------------------------------------------------------


def test_group_order_defines_the_merged_ids():
    """Model class i is the i-th group -- the convention training already uses."""
    mapping, names = build_mapping(_data())
    assert names == ["BSD", "BSV", "bus", "car", "human", "motorbike", "truck"]
    assert mapping == EXPECTED


def test_a_class_in_no_group_is_refused():
    groups = dict(GROUPS)
    groups["truck"] = ["truck_m", "truck_l"]          # truck_xl orphaned
    with pytest.raises(SystemExit, match="truck_xl"):
        build_mapping(_data(groups=groups))


def test_a_class_in_two_groups_is_refused():
    groups = dict(GROUPS)
    groups["car"] = ["car", "bus_m"]                  # bus_m already in 'bus'
    with pytest.raises(SystemExit, match="both"):
        build_mapping(_data(groups=groups))


def test_an_unknown_class_name_is_refused():
    groups = dict(GROUPS)
    groups["car"] = ["car", "sedan"]
    with pytest.raises(SystemExit, match="sedan"):
        build_mapping(_data(groups=groups))


def test_a_yaml_with_no_groups_is_refused():
    d = _data()
    del d["groups"]
    with pytest.raises(SystemExit, match="groups"):
        build_mapping(d)


def test_names_may_be_a_dict():
    d = _data(names={i: n for i, n in enumerate(NAMES)})
    mapping, _ = build_mapping(d)
    assert mapping == EXPECTED


# -- rewriting -------------------------------------------------------------


def test_only_the_class_id_changes():
    text = "3 0.617839 0.686877 0.082031 0.031561\n"
    out, seen, unmapped = remap_text(text, EXPECTED)
    assert out == "2 0.617839 0.686877 0.082031 0.031561\n"
    assert seen == {3: 1} and not unmapped


def test_coordinates_are_never_reformatted():
    """Re-emitting the numbers would rewrite 450k files and lose precision."""
    text = "5 0.1 0.20000000001 1 0.5000\n"
    out, _, _ = remap_text(text, EXPECTED)
    assert out == "3 0.1 0.20000000001 1 0.5000\n"


def test_polygons_survive_intact():
    text = "11 0.1 0.1 0.2 0.1 0.2 0.2 0.15 0.25 0.1 0.2\n"
    out, _, _ = remap_text(text, EXPECTED)
    assert out == "6 0.1 0.1 0.2 0.1 0.2 0.2 0.15 0.25 0.1 0.2\n"


def test_a_confidence_column_survives():
    text = "9 0.5 0.5 0.2 0.2 0.87\n"
    out, _, _ = remap_text(text, EXPECTED)
    assert out == "5 0.5 0.5 0.2 0.2 0.87\n"


@pytest.mark.parametrize("eol", ["\n", "\r\n", ""])
def test_line_endings_are_preserved(eol):
    text = f"0 0.5 0.5 0.1 0.1{eol}"
    out, _, _ = remap_text(text, EXPECTED)
    assert out == f"0 0.5 0.5 0.1 0.1{eol}"


def test_blank_and_whitespace_lines_pass_through():
    text = "\n   \n5 0.5 0.5 0.1 0.1\n\n"
    out, seen, _ = remap_text(text, EXPECTED)
    assert out == "\n   \n3 0.5 0.5 0.1 0.1\n\n"
    assert seen == {5: 1}


def test_a_non_numeric_class_is_left_alone():
    """Not a label line; rewriting it would be a guess."""
    text = "abc 0.5 0.5 0.1 0.1\n"
    out, seen, _ = remap_text(text, EXPECTED)
    assert out == text and not seen


def test_an_unmapped_class_is_reported_and_left_unchanged():
    """data_UK's 13=LP_UK / 14=pickup would land here if ever included."""
    text = "13 0.5 0.5 0.1 0.1\n"
    out, seen, unmapped = remap_text(text, EXPECTED)
    assert out == text
    assert unmapped == {13: 1} and seen == {13: 1}


# -- label directory resolution -------------------------------------------


def test_images_dirs_pair_with_sibling_labels(tmp_path):
    d = _data(train=["x/train/images"], val=["y/valid/images"])
    dirs = label_dirs(tmp_path, d)
    assert dirs == [(tmp_path / "x/train/labels").resolve(),
                    (tmp_path / "y/valid/labels").resolve()]


def test_duplicate_entries_are_visited_once(tmp_path):
    d = _data(train=["x/images", "x/images"], val=["x/images"])
    assert len(label_dirs(tmp_path, d)) == 1


def test_root_falls_back_to_the_yaml_directory(tmp_path):
    y = tmp_path / "data.yaml"
    assert dataset_root(y, {"path": None}) == tmp_path.resolve()
    assert dataset_root(y, {"path": "sub"}) == (tmp_path / "sub").resolve()


# -- end to end ------------------------------------------------------------


@pytest.fixture
def dataset(tmp_path):
    """A miniature dataset in the real layout, plus a stray non-label .txt."""
    root = tmp_path / "ds"
    for split in ("train", "valid"):
        lbl = root / "cam" / split / "labels"
        lbl.mkdir(parents=True)
        (lbl / "0001.txt").write_text("3 0.5 0.5 0.2 0.2\n9 0.1 0.1 0.1 0.1\n")
        (lbl / "0002.txt").write_text("")            # valid negative sample
    # Not referenced by the yaml, and using a foreign taxonomy -- must be skipped.
    stray = root / "data_UK" / "train" / "labels"
    stray.mkdir(parents=True)
    (stray / "x.txt").write_text("13 0.5 0.5 0.1 0.1\n")

    (root / "data.yaml").write_text(yaml.safe_dump(
        _data(train=["cam/train/images"], val=["cam/valid/images"]),
        sort_keys=False))
    return root


def test_end_to_end_writes_merged_labels(dataset, capsys):
    out = dataset / "merged"
    assert main([str(dataset / "data.yaml"), "--out", str(out)]) == 0

    got = (out / "cam/train/labels/0001.txt").read_text()
    assert got == "2 0.5 0.5 0.2 0.2\n5 0.1 0.1 0.1 0.1\n"
    assert (out / "cam/valid/labels/0002.txt").read_text() == ""


def test_the_generated_yaml_describes_the_merged_classes(dataset):
    out = dataset / "merged"
    main([str(dataset / "data.yaml"), "--out", str(out)])
    merged = yaml.safe_load((out / "data_7g.yaml").read_text())
    assert merged["nc"] == 7
    assert merged["names"] == ["BSD", "BSV", "bus", "car", "human", "motorbike", "truck"]
    # A groups: block here would invite a second round of collapsing.
    assert "groups" not in merged
    assert merged["train"] == ["cam/train/images"]


def test_unreferenced_folders_are_never_touched(dataset):
    """data_UK has a different 15-class taxonomy; a blind walk would mis-map it."""
    out = dataset / "merged"
    main([str(dataset / "data.yaml"), "--out", str(out)])
    assert not (out / "data_UK").exists()
    assert (dataset / "data_UK/train/labels/x.txt").read_text() == "13 0.5 0.5 0.1 0.1\n"


def test_dry_run_writes_nothing(dataset):
    out = dataset / "merged"
    main([str(dataset / "data.yaml"), "--out", str(out), "--dry-run"])
    assert not out.exists()


def test_report_only_writes_nothing(dataset):
    out = dataset / "merged"
    main([str(dataset / "data.yaml"), "--out", str(out), "--report-only"])
    assert not out.exists()


def test_the_source_labels_are_never_modified(dataset):
    before = {p: p.read_bytes() for p in dataset.rglob("*.txt")}
    main([str(dataset / "data.yaml"), "--out", str(dataset / "merged")])
    for p, content in before.items():
        assert p.read_bytes() == content, f"{p} was modified in place"


def test_out_of_range_classes_stop_the_run(dataset):
    """Silently mis-mapping 11k boxes is exactly what this must not do."""
    bad = dataset / "cam/train/labels/0003.txt"
    bad.write_text("13 0.5 0.5 0.1 0.1\n")
    with pytest.raises(SystemExit, match="outside the mapping"):
        main([str(dataset / "data.yaml"), "--out", str(dataset / "merged")])


def test_allow_unmapped_lets_them_through_unchanged(dataset):
    bad = dataset / "cam/train/labels/0003.txt"
    bad.write_text("13 0.5 0.5 0.1 0.1\n")
    out = dataset / "merged"
    assert main([str(dataset / "data.yaml"), "--out", str(out),
                 "--allow-unmapped"]) == 0
    assert (out / "cam/train/labels/0003.txt").read_text() == "13 0.5 0.5 0.1 0.1\n"


def test_zip_matches_the_layout_the_importer_reads(dataset):
    """label_transfer.py reads labels from `labels/`, yamls from `yamls/`, and
    sizes the run from metadata's label_files/yaml_files. Getting any of those
    wrong imports silently or reports "0 labels" while writing 45k."""
    import json
    import zipfile

    out, arc = dataset / "merged", dataset / "merged.zip"
    main([str(dataset / "data.yaml"), "--out", str(out), "--zip", str(arc)])
    with zipfile.ZipFile(arc) as zf:
        names = zf.namelist()
        assert "metadata.json" in names
        assert "labels/cam/train/labels/0001.txt" in names
        assert zf.read("labels/cam/train/labels/0001.txt").decode() == \
            "2 0.5 0.5 0.2 0.2\n5 0.1 0.1 0.1 0.1\n"

        # Shipped AS data.yaml so importing leaves the dataset self-consistent.
        assert "yamls/data.yaml" in names
        shipped = yaml.safe_load(zf.read("yamls/data.yaml"))
        assert shipped["nc"] == 7 and "groups" not in shipped

        meta = json.loads(zf.read("metadata.json"))
        assert len(meta["label_files"]) == 4
        assert len(meta["yaml_files"]) == 1
        assert meta["yaml_files"][0]["archive_path"] == "yamls/data.yaml"


def test_the_generated_yaml_pins_the_dataset_root(dataset):
    """It is written into a staging dir, so relative split entries need a root
    or they resolve against the wrong directory and match nothing."""
    out = dataset / "merged"
    main([str(dataset / "data.yaml"), "--out", str(out)])
    merged = yaml.safe_load((out / "data_7g.yaml").read_text())
    assert Path(merged["path"]) == dataset.resolve()


def test_already_merged_labels_are_refused(dataset):
    """Merging twice is silently catastrophic: a merged 'car' (3) would be read
    as fine class 3 (bus_l) and rewritten to 'bus'."""
    out = dataset / "merged"
    main([str(dataset / "data.yaml"), "--out", str(out)])
    # Feed the merged output back in as if it were the source.
    for f in out.rglob("*.txt"):
        (dataset / f.relative_to(out)).write_text(f.read_text())
    with pytest.raises(SystemExit, match="ALREADY MERGED"):
        main([str(dataset / "data.yaml"), "--out", str(dataset / "again")])


def test_force_overrides_the_already_merged_guard(dataset):
    out = dataset / "merged"
    main([str(dataset / "data.yaml"), "--out", str(out)])
    for f in out.rglob("*.txt"):
        (dataset / f.relative_to(out)).write_text(f.read_text())
    assert main([str(dataset / "data.yaml"), "--out", str(dataset / "again"),
                 "--force"]) == 0
