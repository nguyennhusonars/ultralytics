"""Unit tests for core.labelio -- parsing, byte-identical rendering, atomic writes."""

from __future__ import annotations


import pytest

from core import labelio
from core.labelio import (
    delete_label_file,
    parse_label_text,
    read_label_file,
    render_label_text,
    write_label_file,
)
from fixtures import GAUNTLET, MIXED, POLY1, read_exact

# --------------------------------------------------------------------------
# THE acceptance criterion: untouched content survives a round trip exactly.
# This single parametrization covers the segmentation fix, extra columns,
# malformed lines, CRLF and the missing trailing newline all at once.
# --------------------------------------------------------------------------

ROUND_TRIP_CASES = [
    "",                                             # empty file
    "\n",                                           # single newline
    "0 0.5 0.5 0.1 0.1",                            # no trailing newline
    "0 0.5 0.5 0.1 0.1\n",                          # ordinary
    "0 0.5 0.5 0.1 0.1 0.95\n",                     # confidence column
    "0 0.1 0.1 0.2 0.1 0.2 0.2 0.1 0.2\n",          # 4-point polygon
    "  0   0.5 0.5 0.1 0.1  \n",                    # odd whitespace
    "0 0.5 0.5 0.2 0.2\r\n",                        # CRLF
    "99 0.5 0.5 0.1 0.1\n",                         # class past nc
    "-1 0.5 0.5 0.1 0.1\n",                         # negative class
    "0 1.4 0.5 0.2 0.2\n",                          # out of range
    "0 0.5 0.5\n",                                  # too few fields
    "abc 0.5 0.5 0.2 0.2\n",                        # non-numeric class
    "   \n",                                        # whitespace only
    "# a comment\n",                                # comment
    "0 0.5 0.5 0.1 0.1\n\n\n0 0.2 0.2 0.1 0.1\n",   # blank lines between
    GAUNTLET,
    POLY1,
    MIXED,
]


@pytest.mark.parametrize("text", ROUND_TRIP_CASES, ids=lambda t: repr(t[:28]))
def test_untouched_content_round_trips_byte_for_byte(text):
    assert render_label_text(parse_label_text(text)) == text


@pytest.mark.parametrize("text", ROUND_TRIP_CASES, ids=lambda t: repr(t[:28]))
def test_parsing_never_raises(text):
    parse_label_text(text)


def test_malformed_lines_are_preserved_not_dropped():
    """The bug-13 fix: a bad line must not abort the parse or vanish."""
    lines = parse_label_text(GAUNTLET)
    assert len(lines) == len(GAUNTLET.splitlines())
    bad = [ln for ln in lines if not ln.parsed]
    assert bad, "malformed lines are retained as unparsed"
    assert render_label_text(lines) == GAUNTLET


# -- parsing semantics -----------------------------------------------------


def test_bbox_line_parses():
    (ln,) = parse_label_text("2 0.5 0.5 0.2 0.3\n")
    assert ln.cls_id == 2 and ln.parsed and not ln.is_polygon
    assert ln.bbox == (0.5, 0.5, 0.2, 0.3)


def test_confidence_column_is_not_mistaken_for_a_polygon():
    (ln,) = parse_label_text("0 0.5 0.5 0.2 0.2 0.95\n")
    assert not ln.is_polygon
    assert ln.bbox == (0.5, 0.5, 0.2, 0.2)


def test_polygon_is_detected_and_reduced_to_its_extent():
    (ln,) = parse_label_text("0 0.1 0.1 0.2 0.1 0.2 0.2 0.1 0.2\n")
    assert ln.is_polygon
    cx, cy, w, h = ln.bbox
    assert (cx, cy) == pytest.approx((0.15, 0.15))
    assert (w, h) == pytest.approx((0.1, 0.1))


def test_float_class_id_is_normalised_to_int():
    (ln,) = parse_label_text("3.0 0.5 0.5 0.1 0.1\n")
    assert ln.cls_id == 3 and isinstance(ln.cls_id, int)


# -- mutation --------------------------------------------------------------


def test_reassigning_a_polygons_class_keeps_every_point():
    (ln,) = parse_label_text("0 0.1 0.1 0.2 0.1 0.2 0.2 0.1 0.2\n")
    new = ln.with_class(7)
    assert new.cls_id == 7
    assert new.coords == ln.coords
    assert new.render().split()[1:] == [f"{c:.6f}" for c in ln.coords]


def test_only_dirty_lines_are_reserialized():
    lines = parse_label_text(POLY1)
    lines[0] = lines[0].with_class(5)
    out = render_label_text(lines)
    assert out.splitlines()[1] == POLY1.splitlines()[1], "untouched line unchanged"
    assert out.splitlines()[0].startswith("5 ")


def test_edited_lines_get_a_trailing_newline():
    (ln,) = parse_label_text("0 0.5 0.5 0.1 0.1")     # none originally
    assert ln.with_class(1).render().endswith("\n")


# -- atomic writes ---------------------------------------------------------


ORIGINAL = "0 0.500000 0.500000 0.200000 0.200000\n"


def test_write_then_read_round_trips(tmp_path):
    p = tmp_path / "a.txt"
    write_label_file(p, parse_label_text(POLY1))
    assert read_exact(p) == POLY1


def test_write_creates_missing_parent_directories(tmp_path):
    p = tmp_path / "deep" / "nested" / "a.txt"
    write_label_file(p, ORIGINAL)
    assert p.exists()


def test_failure_during_write_leaves_the_original_intact(tmp_path, monkeypatch):
    p = tmp_path / "a.txt"
    p.write_text(ORIGINAL)

    def boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(labelio.os, "fsync", boom)
    with pytest.raises(OSError):
        write_label_file(p, "9 0.1 0.1 0.1 0.1\n")
    assert read_exact(p) == ORIGINAL, "original was truncated"


def test_failure_during_rename_leaves_the_original_intact(tmp_path, monkeypatch):
    p = tmp_path / "a.txt"
    p.write_text(ORIGINAL)

    def boom(*a, **k):
        raise OSError("rename failed")

    monkeypatch.setattr(labelio.os, "replace", boom)
    with pytest.raises(OSError):
        write_label_file(p, "9 0.1 0.1 0.1 0.1\n")
    assert read_exact(p) == ORIGINAL


def test_a_failed_write_leaves_no_temp_files_behind(tmp_path, monkeypatch):
    p = tmp_path / "a.txt"
    p.write_text(ORIGINAL)
    monkeypatch.setattr(labelio.os, "replace", lambda *a, **k: (_ for _ in ()).throw(OSError))
    with pytest.raises(OSError):
        write_label_file(p, "x\n")
    assert [q.name for q in tmp_path.iterdir()] == ["a.txt"]


def test_written_file_is_never_partially_visible(tmp_path):
    """os.replace is atomic: the target is either old content or new, never half."""
    p = tmp_path / "a.txt"
    p.write_text(ORIGINAL)
    big = "".join(f"{i % 4} 0.5 0.5 0.1 0.1\n" for i in range(5000))
    write_label_file(p, big)
    assert read_exact(p) == big


# -- deletion and dry run --------------------------------------------------


def test_delete_reports_whether_it_removed_anything(tmp_path):
    p = tmp_path / "a.txt"
    assert delete_label_file(p) is False
    p.write_text(ORIGINAL)
    assert delete_label_file(p) is True
    assert not p.exists()


def test_dry_run_logs_instead_of_touching_disk(tmp_path, monkeypatch):
    p = tmp_path / "a.txt"
    p.write_text(ORIGINAL)
    monkeypatch.setattr(labelio, "DRY_RUN", True)
    monkeypatch.setattr(labelio, "DRY_RUN_LOG", [])
    write_label_file(p, "9 0.1 0.1 0.1 0.1\n")
    delete_label_file(p)
    assert read_exact(p) == ORIGINAL, "disk untouched"
    assert [op for op, _, _ in labelio.DRY_RUN_LOG] == ["write", "delete"]


def test_reading_a_missing_file_is_empty_not_an_error(tmp_path):
    assert read_label_file(tmp_path / "nope.txt") == []
