"""The 'Find Bad Annotations' configuration dialog and result ordering.

The dialog's job is to make a scan's meaning and cost knowable before it runs.
Most of what is pinned here is therefore refusal behaviour: a scan that cannot
answer the question asked should say so rather than run for an hour and return
an empty list.
"""

from __future__ import annotations

import pytest

from yolo_labeling.core import predcache as _predcache
from yolo_labeling.ui.crop_review import KIND_ORDER, _entry_rank

pytestmark = [pytest.mark.gui, pytest.mark.legacy]


def _entry(kind, score, **kw):
    e = {'kind': kind, 'score': score}
    e.update(kw)
    return e


# -- result ordering -------------------------------------------------------


def test_kinds_are_grouped_before_being_scored():
    """`score` means different things per kind, so a mixed sort on it alone
    put two scales in one ordering."""
    entries = [
        _entry('missing', 1.0),
        _entry('class_mismatch', 0.30),
        _entry('missing', 0.60),
        _entry('class_mismatch', 0.95),
    ]
    entries.sort(key=_entry_rank)
    assert [e['kind'] for e in entries] == [
        'class_mismatch', 'class_mismatch', 'missing', 'missing']


def test_within_a_kind_the_worst_comes_first():
    entries = [_entry('class_mismatch', 0.30), _entry('class_mismatch', 0.95)]
    entries.sort(key=_entry_rank)
    assert [e['score'] for e in entries] == [0.95, 0.30]


def test_an_unknown_kind_sorts_last_rather_than_crashing():
    entries = [_entry('something_new', 1.0), _entry('class_mismatch', 0.1)]
    entries.sort(key=_entry_rank)
    assert entries[0]['kind'] == 'class_mismatch'


def test_a_missing_score_does_not_break_the_sort():
    assert _entry_rank({'kind': 'missing'}) == (KIND_ORDER['missing'], 0.0)


# -- dialog guards ---------------------------------------------------------


@pytest.fixture
def opened(gui_app, monkeypatch):
    """Open the dialog and hand back its Start handler plus its widgets."""
    gui_app.pretrained_model = object()          # the dialog only checks truthiness
    gui_app.model_names = {0: "person"}

    started = {}
    monkeypatch.setattr(type(gui_app), "open_bad_label_review",
                        lambda self, config: started.setdefault('config', config))
    gui_app.show_find_bad_dialog()
    gui_app.root.update_idletasks()

    win = [w for w in gui_app.root.winfo_children()
           if isinstance(w, __import__('tkinter').Toplevel)
           and w.title() == "Find Bad Annotations"][-1]
    return gui_app, win, started


def _widgets(win, cls):
    out = []
    stack = [win]
    while stack:
        w = stack.pop()
        stack.extend(w.winfo_children())
        if isinstance(w, cls):
            out.append(w)
    return out


def _click_start(win):
    import tkinter.ttk as ttk
    for b in _widgets(win, ttk.Button):
        if b.cget('text') == "Start scan":
            b.invoke()
            return
    raise AssertionError("no Start scan button")


def _status(win):
    import tkinter.ttk as ttk
    texts = []
    for lab in _widgets(win, ttk.Label):
        var = lab.cget('textvariable')
        if var:
            texts.append(win.getvar(var))
    return " ".join(t for t in texts if t)


def test_scan_refuses_when_nothing_is_selected_to_report(opened):
    app, win, started = opened
    for cb in _widgets(win, __import__('tkinter').ttk.Checkbutton):
        cb.invoke()                               # both start on, turn both off
    _click_start(win)
    assert 'config' not in started
    assert "report" in _status(win).lower()


def test_a_model_says_restriction_without_wrong_class_is_refused(opened):
    """The restriction only applies to wrong-class entries, so this combination
    can only ever return nothing."""
    app, win, started = opened
    import tkinter.ttk as ttk
    combos = _widgets(win, ttk.Combobox)
    combos[0].set(combos[0].cget('values')[1])    # a concrete predicted class
    for cb in _widgets(win, ttk.Checkbutton):
        if "Wrong class" in cb.cget('text'):
            cb.invoke()                            # switch wrong-class OFF
    _click_start(win)
    assert 'config' not in started
    assert "Wrong class" in _status(win)


@pytest.mark.parametrize("field,value", [
    (0, "abc"), (1, "abc"),          # not numbers
    (0, "0"), (1, "0"),              # out of range low
    (0, "1.5"), (1, "1.5"),          # out of range high
])
def test_out_of_range_thresholds_are_refused(opened, field, value):
    app, win, started = opened
    import tkinter.ttk as ttk
    entries = sorted(_widgets(win, ttk.Entry), key=lambda w: w.winfo_rooty())
    entries[field].delete(0, 'end')
    entries[field].insert(0, value)
    _click_start(win)
    assert 'config' not in started
    assert _status(win), "refused silently, with no reason given"


def test_a_valid_config_starts_the_scan(opened, ds):
    app, win, started = opened
    _click_start(win)
    cfg = started.get('config')
    assert cfg is not None
    assert cfg['conf'] == pytest.approx(0.40)
    assert cfg['iou_match'] == pytest.approx(0.50)
    assert cfg['mismatch'] and cfg['missing']
    assert cfg['paths'] == list(app.filtered_image_paths)


def test_scope_can_be_widened_to_all_images(opened):
    app, win, started = opened
    app.filtered_image_paths = app.image_paths[:2]
    import tkinter.ttk as ttk
    for rb in _widgets(win, ttk.Radiobutton):
        if "All loaded" in rb.cget('text'):
            rb.invoke()
    _click_start(win)
    assert started['config']['paths'] == list(app.image_paths)


def test_a_confidence_below_the_cache_floor_asks_first(opened):
    """Below the floor nothing can be served from cache, so the scan silently
    re-runs the model over everything. The user should hear about it."""
    app, win, started = opened
    import tkinter.ttk as ttk
    entries = sorted(_widgets(win, ttk.Entry), key=lambda w: w.winfo_rooty())
    entries[0].delete(0, 'end')
    entries[0].insert(0, str(_predcache.CONF_FLOOR / 2))

    app.dialogs.answer = False
    _click_start(win)
    assert 'config' not in started, "ran anyway after the user declined"
    assert app.dialogs.asks, "no warning was raised"


def test_accepting_the_cache_warning_proceeds(opened):
    app, win, started = opened
    import tkinter.ttk as ttk
    entries = sorted(_widgets(win, ttk.Entry), key=lambda w: w.winfo_rooty())
    entries[0].delete(0, 'end')
    entries[0].insert(0, str(_predcache.CONF_FLOOR / 2))

    app.dialogs.answer = True
    _click_start(win)
    assert started['config']['conf'] == pytest.approx(_predcache.CONF_FLOOR / 2)


def test_at_or_above_the_floor_no_warning_is_raised(opened):
    app, win, started = opened
    app.dialogs.asks.clear()
    _click_start(win)
    assert not app.dialogs.asks
    assert 'config' in started
