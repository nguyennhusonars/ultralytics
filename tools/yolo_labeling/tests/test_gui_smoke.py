"""The five GUI tests worth having. Everything else is on the manual checklist.

Rendering, crop-review layout, dialog widget trees and matplotlib figures are
deliberately NOT tested: a visual bug is obvious in one second, and such tests
rot faster than they catch anything. What IS worth pinning is the wiring that
breaks silently -- bindings that vanish, class panels that desync, focus guards
that stop guarding.

Run with: xvfb-run -a pytest -m gui
"""

from __future__ import annotations

import pytest

from fixtures import assert_only_changed, snapshot

pytestmark = [pytest.mark.gui, pytest.mark.legacy]


def test_application_constructs_without_error(gui_app):
    """The likeliest package-split regression: import order or widget wiring."""
    assert gui_app.root.winfo_exists()
    assert gui_app.dialogs.errors == [], f"error dialog raised: {gui_app.dialogs.errors}"


def test_all_keyboard_shortcuts_survive(gui_app):
    """A dropped binding is otherwise only discoverable by a user pressing a key
    and getting silence."""
    bound = set(gui_app.root.bind())
    # Tk reports plain-character bindings bare ("s") and special keys in angle
    # brackets ("<Key-Left>"). Both forms are pinned deliberately.
    expected = {
        "s", "d", "a", "f", "h", "?", "P", "L", "U", "k", "K",
        "<Key-Left>", "<Key-Right>", "<Key-Delete>", "<Key-Return>", "<Key-Escape>",
        "<Control-Key-v>", "<Control-Key-z>", "<Control-Key-y>",
        "+", "-", "=", "<Key-KP_Add>", "<Key-KP_Subtract>",
    } | {str(d) for d in range(10)} | {f"<Key-KP_{d}>" for d in range(10)}
    missing = {b for b in expected if b not in bound}
    assert not missing, f"missing bindings: {sorted(missing)}"
    assert len(bound) >= 45, f"only {len(bound)} bindings registered"


def test_class_buttons_match_the_class_registry(gui_app):
    """`setup_class_buttons` rebuilds the panel on every class change."""
    assert gui_app.class_info, "classes loaded from data.yaml"
    names = {c["name"] for c in gui_app.class_info.values()}
    assert {"person", "car", "bus_m", "bus_l"} <= names


def test_entry_focus_guard_classifies_every_text_widget(gui_app, monkeypatch):
    """`d` deletes the current image and `s` saves. If this guard regresses,
    typing in a filter box starts deleting files.

    Real X focus cannot be exercised under Xvfb -- with no window manager,
    `focus_get()` never leaves the toplevel. That plumbing is Tk's job anyway;
    what can actually regress is the predicate below, e.g. someone adds a
    Spinbox filter and the guard silently stops covering it.
    """
    import tkinter as tk
    from tkinter import ttk

    text_widgets = [
        tk.Entry(gui_app.root),
        ttk.Entry(gui_app.root),
        ttk.Combobox(gui_app.root),
        tk.Text(gui_app.root),
        tk.Spinbox(gui_app.root),
    ]
    for w in text_widgets:
        monkeypatch.setattr(gui_app.root, "focus_get", lambda w=w: w)
        assert gui_app._is_entry_focused() is True, f"{type(w).__name__} not guarded"

    for w in (gui_app.canvas, tk.Button(gui_app.root), tk.Frame(gui_app.root)):
        monkeypatch.setattr(gui_app.root, "focus_get", lambda w=w: w)
        assert gui_app._is_entry_focused() is False, f"{type(w).__name__} wrongly guarded"

    monkeypatch.setattr(gui_app.root, "focus_get", lambda: None)
    assert gui_app._is_entry_focused() is False, "no focus must not suppress hotkeys"


def test_batch_view_renders_and_writes_nothing(gui_app, ds):
    """Covers the reduced-decode + thumbnail-cache change: tiles must render,
    boxes must be scaled from FULL-resolution dimensions, and pure browsing
    must not touch a single label file."""
    before = snapshot(ds.root)
    # toggle_batch_view reads batch_size_var; "1x1" means single-image view, so
    # a grid size has to be selected for batch mode to engage at all.
    gui_app.batch_size_var.set("2x2")
    gui_app.toggle_batch_view()
    gui_app.root.update_idletasks()
    assert gui_app.batch_mode is True
    assert gui_app.batch_canvases, "batch canvases were created"

    gui_app.load_batch_images()
    gui_app.root.update_idletasks()
    assert gui_app._batch_thumb_cache, "thumbnails were cached"

    n_cached = len(gui_app._batch_thumb_cache)
    gui_app.load_batch_images()                     # same page again
    assert len(gui_app._batch_thumb_cache) == n_cached, "second pass reused the cache"

    gui_app.batch_size_var.set("1x1")
    gui_app.toggle_batch_view()
    gui_app.root.update_idletasks()
    assert gui_app.batch_mode is False
    assert_only_changed(before, snapshot(ds.root), set())
