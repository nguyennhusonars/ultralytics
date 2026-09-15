"""Resolving an image path to its label file.

The monolith answered this question in six places with three different
algorithms (L1701 load, L1970 save, L3581 delete, L3699 lookup, L3057 filter,
L6929 cache). They disagree, and the disagreement is a data-loss bug: with
images in a folder not named ``images``, *load* finds ``<parent>/../labels``
while *save* creates ``<parent>/labels`` -- so the tool starts reading its own
shadow copy while the real dataset silently diverges.

`LabelPathResolver` gives one deterministic answer, identical for reading and
writing. Two other fixes fall out of that:

* prefix matching respects path boundaries, so ``/ds/set1`` no longer captures
  ``/ds/set10/a.jpg``;
* when several mappings match, the most specific (longest) one wins instead of
  whichever happened to be first in dict order.
"""

from __future__ import annotations

from pathlib import Path

LABELS_DIRNAME = "labels"
IMAGES_DIRNAME = "images"


def is_within(path: str | Path, folder: str | Path) -> bool:
    """Whether `path` lies inside `folder`, respecting path boundaries.

    Unlike ``str.startswith`` this is False for ``/ds/set10/a.jpg`` against
    ``/ds/set1``.
    """
    try:
        Path(path).resolve().relative_to(Path(folder).resolve())
        return True
    except (ValueError, OSError):
        return False


class LabelPathResolver:
    """Maps image paths to label directories.

    Resolution order:

    1. an explicit ``label_folder`` (a single flat directory for everything);
    2. the longest matching entry in ``per_image`` (the per-folder mapping
       built when loading a multi-path ``data.yaml``);
    3. convention -- ``.../images/x.jpg`` pairs with ``.../labels/x.txt``;
       otherwise a ``labels`` directory beside or above the image, preferring
       one that already exists.

    The answer never depends on whether the caller intends to read or write.
    """

    def __init__(
        self,
        label_folder: str | Path | None = None,
        per_image: dict[str, str] | None = None,
    ):
        self.label_folder = Path(label_folder) if label_folder else None
        self.per_image = {str(k): str(v) for k, v in (per_image or {}).items()}

    def resolve_dir(self, img_path: str | Path) -> Path:
        """Return the label directory for `img_path`. Never returns None."""
        if self.label_folder is not None:
            return self.label_folder

        mapped = self._longest_mapping(img_path)
        if mapped is not None:
            return mapped

        p = Path(img_path)
        if p.parent.name == IMAGES_DIRNAME:
            return p.parent.parent / LABELS_DIRNAME
        # Not the conventional layout: prefer an existing directory so we read
        # and write the same place, else default to a sibling.
        for candidate in (p.parent / LABELS_DIRNAME, p.parent.parent / LABELS_DIRNAME):
            if candidate.is_dir():
                return candidate
        return p.parent / LABELS_DIRNAME

    def resolve(self, img_path: str | Path) -> Path:
        """Return the label *file* path for `img_path`."""
        return self.resolve_dir(img_path) / f"{Path(img_path).stem}.txt"

    def exists(self, img_path: str | Path) -> bool:
        return self.resolve(img_path).is_file()

    def _longest_mapping(self, img_path: str | Path) -> Path | None:
        """Most specific matching mapping entry, or None."""
        best: tuple[int, Path] | None = None
        for img_folder, lbl_folder in self.per_image.items():
            if is_within(img_path, img_folder):
                depth = len(Path(img_folder).parts)
                if best is None or depth > best[0]:
                    best = (depth, Path(lbl_folder))
        return best[1] if best else None

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"LabelPathResolver(label_folder={self.label_folder!r}, "
            f"per_image={len(self.per_image)} entries)"
        )
