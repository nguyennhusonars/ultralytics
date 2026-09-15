"""Mapping a model's prediction classes onto the dataset's classes.

A pretrained model rarely shares your dataset's class list or ordering: a COCO
model calls `2` a car, while your dataset might call `2` a `bus_m`. Accepting
such a prediction verbatim writes the model's id into your label file, silently
mislabeling the object.

This module keeps that translation in one place. The map is applied once, at the
point predictions enter the application, so every consumer downstream -- colours,
class-visibility, filters, accept, statistics -- only ever sees dataset ids.

A model class with no entry is *unmapped*: its predictions are dropped rather
than guessed at, because a wrong label is worse than a missing one.
"""

from __future__ import annotations

from dataclasses import dataclass, field


def normalize(name: str) -> str:
    """Loose key for name matching: case, spaces, dashes and underscores."""
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def auto_map_by_name(
    model_names: dict[int, str], class_info: dict[int, dict]
) -> dict[int, int]:
    """Seed a mapping by matching class NAMES between model and dataset.

    Matching is loose (`Bus_M`, `bus m` and `busm` all match), which is what
    makes the common case -- a model fine-tuned on a similar taxonomy -- map
    itself with no manual work. Names that do not match are simply absent.
    """
    by_name: dict[str, int] = {}
    for ds_id, info in class_info.items():
        key = normalize(info.get("name", ""))
        # First definition wins, so duplicate dataset names stay deterministic.
        by_name.setdefault(key, int(ds_id))

    out: dict[int, int] = {}
    for m_id, m_name in (model_names or {}).items():
        ds_id = by_name.get(normalize(m_name))
        if ds_id is not None:
            out[int(m_id)] = ds_id
    return out


@dataclass
class PredictionClassMap:
    """Translates model class ids to dataset class ids.

    Attributes:
        mapping: ``{model_class_id: dataset_class_id}``. Absent keys are unmapped.
        drop_unmapped: When True (the default) predictions whose class has no
            entry are discarded. When False they are kept with their original id,
            which is the legacy behaviour and will write model ids into labels.
        enabled: Master switch, so the map can be turned off without losing it.
    """

    mapping: dict[int, int] = field(default_factory=dict)
    drop_unmapped: bool = True
    enabled: bool = True

    def translate(self, model_cls_id) -> int | None:
        """Dataset id for a model class, or None if it should be dropped."""
        cls = int(model_cls_id)
        if not self.enabled:
            return cls
        if cls in self.mapping:
            return int(self.mapping[cls])
        return None if self.drop_unmapped else cls

    def apply(self, preds):
        """Remap a list of ``(cls_id, x1, y1, x2, y2)`` predictions."""
        out = []
        for cls_id, *box in preds:
            ds = self.translate(cls_id)
            if ds is not None:
                out.append((ds, *box))
        return out

    # -- many-to-one helpers -------------------------------------------------
    def map_all_to(self, model_ids, dataset_id: int) -> None:
        """Point several model classes at one dataset class.

        Useful for collapsing a fine-grained model onto a coarse dataset, e.g.
        COCO car/truck/bus all becoming a single `vehicle`.
        """
        for m in model_ids:
            self.mapping[int(m)] = int(dataset_id)

    def unmap(self, model_ids) -> None:
        for m in model_ids:
            self.mapping.pop(int(m), None)

    # -- persistence ---------------------------------------------------------
    def to_json(self) -> dict:
        return {
            "mapping": {str(k): int(v) for k, v in sorted(self.mapping.items())},
            "drop_unmapped": bool(self.drop_unmapped),
            "enabled": bool(self.enabled),
        }

    @classmethod
    def from_json(cls, data: dict | None) -> "PredictionClassMap":
        data = data or {}
        raw = data.get("mapping", {}) or {}
        mapping = {}
        for k, v in raw.items():
            try:
                mapping[int(k)] = int(v)
            except (TypeError, ValueError):
                continue          # tolerate a hand-edited config
        return cls(
            mapping=mapping,
            drop_unmapped=bool(data.get("drop_unmapped", True)),
            enabled=bool(data.get("enabled", True)),
        )

    def summary(self, model_names: dict[int, str], class_info: dict[int, dict]) -> str:
        """One-line human description, for the status bar."""
        n_model = len(model_names or {})
        n_mapped = len([m for m in self.mapping if m in (model_names or {})])
        if not self.enabled:
            return "prediction class mapping: OFF (model ids used as-is)"
        action = "dropped" if self.drop_unmapped else "kept as-is"
        return (f"prediction class mapping: {n_mapped}/{n_model} model classes mapped, "
                f"rest {action}")
