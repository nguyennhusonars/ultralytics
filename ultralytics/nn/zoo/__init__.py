# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Model zoo: layers for the community YOLO configs in cfg/models/zoo, ported from the ultralytics_sonnn fork.

parse_model asks the zoo about any layer name upstream does not define. Zoo submodules are imported only when a YAML
names one of their layers, so `import ultralytics` stays fast and optional packages (pywt, einops, ...) are needed only
by the configs that use them. Layer code lives in submodules named after the fork's files; _index maps YAML names to
them, _parse holds the fork's parse_model rules.
"""

import importlib

from ultralytics.nn.modules import Index

from ._index import INDEX
from ._parse import BACKBONES, FEATURE, NAMES, parse

__all__ = ("ZooFeature", "expand", "fuse", "handles", "parse", "resolve")


class ZooFeature(Index):
    """One feature map of a multi-output zoo backbone (rows added by expand())."""


def resolve(name):
    """Return the zoo layer (class or backbone factory) a YAML names; KeyError if neither upstream nor zoo has it."""
    if name == FEATURE:
        return ZooFeature
    if name not in INDEX:
        raise KeyError(f"'{name}' is neither an Ultralytics nor a zoo layer")
    mod, attr = INDEX[name]
    return getattr(importlib.import_module(f"{__name__}.{mod}"), attr)


def handles(m):
    """Whether parse_model should build layer `m` with the zoo rules (zoo layers, and RepConv's fork rule)."""
    return getattr(m, "__name__", None) in NAMES or getattr(m, "__module__", "").startswith(__name__)


def expand(rows):
    """Split each multi-output backbone row into the backbone and 4 feature rows (yaml layer indices 1-4 = P2-P5).

    The fork's backbones return 4 or 5 feature maps and occupied 5 layer slots; one row per feature keeps every later
    'from' index of those YAMLs valid without changing upstream's forward pass.
    """
    out = []
    for f, n, m, args in rows:
        out.append([f, n, m, args])
        if m in BACKBONES:
            i = len(out) - 1
            out.extend([i, 1, FEATURE, [k]] for k in range(4))
    return out


# reparameterisation each zoo layer needs before export, as the fork's fuse() did (method name per class name)
_FUSE = {
    **dict.fromkeys(("RepConvN",), "fuse_convs"),
    **dict.fromkeys(("RepConv_v7",), "fuse_repvgg_block"),
    **dict.fromkeys(
        (
            "MobileOneBlock_origin",
            "MobileOne_origin",
            "MobileOneBlockv5",
            "MobileOnev5",
            "FastViT",
            "MobileViG",
            "UniRepLKNetBlock_pro",
        ),
        "reparameterize",
    ),
    **dict.fromkeys(("OverLoCK",), "reparm"),
    **dict.fromkeys(
        (
            "UniRepLKNet",
            "OREPA",
            "RepVGGBlock_OREPA",
            "OREPA_2",
            "VanillaStem",
            "VanillaBlock",
            "DilatedReparamBlock",
            "UniRepLKNetBlock",
            "RepViT",
            "VanillaNet",
        ),
        "switch_to_deploy",
    ),
}


def fuse(model):
    """Reparameterise zoo layers in place (called from BaseModel.fuse after the upstream Conv/BN fusion)."""
    from ultralytics.utils.torch_utils import fuse_conv_and_bn

    for m in model.modules():
        if not type(m).__module__.startswith(__name__):
            continue
        name = type(m).__name__
        if name == "FDConv_cfg" and hasattr(m, "bn"):
            m.conv = fuse_conv_and_bn(m.conv, m.bn)
            delattr(m, "bn")
            m.forward = m.forward_fuse
        elif name in {"RepVGGBlock", "QARepVGGBlock"} and hasattr(m, "rbr_reparam"):  # the fork's condition, kept as is
            m.switch_to_deploy()
        elif name in _FUSE and hasattr(m, _FUSE[name]):
            getattr(m, _FUSE[name])()
            if name == "RepConvN":
                m.forward = m.forward_fuse
