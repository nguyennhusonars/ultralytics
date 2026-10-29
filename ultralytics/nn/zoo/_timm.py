# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""The few timm helpers the zoo modules use, so the zoo does not depend on timm (0.6 breaks on NumPy 2)."""

import collections.abc
import math
from itertools import repeat

import torch.nn as nn

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


def _ntuple(n):
    def parse(x):
        return tuple(x) if isinstance(x, collections.abc.Iterable) and not isinstance(x, str) else tuple(repeat(x, n))

    return parse


to_2tuple, to_3tuple = _ntuple(2), _ntuple(3)


def to_ntuple(n):
    return _ntuple(n)


def drop_path(x, drop_prob: float = 0.0, training: bool = False, scale_by_keep: bool = True):
    """Stochastic depth per sample (timm.layers.drop_path)."""
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Stochastic depth per sample (timm.layers.DropPath)."""

    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        super().__init__()
        self.drop_prob, self.scale_by_keep = drop_prob, scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)


class Mlp(nn.Module):
    """Two-layer MLP (timm.layers.Mlp)."""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0, **kwargs):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        return self.drop2(self.fc2(self.drop1(self.act(self.fc1(x)))))


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    """Truncated normal init; torch's own implementation."""
    return nn.init.trunc_normal_(tensor, mean, std, a, b)


def lecun_normal_(tensor):
    fan_in = nn.init._calculate_fan_in_and_fan_out(tensor)[0]
    return trunc_normal_(tensor, std=math.sqrt(1.0 / fan_in) / 0.87962566103423978)


def register_model(fn):
    """timm's model registry is not used here: return the factory unchanged."""
    return fn


def _cfg(url="", **kwargs):
    return {"url": url, "num_classes": 1000, "input_size": (3, 224, 224), **kwargs}


__all__ = (
    "IMAGENET_DEFAULT_MEAN",
    "IMAGENET_DEFAULT_STD",
    "DropPath",
    "Mlp",
    "_cfg",
    "drop_path",
    "lecun_normal_",
    "register_model",
    "to_2tuple",
    "to_3tuple",
    "to_ntuple",
    "trunc_normal_",
)
