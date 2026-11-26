# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""The few timm helpers the zoo modules use, so the zoo does not depend on timm (0.6 breaks on NumPy 2)."""

import collections.abc
import math
from itertools import repeat

import torch.nn.functional as F
from torch import nn

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


_ACT = {
    "sigmoid": nn.Sigmoid,
    "hard_sigmoid": nn.Hardsigmoid,
    "relu": nn.ReLU,
    "relu6": nn.ReLU6,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
    "swish": nn.SiLU,
    "hard_swish": nn.Hardswish,
    "tanh": nn.Tanh,
    "mish": nn.Mish,
    "elu": nn.ELU,
    "leaky_relu": nn.LeakyReLU,
    "prelu": nn.PReLU,
    "identity": nn.Identity,
}


def create_act_layer(name, inplace=None, **kwargs):
    """Activation module from a timm name or class (timm.layers.create_act_layer)."""
    if name is None:
        return None
    layer = _ACT[name] if isinstance(name, str) else name
    if inplace is not None:
        try:
            return layer(inplace=inplace, **kwargs)
        except TypeError:  # activation without an inplace option (GELU, Sigmoid, ...)
            pass
    return layer(**kwargs)


def make_divisible(v, divisor=8, min_value=None, round_limit=0.9):
    min_value = min_value or divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    return new_v + divisor if new_v < round_limit * v else new_v


class SEModule(nn.Module):
    """Squeeze-and-excitation (timm.layers.SEModule, alias SqueezeExcite)."""

    def __init__(
        self,
        channels,
        rd_ratio=1.0 / 16,
        rd_channels=None,
        rd_divisor=8,
        add_maxpool=False,
        bias=True,
        act_layer=nn.ReLU,
        norm_layer=None,
        gate_layer="sigmoid",
    ):
        super().__init__()
        self.add_maxpool = add_maxpool
        rd_channels = rd_channels or make_divisible(channels * rd_ratio, rd_divisor, round_limit=0.0)
        self.fc1 = nn.Conv2d(channels, rd_channels, kernel_size=1, bias=bias)
        self.bn = norm_layer(rd_channels) if norm_layer else nn.Identity()
        self.act = create_act_layer(act_layer, inplace=True)
        self.fc2 = nn.Conv2d(rd_channels, channels, kernel_size=1, bias=bias)
        self.gate = create_act_layer(gate_layer)

    def forward(self, x):
        x_se = x.mean((2, 3), keepdim=True)
        if self.add_maxpool:
            x_se = 0.5 * x_se + 0.5 * x.amax((2, 3), keepdim=True)
        return x * self.gate(self.fc2(self.act(self.bn(self.fc1(x_se)))))


SqueezeExcite = SEModule


class EffectiveSEModule(nn.Module):
    """Effective squeeze-excitation (timm.layers.EffectiveSEModule)."""

    def __init__(self, channels, add_maxpool=False, gate_layer="hard_sigmoid", **_):
        super().__init__()
        self.add_maxpool = add_maxpool
        self.fc = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
        self.gate = create_act_layer(gate_layer)

    def forward(self, x):
        x_se = x.mean((2, 3), keepdim=True)
        if self.add_maxpool:
            x_se = 0.5 * x_se + 0.5 * x.amax((2, 3), keepdim=True)
        return x * self.gate(self.fc(x_se))


class LayerNorm2d(nn.LayerNorm):
    """LayerNorm over channels of NCHW tensors (timm.layers.LayerNorm2d)."""

    def __init__(self, num_channels, eps=1e-6, affine=True):
        super().__init__(num_channels, eps=eps, elementwise_affine=affine)

    def forward(self, x):
        return F.layer_norm(x.permute(0, 2, 3, 1), self.normalized_shape, self.weight, self.bias, self.eps).permute(
            0, 3, 1, 2
        )


def named_apply(fn, module, name="", depth_first=True, include_root=False):
    """Apply fn(module=, name=) to every submodule (timm.models.named_apply)."""
    if not depth_first and include_root:
        fn(module=module, name=name)
    for child_name, child in module.named_children():
        named_apply(fn, child, f"{name}.{child_name}" if name else child_name, depth_first, include_root=True)
    if depth_first and include_root:
        fn(module=module, name=name)
    return module


def checkpoint_seq(functions, x, every=1, flatten=False, skip_last=False, preserve_rng_state=True):
    """Run modules in sequence with activation checkpointing (timm.models.checkpoint_seq).

    shortcut: checkpoints every module (ignores every/flatten/skip_last); only runs when a model enables grad
    checkpointing.
    """
    from torch.utils.checkpoint import checkpoint

    for f in functions:
        x = checkpoint(f, x, use_reentrant=False, preserve_rng_state=preserve_rng_state)
    return x


def register_model(fn):
    """Timm's model registry is not used here: return the factory unchanged."""
    return fn


def _cfg(url="", **kwargs):
    return {"url": url, "num_classes": 1000, "input_size": (3, 224, 224), **kwargs}


__all__ = (
    "IMAGENET_DEFAULT_MEAN",
    "IMAGENET_DEFAULT_STD",
    "DropPath",
    "EffectiveSEModule",
    "LayerNorm2d",
    "Mlp",
    "SEModule",
    "SqueezeExcite",
    "_cfg",
    "checkpoint_seq",
    "create_act_layer",
    "drop_path",
    "lecun_normal_",
    "make_divisible",
    "named_apply",
    "register_model",
    "to_2tuple",
    "to_3tuple",
    "to_ntuple",
    "trunc_normal_",
)
