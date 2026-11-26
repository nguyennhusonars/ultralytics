# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Ported from ultralytics_sonnn nn/modules/moe/utils.py (model zoo; see ultralytics/nn/zoo/__init__.py)."""

from __future__ import annotations

from torch import nn


class FlopsUtils:
    @staticmethod
    def count_conv2d(layer: nn.Conv2d | nn.Sequential, input_shape: tuple[int, int, int, int]) -> float:
        B, _C, H, W = input_shape
        if isinstance(layer, nn.Sequential):
            total = 0
            curr_shape = input_shape
            for m in layer:
                if isinstance(m, nn.Conv2d):
                    total += FlopsUtils.count_conv2d(m, curr_shape)
                    # Simple shape derivation
                    curr_h = int((curr_shape[2] + 2 * m.padding[0] - m.kernel_size[0]) / m.stride[0] + 1)
                    curr_w = int((curr_shape[3] + 2 * m.padding[1] - m.kernel_size[1]) / m.stride[1] + 1)
                    curr_shape = (B, m.out_channels, curr_h, curr_w)
            return total

        # Single Conv2d compute
        out_h = (H + 2 * layer.padding[0] - layer.dilation[0] * (layer.kernel_size[0] - 1) - 1) // layer.stride[0] + 1
        out_w = (W + 2 * layer.padding[1] - layer.dilation[1] * (layer.kernel_size[1] - 1) - 1) // layer.stride[1] + 1
        ops = (layer.in_channels // layer.groups) * layer.kernel_size[0] * layer.kernel_size[1]
        ops = (ops + (1 if layer.bias is not None else 0)) * layer.out_channels * out_h * out_w
        return ops * 2.0 * B
