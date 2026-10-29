# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Ported from ultralytics_sonnn nn/modules/conv.py (model zoo; see ultralytics/nn/zoo/__init__.py)."""

import torch
import torch.nn as nn


class ImplicitA(nn.Module):
    def __init__(self, channel, mean=0.0, std=0.02):
        super(ImplicitA, self).__init__()
        self.channel = channel
        self.mean = mean
        self.std = std
        self.implicit = nn.Parameter(torch.zeros(1, channel, 1, 1))
        nn.init.normal_(self.implicit, mean=self.mean, std=self.std)

    def forward(self, x):
        return self.implicit + x


class ImplicitM(nn.Module):
    def __init__(self, channel, mean=1.0, std=0.02):
        super(ImplicitM, self).__init__()
        self.channel = channel
        self.mean = mean
        self.std = std
        self.implicit = nn.Parameter(torch.ones(1, channel, 1, 1))
        nn.init.normal_(self.implicit, mean=self.mean, std=self.std)

    def forward(self, x):
        return self.implicit * x
