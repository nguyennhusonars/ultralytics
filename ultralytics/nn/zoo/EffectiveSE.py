# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Ported from ultralytics_sonnn nn/modules/EffectiveSE.py (model zoo; see ultralytics/nn/zoo/__init__.py)."""

from torch import nn

from ultralytics.nn.modules.block import C2f
from ultralytics.nn.modules.conv import Conv


class EffectiveSEModule(nn.Module):
    def __init__(self, channels, add_maxpool=False):
        super().__init__()
        self.add_maxpool = add_maxpool
        self.fc = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
        self.gate = nn.Hardsigmoid()

    def forward(self, x):
        x_se = x.mean((2, 3), keepdim=True)
        if self.add_maxpool:
            # experimental codepath, may remove or change
            x_se = 0.5 * x_se + 0.5 * x.amax((2, 3), keepdim=True)
        x_se = self.fc(x_se)
        return x * self.gate(x_se)


class MBConv(nn.Module):
    def __init__(self, inc, ouc, shortcut=True, e=4, dropout=0.1) -> None:
        super().__init__()
        midc = inc * e
        self.conv_pw_1 = Conv(inc, midc, 1)
        self.conv_dw_1 = Conv(midc, midc, 3, g=midc)
        self.effective_se = EffectiveSEModule(midc)
        self.conv1 = Conv(midc, ouc, 1, act=False)
        self.dropout = nn.Dropout2d(p=dropout)
        self.add = shortcut and inc == ouc

    def forward(self, x):
        return (
            x + self.dropout(self.conv1(self.effective_se(self.conv_dw_1(self.conv_pw_1(x)))))
            if self.add
            else self.dropout(self.conv1(self.effective_se(self.conv_dw_1(self.conv_pw_1(x)))))
        )


class C2f_EMBC(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(MBConv(self.c, self.c, shortcut) for _ in range(n))
