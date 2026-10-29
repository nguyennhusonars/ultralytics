# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Ported from ultralytics_sonnn nn/modules/head.py (model zoo; see ultralytics/nn/zoo/__init__.py)."""

from __future__ import annotations

import torch
import torch.nn as nn

from ultralytics.nn.modules.head import Detect
from ultralytics.nn.zoo.conv import ImplicitA, ImplicitM


class IDetect(Detect):
    """YOLOR detection head: Detect with implicit knowledge, an ImplicitA before and an ImplicitM after each branch."""

    def __init__(self, nc: int = 80, reg_max: int = 16, end2end: bool = False, ch: tuple = ()):
        """Build the Detect branches plus one ImplicitA/ImplicitM pair per detection level for box and class."""
        super().__init__(nc, reg_max, end2end, ch)
        self.ia2 = nn.ModuleList(ImplicitA(x) for x in ch)
        self.ia3 = nn.ModuleList(ImplicitA(x) for x in ch)
        self.im2 = nn.ModuleList(ImplicitM(4 * self.reg_max) for _ in ch)
        self.im3 = nn.ModuleList(ImplicitM(self.nc) for _ in ch)

    def forward_head(
        self, x: list[torch.Tensor], box_head: nn.Module | None = None, cls_head: nn.Module | None = None
    ) -> dict[str, torch.Tensor]:
        """Detect.forward_head with each box/class branch wrapped as im(branch(ia(x)))."""
        if box_head is None or cls_head is None:  # for fused inference
            return {}
        bs = x[0].shape[0]  # batch size
        boxes = [self.im2[i](box_head[i](self.ia2[i](x[i]))).view(bs, 4 * self.reg_max, -1) for i in range(self.nl)]
        scores = [self.im3[i](cls_head[i](self.ia3[i](x[i]))).view(bs, self.nc, -1) for i in range(self.nl)]
        return {"boxes": torch.cat(boxes, dim=-1), "scores": torch.cat(scores, dim=-1), "feats": x}
