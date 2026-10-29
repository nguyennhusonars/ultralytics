# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Ported from ultralytics_sonnn nn/modules/moe/routers.py (model zoo; see ultralytics/nn/zoo/__init__.py)."""

from typing import Dict
from typing import Tuple
from ultralytics.nn.zoo.moe.utils import FlopsUtils
import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseRouter(nn.Module):
    def __init__(self, num_experts, top_k):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.softmax = nn.Softmax(dim=1)

    def _process_logits(
        self, logits: torch.Tensor, noise_std: float, training: bool
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """Unified logic to process logits into Top-K selection."""
        B = logits.shape[0]

        # 1) Add noise during training (simplified Gumbel-Softmax trick)
        if training and noise_std > 0:
            logits = logits + torch.randn_like(logits) * noise_std

        # 2) Compute probabilities
        probs = F.softmax(logits.float(), dim=1).type_as(logits)

        # 3) Select Top-K
        topk_vals, topk_indices = torch.topk(probs, self.top_k, dim=1)

        # 4) Normalize weights
        sum_vals = topk_vals.sum(dim=1, keepdim=True) + 1e-6
        topk_vals = topk_vals / sum_vals

        # 5) Collect loss-related info (train only)
        loss_dict = {}
        if training:
            loss_dict["router_logits"] = logits
            loss_dict["router_probs"] = probs
            loss_dict["topk_indices"] = topk_indices

        return topk_vals, topk_indices, loss_dict


class EfficientSpatialRouter(BaseRouter):
    def __init__(self, in_channels, num_experts, reduction=8, top_k=2, noise_std=1.0, pool_scale=4):
        super().__init__(num_experts, top_k)
        self.noise_std = noise_std
        self.pool_scale = pool_scale
        reduced_channels = max(in_channels // reduction, 8)

        self.router = nn.Sequential(
            nn.Conv2d(in_channels, reduced_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(reduced_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(reduced_channels, num_experts, 1, bias=False),
            nn.BatchNorm2d(num_experts),  # numerical stability
        )

    def forward(self, x):
        B, C, H, W = x.shape
        # Pre-pooling optimization
        if H > self.pool_scale and W > self.pool_scale:
            x_in = F.avg_pool2d(x, kernel_size=self.pool_scale, stride=self.pool_scale)
        else:
            x_in = x

        out = self.router(x_in)  # [B, E, H', W']
        global_logits = torch.mean(out, dim=[2, 3])  # [B, E]

        return self._process_logits(global_logits, self.noise_std, self.training)

    def compute_flops(self, input_shape):
        B, C, H, W = input_shape
        h_down, w_down = max(H // self.pool_scale, 1), max(W // self.pool_scale, 1)
        return FlopsUtils.count_conv2d(self.router, (B, C, h_down, w_down))


class AdaptiveRoutingLayer(BaseRouter):
    def __init__(self, in_channels, num_experts, reduction=8, top_k=2, noise_std=1.0):
        super().__init__(num_experts, top_k)
        self.noise_std = noise_std
        reduced_channels = max(in_channels // reduction, 8)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.router = nn.Sequential(
            nn.Conv2d(in_channels, reduced_channels, 1, bias=False),
            nn.BatchNorm2d(reduced_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(reduced_channels, num_experts, 1, bias=False),
            nn.BatchNorm2d(num_experts),
        )

    def forward(self, x):
        pooled = self.avg_pool(x)
        logits = self.router(pooled).squeeze(-1).squeeze(-1)  # [B, E]
        return self._process_logits(logits, self.noise_std, self.training)

    def compute_flops(self, input_shape):
        # FLOPs here are minimal
        return FlopsUtils.count_conv2d(self.router, (input_shape[0], input_shape[1], 1, 1))


class LocalRoutingLayer(BaseRouter):
    def __init__(self, in_channels, num_experts, reduction=8, top_k=2, noise_std=1.0):
        super().__init__(num_experts, top_k)
        self.noise_std = noise_std
        # Even for local routing, default to 2x downsampling to save FLOPs with minimal texture loss
        self.pool_scale = 2

        reduced_channels = max(in_channels // reduction, 8)
        self.router = nn.Sequential(
            nn.Conv2d(in_channels, reduced_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(reduced_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(reduced_channels, num_experts, 1, bias=False),
            nn.BatchNorm2d(num_experts),
        )

    def forward(self, x):
        # Moderate downsampling to accelerate
        if x.shape[2] > self.pool_scale:
            x_in = F.avg_pool2d(x, kernel_size=self.pool_scale, stride=self.pool_scale)
        else:
            x_in = x

        out = self.router(x_in)
        global_logits = torch.mean(out, dim=[2, 3])
        return self._process_logits(global_logits, self.noise_std, self.training)

    def compute_flops(self, input_shape):
        B, C, H, W = input_shape
        h_d, w_d = max(H // self.pool_scale, 1), max(W // self.pool_scale, 1)
        return FlopsUtils.count_conv2d(self.router, (B, C, h_d, w_d))


class DynamicRoutingLayer(nn.Module):
    def __init__(self, in_channels, num_experts=3, reduction=8, top_k=None):
        """
        Args:
            top_k: Number of active experts; if None uses all experts (Softmax)
        """
        super(DynamicRoutingLayer, self).__init__()
        reduced_channels = max(in_channels // reduction, 8)

        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts) if top_k is not None else num_experts
        self.use_top_k = top_k is not None  # whether to enable Top-K

        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Remove Softmax and control manually
        self.routing_network = nn.Sequential(
            nn.Conv2d(in_channels, reduced_channels, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(reduced_channels, num_experts, kernel_size=1),
        )

    def forward(self, x):
        pooled = self.global_pool(x)
        routing_logits = self.routing_network(pooled)  # [B, num_experts, 1, 1]

        # Choose strategy based on Top-K enablement and train/infer mode
        if not self.use_top_k:
            # No Top-K: direct Softmax
            routing_weights = F.softmax(routing_logits.float(), dim=1).type_as(x)
        elif self.training:
            # Training: soft Top-K (keeps gradients flowing)
            routing_weights = self._soft_top_k(routing_logits)
        else:
            # Inference: hard Top-K (truly sparse)
            routing_weights = self._hard_top_k(routing_logits)

        return routing_weights.repeat(1, 1, x.size(2), x.size(3))

    def _soft_top_k(self, logits):
        """Soft Top-K during training to maintain gradient flow."""
        B, E, H, W = logits.shape
        logits_flat = logits.view(B, E, -1)

        # Compute softmax
        # Fix: Clamp logits to avoid overflow
        logits_flat = logits_flat.clamp(-30.0, 30.0)
        weights = F.softmax(logits_flat.float(), dim=1).type_as(logits)

        # Find Top-K and build mask
        _, topk_indices = torch.topk(weights, self.top_k, dim=1)
        idx = topk_indices.permute(0, 2, 1).contiguous()
        mask_one_hot = F.one_hot(idx, num_classes=E).sum(dim=2)
        mask_one_hot = mask_one_hot.permute(0, 2, 1).contiguous().to(weights.dtype)

        # Apply mask and re-normalize
        weights = weights * mask_one_hot
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-6)

        return weights.view(B, E, H, W)

    def _hard_top_k(self, logits):
        """Hard Top-K during inference for true sparsity."""
        B, E, H, W = logits.shape
        logits_flat = logits.view(B, E, -1)

        # Find Top-K
        topk_values, topk_indices = torch.topk(logits_flat, self.top_k, dim=1)

        # Apply softmax to Top-K logits
        # Fix: Clamp values to avoid overflow before softmax
        topk_values = topk_values.clamp(-30.0, 30.0)
        topk_weights = F.softmax(topk_values.float(), dim=1).type_as(logits)

        # Construct sparse weights
        idx = topk_indices.permute(0, 2, 1).contiguous()
        oh = F.one_hot(idx, num_classes=E)
        tw = topk_weights.permute(0, 2, 1).contiguous()
        weighted = (oh.to(tw.dtype) * tw.unsqueeze(-1)).sum(dim=2)
        weights = weighted.permute(0, 2, 1).contiguous()

        return weights.view(B, E, H, W)
