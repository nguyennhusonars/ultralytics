# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""DEIMv2 (DETR with Improved Matching v2) decoder module for object detection.

Adapted from the DEIMv2 repository (https://github.com/ShihuaHuang95/DEIMv2).
"""

from __future__ import annotations

import copy
import math
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import constant_, xavier_uniform_

from ultralytics.utils.torch_utils import TORCH_1_11


def _bilinear_grid_sample(im, grid):
    """Bilinear grid sample compatible with ONNX opset 11 (no aten::grid_sampler).

    Equivalent to F.grid_sample(im, grid, mode='bilinear', padding_mode='zeros', align_corners=False).

    Args:
        im: (N, C, H, W) input feature map.
        grid: (N, Hout, Wout, 2) grid normalized to [-1, 1].

    Returns:
        (N, C, Hout, Wout) sampled values.
    """
    N, C, H, W = im.shape
    # Extract grid x/y via reshape to keep Gather inputs rank <= 3 for ONNX
    _gshape = grid.shape[:-1]  # (N, Hout, Wout)
    _gflat = grid.reshape(-1, 2)  # (N*Hout*Wout, 2)
    gx = _gflat[:, 0].reshape(_gshape)  # (N, Hout, Wout)
    gy = _gflat[:, 1].reshape(_gshape)
    # Denormalize (align_corners=False)
    gx = ((gx + 1.0) * W - 1.0) / 2.0
    gy = ((gy + 1.0) * H - 1.0) / 2.0
    # Replace NaN/Inf with 0 for robustness (skip for ONNX to avoid IsInf/IsNaN/Where ops)
    if not torch.onnx.is_in_onnx_export():
        gx = torch.where(torch.isfinite(gx), gx, torch.zeros_like(gx))
        gy = torch.where(torch.isfinite(gy), gy, torch.zeros_like(gy))

    gx0 = gx.floor()
    gy0 = gy.floor()
    gx1 = gx0 + 1.0
    gy1 = gy0 + 1.0

    # Bilinear weights
    wa = ((gx1 - gx) * (gy1 - gy)).unsqueeze(1)
    wb = ((gx1 - gx) * (gy - gy0)).unsqueeze(1)
    wc = ((gx - gx0) * (gy1 - gy)).unsqueeze(1)
    wd = ((gx - gx0) * (gy - gy0)).unsqueeze(1)

    # Zero-padding masks using arithmetic (avoids Less/And/Not/Cast ops in ONNX)
    # For integer-valued coords from floor(): clamp(x+1,0,1) ≡ (x>=0), clamp(W-x,0,1) ≡ (x<W)
    mask_x0 = (gx0 + 1).clamp(0, 1) * (W - gx0).clamp(0, 1)  # 0 <= gx0 < W
    mask_x1 = (gx1 + 1).clamp(0, 1) * (W - gx1).clamp(0, 1)  # 0 <= gx1 < W
    mask_y0 = (gy0 + 1).clamp(0, 1) * (H - gy0).clamp(0, 1)  # 0 <= gy0 < H
    mask_y1 = (gy1 + 1).clamp(0, 1) * (H - gy1).clamp(0, 1)  # 0 <= gy1 < H
    ma = (mask_x0 * mask_y0).unsqueeze(1)
    mb = (mask_x0 * mask_y1).unsqueeze(1)
    mc = (mask_x1 * mask_y0).unsqueeze(1)
    md = (mask_x1 * mask_y1).unsqueeze(1)

    # Clamp as float first (Clip op doesn't support int64 in ONNX opset 11), then cast
    gx0i = gx0.clamp(0, W - 1).long()
    gy0i = gy0.clamp(0, H - 1).long()
    gx1i = gx1.clamp(0, W - 1).long()
    gy1i = gy1.clamp(0, H - 1).long()

    # Flatten spatial
    im_flat = im.reshape(N, C, H * W)
    spatial_shape = gx.shape[1:]

    if torch.onnx.is_in_onnx_export():
        # Use index_select (maps to ONNX Gather op) per batch element
        # N = bs*n_head is a concrete int at trace time, so range() unrolls
        def _sample(yi, xi):
            parts = []
            for i in range(N):
                linear_idx = (yi[i] * W + xi[i]).reshape(-1)  # 1D index
                sampled = torch.index_select(im_flat[i], 1, linear_idx)  # (C, pts)
                parts.append(sampled)
            return torch.stack(parts, 0).reshape(N, C, *spatial_shape)
    else:
        def _sample(yi, xi):
            idx = (yi * W + xi).reshape(N, 1, -1).expand(-1, C, -1)
            return im_flat.gather(2, idx).reshape(N, C, *spatial_shape)

    return (
        _sample(gy0i, gx0i) * wa * ma
        + _sample(gy1i, gx0i) * wb * mb
        + _sample(gy0i, gx1i) * wc * mc
        + _sample(gy1i, gx1i) * wd * md
    )

from .utils import bias_init_with_prob

__all__ = ("DEIMv2Decoder",)


# --- Utility functions ---


def inverse_sigmoid(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Compute inverse sigmoid of a tensor."""
    x = x.clamp(min=0.0, max=1.0)
    return torch.log(x.clamp(min=eps) / (1 - x).clamp(min=eps))


def get_activation(act: str | nn.Module | None = "relu") -> nn.Module:
    """Get activation function by name."""
    if act is None:
        return nn.Identity()
    if isinstance(act, nn.Module):
        return act
    act = act.lower()
    if act in ("silu", "swish"):
        return nn.SiLU(inplace=True)
    if act == "relu":
        return nn.ReLU(inplace=True)
    if act == "gelu":
        return nn.GELU()
    if act == "leaky_relu":
        return nn.LeakyReLU(inplace=True)
    raise ValueError(f"Unknown activation: {act}")


def weighting_function(reg_max, up, reg_scale, deploy=False):
    """Generate non-uniform Weighting Function W(n) for bounding box regression."""
    if deploy:
        upper_bound1 = (abs(up[0]) * abs(reg_scale)).item()
        upper_bound2 = (abs(up[0]) * abs(reg_scale) * 2).item()
        step = (upper_bound1 + 1) ** (2 / (reg_max - 2))
        left_values = [-(step) ** i + 1 for i in range(reg_max // 2 - 1, 0, -1)]
        right_values = [(step) ** i - 1 for i in range(1, reg_max // 2)]
        values = [-upper_bound2] + left_values + [torch.zeros_like(up[0][None])] + right_values + [upper_bound2]
        return torch.tensor(values, dtype=up.dtype, device=up.device)
    else:
        upper_bound1 = abs(up[0]) * abs(reg_scale)
        upper_bound2 = abs(up[0]) * abs(reg_scale) * 2
        step = (upper_bound1 + 1) ** (2 / (reg_max - 2))
        left_values = [-(step) ** i + 1 for i in range(reg_max // 2 - 1, 0, -1)]
        right_values = [(step) ** i - 1 for i in range(1, reg_max // 2)]
        values = [-upper_bound2] + left_values + [torch.zeros_like(up[0][None])] + right_values + [upper_bound2]
        return torch.cat(values, 0)


def distance2bbox(points, distance, reg_scale):
    """Decode edge-distances into bounding box coordinates (cxcywh)."""
    reg_scale = abs(reg_scale)
    x1 = points[..., 0] - (0.5 * reg_scale + distance[..., 0]) * (points[..., 2] / reg_scale)
    y1 = points[..., 1] - (0.5 * reg_scale + distance[..., 1]) * (points[..., 3] / reg_scale)
    x2 = points[..., 0] + (0.5 * reg_scale + distance[..., 2]) * (points[..., 2] / reg_scale)
    y2 = points[..., 1] + (0.5 * reg_scale + distance[..., 3]) * (points[..., 3] / reg_scale)
    bboxes = torch.stack([x1, y1, x2, y2], -1)
    # xyxy -> cxcywh
    x0, y0, x1, y1 = bboxes.unbind(-1)
    return torch.stack([(x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0), (y1 - y0)], dim=-1)


def bbox2distance(points, bbox, reg_max, reg_scale, up, eps=0.1):
    """Convert bounding boxes to distances from a reference point."""
    reg_scale_abs = abs(reg_scale)
    left = (points[:, 0] - bbox[:, 0]) / (points[..., 2] / reg_scale_abs + 1e-16) - 0.5 * reg_scale_abs
    top = (points[:, 1] - bbox[:, 1]) / (points[..., 3] / reg_scale_abs + 1e-16) - 0.5 * reg_scale_abs
    right = (bbox[:, 2] - points[:, 0]) / (points[..., 2] / reg_scale_abs + 1e-16) - 0.5 * reg_scale_abs
    bottom = (bbox[:, 3] - points[:, 1]) / (points[..., 3] / reg_scale_abs + 1e-16) - 0.5 * reg_scale_abs
    four_lens = torch.stack([left, top, right, bottom], -1)
    four_lens, weight_right, weight_left = translate_gt(four_lens, reg_max, reg_scale, up)
    if reg_max is not None:
        four_lens = four_lens.clamp(min=0, max=reg_max - eps)
    return four_lens.reshape(-1).detach(), weight_right.detach(), weight_left.detach()


def translate_gt(gt, reg_max, reg_scale, up):
    """Decode GT values into distribution-based GT representations."""
    gt = gt.reshape(-1)
    function_values = weighting_function(reg_max, up, reg_scale)
    diffs = function_values.unsqueeze(0) - gt.unsqueeze(1)
    mask = diffs <= 0
    closest_left_indices = torch.sum(mask, dim=1) - 1
    indices = closest_left_indices.float()
    weight_right = torch.zeros_like(indices)
    weight_left = torch.zeros_like(indices)
    valid_idx_mask = (indices >= 0) & (indices < reg_max)
    valid_indices = indices[valid_idx_mask].long()
    left_values = function_values[valid_indices]
    right_values = function_values[valid_indices + 1]
    left_diffs = torch.abs(gt[valid_idx_mask] - left_values)
    right_diffs = torch.abs(right_values - gt[valid_idx_mask])
    weight_right[valid_idx_mask] = left_diffs / (left_diffs + right_diffs)
    weight_left[valid_idx_mask] = 1.0 - weight_right[valid_idx_mask]
    invalid_idx_mask_neg = indices < 0
    weight_right[invalid_idx_mask_neg] = 0.0
    weight_left[invalid_idx_mask_neg] = 1.0
    indices[invalid_idx_mask_neg] = 0.0
    invalid_idx_mask_pos = indices >= reg_max
    weight_right[invalid_idx_mask_pos] = 1.0
    weight_left[invalid_idx_mask_pos] = 0.0
    indices[invalid_idx_mask_pos] = reg_max - 0.1
    return indices, weight_right, weight_left


def box_cxcywh_to_xyxy(x):
    """Convert boxes from cxcywh to xyxy format."""
    x_c, y_c, w, h = x.unbind(-1)
    b = [
        (x_c - 0.5 * w.clamp(min=0.0)),
        (y_c - 0.5 * h.clamp(min=0.0)),
        (x_c + 0.5 * w.clamp(min=0.0)),
        (y_c + 0.5 * h.clamp(min=0.0)),
    ]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    """Convert boxes from xyxy to cxcywh format."""
    x0, y0, x1, y1 = x.unbind(-1)
    return torch.stack([(x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0), (y1 - y0)], dim=-1)


# --- Building Blocks ---


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self._norm(x.float()).type_as(x) * self.scale


class SwiGLUFFN(nn.Module):
    """SwiGLU Feed-Forward Network."""

    def __init__(self, in_features: int, hidden_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.w12 = nn.Linear(in_features, 2 * hidden_features, bias=bias)
        self.w3 = nn.Linear(hidden_features, out_features, bias=bias)
        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self.w12.weight)
        constant_(self.w12.bias, 0)
        xavier_uniform_(self.w3.weight)
        constant_(self.w3.bias, 0)

    def forward(self, x):
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        return self.w3(F.silu(x1) * x2)


class Gate(nn.Module):
    """Gating mechanism for cross-attention fusion."""

    def __init__(self, d_model, use_rmsnorm=False):
        super().__init__()
        self.gate = nn.Linear(2 * d_model, 2 * d_model)
        bias = bias_init_with_prob(0.5)
        constant_(self.gate.bias, bias)
        constant_(self.gate.weight, 0)
        self.norm = RMSNorm(d_model) if use_rmsnorm else nn.LayerNorm(d_model)

    def forward(self, x1, x2):
        gate_input = torch.cat([x1, x2], dim=-1)
        gates = torch.sigmoid(self.gate(gate_input))
        gate1, gate2 = gates.chunk(2, dim=-1)
        return self.norm(gate1 * x1 + gate2 * x2)


class DEIMv2MLP(nn.Module):
    """Multi-layer perceptron used in DEIMv2."""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3, act="relu"):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.act = get_activation(act)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = self.act(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class Integral(nn.Module):
    """Compute integral of distribution: sum{Pr(n) * W(n)}."""

    def __init__(self, reg_max=32):
        super().__init__()
        self.reg_max = reg_max

    def forward(self, x, project):
        shape = x.shape
        x = F.softmax(x.reshape(-1, self.reg_max + 1), dim=1)
        x = F.linear(x, project.to(x.device)).reshape(-1, 4)
        return x.reshape(list(shape[:-1]) + [-1])


class LQE(nn.Module):
    """Location Quality Estimator."""

    def __init__(self, k, hidden_dim, num_layers, reg_max, act="relu"):
        super().__init__()
        self.k = k
        self.reg_max = reg_max
        self.reg_conf = DEIMv2MLP(4 * (k + 1), hidden_dim, 1, num_layers, act=act)
        constant_(self.reg_conf.layers[-1].bias, 0)
        constant_(self.reg_conf.layers[-1].weight, 0)

    def forward(self, scores, pred_corners):
        B, L, _ = pred_corners.size()
        prob = F.softmax(pred_corners.reshape(B, L, 4, self.reg_max + 1), dim=-1)
        if torch.onnx.is_in_onnx_export():
            # Replace TopK with iterative ReduceMax (platform only supports 1D TopK)
            topk_vals = []
            p = prob
            for _ in range(self.k):
                max_val = p.max(dim=-1, keepdim=True).values
                topk_vals.append(max_val)
                # Soft-mask out the max position using arithmetic only
                eps = 1e-7
                marker = torch.relu(p - max_val + eps)
                mask = (marker * (1.0 / eps)).clamp(0, 1)
                p = p * (1 - mask)
            prob_topk = torch.cat(topk_vals, dim=-1)
        else:
            prob_topk, _ = prob.topk(self.k, dim=-1)
        stat = torch.cat([prob_topk, prob_topk.mean(dim=-1, keepdim=True)], dim=-1)
        quality_score = self.reg_conf(stat.reshape(B, L, -1))
        return scores + quality_score


class MSDeformableAttention(nn.Module):
    """Multi-Scale Deformable Attention module."""

    def __init__(self, embed_dim=256, num_heads=8, num_levels=4, num_points=4, method="default", offset_scale=0.5):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.offset_scale = offset_scale

        if isinstance(num_points, list):
            assert len(num_points) == num_levels
            num_points_list = num_points
        else:
            num_points_list = [num_points for _ in range(num_levels)]

        self.num_points_list = num_points_list
        num_points_scale = [1 / n for n in num_points_list for _ in range(n)]
        self.register_buffer("num_points_scale", torch.tensor(num_points_scale, dtype=torch.float32))

        self.total_points = num_heads * sum(num_points_list)
        self.method = method
        self.head_dim = embed_dim // num_heads

        self.sampling_offsets = nn.Linear(embed_dim, self.total_points * 2)
        self.attention_weights = nn.Linear(embed_dim, self.total_points)
        self._reset_parameters()

        if method == "discrete":
            for p in self.sampling_offsets.parameters():
                p.requires_grad = False

    def _reset_parameters(self):
        constant_(self.sampling_offsets.weight, 0)
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = grid_init / grid_init.abs().max(-1, keepdim=True).values
        grid_init = grid_init.reshape(self.num_heads, 1, 2).tile([1, sum(self.num_points_list), 1])
        scaling = torch.cat([torch.arange(1, n + 1) for n in self.num_points_list]).reshape(1, -1, 1)
        grid_init *= scaling
        self.sampling_offsets.bias.data[...] = grid_init.flatten()
        constant_(self.attention_weights.weight, 0)
        constant_(self.attention_weights.bias, 0)

    def forward(self, query, reference_points, value, value_spatial_shapes):
        bs, Len_q = query.shape[:2]
        n_pts = sum(self.num_points_list)
        sampling_offsets = self.sampling_offsets(query)

        if torch.onnx.is_in_onnx_export():
            # Keep all intermediate tensors at rank <= 4 for embedded platform
            # Offsets: (bs, Len_q, n_head*n_pts*2) → (bs*n_head, Len_q, n_pts, 2) via rank-4 steps
            sampling_offsets = sampling_offsets.reshape(bs, Len_q, self.num_heads, n_pts * 2)
            sampling_offsets = sampling_offsets.permute(0, 2, 1, 3)  # (bs, n_head, Len_q, n_pts*2)
            sampling_offsets = sampling_offsets.reshape(bs * self.num_heads, Len_q, n_pts, 2)

            attention_weights = self.attention_weights(query).reshape(bs, Len_q, self.num_heads, n_pts)
            attention_weights = F.softmax(attention_weights, dim=-1)

            if reference_points.shape[-1] == 4:
                num_points_scale = self.num_points_scale.to(dtype=query.dtype).unsqueeze(-1)
                # Slice ref to xy/wh at rank 4, broadcast with (bs*n_head, Len_q, n_pts, 2)
                ref_xy = reference_points[:, :, :, :2]  # (bs, Len_q, 1, 2)
                ref_wh = reference_points[:, :, :, 2:]  # (bs, Len_q, 1, 2)
                offset = sampling_offsets * num_points_scale * ref_wh * self.offset_scale
                sampling_locations = ref_xy + offset
            else:
                raise ValueError("ONNX export only supports reference_points with shape[-1]==4")
        else:
            sampling_offsets = sampling_offsets.reshape(bs, Len_q, self.num_heads, n_pts, 2)
            attention_weights = self.attention_weights(query).reshape(bs, Len_q, self.num_heads, n_pts)
            attention_weights = F.softmax(attention_weights, dim=-1)

            if reference_points.shape[-1] == 4:
                num_points_scale = self.num_points_scale.to(dtype=query.dtype).unsqueeze(-1)
                offset = sampling_offsets * num_points_scale * reference_points[:, :, None, :, 2:] * self.offset_scale
                sampling_locations = reference_points[:, :, None, :, :2] + offset
            elif reference_points.shape[-1] == 2:
                offset_normalizer = torch.tensor(value_spatial_shapes)
                offset_normalizer = offset_normalizer.flip([1]).reshape(1, 1, 1, self.num_levels, 1, 2)
                sampling_locations = (
                    reference_points.reshape(bs, Len_q, 1, self.num_levels, 1, 2)
                    + sampling_offsets / offset_normalizer
                )
            else:
                raise ValueError(f"Last dim of reference_points must be 2 or 4, got {reference_points.shape[-1]}")

        output = self._deformable_attention_core(
            value, value_spatial_shapes, sampling_locations, attention_weights
        )
        return output

    def _deformable_attention_core(self, value, value_spatial_shapes, sampling_locations, attention_weights):
        bs, n_head, c, _ = value[0].shape

        if sampling_locations.ndim == 5:
            # Training path: (bs, Len_q, n_head, sum_pts, 2)
            _, Len_q, _, _, _ = sampling_locations.shape
            if self.method == "default":
                sampling_grids = 2 * sampling_locations - 1
            elif self.method == "discrete":
                sampling_grids = sampling_locations
            else:
                sampling_grids = 2 * sampling_locations - 1
            sampling_grids = sampling_grids.permute(0, 2, 1, 3, 4).flatten(0, 1)
        else:
            # ONNX path: already (bs*n_head, Len_q, sum_pts, 2) — rank 4
            _, Len_q, _, _ = sampling_locations.shape
            if self.method == "default":
                sampling_grids = 2 * sampling_locations - 1
            elif self.method == "discrete":
                sampling_grids = sampling_locations
            else:
                sampling_grids = 2 * sampling_locations - 1
        sampling_locations_list = sampling_grids.split(self.num_points_list, dim=-2)

        sampling_value_list = []
        for level, (h, w) in enumerate(value_spatial_shapes):
            value_l = value[level].reshape(bs * n_head, c, h, w)
            sampling_grid_l = sampling_locations_list[level]

            if self.method == "default":
                sampling_value_l = _bilinear_grid_sample(value_l, sampling_grid_l)
            elif self.method == "discrete":
                sampling_coord = (
                    sampling_grid_l * torch.tensor([[w, h]], device=value_l.device) + 0.5
                ).to(torch.int64)
                sampling_coord = sampling_coord.clamp(0, h - 1)
                sampling_coord = sampling_coord.reshape(bs * n_head, Len_q * self.num_points_list[level], 2)
                s_idx = (
                    torch.arange(sampling_coord.shape[0], device=value_l.device)
                    .unsqueeze(-1)
                    .repeat(1, sampling_coord.shape[1])
                )
                sampling_value_l = value_l[s_idx, :, sampling_coord[..., 1], sampling_coord[..., 0]]
                sampling_value_l = sampling_value_l.permute(0, 2, 1).reshape(
                    bs * n_head, c, Len_q, self.num_points_list[level]
                )
            else:
                sampling_value_l = _bilinear_grid_sample(value_l, sampling_grid_l)
            sampling_value_list.append(sampling_value_l)

        attn_weights = attention_weights.permute(0, 2, 1, 3).reshape(
            bs * n_head, 1, Len_q, sum(self.num_points_list)
        )
        weighted = torch.cat(sampling_value_list, dim=-1) * attn_weights
        output = weighted.sum(-1).reshape(bs, n_head * c, Len_q)
        return output.permute(0, 2, 1)


# --- Decoder layers ---


class DEIMv2DecoderLayer(nn.Module):
    """Single DEIMv2 transformer decoder layer with self-attention, cross-attention, and SwiGLU FFN."""

    def __init__(
        self,
        d_model=256,
        n_head=8,
        dim_feedforward=1024,
        dropout=0.0,
        activation="relu",
        n_levels=4,
        n_points=4,
        cross_attn_method="default",
        layer_scale=None,
        use_gateway=False,
    ):
        super().__init__()
        if layer_scale is not None:
            dim_feedforward = round(layer_scale * dim_feedforward)
            d_model = round(layer_scale * d_model)

        # Self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = RMSNorm(d_model)

        # Cross attention
        self.cross_attn = MSDeformableAttention(d_model, n_head, n_levels, n_points, method=cross_attn_method)
        self.dropout2 = nn.Dropout(dropout)

        self.use_gateway = use_gateway
        if use_gateway:
            self.gateway = Gate(d_model, use_rmsnorm=True)
        else:
            self.norm2 = RMSNorm(d_model)

        # FFN (SwiGLU)
        self.swish_ffn = SwiGLUFFN(d_model, dim_feedforward // 2, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = RMSNorm(d_model)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, target, reference_points, value, spatial_shapes, attn_mask=None, query_pos_embed=None):
        # Self attention
        q = k = self.with_pos_embed(target, query_pos_embed)
        target2, _ = self.self_attn(q, k, value=target, attn_mask=attn_mask)
        target = target + self.dropout1(target2)
        target = self.norm1(target)

        # Cross attention
        target2 = self.cross_attn(self.with_pos_embed(target, query_pos_embed), reference_points, value, spatial_shapes)
        if self.use_gateway:
            target = self.gateway(target, self.dropout2(target2))
        else:
            target = target + self.dropout2(target2)
            target = self.norm2(target)

        # FFN
        target2 = self.swish_ffn(target)
        target = target + self.dropout4(target2)
        target = self.norm3(target.clamp(min=-65504, max=65504))
        return target


class DEIMv2TransformerDecoder(nn.Module):
    """Transformer Decoder with Fine-grained Distribution Refinement (FDR)."""

    def __init__(self, hidden_dim, decoder_layer, decoder_layer_wide, num_layers, num_head, reg_max, reg_scale, up,
                 eval_idx=-1, layer_scale=2, act="relu"):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.layer_scale = layer_scale
        self.num_head = num_head
        self.eval_idx = eval_idx if eval_idx >= 0 else num_layers + eval_idx
        self.up, self.reg_scale, self.reg_max = up, reg_scale, reg_max
        self.layers = nn.ModuleList(
            [copy.deepcopy(decoder_layer) for _ in range(self.eval_idx + 1)]
            + [copy.deepcopy(decoder_layer_wide) for _ in range(num_layers - self.eval_idx - 1)]
        )
        self.lqe_layers = nn.ModuleList(
            [copy.deepcopy(LQE(4, 64, 2, reg_max, act=act)) for _ in range(num_layers)]
        )

    def value_op(self, memory, value_proj, value_scale, memory_mask, memory_spatial_shapes):
        """Preprocess values for MSDeformableAttention."""
        value = value_proj(memory) if value_proj is not None else memory
        value = F.interpolate(memory, size=value_scale) if value_scale is not None else value
        if memory_mask is not None:
            value = value * memory_mask.to(value.dtype).unsqueeze(-1)
        value = value.reshape(value.shape[0], value.shape[1], self.num_head, -1)
        split_shape = [h * w for h, w in memory_spatial_shapes]
        return value.permute(0, 2, 3, 1).split(split_shape, dim=-1)

    def convert_to_deploy(self):
        """Convert decoder to deploy mode by pruning unused layers."""
        self.project = weighting_function(self.reg_max, self.up, self.reg_scale, deploy=True)
        self.layers = self.layers[:self.eval_idx + 1]
        self.lqe_layers = nn.ModuleList(
            [nn.Identity()] * self.eval_idx + [self.lqe_layers[self.eval_idx]]
        )

    def forward(
        self, target, ref_points_unact, memory, spatial_shapes,
        bbox_head, score_head, query_pos_head, pre_bbox_head, integral, up, reg_scale,
        attn_mask=None, memory_mask=None, dn_meta=None,
    ):
        output = target
        output_detach = pred_corners_undetach = 0
        value = self.value_op(memory, None, None, memory_mask, spatial_shapes)

        dec_out_bboxes = []
        dec_out_logits = []
        dec_out_pred_corners = []
        dec_out_refs = []
        if not hasattr(self, "project"):
            project = weighting_function(self.reg_max, up, reg_scale)
        else:
            project = self.project

        ref_points_detach = F.sigmoid(ref_points_unact)
        query_pos_embed = query_pos_head(ref_points_detach).clamp(min=-10, max=10)

        for i, layer in enumerate(self.layers):
            ref_points_input = ref_points_detach.unsqueeze(2)

            if i >= self.eval_idx + 1 and self.layer_scale > 1:
                query_pos_embed = F.interpolate(query_pos_embed, scale_factor=self.layer_scale)
                value = self.value_op(memory, None, query_pos_embed.shape[-1], memory_mask, spatial_shapes)
                output = F.interpolate(output, size=query_pos_embed.shape[-1])
                output_detach = output.detach()

            output = layer(output, ref_points_input, value, spatial_shapes, attn_mask, query_pos_embed)

            if i == 0:
                pre_bboxes = F.sigmoid(pre_bbox_head(output) + inverse_sigmoid(ref_points_detach))
                pre_scores = score_head[0](output)
                ref_points_initial = pre_bboxes.detach()

            pred_corners = bbox_head[i](output + output_detach) + pred_corners_undetach
            inter_ref_bbox = distance2bbox(ref_points_initial, integral(pred_corners, project), reg_scale)

            if self.training or i == self.eval_idx:
                scores = score_head[i](output)
                scores = self.lqe_layers[i](scores, pred_corners)
                dec_out_logits.append(scores)
                dec_out_bboxes.append(inter_ref_bbox)
                dec_out_pred_corners.append(pred_corners)
                dec_out_refs.append(ref_points_initial)
                if not self.training:
                    break

            pred_corners_undetach = pred_corners
            ref_points_detach = inter_ref_bbox.detach()
            output_detach = output.detach()

        return (
            torch.stack(dec_out_bboxes),
            torch.stack(dec_out_logits),
            torch.stack(dec_out_pred_corners),
            torch.stack(dec_out_refs),
            pre_bboxes,
            pre_scores,
        )


# --- Denoising ---


def get_contrastive_denoising_training_group(
    targets, num_classes, num_queries, class_embed,
    num_denoising=100, label_noise_ratio=0.5, box_noise_scale=1.0,
):
    """Generate contrastive denoising training group for the DEIM decoder."""
    if num_denoising <= 0:
        return None, None, None, None

    num_gts = [len(t["labels"]) for t in targets]
    device = targets[0]["labels"].device
    max_gt_num = max(num_gts)
    if max_gt_num == 0:
        return None, None, None, None

    num_group = num_denoising // max_gt_num
    num_group = 1 if num_group == 0 else num_group
    bs = len(num_gts)

    input_query_class = torch.full([bs, max_gt_num], num_classes, dtype=torch.int32, device=device)
    input_query_bbox = torch.zeros([bs, max_gt_num, 4], device=device)
    pad_gt_mask = torch.zeros([bs, max_gt_num], dtype=torch.bool, device=device)

    for i in range(bs):
        num_gt = num_gts[i]
        if num_gt > 0:
            input_query_class[i, :num_gt] = targets[i]["labels"]
            input_query_bbox[i, :num_gt] = targets[i]["boxes"]
            pad_gt_mask[i, :num_gt] = 1

    input_query_class = input_query_class.tile([1, 2 * num_group])
    input_query_bbox = input_query_bbox.tile([1, 2 * num_group, 1])
    pad_gt_mask = pad_gt_mask.tile([1, 2 * num_group])

    negative_gt_mask = torch.zeros([bs, max_gt_num * 2, 1], device=device)
    negative_gt_mask[:, max_gt_num:] = 1
    negative_gt_mask = negative_gt_mask.tile([1, num_group, 1])
    positive_gt_mask = 1 - negative_gt_mask

    positive_gt_mask = positive_gt_mask.squeeze(-1) * pad_gt_mask
    dn_positive_idx = torch.nonzero(positive_gt_mask)[:, 1]
    dn_positive_idx = torch.split(dn_positive_idx, [n * num_group for n in num_gts])

    num_denoising = int(max_gt_num * 2 * num_group)

    if label_noise_ratio > 0:
        mask = torch.rand_like(input_query_class, dtype=torch.float) < (label_noise_ratio * 0.5)
        new_label = torch.randint_like(mask, 0, num_classes, dtype=input_query_class.dtype)
        input_query_class = torch.where(mask & pad_gt_mask, new_label, input_query_class)

    if box_noise_scale > 0:
        known_bbox = box_cxcywh_to_xyxy(input_query_bbox)
        diff = torch.tile(input_query_bbox[..., 2:] * 0.5, [1, 1, 2]) * box_noise_scale
        rand_sign = torch.randint_like(input_query_bbox, 0, 2) * 2.0 - 1.0
        rand_part = torch.rand_like(input_query_bbox)
        rand_part = (rand_part + 1.0) * negative_gt_mask + rand_part * (1 - negative_gt_mask)
        known_bbox += rand_sign * rand_part * diff
        known_bbox = torch.clip(known_bbox, min=0.0, max=1.0)
        input_query_bbox = box_xyxy_to_cxcywh(known_bbox)
        input_query_bbox[input_query_bbox < 0] *= -1
        input_query_bbox_unact = inverse_sigmoid(input_query_bbox)

    input_query_logits = class_embed(input_query_class)

    tgt_size = num_denoising + num_queries
    attn_mask = torch.full([tgt_size, tgt_size], False, dtype=torch.bool, device=device)
    attn_mask[num_denoising:, :num_denoising] = True
    for i in range(num_group):
        if i == 0:
            attn_mask[max_gt_num * 2 * i : max_gt_num * 2 * (i + 1), max_gt_num * 2 * (i + 1) : num_denoising] = True
        if i == num_group - 1:
            attn_mask[max_gt_num * 2 * i : max_gt_num * 2 * (i + 1), : max_gt_num * i * 2] = True
        else:
            attn_mask[max_gt_num * 2 * i : max_gt_num * 2 * (i + 1), max_gt_num * 2 * (i + 1) : num_denoising] = True
            attn_mask[max_gt_num * 2 * i : max_gt_num * 2 * (i + 1), : max_gt_num * 2 * i] = True

    dn_meta = {
        "dn_positive_idx": dn_positive_idx,
        "dn_num_group": num_group,
        "dn_num_split": [num_denoising, num_queries],
    }
    return input_query_logits, input_query_bbox_unact, attn_mask, dn_meta


# --- Main DEIMv2 Decoder Head ---


class DEIMv2Decoder(nn.Module):
    """DEIMv2 Decoder for object detection, integrating transformer decoder with fine-grained distribution refinement.

    This module serves as the head of the DEIMv2 detection pipeline within the Ultralytics framework.
    It takes multi-scale feature maps from the backbone/neck and produces detection predictions.

    Attributes:
        export (bool): Export mode flag.
        nc (int): Number of classes.
        hidden_dim (int): Hidden dimension size.
        num_queries (int): Number of decoder queries.
        num_levels (int): Number of feature levels.
    """

    export = False
    dynamic = False
    shapes = []
    anchors = torch.empty(0)
    valid_mask = torch.empty(0)

    def __init__(
        self,
        nc: int = 80,
        ch: tuple = (256, 256, 256),
        hd: int = 256,  # hidden dim
        nq: int = 300,  # num queries
        ndl: int = 6,  # num decoder layers
        nl: int = 3,  # num feature levels
        np: int = 4,  # num deformable points
        reg_max: int = 32,
        reg_scale: float = 4.0,
        nh: int = 8,  # num heads
        d_ffn: int = 1024,  # dim of feedforward
        share_bbox_head: bool = False,
        use_gateway: bool = True,
        dropout: float = 0.0,
        act: str = "relu",
        eval_idx: int = -1,
        # Training args
        nd: int = 100,  # num denoising
        label_noise_ratio: float = 0.5,
        box_noise_scale: float = 1.0,
        learnt_init_query: bool = False,
        # DEIMv2-specific
        layer_scale: int = 1,
        cross_attn_method: str = "default",
        query_select_method: str = "default",
        mlp_act: str = "relu",
        share_score_head: bool = False,
    ):
        """Initialize the DEIMv2Decoder.

        Args:
            nc: Number of classes.
            ch: Channel sizes from backbone feature maps.
            hd: Hidden dimension.
            nq: Number of query points.
            ndl: Number of decoder layers.
            nl: Number of feature levels.
            np: Number of deformable attention points.
            reg_max: Max discrete bins for distribution-based regression.
            reg_scale: Scale factor for weighting function.
            nh: Number of attention heads.
            d_ffn: Feed-forward network dimension.
            dropout: Dropout rate.
            act: Activation function name.
            eval_idx: Which decoder layer to use at eval (-1 = last).
            nd: Number of denoising queries.
            label_noise_ratio: Denoising label noise ratio.
            box_noise_scale: Denoising box noise scale.
            learnt_init_query: Whether to use learnable query embeddings.
            layer_scale: Width scaling for later decoder layers.
            cross_attn_method: Cross attention method ('default' or 'discrete').
            query_select_method: Query selection method ('default', 'one2many', 'agnostic').
            mlp_act: MLP activation function name.
            use_gateway: Whether to use gating for cross-attention fusion.
            share_bbox_head: Share bbox head across decoder layers.
            share_score_head: Share score head across decoder layers.
        """
        super().__init__()
        feat_channels = list(ch)
        feat_strides = [8 * (2 ** i) for i in range(len(feat_channels))]

        assert len(feat_channels) <= nl
        assert len(feat_strides) == len(feat_channels)
        for _ in range(nl - len(feat_strides)):
            feat_strides.append(feat_strides[-1] * 2)

        self.hidden_dim = hd
        scaled_dim = round(layer_scale * hd)
        self.nhead = nh
        self.feat_strides = feat_strides
        self.num_levels = nl
        self.nc = nc
        self.nl = len(ch)
        self.num_queries = nq
        self.eps = 1e-2
        self.num_layers = ndl
        self.eval_spatial_size = None
        self.aux_loss = True
        self.reg_max = reg_max
        self.cross_attn_method = cross_attn_method
        self.query_select_method = query_select_method

        # Backbone feature projection
        self.input_proj = nn.ModuleList()
        for in_channels in feat_channels:
            if in_channels == hd:
                self.input_proj.append(nn.Identity())
            else:
                self.input_proj.append(
                    nn.Sequential(
                        OrderedDict([
                            ("conv", nn.Conv2d(in_channels, hd, 1, bias=False)),
                            ("norm", nn.BatchNorm2d(hd)),
                        ])
                    )
                )
        in_channels = feat_channels[-1]
        for _ in range(nl - len(feat_channels)):
            if in_channels == hd:
                self.input_proj.append(nn.Identity())
            else:
                self.input_proj.append(
                    nn.Sequential(
                        OrderedDict([
                            ("conv", nn.Conv2d(in_channels, hd, 3, 2, padding=1, bias=False)),
                            ("norm", nn.BatchNorm2d(hd)),
                        ])
                    )
                )
                in_channels = hd

        # Transformer
        self.up = nn.Parameter(torch.tensor([0.5]), requires_grad=False)
        self.reg_scale_param = nn.Parameter(torch.tensor([reg_scale]), requires_grad=False)
        decoder_layer = DEIMv2DecoderLayer(
            hd, nh, d_ffn, dropout, act, nl, np,
            cross_attn_method=cross_attn_method, use_gateway=use_gateway,
        )
        decoder_layer_wide = DEIMv2DecoderLayer(
            hd, nh, d_ffn, dropout, act, nl, np,
            cross_attn_method=cross_attn_method, layer_scale=layer_scale, use_gateway=use_gateway,
        )
        self.decoder = DEIMv2TransformerDecoder(
            hd, decoder_layer, decoder_layer_wide, ndl, nh,
            reg_max, self.reg_scale_param, self.up, eval_idx, layer_scale, act=act,
        )

        # Denoising
        self.num_denoising = nd
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale
        if nd > 0:
            self.denoising_class_embed = nn.Embedding(nc + 1, hd, padding_idx=nc)
            nn.init.normal_(self.denoising_class_embed.weight[:-1])

        # Decoder embedding
        self.learnt_init_query = learnt_init_query
        if learnt_init_query:
            self.tgt_embed = nn.Embedding(nq, hd)

        if query_select_method == "agnostic":
            self.enc_score_head = nn.Linear(hd, 1)
        else:
            self.enc_score_head = nn.Linear(hd, nc)
        self.enc_bbox_head = DEIMv2MLP(hd, hd, 4, 3, act=mlp_act)
        self.query_pos_head = DEIMv2MLP(4, hd, hd, 3, act=mlp_act)

        # Decoder head
        self.pre_bbox_head = DEIMv2MLP(hd, hd, 4, 3, act=mlp_act)
        self.integral = Integral(reg_max)

        eval_idx_resolved = eval_idx if eval_idx >= 0 else ndl + eval_idx
        dec_score_head = nn.Linear(hd, nc)
        self.dec_score_head = nn.ModuleList(
            [dec_score_head if share_score_head else copy.deepcopy(dec_score_head) for _ in range(eval_idx_resolved + 1)]
            + [copy.deepcopy(dec_score_head) for _ in range(ndl - eval_idx_resolved - 1)]
        )
        dec_bbox_head = DEIMv2MLP(hd, hd, 4 * (reg_max + 1), 3, act=mlp_act)
        self.dec_bbox_head = nn.ModuleList(
            [dec_bbox_head if share_bbox_head else copy.deepcopy(dec_bbox_head) for _ in range(eval_idx_resolved + 1)]
            + [DEIMv2MLP(scaled_dim, scaled_dim, 4 * (reg_max + 1), 3, act=mlp_act)
               for _ in range(ndl - eval_idx_resolved - 1)]
        )

        self._reset_parameters(feat_channels)

    def _reset_parameters(self, feat_channels):
        bias = bias_init_with_prob(0.01)
        constant_(self.enc_score_head.bias, bias)
        constant_(self.enc_bbox_head.layers[-1].weight, 0)
        constant_(self.enc_bbox_head.layers[-1].bias, 0)
        constant_(self.pre_bbox_head.layers[-1].weight, 0)
        constant_(self.pre_bbox_head.layers[-1].bias, 0)

        for cls_, reg_ in zip(self.dec_score_head, self.dec_bbox_head):
            constant_(cls_.bias, bias)
            if hasattr(reg_, "layers"):
                constant_(reg_.layers[-1].weight, 0)
                constant_(reg_.layers[-1].bias, 0)

        if self.learnt_init_query:
            xavier_uniform_(self.tgt_embed.weight)
        xavier_uniform_(self.query_pos_head.layers[0].weight)
        xavier_uniform_(self.query_pos_head.layers[1].weight)
        xavier_uniform_(self.query_pos_head.layers[-1].weight)
        for m, in_channels in zip(self.input_proj, feat_channels):
            if in_channels != self.hidden_dim:
                if hasattr(m, "__getitem__"):
                    xavier_uniform_(m[0].weight)

    def _get_encoder_input(self, feats):
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]
        if self.num_levels > len(proj_feats):
            len_srcs = len(proj_feats)
            for i in range(len_srcs, self.num_levels):
                if i == len_srcs:
                    proj_feats.append(self.input_proj[i](feats[-1]))
                else:
                    proj_feats.append(self.input_proj[i](proj_feats[-1]))

        feat_flatten = []
        spatial_shapes = []
        for feat in proj_feats:
            _, _, h, w = feat.shape
            feat_flatten.append(feat.flatten(2).permute(0, 2, 1))
            spatial_shapes.append([h, w])

        feat_flatten = torch.cat(feat_flatten, 1)
        return feat_flatten, spatial_shapes

    def convert_to_deploy(self):
        """Convert to deploy mode by pruning unused decoder layers and heads."""
        self.decoder.convert_to_deploy()
        eval_idx = self.decoder.eval_idx
        self.dec_score_head = nn.ModuleList(
            [nn.Identity()] * eval_idx + [self.dec_score_head[eval_idx]]
        )
        self.dec_bbox_head = nn.ModuleList(
            [self.dec_bbox_head[i] if i <= eval_idx else nn.Identity()
             for i in range(len(self.dec_bbox_head))]
        )

    def _generate_anchors(self, spatial_shapes=None, grid_size=0.05, dtype=torch.float32, device="cpu"):
        if spatial_shapes is None:
            spatial_shapes = []
            eval_h, eval_w = self.eval_spatial_size
            for s in self.feat_strides:
                spatial_shapes.append([int(eval_h / s), int(eval_w / s)])

        anchors = []
        for lvl, (h, w) in enumerate(spatial_shapes):
            grid_y, grid_x = torch.meshgrid(
                torch.arange(h), torch.arange(w), indexing="ij"
            ) if TORCH_1_11 else torch.meshgrid(torch.arange(h), torch.arange(w))
            grid_xy = torch.stack([grid_x, grid_y], dim=-1)
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / torch.tensor([w, h], dtype=dtype)
            wh = torch.ones_like(grid_xy) * grid_size * (2.0 ** lvl)
            lvl_anchors = torch.cat([grid_xy, wh], dim=-1).reshape(-1, h * w, 4)
            anchors.append(lvl_anchors)

        anchors = torch.cat(anchors, dim=1).to(device)
        valid_mask = ((anchors > self.eps) & (anchors < 1 - self.eps)).all(-1, keepdim=True)
        anchors = torch.log(anchors / (1 - anchors))
        anchors = torch.where(valid_mask, anchors, torch.tensor(float("inf"), device=device))
        return anchors, valid_mask

    def _get_decoder_input(self, memory, spatial_shapes, denoising_logits=None, denoising_bbox_unact=None):
        if self.training or self.eval_spatial_size is None:
            anchors, valid_mask = self._generate_anchors(spatial_shapes, device=memory.device)
        else:
            anchors = self.anchors
            valid_mask = self.valid_mask
        if memory.shape[0] > 1:
            anchors = anchors.repeat(memory.shape[0], 1, 1)

        memory = valid_mask.to(memory.dtype) * memory
        enc_outputs_logits = self.enc_score_head(memory)

        enc_topk_memory, enc_topk_logits, enc_topk_anchors = self._select_topk(
            memory, enc_outputs_logits, anchors, self.num_queries
        )
        enc_topk_bbox_unact = self.enc_bbox_head(enc_topk_memory) + enc_topk_anchors

        enc_topk_bboxes_list, enc_topk_logits_list = [], []
        if self.training:
            enc_topk_bboxes = F.sigmoid(enc_topk_bbox_unact)
            enc_topk_bboxes_list.append(enc_topk_bboxes)
            enc_topk_logits_list.append(enc_topk_logits)

        if self.learnt_init_query:
            content = self.tgt_embed.weight.unsqueeze(0).tile([memory.shape[0], 1, 1])
        else:
            content = enc_topk_memory.detach()

        enc_topk_bbox_unact = enc_topk_bbox_unact.detach()

        if denoising_bbox_unact is not None:
            enc_topk_bbox_unact = torch.cat([denoising_bbox_unact, enc_topk_bbox_unact], dim=1)
            content = torch.cat([denoising_logits, content], dim=1)

        return content, enc_topk_bbox_unact, enc_topk_bboxes_list, enc_topk_logits_list

    def _select_topk(self, memory, outputs_logits, outputs_anchors_unact, topk):
        if torch.onnx.is_in_onnx_export():
            # 1D TopK for ONNX (batch=1, squeeze to 1D so platform gets a 1D TopK node)
            if self.query_select_method == "default":
                topk_input = outputs_logits.max(-1).values[0]  # (N,)
            elif self.query_select_method == "one2many":
                topk_input = outputs_logits[0].flatten()  # (N*nc,)
            elif self.query_select_method == "agnostic":
                topk_input = outputs_logits[0, :, 0]  # (N,)
            _, ind_1d = torch.topk(topk_input, topk, dim=0)  # 1D TopK
            if self.query_select_method == "one2many":
                ind_1d = ind_1d // self.nc
            topk_anchors = torch.index_select(outputs_anchors_unact[0], 0, ind_1d).unsqueeze(0)
            topk_logits = None
            topk_memory = torch.index_select(memory[0], 0, ind_1d).unsqueeze(0)
        else:
            if self.query_select_method == "default":
                _, topk_ind = torch.topk(outputs_logits.max(-1).values, topk, dim=-1)
            elif self.query_select_method == "one2many":
                _, topk_ind = torch.topk(outputs_logits.flatten(1), topk, dim=-1)
                topk_ind = topk_ind // self.nc
            elif self.query_select_method == "agnostic":
                _, topk_ind = torch.topk(outputs_logits.squeeze(-1), topk, dim=-1)
            topk_anchors = outputs_anchors_unact.gather(
                dim=1, index=topk_ind.unsqueeze(-1).repeat(1, 1, outputs_anchors_unact.shape[-1])
            )
            topk_logits = (
                outputs_logits.gather(dim=1, index=topk_ind.unsqueeze(-1).repeat(1, 1, outputs_logits.shape[-1]))
                if self.training
                else None
            )
            topk_memory = memory.gather(
                dim=1, index=topk_ind.unsqueeze(-1).repeat(1, 1, memory.shape[-1])
            )
        return topk_memory, topk_logits, topk_anchors

    def forward(self, x: list[torch.Tensor], batch: dict | None = None) -> tuple | torch.Tensor:
        """Forward pass of the DEIMv2 decoder.

        Args:
            x: List of feature maps from backbone/neck.
            batch: Batch data for training (ground truth).

        Returns:
            Training: dict with pred_logits, pred_boxes, pred_corners, and auxiliary outputs.
            Inference: Tensor of shape (bs, nq, 4+nc) or tuple with extended info.
        """
        # Input projection and embedding
        memory, spatial_shapes = self._get_encoder_input(x)

        # Prepare denoising training
        if self.training and self.num_denoising > 0 and batch is not None:
            targets = self._prepare_targets(batch)
            denoising_logits, denoising_bbox_unact, attn_mask, dn_meta = (
                get_contrastive_denoising_training_group(
                    targets, self.nc, self.num_queries, self.denoising_class_embed,
                    num_denoising=self.num_denoising,
                    label_noise_ratio=self.label_noise_ratio,
                    box_noise_scale=1.0,
                )
            )
        else:
            denoising_logits, denoising_bbox_unact, attn_mask, dn_meta = None, None, None, None

        init_ref_contents, init_ref_points_unact, enc_topk_bboxes_list, enc_topk_logits_list = (
            self._get_decoder_input(memory, spatial_shapes, denoising_logits, denoising_bbox_unact)
        )

        # Decoder
        out_bboxes, out_logits, out_corners, out_refs, pre_bboxes, pre_logits = self.decoder(
            init_ref_contents, init_ref_points_unact, memory, spatial_shapes,
            self.dec_bbox_head, self.dec_score_head, self.query_pos_head,
            self.pre_bbox_head, self.integral, self.up, self.reg_scale_param,
            attn_mask=attn_mask, dn_meta=dn_meta,
        )

        if self.training and dn_meta is not None:
            dn_pre_logits, pre_logits = torch.split(pre_logits, dn_meta["dn_num_split"], dim=1)
            dn_pre_bboxes, pre_bboxes = torch.split(pre_bboxes, dn_meta["dn_num_split"], dim=1)
            dn_out_logits, out_logits = torch.split(out_logits, dn_meta["dn_num_split"], dim=2)
            dn_out_bboxes, out_bboxes = torch.split(out_bboxes, dn_meta["dn_num_split"], dim=2)
            dn_out_corners, out_corners = torch.split(out_corners, dn_meta["dn_num_split"], dim=2)
            dn_out_refs, out_refs = torch.split(out_refs, dn_meta["dn_num_split"], dim=2)

        if self.training:
            out = {
                "pred_logits": out_logits[-1],
                "pred_boxes": out_bboxes[-1],
                "pred_corners": out_corners[-1],
                "ref_points": out_refs[-1],
                "up": self.up,
                "reg_scale": self.reg_scale_param,
            }
            if self.aux_loss:
                out["aux_outputs"] = [
                    {
                        "pred_logits": a, "pred_boxes": b, "pred_corners": c, "ref_points": d,
                        "teacher_corners": out_corners[-1], "teacher_logits": out_logits[-1],
                    }
                    for a, b, c, d in zip(out_logits[:-1], out_bboxes[:-1], out_corners[:-1], out_refs[:-1])
                ]
                out["enc_aux_outputs"] = [
                    {"pred_logits": a, "pred_boxes": b}
                    for a, b in zip(enc_topk_logits_list, enc_topk_bboxes_list)
                ]
                out["pre_outputs"] = {"pred_logits": pre_logits, "pred_boxes": pre_bboxes}
                out["enc_meta"] = {"class_agnostic": self.query_select_method == "agnostic"}

                if dn_meta is not None:
                    out["dn_outputs"] = [
                        {
                            "pred_logits": a, "pred_boxes": b, "pred_corners": c, "ref_points": d,
                            "teacher_corners": dn_out_corners[-1], "teacher_logits": dn_out_logits[-1],
                        }
                        for a, b, c, d in zip(dn_out_logits, dn_out_bboxes, dn_out_corners, dn_out_refs)
                    ]
                    out["dn_pre_outputs"] = {"pred_logits": dn_pre_logits, "pred_boxes": dn_pre_bboxes}
                    out["dn_meta"] = dn_meta
            return out
        else:
            # Inference: return (bs, nq, 4+nc) tensor
            # Use squeeze(0) instead of [-1] to avoid Gather on rank-4 tensor in ONNX
            pred_logits = out_logits.squeeze(0) if torch.onnx.is_in_onnx_export() else out_logits[-1]
            pred_boxes = out_bboxes.squeeze(0) if torch.onnx.is_in_onnx_export() else out_bboxes[-1]
            # boxes are in cxcywh normalized format, scores need sigmoid
            y = torch.cat((pred_boxes, pred_logits.sigmoid()), -1)
            return y if self.export else (y, (out_bboxes, out_logits, out_corners, out_refs, pre_bboxes, pre_logits))

    def _prepare_targets(self, batch):
        """Convert Ultralytics batch format to DEIMv2's target format."""
        bs = batch["img"].shape[0]
        batch_idx = batch["batch_idx"]
        targets = []
        for i in range(bs):
            mask = batch_idx == i
            targets.append({
                "labels": batch["cls"][mask].to(torch.long).view(-1),
                "boxes": batch["bboxes"][mask],
            })
        return targets
