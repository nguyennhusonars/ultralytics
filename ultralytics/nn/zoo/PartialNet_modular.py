# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Ported from ultralytics_sonnn nn/modules/PartialNet_modular.py (model zoo; see ultralytics/nn/zoo/__init__.py)."""

from functools import partial
from torch import Tensor
from ultralytics.nn.zoo._timm import DropPath, LayerNorm2d
import math
import torch
import torch.nn as nn


class RPEConfig:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def _no_op_initializer(x):
    return None


@torch.no_grad()
def piecewise_index(relative_position, alpha, beta, gamma, dtype):
    rp_abs = relative_position.abs()
    mask = rp_abs <= alpha
    not_mask = ~mask
    rp_out = relative_position[not_mask]
    rp_abs_out = rp_abs[not_mask]
    y_out = (
        torch.sign(rp_out)
        * (alpha + torch.log(rp_abs_out / alpha) / math.log(gamma / alpha) * (beta - alpha)).round().clip(max=beta)
    ).to(dtype)

    idx = relative_position.clone()
    if idx.dtype in [torch.float32, torch.float64]:
        idx = idx.round().to(dtype)

    idx[not_mask] = y_out
    return idx


def get_absolute_positions(height, width, dtype, device):
    rows = torch.arange(height, dtype=dtype, device=device).view(height, 1).repeat(1, width)
    cols = torch.arange(width, dtype=dtype, device=device).view(1, width).repeat(height, 1)
    return torch.stack([rows, cols], 2)


class METHOD:
    EUCLIDEAN = 0
    QUANT = 1
    PRODUCT = 3
    CROSS = 4
    CROSS_ROWS = 41
    CROSS_COLS = 42


@torch.no_grad()
def _rp_2d_euclidean(diff, **kwargs):
    dis = diff.square().sum(2).float().sqrt().round()
    return piecewise_index(dis, **kwargs)


@torch.no_grad()
def _rp_2d_quant(diff, **kwargs):
    dis = diff.square().sum(2)
    return piecewise_index(dis, **kwargs)


@torch.no_grad()
def _rp_2d_product(diff, **kwargs):
    beta_int = int(kwargs["beta"])
    S = 2 * beta_int + 1
    r = piecewise_index(diff[:, :, 0], **kwargs) + beta_int
    c = piecewise_index(diff[:, :, 1], **kwargs) + beta_int
    pid = r * S + c
    return pid


@torch.no_grad()
def _rp_2d_cross_rows(diff, **kwargs):
    dis = diff[:, :, 0]
    return piecewise_index(dis, **kwargs)


@torch.no_grad()
def _rp_2d_cross_cols(diff, **kwargs):
    dis = diff[:, :, 1]
    return piecewise_index(dis, **kwargs)


_METHOD_FUNC = {
    METHOD.EUCLIDEAN: _rp_2d_euclidean,
    METHOD.QUANT: _rp_2d_quant,
    METHOD.PRODUCT: _rp_2d_product,
    METHOD.CROSS_ROWS: _rp_2d_cross_rows,
    METHOD.CROSS_COLS: _rp_2d_cross_cols,
}


BUCKET_IDS_BUF = dict()


@torch.no_grad()
def get_bucket_ids_2d_without_skip(
    method, height, width, alpha, beta, gamma, dtype=torch.long, device=torch.device("cpu")
):
    key = (method, alpha, beta, gamma, dtype, device)
    value = BUCKET_IDS_BUF.get(key, None)
    if value is None or value[-2] < height or value[-1] < width:
        if value is None:
            max_height, max_width = height, width
        else:
            max_height = max(value[-2], height)
            max_width = max(value[-1], width)

        func = _METHOD_FUNC.get(method, None)
        pos = get_absolute_positions(max_height, max_width, dtype, device)
        max_L = max_height * max_width
        pos1 = pos.view((max_L, 1, 2))
        pos2 = pos.view((1, max_L, 2))
        diff = pos1 - pos2
        bucket_ids = func(diff, alpha=alpha, beta=beta, gamma=gamma, dtype=dtype)
        beta_int = int(beta)
        if method != METHOD.PRODUCT:
            bucket_ids += beta_int
        bucket_ids = bucket_ids.view(max_height, max_width, max_height, max_width)
        num_buckets = get_num_buckets(method, alpha, beta, gamma)
        value = (bucket_ids, num_buckets, height, width)
        BUCKET_IDS_BUF[key] = value
    L = height * width
    bucket_ids = value[0][:height, :width, :height, :width].reshape(L, L)
    num_buckets = value[1]
    return bucket_ids, num_buckets, L


@torch.no_grad()
def get_bucket_ids_2d(method, height, width, skip, alpha, beta, gamma, dtype=torch.long, device=torch.device("cpu")):
    bucket_ids, num_buckets, L = get_bucket_ids_2d_without_skip(
        method, height, width, alpha, beta, gamma, dtype, device
    )
    if skip > 0:
        new_bids = bucket_ids.new_empty(size=(skip + L, skip + L))
        extra_bucket_id = num_buckets
        num_buckets += 1
        new_bids[:skip] = extra_bucket_id
        new_bids[:, :skip] = extra_bucket_id
        new_bids[skip:, skip:] = bucket_ids
        bucket_ids = new_bids
    bucket_ids = bucket_ids.contiguous()
    return bucket_ids, num_buckets


class iRPE(nn.Module):
    _rp_bucket_buf = (None, None, None)

    def __init__(
        self,
        head_dim,
        num_heads=8,
        mode=None,
        method=None,
        transposed=True,
        num_buckets=None,
        initializer=None,
        rpe_config=None,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.mode = mode
        self.method = method
        self.transposed = transposed
        self.num_buckets = num_buckets
        self.initializer = _no_op_initializer if initializer is None else initializer
        self.rpe_config = rpe_config
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        if self.transposed:
            if self.mode == "bias":
                self.lookup_table_bias = nn.Parameter(torch.zeros(self.num_heads, self.num_buckets))
                self.initializer(self.lookup_table_bias)
            elif self.mode == "contextual":
                self.lookup_table_weight = nn.Parameter(torch.zeros(self.num_heads, self.head_dim, self.num_buckets))
                self.initializer(self.lookup_table_weight)
        else:
            if self.mode == "contextual":
                self.lookup_table_weight = nn.Parameter(torch.zeros(self.num_heads, self.num_buckets, self.head_dim))
                self.initializer(self.lookup_table_weight)

    def forward(self, x, height=None, width=None):
        rp_bucket, self._ctx_rp_bucket_flatten = self._get_rp_bucket(x, height=height, width=width)
        if self.transposed:
            return self.forward_rpe_transpose(x, rp_bucket)
        return self.forward_rpe_no_transpose(x, rp_bucket)

    def _get_rp_bucket(self, x, height=None, width=None):
        B, H, L, D = x.shape
        device = x.device
        if height is None:
            E = int(math.sqrt(L))
            height = width = E
        key = (height, width, device)
        if self._rp_bucket_buf[0] == key:
            return self._rp_bucket_buf[1:3]
        skip = L - height * width
        config = self.rpe_config
        rp_bucket, num_buckets = get_bucket_ids_2d(
            method=self.method,
            height=height,
            width=width,
            skip=skip,
            alpha=config.alpha,
            beta=config.beta,
            gamma=config.gamma,
            dtype=torch.long,
            device=device,
        )
        _ctx_rp_bucket_flatten = None
        if self.mode == "contextual" and self.transposed:
            offset = torch.arange(
                0, L * self.num_buckets, self.num_buckets, dtype=rp_bucket.dtype, device=rp_bucket.device
            ).view(-1, 1)
            _ctx_rp_bucket_flatten = (rp_bucket + offset).flatten()
        self._rp_bucket_buf = (key, rp_bucket, _ctx_rp_bucket_flatten)
        return rp_bucket, _ctx_rp_bucket_flatten

    def forward_rpe_transpose(self, x, rp_bucket):
        B = len(x)
        L_query, L_key = rp_bucket.shape
        if self.mode == "bias":
            return self.lookup_table_bias[:, rp_bucket.flatten()].view(1, self.num_heads, L_query, L_key)
        elif self.mode == "contextual":
            lookup_table = (
                torch.matmul(x.transpose(0, 1).reshape(-1, B * L_query, self.head_dim), self.lookup_table_weight)
                .view(-1, B, L_query, self.num_buckets)
                .transpose(0, 1)
            )
            return lookup_table.flatten(2)[:, :, self._ctx_rp_bucket_flatten].view(B, -1, L_query, L_key)

    def forward_rpe_no_transpose(self, x, rp_bucket):
        L_query, L_key = rp_bucket.shape
        weight = self.lookup_table_weight[:, rp_bucket.flatten()].view(self.num_heads, L_query, L_key, self.head_dim)
        return torch.matmul(x.permute(1, 2, 0, 3), weight).permute(2, 0, 1, 3)


def get_num_buckets(method, alpha, beta, gamma):
    beta_int = int(beta)
    if method == METHOD.PRODUCT:
        return (2 * beta_int + 1) ** 2
    return 2 * beta_int + 1


def get_rpe_config(ratio=1.9, method=METHOD.PRODUCT, mode="contextual", shared_head=True, skip=0, rpe_on="k"):
    if isinstance(method, str):
        method_mapping = dict(euc=METHOD.EUCLIDEAN, quant=METHOD.QUANT, cross=METHOD.CROSS, product=METHOD.PRODUCT)
        method = method_mapping[method.lower()]
    config = RPEConfig()
    kwargs = dict(ratio=ratio, method=method, mode=mode, shared_head=shared_head, skip=skip)

    def get_single_rpe_config(ratio=1.9, method=METHOD.PRODUCT, mode="contextual", shared_head=True, skip=0):
        c = RPEConfig()
        c.shared_head = shared_head
        c.mode = mode
        c.method = method
        c.alpha = 1 * ratio
        c.beta = 2 * ratio
        c.gamma = 8 * ratio
        c.num_buckets = get_num_buckets(method, c.alpha, c.beta, c.gamma)
        if skip > 0:
            c.num_buckets += 1
        return c

    config.rpe_q = get_single_rpe_config(**kwargs) if "q" in rpe_on else None
    config.rpe_k = get_single_rpe_config(**kwargs) if "k" in rpe_on else None
    config.rpe_v = get_single_rpe_config(**kwargs) if "v" in rpe_on else None
    return config


def build_rpe(config, head_dim, num_heads):
    if config is None:
        return None, None, None
    return [
        iRPE(
            head_dim=head_dim,
            num_heads=1 if rpe.shared_head else num_heads,
            mode=rpe.mode,
            method=rpe.method,
            transposed=t,
            num_buckets=rpe.num_buckets,
            rpe_config=rpe,
        )
        if rpe
        else None
        for rpe, t in zip([config.rpe_q, config.rpe_k, config.rpe_v], [True, True, False])
    ]


class RPEAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0, rpe_config=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rpe_q, self.rpe_k, self.rpe_v = build_rpe(rpe_config, head_dim=head_dim, num_heads=num_heads)

    def forward(self, x):
        B, C, h, w = x.shape
        x = x.view(B, C, h * w).transpose(1, 2)
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q *= self.scale
        attn = q @ k.transpose(-2, -1)
        if self.rpe_k is not None:
            attn += self.rpe_k(q, h, w)
        if self.rpe_q is not None:
            attn += self.rpe_q(k * self.scale).transpose(2, 3)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = attn @ v
        if self.rpe_v is not None:
            out += self.rpe_v(attn)
        x = out.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        x = x.transpose(1, 2).view(B, C, h, w)
        return x


class SRM(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.cfc1 = nn.Conv2d(channel, channel, kernel_size=(1, 2), bias=False)
        self.bn = nn.BatchNorm2d(channel)
        self.sigmoid = nn.Hardsigmoid()

    def forward(self, x):
        b, c, h, w = x.shape
        mean = x.reshape(b, c, -1).mean(-1).view(b, c, 1, 1)
        std = x.reshape(b, c, -1).std(-1).view(b, c, 1, 1)
        u = torch.cat([mean, std], dim=-1)
        z = self.cfc1(u)
        z = self.bn(z)
        g = self.sigmoid(z)
        return x * g.reshape(b, c, 1, 1).expand_as(x)


class Partial_conv3(nn.Module):
    def __init__(self, dim, n_div, forward_type, use_attn="", channel_type="", patnet_t0=False):
        super().__init__()
        self.dim_conv3 = dim // n_div
        self.dim = dim
        self.n_div = n_div
        self.dim_untouched = dim - self.dim_conv3
        self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)
        self.use_attn = use_attn
        self.channel_type = channel_type
        if use_attn:
            if channel_type == "self":
                rpe_config = get_rpe_config(ratio=20, method="euc", mode="bias", shared_head=False, skip=0, rpe_on="k")
                num_heads = 4 if patnet_t0 else 6
                self.attn = RPEAttention(
                    self.dim_untouched, num_heads=num_heads, attn_drop=0.1, proj_drop=0.1, rpe_config=rpe_config
                )
                self.norm = LayerNorm2d(self.dim_untouched)
                self.forward = self.forward_atten
            elif channel_type == "se":
                self.attn = SRM(self.dim_untouched)
                self.norm = nn.BatchNorm2d(self.dim_untouched)
                self.forward = self.forward_atten
        else:
            if forward_type == "slicing":
                self.forward = self.forward_slicing
            elif forward_type == "split_cat":
                self.forward = self.forward_split_cat

    def forward_atten(self, x: Tensor) -> Tensor:
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        if self.channel_type == "se":
            x2 = self.attn(x2)
            x2 = self.norm(x2)
        else:
            x2 = self.norm(x2)
            x2 = self.attn(x2)
        return torch.cat((x1, x2), 1)

    def forward_slicing(self, x: Tensor) -> Tensor:
        x1 = x.clone()
        x1[:, : self.dim_conv3, :, :] = self.partial_conv3(x1[:, : self.dim_conv3, :, :])
        return x1

    def forward_split_cat(self, x: Tensor) -> Tensor:
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        return torch.cat((x1, x2), 1)


class partial_spatial_attn_layer_reverse(nn.Module):
    def __init__(self, dim, n_head, partial=0.5):
        super().__init__()
        self.dim_conv = int(partial * dim)
        self.dim_untouched = dim - self.dim_conv
        self.conv = nn.Conv2d(self.dim_conv, self.dim_conv, 1, bias=False)
        self.conv_attn = nn.Conv2d(self.dim_untouched, n_head, 1, bias=False)
        self.norm = nn.BatchNorm2d(self.dim_untouched)
        self.norm2 = nn.BatchNorm2d(self.dim_conv)
        self.act = nn.Hardsigmoid()

    def forward(self, x):
        x1, x2 = torch.split(x, [self.dim_untouched, self.dim_conv], 1)
        weight = self.act(self.conv_attn(x1))
        x1 = self.norm(x1 * weight)
        x2 = self.conv(self.norm2(x2))
        return torch.cat((x1, x2), 1)


class MLPBlock(nn.Module):
    def __init__(
        self,
        dim,
        n_div,
        mlp_ratio,
        drop_path,
        layer_scale_init_value,
        act_layer,
        norm_layer,
        pconv_fw_type,
        use_channel,
        use_spatial,
        channel_type,
        patnet_t0,
    ):
        super().__init__()
        self.split_shortcut = True if channel_type == "self" else False
        self.spatial_mixing = Partial_conv3(dim, n_div, pconv_fw_type, use_channel, channel_type, patnet_t0)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        if use_spatial:
            mlp_layer = [
                nn.Conv2d(dim, mlp_hidden_dim, 1, bias=False),
                norm_layer(mlp_hidden_dim),
                act_layer(),
                nn.Conv2d(mlp_hidden_dim, dim, 1, bias=False),
                partial_spatial_attn_layer_reverse(dim, 1),
            ]
        else:
            mlp_layer = [
                nn.Conv2d(dim, mlp_hidden_dim, 1, bias=False),
                norm_layer(mlp_hidden_dim),
                act_layer(),
                nn.Conv2d(mlp_hidden_dim, dim, 1, bias=False),
            ]
        self.mlp = nn.Sequential(*mlp_layer)

        if layer_scale_init_value > 0:
            self.layer_scale = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            self.forward_impl = self.forward_layer_scale
        else:
            self.forward_impl = self.forward_base

    def forward_base(self, x: Tensor) -> Tensor:
        if self.split_shortcut:
            x = x + self.spatial_mixing(x)
            x = x + self.drop_path(self.mlp(x))
        else:
            shortcut = x
            x = self.spatial_mixing(x)
            x = shortcut + self.drop_path(self.mlp(x))
        return x

    def forward_layer_scale(self, x: Tensor) -> Tensor:
        if self.split_shortcut:
            x = x + self.spatial_mixing(x)
            x = x + self.drop_path(self.layer_scale.unsqueeze(-1).unsqueeze(-1) * self.mlp(x))
        else:
            shortcut = x
            x = self.spatial_mixing(x)
            x = shortcut + self.drop_path(self.layer_scale.unsqueeze(-1).unsqueeze(-1) * self.mlp(x))
        return x

    def forward(self, x):
        return self.forward_impl(x)


class Partial_PatchEmbed(nn.Module):
    """
    Patch Embedding Layer.
    YAML usage: [-1, 1, Partial_PatchEmbed, [96, 4, 4]]  # [c2, kernel_size, stride]
    """

    def __init__(self, c1, c2, patch_size=4, patch_stride=4, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.proj = nn.Conv2d(c1, c2, kernel_size=patch_size, stride=patch_stride, bias=False)
        self.norm = norm_layer(c2) if norm_layer else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        x = self.norm(self.proj(x))
        return x


class Partial_Block(nn.Module):
    """
    BasicStage Wrapper.
    YAML usage: [-1, 1, Partial_Block, [96, 2, 4, 2.0, 0.1, 'se']]
    Args: [c2, depth, n_div, mlp_ratio, drop_path, channel_type]
    Note: c2 must match c1 (input channels) for this block type.
    """

    def __init__(
        self,
        c1,
        c2,
        depth,
        n_div=4,
        mlp_ratio=2.0,
        drop_path_rate=0.0,
        channel_type="se",
        use_channel_attn=True,
        use_spatial_attn=True,
        patnet_t0=False,
    ):
        super().__init__()
        # 確保 c1 == c2，因為 BasicStage 不改變通道數
        # YOLO parser 會傳入 (c1, c2, ...)，這裡 c2 是 YAML 中指定的值
        assert c1 == c2, f"Input channel {c1} must equal output channel {c2} in Partial_Block"

        norm_layer = nn.BatchNorm2d
        act_layer = partial(nn.ReLU, inplace=True)

        # 處理 drop_path
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        blocks_list = [
            MLPBlock(
                dim=c1,
                n_div=n_div,
                mlp_ratio=mlp_ratio,
                drop_path=dpr[i],
                layer_scale_init_value=0,
                norm_layer=norm_layer,
                act_layer=act_layer,
                pconv_fw_type="split_cat",
                use_channel=use_channel_attn,
                use_spatial=use_spatial_attn,
                channel_type=channel_type,
                patnet_t0=patnet_t0,
            )
            for i in range(depth)
        ]
        self.blocks = nn.Sequential(*blocks_list)

    def forward(self, x: Tensor) -> Tensor:
        return self.blocks(x)


class Partial_Downsample(nn.Module):
    """
    PatchMerging Wrapper.
    YAML usage: [-1, 1, Partial_Downsample, [192, 2, 2]] # [c2, size, stride]
    """

    def __init__(self, c1, c2, patch_size=2, patch_stride=2, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.reduction = nn.Conv2d(c1, c2, kernel_size=patch_size, stride=patch_stride, bias=False)
        self.norm = norm_layer(c2) if norm_layer else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        return self.norm(self.reduction(x))
