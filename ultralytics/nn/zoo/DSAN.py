# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Ported from ultralytics_sonnn nn/modules/DSAN.py (model zoo; see ultralytics/nn/zoo/__init__.py)."""

from functools import partial

from torch.nn.init import constant_, xavier_uniform_

from ultralytics.nn.zoo._timm import DropPath, _cfg, register_model, to_2tuple, trunc_normal_

try:
    import DSCN  # optional compiled CUDA op
except ImportError:
    DSCN = None
import math
import warnings

import torch
from torch import nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.cuda.amp import custom_bwd, custom_fwd

# 如果 DSCN 導入失敗，則定義一個虛擬的 Function 以避免錯誤
if DSCN is None:

    class DSCNFunction(Function):
        @staticmethod
        def forward(ctx, *args, **kwargs):
            raise NotImplementedError("DSCN CUDA op is not available or the DSCN package is not installed.")

        @staticmethod
        def backward(ctx, *args, **kwargs):
            raise NotImplementedError("DSCN CUDA op is not available or the DSCN package is not installed.")
else:

    class DSCNFunction(Function):
        @staticmethod
        @custom_fwd
        def forward(
            ctx,
            input,
            offset,
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            pad_h,
            pad_w,
            dilation_h,
            dilation_w,
            group,
            group_channels,
            offset_scale,
            im2col_step,
            remove_center,
            on_x,
        ):
            ctx.kernel_h = kernel_h
            ctx.kernel_w = kernel_w
            ctx.stride_h = stride_h
            ctx.stride_w = stride_w
            ctx.pad_h = pad_h
            ctx.pad_w = pad_w
            ctx.dilation_h = dilation_h
            ctx.dilation_w = dilation_w
            ctx.group = group
            ctx.group_channels = group_channels
            ctx.offset_scale = offset_scale
            ctx.im2col_step = im2col_step
            ctx.remove_center = remove_center
            ctx.on_x = on_x

            args = [
                input,
                offset,
                kernel_h,
                kernel_w,
                stride_h,
                stride_w,
                pad_h,
                pad_w,
                dilation_h,
                dilation_w,
                group,
                group_channels,
                offset_scale,
                ctx.im2col_step,
                ctx.remove_center,
                ctx.on_x,
            ]

            output = DSCN.dscn_forward(*args)
            ctx.save_for_backward(input, offset)

            return output

        @staticmethod
        @once_differentiable
        @custom_bwd
        def backward(ctx, grad_output):
            input, offset = ctx.saved_tensors

            args = [
                input,
                offset,
                ctx.kernel_h,
                ctx.kernel_w,
                ctx.stride_h,
                ctx.stride_w,
                ctx.pad_h,
                ctx.pad_w,
                ctx.dilation_h,
                ctx.dilation_w,
                ctx.group,
                ctx.group_channels,
                ctx.offset_scale,
                grad_output.contiguous(),
                ctx.im2col_step,
                ctx.remove_center,
                ctx.on_x,
            ]

            grad_input, grad_offset = DSCN.dscn_backward(*args)

            return (
                grad_input,
                grad_offset,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            )

        @staticmethod
        def symbolic(
            g,
            input,
            offset,
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            pad_h,
            pad_w,
            dilation_h,
            dilation_w,
            group,
            group_channels,
            offset_scale,
            im2col_step,
            remove_center,
        ):
            """Symbolic function for mmdeploy::DSCN.

            Returns:
                DSCN op for onnx.
            """
            return g.op(
                "mmdeploy::TRTDSCN",
                input,
                offset,
                kernel_h_i=int(kernel_h),
                kernel_w_i=int(kernel_w),
                stride_h_i=int(stride_h),
                stride_w_i=int(stride_w),
                pad_h_i=int(pad_h),
                pad_w_i=int(pad_w),
                dilation_h_i=int(dilation_h),
                dilation_w_i=int(dilation_w),
                group_i=int(group),
                group_channels_i=int(group_channels),
                offset_scale_f=float(offset_scale),
                im2col_step_i=int(im2col_step),
                remove_center=int(remove_center),
            )


class to_channels_first(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.permute(0, 3, 1, 2)


class to_channels_last(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.permute(0, 2, 3, 1)


def build_norm_layer(dim, norm_layer, in_format="channels_last", out_format="channels_last", eps=1e-6):
    layers = []
    if norm_layer == "BN":
        if in_format == "channels_last":
            layers.append(to_channels_first())
        layers.append(nn.BatchNorm2d(dim))
        if out_format == "channels_last":
            layers.append(to_channels_last())
    elif norm_layer == "LN":
        if in_format == "channels_first":
            layers.append(to_channels_last())
        layers.append(nn.LayerNorm(dim, eps=eps))
        if out_format == "channels_first":
            layers.append(to_channels_first())
    else:
        raise NotImplementedError(f"build_norm_layer does not support {norm_layer}")
    return nn.Sequential(*layers)


def build_act_layer(act_layer):
    if act_layer == "ReLU":
        return nn.ReLU(inplace=True)
    elif act_layer == "SiLU":
        return nn.SiLU(inplace=True)
    elif act_layer == "GELU":
        return nn.GELU()

    raise NotImplementedError(f"build_act_layer does not support {act_layer}")


def _is_power_of_2(n):
    if (not isinstance(n, int)) or (n < 0):
        raise ValueError(f"invalid input for _is_power_of_2: {n} (type: {type(n)})")

    return (n & (n - 1) == 0) and n != 0


class DSCNX(nn.Module):
    def __init__(
        self,
        channels=64,
        kernel_size=3,
        dw_kernel_size=None,
        stride=1,
        pad=1,
        dilation=1,
        group=4,
        offset_scale=1.0,
        act_layer="GELU",
        norm_layer="LN",
        remove_center=False,
    ):
        """DSCNX Module."""
        super().__init__()
        if channels % group != 0:
            raise ValueError(f"channels must be divisible by group, but got {channels} and {group}")
        _d_per_group = channels // group
        dw_kernel_size = dw_kernel_size if dw_kernel_size is not None else kernel_size
        if not _is_power_of_2(_d_per_group):
            warnings.warn(
                "You'd better set channels in DSCN to make the dimension of each attention head a power of 2 "
                "which is more efficient in this CUDA implementation."
            )

        self.offset_scale = offset_scale
        self.channels = channels
        self.kernel_size = kernel_size
        self.dw_kernel_size = dw_kernel_size
        self.stride = stride
        self.dilation = dilation
        self.pad = pad
        self.group = group
        self.group_channels = channels // group
        self.offset_scale = offset_scale
        self.remove_center = int(remove_center)

        if self.remove_center and self.kernel_size % 2 == 0:
            raise ValueError("remove_center is only compatible with odd kernel size.")

        self.dw_conv = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=(1, dw_kernel_size),
                stride=1,
                padding=(0, (dw_kernel_size - 1) // 2),
                groups=channels,
            ),
            build_norm_layer(channels, norm_layer, "channels_first", "channels_last"),
            build_act_layer(act_layer),
        )
        self.offset = nn.Linear(channels, group * (kernel_size - remove_center))
        self.input_proj = nn.Linear(channels, channels)
        self._reset_parameters()

    def _reset_parameters(self):
        constant_(self.offset.weight.data, 0.0)
        constant_(self.offset.bias.data, 0.0)
        xavier_uniform_(self.input_proj.weight.data)
        constant_(self.input_proj.bias.data, 0.0)

    def forward(self, input, off_x):
        """
        :param query (N, H, W, C)
        :param off_x (N, C, H, W)
        :return output (N, H, W, C)
        """
        x = self.input_proj(input)

        # === 修改開始 ===
        # 檢查輸入是否在 CUDA 上，並且 DSCN 模組是否可用
        # 這允許模型在 CPU 上進行初始化（例如YOLO框架計算stride），而在 GPU 上進行實際運算
        if input.is_cuda and DSCN is not None:
            x1 = self.dw_conv(off_x)
            offset = self.offset(x1)

            x = DSCNFunction.apply(
                x,
                offset,
                1,
                self.kernel_size,
                1,
                self.stride,
                0,
                self.pad,
                1,
                self.dilation,
                self.group,
                self.group_channels,
                self.offset_scale,
                256,
                self.remove_center,
                True,
            )
        # 如果在 CPU 上，則跳過自定義操作。這對於 stride=1 的情況是安全的，
        # 因為它不會改變特徵圖的形狀。
        # === 修改結束 ===
        return x


class DSCNY(nn.Module):
    def __init__(
        self,
        channels=64,
        kernel_size=3,
        dw_kernel_size=None,
        stride=1,
        pad=1,
        dilation=1,
        group=4,
        offset_scale=1.0,
        act_layer="GELU",
        norm_layer="LN",
        remove_center=False,
    ):
        """DSCNY Module."""
        super().__init__()
        if channels % group != 0:
            raise ValueError(f"channels must be divisible by group, but got {channels} and {group}")
        _d_per_group = channels // group
        dw_kernel_size = dw_kernel_size if dw_kernel_size is not None else kernel_size
        if not _is_power_of_2(_d_per_group):
            warnings.warn(
                "You'd better set channels in DSCN to make the dimension of each attention head a power of 2 "
                "which is more efficient in this CUDA implementation."
            )

        self.offset_scale = offset_scale
        self.channels = channels
        self.kernel_size = kernel_size
        self.dw_kernel_size = dw_kernel_size
        self.stride = stride
        self.dilation = dilation
        self.pad = pad
        self.group = group
        self.group_channels = channels // group
        self.offset_scale = offset_scale
        self.remove_center = int(remove_center)

        if self.remove_center and self.kernel_size % 2 == 0:
            raise ValueError("remove_center is only compatible with odd kernel size.")

        self.dw_conv = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=(dw_kernel_size, 1),
                stride=1,
                padding=((dw_kernel_size - 1) // 2, 0),
                groups=channels,
            ),
            build_norm_layer(channels, norm_layer, "channels_first", "channels_last"),
            build_act_layer(act_layer),
        )

        self.offset = nn.Linear(channels, group * (kernel_size - remove_center))
        self._reset_parameters()

    def _reset_parameters(self):
        constant_(self.offset.weight.data, 0.0)
        constant_(self.offset.bias.data, 0.0)

    def forward(self, input, off_x):
        """
        :param query (N, H, W, C)
        :param off_x (N, C, H, W)
        :return output (N, H, W, C)
        """
        x = input

        # === 修改開始 ===
        # 檢查輸入是否在 CUDA 上，並且 DSCN 模組是否可用
        # 這允許模型在 CPU 上進行初始化（例如YOLO框架計算stride），而在 GPU 上進行實際運算
        if input.is_cuda and DSCN is not None:
            x1 = self.dw_conv(off_x)
            offset = self.offset(x1)

            x = DSCNFunction.apply(
                x,
                offset,
                self.kernel_size,
                1,
                self.stride,
                1,
                self.pad,
                0,
                self.dilation,
                1,
                self.group,
                self.group_channels,
                self.offset_scale,
                256,
                self.remove_center,
                False,
            )
        # 如果在 CPU 上，則跳過自定義操作。這對於 stride=1 的情況是安全的，
        # 因為它不會改變特徵圖的形狀。
        # === 修改結束 ===

        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DSCNPair(nn.Module):
    def __init__(self, d_model, kernel_size, dw_kernel_size, pad, stride, dilation, group):
        super().__init__()
        self.kernel_size = kernel_size
        self.dw_kernel_size = dw_kernel_size
        self.pad = pad
        self.stride = stride
        self.dilation = dilation
        self.group = group
        self.conv0 = nn.Conv2d(d_model, d_model, kernel_size=5, padding=2, groups=d_model)

        self.dscn_x = DSCNX(
            d_model, kernel_size, dw_kernel_size, stride=stride, pad=pad, dilation=dilation, group=group
        )
        self.dscn_y = DSCNY(
            d_model, kernel_size, dw_kernel_size, stride=stride, pad=pad, dilation=dilation, group=group
        )
        self.conv = nn.Conv2d(d_model, d_model, 1)

    def forward(self, x):
        u = x.clone()
        x = self.conv0(x)
        attn = x.permute(0, 2, 3, 1)  # N, C, H, W -> N, H, W, C
        attn = self.dscn_x(attn, x)
        attn = self.dscn_y(attn, x)
        attn = attn.permute(0, 3, 1, 2)  # N, H, W, C -> N, C, H, W
        attn = self.conv(attn)
        return u * attn


class DSA(nn.Module):
    def __init__(self, d_model, kernel_size, dw_kernel_size, pad, stride, dilation, group):
        super().__init__()

        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = DSCNPair(d_model, kernel_size, dw_kernel_size, pad, stride, dilation, group)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)

    def forward(self, x):
        shortcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shortcut
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim,
        kernel_size,
        dw_kernel_size,
        pad,
        stride,
        dilation,
        group,
        mlp_ratio=4.0,
        drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
    ):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(dim)
        self.attn = DSA(dim, kernel_size, dw_kernel_size, pad, stride, dilation, group)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = nn.BatchNorm2d(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        layer_scale_init_value = 1e-2
        self.layer_scale_1 = nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(self.norm2(x)))
        return x


class OverlapPatchEmbed(nn.Module):
    """Image to Patch Embedding."""

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=stride, padding=(patch_size[0] // 2, patch_size[1] // 2)
        )
        self.norm = nn.BatchNorm2d(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = self.norm(x)
        return x, H, W


class DSAN(nn.Module):
    def __init__(
        self,
        img_size=224,
        in_chans=3,
        num_classes=1000,
        embed_dims=None,
        kernel_sizes=None,
        dw_kernel_sizes=None,
        pads=None,
        strides=None,
        dilations=None,
        groups=None,
        mlp_ratios=None,
        drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        depths=None,
        num_stages=4,
        flag=False,
    ):
        if depths is None:
            depths = [3, 4, 6, 3]
        if mlp_ratios is None:
            mlp_ratios = [4, 4, 4, 4]
        if groups is None:
            groups = [1, 4, 8, 8]
        if dilations is None:
            dilations = [1, 1, 1, 1]
        if strides is None:
            strides = [1, 1, 1, 1]
        if pads is None:
            pads = [5, 5, 3, 2]
        if dw_kernel_sizes is None:
            dw_kernel_sizes = [5, 5, 5, 5]
        if kernel_sizes is None:
            kernel_sizes = [11, 11, 7, 5]
        if embed_dims is None:
            embed_dims = [64, 128, 256, 512]
        super().__init__()
        if flag == False:
            self.num_classes = num_classes
        self.depths = depths
        self.num_stages = num_stages

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0

        for i in range(num_stages):
            patch_embed = OverlapPatchEmbed(
                img_size=img_size if i == 0 else img_size // (2 ** (i + 1)),
                patch_size=7 if i == 0 else 3,
                stride=4 if i == 0 else 2,
                in_chans=in_chans if i == 0 else embed_dims[i - 1],
                embed_dim=embed_dims[i],
            )

            block = nn.ModuleList(
                [
                    Block(
                        dim=embed_dims[i],
                        kernel_size=kernel_sizes[i],
                        dw_kernel_size=dw_kernel_sizes[i],
                        pad=pads[i],
                        stride=strides[i],
                        dilation=dilations[i],
                        group=groups[i],
                        mlp_ratio=mlp_ratios[i],
                        drop=drop_rate,
                        drop_path=dpr[cur + j],
                    )
                    for j in range(depths[i])
                ]
            )
            norm = norm_layer(embed_dims[i])
            cur += depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)

        # classification head
        self.head = nn.Linear(embed_dims[3], num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

        # --- width_list 計算 ---
        # 由於 forward 函數已被修改為兼容 CPU，此處的虛擬運算現在更加健壯。
        self.width_list = []
        if torch.cuda.is_available() and DSCN is not None:
            try:
                device = torch.device("cuda")
                original_device = next(self.parameters()).device

                self.to(device)
                self.eval()

                dummy_input = torch.randn(1, in_chans, img_size, img_size, device=device)

                with torch.no_grad():
                    features = self.forward(dummy_input)

                self.width_list = [f.size(1) for f in features]

                self.to(original_device)
                self.train()
                torch.cuda.empty_cache()

            except Exception as e:
                print(f"Error during dummy forward pass for width_list calculation: {e}")
                print("Setting width_list to embed_dims as fallback.")
                self.width_list = embed_dims
                self.train()
        else:
            print(
                "Warning: CUDA not available or DSCN module not found. "
                "Falling back to `embed_dims` for `width_list`. This may be inaccurate."
            )
            self.width_list = embed_dims

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def freeze_patch_emb(self):
        self.patch_embed1.requires_grad = False

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed1", "pos_embed2", "pos_embed3", "pos_embed4", "cls_token"}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=""):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features_for_classification(self, x):
        """用於分類任務的原始 forward_features 邏輯。."""
        B = x.shape[0]
        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            x, H, W = patch_embed(x)
            for blk in block:
                x = blk(x)
            x = x.flatten(2).transpose(1, 2)
            x = norm(x)
            if i != self.num_stages - 1:
                x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        return x.mean(dim=1)

    def forward_head(self, x):
        """分類頭的前向傳播。."""
        x = self.forward_features_for_classification(x)
        x = self.head(x)
        return x

    def forward(self, x):
        """修改後的前向傳播，返回每個階段的特徵圖列表。 這適用於作為檢測/分割模型的骨幹網路。.
        """
        B = x.shape[0]
        feature_outputs = []
        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            x, H, W = patch_embed(x)
            for blk in block:
                x = blk(x)

            x_norm_in = x.flatten(2).transpose(1, 2)
            x_norm_out = norm(x_norm_in)

            x = x_norm_out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            feature_outputs.append(x)

        return feature_outputs


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        x = self.dwconv(x)
        return x


@register_model
def dsan_b(pretrained=False, **kwargs):
    model = DSAN(
        embed_dims=[64, 128, 320, 512],
        mlp_ratios=[8, 8, 4, 4],
        kernel_sizes=[15, 13, 7, 5],
        dw_kernel_sizes=[9, 7, 5, 5],
        pads=[7, 6, 3, 2],
        groups=[4, 8, 16, 16],
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        depths=[2, 2, 5, 3],
        **kwargs,
    )
    model.default_cfg = _cfg()
    if pretrained:
        print("Warning: No pretrained weights available for dsan_b.")
    return model
