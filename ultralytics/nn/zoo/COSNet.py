# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Ported from ultralytics_sonnn nn/modules/COSNet.py (model zoo; see ultralytics/nn/zoo/__init__.py)."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from ultralytics.nn.zoo._timm import DropPath, trunc_normal_

COSNET_SPECS: dict[str, dict[str, Any]] = {
    "cosnet_tiny": {  # Example: A smaller COSNet
        "depths": [2, 2, 6, 2],  # Shallower than original
        "base_dim": 64,  # Smaller base dimension
        "expan_ratio": 4,
        "s_kernel_sizes": [5, 5, 3, 3],  # Kernel sizes per stage for MCFS
        "drop_path_rate": 0.1,
        "layer_scale_init_value": 1e-6,  # Kept from original FSB
    },
    "cosnet_small": {  # This matches the original parameters more closely
        "depths": [3, 3, 12, 3],
        "base_dim": 72,  # Original 'dim'
        "expan_ratio": 4,
        "s_kernel_sizes": [5, 5, 3, 3],
        "drop_path_rate": 0.2,
        "layer_scale_init_value": 1e-6,
    },
    "cosnet_base": {  # Example: A larger COSNet
        "depths": [3, 3, 18, 3],
        "base_dim": 96,
        "expan_ratio": 4,
        "s_kernel_sizes": [5, 5, 3, 3],
        "drop_path_rate": 0.3,
        "layer_scale_init_value": 1e-6,
    },
}


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight.view(-1, 1, 1) * x + self.bias.view(-1, 1, 1)  # Ensure weight and bias are broadcastable
            return x


class MCFS(nn.Module):  # Multi-scale Contextual Feature Sharpening
    def __init__(self, dim: int, s_kernel_size: int = 3):
        super().__init__()

        self.proj_1 = nn.Conv2d(dim, dim, kernel_size=1, padding=0)
        self.proj_2 = nn.Conv2d(dim * 2, dim, kernel_size=1, padding=0)
        self.norm_proj = LayerNorm(dim, eps=1e-6, data_format="channels_first")

        # multiscale spatial context layers
        s_groups = dim // 4 if dim >= 4 else 1  # Ensure groups >= 1
        self.s_ctx_1 = nn.Conv2d(dim, dim, kernel_size=s_kernel_size, padding=s_kernel_size // 2, groups=s_groups)
        self.s_ctx_2 = nn.Conv2d(
            dim, dim, kernel_size=s_kernel_size, dilation=2, padding=(s_kernel_size // 2) * 2, groups=s_groups
        )
        self.norm_s = LayerNorm(dim, eps=1e-6, data_format="channels_first")

        # sharpening module layers
        self.h_ctx = nn.Conv2d(dim, dim, kernel_size=5, padding=2, bias=False, groups=dim)
        self.norm_h = LayerNorm(dim, eps=1e-6, data_format="channels_first")

        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_proj = self.norm_proj(self.act(self.proj_1(x)))  # Renamed to avoid confusion with input 'x'

        # extract multi-scale contextual features
        sx1 = self.act(self.s_ctx_1(x_proj))
        sx2 = self.act(self.s_ctx_2(x_proj))
        sx = self.norm_s(sx1 + sx2)

        # feature enhancement using learnable sharpening factors
        hx = self.act(self.h_ctx(x_proj))
        hx_t_mean = hx.mean(dim=1, keepdim=True)  # Mean over channels for sharpening factor base
        hx_t = x_proj - hx_t_mean  # Difference from mean

        # Softmax over spatial dimensions, then expand to C channels for element-wise mul
        # The original implementation averaged over H, W for softmax, then unsqueezed.
        # Let's refine the softmax application for sharpening factor.
        # Original: hx_t = torch.softmax(hx.mean(dim=[-2,-1]).unsqueeze(-1).unsqueeze(-1), dim=1) * hx_t
        # This applies softmax over channel dimension of spatially-averaged hx, which makes sense as channel-wise attention.
        sharpening_factors = torch.softmax(hx.mean(dim=[-2, -1], keepdim=True), dim=1)
        hx_t_sharpened = sharpening_factors * hx_t
        hx = self.norm_h(hx + hx_t_sharpened)  # Original was hx + hx_t, now hx + hx_t_sharpened

        # combine the multiscale contextual features with the sharpened features
        out = self.act(self.proj_2(torch.cat([sx, hx], dim=1)))
        return out


class MLP(nn.Module):
    def __init__(self, dim: int, mlp_ratio: int = 4):
        super().__init__()
        hidden_dim = dim * mlp_ratio
        self.fc_1 = nn.Conv2d(dim, hidden_dim, kernel_size=1)
        self.pos = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim)  # Depthwise conv
        self.fc_2 = nn.Conv2d(hidden_dim, dim, kernel_size=1)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc_1(x)
        x = self.act(x)
        x = x + self.act(self.pos(x))  # Skip connection within MLP
        x = self.fc_2(x)
        return x


class FSB(nn.Module):  # Feature Sharpening Block
    def __init__(
        self,
        dim: int,
        s_kernel_size: int = 3,
        drop_path: float = 0.1,
        layer_scale_init_value: float = 1e-6,
        expan_ratio: int = 4,
    ):  # layer_scale_init_value not used in this version
        super().__init__()

        self.conv_dw = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.norm_dw = LayerNorm(dim, eps=1e-6, data_format="channels_first")

        self.layer_norm_1 = LayerNorm(dim, eps=1e-6, data_format="channels_first")
        self.layer_norm_2 = LayerNorm(dim, eps=1e-6, data_format="channels_first")

        self.mlp = MLP(dim=dim, mlp_ratio=expan_ratio)
        self.attn = MCFS(dim, s_kernel_size=s_kernel_size)

        self.drop_path_1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.drop_path_2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.act = nn.GELU()

        # Note: layer_scale_init_value was in the original __init__ but not used.
        # If layer scaling is desired, it would typically involve learnable parameters like:
        # self.gamma1 = nn.Parameter(layer_scale_init_value * torch.ones(dim, 1, 1), requires_grad=True)
        # self.gamma2 = nn.Parameter(layer_scale_init_value * torch.ones(dim, 1, 1), requires_grad=True)
        # And then used as: x = x_copy + self.drop_path_1(self.gamma1 * self.attn(normed_x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # First residual connection (convolutional part)
        x = x + self.norm_dw(self.act(self.conv_dw(x)))  # Original had norm after act

        # Second residual connection (attention part)
        x_copy_1 = x
        x_norm_1 = self.layer_norm_1(x_copy_1)
        x_attn = self.attn(x_norm_1)
        out_1 = x_copy_1 + self.drop_path_1(x_attn)

        # Third residual connection (MLP part)
        x_copy_2 = out_1
        x_norm_2 = self.layer_norm_2(x_copy_2)
        x_mlp = self.mlp(x_norm_2)
        out_2 = x_copy_2 + self.drop_path_2(x_mlp)

        return out_2


class COSNet(nn.Module):
    def __init__(
        self,
        model_name: str,
        in_chans: int = 3,
        input_size: int | tuple[int, int] = 224,
        num_classes: int = 1000,  # Kept for compatibility, but not used if only backbone
        head_init_scale: float = 1.0,  # Kept for compatibility
        **kwargs,
    ):  # To catch any other unexpected args
        super().__init__()

        if model_name not in COSNET_SPECS:
            raise ValueError(f"Unknown model_name: {model_name}. Available: {list(COSNET_SPECS.keys())}")

        spec = COSNET_SPECS[model_name]
        depths: list[int] = spec["depths"]
        base_dim: int = spec["base_dim"]
        expan_ratio: int = spec["expan_ratio"]
        s_kernel_sizes: list[int] = spec["s_kernel_sizes"]
        drop_path_rate: float = spec["drop_path_rate"]
        layer_scale_init_value: float = spec["layer_scale_init_value"]

        self.model_name = model_name
        self.num_stages = len(depths)
        self.in_chans = in_chans
        if isinstance(input_size, int):
            self.input_h_w: tuple[int, int] = (input_size, input_size)
        elif isinstance(input_size, tuple) and len(input_size) == 2:
            self.input_h_w = input_size
        else:
            raise ValueError(f"input_size must be int or tuple of 2 ints, got {input_size}")

        self.dims: list[int] = []
        for i in range(self.num_stages):
            self.dims.append(base_dim * (2**i))

        self.downsample_layers = nn.ModuleList()
        # Stem layer
        stem = nn.Conv2d(in_chans, self.dims[0], kernel_size=5, stride=4, padding=2)
        self.downsample_layers.append(stem)

        # Intermediate downsampling layers
        for i in range(self.num_stages - 1):  # 3 intermediate downsamplers for 4 stages
            downsample_layer = nn.Conv2d(self.dims[i], self.dims[i + 1], kernel_size=3, stride=2, padding=1)
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur_dp_idx = 0
        for i in range(self.num_stages):
            stage_blocks = []
            for j in range(depths[i]):
                stage_blocks.append(
                    FSB(
                        dim=self.dims[i],
                        s_kernel_size=s_kernel_sizes[i],
                        drop_path=dp_rates[cur_dp_idx + j],
                        layer_scale_init_value=layer_scale_init_value,
                        expan_ratio=expan_ratio,
                    )
                )
            self.stages.append(nn.Sequential(*stage_blocks))
            cur_dp_idx += depths[i]

        # Classification head (commented out, backbone returns features)
        # self.norm_head = LayerNorm(self.dims[-1], eps=1e-6, data_format="channels_first") # For global avg pool
        # self.head = nn.Linear(self.dims[-1], num_classes)
        # if head_init_scale != 1.0:
        #     self.head.weight.data.mul_(head_init_scale)
        #     self.head.bias.data.mul_(head_init_scale)

        self.apply(self._init_weights)

        # Calculate width_list (output channels of each feature map)
        self.width_list: list[int] = []
        # Temporarily set to eval mode for dummy pass, affects dropout, batchnorm
        # Store original mode
        original_mode_is_training = self.training
        self.eval()
        try:
            # Create a dummy input with the specified size
            dummy_input = torch.randn(1, self.in_chans, *self.input_h_w)
            features = self.forward_features(dummy_input)
            self.width_list = [f.size(1) for f in features]
        except Exception as e:
            print(f"Warning: Error during dummy forward pass for COSNet width_list: {e}.")
            # Fallback: use dims directly, assuming forward_features returns one feature per stage
            self.width_list = list(self.dims)
            print(f"Falling back to width_list based on self.dims: {self.width_list}")
        # Restore original mode
        if original_mode_is_training:
            self.train()

    def _init_weights(self, m: nn.Module):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (LayerNorm, nn.LayerNorm)):  # Handles custom LayerNorm and nn.LayerNorm
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if hasattr(m, "weight") and m.weight is not None:
                nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)

    def init_weights(self, pretrained: str | None = None):
        """Load pretrained weights. Handles shape mismatches in convolutional layers by bilinear interpolation.
        """
        if pretrained is not None:
            try:
                cur_state_dict = self.state_dict()
                checkpoint = torch.load(pretrained, map_location="cpu")

                # Determine the key for the state dictionary
                if "state_dict" in checkpoint:
                    pretrained_state_dict = checkpoint["state_dict"]
                elif "model" in checkpoint:
                    pretrained_state_dict = checkpoint["model"]
                else:
                    pretrained_state_dict = checkpoint

                # Adapt keys (e.g., remove 'module.' prefix from DataParallel)
                # and handle shape mismatches
                new_state_dict = {}
                for k_pretrained, v_pretrained in pretrained_state_dict.items():
                    k_current = k_pretrained.replace("module.", "")  # Remove 'module.' prefix if present

                    if k_current in cur_state_dict:
                        if v_pretrained.shape != cur_state_dict[k_current].shape:
                            # Try to interpolate for convolutional weights if dimensions mismatch (typically H, W)
                            # This is a common case for transferring weights between different input resolutions.
                            # Check if it's a Conv2d weight and has 4 dimensions (O, I, H, W)
                            # And the mismatch is in spatial dimensions (last 2)
                            current_v = cur_state_dict[k_current]
                            if (
                                len(v_pretrained.shape) == 4
                                and len(current_v.shape) == 4
                                and v_pretrained.shape[:2] == current_v.shape[:2]
                                and v_pretrained.shape[2:] != current_v.shape[2:]
                            ):
                                print(f"Interpolating {k_current} from {v_pretrained.shape} to {current_v.shape}")
                                v_pretrained_interpolated = F.interpolate(
                                    v_pretrained, size=current_v.shape[2:], mode="bilinear", align_corners=False
                                )  # Usually False for features
                                new_state_dict[k_current] = v_pretrained_interpolated
                            else:
                                print(
                                    f"Skipping {k_current} due to shape mismatch: "
                                    f"pretrained {v_pretrained.shape} vs current {current_v.shape}"
                                )
                                # If not interpolatable or other dims mismatch, skip or handle as error
                        else:
                            new_state_dict[k_current] = v_pretrained
                    # else: # Key from pretrained not in current model (e.g. classifier head)
                    #     print(f"Skipping {k_pretrained} (as {k_current}) as it's not in the current model.")

                msg = self.load_state_dict(new_state_dict, strict=False)
                print(f"Pretrained weights loaded from {pretrained}. Load message: {msg}")

            except Exception as e:
                print(f"Error loading pretrained weights from {pretrained}: {e}")
        else:
            # Weights are already initialized by self.apply(self._init_weights) in __init__
            print("No pretrained weights provided, using random initialization from _init_weights.")

    def forward_features(self, x: torch.Tensor) -> list[torch.Tensor]:
        features: list[torch.Tensor] = []
        # Stem
        x = self.downsample_layers[0](x)
        x = self.stages[0](x)
        features.append(x)

        # Subsequent stages
        for i in range(1, self.num_stages):
            x = self.downsample_layers[i](x)  # Downsample
            x = self.stages[i](x)  # Pass through FSB blocks
            features.append(x)
        return features

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        # This model is used as a backbone, so it returns a list of feature maps.
        return self.forward_features(x)


def cosnet_tiny(
    input_size: int | tuple[int, int] = 224, in_chans: int = 3, pretrained: str | None = None, **kwargs
) -> COSNet:
    model = COSNet(model_name="cosnet_tiny", in_chans=in_chans, input_size=input_size, **kwargs)
    if pretrained:
        model.init_weights(pretrained)
    return model
