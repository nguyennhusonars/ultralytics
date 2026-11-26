# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Ported from ultralytics_sonnn nn/modules/Conv2Former.py (model zoo; see ultralytics/nn/zoo/__init__.py)."""

import torch
import torch.nn.functional as F
from torch import nn

from ultralytics.nn.zoo._timm import DropPath, _cfg, register_model, trunc_normal_


class LayerNorm(nn.Module):
    r"""LayerNorm that supports two data formats: channels_last (default) or channels_first. The ordering of the
    dimensions in the inputs. channels_last corresponds to inputs with shape (batch_size, height, width, channels)
    while channels_first corresponds to inputs with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        # Ensure normalized_shape is an int for channels_first format
        if data_format == "channels_first":
            if isinstance(normalized_shape, (list, tuple)):
                normalized_shape = normalized_shape[0] if len(normalized_shape) > 0 else 0
            elif not isinstance(normalized_shape, int):
                raise TypeError(
                    f"normalized_shape must be int or tuple/list with one element for LayerNorm with data_format='channels_first', got {type(normalized_shape)}"
                )
        elif isinstance(normalized_shape, int):
            # For channels_last, nn.LayerNorm expects a shape tuple
            normalized_shape = (normalized_shape,)

        if isinstance(normalized_shape, int):  # Should be channels_first case now
            self.ln_weight = nn.Parameter(torch.ones(normalized_shape))
            self.ln_bias = nn.Parameter(torch.zeros(normalized_shape))
            self.normalized_shape_int = normalized_shape
        else:  # channels_last case
            self.ln_weight = nn.Parameter(torch.ones(normalized_shape))
            self.ln_bias = nn.Parameter(torch.zeros(normalized_shape))

        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = normalized_shape

    def forward(self, x):
        if self.data_format == "channels_last":
            # Use functional LayerNorm for channels_last
            return F.layer_norm(x, self.normalized_shape, self.ln_weight, self.ln_bias, self.eps)
        elif self.data_format == "channels_first":
            # Manual implementation for channels_first
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            # Reshape weight/bias for broadcasting: [1, C, 1, 1]
            x = self.ln_weight.view(1, -1, 1, 1) * x + self.ln_bias.view(1, -1, 1, 1)
            return x


class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4):
        super().__init__()
        # Use LayerNorm defined above, explicitly set data_format
        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_first")
        hidden_dim = int(dim * mlp_ratio)
        self.fc1 = nn.Conv2d(dim, hidden_dim, 1)
        # Positional encoding using depth-wise conv
        self.pos = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, groups=hidden_dim)
        self.fc2 = nn.Conv2d(hidden_dim, dim, 1)
        self.act = nn.GELU()

    def forward(self, x):
        _B, _C, _H, _W = x.shape
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        # Add positional encoding contribution
        x = x + self.act(self.pos(x))  # Apply activation to pos encoding as well
        x = self.fc2(x)
        return x


class SpatialAttention(nn.Module):
    def __init__(self, dim, kernel_size):
        super().__init__()
        # Use LayerNorm defined above, explicitly set data_format
        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_first")
        self.att = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.GELU(),
            # Depth-wise conv for spatial attention map
            nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim),
        )
        self.v = nn.Conv2d(dim, dim, 1)  # Value projection
        self.proj = nn.Conv2d(dim, dim, 1)  # Final projection

    def forward(self, x):
        _B, _C, _H, _W = x.shape
        x = self.norm(x)
        # Calculate attention map and multiply element-wise with value projection
        attn_map = self.att(x)
        value = self.v(x)
        x = attn_map * value
        x = self.proj(x)
        return x  # Return projected attention output


class Block(nn.Module):
    # Removed unused num_head, window_size from signature
    def __init__(self, index, dim, kernel_size, mlp_ratio=4.0, drop_path=0.0):
        super().__init__()
        self.attn = SpatialAttention(dim, kernel_size)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.mlp = MLP(dim, mlp_ratio)

        # Layer Scale (learnable scaling factor for residual branches)
        layer_scale_init_value = 1e-6
        self.layer_scale_1 = nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)

    def forward(self, x):
        # Apply Attention block with residual connection and LayerScale
        attn_out = self.attn(x)
        x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * attn_out)

        # Apply MLP block with residual connection and LayerScale
        mlp_out = self.mlp(x)
        x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * mlp_out)
        return x


class Conv2Former(nn.Module):
    # Architecture Zoo like MogaNet
    arch_zoo = {
        "n": {"dims": [64, 128, 256, 512], "depths": [2, 2, 8, 2], "mlp_ratios": [4, 4, 4, 4], "kernel_size": 7},
        "t": {"dims": [72, 144, 288, 576], "depths": [3, 3, 12, 3], "mlp_ratios": [4, 4, 4, 4], "kernel_size": 11},
        "s": {"dims": [72, 144, 288, 576], "depths": [4, 4, 32, 4], "mlp_ratios": [4, 4, 4, 4], "kernel_size": 11},
        "b": {"dims": [96, 192, 384, 768], "depths": [4, 4, 34, 4], "mlp_ratios": [4, 4, 4, 4], "kernel_size": 11},
        "b_22k": {
            "dims": [96, 192, 384, 768],
            "depths": [4, 4, 34, 4],
            "mlp_ratios": [4, 4, 4, 4],
            "kernel_size": 7,
        },  # Example variant
        "l": {"dims": [128, 256, 512, 1024], "depths": [4, 4, 48, 4], "mlp_ratios": [4, 4, 4, 4], "kernel_size": 11},
    }

    def __init__(
        self,
        arch="t",  # Architecture variant name from arch_zoo
        c1=3,  # Input channels (standardized name)
        c2=None,  # Output channels (placeholder, not used by backbone itself)
        num_classes=1000,  # For classification mode
        fork_feat=True,  # True: output list of features per stage; False: output classification logits
        drop_path_rate=0.1,  # Stochastic depth rate
        drop_rate=0.0,  # Dropout rate (used in head if classification) - Currently MLP has no dropout
        layer_scale_init_value=1e-6,  # Initial value for LayerScale
        head_init_scale=1.0,  # Scaling for classification head initialization
        # Removed direct params like dims, depths, kernel_size, mlp_ratios
        **kwargs,
    ):  # Allow extra args
        super().__init__()

        # --- Determine architecture settings ---
        if isinstance(arch, str):
            arch = arch.lower()
            if arch not in self.arch_zoo:
                raise KeyError(f'Architecture "{arch}" is not in arch_zoo: {list(self.arch_zoo.keys())}')
            arch_config = self.arch_zoo[arch]
        else:
            # Allow passing a custom config dict (optional)
            assert isinstance(arch, dict)
            essential_keys = {"dims", "depths", "mlp_ratios", "kernel_size"}
            assert essential_keys.issubset(arch.keys()), f"Custom arch dict must contain keys: {essential_keys}"
            arch_config = arch

        # --- Set parameters from config ---
        self.dims = arch_config["dims"]
        self.depths = arch_config["depths"]
        self.mlp_ratios = arch_config["mlp_ratios"]
        self.kernel_size = arch_config["kernel_size"]  # Kernel size for SpatialAttention
        self.num_stages = len(self.depths)

        self.num_classes = num_classes
        self.fork_feat = fork_feat
        self.in_chans = c1  # Use c1 for input channels

        # --- Set width_list required by some frameworks (e.g., Ultralytics) ---
        # width_list stores the output channels of each stage intended for feature usage
        if self.fork_feat:
            self.width_list = list(self.dims)  # Output channels of each stage match dims
        else:
            self.width_list = None  # Not needed or only final dim if not forking features

        # --- Build Stem and Downsampling Layers ---
        self.downsample_layers = nn.ModuleList()
        # Stem: Conv layers to reduce resolution and increase channels
        stem = nn.Sequential(
            nn.Conv2d(self.in_chans, self.dims[0] // 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GELU(),
            nn.BatchNorm2d(self.dims[0] // 2),  # Using BatchNorm in stem
            nn.Conv2d(self.dims[0] // 2, self.dims[0] // 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GELU(),
            nn.BatchNorm2d(self.dims[0] // 2),
            nn.Conv2d(self.dims[0] // 2, self.dims[0], kernel_size=2, stride=2, bias=False),
        )
        self.downsample_layers.append(stem)

        # Intermediate downsampling layers (between stages)
        for i in range(self.num_stages - 1):
            stride = 2
            downsample_layer = nn.Sequential(
                # Using LayerNorm before downsampling conv
                LayerNorm(self.dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(self.dims[i], self.dims[i + 1], kernel_size=stride, stride=stride),
            )
            self.downsample_layers.append(downsample_layer)

        # --- Build Stages (multiple Blocks) ---
        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))]
        cur_block_idx = 0
        for i in range(self.num_stages):
            stage_blocks = nn.Sequential(
                *[
                    Block(
                        index=cur_block_idx + j,
                        dim=self.dims[i],
                        kernel_size=self.kernel_size,  # Use the kernel size defined for the arch
                        mlp_ratio=self.mlp_ratios[i],
                        drop_path=dp_rates[cur_block_idx + j],
                    )
                    for j in range(self.depths[i])
                ]
            )
            self.stages.append(stage_blocks)
            cur_block_idx += self.depths[i]

        # --- Classifier Head (only used if fork_feat=False) ---
        if not self.fork_feat:
            # Original head structure for classification
            self.head_norm = LayerNorm(self.dims[-1], eps=1e-6, data_format="channels_first")
            # Simple global average pooling will be done in forward_features
            self.head_linear = nn.Linear(self.dims[-1], num_classes) if num_classes > 0 else nn.Identity()
        else:
            # No head needed when extracting features
            self.head_norm = nn.Identity()
            self.head_linear = nn.Identity()

        # --- Initialize Weights ---
        self.apply(self._init_weights)
        if isinstance(self.head_linear, nn.Linear):
            # Apply head initialization scaling if head exists
            trunc_normal_(self.head_linear.weight, std=0.02)
            if self.head_linear.bias is not None:
                nn.init.constant_(self.head_linear.bias, 0)
            self.head_linear.weight.data.mul_(head_init_scale)
            self.head_linear.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        """Initialize weights."""
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        # Check specifically for our LayerNorm implementation and standard Norm layers
        elif isinstance(m, (LayerNorm, nn.LayerNorm, nn.BatchNorm2d, nn.GroupNorm)):
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if hasattr(m, "weight") and m.weight is not None:  # BatchNorm uses 'weight'
                nn.init.constant_(m.weight, 1.0)
            elif hasattr(m, "ln_weight") and m.ln_weight is not None:  # Our LayerNorm uses 'ln_weight'
                nn.init.constant_(m.ln_weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        """Separates parameters for weight decay application."""
        nwd = set()
        for n, p in self.named_parameters():
            is_norm_param = False
            # Check if the parameter belongs to any norm layer based on module type
            module_name_parts = n.split(".")
            if len(module_name_parts) > 1:
                try:
                    module = self
                    for part in module_name_parts[:-1]:
                        module = getattr(module, part)
                    if isinstance(module, (LayerNorm, nn.LayerNorm, nn.BatchNorm2d, nn.GroupNorm)):
                        is_norm_param = True
                except AttributeError:
                    pass  # Parameter might be directly on self, not in a submodule

            # Add bias terms, norm layer parameters, and layer_scale parameters to no_weight_decay set
            if ".bias" in n or is_norm_param or "layer_scale" in n:
                # print(f"Excluding {n} from weight decay") # Debugging print
                nwd.add(n)

        # Verify exclusion (optional debug)
        # print(f"Total params: {len(list(self.named_parameters()))}, NWD params: {len(nwd)}")
        # for name in nwd:
        #     if 'weight' in name and not is_norm_param and 'layer_scale' not in name:
        #          print(f" Potential non-norm weight in NWD: {name}")

        return {name: p for name, p in self.named_parameters() if name in nwd}

    def forward_features(self, x):
        """Forward pass returning features based on fork_feat."""
        outs = []
        # Process through stem (downsample_layers[0]) first
        x = self.downsample_layers[0](x)

        # Iterate through the main stages
        for i in range(self.num_stages):
            # Apply stage blocks
            x = self.stages[i](x)

            # Store feature map if fork_feat is True
            if self.fork_feat:
                # Store output *after* the blocks of the current stage
                outs.append(x)

            # Apply downsampling for the next stage (if not the last stage)
            if i < self.num_stages - 1:
                x = self.downsample_layers[i + 1](x)

        # --- Output Handling ---
        if self.fork_feat:
            # Return list of features from each stage
            # Verify output length matches width_list length if width_list exists
            if self.width_list is not None and len(outs) != len(self.width_list):
                print(
                    f"Warning: Conv2Former forward_features output count ({len(outs)}) "
                    f"mismatches stored width_list length ({len(self.width_list)}). Check stage indexing."
                )
            return outs
        else:
            # For classification: apply final norm and global average pooling
            x = self.head_norm(x)
            return x.mean([-2, -1])  # Global average pooling: (N, C, H, W) -> (N, C)

    def forward(self, x):
        """Default forward pass."""
        features = self.forward_features(x)
        if self.fork_feat:
            # Return the list of feature maps directly
            return features
        else:
            # Apply the final linear layer for classification
            # 'features' is the pooled feature vector (N, C) in this case
            return self.head_linear(features)


@register_model
def conv2former_n(pretrained=False, **kwargs):
    model = Conv2Former(arch="n", **kwargs)
    model.default_cfg = _cfg(url="", **kwargs)  # Add checkpoint URL if available
    return model
