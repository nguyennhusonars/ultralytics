# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Ported from ultralytics_sonnn nn/modules/HorNet.py (model zoo; see ultralytics/nn/zoo/__init__.py)."""

from functools import partial
from ultralytics.nn.zoo._timm import DropPath
from ultralytics.nn.zoo._timm import register_model
from ultralytics.nn.zoo._timm import trunc_normal_
import torch.fft
import torch.nn as nn
import torch.nn.functional as F


def get_dwconv(dim, kernel, bias):
    return nn.Conv2d(dim, dim, kernel_size=kernel, padding=(kernel - 1) // 2, bias=bias, groups=dim)


class LayerNorm(nn.Module):
    r"""LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        # Ensure normalized_shape is correctly handled for layer_norm
        if isinstance(normalized_shape, int):
            self.normalized_shape = (normalized_shape,)
        else:
            self.normalized_shape = tuple(normalized_shape)

    def forward(self, x):
        if self.data_format == "channels_last":
            # Ensure weight and bias match the last dimension
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            # Adjust weight and bias shape for broadcasting
            channels = self.normalized_shape[0]  # Get the channel dimension
            weight = self.weight.view(1, channels, 1, 1)  # Add batch dim for broadcasting
            bias = self.bias.view(1, channels, 1, 1)  # Add batch dim for broadcasting
            x = weight * x + bias
            return x


class gnconv(nn.Module):
    def __init__(self, dim, order=5, gflayer=None, h=14, w=8, s=1.0):
        super().__init__()
        self.order = order
        # Calculation: D_i = dim // 2**(order-1-i) for i=0..order-1? No, stick to code's version.
        # Code version: [dim/16, dim/8, dim/4, dim/2, dim] for order=5
        self.dims = [dim // 2**i for i in range(order)]
        self.dims.reverse()  # [dim/16, dim/8, ..., dim]
        self.proj_in = nn.Conv2d(dim, 2 * dim, 1)  # Projects input C -> 2C

        # The DWConv/GFLayer processes the 'abc' branch which has sum(self.dims) channels
        dw_dim = sum(self.dims)
        if gflayer is None:
            # print(f"[gnconv] Using default dwconv dim={dw_dim}")
            self.dwconv = get_dwconv(dw_dim, 7, True)
        else:
            # print(f"[gnconv] Using gflayer dim={dw_dim}, h={h}, w={w}")
            self.dwconv = gflayer(dw_dim, h=h, w=w)  # Pass h, w here for GFLayer init

        # proj_out takes the output of the iterative process (which has self.dims[-1] = dim channels)
        # and projects it dim -> dim (effectively a final 1x1 conv)
        self.proj_out = nn.Conv2d(self.dims[-1], dim, 1)

        # Pointwise convolutions to change dimensions in the iterative loop
        self.pws = nn.ModuleList([nn.Conv2d(self.dims[i], self.dims[i + 1], 1) for i in range(order - 1)])

        self.scale = s
        # print(f'[gnconv] order={order} input_dim={dim} dims={self.dims} dw_dim={dw_dim} scale={self.scale:.4f} gflayer={"Yes" if gflayer else "No"} h={h} w={w}')

    def forward(self, x, mask=None, dummy=False):
        B, C, H, W = x.shape  # Input C = dim
        # print(f"[gnconv forward] Input shape: {x.shape}, order: {self.order}, self.dims: {self.dims}")

        fused_x = self.proj_in(x)  # B, 2*dim, H, W
        # print(f"[gnconv forward] fused_x shape: {fused_x.shape}")

        # --- CORRECTED SPLIT ---
        # Split fused_x (2*dim channels) into pwa (dims[0] channels) and abc (sum(dims) channels)
        # Check: dims[0] + sum(dims) should equal 2*dim. Verified this holds.
        try:
            split_sizes = (self.dims[0], sum(self.dims))
            if sum(split_sizes) != fused_x.shape[1]:
                raise ValueError(
                    f"Calculated split sizes {split_sizes} sum to {sum(split_sizes)}, but expected {fused_x.shape[1]}"
                )
            pwa, abc = torch.split(fused_x, split_sizes, dim=1)
        except (RuntimeError, ValueError) as e:
            print(f"Error during torch.split in gnconv:")
            print(f"  Input tensor shape (fused_x): {fused_x.shape}")
            print(f"  Target dim size: {fused_x.shape[1]}")
            print(f"  Calculated self.dims: {self.dims}")
            print(f"  Attempted split_sizes: ({self.dims[0]}, {sum(self.dims)})")
            print(f"  Sum of attempted split_sizes: {self.dims[0] + sum(self.dims)}")
            raise e
        # print(f"[gnconv forward] pwa shape: {pwa.shape}, abc shape: {abc.shape}")
        # --- END CORRECTION ---

        # Apply dwconv (either standard DWConv or GlobalLocalFilter) to abc
        # Input to dwconv has sum(dims) channels, which matches dw_dim used in init.
        dw_abc = self.dwconv(abc)
        # print(f"[gnconv forward] dw_abc shape after dwconv: {dw_abc.shape}")

        # Apply scale factor
        dw_abc = dw_abc * self.scale

        # Split the output of dwconv according to self.dims
        # Input to split (dw_abc) has sum(dims) channels.
        # self.dims is the list of sizes, which sums to sum(dims). Correct.
        try:
            if sum(self.dims) != dw_abc.shape[1]:
                raise ValueError(
                    f"dw_abc channel size {dw_abc.shape[1]} does not match sum of self.dims {sum(self.dims)}"
                )
            dw_list = torch.split(dw_abc, self.dims, dim=1)
        except (RuntimeError, ValueError) as e:
            print(f"Error during torch.split (dw_list):")
            print(f"  Input tensor shape (dw_abc): {dw_abc.shape}")
            print(f"  Target dim size: {dw_abc.shape[1]}")
            print(f"  Split sections (self.dims): {self.dims}")
            print(f"  Sum of split sections: {sum(self.dims)}")
            raise e
        # print(f"[gnconv forward] dw_list lengths: {[dw.shape for dw in dw_list]}")

        # Initial multiplication: pwa (dims[0]) * dw_list[0] (dims[0])
        if pwa.shape[1] != dw_list[0].shape[1]:
            raise ValueError(
                f"Shape mismatch for initial multiplication: pwa ({pwa.shape[1]}) vs dw_list[0] ({dw_list[0].shape[1]})"
            )
        x = pwa * dw_list[0]
        # print(f"[gnconv forward] x shape after init mult: {x.shape}")

        # Iterative process
        for i in range(self.order - 1):
            # pws[i] projects dims[i] -> dims[i+1]
            x_proj = self.pws[i](x)
            # dw_list[i+1] has dims[i+1] channels
            if x_proj.shape[1] != dw_list[i + 1].shape[1]:
                raise ValueError(
                    f"Shape mismatch in loop {i}: projected x ({x_proj.shape[1]}) vs dw_list[{i+1}] ({dw_list[i+1].shape[1]})"
                )
            x = x_proj * dw_list[i + 1]
            # print(f"[gnconv forward] x shape after loop {i}: {x.shape}")

        # Final projection: proj_out maps dims[-1] -> dim (where dims[-1] == dim)
        x = self.proj_out(x)
        # print(f"[gnconv forward] final x shape: {x.shape}")

        return x


class Block(nn.Module):
    r"""HorNet block"""

    def __init__(self, dim, drop_path=0.0, layer_scale_init_value=1e-6, gnconv=gnconv):
        super().__init__()

        self.norm1 = LayerNorm(dim, eps=1e-6, data_format="channels_first")  # Input to gnconv needs channels_first norm
        # Pass the 'dim' for the current stage to the partial gnconv function
        self.gnconv = gnconv(dim=dim)
        self.norm2 = LayerNorm(dim, eps=1e-6, data_format="channels_last")  # Input to MLP needs channels_last norm
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)

        self.gamma1 = (
            nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )

        self.gamma2 = (
            nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        # Input x: (B, C, H, W)
        shortcut = x
        x_norm1 = self.norm1(x)  # Apply norm1 (channels_first)
        x_gnconv = self.gnconv(x_norm1)  # Apply gnconv

        if self.gamma1 is not None:
            # Reshape gamma1 for broadcasting: (1, C, 1, 1)
            x_gnconv = self.gamma1.view(1, -1, 1, 1) * x_gnconv

        x = shortcut + self.drop_path(x_gnconv)  # First skip connection

        # FFN part
        shortcut2 = x
        x_permuted = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x_norm2 = self.norm2(x_permuted)  # Apply norm2 (channels_last)
        x_mlp = self.pwconv1(x_norm2)
        x_mlp = self.act(x_mlp)
        x_mlp = self.pwconv2(x_mlp)
        x_mlp_permuted = x_mlp.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        if self.gamma2 is not None:
            # Reshape gamma2 for broadcasting: (1, C, 1, 1)
            x_mlp_permuted = self.gamma2.view(1, -1, 1, 1) * x_mlp_permuted

        x = shortcut2 + self.drop_path(x_mlp_permuted)  # Second skip connection
        return x


class HorNet(nn.Module):
    def __init__(
        self,
        in_chans=3,
        num_classes=1000,  # num_classes kept but head removed
        depths=[3, 3, 9, 3],
        base_dim=96,
        drop_path_rate=0.0,
        layer_scale_init_value=1e-6,
        head_init_scale=1.0,  # head_init_scale unused
        gnconv=gnconv,
        block=Block,
        uniform_init=False,
        out_indices=(0, 1, 2, 3),  # Specifies output stages
        **kwargs,  # Allow passing extra args like pretrained
    ):
        super().__init__()
        dims = [base_dim, base_dim * 2, base_dim * 4, base_dim * 8]  # Dimensions of each stage
        self.num_layers = len(depths)
        self.out_indices = out_indices
        self.uniform_init = uniform_init

        # --- Stem and Downsampling Layers ---
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
        )
        self.downsample_layers.append(stem)
        for i in range(self.num_layers - 1):  # 3 downsampling layers
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        # --- Stages with HorNet Blocks ---
        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # Stochastic depth rates

        # Ensure gnconv config is a list, one per stage
        if not isinstance(gnconv, list):
            gnconv_list = [gnconv for _ in range(self.num_layers)]
        else:
            assert len(gnconv) == self.num_layers
            gnconv_list = gnconv

        cur = 0
        for i in range(self.num_layers):
            stage_gnconv_partial = gnconv_list[i]  # Get the partial function for this stage's gnconv
            # The Block init will call this partial with the correct 'dim' for the stage
            stage = nn.Sequential(
                *[
                    block(
                        dim=dims[i],
                        drop_path=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value,
                        gnconv=stage_gnconv_partial,
                    )  # Pass the partial function
                    for j in range(depths[i])
                ]
            )
            self.stages.append(stage)
            cur += depths[i]

        # --- Output Normalization Layers ---
        self.num_features = [dims[i] for i in range(self.num_layers)]
        for i_layer in out_indices:
            if i_layer < 0 or i_layer >= self.num_layers:
                print(
                    f"Warning: out_indice {i_layer} is out of range for num_layers {self.num_layers}. Skipping norm layer creation."
                )
                continue
            layer = LayerNorm(self.num_features[i_layer], eps=1e-6, data_format="channels_first")
            layer_name = f"norm{i_layer}"
            self.add_module(layer_name, layer)

        # --- Weight Initialization ---
        self.apply(self._init_weights)

        # --- Calculate width_list (Feature Channels) ---
        # This needs to run *after* all layers are defined and weights potentially initialized
        # print("Attempting to calculate width_list...")
        try:
            # Temporarily set eval mode for init forward pass
            # Use a reasonably small dummy input size for faster init
            initial_input_size = 224
            self.eval()
            with torch.no_grad():
                # Check if in_chans is valid
                if in_chans <= 0:
                    raise ValueError(f"in_chans must be positive, got {in_chans}")

                dummy_input = torch.randn(1, in_chans, initial_input_size, initial_input_size)
                # Move dummy input to a potential device if model might be on GPU already (though unlikely during init)
                # This might require knowing the intended device beforehand, tricky during standalone init.
                # If a parameter exists, use its device, otherwise cpu.
                try:
                    p = next(self.parameters())
                    dummy_input = dummy_input.to(p.device)
                    # print(f"  Using device: {p.device} for width_list calculation")
                except StopIteration:
                    print("  Model has no parameters yet? Using CPU for width_list calculation.")
                    dummy_input = dummy_input.to("cpu")

                features = self.forward(dummy_input)  # Perform forward pass
                # Ensure features is a list/tuple before list comprehension
                if isinstance(features, (list, tuple)):
                    self.width_list = [o.shape[1] for o in features]  # Get channel dim (B, C, H, W)
                else:
                    # Handle case where forward might return a single tensor if out_indices has one element?
                    # (Our forward always returns a list based on out_indices)
                    raise TypeError(
                        f"Expected forward pass to return a list or tuple of features, but got {type(features)}"
                    )
            # Set back to train mode
            self.train()
            # print(f"Successfully calculated width_list: {self.width_list}")
        except Exception as e:
            print(f"Warning: Could not calculate width_list during init. Error: {e}")
            # Fallback: Use dimensions defined in 'dims' based on out_indices
            self.width_list = [self.num_features[i] for i in self.out_indices if 0 <= i < self.num_layers]
            print(f"Using fallback width_list based on out_indices: {self.width_list}")

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            if not self.uniform_init:
                trunc_normal_(m.weight, std=0.02)
            else:
                nn.init.xavier_uniform_(m.weight)
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            # Initialize LayerNorm weights to 1 and bias to 0
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        outs = []
        # Iterate through each stage (downsampling + blocks)
        for i in range(self.num_layers):
            # Apply downsampling (stem for i=0, conv layers for i>0)
            x = self.downsample_layers[i](x)
            # Apply the blocks for the current stage
            x = self.stages[i](x)

            # Check if the output of this stage is requested
            if i in self.out_indices:
                # Apply the corresponding normalization layer
                norm_layer = getattr(self, f"norm{i}")
                x_out = norm_layer(x)
                outs.append(x_out)

        # Return a list of feature maps from the specified stages
        return outs


@register_model
def hornet_tiny_7x7(pretrained=False, **kwargs):  # Kept _7x7 for distinction if needed
    s = 1.0 / 3.0
    model = HorNet(
        depths=[2, 3, 18, 2],
        base_dim=64,
        block=Block,
        gnconv=[
            partial(gnconv, order=2, s=s),  # Stage 0: dim=64, order=2
            partial(gnconv, order=3, s=s),  # Stage 1: dim=128, order=3
            partial(gnconv, order=4, s=s),  # Stage 2: dim=256, order=4
            partial(gnconv, order=5, s=s),  # Stage 3: dim=512, order=5
        ],
        **kwargs,
    )
    # Add weight loading logic here if pretrained=True
    return model
