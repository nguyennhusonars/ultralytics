# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Ported from ultralytics_sonnn nn/modules/OrthoNet.py (model zoo; see ultralytics/nn/zoo/__init__.py)."""

from torch import Tensor
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union
import math
import torch
import torch.nn as nn


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def gram_schmidt(input_tensor: Tensor) -> Tensor:  # Added type hint
    def projection(u: Tensor, v: Tensor) -> Tensor:
        return (v * u).sum() / (u * u).sum() * u

    output = []
    for x_i in input_tensor:  # Renamed x to x_i to avoid conflict with outer scope x if any
        for y_j in output:  # Renamed y to y_j
            x_i = x_i - projection(y_j, x_i)
        x_i = x_i / x_i.norm(p=2)
        output.append(x_i)
    return torch.stack(output)


def initialize_orthogonal_filters(c: int, h: int, w: int) -> Tensor:
    if h * w == 0:  # Avoid division by zero if h or w is 0
        # Fallback or error, e.g., return random non-orthogonal or raise error
        # For now, let's assume h*w > 0 as per typical conv filter sizes
        if h * w == 0:
            raise ValueError("Height and width for orthogonal filters must be positive.")

    if h * w < c:
        num_sets = math.ceil(c / (h * w))  # Use math.ceil to ensure enough filters
        gram_list = []
        for _ in range(num_sets):
            # Ensure the number of vectors for Gram-Schmidt is h*w
            # The shape for Gram-Schmidt should be (num_vectors, ...)
            # torch.rand([h * w, 1, h, w]) means h*w vectors, each of shape (1,h,w)
            gram_list.append(gram_schmidt(torch.rand([h * w, 1, h, w])))

        concatenated_filters = torch.cat(gram_list, dim=0)
        return concatenated_filters[:c]  # Take only the first 'c' filters
    else:
        # We need 'c' orthogonal filters. Gram-Schmidt needs at least 'c' input vectors.
        # Each vector here is implicitly a reshaped (1,h,w) filter.
        return gram_schmidt(torch.rand([c, 1, h, w]))


class GramSchmidtTransform(torch.nn.Module):
    # Corrected instance dictionary key type
    instance: Dict[Tuple[int, int], Optional["GramSchmidtTransform"]] = {}
    constant_filter: Tensor

    @staticmethod
    def build(c: int, h: int) -> "GramSchmidtTransform":  # Added return type hint
        if (c, h) not in GramSchmidtTransform.instance or GramSchmidtTransform.instance[
            (c, h)
        ] is None:  # Check for None too
            GramSchmidtTransform.instance[(c, h)] = GramSchmidtTransform(c, h)
        return GramSchmidtTransform.instance[(c, h)]

    def __init__(self, c: int, h: int):
        super().__init__()
        # No device specified here, will inherit from parent module or use default
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # Can be set later by .to(device)
        with torch.no_grad():
            # Using the locally defined initialize_orthogonal_filters
            rand_ortho_filters = initialize_orthogonal_filters(c, h, h).view(c, h, h)
        # self.register_buffer("constant_filter", rand_ortho_filters.to(self.device).detach())
        self.register_buffer("constant_filter", rand_ortho_filters.detach())

    def forward(self, x: Tensor) -> Tensor:  # Added type hints
        _, _, h_in, w_in = x.shape  # Renamed h, w to h_in, w_in
        _, H_filter, W_filter = self.constant_filter.shape  # Renamed H, W to H_filter, W_filter

        # Ensure x is on the same device as the constant_filter
        # This should typically be handled by calling .to(device) on the top-level model
        # x = x.to(self.constant_filter.device)

        if h_in != H_filter or w_in != W_filter:
            x = torch.nn.functional.adaptive_avg_pool2d(x, (H_filter, W_filter))

        # einsum is often clearer for these types of operations:
        # 'bchw, chw -> bc' (sum over h, w) then keepdim for 'bc11'
        # return torch.einsum('bchw,chw->bc', x, self.constant_filter).unsqueeze(-1).unsqueeze(-1)
        # Original approach:
        return (self.constant_filter * x).sum(dim=(-1, -2), keepdim=True)


class Attention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # Let device be handled by .to(device)

    def forward(self, FWT: GramSchmidtTransform, input_tensor: Tensor) -> Tensor:  # Added type hints, renamed input
        # Ensure input_tensor is on the same device as FWT's parameters
        # This should ideally be handled by the parent module's .to(device) call
        # current_input = input_tensor.to(FWT.constant_filter.device)
        current_input = input_tensor

        # happens once in case of BigFilter
        while current_input.size(-1) > 1 and current_input.size(-2) > 1:  # Ensure both H and W are > 1
            current_input = FWT(current_input)
        b = current_input.size(0)
        return current_input.view(b, -1)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, height, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self._process: nn.Module = nn.Sequential(
            conv3x3(inplanes, planes, stride),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            conv3x3(planes, planes),
            nn.BatchNorm2d(planes),
        )
        self.downsample = downsample
        self.stride = stride
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # Let device be handled by .to(device)
        self.planes = planes

        # For Linear layers, device is set if bias=False and device is passed.
        # It's better to let the top-level .to(device) handle this.
        self._excitation = nn.Sequential(
            nn.Linear(in_features=planes, out_features=round(planes / 16), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=round(planes / 16), out_features=planes, bias=False),
            nn.Sigmoid(),
        )
        self.OrthoAttention = Attention()  # Corrected instantiation
        # Using locally defined GramSchmidtTransform
        self.F_C_A = GramSchmidtTransform.build(planes, height)

    def forward(self, x: Tensor) -> Tensor:  # Added type hints
        residual = x if self.downsample is None else self.downsample(x)
        out = self._process(x)

        # Ensure 'out' is on the same device as F_C_A's buffer
        # This should be handled by the model's top-level .to(device) call
        compressed = self.OrthoAttention(self.F_C_A, out)

        b, c = out.size(0), out.size(1)

        # Ensure 'compressed' is on the same device as _excitation layers
        excitation = self._excitation(compressed.to(next(self._excitation.parameters()).device))
        excitation = excitation.view(b, c, 1, 1)

        attention_out = excitation * out  # Renamed 'attention' to 'attention_out' to avoid conflict
        attention_out += residual
        activated = torch.relu(attention_out)
        return activated


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, height, stride=1, downsample=None):
        super(Bottleneck, self).__init__()

        self._process: nn.Module = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False),  # Use self.expansion
            nn.BatchNorm2d(planes * self.expansion),  # Use self.expansion
        )
        self.downsample = downsample
        self.stride = stride
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # Let device be handled by .to(device)
        self.planes = planes  # This is the 'intermediate' planes, output planes is planes * expansion

        output_features_excitation = planes * self.expansion
        self._excitation = nn.Sequential(
            nn.Linear(
                in_features=output_features_excitation, out_features=round(output_features_excitation / 16), bias=False
            ),  # Adjusted for expansion
            nn.ReLU(inplace=True),
            nn.Linear(
                in_features=round(output_features_excitation / 16), out_features=output_features_excitation, bias=False
            ),  # Adjusted for expansion
            nn.Sigmoid(),
        )
        self.OrthoAttention = Attention()  # Corrected instantiation
        # Using locally defined GramSchmidtTransform
        self.F_C_A = GramSchmidtTransform.build(
            planes * self.expansion, height
        )  # Channels for attention are output_features_excitation

    def forward(self, x: Tensor) -> Tensor:  # Added type hints
        residual = x if self.downsample is None else self.downsample(x)
        out = self._process(x)

        # Ensure 'out' is on the same device as F_C_A's buffer
        compressed = self.OrthoAttention(self.F_C_A, out)

        b, c = out.size(0), out.size(1)

        # Ensure 'compressed' is on the same device as _excitation layers
        excitation = self._excitation(compressed.to(next(self._excitation.parameters()).device))
        attention_out = excitation.view(b, c, 1, 1)  # Renamed
        attention_out = attention_out * out
        attention_out += residual
        activated = torch.relu(attention_out)
        return activated


class OrthoNet(nn.Module):
    def __init__(
        self, block: type[Union[BasicBlock, Bottleneck]], layers: List[int], num_classes: int = 1000, **kwargs
    ):  # Added **kwargs
        super(OrthoNet, self).__init__()
        self._device_str = "cuda" if torch.cuda.is_available() else "cpu"  # Renamed to avoid conflict
        self.inplanes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # The 'height' parameter is specific to OrthoNet's block design.
        # These seem to correspond to expected feature map heights at each stage.
        # Input 224 -> conv1 (s2) -> 112 -> maxpool (s2) -> 56. This is input to layer1.
        # Heights for layers:
        # layer1: 56 (passed as 64 to block, maybe for filter init?)
        # layer2: 28 (passed as 32)
        # layer3: 14 (passed as 16)
        # layer4: 7  (passed as 8)
        # For AvgPool(8), it assumes input is 8x8. If layer4 output is 7x7, this needs adjustment.
        # Let's assume the 'height' params for _make_layer are for the OrthoAttention's filter sizes.

        self.layer1 = self._make_layer(
            block, 64, 56, layers[0]
        )  # Original was 64, let's try with actual expected H for filters
        self.layer2 = self._make_layer(block, 128, 28, layers[1], stride=2)  # Original 32
        self.layer3 = self._make_layer(block, 256, 14, layers[2], stride=2)  # Original 16
        self.layer4 = self._make_layer(block, 512, 7, layers[3], stride=2)  # Original 8

        # Adaptive average pooling is more robust to input size variations before FC layer
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # Output size (1,1) for any input size
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        # self.to(self._device_str) # Move to device after all params are initialized.

        # --- Calculate width_list ---
        self.width_list: List[int] = []
        try:
            original_mode = self.training
            self.eval()
            with torch.no_grad():
                # Standard dummy input size, ensure it's on the correct device
                dummy_input = torch.randn(1, 3, 224, 224)  # .to(self._device_str)
                # If model is on GPU, dummy_input should also be. This is handled by example usage.
                # For init, it's safer if dummy_input matches the device of first layer's params
                if next(self.parameters()).is_cuda:
                    dummy_input = dummy_input.cuda()

                features = self._forward_extract(dummy_input)
                self.width_list = [f.size(1) for f in features]
            self.train(original_mode)
        except Exception as e:
            print(f"Warning: Could not compute width_list during OrthoNet init: {e}")
            import traceback

            traceback.print_exc()
            self.width_list = []
        # --- End of width_list calculation ---

    def _make_layer(
        self, block: type[Union[BasicBlock, Bottleneck]], planes: int, height: int, blocks: int, stride: int = 1
    ) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers_list = []  # Renamed from layers to avoid conflict
        layers_list.append(block(self.inplanes, planes, height, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):  # Corrected loop variable i to _
            layers_list.append(block(self.inplanes, planes, height))

        return nn.Sequential(*layers_list)

    def _forward_extract(self, x: Tensor) -> List[Tensor]:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)

        return [f1, f2, f3, f4]

    def forward(self, x: Tensor) -> List[Tensor]:  # Modified to return list of features
        return self._forward_extract(x)

    def get_classification_output(self, x: Tensor) -> Tensor:
        """Helper to get final classification output if needed."""
        features = self._forward_extract(x)
        out = features[-1]  # Take the last feature map
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def orthonet50(**kwargs) -> OrthoNet:
    """Constructs a OrthoNet-50 model.
    Args:
        **kwargs: Keyword arguments, including 'num_classes'.
    """
    model = OrthoNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model
