# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""FDConv (Frequency Dynamic Convolution, CVPR 2025), ported from https://github.com/Linwei-Chen/FDConv (MIT).

The fork's own rewrite (ultralytics_sonnn nn/modules/FDConv.py) could not run once channels exceeded 16; this follows
FDConv_detection/mmdet_custom/FDConv.py with mmcv, plotting and dead options removed.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class StarReLU(nn.Module):
    """StarReLU: s * relu(x) ** 2 + b."""

    def __init__(self, scale_value=1.0, bias_value=0.0):
        super().__init__()
        self.relu = nn.ReLU()
        self.scale = nn.Parameter(scale_value * torch.ones(1))
        self.bias = nn.Parameter(bias_value * torch.ones(1))

    def forward(self, x):
        return self.scale * self.relu(x) ** 2 + self.bias


def get_fft2freq(d1, d2, use_rfft=False):
    """Coordinates (2, d1 * d2') of a (d1, d2) spectrum sorted from low to high frequency, and the frequency grid."""
    freq_h = torch.fft.fftfreq(d1)
    freq_w = torch.fft.rfftfreq(d2) if use_rfft else torch.fft.fftfreq(d2)
    freq_hw = torch.stack(torch.meshgrid(freq_h, freq_w, indexing="ij"), dim=-1)
    _, indices = torch.sort(torch.norm(freq_hw, dim=-1).view(-1))
    if use_rfft:
        d2 = d2 // 2 + 1
    return torch.stack([indices // d2, indices % d2], dim=-1).permute(1, 0), freq_hw


class KernelSpatialModulation_Global(nn.Module):
    """Channel, filter, spatial and kernel attentions from the pooled input (KSM global branch)."""

    def __init__(
        self,
        in_planes,
        out_planes,
        kernel_size,
        groups=1,
        reduction=0.0625,
        kernel_num=4,
        min_channel=16,
        temp=1.0,
        kernel_temp=None,
        att_multi=2.0,
        ksm_only_kernel_att=False,
        stride=1,
        spatial_freq_decompose=False,
        act_type="sigmoid",
    ):
        super().__init__()
        attention_channel = max(int(in_planes * reduction), min_channel)
        self.act_type = act_type
        self.kernel_size = kernel_size
        self.temperature = temp
        self.kernel_temp = kernel_temp
        self.att_multi = att_multi

        self.fc = nn.Conv2d(in_planes, attention_channel, 1, bias=False)
        self.bn = nn.BatchNorm2d(attention_channel)
        self.relu = StarReLU()

        if ksm_only_kernel_att:
            self.func_channel = self.skip
        else:
            c = in_planes * 2 if spatial_freq_decompose and kernel_size > 1 else in_planes
            self.channel_fc = nn.Conv2d(attention_channel, c, 1, bias=True)
            self.func_channel = self.get_channel_attention

        if (in_planes == groups and in_planes == out_planes) or ksm_only_kernel_att:  # depth-wise convolution
            self.func_filter = self.skip
        else:
            c = out_planes * 2 if spatial_freq_decompose else out_planes
            self.filter_fc = nn.Conv2d(attention_channel, c, 1, stride=stride, bias=True)
            self.func_filter = self.get_filter_attention

        if kernel_size == 1 or ksm_only_kernel_att:  # point-wise convolution
            self.func_spatial = self.skip
        else:
            self.spatial_fc = nn.Conv2d(attention_channel, kernel_size * kernel_size, 1, bias=True)
            self.func_spatial = self.get_spatial_attention

        if kernel_num == 1:
            self.func_kernel = self.skip
        else:
            self.kernel_fc = nn.Conv2d(attention_channel, kernel_num, 1, bias=True)
            self.func_kernel = self.get_kernel_attention

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        for name in ("spatial_fc", "kernel_fc", "channel_fc"):  # near-zero init: attentions start neutral
            if hasattr(self, name):
                nn.init.normal_(getattr(self, name).weight, std=1e-6)

    @staticmethod
    def skip(_):
        return 1.0

    def _act(self, att):
        if self.act_type == "sigmoid":
            return torch.sigmoid(att / self.temperature) * self.att_multi
        if self.act_type == "tanh":
            return 1 + torch.tanh(att / self.temperature)
        raise NotImplementedError(self.act_type)

    def get_channel_attention(self, x):
        return self._act(self.channel_fc(x).view(x.size(0), 1, 1, -1, x.size(-2), x.size(-1)))

    def get_filter_attention(self, x):
        return self._act(self.filter_fc(x).view(x.size(0), 1, -1, 1, x.size(-2), x.size(-1)))

    def get_spatial_attention(self, x):
        return self._act(self.spatial_fc(x).view(x.size(0), 1, 1, 1, self.kernel_size, self.kernel_size))

    def get_kernel_attention(self, x):
        att = self.kernel_fc(x).view(x.size(0), -1, 1, 1, 1, 1)
        if self.act_type == "softmax":
            return F.softmax(att / self.kernel_temp, dim=1)
        if self.act_type == "sigmoid":
            return torch.sigmoid(att / self.kernel_temp) * 2 / att.size(1)
        if self.act_type == "tanh":
            return (1 + torch.tanh(att / self.kernel_temp)) / att.size(1)
        raise NotImplementedError(self.act_type)

    def forward(self, x):
        avg_x = self.relu(self.bn(self.fc(x)))
        return self.func_channel(avg_x), self.func_filter(avg_x), self.func_spatial(avg_x), self.func_kernel(avg_x)


class KernelSpatialModulation_Local(nn.Module):
    """ECA-style 1D conv over the pooled channels giving a per-weight attention (KSM local branch)."""

    def __init__(self, channel, kernel_num=1, out_n=1, use_global=False):
        super().__init__()
        self.kn = kernel_num
        self.out_n = out_n
        k_size = round((math.log2(channel) / 2) + 0.5) // 2 * 2 + 1
        self.conv = nn.Conv1d(1, kernel_num * out_n, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        nn.init.constant_(self.conv.weight, 1e-6)
        self.use_global = use_global
        if use_global:
            self.complex_weight = nn.Parameter(torch.randn(1, channel // 2 + 1, 2) * 1e-6)
        self.norm = nn.LayerNorm(channel)

    def forward(self, x):
        x = x.squeeze(-1).transpose(-1, -2)  # b, c, 1, 1 -> b, 1, c
        c = x.shape[-1]
        if self.use_global:
            x_rfft = torch.fft.rfft(x.float(), dim=-1)
            x_real = x_rfft.real * self.complex_weight[..., 0][None]
            x_imag = x_rfft.imag * self.complex_weight[..., 1][None]
            x = x + torch.fft.irfft(torch.view_as_complex(torch.stack([x_real, x_imag], dim=-1)), dim=-1)
        att_logit = self.conv(self.norm(x)).reshape(x.size(0), self.kn, self.out_n, c)
        return att_logit.permute(0, 1, 3, 2)  # b, kn, cin, out_n


class FrequencyBandModulation(nn.Module):
    """Split the input into frequency bands and reweight each high band with a spatial attention."""

    def __init__(
        self,
        in_channels,
        k_list=(2, 4, 8),
        lowfreq_att=False,
        act="sigmoid",
        spatial_group=1,
        spatial_kernel=3,
        max_size=(64, 64),
    ):
        super().__init__()
        self.k_list = k_list
        self.lowfreq_att = lowfreq_att
        self.act = act
        self.spatial_group = in_channels if spatial_group > 64 else spatial_group
        self.freq_weight_conv_list = nn.ModuleList()
        for _ in range(len(k_list) + int(lowfreq_att)):
            conv = nn.Conv2d(
                in_channels, self.spatial_group, spatial_kernel, padding=spatial_kernel // 2, groups=self.spatial_group
            )
            nn.init.normal_(conv.weight, std=1e-6)
            conv.bias.data.zero_()
            self.freq_weight_conv_list.append(conv)
        _, freq = get_fft2freq(*max_size, use_rfft=True)
        freq = freq.abs().max(dim=-1)[0]
        masks = torch.stack([freq < 0.5 / f + 1e-8 for f in k_list]).unsqueeze(1)  # n, 1, max_h, max_w // 2 + 1
        self.register_buffer("cached_masks", masks, persistent=False)

    def sp_act(self, w):
        if self.act == "sigmoid":
            return w.sigmoid() * 2
        if self.act == "tanh":
            return 1 + w.tanh()
        if self.act == "softmax":
            return w.softmax(dim=1) * w.shape[1]
        raise NotImplementedError(self.act)

    def forward(self, x):
        att_feat = x
        x = x.float()
        pre_x = x.clone()
        b, _, h, w = x.shape
        x_fft = torch.fft.rfft2(x, norm="ortho")
        masks = F.interpolate(self.cached_masks.float(), size=(h, w // 2 + 1), mode="nearest")
        x_list = []
        for idx in range(len(self.k_list)):
            low_part = torch.fft.irfft2(x_fft * masks[idx], s=(h, w), norm="ortho")
            high_part = pre_x - low_part
            pre_x = low_part
            att = self.sp_act(self.freq_weight_conv_list[idx](att_feat))
            tmp = att.reshape(b, self.spatial_group, -1, h, w) * high_part.reshape(b, self.spatial_group, -1, h, w)
            x_list.append(tmp.reshape(b, -1, h, w))
        if self.lowfreq_att:
            att = self.sp_act(self.freq_weight_conv_list[len(self.k_list)](att_feat))
            tmp = att.reshape(b, self.spatial_group, -1, h, w) * pre_x.reshape(b, self.spatial_group, -1, h, w)
            x_list.append(tmp.reshape(b, -1, h, w))
        else:
            x_list.append(pre_x)
        return sum(x_list)


class FDConv(nn.Conv2d):
    """Conv2d whose kernel is rebuilt per sample from kernel_num frequency-disjoint weight groups.

    Falls back to a plain Conv2d when min(in, out) <= use_fdconv_if_c_gt or the kernel size is not 1 or 3.
    """

    def __init__(
        self,
        *args,
        reduction=0.0625,
        kernel_num=4,
        use_fdconv_if_c_gt=16,
        use_fdconv_if_k_in=(1, 3),
        use_fbm_if_k_in=(3,),
        use_fbm_for_stride=False,
        kernel_temp=1.0,
        temp=None,
        att_multi=2.0,
        param_ratio=1,
        param_reduction=1.0,
        ksm_only_kernel_att=False,
        use_ksm_local=True,
        ksm_local_act="sigmoid",
        ksm_global_act="sigmoid",
        spatial_freq_decompose=False,
        n_freq_groups=1,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.use_fdconv_if_c_gt = use_fdconv_if_c_gt
        self.use_fdconv_if_k_in = use_fdconv_if_k_in
        self.kernel_num = kernel_num
        self.param_ratio = param_ratio
        self.param_reduction = param_reduction
        self.use_ksm_local = use_ksm_local
        self.att_multi = att_multi
        self.ksm_local_act = ksm_local_act
        assert ksm_local_act in {"sigmoid", "tanh"}
        assert ksm_global_act in {"softmax", "sigmoid", "tanh"}
        if temp is None:
            temp = kernel_temp
        if not self.dynamic:
            return
        self.alpha = min(self.out_channels, self.in_channels) // 2 * kernel_num * param_ratio / param_reduction
        self.KSM_Global = KernelSpatialModulation_Global(
            self.in_channels,
            self.out_channels,
            self.kernel_size[0],
            groups=self.groups,
            temp=temp,
            kernel_temp=kernel_temp,
            reduction=reduction,
            kernel_num=kernel_num * param_ratio,
            att_multi=att_multi,
            ksm_only_kernel_att=ksm_only_kernel_att,
            act_type=ksm_global_act,
            stride=self.stride,
            spatial_freq_decompose=spatial_freq_decompose,
        )
        if self.kernel_size[0] in use_fbm_if_k_in or (use_fbm_for_stride and self.stride[0] > 1):
            self.FBM = FrequencyBandModulation(self.in_channels, spatial_group=n_freq_groups)
        if use_ksm_local:
            self.KSM_Local = KernelSpatialModulation_Local(
                self.in_channels, kernel_num=1, out_n=self.out_channels * self.kernel_size[0] * self.kernel_size[1]
            )
        self.convert2dftweight()

    @property
    def dynamic(self):
        """Whether this layer runs the frequency-dynamic path (otherwise it is a plain Conv2d)."""
        return (
            min(self.in_channels, self.out_channels) > self.use_fdconv_if_c_gt
            and self.kernel_size[0] in self.use_fdconv_if_k_in
        )

    def convert2dftweight(self):
        """Replace self.weight by its 2D rFFT, split into kernel_num groups of frequency coordinates."""
        d1, d2, k1, k2 = self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1]
        freq_indices, _ = get_fft2freq(d1 * k1, d2 * k2, use_rfft=True)
        weight_rfft = torch.fft.rfft2(self.weight.permute(0, 2, 1, 3).reshape(d1 * k1, d2 * k2), dim=(0, 1))
        weight_rfft = torch.stack([weight_rfft.real, weight_rfft.imag], dim=-1)
        if self.param_reduction < 1:  # keep only the lowest frequencies
            freq_indices = freq_indices[:, : int(freq_indices.size(1) * self.param_reduction)]
            weight_rfft = weight_rfft[freq_indices[0, :], freq_indices[1, :]].reshape(-1, 2)
        weight_rfft = weight_rfft[None].repeat(self.param_ratio, *[1] * weight_rfft.dim()) / (min(d1, d2) // 2)
        self.dft_weight = nn.Parameter(weight_rfft)
        del self.weight
        indices = freq_indices.reshape(2, self.kernel_num, -1)
        self.register_buffer("indices", torch.stack([indices] * self.param_ratio), persistent=False)

    def forward(self, x):
        if not self.dynamic:
            return super().forward(x)
        global_x = F.adaptive_avg_pool2d(x, 1)
        channel_attention, filter_attention, spatial_attention, kernel_attention = self.KSM_Global(global_x)
        b, in_planes, height, width = x.size()
        if self.use_ksm_local:
            hr_att_logit = self.KSM_Local(global_x).reshape(
                b, 1, self.in_channels, self.out_channels, self.kernel_size[0], self.kernel_size[1]
            )
            hr_att_logit = hr_att_logit.permute(0, 1, 3, 2, 4, 5)
            hr_att = (
                hr_att_logit.sigmoid() * self.att_multi if self.ksm_local_act == "sigmoid" else 1 + hr_att_logit.tanh()
            )
        else:
            hr_att = 1
        dft_map = torch.zeros(
            (b, self.out_channels * self.kernel_size[0], self.in_channels * self.kernel_size[1] // 2 + 1, 2),
            device=x.device,
        )
        kernel_attention = kernel_attention.reshape(b, self.param_ratio, self.kernel_num, -1)
        for i in range(self.param_ratio):
            indices = self.indices[i]
            if self.param_reduction < 1:
                w = self.dft_weight[i].reshape(self.kernel_num, -1, 2)[None]
            else:
                w = self.dft_weight[i][indices[0, :, :], indices[1, :, :]][None] * self.alpha  # 1, kernel_num, -1, 2
            dft_map[:, indices[0, :, :], indices[1, :, :]] += torch.stack(
                [w[..., 0] * kernel_attention[:, i], w[..., 1] * kernel_attention[:, i]], dim=-1
            )
        adaptive_weights = torch.fft.irfft2(torch.view_as_complex(dft_map), dim=(1, 2)).reshape(
            b, 1, self.out_channels, self.kernel_size[0], self.in_channels, self.kernel_size[1]
        )
        adaptive_weights = adaptive_weights.permute(0, 1, 2, 4, 3, 5)
        if hasattr(self, "FBM"):
            x = self.FBM(x)

        # attentions folded into the kernel when it is cheaper than applying them to the feature maps
        fold_all = (
            self.out_channels * self.in_channels * self.kernel_size[0] * self.kernel_size[1]
            < (in_planes + self.out_channels) * height * width
        )
        if fold_all:
            aggregate_weight = spatial_attention * channel_attention * filter_attention * adaptive_weights * hr_att
        else:
            aggregate_weight = spatial_attention * adaptive_weights * hr_att
            if not isinstance(channel_attention, float):
                x = x * channel_attention.view(b, -1, 1, 1)
        aggregate_weight = torch.sum(aggregate_weight, dim=1).view(
            -1, self.in_channels // self.groups, self.kernel_size[0], self.kernel_size[1]
        )
        output = F.conv2d(
            x.reshape(1, -1, height, width),
            weight=aggregate_weight,
            bias=None,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups * b,
        )
        output = output.view(b, self.out_channels, output.size(-2), output.size(-1))
        if not fold_all and not isinstance(filter_attention, float):
            output = output * filter_attention.view(b, -1, 1, 1)
        if self.bias is not None:
            output = output + self.bias.view(1, -1, 1, 1)
        return output
