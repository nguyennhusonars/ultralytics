# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Ported from ultralytics_sonnn nn/modules/RFAConv.py (model zoo; see ultralytics/nn/zoo/__init__.py)."""

from einops import rearrange
from torch import nn

from ultralytics.nn.modules.conv import Conv


class RFAConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1):
        super().__init__()
        self.kernel_size = kernel_size

        self.get_weight = nn.Sequential(
            nn.AvgPool2d(kernel_size=kernel_size, padding=kernel_size // 2, stride=stride),
            nn.Conv2d(in_channel, in_channel * (kernel_size**2), kernel_size=1, groups=in_channel, bias=False),
        )
        self.generate_feature = nn.Sequential(
            nn.Conv2d(
                in_channel,
                in_channel * (kernel_size**2),
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                stride=stride,
                groups=in_channel,
                bias=False,
            ),
            nn.BatchNorm2d(in_channel * (kernel_size**2)),
            nn.ReLU(),
        )

        # self.conv = nn.Sequential(nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=kernel_size),
        #                           nn.BatchNorm2d(out_channel),
        #                           nn.ReLU())
        self.conv = Conv(in_channel, out_channel, k=kernel_size, s=kernel_size, p=0)

    def forward(self, x):
        b, c = x.shape[0:2]
        weight = self.get_weight(x)
        h, w = weight.shape[2:]
        weighted = weight.view(b, c, self.kernel_size**2, h, w).softmax(2)  # b c*kernel**2,h,w ->  b c k**2 h w
        feature = self.generate_feature(x).view(b, c, self.kernel_size**2, h, w)  # b c*kernel**2,h,w ->  b c k**2 h w
        weighted_data = feature * weighted
        conv_data = rearrange(
            weighted_data,
            "b c (n1 n2) h w -> b c (h n1) (w n2)",
            n1=self.kernel_size,  # b c k**2 h w ->  b c h*k w*k
            n2=self.kernel_size,
        )
        return self.conv(conv_data)
