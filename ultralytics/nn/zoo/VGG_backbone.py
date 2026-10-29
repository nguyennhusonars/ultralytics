# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Ported from ultralytics_sonnn nn/modules/VGG_backbone.py (model zoo; see ultralytics/nn/zoo/__init__.py)."""

from typing import Any
from typing import Dict
from typing import List
from typing import Union
import torch
import torch.nn as nn


vgg_cfgs: Dict[str, List[Union[str, int]]] = {
    "VGG11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG16": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "VGG19": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}


MODEL_SPECS = {
    "VGG11": vgg_cfgs["VGG11"],
    "VGG13": vgg_cfgs["VGG13"],
    "VGG16": vgg_cfgs["VGG16"],
    "VGG19": vgg_cfgs["VGG19"],
}


def _make_layers(cfg: List[Union[str, int]], batch_norm: bool = True) -> nn.Sequential:
    """根據配置列表構建 VGG 的卷積層和池化層"""
    layers: List[nn.Module] = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = int(v)  # 確保 v 是整數
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


class VGG(nn.Module):
    def __init__(self, model_name: str, batch_norm: bool = True, init_weights: bool = True):
        """
        VGG 網絡基礎類

        Args:
            model_name (str): VGG模型的名稱 (例如 'VGG16').
            batch_norm (bool): 是否使用 BatchNorm2d.
            init_weights (bool): 是否初始化權重.
        """
        super().__init__()
        assert model_name in MODEL_SPECS.keys(), f"未知模型名稱: {model_name}"

        self.model_name = model_name
        self.cfg = MODEL_SPECS[self.model_name]

        self.features = _make_layers(self.cfg, batch_norm=batch_norm)

        # --- 模仿 MobileNetV4 的結構 ---
        # 識別需要返回輸出的層的索引（通常是 MaxPool 層之後）
        self._output_indices = []
        for i, layer in enumerate(self.features):
            if isinstance(layer, nn.MaxPool2d):
                self._output_indices.append(i)

        if not self._output_indices:  # 如果沒有池化層，可能需要定義不同的輸出點
            #  print("警告：在 VGG 配置中未找到 MaxPool2d 層，將返回最後一層的輸出。")
            self._output_indices.append(len(self.features) - 1)

        # 初始化權重 (可選)
        if init_weights:
            self._initialize_weights()

        # --- 模仿 MobileNetV4 計算 width_list ---
        # 注意：這裡使用與 MobileNetV4 範例相同的輸入尺寸 (640x640)
        # VGG 通常使用 224x224，但為了與請求保持一致，我們使用 640x640
        # print("正在計算 width_list（可能需要一些時間）...")
        with torch.no_grad():
            # 創建一個符合模型預期的虛擬輸入
            dummy_input = torch.randn(1, 3, 640, 640)
            # 執行一次前向傳播以獲取中間層輸出
            intermediate_outputs = self.forward(dummy_input)
            # 從中間層輸出計算通道數
            self.width_list = [output.size(1) for output in intermediate_outputs]
        # print(f"計算得到的 width_list: {self.width_list}")
        # --- 結束模仿 MobileNetV4 計算 width_list ---

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """執行前向傳播並返回指定的中間層輸出列表"""
        outputs: List[torch.Tensor] = []
        current_output_idx = 0
        for i, layer in enumerate(self.features):
            x = layer(x)
            # 檢查當前層的索引是否是我們需要記錄輸出的索引之一
            if current_output_idx < len(self._output_indices) and i == self._output_indices[current_output_idx]:
                outputs.append(x)
                current_output_idx += 1

        # 如果由於某些原因（例如，配置中沒有足夠的池化層）列表為空，
        # 至少返回最後的特徵圖
        if not outputs and len(self.features) > 0:
            outputs.append(x)

        return outputs

    def _initialize_weights(self) -> None:
        """初始化網絡權重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def VGG19(batch_norm: bool = True, **kwargs: Any) -> VGG:
    """創建 VGG19 模型"""
    return VGG("VGG19", batch_norm=batch_norm, **kwargs)
