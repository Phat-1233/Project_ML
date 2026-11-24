"""PreAct ResNet-18 cho CIFAR, dựng lại ĐÚNG như mã chính thức của tác giả
(facebookresearch/mixup-cifar10, models/resnet.py: hàm ResNet18 dùng PreActBlock),
vốn chuyển thể từ kuangliu/pytorch-cifar.

Chi tiết cài đặt bài báo không nêu (nên ghi trong báo cáo):
  * Stem là conv3x3 -> BN -> ReLU (giống ResNet thường), rồi mới đến các khối
    pre-activation.
  * KHÔNG có BN + ReLU cuối cùng trước average pooling, khác với PreAct ResNet
    "chuẩn" của He et al. (2016b).
  * Shortcut khi đổi kích thước là conv1x1 không có BN.
  * Khởi tạo trọng số để mặc định của PyTorch.
11.171.274 tham số với 10 lớp.

Các khối được đặt tên conv1/bn1/layer1..layer4/linear để `LatentMixer` có thể trộn
biểu diễn ẩn trước "Layer 1-5" như ablation 3.8:
    Layer 1 = trước layer1, ..., Layer 4 = trước layer4, Layer 5 = trước linear.
"""
from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class PreActBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, dropout: float = 0.0):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        # Dropout giữa 2 conv, sau ReLU (dùng cho 3.4, theo Zagoruyko & Komodakis 2016)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False))

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out)
        out = self.conv1(out)
        out = self.conv2(self.drop(F.relu(self.bn2(out))))
        return out + shortcut


class Flatten(nn.Module):
    def forward(self, x):
        return F.avg_pool2d(x, 4).flatten(1)


class PreActResNet(nn.Module):
    def __init__(self, num_blocks=(2, 2, 2, 2), num_classes=10, dropout: float = 0.0):
        super().__init__()
        self.in_planes = 64
        self.conv1 = conv3x3(3, 64)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, num_blocks[0], 1, dropout)
        self.layer2 = self._make_layer(128, num_blocks[1], 2, dropout)
        self.layer3 = self._make_layer(256, num_blocks[2], 2, dropout)
        self.layer4 = self._make_layer(512, num_blocks[3], 2, dropout)
        self.pool = Flatten()
        self.linear = nn.Linear(512, num_classes)

    def _make_layer(self, planes, n, stride, dropout):
        layers = []
        for s in [stride] + [1] * (n - 1):
            layers.append(PreActBlock(self.in_planes, planes, s, dropout))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer4(self.layer3(self.layer2(self.layer1(out))))
        return self.linear(self.pool(out))


def preact_resnet18(num_classes: int = 10, dropout: float = 0.0) -> PreActResNet:
    return PreActResNet((2, 2, 2, 2), num_classes, dropout)


MODELS = {"preact_resnet18": preact_resnet18}

# Ánh xạ tên "Layer k" trong Bảng 5 của bài báo sang tên module cho LatentMixer
LATENT_LAYERS = {1: "layer1", 2: "layer2", 3: "layer3", 4: "layer4", 5: "linear"}



import math

import torch


MODEL_NAMES = ("preact_resnet18", "wideresnet28_10", "densenet_bc190")


class _WideBlock(nn.Module):
    """Two preactivated convolutions with an optional preactivated projection."""

    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.projection = (
            nn.Conv2d(in_channels, out_channels, 1, stride, bias=False)
            if in_channels != out_channels or stride != 1 else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        activated = F.relu(self.bn1(x))
        shortcut = x if self.projection is None else self.projection(activated)
        residual = self.conv1(activated)
        residual = self.conv2(F.relu(self.bn2(residual)))
        return shortcut + residual


class WideResNet(nn.Module):
    """CIFAR WRN (depth=6n+4); defaults are WRN-28-10, with no dropout."""

    def __init__(self, depth: int = 28, width: int = 10, num_classes: int = 10):
        super().__init__()
        if depth < 10 or (depth - 4) % 6:
            raise ValueError("WideResNet depth must be 6n+4 with n >= 1")
        if width < 1 or not isinstance(width, int) or num_classes < 1:
            raise ValueError("width and num_classes must be positive integers")
        blocks_per_group = (depth - 4) // 6
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1, bias=False)
        channels = 16
        groups = []
        for group_index, next_channels in enumerate((16 * width, 32 * width, 64 * width)):
            group = []
            for block_index in range(blocks_per_group):
                stride = 2 if group_index > 0 and block_index == 0 else 1
                group.append(_WideBlock(channels, next_channels, stride))
                channels = next_channels
            groups.append(nn.Sequential(*group))
        self.groups = nn.Sequential(*groups)
        self.bn = nn.BatchNorm2d(channels)
        self.fc = nn.Linear(channels, num_classes)
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.uniform_(module.weight, 0.0, 1.0)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.groups(self.conv1(x))
        x = F.relu(self.bn(x))
        return self.fc(F.avg_pool2d(x, 8).flatten(1))


class _DenseLayer(nn.Module):
    def __init__(self, channels: int, growth: int):
        super().__init__()
        bottleneck_channels = 4 * growth
        self.bottleneck = nn.Sequential(
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, bottleneck_channels, 1, bias=False),
            nn.BatchNorm2d(bottleneck_channels),
            nn.ReLU(),
            nn.Conv2d(bottleneck_channels, growth, 3, padding=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat((x, self.bottleneck(x)), dim=1)


class DenseNetBC(nn.Module):
    """Three CIFAR dense blocks, bottleneck factor 4, and compressed transitions.

    Defaults give DenseNet-BC-190 (growth 40, compression 0.5), with 31 layers in
    each block. Smaller depth/growth values are useful for CPU validation.
    """

    def __init__(self, depth: int = 190, growth: int = 40,
                 reduction: float = 0.5, num_classes: int = 10):
        super().__init__()
        if depth < 10 or (depth - 4) % 6:
            raise ValueError("DenseNet-BC depth must be 6n+4 with n >= 1")
        if growth < 1 or not isinstance(growth, int) or num_classes < 1:
            raise ValueError("growth and num_classes must be positive integers")
        if not 0 < reduction <= 1:
            raise ValueError("reduction must be in (0, 1]")
        layers_per_block = (depth - 4) // 6
        channels = 2 * growth
        self.conv1 = nn.Conv2d(3, channels, 3, padding=1, bias=False)
        blocks = []
        for block_index in range(3):
            dense_layers = []
            for _ in range(layers_per_block):
                dense_layers.append(_DenseLayer(channels, growth))
                channels += growth
            blocks.append(nn.Sequential(*dense_layers))
            if block_index < 2:
                compressed_channels = math.floor(channels * reduction)
                if compressed_channels < 1:
                    raise ValueError("reduction leaves a transition with no channels")
                blocks.append(nn.Sequential(
                    nn.BatchNorm2d(channels),
                    nn.ReLU(),
                    nn.Conv2d(channels, compressed_channels, 1, bias=False),
                    nn.AvgPool2d(2),
                ))
                channels = compressed_channels
        self.blocks = nn.Sequential(*blocks)
        self.bn = nn.BatchNorm2d(channels)
        self.fc = nn.Linear(channels, num_classes)
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                # Reference code keeps nn.Linear's default weight initialization.
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.blocks(self.conv1(x))
        x = F.relu(self.bn(x))
        return self.fc(F.avg_pool2d(x, 8).flatten(1))


def build_model(name: str, num_classes: int = 10) -> nn.Module:
    """Build one Figure 3 architecture, retaining the shared ResNet factory."""
    if name == "preact_resnet18":
        return preact_resnet18(num_classes)
    if name == "wideresnet28_10":
        return WideResNet(num_classes=num_classes)
    if name == "densenet_bc190":
        return DenseNetBC(num_classes=num_classes)
    raise ValueError(f"Unknown CIFAR model {name!r}; choose from {MODEL_NAMES}")
