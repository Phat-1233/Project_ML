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
