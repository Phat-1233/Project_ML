"""Lịch learning rate dùng trong các thực nghiệm của bài báo.

Mọi scheduler ở đây được gọi `.step()` MỘT LẦN SAU MỖI EPOCH.

Ghi chú khi đối chiếu mã gốc: mã chính thức gọi adjust_learning_rate(epoch) sau
epoch `epoch` với điều kiện `epoch >= 100`, nên lr chỉ giảm từ epoch thứ 101
(chỉ số 0). Bài báo viết "divided by 10 after 100 and 150 epochs", tức giảm từ
epoch chỉ số 100. Ta theo bài báo (milestones=[100,150]); muốn khớp mã gốc thì
dùng `official_offset=True`. Đây là một "chi tiết cài đặt" nên nêu trong báo cáo.
"""
from __future__ import annotations

from typing import Sequence

import torch
from torch.optim.lr_scheduler import LambdaLR


def step_schedule(optimizer: torch.optim.Optimizer, milestones: Sequence[int], gamma: float = 0.1,
                  warmup_epochs: int = 0, warmup_start_factor: float = 1.0,
                  official_offset: bool = False) -> LambdaLR:
    """lr = base_lr · gamma^(số milestone đã qua), có tùy chọn warm-up tuyến tính.

    - CIFAR PreAct/DenseNet: milestones=[100,150]
    - CIFAR WideResNet:      milestones=[60,120,180], gamma=0.2 theo WRN gốc (bài báo ghi /10)
    - ImageNet (mục 3.1):    warmup 5 epoch từ 0.1 -> 0.4 (warmup_start_factor=0.25)
    - Speech (mục 3.3):      Adam, chia 10 mỗi 10 epoch -> milestones=[10,20]
    """
    shift = 1 if official_offset else 0
    ms = sorted(int(m) + shift for m in milestones)

    def factor(epoch: int) -> float:
        if warmup_epochs > 0 and epoch < warmup_epochs:
            return warmup_start_factor + (1 - warmup_start_factor) * epoch / warmup_epochs
        return gamma ** sum(epoch >= m for m in ms)

    return LambdaLR(optimizer, lr_lambda=factor)


def constant_schedule(optimizer: torch.optim.Optimizer) -> LambdaLR:
    return LambdaLR(optimizer, lr_lambda=lambda e: 1.0)
