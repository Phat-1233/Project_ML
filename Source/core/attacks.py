"""Tấn công đối kháng cho mục 3.5 (FGSM và I-FGSM), dùng được cho mọi bộ phân loại.

Bài báo dùng FGSM (Goodfellow et al., 2015) và I-FGSM (Kurakin et al., 2017) với
ε = 4 (trên thang pixel 0-255) và I-FGSM 10 bước, mỗi bước 1 pixel. Vì input đã
chuẩn hóa (x - mean)/std, ta tấn công trong không gian pixel [0,1] rồi chuẩn hóa
lại bên trong `forward` để ε có đúng nghĩa như bài báo.
"""
from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class Normalized(nn.Module):
    """Bọc model để nhận ảnh trong [0,1] và tự chuẩn hóa bên trong."""

    def __init__(self, model: nn.Module, mean, std):
        super().__init__()
        self.model = model
        self.register_buffer("mean", torch.tensor(mean).view(1, -1, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, -1, 1, 1))

    def forward(self, x):
        return self.model((x - self.mean) / self.std)


def fgsm(model: nn.Module, x: torch.Tensor, y: torch.Tensor, eps: float,
         clamp: Optional[tuple] = (0.0, 1.0)) -> torch.Tensor:
    """x_adv = x + ε · sign(∇_x L(f(x), y))"""
    x = x.clone().detach().requires_grad_(True)
    loss = F.cross_entropy(model(x), y)
    grad, = torch.autograd.grad(loss, x)
    x_adv = x + eps * grad.sign()
    return x_adv.clamp(*clamp).detach() if clamp else x_adv.detach()


def ifgsm(model: nn.Module, x: torch.Tensor, y: torch.Tensor, eps: float, step: float, iters: int,
          clamp: Optional[tuple] = (0.0, 1.0)) -> torch.Tensor:
    """FGSM lặp, chiếu lại vào quả cầu L∞ bán kính ε quanh x sau mỗi bước."""
    x0 = x.clone().detach()
    x_adv = x0.clone()
    for _ in range(iters):
        x_adv = fgsm(model, x_adv, y, step, clamp)
        x_adv = torch.max(torch.min(x_adv, x0 + eps), x0 - eps)
        if clamp:
            x_adv = x_adv.clamp(*clamp)
    return x_adv.detach()


@torch.no_grad()
def _err(model, x, y, topk=(1, 5)):
    logits = model(x)
    pred = logits.topk(min(max(topk), logits.size(1)), 1).indices
    hit = pred.eq(y.view(-1, 1))
    return {k: (~hit[:, :k].any(1)).sum().item() for k in topk}


def evaluate_attack(model: nn.Module, loader, attack: Callable, device, source_model: Optional[nn.Module] = None):
    """Lỗi top-1/top-5 (%) của `model` trên ví dụ đối kháng.

    source_model=None  -> white-box (sinh ví dụ từ chính model).
    source_model=M     -> black-box (sinh từ M rồi đánh `model`), như bài báo.
    """
    model.eval()
    src = source_model.eval() if source_model is not None else model
    errs, n = {1: 0, 5: 0}, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        x_adv = attack(src, x, y)
        e = _err(model, x_adv, y)
        errs[1] += e[1]
        errs[5] += e[5]
        n += y.size(0)
    return {"top1_err": 100.0 * errs[1] / n, "top5_err": 100.0 * errs[5] / n}
