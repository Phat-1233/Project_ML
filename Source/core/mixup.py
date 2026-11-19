"""Phần lõi của mixup (Zhang et al., ICLR 2018), không phụ thuộc kiểu dữ liệu.

Công thức (Eq. 1 trong bài báo):
    x~ = λ x_i + (1 - λ) x_j
    y~ = λ y_i + (1 - λ) y_j,      λ ~ Beta(α, α),  α > 0

Mọi hàm ở đây chỉ thao tác trên chiều batch (dim 0) nên dùng được cho:
    - ảnh  (B, C, H, W)        -> 3.2 CIFAR, 3.4 nhãn nhiễu, 3.5 FGSM, 3.8 ablation
    - phổ âm thanh (B, 1, F, T) -> 3.3 Speech Commands
    - dữ liệu bảng (B, D)       -> 3.6 UCI
    - ảnh thật/giả của GAN      -> 3.7 (xem `mixup_real_fake`)
    - biểu diễn ẩn (feature map) -> ablation "Layer 1-5" (xem `LatentMixer`)

Chi tiết cài đặt lấy theo mã nguồn chính thức (facebookresearch/mixup-cifar10,
CC-BY-NC), những điểm bài báo KHÔNG nói rõ:
    * Một λ duy nhất cho cả minibatch (không phải mỗi mẫu một λ).
    * Cặp (i, j) tạo bằng cách hoán vị ngẫu nhiên chính minibatch đó, thay vì
      dùng hai data loader như Hình 1a của bài báo.
    * Loss viết dạng λ·CE(ŷ, y_i) + (1-λ)·CE(ŷ, y_j); với cross-entropy, cách này
      tương đương hoàn toàn với CE trên nhãn mềm y~.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------- #
# Lấy mẫu λ
# --------------------------------------------------------------------------- #


def sample_lambda(alpha: float, rng: np.random.Generator, size: Optional[int] = None):
    """λ ~ Beta(α, α). α <= 0 nghĩa là tắt mixup (λ = 1, quay về ERM).

    `rng` là np.random.Generator riêng để kết quả tái lập được và lưu/khôi phục
    được trạng thái khi resume (mã gốc dùng np.random toàn cục và không seed nó).
    """
    if alpha <= 0:
        return 1.0 if size is None else np.ones(size, dtype=np.float32)
    return rng.beta(alpha, alpha, size=size)


# --------------------------------------------------------------------------- #
# Chọn cặp (i, j)
# --------------------------------------------------------------------------- #


def random_pairing(y: torch.Tensor, generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """Hoán vị ngẫu nhiên trên toàn batch (mặc định, ký hiệu AC+RP trong ablation)."""
    return torch.randperm(y.size(0), generator=generator).to(y.device)


def same_class_pairing(y: torch.Tensor, generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """Chỉ trộn các mẫu CÙNG lớp (ablation SC+RP): hoán vị trong từng lớp."""
    y_cpu = y.detach().cpu()
    index = torch.arange(y_cpu.size(0))
    for c in torch.unique(y_cpu):
        pos = (y_cpu == c).nonzero(as_tuple=True)[0]
        index[pos] = pos[torch.randperm(pos.numel(), generator=generator)]
    return index.to(y.device)


PAIRINGS: dict[str, Callable] = {
    "random": random_pairing,
    "same_class": same_class_pairing,
}


# --------------------------------------------------------------------------- #
# Trộn một batch
# --------------------------------------------------------------------------- #


@dataclass
class MixedBatch:
    x: torch.Tensor          # input đã trộn
    y_a: torch.Tensor        # nhãn của x_i (chỉ số lớp, dtype long)
    y_b: torch.Tensor        # nhãn của x_j
    lam: torch.Tensor        # λ: tensor vô hướng, hoặc (B,) nếu per_sample
    index: torch.Tensor      # hoán vị đã dùng (để debug / trực quan hóa)


def _expand(lam: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Đưa λ dạng (B,) về (B,1,1,...) để broadcast với tensor bất kỳ."""
    if lam.dim() == 0:
        return lam
    return lam.view(-1, *([1] * (ref.dim() - 1)))


def mix_tensors(a: torch.Tensor, b: torch.Tensor, lam: torch.Tensor) -> torch.Tensor:
    lam = _expand(lam.to(a.dtype), a)
    return lam * a + (1 - lam) * b


def mixup_batch(
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float,
    rng: np.random.Generator,
    pairing: str = "random",
    per_sample: bool = False,
    index: Optional[torch.Tensor] = None,
    generator: Optional[torch.Generator] = None,
) -> MixedBatch:
    """Trộn một minibatch.

    Args:
        x, y: input và nhãn (chỉ số lớp).
        alpha: tham số α của Beta(α, α). α <= 0 -> ERM.
        pairing: "random" (mặc định của bài báo) hoặc "same_class".
        per_sample: True -> mỗi mẫu một λ (mở rộng, KHÔNG có trong bài báo).
        index: truyền sẵn chỉ số cặp (ví dụ KNN cho ablation) thì bỏ qua `pairing`.
    """
    if alpha <= 0:
        idx = torch.arange(x.size(0), device=x.device)
        one = torch.tensor(1.0, device=x.device)
        return MixedBatch(x, y, y, one, idx)

    if index is None:
        index = PAIRINGS[pairing](y, generator)
    if per_sample:
        lam = torch.as_tensor(sample_lambda(alpha, rng, size=x.size(0)), dtype=torch.float32, device=x.device)
    else:
        lam = torch.tensor(float(sample_lambda(alpha, rng)), device=x.device)

    mixed_x = mix_tensors(x, x[index], lam)
    return MixedBatch(mixed_x, y, y[index], lam, index)


# --------------------------------------------------------------------------- #
# Hàm mất mát
# --------------------------------------------------------------------------- #


def soft_targets(
    mb: MixedBatch,
    num_classes: int,
    label_mode: str = "mix",
    smoothing: float = 0.0,
) -> torch.Tensor:
    """Tạo nhãn mềm (B, K) cho batch đã trộn.

    label_mode:
        "mix"  -> y~ = λ y_i + (1-λ) y_j          (mixup chuẩn)
        "hard" -> one-hot của mẫu có trọng số lớn hơn (ablation "mix input only")
    smoothing: label smoothing kiểu Szegedy et al. (ablation): ε/K cho lớp sai.
    """
    ya = F.one_hot(mb.y_a, num_classes).float()
    yb = F.one_hot(mb.y_b, num_classes).float()
    lam = _expand(mb.lam.float(), ya)
    if label_mode == "mix":
        t = lam * ya + (1 - lam) * yb
    elif label_mode == "hard":
        t = torch.where(lam >= 0.5, ya, yb) if lam.dim() else (ya if float(lam) >= 0.5 else yb)
    else:
        raise ValueError(f"label_mode không hợp lệ: {label_mode}")
    if smoothing > 0:
        t = t * (1 - smoothing) + smoothing / num_classes
    return t


def soft_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """CE với nhãn mềm: -Σ_k t_k log softmax(z)_k, lấy trung bình theo batch."""
    return -(targets * F.log_softmax(logits.float(), dim=1)).sum(dim=1).mean()


def mixup_loss(
    logits: torch.Tensor,
    mb: MixedBatch,
    num_classes: Optional[int] = None,
    label_mode: str = "mix",
    smoothing: float = 0.0,
) -> torch.Tensor:
    """Loss mixup. Trường hợp chuẩn (mix, không smoothing) dùng đúng dạng của mã
    gốc: λ·CE(y_i) + (1-λ)·CE(y_j); các trường hợp khác dùng nhãn mềm."""
    if label_mode == "mix" and smoothing == 0 and mb.lam.dim() == 0:
        lam = mb.lam.float()
        return lam * F.cross_entropy(logits, mb.y_a) + (1 - lam) * F.cross_entropy(logits, mb.y_b)
    if num_classes is None:
        num_classes = logits.size(1)
    return soft_cross_entropy(logits, soft_targets(mb, num_classes, label_mode, smoothing))


def mixed_accuracy(logits: torch.Tensor, mb: MixedBatch) -> float:
    """Độ chính xác "có trọng số" trên batch đã trộn, như mã gốc:
    λ·[ŷ = y_i] + (1-λ)·[ŷ = y_j]. Chỉ để theo dõi, không phải độ đo đánh giá."""
    pred = logits.argmax(1)
    lam = mb.lam.float()
    hit = lam * (pred == mb.y_a).float() + (1 - lam) * (pred == mb.y_b).float()
    return hit.sum().item()


# --------------------------------------------------------------------------- #
# Trộn biểu diễn ẩn (ablation "Layer 1-5" trong mục 3.8)
# --------------------------------------------------------------------------- #


class LatentMixer:
    """Trộn ĐẦU VÀO của một module bất kỳ trong mạng bằng forward pre-hook.

    Dùng được cho mọi kiến trúc mà không phải sửa forward():
        mixer = LatentMixer(model.layer3)   # trộn feature map trước layer3
        mixer.arm(index, lam)               # đặt cặp + λ cho batch này
        logits = model(x)                   # x KHÔNG trộn; hook trộn bên trong
    Hook tự gỡ trạng thái sau một lần forward để không ảnh hưởng lúc eval.
    """

    def __init__(self, module: nn.Module):
        self._index = None
        self._lam = None
        self.handle = module.register_forward_pre_hook(self._hook)

    def arm(self, index: torch.Tensor, lam: torch.Tensor):
        self._index, self._lam = index, lam

    def _hook(self, module, inputs):
        if self._index is None:
            return None
        h = inputs[0]
        mixed = mix_tensors(h, h[self._index], self._lam)
        self._index = self._lam = None
        return (mixed,) + tuple(inputs[1:])

    def remove(self):
        self.handle.remove()


# --------------------------------------------------------------------------- #
# GAN (mục 3.7)
# --------------------------------------------------------------------------- #


def mixup_real_fake(real: torch.Tensor, fake: torch.Tensor, alpha: float, rng: np.random.Generator):
    """mixup cho discriminator của GAN (Eq. trong mục 3.7):
        x~ = λ x_real + (1-λ) G(z),  nhãn của D là λ (1 = thật, 0 = giả).
    Trả về (x~, target) với target có shape (B,) để dùng với BCE.
    """
    b = real.size(0)
    lam = torch.as_tensor(sample_lambda(alpha, rng, size=b), dtype=real.dtype, device=real.device)
    mixed = mix_tensors(real, fake, lam)
    return mixed, lam
