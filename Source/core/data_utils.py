"""Tiện ích dữ liệu không phụ thuộc kiểu dữ liệu."""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


def corrupt_labels(labels: Sequence[int], rate: float, num_classes: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Thay `rate` phần nhãn bằng nhãn ngẫu nhiên đều trên K lớp (mục 3.4).

    Theo cách của Zhang et al. (2017) mà bài báo dùng lại: nhãn mới được lấy
    ngẫu nhiên trên TẤT CẢ các lớp, nên có thể trùng nhãn cũ. Vì vậy tỉ lệ nhãn
    thực sự sai ≈ rate·(K-1)/K (ví dụ 50% -> ~45% với CIFAR-10).

    Trả về (nhãn_mới, mặt_nạ_bị_chọn). Nhãn test luôn giữ nguyên.
    """
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels).copy()
    mask = rng.random(len(labels)) < rate
    labels[mask] = rng.integers(0, num_classes, size=int(mask.sum()))
    return labels, mask


class RelabeledDataset(Dataset):
    """Bọc một dataset bất kỳ và thay nhãn bằng mảng `labels` cho trước.
    Dùng cho nhãn nhiễu (3.4) hoặc để đo lỗi train trên nhãn gốc/nhãn nhiễu."""

    def __init__(self, base: Dataset, labels: Sequence[int], subset: Optional[np.ndarray] = None):
        self.base = base
        self.labels = np.asarray(labels)
        self.subset = np.arange(len(base)) if subset is None else np.asarray(subset)

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, i):
        j = int(self.subset[i])
        x, _ = self.base[j]
        return x, int(self.labels[j])


class TensorClassificationDataset(Dataset):
    """Dataset từ mảng numpy (dùng cho dữ liệu bảng UCI, 3.6)."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.y[i]
