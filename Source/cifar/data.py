"""Dữ liệu CIFAR-10/100 dùng chung cho 3.2, 3.4, 3.5, 3.8.

Tiền xử lý giống mã chính thức: RandomCrop(32, pad=4) + lật ngang, chuẩn hóa với
mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010).
Lưu ý: bộ std này là giá trị phổ biến từ kuangliu/pytorch-cifar, hơi khác std
thật của CIFAR-10 (~0.247, 0.243, 0.261). Ta giữ nguyên để khớp mã gốc.
Mã gốc dùng cùng hằng số này cho CIFAR-100; ta cũng làm vậy.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as T
from torch.utils.data import DataLoader

from core.data_utils import RelabeledDataset, corrupt_labels
from core.utils import loader_generator, seed_worker

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2023, 0.1994, 0.2010)
NUM_CLASSES = {"cifar10": 10, "cifar100": 100}


def transforms(augment: bool = True, normalize: bool = True):
    norm = [T.Normalize(MEAN, STD)] if normalize else []
    train = ([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip()] if augment else []) + [T.ToTensor()] + norm
    test = [T.ToTensor()] + norm
    return T.Compose(train), T.Compose(test)


@dataclass
class CifarData:
    train: DataLoader
    test: DataLoader
    num_classes: int
    extra_eval: dict            # loader phụ để đánh giá (vd train_clean/train_noisy cho 3.4)
    noisy_mask: Optional[np.ndarray] = None


def build_cifar(
    name: str = "cifar10",
    data_dir: str = "./data",
    batch_size: int = 128,
    test_batch_size: int = 100,
    workers: int = 2,
    seed: int = 0,
    augment: bool = True,
    normalize: bool = True,
    noise_rate: float = 0.0,
    noise_seed: Optional[int] = None,
    eval_train_subsets: bool = False,
    download: bool = True,
) -> CifarData:
    """Tạo DataLoader CIFAR.

    noise_rate > 0: thay nhãn train ngẫu nhiên (3.4). noise_seed cố định tập nhãn
        nhiễu, nên để giống nhau giữa ERM/mixup/dropout để so sánh công bằng.
    eval_train_subsets: thêm loader đánh giá lỗi train (không augment) trên
        nhãn thật ("train_real") và trên phần bị nhiễu với nhãn nhiễu ("train_corrupted"),
        như Bảng 2 của bài báo.
    """
    cls = {"cifar10": datasets.CIFAR10, "cifar100": datasets.CIFAR100}[name]
    k = NUM_CLASSES[name]
    tf_train, tf_test = transforms(augment, normalize)
    train_set = cls(data_dir, train=True, download=download, transform=tf_train)
    test_set = cls(data_dir, train=False, download=download, transform=tf_test)

    mask = None
    extra = {}
    if noise_rate > 0:
        clean = np.asarray(train_set.targets)
        noisy, mask = corrupt_labels(clean, noise_rate, k, seed if noise_seed is None else noise_seed)
        train_set.targets = noisy.tolist()
        if eval_train_subsets:
            plain = cls(data_dir, train=True, download=False, transform=tf_test)
            extra["train_real"] = RelabeledDataset(plain, clean)
            extra["train_corrupted"] = RelabeledDataset(plain, noisy, subset=np.where(mask)[0])

    common = dict(num_workers=workers, pin_memory=True, worker_init_fn=seed_worker,
                  persistent_workers=workers > 0)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=False,
                              generator=loader_generator(seed), **common)
    test_loader = DataLoader(test_set, batch_size=test_batch_size, shuffle=False, **common)
    extra_loaders = {n: DataLoader(d, batch_size=test_batch_size, shuffle=False, **common) for n, d in extra.items()}
    return CifarData(train_loader, test_loader, k, extra_loaders, mask)
