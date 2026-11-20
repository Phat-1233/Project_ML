"""Mục 3.8: ablation của mixup trên CIFAR-10 với PreAct ResNet-18 (Bảng 5 của bài báo).

Mỗi dòng của Bảng 5 là một "biến thể" (--variant). Biến thể chỉ đặt sẵn các cờ
mixup của core (--alpha, --pairing, --label-mode, --smoothing, --input-noise,
--latent-module) và thêm KNN. Cờ truyền tay vẫn ghi đè được giá trị của biến thể.
Mọi thiết lập khác giống 3.2 (bài báo: "All the other settings ... are the same as
reported in Section 3.2"), riêng weight decay chạy 2 mức 1e-4 và 5e-4.

Ví dụ (chạy từ thư mục Source):
    python experiments/exp38_ablation.py --list                       # xem các biến thể
    python experiments/exp38_ablation.py --variant mixup --wd 1e-4 --seed 0
    python experiments/exp38_ablation.py --variant input_ac_rp --wd 5e-4 --seed 1
    # rút gọn 100 epoch, dời mốc giảm lr theo tỉ lệ
    python experiments/exp38_ablation.py --variant erm --epochs 100 --milestones 50 75
    # chạy thử nhanh
    python experiments/exp38_ablation.py --variant input_sc_knn --epochs 1 --limit-batches 20

Lựa chọn cài đặt mà bài báo KHÔNG nêu (cần ghi trong báo cáo):
  * α = 1 cho mọi biến thể có trộn (giống 3.2; mục 3.8 không ghi α).
  * Layer 1-4 = trước layer1..layer4 (sau stem conv-BN-ReLU), Layer 5 = trước lớp linear
    (sau avg pool; trộn trước hay sau avg pool là như nhau vì avg pool tuyến tính).
  * "Mix inputs + label smoothing": nhãn = one-hot của mẫu có trọng số lớn hơn rồi làm mềm.
  * Nhiễu Gauss cộng SAU bước chuẩn hóa ảnh (σ tính trên thang đã chuẩn hóa).
  * KNN: khoảng cách Euclid trên ảnh gốc 32x32x3 đã chuẩn hóa (không augment), k = 200,
    loại chính mẫu đó. Cách đưa láng giềng vào batch: xem `KNNPairBatchSampler`.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cifar import build_cifar, preact_resnet18  # noqa: E402
from cifar.data import MEAN, STD  # noqa: E402
from cifar.models import LATENT_LAYERS  # noqa: E402
from core.config import add_common_args, mixup_config_from_args  # noqa: E402
from core.schedules import step_schedule  # noqa: E402
from core.trainer import Trainer  # noqa: E402
from core.utils import RunLogger, count_params, get_device, loader_generator, seed_worker, set_seed  # noqa: E402

# --------------------------------------------------------------------------- #
# Các biến thể = các dòng của Bảng 5
# --------------------------------------------------------------------------- #

_MIX = dict(alpha=1.0, pairing="random", label_mode="mix", smoothing=0.0, input_noise=0.0,
            latent_module=None, knn=None)
_ERM = {**_MIX, "alpha": 0.0}

VARIANTS: dict[str, dict] = {
    # nhóm A: mốc
    "erm": _ERM,
    # nhóm B: mixup (trộn cả input và nhãn)
    "mixup": _MIX,                                              # AC + RP
    "mixup_ac_knn": {**_MIX, "knn": "ac"},                      # AC + KNN
    # nhóm C: trộn nhãn + biểu diễn ẩn (AC + RP)
    **{f"latent_l{k}": {**_MIX, "latent_module": LATENT_LAYERS[k]} for k in range(1, 6)},
    # nhóm D: chỉ trộn input, nhãn = one-hot của mẫu gần hơn
    "input_sc_knn": {**_MIX, "label_mode": "hard", "knn": "sc"},     # SMOTE (Chawla et al.)
    "input_ac_knn": {**_MIX, "label_mode": "hard", "knn": "ac"},
    "input_sc_rp": {**_MIX, "label_mode": "hard", "pairing": "same_class"},
    "input_ac_rp": {**_MIX, "label_mode": "hard"},
    # nhóm E: label smoothing (không trộn)
    **{f"ls_{e:g}": {**_ERM, "smoothing": e} for e in (0.05, 0.1, 0.2)},
    # nhóm F: trộn input (AC + RP) + label smoothing
    **{f"input_ls_{e:g}": {**_MIX, "label_mode": "hard", "smoothing": e} for e in (0.05, 0.1, 0.2, 0.4)},
    # nhóm G: nhiễu Gauss vào input (không trộn)
    **{f"noise_{s:g}": {**_ERM, "input_noise": s} for s in (0.05, 0.1, 0.2)},
}

# --------------------------------------------------------------------------- #
# KNN
# --------------------------------------------------------------------------- #


def raw_images_labels(dataset) -> tuple[np.ndarray, np.ndarray]:
    """Ảnh gốc (N,32,32,3) uint8 và nhãn (N,) của tập train torchvision, không augment."""
    return np.asarray(dataset.data), np.asarray(dataset.targets)


@torch.no_grad()
def compute_knn(images: np.ndarray, labels: np.ndarray, k: int = 200, same_class: bool = False,
                device: Optional[torch.device] = None, chunk: int = 1024) -> np.ndarray:
    """k láng giềng gần nhất (khoảng cách Euclid trên ảnh đã chuẩn hóa) cho từng ảnh.

    same_class=True -> chỉ tìm trong cùng lớp (SC), ngược lại tìm trong mọi lớp (AC).
    Tính theo từng khối `chunk` ảnh để không phải giữ ma trận N x N trong bộ nhớ.
    Trả về mảng (N, k) int32, không chứa chính mẫu đó.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.from_numpy(images).float().div_(255).to(device)               # (N,H,W,C)
    x = (x - torch.tensor(MEAN, device=device)) / torch.tensor(STD, device=device)
    x = x.flatten(1)                                                         # (N, D)
    y = torch.from_numpy(labels).to(device)
    sq = (x * x).sum(1)
    n = x.size(0)
    out = np.empty((n, k), dtype=np.int32)
    for s in range(0, n, chunk):
        xb = x[s:s + chunk]
        d = sq[s:s + chunk, None] + sq[None, :] - 2.0 * xb @ x.T            # khoảng cách bình phương
        rows = torch.arange(xb.size(0), device=device)
        d[rows, rows + s] = float("inf")                                     # bỏ chính nó
        if same_class:
            d[y[s:s + chunk, None] != y[None, :]] = float("inf")
        out[s:s + xb.size(0)] = d.topk(k, dim=1, largest=False).indices.cpu().numpy()
    return out


def load_or_compute_knn(cache_dir: str, images: np.ndarray, labels: np.ndarray, k: int,
                        same_class: bool) -> np.ndarray:
    path = Path(cache_dir) / f"knn_{'sc' if same_class else 'ac'}_k{k}.npy"
    if path.exists():
        knn = np.load(path)
        if knn.shape == (len(images), k):
            print(f"[knn] dùng lại {path}", flush=True)
            return knn
    print(f"[knn] đang tính {path.name} ...", flush=True)
    knn = compute_knn(images, labels, k, same_class)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, knn)
    return knn


class KNNPairBatchSampler(Sampler):
    """Đưa láng giềng KNN vào cùng minibatch để dùng `pair_index_fn` của core mà không sửa core.

    Mỗi batch có B/2 mẫu "neo" (lấy ngẫu nhiên, không lặp trong một epoch) và B/2 mẫu
    "láng giềng": láng giềng của neo i = một phần tử rút ngẫu nhiên trong knn[i].
    `pair_index` ghép neo i <-> láng giềng i, nên mỗi cặp sinh 2 mẫu trộn với hệ số λ và 1-λ.
    Vì Beta(α, α) đối xứng, cả hai đều là "neo trộn với láng giềng của nó".

    Khác bài báo (cần ghi trong báo cáo): bài báo thay MỖI mẫu trong batch bằng một mẫu
    trộn với láng giềng của chính nó (B cặp/batch); ở đây là B/2 cặp/batch, mỗi cặp dùng
    2 lần. Số bước mỗi epoch, kích thước batch và mọi phần còn lại giữ nguyên như 3.2.
    """

    def __init__(self, knn: np.ndarray, batch_size: int, seed: int = 0, drop_last: bool = False):
        assert batch_size % 2 == 0, "batch size phải chẵn"
        self.knn = knn
        self.half = batch_size // 2
        self.n = len(knn)
        self.steps = (self.n + batch_size - 1) // batch_size if not drop_last else self.n // batch_size
        self.gen = loader_generator(seed)

    def __len__(self) -> int:
        return self.steps

    def __iter__(self) -> Iterator[list[int]]:
        order = torch.randperm(self.n, generator=self.gen)
        pick = torch.randint(self.knn.shape[1], (self.n,), generator=self.gen)
        for b in range(self.steps):
            a = order[(b * self.half) % self.n:][:self.half]
            if len(a) < self.half:                                  # cuối hoán vị thì nối vòng
                a = torch.cat([a, order[:self.half - len(a)]])
            partners = self.knn[a.numpy(), pick[a].numpy()]
            yield a.tolist() + partners.tolist()


def knn_pair_index(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """pair_index_fn cho batch do KNNPairBatchSampler tạo: nửa đầu <-> nửa sau."""
    h = y.size(0) // 2
    idx = torch.cat([torch.arange(h, 2 * h), torch.arange(0, h)])
    return idx.to(y.device)


def knn_train_loader(train_loader: DataLoader, knn: np.ndarray, batch_size: int, seed: int) -> DataLoader:
    w = train_loader.num_workers
    return DataLoader(train_loader.dataset, batch_sampler=KNNPairBatchSampler(knn, batch_size, seed),
                      num_workers=w, pin_memory=train_loader.pin_memory, worker_init_fn=seed_worker,
                      persistent_workers=w > 0)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(p)
    g = p.add_argument_group("3.8 ablation")
    g.add_argument("--variant", choices=sorted(VARIANTS), default=None,
                   help="dòng của Bảng 5; đặt sẵn các cờ mixup (cờ truyền tay vẫn ghi đè)")
    g.add_argument("--knn", choices=["ac", "sc"], default=None, help="ghép cặp bằng KNN trong mọi lớp/cùng lớp")
    g.add_argument("--knn-k", type=int, default=200)
    g.add_argument("--knn-cache", default=None, help="thư mục lưu knn_*.npy (mặc định: --data-dir)")
    g.add_argument("--list", action="store_true", help="in danh sách biến thể rồi thoát")
    return p


def parse_args(argv=None) -> argparse.Namespace:
    p = build_parser()
    first, _ = p.parse_known_args(argv)
    if first.variant:                     # biến thể làm giá trị mặc định, cờ truyền tay ghi đè
        p.set_defaults(**VARIANTS[first.variant])
    return p.parse_args(argv)


def run_name(a: argparse.Namespace) -> str:
    if a.run_name:
        return a.run_name
    v = a.variant or "custom"
    return f"exp38_ablation/{v}_wd{a.wd:g}_seed{a.seed}"


def main(argv=None):
    a = parse_args(argv)
    if a.list:
        for name, cfg in VARIANTS.items():
            diff = {k: v for k, v in cfg.items() if _MIX.get(k) != v}
            print(f"{name:16s} {diff}")
        return None
    if a.knn and a.latent_module:
        raise SystemExit("KNN + trộn ẩn không có trong Bảng 5")

    rng = set_seed(a.seed, a.deterministic)
    device = get_device(a.device)
    data = build_cifar("cifar10", a.data_dir, a.batch_size, workers=a.workers, seed=a.seed)
    train_loader, pair_fn = data.train, None
    if a.knn and a.alpha > 0:
        imgs, labels = raw_images_labels(data.train.dataset)
        knn = load_or_compute_knn(a.knn_cache or a.data_dir, imgs, labels, a.knn_k, a.knn == "sc")
        train_loader, pair_fn = knn_train_loader(data.train, knn, a.batch_size, a.seed), knn_pair_index

    model = preact_resnet18(data.num_classes)
    opt = torch.optim.SGD(model.parameters(), lr=a.lr, momentum=a.momentum, weight_decay=a.wd)
    sched = step_schedule(opt, a.milestones, a.gamma, official_offset=a.official_lr_offset)

    name = run_name(a)
    log = RunLogger(a.out_dir, name, vars(a), a.wandb, a.wandb_project, resume=not a.no_resume)
    mix = mixup_config_from_args(a)
    print(f"[exp38] {name} | device={device} | params={count_params(model):,} | {mix} | knn={a.knn}",
          flush=True)

    tr = Trainer(model, opt, sched, train_loader, {"test": data.test}, data.num_classes, mix, log, rng,
                 device, a.epochs, amp=a.amp, pair_index_fn=pair_fn, limit_batches=a.limit_batches)
    summary = tr.fit(resume=not a.no_resume)   # đã ghi summary.json; biến thể/wd/seed nằm trong config.json
    print(json.dumps({"variant": a.variant, "wd": a.wd, "seed": a.seed, **summary}, indent=2))
    return summary


if __name__ == "__main__":
    main()
