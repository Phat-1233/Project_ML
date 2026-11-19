"""Tiện ích dùng chung: seed, thiết bị, log, checkpoint, thông tin môi trường."""
from __future__ import annotations

import csv
import json
import os
import platform
import random
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

# --------------------------------------------------------------------------- #
# Seed và tính tái lập
# --------------------------------------------------------------------------- #


def set_seed(seed: int, deterministic: bool = False) -> np.random.Generator:
    """Cố định mọi nguồn ngẫu nhiên. Trả về np.random.Generator riêng dùng để lấy λ.

    deterministic=True bật thuật toán cuDNN tất định: kết quả lặp lại y hệt giữa
    các lần chạy trên cùng phần cứng nhưng chậm hơn (khoảng 10-30%).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.backends.cudnn.benchmark = True
    return np.random.default_rng(seed)


def seed_worker(worker_id: int):
    """worker_init_fn cho DataLoader để augmentation trong các worker tái lập được."""
    s = torch.initial_seed() % 2**32
    np.random.seed(s)
    random.seed(s)


def loader_generator(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def get_device(prefer: str = "auto") -> torch.device:
    if prefer != "auto":
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# --------------------------------------------------------------------------- #
# Thông tin môi trường (đề yêu cầu ghi phần cứng + phiên bản thư viện)
# --------------------------------------------------------------------------- #


def env_info() -> dict[str, Any]:
    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    try:
        import torchvision

        info["torchvision"] = torchvision.__version__
    except ImportError:
        pass
    if torch.cuda.is_available():
        info["cuda"] = torch.version.cuda
        info["cudnn"] = torch.backends.cudnn.version()
        info["gpu"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    return info


# --------------------------------------------------------------------------- #
# Ghi log: CSV theo epoch + JSON tổng kết + (tùy chọn) Weights & Biases
# --------------------------------------------------------------------------- #


class RunLogger:
    """Mỗi lần chạy có một thư mục riêng:
        <out_dir>/<run_name>/config.json    cấu hình + môi trường
        <out_dir>/<run_name>/history.csv    mỗi epoch một dòng
        <out_dir>/<run_name>/summary.json   kết quả cuối
        <out_dir>/<run_name>/last.pt        checkpoint để resume
    """

    def __init__(self, out_dir: str, run_name: str, config: dict, use_wandb: bool = False,
                 wandb_project: str = "mixup-repro", resume: bool = False):
        self.dir = Path(out_dir) / run_name
        self.dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.dir / "history.csv"
        if not resume and self.csv_path.exists():
            self.csv_path.unlink()
        self._fields: Optional[list[str]] = None
        if self.csv_path.exists():
            with open(self.csv_path) as f:
                self._fields = next(csv.reader(f), None)

        payload = {"config": config, "env": env_info()}
        with open(self.dir / "config.json", "w") as f:
            json.dump(payload, f, indent=2, default=str)

        self.wandb = None
        if use_wandb:
            try:
                import wandb

                self.wandb = wandb.init(project=wandb_project, name=run_name, config=config,
                                        dir=str(self.dir), resume="allow", id=run_name.replace("/", "_"))
            except Exception as e:  # không có wandb hoặc chưa đăng nhập: vẫn chạy tiếp
                print(f"[warn] Không bật được W&B: {e}")

    def log_epoch(self, row: dict[str, Any]):
        if self._fields is None:
            self._fields = list(row.keys())
            with open(self.csv_path, "w", newline="") as f:
                csv.writer(f).writerow(self._fields)
        with open(self.csv_path, "a", newline="") as f:
            csv.writer(f).writerow([row.get(k, "") for k in self._fields])
        if self.wandb is not None:
            self.wandb.log(row, step=int(row.get("epoch", 0)))

    def save_summary(self, summary: dict[str, Any]):
        with open(self.dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=float)
        if self.wandb is not None:
            self.wandb.summary.update(summary)
            self.wandb.finish()


# --------------------------------------------------------------------------- #
# Checkpoint (quan trọng với Kaggle/Colab vì phiên bị ngắt sau 9-12 giờ)
# --------------------------------------------------------------------------- #


def save_checkpoint(path: Path, **state):
    state["rng"] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    tmp = Path(str(path) + ".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)  # ghi nguyên tử: không hỏng file nếu phiên bị ngắt giữa chừng


def load_checkpoint(path: Path) -> dict:
    """Luôn nạp lên CPU. Trạng thái RNG bắt buộc là ByteTensor trên CPU, còn
    model.load_state_dict / optimizer.load_state_dict tự chép tensor sang đúng
    thiết bị của tham số, nên không cần map_location=cuda."""
    state = torch.load(path, map_location="cpu", weights_only=False)
    rng = state.get("rng")
    if rng:
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        torch.set_rng_state(rng["torch"].cpu())
        if rng.get("cuda") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([t.cpu() for t in rng["cuda"]])
    return state


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
