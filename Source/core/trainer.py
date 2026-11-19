"""Vòng lặp huấn luyện/đánh giá dùng chung cho mọi thực nghiệm PHÂN LOẠI
(3.2 CIFAR, 3.3 Speech, 3.4 nhãn nhiễu, 3.6 UCI, 3.8 ablation).
3.7 GAN có vòng lặp riêng nhưng vẫn dùng `core.mixup.mixup_real_fake`.

Trainer không biết gì về dữ liệu: chỉ cần DataLoader trả về (x, y) với y là chỉ
số lớp. Mọi khác biệt giữa các thực nghiệm được đưa vào qua `MixupConfig` và các
tham số khởi tạo, không ai cần sửa file này.
"""
from __future__ import annotations

import math
import statistics
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .mixup import LatentMixer, MixedBatch, mixed_accuracy, mixup_batch, mixup_loss
from .utils import RunLogger, load_checkpoint, save_checkpoint


@dataclass
class MixupConfig:
    alpha: float = 1.0               # α = 0 -> ERM
    pairing: str = "random"          # "random" | "same_class"
    label_mode: str = "mix"          # "mix" | "hard" (ablation: chỉ trộn input)
    smoothing: float = 0.0           # label smoothing ε (ablation)
    per_sample: bool = False         # mỗi mẫu một λ (mở rộng, không có trong bài báo)
    warmup_epochs: int = 0           # số epoch đầu train ERM (speech dùng 5)
    input_noise_std: float = 0.0     # ablation: cộng nhiễu Gaussian vào input
    latent_module: Optional[str] = None  # tên module để trộn biểu diễn ẩn, vd "layer3"

    @property
    def enabled(self) -> bool:
        return self.alpha > 0


@dataclass
class TrainState:
    epoch: int = 0
    history: list = field(default_factory=list)
    best_test_err: float = math.inf


def accuracy_topk(logits: torch.Tensor, y: torch.Tensor, ks=(1, 5)) -> dict[int, int]:
    maxk = min(max(ks), logits.size(1))
    pred = logits.topk(maxk, dim=1).indices
    hit = pred.eq(y.view(-1, 1))
    return {k: hit[:, :min(k, maxk)].any(dim=1).sum().item() for k in ks}


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler,
        train_loader,
        eval_loaders: dict,
        num_classes: int,
        mix: MixupConfig,
        logger: RunLogger,
        rng: np.random.Generator,
        device: torch.device,
        epochs: int,
        amp: bool = False,
        grad_clip: Optional[float] = None,
        pair_index_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        primary_eval: str = "test",
        limit_batches: int = 0,
    ):
        """
        eval_loaders: {"test": loader, ...}. Có thể thêm "train_clean"/"train_noisy"
            cho 3.4 để đo lỗi huấn luyện trên nhãn thật và nhãn bị nhiễu.
        pair_index_fn(x, y) -> index: chọn cặp tùy biến (vd KNN cho ablation).
        primary_eval: loader dùng để tính best/last/median error.
        """
        self.model = model.to(device)
        self.opt = optimizer
        self.sched = scheduler
        self.train_loader = train_loader
        self.eval_loaders = eval_loaders
        self.num_classes = num_classes
        self.mix = mix
        self.log = logger
        self.rng = rng
        self.device = device
        self.epochs = epochs
        self.use_amp = amp and device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.grad_clip = grad_clip
        self.pair_index_fn = pair_index_fn
        self.primary_eval = primary_eval
        self.limit_batches = limit_batches  # >0: chỉ chạy thử vài batch
        self.state = TrainState()
        self.ckpt_path = self.log.dir / "last.pt"

        self.latent = None
        if mix.latent_module:
            self.latent = LatentMixer(dict(model.named_modules())[mix.latent_module])

    # ------------------------------------------------------------------ #
    def _autocast(self):
        return torch.autocast("cuda", dtype=torch.float16) if self.use_amp else nullcontext()

    def train_one_epoch(self, epoch: int) -> dict:
        self.model.train()
        use_mix = self.mix.enabled and epoch >= self.mix.warmup_epochs
        tot_loss, tot_hit, n = 0.0, 0.0, 0
        t0 = time.time()
        for bi, (x, y) in enumerate(self.train_loader):
            if self.limit_batches and bi >= self.limit_batches:
                break
            x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
            if self.mix.input_noise_std > 0:
                x = x + self.mix.input_noise_std * torch.randn_like(x)

            if use_mix:
                index = self.pair_index_fn(x, y) if self.pair_index_fn else None
                if self.latent is not None:
                    # Trộn biểu diễn ẩn: input giữ nguyên, hook trộn bên trong mạng
                    mb = mixup_batch(x, y, self.mix.alpha, self.rng, self.mix.pairing,
                                     self.mix.per_sample, index)
                    self.latent.arm(mb.index, mb.lam)
                    mb = MixedBatch(x, mb.y_a, mb.y_b, mb.lam, mb.index)
                else:
                    mb = mixup_batch(x, y, self.mix.alpha, self.rng, self.mix.pairing,
                                     self.mix.per_sample, index)
            else:
                mb = MixedBatch(x, y, y, torch.tensor(1.0, device=self.device),
                                torch.arange(x.size(0), device=self.device))

            with self._autocast():
                logits = self.model(mb.x)
            if use_mix or self.mix.smoothing > 0:
                loss = mixup_loss(logits, mb, self.num_classes, self.mix.label_mode, self.mix.smoothing)
            else:
                loss = F.cross_entropy(logits.float(), y)

            self.opt.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            if self.grad_clip:
                self.scaler.unscale_(self.opt)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.scaler.step(self.opt)
            self.scaler.update()

            bs = y.size(0)
            tot_loss += loss.item() * bs
            tot_hit += mixed_accuracy(logits.detach(), mb)
            n += bs
        return {
            "train_loss": tot_loss / n,
            "train_mixed_acc": 100.0 * tot_hit / n,
            "lr": self.opt.param_groups[0]["lr"],
            "epoch_time_s": time.time() - t0,
        }

    @torch.no_grad()
    def evaluate(self, loader) -> dict:
        self.model.eval()
        tot_loss, hits1, hits5, n = 0.0, 0, 0, 0
        for bi, (x, y) in enumerate(loader):
            if self.limit_batches and bi >= self.limit_batches:
                break
            x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
            with self._autocast():
                logits = self.model(x)
            tot_loss += F.cross_entropy(logits.float(), y, reduction="sum").item()
            h = accuracy_topk(logits, y)
            hits1 += h[1]
            hits5 += h[5]
            n += y.size(0)
        return {"loss": tot_loss / n, "err": 100.0 * (1 - hits1 / n), "err_top5": 100.0 * (1 - hits5 / n)}

    # ------------------------------------------------------------------ #
    def _save(self):
        save_checkpoint(
            self.ckpt_path,
            model=self.model.state_dict(),
            optimizer=self.opt.state_dict(),
            scheduler=self.sched.state_dict() if self.sched else None,
            scaler=self.scaler.state_dict(),
            np_rng=self.rng.bit_generator.state,
            state=asdict(self.state),
        )

    def try_resume(self) -> bool:
        if not self.ckpt_path.exists():
            return False
        ck = load_checkpoint(self.ckpt_path)
        self.model.load_state_dict(ck["model"])
        self.opt.load_state_dict(ck["optimizer"])
        if self.sched and ck.get("scheduler"):
            self.sched.load_state_dict(ck["scheduler"])
        self.scaler.load_state_dict(ck["scaler"])
        self.rng.bit_generator.state = ck["np_rng"]
        self.state = TrainState(**ck["state"])
        print(f"[resume] tiếp tục từ epoch {self.state.epoch}")
        return True

    def fit(self, resume: bool = True, save_best: bool = True) -> dict:
        if resume:
            self.try_resume()
        for epoch in range(self.state.epoch, self.epochs):
            row = {"epoch": epoch}
            row.update(self.train_one_epoch(epoch))
            for name, loader in self.eval_loaders.items():
                for k, v in self.evaluate(loader).items():
                    row[f"{name}_{k}"] = v
            if self.sched:
                self.sched.step()

            err = row[f"{self.primary_eval}_err"]
            if err < self.state.best_test_err:
                self.state.best_test_err = err
                if save_best:
                    torch.save(self.model.state_dict(), self.log.dir / "best_model.pt")
            self.state.history.append(row)
            self.state.epoch = epoch + 1
            self.log.log_epoch(row)
            self._save()
            print(" | ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()),
                  flush=True)

        torch.save(self.model.state_dict(), self.log.dir / "final_model.pt")
        summary = self.summary()
        self.log.save_summary(summary)
        if self.latent is not None:
            self.latent.remove()
        return summary

    def summary(self) -> dict:
        """Ba cách báo cáo lỗi mà bài báo dùng:
        best (3.4), last (3.4), median 10 epoch cuối (3.8 ablation)."""
        key = f"{self.primary_eval}_err"
        errs = [r[key] for r in self.state.history]
        out = {
            "best_err": min(errs),
            "last_err": errs[-1],
            "median_last10_err": statistics.median(errs[-10:]),
            "epochs_run": len(errs),
            "mean_epoch_time_s": statistics.mean(r["epoch_time_s"] for r in self.state.history),
        }
        for k, v in self.state.history[-1].items():
            if k.endswith("_err") or k.endswith("_err_top5"):
                out[f"last_{k}"] = v
        return out
