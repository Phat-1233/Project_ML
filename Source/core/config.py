"""Tham số dòng lệnh chung để mọi script thực nghiệm có cùng tên cờ.

Mỗi script gọi `add_common_args(parser)` rồi tự thêm cờ riêng của nó.
Quy ước nhóm: KHÔNG đổi giá trị mặc định ở đây; muốn khác thì truyền cờ.
"""
from __future__ import annotations

import argparse

from .trainer import MixupConfig


def add_common_args(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
    g = p.add_argument_group("chung")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--epochs", type=int, default=200)
    g.add_argument("--batch-size", type=int, default=128)
    g.add_argument("--lr", type=float, default=0.1)
    g.add_argument("--momentum", type=float, default=0.9)
    g.add_argument("--wd", type=float, default=1e-4, help="weight decay")
    g.add_argument("--milestones", type=int, nargs="*", default=[100, 150])
    g.add_argument("--gamma", type=float, default=0.1)
    g.add_argument("--official-lr-offset", action="store_true",
                   help="giảm lr trễ 1 epoch như mã gốc thay vì đúng như bài báo")
    g.add_argument("--workers", type=int, default=2)
    g.add_argument("--data-dir", default="./data")
    g.add_argument("--out-dir", default="./results")
    g.add_argument("--run-name", default=None, help="mặc định tự sinh từ cấu hình")
    g.add_argument("--device", default="auto")
    g.add_argument("--amp", action="store_true", help="mixed precision (nhanh hơn trên T4/P100)")
    g.add_argument("--deterministic", action="store_true")
    g.add_argument("--no-resume", action="store_true")
    g.add_argument("--wandb", action="store_true")
    g.add_argument("--wandb-project", default="mixup-repro")
    g.add_argument("--limit-batches", type=int, default=0,
                   help=">0: chỉ dùng chừng ấy batch mỗi epoch (để chạy thử nhanh)")

    m = p.add_argument_group("mixup")
    m.add_argument("--alpha", type=float, default=1.0, help="α của Beta(α,α); 0 = ERM")
    m.add_argument("--pairing", choices=["random", "same_class"], default="random")
    m.add_argument("--label-mode", choices=["mix", "hard"], default="mix")
    m.add_argument("--smoothing", type=float, default=0.0)
    m.add_argument("--per-sample-lambda", action="store_true")
    m.add_argument("--mix-warmup", type=int, default=0, help="số epoch đầu train ERM")
    m.add_argument("--input-noise", type=float, default=0.0)
    m.add_argument("--latent-module", default=None, help="vd layer1..layer4, linear")
    return p


def mixup_config_from_args(a: argparse.Namespace) -> MixupConfig:
    return MixupConfig(
        alpha=a.alpha,
        pairing=a.pairing,
        label_mode=a.label_mode,
        smoothing=a.smoothing,
        per_sample=a.per_sample_lambda,
        warmup_epochs=a.mix_warmup,
        input_noise_std=a.input_noise,
        latent_module=a.latent_module,
    )


def default_run_name(a: argparse.Namespace, prefix: str) -> str:
    method = "erm" if a.alpha <= 0 else f"mixup_a{a.alpha:g}"
    return f"{prefix}/{method}_seed{a.seed}"
