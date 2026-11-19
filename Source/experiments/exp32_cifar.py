"""Mục 3.2: so sánh ERM và mixup trên CIFAR-10/100 với PreAct ResNet-18.

Thiết lập mặc định đúng như bài báo: 200 epoch, batch 128, SGD lr 0.1
(momentum 0.9), chia 10 sau epoch 100 và 150, weight decay 1e-4, α = 1.

Ví dụ (chạy từ thư mục Source):
    python experiments/exp32_cifar.py --alpha 1 --seed 0            # mixup
    python experiments/exp32_cifar.py --alpha 0 --seed 0            # ERM
    python experiments/exp32_cifar.py --dataset cifar100 --alpha 1
    # chạy thử nhanh: 2 epoch, mỗi epoch 50 batch
    python experiments/exp32_cifar.py --epochs 2 --limit-batches 50 --run-name smoke
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cifar import build_cifar, preact_resnet18  # noqa: E402
from core.config import add_common_args, default_run_name, mixup_config_from_args  # noqa: E402
from core.schedules import step_schedule  # noqa: E402
from core.trainer import Trainer  # noqa: E402
from core.utils import RunLogger, count_params, get_device, set_seed  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(p)
    p.add_argument("--dataset", choices=["cifar10", "cifar100"], default="cifar10")
    p.add_argument("--no-augment", action="store_true")
    a = p.parse_args(argv)

    rng = set_seed(a.seed, a.deterministic)
    device = get_device(a.device)
    data = build_cifar(a.dataset, a.data_dir, a.batch_size, workers=a.workers, seed=a.seed,
                       augment=not a.no_augment)
    model = preact_resnet18(data.num_classes)
    opt = torch.optim.SGD(model.parameters(), lr=a.lr, momentum=a.momentum, weight_decay=a.wd)
    sched = step_schedule(opt, a.milestones, a.gamma, official_offset=a.official_lr_offset)

    name = a.run_name or default_run_name(a, f"exp32_{a.dataset}")
    log = RunLogger(a.out_dir, name, vars(a), a.wandb, a.wandb_project, resume=not a.no_resume)
    print(f"[exp32] {name} | device={device} | params={count_params(model):,}", flush=True)

    tr = Trainer(model, opt, sched, data.train, {"test": data.test}, data.num_classes,
                 mixup_config_from_args(a), log, rng, device, a.epochs, amp=a.amp,
                 limit_batches=a.limit_batches)
    summary = tr.fit(resume=not a.no_resume)
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    main()
