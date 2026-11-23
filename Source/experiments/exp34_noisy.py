"""Mục 3.4: học thuộc nhãn nhiễu trên CIFAR-10 với PreAct ResNet-18 (Bảng 2 của bài báo).

So sánh ERM, ERM + dropout, mixup, mixup + dropout khi 20/50/80% nhãn train bị thay
ngẫu nhiên. Mọi thiết lập khác giống 3.2 (200 epoch, lr 0.1 chia 10 sau epoch 100, 150).

Ví dụ (chạy từ thư mục Source):
    python experiments/exp34_noisy.py --noise-rate 0.5 --alpha 0                  # ERM
    python experiments/exp34_noisy.py --noise-rate 0.5 --alpha 0 --dropout 0.8    # ERM + dropout
    python experiments/exp34_noisy.py --noise-rate 0.5 --alpha 32                 # mixup
    python experiments/exp34_noisy.py --noise-rate 0.5 --alpha 8 --dropout 0.3    # mixup + dropout
    # chạy thử nhanh
    python experiments/exp34_noisy.py --noise-rate 0.5 --alpha 8 --dropout 0.3 --epochs 1 --limit-batches 20

Kết quả: results/exp34_noise50/<method>_seed<s>/summary.json, ngoài best/last test error
còn có lỗi train ở epoch cuối trên nhãn thật (`train_real_err`) và trên phần bị làm nhiễu
với nhãn nhiễu (`train_corrupted_err`), như hai cột "Training error" của Bảng 2.

Lựa chọn cài đặt (cần ghi trong báo cáo):
  * Nhãn nhiễu rút đều trên cả 10 lớp nên có thể trùng nhãn cũ: tỉ lệ nhãn sai thật
    ≈ rate·9/10. "Corrupted" tính trên mọi mẫu bị chọn, kể cả mẫu rút trúng nhãn cũ,
    nên lỗi corrupted tối đa ≈ 90%.
  * Mặc định noise_seed = seed: ERM/mixup/dropout cùng seed dùng chung một bộ nhãn nhiễu.
  * Lỗi train đo SAU khi train xong (một lần, không augment, model ở epoch cuối)
    thay vì mỗi epoch, để không tốn gấp đôi thời gian.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cifar import build_cifar, preact_resnet18  # noqa: E402
from core.config import add_common_args, mixup_config_from_args  # noqa: E402
from core.schedules import step_schedule  # noqa: E402
from core.trainer import Trainer  # noqa: E402
from core.utils import RunLogger, count_params, get_device, set_seed  # noqa: E402


def method_name(alpha: float, dropout: float) -> str:
    """erm | dropout_p0.7 | mixup_a8 | mixup_a8_p0.3"""
    parts = (["erm"] if alpha <= 0 else ["mixup", f"a{alpha:g}"]) + ([f"p{dropout:g}"] if dropout > 0 else [])
    if parts[0] == "erm" and dropout > 0:
        parts[0] = "dropout"
    return "_".join(parts)


def run_name(a: argparse.Namespace) -> str:
    ns = "" if a.noise_seed is None or a.noise_seed == a.seed else f"_ns{a.noise_seed}"
    return f"exp34_noise{round(a.noise_rate * 100)}/{method_name(a.alpha, a.dropout)}_seed{a.seed}{ns}"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(p)
    p.add_argument("--noise-rate", type=float, required=True, help="tỉ lệ nhãn train bị thay, vd 0.2 0.5 0.8")
    p.add_argument("--noise-seed", type=int, default=None, help="mặc định = --seed")
    p.add_argument("--dropout", type=float, default=0.0, help="xác suất dropout trong mỗi PreAct block")
    a = p.parse_args(argv)
    if not 0 < a.noise_rate <= 1:
        p.error("--noise-rate phải trong (0, 1]")

    rng = set_seed(a.seed, a.deterministic)
    device = get_device(a.device)
    data = build_cifar("cifar10", a.data_dir, a.batch_size, workers=a.workers, seed=a.seed,
                       noise_rate=a.noise_rate, noise_seed=a.noise_seed, eval_train_subsets=True)
    model = preact_resnet18(data.num_classes, dropout=a.dropout)
    opt = torch.optim.SGD(model.parameters(), lr=a.lr, momentum=a.momentum, weight_decay=a.wd)
    sched = step_schedule(opt, a.milestones, a.gamma, official_offset=a.official_lr_offset)

    name = a.run_name or run_name(a)
    log = RunLogger(a.out_dir, name, vars(a), a.wandb, a.wandb_project, resume=not a.no_resume)
    print(f"[exp34] {name} | device={device} | params={count_params(model):,}", flush=True)

    # Chỉ đánh giá test mỗi epoch; lỗi train trên nhãn thật/nhiễu đo một lần ở cuối.
    tr = Trainer(model, opt, sched, data.train, {"test": data.test}, data.num_classes,
                 mixup_config_from_args(a), log, rng, device, a.epochs, amp=a.amp,
                 limit_batches=a.limit_batches)
    summary = tr.fit(resume=not a.no_resume)

    for split, loader in data.extra_eval.items():          # train_real, train_corrupted
        for k, v in tr.evaluate(loader).items():
            summary[f"{split}_{k}"] = v
    clean = data.extra_eval["train_real"].dataset.labels
    noisy = np.asarray(data.train.dataset.targets)
    summary["noise_frac_selected"] = float(data.noisy_mask.mean())
    summary["noise_frac_wrong"] = float((clean != noisy).mean())
    with open(log.dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    main()
