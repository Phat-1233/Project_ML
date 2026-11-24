"""Tái lập mục 3.2 / Figure 3 của mixup (arXiv:1710.09412v2).

Chạy từ Source, xem experiments/README_exp32.md để biết đầy đủ protocol:
    python experiments/exp32_cifar.py --dataset cifar10 --model preact_resnet18 --alpha 1 --seed 0
    python experiments/exp32_cifar.py --suite --seeds 0 1 2 --dry-run
    python experiments/exp32_cifar.py --suite --seeds 0 1 2 --device cuda
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cifar import MODEL_NAMES, build_cifar, build_model  # noqa: E402
from core.config import add_common_args, default_run_name, mixup_config_from_args  # noqa: E402
from core.schedules import step_schedule  # noqa: E402
from core.trainer import Trainer  # noqa: E402
from core.utils import RunLogger, count_params, get_device, set_seed  # noqa: E402

DATASETS = ("cifar10", "cifar100")
PAPER_URL = "https://arxiv.org/html/1710.09412v2#S3.SS2"
SINGLE_ONLY = {"suite", "datasets", "models", "methods", "seeds", "dry_run"}
RESUME_KEYS = (
    "dataset", "model", "seed", "batch_size", "test_batch_size", "lr", "momentum", "wd",
    "milestones", "gamma", "official_lr_offset", "alpha", "no_augment", "amp", "deterministic",
    "limit_batches", "pairing", "label_mode", "smoothing", "per_sample_lambda", "mix_warmup",
    "input_noise", "latent_module",
)


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(p)
    p.add_argument("--dataset", choices=DATASETS, default="cifar10")
    p.add_argument("--model", choices=MODEL_NAMES, default="preact_resnet18")
    p.add_argument("--no-augment", action="store_true")
    p.add_argument("--no-download", action="store_true", help="chỉ dùng dữ liệu đã tải")
    p.add_argument("--test-batch-size", type=int, default=100)
    p.add_argument("--dry-run", action="store_true", help="in cấu hình, không tải dữ liệu hoặc train")
    group = p.add_argument_group("chạy nhiều cấu hình, mỗi run ở một tiến trình riêng")
    group.add_argument("--suite", action="store_true", help="2 dataset x 3 model x ERM/mixup")
    group.add_argument("--datasets", nargs="+", choices=DATASETS)
    group.add_argument("--models", nargs="+", choices=MODEL_NAMES)
    group.add_argument("--methods", nargs="+", choices=("erm", "mixup"))
    group.add_argument("--seeds", nargs="+", type=int)
    p.set_defaults(milestones=None)
    return p


def paper_deviations(a):
    """Paper settings plus implementation details established by attached train.py."""
    expected = {
        "epochs": 200, "batch_size": 128, "lr": 0.1, "momentum": 0.9, "wd": 1e-4,
        "milestones": [60, 120, 180] if a.model == "wideresnet28_10" else [100, 150],
        "gamma": 0.1, "no_augment": False, "amp": False, "official_lr_offset": False,
        "limit_batches": 0, "pairing": "random", "label_mode": "mix", "smoothing": 0.0,
        "per_sample_lambda": False, "mix_warmup": 0, "input_noise": 0.0, "latent_module": None,
    }
    differences = [f"{key}={getattr(a, key)!r} (expected {value!r})"
                   for key, value in expected.items() if getattr(a, key) != value]
    if a.alpha not in (0.0, 1.0):
        differences.append(f"alpha={a.alpha!r} (expected 0 for ERM or 1 for mixup)")
    return differences


def validate_args(p, a):
    for key in ("lr", "momentum", "wd", "gamma", "alpha", "smoothing", "input_noise"):
        if not math.isfinite(getattr(a, key)):
            p.error(f"--{key.replace('_', '-')} phải là số hữu hạn")
    for key in ("epochs", "batch_size", "test_batch_size"):
        if getattr(a, key) <= 0:
            p.error(f"--{key.replace('_', '-')} phải > 0")
    for key in ("workers", "limit_batches", "seed", "mix_warmup"):
        if getattr(a, key) < 0:
            p.error(f"--{key.replace('_', '-')} phải >= 0")
    if a.lr <= 0 or not 0 < a.gamma <= 1 or a.wd < 0 or not 0 <= a.momentum < 1:
        p.error("cần lr > 0, 0 < gamma <= 1, wd >= 0 và 0 <= momentum < 1")
    if a.alpha < 0 or not 0 <= a.smoothing < 1 or a.input_noise < 0:
        p.error("cần alpha >= 0, 0 <= smoothing < 1 và input-noise >= 0")
    if a.milestones is not None and any(m < 0 for m in a.milestones):
        p.error("milestones phải >= 0")
    if not a.suite and any(getattr(a, key) is not None for key in ("datasets", "models", "methods", "seeds")):
        p.error("--datasets/--models/--methods/--seeds cần --suite")
    for key in ("datasets", "models", "methods", "seeds"):
        values = getattr(a, key)
        if values is not None and len(set(values)) != len(values):
            p.error(f"--{key} không được lặp giá trị")
    if a.seeds and any(seed < 0 for seed in a.seeds):
        p.error("seeds phải >= 0")


def plan_runs(a):
    runs = []
    for dataset in (a.datasets or DATASETS) if a.suite else [a.dataset]:
        for model in (a.models or MODEL_NAMES) if a.suite else [a.model]:
            for alpha in ([0.0 if method == "erm" else 1.0 for method in (a.methods or ["erm", "mixup"])]
                          if a.suite else [a.alpha]):
                for seed in (a.seeds or [a.seed]) if a.suite else [a.seed]:
                    run = copy.deepcopy(a)
                    run.dataset, run.model, run.alpha, run.seed = dataset, model, alpha, seed
                    run.milestones = (list(a.milestones) if a.milestones is not None else
                                      [60, 120, 180] if model == "wideresnet28_10" else [100, 150])
                    deviations = paper_deviations(run)
                    name = default_run_name(run, f"exp32_{dataset}/{model}")
                    if deviations:
                        digest = hashlib.sha256(json.dumps(deviations).encode()).hexdigest()[:8]
                        name += f"_custom_{digest}"
                    run.run_name = (f"{a.run_name}/{name}" if a.suite and a.run_name else
                                    a.run_name or name)
                    for key in SINGLE_ONLY:
                        delattr(run, key)
                    run.paper_protocol = not deviations
                    run.protocol_deviations = deviations
                    run.paper_url = PAPER_URL
                    runs.append(run)
    return runs


def check_run_directory(a):
    """Validate BEFORE RunLogger overwrites config.json or touches history.csv."""
    root = Path(a.out_dir).resolve()
    target = (root / a.run_name).resolve()
    if target == root or not target.is_relative_to(root):
        raise ValueError("run-name phải là đường dẫn con trong out-dir")
    if not target.exists() or not any(target.iterdir()):
        return
    if a.no_resume:
        raise ValueError(f"Run đã tồn tại: {target}. Chọn --run-name mới khi dùng --no-resume.")
    config_path = target / "config.json"
    if not config_path.exists():
        raise ValueError(f"Thiếu config.json để xác minh run cũ: {target}")
    saved = json.loads(config_path.read_text(encoding="utf-8"))["config"]
    saved.setdefault("model", "preact_resnet18")
    saved.setdefault("test_batch_size", 100)
    mismatches = [key for key in RESUME_KEYS if saved.get(key) != getattr(a, key)]
    if mismatches:
        raise ValueError(f"Cấu hình resume khác run cũ ({', '.join(mismatches)}): {target}")
    if saved.get("epochs", 0) > a.epochs:
        raise ValueError("Không thể resume với epochs nhỏ hơn cấu hình cũ")
    if (target / "history.csv").exists() and not (target / "last.pt").exists():
        raise ValueError(f"Có history.csv nhưng thiếu last.pt: {target}. Chọn run-name mới.")


def run_experiment(a):
    check_run_directory(a)

    rng = set_seed(a.seed, a.deterministic)
    device = get_device(a.device)
    data = build_cifar(a.dataset, a.data_dir, a.batch_size, test_batch_size=a.test_batch_size,
                       workers=a.workers, seed=a.seed, augment=not a.no_augment,
                       download=not a.no_download)
    model = build_model(a.model, data.num_classes).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=a.lr, momentum=a.momentum, weight_decay=a.wd)
    sched = step_schedule(opt, a.milestones, a.gamma, official_offset=a.official_lr_offset)

    config = {**vars(a), "num_classes": data.num_classes, "num_parameters": count_params(model)}
    log = RunLogger(a.out_dir, a.run_name, config, a.wandb, a.wandb_project, resume=not a.no_resume)
    print(f"[exp32] {a.run_name} | device={device} | params={count_params(model):,}", flush=True)
    if a.protocol_deviations:
        print("[custom protocol] " + "; ".join(a.protocol_deviations), flush=True)

    tr = Trainer(model, opt, sched, data.train, {"test": data.test}, data.num_classes,
                 mixup_config_from_args(a), log, rng, device, a.epochs, amp=a.amp,
                 limit_batches=a.limit_batches)
    summary = tr.fit(resume=not a.no_resume)
    print(json.dumps(summary, indent=2))
    return summary


def child_command(run):
    """Isolate suite runs, including CUDA memory and DataLoader workers."""
    command = [sys.executable, str(Path(__file__).resolve())]
    for key in ("dataset", "model", "seed", "epochs", "batch_size", "test_batch_size", "lr",
                "momentum", "wd", "gamma", "workers", "data_dir", "out_dir", "run_name",
                "device", "wandb_project", "limit_batches", "alpha", "pairing", "label_mode",
                "smoothing", "mix_warmup", "input_noise"):
        command.extend(["--" + key.replace("_", "-"), str(getattr(run, key))])
    command.append("--milestones")
    command.extend(str(m) for m in run.milestones)
    for key in ("no_augment", "no_download", "official_lr_offset", "amp", "deterministic",
                "no_resume", "wandb", "per_sample_lambda"):
        if getattr(run, key):
            command.append("--" + key.replace("_", "-"))
    if run.latent_module:
        command.extend(["--latent-module", run.latent_module])
    return command


def main(argv=None):
    p = build_parser()
    a = p.parse_args(argv)
    validate_args(p, a)
    runs = plan_runs(a)
    if a.dry_run:
        print(json.dumps([vars(run) for run in runs], indent=2, ensure_ascii=False))
        return runs
    if a.suite:
        for index, run in enumerate(runs, 1):
            print(f"[suite {index}/{len(runs)}] {run.run_name}", flush=True)
            subprocess.run(child_command(run), check=True)
        return None
    return run_experiment(runs[0])


if __name__ == "__main__":
    main()
