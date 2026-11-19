"""Kiểm tra nhanh phần lõi, chạy trên CPU trong vài giây, không cần tải dữ liệu.

    cd Source
    python -m pytest tests -q        # hoặc: python tests/test_core.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cifar.models import LATENT_LAYERS, preact_resnet18  # noqa: E402
from core.attacks import fgsm, ifgsm  # noqa: E402
from core.data_utils import TensorClassificationDataset, corrupt_labels  # noqa: E402
from core.mixup import (LatentMixer, mixup_batch, mixup_loss, mixup_real_fake,  # noqa: E402
                        soft_cross_entropy, soft_targets)
from core.schedules import step_schedule  # noqa: E402
from core.trainer import MixupConfig, Trainer  # noqa: E402
from core.utils import RunLogger, set_seed  # noqa: E402


def test_shapes_for_all_data_types():
    rng = np.random.default_rng(0)
    for shape in [(8, 3, 32, 32), (8, 1, 40, 101), (8, 13)]:  # ảnh, phổ âm thanh, bảng
        x, y = torch.randn(shape), torch.randint(0, 4, (8,))
        mb = mixup_batch(x, y, 1.0, rng)
        assert mb.x.shape == x.shape and 0 <= float(mb.lam) <= 1
        # x~ phải đúng bằng λ x + (1-λ) x[perm]
        assert torch.allclose(mb.x, mb.lam * x + (1 - mb.lam) * x[mb.index])
        mb2 = mixup_batch(x, y, 1.0, rng, per_sample=True)
        assert mb2.lam.shape == (8,)


def test_alpha_zero_is_erm():
    x, y = torch.randn(6, 5), torch.randint(0, 3, (6,))
    mb = mixup_batch(x, y, 0.0, np.random.default_rng(0))
    assert torch.equal(mb.x, x) and float(mb.lam) == 1.0


def test_two_term_loss_equals_soft_label_ce():
    """λ·CE(y_i) + (1-λ)·CE(y_j) == CE với nhãn mềm λ y_i + (1-λ) y_j."""
    rng = np.random.default_rng(1)
    x, y = torch.randn(16, 10), torch.randint(0, 5, (16,))
    mb = mixup_batch(x, y, 0.4, rng)
    logits = torch.randn(16, 5)
    a = mixup_loss(logits, mb)
    b = soft_cross_entropy(logits, soft_targets(mb, 5))
    assert torch.allclose(a, b, atol=1e-6)


def test_hard_labels_and_smoothing():
    rng = np.random.default_rng(2)
    x, y = torch.randn(10, 3), torch.randint(0, 4, (10,))
    mb = mixup_batch(x, y, 1.0, rng)
    t = soft_targets(mb, 4, "hard")
    assert torch.allclose(t.sum(1), torch.ones(10)) and set(t.unique().tolist()) <= {0.0, 1.0}
    t = soft_targets(mb, 4, "mix", smoothing=0.1)
    assert torch.allclose(t.sum(1), torch.ones(10))


def test_same_class_pairing():
    y = torch.tensor([0, 0, 1, 1, 1, 2, 2, 0])
    mb = mixup_batch(torch.randn(8, 2), y, 1.0, np.random.default_rng(3), pairing="same_class")
    assert torch.equal(mb.y_a, mb.y_b)


def test_latent_mixer_changes_hidden_only_once():
    torch.manual_seed(0)
    net = preact_resnet18().eval()
    x = torch.randn(4, 3, 32, 32)
    ref = net(x)
    mixer = LatentMixer(net.layer3)
    mixer.arm(torch.tensor([1, 0, 3, 2]), torch.tensor(0.5))
    mixed = net(x)
    assert not torch.allclose(ref, mixed)
    assert torch.allclose(net(x), ref)  # hook tự tắt sau 1 lần forward
    mixer.remove()
    assert set(LATENT_LAYERS.values()) <= set(dict(net.named_modules()))


def test_model_matches_official_param_count():
    n = sum(p.numel() for p in preact_resnet18(10).parameters())
    assert n == 11_171_274, n  # cấu trúc giống ResNet18() (PreActBlock) của mã gốc


def test_corrupt_labels_rate():
    labels, mask = corrupt_labels(np.zeros(100_000, dtype=int), 0.5, 10, seed=0)
    assert abs(mask.mean() - 0.5) < 0.01
    assert abs((labels != 0).mean() - 0.45) < 0.01  # ~ rate·(K-1)/K


def test_gan_mix():
    real, fake = torch.ones(5, 2), torch.zeros(5, 2)
    mixed, t = mixup_real_fake(real, fake, 0.2, np.random.default_rng(0))
    assert torch.allclose(mixed[:, 0], t)


def test_fgsm_within_eps():
    torch.manual_seed(0)
    net = nn.Sequential(nn.Flatten(), nn.Linear(12, 3))
    x, y = torch.rand(4, 3, 2, 2), torch.randint(0, 3, (4,))
    xa = fgsm(net, x, y, 4 / 255)
    assert (xa - x).abs().max() <= 4 / 255 + 1e-6
    xi = ifgsm(net, x, y, 4 / 255, 1 / 255, 10)
    assert (xi - x).abs().max() <= 4 / 255 + 1e-6


def test_lr_schedule_follows_paper():
    opt = torch.optim.SGD([nn.Parameter(torch.zeros(1))], lr=0.1)
    s = step_schedule(opt, [100, 150])
    lrs = []
    for _ in range(160):
        lrs.append(opt.param_groups[0]["lr"])
        opt.step()
        s.step()
    assert lrs[99] == 0.1 and abs(lrs[100] - 0.01) < 1e-12 and abs(lrs[150] - 0.001) < 1e-12


def _tabular_trainer(tmp, epochs, mix):
    rng = set_seed(0)
    X = np.random.randn(256, 6).astype(np.float32)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    dl = torch.utils.data.DataLoader(TensorClassificationDataset(X, y), batch_size=32, shuffle=True,
                                     generator=torch.Generator().manual_seed(0))
    model = nn.Sequential(nn.Linear(6, 32), nn.ReLU(), nn.Linear(32, 2))
    opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    log = RunLogger(tmp, "run", {"test": True}, resume=True)
    return Trainer(model, opt, step_schedule(opt, [3]), dl, {"test": dl}, 2, mix, log, rng,
                   torch.device("cpu"), epochs)


def test_trainer_end_to_end_and_resume():
    with tempfile.TemporaryDirectory() as tmp:
        tr = _tabular_trainer(tmp, 2, MixupConfig(alpha=1.0))
        s = tr.fit()
        assert s["epochs_run"] == 2 and s["last_err"] < 50
        tr2 = _tabular_trainer(tmp, 4, MixupConfig(alpha=1.0))
        s2 = tr2.fit()  # phải tiếp tục từ epoch 2, không chạy lại từ đầu
        assert s2["epochs_run"] == 4
        assert (Path(tmp) / "run" / "history.csv").read_text().count("\n") == 5


def test_trainer_latent_and_ablation_modes():
    with tempfile.TemporaryDirectory() as tmp:
        for cfg in [MixupConfig(alpha=0), MixupConfig(alpha=1, label_mode="hard"),
                    MixupConfig(alpha=0, smoothing=0.1), MixupConfig(alpha=1, input_noise_std=0.1),
                    MixupConfig(alpha=1, latent_module="2"), MixupConfig(alpha=1, warmup_epochs=1)]:
            tr = _tabular_trainer(tmp + f"/{id(cfg)}", 2, cfg)
            assert np.isfinite(tr.fit(resume=False)["last_err"])


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
