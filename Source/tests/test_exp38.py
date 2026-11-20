"""Kiểm tra script 3.8 trên CPU với dữ liệu giả (không tải CIFAR).

    cd Source
    python -m pytest tests/test_exp38.py -q
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

import exp38_ablation as E  # noqa: E402
import exp38_report as R  # noqa: E402
from cifar.data import MEAN, STD, CifarData  # noqa: E402


class FakeCifar(Dataset):
    """Giống torchvision CIFAR10 ở chỗ có .data (N,32,32,3) uint8 và .targets."""

    def __init__(self, n=64, seed=0):
        r = np.random.default_rng(seed)
        self.data = r.integers(0, 256, (n, 32, 32, 3), dtype=np.uint8)
        self.targets = (np.arange(n) % 10).tolist()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        x = torch.from_numpy(self.data[i]).float().div(255).permute(2, 0, 1)
        x = (x - torch.tensor(MEAN)[:, None, None]) / torch.tensor(STD)[:, None, None]
        return x, self.targets[i]


def fake_build_cifar(name, data_dir, batch_size, workers=0, seed=0, **kw):
    tr, te = FakeCifar(64, 0), FakeCifar(32, 1)
    return CifarData(DataLoader(tr, batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(seed)),
                     DataLoader(te, batch_size=16), 10, {})


def _brute_knn(imgs, labels, k, same_class):
    x = torch.from_numpy(imgs).float().div(255)
    x = ((x - torch.tensor(MEAN)) / torch.tensor(STD)).flatten(1)
    d = torch.cdist(x, x) ** 2
    d.fill_diagonal_(float("inf"))
    if same_class:
        lab = torch.from_numpy(labels)
        d[lab[:, None] != lab[None, :]] = float("inf")
    return d.topk(k, largest=False).indices.numpy()


def test_knn_matches_brute_force():
    ds = FakeCifar(60)
    imgs, labels = E.raw_images_labels(ds)
    for sc in (False, True):
        got = E.compute_knn(imgs, labels, k=4, same_class=sc, device=torch.device("cpu"), chunk=16)
        ref = _brute_knn(imgs, labels, 4, sc)
        assert all(set(a) == set(b) for a, b in zip(got, ref))
        assert not (got == np.arange(60)[:, None]).any()              # không chứa chính nó
        if sc:
            assert (labels[got] == labels[:, None]).all()             # SC: cùng lớp


def test_sampler_and_pair_index():
    knn = np.random.default_rng(0).integers(0, 100, (100, 7))
    s = E.KNNPairBatchSampler(knn, batch_size=10, seed=0)
    batches = list(s)
    assert len(batches) == len(s) == 10
    anchors = [i for b in batches for i in b[:5]]
    assert len(set(anchors)) == len(anchors)                          # không lặp neo trong 1 epoch
    for b in batches:
        assert all(p in knn[a] for a, p in zip(b[:5], b[5:]))         # láng giềng đúng của neo
    idx = E.knn_pair_index(None, torch.zeros(10))
    assert torch.equal(idx[idx], torch.arange(10))                    # neo <-> láng giềng


def test_variant_presets_and_override():
    a = E.parse_args(["--variant", "input_sc_rp"])
    assert (a.alpha, a.pairing, a.label_mode) == (1.0, "same_class", "hard")
    a = E.parse_args(["--variant", "ls_0.1", "--smoothing", "0.3"])
    assert a.alpha == 0 and a.smoothing == 0.3                         # cờ truyền tay ghi đè
    assert E.parse_args(["--variant", "latent_l5"]).latent_module == "linear"
    assert set(R.PAPER) == set(E.VARIANTS)                            # đủ 22 dòng của Bảng 5
    assert len(E.VARIANTS) == 22


def test_end_to_end_variants_and_report(monkeypatch):
    monkeypatch.setattr(E, "build_cifar", fake_build_cifar)
    with tempfile.TemporaryDirectory() as tmp:
        for v in ["erm", "mixup", "latent_l3", "input_sc_knn", "mixup_ac_knn", "input_ls_0.1",
                  "ls_0.1", "noise_0.1"]:
            for seed in (0, 1):
                s = E.main(["--variant", v, "--epochs", "1", "--batch-size", "16", "--workers", "0",
                            "--knn-k", "3", "--seed", str(seed), "--data-dir", tmp, "--out-dir", tmp,
                            "--device", "cpu"])
                assert np.isfinite(s["median_last10_err"])
        assert (Path(tmp) / "knn_sc_k3.npy").exists()
        rows = R.main(["--results", tmp])
        assert {r["variant"] for r in rows} >= {"erm", "mixup", "input_sc_knn"}
        assert all(r["n_seeds"] == 2 for r in rows)
        assert (Path(tmp) / "exp38_ablation" / "report.md").exists()
