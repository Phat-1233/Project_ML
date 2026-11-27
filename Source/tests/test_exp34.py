"""Kiểm tra phần gom bảng của 3.4 trên kết quả giả (không cần torch/GPU).

    cd Source
    python -m pytest tests/test_exp34.py -q     # hoặc: python tests/test_exp34.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.exp34_report import PAPER, main  # noqa: E402


def _fake_run(root: Path, name: str, rate, alpha, p, seed, err, done=True):
    d = root / f"exp34_noise{round(rate * 100)}" / name
    d.mkdir(parents=True)
    (d / "config.json").write_text(json.dumps(
        {"config": {"noise_rate": rate, "alpha": alpha, "dropout": p, "seed": seed}}))
    summ = {"best_err": err, "last_err": err + 1}
    if done:
        summ.update(train_clean_err=1.0, train_corrupted_err=85.0, train_real_err=40.0)
    (d / "summary.json").write_text(json.dumps(summ))


def test_report_groups_seeds_and_skips_unfinished():
    with tempfile.TemporaryDirectory() as t:
        root = Path(t)
        _fake_run(root, "mixup_a32_seed0", 0.5, 32.0, 0.0, 0, 11.0)
        _fake_run(root, "mixup_a32_seed1", 0.5, 32.0, 0.0, 1, 13.0)
        _fake_run(root, "erm_seed0", 0.5, 0.0, 0.0, 0, 20.0)
        _fake_run(root, "erm_seed1", 0.5, 0.0, 0.0, 1, 99.0, done=False)   # chưa xong: bỏ qua
        rows = main(["--results", t, "--no-plot"])
        assert [r["method"] for r in rows] == ["ERM", "mixup (α=32)"]
        erm, mix = rows
        assert erm["n_seeds"] == 1 and erm["Test best"] == "20.00 (18.8)"
        assert mix["n_seeds"] == 2 and mix["Test best"] == "12.00 ± 1.41 (11.3)"
        assert mix["Test last"] == "13.00 ± 1.41 (12.7)"
        assert mix["Train real"] == "1.00 ± 0.00 (5.84)" and mix["Train toàn bộ nhãn thật"] == "40.00 ± 0.00"
        assert (root / "exp34_report.md").exists() and (root / "exp34_report.csv").exists()


def test_paper_table_covers_12_configs():
    assert len(PAPER) == 12
    for rate in (0.2, 0.5, 0.8):
        assert sum(k[0] == rate for k in PAPER) == 4


if __name__ == "__main__":
    test_report_groups_seeds_and_skips_unfinished()
    test_paper_table_covers_12_configs()
    print("OK")
