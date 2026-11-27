"""Gom kết quả 3.4 thành bảng giống Bảng 2 của bài báo (+ biểu đồ lỗi test theo epoch).

Đọc mọi <results>/exp34_noise*/*/summary.json, gom theo (noise_rate, α, dropout) qua các seed.

    python experiments/exp34_report.py --results ./results
    -> results/exp34_report.csv, exp34_report.md, exp34_curves.png

Mỗi ô ghi "nhóm ± std (bài báo)". std chỉ có khi chạy >= 2 seed.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

# Bảng 2: (noise_rate, α, p) -> (test best, test last, train real, train corrupted), đơn vị %.
# Đây cũng là danh sách cấu hình notebook sẽ chạy.
PAPER = {
    (0.2, 0.0, 0.0): (12.7, 16.6, 0.05, 0.28),
    (0.2, 0.0, 0.7): (8.8, 10.4, 5.26, 83.55),
    (0.2, 8.0, 0.0): (5.9, 6.4, 2.27, 86.32),
    (0.2, 4.0, 0.1): (6.2, 6.2, 1.92, 85.02),   # p=0.1 không nằm trong lưới {0.3,0.5,0.7} bài báo nêu
    (0.5, 0.0, 0.0): (18.8, 44.6, 0.26, 0.64),
    (0.5, 0.0, 0.8): (14.1, 15.5, 12.71, 86.98),
    (0.5, 32.0, 0.0): (11.3, 12.7, 5.84, 85.71),
    (0.5, 8.0, 0.3): (10.9, 10.9, 7.56, 87.90),
    (0.8, 0.0, 0.0): (36.5, 73.9, 0.62, 0.83),
    (0.8, 0.0, 0.8): (30.9, 35.1, 29.84, 86.37),
    (0.8, 32.0, 0.0): (25.3, 30.9, 18.92, 85.44),
    (0.8, 8.0, 0.3): (24.0, 24.8, 19.70, 87.67),
}
# "Train real" của bài báo = lỗi trên các mẫu KHÔNG bị làm nhiễu (train_clean_err).
METRICS = ("best_err", "last_err", "train_clean_err", "train_corrupted_err", "train_real_err")
COLS = ("Test best", "Test last", "Train real", "Train corrupted", "Train toàn bộ nhãn thật")


def label(alpha: float, p: float) -> str:
    if alpha <= 0:
        return "ERM" if p <= 0 else f"ERM + dropout (p={p:g})"
    return f"mixup (α={alpha:g})" if p <= 0 else f"mixup + dropout (α={alpha:g}, p={p:g})"


def collect(root: Path) -> dict:
    """{(rate, α, p): [summary, ...]} chỉ lấy lần chạy đã xong (có train_clean_err)."""
    runs: dict[tuple, list[dict]] = {}
    for s in sorted(root.glob("exp34_noise*/*/summary.json")):
        summ = json.loads(s.read_text())
        if "train_clean_err" not in summ:
            continue
        cfg = json.loads((s.parent / "config.json").read_text())["config"]
        key = (float(cfg["noise_rate"]), float(cfg["alpha"]), float(cfg["dropout"]))
        runs.setdefault(key, []).append(summ)
    return runs


def _fmt(vals: list[float], paper) -> str:
    m = statistics.mean(vals)
    s = f"{m:.2f}" + (f" ± {statistics.stdev(vals):.2f}" if len(vals) > 1 else "")
    return s + (f" ({paper:g})" if paper is not None else "")


def build_rows(runs: dict) -> list[dict]:
    rows = []
    for key in sorted(runs, key=lambda k: (k[0], k[1] > 0, k[2] > 0, k[1], k[2])):
        rate, alpha, p = key
        paper = PAPER.get(key, (None,) * 4) + (None,)
        row = {"noise": f"{rate:.0%}", "method": label(alpha, p), "n_seeds": len(runs[key])}
        for col, m, pv in zip(COLS, METRICS, paper):
            row[col] = _fmt([r[m] for r in runs[key]], pv)
        rows.append(row)
    return rows


def to_markdown(rows: list[dict]) -> str:
    if not rows:
        return "_Chưa có lần chạy nào xong._\n"
    head = list(rows[0])
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    lines += ["| " + " | ".join(str(r[h]) for h in head) + " |" for r in rows]
    return "\n".join(lines) + "\nMỗi ô: nhóm (± std qua seed) (bài báo). Đơn vị: % lỗi.\n"


def plot_curves(root: Path, runs: dict, out: Path):
    """Mỗi mức nhiễu một ô: lỗi test theo epoch, mỗi lần chạy (cấu hình, seed) một đường."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rates = sorted({k[0] for k in runs})
    fig, axes = plt.subplots(1, len(rates), figsize=(5 * len(rates), 4), squeeze=False)
    for ax, rate in zip(axes[0], rates):
        for s in sorted(root.glob(f"exp34_noise{round(rate * 100)}/*/history.csv")):
            cfg = json.loads((s.parent / "config.json").read_text())["config"]
            with open(s) as f:
                h = list(csv.DictReader(f))
            ax.plot([int(r["epoch"]) for r in h], [float(r["test_err"]) for r in h],
                    label=f"{label(float(cfg['alpha']), float(cfg['dropout']))} s{cfg['seed']}")
        ax.set(title=f"{rate:.0%} nhãn nhiễu", xlabel="epoch", ylabel="lỗi test (%)")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=150)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", default="./results")
    p.add_argument("--no-plot", action="store_true")
    a = p.parse_args(argv)
    root = Path(a.results)
    runs = collect(root)
    rows = build_rows(runs)
    if rows:
        with open(root / "exp34_report.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    md = to_markdown(rows)
    (root / "exp34_report.md").write_text(md, encoding="utf-8")
    print(md)
    if runs and not a.no_plot:
        plot_curves(root, runs, root / "exp34_curves.png")
    return rows


if __name__ == "__main__":
    main()
