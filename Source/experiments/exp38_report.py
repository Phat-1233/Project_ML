"""Gom kết quả 3.8 thành bảng so sánh với Bảng 5 của bài báo (+ biểu đồ cột).

Đọc mọi thư mục <results>/exp38_ablation/*/ có summary.json, lấy `median_last10_err`
(đúng độ đo của Bảng 5), gom theo (biến thể, weight decay) qua các seed.

    python experiments/exp38_report.py --results ./results
    -> results/exp38_ablation/report.csv, report.md, report.png

Cột "sai khác" = (nhóm - bài báo) / bài báo; "trong nhiễu" = |nhóm - bài báo| <= 2·std
(chỉ có nghĩa khi chạy >= 2 seed).
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

# Bảng 5 (median test error % của 10 epoch cuối), (wd 1e-4, wd 5e-4)
PAPER = {
    "erm": (5.53, 5.18),
    "mixup": (4.24, 4.68), "mixup_ac_knn": (4.98, 5.26),
    "latent_l1": (4.44, 4.51), "latent_l2": (4.56, 4.61), "latent_l3": (5.39, 5.55),
    "latent_l4": (5.95, 5.43), "latent_l5": (5.39, 5.15),
    "input_sc_knn": (5.45, 5.52), "input_ac_knn": (5.43, 5.48),
    "input_sc_rp": (5.23, 5.55), "input_ac_rp": (5.17, 5.72),
    "ls_0.05": (5.25, 5.02), "ls_0.1": (5.33, 5.17), "ls_0.2": (5.34, 5.06),
    "input_ls_0.05": (5.02, 5.40), "input_ls_0.1": (5.08, 5.09),
    "input_ls_0.2": (4.98, 5.06), "input_ls_0.4": (5.25, 5.39),
    "noise_0.05": (5.53, 5.04), "noise_0.1": (6.41, 5.86), "noise_0.2": (7.16, 7.24),
}
GROUP = {"erm": "A ERM", "mixup": "B mixup", "mixup_ac_knn": "B mixup", "latent": "C trộn ẩn",
         "input_ls": "F trộn input + LS", "input": "D chỉ trộn input", "ls": "E label smoothing",
         "noise": "G nhiễu Gauss"}


def group_of(v: str) -> str:
    for prefix in ("mixup_ac_knn", "input_ls", "latent", "input", "noise", "ls", "mixup", "erm"):
        if v.startswith(prefix):
            return GROUP[prefix]
    return "?"


def collect(root: Path) -> dict:
    runs: dict[tuple[str, float], list[float]] = {}
    for s in sorted(root.glob("*/summary.json")):
        cfg = json.loads((s.parent / "config.json").read_text())["config"]
        v = cfg.get("variant") or s.parent.name
        err = json.loads(s.read_text())["median_last10_err"]
        runs.setdefault((v, float(cfg["wd"])), []).append(err)
    return runs


def build_rows(runs: dict) -> list[dict]:
    rows = []
    for (v, wd), errs in sorted(runs.items(), key=lambda kv: (group_of(kv[0][0]), kv[0][0], kv[0][1])):
        mean = statistics.mean(errs)
        std = statistics.stdev(errs) if len(errs) > 1 else float("nan")
        paper = PAPER.get(v, (None, None))[0 if abs(wd - 1e-4) < 1e-12 else 1] if v in PAPER else None
        row = {"group": group_of(v), "variant": v, "wd": f"{wd:g}", "n_seeds": len(errs),
               "ours_mean": round(mean, 2), "ours_std": round(std, 2) if len(errs) > 1 else "",
               "paper": paper if paper is not None else ""}
        if paper is not None:
            row["rel_diff_%"] = round(100 * (mean - paper) / paper, 1)
            row["within_2std"] = (abs(mean - paper) <= 2 * std) if len(errs) > 1 else ""
        rows.append(row)
    return rows


def write_outputs(rows: list[dict], out: Path):
    import csv
    cols = ["group", "variant", "wd", "n_seeds", "ours_mean", "ours_std", "paper", "rel_diff_%", "within_2std"]
    with open(out / "report.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    md = ["| Nhóm | Biến thể | WD | Seed | Nhóm (mean ± std) | Bài báo | Sai khác |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        ours = f"{r['ours_mean']:.2f}" + (f" ± {r['ours_std']:.2f}" if r["ours_std"] != "" else "")
        diff = f"{r['rel_diff_%']:+.1f}%" if "rel_diff_%" in r else ""
        md.append(f"| {r['group']} | {r['variant']} | {r['wd']} | {r['n_seeds']} | {ours} | {r['paper']} | {diff} |")
    (out / "report.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[warn] không có matplotlib, bỏ qua biểu đồ")
        return
    for wd in sorted({r["wd"] for r in rows}):
        sub = [r for r in rows if r["wd"] == wd]
        fig, ax = plt.subplots(figsize=(max(6, 0.55 * len(sub)), 4))
        xs = range(len(sub))
        ax.bar([x - 0.2 for x in xs], [r["paper"] or 0 for r in sub], 0.4, label="Bài báo", color="#B4B2A9")
        ax.bar([x + 0.2 for x in xs], [r["ours_mean"] for r in sub], 0.4, label="Nhóm",
               yerr=[r["ours_std"] or 0 for r in sub], capsize=3, color="#1D9E75")
        erm = next((r for r in sub if r["variant"] == "erm"), None)
        if erm:
            ax.axhline(erm["ours_mean"], ls="--", lw=1, color="#5F5E5A", label="ERM (nhóm)")
        ax.set_xticks(list(xs), [r["variant"] for r in sub], rotation=60, ha="right", fontsize=8)
        ax.set_ylabel("Median test error 10 epoch cuối (%)")
        ax.set_ylim(bottom=3.5)
        ax.set_title(f"Ablation 3.8, CIFAR-10, PreAct ResNet-18, wd = {wd}")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out / f"report_wd{wd}.png", dpi=150)
        plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="./results")
    a = p.parse_args(argv)
    root = Path(a.results) / "exp38_ablation"
    rows = build_rows(collect(root))
    if not rows:
        raise SystemExit(f"không thấy summary.json nào trong {root}")
    write_outputs(rows, root)
    return rows


if __name__ == "__main__":
    main()
