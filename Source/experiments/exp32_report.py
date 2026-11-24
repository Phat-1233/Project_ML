"""Aggregate section 3.2 runs without importing PyTorch.

Figure 3(a) reports test error but does not state the checkpoint-selection rule.
The default ``best_err`` is an explicit reporting choice; compare ``last_err``
and ``median_last10_err`` as well. No missing result is filled from the paper.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import warnings
from collections import defaultdict
from pathlib import Path

PAPER_URL = "https://arxiv.org/abs/1710.09412"
REFERENCE_CODE_URL = "https://github.com/facebookresearch/mixup-cifar10"
MODELS = ("preact_resnet18", "wideresnet28_10", "densenet_bc190")
DATASETS = ("cifar10", "cifar100")
METRICS = ("best_err", "last_err", "median_last10_err")
PAPER_ERRORS = {
    ("cifar10", "preact_resnet18"): (5.6, 4.2),
    ("cifar10", "wideresnet28_10"): (3.8, 2.7),
    ("cifar10", "densenet_bc190"): (3.7, 2.7),
    ("cifar100", "preact_resnet18"): (25.6, 21.1),
    ("cifar100", "wideresnet28_10"): (19.4, 17.5),
    ("cifar100", "densenet_bc190"): (19.0, 16.8),
}
# Values affecting the scientific experiment; paths, workers, devices and seed
# deliberately do not distinguish protocols. Reject mixing these within a cell.
SCIENTIFIC_FIELDS = (
    "epochs", "batch_size", "lr", "momentum", "wd", "milestones", "gamma",
    "no_augment", "amp", "limit_batches", "official_lr_offset", "pairing",
    "label_mode", "smoothing", "per_sample_lambda", "mix_warmup",
    "input_noise", "latent_module", "deterministic",
)
OPTION_DEFAULTS = {
    "no_augment": False, "amp": False, "limit_batches": 0,
    "official_lr_offset": False, "pairing": "random", "label_mode": "mix",
    "smoothing": 0.0, "per_sample_lambda": False, "mix_warmup": 0,
    "input_noise": 0.0, "latent_module": None, "deterministic": False,
}


def _warn(message: str):
    warnings.warn(message, UserWarning, stacklevel=2)


def _protocol_deviations(config: dict, summary: dict) -> list[str]:
    deviations = list(config.get("protocol_deviations") or [])
    expected = {
        "epochs": 200, "batch_size": 128, "lr": 0.1, "momentum": 0.9,
        "wd": 1e-4, "gamma": 0.1,
        "milestones": [60, 120, 180] if config["model"] == "wideresnet28_10" else [100, 150],
        **{key: value for key, value in OPTION_DEFAULTS.items() if key != "deterministic"},
    }
    for key, value in expected.items():
        if config.get(key, OPTION_DEFAULTS.get(key)) != value:
            deviations.append(f"{key} differs from the section 3.2 protocol")
    if float(config["alpha"]) not in (0.0, 1.0):
        deviations.append("alpha is neither 0 (ERM) nor 1 (mixup)")
    if summary.get("epochs_run") != config.get("epochs"):
        deviations.append("run is incomplete")
    if config.get("paper_protocol") is not True:
        deviations.append("config does not declare paper_protocol=true")
    return sorted(set(deviations))


def collect_runs(root, include_nonpaper: bool = False) -> list[dict]:
    """Read experiment artifacts; warn and skip malformed/unrelated runs.

    Complete 200-epoch paper runs are included by default. Explicitly including
    other runs keeps their protocol deviations visible in every summary row.
    """
    runs = []
    for summary_path in sorted(Path(root).rglob("summary.json")):
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            payload = json.loads(summary_path.with_name("config.json").read_text(encoding="utf-8"))
            config = payload["config"]
            if config.get("dataset") not in DATASETS or config.get("model") not in MODELS:
                _warn(f"Skipping unrelated or unrecognized run: {summary_path.parent}")
                continue
            alpha = float(config["alpha"])
            seed = config["seed"]
            if not math.isfinite(alpha) or alpha < 0 or not isinstance(seed, int):
                raise ValueError("alpha must be finite/nonnegative and seed must be an integer")
            if not isinstance(config.get("epochs"), int) or config["epochs"] <= 0:
                raise ValueError("epochs must be a positive integer")
            if not isinstance(summary.get("epochs_run"), int) or summary["epochs_run"] <= 0:
                raise ValueError("epochs_run must be a positive integer")
            deviations = _protocol_deviations(config, summary)
            paper_protocol = not deviations
            if not paper_protocol and not include_nonpaper:
                _warn(f"Skipping nonpaper/partial run {summary_path.parent}: {'; '.join(deviations)}")
                continue
            method = "ERM" if alpha == 0 else ("mixup" if alpha == 1 else f"mixup(alpha={alpha:g})")
            runs.append({
                "dataset": config["dataset"], "model": config["model"],
                "method": method, "alpha": alpha, "seed": seed,
                "config": config, "summary": summary,
                "paper_protocol": paper_protocol, "protocol_deviations": deviations,
                "run_dir": str(summary_path.parent.resolve()),
            })
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
            _warn(f"Skipping malformed run {summary_path.parent}: {exc}")
    return runs


def summarize_runs(runs: list[dict], metric: str = "best_err") -> list[dict]:
    """Group seeds using mean and sample standard deviation of test error (%)."""
    if metric not in METRICS:
        raise ValueError(f"metric must be one of {METRICS}")
    groups = defaultdict(list)
    seen = set()
    for run in runs:
        key = (run["dataset"], run["model"], run["method"])
        identity = (*key, run["seed"])
        if identity in seen:
            raise ValueError(f"Duplicate dataset/model/method/seed: {identity}")
        seen.add(identity)
        try:
            value = float(run["summary"][metric])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Missing or invalid {metric}: {run['run_dir']}") from exc
        if not math.isfinite(value) or not 0 <= value <= 100:
            raise ValueError(f"Invalid test error {metric}={value}: {run['run_dir']}")
        groups[key].append(run)
    rows = []
    for (dataset, model, method), members in sorted(groups.items()):
        signatures = {
            json.dumps({key: run["config"].get(key, OPTION_DEFAULTS.get(key)) for key in SCIENTIFIC_FIELDS},
                       sort_keys=True) for run in members
        }
        if len(signatures) != 1:
            raise ValueError(f"Mixed scientific settings for {dataset}/{model}/{method}; report each protocol separately")
        errors = [float(run["summary"][metric]) for run in members]
        seeds = sorted(run["seed"] for run in members)
        reference = PAPER_ERRORS[(dataset, model)][0 if method == "ERM" else 1] if method in ("ERM", "mixup") else None
        mean = statistics.mean(errors)
        rows.append({
            "dataset": dataset, "model": model, "method": method, "alpha": members[0]["alpha"],
            "metric": metric, "count": len(errors), "seeds": seeds,
            "mean_err": mean, "sample_std_err": statistics.stdev(errors) if len(errors) > 1 else None,
            "paper_err": reference, "gap_to_paper": mean - reference if reference is not None else None,
            "paper_protocol": all(run["paper_protocol"] for run in members),
            "protocol_deviations": sorted({item for run in members for item in run["protocol_deviations"]}),
            "settings": json.loads(next(iter(signatures))),
        })
        if len(seeds) < 2:
            _warn(f"Only one seed for {dataset}/{model}/{method}; sample standard deviation is undefined")
    observed_seeds = {run["seed"] for run in runs}
    for row in rows:
        missing = sorted(observed_seeds - set(row["seeds"]))
        if missing:
            _warn(f"Missing observed seeds {missing} for {row['dataset']}/{row['model']}/{row['method']}")
    for dataset in DATASETS:
        for model in MODELS:
            pair = {row["method"]: row for row in rows if row["dataset"] == dataset and row["model"] == model}
            for method in ("ERM", "mixup"):
                if method not in pair:
                    _warn(f"Missing cell: {dataset}/{model}/{method}")
            if "ERM" in pair and "mixup" in pair and pair["ERM"]["seeds"] != pair["mixup"]["seeds"]:
                _warn(f"Mismatched ERM/mixup seeds for {dataset}/{model}")
    return rows


def _plot_runs(runs: list[dict], destination: Path):
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("--plot requires matplotlib; install the plotting dependency") from exc
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), squeeze=False)
    for di, dataset in enumerate(DATASETS):
        for mi, model in enumerate(MODELS):
            ax = axes[di, mi]
            methods = sorted({run["method"] for run in runs if run["dataset"] == dataset and run["model"] == model})
            for method in methods:
                values = defaultdict(list)
                for run in runs:
                    if (run["dataset"], run["model"], run["method"]) != (dataset, model, method):
                        continue
                    history_path = Path(run["run_dir"]) / "history.csv"
                    if not history_path.exists():
                        _warn(f"Missing learning curve: {history_path}")
                        continue
                    with history_path.open(encoding="utf-8", newline="") as stream:
                        for row in csv.DictReader(stream):
                            values[int(row["epoch"]) + 1].append(float(row["test_err"]))
                if values:
                    epochs = sorted(values)
                    ax.plot(epochs, [statistics.mean(values[e]) for e in epochs], label=method)
            ax.set(title=f"{dataset} / {model}", xlabel="Epoch", ylabel="Test error (%)")
            ax.grid(alpha=0.25)
            if ax.lines:
                ax.legend()
    suffix = " — includes nonpaper runs" if any(not run["paper_protocol"] for run in runs) else ""
    fig.suptitle("Mean test error across available seeds (Figure 3-style curves)" + suffix)
    fig.tight_layout()
    fig.savefig(destination, dpi=160)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Results directory to search recursively")
    parser.add_argument("--metric", choices=METRICS, default="best_err")
    parser.add_argument("--include-nonpaper", action="store_true", help="Include and mark partial or modified protocols")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args(argv)
    if not args.root.is_dir():
        parser.error(f"Results directory does not exist: {args.root}")
    try:
        runs = collect_runs(args.root, args.include_nonpaper)
        if not runs:
            parser.error("No eligible runs found. Check the results directory and protocol warnings; "
                         "use --include-nonpaper to inspect modified or partial runs.")
        rows = summarize_runs(runs, args.metric)
    except (ValueError, KeyError) as exc:
        parser.error(str(exc))
    output = args.output_dir or args.root / "report"
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "metric": args.metric,
        "metric_note": "Checkpoint selection is a reporting choice; section 3.2 does not specify it.",
        "units": "test error percent; gap_to_paper is percentage points",
        "paper_url": PAPER_URL, "reference_code_url": REFERENCE_CODE_URL,
        "include_nonpaper": args.include_nonpaper, "rows": rows,
        "run_directories": [run["run_dir"] for run in runs],
    }
    (output / "comparison.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    fields = ["dataset", "model", "method", "alpha", "metric", "count", "seeds", "mean_err",
              "sample_std_err", "paper_err", "gap_to_paper", "paper_protocol", "protocol_deviations", "settings"]
    with (output / "comparison.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value
                             for key, value in row.items()})
    if args.plot:
        _plot_runs(runs, output / "training_curves.png")
    print(f"Reported {len(runs)} runs in {len(rows)} cells to {output}")
    return report


if __name__ == "__main__":
    main()
