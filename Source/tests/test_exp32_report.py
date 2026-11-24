"""Scientific reporting checks using tiny JSON artifacts; no PyTorch required."""
from __future__ import annotations

import csv
import json
import sys
import warnings
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.exp32_report import collect_runs, main, summarize_runs


def write_run(root, name="run", seed=0, alpha=0, error=5.6, config_updates=None, summary_updates=None):
    directory = root / name
    directory.mkdir(parents=True)
    config = {
        "dataset": "cifar10", "model": "preact_resnet18", "seed": seed,
        "alpha": alpha, "epochs": 200, "batch_size": 128, "lr": 0.1,
        "momentum": 0.9, "wd": 1e-4, "milestones": [100, 150], "gamma": 0.1,
        "paper_protocol": True, "protocol_deviations": [],
    }
    config.update(config_updates or {})
    summary = {"epochs_run": 200, "best_err": error, "last_err": error + 1,
               "median_last10_err": error + 0.5}
    summary.update(summary_updates or {})
    (directory / "config.json").write_text(json.dumps({"config": config, "env": {}}), encoding="utf-8")
    (directory / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    return directory


def quiet_summary(runs, metric="best_err"):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return summarize_runs(runs, metric)


def test_seed_statistics_and_paper_gap(tmp_path):
    write_run(tmp_path, "a", seed=0, error=5.0)
    write_run(tmp_path, "b", seed=1, error=7.0)
    row, = quiet_summary(collect_runs(tmp_path))
    assert row["count"] == 2 and row["seeds"] == [0, 1]
    assert row["mean_err"] == 6.0
    assert row["sample_std_err"] == pytest.approx(2 ** 0.5)
    assert row["paper_err"] == 5.6 and row["gap_to_paper"] == pytest.approx(0.4)
    assert row["paper_protocol"] is True
    assert quiet_summary(collect_runs(tmp_path), "last_err")[0]["mean_err"] == 7.0


def test_nonpaper_partial_and_incomplete_excluded_and_marked(tmp_path):
    write_run(tmp_path, "good")
    write_run(tmp_path, "short", seed=1, config_updates={"epochs": 2}, summary_updates={"epochs_run": 2})
    write_run(tmp_path, "incomplete", seed=2, summary_updates={"epochs_run": 199})
    write_run(tmp_path, "limited", seed=3, config_updates={"limit_batches": 3})
    write_run(tmp_path, "nonpaper", seed=4, config_updates={"paper_protocol": False})
    with pytest.warns(UserWarning, match="Skipping nonpaper/partial"):
        runs = collect_runs(tmp_path)
    assert [run["seed"] for run in runs] == [0]
    included = collect_runs(tmp_path, include_nonpaper=True)
    assert len(included) == 5
    assert sum(run["paper_protocol"] for run in included) == 1
    assert all(run["protocol_deviations"] for run in included if not run["paper_protocol"])


def test_duplicate_seed_and_mixed_protocols_are_rejected(tmp_path):
    write_run(tmp_path, "a")
    write_run(tmp_path, "b")
    with pytest.raises(ValueError, match="Duplicate"):
        summarize_runs(collect_runs(tmp_path))
    second_config = tmp_path / "b" / "config.json"
    payload = json.loads(second_config.read_text())
    payload["config"].update(seed=1, wd=5e-4)
    second_config.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="Mixed scientific settings"):
        summarize_runs(collect_runs(tmp_path, include_nonpaper=True))


def test_warning_for_missing_cells_and_different_seeds(tmp_path):
    write_run(tmp_path, "erm", seed=0)
    write_run(tmp_path, "mixup", seed=1, alpha=1, error=4.2)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        rows = summarize_runs(collect_runs(tmp_path))
    messages = [str(item.message) for item in caught]
    assert any("Missing cell" in message for message in messages)
    assert any("Mismatched ERM/mixup seeds" in message for message in messages)
    assert any("sample standard deviation is undefined" in message for message in messages)
    assert all(row["sample_std_err"] is None for row in rows)


def test_malformed_runs_skipped_invalid_metric_and_errors_rejected(tmp_path):
    directory = write_run(tmp_path)
    (directory / "summary.json").write_text("invalid")
    with pytest.warns(UserWarning, match="Skipping malformed"):
        assert collect_runs(tmp_path) == []
    with pytest.raises(ValueError, match="metric must be"):
        summarize_runs([], "invalid")
    write_run(tmp_path, "bad_error", error=float("nan"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        runs = collect_runs(tmp_path)
    with pytest.raises(ValueError, match="Invalid test error"):
        summarize_runs(runs)


def test_wrn_protocol_milestones_and_csv_json_cli(tmp_path):
    write_run(tmp_path, "wrn", config_updates={"model": "wideresnet28_10", "milestones": [60, 120, 180]}, error=3.8)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report = main([str(tmp_path), "--metric", "median_last10_err"])
    assert report["metric"] == "median_last10_err"
    saved = json.loads((tmp_path / "report" / "comparison.json").read_text())
    assert saved["rows"][0]["paper_err"] == 3.8
    with (tmp_path / "report" / "comparison.csv").open(newline="") as stream:
        row, = list(csv.DictReader(stream))
    assert row["paper_protocol"] == "True" and json.loads(row["seeds"]) == [0]
    assert float(row["mean_err"]) == 4.3


def test_wrn180_accepted_and_wrn160_rejected_even_with_paper_flag(tmp_path):
    write_run(tmp_path, "correct", seed=0,
              config_updates={"model": "wideresnet28_10", "milestones": [60, 120, 180]})
    write_run(tmp_path, "incorrect", seed=1,
              config_updates={"model": "wideresnet28_10", "milestones": [60, 120, 160]})
    with pytest.warns(UserWarning, match="milestones differs"):
        runs = collect_runs(tmp_path)
    assert [run["seed"] for run in runs] == [0]
    included = collect_runs(tmp_path, include_nonpaper=True)
    assert included[0]["paper_protocol"] is True
    assert included[1]["paper_protocol"] is False


def test_empty_or_only_excluded_runs_fail_cli_without_writing_report(tmp_path, capsys):
    with pytest.raises(SystemExit) as exc:
        main([str(tmp_path)])
    assert exc.value.code == 2
    assert "No eligible runs" in capsys.readouterr().err
    assert not (tmp_path / "report").exists()
    write_run(tmp_path, "smoke", config_updates={"epochs": 2}, summary_updates={"epochs_run": 2})
    with pytest.warns(UserWarning, match="Skipping nonpaper/partial"):
        with pytest.raises(SystemExit) as exc:
            main([str(tmp_path)])
    assert exc.value.code == 2
    assert not (tmp_path / "report").exists()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report = main([str(tmp_path), "--include-nonpaper"])
    assert report["rows"][0]["paper_protocol"] is False
    assert report["rows"][0]["protocol_deviations"]


def test_missing_or_null_selected_metric_reports_clear_validation_error(tmp_path):
    write_run(tmp_path, summary_updates={"best_err": None})
    with pytest.raises(ValueError, match="Missing or invalid best_err"):
        summarize_runs(collect_runs(tmp_path))
