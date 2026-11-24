"""Section 3.2 planning, safe resume, and real Trainer integration; no downloads."""
from __future__ import annotations

import csv
import itertools
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cifar import preact_resnet18  # noqa: E402
from experiments import exp32_cifar as experiment  # noqa: E402
from cifar import models  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def single_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def planned(*argv):
    parser = experiment.build_parser()
    args = parser.parse_args(list(argv))
    experiment.validate_args(parser, args)
    return experiment.plan_runs(args)


@pytest.mark.parametrize("seeds", [(0,), (0, 1, 2)])
def test_paper_grid_has_all_configurations_and_model_specific_schedule(seeds, monkeypatch, capsys):
    def no_dataset(*args, **kwargs):
        pytest.fail("dry-run must not construct or download a dataset")

    monkeypatch.setattr(experiment, "build_cifar", no_dataset)
    runs = experiment.main(["--suite", "--seeds", *map(str, seeds), "--dry-run"])
    assert len(runs) == 12 * len(seeds)
    assert len({run.run_name for run in runs}) == len(runs)
    assert {(r.dataset, r.model, r.alpha, r.seed) for r in runs} == set(itertools.product(
        ("cifar10", "cifar100"), models.MODEL_NAMES, (0.0, 1.0), seeds))
    assert len(json.loads(capsys.readouterr().out)) == len(runs)
    for run in runs:
        assert run.paper_protocol and run.protocol_deviations == []
        assert (run.epochs, run.batch_size, run.lr, run.wd) == (200, 128, 0.1, 1e-4)
        assert run.gamma == 0.1
        assert run.milestones == ([60, 120, 180] if run.model == "wideresnet28_10" else [100, 150])


def test_default_model_reuses_shared_factory(monkeypatch):
    assert planned()[0].model == "preact_resnet18"
    assert models.preact_resnet18 is preact_resnet18
    calls = []
    sentinel = object()

    def shared_factory(num_classes):
        calls.append(num_classes)
        return sentinel

    monkeypatch.setattr(models, "preact_resnet18", shared_factory)
    assert experiment.build_model("preact_resnet18", 100) is sentinel
    assert calls == [100]


def test_suite_child_command_preserves_all_single_run_arguments(tmp_path):
    run, = planned(
        "--suite", "--datasets", "cifar100", "--models", "wideresnet28_10",
        "--methods", "mixup", "--seeds", "7", "--data-dir", str(tmp_path / "data with spaces"),
        "--run-name", "suite with spaces", "--milestones", "11", "17", "--no-download",
        "--deterministic", "--amp", "--no-resume", "--wandb", "--no-augment",
        "--official-lr-offset", "--per-sample-lambda", "--latent-module", "layer1",
        "--pairing", "same_class", "--label-mode", "hard", "--smoothing", ".05",
        "--input-noise", ".01", "--mix-warmup", "3",
    )
    command = experiment.child_command(run)
    assert command[0] == sys.executable
    assert Path(command[1]).resolve() == Path(experiment.__file__).resolve()
    reconstructed, = planned(*command[2:])
    assert vars(reconstructed) == vars(run)


@pytest.mark.parametrize("argv", [
    ["--epochs", "0"], ["--batch-size", "0"], ["--test-batch-size", "0"],
    ["--workers", "-1"], ["--seed", "-1"], ["--alpha", "-1"],
    ["--gamma", "0"], ["--momentum", "1"], ["--smoothing", "1"],
    ["--milestones", "-1"], ["--seeds", "0", "1"],
    ["--suite", "--seeds", "0", "0"], ["--suite", "--seeds", "-1"],
])
def test_invalid_arguments_are_rejected(argv):
    with pytest.raises(SystemExit) as error:
        planned(*argv)
    assert error.value.code == 2


@pytest.mark.parametrize("flag,value", [
    ("--lr", "nan"), ("--wd", "nan"), ("--alpha", "nan"), ("--input-noise", "nan"),
    ("--lr", "inf"), ("--alpha", "inf"), ("--momentum", "nan"), ("--gamma", "inf"),
])
def test_nonfinite_arguments_are_rejected(flag, value):
    with pytest.raises(SystemExit) as error:
        planned(flag, value)
    assert error.value.code == 2


def test_smoke_configuration_is_distinguished_from_paper_run():
    paper, = planned()
    smoke, = planned("--epochs", "2", "--limit-batches", "1")
    assert paper.paper_protocol
    assert not smoke.paper_protocol
    assert any("epochs=" in reason for reason in smoke.protocol_deviations)
    assert any("limit_batches=" in reason for reason in smoke.protocol_deviations)
    assert smoke.run_name != paper.run_name


@pytest.mark.parametrize("change", [
    ["--alpha", "0"], ["--lr", ".2"], ["--model", "densenet_bc190"],
    ["--milestones", "60", "120", "180"], ["--epochs", "199"], ["--no-resume"],
])
def test_incompatible_resume_preserves_every_existing_file(tmp_path, change, monkeypatch):
    common = ["--out-dir", str(tmp_path), "--run-name", "saved"]
    old, = planned(*common)
    directory = tmp_path / old.run_name
    directory.mkdir()
    (directory / "config.json").write_text(json.dumps({"config": vars(old)}), encoding="utf-8")
    (directory / "history.csv").write_text("epoch,test_err\n0,10\n", encoding="utf-8")
    (directory / "last.pt").write_bytes(b"checkpoint must stay untouched")
    before = {path.name: path.read_bytes() for path in directory.iterdir()}

    def no_dataset(*args, **kwargs):
        pytest.fail("resume compatibility must be checked before building data")

    monkeypatch.setattr(experiment, "build_cifar", no_dataset)
    changed, = planned(*common, *change)
    with pytest.raises(ValueError):
        experiment.run_experiment(changed)
    assert {path.name: path.read_bytes() for path in directory.iterdir()} == before


@pytest.mark.parametrize("name", ["..", "../outside", "."])
def test_run_name_stays_inside_output_directory(tmp_path, name):
    run, = planned("--out-dir", str(tmp_path), "--run-name", name)
    with pytest.raises(ValueError, match="run-name"):
        experiment.check_run_directory(run)


@pytest.fixture
def synthetic_cifar(monkeypatch):
    """Replace expensive inputs while retaining real SGD, Trainer and RunLogger."""
    calls = []

    def build_cifar(name, data_dir, batch_size, **kwargs):
        classes = 10 if name == "cifar10" else 100
        images = torch.arange(4 * 3 * 8 * 8, dtype=torch.float32).reshape(4, 3, 8, 8) / 512
        labels = torch.tensor([0, classes - 1, 1, 2])
        dataset = TensorDataset(images, labels)
        calls.append((name, kwargs))
        return SimpleNamespace(
            train=DataLoader(dataset, batch_size=batch_size, shuffle=False),
            test=DataLoader(dataset, batch_size=kwargs["test_batch_size"], shuffle=False),
            num_classes=classes,
        )

    def build_model(name, num_classes):
        return nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.ReLU(),
                             nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(4, num_classes))

    monkeypatch.setattr(experiment, "build_cifar", build_cifar)
    monkeypatch.setattr(experiment, "build_model", build_model)
    return calls


def smoke_args(tmp_path, name, epochs=1, dataset="cifar10", alpha=1):
    return ["--out-dir", str(tmp_path), "--run-name", name, "--device", "cpu",
            "--dataset", dataset, "--alpha", str(alpha), "--epochs", str(epochs),
            "--batch-size", "2", "--test-batch-size", "2", "--workers", "0",
            "--limit-batches", "1", "--no-download", "--seed", "3"]


@pytest.mark.parametrize("dataset,classes", [("cifar10", 10), ("cifar100", 100)])
@pytest.mark.parametrize("alpha", [0, 1])
def test_single_run_real_trainer_writes_complete_artifacts(tmp_path, synthetic_cifar, dataset, classes, alpha):
    summary = experiment.main(smoke_args(tmp_path, "smoke", dataset=dataset, alpha=alpha))
    directory = tmp_path / "smoke"
    assert {"config.json", "history.csv", "summary.json", "last.pt", "best_model.pt", "final_model.pt"} <= {
        path.name for path in directory.iterdir()}
    config = json.loads((directory / "config.json").read_text())["config"]
    assert config["num_classes"] == classes and config["alpha"] == alpha
    assert not config["paper_protocol"] and config["num_parameters"] > 0
    assert synthetic_cifar[0][0] == dataset and synthetic_cifar[0][1]["download"] is False
    assert json.loads((directory / "summary.json").read_text()) == summary
    assert summary["epochs_run"] == 1 and math.isfinite(summary["last_err"])
    checkpoint = torch.load(directory / "last.pt", map_location="cpu", weights_only=False)
    assert checkpoint["state"]["epoch"] == 1
    assert checkpoint["optimizer"]["state"]  # SGD momentum has actually been saved.
    assert checkpoint["scheduler"] is not None and "rng" in checkpoint and "np_rng" in checkpoint
    with (directory / "history.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1 and rows[0]["epoch"] == "0"


def test_real_resume_extends_history_and_restores_optimizer(tmp_path, synthetic_cifar):
    experiment.main(smoke_args(tmp_path, "resumed"))
    directory = tmp_path / "resumed"
    first_history = (directory / "history.csv").read_bytes()
    summary = experiment.main(smoke_args(tmp_path, "resumed", epochs=2))
    assert summary["epochs_run"] == 2
    assert (directory / "history.csv").read_bytes().startswith(first_history)
    with (directory / "history.csv").open(newline="") as stream:
        assert [row["epoch"] for row in csv.DictReader(stream)] == ["0", "1"]
    assert json.loads((directory / "config.json").read_text())["config"]["epochs"] == 2

    # A resumed run must match an uninterrupted run, including momentum buffers.
    experiment.main(smoke_args(tmp_path, "uninterrupted", epochs=2))
    resumed = torch.load(directory / "last.pt", map_location="cpu", weights_only=False)
    uninterrupted = torch.load(tmp_path / "uninterrupted" / "last.pt", map_location="cpu", weights_only=False)
    for key, value in resumed["model"].items():
        torch.testing.assert_close(value, uninterrupted["model"][key], rtol=0, atol=0)
    assert resumed["scheduler"] == uninterrupted["scheduler"]
    for index, state in resumed["optimizer"]["state"].items():
        torch.testing.assert_close(state["momentum_buffer"],
                                   uninterrupted["optimizer"]["state"][index]["momentum_buffer"],
                                   rtol=0, atol=0)
