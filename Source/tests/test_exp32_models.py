"""Architecture checks without downloading CIFAR or training full-size models."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cifar.models import PreActResNet  # noqa: E402
from cifar.models import (  # noqa: E402
    DenseNetBC, WideResNet, _WideBlock, build_model,
)


@pytest.mark.parametrize("name,classes,parameters", [
    ("preact_resnet18", 10, 11_171_274),
    ("preact_resnet18", 100, 11_217_444),
    ("wideresnet28_10", 10, 36_479_194),
    ("wideresnet28_10", 100, 36_536_884),
    ("densenet_bc190", 10, 25_624_430),
    ("densenet_bc190", 100, 25_821_620),
])
def test_paper_architecture_sizes_and_outputs(name, classes, parameters):
    # Meta tensors check all full-size layers and shapes without allocating weights.
    with torch.device("meta"):
        model = build_model(name, classes).eval()
        logits = model(torch.empty(2, 3, 32, 32))
    assert sum(p.numel() for p in model.parameters()) == parameters
    assert logits.shape == (2, classes)
    assert not any(isinstance(layer, nn.Dropout) for layer in model.modules())
    if name == "preact_resnet18":
        assert isinstance(model, PreActResNet)


@pytest.mark.parametrize("factory", [
    lambda: WideResNet(depth=10, width=1, num_classes=100),
    lambda: DenseNetBC(depth=10, growth=4, num_classes=100),
])
def test_small_architectures_forward_and_backward(factory):
    model = factory()
    logits = model(torch.randn(2, 3, 32, 32))
    assert logits.shape == (2, 100)
    nn.functional.cross_entropy(logits, torch.tensor([0, 99])).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_wrn_identity_shortcut_preserves_raw_input():
    block = _WideBlock(16, 16, 1).eval()
    nn.init.zeros_(block.conv1.weight)
    nn.init.zeros_(block.conv2.weight)
    x = torch.randn(2, 16, 8, 8) - 1
    assert torch.equal(block(x), x)


def test_wrn_reference_batchnorm_initialization():
    torch.manual_seed(0)
    model = WideResNet(depth=10, width=1)
    scales = torch.cat([m.weight.detach() for m in model.modules() if isinstance(m, nn.BatchNorm2d)])
    assert torch.all((scales >= 0) & (scales <= 1))
    assert scales.std() > 0.1


@pytest.mark.parametrize("factory", [
    lambda: WideResNet(depth=18),
    lambda: WideResNet(width=0),
    lambda: DenseNetBC(depth=40 + 1),
    lambda: DenseNetBC(growth=0),
    lambda: DenseNetBC(reduction=0),
    lambda: build_model("unknown"),
])
def test_invalid_model_configuration(factory):
    with pytest.raises(ValueError):
        factory()
