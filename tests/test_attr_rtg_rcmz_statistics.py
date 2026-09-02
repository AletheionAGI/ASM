"""Deterministic synthetic goldens for ATTR-RTG-RCMZ V1 statistics."""

import pytest
import torch

from attr_rtg_rcmz import (
    TEMPERATURE_GRID,
    calibrated_probabilities,
    contrast_gate,
    decide,
    ece15,
    fit_temperature,
    h8_nll,
    paired_bootstrap,
    safe_q95,
    simultaneous_bounds,
    type7,
)
from attr_rtg_rcmz.bootstrap import hierarchical_weights


def test_temperature_nll_ece_and_type7_goldens():
    logits = torch.tensor([[[[-2.0, 2.0]]]], dtype=torch.float64)
    labels = torch.tensor([[[[0.0, 1.0]]]], dtype=torch.float64)
    temperature, scores = fit_temperature(logits, labels)
    assert TEMPERATURE_GRID == tuple(i / 4 for i in range(1, 17))
    assert temperature == 0.25 and scores.shape == (16,)
    probs = calibrated_probabilities(torch.zeros_like(logits), 1.0)
    assert h8_nll(probs, labels).item() == pytest.approx(0.6931471805599453)
    assert ece15(probs, labels).item() == pytest.approx(0.0)
    assert type7([0.0, 10.0], 0.25).item() == pytest.approx(2.5)
    assert safe_q95([0.1, 0.9, 0.3], [0, 1, 0]).item() == pytest.approx(0.29)


def test_decision_ties_abstain_and_fail_closed():
    assert decide([0.2] * 6, [1] * 6, 0.2).executed == "U"
    abstain = decide([0.3] * 6, [1] * 6, 0.2)
    assert (abstain.outcome, abstain.selected, abstain.executed, abstain.coverage) == (
        "ABSTAIN",
        "U",
        "BRAKE",
        0,
    )
    assert decide([0.1] * 6, [1, 1, 1, 1, 0, 1], 0.2).outcome == "BLOCK"
    assert decide([float("nan")] + [0.1] * 5, [1] * 6, 0.2).outcome == "BLOCK"


def test_shared_hierarchical_bootstrap_and_frozen_bounds():
    shape = (5, 3, 2, 2)
    base = torch.arange(torch.tensor(shape).prod(), dtype=torch.float64).reshape(shape)
    endpoints = {"R": base, "CM": base - 1, "Z": base + 2, "T": base + 3}
    first = paired_bootstrap(endpoints, replicates=1000, device="cpu", seed=7)
    second = paired_bootstrap(endpoints, replicates=1000, device="cpu", seed=7)
    assert torch.equal(first["CM-R"], second["CM-R"])
    assert torch.allclose(
        first["CM-R"], torch.full((1000, 3), -1.0, dtype=torch.float64)
    )
    lower, upper = simultaneous_bounds(first["CM-R"])
    assert torch.allclose(lower, torch.full((3,), -1.0, dtype=torch.float64))
    assert torch.allclose(lower, upper)
    sw, ww, ew = hierarchical_weights(shape, replicates=2, device="cpu", seed=9)
    assert torch.allclose(sw.sum(-1), torch.ones(2, dtype=torch.float64))
    assert torch.allclose(ww.sum(-1), torch.ones((2, 5, 3), dtype=torch.float64))
    assert torch.allclose(ew.sum(-1), torch.ones((2, 5, 3, 2), dtype=torch.float64))


def test_exact_gate_thresholds_and_all_marginals():
    lower = torch.tensor([[-0.2, -0.2, -0.1, -0.02, -0.02]] * 3, dtype=torch.float64)
    upper = torch.tensor([[-0.1, -0.1, 0.02, 0.1, 0.1]] * 3, dtype=torch.float64)
    raw = torch.tensor(
        [[[-0.1, -0.1, 0.02, -0.02, -0.02]] * 3] * 5, dtype=torch.float64
    )
    assert contrast_gate(lower, upper, raw).passed
    raw[4, 2, 0] = 0
    result = contrast_gate(lower, upper, raw)
    assert not result.passed and result.bounds_pass and not result.marginals_pass
