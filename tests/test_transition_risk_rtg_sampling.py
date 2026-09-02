"""Synthetic checks for the frozen ATTR-RTG CRN sampler."""

import hashlib

import pytest
import torch

from aletheion_state_models.benchmarks.transition_risk.rtg_sampling import (
    RISK_SAMPLES,
    crn_uniform,
    estimate_g_risk,
    inverse_cdf,
    smoothed_g_logit,
)
from aletheion_state_models.benchmarks.transition_risk.rtg_types import (
    Y_COMMON_CARDINALITIES,
)


def _peaked_logits(categories: tuple[int, ...]) -> torch.Tensor:
    groups = []
    for cardinality, category in zip(Y_COMMON_CARDINALITIES, categories):
        values = torch.full((cardinality,), -1000.0)
        values[category] = 1000.0
        groups.append(values)
    return torch.cat(groups)


def test_crn_matches_registered_sha_key_and_is_open() -> None:
    fields = ("toy-split", 29, "world-1", "episode-2", 3, 4, 5, 6)
    key = "ATTR-RTG-RISK-V1|toy-split|29|world-1|episode-2|3|4|5|6"
    integer = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big")
    expected = (integer + 0.5) / 2**64
    actual = crn_uniform(*fields)
    assert actual == expected
    assert 0.0 < actual < 1.0


def test_inverse_cdf_uses_strict_boundary_and_float64_cpu() -> None:
    logits = torch.tensor([0.0, 0.0], dtype=torch.float32)
    assert inverse_cdf(logits, 0.5) == 1
    assert inverse_cdf(logits, 0.25) == 0


def test_joint_risk_is_exact_for_peaked_safe_and_unsafe_targets() -> None:
    identity = {
        "split_id": "toy-split", "training_seed": 29, "world_id": "w",
        "episode_id": "e", "t": 1, "action_index": 0, "failure_delay": 3,
    }
    safe = (0, 1, 2, 3, 4, 1, 1, 63, 0, 0, 0)
    unsafe = (0, 1, 2, 0, 4, 1, 1, 63, 0, 0, 0)
    assert estimate_g_risk(_peaked_logits(safe), **identity) == (0.0, 0)
    assert estimate_g_risk(_peaked_logits(unsafe), **identity) == (1.0, RISK_SAMPLES)


def test_malformed_schema_fails_closed_and_k_is_frozen() -> None:
    identity = {
        "split_id": "x", "training_seed": 29, "world_id": "w",
        "episode_id": "e", "t": 1, "action_index": 0, "failure_delay": 3,
    }
    assert estimate_g_risk(torch.zeros(484), **identity) == (1.0, RISK_SAMPLES)
    with pytest.raises(ValueError, match="K=128"):
        estimate_g_risk(torch.zeros(485), samples=127, **identity)
    assert smoothed_g_logit(0) < 0 < smoothed_g_logit(128)
