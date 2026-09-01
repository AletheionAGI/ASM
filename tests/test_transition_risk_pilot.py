import torch
import pytest
from aletheion_state_models.benchmarks.transition_risk.dataset import HazardEpisode
from aletheion_state_models.benchmarks.transition_risk.pilot import (
    event_prevalence,
    write_pilot_manifest,
)


def _episode(labels, unsafe):
    steps = len(labels)
    return HazardEpisode(
        "episode",
        "world",
        torch.zeros(steps * 4, dtype=torch.long),
        torch.arange(3, steps * 4, 4),
        torch.zeros(steps, 6),
        torch.tensor(labels, dtype=torch.float32),
        torch.zeros(steps),
        torch.zeros(steps),
        torch.tensor(unsafe),
        ("BRAKE",) * steps,
    )


def test_event_prevalence_counts_each_horizon_and_unsafe_episode():
    episodes = [
        _episode([[0, 1, 1, 1], [0, 0, 1, 1]], [False, True]),
        _episode([[0, 0, 0, 0]], [False]),
    ]
    result = event_prevalence(episodes)
    assert result["episodes"] == 2
    assert result["transitions"] == 3
    assert result["unsafe_episodes"] == 1
    assert result["by_horizon"]["8"] == {
        "positives": 2,
        "prevalence": pytest.approx(2 / 3),
    }


def test_manifest_fails_closed_if_test_was_generated(tmp_path):
    summary = {"test_worlds_generated": True}
    with pytest.raises(ValueError, match="sealed"):
        write_pilot_manifest(tmp_path, tmp_path, summary)
