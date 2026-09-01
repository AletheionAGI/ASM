import pytest

from aletheion_state_models.benchmarks.transition_risk import (
    FeatureAvailability,
    audit_episode_splits,
    audit_feature_availability,
)


def test_leakage_audit_detects_privileged_future_and_test_fit():
    audit = audit_feature_availability(
        [
            FeatureAvailability("hidden_mode", 3, 3),
            FeatureAvailability("local_sensor", 4, 3),
        ],
        {"normalizer": "test"},
        threshold_selection_split="test-ood",
    )
    assert not audit.passed
    assert {item.code for item in audit.violations} == {
        "forbidden_feature",
        "future_feature",
        "test_fit",
        "threshold_selection",
    }
    with pytest.raises(ValueError, match="leakage audit failed"):
        audit.require_pass()


def test_episode_split_audit():
    assert audit_episode_splits([1, 1, 2], ["train", "train", "test"]).passed
    assert not audit_episode_splits([1, 1], ["train", "test"]).passed
