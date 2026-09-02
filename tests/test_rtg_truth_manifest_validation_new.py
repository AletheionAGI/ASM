import pytest
import torch

from aletheion_state_models.benchmarks.transition_risk.rtg_artifacts import (
    canonical_sha256,
)
from aletheion_state_models.benchmarks.transition_risk.rtg_truth_manifest_validation import (
    _attached_truth_sha256,
    _split_counts,
    _verify_content,
)


def rows(origin_count):
    return {
        "origin_count": origin_count,
        "candidate_count": origin_count * 6,
    }


def test_truth_counts_must_match_allowed_and_classes():
    allowed = {"splits": [
        {"name": name, "audit": rows(count)}
        for name, count in (("train", 500), ("validation", 100), ("calibration", 100))
    ]}
    truth = {"splits": [
        {"name": name, **rows(count), "positive_labels": count * 3,
         "negative_labels": count * 3, "truth_sha256": "a" * 64}
        for name, count in (("train", 500), ("validation", 100), ("calibration", 100))
    ]}
    assert _split_counts(allowed, truth)["train"] == 3000
    truth["splits"][0]["candidate_count"] -= 1
    with pytest.raises(ValueError):
        _split_counts(allowed, truth)


def test_truth_content_digest_is_recomputed():
    payload = {"schema": "ATTR-RTG-TRUTH-MANIFEST-V1", "splits": []}
    payload["content_sha256"] = canonical_sha256(payload)
    _verify_content(payload, "ATTR-RTG-TRUTH-MANIFEST-V1")
    payload["splits"].append({"stale": True})
    with pytest.raises(ValueError, match="content digest"):
        _verify_content(payload, "ATTR-RTG-TRUTH-MANIFEST-V1")


def test_attached_ledger_truth_digest_is_recomputed_from_values():
    safe = [0, 1, 2, 3, 4, 1, 1, 10, 0, 0, 0]
    payload = {
        "identity": [["train", "w", "e", 1, index] for index in range(6)],
        "physical_target": torch.tensor([safe] * 6),
        "persistence_target": torch.tensor([safe] * 6),
        "unsafe": torch.zeros(6, dtype=torch.bool),
        "failure_delay": torch.full((6,), 3),
    }
    digest = _attached_truth_sha256(payload)
    payload["physical_target"][0, 3] = 0
    assert _attached_truth_sha256(payload) != digest
