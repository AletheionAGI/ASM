"""Pure frozen-head evaluation of pre-materialized ATTR-RTG candidate records."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from .rtg_calibration import RtgCalibration, expected_calibration_error
from .rtg_decision import decide
from .rtg_metrics_governance import governance_metrics
from .rtg_metrics_state import (
    consequence_macro_accuracy,
    consequence_nll,
    transition_state_metrics,
    validate_six_candidate_clusters,
)
from .rtg_sampling import estimate_g_risk, smoothed_g_logit, split_group_logits
from .rtg_types import Y_COMMON_CARDINALITIES

Record = Mapping[str, Any]
Head = Callable[[torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class EvaluationBatch:
    records: tuple[dict[str, Any], ...]
    metrics: Mapping[str, float]


def _vector(value: Any, size: int, name: str) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float32, device="cpu")
    if tensor.shape != (size,) or not torch.isfinite(tensor).all():
        raise ValueError(f"{name} must be a finite vector of size {size}")
    return tensor


def _frozen_vector(head: Head, inputs: torch.Tensor, size: int, name: str) -> torch.Tensor:
    if isinstance(head, torch.nn.Module) and head.training:
        raise ValueError(f"{name} must be in eval mode")
    with torch.no_grad():
        output = head(inputs.unsqueeze(0))
    tensor = torch.as_tensor(output).detach().cpu()
    if tensor.shape == (1, size):
        tensor = tensor[0]
    if tensor.shape != (size,) or not torch.isfinite(tensor).all():
        raise ValueError(f"{name} returned an invalid vector")
    return tensor


def _input(row: Record) -> torch.Tensor:
    state = _vector(row.get("normalized_state"), 28, "normalized_state")
    frame = _vector(row.get("fixed_frame"), 32, "fixed_frame")
    return torch.cat((state, frame))


def _identity(row: Record) -> dict[str, Any]:
    required = ("seed", "split_id", "world_id", "episode_id", "t", "action_index")
    if any(name not in row for name in required):
        raise ValueError("record is missing a CRN identity field")
    return {
        "split_id": row["split_id"],
        "training_seed": row["seed"],
        "world_id": row["world_id"],
        "episode_id": row["episode_id"],
        "t": row["t"],
        "action_index": row["action_index"],
    }


def _truth(row: Record) -> tuple[int, ...]:
    raw = row.get("y_common")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or len(raw) != 11:
        raise ValueError("G evaluation requires eleven y_common categories")
    truth = tuple(raw)
    if any(type(value) is not int or value not in range(cardinality)
           for value, cardinality in zip(truth, Y_COMMON_CARDINALITIES)):
        raise ValueError("y_common category is outside its registered group")
    return truth


def _physical_scores(logits: torch.Tensor, truth: Sequence[int]) -> tuple[list[float], list[int]]:
    nll, predictions = [], []
    for group, target in zip(split_group_logits(logits), truth):
        log_probabilities = torch.log_softmax(group, dim=0)
        value = float(-log_probabilities[target])
        if not math.isfinite(value):
            raise ValueError("physical NLL is nonfinite")
        nll.append(value)
        predictions.append(int(torch.argmax(group).item()))
    return nll, predictions




def _persistence_scores(
    persisted: Sequence[int], truth: Sequence[int]
) -> tuple[list[float], list[int]]:
    if len(persisted) != 11:
        raise ValueError("physical persistence requires eleven categories")
    nll = []
    for category, target, cardinality in zip(
        persisted, truth, Y_COMMON_CARDINALITIES, strict=True
    ):
        if type(category) is not int or category not in range(cardinality):
            raise ValueError("physical persistence category is outside its group")
        probability = 1 - (cardinality - 1) * 1e-4 if category == target else 1e-4
        nll.append(-math.log(probability))
    return nll, list(persisted)


def _common_output(row: Record, risk: float, calibration: RtgCalibration) -> dict[str, Any]:
    calibrated = calibration.probability(risk)
    item = dict(row)
    item.update({"raw_calibration_logit": risk, "risk": calibrated,
                 "decision": decide(calibrated, calibration.q95).value})
    return item


def evaluate_g_records(
    records: Iterable[Record], g: Head, d: Head, calibration: RtgCalibration,
) -> EvaluationBatch:
    """Evaluate G→D with frozen CRN on records; never create or advance worlds."""
    rows = validate_six_candidate_clusters(records)
    output = []
    for row in rows:
        predicted = _frozen_vector(g, _input(row), 28, "G")
        logits = _frozen_vector(d, predicted, 485, "D")
        identity = _identity(row)
        risk, hits = estimate_g_risk(
            logits, failure_delay=row.get("failure_delay"), **identity
        )
        item = _common_output(row, smoothed_g_logit(hits), calibration)
        truth = _truth(row)
        group_nll, group_predictions = _physical_scores(logits, truth)
        if "true_next_state" not in row or "persistence_state" not in row:
            raise ValueError("G evaluation requires true and persistence normalized states")
        true_state = _vector(row["true_next_state"], 28, "true_next_state")
        true_logits = _frozen_vector(d, true_state, 485, "D")
        d_group_nll, d_group_predictions = _physical_scores(true_logits, truth)
        persistence_nll, persistence_predictions = _persistence_scores(
            row.get("persistence_target"), truth
        )
        item.update({
            "predicted_state": predicted.tolist(),
            "raw_risk": risk,
            "risk_hits": hits,
            "group_nll": group_nll,
            "group_predictions": group_predictions,
            "d_group_nll": d_group_nll,
            "d_group_predictions": d_group_predictions,
            "persistence_nll": persistence_nll,
            "persistence_predictions": persistence_predictions,
            "group_targets": list(truth),
            "true_state": true_state.tolist(),
            "persistence_state": _vector(
                row["persistence_state"], 28, "persistence_state"
            ).tolist(),
        })
        output.append(item)
    metrics = governance_metrics(output)
    metrics.update(transition_state_metrics(output))
    metrics["consequence_nll"] = consequence_nll(output)
    metrics["physical_persistence_nll"] = consequence_nll(
        output, field="persistence_nll"
    )
    metrics["d_true_next_nll"] = consequence_nll(output, field="d_group_nll")
    metrics["d_macro_accuracy"] = consequence_macro_accuracy(output)
    metrics["d_true_next_macro_accuracy"] = consequence_macro_accuracy(
        output, prediction_field="d_group_predictions"
    )
    metrics["physical_persistence_macro_accuracy"] = consequence_macro_accuracy(
        output, prediction_field="persistence_predictions"
    )
    metrics["ece"] = expected_calibration_error(
        [item["risk"] for item in output],
        [int(item["candidate_unsafe"]) for item in output],
    )
    return EvaluationBatch(tuple(output), metrics)


def evaluate_c_records(
    records: Iterable[Record], c: Head, calibration: RtgCalibration,
) -> EvaluationBatch:
    """Evaluate direct C logits and the common frozen decision/accounting rule."""
    rows = validate_six_candidate_clusters(records)
    output = []
    for row in rows:
        logit = float(_frozen_vector(c, _input(row), 1, "C")[0])
        output.append(_common_output(row, logit, calibration))
    metrics = governance_metrics(output)
    metrics["ece"] = expected_calibration_error(
        [item["risk"] for item in output],
        [int(item["candidate_unsafe"]) for item in output],
    )
    return EvaluationBatch(tuple(output), metrics)
