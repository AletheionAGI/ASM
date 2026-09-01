"""Small dependency-free prediction and early-warning metrics."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from math import fsum


def _pairs(
    labels: Iterable[int | bool], scores: Iterable[float]
) -> list[tuple[int, float]]:
    y = [int(value) for value in labels]
    p = [float(value) for value in scores]
    if len(y) != len(p) or not y:
        raise ValueError("labels and scores must be non-empty and have equal length")
    if any(value not in (0, 1) for value in y):
        raise ValueError("labels must be binary")
    if any(not 0.0 <= value <= 1.0 for value in p):
        raise ValueError("scores must be probabilities in [0, 1]")
    return list(zip(y, p))


def average_precision(labels: Iterable[int | bool], scores: Iterable[float]) -> float:
    """Compute step-wise area under the precision-recall curve (AUPRC/AP)."""
    pairs = _pairs(labels, scores)
    positives = sum(label for label, _ in pairs)
    if positives == 0:
        return 0.0
    pairs.sort(key=lambda item: item[1], reverse=True)
    true_positives = 0
    area = 0.0
    for rank, (label, _score) in enumerate(pairs, start=1):
        if label:
            true_positives += 1
            area += true_positives / rank
    return area / positives


def auprc(labels: Iterable[int | bool], scores: Iterable[float]) -> float:
    """Alias for :func:`average_precision`."""
    return average_precision(labels, scores)


def brier_score(labels: Iterable[int | bool], probabilities: Iterable[float]) -> float:
    """Return mean squared probabilistic error."""
    pairs = _pairs(labels, probabilities)
    return fsum((probability - label) ** 2 for label, probability in pairs) / len(pairs)


def recall_at_false_positive_rate(
    labels: Iterable[int | bool], scores: Iterable[float], max_fpr: float = 0.01
) -> tuple[float, float]:
    """Return best attainable recall and its threshold under an FPR budget."""
    if not 0.0 <= max_fpr <= 1.0:
        raise ValueError("max_fpr must be in [0, 1]")
    pairs = _pairs(labels, scores)
    positives = sum(label for label, _ in pairs)
    negatives = len(pairs) - positives
    if positives == 0:
        return 0.0, 1.0
    thresholds = sorted({score for _, score in pairs}, reverse=True)
    best_recall, best_threshold = 0.0, 1.0
    for threshold in thresholds:
        tp = sum(label for label, score in pairs if score >= threshold)
        fp = sum(1 - label for label, score in pairs if score >= threshold)
        fpr = fp / negatives if negatives else 0.0
        recall = tp / positives
        if fpr <= max_fpr and recall > best_recall:
            best_recall, best_threshold = recall, threshold
    return best_recall, best_threshold


def first_sustained_alarm(
    scores: Sequence[float],
    threshold: float,
    sustained_steps: int = 1,
    stop: int | None = None,
) -> int | None:
    """Find the first run of alarms that starts before ``stop``."""
    if sustained_steps < 1:
        raise ValueError("sustained_steps must be positive")
    limit = len(scores) if stop is None else min(stop + 1, len(scores))
    for start in range(max(0, limit - sustained_steps + 1)):
        if all(
            float(scores[index]) >= threshold
            for index in range(start, start + sustained_steps)
        ):
            return start
    return None


def useful_lead_time(
    scores: Sequence[float],
    event_step: int,
    last_effective_intervention_step: int,
    threshold: float,
    sustained_steps: int = 1,
) -> int | None:
    """Steps from the first sustained actionable alarm to an unsafe entry.

    ``None`` means no alarm began by the last effective intervention time.
    """
    if not 0 <= last_effective_intervention_step < event_step <= len(scores):
        raise ValueError("expected intervention_step < event_step <= len(scores)")
    alarm = first_sustained_alarm(
        scores, threshold, sustained_steps, stop=last_effective_intervention_step
    )
    return None if alarm is None else event_step - alarm


@dataclass(frozen=True)
class BasicRiskMetrics:
    auprc: float
    brier: float


def basic_risk_metrics(
    labels: Iterable[int | bool], scores: Iterable[float]
) -> BasicRiskMetrics:
    """Compute the two basic ATTR-P0 probabilistic metrics."""
    y, p = zip(*_pairs(labels, scores))
    return BasicRiskMetrics(auprc=average_precision(y, p), brier=brier_score(y, p))
