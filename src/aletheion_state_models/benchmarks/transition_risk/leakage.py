"""Integrity checks that fail closed on common transition-risk leakage."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

FORBIDDEN_FEATURE_TOKENS = (
    "hidden_mode",
    "failure_countdown",
    "countdown",
    "distance_to_failure",
    "future_padding",
    "future_suffix",
    "simulator_seed",
    "seed_id",
    "generator_parameter",
)
ALLOWED_FIT_SPLITS = frozenset({"train", "validation", "val"})


@dataclass(frozen=True)
class FeatureAvailability:
    """Provenance of a single model input."""

    name: str
    available_at: int
    prediction_at: int
    source_split: str = "train"


@dataclass(frozen=True)
class LeakageViolation:
    code: str
    detail: str


@dataclass(frozen=True)
class LeakageAudit:
    violations: tuple[LeakageViolation, ...]

    @property
    def passed(self) -> bool:
        return not self.violations

    def require_pass(self) -> None:
        if self.violations:
            details = "; ".join(item.detail for item in self.violations)
            raise ValueError(f"transition-risk leakage audit failed: {details}")


def audit_feature_availability(
    features: Iterable[FeatureAvailability],
    fitted_artifact_splits: Mapping[str, str] | None = None,
    threshold_selection_split: str | None = None,
) -> LeakageAudit:
    """Audit causal availability, forbidden names, and fit/selection splits."""
    violations: list[LeakageViolation] = []
    for feature in features:
        normalized = feature.name.casefold().replace("-", "_").replace(" ", "_")
        token = next(
            (item for item in FORBIDDEN_FEATURE_TOKENS if item in normalized), None
        )
        if token is not None:
            violations.append(
                LeakageViolation(
                    "forbidden_feature", f"{feature.name!r} exposes {token}"
                )
            )
        if feature.available_at > feature.prediction_at:
            violations.append(
                LeakageViolation(
                    "future_feature",
                    f"{feature.name!r} is available at {feature.available_at}, after prediction {feature.prediction_at}",
                )
            )
        if feature.source_split.casefold() in {
            "test",
            "test-id",
            "test-shift",
            "test-ood",
        }:
            violations.append(
                LeakageViolation(
                    "test_input_provenance",
                    f"{feature.name!r} was fitted from {feature.source_split}",
                )
            )
    for artifact, split in (fitted_artifact_splits or {}).items():
        if split.casefold() not in ALLOWED_FIT_SPLITS:
            violations.append(
                LeakageViolation(
                    "test_fit", f"{artifact!r} was fitted on disallowed split {split!r}"
                )
            )
    if (
        threshold_selection_split is not None
        and threshold_selection_split.casefold() not in {"validation", "val"}
    ):
        violations.append(
            LeakageViolation(
                "threshold_selection",
                "alarm threshold must be selected on validation only",
            )
        )
    return LeakageAudit(tuple(violations))


def audit_episode_splits(
    episode_ids: Sequence[object], split_names: Sequence[str]
) -> LeakageAudit:
    """Reject episodes or transition windows assigned to multiple splits."""
    if len(episode_ids) != len(split_names):
        raise ValueError("episode_ids and split_names must have equal length")
    assigned: dict[object, str] = {}
    violations: list[LeakageViolation] = []
    for episode, split in zip(episode_ids, split_names):
        prior = assigned.setdefault(episode, split)
        if prior != split:
            violations.append(
                LeakageViolation(
                    "episode_split_overlap",
                    f"episode {episode!r} occurs in both {prior!r} and {split!r}",
                )
            )
    return LeakageAudit(tuple(violations))


def audit_leakage(
    feature_names: Iterable[str],
    feature_times: Iterable[int] | None = None,
    prediction_times: Iterable[int] | None = None,
) -> LeakageAudit:
    """Convenience audit for parallel arrays used by lightweight pipelines."""
    names = list(feature_names)
    available = list(feature_times) if feature_times is not None else [0] * len(names)
    predictions = list(prediction_times) if prediction_times is not None else available
    if not (len(names) == len(available) == len(predictions)):
        raise ValueError("feature audit arrays must have equal length")
    return audit_feature_availability(
        FeatureAvailability(name, at, prediction)
        for name, at, prediction in zip(names, available, predictions)
    )
