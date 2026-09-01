"""Pure rank curriculum schedule for ASM-VR Phase 2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class RankCurriculumConfig(Protocol):
    d_state: int
    variable_rank_target_fraction: float
    variable_rank_temperature_initial: float
    variable_rank_temperature_final: float
    variable_rank_warmup_steps: int
    variable_rank_budget_ramp_steps: int
    variable_rank_hardening_steps: int
    lambda_variable_rank_budget: float
    lambda_variable_rank_binary: float
    lambda_variable_rank_switch: float


@dataclass(frozen=True)
class RankCurriculumState:
    target_rank: float
    temperature: float
    budget_weight: float
    binary_weight: float
    switch_weight: float


def _progress(step: int, start: int, duration: int) -> float:
    if duration == 0:
        return 1.0
    return min(max((step - start) / duration, 0.0), 1.0)


def phase2_rank_curriculum(
    step: int, config: RankCurriculumConfig
) -> RankCurriculumState:
    """Return monotonic warm-up, budget-ramp, and hardening controls."""
    if step < 0:
        raise ValueError("step must be non-negative")
    width = float(config.d_state)
    final_target = width * config.variable_rank_target_fraction
    warmup_end = config.variable_rank_warmup_steps
    ramp_end = warmup_end + config.variable_rank_budget_ramp_steps
    hardening_end = ramp_end + config.variable_rank_hardening_steps
    if step < warmup_end:
        return RankCurriculumState(width, config.variable_rank_temperature_initial, 0.0, 0.0, 0.0)
    if step < ramp_end:
        progress = _progress(step, warmup_end, config.variable_rank_budget_ramp_steps)
        target = width + progress * (final_target - width)
        return RankCurriculumState(
            target,
            config.variable_rank_temperature_initial,
            progress * config.lambda_variable_rank_budget,
            0.0,
            progress * config.lambda_variable_rank_switch,
        )
    progress = _progress(step, ramp_end, config.variable_rank_hardening_steps)
    temperature = config.variable_rank_temperature_initial + progress * (
        config.variable_rank_temperature_final
        - config.variable_rank_temperature_initial
    )
    return RankCurriculumState(
        final_target,
        temperature,
        config.lambda_variable_rank_budget,
        progress * config.lambda_variable_rank_binary,
        config.lambda_variable_rank_switch,
    )


__all__ = ["RankCurriculumState", "phase2_rank_curriculum"]
