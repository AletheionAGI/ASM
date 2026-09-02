"""Synthetic checks for disjoint ATTR-RTG calibration."""

import math

import pytest

from aletheion_state_models.benchmarks.transition_risk.rtg_calibration import (
    RtgCalibration,
    calibrated_probabilities,
    empirical_q95,
    expected_calibration_error,
    fit_disjoint_calibration,
    fit_temperature,
    partition_calibration_worlds,
)


def test_world_partition_is_sorted_and_disjoint() -> None:
    worlds = [f"w{index:02d}" for index in reversed(range(16))]
    temperature, residual = partition_calibration_worlds(worlds)
    assert temperature == tuple(f"w{index:02d}" for index in range(8))
    assert residual == tuple(f"w{index:02d}" for index in range(8, 16))
    assert set(temperature).isdisjoint(residual)


def test_integer_world_ids_use_native_numeric_order() -> None:
    temperature, residual = partition_calibration_worlds(list(reversed(range(16))))
    assert temperature == tuple(range(8))
    assert residual == tuple(range(8, 16))
    with pytest.raises(ValueError, match="homogeneous"):
        partition_calibration_worlds([*range(15), "15"])


def test_temperature_uses_frozen_grid_and_improves_over_large_t() -> None:
    labels = [0, 1] * 30
    logits = [-4.0, 4.0] * 30
    temperature = fit_temperature(logits, labels, origin_count=50)
    assert math.isclose(math.log(temperature), -4.0, abs_tol=1e-12)
    probabilities = calibrated_probabilities(logits, temperature)
    assert max(probabilities[::2]) < 1e-10
    assert min(probabilities[1::2]) > 1 - 1e-10


def test_q95_uses_registered_n_plus_one_order_statistic() -> None:
    labels = [0] * 45 + [1] * 15
    probabilities = [0.1] * 45 + [0.2] * 15
    assert empirical_q95(probabilities, labels, origin_count=50) == pytest.approx(0.8)


def test_orchestrator_enforces_disjoint_canonical_halves() -> None:
    labels = [0, 1] * 30
    temperature_worlds = [index % 8 for index in range(60)]
    residual_worlds = [8 + index % 8 for index in range(60)]
    calibration = fit_disjoint_calibration(
        [-2.0, 2.0] * 30,
        labels,
        temperature_worlds,
        [0.0] * 60,
        labels,
        residual_worlds,
        temperature_origin_count=50,
        residual_origin_count=50,
    )
    assert calibration.q95 == pytest.approx(0.5)
    with pytest.raises(ValueError, match="canonical"):
        fit_disjoint_calibration(
            [-2.0, 2.0] * 30,
            labels,
            residual_worlds,
            [0.0] * 60,
            labels,
            temperature_worlds,
            temperature_origin_count=50,
            residual_origin_count=50,
        )


def test_ece15_handles_edges_and_weights_by_count() -> None:
    probabilities = [0.0, 1.0, 0.5, 0.5]
    labels = [0, 1, 0, 1]
    assert expected_calibration_error(probabilities, labels) == pytest.approx(0.0)


def test_calibration_half_requirements_fail_closed() -> None:
    with pytest.raises(ValueError, match="50 origins"):
        fit_temperature([0.0] * 30, [0, 1] * 15, origin_count=49)
    with pytest.raises(ValueError, match="15 labels"):
        empirical_q95([0.2] * 60, [0] * 50 + [1] * 10, origin_count=50)


def test_calibration_artifact_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="temperature"):
        RtgCalibration(0.0, 0.5)
    with pytest.raises(ValueError, match="q95"):
        RtgCalibration(1.0, float("nan"))
