from __future__ import annotations

from copy import deepcopy

import pytest

from aletheion_state_models.benchmarks.transition_risk.rtg_evaluator import (
    EvaluationBatch,
)
from aletheion_state_models.benchmarks.transition_risk.rtg_gates import (
    evaluate_rtg3_regimes,
    gate_rtg2_absolute,
    gate_rtg3,
    gate_rtg3_comparative,
)
from aletheion_state_models.benchmarks.transition_risk.rtg_pairing import (
    canonical_records,
    require_byte_equivalent,
)
from aletheion_state_models.benchmarks.transition_risk.rtg_registered_evaluation import (
    _batch_payload,
)

SEEDS = (29, 43, 71, 89, 107)


def _bootstrap():
    return {"bootstrap_seed": 20260903, "replicates": 1000,
            "bit_generator": "PCG64", "cluster_order": "seed-world-episode"}


def _absolute():
    seed = {"reduction": 0.2, "relative_reduction": 0.6,
            "safe_service": 0.95, "coverage": 0.9}
    return {"relative_reduction": 0.6, "reduction_ci95": [0.1, 0.3],
            "safe_service": 0.96, "safe_service_ci95": [0.94, 0.98],
            "coverage": 0.9, "coverage_ci95": [0.8, 0.95],
            "per_seed": {seed: dict(seed_metrics) for seed, seed_metrics in
                         ((value, seed) for value in SEEDS)},
            "bootstrap": _bootstrap()}


def _comparative():
    seed = {"delta_safety": 0.03, "delta_safe_service": -0.01,
            "coverage_difference": 0.01}
    return {"delta_safety": 0.03, "delta_safety_ci95": [0.01, 0.05],
            "delta_safe_service_ci95": [-0.015, 0.01],
            "coverage_difference": 0.01, "per_seed": {value: seed for value in SEEDS},
            "bootstrap": _bootstrap()}


def _regime():
    z_seed = {seed: {"transition_nmse": 0.8} for seed in SEEDS}
    y_seed = {seed: {"delta_nll": -0.2} for seed in SEEDS}
    return {
        "d_fidelity": {"macro_accuracy": 0.8, "persistence_macro_accuracy": 0.7,
                       "nll": 1.0, "bootstrap": _bootstrap()},
        "rtg1_z": {"mse_state_persistence": 1.0, "transition_nmse": 0.8,
                   "transition_nmse_ci95": [0.7, 0.9], "per_seed": z_seed,
                   "bootstrap": _bootstrap()},
        "rtg1_y": {"nll": 0.7, "persistence_nll": 1.0, "ece": 0.01,
                   "delta_nll_ci95": [-0.3, -0.1], "per_seed": y_seed,
                   "bootstrap": _bootstrap()},
        "rtg2_g": _absolute(), "rtg2_c": _absolute(), "rtg2_v": _comparative(),
    }


def _row(action: int):
    return {"seed": 29, "world_id": "w", "episode_id": "e", "t": 1,
            "action_index": action, "fixed_frame": [0.0, 1.0], "y_common": [1, 2],
            "persistence_target": [1, 2], "candidate_unsafe": action == 0,
            "brake_unsafe": False, "failure_delay": None}


def test_pairing_requires_order_and_byte_equivalent_frozen_fields():
    left = [_row(action) for action in range(6)]
    right = deepcopy(left)
    for row in right:
        row["group_targets"] = row.pop("y_common")
    require_byte_equivalent(left, right)
    with pytest.raises(ValueError, match="lexicographic"):
        canonical_records(tuple(reversed(left)))
    right[2]["fixed_frame"] = [0.0, 2.0]
    with pytest.raises(ValueError, match="fixed_frame"):
        require_byte_equivalent(left, right)


def test_gates_authenticate_bootstrap_and_exact_registered_seeds():
    assert gate_rtg2_absolute(_absolute())
    stale = _absolute()
    stale["bootstrap"]["replicates"] = 999
    assert not gate_rtg2_absolute(stale)
    wrong_seeds = _absolute()
    wrong_seeds["per_seed"] = {seed: next(iter(wrong_seeds["per_seed"].values()))
                               for seed in range(5)}
    assert not gate_rtg2_absolute(wrong_seeds)


def test_rtg3_reexecutes_complete_local_chain_per_regime_and_v():
    evidence = {"shift": _regime(), "ood": _regime()}
    assert gate_rtg3(evidence)
    assert gate_rtg3_comparative(evidence)
    evidence["shift"]["rtg2_g"]["relative_reduction"] = 0.1
    independent = evaluate_rtg3_regimes(evidence)
    assert not independent["g_shift"] and independent["c_shift"]
    assert not independent["v_shift"]
    evidence = {"shift": _regime(), "ood": _regime()}
    evidence["ood"]["rtg1_z"]["transition_nmse"] = 0.95
    assert not gate_rtg3(evidence)
    assert not gate_rtg3_comparative(evidence)


def test_registered_batch_drops_stale_precomputed_metrics():
    batch = EvaluationBatch(records=({"x": 1},), metrics={"unsafe_rate": 999.0})
    assert _batch_payload(batch) == {"records": [{"x": 1}]}
