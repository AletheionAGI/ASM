from __future__ import annotations

from aletheion_state_models.benchmarks.transition_risk.rtg_bootstrap import (
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    hierarchical_bootstrap,
    percentile_type7,
)
from aletheion_state_models.benchmarks.transition_risk.rtg_gates import (
    evaluate_gate_dag,
    gate_rtg1_z,
    gate_rtg2_absolute,
    gate_rtg2_comparative,
)
from aletheion_state_models.benchmarks.transition_risk.rtg_metrics_governance import (
    governance_metrics,
)


def _rows():
    return [
        {"seed": seed, "world_id": "w", "episode_id": "e", "t": 0,
         "action_index": action, "value": float(seed + action)}
        for seed in (29, 43, 71, 89, 107) for action in range(6)
    ]


def test_frozen_pcg64_hierarchical_bootstrap_is_reproducible():
    estimator = lambda rows: {"mean": sum(row["value"] for row in rows) / len(rows)}
    first = hierarchical_bootstrap(_rows(), estimator)
    second = hierarchical_bootstrap(_rows(), estimator)
    assert first == second
    assert first["bootstrap_seed"] == BOOTSTRAP_SEED == 20260903
    assert first["replicates"] == BOOTSTRAP_REPLICATES == 1000
    assert first["bit_generator"] == "PCG64"
    assert first["cluster_order"] == "seed-world-episode"


def test_bootstrap_materializes_unique_cluster_occurrences_for_governance():
    rows = _rows()
    for row in rows:
        unsafe = row["action_index"] < 2
        row.update({
            "candidate_unsafe": unsafe,
            "brake_unsafe": False,
            "decision": "BLOCK" if unsafe else "ALLOW",
        })
    result = hierarchical_bootstrap(rows, governance_metrics)
    assert result["metrics"]["relative_reduction"]["estimate"] == 1.0
    assert result["metrics"]["unsafe_rate"]["ci95"] == [0.0, 0.0]


def test_percentile_is_linear_type_seven():
    assert percentile_type7([0.0, 10.0], 0.25) == 2.5


def _bootstrap():
    return {"bootstrap_seed": 20260903, "replicates": 1000,
            "bit_generator": "PCG64", "cluster_order": "seed-world-episode"}


def _absolute():
    seed = {"reduction": 0.2, "relative_reduction": 0.6,
            "safe_service": 0.95, "coverage": 0.9}
    return {"relative_reduction": 0.6, "reduction_ci95": [0.1, 0.3],
            "safe_service": 0.96, "safe_service_ci95": [0.94, 0.98],
            "coverage": 0.9, "coverage_ci95": [0.8, 0.95],
            "per_seed": {index: seed for index in (29, 43, 71, 89, 107)},
            "bootstrap": _bootstrap()}


def _comparative():
    seed = {"delta_safety": 0.03, "delta_safe_service": -0.01,
            "coverage_difference": 0.01}
    return {"delta_safety": 0.03, "delta_safety_ci95": [0.01, 0.05],
            "delta_safe_service_ci95": [-0.015, 0.01],
            "coverage_difference": 0.01,
            "per_seed": {index: seed for index in (29, 43, 71, 89, 107)},
            "bootstrap": _bootstrap()}


def test_exact_gates_require_five_of_five_and_fail_closed():
    assert gate_rtg2_absolute(_absolute())
    assert gate_rtg2_comparative(_comparative())
    failed = _absolute()
    failed["per_seed"] = {index: failed["per_seed"][index]
                          for index in (29, 43, 71, 89)}
    assert not gate_rtg2_absolute(failed)
    assert not gate_rtg1_z({"transition_nmse": 0.8})


def test_gate_dag_blocks_dependents():
    result = evaluate_gate_dag({
        "integrity": True,
        "d_fidelity": {"macro_accuracy": 0.8,
                       "persistence_macro_accuracy": 0.7, "nll": 1.0,
                       "bootstrap": _bootstrap()},
        "rtg1_z": {"mse_state_persistence": 1.0, "transition_nmse": 0.8,
                   "transition_nmse_ci95": [0.7, 0.9],
                   "per_seed": {i: {"transition_nmse": 0.8}
                                for i in (29, 43, 71, 89, 107)},
                   "bootstrap": _bootstrap()},
        # Missing RTG1-Y must block G and every dependent G/V claim.
        "rtg2_g": _absolute(), "rtg2_c": _absolute(), "rtg2_v": _comparative(),
    })
    assert result["integrity"] and result["d_fidelity"] and result["rtg1_z"]
    assert result["rtg2_c"]
    assert not result["rtg1_y"] and not result["rtg2_g"]
    assert not result["rtg2_v"] and not result["rtg3_v"]
