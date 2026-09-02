"""Fail-closed statistical summary for registered ATTR-RTG evaluation."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from .rtg_bootstrap import hierarchical_bootstrap, paired_hierarchical_bootstrap
from .rtg_config import TRAINING_SEEDS
from .rtg_gates import (
    evaluate_gate_dag,
    gate_d_fidelity,
    gate_rtg1_y,
    gate_rtg1_z,
    gate_rtg2_absolute,
)
from .rtg_integrity import IntegrityEvidence, require_integrity_evidence
from .rtg_metrics_governance import (
    comparative_metrics,
    governance_metrics,
    metrics_by_seed,
)
from .rtg_metrics_state import (
    consequence_macro_accuracy,
    consequence_nll,
    expected_calibration_error,
    transition_state_metrics,
)
from .rtg_pipeline_seal import BACKBONES
from .rtg_summary_validation import validate_registered_evaluation

Record = Mapping[str, Any]
_SPLITS = ("test_id", "test_shift", "test_ood")
_HEADS = ("G", "C")


def _g_metrics(rows: Iterable[Record]) -> dict[str, float]:
    records = tuple(rows)
    result = dict(governance_metrics(records))
    result.update(transition_state_metrics(records))
    result.update(
        {
            "nll": consequence_nll(records),
            "persistence_nll": consequence_nll(records, field="persistence_nll"),
            "ece": expected_calibration_error(
                records, score_field="risk", label_field="candidate_unsafe"
            ),
            "macro_accuracy": consequence_macro_accuracy(
                records, prediction_field="d_group_predictions"
            ),
            "persistence_macro_accuracy": consequence_macro_accuracy(
                records, prediction_field="persistence_predictions"
            ),
            "d_nll": consequence_nll(records, field="d_group_nll"),
        }
    )
    result["delta_nll"] = result["nll"] - result["persistence_nll"]
    return result


def _flatten(
    bootstrap: Mapping[str, Any], per_seed: Mapping[int, Mapping[str, float]]
) -> dict[str, Any]:
    metrics = bootstrap["metrics"]
    result: dict[str, Any] = {
        name: value["estimate"] for name, value in metrics.items()
    }
    result.update({f"{name}_ci95": value["ci95"] for name, value in metrics.items()})
    result["per_seed"] = {str(seed): dict(values) for seed, values in per_seed.items()}
    result["bootstrap"] = {
        name: bootstrap[name]
        for name in ("bootstrap_seed", "replicates", "bit_generator", "cluster_order")
    }
    return result


def _g_per_seed(rows: tuple[Record, ...]) -> dict[int, dict[str, float]]:
    seeds = sorted({row["seed"] for row in rows})
    if tuple(seeds) != tuple(sorted(TRAINING_SEEDS)):
        raise ValueError("summary requires exactly the five registered seeds")
    return {
        seed: _g_metrics(row for row in rows if row["seed"] == seed) for seed in seeds
    }


def _side(rows: Iterable[Record], name: str) -> tuple[dict[str, Any], ...]:
    output = []
    for wrapper in rows:
        item = dict(wrapper[name])
        for field in ("seed", "world_id", "episode_id"):
            item[field] = wrapper[field]
        output.append(item)
    return tuple(output)


def _paired_g(rows: Iterable[Record]) -> dict[str, float]:
    wrappers = tuple(rows)
    asm, transformer = _side(wrappers, "left"), _side(wrappers, "right")
    left, right = _g_metrics(asm), _g_metrics(transformer)
    result = comparative_metrics(left, right)
    result.update(
        {
            "nmse_asm": left["transition_nmse"],
            "nmse_transformer": right["transition_nmse"],
            "delta_nmse": left["transition_nmse"] - right["transition_nmse"],
            "nll_asm": left["nll"],
            "nll_transformer": right["nll"],
            "delta_nll": left["nll"] - right["nll"],
        }
    )
    return result


def _paired_governance(rows: Iterable[Record]) -> dict[str, float]:
    wrappers = tuple(rows)
    return comparative_metrics(
        governance_metrics(_side(wrappers, "left")),
        governance_metrics(_side(wrappers, "right")),
    )


def _paired_per_seed(
    left: tuple[Record, ...],
    right: tuple[Record, ...],
    estimator: Any,
) -> dict[int, dict[str, float]]:
    result = {}
    for seed in TRAINING_SEEDS:
        wrappers = []
        for left_row, right_row in zip(left, right, strict=True):
            if left_row["seed"] == seed and right_row["seed"] == seed:
                item = dict(left_row)
                item.update({"left": left_row, "right": right_row})
                wrappers.append(item)
        result[seed] = estimator(wrappers)
    return result


def _absolute(rows: tuple[Record, ...], head: str) -> dict[str, Any]:
    estimator = _g_metrics if head == "G" else governance_metrics
    per_seed = _g_per_seed(rows) if head == "G" else metrics_by_seed(rows)
    return _flatten(hierarchical_bootstrap(rows, estimator), per_seed)


def _paired(
    left: tuple[Record, ...],
    right: tuple[Record, ...],
    estimator: Any,
) -> dict[str, Any]:
    bootstrap = paired_hierarchical_bootstrap(left, right, estimator)
    return _flatten(bootstrap, _paired_per_seed(left, right, estimator))


def _fidelity(g: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "macro_accuracy": g["macro_accuracy"],
        "persistence_macro_accuracy": g["persistence_macro_accuracy"],
        "nll": g["d_nll"],
        "bootstrap": g["bootstrap"],
    }


def _regime_evidence(
    split: str,
    kind: str,
    absolute: Mapping[str, Any],
    versus: Mapping[str, Any],
) -> dict[str, Any]:
    g = absolute[split][kind]["G"]
    return {
        "d_fidelity": _fidelity(g),
        "rtg1_z": g,
        "rtg1_y": g,
        "rtg2_g": g,
        "rtg2_c": absolute[split][kind]["C"],
        "rtg2_v": versus[kind][split],
    }


def _dag_input(
    kind: str,
    absolute: Mapping[str, Any],
    architecture: Mapping[str, Any],
    versus: Mapping[str, Any],
    *,
    integrity_verified: bool,
) -> dict[str, Any]:
    arm = absolute["test_id"][kind]
    d = _fidelity(arm["G"])
    other = "transformer" if kind == "asm" else "asm"
    other_g = absolute["test_id"][other]["G"]
    other_d = _fidelity(other_g)
    other_z = gate_rtg1_z(other_g)
    other_y = gate_d_fidelity(other_d) and other_z and gate_rtg1_y(other_g)
    rtg3 = {
        "shift": _regime_evidence("test_shift", kind, absolute, versus),
        "ood": _regime_evidence("test_ood", kind, absolute, versus),
    }
    return {
        "integrity": integrity_verified,
        "d_fidelity": d,
        "rtg1_z": arm["G"],
        "rtg1_z_arch": architecture["test_id"],
        "rtg1_y": arm["G"],
        "rtg1_y_arch": architecture["test_id"],
        "rtg2_g": arm["G"],
        "rtg2_c": arm["C"],
        "rtg2_g_arch": architecture["test_id"],
        "rtg2_v": versus[kind]["test_id"],
        "rtg3": rtg3,
        "transformer_rtg1_z": other_z,
        "transformer_rtg1_y": other_y,
        "transformer_rtg2_g": other_y and gate_rtg2_absolute(other_g),
    }


def summarize_registered_evaluation(
    evaluation: Mapping[str, Any], evidence: IntegrityEvidence
) -> dict[str, Any]:
    """Summarize a complete result only inside the sealed evaluation flow."""
    integrity_verified = require_integrity_evidence(evidence)
    rows = validate_registered_evaluation(evaluation)
    absolute = {
        split: {
            kind: {head: _absolute(rows[split][kind][head], head) for head in _HEADS}
            for kind in BACKBONES
        }
        for split in _SPLITS
    }
    architecture = {
        split: _paired(
            rows[split]["asm"]["G"], rows[split]["transformer"]["G"], _paired_g
        )
        for split in _SPLITS
    }
    versus = {
        kind: {
            split: _paired(
                rows[split][kind]["G"], rows[split][kind]["C"], _paired_governance
            )
            for split in _SPLITS
        }
        for kind in BACKBONES
    }
    dags = {
        kind: evaluate_gate_dag(
            _dag_input(
                kind, absolute, architecture, versus,
                integrity_verified=integrity_verified,
            )
        )
        for kind in BACKBONES
    }
    labels = {
        "d_fidelity": "D",
        "rtg1_z": "RTG1-Z",
        "rtg1_y": "RTG1-Y",
        "rtg2_g": "RTG2-G",
        "rtg2_c": "RTG2-C",
        "rtg2_v": "RTG2-V",
        "rtg3_g_shift": "RTG3-G-SHIFT",
        "rtg3_g_ood": "RTG3-G-OOD",
        "rtg3_g": "RTG3-G",
        "rtg3_c_shift": "RTG3-C-SHIFT",
        "rtg3_c_ood": "RTG3-C-OOD",
        "rtg3_c": "RTG3-C",
        "rtg3_v_shift": "RTG3-V-SHIFT",
        "rtg3_v_ood": "RTG3-V-OOD",
        "rtg3_v": "RTG3-V",
    }
    display = {"asm": "ASM", "transformer": "Transformer"}
    gates = {
        f"{display[kind]}.{label}": dags[kind][key]
        for kind in BACKBONES
        for key, label in labels.items()
    }
    gates.update(
        {
            "RTG1-Z-ARCH": dags["asm"]["rtg1_z_arch"],
            "RTG1-Y-ARCH": dags["asm"]["rtg1_y_arch"],
            "RTG2-G-ARCH": dags["asm"]["rtg2_g_arch"],
        }
    )
    return {
        "kind": "attr_rtg_registered_summary",
        "schema_version": 1,
        "evidence": {
            "absolute": absolute,
            "architecture": architecture,
            "versus_c": versus,
        },
        "gates": gates,
    }
