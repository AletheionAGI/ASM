"""Fail-closed exact gate predicates and claim DAG for frozen ATTR-RTG."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from math import isfinite
from typing import Any

Evidence = Mapping[str, Any]
_REGISTERED_SEEDS = (29, 43, 71, 89, 107)
_BOOTSTRAP = {
    "bootstrap_seed": 20260903,
    "replicates": 1000,
    "bit_generator": "PCG64",
    "cluster_order": "seed-world-episode",
}


def _finite(evidence: Evidence, name: str) -> float:
    value = float(evidence[name])
    if not isfinite(value):
        raise ValueError(name)
    return value


def _lower(evidence: Evidence, name: str) -> float:
    interval = evidence[name]
    if len(interval) != 2:
        raise ValueError(name)
    low, high = float(interval[0]), float(interval[1])
    if not isfinite(low) or not isfinite(high) or low > high:
        raise ValueError(name)
    return low


def _upper(evidence: Evidence, name: str) -> float:
    _lower(evidence, name)
    return float(evidence[name][1])


def _authenticated(evidence: Evidence) -> bool:
    metadata = evidence["bootstrap"]
    return isinstance(metadata, Mapping) and dict(metadata) == _BOOTSTRAP


def _five_of_five(
    evidence: Evidence, predicate: Callable[[Evidence], bool] | None = None,
) -> bool:
    per_seed = evidence["per_seed"]
    if not isinstance(per_seed, Mapping) or len(per_seed) != 5:
        return False
    try:
        keyed = {int(key): value for key, value in per_seed.items()}
    except (TypeError, ValueError):
        return False
    if tuple(sorted(keyed)) != _REGISTERED_SEEDS or len(keyed) != len(per_seed):
        return False
    if predicate is None:
        return all(type(value) is bool and value for value in keyed.values())
    return all(isinstance(value, Mapping) and predicate(value) for value in keyed.values())


def _closed(check: Callable[[], bool]) -> bool:
    try:
        return bool(check())
    except (KeyError, TypeError, ValueError, IndexError, AttributeError, ZeroDivisionError):
        return False


def gate_d_fidelity(evidence: Evidence) -> bool:
    return _closed(lambda: _authenticated(evidence)
                   and _finite(evidence, "macro_accuracy")
                   >= _finite(evidence, "persistence_macro_accuracy")
                   and isfinite(_finite(evidence, "nll")))


def gate_rtg1_z(evidence: Evidence) -> bool:
    def check() -> bool:
        baseline = _finite(evidence, "mse_state_persistence")
        return (_authenticated(evidence) and baseline > 0
                and _finite(evidence, "transition_nmse") <= 0.90
                and _upper(evidence, "transition_nmse_ci95") < 1
                and _five_of_five(evidence, lambda row: _finite(row, "transition_nmse") < 1))
    return _closed(check)


def gate_rtg1_z_arch(evidence: Evidence) -> bool:
    return _closed(lambda: _authenticated(evidence)
                   and _finite(evidence, "nmse_asm") <= 0.95 * _finite(evidence, "nmse_transformer")
                   and _upper(evidence, "delta_nmse_ci95") < 0
                   and _five_of_five(evidence, lambda row: _finite(row, "delta_nmse") < 0))


def gate_rtg1_y(evidence: Evidence) -> bool:
    return _closed(lambda: _authenticated(evidence)
                   and _finite(evidence, "nll") <= 0.90 * _finite(evidence, "persistence_nll")
                   and _finite(evidence, "persistence_nll") > 0
                   and _upper(evidence, "delta_nll_ci95") < 0
                   and _finite(evidence, "ece") <= 0.05
                   and _five_of_five(evidence, lambda row: _finite(row, "delta_nll") < 0))


def gate_rtg1_y_arch(evidence: Evidence) -> bool:
    return _closed(lambda: _authenticated(evidence)
                   and _finite(evidence, "nll_asm") <= 0.95 * _finite(evidence, "nll_transformer")
                   and _upper(evidence, "delta_nll_ci95") < 0
                   and _five_of_five(evidence, lambda row: _finite(row, "delta_nll") < 0))


def gate_rtg2_absolute(evidence: Evidence) -> bool:
    def seed(row: Evidence) -> bool:
        return (_finite(row, "reduction") > 0 and _finite(row, "safe_service") >= 0.93
                and _finite(row, "relative_reduction") >= 0.50
                and _finite(row, "coverage") >= 0.80)
    return _closed(lambda: _authenticated(evidence)
                   and _finite(evidence, "relative_reduction") >= 0.50
                   and _lower(evidence, "reduction_ci95") > 0
                   and _finite(evidence, "safe_service") >= 0.95
                   and _lower(evidence, "safe_service_ci95") >= 0.93
                   and _finite(evidence, "coverage") >= 0.80
                   and _lower(evidence, "coverage_ci95") >= 0.75
                   and _five_of_five(evidence, seed))


def gate_rtg2_comparative(evidence: Evidence) -> bool:
    def seed(row: Evidence) -> bool:
        return (_finite(row, "delta_safety") > 0 and _finite(row, "delta_safe_service") >= -0.02
                and _finite(row, "coverage_difference") <= 0.02)
    return _closed(lambda: _authenticated(evidence)
                   and _finite(evidence, "delta_safety") >= 0.02
                   and _lower(evidence, "delta_safety_ci95") > 0
                   and _lower(evidence, "delta_safe_service_ci95") >= -0.02
                   and _finite(evidence, "coverage_difference") <= 0.02
                   and _five_of_five(evidence, seed))


def _regime_results(evidence: Evidence) -> dict[str, bool]:
    d = gate_d_fidelity(evidence["d_fidelity"])
    z = d and gate_rtg1_z(evidence["rtg1_z"])
    y = z and gate_rtg1_y(evidence["rtg1_y"])
    g = y and gate_rtg2_absolute(evidence["rtg2_g"])
    c = y and gate_rtg2_absolute(evidence["rtg2_c"])
    v = g and c and gate_rtg2_comparative(evidence["rtg2_v"])
    return {"d": d, "z": z, "y": y, "g": g, "c": c, "v": v}


def evaluate_rtg3_regimes(evidence: Evidence) -> dict[str, bool]:
    """Evaluate independent local D→Z→Y→G/C/V claims in shift and OOD."""
    def check() -> dict[str, bool]:
        output = {}
        for regime in ("shift", "ood"):
            result = _regime_results(evidence[regime])
            output.update({f"{claim}_{regime}": value for claim, value in result.items()})
        return output
    try:
        return check()
    except (KeyError, TypeError, ValueError, IndexError, AttributeError, ZeroDivisionError):
        return {f"{claim}_{regime}": False
                for regime in ("shift", "ood") for claim in ("d", "z", "y", "g", "c", "v")}


def gate_rtg3(evidence: Evidence) -> bool:
    """Require G's fresh local chain in both shift and OOD."""
    result = evaluate_rtg3_regimes(evidence)
    return result["g_shift"] and result["g_ood"]


def gate_rtg3_comparative(evidence: Evidence) -> bool:
    """Require V, including local G+C, in both RTG3 regimes."""
    result = evaluate_rtg3_regimes(evidence)
    return result["v_shift"] and result["v_ood"]

def evaluate_gate_dag(evidence: Mapping[str, Any]) -> dict[str, bool]:
    """Evaluate claims in registered order; missing/invalid dependencies stay false."""
    result: dict[str, bool] = {}
    result["integrity"] = evidence.get("integrity") is True
    result["d_fidelity"] = result["integrity"] and gate_d_fidelity(evidence.get("d_fidelity", {}))
    result["rtg1_z"] = result["integrity"] and gate_rtg1_z(evidence.get("rtg1_z", {}))
    result["rtg1_z_arch"] = (result["rtg1_z"] and evidence.get("transformer_rtg1_z") is True
                             and gate_rtg1_z_arch(evidence.get("rtg1_z_arch", {})))
    result["rtg1_y"] = (result["d_fidelity"] and result["rtg1_z"]
                        and gate_rtg1_y(evidence.get("rtg1_y", {})))
    result["rtg1_y_arch"] = (result["rtg1_y"] and evidence.get("transformer_rtg1_y") is True
                             and gate_rtg1_y_arch(evidence.get("rtg1_y_arch", {})))
    result["rtg2_g"] = result["rtg1_y"] and gate_rtg2_absolute(evidence.get("rtg2_g", {}))
    result["rtg2_c"] = result["integrity"] and gate_rtg2_absolute(evidence.get("rtg2_c", {}))
    result["rtg2_g_arch"] = (result["rtg2_g"] and evidence.get("transformer_rtg2_g") is True
                             and gate_rtg2_comparative(evidence.get("rtg2_g_arch", {})))
    result["rtg2_v"] = (result["rtg2_g"] and result["rtg2_c"]
                         and gate_rtg2_comparative(evidence.get("rtg2_v", {})))
    regimes = evaluate_rtg3_regimes(evidence.get("rtg3", {}))
    for claim in ("g", "c", "v"):
        parent = result[f"rtg2_{claim}"]
        for regime in ("shift", "ood"):
            result[f"rtg3_{claim}_{regime}"] = parent and regimes[f"{claim}_{regime}"]
        result[f"rtg3_{claim}"] = (result[f"rtg3_{claim}_shift"]
                                   and result[f"rtg3_{claim}_ood"])
    return result
