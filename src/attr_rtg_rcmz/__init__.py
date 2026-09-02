"""ATTR-RTG-RCMZ components with lazy public imports.

Lazy loading keeps the synthetic terminal fast path independent of CUDA/model code.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "main": ("cli", "main"),
    "CANDIDATES": ("constants", "CANDIDATES"),
    "H8Truth": ("data_contracts", "H8Truth"),
    "ModelProcessInput": ("data_contracts", "ModelProcessInput"),
    "OriginKey": ("data_contracts", "OriginKey"),
    "Manifest": ("manifests", "Manifest"),
    "candidate_frames": ("adapters", "candidate_frames"),
    "batch_manifest": ("manifests", "batch_manifest"),
    "candidate_manifest": ("manifests", "candidate_manifest"),
    "split_manifest": ("manifests", "split_manifest"),
    "h8_all_candidates": ("h8", "h8_all_candidates"),
    "h8_truth": ("h8", "h8_truth"),
    "hazard_world": ("adapters", "hazard_world"),
    "model_process_input": ("adapters", "model_process_input"),
    "DraftEngine": ("engine", "DraftEngine"),
    "RunReceipt": ("engine", "RunReceipt"),
    "DraftRunConfig": ("policy", "DraftRunConfig"),
    "derive_seed64": ("policy", "derive_seed64"),
    "ACTIONS": ("constants", "ACTIONS"),
    "ARMS": ("constants", "ARMS"),
    "BOOTSTRAP_REPLICATES": ("constants", "BOOTSTRAP_REPLICATES"),
    "CONTRASTS": ("constants", "CONTRASTS"),
    "REGIMES": ("constants", "REGIMES"),
    "SEEDS": ("constants", "SEEDS"),
    "TEMPERATURE_GRID": ("constants", "TEMPERATURE_GRID"),
    "Decision": ("decision", "Decision"),
    "decide": ("decision", "decide"),
    "GateResult": ("gates", "GateResult"),
    "contrast_gate": ("gates", "contrast_gate"),
    "bootstrap_seed64": ("bootstrap", "bootstrap_seed64"),
    "paired_bootstrap": ("bootstrap", "paired_bootstrap"),
    "simultaneous_bounds": ("bootstrap", "simultaneous_bounds"),
    "calibrated_probabilities": ("calibration", "calibrated_probabilities"),
    "fit_temperature": ("calibration", "fit_temperature"),
    "candidate_nll": ("nll", "candidate_nll"),
    "h8_nll": ("nll", "h8_nll"),
    "ece15": ("ece", "ece15"),
    "origin_ece15": ("ece", "origin_ece15"),
    "equal_fold": ("folds", "equal_fold"),
    "safe_q95": ("quantiles", "safe_q95"),
    "type7": ("quantiles", "type7"),
    "six_contrasts": ("contrasts", "six_contrasts"),
    "decision_origin_metrics": ("metrics", "decision_origin_metrics"),
    "fold_decision_metric": ("metrics", "fold_decision_metric"),
    "ModelConfig": ("config", "ModelConfig"),
    "load_config": ("config", "load_config"),
    "registered_config_paths": ("config", "registered_config_paths"),
    "RiskAdapter": ("adapter", "RiskAdapter"),
    "build_adapter": ("models", "build_adapter"),
    "InferenceMessage": ("contracts", "InferenceMessage"),
    "InferenceResult": ("contracts", "InferenceResult"),
    "FourFieldInference": ("contracts", "FourFieldInference"),
    "MESSAGE_FIELDS": ("contracts", "MESSAGE_FIELDS"),
    "audit_strict_z": ("z", "audit_strict_z"),
}

__all__ = sorted(_EXPORTS)  # noqa: PLE0605 — lazy exports are generated from one registry


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as error:
        raise AttributeError(name) from error
    value = getattr(import_module(f"{__name__}.{module_name}"), attribute)
    globals()[name] = value
    return value
