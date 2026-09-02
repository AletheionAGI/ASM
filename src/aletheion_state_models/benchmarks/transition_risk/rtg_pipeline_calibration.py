"""Allowed calibration orchestration over attached post-truth state ledgers."""

from __future__ import annotations

from pathlib import Path

from .rtg_artifacts import atomic_write_json
from .rtg_calibration import fit_disjoint_calibration, partition_calibration_worlds
from .rtg_checkpoint import load_terminal_checkpoint
from .rtg_heads import DirectC, PhysicalD, TransitionG
from .rtg_normalization import StateNormalization
from .rtg_pipeline_train import (
    ARMS,
    _arm_dir,
    _load_normalization,
    _load_records,
    _metadata,
    _verified_allowed_data,
)
from .rtg_validation import extract_calibration_scores


def _fit_score_calibrations(scores):
    worlds = sorted({str(row["world_id"]) for row in scores})
    first_worlds, second_worlds = partition_calibration_worlds(worlds)
    groups = []
    for selected in (set(first_worlds), set(second_worlds)):
        rows = [row for row in scores if str(row["world_id"]) in selected]
        origins = len({(row["world_id"], row["episode_id"], row["t"]) for row in rows})
        groups.append((rows, origins))
    fitted = {}
    for system, field in (("G", "g_logit"), ("C", "c_logit")):
        first, second = groups
        value = fit_disjoint_calibration(
            [float(row[field]) for row in first[0]], [int(row["unsafe"]) for row in first[0]], [str(row["world_id"]) for row in first[0]],
            [float(row[field]) for row in second[0]], [int(row["unsafe"]) for row in second[0]], [str(row["world_id"]) for row in second[0]],
            temperature_origin_count=first[1], residual_origin_count=second[1])
        fitted[system] = {"temperature": value.temperature, "q95": value.q95}
    return fitted


def run_calibration_extraction(root: str | Path, output: str | Path, *, device="cpu") -> tuple[Path, ...]:
    _verified_allowed_data(root, output); output = Path(output); paths = []
    for kind, seed in ARMS:
        arm = _arm_dir(output, kind, seed)
        normalization: StateNormalization = _load_normalization(arm / "normalization.json")
        records = _load_records(arm / "states_calibration.pt")
        modules = [load_terminal_checkpoint(arm / f"{name}.pt", module,
            expected_kind=f"{kind}-{name}", expected_seed=seed,
            expected_metadata=_metadata()) for name, module in
            (("G", TransitionG()), ("D", PhysicalD()), ("C", DirectC()))]
        scores = extract_calibration_scores(records, normalization, *modules, seed,
                                             failure_delay=3, device=device)
        fitted = _fit_score_calibrations(scores)
        paths.append(atomic_write_json(arm / "calibration_scores.json", list(scores)))
        for system in ("G", "C"):
            paths.append(atomic_write_json(arm / f"calibration_{system}.json", fitted[system]))
    return tuple(paths)
