"""ATTR-TG1 aggregation, paired statistics, and trajectory-only plots."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

from .trajectory_checkpoint import FileDigest, digest_files
from .trajectory_evaluation import TrajectoryRecord, read_records_jsonl
from .trajectory_manifests import (
    ARMS,
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    OPTIMIZER_SEEDS,
)
from .trajectory_plots import render_trajectory_grounded
from .trajectory_statistics import (
    build_summary,
    fit_fpr_thresholds,
    paired_hierarchical_bootstrap,
)

DEFAULT_OUTPUT = Path(
    "docs/benchmarks/asm_transformer_transition_risk/trajectory_grounded_tg1"
)


def _verify_prediction_manifest(predictions: Path) -> None:
    path = predictions.parent / "trajectory_prediction_manifest.json"
    value = json.loads(path.read_text())
    if not isinstance(value, dict) or set(value) != {"seal", "files"}:
        raise ValueError("invalid ATTR-TG1 prediction manifest")
    records = tuple(FileDigest(**item) for item in value["files"])
    paths = {item.name: predictions / item.name for item in records}
    if len(records) != 40 or records != digest_files(paths):
        raise ValueError("ATTR-TG1 prediction manifest is incomplete or altered")


def load_matrix(
    predictions: str | Path, splits: Iterable[str]
) -> dict[tuple[str, str], tuple[TrajectoryRecord, ...]]:
    """Load the complete seed matrix and group records by arm and split."""
    root = Path(predictions)
    _verify_prediction_manifest(root)
    output = {}
    for arm in ARMS:
        for split in splits:
            rows = []
            for seed in OPTIMIZER_SEEDS:
                rows.extend(read_records_jsonl(root / f"{arm}-{seed}-{split}.jsonl"))
            output[(arm, split)] = tuple(rows)
    return output


def summarize_trajectory_grounded(
    predictions: str | Path,
    output: str | Path = DEFAULT_OUTPUT,
    *,
    replicates: int = BOOTSTRAP_REPLICATES,
) -> tuple[dict, tuple[Path, ...]]:
    """Compare ASM against transformer with paired seed/world/episode statistics."""
    splits = ("validation", "test_id", "test_shift", "test_ood")
    matrix = load_matrix(predictions, splits)
    thresholds = {arm: fit_fpr_thresholds(matrix[(arm, "validation")]) for arm in ARMS}
    by_split = {}
    for split in splits[1:]:
        asm = matrix[("asm_x_base", split)]
        transformer = matrix[("transformer_base", split)]
        by_split[split] = {
            "asm_x_base": build_summary(asm, thresholds=thresholds["asm_x_base"]),
            "transformer_base": build_summary(
                transformer, thresholds=thresholds["transformer_base"]
            ),
            "paired_deltas": paired_hierarchical_bootstrap(
                asm, transformer, replicates=replicates, seed=BOOTSTRAP_SEED
            ),
        }
    id_h8 = by_split["test_id"]["paired_deltas"]["by_horizon"]["H8"]
    tg2 = (
        id_h8["auprc"]["delta"] >= 0.03
        and id_h8["auprc"]["ci95"][0] > 0.0
        and id_h8["brier"]["ci95"][1] <= 0.01
    )
    robustness = all(
        by_split[split]["paired_deltas"]["by_horizon"]["H8"]["auprc"]["ci95"][0] >= 0.0
        and by_split[split]["paired_deltas"]["by_horizon"]["H8"]["brier"]["ci95"][1]
        <= 0.01
        for split in ("test_shift", "test_ood")
    )
    summary = dict(by_split["test_id"]["asm_x_base"])
    summary.update(
        {
            "protocol": "ATTR-TG1",
            "comparison": "asm_x_base minus transformer_base",
            "mechanism": "representation -> physical trajectory -> fixed unsafe predicate",
            "hazard_classifier_parameters": 0,
            "validation_thresholds": thresholds,
            "splits": by_split,
            "paired_deltas": by_split["test_id"]["paired_deltas"],
            "gates": {
                "TG0_integrity": True,
                "TG2_trajectory_anticipation_id": tg2,
                "robustness_shift_ood_diagnostic": robustness,
                "TG4_causal_intervention": None,
            },
            "predictive_passed": tg2 and robustness,
        }
    )
    paths = tuple(render_trajectory_grounded(summary, Path(output)))
    return summary, paths


__all__ = ["DEFAULT_OUTPUT", "load_matrix", "summarize_trajectory_grounded"]
