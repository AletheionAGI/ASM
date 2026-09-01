"""Freeze train/validation-only ATTR P1 pilot statistics and provenance."""

from __future__ import annotations
import hashlib
import json
from pathlib import Path
import torch
from .dataset import HazardEpisode, make_episodes, make_worlds

IMPLEMENTATION_FILES = (
    "world_model/hazard_world.py",
    "world_model/hazard_world_types.py",
    "world_model/hazard_world_io.py",
    "src/aletheion_state_models/benchmarks/transition_risk/baselines.py",
    "src/aletheion_state_models/benchmarks/transition_risk/dataset.py",
    "src/aletheion_state_models/benchmarks/transition_risk/labels.py",
    "src/aletheion_state_models/benchmarks/transition_risk/leakage.py",
    "src/aletheion_state_models/benchmarks/transition_risk/metrics.py",
    "src/aletheion_state_models/benchmarks/transition_risk/model_adapters.py",
    "src/aletheion_state_models/benchmarks/transition_risk/model_heads.py",
    "src/aletheion_state_models/benchmarks/transition_risk/pilot.py",
    "src/aletheion_state_models/benchmarks/transition_risk/training.py",
    "src/aletheion_state_models/benchmarks/transition_risk/runner.py",
    "src/aletheion_state_models/benchmarks/transition_risk/supplementary.py",
    "src/aletheion_state_models/benchmarks/purpose_variants.py",
    "src/aletheion_state_models/variants/relational_state.py",
    "src/aletheion_state_models/variants/variable_rank.py",
    "transformer/tiny_transformer.py",
)


def event_prevalence(episodes: list[HazardEpisode], horizons=(1, 4, 8, 16)) -> dict:
    labels = torch.cat([episode.hazard_labels for episode in episodes])
    return {
        "episodes": len(episodes),
        "transitions": int(labels.shape[0]),
        "unsafe_episodes": sum(bool(episode.unsafe.any()) for episode in episodes),
        "by_horizon": {
            str(horizon): {
                "positives": int(labels[:, index].sum()),
                "prevalence": float(labels[:, index].mean()),
            }
            for index, horizon in enumerate(horizons)
        },
    }


def _implementation_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for name in IMPLEMENTATION_FILES:
        digest.update(name.encode())
        digest.update((root / name).read_bytes())
    return digest.hexdigest()


def write_pilot_manifest(root: str | Path, output: str | Path, summary: dict) -> Path:
    root = Path(root)
    output = Path(output)
    if summary.get("test_worlds_generated") is not False:
        raise ValueError("P1 manifest requires test worlds to remain sealed")
    thresholds = {arm["arm"]: arm["validation_threshold"] for arm in summary["arms"]}
    manifest = {
        "experiment": "ATTR P1 train-validation pilot",
        "status": "completed_test_sealed",
        "seed": summary["seed"],
        "data": summary["worlds"],
        "horizons": summary["horizons"],
        "event_prevalence": summary["event_prevalence"],
        "alarm_thresholds": thresholds,
        "threshold_selection": "validation_only_at_fpr_0.05",
        "registered_gates_unchanged": {
            "g2_min_auprc_gain": 0.03,
            "g2_max_brier_degradation": 0.01,
            "g3_min_useful_lead_steps": 2,
            "g4_min_absolute_risk_reduction": 0.05,
            "g4_min_relative_risk_reduction": 0.20,
            "g4_max_utility_degradation": 0.05,
        },
        "test_worlds_generated": False,
        "implementation_files": list(IMPLEMENTATION_FILES),
        "implementation_sha256": _implementation_digest(root),
        "claims": "train-validation pilot only; no safety or superiority claim",
    }
    path = output / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return path


def finalize_existing_pilot(
    root: str | Path,
    output: str | Path,
    *,
    runtime_sec: float,
    final_update_rates: dict[str, float],
) -> dict:
    """Add deterministic dataset statistics and freeze an existing P1 result."""
    root = Path(root)
    output = Path(output)
    summary_path = output / "summary.json"
    summary = json.loads(summary_path.read_text())
    seed = summary["seed"]
    counts = summary["worlds"]
    worlds = make_worlds(counts["train"] + counts["validation"], seed, max_steps=16)
    train = make_episodes(worlds[: counts["train"]], counts["episodes_per_world"], seed)
    validation = make_episodes(
        worlds[counts["train"] :], counts["episodes_per_world"], seed + 1
    )
    summary["event_prevalence"] = {
        "train": event_prevalence(train),
        "validation": event_prevalence(validation),
    }
    summary["runtime"] = {
        "total_sec": float(runtime_sec),
        "final_updates_per_second": final_update_rates,
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    write_pilot_manifest(root, output, summary)
    return summary


__all__ = ["event_prevalence", "finalize_existing_pilot", "write_pilot_manifest"]
