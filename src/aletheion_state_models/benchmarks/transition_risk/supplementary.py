"""Supplementary ATTR P1 arms outside the registered main model pair."""

from __future__ import annotations
import json
from pathlib import Path
import torch
from aletheion_state_models.benchmarks.phase3a_variants import phase3a_config
from aletheion_state_models.benchmarks.purpose_variants import build_purpose_variant
from aletheion_state_models.variants import build_relational_state
from .dataset import make_episodes, make_worlds
from .model_adapters import ASMModelAdapter
from .model_heads import TransitionRiskHeads
from .pilot import write_pilot_manifest
from .render import render_summary
from .training import train_arm

SUPPLEMENTARY_ARMS = (
    "asm_cm_durable",
    "asm_vr_s_full64",
    "asm_vr_s_fixed32",
    "asm_r_240k_control",
)


def _build_backbone(arm: str, seed: int):
    torch.manual_seed(seed)
    if arm == "asm_cm_durable":
        return build_purpose_variant("asm_cm", seed)[0], None, True
    if arm == "asm_vr_s_full64":
        return build_purpose_variant("asm_vr_s_full", seed)[0], 64, False
    if arm == "asm_vr_s_fixed32":
        return build_purpose_variant("asm_vr_s_fixed_32", seed)[0], 32, False
    if arm == "asm_r_240k_control":
        return build_relational_state(phase3a_config(seed)), None, False
    raise ValueError(f"unknown supplementary ATTR arm: {arm}")


def _build_arm(arm: str, seed: int, updates: int):
    model, logical_rank, explicit_memory = _build_backbone(arm, seed)
    torch.manual_seed(seed + 50_000)
    heads = TransitionRiskHeads(model.config.d_state, 6, hidden_dim=32)
    adapter = ASMModelAdapter(model, global_step=updates)
    metadata = {
        "comparison_role": "supplementary_unmatched",
        "logical_rank": logical_rank,
        "explicit_associative_memory": explicit_memory,
        "backbone_trainable_parameters": sum(
            p.numel() for p in model.parameters() if p.requires_grad
        ),
    }
    return adapter, heads, metadata


def _write(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def run_supplementary_p1(
    root: str | Path,
    output: str | Path,
    *,
    device: str = "cuda",
) -> dict:
    """Train requested extra arms on the exact frozen P1 train/validation data."""
    root = Path(root)
    output = Path(output)
    summary_path = output / "summary.json"
    summary = json.loads(summary_path.read_text())
    if summary.get("test_worlds_generated") is not False:
        raise ValueError("supplementary P1 requires test to remain sealed")
    counts = summary["worlds"]
    seed = summary["seed"]
    updates = summary["updates"]
    worlds = make_worlds(counts["train"] + counts["validation"], seed, max_steps=16)
    train_worlds = worlds[: counts["train"]]
    validation_worlds = worlds[counts["train"] :]
    train = make_episodes(train_worlds, counts["episodes_per_world"], seed)
    validation = make_episodes(
        validation_worlds, counts["episodes_per_world"], seed + 1
    )
    progress_path = output / "supplementary_results.json"
    completed = json.loads(progress_path.read_text()) if progress_path.exists() else []
    completed_by_arm = {item["arm"]: item for item in completed}
    for arm in SUPPLEMENTARY_ARMS:
        if arm in completed_by_arm:
            continue
        adapter, heads, metadata = _build_arm(arm, seed, updates)
        result = train_arm(
            arm,
            adapter,
            heads,
            train,
            validation,
            updates=updates,
            batch_size=4,
            seed=seed,
            device=device,
        ).to_dict()
        result.update(metadata)
        completed.append(result)
        completed_by_arm[arm] = result
        _write(progress_path, completed)
    main_names = {"asm_x_directional", "tiny_transformer_220k"}
    main = [item for item in summary["arms"] if item["arm"] in main_names]
    for item in main:
        item["comparison_role"] = "registered_main_pair"
        item["backbone_trainable_parameters"] = item["backbone_parameters"]
        item["total_parameters"] = item.get(
            "total_parameters", item["trainable_parameters"]
        )
        rate = (
            summary.get("runtime", {})
            .get("final_updates_per_second", {})
            .get(item["arm"])
        )
        if rate is not None:
            item["updates_per_second"] = rate
            item["arm_train_elapsed_sec"] = updates / rate
    summary["arms"] = main + [completed_by_arm[arm] for arm in SUPPLEMENTARY_ARMS]
    cm = completed_by_arm["asm_cm_durable"]
    summary["supplementary_comparisons"] = {
        "cm_vs_vr_s_full_total_parameter_mismatch": abs(
            cm["backbone_parameters"]
            - completed_by_arm["asm_vr_s_full64"]["backbone_parameters"]
        )
        / max(
            cm["backbone_parameters"],
            completed_by_arm["asm_vr_s_full64"]["backbone_parameters"],
        ),
        "cm_vs_vr_s_fixed32_total_parameter_mismatch": abs(
            cm["backbone_parameters"]
            - completed_by_arm["asm_vr_s_fixed32"]["backbone_parameters"]
        )
        / max(
            cm["backbone_parameters"],
            completed_by_arm["asm_vr_s_fixed32"]["backbone_parameters"],
        ),
        "interpretation": "descriptive validation-only supplementary arms; not the registered ATTR gate pair",
    }
    _write(summary_path, summary)
    write_pilot_manifest(root, output, summary)
    render_summary(summary_path)
    return summary


__all__ = ["SUPPLEMENTARY_ARMS", "run_supplementary_p1"]
