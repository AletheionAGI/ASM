"""Aggregate sealed ATTR P2 predictions, uncertainty, and registered gates."""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path

from .p2_conditioned_metrics import next_state_nll_by_hazard
from .p2_evaluation import (
    EpisodePrediction,
    compute_aggregate_metrics,
    read_episode_records_jsonl,
)
from .p2_models import MAIN_ARMS, P2_ARMS
from .p2_runner import P2_SEEDS, prediction_path, result_path
from .p2_statistics import evaluate_g2_g5, paired_hierarchical_bootstrap

P2_SPLITS = ("validation", "test_id", "test_shift", "test_ood")


def _tagged(records, seed: int, arm: str, split: str) -> list[dict]:
    output = []
    for record in records:
        labels = [list(column) for column in zip(*record.hazard_labels)]
        probabilities = [list(column) for column in zip(*record.hazard_probabilities)]
        output.append(
            {
                "seed": seed,
                "arm": arm,
                "split": split,
                "world_id": record.world_id,
                "episode_id": record.episode_id,
                "horizons": list(record.horizons),
                "hazard_labels": labels,
                "hazard_probabilities": probabilities,
            }
        )
    return output


def _load_records(run_root: Path, arm: str, split: str):
    tagged = []
    raw: list[tuple[int, EpisodePrediction]] = []
    for seed in P2_SEEDS:
        records = read_episode_records_jsonl(
            prediction_path(run_root, split, seed, arm)
        )
        tagged.extend(_tagged(records, seed, arm, split))
        raw.extend((seed, record) for record in records)
    return tagged, raw


def _threshold_metrics(
    raw, thresholds: dict[int, float], horizon_index: int = 2
) -> dict:
    by_seed = {}
    for seed in P2_SEEDS:
        labels = []
        scores = []
        for record_seed, record in raw:
            if record_seed != seed:
                continue
            labels.extend(row[horizon_index] for row in record.hazard_labels)
            scores.extend(row[horizon_index] for row in record.hazard_probabilities)
        threshold = thresholds[seed]
        tp = sum(label and score >= threshold for label, score in zip(labels, scores))
        fp = sum(
            not label and score >= threshold for label, score in zip(labels, scores)
        )
        positives = sum(labels)
        negatives = len(labels) - positives
        by_seed[str(seed)] = {
            "threshold": threshold,
            "recall": tp / positives if positives else 0.0,
            "fpr": fp / negatives if negatives else 0.0,
            "positives": positives,
            "negatives": negatives,
        }
    return {
        "by_seed": by_seed,
        "recall_mean": sum(item["recall"] for item in by_seed.values()) / len(by_seed),
        "fpr_mean": sum(item["fpr"] for item in by_seed.values()) / len(by_seed),
    }


def _nll_tree(raw):
    tree = defaultdict(lambda: defaultdict(list))
    values = {}
    for seed, record in raw:
        key = (seed, record.world_id, record.episode_id)
        values[key] = tuple(record.next_state_nll)
        tree[seed][record.world_id].append(key)
    return values, tree


def _mean_nll(values, keys) -> float:
    flattened = [value for key in keys for value in values[key]]
    return math.fsum(flattened) / len(flattened)


def _percentile(values, probability):
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    fraction = position - lower
    return (
        ordered[lower] * (1 - fraction)
        + ordered[min(lower + 1, len(ordered) - 1)] * fraction
    )


def paired_nll_bootstrap(
    left_raw, right_raw, *, replicates=1000, seed=20260901
) -> dict:
    left, tree = _nll_tree(left_raw)
    right, right_tree = _nll_tree(right_raw)
    if left.keys() != right.keys() or tree.keys() != right_tree.keys():
        raise ValueError("next-state NLL records are not paired")
    keys = list(left)
    observed = _mean_nll(left, keys) - _mean_nll(right, keys)
    rng = random.Random(seed)
    draws = []
    for _ in range(replicates):
        selected = []
        seeds = sorted(tree)
        for sampled_seed in rng.choices(seeds, k=len(seeds)):
            worlds = sorted(tree[sampled_seed])
            for world in rng.choices(worlds, k=len(worlds)):
                episodes = tree[sampled_seed][world]
                selected.extend(rng.choices(episodes, k=len(episodes)))
        draws.append(_mean_nll(left, selected) - _mean_nll(right, selected))
    return {
        "mean_delta_nll": observed,
        "delta_nll_ci95": [_percentile(draws, 0.025), _percentile(draws, 0.975)],
        "replicates": replicates,
    }


def build_p2_summary(
    root: str | Path,
    output: str | Path,
    run_root: str | Path,
    *,
    replicates: int = 1000,
) -> dict:
    root = Path(root)
    output = Path(output)
    run_root = Path(run_root)
    results = {
        (seed, arm): json.loads(result_path(run_root, seed, arm).read_text())
        for seed in P2_SEEDS
        for arm in P2_ARMS
    }
    arm_records = {}
    arms = {}
    record_counts = {}
    for arm in P2_ARMS:
        thresholds = {
            seed: results[(seed, arm)]["validation_threshold"] for seed in P2_SEEDS
        }
        arms[arm] = {
            "comparison_role": results[(P2_SEEDS[0], arm)]["comparison_role"],
            "parameters": {
                "total": results[(P2_SEEDS[0], arm)]["total_parameters"],
                "trainable": results[(P2_SEEDS[0], arm)]["trainable_parameters"],
            },
            "splits": {},
        }
        for split in P2_SPLITS:
            tagged, raw = _load_records(run_root, arm, split)
            arm_records[(arm, split)] = (tagged, raw)
            record_counts[(arm, split)] = len(raw)
            aggregate = compute_aggregate_metrics([record for _, record in raw])
            aggregate["next_state_nll_by_hazard_h8"] = next_state_nll_by_hazard(
                [record for _, record in raw], horizon=8
            )
            aggregate["by_seed"] = {
                str(seed): compute_aggregate_metrics(
                    [record for record_seed, record in raw if record_seed == seed]
                )
                for seed in P2_SEEDS
            }
            aggregate["threshold_metrics_h8"] = _threshold_metrics(raw, thresholds)
            arms[arm]["splits"][split] = aggregate
    comparisons = {}
    for split in P2_SPLITS[1:]:
        left, left_raw = arm_records[(MAIN_ARMS[0], split)]
        right, right_raw = arm_records[(MAIN_ARMS[1], split)]
        hazard = paired_hierarchical_bootstrap(
            left, right, horizon=8, replicates=replicates, seed=20260901
        )
        comparisons[split] = {
            "asm_x_minus_transformer": hazard,
            "next_state_nll": paired_nll_bootstrap(
                left_raw, right_raw, replicates=replicates
            ),
        }
    supplementary_pairs = {
        "cm_minus_vr_s_full": ("asm_cm_durable", "asm_vr_s_full64"),
        "cm_minus_vr_s_fixed32": ("asm_cm_durable", "asm_vr_s_fixed32"),
        "vr_s_fixed32_minus_full": ("asm_vr_s_fixed32", "asm_vr_s_full64"),
    }
    supplementary = {}
    for name, (left_arm, right_arm) in supplementary_pairs.items():
        supplementary[name] = paired_hierarchical_bootstrap(
            arm_records[(left_arm, "test_id")][0],
            arm_records[(right_arm, "test_id")][0],
            horizon=8,
            replicates=replicates,
            seed=20260901,
        )
    registered = comparisons["test_id"]["asm_x_minus_transformer"]
    partial_gates = evaluate_g2_g5(registered, critical_ood_floors=None)
    nll = comparisons["test_id"]["next_state_nll"]
    expected_counts = {
        "validation": len(P2_SEEDS) * 64,
        "test_id": len(P2_SEEDS) * 128,
        "test_shift": len(P2_SEEDS) * 128,
        "test_ood": len(P2_SEEDS) * 128,
    }
    implementation = json.loads(
        (output / "training_implementation_manifest.json").read_text()
    )
    patch = implementation.get("post_training_evaluation_patch", {})
    implementation_matches = all(
        hashlib.sha256((root / name).read_bytes()).hexdigest() == expected
        or (
            name == patch.get("file")
            and hashlib.sha256((root / name).read_bytes()).hexdigest()
            == patch.get("evaluation_sha256")
            and patch.get("training_sha256") == expected
        )
        for name, expected in implementation["files"].items()
    )
    g0 = (
        implementation_matches
        and all(
            record_counts[(arm, split)] == expected_counts[split]
            for arm in P2_ARMS
            for split in P2_SPLITS
        )
        and (output / "dataset_seal.json").is_file()
        and (output / "test_spec_preseal.json").is_file()
        and (output / "test_open_event.json").is_file()
        and len(results) == len(P2_SEEDS) * len(P2_ARMS)
    )
    positive_directions = sum(
        bool(item["positive"]) for item in registered["per_seed"].values()
    )
    gates = {
        "g0_integrity": g0,
        "g1_next_state_noninferiority": nll["delta_nll_ci95"][1] <= 0.02,
        "g2_anticipation": partial_gates["g2"],
        "g3_actionable_warning": None,
        "g4_causal_intervention": None,
        "g5_robustness": False,
    }
    main_left = arms[MAIN_ARMS[0]]["parameters"]["total"]
    main_right = arms[MAIN_ARMS[1]]["parameters"]["total"]
    cm_total = arms["asm_cm_durable"]["parameters"]["total"]
    vr_total = arms["asm_vr_s_full64"]["parameters"]["total"]
    parameter_audits = {
        "main_total_mismatch_fraction": abs(main_left - main_right)
        / max(main_left, main_right),
        "cm_vr_s_total_mismatch_fraction": abs(cm_total - vr_total)
        / max(cm_total, vr_total),
        "vr_frozen_parameters": arms["asm_vr_s_full64"]["parameters"]["total"]
        - arms["asm_vr_s_full64"]["parameters"]["trainable"],
    }
    summary = {
        "experiment": "ATTR P2 sealed predictive benchmark",
        "training_seeds": list(P2_SEEDS),
        "parameter_audits": parameter_audits,
        "arms": arms,
        "comparisons": comparisons,
        "supplementary_test_id": supplementary,
        "g5_direction_diagnostic": {
            "asm_x_positive_seeds": positive_directions,
            "required": 4,
            "direction_only_passed": positive_directions >= 4,
            "full_g5_requires_registered_ood_floor": True,
        },
        "gates": gates,
        "predictive_passed": bool(
            gates["g0_integrity"]
            and gates["g1_next_state_noninferiority"]
            and gates["g2_anticipation"]
        ),
        "test_opened_after_all_checkpoints": True,
        "claims": "benchmark-specific predictive evidence only; G3/G4 not evaluated and no safety claim",
    }
    _write = output / "summary.json"
    _write.write_text(json.dumps(summary, indent=2) + "\n")
    return summary
