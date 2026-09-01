"""Statistics for the post-hoc ASM-X native-risk-mass diagnostic."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from .p2_risk_mass_models import BASELINE_ARM, RISK_MASS_ARM, build_risk_mass_arm
from .p2_runner import P2_SEEDS, result_path
from .p2_statistics import paired_hierarchical_bootstrap
from .p2_summary import (
    P2_SPLITS,
    _load_records,
    _threshold_metrics,
    paired_nll_bootstrap,
)

PUBLIC_NAMES = {BASELINE_ARM: "ASM-X Base", RISK_MASS_ARM: "ASM-X + Native Risk Mass"}


def _learned_risk_parameter_audit(root: Path, run_root: Path) -> dict:
    output = {}
    for seed in P2_SEEDS:
        adapter, _, _ = build_risk_mass_arm(root, seed, 1000)
        initial = {
            name: value.detach().cpu()
            for name, value in adapter.model.risk.state_dict().items()
        }
        payload = torch.load(
            result_path(run_root, seed, RISK_MASS_ARM).parent.parent
            / "checkpoints"
            / f"seed_{seed}__{RISK_MASS_ARM}.pt",
            map_location="cpu",
            weights_only=False,
        )
        trained = {
            name.removeprefix("risk."): value
            for name, value in payload["model_state"].items()
            if name.startswith("risk.")
        }
        squared = sum(
            (trained[name] - initial[name]).double().square().sum().item()
            for name in initial
        )
        output[str(seed)] = {
            "risk_parameter_l2_change": squared**0.5,
            "changed": squared > 0.0,
        }
    return output


def build_risk_mass_summary(
    root: str | Path,
    output: str | Path,
    run_root: str | Path,
    *,
    replicates: int = 1000,
) -> dict:
    """Compare the new post-hoc arm with the untouched ASM-X checkpoints."""
    root, output, run_root = Path(root), Path(output), Path(run_root)
    records = {}
    arms = {}
    for arm in (BASELINE_ARM, RISK_MASS_ARM):
        results = {
            seed: json.loads(result_path(run_root, seed, arm).read_text())
            for seed in P2_SEEDS
        }
        thresholds = {seed: results[seed]["validation_threshold"] for seed in P2_SEEDS}
        arms[arm] = {
            "public_name": PUBLIC_NAMES[arm],
            "parameters": {
                "total": results[P2_SEEDS[0]]["total_parameters"],
                "trainable": results[P2_SEEDS[0]]["trainable_parameters"],
            },
            "splits": {},
        }
        for split in P2_SPLITS:
            tagged, raw = _load_records(run_root, arm, split)
            records[(arm, split)] = (tagged, raw)
            from .p2_evaluation import compute_aggregate_metrics

            aggregate = compute_aggregate_metrics([record for _, record in raw])
            aggregate["by_seed"] = {
                str(seed): compute_aggregate_metrics(
                    [record for record_seed, record in raw if record_seed == seed]
                )
                for seed in P2_SEEDS
            }
            aggregate["threshold_metrics_h8"] = _threshold_metrics(raw, thresholds)
            arms[arm]["splits"][split] = aggregate
    comparisons = {}
    for split in P2_SPLITS:
        baseline, baseline_raw = records[(BASELINE_ARM, split)]
        variant, variant_raw = records[(RISK_MASS_ARM, split)]
        comparisons[split] = {
            "risk_mass_minus_base_by_horizon": {
                str(horizon): paired_hierarchical_bootstrap(
                    variant,
                    baseline,
                    horizon=horizon,
                    replicates=replicates,
                    seed=20260901,
                )
                for horizon in (1, 4, 8, 16)
            },
            "next_state_nll": paired_nll_bootstrap(
                variant_raw, baseline_raw, replicates=replicates, seed=20260901
            ),
        }
    summary = {
        "experiment": "ATTR P2 post-hoc native-risk-mass diagnostic",
        "confirmatory_status": "exploratory_posthoc_does_not_modify_registered_p2_gates",
        "public_names": PUBLIC_NAMES,
        "training_seeds": list(P2_SEEDS),
        "sole_config_delta": {
            "use_powerlaw_risk": {"baseline": False, "variant": True}
        },
        "arms": arms,
        "comparisons": comparisons,
        "learned_risk_parameter_audit": _learned_risk_parameter_audit(root, run_root),
        "registered_p2_gates_unchanged": True,
        "claims": "diagnostic evidence only; test was already open before selection",
    }
    (output / "risk_mass_extension_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    return summary


__all__ = ["PUBLIC_NAMES", "build_risk_mass_summary"]
