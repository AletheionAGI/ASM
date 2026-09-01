"""Offline figures and dashboard for sealed ATTR P2 results."""

from __future__ import annotations

import html
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

LABELS = {
    "asm_x_directional": "ASM-X Base",
    "tiny_transformer_220k": "Transformer",
    "asm_cm_durable": "ASM-CM",
    "asm_vr_s_full64": "VR-S full",
    "asm_vr_s_fixed32": "VR-S fixed-32",
    "asm_r_240k_control": "ASM-R 240K",
}
COLORS = ("#4472C4", "#ED7D31", "#70AD47", "#A5A5A5", "#FFC000", "#5B9BD5")
SPLITS = ("test_id", "test_shift", "test_ood")


def _save(fig, output: Path, name: str) -> None:
    fig.tight_layout()
    for extension in ("png", "svg"):
        fig.savefig(output / f"{name}.{extension}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def render_p2(summary_or_path, output: str | Path | None = None) -> list[Path]:
    if isinstance(summary_or_path, (str, Path)):
        source = Path(summary_or_path)
        summary = json.loads(source.read_text())
        target = Path(output) if output else source.parent
    else:
        summary = summary_or_path
        target = Path(output or ".")
    target.mkdir(parents=True, exist_ok=True)
    arms = list(summary["arms"])
    names = [LABELS.get(arm, arm) for arm in arms]
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    for row, split in enumerate(SPLITS):
        values = (
            [
                summary["arms"][arm]["splits"][split]["hazard"]["8"]["auprc"]
                for arm in arms
            ],
            [
                summary["arms"][arm]["splits"][split]["hazard"]["8"]["brier"]
                for arm in arms
            ],
            [
                summary["arms"][arm]["splits"][split]["mean_next_state_nll"]
                for arm in arms
            ],
        )
        for axis, metric, data in zip(
            axes[row],
            (
                "Direct hazard-head AUPRC H8",
                "Direct hazard-head Brier H8 ↓",
                "Next-state NLL ↓",
            ),
            values,
        ):
            axis.bar(range(len(arms)), data, color=COLORS)
            axis.set_xticks(range(len(arms)), names, rotation=27, ha="right")
            axis.set_title(f"{split}: {metric}")
            axis.grid(axis="y", alpha=0.25)
    fig.suptitle("ATTR P2 sealed metrics: direct hazard heads and one-step dynamics")
    _save(fig, target, "sealed_metrics")
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    for column, split in enumerate(SPLITS):
        conditioned = [
            summary["arms"][arm]["splits"][split]["next_state_nll_by_hazard_h8"]
            for arm in arms
        ]
        for row, group in enumerate(("positive", "negative")):
            data = [item[group]["mean"] for item in conditioned]
            axis = axes[row, column]
            axis.bar(range(len(arms)), data, color=COLORS)
            axis.set_xticks(range(len(arms)), names, rotation=27, ha="right")
            axis.set_title(
                f"{split}: Next-state NLL | H8={'1' if group == 'positive' else '0'} ↓"
            )
            axis.grid(axis="y", alpha=0.25)
    fig.suptitle(
        "Hazard-conditioned one-step dynamics (evaluation stratification only)"
    )
    _save(fig, target, "hazard_conditioned_dynamics")
    fig, axis = plt.subplots(figsize=(10, 5.5))
    for arm, color in zip(arms, COLORS):
        values = [
            summary["arms"][arm]["splits"]["test_id"]["by_seed"][str(seed)]["hazard"][
                "8"
            ]["auprc"]
            for seed in summary["training_seeds"]
        ]
        axis.plot(
            summary["training_seeds"],
            values,
            marker="o",
            label=LABELS.get(arm, arm),
            color=color,
        )
    axis.set_xlabel("Training seed")
    axis.set_ylabel("AUPRC H8 test_id")
    axis.set_title("Per-seed sealed ID results")
    axis.grid(alpha=0.25)
    axis.legend(ncol=2)
    _save(fig, target, "test_id_multiseed")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for axis, split in zip(axes, SPLITS):
        for arm, color in zip(arms, COLORS):
            metrics = summary["arms"][arm]["splits"][split]
            x = metrics["mean_next_state_nll"]
            y = metrics["hazard"]["8"]["auprc"]
            axis.scatter(x, y, color=color, s=48)
            axis.annotate(
                LABELS.get(arm, arm),
                (x, y),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8,
            )
        axis.set_xlabel("Next-state NLL ↓")
        axis.set_title(split)
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("AUPRC H8 ↑")
    fig.suptitle("Dynamics prediction versus hazard anticipation (descriptive)")
    _save(fig, target, "dynamics_vs_anticipation")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    deltas = [
        summary["comparisons"][split]["asm_x_minus_transformer"] for split in SPLITS
    ]
    means = [item["mean_delta_auprc"] for item in deltas]
    lower = [mean - item["delta_auprc_ci95"][0] for mean, item in zip(means, deltas)]
    upper = [item["delta_auprc_ci95"][1] - mean for mean, item in zip(means, deltas)]
    axes[0].errorbar(SPLITS, means, yerr=[lower, upper], marker="o", capsize=5)
    axes[0].axhline(0, color="black")
    axes[0].axhline(0.03, color="green", linestyle="--", label="G2 mean ≥ .03")
    axes[0].set_title("ASM-X Base − Transformer AUPRC H8")
    axes[0].legend()
    axes[0].grid(alpha=0.25)
    nll = [summary["comparisons"][split]["next_state_nll"] for split in SPLITS]
    nmeans = [item["mean_delta_nll"] for item in nll]
    nlower = [mean - item["delta_nll_ci95"][0] for mean, item in zip(nmeans, nll)]
    nupper = [item["delta_nll_ci95"][1] - mean for mean, item in zip(nmeans, nll)]
    axes[1].errorbar(SPLITS, nmeans, yerr=[nlower, nupper], marker="o", capsize=5)
    axes[1].axhline(0.02, color="green", linestyle="--", label="G1 margin .02")
    axes[1].axhline(0, color="black")
    axes[1].set_title("ASM-X Base − Transformer next-state NLL")
    axes[1].legend()
    axes[1].grid(alpha=0.25)
    _save(fig, target, "registered_pair_deltas")
    gates = "".join(
        f"<li><b>{html.escape(name)}</b>: {('NOT EVALUATED' if value is None else 'PASS' if value else 'FAIL')}</li>"
        for name, value in summary["gates"].items()
    )
    figures = "".join(
        f'<section><h2>{name.replace("_", " ")}</h2><a href="{name}.svg"><img src="{name}.svg"></a></section>'
        for name in (
            "sealed_metrics",
            "hazard_conditioned_dynamics",
            "test_id_multiseed",
            "dynamics_vs_anticipation",
            "registered_pair_deltas",
        )
    )
    page = (
        "<!doctype html><meta charset='utf-8'><title>ATTR P2 sealed predictive benchmark</title><style>body{font:16px system-ui;max-width:1200px;margin:2rem auto;padding:0 1rem}.warning{background:#fff3cd;padding:1rem;border-left:5px solid #d39e00}img{max-width:100%}</style><h1>ATTR P2 sealed predictive benchmark</h1><p class='warning'>Benchmark-specific prediction only. G3/G4 were not evaluated; no safety or causal-intervention claim is permitted.</p><ul>"
        + gates
        + "</ul>"
        + figures
        + "<p><a href='README.md'>Result notes</a> · <a href='summary.json'>Summary JSON</a> · <a href='dataset_seal.json'>Dataset seal</a> · <a href='test_spec_preseal.json'>Preseal</a></p>"
    )
    (target / "index.html").write_text(page)
    return [
        target / f"{name}.{extension}"
        for name in (
            "sealed_metrics",
            "hazard_conditioned_dynamics",
            "test_id_multiseed",
            "dynamics_vs_anticipation",
            "registered_pair_deltas",
        )
        for extension in ("png", "svg")
    ] + [target / "index.html"]


__all__ = ["render_p2"]
