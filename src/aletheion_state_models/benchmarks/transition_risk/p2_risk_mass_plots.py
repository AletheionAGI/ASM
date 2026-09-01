"""Figures and dashboard integration for the post-hoc risk-mass diagnostic."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .p2_risk_mass_models import BASELINE_ARM, RISK_MASS_ARM

SPLITS = ("test_id", "test_shift", "test_ood")
NAMES = {BASELINE_ARM: "ASM-X Base", RISK_MASS_ARM: "ASM-X + Native Risk Mass"}
COLORS = {BASELINE_ARM: "#4472C4", RISK_MASS_ARM: "#C00000"}
MARKER_START = "<!-- risk-mass-extension:start -->"
MARKER_END = "<!-- risk-mass-extension:end -->"


def _save(fig, output: Path, name: str) -> None:
    fig.tight_layout()
    for extension in ("png", "svg"):
        fig.savefig(output / f"{name}.{extension}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _metric(summary, arm, split, name):
    metrics = summary["arms"][arm]["splits"][split]
    if name == "auprc":
        return metrics["hazard"]["8"]["auprc"]
    if name == "brier":
        return metrics["hazard"]["8"]["brier"]
    return metrics["mean_next_state_nll"]


def _render_metric_grid(summary, output: Path) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(14, 11))
    for row, split in enumerate(SPLITS):
        for axis, metric, title in zip(
            axes[row],
            ("auprc", "brier", "nll"),
            ("AUPRC H8 ↑", "Brier H8 ↓", "Next-state NLL ↓"),
        ):
            values = [
                _metric(summary, arm, split, metric)
                for arm in (BASELINE_ARM, RISK_MASS_ARM)
            ]
            axis.bar(
                (0, 1), values, color=[COLORS[BASELINE_ARM], COLORS[RISK_MASS_ARM]]
            )
            axis.set_xticks(
                (0, 1),
                [NAMES[BASELINE_ARM], NAMES[RISK_MASS_ARM]],
                rotation=12,
                ha="right",
            )
            axis.set_title(f"{split}: {title}")
            axis.grid(axis="y", alpha=0.25)
    fig.suptitle("ATTR P2 post-hoc diagnostic: native risk mass")
    _save(fig, output, "risk_mass_metrics")


def _render_horizons(summary, output: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharex=True)
    horizons = (1, 4, 8, 16)
    for axis, split in zip(axes, SPLITS):
        for arm in (BASELINE_ARM, RISK_MASS_ARM):
            values = [
                summary["arms"][arm]["splits"][split]["hazard"][str(horizon)]["auprc"]
                for horizon in horizons
            ]
            axis.plot(horizons, values, marker="o", label=NAMES[arm], color=COLORS[arm])
        axis.set_title(split)
        axis.set_xlabel("Hazard horizon")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("AUPRC")
    axes[-1].legend()
    fig.suptitle("AUPRC by horizon")
    _save(fig, output, "risk_mass_horizons")


def _render_deltas(summary, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    stats = [
        summary["comparisons"][split]["risk_mass_minus_base_by_horizon"]["8"]
        for split in SPLITS
    ]
    means = [item["mean_delta_auprc"] for item in stats]
    lower = [
        max(0.0, mean - item["delta_auprc_ci95"][0]) for mean, item in zip(means, stats)
    ]
    upper = [
        max(0.0, item["delta_auprc_ci95"][1] - mean) for mean, item in zip(means, stats)
    ]
    axes[0].errorbar(
        SPLITS,
        means,
        yerr=[lower, upper],
        marker="o",
        capsize=5,
        color=COLORS[RISK_MASS_ARM],
    )
    axes[0].axhline(0, color="black")
    axes[0].set_title("Native Risk Mass − Base: AUPRC H8")
    axes[0].grid(alpha=0.25)
    nll = [summary["comparisons"][split]["next_state_nll"] for split in SPLITS]
    means = [item["mean_delta_nll"] for item in nll]
    lower = [
        max(0.0, mean - item["delta_nll_ci95"][0]) for mean, item in zip(means, nll)
    ]
    upper = [
        max(0.0, item["delta_nll_ci95"][1] - mean) for mean, item in zip(means, nll)
    ]
    axes[1].errorbar(
        SPLITS,
        means,
        yerr=[lower, upper],
        marker="o",
        capsize=5,
        color=COLORS[RISK_MASS_ARM],
    )
    axes[1].axhline(0, color="black")
    axes[1].set_title("Native Risk Mass − Base: NLL")
    axes[1].grid(alpha=0.25)
    fig.suptitle("Post-hoc paired hierarchical bootstrap (95% CI)")
    _save(fig, output, "risk_mass_deltas")


def _render_per_seed(summary, output: Path) -> None:
    fig, axis = plt.subplots(figsize=(9, 5.2))
    seeds = summary["training_seeds"]
    for arm in (BASELINE_ARM, RISK_MASS_ARM):
        values = [
            summary["arms"][arm]["splits"]["test_id"]["by_seed"][str(seed)]["hazard"][
                "8"
            ]["auprc"]
            for seed in seeds
        ]
        axis.plot(seeds, values, marker="o", label=NAMES[arm], color=COLORS[arm])
    axis.set_xlabel("Training seed")
    axis.set_ylabel("AUPRC H8 test_id")
    axis.set_title("Per-seed post-hoc comparison")
    axis.grid(alpha=0.25)
    axis.legend()
    _save(fig, output, "risk_mass_test_id_multiseed")


def _update_dashboard(output: Path) -> None:
    path = output / "index.html"
    page = path.read_text()
    if MARKER_START in page:
        page = (
            page[: page.index(MARKER_START)]
            + page[page.index(MARKER_END) + len(MARKER_END) :]
        )
    section = (
        MARKER_START
        + "<section><h2>ASM-X Base vs ASM-X + Native Risk Mass</h2><p class='warning'>Exploratory and selected after P2 test results were observed. It does not modify registered G0–G5.</p>"
        + "".join(
            f'<h3>{name.replace("_", " ")}</h3><a href="{name}.svg"><img src="{name}.svg"></a>'
            for name in (
                "risk_mass_metrics",
                "risk_mass_horizons",
                "risk_mass_deltas",
                "risk_mass_test_id_multiseed",
            )
        )
        + "<p><a href='risk_mass_extension_summary.json'>Diagnostic summary JSON</a> · <a href='risk_mass_extension_pretrain_manifest.json'>Pretrain manifest</a> · <a href='risk_mass_extension_checkpoint_seal.json'>Checkpoint seal</a></p></section>"
        + MARKER_END
    )
    path.write_text(page + section)


def render_risk_mass_extension(
    summary_or_path, output: str | Path | None = None
) -> list[Path]:
    if isinstance(summary_or_path, (str, Path)):
        source = Path(summary_or_path)
        summary = json.loads(source.read_text())
        target = Path(output) if output else source.parent
    else:
        summary = summary_or_path
        target = Path(output or ".")
    _render_metric_grid(summary, target)
    _render_horizons(summary, target)
    _render_deltas(summary, target)
    _render_per_seed(summary, target)
    _update_dashboard(target)
    names = (
        "risk_mass_metrics",
        "risk_mass_horizons",
        "risk_mass_deltas",
        "risk_mass_test_id_multiseed",
    )
    return [
        target / f"{name}.{extension}" for name in names for extension in ("png", "svg")
    ]


__all__ = ["render_risk_mass_extension"]
