"""Matplotlib renderers matching the four legacy RTG artifact names."""

from __future__ import annotations

import math
from pathlib import Path
from statistics import fmean, stdev
from typing import Any

ARMS = ("R", "CM", "Z", "T")
REGIMES = ("ID", "shift", "OOD")
CONTRASTS = (("CM", "R"), ("CM", "Z"), ("CM", "T"), ("R", "Z"), ("R", "T"), ("Z", "T"))
COLORS = {"R": "#29c7ac", "CM": "#ffb45b", "Z": "#55a868", "T": "#c44e52"}


def render_rtg_charts(rows: list[dict[str, Any]], output_dir: Path) -> list[Path]:
    """Create PNG and SVG versions of the four frozen RTG chart names."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matplotlib.rcParams.update(
        {
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "svg.hashsalt": "attr-rtg-rcmz-v1",
        }
    )

    scalar = [row for row in rows if row.get("arm") in ARMS and "seed" in row]
    figures = {
        "architecture_quality": _architecture(plt, scalar),
        "governance": _governance(plt, scalar, "RCMZ governance outcomes"),
        "g_vs_c": _governance(
            plt, scalar, "RCMZ decision tradeoffs (legacy g_vs_c filename)"
        ),
        "seed_differences": _seed_differences(plt, scalar),
    }
    paths: list[Path] = []
    for name, figure in figures.items():
        for suffix in ("png", "svg"):
            path = output_dir / f"{name}.{suffix}"
            figure.savefig(
                path, dpi=170 if suffix == "png" else None, bbox_inches="tight"
            )
            paths.append(path)
        plt.close(figure)
    return paths


def _architecture(plt: Any, rows: list[dict[str, Any]]) -> Any:
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharex=True)
    _regime_errorbars(axes[0], rows, "h8_nll", "H8 NLL", lower_better=True)
    _regime_errorbars(axes[1], rows, "ece", "ECE-15", lower_better=True)
    figure.suptitle("RCMZ architecture quality · estimate and 95% CI")
    figure.tight_layout()
    return figure


def _governance(plt: Any, rows: list[dict[str, Any]], title: str) -> Any:
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.4), sharex=True)
    for axis, metric, label, lower in zip(
        axes,
        ("unsafe_selection", "safe_service", "coverage"),
        ("Unsafe selection", "Safe service", "Coverage"),
        (True, False, False),
    ):
        _regime_errorbars(axis, rows, metric, label, lower_better=lower)
    figure.suptitle(title + " · estimate and 95% CI")
    figure.tight_layout()
    return figure


def _regime_errorbars(
    axis: Any,
    rows: list[dict[str, Any]],
    metric: str,
    label: str,
    *,
    lower_better: bool,
) -> None:
    width = 0.19
    offsets = (-1.5 * width, -0.5 * width, 0.5 * width, 1.5 * width)
    found = False
    for arm, offset in zip(ARMS, offsets):
        means, errors = [], []
        for regime in REGIMES:
            values = _values(rows, arm, regime, metric)
            mean, error = _mean_ci(values)
            means.append(mean)
            errors.append(error)
            found |= bool(values)
        axis.bar(
            [x + offset for x in range(3)],
            means,
            width,
            yerr=errors,
            capsize=3,
            color=COLORS[arm],
            label=arm,
            edgecolor="white",
            linewidth=0.5,
            error_kw={"ecolor": "black", "elinewidth": 1, "capthick": 1},
        )
    axis.set_xticks(range(3), REGIMES)
    axis.set_xlabel("Regime")
    axis.set_ylabel(
        label + (" (lower is better)" if lower_better else " (higher is better)")
    )
    axis.grid(axis="y", alpha=0.25)
    axis.set_axisbelow(True)
    if found:
        axis.legend(title="Arm", fontsize=8, frameon=False)
    else:
        axis.text(
            0.5, 0.5, "No eligible scalar rows", ha="center", transform=axis.transAxes
        )


def _seed_differences(plt: Any, rows: list[dict[str, Any]]) -> Any:
    figure, axes = plt.subplots(1, 2, figsize=(14, 5.2), sharex=True)
    seeds = sorted({int(row["seed"]) for row in rows})
    palette = plt.get_cmap("tab10")
    for axis, metric, title in zip(
        axes,
        ("h8_nll", "unsafe_selection"),
        ("H8 NLL pairwise deltas", "Unsafe-rate pairwise deltas"),
    ):
        width = 0.78 / max(1, len(seeds))
        for seed_index, seed in enumerate(seeds):
            deltas = [
                _delta(rows, left, right, seed, metric) for left, right in CONTRASTS
            ]
            offset = (seed_index - (len(seeds) - 1) / 2) * width
            axis.bar(
                [x + offset for x in range(6)],
                deltas,
                width,
                color=palette(seed_index),
                label=f"seed {seed}",
                edgecolor="white",
                linewidth=0.4,
            )
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_xticks(
            range(6), [f"{a}−{b}" for a, b in CONTRASTS], rotation=30, ha="right"
        )
        axis.set_ylabel("Left − right")
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
        axis.set_axisbelow(True)
        if seeds:
            axis.legend(fontsize=8, ncol=min(3, len(seeds)), frameon=False)
        else:
            axis.text(
                0.5,
                0.5,
                "No eligible seed pairs",
                ha="center",
                transform=axis.transAxes,
            )
    figure.suptitle("Six frozen contrasts · variation across five registered seeds")
    figure.tight_layout()
    return figure


def _values(
    rows: list[dict[str, Any]], arm: str, regime: str, metric: str
) -> list[float]:
    values = []
    for row in rows:
        if row.get("arm") == arm and str(row.get("regime")) == regime and metric in row:
            value = float(row[metric])
            if math.isfinite(value):
                values.append(value)
    return values


def _mean_ci(values: list[float]) -> tuple[float, float]:
    if not values:
        return math.nan, 0.0
    error = 1.96 * stdev(values) / math.sqrt(len(values)) if len(values) > 1 else 0.0
    return fmean(values), error


def _delta(
    rows: list[dict[str, Any]], left: str, right: str, seed: int, metric: str
) -> float:
    def arm_mean(arm: str) -> float:
        values = [
            float(row[metric])
            for row in rows
            if row.get("arm") == arm
            and int(row.get("seed", -1)) == seed
            and metric in row
        ]
        return fmean(values) if values else math.nan

    return arm_mean(left) - arm_mean(right)
