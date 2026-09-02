"""Render static ATTR-RTG charts from flattened registered evidence."""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

mpl.rcParams["svg.hashsalt"] = "attr-rtg-registered-v1"

COLORS = {"ASM-X": "#29c7ac", "Transformer": "#ffb45b"}
REGIMES = ("ID", "Shift", "OOD")


def _save(fig, output: Path, name: str) -> None:
    fig.tight_layout()
    for extension in ("png", "svg"):
        path = output / f"{name}.{extension}"
        metadata = {"Date": None} if extension == "svg" else None
        fig.savefig(path, dpi=180, bbox_inches="tight", metadata=metadata)
        if extension == "svg":
            lines = path.read_text(encoding="utf-8").splitlines()
            path.write_text("\n".join(line.rstrip() for line in lines) + "\n", encoding="utf-8")
    plt.close(fig)


def architecture_chart(rows: list[dict], output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    x, width = np.arange(3), 0.34
    for axis, metric, title in zip(axes, ("nmse", "nll"), ("NMSE de transição ↓", "NLL ↓")):
        for offset, model in zip((-width / 2, width / 2), COLORS):
            selected = [next(row for row in rows if row["regime"] == regime and row["model"] == model) for regime in REGIMES]
            values = np.array([row[metric] for row in selected])
            errors = np.array([[row[metric] - row[f"{metric}_ci_low"] for row in selected],
                               [row[f"{metric}_ci_high"] - row[metric] for row in selected]])
            axis.bar(x + offset, values, width, label=model, color=COLORS[model], yerr=errors, capsize=3)
        axis.set(title=title, xticks=x, xticklabels=REGIMES)
        axis.grid(axis="y", alpha=.2)
    axes[0].legend(frameon=False)
    fig.suptitle("ASM-X vs Transformer · estimativa e IC 95%")
    _save(fig, output, "architecture_quality")


def governance_chart(rows: list[dict], output: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3))
    specs = (("unsafe_rate", "Taxa unsafe ↓"), ("safe_service", "Safe-service ↑"), ("coverage", "Cobertura ↑"))
    x, width = np.arange(3), .18
    series = [(model, head) for model in COLORS for head in ("G", "C")]
    for axis, (metric, title) in zip(axes, specs):
        for index, (model, head) in enumerate(series):
            selected = [next(row for row in rows if row["regime"] == regime and row["model"] == model and row["head"] == head) for regime in REGIMES]
            values = np.array([row[metric] for row in selected])
            errors = np.array([[row[metric] - row[f"{metric}_ci_low"] for row in selected],
                               [row[f"{metric}_ci_high"] - row[metric] for row in selected]])
            axis.errorbar(x + (index - 1.5) * width, values, yerr=errors, marker="o" if head == "G" else "s",
                          linestyle="-", capsize=2, color=COLORS[model], label=f"{model} · {head}")
        axis.set(title=title, xticks=x, xticklabels=REGIMES)
        axis.grid(axis="y", alpha=.2)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Governança por cabeça G/C · estimativa e IC 95%")
    _save(fig, output, "governance")


def comparison_chart(rows: list[dict], output: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    specs = (("delta_safety", "C−G unsafe (↓ favorece C)"),
             ("delta_safe_service", "G−C safe-service (↑ favorece G)"),
             ("coverage_difference", "G−C cobertura (↑ favorece G)"))
    x, width = np.arange(3), .32
    for axis, (metric, title) in zip(axes, specs):
        for offset, model in zip((-width / 2, width / 2), COLORS):
            selected = [next(row for row in rows if row["regime"] == regime and row["model"] == model) for regime in REGIMES]
            values = np.array([row[metric] for row in selected])
            errors = np.array([[row[metric] - row[f"{metric}_ci_low"] for row in selected],
                               [row[f"{metric}_ci_high"] - row[metric] for row in selected]])
            axis.bar(x + offset, values, width, yerr=errors, capsize=3, color=COLORS[model], label=model)
        axis.axhline(0, color="#64748b", linewidth=.8)
        axis.set(title=title, xticks=x, xticklabels=REGIMES)
        axis.grid(axis="y", alpha=.2)
    axes[0].legend(frameon=False)
    fig.suptitle("Trade-off G vs C · estimativa e IC 95%")
    _save(fig, output, "g_vs_c")


def seeds_chart(rows: list[dict], output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for axis, metric, title in zip(axes, ("delta_nmse", "delta_nll"),
                                   ("ΔNMSE ASM-X−Transformer", "ΔNLL ASM-X−Transformer")):
        for regime, marker in zip(REGIMES, ("o", "s", "^")):
            selected = [row for row in rows if row["regime"] == regime]
            axis.plot([row["seed"] for row in selected], [row[metric] for row in selected], marker=marker, label=regime)
        axis.axhline(0, color="#64748b", linewidth=.8)
        axis.set(title=title, xlabel="Seed", ylabel="Diferença (menor é melhor)")
        axis.grid(alpha=.2)
    axes[0].legend(frameon=False)
    fig.suptitle("Variação entre as 5 seeds registradas")
    _save(fig, output, "seed_differences")


def render_all(tables: dict, output: Path) -> None:
    architecture_chart(tables["architecture"], output)
    governance_chart(tables["governance"], output)
    comparison_chart(tables["g_vs_c"], output)
    seeds_chart(tables["seeds"], output)
