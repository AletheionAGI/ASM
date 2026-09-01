"""Separate ATTR-TG1 figures and a small trajectory-only dashboard."""

from __future__ import annotations

import html
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def _pyplot():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("trajectory plots require the 'viz' extra") from exc
    return plt


def _save(figure, target: Path, stem: str) -> list[Path]:
    paths = []
    for suffix in ("png", "svg"):
        path = target / f"{stem}.{suffix}"
        figure.savefig(path, bbox_inches="tight", dpi=180 if suffix == "png" else None)
        paths.append(path)
    return paths


def _quality_figure(summary: Mapping[str, Any]):
    plt = _pyplot()
    labels = [f"H{horizon}" for horizon in (1, 4, 8)]
    figure, axes = plt.subplots(1, 3, figsize=(10.5, 3.2))
    arms = (("asm_x_base", "ASM-X Base"), ("transformer_base", "Transformer"))
    metrics = (
        ("auprc", "AUPRC", False),
        ("brier", "Brier ↓", False),
        ("joint", "Joint trajectory NLL ↓", True),
    )
    for axis, (field, title, is_nll) in zip(axes, metrics):
        for arm, name in arms:
            quality = summary["splits"]["test_id"][arm]["quality_by_horizon"]
            values = [
                quality[label]["trajectory_nll"][field]
                if is_nll
                else quality[label][field]
                for label in labels
            ]
            axis.plot(labels, values, marker="o", label=name)
        axis.set_title(title)
        axis.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    figure.suptitle("ATTR-TG1 ID quality by predicted-trajectory horizon")
    return figure


def _delta_figure(paired: Mapping[str, Any] | None):
    plt = _pyplot()
    figure, axes = plt.subplots(1, 3, figsize=(10.5, 3.2))
    metrics = ("auprc", "brier", "event_logloss")
    if not paired:
        for axis in axes:
            axis.text(0.5, 0.5, "No paired comparison", ha="center", va="center")
            axis.set_axis_off()
        figure.suptitle("Paired deltas")
        return figure
    rows = paired.get("by_horizon", {})
    for axis, metric in zip(axes, metrics):
        values, lows, highs = [], [], []
        for label in ("H1", "H4", "H8"):
            item = rows[label][metric]
            value, interval = item["delta"], item["ci95"]
            values.append(value)
            lows.append(max(0, value - interval[0]))
            highs.append(max(0, interval[1] - value))
        axis.errorbar((1, 4, 8), values, yerr=(lows, highs), marker="o", capsize=3)
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_title(f"Δ {metric}")
        axis.set_xlabel("Horizon")
        axis.grid(alpha=0.25)
    direction = f"{paired.get('left_arm', 'candidate')} minus {paired.get('right_arm', 'reference')}"
    figure.suptitle(f"Paired hierarchical bootstrap: {direction}")
    return figure


def _anticipation_figure(summary: Mapping[str, Any]):
    """Compare both arms using risk derived only from their predicted trajectories."""
    plt = _pyplot()
    splits = ("test_id", "test_shift", "test_ood")
    labels = ("ID", "shift", "OOD")
    arms = ("asm_x_base", "transformer_base")
    names = ("ASM-X Base", "Transformer")
    colors = ("#3569b7", "#ef7d28")
    figure, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    x = list(range(len(splits)))
    width = 0.36
    metrics = (
        ("auprc", "Trajectory-derived AUPRC H8", False),
        ("brier", "Trajectory-derived Brier H8 ↓", False),
        ("joint", "Joint trajectory NLL H8 ↓", True),
    )
    for axis, (metric, title, is_nll) in zip(axes, metrics):
        for offset, (arm, name, color) in enumerate(zip(arms, names, colors)):
            values = []
            for split in splits:
                quality = summary["splits"][split][arm]["quality_by_horizon"]["H8"]
                values.append(
                    quality["trajectory_nll"][metric] if is_nll else quality[metric]
                )
            positions = [item + (offset - 0.5) * width for item in x]
            axis.bar(positions, values, width=width, label=name, color=color)
        if metric == "auprc":
            prevalence = [
                summary["splits"][split][arms[0]]["quality_by_horizon"]["H8"][
                    "prevalence"
                ]
                for split in splits
            ]
            axis.scatter(
                x, prevalence, color="black", marker="_", s=130, label="prevalence"
            )
        axis.set_xticks(x, labels)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
    axes[0].legend(fontsize=8)
    figure.suptitle(
        "ATTR-TG1: representation → predicted physical trajectory → fixed unsafe predicate"
    )
    return figure


def _dashboard(summary: Mapping[str, Any], target: Path, assets: list[Path]) -> Path:
    rows = "".join(
        f"<tr><th>{split}</th><th>{arm}</th>"
        f"<td>{summary['splits'][split][arm]['quality_by_horizon']['H8']['auprc']:.6f}</td>"
        f"<td>{summary['splits'][split][arm]['quality_by_horizon']['H8']['brier']:.6f}</td>"
        f"<td>{summary['splits'][split][arm]['quality_by_horizon']['H8']['trajectory_nll']['joint']:.6f}</td>"
        f"<td>{summary['splits'][split][arm]['quality_by_horizon']['H8']['prevalence']:.6f}</td></tr>"
        for split in ("test_id", "test_shift", "test_ood")
        for arm in ("asm_x_base", "transformer_base")
    )
    gates = "".join(
        f"<li><b>{html.escape(name)}</b>: {html.escape(str(value))}</li>"
        for name, value in summary.get("gates", {}).items()
    )
    pngs = [path for path in assets if path.suffix == ".png"]
    images = "".join(
        f'<figure><img src="{html.escape(path.name)}" alt="{html.escape(path.stem)}"></figure>'
        for path in pngs
    )
    content = f"""<!doctype html><html><head><meta charset="utf-8">
<title>ATTR-TG1 trajectory-grounded anticipation</title>
<style>body{{font-family:sans-serif;max-width:1100px;margin:auto}}img{{max-width:100%}}td,th{{padding:.35rem;border:1px solid #bbb}}table{{border-collapse:collapse}}</style></head>
<body><h1>ATTR-TG1 trajectory-grounded anticipation</h1>
<p>Risk is derived only from sampled physical trajectories and a fixed unsafe predicate. No HazardHead score is used.</p>
<ul>{gates}</ul>
<table><thead><tr><th>Split</th><th>Arm</th><th>AUPRC H8</th><th>Brier H8</th><th>Joint trajectory NLL H8</th><th>Prevalence H8</th></tr></thead><tbody>{rows}</tbody></table>
{images}</body></html>"""
    path = target / "index.html"
    path.write_text(content, encoding="utf-8")
    return path


def render_trajectory_grounded(
    summary: Mapping[str, Any] | str | Path, docs_target: str | Path
) -> tuple[Path, ...]:
    """Render TG1 assets into a new, separate documentation target."""
    if isinstance(summary, (str, Path)):
        summary = json.loads(Path(summary).read_text(encoding="utf-8"))
    if not isinstance(summary, Mapping) or "quality_by_horizon" not in summary:
        raise ValueError("invalid trajectory summary")
    target = Path(docs_target)
    target.mkdir(parents=True, exist_ok=True)
    figures = (
        ("trajectory_grounded_anticipation", _anticipation_figure(summary)),
        ("quality_by_horizon", _quality_figure(summary)),
        ("paired_deltas", _delta_figure(summary.get("paired_deltas"))),
    )
    paths = []
    plt = _pyplot()
    for stem, figure in figures:
        paths.extend(_save(figure, target, stem))
        plt.close(figure)
    summary_path = target / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    paths.append(summary_path)
    paths.append(_dashboard(summary, target, paths))
    return tuple(paths)
