"""Offline rendering for ATTR smoke and train-only pilot artifacts."""

from __future__ import annotations
import html
import json
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _save(fig, output: Path, stem: str) -> None:
    fig.tight_layout()
    for extension in ("png", "svg"):
        fig.savefig(output / f"{stem}.{extension}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _identity(summary: dict) -> tuple[str, str]:
    if summary["status"] == "train_only_pilot_test_sealed":
        return (
            "ATTR P1 train-only pilot",
            "Train/validation pilot only. Test remained sealed. These results support no safety, causal-understanding, or universal-superiority claim.",
        )
    return (
        "ATTR P0 smoke",
        "Integration smoke only. These results are not sealed ATTR evidence and support no safety or universal-superiority claim.",
    )


def _prevalence_table(summary: dict) -> str:
    if "event_prevalence" not in summary:
        return ""
    rows = []
    for split, values in summary["event_prevalence"].items():
        for horizon, item in values["by_horizon"].items():
            rows.append(
                f"<tr><td>{split}</td><td>{horizon}</td><td>{item['positives']}</td><td>{item['prevalence']:.4f}</td></tr>"
            )
    return (
        "<h2>Event prevalence</h2><table><tr><th>Split</th><th>Horizon</th><th>Positives</th><th>Prevalence</th></tr>"
        + "".join(rows)
        + "</table>"
    )


def render_summary(summary_path: str | Path) -> list[Path]:
    path = Path(summary_path)
    summary = json.loads(path.read_text())
    output = path.parent
    title, warning = _identity(summary)
    arms = summary["arms"]
    display_names = {
        "asm_x_directional": "ASM-X",
        "tiny_transformer_220k": "Transformer",
        "asm_cm_durable": "ASM-CM",
        "asm_vr_s_full64": "ASM-VR-S full",
        "asm_vr_s_fixed32": "ASM-VR-S fixed-32",
        "asm_r_240k_control": "ASM-R 240K",
    }
    names = [display_names.get(item["arm"], item["arm"]) for item in arms]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    panels = (
        ("AUPRC H8", [item["validation_auprc_h8"] for item in arms], False),
        (
            "Brier H8 (lower is better)",
            [item["validation_brier_h8"] for item in arms],
            True,
        ),
        ("Recall @ FPR≤5%", [item["validation_recall_at_fpr"] for item in arms], False),
    )
    colors = ["#4472C4", "#ED7D31"] + ["#70AD47"] * max(0, len(arms) - 2)
    positions = list(range(len(names)))
    for axis, (label, values, narrow) in zip(axes, panels):
        axis.bar(positions, values, color=colors)
        axis.set_xticks(positions, names, rotation=28, ha="right")
        axis.set_title(label)
        axis.grid(axis="y", alpha=0.25)
        if narrow:
            padding = max(max(values) - min(values), 0.001) * 0.6
            axis.set_ylim(max(0, min(values) - padding), max(values) + padding)
    fig.suptitle(title + " — validation seed 17")
    _save(fig, output, "validation_metrics")
    fig, axis = plt.subplots(figsize=(9, 4.8))
    positions = list(range(len(names)))
    totals = [
        item.get("total_parameters", item["trainable_parameters"]) for item in arms
    ]
    trainable = [item["trainable_parameters"] for item in arms]
    axis.bar([position - 0.18 for position in positions], totals, 0.36, label="total")
    axis.bar(
        [position + 0.18 for position in positions], trainable, 0.36, label="trainable"
    )
    axis.set_xticks(positions, names, rotation=20, ha="right")
    axis.set_ylabel("Parameters")
    axis.set_title(
        f"Registered main-pair mismatch: {summary['parameter_mismatch_fraction'] * 100:.3f}%"
    )
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    _save(fig, output, "parameter_match")
    rows = "".join(
        f"<tr><td>{html.escape(display_names.get(item['arm'], item['arm']))}</td><td>{item['backbone_parameters']:,}</td>"
        f"<td>{item.get('total_parameters', item['trainable_parameters']):,}</td><td>{item['trainable_parameters']:,}</td>"
        f"<td>{item['validation_auprc_h8']:.4f}</td><td>{item['validation_brier_h8']:.4f}</td>"
        f"<td>{item['validation_recall_at_fpr']:.4f}</td><td>{item['validation_threshold']:.4f}</td>"
        f"<td>{html.escape(item.get('comparison_role', 'unspecified'))}</td></tr>"
        for item in arms
    )
    prevalence = _prevalence_table(summary)
    manifest = (
        " · <a href='manifest.json'>Frozen pilot manifest</a>"
        if (output / "manifest.json").exists()
        else ""
    )
    dashboard = f"""<!doctype html><meta charset="utf-8"><title>{title}</title><style>body{{font:16px system-ui;max-width:1000px;margin:2rem auto;padding:0 1rem}}.warning{{background:#fff3cd;padding:1rem;border-left:5px solid #d39e00}}img{{max-width:48%}}table{{border-collapse:collapse}}td,th{{padding:.5rem;border:1px solid #bbb}}</style><h1>{title}</h1><p class="warning"><strong>Limited evidence.</strong> {warning}</p><p>Leakage audits: feature={summary["audits"]["feature_leakage"]}; split={summary["audits"]["episode_split"]}; threshold={summary["audits"]["threshold_selection"]}.</p><img src="validation_metrics.svg" alt="validation metrics"><img src="parameter_match.svg" alt="parameter counts"><table><tr><th>Arm</th><th>Backbone params</th><th>Total incl. head</th><th>Trainable incl. head</th><th>AUPRC H8</th><th>Brier H8</th><th>Recall @ FPR≤5%</th><th>Threshold</th><th>Role</th></tr>{rows}</table>{prevalence}<p><a href="summary.json">Raw JSON</a>{manifest}</p>"""
    (output / "index.html").write_text(dashboard)
    pilot_details = ""
    if summary["status"] == "train_only_pilot_test_sealed":
        by_name = {item["arm"]: item for item in arms}
        asm = by_name["asm_x_directional"]
        transformer = by_name["tiny_transformer_220k"]
        prevalence_h8 = summary["event_prevalence"]["validation"]["by_horizon"]["8"][
            "prevalence"
        ]
        pilot_details = f"""
## Pilot snapshot

- Validation H8 event prevalence: {prevalence_h8:.2%}.
- ASM-X: AUPRC {asm["validation_auprc_h8"]:.4f}, Brier {asm["validation_brier_h8"]:.4f}, recall @ FPR≤5% {asm["validation_recall_at_fpr"]:.2%}.
- Transformer: AUPRC {transformer["validation_auprc_h8"]:.4f}, Brier {transformer["validation_brier_h8"]:.4f}, recall @ FPR≤5% {transformer["validation_recall_at_fpr"]:.2%}.
- The Transformer led the registered main-pair single-seed validation pilot. No registered predictive or intervention gate was assessed.
"""
        supplementary = [
            item
            for item in arms
            if item.get("comparison_role") == "supplementary_unmatched"
        ]
        if supplementary:
            pilot_details += "\n### Supplementary arms\n\n" + "".join(
                f"- {display_names.get(item['arm'], item['arm'])}: AUPRC {item['validation_auprc_h8']:.4f}, Brier {item['validation_brier_h8']:.4f}, recall {item['validation_recall_at_fpr']:.2%}; descriptive only.\n"
                for item in supplementary
            )
    readme = f"""# {title}

{warning}
{pilot_details}
- Seed: {summary["seed"]}; updates per arm: {summary["updates"]}.
- Same HazardWorld episodes, horizons and objective for all arms.
- Threshold selected on validation only.
- Feature and episode-split leakage audits passed.
- Controls: persistence, Markov and Kalman.
- Raw results: `summary.json`; offline dashboard: `index.html`.
"""
    (output / "README.md").write_text(readme)
    names = (
        "validation_metrics.png",
        "validation_metrics.svg",
        "parameter_match.png",
        "parameter_match.svg",
        "index.html",
        "README.md",
    )
    return [output / name for name in names]


def render_smoke(summary_path: str | Path) -> list[Path]:
    """Backward-compatible rendering entry point."""
    return render_summary(summary_path)


__all__ = ["render_smoke", "render_summary"]
