"""Offline charts for long-context ASM-CM-VR gates."""

from pathlib import Path
import html
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

LABELS = {
    "cm_vr_full64": "full-64",
    "cm_vr_fixed32": "fixed-32",
    "cm_vr_adaptive32": "adaptive-32",
}
COLORS = {
    "cm_vr_full64": "tab:blue",
    "cm_vr_fixed32": "tab:orange",
    "cm_vr_adaptive32": "tab:green",
}


def _save(fig, target, name):
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(target / f"{name}.{ext}", dpi=180, bbox_inches="tight")
    plt.close(fig)
    return name


def _failure_table(summary):
    rows = []
    for arm, data in summary["arms"].items():
        for run in data["runs"]:
            for item in run["test"]:
                if item.get("status") == "failed" or not item.get("ce_finite", False):
                    error = item.get("error", "non-finite cross entropy")
                    rows.append((arm, run["seed"], "MQAR", item["length"], error))
            for item in run["streaming"]:
                if item.get("status") == "failed":
                    position = item.get("failed_at_position", "unknown")
                    error = f"{item.get('error', 'failed')} at token {position}"
                    rows.append((arm, run["seed"], "streaming", item["length"], error))
    if not rows:
        return "<p>No recorded numerical failures.</p>"
    body = "".join(
        "<tr>"
        + "".join(f"<td>{html.escape(str(value))}</td>" for value in row)
        + "</tr>"
        for row in rows
    )
    return (
        "<table><tr><th>Arm</th><th>Seed</th><th>Axis</th><th>Length</th><th>Failure</th></tr>"
        + body
        + "</table>"
    )


def render(summary, directory):
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    charts = []
    fig, ax = plt.subplots(figsize=(9, 5.2))
    for arm, data in summary["arms"].items():
        for row in data["runs"]:
            ok = [item for item in row["test"] if item.get("status") != "failed"]
            ax.plot(
                [item["length"] for item in ok],
                [item["accuracy"] for item in ok],
                marker="o",
                color=COLORS[arm],
                alpha=0.45,
                label=f"{LABELS[arm]} seed {row['seed']}",
            )
    ax.axhline(0.8, color="black", linestyle="--", label="gate 80%")
    ax.set_xscale("log", base=2)
    ax.set(
        xlabel="Comprimento MQAR",
        ylabel="Acurácia held-out",
        title="Generalização associativa longa",
    )
    ax.grid(alpha=0.25)
    ax.legend(ncol=2)
    charts.append(_save(fig, target, "mqar_long_accuracy"))
    fig, ax = plt.subplots(figsize=(9, 5.2))
    for arm, data in summary["arms"].items():
        for row in data["runs"]:
            ax.plot(
                [item["length"] for item in row["history"]],
                [item["validation_accuracy"] for item in row["history"]],
                marker="o",
                color=COLORS[arm],
                alpha=0.45,
                label=f"{LABELS[arm]} seed {row['seed']}",
            )
    ax.set_xscale("log", base=2)
    ax.set(
        xlabel="Comprimento do estágio",
        ylabel="Acurácia de validação",
        title="Currículo por seed",
    )
    ax.grid(alpha=0.25)
    ax.legend(ncol=2)
    charts.append(_save(fig, target, "curriculum_validation"))
    for metric, ylabel, name in (
        ("tokens_per_second_mean", "Tokens/s", "streaming_throughput"),
        ("retained_state_bytes_mean", "Bytes retidos", "streaming_state_bytes"),
    ):
        fig, ax = plt.subplots(figsize=(8.5, 5))
        for arm, data in summary["arms"].items():
            items = data["streaming"]
            lengths = sorted(int(key) for key in items)
            values = [items[str(length)][metric] for length in lengths]
            ax.plot(lengths, values, marker="o", label=LABELS[arm])
        ax.set_xscale("log", base=2)
        ax.set(
            xlabel="Comprimento do stream",
            ylabel=ylabel,
            title=ylabel + " no streaming",
        )
        ax.grid(alpha=0.25)
        ax.legend()
        charts.append(_save(fig, target, name))
    fig, ax = plt.subplots(figsize=(9, 5))
    labels = []
    normal = []
    no_read = []
    no_write = []
    for arm, data in summary["arms"].items():
        for row in data["runs"]:
            labels.append(f"{LABELS[arm]} s{row['seed']}")
            normal.append(
                next(item["accuracy"] for item in row["test"] if item["length"] == 40)
            )
            no_read.append(row["no_read"]["accuracy"])
            no_write.append(row["no_write"]["accuracy"])
    x = list(range(len(labels)))
    ax.bar([i - 0.25 for i in x], normal, 0.25, label="normal")
    ax.bar(x, no_read, 0.25, label="sem leitura")
    ax.bar([i + 0.25 for i in x], no_write, 0.25, label="sem escrita")
    ax.set_xticks(x, labels, rotation=25)
    ax.set_ylabel("Acurácia MQAR-40")
    ax.set_title("Canários causais multiseed")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    charts.append(_save(fig, target, "memory_ablations_multiseed"))
    adaptive = summary["arms"]["cm_vr_adaptive32"]["runs"]
    if adaptive:
        fig, ax = plt.subplots(figsize=(9, 5.2))
        for row in adaptive:
            ax.plot(
                [item["length"] for item in row["history"]],
                [item["mean_rank"] for item in row["history"]],
                marker="o",
                label=f"seed {row['seed']}",
            )
        ax.axhline(32, color="black", linestyle="--", label="target 32")
        ax.set_xscale("log", base=2)
        ax.set(
            xlabel="Comprimento do estágio",
            ylabel="Rank lógico médio",
            title="Controller adaptive-32",
        )
        ax.grid(alpha=0.25)
        ax.legend()
        charts.append(_save(fig, target, "adaptive_rank"))
    gates = "".join(
        f"<li><b>{html.escape(k)}</b>: {'PASS' if v else 'FAIL'}</li>"
        for k, v in summary["gates"].items()
    )
    figures = "".join(
        f'<section><h2>{name.replace("_", " ")}</h2><a href="{name}.svg"><img src="{name}.svg"></a></section>'
        for name in charts
    )
    failures = _failure_table(summary)
    page = (
        "<!doctype html><html><head><meta charset='utf-8'><title>ASM-CM-VR long gates</title>"
        "<style>body{font:16px sans-serif;max-width:1200px;margin:auto;padding:2rem}img{width:100%}"
        "table{border-collapse:collapse;width:100%}td,th{border:1px solid #bbb;padding:.5rem;text-align:left}"
        ".warning{background:#fff3cd;border-left:5px solid #d39e00;padding:1rem}</style></head>"
        "<body><h1>ASM-CM-VR fixed-32 long multiseed</h1>"
        "<p class='warning'><b>Not promoted.</b> Logical rank did not reduce physical state bytes, "
        "and full/fixed seed 29 recorded non-finite 32K failures.</p><ul>"
        + gates
        + "</ul><h2>Recorded failures</h2>"
        + failures
        + figures
        + "<p><a href='summary.json'>Raw summary JSON</a> · <a href='manifest.json'>Manifest</a></p>"
        + "</body></html>"
    )
    (target / "index.html").write_text(page, encoding="utf-8")


__all__ = ["render"]
