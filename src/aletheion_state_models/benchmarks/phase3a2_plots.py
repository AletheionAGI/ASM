"""Comparative charts and dashboard for ASM-VR-R versus ASM-VR-S."""
from pathlib import Path
import html
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from .phase3a2_variants import BASES, RANK_ARMS


def _save(fig, target, name):
    fig.tight_layout()
    for ext in ("png", "svg"): fig.savefig(target / f"{name}.{ext}", dpi=180, bbox_inches="tight")
    plt.close(fig); return name


def _learning(summary, target):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), sharey=True)
    for ax, base in zip(axes, BASES):
        for arm in RANK_ARMS:
            runs = summary["variants"][f"{base}_{arm}"]["runs"]
            tokens = np.asarray([point["tokens"] for point in runs[0]["history"]])
            curves = np.asarray([[point["validation_ce"] for point in run["history"]] for run in runs])
            ax.plot(tokens, curves.mean(0), label=arm); ax.fill_between(tokens, curves.min(0), curves.max(0), alpha=.1)
        ax.set_title(base.upper()); ax.set_xlabel("Tokens"); ax.grid(alpha=.25); ax.legend(fontsize=7)
    axes[0].set_ylabel("CE de validação"); fig.suptitle("3A.2 — aprendizado por base e rank")
    return _save(fig, target, "validation_ce_by_base_rank")


def _quality_rank(summary, target):
    fig, ax = plt.subplots(figsize=(9, 5.6)); colors = {"vr_r": "#2980b9", "vr_s": "#e67e22"}
    for base in BASES:
        fixed = []
        for arm in RANK_ARMS:
            value = summary["variants"][f"{base}_{arm}"]; marker = "*" if arm == "adaptive_32" else "o"
            ax.errorbar(value["mean_rank_mean"], value["test_ce_mean"], xerr=value["mean_rank_std"], yerr=value["test_ce_std"], marker=marker, color=colors[base], capsize=3, label=f"{base}:{arm}")
            if arm != "adaptive_32": fixed.append((value["mean_rank_mean"], value["test_ce_mean"]))
        ax.plot(*zip(*sorted(fixed)), linestyle="--", color=colors[base])
    ax.set(xlabel="Rank hard médio", ylabel="CE de test", title="Fronteiras fixed e adaptativas")
    ax.grid(alpha=.25); ax.legend(fontsize=7, ncol=2)
    return _save(fig, target, "quality_vs_rank_frontiers")


def _paired_base(summary, target):
    comparisons = summary["paired_base_comparisons"]; x = np.arange(len(RANK_ARMS)); fig, ax = plt.subplots(figsize=(10, 5.2))
    for seed_index in range(3):
        values = [comparisons[arm][seed_index]["s_minus_r_test_ce"] for arm in RANK_ARMS]
        ax.plot(x, values, marker="o", alpha=.55, label=f"seed {comparisons[RANK_ARMS[0]][seed_index]['seed']}")
    means = [np.mean([item["s_minus_r_test_ce"] for item in comparisons[arm]]) for arm in RANK_ARMS]
    ax.plot(x, means, color="black", linewidth=2.5, marker="s", label="média")
    ax.axhline(0, color="black", linewidth=1); ax.set_xticks(x, RANK_ARMS, rotation=25, ha="right")
    ax.set(ylabel="CE S − R", title="Efeito pareado da base; negativo favorece S"); ax.grid(alpha=.25); ax.legend()
    return _save(fig, target, "paired_s_minus_r_deltas")


def _cost(summary, target):
    names = [f"{base}_{arm}" for base in BASES for arm in RANK_ARMS]; x = np.arange(len(names)); fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    axes[0].bar(x, [summary["variants"][name]["tokens_per_second_mean"] for name in names]); axes[0].set_ylabel("Tokens/s")
    axes[1].bar(x, [summary["variants"][name]["peak_memory_mb_mean"] for name in names]); axes[1].set_ylabel("Pico CUDA MiB")
    for ax in axes: ax.set_xticks(x, names, rotation=45, ha="right", fontsize=7); ax.grid(axis="y", alpha=.25)
    fig.suptitle("Custo observado denso por base e rank")
    return _save(fig, target, "observed_dense_cost")


def _adaptive_ranges(summary, target):
    fig, ax = plt.subplots(figsize=(8.5, 5)); offsets = {"vr_r": -.12, "vr_s": .12}; x = np.arange(3)
    for base in BASES:
        runs = summary["variants"][f"{base}_adaptive_32"]["runs"]; means = np.asarray([run["mean_rank"] for run in runs])
        low = means - np.asarray([run["rank_min"] for run in runs]); high = np.asarray([run["rank_max"] for run in runs]) - means
        ax.errorbar(x + offsets[base], means, yerr=np.vstack([low, high]), fmt="o", capsize=5, label=base)
    ax.axhline(32, linestyle="--", color="black"); ax.set_xticks(x, [17, 29, 43]); ax.set(xlabel="Seed", ylabel="Rank", title="Faixa de rank adaptativo")
    ax.grid(alpha=.25); ax.legend(); return _save(fig, target, "adaptive_rank_ranges")


def _heatmap(summary, target):
    matrix = np.asarray([[summary["variants"][f"{base}_{arm}"]["test_ce_mean"] for arm in RANK_ARMS] for base in BASES])
    fig, ax = plt.subplots(figsize=(9, 3.5)); image = ax.imshow(matrix, cmap="viridis_r", aspect="auto")
    for row in range(2):
        for col in range(len(RANK_ARMS)): ax.text(col, row, f"{matrix[row,col]:.4f}", ha="center", va="center", color="white" if matrix[row,col] > matrix.mean() else "black")
    ax.set_xticks(range(len(RANK_ARMS)), RANK_ARMS, rotation=25, ha="right"); ax.set_yticks(range(2), [b.upper() for b in BASES]); ax.set_title("Test CE por base × rank"); fig.colorbar(image, ax=ax)
    return _save(fig, target, "test_ce_heatmap")


def _efficiency(summary, target):
    fig, ax = plt.subplots(figsize=(8.5, 5.3))
    for base, color in (("vr_r", "#2980b9"), ("vr_s", "#e67e22")):
        for arm in RANK_ARMS:
            value = summary["variants"][f"{base}_{arm}"]; ax.scatter(value["tokens_per_second_mean"], value["test_ce_mean"], s=value["peak_memory_mb_mean"] * 2, color=color, alpha=.65)
            ax.annotate(arm, (value["tokens_per_second_mean"], value["test_ce_mean"]), fontsize=7)
    ax.set(xlabel="Tokens/s observados", ylabel="CE de test", title="Qualidade × throughput; tamanho = pico CUDA")
    ax.grid(alpha=.25); return _save(fig, target, "quality_vs_observed_throughput")


def render_phase3a2_charts(summary, directory):
    target = Path(directory); target.mkdir(parents=True, exist_ok=True)
    charts = [_learning(summary, target), _quality_rank(summary, target), _paired_base(summary, target), _cost(summary, target), _adaptive_ranges(summary, target), _heatmap(summary, target), _efficiency(summary, target)]
    gates = {**summary["base_gates"]}
    for base in BASES:
        gates.update({f"{base}:{name}": value for name, value in summary["adaptive"][base]["gates"].items()})
    gate_html = "".join(f"<li><b>{html.escape(name)}</b>: {'PASS' if value else 'FAIL'}</li>" for name, value in gates.items())
    figures = "".join(f'<section><h2>{name.replace("_", " ")}</h2><a href="{name}.svg"><img src="{name}.svg"></a></section>' for name in charts)
    page = "<!doctype html><html><head><meta charset='utf-8'><title>ASM-VR 3A.2</title><style>body{font:16px sans-serif;max-width:1250px;margin:auto;padding:2rem}img{width:100%}section{margin:2rem 0}</style></head><body><h1>ASM-VR-R versus ASM-VR-S</h1><p>Custos observados em caminhos densos; não são speedup causal por rank.</p><ul>" + gate_html + "</ul>" + figures + "</body></html>"
    (target / "index.html").write_text(page, encoding="utf-8")


__all__ = ["render_phase3a2_charts"]
