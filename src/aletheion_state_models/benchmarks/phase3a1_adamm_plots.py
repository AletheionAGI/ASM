'''Charts for the matched AdamW versus AdamM Phase 3A.1 ablation.'''
from pathlib import Path
import html
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _save(fig, target, name):
    fig.tight_layout()
    for ext in ("png", "svg"): fig.savefig(target / f"{name}.{ext}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _arms(summary):
    return {
        "AdamW fixed-32": summary["adamw_baseline"]["selected_fixed_32"],
        "AdamM fixed-32": summary["variants"]["adamm_fixed_32"],
        "AdamW adaptive": summary["adamw_baseline"]["selected_adaptive_32"],
        "AdamM adaptive": summary["variants"]["adamm_adaptive_32"],
    }


def _quality(summary, target):
    arms = _arms(summary); fig, ax = plt.subplots(figsize=(8.5, 5.3))
    for name, value in arms.items():
        ax.errorbar(value["mean_rank_mean"], value["test_ce_mean"], xerr=value["mean_rank_std"], yerr=value["test_ce_std"], marker="o", capsize=4, label=name)
    ax.set(xlabel="Rank hard médio", ylabel="CE de test", title="3A.1-AdamM — qualidade versus rank")
    ax.grid(alpha=.25); ax.legend(); _save(fig, target, "quality_vs_rank_optimizer")
    return "quality_vs_rank_optimizer"


def _paired(summary, target):
    comparisons = summary["optimizer_comparisons"]; labels = [str(x["seed"]) for x in comparisons["adamm_fixed_32"]]
    fixed = [x["adamm_minus_adamw_test_ce"] for x in comparisons["adamm_fixed_32"]]
    adaptive = [x["adamm_minus_adamw_test_ce"] for x in comparisons["adamm_adaptive_32"]]
    x = np.arange(len(labels)); fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.bar(x-.2, fixed, .4, label="fixed-32"); ax.bar(x+.2, adaptive, .4, label="adaptive")
    ax.axhline(0, color="black"); ax.set_xticks(x, labels); ax.set(xlabel="Seed", ylabel="CE AdamM − AdamW", title="Efeito pareado do otimizador")
    ax.grid(axis="y", alpha=.25); ax.legend(); _save(fig, target, "paired_optimizer_deltas")
    return "paired_optimizer_deltas"


def _learning(summary, target):
    arms = _arms(summary); fig, ax = plt.subplots(figsize=(10, 5.3))
    for name, value in arms.items():
        histories = value["runs"]; tokens = np.asarray([point["tokens"] for point in histories[0]["history"]])
        curves = np.asarray([[point["validation_ce"] for point in run["history"]] for run in histories])
        ax.plot(tokens, curves.mean(0), label=name); ax.fill_between(tokens, curves.min(0), curves.max(0), alpha=.12)
    ax.set(xlabel="Tokens", ylabel="CE de validação", title="AdamW versus AdamM — curvas pareadas")
    ax.grid(alpha=.25); ax.legend(); _save(fig, target, "validation_ce_optimizer")
    return "validation_ce_optimizer"


def _cost(summary, target):
    arms = _arms(summary); names = list(arms); x = np.arange(len(names)); fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    axes[0].bar(x, [arms[n]["tokens_per_second_mean"] for n in names]); axes[0].set_ylabel("Tokens/s")
    axes[1].bar(x, [arms[n]["peak_memory_mb_mean"] for n in names]); axes[1].set_ylabel("Pico CUDA MiB")
    optimizer = summary["optimizer"]
    state = [optimizer["adamw_state_bytes_by_variant"]["selected_fixed_32"]["mean"], optimizer["state_bytes_by_variant"]["adamm_fixed_32"]["mean"], optimizer["adamw_state_bytes_by_variant"]["selected_adaptive_32"]["mean"], optimizer["state_bytes_by_variant"]["adamm_adaptive_32"]["mean"]]
    axes[2].bar(x, np.asarray(state) / 2**20); axes[2].set_ylabel("Estado do otimizador MiB")
    for ax in axes: ax.set_xticks(x, names, rotation=30, ha="right"); ax.grid(axis="y", alpha=.25)
    fig.suptitle("Custo observado do otimizador"); _save(fig, target, "optimizer_observed_cost")
    return "optimizer_observed_cost"


def render_adamm_charts(summary, directory):
    target = Path(directory); target.mkdir(parents=True, exist_ok=True)
    charts = [_quality(summary, target), _paired(summary, target), _learning(summary, target), _cost(summary, target)]
    gates = "".join(f"<li><b>{html.escape(name)}</b>: {'PASS' if value else 'FAIL'}</li>" for name, value in summary["gates"].items())
    figures = "".join(f'<section><h2>{name.replace("_", " ")}</h2><a href="{name}.svg"><img src="{name}.svg"></a></section>' for name in charts)
    page = f'''<!doctype html><html><head><meta charset="utf-8"><title>3A.1-AdamM</title><style>body{{font:16px sans-serif;max-width:1200px;margin:auto;padding:2rem}}img{{width:100%}}section{{margin:2rem 0}}</style></head><body><h1>ASM-VR 3A.1-AdamM</h1><p>Troca controlada apenas do otimizador. Caminhos do modelo permanecem densos.</p><ul>{gates}</ul>{figures}</body></html>'''
    (target / "index.html").write_text(page, encoding="utf-8")


__all__ = ["render_adamm_charts"]
