'''PNG/SVG charts and offline dashboards for both Phase 3A.1 stages.'''
from pathlib import Path
import html
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _save(fig, target, name):
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(target / f"{name}.{ext}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _learning(summary, target, order):
    fig, ax = plt.subplots(figsize=(10, 5.4))
    for name in order:
        runs = summary["variants"][name]["runs"]
        tokens = np.asarray([point["tokens"] for point in runs[0]["history"]])
        curves = np.asarray([[point["validation_ce"] for point in run["history"]] for run in runs])
        ax.plot(tokens, curves.mean(0), label=name)
        ax.fill_between(tokens, curves.min(0), curves.max(0), alpha=.12)
    ax.set(xlabel="Tokens de treino", ylabel="CE de validação", title=f"{summary['stage']} — aprendizado pareado")
    ax.grid(alpha=.25); ax.legend(fontsize=7, ncol=2)
    _save(fig, target, "validation_ce_by_tokens"); return "validation_ce_by_tokens"


def _cost(summary, target, order):
    x = np.arange(len(order)); fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].bar(x, [summary["variants"][n]["tokens_per_second_mean"] for n in order], color="#3498db")
    axes[0].set_ylabel("Tokens/s")
    axes[1].bar(x, [summary["variants"][n]["peak_memory_mb_mean"] for n in order], color="#e67e22")
    axes[1].set_ylabel("Pico CUDA (MiB)")
    for ax in axes:
        ax.set_xticks(x, order, rotation=35, ha="right"); ax.grid(axis="y", alpha=.25)
    fig.suptitle(f"{summary['stage']} — custo observado denso (não é speedup por rank)")
    _save(fig, target, "observed_dense_cost"); return "observed_dense_cost"


def _stage_a_quality(summary, target, order):
    x = np.arange(len(order)); selected = summary["selection"]["variant"]
    values = [summary["variants"][n]["validation_ce_mean"] for n in order]
    errors = [summary["variants"][n]["validation_ce_std"] for n in order]
    colors = ["#27ae60" if n == selected else "#7f8c8d" for n in order]
    fig, ax = plt.subplots(figsize=(11, 5.2)); ax.bar(x, values, yerr=errors, color=colors, capsize=4)
    ax.set_xticks(x, order, rotation=35, ha="right"); ax.set_ylabel("CE de validação")
    ax.set_title("3A.1-A — ablação fatorial e scaffold selecionado"); ax.grid(axis="y", alpha=.25)
    _save(fig, target, "scaffold_validation_ce"); return "scaffold_validation_ce"


def _factorial(summary, target):
    effects = summary["factorial_effects"]; names = list(effects)
    values = [effects[n]["validation_ce_effect_mean"] for n in names]
    fig, ax = plt.subplots(figsize=(9, 5.2)); y = np.arange(len(names))
    ax.barh(y, values, color=["#27ae60" if v < 0 else "#c0392b" for v in values], alpha=.75)
    for index, name in enumerate(names):
        points = effects[name]["by_seed"]; ax.scatter(points, [index] * len(points), color="black", s=18)
    ax.axvline(0, color="black"); ax.set_yticks(y, names); ax.invert_yaxis()
    ax.set_xlabel("Efeito em CE de validação (negativo melhora)"); ax.set_title("3A.1-A — efeitos e interações")
    _save(fig, target, "factorial_effects"); return "factorial_effects"


def _quality_rank(summary, target, order):
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for name in order:
        value = summary["variants"][name]
        ax.errorbar(value["mean_rank_mean"], value["test_ce_mean"], xerr=value["mean_rank_std"], yerr=value["test_ce_std"], marker="o", capsize=3, label=name)
        for run in value["runs"]:
            ax.scatter(run["mean_rank"], run["test_ce"], alpha=.35, s=18)
    fixed = sorted((summary["variants"][n]["mean_rank_mean"], summary["variants"][n]["test_ce_mean"]) for n in order if "adaptive" not in n)
    ax.plot(*zip(*fixed), linestyle="--", color="#7f8c8d", label="fronteira fixa")
    ax.set(xlabel="Rank hard médio", ylabel="CE de test", title="3A.1-B — qualidade versus rank")
    ax.grid(alpha=.25); ax.legend(fontsize=7)
    _save(fig, target, "quality_vs_mean_rank"); return "quality_vs_mean_rank"


def _rank_range(summary, target):
    runs = summary["variants"]["selected_adaptive_32"]["runs"]
    labels = [str(r["seed"]) for r in runs]; means = np.asarray([r["mean_rank"] for r in runs])
    low = means - np.asarray([r["rank_min"] for r in runs]); high = np.asarray([r["rank_max"] for r in runs]) - means
    fig, ax = plt.subplots(figsize=(7.5, 4.8)); ax.errorbar(labels, means, yerr=np.vstack([low, high]), fmt="o", capsize=6)
    ax.axhline(32, linestyle="--", color="#34495e", label="alvo 32")
    ax.set(xlabel="Seed", ylabel="Rank hard", title="3A.1-B — faixa do rank adaptativo"); ax.legend(); ax.grid(alpha=.25)
    _save(fig, target, "adaptive_rank_range"); return "adaptive_rank_range"


def _paired(summary, target):
    items = summary["paired_quality"]; labels = [str(x["seed"]) for x in items]
    values = [x["adaptive_minus_fixed32_test_ce"] for x in items]
    fig, ax = plt.subplots(figsize=(7.5, 4.8)); ax.bar(labels, values, color=["#27ae60" if v < 0 else "#c0392b" for v in values])
    ax.axhline(0, color="black"); ax.set(xlabel="Seed", ylabel="CE adaptativo − fixed-32", title="3A.1-B — deltas pareados")
    ax.grid(axis="y", alpha=.25); _save(fig, target, "paired_adaptive_deltas"); return "paired_adaptive_deltas"


def _dashboard(summary, target, charts):
    gates = "".join(f"<li><b>{html.escape(n)}</b>: {'PASS' if p else 'FAIL'}</li>" for n, p in summary["gates"].items())
    figures = "".join(f'<section><h2>{html.escape(n.replace("_", " "))}</h2><a href="{n}.svg"><img src="{n}.svg"></a></section>' for n in charts)
    document = f'''<!doctype html><html><head><meta charset="utf-8"><title>{summary['stage']}</title><style>body{{font:16px sans-serif;max-width:1200px;margin:auto;padding:2rem}}img{{width:100%}}section{{margin:2rem 0}}.note{{background:#fff3cd;padding:1rem}}</style></head><body><h1>ASM-VR {summary['stage']}</h1><div class="note">Custos observados em caminhos densos; não representam speedup causal por rank.</div><h2>Gates</h2><ul>{gates}</ul>{figures}</body></html>'''
    (target / "index.html").write_text(document, encoding="utf-8")


def render_stage_a(summary, directory):
    target = Path(directory); target.mkdir(parents=True, exist_ok=True); order = tuple(summary["variants"])
    charts = [_learning(summary, target, order), _stage_a_quality(summary, target, order), _factorial(summary, target), _cost(summary, target, order)]
    _dashboard(summary, target, charts)


def render_stage_b(summary, directory):
    target = Path(directory); target.mkdir(parents=True, exist_ok=True); order = tuple(summary["variants"])
    charts = [_learning(summary, target, order), _quality_rank(summary, target, order), _rank_range(summary, target), _paired(summary, target), _cost(summary, target, order)]
    _dashboard(summary, target, charts)


__all__ = ["render_stage_a", "render_stage_b"]
