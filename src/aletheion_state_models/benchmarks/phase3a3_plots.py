"""Charts for the R/S/RS full-rank comparison."""
from pathlib import Path
import html
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ORDER = ("vr_r_full", "vr_s_full", "vr_rs_full")


def _save(fig, target, name):
    fig.tight_layout()
    for ext in ("png", "svg"): fig.savefig(target / f"{name}.{ext}", dpi=180, bbox_inches="tight")
    plt.close(fig); return name


def _learning(summary, target):
    fig, ax = plt.subplots(figsize=(9, 5.2))
    for name in ORDER:
        runs = summary["variants"][name]["runs"]; tokens = np.asarray([p["tokens"] for p in runs[0]["history"]]); curves = np.asarray([[p["validation_ce"] for p in run["history"]] for run in runs])
        ax.plot(tokens, curves.mean(0), label=name); ax.fill_between(tokens, curves.min(0), curves.max(0), alpha=.12)
    ax.set(xlabel="Tokens", ylabel="CE de validação", title="R/S/RS full — aprendizado pareado"); ax.grid(alpha=.25); ax.legend()
    return _save(fig, target, "validation_ce_full_bases")


def _quality(summary, target):
    fig, ax = plt.subplots(figsize=(8.5, 5)); x = np.arange(3); values = [summary["variants"][n]["test_ce_mean"] for n in ORDER]; errors = [summary["variants"][n]["test_ce_std"] for n in ORDER]
    ax.bar(x, values, yerr=errors, capsize=4); ax.set_xticks(x, ORDER); ax.set_ylabel("CE de test"); ax.set_title("Qualidade full-rank por base"); ax.grid(axis="y", alpha=.25)
    return _save(fig, target, "full_base_test_ce")


def _cost(summary, target):
    x = np.arange(3); fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].bar(x, [summary["variants"][n]["tokens_per_second_mean"] for n in ORDER]); axes[0].set_ylabel("Tokens/s")
    axes[1].bar(x, [summary["variants"][n]["peak_memory_mb_mean"] for n in ORDER]); axes[1].set_ylabel("Pico CUDA MiB")
    for ax in axes: ax.set_xticks(x, ORDER, rotation=20); ax.grid(axis="y", alpha=.25)
    fig.suptitle("Custo observado full-rank"); return _save(fig, target, "full_base_observed_cost")


def _parameters(summary, target):
    fig, ax = plt.subplots(figsize=(8, 5))
    for name in ORDER:
        value = summary["variants"][name]; ax.scatter(value["parameter_count_mean"], value["test_ce_mean"], s=90, label=name)
    ax.set(xlabel="Parâmetros", ylabel="CE de test", title="Qualidade versus capacidade física"); ax.grid(alpha=.25); ax.legend()
    return _save(fig, target, "quality_vs_parameters")


def render_phase3a3_charts(summary, directory):
    target = Path(directory); target.mkdir(parents=True, exist_ok=True); charts = [_learning(summary, target), _quality(summary, target), _cost(summary, target), _parameters(summary, target)]
    gates = "".join(f"<li><b>{html.escape(n)}</b>: {'PASS' if v else 'FAIL'}</li>" for n, v in summary["gates"].items()); figures = "".join(f'<section><h2>{n.replace("_", " ")}</h2><a href="{n}.svg"><img src="{n}.svg"></a></section>' for n in charts)
    page = "<!doctype html><html><head><meta charset='utf-8'><title>ASM-VR-RS</title><style>body{font:16px sans-serif;max-width:1200px;margin:auto;padding:2rem}img{width:100%}</style></head><body><h1>ASM-VR-RS full</h1><ul>" + gates + "</ul>" + figures + "</body></html>"; (target / "index.html").write_text(page, encoding="utf-8")


__all__ = ["render_phase3a3_charts"]
