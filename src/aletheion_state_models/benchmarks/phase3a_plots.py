"""Matplotlib comparison charts for ASM-VR Phase 3A."""
from __future__ import annotations
from collections import defaultdict
import html
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ORDER=("asm_r","vr_full","vr_fixed_16","vr_fixed_32","vr_fixed_48","vr_adaptive_32")
COLORS={"asm_r":"#222222","vr_full":"#0072B2","vr_fixed_16":"#E69F00","vr_fixed_32":"#009E73","vr_fixed_48":"#CC79A7","vr_adaptive_32":"#D55E00"}


def _save(fig: object, directory: Path, name: str) -> None:
    fig.tight_layout()
    fig.savefig(directory/f"{name}.png",dpi=180,bbox_inches="tight",facecolor="white")
    fig.savefig(directory/f"{name}.svg",bbox_inches="tight",facecolor="white")
    plt.close(fig)


def _learning(summary: dict[str,object], directory: Path) -> None:
    fig,ax=plt.subplots(figsize=(11,6.5)); grouped=defaultdict(list)
    for row in summary["runs"]:
        grouped[row["variant"]].append(row); history=row["history"]
        ax.plot([p["tokens"] for p in history],[p["validation_ce"] for p in history],color=COLORS[row["variant"]],alpha=.2,lw=1,marker="o",ms=3)
    for variant in ORDER:
        rows=grouped.get(variant,[])
        if not rows: continue
        x=[p["tokens"] for p in rows[0]["history"]]
        y=np.asarray([[p["validation_ce"] for p in row["history"]] for row in rows])
        ax.plot(x,y.mean(0),color=COLORS[variant],lw=2.8,label=variant,marker="o",ms=4)
    ax.set(title="Validation CE by training tokens",xlabel="Training tokens",ylabel="Cross-entropy (lower is better)");ax.grid(alpha=.25);ax.legend(ncol=2)
    _save(fig,directory,"validation_ce_by_tokens")


def _final_ce(summary: dict[str,object], directory: Path) -> None:
    fig,ax=plt.subplots(figsize=(11,6.5))
    for index,variant in enumerate(ORDER):
        values=[r["test_ce"] for r in summary["runs"] if r["variant"]==variant]
        ax.scatter([index]*len(values),values,color=COLORS[variant],s=55,alpha=.75)
        if values: ax.scatter(index,np.mean(values),color="white",edgecolor=COLORS[variant],lw=2.5,s=150,zorder=3)
    ax.set_xticks(range(len(ORDER)),ORDER,rotation=20,ha="right");ax.set(title="Final test CE by variant and seed",ylabel="Cross-entropy (lower is better)");ax.grid(axis="y",alpha=.25)
    _save(fig,directory,"final_test_ce_by_variant")


def _quality_rank(summary: dict[str,object], directory: Path) -> None:
    fig,ax=plt.subplots(figsize=(10,6.5))
    for variant in ORDER:
        rows=[r for r in summary["runs"] if r["variant"]==variant]
        ax.scatter([r["mean_rank"] for r in rows],[r["test_ce"] for r in rows],color=COLORS[variant],s=65,label=variant)
    ax.set(title="Language quality versus logical mean rank",xlabel="Mean hard rank (not FLOPs)",ylabel="Test CE (lower is better)");ax.grid(alpha=.25);ax.legend(ncol=2)
    _save(fig,directory,"quality_vs_mean_rank")


def _ranks(summary: dict[str,object], directory: Path) -> None:
    fig,ax=plt.subplots(figsize=(11,6.5))
    for index,variant in enumerate(ORDER):
        rows=[r for r in summary["runs"] if r["variant"]==variant]
        for offset,row in enumerate(rows):
            x=index+(offset-1)*.08;ax.vlines(x,row["rank_min"],row["rank_max"],color=COLORS[variant],alpha=.55);ax.scatter(x,row["mean_rank"],color=COLORS[variant],s=45)
    ax.axhline(32,color="#666",ls="--",label="adaptive target");ax.set_xticks(range(len(ORDER)),ORDER,rotation=20,ha="right");ax.set(title="Hard-rank range and mean by seed",ylabel="Logical rank");ax.grid(axis="y",alpha=.25);ax.legend()
    _save(fig,directory,"rank_distribution")


def _cost(summary: dict[str,object], directory: Path) -> None:
    fig,axes=plt.subplots(1,2,figsize=(13,6)); values=summary["variants"]; colors=[COLORS[n] for n in ORDER]
    axes[0].bar(ORDER,[values[n]["tokens_per_second_mean"] for n in ORDER],color=colors)
    axes[1].bar(ORDER,[values[n]["peak_memory_mb_mean"] for n in ORDER],color=colors)
    axes[0].set(title="Observed dense-path throughput",ylabel="Tokens/second");axes[1].set(title="Observed peak CUDA memory",ylabel="MiB")
    for ax in axes: ax.tick_params(axis="x",rotation=35);ax.grid(axis="y",alpha=.2)
    _save(fig,directory,"observed_cost")


def _deltas(summary: dict[str,object], directory: Path) -> None:
    rows=summary["paired_quality"];fig,ax=plt.subplots(figsize=(9,5.5));values=[r["adaptive_minus_fixed32_test_ce"] for r in rows]
    ax.bar([f"seed {r['seed']}" for r in rows],values,color="#D55E00");ax.axhline(.05,color="#222",ls="--",label="mean gate +0.05");ax.axhline(0,color="#666",lw=1)
    ax.set(title="Paired adaptive minus fixed-32 test CE",ylabel="CE delta (lower is better)");ax.legend();ax.grid(axis="y",alpha=.2)
    _save(fig,directory,"paired_seed_deltas")


def _dashboard(summary: dict[str,object], directory: Path) -> None:
    names=("validation_ce_by_tokens","final_test_ce_by_variant","quality_vs_mean_rank","rank_distribution","observed_cost","paired_seed_deltas")
    gates="".join(f"<tr><td>{html.escape(k)}</td><td>{'PASS' if v else 'FAIL'}</td></tr>" for k,v in summary["gates"].items())
    images="".join(f'<figure><img src="{n}.svg" alt="{n}"><figcaption>{n}</figcaption></figure>' for n in names)
    document=('<!doctype html><meta charset="utf-8"><title>ASM-VR Phase 3A</title>'
      "<style>body{font:16px system-ui;max-width:1250px;margin:auto;padding:24px}img{width:100%}figure{margin:32px 0}table{border-collapse:collapse}td{border:1px solid #bbb;padding:8px}</style>"
      f"<h1>ASM-VR Phase 3A dashboard</h1><p>Small-scale language gate. Dense execution; no hardware speedup claim.</p><h2>Acceptance gates</h2><table>{gates}</table>{images}"
      '<p><a href="summary.json">summary.json</a> · <a href="manifest.json">manifest.json</a></p>')
    (directory/"index.html").write_text(document,encoding="utf-8")


def render_phase3a_charts(summary: dict[str,object], output_directory: str|Path) -> None:
    """Render PNG/SVG comparisons and an offline HTML dashboard."""
    directory=Path(output_directory);directory.mkdir(parents=True,exist_ok=True)
    for renderer in (_learning,_final_ce,_quality_rank,_ranks,_cost,_deltas,_dashboard): renderer(summary,directory)


__all__=["render_phase3a_charts"]
