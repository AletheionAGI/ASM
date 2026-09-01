"""Charts for new-seed AdamM confirmation of the Phase 3A.2 base effect."""
from pathlib import Path
import html
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _save(fig, target, name):
    fig.tight_layout()
    for ext in ("png", "svg"): fig.savefig(target / f"{name}.{ext}", dpi=180, bbox_inches="tight")
    plt.close(fig); return name


def render_phase3a2_adamm_confirmation(summary, directory):
    target = Path(directory); target.mkdir(parents=True, exist_ok=True); charts = []
    fig, ax = plt.subplots(figsize=(9,5))
    for base in ("vr_r", "vr_s"):
        runs=summary["variants"][base]["runs"]; tokens=np.asarray([p["tokens"] for p in runs[0]["history"]]); curves=np.asarray([[p["validation_ce"] for p in run["history"]] for run in runs]); ax.plot(tokens,curves.mean(0),label=f"AdamM {base}");ax.fill_between(tokens,curves.min(0),curves.max(0),alpha=.12)
    ax.set(xlabel="Tokens",ylabel="CE de validação",title="Confirmação AdamM em seeds novas");ax.grid(alpha=.25);ax.legend();charts.append(_save(fig,target,"validation_ce_adamm_new_seeds"))
    fig,ax=plt.subplots(figsize=(8,5));items=summary["paired"];labels=[str(x["seed"]) for x in items];values=[x["s_minus_r_test_ce"] for x in items];ax.bar(labels,values);ax.axhline(0,color="black");ax.set(xlabel="Seed",ylabel="CE S − R",title="Efeito pareado da base sob AdamM");ax.grid(axis="y",alpha=.25);charts.append(_save(fig,target,"paired_s_minus_r_adamm"))
    fig,ax=plt.subplots(figsize=(8,5));names=("AdamW R","AdamW S","AdamM R","AdamM S");values=(summary["adamw_context"]["vr_r_full"]["test_ce_mean"],summary["adamw_context"]["vr_s_full"]["test_ce_mean"],summary["variants"]["vr_r"]["test_ce_mean"],summary["variants"]["vr_s"]["test_ce_mean"]);ax.bar(names,values);ax.set_ylabel("CE de test");ax.set_title("Contexto AdamW e confirmação AdamM");ax.tick_params(axis="x",rotation=20);ax.grid(axis="y",alpha=.25);charts.append(_save(fig,target,"quality_optimizer_context"))
    fig,axes=plt.subplots(1,2,figsize=(11,4.7));names=("AdamM R","AdamM S");axes[0].bar(names,[summary["variants"][b]["tokens_per_second_mean"] for b in ("vr_r","vr_s")]);axes[0].set_ylabel("Tokens/s");axes[1].bar(names,[summary["variants"][b]["peak_memory_mb_mean"] for b in ("vr_r","vr_s")]);axes[1].set_ylabel("Pico CUDA MiB");fig.suptitle("Custo observado da confirmação");charts.append(_save(fig,target,"adamm_confirmation_cost"))
    gates="".join(f"<li><b>{html.escape(n)}</b>: {'PASS' if v else 'FAIL'}</li>" for n,v in summary["gates"].items());figures="".join(f'<section><h2>{n.replace("_"," ")}</h2><a href="{n}.svg"><img src="{n}.svg"></a></section>' for n in charts);page="<!doctype html><html><head><meta charset='utf-8'><title>3A.2 AdamM</title><style>body{font:16px sans-serif;max-width:1200px;margin:auto;padding:2rem}img{width:100%}</style></head><body><h1>ASM-VR-S AdamM confirmation</h1><ul>"+gates+"</ul>"+figures+"</body></html>";(target/"index.html").write_text(page,encoding="utf-8")


__all__=["render_phase3a2_adamm_confirmation"]
