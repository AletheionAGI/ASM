"""Charts and offline dashboard for the PMCS-64 capability suite."""
from pathlib import Path
import html
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from .purpose_variants import PURPOSE_VARIANTS

LABELS={"asm_cm":"ASM-CM","asm_vr_s_full":"ASM-VR-S full","asm_vr_s_fixed_32":"ASM-VR-S fixed-32"}


def _save(fig,target,name):
    fig.tight_layout()
    for ext in ("png","svg"):fig.savefig(target/f"{name}.{ext}",dpi=180,bbox_inches="tight")
    plt.close(fig);return name


def render_purpose_charts(summary,directory):
    target=Path(directory);target.mkdir(parents=True,exist_ok=True);charts=[]
    fig,ax=plt.subplots(figsize=(9,5.2))
    for name in PURPOSE_VARIANTS:
        runs=summary["language"][name]["runs"];tokens=np.asarray([p["tokens"] for p in runs[0]["history"]]);curves=np.asarray([[p["validation_ce"] for p in run["history"]] for run in runs]);ax.plot(tokens,curves.mean(0),label=LABELS[name]);ax.fill_between(tokens,curves.min(0),curves.max(0),alpha=.12)
    ax.set(xlabel="Tokens",ylabel="CE de validação",title="Linguagem pareada");ax.grid(alpha=.25);ax.legend();charts.append(_save(fig,target,"language_validation_ce"))
    fig,ax=plt.subplots(figsize=(8,5));x=np.arange(3);values=[summary["language"][n]["test_ce_mean"] for n in PURPOSE_VARIANTS];errors=[summary["language"][n]["test_ce_std"] for n in PURPOSE_VARIANTS];ax.bar(x,values,yerr=errors,capsize=4);ax.set_xticks(x,[LABELS[n] for n in PURPOSE_VARIANTS],rotation=20);ax.set_ylabel("CE de test");ax.set_title("Qualidade de linguagem");ax.grid(axis="y",alpha=.25);charts.append(_save(fig,target,"language_test_ce"))
    for metric,ylabel,name in (("accuracy_mean","Acurácia","mqar_accuracy_by_length"),("ce_mean","CE nos targets","mqar_ce_by_length")):
        fig,ax=plt.subplots(figsize=(9,5.2))
        for variant in PURPOSE_VARIANTS:
            items=summary["mqar"][variant]["by_length"];lengths=sorted(int(k) for k in items);ax.plot(lengths,[items[str(k)][metric] for k in lengths],marker="o",label=LABELS[variant])
        ax.set_xscale("log",base=2);ax.set(xlabel="Comprimento MQAR",ylabel=ylabel,title=ylabel+" MQAR após especialização 80/20");ax.grid(alpha=.25);ax.legend();charts.append(_save(fig,target,name))
    fig,ax=plt.subplots(figsize=(8,5));before=[summary["language"][n]["test_ce_mean"] for n in PURPOSE_VARIANTS];after=[summary["mqar"][n]["language_test_ce_after_mean"] for n in PURPOSE_VARIANTS];x=np.arange(3);ax.bar(x-.2,before,.4,label="antes");ax.bar(x+.2,after,.4,label="após 80/20");ax.set_xticks(x,[LABELS[n] for n in PURPOSE_VARIANTS],rotation=20);ax.set_ylabel("CE de test linguagem");ax.set_title("Retenção de linguagem");ax.legend();ax.grid(axis="y",alpha=.25);charts.append(_save(fig,target,"language_retention_after_mqar"))
    for metric,ylabel,name in (("retained_state_bytes_mean","Bytes retidos","streaming_retained_state"),("tokens_per_second_mean","Tokens/s","streaming_throughput")):
        fig,ax=plt.subplots(figsize=(9,5.2))
        for variant in PURPOSE_VARIANTS:
            items=summary["streaming"][variant]["by_length"];lengths=sorted(int(k) for k in items);ax.plot(lengths,[items[str(k)][metric] for k in lengths],marker="o",label=LABELS[variant])
        ax.set_xscale("log",base=2);ax.set(xlabel="Comprimento do stream",ylabel=ylabel,title=ylabel+" por comprimento");ax.grid(alpha=.25);ax.legend();charts.append(_save(fig,target,name))
    fig,axes=plt.subplots(1,2,figsize=(12,4.8));x=np.arange(3);axes[0].bar(x,[summary["language"][n]["tokens_per_second_mean"] for n in PURPOSE_VARIANTS]);axes[0].set_ylabel("Tokens/s de treino");axes[1].bar(x,[summary["language"][n]["peak_memory_mb_mean"] for n in PURPOSE_VARIANTS]);axes[1].set_ylabel("Pico CUDA MiB")
    for ax in axes:ax.set_xticks(x,[LABELS[n] for n in PURPOSE_VARIANTS],rotation=20);ax.grid(axis="y",alpha=.25)
    charts.append(_save(fig,target,"observed_training_cost"))
    fig,ax=plt.subplots(figsize=(8,5.2)); recall_length=max(int(key) for key in summary["mqar"][PURPOSE_VARIANTS[0]]["by_length"])
    for variant in PURPOSE_VARIANTS:
        xval=summary["language"][variant]["test_ce_mean"];yval=summary["mqar"][variant]["by_length"][str(recall_length)]["accuracy_mean"];ax.scatter(xval,yval,s=120,label=LABELS[variant])
    ax.set(xlabel="CE de linguagem (menor melhor)",ylabel=f"Acurácia MQAR {recall_length}",title="Mapa de finalidade prática");ax.grid(alpha=.25);ax.legend();charts.append(_save(fig,target,"language_vs_durable_recall"))
    gates="".join(f"<li><b>{html.escape(n)}</b>: {'PASS' if v else 'FAIL'}</li>" for n,v in summary["gates"].items());figures="".join(f'<section><h2>{n.replace("_"," ")}</h2><a href="{n}.svg"><img src="{n}.svg"></a></section>' for n in charts);page="<!doctype html><html><head><meta charset='utf-8'><title>PMCS-64</title><style>body{font:16px sans-serif;max-width:1200px;margin:auto;padding:2rem}img{width:100%}</style></head><body><h1>ASM-CM vs ASM-VR-S PMCS-64</h1><p>Variable Rank permanece execução densa neste protocolo.</p><ul>"+gates+"</ul>"+figures+"</body></html>";(target/"index.html").write_text(page,encoding="utf-8")


__all__=["render_purpose_charts"]
