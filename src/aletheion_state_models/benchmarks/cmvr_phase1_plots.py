"""Offline artifacts for the ASM-CM-VR fixed-32 Phase 1 smoke."""
from pathlib import Path
import html
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LABELS={"cm_vr_full64":"ASM-CM-VR full-64","cm_vr_fixed32":"ASM-CM-VR fixed-32"}


def _save(fig,target,name):
    fig.tight_layout()
    for ext in ("png","svg"):fig.savefig(target/f"{name}.{ext}",dpi=180,bbox_inches="tight")
    plt.close(fig);return name


def render_cmvr_phase1(summary,directory):
    target=Path(directory);target.mkdir(parents=True,exist_ok=True);charts=[]
    fig,ax=plt.subplots(figsize=(8.5,5))
    for result in summary["arms"]:ax.plot([row["step"] for row in result["history"]],[row["validation_accuracy"] for row in result["history"]],marker="o",label=LABELS[result["arm"]])
    ax.axhline(.95,color="black",linestyle="--",label="gate 95%");ax.set(xlabel="Update",ylabel="Acurácia MQAR-40",title="Aprendizagem associativa curta");ax.grid(alpha=.25);ax.legend();charts.append(_save(fig,target,"mqar_learning"))
    fig,ax=plt.subplots(figsize=(8.5,5));names=[];normal=[];no_read=[];no_write=[]
    for result in summary["arms"]:names.append(LABELS[result["arm"]]);normal.append(result["test"]["accuracy"]);no_read.append(result["no_read"]["accuracy"]);no_write.append(result["no_write"]["accuracy"])
    x=range(len(names));ax.bar([i-.25 for i in x],normal,.25,label="normal");ax.bar(x,no_read,.25,label="sem leitura");ax.bar([i+.25 for i in x],no_write,.25,label="sem escrita");ax.set_xticks(list(x),names);ax.set_ylabel("Acurácia MQAR-40");ax.set_title("Canários causais da memória");ax.grid(axis="y",alpha=.25);ax.legend();charts.append(_save(fig,target,"memory_causal_ablations"))
    fig,ax=plt.subplots(figsize=(8.5,5))
    for result in summary["arms"]:
        rows=[row for row in result["streaming"] if row.get("status")!="failed"];ax.plot([row["length"] for row in rows],[row["tokens_per_second"] for row in rows],marker="o",label=LABELS[result["arm"]])
    ax.set_xscale("log",base=2);ax.set(xlabel="Comprimento",ylabel="Tokens/s",title="Streaming com estado retido");ax.grid(alpha=.25);ax.legend();charts.append(_save(fig,target,"streaming_throughput"))
    gates="".join(f"<li><b>{html.escape(name)}</b>: {'PASS' if value else 'FAIL'}</li>" for name,value in summary["gates"].items());figures="".join(f'<section><h2>{name.replace("_"," ")}</h2><a href="{name}.svg"><img src="{name}.svg"></a></section>' for name in charts);page="<!doctype html><html><head><meta charset='utf-8'><title>ASM-CM-VR Phase 1</title><style>body{font:16px sans-serif;max-width:1100px;margin:auto;padding:2rem}img{width:100%}</style></head><body><h1>ASM-CM-VR fixed-32 Phase 1</h1><ul>"+gates+"</ul>"+figures+"</body></html>";(target/"index.html").write_text(page,encoding="utf-8")


__all__=["render_cmvr_phase1"]
