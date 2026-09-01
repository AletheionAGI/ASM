"""Confirm the ASM-VR-S base effect under AdamM on new seeds."""
from dataclasses import asdict
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from statistics import mean, pstdev
from aletheion_state_models.benchmarks.phase3a_checkpoint import write_result
from aletheion_state_models.benchmarks.phase3a_data import load_document_hash_splits
from aletheion_state_models.benchmarks.phase3a_training import train_phase3a_run
from aletheion_state_models.benchmarks.phase3a2_adamm_plots import render_phase3a2_adamm_confirmation
from aletheion_state_models.benchmarks.phase3a2_variants import build_phase3a2_variant

SEEDS=(71,89,107);VARIANTS=("adamm_vr_r_full","adamm_vr_s_full");ADAMM_SHA="79495581868147a5bed69acc3e3a85e838634c3ced0aeb9ab98b35223722c877";ADAMM_COMMIT="980d84ce96825c3d11d6bc8dd98f0c5168897643"


def _builder(variant,seed): return build_phase3a2_variant(variant.removeprefix("adamm_"),seed)


def _load(path):
    if hashlib.sha256(path.read_bytes()).hexdigest()!=ADAMM_SHA: raise RuntimeError("AdamM source hash mismatch")
    spec=importlib.util.spec_from_file_location("asm_vr_adamm_confirm",path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module.AdamM


def _aggregate(runs):
    output={"runs":[asdict(run) for run in runs]}
    for field in ("validation_ce","test_ce","tokens_per_second","peak_memory_mb","streaming_error"):
        values=[float(getattr(run,field)) for run in runs];output[f"{field}_mean"]=mean(values);output[f"{field}_std"]=pstdev(values)
    return output


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--corpus",type=Path,default=Path("data/wikipedia_en_20231101_sample.txt"));parser.add_argument("--phase3a2",type=Path,default=Path("docs/benchmarks/asm_vr_phase3a2/summary.json"));parser.add_argument("--adamm-source",type=Path,default=Path("../AdamM/adamm.py"));parser.add_argument("--run-root",type=Path,default=Path("runs/asm_vr_phase3a2_adamm_confirm"));parser.add_argument("--output",type=Path,default=Path("docs/benchmarks/asm_vr_phase3a2_adamm_confirm"));parser.add_argument("--device",default="cuda");args=parser.parse_args()
    phase=json.loads(args.phase3a2.read_text());assert phase["base_selection"]["promoted"]=="vr_s";assert phase["validation_policy_selection"]["common_policy"]=="full"
    AdamM=_load(args.adamm_source);factory=lambda parameters,lr:AdamM(parameters,lr=lr,weight_decay=.01);splits=load_document_hash_splits(args.corpus);results=[]
    for seed in SEEDS:
        for variant in VARIANTS:
            print(f"training {variant} seed={seed}",flush=True);result=train_phase3a_run(variant,seed,splits,output_directory=args.run_root,steps=489,batch_size=16,sequence_length=256,evaluation_batches=16,device=args.device,variant_builder=_builder,adaptive_variants=frozenset(),optimizer_factory=factory);results.append(result);print(f"done {variant} seed={seed} test={result.test_ce:.4f}",flush=True)
    grouped={base:sorted((run for run in results if f"_{base}_" in run.variant),key=lambda run:run.seed) for base in ("vr_r","vr_s")};variants={base:_aggregate(runs) for base,runs in grouped.items()};r={run.seed:run for run in grouped["vr_r"]};paired=[{"seed":run.seed,"s_minus_r_test_ce":run.test_ce-r[run.seed].test_ce} for run in grouped["vr_s"]];delta=mean(item["s_minus_r_test_ce"] for item in paired);gates={"complete":all(len(runs)==3 for runs in grouped.values()),"finite":all(run.finite for run in results),"streaming":max(run.streaming_error for run in results)<=1e-4,"s_quality_superior":delta<=-.02 and all(item["s_minus_r_test_ce"]<0 for item in paired)};summary={"experiment":"3A.2-AdamM-confirmation","seeds":list(SEEDS),"policy":"full","variants":variants,"paired":paired,"s_minus_r_test_ce":delta,"adamw_context":{name:phase["variants"][name] for name in ("vr_r_full","vr_s_full")},"optimizer":{"name":"AdamM","lr":3e-4,"commit":ADAMM_COMMIT,"sha256":ADAMM_SHA},"gates":gates,"passed":all(gates.values())};args.output.mkdir(parents=True,exist_ok=True);write_result(args.output/"summary.json",summary);render_phase3a2_adamm_confirmation(summary,args.output);write_result(args.output/"manifest.json",{"corpus":splits.manifest,"seeds":list(SEEDS),"tokens_per_run":489*16*256,"policy":"full","optimizer":summary["optimizer"],"test_role":"confirmation after AdamW selection","repository_dirty":True});print(json.dumps({"delta":delta,"gates":gates},indent=2));raise SystemExit(0 if all(gates[name] for name in ("complete","finite","streaming")) else 1)


if __name__=="__main__":main()
