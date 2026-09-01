"""Run PMCS-64: parameter-matched ASM-CM versus ASM-VR-S capability suite."""
from dataclasses import replace
import argparse
import hashlib
import json
from pathlib import Path
import torch
from aletheion_state_models.benchmarks.phase3a_checkpoint import atomic_torch_save, write_result
from aletheion_state_models.benchmarks.phase3a_data import load_document_hash_splits
from aletheion_state_models.benchmarks.phase3a_training import evaluate_language_model, train_phase3a_run
from aletheion_state_models.benchmarks.purpose_mqar import specialize_and_evaluate
from aletheion_state_models.benchmarks.purpose_plots import render_purpose_charts
from aletheion_state_models.benchmarks.purpose_streaming import probe_streaming
from aletheion_state_models.benchmarks.purpose_summary import summarize_purpose_suite
from aletheion_state_models.benchmarks.purpose_variants import PURPOSE_VARIANTS, build_purpose_variant, parameter_inventory

SEEDS=(17,29,43)


def _arguments():
    parser=argparse.ArgumentParser();parser.add_argument("--corpus",type=Path,default=Path("data/wikipedia_en_20231101_sample.txt"));parser.add_argument("--run-root",type=Path,default=Path("runs/asm_cm_vs_vr_s_pmcs64"));parser.add_argument("--output",type=Path,default=Path("docs/benchmarks/asm_cm_vs_vr_s_pmcs64"));parser.add_argument("--language-steps",type=int,default=489);parser.add_argument("--mqar-steps",type=int,default=1000);parser.add_argument("--evaluation-batches",type=int,default=16);parser.add_argument("--mqar-test-lengths",default="40,512,4096,32768");parser.add_argument("--stream-lengths",default="512,4096");parser.add_argument("--stream-long-lengths",default="512,4096,32768");parser.add_argument("--device",default="cuda");return parser.parse_args()


def _load_model(run_root,variant,seed,device):
    directory=run_root/variant/f"seed_{seed}";checkpoint=torch.load(directory/"best.pt",map_location=device,weights_only=False);model,_=build_purpose_variant(variant,seed);model.load_state_dict(checkpoint["model_state"]);model.to(device).eval();return model,checkpoint,directory


def _source_hash():
    files=("src/aletheion_state_models/benchmarks/purpose_variants.py","src/aletheion_state_models/benchmarks/purpose_mqar.py","src/aletheion_state_models/benchmarks/purpose_streaming.py","src/aletheion_state_models/benchmarks/purpose_summary.py","scripts/run_asm_cm_vs_vr_s_pmcs64.py");digest=hashlib.sha256()
    for name in files:digest.update(name.encode());digest.update(Path(name).read_bytes())
    return {"sha256":digest.hexdigest(),"files":list(files)}


def main():
    args=_arguments();splits=load_document_hash_splits(args.corpus);device=torch.device(args.device);language=[]
    for seed in SEEDS:
        for variant in PURPOSE_VARIANTS:
            print(f"language {variant} seed={seed}",flush=True);result=train_phase3a_run(variant,seed,splits,output_directory=args.run_root,steps=args.language_steps,batch_size=16,sequence_length=256,evaluation_batches=args.evaluation_batches,milestones=tuple(sorted({100,200,300,400,args.language_steps})),device=args.device,variant_builder=build_purpose_variant,adaptive_variants=frozenset(),evaluate_test=False);language.append(result)
    manifest={"experiment":"PMCS-64","corpus":splits.manifest,"seeds":list(SEEDS),"language_steps":args.language_steps,"language_tokens_per_run":args.language_steps*16*256,"mqar_steps":args.mqar_steps,"parameters":parameter_inventory(),"test_protocol":{"language_test_opened_after_all_checkpoints":False},"provenance":{"repository_dirty":True,"implementation":_source_hash()}};args.output.mkdir(parents=True,exist_ok=True);write_result(args.output/"manifest.json",manifest)
    finalized=[]
    for result in language:
        model,checkpoint,directory=_load_model(args.run_root,result.variant,result.seed,device);test=evaluate_language_model(model,splits.test,seed=20_000+result.seed,batches=args.evaluation_batches,batch_size=16,sequence_length=256,device=device);final=replace(result,test_ce=test["ce"],test_ppl=test["ppl"],mean_rank=test["mean_rank"],rank_std=test["rank_std"],rank_min=test["rank_min"],rank_max=test["rank_max"],rank_ce_correlation=test["rank_ce_correlation"]);checkpoint["test_protocol"]={"opened_after_all_checkpoints":True};write_result(directory/"result.json",final);atomic_torch_save(directory/"best.pt",checkpoint);finalized.append(final)
    manifest["test_protocol"]["language_test_opened_after_all_checkpoints"]=True;write_result(args.output/"manifest.json",manifest)
    mqar=[];test_lengths=tuple(int(value) for value in args.mqar_test_lengths.split(","))
    for seed in SEEDS:
        for variant in PURPOSE_VARIANTS:
            print(f"specialization {variant} seed={seed}",flush=True);model,_,_=_load_model(args.run_root,variant,seed,device);mqar.append(specialize_and_evaluate(model,splits,variant=variant,seed=seed,output_directory=args.run_root,steps=args.mqar_steps,test_lengths=test_lengths,device=args.device))
    streaming=[];short_lengths=tuple(int(value) for value in args.stream_lengths.split(","));long_lengths=tuple(int(value) for value in args.stream_long_lengths.split(","))
    for seed in SEEDS:
        for variant in PURPOSE_VARIANTS:
            lengths=long_lengths if seed==SEEDS[0] else short_lengths;print(f"streaming {variant} seed={seed} lengths={lengths}",flush=True);model,_,_= _load_model(args.run_root,variant,seed,device);rows=probe_streaming(model,lengths,seed=50_000+seed,device=args.device);streaming.append({"variant":variant,"seed":seed,"streaming":[dict(row,seed=seed) for row in rows]});write_result(args.run_root/variant/f"seed_{seed}"/"streaming_result.json",streaming[-1])
    summary=summarize_purpose_suite(finalized,mqar,streaming,parameter_inventory());write_result(args.output/"summary.json",summary);render_purpose_charts(summary,args.output);print(json.dumps({"gates":summary["gates"],"language_winner":summary["language_winner"],"mqar_winner":summary["mqar_32k_winner"]},indent=2));raise SystemExit(0 if summary["technical_passed"] else 1)


if __name__=="__main__":main()
