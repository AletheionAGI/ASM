"""Calibrate the Phase 3A hard-rank budget and regenerate final artifacts."""
from __future__ import annotations
from dataclasses import asdict, replace
import argparse
import json
from pathlib import Path
import torch
from aletheion_state_models.benchmarks.phase3a_checkpoint import write_result
from aletheion_state_models.benchmarks.phase3a_data import load_document_hash_splits
from aletheion_state_models.benchmarks.phase3a_plots import render_phase3a_charts
from aletheion_state_models.benchmarks.phase3a_summary import summarize_phase3a
from aletheion_state_models.benchmarks.phase3a_training import Phase3ARunResult,evaluate_language_model,measure_streaming_error
from aletheion_state_models.benchmarks.phase3a_variants import PHASE3A_VARIANTS,build_phase3a_variant


def main() -> None:
 parser=argparse.ArgumentParser();parser.add_argument("--corpus",type=Path,default=Path("data/wikipedia_en_20231101_sample.txt"));parser.add_argument("--run-root",type=Path,default=Path("runs/asm_vr_phase3a"));parser.add_argument("--output",type=Path,default=Path("docs/benchmarks/asm_vr_phase3a"));parser.add_argument("--threshold",type=float,default=.8);parser.add_argument("--evaluation-batches",type=int,default=16);parser.add_argument("--device",default="cuda");args=parser.parse_args()
 splits=load_document_hash_splits(args.corpus);device=torch.device(args.device);results=[]
 for seed in (17,29,43):
  for variant in PHASE3A_VARIANTS:
   run_dir=args.run_root/variant/f"seed_{seed}";result=Phase3ARunResult(**json.loads((run_dir/"result.json").read_text()));checkpoint=torch.load(run_dir/"best.pt",map_location=device,weights_only=False);model,_=build_phase3a_variant(variant,seed);model.load_state_dict(checkpoint["model_state"]);model.to(device).eval()
   if variant=="vr_adaptive_32":
    model.variable_rank_core.controller.threshold=args.threshold
    validation=evaluate_language_model(model,splits.validation,seed=10000+seed,batches=args.evaluation_batches,batch_size=16,sequence_length=256,device=device)
    test=evaluate_language_model(model,splits.test,seed=20000+seed,batches=args.evaluation_batches,batch_size=16,sequence_length=256,device=device)
    result=replace(result,validation_ce=validation["ce"],validation_ppl=validation["ppl"],test_ce=test["ce"],test_ppl=test["ppl"],mean_rank=test["mean_rank"],rank_std=test["rank_std"],rank_min=test["rank_min"],rank_max=test["rank_max"],rank_ce_correlation=test["rank_ce_correlation"])
   result=replace(result,streaming_error=measure_streaming_error(model,splits.validation,device));write_result(run_dir/"result.json",result);results.append(result)
 summary=summarize_phase3a(results);write_result(args.output/"summary.json",summary);manifest=json.loads((args.output/"manifest.json").read_text());manifest["hard_budget_calibration"]={"threshold":args.threshold,"source":"combined validation score distributions; no test labels","target_rank":32};write_result(args.output/"manifest.json",manifest);render_phase3a_charts(summary,args.output);print(json.dumps(summary["gates"],indent=2,sort_keys=True));raise SystemExit(0 if summary["passed"] else 1)


if __name__=="__main__":main()
