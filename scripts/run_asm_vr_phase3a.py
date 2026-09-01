"""Run the paired small-scale ASM-VR Phase 3A language matrix."""
from __future__ import annotations
import argparse
from dataclasses import asdict, replace
import gc
import hashlib
import json
from pathlib import Path
import platform
import torch
from aletheion_state_models.benchmarks.phase3a_data import load_document_hash_splits
from aletheion_state_models.benchmarks.phase3a_plots import render_phase3a_charts
from aletheion_state_models.benchmarks.phase3a_summary import summarize_phase3a
from aletheion_state_models.benchmarks.phase3a_training import Phase3ARunResult, measure_streaming_error, train_phase3a_run
from aletheion_state_models.benchmarks.phase3a_variants import PHASE3A_VARIANTS, build_phase3a_variant


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
    temporary.replace(path)


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--corpus",type=Path,default=Path("data/wikipedia_en_20231101_sample.txt"))
    parser.add_argument("--steps",type=int,default=489)
    parser.add_argument("--batch-size",type=int,default=16)
    parser.add_argument("--sequence-length",type=int,default=256)
    parser.add_argument("--evaluation-batches",type=int,default=16)
    parser.add_argument("--seeds",type=int,nargs="+",default=[17,29,43])
    parser.add_argument("--variants",nargs="+",choices=PHASE3A_VARIANTS,default=list(PHASE3A_VARIANTS))
    parser.add_argument("--device",default="cuda")
    parser.add_argument("--reuse-completed",action="store_true")
    parser.add_argument("--run-root",type=Path,default=Path("runs/asm_vr_phase3a"))
    parser.add_argument("--output",type=Path,default=Path("docs/benchmarks/asm_vr_phase3a"))
    args=parser.parse_args()
    splits=load_document_hash_splits(args.corpus)
    protocol={
      "id":"asm-vr-phase3a-small-language-v1","corpus":splits.manifest,
      "variants":args.variants,"seeds":args.seeds,"steps":args.steps,
      "batch_size":args.batch_size,"sequence_length":args.sequence_length,
      "tokens_per_run":args.steps*args.batch_size*args.sequence_length,
      "evaluation_batches":args.evaluation_batches,"device":args.device,
      "precision":"bf16 autocast on CUDA; fp32 parameters/loss",
      "environment":{"python":platform.python_version(),"torch":torch.__version__,"cuda":torch.version.cuda,"gpu":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},
      "claims":{"small_scale_language_gate":True,"scaling_confirmation":False,"hardware_speedup":False},
      "development_disclosure":["Initial matrix with budget weight 0.00125 missed the adaptive budget gate.","A second adaptive matrix with budget weight 0.0025 still missed the hard-rank gate at threshold 0.5.","Threshold 0.7 was calibrated from validation score distributions without test labels and frozen for final confirmation.","All non-adaptive runs are reused unchanged.","Streaming parity is measured in FP32; BF16 exploratory errors were quantized at 0.03125/0.0625."],
    }
    protocol["protocol_sha256"]=hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest()
    _write_json(args.output/"manifest.json",protocol)
    results=[]
    for seed in args.seeds:
        for variant in args.variants:
            print(f"START {variant} seed={seed}",flush=True)
            result_path=args.run_root/variant/f"seed_{seed}"/"result.json"
            if args.reuse_completed and result_path.exists():
                result=Phase3ARunResult(**json.loads(result_path.read_text()))
                checkpoint=torch.load(result_path.with_name("best.pt"),map_location=args.device,weights_only=False)
                model,_=build_phase3a_variant(variant,seed);model.load_state_dict(checkpoint["model_state"]);model.to(args.device)
                result=replace(result,streaming_error=measure_streaming_error(model,splits.validation,torch.device(args.device)))
                print(f"REUSED {variant} seed={seed}",flush=True)
            else:
                result=train_phase3a_run(
                  variant,seed,splits,output_directory=args.run_root,steps=args.steps,
                  batch_size=args.batch_size,sequence_length=args.sequence_length,
                  evaluation_batches=args.evaluation_batches,device=args.device,
                  milestones=tuple(sorted({max(1,round(args.steps*f)) for f in (.2,.4,.6,.8,1.0)})),
                )
            results.append(result)
            print(f"DONE {variant} seed={seed} test_ce={result.test_ce:.4f} rank={result.mean_rank:.2f} tok_s={result.tokens_per_second:.0f}",flush=True)
            _write_json(args.output/"partial_runs.json",[asdict(row) for row in results])
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
    summary=summarize_phase3a(results)
    _write_json(args.output/"summary.json",summary)
    render_phase3a_charts(summary,args.output)
    (args.output/"partial_runs.json").unlink(missing_ok=True)
    print(json.dumps(summary["gates"],indent=2,sort_keys=True))
    if not summary["passed"]: raise SystemExit(1)


if __name__=="__main__": main()
