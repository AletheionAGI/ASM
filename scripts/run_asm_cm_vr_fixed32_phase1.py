"""Run the first fixed-rank ASM-CM-VR structural and MQAR gate."""
import argparse
import hashlib
import json
from pathlib import Path
from aletheion_state_models.benchmarks.cmvr_phase1 import run_phase1
from aletheion_state_models.benchmarks.cmvr_phase1_plots import render_cmvr_phase1
from aletheion_state_models.benchmarks.phase3a_checkpoint import write_result


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--seed",type=int,default=17);parser.add_argument("--steps",type=int,default=1000);parser.add_argument("--run-root",default="runs/asm_cm_vr_fixed32_phase1");parser.add_argument("--output",default="docs/benchmarks/asm_cm_vr_fixed32_phase1");parser.add_argument("--device",default="cuda");args=parser.parse_args();output=Path(args.output)
    summary=run_phase1(seed=args.seed,steps=args.steps,run_root=args.run_root,device=args.device);files=("src/drm_language_emitter/rank_aware_memory.py","src/drm_language_emitter/fast_weight_memory.py","src/aletheion_state_models/variants/compact_variable_rank.py","src/aletheion_state_models/benchmarks/cmvr_phase1.py","scripts/run_asm_cm_vr_fixed32_phase1.py");digest=hashlib.sha256()
    for name in files:digest.update(name.encode());digest.update(Path(name).read_bytes())
    manifest={"experiment":summary["experiment"],"seed":args.seed,"steps":args.steps,"arms":{"cm_vr_full64":64,"cm_vr_fixed32":32},"test_protocol":"fixed terminal checkpoint and held-out deterministic MQAR streams","controller":"input-only, prefix mask, frozen","memory_contract":"state and fast-weight value payload projected before write and after read; token/key control plane allowed","implementation":{"files":list(files),"sha256":digest.hexdigest()},"hardware":"NVIDIA GeForce RTX 4090"};write_result(output/"manifest.json",manifest);write_result(output/"summary.json",summary);render_cmvr_phase1(summary,output);print(json.dumps({"gates":summary["gates"],"passed":summary["passed"]},indent=2));raise SystemExit(0 if summary["passed"] else 1)


if __name__=="__main__":main()
