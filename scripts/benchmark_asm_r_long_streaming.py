from __future__ import annotations

import argparse
import json
import math
import sys
from contextlib import nullcontext
from dataclasses import fields, is_dataclass
from pathlib import Path
from time import perf_counter

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))

from drm_language_emitter.checkpoint import load_model


def parse_lengths(raw: str) -> list[int]:
    values=sorted({int(value.strip()) for value in raw.split(',') if value.strip()})
    if not values or values[0] <= 0: raise ValueError("lengths must be positive")
    return values


def wilson_interval(correct: int, total: int, z: float = 1.96) -> tuple[float, float]:
    proportion=correct/total; denominator=1+z*z/total
    center=(proportion+z*z/(2*total))/denominator
    radius=z*math.sqrt(proportion*(1-proportion)/total+z*z/(4*total*total))/denominator
    return max(0.0,center-radius),min(1.0,center+radius)


def precision_context(device: torch.device):
    return torch.autocast("cuda",dtype=torch.bfloat16) if device.type=="cuda" else nullcontext()


def sync(device: torch.device) -> None:
    if device.type=="cuda": torch.cuda.synchronize(device)


def tensor_bytes(value) -> int:
    if torch.is_tensor(value): return value.numel()*value.element_size()
    if is_dataclass(value): return sum(tensor_bytes(getattr(value,item.name)) for item in fields(value))
    if isinstance(value,(tuple,list)): return sum(tensor_bytes(item) for item in value)
    if isinstance(value,dict): return sum(tensor_bytes(item) for item in value.values())
    return 0


def save_partial(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp'); temporary.write_text(json.dumps(payload,indent=2)+'\n',encoding='utf-8'); temporary.replace(path)


@torch.inference_mode()
def streaming_probe(model, lengths: list[int], prompt_tokens: int, device: torch.device, seed: int, partial: Path) -> list[dict]:
    if prompt_tokens >= lengths[0]: raise ValueError("prompt-tokens must be below the first length")
    generator=torch.Generator(device=device).manual_seed(seed)
    prompt=torch.randint(0,model.config.vocab_size,(1,prompt_tokens),generator=generator,device=device)
    with precision_context(device): _,state=model.prefill(prompt)
    sync(device); rows=[]; position=prompt_tokens; segment_started=perf_counter(); segment_start=position
    if device.type=='cuda': torch.cuda.reset_peak_memory_stats(device)
    for target in lengths:
        try:
            while position < target:
                token=torch.randint(0,model.config.vocab_size,(1,),generator=generator,device=device)
                with precision_context(device): _,state=model.decode_step(token,state)
                position += 1
            sync(device); elapsed=perf_counter()-segment_started
            rows.append({"sequence_length":target,"segment_tokens":target-segment_start,"segment_elapsed_sec":elapsed,"segment_tokens_per_sec":(target-segment_start)/elapsed,"state_sequence_length":state.sequence_length,"cache_tensor_bytes":tensor_bytes(state),"cache_mebibytes":tensor_bytes(state)/2**20,"cuda_allocated_mb":torch.cuda.memory_allocated(device)/2**20 if device.type=='cuda' else None,"cuda_peak_mb":torch.cuda.max_memory_allocated(device)/2**20 if device.type=='cuda' else None,"open_block_tokens":int(state.block_tokens.shape[1]) if state.block_tokens is not None else None})
            print(json.dumps({"streaming":rows[-1]}),flush=True); save_partial(partial,{"streaming":rows})
            segment_started=perf_counter(); segment_start=target
            if device.type=='cuda': torch.cuda.reset_peak_memory_stats(device)
        except (torch.OutOfMemoryError,RuntimeError) as exc:
            rows.append({"sequence_length":target,"status":"failed","error":str(exc)})
            save_partial(partial,{"streaming":rows}); break
    return rows


def delayed_mqar_batch(length: int,batch_size: int,n_pairs: int,n_queries: int,generator: torch.Generator,device: torch.device):
    key_offset=2; value_offset=34; query_token=1; sequences=[]; masks=[]
    minimum=2*n_pairs+3*n_queries
    if length < minimum: raise ValueError(f"MQAR length must be at least {minimum}")
    filler_count=length-minimum
    for _ in range(batch_size):
        keys=torch.randperm(32,generator=generator)[:n_pairs]+key_offset
        values=torch.randint(0,64,(n_pairs,),generator=generator)+value_offset
        queries=torch.randperm(n_pairs,generator=generator)[:n_queries]
        tokens=[]
        for key,value in zip(keys.tolist(),values.tolist()): tokens += [key,value]
        tokens += torch.randint(98,256,(filler_count,),generator=generator).tolist()
        answers=[]
        for index in queries.tolist(): tokens += [query_token,int(keys[index]),int(values[index])]; answers.append(len(tokens)-2)
        sequence=torch.tensor(tokens); mask=torch.zeros(length-1,dtype=torch.bool); mask[torch.tensor(answers)]=True
        sequences.append(sequence); masks.append(mask)
    stacked=torch.stack(sequences).to(device)
    return stacked[:,:-1],stacked[:,1:],torch.stack(masks).to(device)


def adapt_mqar(model,steps: int,batch_size: int,device: torch.device,seed: int) -> float:
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=.01); generator=torch.Generator().manual_seed(seed); model.train(); started=perf_counter()
    for step in range(1,steps+1):
        x,y,mask=delayed_mqar_batch(40,batch_size,8,8,generator,device)
        with precision_context(device): selected=model(x,collect_diagnostics=False)["logits"][mask]
        loss=F.cross_entropy(selected.float(),y[mask]); optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0,error_if_nonfinite=True); optimizer.step()
        if step==1 or step%500==0: print(json.dumps({"mqar_adaptation_step":step,"train_ce":float(loss)}),flush=True)
    return perf_counter()-started


@torch.inference_mode()
def mqar_retention(model,lengths: list[int],batches: int,batch_size: int,device: torch.device,seed: int,partial: Path,streaming_rows: list[dict],adaptation_sec: float) -> list[dict]:
    rows=[]; model.eval()
    for length in lengths:
        losses=[]; correct=total=0; generator=torch.Generator().manual_seed(seed+length); started=perf_counter()
        try:
            for _ in range(batches):
                x,y,mask=delayed_mqar_batch(length,batch_size,8,8,generator,device)
                with precision_context(device): selected=model(x,collect_diagnostics=False)["logits"][mask]
                targets=y[mask]; losses.append(float(F.cross_entropy(selected.float(),targets))); correct += int((selected.argmax(-1)==targets).sum()); total += targets.numel()
            sync(device); low,high=wilson_interval(correct,total); rows.append({"sequence_length":length,"validation_ce":sum(losses)/len(losses),"validation_accuracy":correct/total,"correct":correct,"targets":total,"accuracy_ci95_low":low,"accuracy_ci95_high":high,"elapsed_sec":perf_counter()-started})
            print(json.dumps({"mqar":rows[-1]}),flush=True)
        except (torch.OutOfMemoryError,RuntimeError) as exc: rows.append({"sequence_length":length,"status":"failed","error":str(exc)})
        save_partial(partial,{"streaming":streaming_rows,"mqar_adaptation_sec":adaptation_sec,"mqar":rows})
    return rows


def main() -> None:
    parser=argparse.ArgumentParser(description="ASM-R long-streaming cache and delayed-MQAR probe.")
    parser.add_argument("--checkpoint",type=Path,required=True); parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--lengths",default="512,1024,2048,4096,8192,16384,32768"); parser.add_argument("--prompt-tokens",type=int,default=64)
    parser.add_argument("--mqar-steps",type=int,default=5000); parser.add_argument("--mqar-batches",type=int,default=8); parser.add_argument("--mqar-batch-size",type=int,default=1)
    parser.add_argument("--mqar-control-length",type=int,default=40); parser.add_argument("--mqar-control-threshold",type=float,default=.8)
    parser.add_argument("--skip-mqar", action="store_true", help="Measure streaming throughput and memory without retraining or evaluating MQAR.")
    parser.add_argument("--compact",action="store_true")
    parser.add_argument("--seed",type=int,default=1234); parser.add_argument("--device",default="cuda")
    args=parser.parse_args(); lengths=parse_lengths(args.lengths); device=torch.device(args.device)
    if device.type=='cuda' and not torch.cuda.is_available(): raise SystemExit("CUDA requested but unavailable")
    model=load_model(args.checkpoint).to(device).eval(); model.config.compact_streaming_inference=args.compact; partial=args.output.with_name('partial.json')
    streaming=streaming_probe(model,lengths,args.prompt_tokens,device,args.seed,partial)
    if args.skip_mqar:
        payload={"checkpoint":str(args.checkpoint),"protocol":{"lengths":lengths,"prompt_tokens":args.prompt_tokens,"compact":args.compact,"skip_mqar":True,"device":str(device),"seed":args.seed},"streaming":streaming,"mqar":[],"interpretation":{"streaming":"Actual decode_step path; compact mode retains only counter, completed state, and open block.","mqar":"Not repeated: this run isolates post-FP32 CE, throughput, and VRAM remeasurement."}}
        save_partial(args.output,payload); print(f"saved={args.output}"); return
    del model
    if device.type=='cuda': torch.cuda.empty_cache()
    model=load_model(args.checkpoint).to(device); model.config.compact_streaming_inference=args.compact; adaptation_sec=adapt_mqar(model,args.mqar_steps,4,device,args.seed)
    mqar_lengths=[args.mqar_control_length]+[value for value in lengths if value!=args.mqar_control_length]
    mqar=mqar_retention(model,mqar_lengths,args.mqar_batches,args.mqar_batch_size,device,args.seed+100000,partial,streaming,adaptation_sec)
    control=next(row for row in mqar if row.get('sequence_length')==args.mqar_control_length)
    control_passed=control.get('validation_accuracy',0)>=args.mqar_control_threshold
    payload={"checkpoint":str(args.checkpoint),"protocol":{"lengths":lengths,"prompt_tokens":args.prompt_tokens,"compact":args.compact,"mqar_steps":args.mqar_steps,"mqar_batches":args.mqar_batches,"mqar_batch_size":args.mqar_batch_size,"mqar_control_length":args.mqar_control_length,"mqar_control_threshold":args.mqar_control_threshold,"device":str(device),"seed":args.seed},"streaming":streaming,"mqar_adaptation_sec":adaptation_sec,"mqar_control_passed":control_passed,"mqar_retention_interpretable":control_passed,"mqar":mqar,"interpretation":{"streaming":"Actual decode_step path; compact mode retains only counter, completed state, and open block.","mqar":"Retention is interpretable only if the post-adaptation short control reaches its threshold."}}
    save_partial(args.output,payload); print(f"saved={args.output}")


if __name__=="__main__": main()
