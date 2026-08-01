from __future__ import annotations

import argparse
import json
import math
import sys
from contextlib import nullcontext
from pathlib import Path
from time import perf_counter

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from drm_language_emitter.checkpoint import load_model
from drm_language_emitter.data import MemmapTokenDataset
from drm_language_emitter.tokenizer import ByteTokenizer
from scripts.evaluate_frozen_test import load_gpt2, sha256_file


PROMPTS = ("Aletheion State Models ", "The geometry of language ", "In a causal state model, memory ")


def precision_context(device: torch.device):
    return torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()


def sync(device: torch.device) -> None:
    if device.type == "cuda": torch.cuda.synchronize(device)


def reset_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(device)


def peak_mb(device: torch.device) -> float | None:
    return torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None


def logits(model, family: str, x: torch.Tensor) -> torch.Tensor:
    return model(x, collect_diagnostics=False)["logits"] if family == "asm_r" else model(input_ids=x).logits


@torch.inference_mode()
def context_probe(models: dict, dataset: MemmapTokenDataset, lengths: list[int], batches: int, device: torch.device) -> list[dict]:
    rows=[]
    for length in lengths:
        starts=[round(index*(len(dataset)-length-1)/max(batches-1,1)) for index in range(batches)]
        windows=[dataset.window(start,length) for start in starts]
        for family,model in models.items():
            supported = family == "asm_r" or length <= int(model.config.n_positions)
            if not supported:
                rows.append({"family":family,"context_length":length,"supported":False,"reason":"exceeds learned absolute-position limit"}); continue
            reset_memory(device); total_loss=0.0; total_tokens=0; started=perf_counter()
            for x,y in windows:
                x=x.unsqueeze(0).to(device); y=y.unsqueeze(0).to(device)
                with precision_context(device): out=logits(model,family,x)
                total_loss += float(F.cross_entropy(out.float().reshape(-1,out.shape[-1]),y.reshape(-1),reduction="sum")); total_tokens += y.numel()
            sync(device); elapsed=perf_counter()-started
            rows.append({"family":family,"context_length":length,"supported":True,"ce":total_loss/total_tokens,"ppl":math.exp(min(total_loss/total_tokens,20)),"tokens":total_tokens,"elapsed_sec":elapsed,"tokens_per_sec":total_tokens/elapsed,"peak_memory_mb":peak_mb(device)})
    return rows


@torch.inference_mode()
def speed_probe(models: dict, lengths: list[int], decode_tokens: int, repeats: int, device: torch.device, seed: int) -> list[dict]:
    rows=[]; generator=torch.Generator(device=device).manual_seed(seed)
    for length in lengths:
        transformer_limit = int(models["transformer"].config.n_positions)
        paired_decode_tokens = min(decode_tokens, max(transformer_limit - length, 0))
        prompt=torch.randint(0,256,(1,length),generator=generator,device=device); continuation=torch.randint(0,256,(1,max(paired_decode_tokens,1)),generator=generator,device=device)
        for family,model in models.items():
            if family == "transformer" and length > int(model.config.n_positions): continue
            reset_memory(device)
            with precision_context(device): logits(model,family,prompt)
            sync(device); started=perf_counter()
            for _ in range(repeats):
                with precision_context(device): logits(model,family,prompt)
            sync(device); prefill_sec=(perf_counter()-started)/repeats
            prefill_peak=peak_mb(device); reset_memory(device)
            if paired_decode_tokens == 0:
                rows.append({"family":family,"prompt_tokens":length,"requested_decode_tokens":decode_tokens,"decode_tokens":0,"decode_supported":False,"decode_reason":"no shared positional capacity remains after prefill","prefill_sec":prefill_sec,"prefill_tokens_per_sec":length/prefill_sec,"prefill_peak_memory_mb":prefill_peak,"decode_sec":None,"decode_tokens_per_sec":None,"decode_peak_memory_mb":None})
                continue
            if family == "asm_r":
                with precision_context(device): _,state=model.prefill(prompt)
                sync(device); started=perf_counter()
                for position in range(paired_decode_tokens):
                    with precision_context(device): _,state=model.decode_step(continuation[:,position],state)
            else:
                with precision_context(device): output=model(input_ids=prompt,use_cache=True); cache=output.past_key_values
                sync(device); started=perf_counter()
                for position in range(paired_decode_tokens):
                    with precision_context(device): output=model(input_ids=continuation[:,position:position+1],past_key_values=cache,use_cache=True); cache=output.past_key_values
            sync(device); decode_sec=perf_counter()-started
            rows.append({"family":family,"prompt_tokens":length,"requested_decode_tokens":decode_tokens,"decode_tokens":paired_decode_tokens,"decode_supported":True,"prefill_sec":prefill_sec,"prefill_tokens_per_sec":length/prefill_sec,"prefill_peak_memory_mb":prefill_peak,"decode_sec":decode_sec,"decode_tokens_per_sec":paired_decode_tokens/decode_sec,"decode_peak_memory_mb":peak_mb(device)})
    return rows


@torch.inference_mode()
def generations(models: dict, device: torch.device, max_new: int, seed: int) -> list[dict]:
    tokenizer=ByteTokenizer(); rows=[]
    for prompt_index,prompt in enumerate(PROMPTS):
        base=torch.tensor([tokenizer.encode(prompt)],device=device)
        for family,model in models.items():
            torch.manual_seed(seed+prompt_index); generated=base.clone()
            if family == "asm_r":
                with precision_context(device): out,state=model.prefill(generated)
                next_logits=out[:,-1]
                for _ in range(max_new):
                    probs=torch.softmax(next_logits.float()/0.8,dim=-1); token=torch.multinomial(probs,1); generated=torch.cat((generated,token),1)
                    with precision_context(device): next_logits,state=model.decode_step(token[:,0],state)
            else:
                cache=None
                for _ in range(max_new):
                    inputs=generated if cache is None else generated[:,-1:]
                    with precision_context(device): out=model(input_ids=inputs,past_key_values=cache,use_cache=True)
                    cache=out.past_key_values; probs=torch.softmax(out.logits[:,-1].float()/0.8,dim=-1); token=torch.multinomial(probs,1); generated=torch.cat((generated,token),1)
            rows.append({"family":family,"prompt":prompt,"text":tokenizer.decode(generated[0].tolist())})
    return rows


def main() -> None:
    parser=argparse.ArgumentParser(description="Paired ASM-R versus Transformer inference/context suite.")
    parser.add_argument("--asm-checkpoint",type=Path,required=True); parser.add_argument("--transformer-checkpoint",type=Path,required=True)
    parser.add_argument("--manifest",type=Path,required=True); parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--split",default="validation")
    parser.add_argument("--context-lengths",default="64,128,256,512,1024,2048"); parser.add_argument("--context-batches",type=int,default=8)
    parser.add_argument("--speed-lengths",default="64,128,256,512"); parser.add_argument("--speed-repeats",type=int,default=5); parser.add_argument("--decode-tokens",type=int,default=128)
    parser.add_argument("--generation-tokens",type=int,default=128); parser.add_argument("--seed",type=int,default=1234); parser.add_argument("--device",default="cuda")
    parser.add_argument("--skip-speed",action="store_true"); parser.add_argument("--skip-generation",action="store_true")
    parser.add_argument("--asm-compact",action="store_true")
    args=parser.parse_args(); device=torch.device(args.device)
    if device.type=="cuda" and not torch.cuda.is_available(): raise SystemExit("CUDA requested but unavailable")
    models={"asm_r":load_model(args.asm_checkpoint).to(device).eval(),"transformer":load_gpt2(args.transformer_checkpoint).to(device).eval()}
    models["asm_r"].config.compact_streaming_inference=args.asm_compact
    with MemmapTokenDataset(args.manifest,split=args.split) as dataset:
        context=context_probe(models,dataset,[int(x) for x in args.context_lengths.split(',')],args.context_batches,device)
    result={"protocol":{"manifest":str(args.manifest),"manifest_sha256":sha256_file(args.manifest),"split":args.split,"asm_compact":args.asm_compact,"context_batches":args.context_batches,"speed_repeats":args.speed_repeats,"decode_tokens":args.decode_tokens,"device":str(device)},"checkpoints":{"asm_r":{"path":str(args.asm_checkpoint),"sha256":sha256_file(args.asm_checkpoint)},"transformer":{"path":str(args.transformer_checkpoint),"sha256":sha256_file(args.transformer_checkpoint)}},"context":context,"speed":[] if args.skip_speed else speed_probe(models,[int(x) for x in args.speed_lengths.split(',')],args.decode_tokens,args.speed_repeats,device,args.seed),"generations":[] if args.skip_generation else generations(models,device,args.generation_tokens,args.seed)}
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8'); print(f"saved={args.output}")


if __name__ == "__main__": main()
