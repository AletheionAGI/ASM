from __future__ import annotations

import argparse
import json
from contextlib import nullcontext
from pathlib import Path

import torch

from drm_language_emitter.checkpoint import load_model


def main() -> None:
    parser=argparse.ArgumentParser(description="Measure ASM-C compact decode parity against full ASM-R forward.")
    parser.add_argument("--checkpoint",type=Path,required=True); parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--batch-size",type=int,default=2); parser.add_argument("--prompt-tokens",type=int,default=64); parser.add_argument("--decode-tokens",type=int,default=256); parser.add_argument("--seed",type=int,default=1234); parser.add_argument("--device",default="cuda"); parser.add_argument("--precision",choices=("fp32","bf16"),default="bf16")
    args=parser.parse_args(); device=torch.device(args.device); model=load_model(args.checkpoint).to(device).eval(); generator=torch.Generator(device=device).manual_seed(args.seed)
    tokens=torch.randint(0,model.config.vocab_size,(args.batch_size,args.prompt_tokens+args.decode_tokens),generator=generator,device=device)
    context=torch.autocast("cuda",dtype=torch.bfloat16) if device.type=="cuda" and args.precision=="bf16" else nullcontext()
    with torch.inference_mode(),context: reference=model(tokens,collect_diagnostics=False)["logits"][:,args.prompt_tokens:]
    model.config.compact_streaming_inference=True; rows=[]
    with torch.inference_mode(),context:
        _,state=model.prefill(tokens[:,:args.prompt_tokens])
        for position in range(args.prompt_tokens,tokens.shape[1]): logits,state=model.decode_step(tokens[:,position],state); rows.append(logits[:,None])
    actual=torch.cat(rows,1); error=(actual.float()-reference.float()).abs(); mismatch=(actual.argmax(-1)!=reference.argmax(-1))
    payload={"precision":args.precision,"batch_size":args.batch_size,"prompt_tokens":args.prompt_tokens,"decode_tokens":args.decode_tokens,"stable_compact_fp32_core":bool(model.config.compact_streaming_inference and model.config.fast_weight_compute_fp32),"max_abs_error":float(error.max()),"mean_abs_error":float(error.mean()),"argmax_mismatches":int(mismatch.sum()),"argmax_total":mismatch.numel(),"argmax_mismatch_rate":float(mismatch.float().mean()),"final_cache_tokens":state.input_ids.numel(),"final_sequence_length":state.sequence_length}
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(payload,indent=2)+'\n'); print(json.dumps(payload,indent=2))


if __name__=="__main__": main()
