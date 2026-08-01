from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser=argparse.ArgumentParser(description="Summarize paired ASM-R/Transformer suite.")
    parser.add_argument("--root",type=Path,required=True); args=parser.parse_args()
    scaling=json.loads((args.root/"milestone_rescoring.json").read_text(encoding="utf-8")); bench=json.loads((args.root/"paired_benchmark.json").read_text(encoding="utf-8"))
    milestone_rows=[]
    for tokens in scaling["protocol"]["milestones"]:
        by={row["variant"]:row for row in scaling["rows"] if int(row["milestone_tokens"])==tokens}
        milestone_rows.append(f"| {tokens/1e6:g}M | {by['ASM_R']['validation_ce']:.6f} | {by['TRANSFORMER']['validation_ce']:.6f} | {by['ASM_R']['validation_ce']-by['TRANSFORMER']['validation_ce']:+.6f} |")
    context_rows=[]
    lengths=sorted({row["context_length"] for row in bench["context"]})
    for length in lengths:
        by={row["family"]:row for row in bench["context"] if row["context_length"]==length}
        asm=by["asm_r"]; tr=by["transformer"]
        tr_ce=f"{tr['ce']:.6f}" if tr.get("supported") else "unsupported"
        context_rows.append(f"| {length} | {asm['ce']:.6f} | {tr_ce} | {asm['peak_memory_mb']:.1f} | {tr.get('peak_memory_mb', 0):.1f} |")
    speed_rows=[]
    for length in sorted({row["prompt_tokens"] for row in bench["speed"]}):
        by={row["family"]:row for row in bench["speed"] if row["prompt_tokens"]==length}
        asm_decode=f"{by['asm_r']['decode_tokens_per_sec']:.1f}" if by['asm_r'].get('decode_supported') else "unsupported"
        tr_decode=f"{by['transformer']['decode_tokens_per_sec']:.1f}" if by['transformer'].get('decode_supported') else "unsupported"
        speed_rows.append(f"| {length} | {by['asm_r']['prefill_tokens_per_sec']:.1f} | {by['transformer']['prefill_tokens_per_sec']:.1f} | {asm_decode} | {tr_decode} |")
    generations='\n\n'.join(f"### {row['family']} — `{row['prompt']}`\n\n```text\n{row['text']}\n```" for row in bench["generations"])
    report=f"""# ASM-R versus Transformer — paired 100M suite

## Frozen milestone rescoring

| Tokens | ASM-R CE | Transformer CE | ASM-R − Transformer |
|---:|---:|---:|---:|
{chr(10).join(milestone_rows)}

Positive gaps favor the Transformer. Every checkpoint is scored over the same
continuous validation stream.

## Context length, CE, and peak VRAM

| Context | ASM-R CE | Transformer CE | ASM-R MiB | Transformer MiB |
|---:|---:|---:|---:|---:|
{chr(10).join(context_rows)}

Transformer contexts beyond 512 are unsupported because the matched checkpoint
uses learned absolute positions with a training limit of 512. ASM-R results above
512 are extrapolation probes, not a paired victory.

## Prefill and cached decode

| Prompt | ASM-R prefill tok/s | Transformer prefill tok/s | ASM-R decode tok/s | Transformer decode tok/s |
|---:|---:|---:|---:|---:|
{chr(10).join(speed_rows)}

## Fixed-prompt qualitative generation

{generations}
"""
    (args.root/"report.md").write_text(report,encoding="utf-8"); print(f"saved={args.root/'report.md'}")


if __name__ == "__main__": main()
