from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser=argparse.ArgumentParser(description="Compare ASM-R and ASM-C streaming results.")
    parser.add_argument("--asm-r",type=Path,required=True); parser.add_argument("--asm-c",type=Path,required=True); parser.add_argument("--output",type=Path,required=True); parser.add_argument("--report",type=Path,required=True)
    args=parser.parse_args(); r=json.loads(args.asm_r.read_text()); c=json.loads(args.asm_c.read_text())
    rrows={row['sequence_length']:row for row in r['streaming']}; crows={row['sequence_length']:row for row in c['streaming']}; rows=[]
    for length in sorted(rrows.keys() & crows.keys()):
        old,new=rrows[length],crows[length]; rows.append({"sequence_length":length,"asm_r_tokens_per_sec":old['segment_tokens_per_sec'],"asm_c_tokens_per_sec":new['segment_tokens_per_sec'],"speedup":new['segment_tokens_per_sec']/old['segment_tokens_per_sec'],"asm_r_cache_bytes":old['cache_tensor_bytes'],"asm_c_cache_bytes":new['cache_tensor_bytes'],"cache_reduction":1-new['cache_tensor_bytes']/old['cache_tensor_bytes'],"asm_r_peak_mb":old['cuda_peak_mb'],"asm_c_peak_mb":new['cuda_peak_mb']})
    by={row['sequence_length']:row for row in rows}; c4,c32=by[4096],by[32768]
    criteria={"cache_bounded_4k_to_32k":c32['asm_c_cache_bytes']<=c4['asm_c_cache_bytes']*1.10,"peak_vram_growth_at_most_10pct":c32['asm_c_peak_mb']<=c4['asm_c_peak_mb']*1.10,"throughput_retention_at_least_90pct":c32['asm_c_tokens_per_sec']>=c4['asm_c_tokens_per_sec']*.90,"mqar_short_control_at_least_80pct":bool(c.get('mqar_control_passed'))}
    payload={"rows":rows,"criteria":criteria,"all_streaming_criteria_passed":all(criteria.values()),"mqar_retention_interpretable":c.get('mqar_retention_interpretable',False)}
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(payload,indent=2)+'\n')
    table='\n'.join(f"| {row['sequence_length']} | {row['asm_r_tokens_per_sec']:.1f} | {row['asm_c_tokens_per_sec']:.1f} | {row['speedup']:.2f}x | {row['asm_r_cache_bytes']} | {row['asm_c_cache_bytes']} | {row['asm_r_peak_mb']:.1f} | {row['asm_c_peak_mb']:.1f} |" for row in rows)
    checks='\n'.join(f"- {name}: **{'PASS' if passed else 'FAIL'}**" for name,passed in criteria.items())
    args.report.write_text(f"""# ASM-C streaming validation

| Length | ASM-R tok/s | ASM-C tok/s | Speedup | ASM-R cache B | ASM-C cache B | ASM-R peak MiB | ASM-C peak MiB |
|---:|---:|---:|---:|---:|---:|---:|---:|
{table}

## Promotion criteria

{checks}

MQAR retention is interpretable: **{payload['mqar_retention_interpretable']}**.
ASM-C is experimental and does not alter the promoted ASM-R checkpoint weights.
""",encoding='utf-8'); print(f"saved={args.output}"); print(f"saved={args.report}")


if __name__=="__main__": main()
