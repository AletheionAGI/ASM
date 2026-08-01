from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from plot_asm_scaling_law import svg_line_chart


def fmt_length(value: float) -> str:
    return f"{int(value/1024)}K" if value >= 1024 else str(int(value))


def main() -> None:
    parser=argparse.ArgumentParser(description="Plot ASM-R long-streaming results.")
    parser.add_argument("--results",type=Path,required=True); parser.add_argument("--output-root",type=Path,required=True)
    args=parser.parse_args(); payload=json.loads(args.results.read_text()); args.output_root.mkdir(parents=True,exist_ok=True)
    streaming=[row for row in payload["streaming"] if row.get("status")!="failed"]
    mqar=[row for row in payload["mqar"] if row.get("status")!="failed"]
    stream_series={"ASM_R_STREAM":streaming}
    svg_line_chart(args.output_root/"decode_throughput_by_length.svg",stream_series,"sequence_length","segment_tokens_per_sec","ASM-R long streaming: decode throughput","Actual decode_step path through 32K; higher is better.","Sequence length (log scale)","Decode tokens per second",fmt_length,True)
    svg_line_chart(args.output_root/"cache_size_by_length.svg",stream_series,"sequence_length","cache_mebibytes","ASM-R long streaming: persistent cache size","Tensor bytes retained by the inference state; lower and flatter is better.","Sequence length (log scale)","Cache size (MiB)",fmt_length,True)
    svg_line_chart(args.output_root/"peak_vram_by_length.svg",stream_series,"sequence_length","cuda_peak_mb","ASM-R long streaming: peak CUDA memory","Peak allocated CUDA memory within each decode segment.","Sequence length (log scale)","Peak CUDA memory (MiB)",fmt_length,True)
    chance=[{"sequence_length":row["sequence_length"],"validation_accuracy":1/64} for row in mqar]
    svg_line_chart(args.output_root/"mqar_accuracy_by_distance.svg",{"ASM_R_STREAM":mqar,"MQAR_CHANCE":chance},"sequence_length","validation_accuracy","Delayed MQAR accuracy by sequence length","Fine-tuned at length 40 and evaluated after increasing filler; 64 targets per point.","Sequence length (log scale)","Validation accuracy",fmt_length,True)
    svg_line_chart(args.output_root/"mqar_ce_by_distance.svg",{"ASM_R_STREAM":mqar},"sequence_length","validation_ce","Delayed MQAR cross-entropy by sequence length","Fine-tuned at length 40; lower is better.","Sequence length (log scale)","Cross-entropy",fmt_length,True)
    with (args.output_root/"streaming_data.csv").open('w',newline='',encoding='utf-8') as handle:
        fields=list(streaming[0]); writer=csv.DictWriter(handle,fieldnames=fields); writer.writeheader(); writer.writerows(streaming)
    with (args.output_root/"mqar_data.csv").open('w',newline='',encoding='utf-8') as handle:
        fields=list(mqar[0]); writer=csv.DictWriter(handle,fieldnames=fields); writer.writeheader(); writer.writerows(mqar)
    print(f"saved={args.output_root}")


if __name__=="__main__": main()
