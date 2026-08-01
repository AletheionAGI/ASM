from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from plot_asm_scaling_law import svg_line_chart


def fmt_length(value: float) -> str:
    return f"{int(value / 1024)}K" if value >= 1024 else str(int(value))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot the complete ASM-C validation suite.")
    parser.add_argument("--asm-c-results", type=Path, required=True)
    parser.add_argument("--asm-r-results", type=Path, required=True)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--paired", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    asm_c = json.loads(args.asm_c_results.read_text())
    asm_r = json.loads(args.asm_r_results.read_text())
    comparison = json.loads(args.comparison.read_text())
    paired = json.loads(args.paired.read_text())
    args.output_root.mkdir(parents=True, exist_ok=True)

    c_stream = asm_c["streaming"]
    r_stream = asm_r["streaming"]
    stream = {"ASM-C": c_stream, "ASM-R": r_stream}
    svg_line_chart(args.output_root / "streaming_throughput.svg", stream,
                   "sequence_length", "segment_tokens_per_sec",
                   "ASM-C versus ASM-R: streaming throughput",
                   "Actual incremental decode through 32K; higher is better.",
                   "Sequence length (log scale)", "Tokens per second", fmt_length, True)
    svg_line_chart(args.output_root / "streaming_cache.svg", stream,
                   "sequence_length", "cache_mebibytes",
                   "ASM-C versus ASM-R: persistent cache",
                   "Tensor storage retained by the inference state; lower is better.",
                   "Sequence length (log scale)", "Cache (MiB)", fmt_length, True)
    svg_line_chart(args.output_root / "streaming_peak_vram.svg", stream,
                   "sequence_length", "cuda_peak_mb",
                   "ASM-C versus ASM-R: peak CUDA memory",
                   "Peak allocated memory per streaming segment; lower is better.",
                   "Sequence length (log scale)", "Peak CUDA memory (MiB)", fmt_length, True)
    svg_line_chart(args.output_root / "asm_c_speedup.svg", {"ASM-C / ASM-R": comparison["rows"]},
                   "sequence_length", "speedup", "ASM-C streaming speedup",
                   "Ratio relative to the reference ASM-R streaming path.",
                   "Sequence length (log scale)", "Speedup (x)", fmt_length, True)
    reduction = [dict(row, cache_reduction_percent=100 * row["cache_reduction"])
                 for row in comparison["rows"]]
    svg_line_chart(args.output_root / "asm_c_cache_reduction.svg", {"ASM-C": reduction},
                   "sequence_length", "cache_reduction_percent", "ASM-C cache reduction",
                   "Reduction relative to the reference ASM-R inference cache.",
                   "Sequence length (log scale)", "Cache reduction (%)", fmt_length, True)

    mqar = asm_c["mqar"]
    chance = [{"sequence_length": row["sequence_length"], "validation_accuracy": 1 / 64}
              for row in mqar]
    svg_line_chart(args.output_root / "mqar_accuracy.svg",
                   {"ASM-C": mqar, "Chance (1/64)": chance},
                   "sequence_length", "validation_accuracy", "ASM-C delayed MQAR accuracy",
                   "4096 targets per point; the short control failed the 80% gate.",
                   "Sequence length (log scale)", "Accuracy", fmt_length, True)
    svg_line_chart(args.output_root / "mqar_cross_entropy.svg", {"ASM-C": mqar},
                   "sequence_length", "validation_ce", "ASM-C delayed MQAR cross-entropy",
                   "Lower is better; long-range retention is not interpretable after control failure.",
                   "Sequence length (log scale)", "Cross-entropy", fmt_length, True)

    context: dict[str, list[dict]] = {}
    for row in paired["context"]:
        if row.get("supported"):
            label = "ASM-C" if row["family"] == "asm_r" else "Transformer"
            context.setdefault(label, []).append(row)
    svg_line_chart(args.output_root / "paired_context_ce.svg", context,
                   "context_length", "ce", "ASM-C and Transformer: context CE",
                   "Same Wikipedia validation protocol; lower is better.",
                   "Context length (log scale)", "Cross-entropy", fmt_length, True)
    svg_line_chart(args.output_root / "paired_context_throughput.svg", context,
                   "context_length", "tokens_per_sec", "ASM-C and Transformer: context throughput",
                   "Full-context evaluation throughput; higher is better.",
                   "Context length (log scale)", "Tokens per second", fmt_length, True)
    svg_line_chart(args.output_root / "paired_context_vram.svg", context,
                   "context_length", "peak_memory_mb", "ASM-C and Transformer: context peak memory",
                   "Unsupported Transformer lengths are omitted.",
                   "Context length (log scale)", "Peak CUDA memory (MiB)", fmt_length, True)

    speed: dict[str, list[dict]] = {}
    for row in paired["speed"]:
        if row.get("decode_supported"):
            label = "ASM-C" if row["family"] == "asm_r" else "Transformer"
            speed.setdefault(label, []).append(row)
    svg_line_chart(args.output_root / "paired_decode_throughput.svg", speed,
                   "prompt_tokens", "decode_tokens_per_sec", "ASM-C and Transformer: decode throughput",
                   "128-token incremental decode after each prompt; higher is better.",
                   "Prompt length", "Decode tokens per second", fmt_length, False)

    write_csv(args.output_root / "streaming_comparison.csv", comparison["rows"])
    write_csv(args.output_root / "asm_c_mqar.csv", mqar)
    write_csv(args.output_root / "paired_context.csv", paired["context"])
    write_csv(args.output_root / "paired_speed.csv", paired["speed"])
    print(f"saved={args.output_root}")


if __name__ == "__main__":
    main()
