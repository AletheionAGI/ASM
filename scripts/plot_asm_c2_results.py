from __future__ import annotations

import argparse
import json
from pathlib import Path

from plot_asm_scaling_law import svg_line_chart


def fmt_length(value: float) -> str:
    return f"{int(value / 1024)}K" if value >= 1024 else str(int(value))


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot ASM-C2 short and long validation.")
    parser.add_argument("--short-results", type=Path, required=True)
    parser.add_argument("--long-results", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    short = json.loads(args.short_results.read_text())
    series = {result["variant"]: result["rows"] for result in short["results"]}
    args.output_root.mkdir(parents=True, exist_ok=True)
    svg_line_chart(args.output_root / "short_mqar_accuracy.svg", series, "step", "validation_accuracy", "ASM-C2 short MQAR", "Paired batches; 80% gate; higher is better.", "Steps", "Accuracy", lambda x: f"{int(x/1000)}K" if x else "0")
    svg_line_chart(args.output_root / "short_mqar_ce.svg", series, "step", "validation_ce", "ASM-C2 short MQAR cross-entropy", "Paired batches; lower is better.", "Steps", "Cross-entropy", lambda x: f"{int(x/1000)}K" if x else "0")
    if args.long_results and args.long_results.exists():
        long = json.loads(args.long_results.read_text())
        streaming = [row for row in long["streaming"] if row.get("status") != "failed"]
        mqar = [row for row in long["mqar"] if row.get("status") != "failed"]
        label = "ASM_C2_32"
        svg_line_chart(args.output_root / "long_streaming_throughput.svg", {label: streaming}, "sequence_length", "segment_tokens_per_sec", "ASM-C2 long streaming throughput", "Actual compact decode; higher is better.", "Sequence length", "Tokens per second", fmt_length, True)
        svg_line_chart(args.output_root / "long_streaming_cache.svg", {label: streaming}, "sequence_length", "cache_mebibytes", "ASM-C2 persistent cache", "Fixed slots plus compact recurrent state.", "Sequence length", "Cache (MiB)", fmt_length, True)
        svg_line_chart(args.output_root / "long_mqar_accuracy.svg", {label: mqar}, "sequence_length", "validation_accuracy", "ASM-C2 delayed MQAR", "Interpretable only after the short control passes.", "Sequence length", "Accuracy", fmt_length, True)
    print(f"saved={args.output_root}")


if __name__ == "__main__":
    main()
