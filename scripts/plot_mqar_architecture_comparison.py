from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from plot_asm_scaling_law import svg_line_chart


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot paired MQAR learning curves.")
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.results.read_text(encoding="utf-8"))
    series = {result["variant"]: result["rows"] for result in payload["results"]}
    args.output_root.mkdir(parents=True, exist_ok=True)
    svg_line_chart(
        args.output_root / "mqar_accuracy_by_steps.svg",
        series,
        "step",
        "validation_accuracy",
        "MQAR short-control learning curve",
        "Same training and frozen validation batches; higher is better.",
        "Adaptation steps",
        "Validation accuracy",
        lambda value: f"{int(value / 1000)}K" if value else "0",
    )
    svg_line_chart(
        args.output_root / "mqar_ce_by_steps.svg",
        series,
        "step",
        "validation_ce",
        "MQAR short-control cross-entropy",
        "Same training and frozen validation batches; lower is better.",
        "Adaptation steps",
        "Validation cross-entropy",
        lambda value: f"{int(value / 1000)}K" if value else "0",
    )
    rows = [dict(row, variant=result["variant"]) for result in payload["results"] for row in result["rows"]]
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (args.output_root / "mqar_learning_curves.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved={args.output_root}")


if __name__ == "__main__":
    main()
