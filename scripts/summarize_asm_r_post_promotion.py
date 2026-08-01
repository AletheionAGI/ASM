from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Consolidate the ASM-R post-promotion suite.")
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    checkpoint = load_json(root / "checkpoint_evaluation.json")
    causality = load_json(root / "causality.json")
    decode = load_json(root / "incremental_decode.json")
    validation = load_json(root / "validation.json")
    summary = {
        "status": "passed" if causality.get("passed") else "failed",
        "checkpoint": checkpoint["checkpoint"],
        "checkpoint_audit": checkpoint["audit"],
        "validation": {
            "ce": validation.get("test_ce"),
            "ppl": validation.get("test_ppl"),
            "tokens": validation.get("test_tokens"),
        },
        "causality": {
            "passed": causality.get("passed"),
            "max_logit_abs_diff": causality.get("max_logit_abs_diff"),
            "max_state_abs_diff": causality.get("max_state_abs_diff"),
        },
        "incremental_decode": decode,
        "context": checkpoint["context"],
        "mqar": checkpoint["mqar"],
        "generations": checkpoint["generations"],
    }
    (root / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    context_rows = "\n".join(
        f"| {row['context_length']} | {row['ce']:.6f} | {row['ppl']:.4f} | {row['tokens_per_sec']:.1f} |"
        for row in summary["context"]
    )
    generations = "\n\n".join(
        f"### Prompt: `{row['prompt']}`\n\n```text\n{row['text']}\n```"
        for row in summary["generations"]
    )
    mqar = summary["mqar"]
    report = f"""# ASM-R post-promotion evaluation

## Outcome

- Suite status: **{summary['status']}**
- Frozen validation CE: **{summary['validation']['ce']:.6f}**
- Frozen validation PPL: **{summary['validation']['ppl']:.4f}**
- Causality passed: **{summary['causality']['passed']}**
- Cached decode speedup: **{summary['incremental_decode']['speedup']:.3f}x**
- Cached decode parity max error: **{summary['incremental_decode']['max_abs_error']:.6g}**

## Context-length evaluation

| Context | CE | PPL | tokens/s |
|---:|---:|---:|---:|
{context_rows}

Lengths beyond the training context are extrapolation probes, not evidence of
training at those lengths.

## MQAR adaptation probe

- Interpretation: {mqar['interpretation']}
- Steps: {mqar['steps']}
- Accuracy before adaptation: {mqar['before']['accuracy']:.6f}
- Accuracy after adaptation: {mqar['after']['accuracy']:.6f}
- CE after adaptation: {mqar['after']['ce']:.6f}

## Fixed-prompt generations

{generations}
"""
    (root / "report.md").write_text(report, encoding="utf-8")
    print(f"saved={root / 'summary.json'}")
    print(f"saved={root / 'report.md'}")


if __name__ == "__main__":
    main()
