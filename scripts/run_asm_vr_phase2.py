"""Run the deterministic ASM-VR Phase 2 synthetic benchmark."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile

from aletheion_state_models.synthetic.phase2_benchmark import (
    VARIANTS,
    summarize_phase2,
    train_phase2_run,
)


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--evaluation-batches", type=int, default=16)
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 29, 43])
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/benchmarks/asm_vr_phase2/summary.json"),
    )
    args = parser.parse_args()
    results = []
    for seed in args.seeds:
        for variant in args.variants:
            result = train_phase2_run(
                variant,
                seed,
                steps=args.steps,
                batch_size=args.batch_size,
                evaluation_batches=args.evaluation_batches,
            )
            results.append(result)
            print(
                f"{variant} seed={seed} accuracy={result.validation_accuracy:.4f} "
                f"rank={result.mean_rank:.2f} corr={result.rank_difficulty_correlation:.3f}"
            )
    summary = summarize_phase2(results)
    _atomic_json(args.output, summary)
    print(json.dumps(summary["gates"], indent=2, sort_keys=True))
    if not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
