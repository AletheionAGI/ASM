from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import time
from pathlib import Path
from typing import Any

import torch

from drm_language_emitter.checkpoint import load_model
from drm_language_emitter.data import MemmapTokenDataset
from drm_language_emitter.utils import save_json
from evaluate_frozen_test import current_commit, evaluate_sequential, sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rescore DRM-fix variants on one continuous validation sequence."
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--variants", default="f,i")
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/benchmark_125m_wikipedia/validation/manifest.json"),
    )
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--max-tokens", type=int, default=10_000_000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()

    variants = [item.strip().lower() for item in args.variants.split(",") if item.strip()]
    seeds = [int(item) for item in args.seeds.split(",") if item.strip()]
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device_name == "auto":
        device_name = "cpu"
    device = torch.device(device_name)
    rows: list[dict[str, Any]] = []

    with MemmapTokenDataset(args.manifest, split="validation") as dataset:
        for variant in variants:
            for seed in seeds:
                run_dir = args.root / f"variant_{variant}_seed_{seed}"
                checkpoint = run_dir / "checkpoint_best.pt"
                if not checkpoint.is_file():
                    raise FileNotFoundError(checkpoint)
                print(f"rescoring variant={variant.upper()} seed={seed}", flush=True)
                model = load_model(checkpoint).to(device).eval()
                started = time.perf_counter()
                ce, tokens, batches = evaluate_sequential(
                    model,
                    dataset,
                    "drm",
                    args.seq_len,
                    args.max_tokens,
                    args.batch_size,
                    device,
                )
                elapsed = time.perf_counter() - started
                row = {
                    "variant": variant.upper(),
                    "seed": seed,
                    "checkpoint": str(checkpoint),
                    "checkpoint_sha256": sha256_file(checkpoint),
                    "validation_ce": ce,
                    "validation_ppl": math.exp(min(ce, 20.0)),
                    "validation_tokens": tokens,
                    "batches": batches,
                    "elapsed_sec": elapsed,
                    "tokens_per_sec": tokens / elapsed,
                }
                save_json(run_dir / "validation_full.json", row)
                rows.append(row)
                print(json.dumps(row, indent=2), flush=True)
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    aggregate = []
    for variant in variants:
        values = [row["validation_ce"] for row in rows if row["variant"] == variant.upper()]
        aggregate.append(
            {
                "variant": variant.upper(),
                "seeds": len(values),
                "validation_ce_mean": statistics.mean(values),
                "validation_ce_std": statistics.pstdev(values),
                "validation_ce_min": min(values),
                "validation_ce_max": max(values),
            }
        )
    result = {
        "manifest": str(args.manifest),
        "manifest_sha256": sha256_file(args.manifest),
        "split": "validation",
        "commit": current_commit(),
        "source_worktree_may_be_dirty": True,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device": str(device),
        "runs": rows,
        "aggregate": aggregate,
    }
    save_json(args.root / "paired_validation_summary.json", result)
    print(json.dumps({"aggregate": aggregate}, indent=2), flush=True)


if __name__ == "__main__":
    main()
