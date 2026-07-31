from __future__ import annotations

import argparse
import json
import math
import platform
from pathlib import Path
from typing import Any

import torch

from drm_language_emitter.checkpoint import load_model
from drm_language_emitter.data import MemmapTokenDataset
from drm_language_emitter.utils import save_json
try:
    from scripts.evaluate_frozen_test import current_commit, evaluate_sequential, sha256_file
except ModuleNotFoundError:  # Direct execution adds scripts/ rather than the repo root.
    from evaluate_frozen_test import current_commit, evaluate_sequential, sha256_file


def parse_csv(raw: str, transform=str):
    return [transform(item.strip()) for item in raw.split(",") if item.strip()]


def nearest_elapsed(metrics_path: Path, milestone: int) -> float | None:
    if not metrics_path.is_file():
        return None
    history = json.loads(metrics_path.read_text(encoding="utf-8")).get("history", [])
    candidates = [row for row in history if int(row.get("tokens_seen", 0)) >= milestone]
    if not candidates:
        return None
    return float(min(candidates, key=lambda row: int(row["tokens_seen"]))["elapsed_sec"])


def fit_power_law(rows: list[dict[str, Any]]) -> dict[str, float] | None:
    if len(rows) < 4:
        return None
    xs = [math.log(float(row["milestone_tokens"])) for row in rows]
    losses = [float(row["validation_ce"]) for row in rows]
    minimum = min(losses)
    best: tuple[float, float, float, float] | None = None
    for index in range(1, 2001):
        asymptote = max(0.0, minimum - 1.0) + (minimum - 1e-6 - max(0.0, minimum - 1.0)) * index / 2001
        residuals = [loss - asymptote for loss in losses]
        if min(residuals) <= 0:
            continue
        ys = [math.log(value) for value in residuals]
        mean_x = sum(xs) / len(xs)
        mean_y = sum(ys) / len(ys)
        denominator = sum((x - mean_x) ** 2 for x in xs)
        slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / denominator
        alpha = -slope
        if alpha <= 0:
            continue
        log_a = mean_y + alpha * mean_x
        coefficient = math.exp(log_a)
        error = sum(
            (loss - (asymptote + coefficient * tokens ** (-alpha))) ** 2
            for loss, tokens in zip(losses, (math.exp(x) for x in xs))
        )
        candidate = (error, asymptote, coefficient, alpha)
        if best is None or candidate < best:
            best = candidate
    if best is None:
        return None
    error, asymptote, coefficient, alpha = best
    return {
        "loss_asymptote": asymptote,
        "coefficient": coefficient,
        "alpha": alpha,
        "squared_error": error,
    }


def observed_crossovers(rows: list[dict[str, Any]], variants: list[str]) -> list[dict[str, Any]]:
    lookup = {
        (row["variant"], int(row["milestone_tokens"])): float(row["validation_ce"])
        for row in rows
    }
    milestones = sorted({int(row["milestone_tokens"]) for row in rows})
    results = []
    for left_index, left in enumerate(variants):
        for right in variants[left_index + 1 :]:
            previous = None
            for milestone in milestones:
                if (left, milestone) not in lookup or (right, milestone) not in lookup:
                    continue
                difference = lookup[(left, milestone)] - lookup[(right, milestone)]
                if previous is not None and difference * previous[1] < 0:
                    x0, d0 = previous
                    fraction = abs(d0) / (abs(d0) + abs(difference))
                    estimate = math.exp(math.log(x0) + fraction * (math.log(milestone) - math.log(x0)))
                    results.append(
                        {
                            "left": left,
                            "right": right,
                            "between_tokens": [x0, milestone],
                            "estimated_crossover_tokens": estimate,
                        }
                    )
                previous = (milestone, difference)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Frozen rescoring and curve fitting for DRM scaling runs.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--variants", required=True)
    parser.add_argument("--milestones", default="1000000,2000000,5000000,10000000,20000000,30000000,50000000,100000000")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--manifest", type=Path, default=Path("data/benchmark_125m_wikipedia/validation/manifest.json"))
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--max-tokens", type=int, default=10_000_000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()

    variants = [item.upper() for item in parse_csv(args.variants)]
    milestones = parse_csv(args.milestones, int)
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device_name == "auto":
        device_name = "cpu"
    device = torch.device(device_name)
    rows: list[dict[str, Any]] = []

    with MemmapTokenDataset(args.manifest, split="validation") as dataset:
        for variant in variants:
            run_dir = args.root / f"variant_{variant.lower()}_seed_{args.seed}"
            for milestone in milestones:
                checkpoint = run_dir / f"checkpoint_milestone_{milestone}.pt"
                if not checkpoint.is_file():
                    raise FileNotFoundError(checkpoint)
                print(f"rescoring variant={variant} milestone={milestone}", flush=True)
                model = load_model(checkpoint).to(device).eval()
                ce, tokens, batches = evaluate_sequential(
                    model,
                    dataset,
                    "drm",
                    args.seq_len,
                    args.max_tokens,
                    args.batch_size,
                    device,
                )
                elapsed = nearest_elapsed(run_dir / "metrics_latest.json", milestone)
                row = {
                    "variant": variant,
                    "seed": args.seed,
                    "milestone_tokens": milestone,
                    "checkpoint": str(checkpoint),
                    "checkpoint_sha256": sha256_file(checkpoint),
                    "validation_ce": ce,
                    "validation_ppl": math.exp(min(ce, 20.0)),
                    "validation_tokens": tokens,
                    "validation_batches": batches,
                    "training_elapsed_sec": elapsed,
                    "training_gpu_hours": elapsed / 3600 if elapsed is not None else None,
                }
                rows.append(row)
                print(json.dumps(row, indent=2), flush=True)
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    fits = {}
    for variant in variants:
        fits[variant] = fit_power_law([row for row in rows if row["variant"] == variant])
    result = {
        "manifest": str(args.manifest),
        "manifest_sha256": sha256_file(args.manifest),
        "commit": current_commit(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device": str(device),
        "rows": rows,
        "power_law_fits": fits,
        "observed_crossovers": observed_crossovers(rows, variants),
    }
    save_json(args.root / "scaling_law_summary.json", result)
    print(json.dumps({"power_law_fits": fits, "observed_crossovers": result["observed_crossovers"]}, indent=2))


if __name__ == "__main__":
    main()
