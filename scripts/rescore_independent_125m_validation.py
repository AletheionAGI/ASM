from __future__ import annotations

import argparse
import json
import math
import platform
import time
from pathlib import Path
from typing import Any

import torch

from drm_language_emitter.checkpoint import load_model
from drm_language_emitter.data import MemmapTokenDataset
from drm_language_emitter.utils import save_json
from evaluate_frozen_test import current_commit, evaluate_sequential, load_gpt2, sha256_file


def checkpoint_metadata(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError(f"invalid checkpoint payload: {path}")
    return {
        "path": str(path),
        "step": int(payload["step"]),
        "tokens_seen": int(payload["tokens_seen"]),
        "online_best_val_ce": float(payload["best_val_ce"]),
    }


def candidate_checkpoints(run_dir: Path) -> list[dict[str, Any]]:
    paths = [run_dir / "checkpoint_best.pt", *sorted(run_dir.glob("checkpoint_tokens_*.pt"))]
    candidates: list[dict[str, Any]] = []
    seen_training_states: set[tuple[int, int]] = set()
    for path in paths:
        if not path.is_file():
            continue
        metadata = checkpoint_metadata(path)
        state = (metadata["step"], metadata["tokens_seen"])
        if state in seen_training_states:
            continue
        seen_training_states.add(state)
        candidates.append(metadata)
    if not candidates:
        raise FileNotFoundError(f"no candidate checkpoints found in {run_dir}")
    return candidates


def discover_runs(root: Path) -> list[tuple[str, int, Path]]:
    runs: list[tuple[str, int, Path]] = []
    for family, pattern in (
        ("drm", "drm_125m_real_local_mixer_seed_*"),
        ("gpt2", "gpt2_125m_real_next_token_seed_*"),
    ):
        for run_dir in sorted(root.glob(pattern)):
            seed = int(run_dir.name.rsplit("_", 1)[-1])
            runs.append((family, seed, run_dir))
    if len(runs) != 6:
        raise ValueError(f"expected six runs, found {len(runs)} in {root}")
    return runs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Deterministically rescore independent 125M checkpoints on validation only."
    )
    parser.add_argument("--root", type=Path, default=Path("runs/independent_125m_frozen"))
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/benchmark_125m_wikipedia/validation/manifest.json"),
    )
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--max-tokens", type=int, default=10_000_000)
    parser.add_argument("--drm-batch-size", type=int, default=4)
    parser.add_argument("--gpt2-batch-size", type=int, default=8)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()

    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device_name == "auto":
        device_name = "cpu"
    device = torch.device(device_name)
    output_dir = args.root / "validation_rescore"
    output_dir.mkdir(parents=True, exist_ok=True)

    with MemmapTokenDataset(args.manifest, split="validation") as dataset:
        validation_tokens = min(len(dataset) - 1, args.max_tokens)
        all_results: list[dict[str, Any]] = []
        selections: list[dict[str, Any]] = []
        for family, seed, run_dir in discover_runs(args.root):
            run_results: list[dict[str, Any]] = []
            candidates = candidate_checkpoints(run_dir)
            for index, metadata in enumerate(candidates, start=1):
                checkpoint = Path(metadata["path"])
                print(
                    f"[{family} seed {seed}] candidate {index}/{len(candidates)}: "
                    f"{checkpoint.name} ({metadata['tokens_seen']} tokens)",
                    flush=True,
                )
                model = (load_model(checkpoint) if family == "drm" else load_gpt2(checkpoint)).to(device)
                model.eval()
                started = time.perf_counter()
                ce, tokens, batches = evaluate_sequential(
                    model,
                    dataset,
                    family,
                    args.seq_len,
                    args.max_tokens,
                    args.drm_batch_size if family == "drm" else args.gpt2_batch_size,
                    device,
                )
                elapsed = time.perf_counter() - started
                result = {
                    "family": family,
                    "seed": seed,
                    **metadata,
                    "checkpoint_sha256": sha256_file(checkpoint),
                    "validation_ce": ce,
                    "validation_ppl": math.exp(min(ce, 20.0)),
                    "validation_tokens": tokens,
                    "batches": batches,
                    "elapsed_sec": elapsed,
                    "tokens_per_sec": tokens / elapsed,
                }
                print(json.dumps(result, indent=2), flush=True)
                run_results.append(result)
                all_results.append(result)
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            selected = min(run_results, key=lambda item: (item["validation_ce"], item["tokens_seen"]))
            selections.append(selected)
            save_json(output_dir / f"{family}_seed_{seed}.json", {"candidates": run_results, "selected": selected})

    selection = {
        "status": "selected_by_validation",
        "external_test_accessed": False,
        "selection_rule": "minimum deterministic sequential validation CE per seed",
        "manifest": str(args.manifest),
        "manifest_sha256": sha256_file(args.manifest),
        "validation_tokens": validation_tokens,
        "seq_len": args.seq_len,
        "device": str(device),
        "commit": current_commit(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "selected": selections,
        "all_candidates": all_results,
    }
    save_json(output_dir / "selection.json", selection)
    print(f"saved={output_dir / 'selection.json'}", flush=True)
    print("PG-19 was not accessed.", flush=True)


if __name__ == "__main__":
    main()
