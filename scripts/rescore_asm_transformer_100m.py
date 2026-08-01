from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from drm_language_emitter.data import MemmapTokenDataset
from scripts.evaluate_frozen_test import evaluate_sequential, load_gpt2, sha256_file
from scripts.rescore_drm_scaling_law import fit_power_law, observed_crossovers


def parse_csv_ints(value: str) -> list[int]:
    values = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    if not values or any(item <= 0 for item in values):
        raise ValueError("milestones must be positive")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description="Frozen milestone rescoring for paired ASM-R/Transformer 100M runs.")
    parser.add_argument("--asm-summary", type=Path, required=True)
    parser.add_argument("--transformer-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--milestones", default="1000000,2000000,5000000,10000000,20000000,30000000,50000000,100000000")
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--max-tokens", type=int, default=10_000_000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    milestones = parse_csv_ints(args.milestones)
    asm_payload = json.loads(args.asm_summary.read_text(encoding="utf-8"))
    asm_rows = [
        {**row, "variant": "ASM_R"}
        for row in asm_payload["rows"]
        if row["variant"] == "J_NO_DIRECTION" and int(row["milestone_tokens"]) in milestones
    ]
    metrics = json.loads((args.transformer_root / "metrics_latest.json").read_text(encoding="utf-8"))["history"]
    device = torch.device(args.device)
    transformer_rows = []
    with MemmapTokenDataset(args.manifest, split="validation") as dataset:
        for milestone in milestones:
            checkpoint = args.transformer_root / f"checkpoint_milestone_{milestone}.pt"
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            print(f"rescoring Transformer milestone={milestone}", flush=True)
            model = load_gpt2(checkpoint).to(device).eval()
            ce, tokens, batches = evaluate_sequential(model, dataset, "gpt2", args.seq_len, args.max_tokens, args.batch_size, device)
            nearest = min(metrics, key=lambda row: abs(int(row["tokens_seen"]) - milestone))
            transformer_rows.append({
                "variant": "TRANSFORMER",
                "seed": 1,
                "milestone_tokens": milestone,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": sha256_file(checkpoint),
                "validation_ce": ce,
                "validation_ppl": math.exp(min(ce, 20.0)),
                "validation_tokens": tokens,
                "validation_batches": batches,
                "training_elapsed_sec": float(nearest["elapsed_sec"]),
                "training_gpu_hours": float(nearest["elapsed_sec"]) / 3600,
            })
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
    rows = sorted(asm_rows + transformer_rows, key=lambda row: (row["variant"], row["milestone_tokens"]))
    fits = {variant: fit_power_law([row for row in rows if row["variant"] == variant]) for variant in ("ASM_R", "TRANSFORMER")}
    payload = {
        "protocol": {
            "manifest": str(args.manifest),
            "manifest_sha256": sha256_file(args.manifest),
            "validation_tokens": transformer_rows[0]["validation_tokens"],
            "milestones": milestones,
            "seq_len": args.seq_len,
            "device": str(device),
        },
        "rows": rows,
        "power_law_fits": fits,
        "observed_crossovers": observed_crossovers(rows, ["ASM_R", "TRANSFORMER"]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"saved={args.output}")


if __name__ == "__main__":
    main()
