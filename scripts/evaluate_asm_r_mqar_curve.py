from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from drm_language_emitter.checkpoint import load_model
from drm_language_emitter.mqar import make_mqar_batch
from scripts.evaluate_asm_r_checkpoint import checkpoint_audit, evaluate_mqar, precision_context


def parse_milestones(raw: str) -> list[int]:
    values = sorted({int(value.strip()) for value in raw.split(",") if value.strip()})
    if not values or values[0] <= 0:
        raise ValueError("milestones must contain positive integers")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure supervised MQAR adaptation of an ASM-R checkpoint."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--milestones", default="200,500,1000,2000,5000")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-batches", type=int, default=64)
    parser.add_argument("--n-pairs", type=int, default=8)
    parser.add_argument("--n-queries", type=int, default=8)
    parser.add_argument("--key-vocab-size", type=int, default=32)
    parser.add_argument("--value-vocab-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    milestones = parse_milestones(args.milestones)
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    if not 0 < args.n_queries <= args.n_pairs <= args.key_vocab_size:
        raise ValueError("require 0 < n-queries <= n-pairs <= key-vocab-size")
    if min(args.batch_size, args.eval_batches, args.value_vocab_size) <= 0:
        raise ValueError("batch sizes and vocabulary sizes must be positive")

    device = torch.device(args.device)
    audit = checkpoint_audit(args.checkpoint)
    model = load_model(args.checkpoint).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    train_generator = torch.Generator().manual_seed(args.seed)
    validation_seed = args.seed + 100_000
    rows = []

    def evaluate(step: int, train_ce: float | None, elapsed: float) -> None:
        metrics = evaluate_mqar(
            model,
            args.eval_batches,
            args.batch_size,
            args.n_pairs,
            args.n_queries,
            validation_seed,
            device,
            args.precision,
            args.key_vocab_size,
            args.value_vocab_size,
        )
        row = {
            "step": step,
            "examples_seen": step * args.batch_size,
            "query_targets_seen": step * args.batch_size * args.n_queries,
            "last_train_ce": train_ce,
            "validation_ce": metrics["ce"],
            "validation_accuracy": metrics["accuracy"],
            "validation_targets": metrics["targets"],
            "training_elapsed_sec": elapsed,
        }
        rows.append(row)
        print(json.dumps(row), flush=True)

    evaluate(0, None, 0.0)
    model.train()
    started = perf_counter()
    last_loss = None
    milestone_set = set(milestones)
    for step in range(1, milestones[-1] + 1):
        x, y, mask = make_mqar_batch(
            args.batch_size,
            args.n_pairs,
            args.n_queries,
            args.key_vocab_size,
            args.value_vocab_size,
            train_generator,
            device,
        )
        with precision_context(device, args.precision):
            logits = model(x, collect_diagnostics=False)["logits"]
        loss = F.cross_entropy(logits[mask].float(), y[mask])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), args.max_grad_norm, error_if_nonfinite=True
        )
        optimizer.step()
        last_loss = float(loss.detach())
        if step in milestone_set:
            evaluate(step, last_loss, perf_counter() - started)
            model.train()

    payload = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_audit": audit,
        "interpretation": "continuous supervised MQAR adaptation curve; not zero-shot corpus recall",
        "protocol": {
            "milestones": milestones,
            "batch_size": args.batch_size,
            "eval_batches": args.eval_batches,
            "n_pairs": args.n_pairs,
            "n_queries": args.n_queries,
            "key_vocab_size": args.key_vocab_size,
            "value_vocab_size": args.value_vocab_size,
            "random_full_vocab_accuracy": 1.0 / model.config.vocab_size,
            "random_value_set_accuracy": 1.0 / args.value_vocab_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "precision": args.precision,
            "device": str(device),
            "seed": args.seed,
        },
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"saved={args.output}")


if __name__ == "__main__":
    main()
