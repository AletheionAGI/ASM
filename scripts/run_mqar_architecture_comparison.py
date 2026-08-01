from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from drm_language_emitter.checkpoint import load_model
from drm_language_emitter.config import DRMConfig
from drm_language_emitter.model import DRMEmitterModel
from drm_language_emitter.mqar import make_mqar_batch
from scripts.evaluate_asm_r_checkpoint import precision_context, sha256_file
from scripts.evaluate_asm_r_mqar_curve import parse_milestones
from scripts.evaluate_frozen_test import load_gpt2


VARIANT_NAMES = (
    "ASM_R_PRETRAINED",
    "ASM_R_RANDOM",
    "ASM_R_NO_MEMORY",
    "ASM_S_PRETRAINED",
    "TRANSFORMER_PRETRAINED",
    "TRANSFORMER_RANDOM",
)


def finite_checkpoint_metadata(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or not isinstance(payload.get("model"), dict):
        raise ValueError(f"invalid checkpoint: {path}")
    invalid = sum(
        int((~torch.isfinite(value)).sum())
        for value in payload["model"].values()
        if torch.is_tensor(value) and value.is_floating_point()
    )
    if invalid:
        raise ValueError(f"checkpoint contains {invalid} non-finite values: {path}")
    return {"path": str(path), "sha256": sha256_file(path), "invalid_values": invalid}


def build_transformer(vocab_size: int, max_seq_len: int):
    try:
        from transformers import GPT2Config, GPT2LMHeadModel
    except ImportError as exc:
        raise SystemExit('Transformer control requires: pip install -e ".[hf]"') from exc
    config = GPT2Config(
        vocab_size=vocab_size,
        n_positions=max_seq_len,
        n_ctx=max_seq_len,
        n_embd=756,
        n_layer=12,
        n_head=12,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        bos_token_id=0,
        eos_token_id=0,
    )
    return GPT2LMHeadModel(config)


def load_asm_r_without_memory(path: Path) -> tuple[DRMEmitterModel, list[str]]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    config = DRMConfig.from_dict(payload["config"])
    config.selective_memory = False
    model = DRMEmitterModel(config.validated_copy())
    incompatible = model.load_state_dict(payload["model"], strict=False)
    unexpected = sorted(incompatible.unexpected_keys)
    missing = sorted(incompatible.missing_keys)
    if missing or any(not key.startswith("selective_memory.") for key in unexpected):
        raise ValueError(
            f"unexpected no-memory migration mismatch: missing={missing}, unexpected={unexpected}"
        )
    return model, unexpected


def build_variant(
    name: str,
    asm_r_checkpoint: Path,
    asm_s_checkpoint: Path,
    transformer_checkpoint: Path | None,
    asm_r_config: dict[str, Any],
    max_seq_len: int,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    if name == "ASM_R_PRETRAINED":
        model = load_model(asm_r_checkpoint)
        metadata = {"initialization": "Wikipedia 100M checkpoint", "geometry": True, "selective_memory": True}
    elif name == "ASM_R_RANDOM":
        model = DRMEmitterModel(DRMConfig.from_dict(asm_r_config))
        metadata = {"initialization": "random canonical ASM-R", "geometry": True, "selective_memory": True}
    elif name == "ASM_R_NO_MEMORY":
        model, removed = load_asm_r_without_memory(asm_r_checkpoint)
        metadata = {"initialization": "Wikipedia 100M shared weights", "geometry": True, "selective_memory": False, "removed_keys": removed}
    elif name == "ASM_S_PRETRAINED":
        model = load_model(asm_s_checkpoint)
        metadata = {"initialization": "Wikipedia 100M checkpoint", "geometry": False, "selective_memory": True}
    elif name == "TRANSFORMER_PRETRAINED":
        if transformer_checkpoint is None:
            raise ValueError("TRANSFORMER_PRETRAINED requires --transformer-checkpoint")
        model = load_gpt2(transformer_checkpoint)
        metadata = {"initialization": "Wikipedia 100M checkpoint", "geometry": False, "selective_memory": False, "attention": True}
    elif name == "TRANSFORMER_RANDOM":
        model = build_transformer(int(asm_r_config["vocab_size"]), max_seq_len)
        metadata = {"initialization": "random", "geometry": False, "selective_memory": False, "attention": True}
    else:
        raise ValueError(f"unknown variant {name}")
    metadata["parameter_count"] = sum(parameter.numel() for parameter in model.parameters())
    return model, metadata


def model_logits(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    output = model(x, collect_diagnostics=False) if isinstance(model, DRMEmitterModel) else model(input_ids=x)
    return output["logits"] if isinstance(output, dict) else output.logits


@torch.inference_mode()
def evaluate(
    model: torch.nn.Module,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, float | int]:
    model.eval()
    generator = torch.Generator().manual_seed(args.seed + 100_000)
    losses = []
    correct = 0
    total = 0
    for _ in range(args.eval_batches):
        x, y, mask = make_mqar_batch(
            args.batch_size,
            args.n_pairs,
            args.n_queries,
            args.key_vocab_size,
            args.value_vocab_size,
            generator,
            device,
        )
        with precision_context(device, args.precision):
            selected = model_logits(model, x)[mask]
        targets = y[mask]
        losses.append(float(F.cross_entropy(selected.float(), targets)))
        correct += int((selected.argmax(dim=-1) == targets).sum())
        total += targets.numel()
    return {"validation_ce": sum(losses) / len(losses), "validation_accuracy": correct / total, "validation_targets": total}


def train_variant(
    name: str,
    model: torch.nn.Module,
    metadata: dict[str, Any],
    args: argparse.Namespace,
    milestones: list[int],
    device: torch.device,
) -> dict[str, Any]:
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_generator = torch.Generator().manual_seed(args.seed)
    rows = [{"step": 0, "last_train_ce": None, "training_elapsed_sec": 0.0, **evaluate(model, args, device)}]
    print(json.dumps({"variant": name, **rows[-1]}), flush=True)
    model.train()
    started = perf_counter()
    milestone_set = set(milestones)
    last_loss = None
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
            logits = model_logits(model, x)
        loss = F.cross_entropy(logits[mask].float(), y[mask])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm, error_if_nonfinite=True)
        optimizer.step()
        last_loss = float(loss.detach())
        if step in milestone_set:
            row = {
                "step": step,
                "examples_seen": step * args.batch_size,
                "query_targets_seen": step * args.batch_size * args.n_queries,
                "last_train_ce": last_loss,
                "training_elapsed_sec": perf_counter() - started,
                **evaluate(model, args, device),
            }
            rows.append(row)
            print(json.dumps({"variant": name, **row}), flush=True)
            model.train()
    return {"variant": name, **metadata, "rows": rows}


def summarize(results: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    thresholds = (0.5, 0.8, 0.9)
    ranking = sorted(
        (
            {"variant": result["variant"], **result["rows"][-1]}
            for result in results
        ),
        key=lambda row: (-row["validation_accuracy"], row["validation_ce"]),
    )
    threshold_steps = {}
    for result in results:
        threshold_steps[result["variant"]] = {
            str(threshold): next(
                (row["step"] for row in result["rows"] if row["validation_accuracy"] >= threshold),
                None,
            )
            for threshold in thresholds
        }
    return {
        "final_ranking": ranking,
        "steps_to_accuracy": threshold_steps,
        "random_full_vocab_accuracy": 1.0 / 256,
        "random_value_set_accuracy": 1.0 / args.value_vocab_size,
    }


def write_report(path: Path, payload: dict[str, Any]) -> None:
    rows = "\n".join(
        f"| {index} | {row['variant']} | {row['validation_accuracy']:.4%} | {row['validation_ce']:.6f} | {row['training_elapsed_sec']:.1f} s |"
        for index, row in enumerate(payload["summary"]["final_ranking"], 1)
    )
    text = f"""# MQAR architecture comparison

## Final ranking at {payload['protocol']['milestones'][-1]:,} steps

| Rank | Variant | Accuracy | CE | Training time |
|---:|---|---:|---:|---:|
{rows}

## Interpretation boundaries

- All variants receive the same continuous training batches and frozen validation batches.
- `ASM_R_NO_MEMORY` preserves compatible 100M pretrained weights and removes only selective-memory parameters.
- `TRANSFORMER_PRETRAINED`, when selected, uses the parameter-near 100M-token Wikipedia checkpoint.
- `TRANSFORMER_RANDOM` isolates the same architecture without language pretraining.
- This is supervised MQAR adaptation, not zero-shot corpus recall.
- Results measure sample efficiency on this synthetic task, not general language quality.
"""
    path.write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the decisive ASM MQAR architecture comparison.")
    parser.add_argument("--asm-r-checkpoint", type=Path, required=True)
    parser.add_argument("--asm-s-checkpoint", type=Path, required=True)
    parser.add_argument("--transformer-checkpoint", type=Path)
    parser.add_argument("--asm-r-config", type=Path, default=Path("configs/asm_r_125m.json"))
    parser.add_argument("--variants", default=",".join(VARIANT_NAMES))
    parser.add_argument("--milestones", default="200,500,1000,2000,5000,10000,20000")
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
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    milestones = parse_milestones(args.milestones)
    names = [value.strip().upper() for value in args.variants.split(",") if value.strip()]
    unknown = sorted(set(names) - set(VARIANT_NAMES))
    if unknown:
        raise ValueError(f"unknown variants: {', '.join(unknown)}")
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    r_checkpoint = finite_checkpoint_metadata(args.asm_r_checkpoint)
    s_checkpoint = finite_checkpoint_metadata(args.asm_s_checkpoint)
    transformer_checkpoint = (
        finite_checkpoint_metadata(args.transformer_checkpoint)
        if args.transformer_checkpoint is not None
        else None
    )
    config = json.loads(args.asm_r_config.read_text(encoding="utf-8"))
    sequence_len = 2 * args.n_pairs + 3 * args.n_queries - 1
    device = torch.device(args.device)
    args.output_root.mkdir(parents=True, exist_ok=True)
    results = []
    for name in names:
        model, metadata = build_variant(
            name,
            args.asm_r_checkpoint,
            args.asm_s_checkpoint,
            args.transformer_checkpoint,
            config,
            sequence_len,
        )
        result = train_variant(name, model, metadata, args, milestones, device)
        results.append(result)
        partial = {"results": results}
        (args.output_root / "partial.json").write_text(json.dumps(partial, indent=2) + "\n", encoding="utf-8")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    payload = {
        "protocol": {
            "milestones": milestones,
            "batch_size": args.batch_size,
            "eval_batches": args.eval_batches,
            "n_pairs": args.n_pairs,
            "n_queries": args.n_queries,
            "key_vocab_size": args.key_vocab_size,
            "value_vocab_size": args.value_vocab_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "seed": args.seed,
            "precision": args.precision,
            "device": str(device),
        },
        "checkpoints": {
            "asm_r": r_checkpoint,
            "asm_s": s_checkpoint,
            "transformer": transformer_checkpoint,
        },
        "results": results,
    }
    payload["summary"] = summarize(results, args)
    output = args.output_root / "results.json"
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    write_report(args.output_root / "report.md", payload)
    print(f"saved={output}")
    print(f"saved={args.output_root / 'report.md'}")


if __name__ == "__main__":
    main()
