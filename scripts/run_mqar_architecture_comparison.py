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
    "ASM_C_PRETRAINED",
    "ASM_C_MEMORY_2X",
    "ASM_C2_16",
    "ASM_C2_32",
    "ASM_C2_64",
    "ASM_C2_NOREAD",
    "ASM_C2_NOWRITE",
    "ASM_C2_SHUFFLED",
    "ASM_C2_SPARSE_16",
    "ASM_C2_SPARSE_32",
    "ASM_C2_SPARSE_64",
    "ASM_C2_FW",
    "ASM_C2_FW_NOREAD",
    "ASM_C2_FW_NOWRITE",
    "ASM_C2_FW_SHUFFLED",
    "ASM_C2_FW_DURABLE",
    "ASM_C2_FW_DURABLE_NOREAD",
    "ASM_C2_FW_DURABLE_NOWRITE",
    "ASM_C2_FW_DURABLE_SHUFFLED",
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


def load_asm_c_with_wider_memory(
    path: Path,
    multiplier: int = 2,
) -> tuple[DRMEmitterModel, list[str]]:
    """Reuse pretrained ASM-R weights while widening and resetting only memory."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    config = DRMConfig.from_dict(payload["config"])
    config.compact_streaming_inference = True
    config.selective_memory_hidden_size *= multiplier
    model = DRMEmitterModel(config.validated_copy())
    shared = {
        key: value
        for key, value in payload["model"].items()
        if not key.startswith("selective_memory.")
    }
    incompatible = model.load_state_dict(shared, strict=False)
    missing = sorted(incompatible.missing_keys)
    unexpected = sorted(incompatible.unexpected_keys)
    if unexpected or any(not key.startswith("selective_memory.") for key in missing):
        raise ValueError(
            f"unexpected wider-memory migration mismatch: missing={missing}, "
            f"unexpected={unexpected}"
        )
    return model, missing


def load_asm_c2(
    path: Path,
    *,
    slots: int,
    read_enabled: bool = True,
    write_enabled: bool = True,
    shuffle_on_eval: bool = False,
    sparse: bool = False,
    backend: str = "slots",
    durable: bool = False,
) -> tuple[DRMEmitterModel, list[str]]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    config = DRMConfig.from_dict(payload["config"])
    config.compact_streaming_inference = True
    config.addressable_memory = True
    config.addressable_memory_backend = backend
    config.fast_weight_durable_memory = durable
    config.fast_weight_state_fp32 = durable
    config.fast_weight_compute_fp32 = durable
    config.fast_weight_hard_write_threshold = 0.5 if durable else 0.0
    config.addressable_memory_slots = slots
    config.addressable_memory_read_enabled = read_enabled
    config.addressable_memory_write_enabled = write_enabled
    config.addressable_memory_shuffle_on_eval = shuffle_on_eval
    if sparse:
        config.addressable_memory_temperature = 0.25
        config.addressable_memory_read_top_k = min(2, slots)
        config.addressable_memory_write_top_k = 1
        config.addressable_memory_use_previous_token_key = True
        config.lambda_addressable_read_entropy = 0.001
        config.lambda_addressable_write_entropy = 0.001
    model = DRMEmitterModel(config.validated_copy())
    incompatible = model.load_state_dict(payload["model"], strict=False)
    missing = sorted(incompatible.missing_keys)
    unexpected = sorted(incompatible.unexpected_keys)
    if unexpected or any(not key.startswith("addressable_memory.") for key in missing):
        raise ValueError(
            f"unexpected ASM-C2 migration mismatch: missing={missing}, "
            f"unexpected={unexpected}"
        )
    return model, missing


def build_variant(
    name: str,
    asm_r_checkpoint: Path,
    asm_s_checkpoint: Path,
    transformer_checkpoint: Path | None,
    asm_r_config: dict[str, Any],
    max_seq_len: int,
    asm_c2_ablation_slots: int,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    if name == "ASM_C_PRETRAINED":
        model = load_model(asm_r_checkpoint)
        model.config.compact_streaming_inference = True
        metadata = {
            "initialization": "Wikipedia 100M ASM-R checkpoint",
            "geometry": True,
            "selective_memory": True,
            "compact_streaming": True,
            "memory_width_multiplier": 1,
        }
    elif name == "ASM_C_MEMORY_2X":
        model, reset = load_asm_c_with_wider_memory(asm_r_checkpoint, multiplier=2)
        metadata = {
            "initialization": "Wikipedia 100M shared weights; selective memory reset",
            "geometry": True,
            "selective_memory": True,
            "compact_streaming": True,
            "memory_width_multiplier": 2,
            "reset_keys": reset,
        }
    elif name.startswith("ASM_C2_"):
        specification = {
            "ASM_C2_16": (16, True, True),
            "ASM_C2_32": (32, True, True),
            "ASM_C2_64": (64, True, True),
            "ASM_C2_NOREAD": (asm_c2_ablation_slots, False, True, False),
            "ASM_C2_NOWRITE": (asm_c2_ablation_slots, True, False, False),
            "ASM_C2_SHUFFLED": (asm_c2_ablation_slots, True, True, True),
            "ASM_C2_SPARSE_16": (16, True, True, False, True),
            "ASM_C2_SPARSE_32": (32, True, True, False, True),
            "ASM_C2_SPARSE_64": (64, True, True, False, True),
            "ASM_C2_FW": (32, True, True, False, False, "fast_weight"),
            "ASM_C2_FW_NOREAD": (32, False, True, False, False, "fast_weight"),
            "ASM_C2_FW_NOWRITE": (32, True, False, False, False, "fast_weight"),
            "ASM_C2_FW_SHUFFLED": (32, True, True, True, False, "fast_weight"),
            "ASM_C2_FW_DURABLE": (32, True, True, False, False, "fast_weight", True),
            "ASM_C2_FW_DURABLE_NOREAD": (32, False, True, False, False, "fast_weight", True),
            "ASM_C2_FW_DURABLE_NOWRITE": (32, True, False, False, False, "fast_weight", True),
            "ASM_C2_FW_DURABLE_SHUFFLED": (32, True, True, True, False, "fast_weight", True),
        }[name]
        if len(specification) == 3:
            slots, read_enabled, write_enabled = specification
            shuffle_on_eval = False
            sparse = False
        elif len(specification) == 4:
            slots, read_enabled, write_enabled, shuffle_on_eval = specification
            sparse = False
        else:
            slots, read_enabled, write_enabled, shuffle_on_eval, sparse, *backend_values = specification
        backend = backend_values[0] if 'backend_values' in locals() and backend_values else "slots"
        durable = bool(backend_values[1]) if 'backend_values' in locals() and len(backend_values) > 1 else False
        model, initialized = load_asm_c2(
            asm_r_checkpoint,
            slots=slots,
            read_enabled=read_enabled,
            write_enabled=write_enabled,
            shuffle_on_eval=shuffle_on_eval,
            sparse=sparse,
            backend=backend,
            durable=durable,
        )
        metadata = {
            "initialization": "Wikipedia 100M ASM-R weights; addressable memory new",
            "geometry": True,
            "selective_memory": True,
            "compact_streaming": True,
            "addressable_memory": True,
            "addressable_slots": slots,
            "addressable_read_enabled": read_enabled,
            "addressable_write_enabled": write_enabled,
            "addressable_shuffle_on_eval": shuffle_on_eval,
            "addressable_sparse": sparse,
            "addressable_backend": backend,
            "fast_weight_durable": durable,
            "initialized_keys": initialized,
        }
    elif name == "ASM_R_PRETRAINED":
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
    short_gate = {
        result["variant"]: {
            "passed": result["rows"][-1]["validation_accuracy"] >= 0.8,
            "final_accuracy": result["rows"][-1]["validation_accuracy"],
            "first_passing_step": next(
                (row["step"] for row in result["rows"] if row["validation_accuracy"] >= 0.8),
                None,
            ),
        }
        for result in results
    }
    return {
        "final_ranking": ranking,
        "steps_to_accuracy": threshold_steps,
        "random_full_vocab_accuracy": 1.0 / 256,
        "random_value_set_accuracy": 1.0 / args.value_vocab_size,
        "short_control_gate": {"threshold": 0.8, "variants": short_gate},
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
- `ASM_C_MEMORY_2X` reuses all compatible pretrained weights but intentionally
  resets a selective-memory module with twice the hidden width.
- The 80% short-control gate must pass before any long-distance MQAR retention
  result is interpreted.
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
    parser.add_argument("--save-final-checkpoints", action="store_true")
    parser.add_argument("--asm-c2-ablation-slots", type=int, default=32)
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
            args.asm_c2_ablation_slots,
        )
        result = train_variant(name, model, metadata, args, milestones, device)
        results.append(result)
        if args.save_final_checkpoints and isinstance(model, DRMEmitterModel):
            checkpoint_path = args.output_root / f"{name.lower()}_final.pt"
            torch.save(model.state_dict_with_config(), checkpoint_path)
            result["final_checkpoint"] = str(checkpoint_path)
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
