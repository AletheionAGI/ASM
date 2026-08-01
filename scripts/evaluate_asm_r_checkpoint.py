from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
from contextlib import nullcontext
from pathlib import Path
from time import perf_counter
from typing import Any

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from drm_language_emitter.checkpoint import load_model
from drm_language_emitter.data import MemmapTokenDataset
from drm_language_emitter.generation import generate
from drm_language_emitter.mqar import make_mqar_batch
from drm_language_emitter.tokenizer import ByteTokenizer


DEFAULT_PROMPTS = (
    "Aletheion State Models ",
    "The geometry of language ",
    "In a causal state model, memory ",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def precision_context(device: torch.device, precision: str):
    if precision == "bf16" and device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


def checkpoint_audit(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or not isinstance(payload.get("model"), dict):
        raise ValueError("checkpoint must contain a model state dictionary")
    config = payload.get("config")
    if not isinstance(config, dict):
        raise ValueError("checkpoint must contain a config dictionary")
    invalid_tensors = 0
    invalid_values = 0
    parameters = 0
    for value in payload["model"].values():
        if not torch.is_tensor(value):
            continue
        parameters += value.numel()
        invalid = (~torch.isfinite(value)).sum().item() if value.is_floating_point() else 0
        if invalid:
            invalid_tensors += 1
            invalid_values += int(invalid)
    if invalid_values:
        raise ValueError(
            f"checkpoint contains {invalid_values} non-finite values across "
            f"{invalid_tensors} tensors"
        )
    is_asm_r = (
        config.get("use_direction_field") is False
        and config.get("selective_memory") is True
        and config.get("use_drm_geometry", True) is True
    )
    if not is_asm_r:
        raise ValueError("checkpoint config is not the promoted ASM-R/J_NO_DIRECTION variant")
    return {
        "sha256": sha256_file(path),
        "parameter_count": parameters,
        "invalid_tensors": invalid_tensors,
        "invalid_values": invalid_values,
        "is_asm_r": is_asm_r,
        "sequence_mode": config.get("sequence_mode"),
        "training_max_seq_len": config.get("max_seq_len"),
    }


@torch.inference_mode()
def evaluate_context_lengths(
    model,
    manifest: Path,
    lengths: list[int],
    batches: int,
    device: torch.device,
    precision: str,
) -> list[dict[str, Any]]:
    rows = []
    with MemmapTokenDataset(manifest, split="validation") as dataset:
        for length in lengths:
            if length <= 0 or dataset.total_tokens < length + 1:
                raise ValueError(f"invalid context length {length}")
            losses = []
            tokens = 0
            started = perf_counter()
            max_start = dataset.total_tokens - length - 1
            for index in range(batches):
                start = 0 if batches == 1 else round(index * max_start / (batches - 1))
                x, y = dataset.window(start, length)
                x = x.unsqueeze(0).to(device)
                y = y.unsqueeze(0).to(device)
                with precision_context(device, precision):
                    logits = model(x, collect_diagnostics=False)["logits"]
                loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), y.reshape(-1))
                losses.append(float(loss))
                tokens += y.numel()
            elapsed = perf_counter() - started
            ce = sum(losses) / len(losses)
            rows.append({
                "context_length": length,
                "batches": batches,
                "tokens": tokens,
                "ce": ce,
                "ppl": math.exp(min(ce, 50.0)),
                "elapsed_sec": elapsed,
                "tokens_per_sec": tokens / elapsed,
                "beyond_training_context": length > model.config.max_seq_len,
            })
    return rows


@torch.inference_mode()
def generate_samples(model, device: torch.device, precision: str, max_new_tokens: int, seed: int) -> list[dict[str, str]]:
    tokenizer = ByteTokenizer()
    rows = []
    for index, prompt in enumerate(DEFAULT_PROMPTS):
        torch.manual_seed(seed + index)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed + index)
        ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
        with precision_context(device, precision):
            output = generate(model, ids, max_new_tokens=max_new_tokens, temperature=0.8, top_k=40)
        rows.append({"prompt": prompt, "text": tokenizer.decode(output[0].tolist())})
    return rows


@torch.inference_mode()
def evaluate_mqar(
    model,
    batches: int,
    batch_size: int,
    n_pairs: int,
    n_queries: int,
    seed: int,
    device: torch.device,
    precision: str,
    key_vocab_size: int = 32,
    value_vocab_size: int = 64,
) -> dict[str, float]:
    losses = []
    correct = 0
    total = 0
    generator = torch.Generator().manual_seed(seed)
    model.eval()
    for _ in range(batches):
        x, y, mask = make_mqar_batch(
            batch_size,
            n_pairs,
            n_queries,
            key_vocab_size,
            value_vocab_size,
            generator,
            device,
        )
        with precision_context(device, precision):
            logits = model(x, collect_diagnostics=False)["logits"]
        selected = logits[mask]
        targets = y[mask]
        losses.append(float(F.cross_entropy(selected.float(), targets)))
        correct += int((selected.argmax(dim=-1) == targets).sum())
        total += targets.numel()
    return {"ce": sum(losses) / len(losses), "accuracy": correct / total, "targets": total}


def adapt_mqar(model, steps: int, batch_size: int, n_pairs: int, n_queries: int, seed: int, device: torch.device, precision: str) -> dict[str, Any]:
    before = evaluate_mqar(model, 16, batch_size, n_pairs, n_queries, seed + 100_000, device, precision)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    generator = torch.Generator().manual_seed(seed)
    model.train()
    started = perf_counter()
    final_loss = None
    for _step in range(steps):
        x, y, mask = make_mqar_batch(batch_size, n_pairs, n_queries, 32, 64, generator, device)
        with precision_context(device, precision):
            logits = model(x, collect_diagnostics=False)["logits"]
        loss = F.cross_entropy(logits[mask].float(), y[mask])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        final_loss = float(loss.detach())
    elapsed = perf_counter() - started
    after = evaluate_mqar(model, 32, batch_size, n_pairs, n_queries, seed + 200_000, device, precision)
    return {
        "interpretation": "supervised adaptation probe; not zero-shot Wikipedia recall",
        "steps": steps,
        "batch_size": batch_size,
        "n_pairs": n_pairs,
        "n_queries": n_queries,
        "before": before,
        "final_train_ce": final_loss,
        "after": after,
        "elapsed_sec": elapsed,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a promoted ASM-R checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--context-lengths", default="64,128,256,512,1024")
    parser.add_argument("--context-batches", type=int, default=4)
    parser.add_argument("--generation-tokens", type=int, default=64)
    parser.add_argument("--mqar-steps", type=int, default=20)
    parser.add_argument("--mqar-batch-size", type=int, default=2)
    parser.add_argument("--mqar-pairs", type=int, default=8)
    parser.add_argument("--mqar-queries", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    if min(args.context_batches, args.mqar_batch_size, args.mqar_pairs, args.mqar_queries) <= 0:
        raise ValueError("batch sizes, pairs, queries, and context batches must be positive")
    if args.mqar_steps < 0 or args.generation_tokens < 0:
        raise ValueError("mqar-steps and generation-tokens must be non-negative")
    lengths = sorted({int(value) for value in args.context_lengths.split(",") if value})
    device = torch.device(args.device)
    audit = checkpoint_audit(args.checkpoint)
    model = load_model(args.checkpoint).to(device).eval()
    result: dict[str, Any] = {
        "checkpoint": str(args.checkpoint),
        "manifest": str(args.manifest),
        "device": str(device),
        "precision": args.precision,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "audit": audit,
    }
    result["context"] = evaluate_context_lengths(model, args.manifest, lengths, args.context_batches, device, args.precision)
    result["generations"] = generate_samples(model, device, args.precision, args.generation_tokens, args.seed)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    adaptation_model = load_model(args.checkpoint).to(device)
    result["mqar"] = adapt_mqar(adaptation_model, args.mqar_steps, args.mqar_batch_size, args.mqar_pairs, args.mqar_queries, args.seed, device, args.precision)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
