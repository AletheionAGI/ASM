from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from drm_language_emitter.checkpoint import load_model
from drm_language_emitter.data import MemmapTokenDataset
from drm_language_emitter.utils import save_json


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_gpt2(checkpoint: Path) -> torch.nn.Module:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or not isinstance(payload.get("config"), dict):
        raise ValueError("GPT-2 checkpoint must contain a config dictionary")
    try:
        from transformers import GPT2Config, GPT2LMHeadModel
    except ImportError as exc:
        raise SystemExit('GPT-2 evaluation requires: pip install -e ".[hf]"') from exc
    data: dict[str, Any] = payload["config"]
    config = GPT2Config(
        vocab_size=int(data.get("vocab_size", 256)),
        n_positions=int(data.get("max_seq_len", 512)),
        n_ctx=int(data.get("max_seq_len", 512)),
        n_embd=int(data["n_embd"]),
        n_layer=int(data["n_layer"]),
        n_head=int(data["n_head"]),
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        bos_token_id=0,
        eos_token_id=0,
    )
    model = GPT2LMHeadModel(config)
    model.load_state_dict(payload["model"])
    return model


def current_commit() -> str | None:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        return f"{commit}-dirty" if dirty else commit
    except (OSError, subprocess.CalledProcessError):
        return None


@torch.inference_mode()
def evaluate_sequential(
    model: torch.nn.Module,
    dataset: MemmapTokenDataset,
    family: str,
    seq_len: int,
    max_tokens: int,
    batch_size: int,
    device: torch.device,
) -> tuple[float, int, int]:
    if seq_len <= 0 or batch_size <= 0 or max_tokens <= 0:
        raise ValueError("seq_len, batch_size, and max_tokens must be positive")
    token_limit = min(len(dataset) - 1, max_tokens)
    windows = [
        (start, min(seq_len, len(dataset) - start - 1, token_limit - start))
        for start in range(0, token_limit, seq_len)
    ]
    total_loss = 0.0
    total_tokens = 0
    batches = 0
    for length in sorted({length for _, length in windows}, reverse=True):
        equal_length_starts = [start for start, window_length in windows if window_length == length]
        for batch_start in range(0, len(equal_length_starts), batch_size):
            rows = [
                dataset.window(start, length)
                for start in equal_length_starts[batch_start : batch_start + batch_size]
            ]
            x = torch.stack([row[0] for row in rows]).to(device)
            y = torch.stack([row[1] for row in rows]).to(device)
            if family == "drm":
                logits = model(x, collect_diagnostics=False)["logits"]
            else:
                logits = model(input_ids=x).logits
            total_loss += float(
                F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    y.reshape(-1),
                    reduction="sum",
                )
            )
            total_tokens += y.numel()
            batches += 1
    return total_loss / total_tokens, total_tokens, batches


def main() -> None:
    parser = argparse.ArgumentParser(description="Deterministic frozen-test evaluation.")
    parser.add_argument("--family", choices=["drm", "gpt2"], required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--max-tokens", type=int, default=10_000_000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device_name == "auto":
        device_name = "cpu"
    device = torch.device(device_name)
    model = (load_model(args.checkpoint) if args.family == "drm" else load_gpt2(args.checkpoint)).to(device)
    model.eval()
    with MemmapTokenDataset(args.manifest, split=args.split) as dataset:
        ce, tokens, batches = evaluate_sequential(
            model, dataset, args.family, args.seq_len, args.max_tokens, args.batch_size, device
        )
    result = {
        "family": args.family,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "manifest": str(args.manifest),
        "manifest_sha256": sha256_file(args.manifest),
        "split": args.split,
        "test_tokens": tokens,
        "batches": batches,
        "test_ce": ce,
        "test_ppl": math.exp(min(ce, 20.0)),
        "commit": current_commit(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device": str(device),
    }
    save_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
