from __future__ import annotations

import argparse
import json
import sys
from contextlib import nullcontext
from pathlib import Path
from time import perf_counter

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from drm_language_emitter.checkpoint import load_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare reference prefix recomputation with cached ASM decode."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt-tokens", type=int, default=128)
    parser.add_argument("--decode-tokens", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main() -> None:
    args = parse_args()
    if args.prompt_tokens <= 0 or args.decode_tokens <= 0 or args.batch_size <= 0:
        raise ValueError("prompt-tokens, decode-tokens, and batch-size must be positive")
    device = torch.device(args.device)
    model = load_model(args.checkpoint).to(device).eval()
    generator = torch.Generator(device=device).manual_seed(args.seed)
    prompt = torch.randint(
        0,
        model.config.vocab_size,
        (args.batch_size, args.prompt_tokens),
        generator=generator,
        device=device,
    )
    continuation = torch.randint(
        0,
        model.config.vocab_size,
        (args.batch_size, args.decode_tokens),
        generator=generator,
        device=device,
    )
    autocast = (
        torch.autocast(device.type, dtype=torch.bfloat16)
        if args.precision == "bf16"
        else nullcontext()
    )
    with torch.inference_mode(), autocast:
        model(prompt, collect_diagnostics=False)
        synchronize(device)

        prefix = prompt
        reference_logits = []
        started = perf_counter()
        for position in range(args.decode_tokens):
            prefix = torch.cat(
                [prefix, continuation[:, position : position + 1]],
                dim=1,
            )
            reference_logits.append(
                model(prefix, collect_diagnostics=False)["logits"][:, -1]
            )
        synchronize(device)
        reference_sec = perf_counter() - started

        _prompt_logits, state = model.prefill(prompt)
        cached_logits = []
        synchronize(device)
        started = perf_counter()
        for position in range(args.decode_tokens):
            logits, state = model.decode_step(continuation[:, position], state)
            cached_logits.append(logits)
        synchronize(device)
        cached_sec = perf_counter() - started

    error = (
        torch.stack(reference_logits).float()
        - torch.stack(cached_logits).float()
    ).abs()
    result = {
        "checkpoint": args.checkpoint,
        "device": str(device),
        "precision": args.precision,
        "batch_size": args.batch_size,
        "prompt_tokens": args.prompt_tokens,
        "decode_tokens": args.decode_tokens,
        "uses_block_cache": state.uses_block_cache,
        "block_size": state.block_size,
        "open_block_tokens": (
            int(state.block_tokens.shape[1])
            if state.block_tokens is not None
            else None
        ),
        "reference_sec": reference_sec,
        "cached_sec": cached_sec,
        "speedup": reference_sec / cached_sec,
        "reference_tokens_per_sec": args.decode_tokens / reference_sec,
        "cached_tokens_per_sec": args.decode_tokens / cached_sec,
        "max_abs_error": float(error.max()),
        "mean_abs_error": float(error.mean()),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
