from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from drm_language_emitter.model import DRMEmitterModel
from drm_language_emitter.mqar import make_mqar_batch
from drm_language_emitter.utils import save_json
from run_drm_fix_ablation import parse_variants, resolve_variant


def probe_config(config, seed: int):
    config.vocab_size = 128
    config.d_token = 128
    config.d_state = 128
    config.n_directions = 16
    config.metric_rank = 8
    config.hidden_size = 256
    config.max_seq_len = 128
    config.direction_basis_size = 32
    config.metric_u_basis_size = 32
    config.directional_cumsum_block_size = 64
    config.directional_local_mixer_hidden_size = 64
    config.selective_memory_hidden_size = 64
    config.seed = seed
    return config.validated_copy()


def evaluate(
    model: DRMEmitterModel,
    batches: int,
    batch_size: int,
    generator: torch.Generator,
    device: torch.device,
    n_pairs: int,
    n_queries: int,
) -> tuple[float, float]:
    model.eval()
    losses = []
    correct = 0
    total = 0
    with torch.inference_mode():
        for _ in range(batches):
            x, y, mask = make_mqar_batch(
                batch_size,
                n_pairs,
                n_queries,
                32,
                64,
                generator,
                device,
            )
            logits = model(x, collect_diagnostics=False)["logits"]
            selected_logits = logits[mask]
            selected_targets = y[mask]
            losses.append(float(F.cross_entropy(selected_logits, selected_targets)))
            correct += int((selected_logits.argmax(dim=-1) == selected_targets).sum())
            total += selected_targets.numel()
    return sum(losses) / len(losses), correct / total


def main() -> None:
    parser = argparse.ArgumentParser(description="Train tiny DRM variants on MQAR.")
    parser.add_argument("--matrix", type=Path, default=Path("configs/drm_fix_ablation_variants.json"))
    parser.add_argument("--variants", default="F,I,J,J_DILATED")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--n-pairs", type=int, default=8)
    parser.add_argument("--n-queries", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output", type=Path, default=Path("runs/drm_fix_mqar/results.json"))
    args = parser.parse_args()

    matrix: dict[str, Any] = json.loads(args.matrix.read_text(encoding="utf-8"))
    names = parse_variants(args.variants, matrix["variants"])
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device_name == "auto":
        device_name = "cpu"
    device = torch.device(device_name)
    rows = []

    for name in names:
        config, description = resolve_variant(matrix, name)
        config = probe_config(config, args.seed)
        model = DRMEmitterModel(config).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
        train_generator = torch.Generator().manual_seed(args.seed)
        started = time.perf_counter()
        model.train()
        for step in range(1, args.steps + 1):
            x, y, mask = make_mqar_batch(
                args.batch_size,
                args.n_pairs,
                args.n_queries,
                32,
                64,
                train_generator,
                device,
            )
            logits = model(x, collect_diagnostics=False)["logits"]
            loss = F.cross_entropy(logits[mask], y[mask])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if step == 1 or step % 100 == 0:
                print(
                    f"variant={name} step={step} "
                    f"train_ce={float(loss.detach()):.6f}",
                    flush=True,
                )
        validation_ce, accuracy = evaluate(
            model,
            32,
            args.batch_size,
            torch.Generator().manual_seed(100_000 + args.seed),
            device,
            args.n_pairs,
            args.n_queries,
        )
        elapsed = time.perf_counter() - started
        row = {
            "variant": name,
            "description": description,
            "seed": args.seed,
            "steps": args.steps,
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "validation_ce": validation_ce,
            "accuracy": accuracy,
            "elapsed_sec": elapsed,
        }
        rows.append(row)
        print(json.dumps(row, indent=2), flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    save_json(args.output, {"results": rows})


if __name__ == "__main__":
    main()
