from __future__ import annotations

import argparse
import json
import sys
from contextlib import nullcontext
from pathlib import Path
from time import perf_counter

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from drm_language_emitter.data import MemmapTokenDataset
from scripts.benchmark_asm_r_long_streaming import delayed_mqar_batch, wilson_interval
from scripts.run_mqar_architecture_comparison import load_asm_c2


def precision_context(device: torch.device):
    return torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()


def parse_curriculum(raw: str) -> list[tuple[int, int]]:
    stages = []
    for item in raw.split(","):
        length, steps = item.split(":", 1)
        stages.append((int(length), int(steps)))
    if not stages or any(length < 40 or steps <= 0 for length, steps in stages):
        raise ValueError("curriculum requires LENGTH:STEPS pairs with length >= 40")
    return stages


@torch.inference_mode()
def evaluate_lengths(model, lengths, args, device, seed_offset=0):
    model.eval()
    rows = []
    for length in lengths:
        generator = torch.Generator().manual_seed(args.seed + seed_offset + length)
        losses = []
        correct = total = 0
        batch_size = min(args.batch_size, 2 if length >= 512 else args.batch_size)
        for _ in range(args.eval_batches):
            x, y, mask = delayed_mqar_batch(
                length, batch_size, 8, 8, generator, device
            )
            with precision_context(device):
                selected = model(x, collect_diagnostics=False)["logits"][mask]
            target = y[mask]
            losses.append(float(F.cross_entropy(selected.float(), target)))
            correct += int((selected.argmax(-1) == target).sum())
            total += target.numel()
        low, high = wilson_interval(correct, total)
        rows.append({
            "sequence_length": length,
            "validation_ce": sum(losses) / len(losses),
            "validation_accuracy": correct / total,
            "targets": total,
            "accuracy_ci95_low": low,
            "accuracy_ci95_high": high,
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Train durable ASM-C2-FW with MQAR distance curriculum and language replay.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--curriculum", default="40:1000,80:500,160:500,320:300,512:200,1024:100,4096:25")
    parser.add_argument("--language-manifest", type=Path, default=Path("data/benchmark_125m_wikipedia/train/manifest.json"))
    parser.add_argument("--language-replay-probability", type=float, default=0.2)
    parser.add_argument("--language-seq-len", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-batches", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if not 0 <= args.language_replay_probability < 1:
        raise ValueError("language replay probability must be in [0, 1)")
    device = torch.device(args.device)
    stages = parse_curriculum(args.curriculum)
    model, initialized = load_asm_c2(
        args.checkpoint, slots=32, backend="fast_weight", durable=True
    )
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    mqar_generator = torch.Generator().manual_seed(args.seed)
    language_generator = torch.Generator().manual_seed(args.seed + 1)
    choice_generator = torch.Generator().manual_seed(args.seed + 2)
    args.output_root.mkdir(parents=True, exist_ok=True)
    history = []
    global_step = 0
    started = perf_counter()
    with MemmapTokenDataset(args.language_manifest, split="train", verify_integrity=False) as language:
        for length, stage_steps in stages:
            for _ in range(stage_steps):
                global_step += 1
                use_language = float(torch.rand((), generator=choice_generator)) < args.language_replay_probability
                model.train()
                if use_language:
                    x, y = language.make_batch(
                        1, args.language_seq_len, device, language_generator
                    )
                    with precision_context(device):
                        output = model(x, targets=y, collect_diagnostics=False)
                    loss = output["loss"]
                    task = "language"
                else:
                    batch_size = min(args.batch_size, 1 if length >= 1024 else 2 if length >= 512 else args.batch_size)
                    x, y, mask = delayed_mqar_batch(
                        length, batch_size, 8, 8, mqar_generator, device
                    )
                    with precision_context(device):
                        selected = model(x, collect_diagnostics=False)["logits"][mask]
                    loss = F.cross_entropy(selected.float(), y[mask])
                    task = "mqar"
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
                if global_step == 1 or global_step % 100 == 0:
                    print(json.dumps({
                        "step": global_step,
                        "stage_length": length,
                        "task": task,
                        "train_ce": float(loss.detach()),
                        "elapsed_sec": perf_counter() - started,
                    }), flush=True)
            evaluation = evaluate_lengths(model, [value for value, _ in stages if value <= length], args, device, 100_000)
            row = {"step": global_step, "stage_length": length, "evaluation": evaluation}
            history.append(row)
            print(json.dumps(row), flush=True)
            (args.output_root / "partial.json").write_text(json.dumps({"history": history}, indent=2) + "\n")
    checkpoint = args.output_root / "checkpoint_final.pt"
    torch.save(model.state_dict_with_config(), checkpoint)
    final = evaluate_lengths(model, [length for length, _ in stages], args, device, 200_000)
    gate = {f"mqar_{row['sequence_length']}_at_least_80pct": row["validation_accuracy"] >= 0.8 for row in final}
    protocol = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    protocol["curriculum"] = stages
    payload = {
        "protocol": protocol,
        "initialized_keys": initialized,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "history": history,
        "final": final,
        "gate": gate,
        "passed": all(gate.values()),
        "saved_checkpoint": str(checkpoint),
    }
    (args.output_root / "results.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"gate": gate, "passed": payload["passed"], "saved": str(checkpoint)}, indent=2))


if __name__ == "__main__":
    main()
