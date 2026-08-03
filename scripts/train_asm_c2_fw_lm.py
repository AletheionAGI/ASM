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

from drm_language_emitter.checkpoint import load_model
from drm_language_emitter.data import MemmapTokenDataset
from scripts.benchmark_asm_r_long_streaming import delayed_mqar_batch
from scripts.run_mqar_architecture_comparison import load_asm_c2
from scripts.train_asm_c2_fw_durable import evaluate_lengths, parse_curriculum


def precision_context(device: torch.device):
    if device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    student = F.log_softmax(student_logits.float() / temperature, dim=-1)
    teacher = F.softmax(teacher_logits.float() / temperature, dim=-1)
    token_count = max(1, student_logits.numel() // student_logits.shape[-1])
    return (
        F.kl_div(student, teacher, reduction="sum")
        * temperature**2
        / token_count
    )


def optimizer_groups(model, backbone_lr: float, memory_lr: float, weight_decay: float):
    memory, backbone = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        (memory if name.startswith("addressable_memory.") else backbone).append(parameter)
    if not memory or not backbone:
        raise RuntimeError("expected non-empty memory and backbone parameter groups")
    return [
        {
            "params": backbone,
            "lr": backbone_lr,
            "weight_decay": weight_decay,
            "group_name": "language_backbone",
        },
        {
            "params": memory,
            "lr": memory_lr,
            "weight_decay": weight_decay,
            "group_name": "fast_weight_memory_and_gates",
        },
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train ASM-C2-FW-LM with language/MQAR replay and ASM-R distillation."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--curriculum",
        default="40:1000,80:500,160:500,320:300,512:200,1024:100,4096:25",
    )
    parser.add_argument(
        "--language-manifest",
        type=Path,
        default=Path("data/benchmark_125m_wikipedia/train/manifest.json"),
    )
    parser.add_argument("--language-probability", type=float, default=0.8)
    parser.add_argument("--language-seq-len", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-batches", type=int, default=32)
    parser.add_argument("--backbone-lr", type=float, default=1e-5)
    parser.add_argument("--memory-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--distillation-weight", type=float, default=0.5)
    parser.add_argument("--distillation-temperature", type=float, default=2.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--epistemic-memory-gating",
        action="store_true",
        help="train ASM-CM-E confidence gates on fast-weight reads and writes",
    )
    args = parser.parse_args()
    if not 0.0 < args.language_probability < 1.0:
        raise ValueError("language probability must be in (0, 1)")
    if args.backbone_lr <= 0 or args.memory_lr <= 0:
        raise ValueError("learning rates must be positive")
    if args.distillation_weight < 0 or args.distillation_temperature <= 0:
        raise ValueError("invalid distillation settings")

    device = torch.device(args.device)
    stages = parse_curriculum(args.curriculum)
    student, initialized = load_asm_c2(
        args.checkpoint,
        slots=32,
        backend="fast_weight",
        durable=True,
        epistemic_memory_gating=args.epistemic_memory_gating,
    )
    student.config.fast_weight_compute_fp32 = True
    assert student.addressable_memory is not None
    student.addressable_memory.compute_fp32 = True
    student = student.to(device)

    teacher = load_model(args.checkpoint).to(device).eval()
    teacher.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        optimizer_groups(
            student, args.backbone_lr, args.memory_lr, args.weight_decay
        )
    )
    mqar_generator = torch.Generator().manual_seed(args.seed)
    language_generator = torch.Generator().manual_seed(args.seed + 1)
    choice_generator = torch.Generator().manual_seed(args.seed + 2)
    args.output_root.mkdir(parents=True, exist_ok=True)
    history: list[dict] = []
    task_counts = {"language": 0, "mqar": 0}
    global_step = 0
    started = perf_counter()

    with MemmapTokenDataset(
        args.language_manifest, split="train", verify_integrity=False
    ) as language:
        for length, stage_steps in stages:
            for _ in range(stage_steps):
                global_step += 1
                use_language = bool(
                    torch.rand((), generator=choice_generator)
                    < args.language_probability
                )
                student.train()
                optimizer.zero_grad(set_to_none=True)
                if use_language:
                    x, y = language.make_batch(
                        1, args.language_seq_len, device, language_generator
                    )
                    with torch.inference_mode(), precision_context(device):
                        teacher_logits = teacher(
                            x, collect_diagnostics=False
                        )["logits"]
                    with precision_context(device):
                        output = student(
                            x, targets=y, collect_diagnostics=False
                        )
                    ce = output["loss"].float()
                    kd = distillation_loss(
                        output["logits"],
                        teacher_logits,
                        args.distillation_temperature,
                    )
                    loss = ce + args.distillation_weight * kd
                    metrics = {
                        "language_ce": float(ce.detach()),
                        "distillation_kl": float(kd.detach()),
                    }
                    task = "language"
                else:
                    batch_size = min(
                        args.batch_size,
                        1 if length >= 1024 else 2 if length >= 512 else args.batch_size,
                    )
                    x, y, mask = delayed_mqar_batch(
                        length, batch_size, 8, 8, mqar_generator, device
                    )
                    with precision_context(device):
                        selected = student(
                            x, collect_diagnostics=False
                        )["logits"][mask]
                    loss = F.cross_entropy(selected.float(), y[mask])
                    metrics = {"mqar_ce": float(loss.detach())}
                    task = "mqar"
                task_counts[task] += 1
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    student.parameters(), args.max_grad_norm, error_if_nonfinite=True
                )
                optimizer.step()
                if global_step == 1 or global_step % 100 == 0:
                    print(json.dumps({
                        "step": global_step,
                        "stage_length": length,
                        "task": task,
                        "loss": float(loss.detach()),
                        **metrics,
                        "task_counts": task_counts,
                        "elapsed_sec": perf_counter() - started,
                    }), flush=True)
            evaluation = evaluate_lengths(
                student,
                [value for value, _ in stages if value <= length],
                args,
                device,
                100_000,
            )
            row = {
                "step": global_step,
                "stage_length": length,
                "task_counts": dict(task_counts),
                "evaluation": evaluation,
            }
            history.append(row)
            print(json.dumps(row), flush=True)
            (args.output_root / "partial.json").write_text(
                json.dumps({"history": history}, indent=2) + "\n"
            )

    checkpoint = args.output_root / "checkpoint_final.pt"
    torch.save(student.state_dict_with_config(), checkpoint)
    final = evaluate_lengths(
        student, [length for length, _ in stages], args, device, 200_000
    )
    gate = {
        f"mqar_{row['sequence_length']}_at_least_80pct":
            row["validation_accuracy"] >= 0.8
        for row in final
    }
    protocol = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    protocol["curriculum"] = stages
    payload = {
        "variant": "ASM-CM-E" if args.epistemic_memory_gating else "ASM-C2-FW-LM",
        "protocol": protocol,
        "initialized_keys": initialized,
        "parameter_count": sum(p.numel() for p in student.parameters()),
        "optimizer_groups": [
            {"name": group["group_name"], "lr": group["lr"]}
            for group in optimizer.param_groups
        ],
        "task_counts": task_counts,
        "observed_language_fraction": task_counts["language"] / global_step,
        "history": history,
        "final": final,
        "gate": gate,
        "passed": all(gate.values()),
        "saved_checkpoint": str(checkpoint),
    }
    (args.output_root / "results.json").write_text(
        json.dumps(payload, indent=2) + "\n"
    )
    print(json.dumps({
        "gate": gate,
        "passed": payload["passed"],
        "task_counts": task_counts,
        "saved": str(checkpoint),
    }, indent=2))


if __name__ == "__main__":
    main()
