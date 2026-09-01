"""Long delayed-MQAR and 32K gates for strict ASM-CM-VR."""

from __future__ import annotations
from copy import deepcopy
from pathlib import Path
import math
import time
import torch
from torch.nn import functional as F
from .phase3a_checkpoint import atomic_torch_save, write_result
from .phase3a_training import measure_streaming_error
from .phase3a_variants import phase3a_config
from .purpose_mqar import delayed_mqar_batch, evaluate_mqar
from .purpose_streaming import probe_streaming
from aletheion_state_models.variants import (
    build_compact_memory_adaptive_rank,
    build_compact_memory_variable_rank,
)

CURRICULUM = (
    (40, 1000, 4),
    (80, 500, 4),
    (160, 500, 4),
    (320, 300, 4),
    (512, 200, 2),
    (1024, 100, 1),
    (4096, 25, 1),
)
TEST_LENGTHS = (40, 512, 4096, 32768)
ARMS = {"cm_vr_full64": 64, "cm_vr_fixed32": 32, "cm_vr_adaptive32": None}


def _autocast(device):
    return torch.autocast(
        device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
    )


def _checkpoint_path(root, arm, seed):
    return Path(root) / arm / f"seed_{seed}" / "curriculum.pt"


def _build(arm, seed, device):
    torch.manual_seed(seed)
    base = phase3a_config(seed)
    model = (
        build_compact_memory_adaptive_rank(base, target_rank=32)
        if arm == "cm_vr_adaptive32"
        else build_compact_memory_variable_rank(base, fixed_rank=ARMS[arm])
    )
    return model.to(device)


def train_curriculum(arm, seed, *, run_root, device):
    """Train or resume one arm, checkpointing only validation-stage boundaries."""
    path = _checkpoint_path(run_root, arm, seed)
    model = _build(arm, seed, device)
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=3e-4, weight_decay=0.01
    )
    generator = torch.Generator().manual_seed(seed)
    history = []
    completed = -1
    updates = 0
    gradient_checks = gradient_hits = 0
    if path.exists():
        saved = torch.load(path, map_location=device, weights_only=False)
        model.load_state_dict(saved["model_state"])
        optimizer.load_state_dict(saved["optimizer_state"])
        generator.set_state(saved["generator_state"].cpu())
        history = saved["history"]
        completed = saved["completed_stage"]
        updates = saved["updates"]
        gradient_checks = saved.get("gradient_checks", 0)
        gradient_hits = saved.get("gradient_hits", 0)
    started = time.perf_counter()
    for stage, (length, batches, batch_size) in enumerate(CURRICULUM):
        if stage <= completed:
            continue
        model.train()
        losses = []
        for _ in range(batches):
            updates += 1
            x, y, mask = delayed_mqar_batch(length, batch_size, generator, device)
            optimizer.zero_grad(set_to_none=True)
            with _autocast(device):
                output = model(x, collect_diagnostics=False, global_step=updates)
                selected = output["logits"][mask]
                task_loss = F.cross_entropy(selected.float(), y[mask])
                loss = task_loss
                if arm == "cm_vr_adaptive32":
                    loss = loss + output["aux_losses"]["variable_rank_regularization"]
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite {arm} seed={seed} length={length}")
            loss.backward()
            if (
                arm == "cm_vr_adaptive32"
                and updates > model.config.variable_rank_warmup_steps
            ):
                gradient_checks += 1
                gradient = model.variable_rank_core.controller.score_head.weight.grad
                if (
                    gradient is not None
                    and torch.linalg.vector_norm(gradient).item() > 1e-8
                ):
                    gradient_hits += 1
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0, error_if_nonfinite=True
            )
            optimizer.step()
            losses.append(float(task_loss.detach()))
        model.eval()
        control = evaluate_mqar(
            model, 40, seed=90_000 + seed + length, batches=8, device=device
        )
        current = evaluate_mqar(
            model,
            length,
            seed=91_000 + seed + length,
            batches=8 if length <= 512 else 4,
            device=device,
        )
        row = {
            "stage": stage,
            "length": length,
            "updates": updates,
            "train_ce_mean": sum(losses) / len(losses),
            "control_40_accuracy": control["accuracy"],
            "validation_accuracy": current["accuracy"],
            "validation_ce": current["ce"],
            "mean_rank": current["mean_rank"],
            "rank_std": current["rank_std"],
            "rank_min": current["rank_min"],
            "rank_max": current["rank_max"],
        }
        history.append(row)
        print({"arm": arm, "seed": seed, **row}, flush=True)
        atomic_torch_save(
            path,
            {
                "model_state": deepcopy(
                    {
                        name: value.detach().cpu()
                        for name, value in model.state_dict().items()
                    }
                ),
                "optimizer_state": optimizer.state_dict(),
                "generator_state": generator.get_state(),
                "history": history,
                "completed_stage": stage,
                "updates": updates,
                "gradient_checks": gradient_checks,
                "gradient_hits": gradient_hits,
            },
        )
    return (
        model.eval(),
        optimizer,
        history,
        updates,
        time.perf_counter() - started,
        gradient_checks,
        gradient_hits,
    )


def evaluate_frozen(
    model,
    arm,
    seed,
    *,
    run_root,
    device,
    history,
    updates,
    elapsed,
    gradient_checks,
    gradient_hits,
):
    tests = []
    for length in TEST_LENGTHS:
        batches = 16 if length <= 512 else (8 if length <= 4096 else 4)
        try:
            tests.append(
                evaluate_mqar(
                    model, length, seed=100_000 + seed, batches=batches, device=device
                )
            )
        except (RuntimeError, ValueError) as exc:
            tests.append({"length": length, "status": "failed", "error": str(exc)})
    memory = model.addressable_memory
    read, write = memory.read_enabled, memory.write_enabled
    memory.read_enabled = False
    no_read = evaluate_mqar(model, 40, seed=100_000 + seed, batches=16, device=device)
    memory.read_enabled = read
    memory.write_enabled = False
    no_write = evaluate_mqar(model, 40, seed=100_000 + seed, batches=16, device=device)
    memory.write_enabled = write
    corpus = torch.arange(256, dtype=torch.uint8).repeat(8)
    parity = measure_streaming_error(model, corpus, device)
    streaming = probe_streaming(
        model, (512, 4096, 32768), seed=110_000 + seed, device=str(device)
    )
    directory = Path(run_root) / arm / f"seed_{seed}"
    test_finite = all(item.get("ce_finite", False) for item in tests)
    stream_finite = all(item.get("status") != "failed" for item in streaming)
    result = {
        "arm": arm,
        "seed": seed,
        "logical_rank": ARMS[arm],
        "adaptive_target_rank": 32 if arm == "cm_vr_adaptive32" else None,
        "updates": updates,
        "elapsed_sec": elapsed,
        "parameters_total": sum(p.numel() for p in model.parameters()),
        "parameters_trainable": sum(
            p.numel() for p in model.parameters() if p.requires_grad
        ),
        "history": history,
        "test": tests,
        "no_read": no_read,
        "no_write": no_write,
        "streaming_error": parity,
        "streaming": streaming,
        "finite": math.isfinite(parity) and test_finite and stream_finite,
        "controller_gradient_checks": gradient_checks,
        "controller_gradient_hits": gradient_hits,
    }
    write_result(directory / "result.json", result)
    return result


def run_arm(arm, seed, *, run_root, device="cuda"):
    torch_device = torch.device(device)
    result_path = Path(run_root) / arm / f"seed_{seed}" / "result.json"
    if result_path.exists():
        return __import__("json").loads(result_path.read_text())
    model, _, history, updates, elapsed, checks, hits = train_curriculum(
        arm, seed, run_root=run_root, device=torch_device
    )
    return evaluate_frozen(
        model,
        arm,
        seed,
        run_root=run_root,
        device=torch_device,
        history=history,
        updates=updates,
        elapsed=elapsed,
        gradient_checks=checks,
        gradient_hits=hits,
    )


def arm_passed(result):
    successful = [
        row
        for row in result["test"]
        if row.get("status") != "failed" and row.get("ce_finite", False)
    ]
    quality = len(successful) == len(TEST_LENGTHS) and all(
        row["accuracy"] >= 0.80 for row in successful
    )
    stream = any(
        row.get("length") == 32768 and row.get("status") != "failed"
        for row in result["streaming"]
    )
    finite = (
        math.isfinite(result["streaming_error"])
        and len(successful) == len(TEST_LENGTHS)
        and all(row.get("status") != "failed" for row in result["streaming"])
    )
    return quality and stream and result["streaming_error"] <= 1e-4 and finite


__all__ = ["ARMS", "CURRICULUM", "TEST_LENGTHS", "arm_passed", "run_arm"]
