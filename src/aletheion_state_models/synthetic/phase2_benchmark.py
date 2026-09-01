"""Small multiseed benchmark for ASM-VR Phase 2."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import time

import torch

from aletheion_state_models.variants import (
    build_relational_state,
    build_variable_rank_phase2,
)
from drm_language_emitter import DRMConfig

from .variable_capacity_copy import (
    generate_variable_capacity_copy_batch,
    masked_copy_metrics,
    rank_difficulty_correlation,
)

VARIANTS = ("asm_r", "vr_full", "vr_fixed_low", "vr_fixed_mid", "vr_adaptive")


@dataclass(frozen=True)
class Phase2RunResult:
    variant: str
    seed: int
    steps: int
    validation_loss: float
    validation_accuracy: float
    low_accuracy: float
    high_accuracy: float
    mean_rank: float
    rank_std: float
    low_rank: float
    high_rank: float
    rank_difficulty_correlation: float
    controller_gradient_fraction: float
    seconds: float
    finite: bool


def phase2_config(seed: int) -> DRMConfig:
    """Return the frozen tiny-model recipe used by the synthetic gate."""
    return DRMConfig(
        vocab_size=32,
        d_token=16,
        d_state=16,
        n_directions=4,
        metric_rank=4,
        hidden_size=32,
        sequence_mode="directional_block_cumsum",
        directional_cumsum_step_mode="velocity",
        directional_cumsum_block_size=4,
        bounded_state=False,
        dropout=0.0,
        seed=seed,
        lambda_action=0.0,
        lambda_dim_sparsity=0.0,
        lambda_dim_entropy=0.0,
        lambda_dim_variance=0.0,
        lambda_metric_reg=0.0,
        lambda_metric_diversity=0.0,
        lambda_active_fraction=0.0,
        lambda_condition=0.0,
        lambda_metric_u_floor=0.0,
        variable_rank_min_rank=4,
        variable_rank_target_fraction=0.5,
        variable_rank_warmup_steps=20,
        variable_rank_budget_ramp_steps=60,
        variable_rank_hardening_steps=80,
        variable_rank_temperature_initial=2.0,
        variable_rank_temperature_final=0.5,
        lambda_variable_rank_budget=0.02,
        lambda_variable_rank_binary=0.01,
        lambda_variable_rank_switch=0.0,
        variable_rank_open_probability=0.95,
    )


def _fix_rank(model, rank: int) -> None:
    controller = model.variable_rank_core.controller
    with torch.no_grad():
        controller.score_head.weight.zero_()
        controller.score_head.bias.fill_(-20.0)
        controller.score_head.bias[:rank].fill_(20.0)
    for parameter in controller.parameters():
        parameter.requires_grad_(False)


def build_phase2_variant(variant: str, seed: int):
    """Build one paired benchmark arm with stable shared-module seeds."""
    if variant not in VARIANTS:
        raise ValueError(f"unknown Phase 2 variant: {variant}")
    config = phase2_config(seed)
    if variant == "asm_r":
        return build_relational_state(config)
    if variant != "vr_adaptive":
        config = DRMConfig.from_dict(
            config.to_dict()
            | {
                "lambda_variable_rank_budget": 0.0,
                "lambda_variable_rank_binary": 0.0,
                "lambda_variable_rank_switch": 0.0,
            }
        )
    model = build_variable_rank_phase2(config)
    fixed_rank = {
        "vr_full": 16,
        "vr_fixed_low": 4,
        "vr_fixed_mid": 8,
    }.get(variant)
    if fixed_rank is not None:
        _fix_rank(model, fixed_rank)
    return model


def _rank_for_batch(output: dict[str, object], difficulty: torch.Tensor) -> torch.Tensor:
    ranks = output.get("variable_rank_ranks")
    if ranks is None:
        return difficulty.new_full(difficulty.shape, 16, dtype=torch.float32)
    return ranks[:, 4].float()


def _evaluate(model, *, seed: int, batches: int, batch_size: int) -> dict[str, float]:
    model.eval()
    losses, accuracies, ranks, difficulties = [], [], [], []
    low_correct = high_correct = low_count = high_count = 0
    with torch.no_grad():
        for index in range(batches):
            batch = generate_variable_capacity_copy_batch(
                batch_size=batch_size,
                vocab_size=model.config.vocab_size,
                seed=seed + 10_000,
                step=index,
            )
            output = model(batch.input_ids, collect_diagnostics=False)
            loss, accuracy = masked_copy_metrics(
                output["logits"], batch.targets, batch.loss_mask
            )
            predictions = output["logits"].argmax(dim=-1)
            correct = (predictions == batch.targets) & batch.loss_mask
            low = batch.difficulty == 1
            high = batch.difficulty == 3
            low_correct += int(correct[low].sum())
            high_correct += int(correct[high].sum())
            low_count += int(batch.loss_mask[low].sum())
            high_count += int(batch.loss_mask[high].sum())
            losses.append(loss)
            accuracies.append(accuracy)
            ranks.append(_rank_for_batch(output, batch.difficulty))
            difficulties.append(batch.difficulty.float())
    all_ranks = torch.cat(ranks)
    all_difficulties = torch.cat(difficulties)
    low = all_difficulties == 1
    high = all_difficulties == 3
    return {
        "loss": torch.stack(losses).mean().item(),
        "accuracy": torch.stack(accuracies).mean().item(),
        "low_accuracy": low_correct / max(low_count, 1),
        "high_accuracy": high_correct / max(high_count, 1),
        "mean_rank": all_ranks.mean().item(),
        "rank_std": all_ranks.std(unbiased=False).item(),
        "low_rank": all_ranks[low].mean().item(),
        "high_rank": all_ranks[high].mean().item(),
        "correlation": rank_difficulty_correlation(
            all_ranks, all_difficulties
        ).item(),
    }


def train_phase2_run(
    variant: str,
    seed: int,
    *,
    steps: int = 200,
    batch_size: int = 64,
    learning_rate: float = 3e-3,
    evaluation_batches: int = 16,
) -> Phase2RunResult:
    """Train and evaluate one deterministic synthetic benchmark arm."""
    torch.manual_seed(seed)
    model = build_phase2_variant(variant, seed).train()
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=learning_rate,
        weight_decay=0.0,
    )
    gradient_hits = gradient_checks = 0
    finite = True
    started = time.perf_counter()
    for step in range(steps):
        batch = generate_variable_capacity_copy_batch(
            batch_size=batch_size,
            vocab_size=model.config.vocab_size,
            seed=seed,
            step=step,
        )
        optimizer.zero_grad(set_to_none=True)
        output = model(
            batch.input_ids,
            global_step=step,
            collect_diagnostics=False,
        )
        copy_loss, _ = masked_copy_metrics(
            output["logits"], batch.targets, batch.loss_mask
        )
        rank_loss = output["aux_losses"].get(
            "variable_rank_regularization", copy_loss.new_tensor(0.0)
        )
        loss = copy_loss + rank_loss
        if not bool(torch.isfinite(loss)):
            finite = False
            break
        loss.backward()
        if variant == "vr_adaptive" and step >= model.config.variable_rank_warmup_steps:
            gradient_checks += 1
            gradient = model.variable_rank_core.controller.score_head.weight.grad
            if gradient is not None and torch.linalg.vector_norm(gradient).item() > 1e-8:
                gradient_hits += 1
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    elapsed = time.perf_counter() - started
    metrics = _evaluate(
        model, seed=seed, batches=evaluation_batches, batch_size=batch_size
    )
    gradient_fraction = gradient_hits / max(gradient_checks, 1)
    return Phase2RunResult(
        variant=variant,
        seed=seed,
        steps=steps,
        validation_loss=metrics["loss"],
        validation_accuracy=metrics["accuracy"],
        low_accuracy=metrics["low_accuracy"],
        high_accuracy=metrics["high_accuracy"],
        mean_rank=metrics["mean_rank"],
        rank_std=metrics["rank_std"],
        low_rank=metrics["low_rank"],
        high_rank=metrics["high_rank"],
        rank_difficulty_correlation=metrics["correlation"],
        controller_gradient_fraction=gradient_fraction,
        seconds=elapsed,
        finite=finite and all(math.isfinite(value) for value in metrics.values()),
    )


def summarize_phase2(results: list[Phase2RunResult]) -> dict[str, object]:
    """Aggregate variants and evaluate the predeclared synthetic gates."""
    grouped: dict[str, list[Phase2RunResult]] = {}
    for result in results:
        grouped.setdefault(result.variant, []).append(result)
    summary = {}
    for variant, rows in grouped.items():
        summary[variant] = {
            "runs": len(rows),
            "validation_loss_mean": sum(row.validation_loss for row in rows) / len(rows),
            "validation_accuracy_mean": sum(row.validation_accuracy for row in rows) / len(rows),
            "mean_rank": sum(row.mean_rank for row in rows) / len(rows),
            "rank_std_mean": sum(row.rank_std for row in rows) / len(rows),
            "rank_difficulty_correlation_mean": sum(
                row.rank_difficulty_correlation for row in rows
            ) / len(rows),
        }
    adaptive = grouped.get("vr_adaptive", [])
    fixed_mid = grouped.get("vr_fixed_mid", [])
    gates = {
        "all_runs_finite": bool(results) and all(row.finite for row in results),
        "three_adaptive_seeds": len(adaptive) == 3,
        "adaptive_budget": bool(adaptive)
        and all(5.0 <= row.mean_rank <= 11.0 for row in adaptive),
        "adaptive_rank_variation": bool(adaptive)
        and all(row.rank_std >= 0.5 for row in adaptive),
        "adaptive_tracks_difficulty": bool(adaptive)
        and sum(row.rank_difficulty_correlation for row in adaptive) / len(adaptive) > 0.1,
        "controller_receives_gradient": bool(adaptive)
        and all(row.controller_gradient_fraction >= 0.9 for row in adaptive),
        "quality_near_fixed_mid": bool(adaptive and fixed_mid)
        and (
            sum(row.validation_accuracy for row in adaptive) / len(adaptive)
            >= sum(row.validation_accuracy for row in fixed_mid) / len(fixed_mid) - 0.05
        ),
    }
    return {
        "passed": all(gates.values()),
        "gates": gates,
        "variants": summary,
        "runs": [asdict(result) for result in results],
        "claims": {"quality_or_capacity": True, "hardware_speedup": False},
    }

__all__ = [
    "Phase2RunResult",
    "VARIANTS",
    "build_phase2_variant",
    "phase2_config",
    "summarize_phase2",
    "train_phase2_run",
]
