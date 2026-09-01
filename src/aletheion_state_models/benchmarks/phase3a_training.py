"""Single-run training and evaluation for ASM-VR Phase 3A."""
from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass
import math
from pathlib import Path
import time
from typing import Callable
import torch
from torch import Tensor

from .phase3a_checkpoint import atomic_torch_save, write_result
from .phase3a_data import ByteCorpusSplits, sample_byte_windows
from .phase3a_variants import build_phase3a_variant

@dataclass(frozen=True)
class Phase3ARunResult:
    variant: str
    seed: int
    steps: int
    tokens_seen: int
    best_step: int
    validation_ce: float
    validation_ppl: float
    test_ce: float
    test_ppl: float
    mean_rank: float
    rank_std: float
    rank_min: float
    rank_max: float
    rank_ce_correlation: float
    controller_gradient_fraction: float
    tokens_per_second: float
    peak_memory_mb: float
    parameter_count: int
    streaming_error: float
    finite: bool
    history: list[dict[str, float]]

def _autocast(device: torch.device):
    return torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    )

def _correlation(first: Tensor, second: Tensor) -> float:
    first = first.float().reshape(-1)
    second = second.float().reshape(-1)
    first = first - first.mean()
    second = second - second.mean()
    denominator = torch.linalg.vector_norm(first) * torch.linalg.vector_norm(second)
    if denominator == 0:
        return 0.0
    return (torch.dot(first, second) / denominator).item()

def evaluate_language_model(
    model,
    tokens: Tensor,
    *,
    seed: int,
    batches: int,
    batch_size: int,
    sequence_length: int,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate CE and hard-rank behavior on frozen deterministic windows."""
    model.eval()
    losses, ranks, block_losses = [], [], []
    with torch.no_grad():
        for index in range(batches):
            inputs, targets = sample_byte_windows(
                tokens,
                batch_size=batch_size,
                sequence_length=sequence_length,
                seed=seed,
                step=index,
                device=device,
            )
            with _autocast(device):
                output = model(inputs, collect_diagnostics=False)
                token_loss = torch.nn.functional.cross_entropy(
                    output["logits"].float().reshape(-1, 256),
                    targets.reshape(-1),
                    reduction="none",
                ).reshape(batch_size, sequence_length)
            losses.append(token_loss.mean())
            if "variable_rank_ranks" in output:
                token_ranks = output["variable_rank_ranks"].float()
            else:
                token_ranks = token_loss.new_full(token_loss.shape, model.config.d_state)
            ranks.append(token_ranks[:, :: model.config.directional_cumsum_block_size])
            block_losses.append(
                token_loss.reshape(
                    batch_size,
                    -1,
                    model.config.directional_cumsum_block_size,
                ).mean(dim=-1)
            )
    all_ranks = torch.cat([value.flatten().cpu() for value in ranks])
    all_block_losses = torch.cat([value.flatten().cpu() for value in block_losses])
    ce = torch.stack(losses).mean().item()
    return {
        "ce": ce,
        "ppl": math.exp(min(ce, 20.0)),
        "mean_rank": all_ranks.mean().item(),
        "rank_std": all_ranks.std(unbiased=False).item(),
        "rank_min": all_ranks.min().item(),
        "rank_max": all_ranks.max().item(),
        "rank_ce_correlation": _correlation(all_ranks, all_block_losses),
    }

def measure_streaming_error(model, tokens: Tensor, device: torch.device) -> float:
    """Compare a short full forward with public prefill/decode inference."""
    inputs, _ = sample_byte_windows(
        tokens,
        batch_size=1,
        sequence_length=64,
        seed=404,
        step=0,
        device=device,
    )
    model.eval()
    with torch.no_grad():
        expected = model(inputs, collect_diagnostics=False)["logits"]
        first, state = model.prefill(inputs[:, :16])
        observed = [first]
        for position in range(16, inputs.shape[1]):
            current, state = model.decode_step(inputs[:, position], state)
            observed.append(current.unsqueeze(1))
        actual = torch.cat(observed, dim=1)
    return torch.max(torch.abs(actual.float() - expected.float())).item()

def train_phase3a_run(
    variant: str,
    seed: int,
    splits: ByteCorpusSplits,
    *,
    output_directory: str | Path,
    steps: int = 489,
    batch_size: int = 16,
    sequence_length: int = 256,
    learning_rate: float = 3e-4,
    evaluation_batches: int = 16,
    milestones: tuple[int, ...] = (100, 200, 300, 400, 489),
    device: str = "cuda",
    variant_builder: Callable[[str, int], tuple[object, int | None]] = build_phase3a_variant,
    adaptive_variants: frozenset[str] = frozenset({"vr_adaptive_32"}),
    optimizer_factory: Callable[[list[torch.nn.Parameter], float], torch.optim.Optimizer] | None = None,
    evaluate_test: bool = True,
) -> Phase3ARunResult:
    """Train one paired run, select by validation CE, then evaluate test once."""
    torch.manual_seed(seed)
    torch_device = torch.device(device)
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    model, _ = variant_builder(variant, seed)
    model.to(torch_device).train()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = (
        torch.optim.AdamW(trainable, lr=learning_rate, weight_decay=0.01)
        if optimizer_factory is None
        else optimizer_factory(trainable, learning_rate)
    )
    if torch_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(torch_device)
        torch.cuda.synchronize(torch_device)
    started = time.perf_counter()
    best_ce = float("inf")
    best_step = 0
    best_state = None
    history: list[dict[str, float]] = []
    gradient_hits = gradient_checks = 0
    finite = True
    milestone_set = set(milestones) | {steps}
    for step in range(1, steps + 1):
        inputs, targets = sample_byte_windows(
            splits.train,
            batch_size=batch_size,
            sequence_length=sequence_length,
            seed=seed,
            step=step - 1,
            device=torch_device,
        )
        optimizer.zero_grad(set_to_none=True)
        with _autocast(torch_device):
            output = model(
                inputs,
                targets=targets,
                global_step=step - 1,
                collect_diagnostics=False,
            )
            loss = output["loss"]
        if not bool(torch.isfinite(loss)):
            finite = False
            break
        loss.backward()
        if variant in adaptive_variants and step > model.config.variable_rank_warmup_steps:
            gradient_checks += 1
            gradient = model.variable_rank_core.controller.score_head.weight.grad
            if gradient is not None and torch.linalg.vector_norm(gradient).item() > 1e-8:
                gradient_hits += 1
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step in milestone_set:
            metrics = evaluate_language_model(
                model,
                splits.validation,
                seed=10_000 + seed,
                batches=evaluation_batches,
                batch_size=batch_size,
                sequence_length=sequence_length,
                device=torch_device,
            )
            history.append(
                {
                    "step": float(step),
                    "tokens": float(step * batch_size * sequence_length),
                    "train_loss": float(loss.detach()),
                    "validation_ce": metrics["ce"],
                    "mean_rank": metrics["mean_rank"],
                }
            )
            if metrics["ce"] < best_ce:
                best_ce = metrics["ce"]
                best_step = step
                best_state = deepcopy(
                    {name: value.detach().cpu() for name, value in model.state_dict().items()}
                )
            model.train()
    if torch_device.type == "cuda":
        torch.cuda.synchronize(torch_device)
    elapsed = time.perf_counter() - started
    if best_state is None:
        raise RuntimeError("run produced no finite validation checkpoint")
    model.load_state_dict(best_state)
    validation = evaluate_language_model(
        model,
        splits.validation,
        seed=10_000 + seed,
        batches=evaluation_batches,
        batch_size=batch_size,
        sequence_length=sequence_length,
        device=torch_device,
    )
    test = (
        evaluate_language_model(
            model, splits.test, seed=20_000 + seed,
            batches=evaluation_batches, batch_size=batch_size,
            sequence_length=sequence_length, device=torch_device,
        )
        if evaluate_test else None
    )
    rank_evaluation = test if test is not None else validation
    streaming_error = measure_streaming_error(model, splits.validation, torch_device)
    tokens_seen = steps * batch_size * sequence_length
    peak_memory = (
        torch.cuda.max_memory_allocated(torch_device) / 2**20
        if torch_device.type == "cuda"
        else 0.0
    )
    checkpoint = {
        "variant": variant,
        "seed": seed,
        "best_step": best_step,
        "config": model.config.to_dict(),
        "model_state": best_state,
        "optimizer_state": optimizer.state_dict(),
        "corpus_manifest": splits.manifest,
    }
    run_directory = Path(output_directory) / variant / f"seed_{seed}"
    atomic_torch_save(run_directory / "best.pt", checkpoint)
    result = Phase3ARunResult(
        variant=variant,
        seed=seed,
        steps=steps,
        tokens_seen=tokens_seen,
        best_step=best_step,
        validation_ce=validation["ce"],
        validation_ppl=validation["ppl"],
        test_ce=test["ce"] if test is not None else float("nan"),
        test_ppl=test["ppl"] if test is not None else float("nan"),
        mean_rank=rank_evaluation["mean_rank"],
        rank_std=rank_evaluation["rank_std"],
        rank_min=rank_evaluation["rank_min"],
        rank_max=rank_evaluation["rank_max"],
        rank_ce_correlation=rank_evaluation["rank_ce_correlation"],
        controller_gradient_fraction=gradient_hits / max(gradient_checks, 1),
        tokens_per_second=tokens_seen / max(elapsed, 1e-9),
        peak_memory_mb=peak_memory,
        parameter_count=sum(parameter.numel() for parameter in model.parameters()),
        streaming_error=streaming_error,
        finite=finite,
        history=history,
    )
    write_result(run_directory / "result.json", result)
    return result

__all__ = ["Phase3ARunResult", "evaluate_language_model", "train_phase3a_run"]
