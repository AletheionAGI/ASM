"""CUDA model training and one-pass evaluation for the official orchestrator."""

from __future__ import annotations

import io
import time
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

from .checkpoint import write_exact_checkpoint
from .constants import ARMS, BACKBONE_UPDATES, TRAINING_SEEDS
from .official_data import (
    Origin,
    TruthCache,
    materialize,
    training_order,
    truths_after_forward,
)
from .policy import configure_torch, derive_seed64

Progress = Callable[[dict[str, object]], None]


def train_and_score(
    data: dict[str, tuple[Origin, ...]],
    output_dir: Path,
    progress: Progress,
    *,
    updates: int = BACKBONE_UPDATES,
    batch_size: int = 64,
    miniature: bool = False,
    lock: dict[str, object] | None = None,
    recovered: dict[int, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    if not miniature:
        from .lock_guard import verify_runtime_lock

        verify_runtime_lock(lock)
    import torch

    from .models import build_adapter

    if not miniature and (not torch.cuda.is_available() or updates != BACKBONE_UPDATES):
        raise RuntimeError("official execution requires CUDA and exactly 2,000 updates")
    device = "cpu" if miniature else "cuda"
    configure_torch(torch)
    if not miniature:
        from .official_isolated import train_and_score_isolated

        return train_and_score_isolated(
            data,
            output_dir,
            progress,
            updates=updates,
            batch_size=batch_size,
            device=device,
            recovered=recovered,
        )
    rows = []
    truth_cache = TruthCache()
    for seed in TRAINING_SEEDS if not miniature else TRAINING_SEEDS[:1]:
        order = training_order(data["train"], seed)
        for arm in ARMS if not miniature else ARMS[:1]:
            stream = None if miniature else torch.cuda.Stream()
            context = _stream_context(torch, stream)
            with context:
                config = _authorized_config(arm, seed)
                model = build_adapter(config).to(device=device, dtype=torch.float32)
                optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
            started = time.monotonic()
            _train(
                model,
                optimizer,
                data["train"],
                order,
                device,
                stream,
                seed,
                updates,
                batch_size,
                progress,
                arm,
                truth_cache,
            )
            checkpoint = _checkpoint_bytes(torch, model, optimizer, config, updates)
            write_exact_checkpoint(
                output_dir / "checkpoints",
                arm,
                seed,
                updates,
                checkpoint,
                synthetic=miniature,
            )
            peak = (
                0
                if miniature
                else max(
                    int(torch.cuda.max_memory_allocated()),
                    int(torch.cuda.max_memory_reserved()),
                )
            )
            calibration = _evaluate(
                model, data["calibration"], device, stream, batch_size, truth_cache
            )
            from .official_stats import calibrate, summarize

            temperature, tau = calibrate(calibration)
            for split, regime in (
                ("test_id", "ID"),
                ("test_shift", "shift"),
                ("test_ood", "OOD"),
            ):
                records = _evaluate(
                    model, data[split], device, stream, batch_size, truth_cache
                )
                rows.append(
                    summarize(
                        records,
                        temperature,
                        tau,
                        arm=arm,
                        seed=seed,
                        regime=regime,
                        peak_bytes=peak,
                        elapsed=time.monotonic() - started,
                    )
                )
            progress(
                {"phase": "arm-complete", "seed": seed, "arm": arm, "update": updates}
            )
            del optimizer, model
            if not miniature:
                stream.synchronize()
                torch.cuda.empty_cache()
    return rows


def _train(
    model: Any,
    optimizer: Any,
    origins: tuple[Origin, ...],
    order: tuple[int, ...],
    device: str,
    stream: Any,
    seed: int,
    updates: int,
    batch_size: int,
    progress: Progress,
    arm: str,
    truth_cache: TruthCache,
) -> None:
    import torch

    from .contracts import InferenceMessage

    generator = torch.Generator(device=device).manual_seed(derive_seed64("train", seed))
    del generator  # Reserved independent domain; fixed manifest defines batch order.
    model.train()
    for update in range(1, updates + 1):
        offset = ((update - 1) * batch_size) % len(order)
        indices = [order[(offset + slot) % len(order)] for slot in range(batch_size)]
        batch = materialize(origins, indices, device)
        with (
            _stream_context(torch, stream),
            torch.autocast(device_type=device, enabled=False),
        ):
            result = model(InferenceMessage.from_mapping(batch["message"]))
            labels = truths_after_forward(batch["origins"], device, truth_cache)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                result.logits, labels
            )
            if not bool(torch.isfinite(loss)):
                raise RuntimeError(f"nonfinite loss at update {update}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        if update == 1 or update % 100 == 0 or update == updates:
            progress(
                {
                    "phase": "training",
                    "seed": seed,
                    "arm": arm,
                    "update": update,
                    "total_updates": updates,
                }
            )


def _evaluate(
    model: Any,
    origins: tuple[Origin, ...],
    device: str,
    stream: Any,
    batch_size: int,
    truth_cache: TruthCache,
) -> list[dict[str, Any]]:
    import torch

    from .contracts import InferenceMessage

    model.eval()
    rows = []
    with torch.no_grad():
        for offset in range(0, len(origins), batch_size):
            indices = list(range(offset, min(offset + batch_size, len(origins))))
            batch = materialize(origins, indices, device)
            with (
                _stream_context(torch, stream),
                torch.autocast(device_type=device, enabled=False),
            ):
                result = model(InferenceMessage.from_mapping(batch["message"]))
            labels = truths_after_forward(batch["origins"], device, truth_cache)
            for origin, logits, truth in zip(
                batch["origins"], result.logits, labels, strict=True
            ):
                rows.append(
                    {
                        "world": origin.world_index,
                        "episode": origin.episode,
                        "origin": origin.origin,
                        "logits": logits.detach(),
                        "labels": truth,
                        "valid": True,
                    }
                )
    return rows


def _authorized_config(arm: str, seed: int) -> Any:
    from .config import load_config

    path = (
        Path(__file__).resolve().parents[2]
        / "configs"
        / "attr_rtg_rcmz_v1"
        / f"{arm.lower()}_seed{seed}.yaml"
    )
    config = load_config(path)
    return replace(config, synthetic_only=False, official_operations_allowed=True)


def _checkpoint_bytes(
    torch: Any, model: Any, optimizer: Any, config: Any, update: int
) -> bytes:
    target = io.BytesIO()
    torch.save(
        {
            "update": update,
            "config": config.__dict__,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        target,
    )
    return target.getvalue()


def _stream_context(torch: Any, stream: Any) -> Any:
    from contextlib import nullcontext

    return nullcontext() if stream is None else torch.cuda.stream(stream)
