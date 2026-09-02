"""Official sequential training with spawn-isolated all-arm evaluation."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from .constants import ARMS, TRAINING_SEEDS
from .evaluation_broker import ScorerRefs, freeze_all_arms_then_join_truth
from .official_data import Origin, TruthCache, training_order


def train_and_score_isolated(
    data: dict[str, tuple[Origin, ...]],
    output_dir: Path,
    progress: Any,
    *,
    updates: int,
    batch_size: int,
    device: str,
) -> list[dict[str, Any]]:
    """Freeze all R/CM/Z/T scores before any calibration/test truth join."""
    import torch

    from .checkpoint import write_exact_checkpoint
    from .models import build_adapter
    from .official_stats import calibrate, invalid_row, summarize
    from .official_training import (
        _authorized_config,
        _checkpoint_bytes,
        _stream_context,
        _train,
    )

    rows = []
    truth_cache = TruthCache()
    for seed in TRAINING_SEEDS:
        order = training_order(data["train"], seed)
        refs: dict[str, ScorerRefs] = {}
        metadata: dict[str, tuple[int, float]] = {}
        for arm in ARMS:
            stream = torch.cuda.Stream()
            with _stream_context(torch, stream):
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
            payload = _checkpoint_bytes(torch, model, optimizer, config, updates)
            checkpoint_path, _ = write_exact_checkpoint(
                output_dir / "checkpoints", arm, seed, updates, payload, synthetic=False
            )
            peak = max(
                int(torch.cuda.max_memory_allocated()),
                int(torch.cuda.max_memory_reserved()),
            )
            config_path = (
                Path(__file__).resolve().parents[2]
                / "configs"
                / "attr_rtg_rcmz_v1"
                / f"{arm.lower()}_seed{seed}.yaml"
            )
            refs[arm] = ScorerRefs(str(config_path), str(checkpoint_path))
            metadata[arm] = (peak, started)
            progress(
                {"phase": "arm-trained", "seed": seed, "arm": arm, "update": updates}
            )
            del optimizer, model
            stream.synchronize()
            torch.cuda.empty_cache()

        calibration = freeze_all_arms_then_join_truth(
            data["calibration"],
            refs,
            device=device,
            batch_size=batch_size,
            truth_cache=truth_cache,
        )
        parameters: dict[str, tuple[float, float] | None] = {}
        reasons: dict[str, str] = {}
        for arm in ARMS:
            try:
                parameters[arm] = calibrate(_records(calibration, arm, device))
            except (RuntimeError, ValueError, TypeError) as error:
                parameters[arm] = None
                reasons[arm] = f"calibration: {error}"

        for split, regime in (
            ("test_id", "ID"),
            ("test_shift", "shift"),
            ("test_ood", "OOD"),
        ):
            evaluation = freeze_all_arms_then_join_truth(
                data[split],
                refs,
                device=device,
                batch_size=batch_size,
                truth_cache=truth_cache,
            )
            for arm in ARMS:
                peak, started = metadata[arm]
                fitted = parameters[arm]
                if fitted is None:
                    row = invalid_row(
                        arm=arm,
                        seed=seed,
                        regime=regime,
                        reason=reasons[arm],
                        peak_bytes=peak,
                        elapsed=time.monotonic() - started,
                    )
                else:
                    temperature, tau = fitted
                    try:
                        row = summarize(
                            _records(evaluation, arm, device),
                            temperature,
                            tau,
                            arm=arm,
                            seed=seed,
                            regime=regime,
                            peak_bytes=peak,
                            elapsed=time.monotonic() - started,
                        )
                    except (RuntimeError, ValueError, TypeError) as error:
                        row = invalid_row(
                            arm=arm,
                            seed=seed,
                            regime=regime,
                            reason=f"metrics: {error}",
                            peak_bytes=peak,
                            elapsed=time.monotonic() - started,
                            temperature=temperature,
                            tau=tau,
                        )
                rows.append(row)
                progress(
                    {
                        "phase": "arm-complete",
                        "seed": seed,
                        "arm": arm,
                        "update": updates,
                        "regime": regime,
                    }
                )
    return rows


def _records(evaluation: Any, arm: str, device: str) -> list[dict[str, Any]]:
    import torch

    frozen = next(item for item in evaluation.scores if item.arm == arm)
    logits = [row for batch in frozen.batches for row in batch.logits]
    if len(logits) != len(evaluation.truth):
        raise RuntimeError("frozen score and truth counts differ")
    return [
        {
            "world": identity[1],
            "episode": identity[2],
            "origin": identity[3],
            "logits": torch.tensor(score, dtype=torch.float32, device=device),
            "labels": torch.tensor(truth, dtype=torch.float32, device=device),
            "valid": True,
        }
        for identity, score, truth in zip(
            evaluation.identities, logits, evaluation.truth, strict=True
        )
    ]
