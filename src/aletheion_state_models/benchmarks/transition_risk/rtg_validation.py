"""Allowed-split terminal validation and calibration-score extraction."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .rtg_losses import PHYSICAL_CARDINALITIES, PHYSICAL_OFFSETS
from .rtg_metrics_state import (
    consequence_macro_accuracy,
    consequence_nll,
    transition_state_metrics,
)
from .rtg_normalization import StateNormalization
from .rtg_physical_targets import unsafe_predicate
from .rtg_sampling import estimate_g_risk, smoothed_g_logit
from .rtg_state_records import CandidateStateRecord
from .rtg_train_backbone import _logits
from .rtg_training_data import BehavioralEpisode, collate_ce_episodes


def terminal_ce(
    model: nn.Module, episodes: tuple[BehavioralEpisode, ...], *, batch_size: int = 4,
    device: torch.device | str = "cpu",
) -> float:
    """Evaluate next-byte CE without episode-boundary or padding targets."""
    if not episodes or batch_size < 1:
        raise ValueError("terminal CE requires episodes and a positive batch size")
    model.to(device).eval()
    total = 0.0
    targets_count = 0
    with torch.no_grad():
        for offset in range(0, len(episodes), batch_size):
            batch = collate_ce_episodes(episodes[offset : offset + batch_size])
            inputs, targets = batch.input_ids.to(device), batch.targets.to(device)
            logits = _logits(model, inputs, 1_000)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), targets.reshape(-1),
                ignore_index=-100, reduction="sum",
            )
            count = int((targets != -100).sum())
            if not torch.isfinite(loss) or count < 1:
                raise FloatingPointError("validation CE is non-finite or empty")
            total += float(loss.cpu())
            targets_count += count
    return total / targets_count


def _identity(record: CandidateStateRecord, seed: int) -> dict[str, object]:
    return {
        "seed": seed, "world_id": record.world_id,
        "episode_id": record.episode_id, "t": record.t,
        "action_index": record.action_index,
    }


def _group_values(logits: torch.Tensor, target: tuple[int, ...]) -> tuple[list[float], list[int]]:
    losses, predictions = [], []
    for index, classes in enumerate(PHYSICAL_CARDINALITIES):
        start, stop = PHYSICAL_OFFSETS[index : index + 2]
        group = logits[start:stop].double()
        if not torch.isfinite(group).all() or target[index] not in range(classes):
            raise FloatingPointError("invalid physical validation output")
        losses.append(float(-F.log_softmax(group, dim=0)[target[index]]))
        predictions.append(int(torch.argmax(group)))
    return losses, predictions


def preliminary_rtg1(
    records: tuple[CandidateStateRecord, ...], normalization: StateNormalization,
    g: nn.Module, d: nn.Module, training_seed: int,
    *, device: torch.device | str = "cpu",
) -> dict[str, float]:
    """Compute validation RTG1-Z/Y and D metrics from terminal heads."""
    if not records:
        raise ValueError("validation state records must not be empty")
    g.to(device).eval()
    d.to(device).eval()
    rows = []
    with torch.no_grad():
        for record in records:
            persisted = record.persistence_target
            pre_raw = record.pre_state.to(device)
            pre = normalization.normalize_pre(pre_raw)
            true_next = normalization.normalize_next(record.next_state.to(device))
            frame = record.fixed_frame.to(device)
            predicted = g(torch.cat((pre, frame))).reshape(28)
            physical_logits = d(predicted).reshape(485)
            group_nll, group_predictions = _group_values(physical_logits, record.physical_target)
            true_logits = d(true_next).reshape(485)
            d_group_nll, d_group_predictions = _group_values(true_logits, record.physical_target)
            persistence_nll = []
            for index, classes in enumerate(PHYSICAL_CARDINALITIES):
                if len(persisted) != 11 or persisted[index] not in range(classes):
                    raise ValueError("physical persistence target is invalid")
                probability = 1 - (classes - 1) * 1e-4 if persisted[index] == record.physical_target[index] else 1e-4
                persistence_nll.append(-math.log(probability))
            row = _identity(record, training_seed)
            row.update({
                "predicted_state": predicted.cpu().tolist(),
                "true_state": true_next.cpu().tolist(),
                "persistence_state": ((pre_raw - normalization.next_mean.to(device)) / normalization.next_std.to(device)).cpu().tolist(),
                "group_nll": group_nll,
                "persistence_nll": persistence_nll,
                "group_predictions": group_predictions,
                "persistence_predictions": list(persisted),
                "group_targets": list(record.physical_target),
                "d_group_nll": d_group_nll,
                "d_group_predictions": d_group_predictions,
            })
            rows.append(row)
    result = transition_state_metrics(rows)
    result.update({
        "nll_dg": consequence_nll(rows),
        "nll_physical_persistence": consequence_nll(rows, "persistence_nll"),
        "macro_accuracy_dg": consequence_macro_accuracy(rows),
        "macro_accuracy_physical_persistence": consequence_macro_accuracy(
            rows, prediction_field="persistence_predictions"
        ),
        "nll_d_true_next": consequence_nll(rows, "d_group_nll"),
        "macro_accuracy_d_true_next": consequence_macro_accuracy(
            rows, prediction_field="d_group_predictions"
        ),
    })
    return result


def extract_calibration_scores(
    records: tuple[CandidateStateRecord, ...], normalization: StateNormalization,
    g: nn.Module, d: nn.Module, c: nn.Module, training_seed: int,
    *, failure_delay: int, device: torch.device | str = "cpu",
) -> tuple[dict[str, object], ...]:
    """Extract finite G smoothed logits and raw C logits; do not fit on tests."""
    g.to(device).eval(); d.to(device).eval(); c.to(device).eval()
    output = []
    with torch.no_grad():
        for record in records:
            if unsafe_predicate(record.physical_target, failure_delay) != record.unsafe:
                raise ValueError("calibration label differs from frozen unsafe predicate")
            pre = normalization.normalize_pre(record.pre_state.to(device))
            frame = record.fixed_frame.to(device)
            combined = torch.cat((pre, frame))
            physical_logits = d(g(combined)).reshape(485)
            _, hits = estimate_g_risk(
                physical_logits, split_id=record.split_id,
                training_seed=training_seed, world_id=record.world_id,
                episode_id=record.episode_id, t=record.t,
                action_index=record.action_index, failure_delay=failure_delay,
            )
            c_logit = float(c(combined).reshape(()).cpu())
            if not math.isfinite(c_logit):
                raise FloatingPointError("non-finite C calibration score")
            row = _identity(record, training_seed)
            row.update({"split_id": record.split_id, "g_logit": smoothed_g_logit(hits),
                        "c_logit": c_logit, "unsafe": record.unsafe})
            output.append(row)
    return tuple(output)
