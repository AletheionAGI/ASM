"""Frozen SHA-256 common-random-number sampling for ATTR-RTG."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence

import torch

from .rtg_physical_targets import unsafe_predicate
from .rtg_types import Y_COMMON_CARDINALITIES

RISK_SAMPLES = 128
_RISK_DOMAIN = "ATTR-RTG-RISK-V1"


def crn_uniform(
    split_id: str | int,
    training_seed: int,
    world_id: str | int,
    episode_id: str | int,
    t: int,
    action_index: int,
    sample_index: int,
    group_index: int,
) -> float:
    """Return the registered architecture-independent open-interval uniform."""
    fields = (split_id, training_seed, world_id, episode_id, t, action_index,
              sample_index, group_index)
    if any("|" in str(field) for field in fields):
        raise ValueError("CRN key fields must not contain '|'")
    if any(type(value) is not int or value < 0 for value in
           (training_seed, t, action_index, sample_index, group_index)):
        raise ValueError("CRN numeric indices must be non-negative integers")
    key = "|".join((_RISK_DOMAIN, *(str(field) for field in fields)))
    integer = int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "big")
    uniform = (integer + 0.5) / 2**64
    return min(math.nextafter(1.0, 0.0), max(math.nextafter(0.0, 1.0), uniform))


def inverse_cdf(logits: torch.Tensor | Sequence[float], uniform: float) -> int:
    """Sample one category using CPU/float64 softmax and the first CDF value > u."""
    values = torch.as_tensor(logits, dtype=torch.float64, device="cpu")
    if values.ndim != 1 or values.numel() < 1 or not torch.isfinite(values).all():
        raise ValueError("categorical logits must be a finite, non-empty vector")
    if not 0.0 < uniform < 1.0 or not math.isfinite(uniform):
        raise ValueError("uniform must be finite and strictly between zero and one")
    cdf = torch.softmax(values, dim=0).cumsum(dim=0)
    category = int(torch.searchsorted(cdf, torch.tensor(uniform, dtype=torch.float64), right=True))
    if category >= values.numel():
        category = values.numel() - 1
    return category


def split_group_logits(logits: torch.Tensor | Sequence[float]) -> tuple[torch.Tensor, ...]:
    """Validate and split the frozen concatenated 485-logit physical schema."""
    values = torch.as_tensor(logits, dtype=torch.float64, device="cpu")
    if values.ndim != 1 or values.numel() != sum(Y_COMMON_CARDINALITIES):
        raise ValueError("D logits must be a flat vector with exactly 485 entries")
    return tuple(values.split(Y_COMMON_CARDINALITIES))


def estimate_g_risk(
    logits: torch.Tensor | Sequence[float],
    *,
    split_id: str | int,
    training_seed: int,
    world_id: str | int,
    episode_id: str | int,
    t: int,
    action_index: int,
    failure_delay: int,
    samples: int = RISK_SAMPLES,
) -> tuple[float, int]:
    """Return unsafe Monte Carlo risk and hits under the frozen joint sampler."""
    if samples != RISK_SAMPLES:
        raise ValueError("ATTR-RTG risk estimation requires exactly K=128 samples")
    if failure_delay not in {1, 3}:
        raise ValueError("failure_delay must be 1 or 3")
    try:
        groups = split_group_logits(logits)
        if not all(torch.isfinite(group).all() for group in groups):
            raise ValueError("non-finite logits")
    except (TypeError, ValueError, RuntimeError):
        return 1.0, samples
    hits = 0
    for sample_index in range(samples):
        categories = tuple(
            inverse_cdf(
                group,
                crn_uniform(split_id, training_seed, world_id, episode_id, t,
                            action_index, sample_index, group_index),
            )
            for group_index, group in enumerate(groups)
        )
        hits += int(unsafe_predicate(categories, failure_delay))
    return hits / samples, hits


def smoothed_g_logit(hits: int, *, samples: int = RISK_SAMPLES) -> float:
    """Convert G hits to the preregistered finite calibration logit."""
    if samples != RISK_SAMPLES or type(hits) is not int or not 0 <= hits <= samples:
        raise ValueError("hits must be an integer in [0, 128]")
    probability = (hits + 0.5) / (samples + 1)
    return math.log(probability / (1.0 - probability))
