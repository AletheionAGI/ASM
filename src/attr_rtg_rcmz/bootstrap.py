"""Paired hierarchical Bayesian bootstrap with shared Exp(1) weights."""

from __future__ import annotations

import hashlib

import torch

from .constants import BOOTSTRAP_REPLICATES, CONTRASTS
from .quantiles import type7
from .validation import fp64, statistics_device


def _lp(value: str) -> bytes:
    raw = value.encode("utf-8")
    return len(raw).to_bytes(8, "big") + raw


def bootstrap_seed64() -> int:
    payload = _lp("ATTR-RTG-RCMZ-V1") + _lp("bootstrap") + _lp("0")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _exp(shape, generator, device):
    return torch.empty(shape, dtype=torch.float64, device=device).exponential_(
        1, generator=generator
    )


def _balanced_sum(values: torch.Tensor, dim: int) -> torch.Tensor:
    """Ascending-index binary-tree sum padded with identity zero."""
    x = values.movedim(dim, -1)
    target = 1 << (x.shape[-1] - 1).bit_length()
    if target != x.shape[-1]:
        x = torch.nn.functional.pad(x, (0, target - x.shape[-1]))
    while x.shape[-1] > 1:
        x = x[..., 0::2] + x[..., 1::2]
    return x[..., 0]


def hierarchical_weights(
    shape: tuple[int, int, int, int],
    *,
    replicates=BOOTSTRAP_REPLICATES,
    device=None,
    seed=None,
):
    """Return normalized seed/world/episode weights for S,R,W,E."""
    s, r, w, e = shape
    device = torch.device(device or statistics_device())
    generator = torch.Generator(device=device)
    generator.manual_seed(seed or bootstrap_seed64())
    seed_w = _exp((replicates, s), generator, device)
    seed_w /= _balanced_sum(seed_w, -1).unsqueeze(-1)
    world_w = _exp((replicates, s, r, w), generator, device)
    world_w /= _balanced_sum(world_w, -1).unsqueeze(-1)
    episode_w = _exp((replicates, s, r, w, e), generator, device)
    episode_w /= _balanced_sum(episode_w, -1).unsqueeze(-1)
    if not all(bool(torch.isfinite(x).all()) for x in (seed_w, world_w, episode_w)):
        raise ValueError("nonfinite bootstrap weight")
    return seed_w, world_w, episode_w


def paired_bootstrap(
    endpoints: dict[str, object],
    *,
    eligible=None,
    replicates=BOOTSTRAP_REPLICATES,
    device=None,
    seed=None,
):
    """Bootstrap endpoints shaped S,R,W,E or S,R,W,E,K.

    All arms share raw weights at matching canonical indices. Eligibility must be
    common to every arm; partial clusters fail closed rather than renormalize.
    """
    device = torch.device(device or statistics_device())
    values = {arm: fp64(value, device=device) for arm, value in endpoints.items()}
    required = {arm for pair in CONTRASTS for arm in pair}
    if required - values.keys():
        raise ValueError("all four arms are required")
    shape = values["R"].shape
    if len(shape) not in (4, 5) or any(x.shape != shape for x in values.values()):
        raise ValueError("endpoint shape must be common S,R,W,E[,K]")
    cluster_shape = shape[:4]
    if eligible is not None:
        mask = torch.as_tensor(eligible, dtype=torch.bool, device=device)
        if mask.shape != cluster_shape or not bool(mask.all()):
            raise ValueError("empty required bootstrap cell")
    sw, ww, ew = hierarchical_weights(
        cluster_shape, replicates=replicates, device=device, seed=seed
    )
    output = {}
    for a, b in CONTRASTS:
        difference = (values[a] - values[b]).unsqueeze(0)
        episode_weight = ew[..., None] if len(shape) == 5 else ew
        episode_fold = _balanced_sum(episode_weight * difference, 4)
        world_weight = ww[..., None] if len(shape) == 5 else ww
        world_fold = _balanced_sum(world_weight * episode_fold, 3)
        seed_weight = sw[:, :, None, None] if len(shape) == 5 else sw[:, :, None]
        output[f"{a}-{b}"] = _balanced_sum(seed_weight * world_fold, 1)
    return output


def simultaneous_bounds(replicates: torch.Tensor):
    reps = fp64(replicates)
    if reps.shape[0] != BOOTSTRAP_REPLICATES:
        raise ValueError("exactly 1000 replicates required")
    flat = reps.reshape(reps.shape[0], -1)
    lower = torch.stack([type7(flat[:, i], 1 / 120) for i in range(flat.shape[1])])
    upper = torch.stack([type7(flat[:, i], 119 / 120) for i in range(flat.shape[1])])
    return lower.reshape(reps.shape[1:]), upper.reshape(reps.shape[1:])
