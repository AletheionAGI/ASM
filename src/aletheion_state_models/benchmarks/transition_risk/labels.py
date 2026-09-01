"""Causal multi-horizon labels for unsafe-set entry."""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Sequence

from .types import DEFAULT_HORIZONS, HorizonLabels


def validate_horizons(horizons: Iterable[int]) -> tuple[int, ...]:
    """Return unique, increasing, strictly positive horizons."""
    values = tuple(sorted(set(int(value) for value in horizons)))
    if not values or values[0] <= 0:
        raise ValueError("horizons must contain positive integers")
    return values


def unsafe_entry_steps(
    unsafe: Sequence[bool], episode_ids: Sequence[Hashable] | None = None
) -> tuple[bool, ...]:
    """Mark safe-to-unsafe entries without crossing episode boundaries."""
    if episode_ids is not None and len(episode_ids) != len(unsafe):
        raise ValueError("episode_ids and unsafe must have equal length")
    entries: list[bool] = []
    for index, value in enumerate(unsafe):
        new_episode = index == 0 or (
            episode_ids is not None and episode_ids[index] != episode_ids[index - 1]
        )
        previously_unsafe = False if new_episode else bool(unsafe[index - 1])
        entries.append(bool(value) and not previously_unsafe)
    return tuple(entries)


def multi_horizon_labels(
    unsafe: Sequence[bool],
    horizons: Iterable[int] = DEFAULT_HORIZONS,
    episode_ids: Sequence[Hashable] | None = None,
) -> tuple[HorizonLabels, ...]:
    """Build ``y_t(H)`` labels using only future entries in ``(t, t+H]``.

    Windows are clipped at episode boundaries. An unsafe state at ``t`` is not
    itself a future entry and therefore cannot label the same timestep.
    """
    checked = validate_horizons(horizons)
    entries = unsafe_entry_steps(unsafe, episode_ids)
    output: list[HorizonLabels] = []
    for step in range(len(unsafe)):
        episode = None if episode_ids is None else episode_ids[step]
        labels: dict[int, int] = {}
        for horizon in checked:
            end = min(len(unsafe), step + horizon + 1)
            future = range(step + 1, end)
            labels[horizon] = int(
                any(
                    entries[index]
                    and (episode_ids is None or episode_ids[index] == episode)
                    for index in future
                )
            )
        output.append(HorizonLabels(step=step, labels=labels, episode_id=episode))
    return tuple(output)


def labels_by_horizon(
    unsafe: Sequence[bool],
    horizons: Iterable[int] = DEFAULT_HORIZONS,
    episode_ids: Sequence[Hashable] | None = None,
) -> dict[int, tuple[int, ...]]:
    """Return labels in the convenient ``horizon -> time series`` form."""
    rows = multi_horizon_labels(unsafe, horizons, episode_ids)
    checked = validate_horizons(horizons)
    return {horizon: tuple(row[horizon] for row in rows) for horizon in checked}


# Explicit alias used by benchmark orchestration code.
build_multi_horizon_labels = labels_by_horizon
