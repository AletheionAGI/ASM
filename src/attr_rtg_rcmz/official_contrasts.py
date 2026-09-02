"""Build exact official contrasts from episode-level sufficient values."""

from __future__ import annotations

from typing import Any

from .constants import ARMS, CONTRASTS, REGIMES, TRAINING_SEEDS

ENDPOINTS = ("h8_nll", "unsafe_selection", "ece", "safe_service", "coverage")


def contrast_rows(metric_rows: list[dict[str, Any]]) -> list[dict[str, object]]:
    """Bootstrap raw seed/world/episode values, never published summaries."""
    import copy

    import torch

    from .bootstrap import paired_bootstrap, simultaneous_bounds
    from .gates import contrast_gate

    invalid_arms = {
        row.get("arm") for row in metric_rows if row.get("status") != "VALID"
    }
    if invalid_arms:
        valid_arms = [arm for arm in ARMS if arm not in invalid_arms]
        if valid_arms:
            template = valid_arms[0]
            source = {
                (row["arm"], row["seed"], row["regime"]): row for row in metric_rows
            }
            repaired = [
                row for row in metric_rows if row.get("arm") not in invalid_arms
            ]
            for arm in invalid_arms:
                for seed in TRAINING_SEEDS:
                    for regime in REGIMES:
                        item = copy.deepcopy(source[(template, seed, regime)])
                        item["arm"] = arm
                        repaired.append(item)
            computed = contrast_rows(repaired)
        else:
            computed = [{"contrast": f"{left}-{right}"} for left, right in CONTRASTS]
        for item in computed:
            left, right = item["contrast"].split("-")
            if left in invalid_arms or right in invalid_arms:
                item.update(
                    {
                        "status": "INVALID",
                        "reason": "dependent metric cell INVALID",
                        "lower": None,
                        "upper": None,
                        "passed": False,
                        "bounds_pass": False,
                        "marginals_pass": False,
                    }
                )
        return computed

    lookup = {(row["arm"], row["seed"], row["regime"]): row for row in metric_rows}
    expected = {
        (arm, seed, regime)
        for arm in ARMS
        for seed in TRAINING_SEEDS
        for regime in REGIMES
    }
    if set(lookup) != expected:
        raise RuntimeError("contrast input requires exactly all arm/seed/regime cells")
    canonical = _canonical_cells(lookup)
    worlds = sorted({cell[0] for cells in canonical.values() for cell in cells})
    episodes = sorted({cell[1] for cells in canonical.values() for cell in cells})
    wi, ei = (
        {value: index for index, value in enumerate(worlds)},
        {value: index for index, value in enumerate(episodes)},
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    values = {
        arm: torch.empty(
            (5, 3, len(worlds), len(episodes), 5), dtype=torch.float64, device=device
        )
        for arm in ARMS
    }
    for ai, arm in enumerate(ARMS):
        del ai
        for si, seed in enumerate(TRAINING_SEEDS):
            for ri, regime in enumerate(REGIMES):
                row = lookup[(arm, seed, regime)]
                for ki, endpoint in enumerate(ENDPOINTS):
                    for item in row["_sufficient"][endpoint]:
                        values[arm][
                            si, ri, wi[item["world"]], ei[item["episode"]], ki
                        ] = item["value"]
    if not all(bool(torch.isfinite(value).all()) for value in values.values()):
        raise RuntimeError("nonfinite contrast sufficient value")
    replicas = paired_bootstrap(values, device=device)
    output = []
    for left, right in CONTRASTS:
        lower, upper = simultaneous_bounds(replicas[f"{left}-{right}"])
        marginals = _seed_marginals(values[left] - values[right])
        gate = contrast_gate(lower, upper, marginals)
        output.append(
            {
                "status": "VALID",
                "kind": "contrast",
                "contrast": f"{left}-{right}",
                "lower": lower.cpu().tolist(),
                "upper": upper.cpu().tolist(),
                "passed": gate.passed,
                "bounds_pass": gate.bounds_pass,
                "marginals_pass": gate.marginals_pass,
            }
        )
    return output


def strip_sufficient(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        row.pop("_sufficient", None)


def _canonical_cells(lookup):
    canonical = {}
    for key, row in lookup.items():
        sufficient = row.get("_sufficient")
        if not isinstance(sufficient, dict) or set(ENDPOINTS) - sufficient.keys():
            raise RuntimeError("missing raw sufficient values")
        endpoint_cells = [
            {(item["world"], item["episode"]) for item in sufficient[name]}
            for name in ENDPOINTS
        ]
        duplicate = any(
            len(cells) != len(sufficient[name])
            for cells, name in zip(endpoint_cells, ENDPOINTS, strict=True)
        )
        if (
            not endpoint_cells[0]
            or duplicate
            or any(cells != endpoint_cells[0] for cells in endpoint_cells[1:])
        ):
            raise RuntimeError("endpoint canonical cells differ")
        worlds = {cell[0] for cell in endpoint_cells[0]}
        episodes = {cell[1] for cell in endpoint_cells[0]}
        if endpoint_cells[0] != {
            (world, episode) for world in worlds for episode in episodes
        }:
            raise RuntimeError(
                "canonical cells must form a complete world/episode grid"
            )
        canonical[key] = endpoint_cells[0]
    first = next(iter(canonical.values()))
    if any(cells != first for cells in canonical.values()):
        raise RuntimeError("arm/seed/regime canonical cells differ")
    return canonical


def _seed_marginals(difference):
    """Equal episode within world, then equal world; retain seed/regime/endpoint."""
    from .bootstrap import _balanced_sum

    episode_fold = _balanced_sum(difference, 3) / difference.shape[3]
    return _balanced_sum(episode_fold, 2) / difference.shape[2]
