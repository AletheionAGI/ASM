"""Build exact official contrasts from episode-level sufficient values."""

from __future__ import annotations

from typing import Any

from .constants import ARMS, CONTRASTS, REGIMES, TRAINING_SEEDS

ENDPOINTS = ("h8_nll", "unsafe_selection", "ece", "safe_service", "coverage")


def contrast_rows(metric_rows: list[dict[str, Any]]) -> list[dict[str, object]]:
    """Bootstrap each available endpoint from its raw hierarchical cells.

    Endpoint availability is independent. A missing endpoint fails its gates
    closed without discarding bounds for the other endpoints.
    """
    import torch

    from .bootstrap import paired_bootstrap, simultaneous_bounds
    from .gates import contrast_gate

    lookup = {(row["arm"], row["seed"], row["regime"]): row for row in metric_rows}
    expected = {
        (arm, seed, regime)
        for arm in ARMS
        for seed in TRAINING_SEEDS
        for regime in REGIMES
    }
    if set(lookup) != expected:
        raise RuntimeError("contrast input requires exactly all arm/seed/regime cells")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    endpoint_results: dict[str, dict[str, tuple[object, object, object]]] = {}
    availability: dict[str, set[str]] = {}
    for endpoint in ENDPOINTS:
        available = {
            arm
            for arm in ARMS
            if all(
                _has_endpoint(lookup[(arm, seed, regime)], endpoint)
                for seed in TRAINING_SEEDS
                for regime in REGIMES
            )
        }
        availability[endpoint] = available
        if not available:
            endpoint_results[endpoint] = {}
            continue
        cells = _canonical_endpoint_cells(lookup, endpoint, available)
        worlds = sorted({cell[0] for cell in cells})
        episodes = sorted({cell[1] for cell in cells})
        wi = {value: index for index, value in enumerate(worlds)}
        ei = {value: index for index, value in enumerate(episodes)}
        values = {
            arm: _endpoint_tensor(
                lookup, arm, endpoint, worlds, episodes, wi, ei, device
            )
            for arm in available
        }
        # paired_bootstrap requires the complete arm family. Filling an absent
        # arm only supplies unused contrasts; availability below still governs.
        template = next(iter(values.values()))
        bootstrap_values = {
            arm: values.get(arm, template.clone()) for arm in ARMS
        }
        replicas = paired_bootstrap(bootstrap_values, device=device)
        computed = {}
        for left, right in CONTRASTS:
            if left not in available or right not in available:
                continue
            lower, upper = simultaneous_bounds(replicas[f"{left}-{right}"])
            marginal = _seed_marginals(values[left] - values[right])
            computed[f"{left}-{right}"] = (lower, upper, marginal)
        endpoint_results[endpoint] = computed

    output = []
    for left, right in CONTRASTS:
        contrast = f"{left}-{right}"
        lower = [[None] * len(ENDPOINTS) for _ in REGIMES]
        upper = [[None] * len(ENDPOINTS) for _ in REGIMES]
        missing = []
        marginals = []
        for ki, endpoint in enumerate(ENDPOINTS):
            result = endpoint_results[endpoint].get(contrast)
            if result is None:
                missing.append(endpoint)
                marginals.append(None)
                continue
            lo, hi, raw = result
            for ri in range(len(REGIMES)):
                lower[ri][ki] = float(lo[ri].item())
                upper[ri][ki] = float(hi[ri].item())
            marginals.append(raw)
        if missing:
            gate_values = (False, False, False)
        else:
            lower_tensor = torch.tensor(lower, dtype=torch.float64, device=device)
            upper_tensor = torch.tensor(upper, dtype=torch.float64, device=device)
            marginal_tensor = torch.stack(marginals, dim=-1)
            gate = contrast_gate(lower_tensor, upper_tensor, marginal_tensor)
            gate_values = (gate.passed, gate.bounds_pass, gate.marginals_pass)
        row: dict[str, object] = {
            "status": "INVALID" if missing else "VALID",
            "kind": "contrast",
            "contrast": contrast,
            "lower": lower,
            "upper": upper,
            "passed": gate_values[0],
            "bounds_pass": gate_values[1],
            "marginals_pass": gate_values[2],
        }
        if missing:
            row["reason"] = "dependent endpoint unavailable: " + ",".join(missing)
        output.append(row)
    return output


def _has_endpoint(row: dict[str, Any], endpoint: str) -> bool:
    sufficient = row.get("_sufficient")
    return isinstance(sufficient, dict) and isinstance(sufficient.get(endpoint), list)


def _canonical_endpoint_cells(lookup, endpoint: str, arms: set[str]):
    canonical = []
    for arm in ARMS:
        if arm not in arms:
            continue
        for seed in TRAINING_SEEDS:
            for regime in REGIMES:
                rows = lookup[(arm, seed, regime)]["_sufficient"][endpoint]
                cells = {(item["world"], item["episode"]) for item in rows}
                if not cells or len(cells) != len(rows):
                    raise RuntimeError(f"invalid raw sufficient values for {endpoint}")
                worlds = {cell[0] for cell in cells}
                episodes = {cell[1] for cell in cells}
                if cells != {
                    (world, episode) for world in worlds for episode in episodes
                }:
                    raise RuntimeError(
                        "canonical cells must form a complete world/episode grid"
                    )
                canonical.append(cells)
    first = canonical[0]
    if any(cells != first for cells in canonical[1:]):
        raise RuntimeError("endpoint canonical cells differ")
    return first


def _endpoint_tensor(lookup, arm, endpoint, worlds, episodes, wi, ei, device):
    import torch

    value = torch.empty(
        (len(TRAINING_SEEDS), len(REGIMES), len(worlds), len(episodes)),
        dtype=torch.float64,
        device=device,
    )
    for si, seed in enumerate(TRAINING_SEEDS):
        for ri, regime in enumerate(REGIMES):
            for item in lookup[(arm, seed, regime)]["_sufficient"][endpoint]:
                value[si, ri, wi[item["world"]], ei[item["episode"]]] = item["value"]
    if not bool(torch.isfinite(value).all()):
        raise RuntimeError(f"nonfinite contrast sufficient value for {endpoint}")
    return value


def strip_sufficient(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        row.pop("_sufficient", None)


def _seed_marginals(difference):
    """Equal episode within world, then equal world; retain seed/regime."""
    from .bootstrap import _balanced_sum

    episode_fold = _balanced_sum(difference, 3) / difference.shape[3]
    return _balanced_sum(episode_fold, 2) / difference.shape[2]
