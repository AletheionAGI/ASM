"""Run the integrated ASM-VR Phase 1 acceptance experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from aletheion_state_models.variants import build_variable_rank_phase1
from drm_language_emitter import DRMConfig


def _config(seed: int) -> DRMConfig:
    return DRMConfig(
        vocab_size=19,
        d_token=8,
        d_state=8,
        n_directions=4,
        metric_rank=2,
        hidden_size=12,
        sequence_mode="directional_block_cumsum",
        directional_cumsum_step_mode="velocity",
        directional_cumsum_block_size=2,
        bounded_state=False,
        seed=seed,
    )


def _set_rank(model, rank: int) -> None:
    controller = model.variable_rank_core.controller
    with torch.no_grad():
        controller.score_head.weight.zero_()
        controller.score_head.bias.fill_(-20.0)
        controller.score_head.bias[:rank].fill_(20.0)


def _initial_pair() -> torch.Tensor:
    return torch.tensor(
        [
            [1.0, 2.0, 3.0, 10.0, 20.0, 30.0, 40.0, 50.0],
            [1.0, 2.0, 3.0, -10.0, -20.0, -30.0, -40.0, -50.0],
        ]
    )


def run(seed: int) -> dict[str, object]:
    torch.manual_seed(seed)
    model = build_variable_rank_phase1(_config(seed)).eval()
    _set_rank(model, 3)
    tokens = torch.randint(0, model.config.vocab_size, (1, 6)).expand(2, -1)
    embeddings = model.token_embedding(tokens)
    paired = model._forward_directional_cumsum(
        _initial_pair(), embeddings, None, True, None, False
    )
    logit_difference = torch.max(torch.abs(paired["logits"][0] - paired["logits"][1])).item()
    state_difference = torch.max(torch.abs(paired["states"][0] - paired["states"][1])).item()

    point = _initial_pair()[0].detach().requires_grad_(True)
    one_embedding = embeddings[:1]

    def future_logits(initial: torch.Tensor) -> torch.Tensor:
        result = model._forward_directional_cumsum(
            initial.unsqueeze(0), one_embedding, None, False, None, False
        )
        return result["logits"].flatten()

    jacobian = torch.autograd.functional.jacobian(future_logits, point, vectorize=True)
    complement_jacobian = torch.linalg.matrix_norm(jacobian[:, 3:]).item()

    stream_tokens = torch.randint(0, model.config.vocab_size, (2, 7))
    expected = model(stream_tokens, collect_diagnostics=False)["logits"]
    first, cache = model.prefill(stream_tokens[:, :1])
    observed = [first]
    maximum_cached_inactive = 0.0
    for position in range(1, stream_tokens.shape[1]):
        step, cache = model.decode_step(stream_tokens[:, position], cache)
        observed.append(step.unsqueeze(1))
        inactive = cache.variable_rank_state.effective_coordinates.masked_select(
            ~cache.variable_rank_state.active_mask
        )
        if inactive.numel():
            maximum_cached_inactive = max(
                maximum_cached_inactive, torch.max(torch.abs(inactive)).item()
            )
    streaming_error = torch.max(torch.abs(torch.cat(observed, dim=1) - expected)).item()
    passed = (
        logit_difference == 0.0
        and state_difference == 0.0
        and complement_jacobian == 0.0
        and maximum_cached_inactive == 0.0
        and streaming_error < 1e-6
        and cache.completed_state is None
        and cache.input_ids.numel() == 0
    )
    return {
        "passed": passed,
        "paired_logit_difference": logit_difference,
        "paired_state_difference": state_difference,
        "discarded_complement_jacobian_norm": complement_jacobian,
        "maximum_cached_inactive_coordinate": maximum_cached_inactive,
        "streaming_maximum_error": streaming_error,
        "cache_retains_ambient_state": cache.completed_state is not None,
        "cache_retains_prefix": cache.input_ids.numel() != 0,
        "controller_policy": "first-block-token-only",
        "frame": "fixed-identity",
        "transition_memory_enabled": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run(args.seed)
    payload = json.dumps(result, indent=2, sort_keys=True)
    print(payload)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
