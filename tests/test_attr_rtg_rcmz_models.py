"""Synthetic-only fast-path receipts for ATTR-RTG-RCMZ-V1 models."""

from __future__ import annotations

import pytest
import torch

from attr_rtg_rcmz.config import load_config, registered_config_paths
from attr_rtg_rcmz.contracts import MESSAGE_FIELDS, FourFieldInference, InferenceMessage
from attr_rtg_rcmz.models import build_adapter
from attr_rtg_rcmz.z import audit_strict_z
from drm_language_emitter.asm_z_core import ASMZCore, ScalarPotential

ROOT = "."


def message(lengths=(3, 5), *, padding_value=0):
    history = torch.full((len(lengths), 256), padding_value, dtype=torch.long)
    for row, length in enumerate(lengths):
        history[row, :length] = torch.arange(length) + row
    return InferenceMessage(
        history_bytes=history,
        candidate4s=torch.randint(0, 8, (len(lengths), 6, 4)).float(),
        masks=torch.ones(len(lengths), 6, dtype=torch.bool),
        logical_lengths=torch.tensor(lengths),
    )


def test_exactly_twenty_complete_configs_and_matched_counts():
    paths = registered_config_paths(ROOT)
    assert len(paths) == 20 and len(set(paths)) == 20
    counts = []
    for path in paths:
        config = load_config(path)
        assert config.context_length == 256
        model = build_adapter(config)
        counts.append(model.trainable_parameter_count())
    assert set(counts) == {50_000}
    assert (max(counts) - min(counts)) / min(counts) <= 0.001


def test_all_trainable_parameters_are_graph_active():
    for arm in ("r", "cm", "z", "t"):
        model = build_adapter(f"configs/attr_rtg_rcmz_v1/{arm}_seed29.yaml")
        result = model(message())
        result.logits.sum().backward()
        assert model.graph_active_parameter_count() == 50_000
        assert all(parameter.grad is not None for parameter in model.parameters())


def test_exact_four_field_api_and_arm_is_bound_outside_message():
    model = build_adapter("configs/attr_rtg_rcmz_v1/r_seed29.yaml")
    process = FourFieldInference(model)
    msg = message((4,))
    mapping = {name: getattr(msg, name) for name in MESSAGE_FIELDS}
    result = process(mapping)
    assert result.logits.shape == (1, 6)
    assert result.common24.shape == (1, 6, 24)
    assert result.native_state.shape == (1, 6, 28)
    with pytest.raises(ValueError, match="exactly"):
        process({**mapping, "arm": "R"})
    with pytest.raises(TypeError):
        model(mapping)


def test_right_padding_does_not_change_exports_or_scores():
    torch.manual_seed(3)
    first = message((7,), padding_value=0)
    second = InferenceMessage(
        history_bytes=first.history_bytes.clone(),
        candidate4s=first.candidate4s,
        masks=first.masks,
        logical_lengths=first.logical_lengths,
    )
    second.history_bytes[:, 7:] = 255
    for arm in ("r", "cm", "z", "t"):
        model = build_adapter(f"configs/attr_rtg_rcmz_v1/{arm}_seed29.yaml").eval()
        with torch.no_grad():
            first_result, second_result = model(first), model(second)
        assert torch.equal(first_result.native_state, second_result.native_state)
        assert torch.equal(first_result.common24, second_result.common24)
        assert torch.equal(first_result.logits, second_result.logits)


def test_history_schedule_has_no_data_dependent_host_extraction():
    import inspect

    from attr_rtg_rcmz.adapter import RiskAdapter

    source = inspect.getsource(RiskAdapter._history_snapshot)
    assert "torch.unique" not in source
    assert ".tolist" not in source
    assert ".item" not in source
    assert "range(self.config.context_length)" in source


def test_history_uses_one_fixed_pass_and_candidates_fork_after_snapshot():
    model = build_adapter("configs/attr_rtg_rcmz_v1/r_seed29.yaml")
    calls = []
    hook = model.backbone.token_embedding.register_forward_pre_hook(
        lambda _m, args: calls.append(args[0].shape)
    )
    model(message((5, 4)))
    hook.remove()
    assert calls == [torch.Size([2, 256]), torch.Size([12, 4])]


def test_six_forks_match_independent_candidates_and_candidates_influence_state():
    for arm in ("r", "cm", "z", "t"):
        model = build_adapter(f"configs/attr_rtg_rcmz_v1/{arm}_seed29.yaml").eval()
        full = message((5,))
        with torch.no_grad():
            combined = model(full)
            for candidate in range(6):
                isolated = InferenceMessage(
                    history_bytes=full.history_bytes,
                    candidate4s=full.candidate4s[:, candidate : candidate + 1]
                    .expand(-1, 6, -1)
                    .clone(),
                    masks=full.masks,
                    logical_lengths=full.logical_lengths,
                )
                single = model(isolated)
                assert torch.allclose(
                    combined.native_state[:, candidate], single.native_state[:, 0]
                )
                assert torch.allclose(
                    combined.common24[:, candidate], single.common24[:, 0]
                )
                assert torch.allclose(
                    combined.logits[:, candidate], single.logits[:, 0]
                )
        changed = InferenceMessage(
            full.history_bytes,
            full.candidate4s.clone(),
            full.masks,
            full.logical_lengths,
        )
        changed.candidate4s[:, 0] = torch.tensor(
            [251, 252, 253, 254], dtype=torch.float32
        )
        with torch.no_grad():
            altered = model(changed)
        assert not torch.equal(combined.native_state[:, 0], altered.native_state[:, 0])
        assert torch.equal(combined.native_state[:, 1:], altered.native_state[:, 1:])


def test_strict_z_potential_metric_and_single_step(monkeypatch):
    adapter = build_adapter("configs/attr_rtg_rcmz_v1/z_seed29.yaml")
    audit_strict_z(adapter)
    core = next(module for module in adapter.modules() if isinstance(module, ASMZCore))
    assert isinstance(core.potential, ScalarPotential)
    state = torch.randn(2, 28)
    token = torch.randn(2, 28)
    with torch.no_grad():
        for parameter in core.potential.net.parameters():
            parameter.zero_()
    expected = 0.5 * adapter.config.z_lambda * state.square().sum(-1)
    assert torch.allclose(core.potential(state, token), expected)
    import drm_language_emitter.asm_z_core as z_core

    calls = []
    original = z_core.solve_spd_metric

    def counted(diagonal, low_rank, gradient):
        calls.append((diagonal, low_rank))
        return original(diagonal, low_rank, gradient)

    monkeypatch.setattr(z_core, "solve_spd_metric", counted)
    next_state, geometry = core(state, token)
    assert len(calls) == 1
    assert torch.allclose(
        geometry.metric,
        torch.diag_embed(geometry.diagonal)
        + geometry.low_rank @ geometry.low_rank.transpose(-1, -2),
    )
    solved = original(geometry.diagonal, geometry.low_rank, geometry.gradient)
    assert torch.allclose(next_state, state - adapter.config.z_eta * solved)
    assert adapter.backbone.config.n_flow_steps == 1
