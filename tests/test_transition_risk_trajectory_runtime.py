from __future__ import annotations

from dataclasses import replace

import torch
from torch import nn

from aletheion_state_models.benchmarks.transition_risk.trajectory_evaluation import (
    TrajectoryIdentity,
)
from aletheion_state_models.benchmarks.transition_risk.trajectory_manifests import (
    default_splits,
)
from aletheion_state_models.benchmarks.transition_risk.trajectory_runtime import (
    crn_uniforms,
    dynamic_family,
    evaluate_free_running,
    generate_split,
)


def test_split_family_mapping_and_test_generation_gate():
    assert dynamic_family("common_fixed") == dynamic_family("id") == "baseline"
    test = next(item for item in default_splits() if item.name == "test_id")
    try:
        generate_split(replace(test, world_count=1, episodes_per_world=1))
    except PermissionError:
        pass
    else:
        raise AssertionError("test generation was not gated")


def test_crn_is_independent_of_arm_and_field_streams_differ():
    a = TrajectoryIdentity(29, "asm_x_base", "validation", "w", "e", 0)
    b = replace(a, arm="transformer_base")
    one = crn_uniforms(a, 1, 1, 4, device=torch.device("cpu"), dtype=torch.float32)
    two = crn_uniforms(b, 1, 1, 4, device=torch.device("cpu"), dtype=torch.float32)
    assert all(torch.equal(one[name], two[name]) for name in one)
    assert not torch.equal(one["agent_cell"], one["energy_bin"])


class _OracleHead(nn.Module):
    def sample(self, context, plans, uniforms):
        shape = (context.shape[0], context.shape[1], 8)
        zeros = torch.zeros(shape, dtype=torch.long, device=context.device)
        return {
            "trap_cells": torch.zeros(
                context.shape[0],
                context.shape[1],
                3,
                dtype=torch.long,
                device=context.device,
            ),
            "agent_cell": zeros,
            "moving_hazard_cell": zeros + 80,
            "velocity_row": zeros,
            "velocity_col": zeros,
            "energy_bin": zeros,
            "low_energy_steps": zeros,
            "recovery_left": zeros + 1,
            "hidden_mode": zeros,
            "safe_terminal": zeros,
        }

    def forward(self, context, plans, targets=None, teacher_forcing=None):
        b, t, h = plans.shape
        result = {"trap_cells": torch.zeros(b, t, 3, 81, device=context.device)}
        cards = {
            "agent_cell": 81,
            "moving_hazard_cell": 81,
            "velocity_row": 3,
            "velocity_col": 3,
            "energy_bin": 64,
            "low_energy_steps": 4,
            "recovery_left": 4,
            "hidden_mode": 3,
            "safe_terminal": 2,
        }
        result.update(
            {
                name: torch.zeros(b, t, h, card, device=context.device)
                for name, card in cards.items()
            }
        )
        return result


class _OracleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(()))
        self.head = _OracleHead()
        self.encode_calls = 0

    def encode_steps(self, input_ids, step_positions):
        self.encode_calls += 1
        return torch.zeros(
            input_ids.shape[0], step_positions.shape[1], 72, device=input_ids.device
        )

    def forward(
        self, input_ids, step_positions, plans, targets=None, teacher_forcing=None
    ):
        return self.head(
            self.encode_steps(input_ids, step_positions),
            plans,
            targets,
            teacher_forcing,
        )


def test_oracle_free_run_crn_and_zero_hazard_head_is_irrelevant():
    spec = replace(default_splits()[1], world_count=1, episodes_per_world=1)
    episodes = generate_split(spec)
    model = _OracleModel()
    records = evaluate_free_running(
        model, episodes, arm="asm_x_base", seed=29, split="validation", samples=4
    )
    # Duplicate predicted traps are invalid and unsafe, regardless of true trap config.
    assert records and all(
        row.risk_by_horizon == {1: 1.0, 4: 1.0, 8: 1.0} for row in records
    )
    # The causal backbone is encoded once and reused by MC and joint CE NLL.
    assert model.encode_calls == 1
    assert (
        "HazardHead"
        not in __import__(
            "aletheion_state_models.benchmarks.transition_risk.trajectory_runtime",
            fromlist=["dummy"],
        ).__dict__
    )
