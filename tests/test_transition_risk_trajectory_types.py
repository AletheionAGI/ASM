import pytest
import torch

from aletheion_state_models.benchmarks.transition_risk.trajectory_types import (
    PLAN_HORIZON,
    TARGET_CARDINALITIES,
    TRAINING_SEEDS,
    TrajectoryTargets,
)


def test_attr_tg1_contract_and_training_seeds_are_frozen():
    assert PLAN_HORIZON == 8
    assert TRAINING_SEEDS == (29, 43, 71, 89, 107)
    assert TARGET_CARDINALITIES == {
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


def test_targets_reject_wrong_categorical_ranges():
    values = {
        name: torch.zeros(2, 8, dtype=torch.long) for name in TARGET_CARDINALITIES
    }
    targets = TrajectoryTargets(**values)
    targets.validate((2, 8))
    values["safe_terminal"] = torch.full((2, 8), 2, dtype=torch.long)
    with pytest.raises(ValueError):
        TrajectoryTargets(**values).validate((2, 8))
