"""Contract tests for the classifier-free ATTR-TG1 trajectory model."""

from pathlib import Path

import torch

from aletheion_state_models.benchmarks.transition_risk.trajectory_head import (
    FIELD_CARDINALITIES,
    TrajectoryHead,
)
from aletheion_state_models.benchmarks.transition_risk.trajectory_models import (
    TG1_TRAINING_SEEDS,
    build_trajectory_arm,
)
from aletheion_state_models.benchmarks.transition_risk.trajectory_training import (
    trajectory_loss,
)


def _physical_batch(batch_size=2, steps=3):
    generator = torch.Generator().manual_seed(7)
    targets = {
        name: torch.randint(size, (batch_size, steps, 8), generator=generator)
        for name, size in FIELD_CARDINALITIES.items()
    }
    valid_mask = torch.ones(batch_size, steps, 8, dtype=torch.bool)
    return {
        "step_mask": torch.ones(batch_size, steps, dtype=torch.bool),
        "plan_actions": torch.randint(7, (batch_size, steps, 8), generator=generator),
        "trap_cells": torch.randint(81, (batch_size, steps, 3), generator=generator),
        "targets": targets,
        "valid_mask": valid_mask,
    }


def test_head_has_only_physical_outputs_and_finite_gradients():
    head = TrajectoryHead()
    batch = _physical_batch()
    context = torch.randn(2, 3, 72)
    predictions = head(context, batch["plan_actions"], batch["targets"])
    assert set(predictions) == {"trap_cells", *FIELD_CARDINALITIES}
    forbidden = {"unsafe", "severity", "time_to_hazard", "hazard_logits"}
    assert not forbidden.intersection(predictions)
    losses = trajectory_loss(predictions, batch)
    assert torch.isfinite(losses.total)
    losses.total.backward()
    gradients = [
        parameter.grad for parameter in head.parameters() if parameter.grad is not None
    ]
    assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)
    assert any(gradient.abs().sum() > 0 for gradient in gradients)


def test_free_running_inverse_cdf_sample_shapes():
    head = TrajectoryHead().eval()
    batch = _physical_batch()
    context = torch.randn(2, 3, 72)
    uniforms = {"trap_cells": torch.rand(2, 3, 3)} | {
        name: torch.rand(2, 3, 8) for name in FIELD_CARDINALITIES
    }
    sampled = head.sample(context, batch["plan_actions"], uniforms)
    assert sampled["trap_cells"].shape == (2, 3, 3)
    assert all(sampled[name].shape == (2, 3, 8) for name in FIELD_CARDINALITIES)


def test_registered_arms_share_exact_head_and_p2_optimizer_seeds():
    root = Path(__file__).resolve().parents[1]
    asm = build_trajectory_arm(root, "asm_x_base", seed=29)[0].head
    transformer = build_trajectory_arm(root, "transformer_base", seed=29)[0].head
    assert sum(p.numel() for p in asm.parameters()) == sum(
        p.numel() for p in transformer.parameters()
    )
    assert all(
        torch.equal(asm.state_dict()[name], transformer.state_dict()[name])
        for name in asm.state_dict()
    )
    assert TG1_TRAINING_SEEDS == (29, 43, 71, 89, 107)
