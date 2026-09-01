import torch

from aletheion_state_models.synthetic.phase2_benchmark import (
    Phase2RunResult,
    build_phase2_variant,
    summarize_phase2,
    train_phase2_run,
)


def test_fixed_phase2_control_has_exact_rank():
    model = build_phase2_variant("vr_fixed_low", 17).eval()
    tokens = torch.randint(0, model.config.vocab_size, (6, 7))
    output = model(tokens, collect_diagnostics=False)

    assert torch.all(output["variable_rank_ranks"] == 4)
    masks = output["variable_rank_masks"]
    assert torch.all(masks[..., :4])
    assert not torch.any(masks[..., 4:])


def test_phase2_benchmark_smoke_is_finite():
    result = train_phase2_run(
        "vr_fixed_mid", 17, steps=2, batch_size=8, evaluation_batches=1
    )

    assert result.finite
    assert result.mean_rank == 8.0
    assert 0.0 <= result.validation_accuracy <= 1.0


def _result(variant: str, seed: int, **overrides) -> Phase2RunResult:
    values = dict(
        variant=variant,
        seed=seed,
        steps=10,
        validation_loss=1.0,
        validation_accuracy=0.50,
        low_accuracy=0.50,
        high_accuracy=0.50,
        mean_rank=8.0,
        rank_std=2.0,
        low_rank=4.0,
        high_rank=12.0,
        rank_difficulty_correlation=1.0,
        controller_gradient_fraction=1.0,
        seconds=1.0,
        finite=True,
    )
    values.update(overrides)
    return Phase2RunResult(**values)


def test_phase2_summary_requires_multiseed_adaptation_and_quality():
    results = []
    for seed in (17, 29, 43):
        results.append(_result("vr_adaptive", seed))
        results.append(
            _result(
                "vr_fixed_mid",
                seed,
                rank_std=0.0,
                rank_difficulty_correlation=0.0,
                controller_gradient_fraction=0.0,
            )
        )
    summary = summarize_phase2(results)

    assert summary["passed"]
    assert all(summary["gates"].values())
    assert summary["claims"]["hardware_speedup"] is False
