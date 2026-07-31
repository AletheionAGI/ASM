import torch

from aletheion_state_models import StateModel
from aletheion_state_models.variants import (
    build_direct_state,
    build_explicit_drm,
    build_metric_subspace,
    build_relational_state,
    build_selective_state,
)
from drm_language_emitter.config import DRMConfig
from drm_language_emitter.model import DRMEmitterModel


def _base_config() -> DRMConfig:
    return DRMConfig(
        vocab_size=32,
        d_token=8,
        d_state=12,
        n_directions=4,
        metric_rank=2,
        hidden_size=16,
        max_seq_len=8,
        sequence_mode="directional_block_cumsum",
        directional_cumsum_step_mode="velocity",
        directional_cumsum_block_size=4,
        directional_local_mixer="none",
        bounded_state=False,
    )


def test_state_model_alias_preserves_checkpoint_compatible_class():
    assert StateModel is DRMEmitterModel


def test_asm_variant_builders_select_expected_components():
    base = _base_config()
    explicit = build_explicit_drm(base)
    subspace = build_metric_subspace(base)
    relational = build_relational_state(base)
    direct = build_direct_state(base)
    selective = build_selective_state(base, memory_hidden_size=10)

    assert explicit.direction_field is not None and explicit.metric is not None
    assert explicit.config.directional_metric_composition == "post_naturalize"
    assert subspace.direction_field is not None and subspace.metric is not None
    assert subspace.config.directional_metric_composition == "metric_subspace"
    assert relational.direction_field is None and relational.metric is not None
    assert relational.direct_transition is not None
    assert direct.direction_field is None and direct.metric is None
    assert direct.direct_transition is not None
    assert selective.metric is None and selective.selective_memory is not None
    assert selective.config.selective_memory_hidden_size == 10


def test_asm_variants_run_finite_causal_forwards():
    base = _base_config()
    tokens = torch.randint(0, base.vocab_size, (2, 8))
    for builder in (
        build_explicit_drm,
        build_metric_subspace,
        build_relational_state,
        build_direct_state,
        build_selective_state,
    ):
        model = builder(base)
        output = model(tokens, tokens, collect_diagnostics=False)
        assert torch.isfinite(output["loss"])
