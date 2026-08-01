import pytest

from drm_language_emitter.config import DRMConfig


def test_config_rejects_unknown_keys():
    with pytest.raises(ValueError, match="unknown DRMConfig field"):
        DRMConfig.from_dict({"vocab_size": 32, "not_a_real_field": True})


def test_config_validates_risk_exponent_range():
    with pytest.raises(ValueError, match="risk_exponent_min"):
        DRMConfig(risk_exponent_min=2.0, risk_exponent_max=1.0)


def test_config_validates_geometry_update_interval():
    assert DRMConfig.from_dict({"vocab_size": 32}).geometry_update_interval == 1
    with pytest.raises(ValueError, match="geometry_update_interval"):
        DRMConfig(geometry_update_interval=0)


def test_config_validates_factorized_basis_sizes():
    config = DRMConfig(direction_basis_size=4, metric_u_basis_size=5)
    assert config.direction_basis_size == 4
    assert config.metric_u_basis_size == 5
    with pytest.raises(ValueError, match="direction_basis_size"):
        DRMConfig(direction_basis_size=-1)


def test_config_validates_bptt_truncate_interval():
    assert DRMConfig(bptt_truncate_interval=8).bptt_truncate_interval == 8
    with pytest.raises(ValueError, match="bptt_truncate_interval"):
        DRMConfig(bptt_truncate_interval=-1)


def test_config_validates_public_generation_and_emitter_fields():
    assert DRMConfig().schema_version == 2
    with pytest.raises(ValueError, match="top_k"):
        DRMConfig(top_k=-1)
    with pytest.raises(ValueError, match="emitter_layers"):
        DRMConfig(emitter_layers=0)
    with pytest.raises(ValueError, match="tokenizer_type"):
        DRMConfig(tokenizer_type="unknown")


def test_config_rejects_direct_transition_in_local_mode():
    with pytest.raises(ValueError, match="direct state transitions"):
        DRMConfig(
            use_direction_field=False,
            directional_cumsum_step_mode="velocity",
        )


def test_validated_copy_rechecks_runtime_mutations():
    config = DRMConfig()
    config.max_seq_len = 0
    with pytest.raises(ValueError, match="max_seq_len"):
        config.validated_copy()


def test_config_validates_sequence_mode_and_geodesic_fields():
    config = DRMConfig(sequence_mode="geodesic_step", geodesic_solver_steps=2, geodesic_lr=0.01)
    assert config.sequence_mode == "geodesic_step"
    assert config.geodesic_solver_steps == 2
    candidate_config = DRMConfig(sequence_mode="directional_candidates", directional_candidate_temperature=0.5)
    assert candidate_config.sequence_mode == "directional_candidates"
    cumsum_config = DRMConfig(sequence_mode="directional_cumsum")
    assert cumsum_config.sequence_mode == "directional_cumsum"
    velocity_cumsum_config = DRMConfig(directional_cumsum_step_mode="velocity")
    assert velocity_cumsum_config.directional_cumsum_step_mode == "velocity"
    block_config = DRMConfig(sequence_mode="directional_block_cumsum", directional_cumsum_block_size=4)
    assert block_config.directional_cumsum_block_size == 4
    superblock_config = DRMConfig(
        sequence_mode="directional_superblock_cumsum",
        directional_superblock_size=16,
        directional_superblock_local_size=4,
    )
    assert superblock_config.sequence_mode == "directional_superblock_cumsum"
    assert superblock_config.directional_superblock_size == 16
    assert superblock_config.directional_superblock_local_size == 4
    endpoint_config = DRMConfig(directional_endpoint_correction_weight=0.5, directional_endpoint_correction_power=2.0)
    assert endpoint_config.directional_endpoint_correction_weight == 0.5
    inner_config = DRMConfig(directional_cumsum_inner_block_size=2)
    assert inner_config.directional_cumsum_inner_block_size == 2
    anderson_config = DRMConfig(directional_anderson_iterations=2, directional_anderson_history_size=3)
    assert anderson_config.directional_anderson_iterations == 2
    velocity_anderson_config = DRMConfig(directional_anderson_transition_mode="velocity")
    assert velocity_anderson_config.directional_anderson_transition_mode == "velocity"
    endpoint_anderson_config = DRMConfig(directional_anderson_scope="endpoint")
    assert endpoint_anderson_config.directional_anderson_scope == "endpoint"
    local_mixer_config = DRMConfig(
        directional_local_mixer="causal_conv",
        directional_local_mixer_hidden_size=32,
        directional_local_mixer_kernel_size=4,
        directional_local_mixer_layers=2,
        directional_local_mixer_scale=0.2,
    )
    assert local_mixer_config.directional_local_mixer == "causal_conv"
    assert local_mixer_config.directional_local_mixer_hidden_size == 32
    assert local_mixer_config.directional_local_mixer_kernel_size == 4
    assert local_mixer_config.directional_local_mixer_layers == 2
    assert local_mixer_config.directional_local_mixer_scale == 0.2
    consistency_config = DRMConfig(lambda_block_consistency=0.1)
    assert consistency_config.lambda_block_consistency == 0.1
    sampled_consistency_config = DRMConfig(
        lambda_sampled_block_consistency=0.1,
        sampled_block_consistency_interval=2,
        sampled_block_consistency_local_size=4,
        sampled_block_consistency_teacher_mode="velocity",
    )
    assert sampled_consistency_config.lambda_sampled_block_consistency == 0.1
    assert sampled_consistency_config.sampled_block_consistency_interval == 2
    assert sampled_consistency_config.sampled_block_consistency_local_size == 4
    assert sampled_consistency_config.sampled_block_consistency_teacher_mode == "velocity"
    with pytest.raises(ValueError, match="sequence_mode"):
        DRMConfig(sequence_mode="not_a_solver")
    with pytest.raises(ValueError, match="geodesic_solver_steps"):
        DRMConfig(geodesic_solver_steps=-1)
    with pytest.raises(ValueError, match="directional_candidate_temperature"):
        DRMConfig(directional_candidate_temperature=0.0)
    with pytest.raises(ValueError, match="directional_cumsum_step_mode"):
        DRMConfig(directional_cumsum_step_mode="not_a_step")
    with pytest.raises(ValueError, match="directional_cumsum_block_size"):
        DRMConfig(directional_cumsum_block_size=-1)
    with pytest.raises(ValueError, match="directional_superblock_size"):
        DRMConfig(directional_superblock_size=-1)
    with pytest.raises(ValueError, match="directional_superblock_local_size"):
        DRMConfig(directional_superblock_local_size=0)
    with pytest.raises(ValueError, match="directional_cumsum_inner_block_size"):
        DRMConfig(directional_cumsum_inner_block_size=-1)
    with pytest.raises(ValueError, match="directional_anderson_transition_mode"):
        DRMConfig(directional_anderson_transition_mode="not_a_transition")
    with pytest.raises(ValueError, match="directional_anderson_scope"):
        DRMConfig(directional_anderson_scope="not_a_scope")
    with pytest.raises(ValueError, match="directional_local_mixer"):
        DRMConfig(directional_local_mixer="not_a_mixer")
    with pytest.raises(ValueError, match="directional_local_mixer_hidden_size"):
        DRMConfig(directional_local_mixer_hidden_size=0)
    with pytest.raises(ValueError, match="directional_local_mixer_kernel_size"):
        DRMConfig(directional_local_mixer_kernel_size=0)
    with pytest.raises(ValueError, match="directional_local_mixer_layers"):
        DRMConfig(directional_local_mixer_layers=0)
    with pytest.raises(ValueError, match="sampled_block_consistency_interval"):
        DRMConfig(sampled_block_consistency_interval=0)
    with pytest.raises(ValueError, match="sampled_block_consistency_local_size"):
        DRMConfig(sampled_block_consistency_local_size=0)
    with pytest.raises(ValueError, match="sampled_block_consistency_teacher_mode"):
        DRMConfig(sampled_block_consistency_teacher_mode="not_a_teacher")
