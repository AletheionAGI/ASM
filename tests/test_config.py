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


def test_config_validates_sequence_mode_and_geodesic_fields():
    config = DRMConfig(sequence_mode="geodesic_step", geodesic_solver_steps=2, geodesic_lr=0.01)
    assert config.sequence_mode == "geodesic_step"
    assert config.geodesic_solver_steps == 2
    candidate_config = DRMConfig(sequence_mode="directional_candidates", directional_candidate_temperature=0.5)
    assert candidate_config.sequence_mode == "directional_candidates"
    cumsum_config = DRMConfig(sequence_mode="directional_cumsum")
    assert cumsum_config.sequence_mode == "directional_cumsum"
    block_config = DRMConfig(sequence_mode="directional_block_cumsum", directional_cumsum_block_size=4)
    assert block_config.directional_cumsum_block_size == 4
    endpoint_config = DRMConfig(directional_endpoint_correction_weight=0.5, directional_endpoint_correction_power=2.0)
    assert endpoint_config.directional_endpoint_correction_weight == 0.5
    inner_config = DRMConfig(directional_cumsum_inner_block_size=2)
    assert inner_config.directional_cumsum_inner_block_size == 2
    anderson_config = DRMConfig(directional_anderson_iterations=2, directional_anderson_history_size=3)
    assert anderson_config.directional_anderson_iterations == 2
    consistency_config = DRMConfig(lambda_block_consistency=0.1)
    assert consistency_config.lambda_block_consistency == 0.1
    with pytest.raises(ValueError, match="sequence_mode"):
        DRMConfig(sequence_mode="not_a_solver")
    with pytest.raises(ValueError, match="geodesic_solver_steps"):
        DRMConfig(geodesic_solver_steps=-1)
    with pytest.raises(ValueError, match="directional_candidate_temperature"):
        DRMConfig(directional_candidate_temperature=0.0)
    with pytest.raises(ValueError, match="directional_cumsum_block_size"):
        DRMConfig(directional_cumsum_block_size=-1)
    with pytest.raises(ValueError, match="directional_cumsum_inner_block_size"):
        DRMConfig(directional_cumsum_inner_block_size=-1)
