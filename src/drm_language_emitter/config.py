from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class DRMConfig:
    schema_version: int = 2
    vocab_size: int = 256
    d_token: int = 64
    d_state: int = 96
    n_directions: int = 12
    metric_rank: int = 8
    hidden_size: int = 128
    n_flow_steps: int = 1
    dt: float = 0.1
    asm_z_eta: float = 0.1
    asm_z_lambda: float = 0.01
    asm_z_metric_d_min: float = 0.1
    asm_z_metric_d_max: float = 2.0
    asm_z_metric_u_bound: float = 1.0
    max_seq_len: int = 256
    dropout: float = 0.0
    bounded_state: bool = True
    use_toroidal_state: bool = False
    use_powerlaw_risk: bool = False
    instantiate_disabled_risk: bool = True
    use_drm_geometry: bool = True
    use_direction_field: bool = True
    use_relational_metric: bool = True
    lambda_action: float = 0.01
    lambda_dim_sparsity: float = 0.001
    lambda_dim_entropy: float = 0.001
    lambda_dim_variance: float = 0.01
    target_dim_std: float = 0.15
    lambda_metric_reg: float = 0.001
    lambda_metric_diversity: float = 0.001
    lambda_recurrence: float = 0.0
    lambda_stability: float = 0.0
    lambda_blindspot: float = 0.0
    risk_mass_max: float = 10.0
    risk_exponent_min: float = 0.25
    risk_exponent_max: float = 4.0
    risk_alpha_max: float = 10.0
    generation_temperature: float = 1.0
    top_k: int = 40
    metric_eps: float = 1e-4
    state_clip_norm: float = 8.0
    direction_norm: bool = True
    tie_embeddings: bool = False
    tokenizer_type: str = "byte"
    seed: int = 1337
    gate_temperature: float = 1.5
    gate_logit_bias: float = -1.0
    gate_top_k: int = 0
    gate_top_k_renorm: bool = False
    active_fraction_loss_mode: str = "upper_bound"
    lambda_active_fraction: float = 0.01
    target_active_fraction: float = 0.65
    use_metric_naturalization: bool = True
    directional_metric_composition: str = "post_naturalize"
    metric_naturalization_strength: float = 0.5
    metric_naturalization_warmup_steps: int = 500
    metric_damping: float = 0.3
    metric_u_min_norm: float = 0.05
    lambda_metric_u_floor: float = 0.001
    metric_u_target_norm: float = 1.0
    lambda_metric_u_target: float = 0.001
    target_condition: float = 100.0
    lambda_condition: float = 0.001
    emitter_layers: int = 1
    emitter_swiglu: bool = False
    emitter_residual: bool = False
    use_torch_compile: bool = False
    sequence_mode: str = "local_step"
    geodesic_solver_steps: int = 0
    geodesic_lr: float = 0.05
    geodesic_anchor_weight: float = 1.0
    geodesic_metric_weight: float = 0.1
    geodesic_risk_weight: float = 0.01
    directional_candidate_temperature: float = 1.0
    directional_candidate_scale: float = 1.0
    directional_cumsum_step_mode: str = "candidate"
    directional_cumsum_block_size: int = 0
    directional_superblock_size: int = 0
    directional_superblock_local_size: int = 8
    directional_endpoint_correction_weight: float = 0.0
    directional_endpoint_correction_power: float = 1.0
    directional_cumsum_inner_block_size: int = 0
    directional_anderson_iterations: int = 0
    directional_anderson_history_size: int = 4
    directional_anderson_ridge: float = 1e-4
    directional_anderson_relaxation: float = 1.0
    directional_anderson_transition_mode: str = "candidate"
    directional_anderson_block_stride: int = 1
    directional_anderson_scope: str = "trajectory"
    directional_fixed_point_iterations: int = 0
    directional_fixed_point_relaxation: float = 1.0
    directional_local_mixer: str = "none"
    directional_local_mixer_hidden_size: int = 256
    directional_local_mixer_kernel_size: int = 8
    directional_local_mixer_layers: int = 1
    directional_local_mixer_dilation_growth: int = 1
    directional_local_mixer_scale: float = 0.1
    token_state_residual: bool = False
    token_state_residual_scale: float = 0.1
    selective_memory: bool = False
    selective_memory_hidden_size: int = 256
    selective_memory_scale: float = 0.1
    selective_memory_forget_bias: float = 2.0
    directional_refinement_layers: int = 0
    directional_refinement_scale: float = 0.1
    lambda_block_consistency: float = 0.0
    block_consistency_weight: float = 1.0
    lambda_sampled_block_consistency: float = 0.0
    sampled_block_consistency_interval: int = 8
    sampled_block_consistency_local_size: int = 8
    sampled_block_consistency_teacher_mode: str = "candidate"
    geometry_update_interval: int = 1
    direction_basis_size: int = 0
    metric_u_basis_size: int = 0
    bptt_truncate_interval: int = 0
    compact_streaming_inference: bool = False
    addressable_memory: bool = False
    addressable_memory_backend: str = "slots"
    addressable_memory_slots: int = 32
    addressable_memory_dim: int = 128
    addressable_memory_value_dim: int = 0
    addressable_memory_heads: int = 1
    addressable_memory_read_scale: float = 0.1
    addressable_memory_write_bias: float = -2.0
    addressable_memory_temperature: float = 1.0
    addressable_memory_usage_decay: float = 0.99
    addressable_memory_age_bias: float = 1.0
    addressable_memory_read_enabled: bool = True
    addressable_memory_write_enabled: bool = True
    addressable_memory_shuffle_on_eval: bool = False
    addressable_memory_read_top_k: int = 0
    addressable_memory_write_top_k: int = 0
    addressable_memory_use_previous_token_key: bool = False
    fast_weight_durable_memory: bool = False
    fast_weight_state_fp32: bool = False
    fast_weight_compute_fp32: bool = False
    fast_weight_hard_write_threshold: float = 0.0
    fast_weight_consolidation_scale: float = 0.25
    fast_weight_slow_read_scale: float = 1.0
    epistemic_memory_gating: bool = False
    epistemic_gate_hidden_dim: int = 64
    epistemic_gate_num_layers: int = 2
    epistemic_gate_dropout: float = 0.0
    epistemic_gate_initial_confidence: float = 0.9
    variable_rank_mode: str = "off"
    variable_rank_threshold: float = 0.5
    variable_rank_min_rank: int = 1
    variable_rank_target_fraction: float = 0.5
    variable_rank_temperature_initial: float = 2.0
    variable_rank_temperature_final: float = 0.5
    variable_rank_warmup_steps: int = 25
    variable_rank_budget_ramp_steps: int = 75
    variable_rank_hardening_steps: int = 100
    lambda_variable_rank_budget: float = 1.0
    lambda_variable_rank_binary: float = 0.01
    lambda_variable_rank_switch: float = 0.01
    variable_rank_open_probability: float = 0.95
    variable_rank_scaffold_projection: bool = False
    variable_rank_memory_policy: str = "forbid"
    lambda_addressable_read_entropy: float = 0.0
    lambda_addressable_write_entropy: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def validated_copy(self) -> DRMConfig:
        """Return a freshly validated copy after runtime overrides."""
        return type(self).from_dict(self.to_dict())

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------
    def _validate(self) -> None:
        from .config_validation_modes import validate_modes_and_constraints
        from .config_validation_numeric import validate_numeric_fields

        validate_numeric_fields(self)
        validate_modes_and_constraints(self)

    def __post_init__(self) -> None:  # pragma: no cover
        """Automatically validate configuration on instantiation."""
        self._validate()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DRMConfig:
        allowed = {field.name for field in cls.__dataclass_fields__.values()}
        unknown = sorted(set(data) - allowed)
        if unknown:
            raise ValueError(f"unknown DRMConfig field(s): {', '.join(unknown)}")
        return cls(**data)
