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

    def validated_copy(self) -> "DRMConfig":
        """Return a freshly validated copy after runtime overrides."""
        return type(self).from_dict(self.to_dict())

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------
    def _validate(self) -> None:
        """Validate configuration values and raise ValueError if any check fails."""

        # Integer / positive checks
        int_fields = [
            ("schema_version", 1, None),
            ("vocab_size", 1, None),
            ("d_token", 1, None),
            ("d_state", 1, None),
            ("n_directions", 1, None),
            ("metric_rank", 0, None),  # rank can be zero
            ("hidden_size", 1, None),
            ("n_flow_steps", 1, None),
            ("max_seq_len", 1, None),
            ("top_k", 0, None),
            ("emitter_layers", 1, None),
            ("gate_top_k", 0, None),
            ("geodesic_solver_steps", 0, None),
            ("directional_cumsum_block_size", 0, None),
            ("directional_superblock_size", 0, None),
            ("directional_superblock_local_size", 1, None),
            ("directional_cumsum_inner_block_size", 0, None),
            ("directional_anderson_iterations", 0, None),
            ("directional_anderson_history_size", 1, None),
            ("directional_anderson_block_stride", 1, None),
            ("directional_fixed_point_iterations", 0, None),
            ("directional_local_mixer_hidden_size", 1, None),
            ("directional_local_mixer_kernel_size", 1, None),
            ("directional_local_mixer_layers", 1, None),
            ("directional_local_mixer_dilation_growth", 1, None),
            ("directional_refinement_layers", 0, None),
            ("selective_memory_hidden_size", 1, None),
            ("sampled_block_consistency_interval", 1, None),
            ("sampled_block_consistency_local_size", 1, None),
            ("geometry_update_interval", 1, None),
            ("direction_basis_size", 0, None),
            ("metric_u_basis_size", 0, None),
            ("bptt_truncate_interval", 0, None),
            ("addressable_memory_slots", 1, None),
            ("addressable_memory_dim", 1, None),
            ("addressable_memory_value_dim", 0, None),
            ("addressable_memory_heads", 1, None),
            ("epistemic_gate_hidden_dim", 1, None),
            ("epistemic_gate_num_layers", 0, None),
            ("variable_rank_min_rank", 0, None),
            ("variable_rank_warmup_steps", 0, None),
            ("variable_rank_budget_ramp_steps", 0, None),
            ("variable_rank_hardening_steps", 0, None),
            ("addressable_memory_read_top_k", 0, None),
            ("addressable_memory_write_top_k", 0, None),
        ]
        for name, min_val, max_val in int_fields:
            val = getattr(self, name)
            if not isinstance(val, int):
                raise ValueError(f"'{name}' must be an integer, got {type(val).__name__}")
            if val < min_val or (max_val is not None and val > max_val):
                raise ValueError(
                    f"'{name}' must be between {min_val}"
                    + (f" and {max_val}" if max_val is not None else "")
                    + f", got {val}"
                )

        # Float / non-negative checks
        float_fields = [
            ("dt", 0.0, None),
            ("dropout", 0.0, 1.0),
            ("lambda_action", 0.0, None),
            ("lambda_dim_sparsity", 0.0, None),
            ("lambda_dim_entropy", 0.0, None),
            ("lambda_dim_variance", 0.0, None),
            ("target_dim_std", 0.0, None),
            ("lambda_metric_reg", 0.0, None),
            ("lambda_metric_diversity", 0.0, None),
            ("lambda_recurrence", 0.0, None),
            ("lambda_stability", 0.0, None),
            ("lambda_blindspot", 0.0, None),
            ("risk_mass_max", 0.0, None),
            ("risk_exponent_min", 0.0, None),
            ("risk_exponent_max", 0.0, None),
            ("risk_alpha_max", 0.0, None),
            ("generation_temperature", 0.1, None),   # temperature > 0
            ("metric_eps", 0.0, None),
            ("state_clip_norm", 0.0, None),
            ("gate_temperature", 0.01, None),       # temperature > 0
            ("gate_logit_bias", -10.0, 10.0),
            ("lambda_active_fraction", 0.0, 1.0),
            ("target_active_fraction", 0.0, 1.0),
            ("metric_naturalization_strength", 0.0, None),
            ("metric_naturalization_warmup_steps", 0, None),
            ("metric_damping", 0.0, None),
            ("metric_u_min_norm", 0.0, None),
            ("lambda_metric_u_floor", 0.0, None),
            ("metric_u_target_norm", 0.0, None),
            ("lambda_metric_u_target", 0.0, None),
            ("target_condition", 1.0, None),
            ("lambda_condition", 0.0, None),
            ("geodesic_lr", 0.0, None),
            ("geodesic_anchor_weight", 0.0, None),
            ("geodesic_metric_weight", 0.0, None),
            ("geodesic_risk_weight", 0.0, None),
            ("directional_candidate_temperature", 0.01, None),
            ("directional_candidate_scale", 0.0, None),
            ("directional_endpoint_correction_weight", 0.0, None),
            ("directional_endpoint_correction_power", 0.0, None),
            ("directional_anderson_ridge", 0.0, None),
            ("directional_anderson_relaxation", 0.0, None),
            ("directional_fixed_point_relaxation", 0.0, None),
            ("directional_local_mixer_scale", 0.0, None),
            ("token_state_residual_scale", 0.0, None),
            ("directional_refinement_scale", 0.0, None),
            ("selective_memory_scale", 0.0, None),
            ("selective_memory_forget_bias", -10.0, 10.0),
            ("addressable_memory_read_scale", 0.0, None),
            ("addressable_memory_write_bias", -10.0, 10.0),
            ("addressable_memory_temperature", 0.01, None),
            ("addressable_memory_usage_decay", 0.0, 1.0),
            ("addressable_memory_age_bias", 0.0, None),
            ("fast_weight_hard_write_threshold", 0.0, 1.0),
            ("fast_weight_consolidation_scale", 0.0, 1.0),
            ("fast_weight_slow_read_scale", 0.0, None),
            ("epistemic_gate_dropout", 0.0, 1.0),
            ("epistemic_gate_initial_confidence", 0.000001, 0.999999),
            ("variable_rank_threshold", 0.0, 1.0),
            ("variable_rank_target_fraction", 0.0, 1.0),
            ("variable_rank_temperature_initial", 0.000001, None),
            ("variable_rank_temperature_final", 0.000001, None),
            ("lambda_variable_rank_budget", 0.0, None),
            ("lambda_variable_rank_binary", 0.0, None),
            ("lambda_variable_rank_switch", 0.0, None),
            ("variable_rank_open_probability", 0.000001, 0.999999),
            ("lambda_addressable_read_entropy", 0.0, None),
            ("lambda_addressable_write_entropy", 0.0, None),
            ("lambda_block_consistency", 0.0, None),
            ("block_consistency_weight", 0.0, None),
            ("lambda_sampled_block_consistency", 0.0, None),
        ]
        for name, min_val, max_val in float_fields:
            val = getattr(self, name)
            if not isinstance(val, (float, int)):
                raise ValueError(f"'{name}' must be a number, got {type(val).__name__}")
            if val < min_val or (max_val is not None and val > max_val):
                raise ValueError(
                    f"'{name}' must be between {min_val}"
                    + (f" and {max_val}" if max_val is not None else "")
                    + f", got {val}"
                )

        # Boolean checks
        bool_fields = [
            "bounded_state",
            "use_toroidal_state",
            "use_powerlaw_risk",
            "instantiate_disabled_risk",
            "use_drm_geometry",
            "use_direction_field",
            "use_relational_metric",
            "direction_norm",
            "tie_embeddings",
            "use_metric_naturalization",
            "gate_top_k_renorm",
            "token_state_residual",
            "selective_memory",
            "emitter_swiglu",
            "emitter_residual",
            "use_torch_compile",
            "compact_streaming_inference",
            "variable_rank_scaffold_projection",
            "addressable_memory",
            "addressable_memory_read_enabled",
            "addressable_memory_write_enabled",
            "addressable_memory_shuffle_on_eval",
            "addressable_memory_use_previous_token_key",
            "fast_weight_durable_memory",
            "fast_weight_state_fp32",
            "fast_weight_compute_fp32",
            "epistemic_memory_gating",
        ]
        for name in bool_fields:
            val = getattr(self, name)
            if not isinstance(val, bool):
                raise ValueError(f"'{name}' must be a boolean, got {type(val).__name__}")
        if self.risk_exponent_min > self.risk_exponent_max:
            raise ValueError("'risk_exponent_min' must be <= 'risk_exponent_max'")
        if self.sequence_mode not in {
            "local_step",
            "geodesic_step",
            "directional_candidates",
            "directional_cumsum",
            "directional_block_cumsum",
            "directional_superblock_cumsum",
        }:
            raise ValueError(
                "'sequence_mode' must be one of: local_step, geodesic_step, "
                "directional_candidates, directional_cumsum, directional_block_cumsum, "
                "directional_superblock_cumsum"
            )
        if self.directional_cumsum_step_mode not in {"candidate", "velocity"}:
            raise ValueError("'directional_cumsum_step_mode' must be one of: candidate, velocity")
        if self.directional_metric_composition not in {
            "post_naturalize",
            "metric_subspace",
            "metric_orthonormal",
        }:
            raise ValueError(
                "'directional_metric_composition' must be one of: "
                "post_naturalize, metric_subspace, metric_orthonormal"
            )
        if (
            self.directional_metric_composition != "post_naturalize"
            and (not self.use_direction_field or not self.use_relational_metric)
        ):
            raise ValueError(
                "metric-aware directional composition requires both a "
                "direction field and relational metric"
            )
        if self.directional_anderson_transition_mode not in {"candidate", "velocity"}:
            raise ValueError("'directional_anderson_transition_mode' must be one of: candidate, velocity")
        if self.directional_anderson_scope not in {"trajectory", "endpoint"}:
            raise ValueError("'directional_anderson_scope' must be one of: trajectory, endpoint")
        if self.directional_local_mixer not in {"none", "causal_conv"}:
            raise ValueError("'directional_local_mixer' must be one of: none, causal_conv")
        if self.active_fraction_loss_mode not in {"upper_bound", "target"}:
            raise ValueError("'active_fraction_loss_mode' must be one of: upper_bound, target")
        if self.sampled_block_consistency_teacher_mode not in {"candidate", "velocity"}:
            raise ValueError("'sampled_block_consistency_teacher_mode' must be one of: candidate, velocity")
        if self.tokenizer_type not in {"byte", "char"}:
            raise ValueError("'tokenizer_type' must be one of: byte, char")
        if not self.use_drm_geometry and (
            self.use_direction_field or self.use_relational_metric
        ):
            raise ValueError(
                "geometry-free models must disable use_direction_field and "
                "use_relational_metric"
            )
        if not self.use_direction_field and self.directional_cumsum_step_mode != "velocity":
            raise ValueError(
                "models without a direction field require "
                "directional_cumsum_step_mode='velocity'"
            )
        if self.use_drm_geometry and not self.use_direction_field and self.sequence_mode in {
            "local_step",
            "geodesic_step",
            "directional_candidates",
        }:
            raise ValueError(
                "direct state transitions require a cumsum/block-cumsum sequence mode"
            )
        if self.variable_rank_mode not in {"off", "phase1_input_hard", "phase2_input_ste", "phase3a1_projected"}:
            raise ValueError("invalid variable_rank_mode")
        if self.variable_rank_memory_policy not in {"forbid", "project_io"}:
            raise ValueError("invalid variable_rank_memory_policy")
        rank_memory_is_aligned = self.variable_rank_mode != "off" and self.addressable_memory and self.addressable_memory_backend == "fast_weight" and (self.addressable_memory_value_dim or self.addressable_memory_dim) == self.d_state
        if self.variable_rank_memory_policy == "project_io" and not rank_memory_is_aligned:
            raise ValueError("project_io requires variable rank and state-aligned fast-weight memory")
        if self.variable_rank_min_rank > self.d_state:
            raise ValueError("variable_rank_min_rank cannot exceed d_state")
        if self.variable_rank_mode == "off" and self.variable_rank_scaffold_projection:
            raise ValueError("projected scaffold requires variable rank to be enabled")
        if self.variable_rank_mode != "off":
            if self.sequence_mode != "directional_block_cumsum":
                raise ValueError("ASM-VR Phase 1 requires directional_block_cumsum")
            if self.directional_cumsum_block_size <= 0:
                raise ValueError("ASM-VR Phase 1 requires a positive block size")
            if self.use_direction_field:
                raise ValueError("ASM-VR Phase 1 requires the ASM-R direct transition")
            scaffold_routes = (
                self.directional_local_mixer != "none",
                self.token_state_residual,
                self.selective_memory,
            )
            forbidden_routes = (
                self.addressable_memory and self.variable_rank_memory_policy == "forbid",
                bool(self.directional_refinement_layers),
                bool(self.directional_endpoint_correction_weight),
                bool(self.directional_fixed_point_iterations),
                bool(self.directional_anderson_iterations),
            )
            if any(forbidden_routes):
                raise ValueError(
                    "ASM-VR forbids unprojected addressable memory, refinement, and solver bypasses"
                )
            if any(scaffold_routes) and not self.variable_rank_scaffold_projection:
                raise ValueError("ASM-VR forbids scaffold routes without projection after each component")
            if self.variable_rank_scaffold_projection != (self.variable_rank_mode == "phase3a1_projected"):
                raise ValueError("projected scaffold requires variable_rank_mode='phase3a1_projected'")
            if not self.compact_streaming_inference:
                raise ValueError("ASM-VR Phase 1 requires compact streaming inference")
        if self.addressable_memory and (
            not self.compact_streaming_inference
            or self.sequence_mode != "directional_block_cumsum"
        ):
            raise ValueError(
                "addressable memory requires compact directional_block_cumsum inference"
            )
        if self.addressable_memory_backend not in {"slots", "fast_weight"}:
            raise ValueError("addressable_memory_backend must be 'slots' or 'fast_weight'")
        if self.epistemic_memory_gating and (
            not self.addressable_memory or self.addressable_memory_backend != "fast_weight"
        ):
            raise ValueError(
                "epistemic memory gating requires the fast_weight addressable memory backend"
            )
        if self.addressable_memory_heads != 1:
            raise ValueError("addressable_memory_heads currently supports only 1")
        for name in ("addressable_memory_read_top_k", "addressable_memory_write_top_k"):
            value = getattr(self, name)
            if value > self.addressable_memory_slots:
                raise ValueError(f"{name} cannot exceed addressable_memory_slots")

    def __post_init__(self) -> None:  # pragma: no cover
        """Automatically validate configuration on instantiation."""
        self._validate()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DRMConfig":
        allowed = {field.name for field in cls.__dataclass_fields__.values()}
        unknown = sorted(set(data) - allowed)
        if unknown:
            raise ValueError(f"unknown DRMConfig field(s): {', '.join(unknown)}")
        return cls(**data)
