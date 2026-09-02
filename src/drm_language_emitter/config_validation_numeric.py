from __future__ import annotations

import math


def validate_numeric_fields(config) -> None:
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
        val = getattr(config, name)
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
        ("asm_z_eta", 0.0, None),
        ("asm_z_lambda", 0.0, None),
        ("asm_z_metric_d_min", 0.000001, None),
        ("asm_z_metric_d_max", 0.000001, None),
        ("asm_z_metric_u_bound", 0.0, None),
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
        val = getattr(config, name)
        if not isinstance(val, (float, int)):
            raise ValueError(f"'{name}' must be a number, got {type(val).__name__}")
        if not math.isfinite(val):
            raise ValueError(f"'{name}' must be finite, got {val}")
        if val < min_val or (max_val is not None and val > max_val):
            raise ValueError(
                f"'{name}' must be between {min_val}"
                + (f" and {max_val}" if max_val is not None else "")
                + f", got {val}"
            )
