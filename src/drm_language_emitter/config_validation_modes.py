from __future__ import annotations


def validate_modes_and_constraints(config) -> None:
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
        val = getattr(config, name)
        if not isinstance(val, bool):
            raise ValueError(f"'{name}' must be a boolean, got {type(val).__name__}")
    if config.risk_exponent_min > config.risk_exponent_max:
        raise ValueError("'risk_exponent_min' must be <= 'risk_exponent_max'")
    if config.sequence_mode not in {
        "local_step",
        "geodesic_step",
        "directional_candidates",
        "directional_cumsum",
        "directional_block_cumsum",
        "directional_superblock_cumsum",
        "asm_z",
    }:
        raise ValueError(
            "'sequence_mode' must be one of: local_step, geodesic_step, "
            "directional_candidates, directional_cumsum, directional_block_cumsum, "
            "directional_superblock_cumsum, asm_z"
        )
    if config.directional_cumsum_step_mode not in {"candidate", "velocity"}:
        raise ValueError("'directional_cumsum_step_mode' must be one of: candidate, velocity")
    if config.directional_metric_composition not in {
        "post_naturalize",
        "metric_subspace",
        "metric_orthonormal",
    }:
        raise ValueError(
            "'directional_metric_composition' must be one of: "
            "post_naturalize, metric_subspace, metric_orthonormal"
        )
    if (
        config.directional_metric_composition != "post_naturalize"
        and (not config.use_direction_field or not config.use_relational_metric)
    ):
        raise ValueError(
            "metric-aware directional composition requires both a "
            "direction field and relational metric"
        )
    if config.directional_anderson_transition_mode not in {"candidate", "velocity"}:
        raise ValueError("'directional_anderson_transition_mode' must be one of: candidate, velocity")
    if config.directional_anderson_scope not in {"trajectory", "endpoint"}:
        raise ValueError("'directional_anderson_scope' must be one of: trajectory, endpoint")
    if config.directional_local_mixer not in {"none", "causal_conv"}:
        raise ValueError("'directional_local_mixer' must be one of: none, causal_conv")
    if config.active_fraction_loss_mode not in {"upper_bound", "target"}:
        raise ValueError("'active_fraction_loss_mode' must be one of: upper_bound, target")
    if config.sampled_block_consistency_teacher_mode not in {"candidate", "velocity"}:
        raise ValueError("'sampled_block_consistency_teacher_mode' must be one of: candidate, velocity")
    if config.tokenizer_type not in {"byte", "char"}:
        raise ValueError("'tokenizer_type' must be one of: byte, char")
    if not config.use_drm_geometry and (
        config.use_direction_field or config.use_relational_metric
    ):
        raise ValueError(
            "geometry-free models must disable use_direction_field and "
            "use_relational_metric"
        )
    if (
        config.sequence_mode != "asm_z"
        and not config.use_direction_field
        and config.directional_cumsum_step_mode != "velocity"
    ):
        raise ValueError(
            "models without a direction field require "
            "directional_cumsum_step_mode='velocity'"
        )
    if config.use_drm_geometry and not config.use_direction_field and config.sequence_mode in {
        "local_step",
        "geodesic_step",
        "directional_candidates",
    }:
        raise ValueError(
            "direct state transitions require a cumsum/block-cumsum sequence mode"
        )
    if config.sequence_mode == "asm_z":
        if config.use_direction_field or config.use_relational_metric:
            raise ValueError("ASM-Z forbids legacy direction and metric catalogs")
        if config.bounded_state:
            raise ValueError("ASM-Z forbids post-update state bounding")
        if not config.use_drm_geometry:
            raise ValueError("ASM-Z requires its learned potential and SPD metric")
        if config.asm_z_eta <= 0:
            raise ValueError("ASM-Z requires asm_z_eta > 0")
        if config.asm_z_metric_d_min >= config.asm_z_metric_d_max:
            raise ValueError("ASM-Z metric bounds require d_min < d_max")
        if config.dropout != 0.0:
            raise ValueError("ASM-Z requires dropout=0 for a deterministic recurrence")
        if config.n_flow_steps != 1:
            raise ValueError("ASM-Z requires exactly one update per input")
        if config.variable_rank_mode != "off":
            raise ValueError("ASM-Z forbids variable-rank routing")
        if any((
            config.addressable_memory,
            config.selective_memory,
            config.token_state_residual,
            config.directional_local_mixer != "none",
            bool(config.directional_refinement_layers),
        )):
            raise ValueError("ASM-Z forbids alternate state and memory bypasses")
    if config.variable_rank_mode not in {"off", "phase1_input_hard", "phase2_input_ste", "phase3a1_projected"}:
        raise ValueError("invalid variable_rank_mode")
    if config.variable_rank_memory_policy not in {"forbid", "project_io"}:
        raise ValueError("invalid variable_rank_memory_policy")
    rank_memory_is_aligned = config.variable_rank_mode != "off" and config.addressable_memory and config.addressable_memory_backend == "fast_weight" and (config.addressable_memory_value_dim or config.addressable_memory_dim) == config.d_state
    if config.variable_rank_memory_policy == "project_io" and not rank_memory_is_aligned:
        raise ValueError("project_io requires variable rank and state-aligned fast-weight memory")
    if config.variable_rank_min_rank > config.d_state:
        raise ValueError("variable_rank_min_rank cannot exceed d_state")
    if config.variable_rank_mode == "off" and config.variable_rank_scaffold_projection:
        raise ValueError("projected scaffold requires variable rank to be enabled")
    if config.variable_rank_mode != "off":
        if config.sequence_mode != "directional_block_cumsum":
            raise ValueError("ASM-VR Phase 1 requires directional_block_cumsum")
        if config.directional_cumsum_block_size <= 0:
            raise ValueError("ASM-VR Phase 1 requires a positive block size")
        if config.use_direction_field:
            raise ValueError("ASM-VR Phase 1 requires the ASM-R direct transition")
        scaffold_routes = (
            config.directional_local_mixer != "none",
            config.token_state_residual,
            config.selective_memory,
        )
        forbidden_routes = (
            config.addressable_memory and config.variable_rank_memory_policy == "forbid",
            bool(config.directional_refinement_layers),
            bool(config.directional_endpoint_correction_weight),
            bool(config.directional_fixed_point_iterations),
            bool(config.directional_anderson_iterations),
        )
        if any(forbidden_routes):
            raise ValueError(
                "ASM-VR forbids unprojected addressable memory, refinement, and solver bypasses"
            )
        if any(scaffold_routes) and not config.variable_rank_scaffold_projection:
            raise ValueError("ASM-VR forbids scaffold routes without projection after each component")
        if config.variable_rank_scaffold_projection != (config.variable_rank_mode == "phase3a1_projected"):
            raise ValueError("projected scaffold requires variable_rank_mode='phase3a1_projected'")
        if not config.compact_streaming_inference:
            raise ValueError("ASM-VR Phase 1 requires compact streaming inference")
    if config.addressable_memory and (
        not config.compact_streaming_inference
        or config.sequence_mode != "directional_block_cumsum"
    ):
        raise ValueError(
            "addressable memory requires compact directional_block_cumsum inference"
        )
    if config.addressable_memory_backend not in {"slots", "fast_weight"}:
        raise ValueError("addressable_memory_backend must be 'slots' or 'fast_weight'")
    if config.epistemic_memory_gating and (
        not config.addressable_memory or config.addressable_memory_backend != "fast_weight"
    ):
        raise ValueError(
            "epistemic memory gating requires the fast_weight addressable memory backend"
        )
    if config.addressable_memory_heads != 1:
        raise ValueError("addressable_memory_heads currently supports only 1")
    for name in ("addressable_memory_read_top_k", "addressable_memory_write_top_k"):
        value = getattr(config, name)
        if value > config.addressable_memory_slots:
            raise ValueError(f"{name} cannot exceed addressable_memory_slots")
