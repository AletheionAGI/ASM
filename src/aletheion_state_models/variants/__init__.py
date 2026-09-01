"""Named constructors for the Aletheion State Models research variants."""

from .causal_memory import build_causal_memory
from .compact_streaming import build_compact_streaming
from .compact_addressable import (
    build_compact_addressable,
    build_compact_addressable_sparse,
    build_compact_fast_weight,
    build_compact_durable_fast_weight,
    build_compact_epistemic_memory,
)
from .compact_variable_rank import (
    build_compact_memory_adaptive_rank,
    build_compact_memory_variable_rank,
)
from .direct_state import build_direct_state
from .explicit_drm import build_explicit_drm
from .metric_frame import build_metric_frame
from .metric_subspace import build_metric_subspace
from .relational_state import build_relational_state
from .relational_selective_state import (
    build_relational_selective_state,
    relational_selective_state_config,
)
from .selective_state import build_selective_state, selective_state_config
from .variable_rank import (
    build_variable_rank_phase1,
    build_variable_rank_phase2,
    build_variable_rank_phase3a1,
)

__all__ = [
    "build_causal_memory",
    "build_compact_streaming",
    "build_compact_addressable",
    "build_compact_addressable_sparse",
    "build_compact_fast_weight",
    "build_compact_durable_fast_weight",
    "build_compact_epistemic_memory",
    "build_compact_memory_adaptive_rank",
    "build_compact_memory_variable_rank",
    "build_direct_state",
    "build_explicit_drm",
    "build_metric_subspace",
    "build_metric_frame",
    "build_relational_state",
    "build_relational_selective_state",
    "relational_selective_state_config",
    "build_selective_state",
    "selective_state_config",
    "build_variable_rank_phase1",
    "build_variable_rank_phase2",
    "build_variable_rank_phase3a1",
]
