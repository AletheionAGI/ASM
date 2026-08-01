"""Aletheion State Models research family.

This package is additive while the winning architecture is still under
evaluation. The existing ``drm_language_emitter`` package remains the source of
the tested implementations and checkpoint-compatible class definitions.
"""

from drm_language_emitter.inference import InferenceState

from .checkpoint import load_state_model
from .config import ASMConfig
from .core.interfaces import StateModelProtocol
from .core.state_model import StateModel

__all__ = [
    "ASMConfig",
    "InferenceState",
    "StateModel",
    "StateModelProtocol",
    "load_state_model",
]
