"""State-transition mechanisms shared by ASM variants."""

from drm_language_emitter.dynamics import DRMFlow
from drm_language_emitter.model_components import DirectStateTransition

ExplicitDirectionalTransition = DRMFlow
DirectContextualTransition = DirectStateTransition

__all__ = ["DirectContextualTransition", "ExplicitDirectionalTransition"]
