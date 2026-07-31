"""Neutral public name for the current checkpoint-compatible state model."""

from drm_language_emitter.model import DRMEmitterModel

# An alias, rather than a subclass, preserves state-dict keys and exact runtime
# behavior while the ASM family is evaluated.
StateModel = DRMEmitterModel

__all__ = ["StateModel"]
