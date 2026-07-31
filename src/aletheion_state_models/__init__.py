"""Aletheion State Models research family.

This package is additive while the winning architecture is still under
evaluation. The existing ``drm_language_emitter`` package remains the source of
the tested implementations and checkpoint-compatible class definitions.
"""

from .core.state_model import StateModel

__all__ = ["StateModel"]
