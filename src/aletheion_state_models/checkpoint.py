"""Checkpoint compatibility entry points for ASM."""

from drm_language_emitter.checkpoint import load_model

load_state_model = load_model

__all__ = ["load_state_model"]
