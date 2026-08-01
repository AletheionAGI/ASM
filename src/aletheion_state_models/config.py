"""Versioned, neutral configuration entry point for ASM."""

from drm_language_emitter.config import DRMConfig


class ASMConfig(DRMConfig):
    """ASM public name retaining exact DRMConfig checkpoint compatibility."""


__all__ = ["ASMConfig"]
