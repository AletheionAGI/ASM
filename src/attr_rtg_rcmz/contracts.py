"""Strict process-message and result contracts for ATTR-RTG-RCMZ-V1."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch

MESSAGE_FIELDS = frozenset({"history_bytes", "candidate4s", "masks", "logical_lengths"})


@dataclass(frozen=True)
class InferenceMessage:
    history_bytes: torch.Tensor
    candidate4s: torch.Tensor
    masks: torch.Tensor
    logical_lengths: torch.Tensor

    @classmethod
    def from_mapping(cls, value: Mapping[str, torch.Tensor]) -> InferenceMessage:
        if set(value) != MESSAGE_FIELDS:
            raise ValueError(
                f"inference message must have exactly {sorted(MESSAGE_FIELDS)}"
            )
        return cls(**value)

    def validate(self, context_length: int = 256) -> None:
        if (
            self.history_bytes.ndim != 2
            or self.history_bytes.shape[1] != context_length
        ):
            raise ValueError(f"history_bytes must have shape [batch,{context_length}]")
        batch = self.history_bytes.shape[0]
        if self.candidate4s.shape != (batch, 6, 4):
            raise ValueError("candidate4s must have shape [batch,6,4]")
        if self.masks.shape != (batch, 6) or self.masks.dtype is not torch.bool:
            raise ValueError("masks must be bool [batch,6]")
        if self.logical_lengths.shape != (batch,):
            raise ValueError("logical_lengths must have shape [batch]")
        if self.history_bytes.dtype not in (torch.uint8, torch.int32, torch.int64):
            raise TypeError("history_bytes must contain integer bytes")
        if torch.any(self.history_bytes < 0) or torch.any(self.history_bytes > 255):
            raise ValueError("history_bytes values must be bytes")
        if torch.any(self.logical_lengths < 1) or torch.any(
            self.logical_lengths > context_length
        ):
            raise ValueError("logical_lengths must be in [1,context_length]")
        if not torch.isfinite(self.candidate4s.float()).all():
            raise ValueError("candidate4s must be finite")
        if torch.any(self.candidate4s < 0) or torch.any(self.candidate4s > 255):
            raise ValueError("candidate4s values must be bytes")
        if self.candidate4s.is_floating_point() and not torch.equal(
            self.candidate4s, self.candidate4s.round()
        ):
            raise ValueError("candidate4s must contain integral byte values")


@dataclass(frozen=True)
class InferenceResult:
    logits: torch.Tensor
    common24: torch.Tensor
    native_state: torch.Tensor


class FourFieldInference:
    """Bind an arm adapter outside the untrusted four-field process message."""

    def __init__(self, adapter: torch.nn.Module) -> None:
        self.adapter = adapter

    def __call__(self, message: Mapping[str, torch.Tensor]) -> InferenceResult:
        return self.adapter(InferenceMessage.from_mapping(message))
