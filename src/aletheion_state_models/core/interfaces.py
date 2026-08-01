"""Architecture-neutral structural interfaces for the ASM family."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import torch

from drm_language_emitter.inference import InferenceState


@runtime_checkable
class StateModelProtocol(Protocol):
    config: Any

    def forward(self, input_ids: torch.Tensor, *args: Any, **kwargs: Any) -> dict[str, Any]: ...

    def init_inference_state(
        self,
        batch_size: int,
        device: torch.device | str,
    ) -> InferenceState: ...

    def prefill(
        self,
        input_ids: torch.Tensor,
        state: InferenceState | None = None,
    ) -> tuple[torch.Tensor, InferenceState]: ...

    def decode_step(
        self,
        input_ids: torch.Tensor,
        state: InferenceState,
    ) -> tuple[torch.Tensor, InferenceState]: ...


__all__ = ["StateModelProtocol"]
