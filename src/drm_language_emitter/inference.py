from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class InferenceState:
    """Reference causal inference state.

    The first implementation retains the exact token prefix and recomputes the
    tested forward path. This is intentionally correctness-first; variant-specific
    incremental caches can replace the prefix without changing the public API.
    """

    input_ids: torch.Tensor

    @property
    def batch_size(self) -> int:
        return int(self.input_ids.shape[0])

    @property
    def sequence_length(self) -> int:
        return int(self.input_ids.shape[1])
