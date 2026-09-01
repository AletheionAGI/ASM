"""Common causal-representation adapters for ATTR model backbones."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
from torch import nn


class ModelAdapter(nn.Module, ABC):
    """Uniform interface used by common prediction heads."""

    def __init__(self, model: nn.Module, representation_dim: int) -> None:
        super().__init__()
        if representation_dim is None or representation_dim <= 0:
            raise ValueError(
                "representation_dim must be positive or available on model.config"
            )
        self.model = model
        self.representation_dim = representation_dim

    @abstractmethod
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Return one causal representation per input timestep."""

    def encode(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self(input_ids)

    @staticmethod
    def _validate(states: Any, input_ids: torch.Tensor, source: str) -> torch.Tensor:
        if not isinstance(states, torch.Tensor):
            raise TypeError(f"{source} did not return a tensor representation")
        if states.ndim != 3:
            raise ValueError(
                f"{source} representation must have shape [batch, time, features]"
            )
        if states.shape[:2] != input_ids.shape:
            raise ValueError(f"{source} representation does not align with input_ids")
        return states


class ASMModelAdapter(ModelAdapter):
    """Expose the causal state sequence already produced by an ASM model."""

    def __init__(
        self,
        model: nn.Module,
        representation_dim: int | None = None,
        *,
        global_step: int | None = None,
    ) -> None:
        config = getattr(model, "config", None)
        inferred = getattr(config, "d_state", None)
        super().__init__(
            model, representation_dim if representation_dim is not None else inferred
        )
        self.global_step = global_step

    def set_global_step(self, global_step: int) -> None:
        self.global_step = int(global_step)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        options = {
            "return_states": True,
            "collect_diagnostics": False,
        }
        if self.global_step is not None:
            options["global_step"] = self.global_step
        output = self.model(input_ids, **options)
        if not isinstance(output, dict) or "states" not in output:
            raise KeyError("ASM output must contain 'states' when return_states=True")
        return self._validate(output["states"], input_ids, "ASM")


class TransformerModelAdapter(ModelAdapter):
    """Expose final causal hidden states from the repository Transformer."""

    def __init__(self, model: nn.Module, representation_dim: int | None = None) -> None:
        config = getattr(model, "config", None)
        inferred = getattr(config, "d_model", None)
        super().__init__(
            model, representation_dim if representation_dim is not None else inferred
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        output = self.model(input_ids, return_hidden_states=True)
        if not isinstance(output, dict) or "hidden_states" not in output:
            raise KeyError(
                "Transformer output must contain 'hidden_states' when "
                "return_hidden_states=True"
            )
        return self._validate(output["hidden_states"], input_ids, "Transformer")


__all__ = ["ASMModelAdapter", "ModelAdapter", "TransformerModelAdapter"]
