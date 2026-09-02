"""Causal pre/post state export from frozen ATTR-RTG backbones."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn


class CausalStateExporter(ABC):
    """Recompute complete prefixes and expose only their final causal state."""

    representation_dim: int

    def __init__(self, model: nn.Module, max_seq_len: int = 64) -> None:
        self.model = model
        self.max_seq_len = max_seq_len
        self.model.eval()

    def _validate_prefix(self, input_ids: torch.Tensor) -> None:
        if input_ids.ndim != 2 or input_ids.shape[1] < 1:
            raise ValueError("causal prefix must have shape [batch,time] with time >= 1")
        if input_ids.shape[1] > self.max_seq_len:
            raise ValueError("causal prefix exceeds registered context 64")
        if input_ids.dtype != torch.long:
            raise ValueError("causal prefix tokens must be torch.long")

    @abstractmethod
    def _forward_states(self, input_ids: torch.Tensor) -> torch.Tensor:
        pass

    @torch.no_grad()
    def export(self, input_ids: torch.Tensor) -> torch.Tensor:
        self._validate_prefix(input_ids)
        self.model.eval()
        parameter = next(self.model.parameters(), None)
        device = input_ids.device if parameter is None else parameter.device
        states = self._forward_states(input_ids.to(device))
        if states.shape != input_ids.shape + (self.representation_dim,):
            raise ValueError("backbone state sequence does not align with causal prefix")
        return states[:, -1, :]

    def export_transition(
        self, history: torch.Tensor, candidate_frame: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if candidate_frame.ndim != 2 or candidate_frame.shape != (history.shape[0], 4):
            raise ValueError("candidate frame must contain exactly four bytes per history")
        pre = self.export(history)
        post = self.export(torch.cat((history, candidate_frame), dim=1))
        return pre, post


class ASMStateExporter(CausalStateExporter):
    representation_dim = 28

    def _forward_states(self, input_ids: torch.Tensor) -> torch.Tensor:
        output = self.model(
            input_ids,
            return_states=True,
            global_step=1_000,
            collect_diagnostics=False,
        )
        if not isinstance(output, dict) or not isinstance(output.get("states"), torch.Tensor):
            raise TypeError("ASM did not return its recurrent states")
        return output["states"]


class TransformerReadoutExporter(CausalStateExporter):
    representation_dim = 32

    def _forward_states(self, input_ids: torch.Tensor) -> torch.Tensor:
        output = self.model(input_ids, return_hidden_states=True)
        if not isinstance(output, dict) or not isinstance(
            output.get("hidden_states"), torch.Tensor
        ):
            raise TypeError("Transformer did not return post-normalization readouts")
        return output["hidden_states"]
