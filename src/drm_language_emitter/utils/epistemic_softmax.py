"""Epistemic Softmax operators adapted for ASM.

Derived from ``gnai-creator/Epistemic_Softmax/epistemic_softmax.py`` by
Felipe Maya Muniz (2025).  The original probability operator is retained, and
``EpistemicConfidenceGate`` exposes its confidence factor without forcing a
uniform mixture.  ASM-CM-E uses that factor to abstain from uncertain memory
reads/writes rather than averaging unrelated memories.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class GateNetwork(nn.Module):
    """MLP producing a scalar gate in ``[0, 1]``."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        current_dim = input_dim
        for _ in range(num_layers):
            layers.extend((nn.Linear(current_dim, hidden_dim), nn.ReLU()))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            current_dim = hidden_dim
        layers.extend((nn.Linear(current_dim, 1), nn.Sigmoid()))
        self.network = nn.Sequential(*layers)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        return self.network(context)


class EpistemicConfidenceGate(nn.Module):
    """Two-factor evidence gate shared by Epistemic Softmax and ASM-CM-E.

    ``q_local`` and ``q_consensus`` are deliberately defined as positive
    evidence/reliability, not uncertainty.  Their product is therefore a
    confidence value and ``1 - confidence`` is the reported uncertainty.
    """

    def __init__(
        self,
        context_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.0,
        initial_confidence: float = 0.9,
    ) -> None:
        super().__init__()
        self.local_evidence = GateNetwork(context_dim, hidden_dim, num_layers, dropout)
        self.context_consensus = GateNetwork(context_dim, hidden_dim, num_layers, dropout)
        factor = math.sqrt(initial_confidence)
        bias = math.log(factor / (1.0 - factor))
        for network in (self.local_evidence, self.context_consensus):
            output = next(
                module for module in reversed(network.network) if isinstance(module, nn.Linear)
            )
            nn.init.zeros_(output.weight)
            nn.init.constant_(output.bias, bias)

    def forward(
        self, context: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        q_local = self.local_evidence(context).squeeze(-1)
        q_consensus = self.context_consensus(context).squeeze(-1)
        confidence = (q_local * q_consensus).clamp(0.0, 1.0)
        return confidence, 1.0 - confidence, q_local, q_consensus


class EpistemicSoftmax(nn.Module):
    """Temperature-scaled softmax mixed with uniform mass by confidence.

    This operator remains available for controlled output-head experiments.
    It is not used to address ASM-CM-E memory because a uniform fallback would
    average unrelated memories instead of abstaining.
    """

    def __init__(
        self,
        context_dim: int,
        base_temperature: float = 1.0,
        tau_threshold: float = 0.5,
        epsilon: float = 1e-8,
        gate_hidden_dim: int = 64,
        gate_num_layers: int = 2,
        gate_dropout: float = 0.1,
        initial_confidence: float = 0.9,
    ) -> None:
        super().__init__()
        self.base_temperature = base_temperature
        self.tau_threshold = tau_threshold
        self.epsilon = epsilon
        self.confidence_gate = EpistemicConfidenceGate(
            context_dim,
            gate_hidden_dim,
            gate_num_layers,
            gate_dropout,
            initial_confidence,
        )

    def forward(
        self, logits: torch.Tensor, context: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        confidence, uncertainty, q_local, q_consensus = self.confidence_gate(context)
        safe_confidence = confidence.clamp_min(self.epsilon)
        tau = torch.where(
            safe_confidence < self.tau_threshold,
            self.base_temperature / safe_confidence,
            torch.full_like(safe_confidence, self.base_temperature),
        ).unsqueeze(-1)
        probabilities = F.softmax(logits / tau, dim=-1)
        uniform = torch.full_like(probabilities, 1.0 / logits.shape[-1])
        gated = confidence.unsqueeze(-1) * probabilities + uncertainty.unsqueeze(-1) * uniform
        return gated, uncertainty, q_local, q_consensus


__all__ = ["EpistemicConfidenceGate", "EpistemicSoftmax", "GateNetwork"]
