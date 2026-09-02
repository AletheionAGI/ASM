"""Exact budget-matched G, D, and C heads registered for ATTR-RTG."""

from __future__ import annotations

import torch
from torch import nn

HEAD_PARAMETER_COUNTS = {"G": 5_724, "D": 33_381, "G+D": 39_105, "C": 39_123}
HEAD_SEED_OFFSETS = {"G": 60_000, "D": 70_000, "C": 80_000}


class TwoLinearGELU(nn.Module):
    """A two-linear MLP with the only registered parameter-free nonlinearity."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.input = nn.Linear(input_dim, hidden_dim, bias=True)
        self.output = nn.Linear(hidden_dim, output_dim, bias=True)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.input.weight)
        nn.init.zeros_(self.input.bias)
        nn.init.xavier_uniform_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.output(torch.nn.functional.gelu(self.input(values)))


class TransitionG(TwoLinearGELU):
    def __init__(self) -> None:
        super().__init__(60, 64, 28)


class PhysicalD(TwoLinearGELU):
    def __init__(self) -> None:
        super().__init__(28, 64, 485)


class DirectC(TwoLinearGELU):
    """Return the registered unsafe logit; sigmoid is applied only at inference."""

    def __init__(self) -> None:
        super().__init__(60, 631, 1)


def count_parameters(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def validate_head_budgets(g: nn.Module, d: nn.Module, c: nn.Module) -> None:
    actual = {
        "G": count_parameters(g),
        "D": count_parameters(d),
        "G+D": count_parameters(g) + count_parameters(d),
        "C": count_parameters(c),
    }
    if actual != HEAD_PARAMETER_COUNTS:
        raise ValueError(f"ATTR-RTG head budgets differ: {actual}")


def build_registered_heads(training_seed: int) -> tuple[TransitionG, PhysicalD, DirectC]:
    """Build the three heads with their independent registered RNG namespaces."""
    torch.manual_seed(HEAD_SEED_OFFSETS["G"] + training_seed)
    g = TransitionG()
    torch.manual_seed(HEAD_SEED_OFFSETS["D"] + training_seed)
    d = PhysicalD()
    torch.manual_seed(HEAD_SEED_OFFSETS["C"] + training_seed)
    c = DirectC()
    validate_head_budgets(g, d, c)
    return g, d, c
