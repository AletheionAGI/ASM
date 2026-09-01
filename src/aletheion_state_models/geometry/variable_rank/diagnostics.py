"""Linear and local diagnostics for controlled ASM-VR cycles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import torch
from torch import Tensor

LinearOperator = Tensor | Callable[[Tensor], Tensor]


@dataclass(frozen=True)
class CycleDiagnostics:
    """Measurements of an ordered linearized cycle."""

    holonomy: Tensor
    singular_values: Tensor
    numerical_rank: int
    initial_rank: int
    rank_deficit: int
    relative_tolerance: float
    frobenius_deviation: Tensor
    dissipation: Tensor
    minimum_dissipation_eigenvalue: Tensor
    ranks_along_cycle: tuple[int, ...]


def ordered_composition(operators: Sequence[Tensor]) -> Tensor:
    """Compose chronological maps, returning ``J_n @ ... @ J_1``.

    Each map's input width must match the preceding map's output height. Empty
    compositions are ambiguous and therefore rejected.
    """
    if not operators:
        raise ValueError("at least one operator is required")
    result = operators[0]
    if result.ndim != 2:
        raise ValueError("cycle operators must be two-dimensional")
    for operator in operators[1:]:
        if operator.ndim != 2:
            raise ValueError("cycle operators must be two-dimensional")
        if operator.shape[1] != result.shape[0]:
            raise ValueError("adjacent cycle operator dimensions do not match")
        result = operator @ result
    return result


def dense_linear_operator(
    operator: LinearOperator,
    input_dimension: int,
    *,
    reference: Tensor | None = None,
) -> Tensor:
    """Materialize a matrix or a matrix-free linear callable on basis vectors."""
    if isinstance(operator, Tensor):
        if operator.ndim != 2 or operator.shape[1] != input_dimension:
            raise ValueError("linear operator has an incompatible shape")
        return operator
    if input_dimension < 1:
        raise ValueError("input_dimension must be positive")
    template = reference if reference is not None else torch.empty((), dtype=torch.float32)
    basis = torch.eye(input_dimension, device=template.device, dtype=template.dtype)
    columns = [operator(basis[index]) for index in range(input_dimension)]
    return torch.stack(columns, dim=-1)


def jacobian_operator(loop_map: Callable[[Tensor], Tensor], point: Tensor) -> Tensor:
    """Return the dense autograd Jacobian of ``loop_map`` at ``point``.

    Input and output are flattened so diagnostics always receive a matrix. The
    returned Jacobian remains differentiable when PyTorch supports the requested
    higher-order derivatives.
    """
    if not (point.is_floating_point() or point.is_complex()):
        raise TypeError("Jacobian points must use a floating or complex dtype")
    jacobian = torch.autograd.functional.jacobian(
        loop_map,
        point,
        create_graph=torch.is_grad_enabled(),
        strict=False,
        vectorize=True,
    )
    output_size = loop_map(point).numel()
    return jacobian.reshape(output_size, point.numel())


def numerical_rank(matrix: Tensor, relative_tolerance: float = 1e-6) -> int:
    """Compute rank using a threshold relative to the largest singular value."""
    if matrix.ndim != 2:
        raise ValueError("numerical_rank expects a matrix")
    if relative_tolerance < 0:
        raise ValueError("relative_tolerance must be non-negative")
    singular_values = torch.linalg.svdvals(matrix.to(dtype=_diagnostic_dtype(matrix)))
    if singular_values.numel() == 0:
        return 0
    threshold = relative_tolerance * singular_values.max()
    return int(torch.count_nonzero(singular_values > threshold).item())


def diagnose_cycle(
    *,
    operators: Sequence[Tensor] | None = None,
    loop_map: Callable[[Tensor], Tensor] | None = None,
    point: Tensor | None = None,
    metric: Tensor | None = None,
    relative_tolerance: float = 1e-6,
) -> CycleDiagnostics:
    """Diagnose an ordered linear cycle or a nonlinear loop via autograd.

    Supply either chronological ``operators`` or both ``loop_map`` and ``point``.
    Forcing belongs inside ``loop_map`` only when its state dependence is intended;
    constant forcing naturally vanishes from the Jacobian.
    """
    if operators is not None:
        if loop_map is not None or point is not None:
            raise ValueError("choose operators or loop_map/point, not both")
        holonomy = ordered_composition(operators)
        ranks = tuple(numerical_rank(item, relative_tolerance) for item in operators)
    elif loop_map is not None and point is not None:
        holonomy = jacobian_operator(loop_map, point)
        ranks = (numerical_rank(holonomy, relative_tolerance),)
    else:
        raise ValueError("supply operators or both loop_map and point")

    if holonomy.shape[0] != holonomy.shape[1]:
        raise ValueError("a closed cycle must have equal initial and final dimensions")
    work = holonomy.to(dtype=_diagnostic_dtype(holonomy))
    dimension = work.shape[0]
    identity = torch.eye(dimension, dtype=work.dtype, device=work.device)
    cycle_metric = identity if metric is None else metric.to(device=work.device, dtype=work.dtype)
    if cycle_metric.shape != (dimension, dimension):
        raise ValueError("metric must be square with the cycle dimension")
    dissipation = cycle_metric - work.transpose(-2, -1) @ cycle_metric @ work
    singular_values = torch.linalg.svdvals(work)
    rank = numerical_rank(work, relative_tolerance)
    deviation = torch.linalg.matrix_norm(work - identity, ord="fro") / max(dimension, 1) ** 0.5
    min_dissipation = torch.linalg.eigvalsh((dissipation + dissipation.T) * 0.5).min()
    return CycleDiagnostics(
        holonomy=work,
        singular_values=singular_values,
        numerical_rank=rank,
        initial_rank=dimension,
        rank_deficit=dimension - rank,
        relative_tolerance=relative_tolerance,
        frobenius_deviation=deviation,
        dissipation=dissipation,
        minimum_dissipation_eigenvalue=min_dissipation,
        ranks_along_cycle=ranks,
    )


def commutator(first: Tensor, second: Tensor) -> Tensor:
    """Return the ordered commutator ``first @ second - second @ first``."""
    if first.ndim != 2 or first.shape != second.shape or first.shape[0] != first.shape[1]:
        raise ValueError("commutator operands must be square matrices of equal shape")
    return first @ second - second @ first


def _diagnostic_dtype(tensor: Tensor) -> torch.dtype:
    if tensor.is_complex():
        return torch.complex64 if tensor.dtype == torch.complex32 else tensor.dtype
    return torch.float64 if tensor.dtype == torch.float64 else torch.float32


__all__ = [
    "CycleDiagnostics",
    "LinearOperator",
    "commutator",
    "dense_linear_operator",
    "diagnose_cycle",
    "jacobian_operator",
    "numerical_rank",
    "ordered_composition",
]
