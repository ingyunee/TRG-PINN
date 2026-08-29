"""Loss primitives shared by PINN and TRG-PINN."""

from __future__ import annotations

import torch


def normalized_weighted_residual_loss(
    residual: torch.Tensor,
    weight: torch.Tensor,
    *,
    epsilon: float = 1.0e-8,
) -> torch.Tensor:
    if residual.shape[0] != weight.shape[0]:
        raise ValueError("Residual and weight batch dimensions do not match.")
    return (weight * residual.pow(2)).sum() / (weight.sum() + float(epsilon))


def composite_loss(
    *,
    initial_loss: torch.Tensor,
    boundary_loss: torch.Tensor,
    pde_loss: torch.Tensor,
    initial_weight: float,
    boundary_weight: float,
    pde_weight: float,
) -> torch.Tensor:
    return (
        float(initial_weight) * initial_loss
        + float(boundary_weight) * boundary_loss
        + float(pde_weight) * pde_loss
    )
