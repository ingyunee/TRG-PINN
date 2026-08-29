"""Sampling routines with the exact random-call order used in the manuscript runs."""

from __future__ import annotations

from typing import Protocol

import torch


class Burgers1DConfigLike(Protocol):
    x_min: float
    x_max: float
    t_min: float
    t_max: float
    x0: float
    uL: float
    uR: float


def sample_burgers_1d_initial(
    n: int,
    cfg: Burgers1DConfigLike,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n_left = int(n) // 2
    n_right = int(n) - n_left

    x_left = cfg.x_min + (cfg.x0 - cfg.x_min) * torch.rand(
        n_left, 1, device=device, dtype=dtype
    )
    x_right = cfg.x0 + (cfg.x_max - cfg.x0) * torch.rand(
        n_right, 1, device=device, dtype=dtype
    )
    x = torch.cat([x_left, x_right], dim=0)
    t = torch.zeros_like(x)
    target = torch.cat(
        [torch.full_like(x_left, cfg.uL), torch.full_like(x_right, cfg.uR)], dim=0
    )

    permutation = torch.randperm(int(n), device=device)
    return x[permutation], t[permutation], target[permutation]


def sample_burgers_1d_boundary(
    n: int,
    cfg: Burgers1DConfigLike,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n_left = int(n) // 2
    n_right = int(n) - n_left

    t_left = cfg.t_min + (cfg.t_max - cfg.t_min) * torch.rand(
        n_left, 1, device=device, dtype=dtype
    )
    t_right = cfg.t_min + (cfg.t_max - cfg.t_min) * torch.rand(
        n_right, 1, device=device, dtype=dtype
    )
    x_left = torch.full_like(t_left, cfg.x_min)
    x_right = torch.full_like(t_right, cfg.x_max)
    target_left = torch.full_like(t_left, cfg.uL)
    target_right = torch.full_like(t_right, cfg.uR)

    x = torch.cat([x_left, x_right], dim=0)
    t = torch.cat([t_left, t_right], dim=0)
    target = torch.cat([target_left, target_right], dim=0)

    permutation = torch.randperm(int(n), device=device)
    return x[permutation], t[permutation], target[permutation]


def sample_burgers_1d_interior(
    n: int,
    cfg: Burgers1DConfigLike,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    x = cfg.x_min + (cfg.x_max - cfg.x_min) * torch.rand(
        int(n), 1, device=device, dtype=dtype
    )
    t = cfg.t_min + (cfg.t_max - cfg.t_min) * torch.rand(
        int(n), 1, device=device, dtype=dtype
    )
    return x, t


class Euler1DConfigLike(Protocol):
    x_min: float
    x_max: float
    t_min: float
    t_max: float
    x0: float
    rhoL: float
    uL: float
    pL: float
    rhoR: float
    uR: float
    pR: float


def _euler_primitive_state(
    n: int,
    *,
    rho: float,
    velocity: float,
    pressure: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    rho_values = torch.full((int(n), 1), float(rho), device=device, dtype=dtype)
    velocity_values = torch.full(
        (int(n), 1), float(velocity), device=device, dtype=dtype
    )
    pressure_values = torch.full(
        (int(n), 1), float(pressure), device=device, dtype=dtype
    )
    return torch.cat([rho_values, velocity_values, pressure_values], dim=1)


def sample_euler_1d_initial(
    n: int,
    cfg: Euler1DConfigLike,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n_left = int(n) // 2
    n_right = int(n) - n_left

    x_left = cfg.x_min + (cfg.x0 - cfg.x_min) * torch.rand(
        n_left, 1, device=device, dtype=dtype
    )
    x_right = cfg.x0 + (cfg.x_max - cfg.x0) * torch.rand(
        n_right, 1, device=device, dtype=dtype
    )
    x = torch.cat([x_left, x_right], dim=0)
    t = torch.zeros_like(x)
    target = torch.cat(
        [
            _euler_primitive_state(
                n_left,
                rho=cfg.rhoL,
                velocity=cfg.uL,
                pressure=cfg.pL,
                device=device,
                dtype=dtype,
            ),
            _euler_primitive_state(
                n_right,
                rho=cfg.rhoR,
                velocity=cfg.uR,
                pressure=cfg.pR,
                device=device,
                dtype=dtype,
            ),
        ],
        dim=0,
    )

    permutation = torch.randperm(int(n), device=device)
    return x[permutation], t[permutation], target[permutation]


def sample_euler_1d_boundary(
    n: int,
    cfg: Euler1DConfigLike,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample the fixed reservoir states used by the reported Sod benchmark."""

    n_left = int(n) // 2
    n_right = int(n) - n_left

    t_left = cfg.t_min + (cfg.t_max - cfg.t_min) * torch.rand(
        n_left, 1, device=device, dtype=dtype
    )
    t_right = cfg.t_min + (cfg.t_max - cfg.t_min) * torch.rand(
        n_right, 1, device=device, dtype=dtype
    )
    x_left = torch.full_like(t_left, cfg.x_min)
    x_right = torch.full_like(t_right, cfg.x_max)
    x = torch.cat([x_left, x_right], dim=0)
    t = torch.cat([t_left, t_right], dim=0)
    target = torch.cat(
        [
            _euler_primitive_state(
                n_left,
                rho=cfg.rhoL,
                velocity=cfg.uL,
                pressure=cfg.pL,
                device=device,
                dtype=dtype,
            ),
            _euler_primitive_state(
                n_right,
                rho=cfg.rhoR,
                velocity=cfg.uR,
                pressure=cfg.pR,
                device=device,
                dtype=dtype,
            ),
        ],
        dim=0,
    )

    permutation = torch.randperm(int(n), device=device)
    return x[permutation], t[permutation], target[permutation]


def sample_euler_1d_interior(
    n: int,
    cfg: Euler1DConfigLike,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    x = cfg.x_min + (cfg.x_max - cfg.x_min) * torch.rand(
        int(n), 1, device=device, dtype=dtype
    )
    t = cfg.t_min + (cfg.t_max - cfg.t_min) * torch.rand(
        int(n), 1, device=device, dtype=dtype
    )
    return x, t


class ShallowWater1DConfigLike(Protocol):
    x_min: float
    x_max: float
    t_min: float
    t_max: float
    x0: float
    hL: float
    qL: float
    hR: float
    qR: float


def _shallowwater_state(
    count: int,
    *,
    depth: float,
    discharge: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    return torch.cat(
        [
            torch.full(
                (int(count), 1),
                float(depth),
                device=device,
                dtype=dtype,
            ),
            torch.full(
                (int(count), 1),
                float(discharge),
                device=device,
                dtype=dtype,
            ),
        ],
        dim=1,
    )


def sample_shallowwater_1d_initial(
    n: int,
    cfg: ShallowWater1DConfigLike,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n_left = int(n) // 2
    n_right = int(n) - n_left

    x_left = cfg.x_min + (cfg.x0 - cfg.x_min) * torch.rand(
        n_left, 1, device=device, dtype=dtype
    )
    x_right = cfg.x0 + (cfg.x_max - cfg.x0) * torch.rand(
        n_right, 1, device=device, dtype=dtype
    )
    x = torch.cat([x_left, x_right], dim=0)
    t = torch.zeros_like(x)
    target = torch.cat(
        [
            _shallowwater_state(
                n_left,
                depth=cfg.hL,
                discharge=cfg.qL,
                device=device,
                dtype=dtype,
            ),
            _shallowwater_state(
                n_right,
                depth=cfg.hR,
                discharge=cfg.qR,
                device=device,
                dtype=dtype,
            ),
        ],
        dim=0,
    )

    permutation = torch.randperm(int(n), device=device)
    return x[permutation], t[permutation], target[permutation]


def sample_shallowwater_1d_boundary(
    n: int,
    cfg: ShallowWater1DConfigLike,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n_left = int(n) // 2
    n_right = int(n) - n_left

    t_left = cfg.t_min + (cfg.t_max - cfg.t_min) * torch.rand(
        n_left, 1, device=device, dtype=dtype
    )
    t_right = cfg.t_min + (cfg.t_max - cfg.t_min) * torch.rand(
        n_right, 1, device=device, dtype=dtype
    )
    x_left = torch.full_like(t_left, cfg.x_min)
    x_right = torch.full_like(t_right, cfg.x_max)

    x = torch.cat([x_left, x_right], dim=0)
    t = torch.cat([t_left, t_right], dim=0)
    target = torch.cat(
        [
            _shallowwater_state(
                n_left,
                depth=cfg.hL,
                discharge=cfg.qL,
                device=device,
                dtype=dtype,
            ),
            _shallowwater_state(
                n_right,
                depth=cfg.hR,
                discharge=cfg.qR,
                device=device,
                dtype=dtype,
            ),
        ],
        dim=0,
    )

    permutation = torch.randperm(int(n), device=device)
    return x[permutation], t[permutation], target[permutation]


def sample_shallowwater_1d_interior(
    n: int,
    cfg: ShallowWater1DConfigLike,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    x = cfg.x_min + (cfg.x_max - cfg.x_min) * torch.rand(
        int(n), 1, device=device, dtype=dtype
    )
    t = cfg.t_min + (cfg.t_max - cfg.t_min) * torch.rand(
        int(n), 1, device=device, dtype=dtype
    )
    return x, t
