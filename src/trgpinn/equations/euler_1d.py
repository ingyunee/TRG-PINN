"""Canonical 1D compressible Euler implementation used for the manuscript.

The module preserves the reported Sod benchmark's network parameterization,
sampling order, residual scaling, detached trace-ratio gate, paired RNG
restoration, optimizer/scheduler order, and final-checkpoint metrics.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
import math
import platform
import time
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

from trgpinn.gate import (
    centered_euler_trace_ratio_gate_1d,
    residual_weight,
    trace_schedule_1d,
    trace_schedule_from_iteration_1d,
)
from trgpinn.models import PrimitiveEulerMLP1D, trainable_parameter_count
from trgpinn.sampling import (
    sample_euler_1d_boundary,
    sample_euler_1d_initial,
    sample_euler_1d_interior,
)
from trgpinn.utils import (
    capture_rng_state,
    clone_state_dict_cpu,
    configure_torch_runtime,
    ensure_unprotected_output,
    load_state_dict_to_model,
    resolve_device,
    restore_rng_state,
    set_seed,
    write_json_atomic,
)


@dataclass
class Euler1DConfig:
    seed: int = 2026
    device: str = "auto"
    dtype: str = "float32"
    output_dir: str = "runs_euler_sod_trace_ratio_paper"
    experiment_name: str = "euler_sod_trace_ratio_original_schedule"
    save_outputs: bool = True

    x_min: float = -0.5
    x_max: float = 0.5
    t_min: float = 0.0
    t_max: float = 0.2
    gamma: float = 1.4
    rhoL: float = 1.0
    uL: float = 0.0
    pL: float = 1.0
    rhoR: float = 0.125
    uR: float = 0.0
    pR: float = 0.1
    x0: float = 0.0

    width: int = 128
    depth: int = 6
    activation: str = "tanh"
    rho_floor: float = 1.0e-4
    p_floor: float = 1.0e-4

    warmup_iters: int = 4000
    gated_iters: int = 6000
    lr_warmup: float = 8.0e-4
    lr_gated: float = 3.0e-4
    weight_decay: float = 1.0e-2
    grad_clip: float = 1.0

    n_f: int = 12000
    n_ic: int = 1500
    n_bc: int = 1500

    w_ic: float = 100.0
    w_bc: float = 30.0
    w_pde: float = 1.0

    h_max_factor: float = 5.0
    h_min_factor: float = 2.0
    cmin_start: float = 0.50
    cmin_end: float = 0.70
    beta: float = 0.05
    residual_floor: float = 0.02

    trace_norm_epsilon: float = 1.0e-12
    trace_ratio_epsilon: float = 1.0e-6
    batch_mean_epsilon: float = 1.0e-8
    weighted_loss_epsilon: float = 1.0e-8
    relative_error_epsilon: float = 1.0e-12

    print_every: int = 500
    history_every: int = 50

    eval_nx: int = 700
    eval_nt: int = 301
    slice_nx: int = 1600
    t_plot: float = 0.2

    n_control_volumes: int = 64
    cv_quad_nx: int = 80
    cv_quad_nt: int = 80
    cv_min_width: float = 0.08
    cv_min_duration: float = 0.04

    @classmethod
    def from_legacy_mapping(
        cls,
        mapping: dict[str, Any],
        **overrides: Any,
    ) -> "Euler1DConfig":
        valid = {field.name for field in fields(cls)}
        values = {key: value for key, value in mapping.items() if key in valid}
        values.update(overrides)
        return cls(**values)

    def smoke_copy(
        self,
        *,
        seed: int = 2026,
        device: str = "auto",
    ) -> "Euler1DConfig":
        return replace(
            self,
            seed=int(seed),
            device=device,
            warmup_iters=2,
            gated_iters=2,
            n_f=64,
            n_ic=32,
            n_bc=32,
            print_every=1,
            history_every=1,
            eval_nx=80,
            eval_nt=40,
            slice_nx=80,
            n_control_volumes=4,
            cv_quad_nx=12,
            cv_quad_nt=12,
            cv_min_width=0.08,
            cv_min_duration=0.04,
        )


def build_model(cfg: Euler1DConfig) -> PrimitiveEulerMLP1D:
    return PrimitiveEulerMLP1D(
        x_min=cfg.x_min,
        x_max=cfg.x_max,
        t_min=cfg.t_min,
        t_max=cfg.t_max,
        hidden_width=cfg.width,
        hidden_layers=cfg.depth,
        activation=cfg.activation,
        rho_floor=cfg.rho_floor,
        p_floor=cfg.p_floor,
        left_state=(cfg.rhoL, cfg.uL, cfg.pL),
        right_state=(cfg.rhoR, cfg.uR, cfg.pR),
    )


def cat_xt(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    return torch.cat([x, t], dim=1)


def scales(cfg: Euler1DConfig) -> tuple[float, float, float, float, float]:
    rho_scale = max(cfg.rhoL, cfg.rhoR, 1.0)
    p_scale = max(cfg.pL, cfg.pR, 1.0)
    c_left = math.sqrt(cfg.gamma * cfg.pL / cfg.rhoL)
    c_right = math.sqrt(cfg.gamma * cfg.pR / cfg.rhoR)
    velocity_scale = max(abs(cfg.uL), abs(cfg.uR), c_left, c_right, 1.0)
    momentum_scale = max(rho_scale * velocity_scale, 1.0)
    energy_scale = (
        p_scale / (cfg.gamma - 1.0)
        + 0.5 * rho_scale * velocity_scale**2
    )
    return (
        rho_scale,
        velocity_scale,
        p_scale,
        momentum_scale,
        energy_scale,
    )


def h0(cfg: Euler1DConfig) -> float:
    return (cfg.x_max - cfg.x_min) / math.sqrt(cfg.n_f)


def schedule_from_progress(
    progress: float,
    cfg: Euler1DConfig,
) -> tuple[float, float, float]:
    return trace_schedule_1d(progress, cfg)


def schedule_from_iter(
    iteration: int,
    total_iterations: int,
    cfg: Euler1DConfig,
) -> tuple[float, float, float]:
    return trace_schedule_from_iteration_1d(iteration, total_iterations, cfg)


def prefun(p, rho_k, p_k, c_k, gamma):
    if p > p_k:  # shock
        A = 2.0 / ((gamma + 1.0) * rho_k)
        B = (gamma - 1.0) / (gamma + 1.0) * p_k
        f = (p - p_k) * math.sqrt(A / (p + B))
        fd = math.sqrt(A / (p + B)) * (1.0 - 0.5 * (p - p_k) / (p + B))
    else:        # rarefaction
        pr = p / p_k
        f = 2.0 * c_k / (gamma - 1.0) * (pr ** ((gamma - 1.0) / (2.0 * gamma)) - 1.0)
        fd = (1.0 / (rho_k * c_k)) * pr ** (-(gamma + 1.0) / (2.0 * gamma))
    return f, fd


def solve_star_region(cfg: Euler1DConfig):
    g = cfg.gamma
    cL = math.sqrt(g * cfg.pL / cfg.rhoL)
    cR = math.sqrt(g * cfg.pR / cfg.rhoR)

    p_old = max(
        1.0e-8,
        0.5 * (cfg.pL + cfg.pR) - 0.125 * (cfg.uR - cfg.uL) * (cfg.rhoL + cfg.rhoR) * (cL + cR),
    )

    for _ in range(100):
        fL, fdL = prefun(p_old, cfg.rhoL, cfg.pL, cL, g)
        fR, fdR = prefun(p_old, cfg.rhoR, cfg.pR, cR, g)
        p_new = p_old - (fL + fR + cfg.uR - cfg.uL) / (fdL + fdR)
        p_new = max(1.0e-10, p_new)
        if abs(p_new - p_old) / (p_new + p_old + 1.0e-12) < 1.0e-12:
            p_old = p_new
            break
        p_old = p_new

    p_star = p_old
    fL, _ = prefun(p_star, cfg.rhoL, cfg.pL, cL, g)
    fR, _ = prefun(p_star, cfg.rhoR, cfg.pR, cR, g)
    u_star = 0.5 * (cfg.uL + cfg.uR + fR - fL)
    return p_star, u_star, cL, cR


def sod_wave_speeds(cfg: Euler1DConfig):
    g = cfg.gamma
    gm1 = g - 1.0
    gp1 = g + 1.0
    pstar, ustar, cL, cR = solve_star_region(cfg)

    # Left wave
    if pstar > cfg.pL:
        p_ratio = pstar / cfg.pL
        left_head = cfg.uL - cL * math.sqrt((gp1 / (2.0 * g)) * p_ratio + gm1 / (2.0 * g))
        left_tail = left_head
    else:
        cstarL = cL * (pstar / cfg.pL) ** (gm1 / (2.0 * g))
        left_head = cfg.uL - cL
        left_tail = ustar - cstarL

    # Right wave
    if pstar > cfg.pR:
        p_ratio = pstar / cfg.pR
        right_head = cfg.uR + cR * math.sqrt((gp1 / (2.0 * g)) * p_ratio + gm1 / (2.0 * g))
        right_tail = right_head
    else:
        cstarR = cR * (pstar / cfg.pR) ** (gm1 / (2.0 * g))
        right_head = cfg.uR + cR
        right_tail = ustar + cstarR

    contact = ustar
    return {
        "p_star": pstar,
        "u_star": ustar,
        "left_head": left_head,
        "left_tail": left_tail,
        "contact": contact,
        "right_head": right_head,
        "right_tail": right_tail,
    }


def euler_exact_np(X, T, cfg: Euler1DConfig):
    X = np.asarray(X)
    T = np.asarray(T)
    rho = np.zeros_like(X, dtype=np.float64)
    u = np.zeros_like(X, dtype=np.float64)
    p = np.zeros_like(X, dtype=np.float64)

    init = T <= 1.0e-14
    rho[init] = np.where(X[init] < cfg.x0, cfg.rhoL, cfg.rhoR)
    u[init] = np.where(X[init] < cfg.x0, cfg.uL, cfg.uR)
    p[init] = np.where(X[init] < cfg.x0, cfg.pL, cfg.pR)

    mask = ~init
    if not np.any(mask):
        return rho, u, p

    xi = np.zeros_like(X, dtype=np.float64)
    xi[mask] = (X[mask] - cfg.x0) / T[mask]

    g = cfg.gamma
    gm1 = g - 1.0
    gp1 = g + 1.0
    pstar, ustar, cL, cR = solve_star_region(cfg)

    left = mask & (xi <= ustar)
    right = mask & (xi > ustar)

    # Left side
    if pstar > cfg.pL:  # left shock
        p_ratio = pstar / cfg.pL
        SL = cfg.uL - cL * math.sqrt((gp1 / (2.0 * g)) * p_ratio + gm1 / (2.0 * g))
        left_state = left & (xi <= SL)
        left_star = left & (xi > SL)
        rho[left_state] = cfg.rhoL
        u[left_state] = cfg.uL
        p[left_state] = cfg.pL
        rho_star_L = cfg.rhoL * ((p_ratio + gm1 / gp1) / ((gm1 / gp1) * p_ratio + 1.0))
        rho[left_star] = rho_star_L
        u[left_star] = ustar
        p[left_star] = pstar
    else:  # left rarefaction
        cstarL = cL * (pstar / cfg.pL) ** (gm1 / (2.0 * g))
        headL = cfg.uL - cL
        tailL = ustar - cstarL
        left_state = left & (xi <= headL)
        left_star = left & (xi >= tailL)
        fan = left & (xi > headL) & (xi < tailL)
        rho[left_state] = cfg.rhoL
        u[left_state] = cfg.uL
        p[left_state] = cfg.pL
        rho_star_L = cfg.rhoL * (pstar / cfg.pL) ** (1.0 / g)
        rho[left_star] = rho_star_L
        u[left_star] = ustar
        p[left_star] = pstar
        xi_f = xi[fan]
        u_f = 2.0 / gp1 * (cL + 0.5 * gm1 * cfg.uL + xi_f)
        c_f = 2.0 / gp1 * (cL + 0.5 * gm1 * (cfg.uL - xi_f))
        rho[fan] = cfg.rhoL * (c_f / cL) ** (2.0 / gm1)
        u[fan] = u_f
        p[fan] = cfg.pL * (c_f / cL) ** (2.0 * g / gm1)

    # Right side
    if pstar > cfg.pR:  # right shock
        p_ratio = pstar / cfg.pR
        SR = cfg.uR + cR * math.sqrt((gp1 / (2.0 * g)) * p_ratio + gm1 / (2.0 * g))
        right_state = right & (xi >= SR)
        right_star = right & (xi < SR)
        rho[right_state] = cfg.rhoR
        u[right_state] = cfg.uR
        p[right_state] = cfg.pR
        rho_star_R = cfg.rhoR * ((p_ratio + gm1 / gp1) / ((gm1 / gp1) * p_ratio + 1.0))
        rho[right_star] = rho_star_R
        u[right_star] = ustar
        p[right_star] = pstar
    else:  # right rarefaction; not used in the standard Sod state, but kept general
        cstarR = cR * (pstar / cfg.pR) ** (gm1 / (2.0 * g))
        headR = cfg.uR + cR
        tailR = ustar + cstarR
        right_state = right & (xi >= headR)
        right_star = right & (xi <= tailR)
        fan = right & (xi < headR) & (xi > tailR)
        rho[right_state] = cfg.rhoR
        u[right_state] = cfg.uR
        p[right_state] = cfg.pR
        rho_star_R = cfg.rhoR * (pstar / cfg.pR) ** (1.0 / g)
        rho[right_star] = rho_star_R
        u[right_star] = ustar
        p[right_star] = pstar
        xi_f = xi[fan]
        u_f = 2.0 / gp1 * (-cR + 0.5 * gm1 * cfg.uR + xi_f)
        c_f = 2.0 / gp1 * (cR - 0.5 * gm1 * (cfg.uR - xi_f))
        rho[fan] = cfg.rhoR * (c_f / cR) ** (2.0 / gm1)
        u[fan] = u_f
        p[fan] = cfg.pR * (c_f / cR) ** (2.0 * g / gm1)

    return rho, u, p


def primitive_to_conserved_np(rho, u, p, cfg: Euler1DConfig):
    m = rho * u
    E = p / (cfg.gamma - 1.0) + 0.5 * rho * u**2
    return rho, m, E


def flux_np(rho, u, p, cfg: Euler1DConfig):
    r, m, E = primitive_to_conserved_np(rho, u, p, cfg)
    F1 = m
    F2 = m * u + p
    F3 = u * (E + p)
    return F1, F2, F3


def primitive_to_conserved_torch(
    primitive: torch.Tensor,
    cfg: Euler1DConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rho = primitive[:, 0:1]
    velocity = primitive[:, 1:2]
    pressure = primitive[:, 2:3]
    momentum = rho * velocity
    energy = (
        pressure / (cfg.gamma - 1.0)
        + 0.5 * rho * velocity.pow(2)
    )
    return rho, momentum, energy


def flux_torch(
    primitive: torch.Tensor,
    cfg: Euler1DConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rho, momentum, energy = primitive_to_conserved_torch(primitive, cfg)
    velocity = primitive[:, 1:2]
    pressure = primitive[:, 2:3]
    return (
        momentum,
        momentum * velocity + pressure,
        velocity * (energy + pressure),
    )


def euler_residual(
    model: nn.Module,
    x: torch.Tensor,
    t: torch.Tensor,
    cfg: Euler1DConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    x = x.detach().clone().requires_grad_(True)
    t = t.detach().clone().requires_grad_(True)
    primitive = model(cat_xt(x, t))
    rho, momentum, energy = primitive_to_conserved_torch(primitive, cfg)
    flux_mass, flux_momentum, flux_energy = flux_torch(primitive, cfg)

    rho_t = torch.autograd.grad(
        rho,
        t,
        grad_outputs=torch.ones_like(rho),
        create_graph=True,
        retain_graph=True,
    )[0]
    momentum_t = torch.autograd.grad(
        momentum,
        t,
        grad_outputs=torch.ones_like(momentum),
        create_graph=True,
        retain_graph=True,
    )[0]
    energy_t = torch.autograd.grad(
        energy,
        t,
        grad_outputs=torch.ones_like(energy),
        create_graph=True,
        retain_graph=True,
    )[0]
    mass_flux_x = torch.autograd.grad(
        flux_mass,
        x,
        grad_outputs=torch.ones_like(flux_mass),
        create_graph=True,
        retain_graph=True,
    )[0]
    momentum_flux_x = torch.autograd.grad(
        flux_momentum,
        x,
        grad_outputs=torch.ones_like(flux_momentum),
        create_graph=True,
        retain_graph=True,
    )[0]
    energy_flux_x = torch.autograd.grad(
        flux_energy,
        x,
        grad_outputs=torch.ones_like(flux_energy),
        create_graph=True,
        retain_graph=True,
    )[0]

    residual_mass = rho_t + mass_flux_x
    residual_momentum = momentum_t + momentum_flux_x
    residual_energy = energy_t + energy_flux_x

    rho_scale, _, _, momentum_scale, energy_scale = scales(cfg)
    residual_norm_squared = (
        (residual_mass / rho_scale).pow(2)
        + (residual_momentum / momentum_scale).pow(2)
        + (residual_energy / energy_scale).pow(2)
    )
    return (
        primitive,
        residual_mass,
        residual_momentum,
        residual_energy,
        residual_norm_squared,
    )


def scaled_primitive_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    cfg: Euler1DConfig,
) -> torch.Tensor:
    rho_scale, velocity_scale, pressure_scale, _, _ = scales(cfg)
    return (
        ((prediction[:, 0:1] - target[:, 0:1]) / rho_scale).pow(2)
        + ((prediction[:, 1:2] - target[:, 1:2]) / velocity_scale).pow(2)
        + ((prediction[:, 2:3] - target[:, 2:3]) / pressure_scale).pow(2)
    ).mean()


def _model_runtime(model: nn.Module) -> tuple[torch.device, torch.dtype]:
    parameter = next(model.parameters())
    return parameter.device, parameter.dtype


def pinn_loss(
    model: nn.Module,
    cfg: Euler1DConfig,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    device, dtype = _model_runtime(model)

    x_ic, t_ic, target_ic = sample_euler_1d_initial(
        cfg.n_ic, cfg, device=device, dtype=dtype
    )
    x_bc, t_bc, target_bc = sample_euler_1d_boundary(
        cfg.n_bc, cfg, device=device, dtype=dtype
    )
    x_f, t_f = sample_euler_1d_interior(
        cfg.n_f, cfg, device=device, dtype=dtype
    )

    prediction_ic = model(cat_xt(x_ic, t_ic))
    prediction_bc = model(cat_xt(x_bc, t_bc))
    _, _, _, _, residual_norm_squared = euler_residual(
        model, x_f, t_f, cfg
    )

    loss_ic = scaled_primitive_mse(prediction_ic, target_ic, cfg)
    loss_bc = scaled_primitive_mse(prediction_bc, target_bc, cfg)
    loss_pde = residual_norm_squared.mean()
    total = (
        cfg.w_ic * loss_ic
        + cfg.w_bc * loss_bc
        + cfg.w_pde * loss_pde
    )
    return total, {
        "ic": loss_ic.detach(),
        "bc": loss_bc.detach(),
        "pde": loss_pde.detach(),
        "weighted_pde": loss_pde.detach(),
    }


def trg_loss(
    model: nn.Module,
    iteration: int,
    total_iterations: int,
    cfg: Euler1DConfig,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    device, dtype = _model_runtime(model)
    h_probe, cmin, progress = schedule_from_iter(
        iteration, total_iterations, cfg
    )

    x_ic, t_ic, target_ic = sample_euler_1d_initial(
        cfg.n_ic, cfg, device=device, dtype=dtype
    )
    x_bc, t_bc, target_bc = sample_euler_1d_boundary(
        cfg.n_bc, cfg, device=device, dtype=dtype
    )
    x_f, t_f = sample_euler_1d_interior(
        cfg.n_f, cfg, device=device, dtype=dtype
    )

    prediction_ic = model(cat_xt(x_ic, t_ic))
    prediction_bc = model(cat_xt(x_bc, t_bc))
    _, _, _, _, residual_norm_squared = euler_residual(
        model, x_f, t_f, cfg
    )

    rho_scale, velocity_scale, pressure_scale, _, _ = scales(cfg)
    gate_result = centered_euler_trace_ratio_gate_1d(
        model,
        x_f,
        t_f,
        h_probe=h_probe,
        cmin=cmin,
        x_min=cfg.x_min,
        x_max=cfg.x_max,
        rho_scale=rho_scale,
        velocity_scale=velocity_scale,
        pressure_scale=pressure_scale,
        beta=cfg.beta,
        norm_epsilon=cfg.trace_norm_epsilon,
        ratio_epsilon=cfg.trace_ratio_epsilon,
        mean_epsilon=cfg.batch_mean_epsilon,
    )
    weight = residual_weight(gate_result.gate, cfg.residual_floor)

    loss_ic = scaled_primitive_mse(prediction_ic, target_ic, cfg)
    loss_bc = scaled_primitive_mse(prediction_bc, target_bc, cfg)
    loss_pde_raw = residual_norm_squared.mean()
    loss_pde_weighted = (
        (weight * residual_norm_squared).sum()
        / (weight.sum() + cfg.weighted_loss_epsilon)
    )
    total = (
        cfg.w_ic * loss_ic
        + cfg.w_bc * loss_bc
        + cfg.w_pde * loss_pde_weighted
    )

    return total, {
        "ic": loss_ic.detach(),
        "bc": loss_bc.detach(),
        "pde": loss_pde_raw.detach(),
        "weighted_pde": loss_pde_weighted.detach(),
        "gate_mean": gate_result.gate.mean().detach(),
        "gate_max": gate_result.gate.max().detach(),
        "gate_active_gt_0p5": (
            gate_result.gate > 0.5
        ).to(dtype).mean().detach(),
        "gate_active_gt_0p1": (
            gate_result.gate > 0.1
        ).to(dtype).mean().detach(),
        "C_mean": gate_result.ratio.mean().detach(),
        "C_max": gate_result.ratio.max().detach(),
        "Jhat_mean": gate_result.normalized_jump.mean().detach(),
        "Jhat_max": gate_result.normalized_jump.max().detach(),
        "Jbar": gate_result.jump_mean.detach(),
        "W_mean": weight.mean().detach(),
        "W_min": weight.min().detach(),
        "h": torch.tensor(h_probe, device=device, dtype=dtype),
        "cmin": torch.tensor(cmin, device=device, dtype=dtype),
        "progress": torch.tensor(progress, device=device, dtype=dtype),
        "valid_frac": gate_result.valid.mean().detach(),
    }


def _train_phase(
    model: nn.Module,
    cfg: Euler1DConfig,
    *,
    phase: str,
) -> tuple[nn.Module, pd.DataFrame]:
    if phase == "warmup":
        iterations = cfg.warmup_iters
        learning_rate = cfg.lr_warmup
        eta_fraction = 0.05
        loss_function = lambda iteration: pinn_loss(model, cfg)
        total_offset = 0
    elif phase == "pinn":
        iterations = cfg.gated_iters
        learning_rate = cfg.lr_gated
        eta_fraction = 0.03
        loss_function = lambda iteration: pinn_loss(model, cfg)
        total_offset = cfg.warmup_iters
    elif phase == "trg_pinn":
        iterations = cfg.gated_iters
        learning_rate = cfg.lr_gated
        eta_fraction = 0.03
        loss_function = lambda iteration: trg_loss(
            model, iteration, cfg.gated_iters, cfg
        )
        total_offset = cfg.warmup_iters
    else:
        raise ValueError(f"Unknown training phase: {phase}")

    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=iterations,
        eta_min=learning_rate * eta_fraction,
    )

    history: list[dict[str, Any]] = []
    for iteration in range(1, iterations + 1):
        optimizer.zero_grad(set_to_none=True)
        loss, parts = loss_function(iteration)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        scheduler.step()

        if (
            iteration == 1
            or iteration % cfg.history_every == 0
            or iteration == iterations
        ):
            row: dict[str, Any] = {
                "phase": phase,
                "iter": iteration,
                "total_iter": total_offset + iteration,
                "loss": float(loss.detach().cpu()),
                "ic": float(parts["ic"].cpu()),
                "bc": float(parts["bc"].cpu()),
                "pde": float(parts["pde"].cpu()),
                "weighted_pde": float(parts["weighted_pde"].cpu()),
                "lr": float(optimizer.param_groups[0]["lr"]),
            }
            for key in (
                "gate_mean",
                "gate_max",
                "gate_active_gt_0p5",
                "gate_active_gt_0p1",
                "C_mean",
                "C_max",
                "Jhat_mean",
                "Jhat_max",
                "Jbar",
                "W_mean",
                "W_min",
                "h",
                "cmin",
                "progress",
                "valid_frac",
            ):
                if key in parts:
                    row[key] = float(parts[key].cpu())
            history.append(row)

        if (
            iteration == 1
            or iteration % cfg.print_every == 0
            or iteration == iterations
        ):
            print(
                f"[{phase}] {iteration:6d}/{iterations} "
                f"loss={float(loss.detach().cpu()):.3e} "
                f"ic={parts['ic'].item():.1e} "
                f"bc={parts['bc'].item():.1e} "
                f"pde={parts['pde'].item():.1e}"
            )

    return model, pd.DataFrame(history)


def train_warmup(
    model: nn.Module,
    cfg: Euler1DConfig,
) -> tuple[nn.Module, pd.DataFrame]:
    return _train_phase(model, cfg, phase="warmup")


def train_pinn_continuation(
    model: nn.Module,
    cfg: Euler1DConfig,
) -> tuple[nn.Module, pd.DataFrame]:
    return _train_phase(model, cfg, phase="pinn")


def train_trg_continuation(
    model: nn.Module,
    cfg: Euler1DConfig,
) -> tuple[nn.Module, pd.DataFrame]:
    return _train_phase(model, cfg, phase="trg_pinn")


@torch.no_grad()
def predict_primitive_points(
    model: nn.Module,
    x_values: np.ndarray,
    t_values: np.ndarray,
    *,
    batch_size: int = 65536,
) -> np.ndarray:
    x_array = np.asarray(x_values, dtype=np.float64)
    t_array = np.asarray(t_values, dtype=np.float64)
    x_flat = x_array.reshape(-1)
    t_flat = t_array.reshape(-1)
    if x_flat.shape != t_flat.shape:
        raise ValueError("x and t arrays must have the same shape.")

    parameter = next(model.parameters())
    outputs = []
    for start in range(0, x_flat.size, int(batch_size)):
        stop = min(start + int(batch_size), x_flat.size)
        coordinates = torch.tensor(
            np.stack([x_flat[start:stop], t_flat[start:stop]], axis=1),
            device=parameter.device,
            dtype=parameter.dtype,
        )
        outputs.append(model(coordinates).detach().cpu().numpy())

    primitive = np.concatenate(outputs, axis=0)
    return primitive.reshape(*x_array.shape, 3)


@torch.no_grad()
def predict_grid(
    model: nn.Module,
    cfg: Euler1DConfig,
    *,
    nx: int | None = None,
    nt: int | None = None,
) -> dict[str, np.ndarray]:
    nx = cfg.eval_nx if nx is None else int(nx)
    nt = cfg.eval_nt if nt is None else int(nt)
    x = np.linspace(cfg.x_min, cfg.x_max, nx)
    t = np.linspace(cfg.t_min, cfg.t_max, nt)
    X, T = np.meshgrid(x, t)
    primitive = predict_primitive_points(model, X, T)
    rho = primitive[:, :, 0]
    velocity = primitive[:, :, 1]
    pressure = primitive[:, :, 2]
    rho_exact, velocity_exact, pressure_exact = euler_exact_np(X, T, cfg)
    return {
        "x": x,
        "t": t,
        "X": X,
        "T": T,
        "rho": rho,
        "u": velocity,
        "p": pressure,
        "rho_exact": rho_exact,
        "u_exact": velocity_exact,
        "p_exact": pressure_exact,
    }


@torch.no_grad()
def predict_slice(
    model: nn.Module,
    cfg: Euler1DConfig,
    *,
    t_value: float | None = None,
    nx: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    time_value = cfg.t_plot if t_value is None else float(t_value)
    points = cfg.slice_nx if nx is None else int(nx)
    x = np.linspace(cfg.x_min, cfg.x_max, points)
    t = np.full_like(x, time_value)
    primitive = predict_primitive_points(model, x, t)
    return x, primitive[:, 0], primitive[:, 1], primitive[:, 2]


def compute_error_metrics(
    model: nn.Module,
    cfg: Euler1DConfig,
    *,
    grid: dict[str, np.ndarray] | None = None,
) -> dict[str, float]:
    if grid is None:
        grid = predict_grid(model, cfg)

    rho = grid["rho"]
    velocity = grid["u"]
    pressure = grid["p"]
    rho_exact = grid["rho_exact"]
    velocity_exact = grid["u_exact"]
    pressure_exact = grid["p_exact"]

    output: dict[str, float] = {}
    for name, prediction, exact in (
        ("rho", rho, rho_exact),
        ("u", velocity, velocity_exact),
        ("p", pressure, pressure_exact),
    ):
        error = prediction - exact
        output[f"{name}_space_time_l1"] = float(np.mean(np.abs(error)))
        output[f"{name}_space_time_l2"] = float(np.sqrt(np.mean(error**2)))
        output[f"{name}_space_time_rel_l2"] = float(
            np.sqrt(np.mean(error**2))
            / (
                np.sqrt(np.mean(exact**2))
                + cfg.relative_error_epsilon
            )
        )
        final_error = error[-1]
        final_exact = exact[-1]
        output[f"{name}_final_l1"] = float(np.mean(np.abs(final_error)))
        output[f"{name}_final_l2"] = float(
            np.sqrt(np.mean(final_error**2))
        )
        output[f"{name}_final_rel_l2"] = float(
            np.sqrt(np.mean(final_error**2))
            / (
                np.sqrt(np.mean(final_exact**2))
                + cfg.relative_error_epsilon
            )
        )
        output[f"{name}_min"] = float(prediction.min())
        output[f"{name}_max"] = float(prediction.max())

    rho_scale, velocity_scale, pressure_scale, _, _ = scales(cfg)
    scaled_error_squared = (
        ((rho - rho_exact) / rho_scale) ** 2
        + ((velocity - velocity_exact) / velocity_scale) ** 2
        + ((pressure - pressure_exact) / pressure_scale) ** 2
    )
    scaled_exact_squared = (
        (rho_exact / rho_scale) ** 2
        + (velocity_exact / velocity_scale) ** 2
        + (pressure_exact / pressure_scale) ** 2
    )
    output["primitive_scaled_space_time_rel_l2"] = float(
        np.sqrt(np.mean(scaled_error_squared))
        / (
            np.sqrt(np.mean(scaled_exact_squared))
            + cfg.relative_error_epsilon
        )
    )

    rho_error_final = ((rho[-1] - rho_exact[-1]) / rho_scale) ** 2
    velocity_error_final = (
        (velocity[-1] - velocity_exact[-1]) / velocity_scale
    ) ** 2
    pressure_error_final = (
        (pressure[-1] - pressure_exact[-1]) / pressure_scale
    ) ** 2
    rho_exact_final = (rho_exact[-1] / rho_scale) ** 2
    velocity_exact_final = (
        velocity_exact[-1] / velocity_scale
    ) ** 2
    pressure_exact_final = (
        pressure_exact[-1] / pressure_scale
    ) ** 2
    output["primitive_scaled_final_rel_l2"] = float(
        np.sqrt(
            np.mean(
                rho_error_final
                + velocity_error_final
                + pressure_error_final
            )
        )
        / (
            np.sqrt(
                np.mean(
                    rho_exact_final
                    + velocity_exact_final
                    + pressure_exact_final
                )
            )
            + cfg.relative_error_epsilon
        )
    )

    output["rho_positivity_violation"] = float(
        max(0.0, cfg.rho_floor - rho.min())
    )
    output["p_positivity_violation"] = float(
        max(0.0, cfg.p_floor - pressure.min())
    )

    for name, prediction, exact in (
        ("rho", rho, rho_exact),
        ("u", velocity, velocity_exact),
        ("p", pressure, pressure_exact),
    ):
        total_variation_prediction = np.sum(
            np.abs(np.diff(prediction[-1]))
        )
        total_variation_exact = np.sum(np.abs(np.diff(exact[-1])))
        output[f"{name}_tv_final"] = float(total_variation_prediction)
        output[f"{name}_tv_excess_final"] = float(
            max(
                0.0,
                total_variation_prediction - total_variation_exact,
            )
        )

    return output


def compute_global_conservation_metrics(
    model: nn.Module,
    cfg: Euler1DConfig,
    *,
    grid: dict[str, np.ndarray] | None = None,
) -> dict[str, float]:
    if grid is None:
        grid = predict_grid(model, cfg)

    x = grid["x"]
    t = grid["t"]
    rho = grid["rho"]
    velocity = grid["u"]
    pressure = grid["p"]

    mass, momentum, energy = primitive_to_conserved_np(
        rho, velocity, pressure, cfg
    )
    mass_flux_left, momentum_flux_left, energy_flux_left = flux_np(
        rho[:, 0], velocity[:, 0], pressure[:, 0], cfg
    )
    mass_flux_right, momentum_flux_right, energy_flux_right = flux_np(
        rho[:, -1], velocity[:, -1], pressure[:, -1], cfg
    )

    results: dict[str, float] = {}
    for name, state, flux_left, flux_right in (
        ("mass", mass, mass_flux_left, mass_flux_right),
        ("mom", momentum, momentum_flux_left, momentum_flux_right),
        ("energy", energy, energy_flux_left, energy_flux_right),
    ):
        integral = np.trapz(state, x, axis=1)
        flux_difference = flux_right - flux_left
        conservation_values = []
        relative_values = []
        for index in range(len(t)):
            flux_integral = (
                np.trapz(
                    flux_difference[: index + 1],
                    t[: index + 1],
                )
                if index > 0
                else 0.0
            )
            value = integral[index] - integral[0] + flux_integral
            denominator = (
                abs(integral[index] - integral[0])
                + abs(flux_integral)
                + cfg.relative_error_epsilon
            )
            conservation_values.append(value)
            relative_values.append(abs(value) / denominator)

        conservation_values = np.asarray(conservation_values)
        relative_values = np.asarray(relative_values)
        results[f"global_{name}_cons_mean_abs"] = float(
            np.mean(np.abs(conservation_values))
        )
        results[f"global_{name}_cons_final_abs"] = float(
            abs(conservation_values[-1])
        )
        results[f"global_{name}_cons_mean_rel"] = float(
            np.mean(relative_values[1:])
        )

    return results


def compute_local_conservation_metrics(
    model: nn.Module,
    cfg: Euler1DConfig,
) -> dict[str, float]:
    rng = np.random.default_rng(cfg.seed + 271828)
    values = {"mass": [], "mom": [], "energy": []}
    relative_values = {"mass": [], "mom": [], "energy": []}

    for _ in range(cfg.n_control_volumes):
        for _try in range(100):
            left, right = np.sort(
                rng.uniform(cfg.x_min, cfg.x_max, size=2)
            )
            time_start, time_end = np.sort(
                rng.uniform(cfg.t_min, cfg.t_max, size=2)
            )
            if (
                (right - left) >= cfg.cv_min_width
                and (time_end - time_start) >= cfg.cv_min_duration
            ):
                break

        x_quadrature = np.linspace(left, right, cfg.cv_quad_nx)
        t_quadrature = np.linspace(
            time_start, time_end, cfg.cv_quad_nt
        )

        primitive_start = predict_primitive_points(
            model,
            x_quadrature,
            np.full_like(x_quadrature, time_start),
        )
        primitive_end = predict_primitive_points(
            model,
            x_quadrature,
            np.full_like(x_quadrature, time_end),
        )

        rho_start = primitive_start[:, 0]
        velocity_start = primitive_start[:, 1]
        pressure_start = primitive_start[:, 2]
        rho_end = primitive_end[:, 0]
        velocity_end = primitive_end[:, 1]
        pressure_end = primitive_end[:, 2]
        state_start = primitive_to_conserved_np(
            rho_start, velocity_start, pressure_start, cfg
        )
        state_end = primitive_to_conserved_np(
            rho_end, velocity_end, pressure_end, cfg
        )

        primitive_left = predict_primitive_points(
            model,
            np.full_like(t_quadrature, left),
            t_quadrature,
        )
        primitive_right = predict_primitive_points(
            model,
            np.full_like(t_quadrature, right),
            t_quadrature,
        )
        flux_left = flux_np(
            primitive_left[:, 0],
            primitive_left[:, 1],
            primitive_left[:, 2],
            cfg,
        )
        flux_right = flux_np(
            primitive_right[:, 0],
            primitive_right[:, 1],
            primitive_right[:, 2],
            cfg,
        )

        for component, name in enumerate(("mass", "mom", "energy")):
            integral_end = np.trapz(
                state_end[component], x_quadrature
            )
            integral_start = np.trapz(
                state_start[component], x_quadrature
            )
            flux_integral = np.trapz(
                flux_right[component] - flux_left[component],
                t_quadrature,
            )
            residual = (
                integral_end - integral_start + flux_integral
            )
            denominator = (
                abs(integral_end)
                + abs(integral_start)
                + abs(flux_integral)
                + cfg.relative_error_epsilon
            )
            values[name].append(abs(residual))
            relative_values[name].append(abs(residual) / denominator)

    output: dict[str, float] = {}
    for name in ("mass", "mom", "energy"):
        value_array = np.asarray(values[name])
        relative_array = np.asarray(relative_values[name])
        output[f"local_{name}_cons_cv_mean_abs"] = float(
            value_array.mean()
        )
        output[f"local_{name}_cons_cv_median_abs"] = float(
            np.median(value_array)
        )
        output[f"local_{name}_cons_cv_max_abs"] = float(
            value_array.max()
        )
        output[f"local_{name}_cons_cv_mean_rel"] = float(
            relative_array.mean()
        )

    return output


@torch.no_grad()
def compute_gate_diagnostics(
    model: nn.Module,
    cfg: Euler1DConfig,
    *,
    progress: float = 1.0,
) -> dict[str, float]:
    h_probe, cmin, normalized_progress = schedule_from_progress(
        progress, cfg
    )
    x = np.linspace(cfg.x_min, cfg.x_max, cfg.eval_nx)
    t = np.linspace(cfg.t_min, cfg.t_max, cfg.eval_nt)
    X, T = np.meshgrid(x, t)
    parameter = next(model.parameters())
    coordinates = torch.tensor(
        np.stack([X.reshape(-1), T.reshape(-1)], axis=1),
        device=parameter.device,
        dtype=parameter.dtype,
    )

    rho_scale, velocity_scale, pressure_scale, _, _ = scales(cfg)
    gate_result = centered_euler_trace_ratio_gate_1d(
        model,
        coordinates[:, 0:1],
        coordinates[:, 1:2],
        h_probe=h_probe,
        cmin=cmin,
        x_min=cfg.x_min,
        x_max=cfg.x_max,
        rho_scale=rho_scale,
        velocity_scale=velocity_scale,
        pressure_scale=pressure_scale,
        beta=cfg.beta,
        norm_epsilon=cfg.trace_norm_epsilon,
        ratio_epsilon=cfg.trace_ratio_epsilon,
        mean_epsilon=cfg.batch_mean_epsilon,
    )

    gate = (
        gate_result.gate.detach().cpu().numpy().reshape(
            cfg.eval_nt, cfg.eval_nx
        )
    )
    ratio = (
        gate_result.ratio.detach().cpu().numpy().reshape(
            cfg.eval_nt, cfg.eval_nx
        )
    )
    normalized_jump = (
        gate_result.normalized_jump.detach().cpu().numpy().reshape(
            cfg.eval_nt, cfg.eval_nx
        )
    )
    valid = (
        gate_result.valid.detach().cpu().numpy().reshape(
            cfg.eval_nt, cfg.eval_nx
        ).astype(bool)
    )
    if valid.sum() == 0:
        valid = np.ones_like(valid, dtype=bool)

    return {
        "gate_progress": float(normalized_progress),
        "gate_h": float(h_probe),
        "gate_cmin": float(cmin),
        "gate_mean": float(gate[valid].mean()),
        "gate_max": float(gate[valid].max()),
        "gate_active_frac_gt_0p5": float(
            ((gate > 0.5) & valid).sum() / valid.sum()
        ),
        "gate_active_frac_gt_0p1": float(
            ((gate > 0.1) & valid).sum() / valid.sum()
        ),
        "gate_C_mean": float(ratio[valid].mean()),
        "gate_C_max": float(ratio[valid].max()),
        "gate_Jhat_mean": float(normalized_jump[valid].mean()),
        "gate_Jhat_max": float(normalized_jump[valid].max()),
        "gate_Jbar": float(gate_result.jump_mean.detach().cpu()),
    }


def evaluate_model(
    model: nn.Module,
    method_name: str,
    cfg: Euler1DConfig,
) -> dict[str, Any]:
    model.eval()
    grid = predict_grid(model, cfg)
    row: dict[str, Any] = {
        "model": method_name,
        "seed": cfg.seed,
        **compute_error_metrics(model, cfg, grid=grid),
        **compute_global_conservation_metrics(model, cfg, grid=grid),
        **compute_local_conservation_metrics(model, cfg),
        **compute_gate_diagnostics(model, cfg, progress=1.0),
    }
    return row


def _save_checkpoint(
    model: nn.Module,
    path: Path,
    extra: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "extra": extra,
        },
        path,
    )


def run_paired_experiment(
    cfg: Euler1DConfig,
    *,
    output_root: str | Path,
    protected_reported_root: str | Path,
    overwrite: bool = False,
) -> pd.DataFrame:
    """Run a fresh shared-warmup PINN/TRG-PINN reproduction experiment."""

    device = resolve_device(cfg.device)
    dtype = configure_torch_runtime(cfg.dtype)
    set_seed(cfg.seed)

    root = ensure_unprotected_output(
        output_root,
        protected_reported_root,
    )
    seed_root = root / "1d_euler" / f"seed_{cfg.seed}"
    if seed_root.exists() and any(seed_root.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Reproduction output already exists: {seed_root}. "
            "Use overwrite explicitly."
        )
    seed_root.mkdir(parents=True, exist_ok=True)

    write_json_atomic(
        {
            "equation": "1d_euler",
            "seed": cfg.seed,
            "config": asdict(cfg),
            "protocol": (
                "shared warm-up; paired PINN and TRG-PINN continuation; "
                "same RNG state restored before each continuation; no validation; "
                "no early stopping; final checkpoint evaluation"
            ),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(device),
        },
        seed_root / "config.json",
    )

    warmup_model = build_model(cfg).to(device=device, dtype=dtype)
    warmup_start = time.time()
    warmup_model, warmup_history = train_warmup(warmup_model, cfg)
    warmup_seconds = time.time() - warmup_start
    warmup_state = clone_state_dict_cpu(warmup_model)
    continuation_state = capture_rng_state()

    pinn_model = build_model(cfg).to(device=device, dtype=dtype)
    load_state_dict_to_model(pinn_model, warmup_state)
    restore_rng_state(continuation_state)
    pinn_start = time.time()
    pinn_model, pinn_history = train_pinn_continuation(pinn_model, cfg)
    pinn_seconds = time.time() - pinn_start

    trg_model = build_model(cfg).to(device=device, dtype=dtype)
    load_state_dict_to_model(trg_model, warmup_state)
    restore_rng_state(continuation_state)
    trg_start = time.time()
    trg_model, trg_history = train_trg_continuation(trg_model, cfg)
    trg_seconds = time.time() - trg_start

    pinn_metrics = evaluate_model(pinn_model, "PINN", cfg)
    trg_metrics = evaluate_model(trg_model, "TRG-PINN", cfg)

    for metrics, continuation_seconds in (
        (pinn_metrics, pinn_seconds),
        (trg_metrics, trg_seconds),
    ):
        metrics.update(
            {
                "equation": "1d_euler",
                "method": metrics["model"],
                "warmup_iters": cfg.warmup_iters,
                "continuation_iters": cfg.gated_iters,
                "adam_total_iters": (
                    cfg.warmup_iters + cfg.gated_iters
                ),
                "optimizer_main": "AdamW",
                "lbfgs_used": False,
                "wall_clock_sec_warmup": warmup_seconds,
                "wall_clock_sec_continuation": continuation_seconds,
                "wall_clock_sec_total": (
                    warmup_seconds + continuation_seconds
                ),
                "num_parameters": trainable_parameter_count(pinn_model),
            }
        )

    warmup_history.to_csv(
        seed_root / "history_warmup.csv",
        index=False,
    )
    pinn_history.to_csv(
        seed_root / "history_pinn.csv",
        index=False,
    )
    trg_history.to_csv(
        seed_root / "history_trg_pinn.csv",
        index=False,
    )
    pd.DataFrame([pinn_metrics, trg_metrics]).to_csv(
        seed_root / "metrics_final.csv",
        index=False,
    )
    write_json_atomic(
        pinn_metrics,
        seed_root / "metrics_pinn.json",
    )
    write_json_atomic(
        trg_metrics,
        seed_root / "metrics_trg_pinn.json",
    )
    torch.save(
        {
            "model_state_dict": warmup_state,
            "equation": "1d_euler",
            "method": "shared_warmup",
            "seed": cfg.seed,
        },
        seed_root / "model_warmup.pt",
    )
    _save_checkpoint(
        pinn_model,
        seed_root / "model_pinn_final.pt",
        pinn_metrics,
    )
    _save_checkpoint(
        trg_model,
        seed_root / "model_trg_pinn_final.pt",
        trg_metrics,
    )
    (seed_root / "_SUCCESS").write_text(
        "success\n",
        encoding="utf-8",
    )
    return pd.DataFrame([pinn_metrics, trg_metrics])
