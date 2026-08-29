"""2D scalar inviscid Burgers benchmark used in the TRG-PINN manuscript.

The public implementation preserves the reported paired protocol, metric
definitions, multidirectional trace-ratio gate, and the corrected all-probe
valid-mask definition.  Fresh runs are separate from immutable reported
artifacts.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
import copy
import math
import platform
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from trgpinn.models import CoordinateMLP
from trgpinn.utils import (
    capture_rng_state,
    clone_state_dict_cpu,
    configure_torch_runtime,
    ensure_unprotected_output,
    load_state_dict_to_model,
    restore_rng_state,
    set_seed,
    write_json_atomic,
)


@dataclass
class Burgers2DConfig:
    seed: int = 2026
    device: str = "auto"
    dtype: str = "float32"

    x_min: float = -1.0
    x_max: float = 1.0
    y_min: float = -1.0
    y_max: float = 1.0
    t_min: float = 0.0
    t_max: float = 0.5

    uL: float = 1.0
    uR: float = 0.0
    theta_deg: float = 30.0
    eta0: float = -0.25

    width: int = 128
    depth: int = 6
    activation: str = "tanh"

    warmup_iters: int = 3000
    gated_iters: int = 7000
    lr_warmup: float = 8.0e-4
    lr_gated: float = 3.0e-4
    weight_decay: float = 1.0e-2
    grad_clip: float = 1.0

    n_f: int = 16000
    n_ic: int = 2500
    n_bc: int = 2500

    w_ic: float = 100.0
    w_bc: float = 30.0
    w_pde: float = 1.0

    h_max_factor: float = 5.0
    h_min_factor: float = 2.0
    cmin_start: float = 0.50
    cmin_end: float = 0.70
    beta: float = 0.05
    residual_floor: float = 0.02
    trace_ratio_epsilon: float = 1.0e-6
    batch_mean_epsilon: float = 1.0e-8
    weighted_loss_epsilon: float = 1.0e-8
    relative_error_epsilon: float = 1.0e-12

    use_four_directions: bool = True

    print_every: int = 500
    history_every: int = 50

    eval_nxy: int = 180
    eval_nt: int = 45
    eval_space_nxy: int = 90
    line_n: int = 900

    cons_nxy: int = 120
    cons_nt: int = 45
    n_control_volumes: int = 48
    cv_quad_nx: int = 48
    cv_quad_ny: int = 48
    cv_quad_nt: int = 48
    cv_min_width: float = 0.25
    cv_min_duration: float = 0.08

    @classmethod
    def from_legacy_mapping(
        cls,
        mapping: dict[str, Any],
        *,
        seed: int | None = None,
        device: str | None = None,
    ) -> "Burgers2DConfig":
        field_names = {item.name for item in fields(cls)}
        values = {
            key: value
            for key, value in dict(mapping).items()
            if key in field_names
        }
        if seed is not None:
            values["seed"] = int(seed)
        if device is not None:
            values["device"] = str(device)
        values.setdefault("weight_decay", 1.0e-2)
        values.setdefault("trace_ratio_epsilon", 1.0e-6)
        values.setdefault("batch_mean_epsilon", 1.0e-8)
        values.setdefault("weighted_loss_epsilon", 1.0e-8)
        values.setdefault("relative_error_epsilon", 1.0e-12)
        return cls(**values)

    def smoke_copy(
        self,
        *,
        seed: int = 2026,
        device: str = "auto",
    ) -> "Burgers2DConfig":
        return replace(
            self,
            seed=int(seed),
            device=str(device),
            warmup_iters=2,
            gated_iters=2,
            n_f=64,
            n_ic=32,
            n_bc=32,
            print_every=1,
            history_every=1,
            eval_nxy=40,
            eval_nt=8,
            eval_space_nxy=30,
            line_n=240,
            cons_nxy=30,
            cons_nt=8,
            n_control_volumes=2,
            cv_quad_nx=12,
            cv_quad_ny=12,
            cv_quad_nt=12,
            cv_min_width=0.15,
            cv_min_duration=0.05,
        )


def resolve_device(value: str | torch.device) -> torch.device:
    if isinstance(value, torch.device):
        requested = value
    else:
        text = str(value).strip().lower()
        if text == "auto":
            requested = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            requested = torch.device(text)
    if requested.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA device {requested} was requested but CUDA is unavailable."
        )
    return requested


def dtype_from_config(cfg: Burgers2DConfig) -> torch.dtype:
    return configure_torch_runtime(cfg.dtype)


def build_model(cfg: Burgers2DConfig) -> CoordinateMLP:
    return CoordinateMLP(
        input_dimension=3,
        output_dimension=1,
        hidden_width=cfg.width,
        hidden_layers=cfg.depth,
        activation=cfg.activation,
        lower_bounds=(cfg.x_min, cfg.y_min, cfg.t_min),
        upper_bounds=(cfg.x_max, cfg.y_max, cfg.t_max),
        output_bias=0.5 * (cfg.uL + cfg.uR),
    )


def count_parameters(model: nn.Module) -> int:
    return int(
        sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )
    )


def normal_vector(cfg: Burgers2DConfig) -> np.ndarray:
    theta = math.radians(float(cfg.theta_deg))
    return np.asarray(
        [math.cos(theta), math.sin(theta)],
        dtype=np.float64,
    )


def tangent_vector(cfg: Burgers2DConfig) -> np.ndarray:
    normal = normal_vector(cfg)
    return np.asarray([-normal[1], normal[0]], dtype=np.float64)


def flux_direction_factor(cfg: Burgers2DConfig) -> float:
    return float(normal_vector(cfg).sum())


def state_scale(cfg: Burgers2DConfig) -> float:
    return max(abs(float(cfg.uL) - float(cfg.uR)), 1.0e-8)


def shock_speed_eta(cfg: Burgers2DConfig) -> float:
    return (
        0.5
        * (float(cfg.uL) + float(cfg.uR))
        * flux_direction_factor(cfg)
    )


def shock_eta(t, cfg: Burgers2DConfig):
    return float(cfg.eta0) + shock_speed_eta(cfg) * np.asarray(t)


def eta_numpy(X, Y, cfg: Burgers2DConfig):
    normal = normal_vector(cfg)
    return normal[0] * np.asarray(X) + normal[1] * np.asarray(Y)


def exact_solution(X, Y, T, cfg: Burgers2DConfig):
    eta = eta_numpy(X, Y, cfg)
    eta_shock = float(cfg.eta0) + shock_speed_eta(cfg) * np.asarray(T)
    return np.where(
        eta < eta_shock,
        float(cfg.uL),
        float(cfg.uR),
    ).astype(np.float64)


def flux_numpy(u):
    return 0.5 * np.asarray(u) ** 2


def flux_torch(u: torch.Tensor) -> torch.Tensor:
    return 0.5 * u.pow(2)


def characteristic_spacing(cfg: Burgers2DConfig) -> float:
    area = (
        (float(cfg.x_max) - float(cfg.x_min))
        * (float(cfg.y_max) - float(cfg.y_min))
    )
    return math.sqrt(area) / math.sqrt(float(cfg.n_f))


def trace_schedule(
    progress: float,
    cfg: Burgers2DConfig,
) -> tuple[float, float, float]:
    normalized = float(np.clip(progress, 0.0, 1.0))
    base = characteristic_spacing(cfg)
    h_probe = base * (
        float(cfg.h_min_factor)
        + (
            float(cfg.h_max_factor)
            - float(cfg.h_min_factor)
        )
        * (1.0 - normalized) ** 2
    )
    cmin = float(cfg.cmin_start) + (
        float(cfg.cmin_end) - float(cfg.cmin_start)
    ) * normalized
    return h_probe, cmin, normalized


def trace_schedule_from_iteration(
    iteration: int,
    total_iterations: int,
    cfg: Burgers2DConfig,
) -> tuple[float, float, float]:
    return trace_schedule(
        float(iteration) / max(1, int(total_iterations)),
        cfg,
    )


def line_eta_limits(
    cfg: Burgers2DConfig,
    safety: float = 0.98,
) -> tuple[float, float]:
    normal = normal_vector(cfg)
    intervals = []

    for component, lower, upper in (
        (normal[0], cfg.x_min, cfg.x_max),
        (normal[1], cfg.y_min, cfg.y_max),
    ):
        if abs(component) < 1.0e-14:
            intervals.append((-np.inf, np.inf))
        else:
            values = [lower / component, upper / component]
            intervals.append((min(values), max(values)))

    lower = max(interval[0] for interval in intervals)
    upper = min(interval[1] for interval in intervals)
    midpoint = 0.5 * (lower + upper)
    half_width = 0.5 * (upper - lower) * float(safety)
    return midpoint - half_width, midpoint + half_width


def _model_device_dtype(
    model: nn.Module,
) -> tuple[torch.device, torch.dtype]:
    parameter = next(model.parameters())
    return parameter.device, parameter.dtype


def sample_initial(
    n: int,
    cfg: Burgers2DConfig,
    *,
    device: torch.device,
    dtype: torch.dtype,
):
    x = cfg.x_min + (cfg.x_max - cfg.x_min) * torch.rand(
        int(n), 1, device=device, dtype=dtype
    )
    y = cfg.y_min + (cfg.y_max - cfg.y_min) * torch.rand(
        int(n), 1, device=device, dtype=dtype
    )
    t = torch.zeros_like(x)
    target_numpy = exact_solution(
        x.detach().cpu().numpy(),
        y.detach().cpu().numpy(),
        np.zeros((int(n), 1)),
        cfg,
    )
    target = torch.tensor(
        target_numpy,
        device=device,
        dtype=dtype,
    )
    return x, y, t, target


def sample_boundary(
    n: int,
    cfg: Burgers2DConfig,
    *,
    device: torch.device,
    dtype: torch.dtype,
):
    n_each = int(n) // 4
    remainder = int(n) - 4 * n_each
    counts = [n_each, n_each, n_each, n_each + remainder]

    xs, ys, ts = [], [], []

    for side, count in enumerate(counts):
        time_values = cfg.t_min + (
            cfg.t_max - cfg.t_min
        ) * torch.rand(
            count,
            1,
            device=device,
            dtype=dtype,
        )

        if side == 0:
            x = torch.full(
                (count, 1),
                cfg.x_min,
                device=device,
                dtype=dtype,
            )
            y = cfg.y_min + (
                cfg.y_max - cfg.y_min
            ) * torch.rand(
                count,
                1,
                device=device,
                dtype=dtype,
            )
        elif side == 1:
            x = torch.full(
                (count, 1),
                cfg.x_max,
                device=device,
                dtype=dtype,
            )
            y = cfg.y_min + (
                cfg.y_max - cfg.y_min
            ) * torch.rand(
                count,
                1,
                device=device,
                dtype=dtype,
            )
        elif side == 2:
            x = cfg.x_min + (
                cfg.x_max - cfg.x_min
            ) * torch.rand(
                count,
                1,
                device=device,
                dtype=dtype,
            )
            y = torch.full(
                (count, 1),
                cfg.y_min,
                device=device,
                dtype=dtype,
            )
        else:
            x = cfg.x_min + (
                cfg.x_max - cfg.x_min
            ) * torch.rand(
                count,
                1,
                device=device,
                dtype=dtype,
            )
            y = torch.full(
                (count, 1),
                cfg.y_max,
                device=device,
                dtype=dtype,
            )

        xs.append(x)
        ys.append(y)
        ts.append(time_values)

    x = torch.cat(xs, dim=0)
    y = torch.cat(ys, dim=0)
    t = torch.cat(ts, dim=0)

    target_numpy = exact_solution(
        x.detach().cpu().numpy(),
        y.detach().cpu().numpy(),
        t.detach().cpu().numpy(),
        cfg,
    )
    target = torch.tensor(
        target_numpy,
        device=device,
        dtype=dtype,
    )

    permutation = torch.randperm(x.shape[0], device=device)
    return (
        x[permutation],
        y[permutation],
        t[permutation],
        target[permutation],
    )


def sample_interior(
    n: int,
    cfg: Burgers2DConfig,
    *,
    device: torch.device,
    dtype: torch.dtype,
):
    x = cfg.x_min + (cfg.x_max - cfg.x_min) * torch.rand(
        int(n), 1, device=device, dtype=dtype
    )
    y = cfg.y_min + (cfg.y_max - cfg.y_min) * torch.rand(
        int(n), 1, device=device, dtype=dtype
    )
    t = cfg.t_min + (cfg.t_max - cfg.t_min) * torch.rand(
        int(n), 1, device=device, dtype=dtype
    )
    return x, y, t


def burgers_residual(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    t: torch.Tensor,
):
    x = x.detach().clone().requires_grad_(True)
    y = y.detach().clone().requires_grad_(True)
    t = t.detach().clone().requires_grad_(True)

    state = model(torch.cat([x, y, t], dim=1))

    state_t = torch.autograd.grad(
        state,
        t,
        grad_outputs=torch.ones_like(state),
        create_graph=True,
        retain_graph=True,
    )[0]

    flux_x = flux_torch(state)
    flux_y = flux_torch(state)

    flux_x_x = torch.autograd.grad(
        flux_x,
        x,
        grad_outputs=torch.ones_like(flux_x),
        create_graph=True,
        retain_graph=True,
    )[0]
    flux_y_y = torch.autograd.grad(
        flux_y,
        y,
        grad_outputs=torch.ones_like(flux_y),
        create_graph=True,
        retain_graph=True,
    )[0]

    residual = state_t + flux_x_x + flux_y_y
    return state, residual


def direction_tensor(
    cfg: Burgers2DConfig,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    directions = [
        [1.0, 0.0],
        [0.0, 1.0],
    ]
    if cfg.use_four_directions:
        inverse_root_two = 1.0 / math.sqrt(2.0)
        directions.extend(
            [
                [inverse_root_two, inverse_root_two],
                [inverse_root_two, -inverse_root_two],
            ]
        )
    return torch.tensor(
        directions,
        device=device,
        dtype=dtype,
    )


@torch.no_grad()
def trace_ratio_gate_2d(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    t: torch.Tensor,
    h_probe: float,
    cmin: float,
    cfg: Burgers2DConfig,
):
    device = x.device
    dtype = x.dtype

    directions = direction_tensor(
        cfg,
        device=device,
        dtype=dtype,
    )
    n_points = x.shape[0]
    n_directions = directions.shape[0]

    dx = directions[:, 0].view(1, n_directions)
    dy = directions[:, 1].view(1, n_directions)

    x0 = x.view(n_points, 1)
    y0 = y.view(n_points, 1)
    t0 = t.view(n_points, 1)

    xm1 = x0 - h_probe * dx
    xp1 = x0 + h_probe * dx
    ym1 = y0 - h_probe * dy
    yp1 = y0 + h_probe * dy

    xm2 = x0 - 2.0 * h_probe * dx
    xp2 = x0 + 2.0 * h_probe * dx
    ym2 = y0 - 2.0 * h_probe * dy
    yp2 = y0 + 2.0 * h_probe * dy

    # Canonical post-fix validity definition:
    # every +/-2h x/y coordinate must satisfy both lower and upper bounds.
    valid = (
        (xm2 >= cfg.x_min)
        & (xm2 <= cfg.x_max)
        & (xp2 >= cfg.x_min)
        & (xp2 <= cfg.x_max)
        & (ym2 >= cfg.y_min)
        & (ym2 <= cfg.y_max)
        & (yp2 >= cfg.y_min)
        & (yp2 <= cfg.y_max)
    ).to(dtype)

    xm1c = xm1.clamp(cfg.x_min, cfg.x_max)
    xp1c = xp1.clamp(cfg.x_min, cfg.x_max)
    ym1c = ym1.clamp(cfg.y_min, cfg.y_max)
    yp1c = yp1.clamp(cfg.y_min, cfg.y_max)

    xm2c = xm2.clamp(cfg.x_min, cfg.x_max)
    xp2c = xp2.clamp(cfg.x_min, cfg.x_max)
    ym2c = ym2.clamp(cfg.y_min, cfg.y_max)
    yp2c = yp2.clamp(cfg.y_min, cfg.y_max)

    repeated_t = t0.repeat(1, n_directions)

    def evaluate(xx, yy):
        coordinates = torch.cat(
            [
                xx.reshape(-1, 1),
                yy.reshape(-1, 1),
                repeated_t.reshape(-1, 1),
            ],
            dim=1,
        )
        return model(coordinates).reshape(
            n_points,
            n_directions,
        )

    u_m1 = evaluate(xm1c, ym1c)
    u_p1 = evaluate(xp1c, yp1c)
    u_m2 = evaluate(xm2c, ym2c)
    u_p2 = evaluate(xp2c, yp2c)

    jump_h = torch.abs(u_m1 - u_p1) / state_scale(cfg)
    jump_2h = torch.abs(u_m2 - u_p2) / state_scale(cfg)

    ratio = torch.clamp(
        jump_h / (jump_2h + cfg.trace_ratio_epsilon),
        0.0,
        2.0,
    )

    valid_sum = valid.sum()
    jump_mean_valid = (
        (jump_h * valid).sum()
        / (valid_sum + cfg.batch_mean_epsilon)
    )
    jump_mean_all = jump_h.mean()

    jump_mean = torch.where(
        valid_sum > 0,
        jump_mean_valid,
        jump_mean_all,
    ).clamp_min(cfg.batch_mean_epsilon)

    normalized_jump = jump_h / jump_mean

    gate_jump = torch.sigmoid(
        (normalized_jump - 1.0) / cfg.beta
    )
    gate_ratio = torch.sigmoid(
        (ratio - cmin) / cfg.beta
    )
    gate_direction = gate_jump * gate_ratio * valid

    gate = gate_direction.max(
        dim=1,
        keepdim=True,
    ).values
    ratio_max = (ratio * valid).max(
        dim=1,
        keepdim=True,
    ).values
    normalized_jump_max = (
        normalized_jump * valid
    ).max(
        dim=1,
        keepdim=True,
    ).values

    return (
        gate,
        ratio_max,
        normalized_jump_max,
        jump_mean,
        gate_direction,
        valid,
    )


def residual_weight(
    gate: torch.Tensor,
    cfg: Burgers2DConfig,
) -> torch.Tensor:
    return (
        cfg.residual_floor
        + (1.0 - cfg.residual_floor)
        * (1.0 - gate)
    )


def vanilla_loss(
    model: nn.Module,
    cfg: Burgers2DConfig,
):
    device, dtype = _model_device_dtype(model)

    x_ic, y_ic, t_ic, u_ic = sample_initial(
        cfg.n_ic,
        cfg,
        device=device,
        dtype=dtype,
    )
    x_bc, y_bc, t_bc, u_bc = sample_boundary(
        cfg.n_bc,
        cfg,
        device=device,
        dtype=dtype,
    )
    x_f, y_f, t_f = sample_interior(
        cfg.n_f,
        cfg,
        device=device,
        dtype=dtype,
    )

    u_ic_prediction = model(
        torch.cat([x_ic, y_ic, t_ic], dim=1)
    )
    u_bc_prediction = model(
        torch.cat([x_bc, y_bc, t_bc], dim=1)
    )
    _, residual = burgers_residual(
        model,
        x_f,
        y_f,
        t_f,
    )

    loss_ic = F.mse_loss(
        u_ic_prediction,
        u_ic,
    )
    loss_bc = F.mse_loss(
        u_bc_prediction,
        u_bc,
    )
    loss_pde = residual.pow(2).mean()

    loss = (
        cfg.w_ic * loss_ic
        + cfg.w_bc * loss_bc
        + cfg.w_pde * loss_pde
    )

    return loss, {
        "ic": loss_ic.detach(),
        "bc": loss_bc.detach(),
        "pde": loss_pde.detach(),
        "weighted_pde": loss_pde.detach(),
    }


def trg_loss(
    model: nn.Module,
    iteration: int,
    total_iterations: int,
    cfg: Burgers2DConfig,
):
    h_probe, cmin, progress = (
        trace_schedule_from_iteration(
            iteration,
            total_iterations,
            cfg,
        )
    )

    device, dtype = _model_device_dtype(model)

    x_ic, y_ic, t_ic, u_ic = sample_initial(
        cfg.n_ic,
        cfg,
        device=device,
        dtype=dtype,
    )
    x_bc, y_bc, t_bc, u_bc = sample_boundary(
        cfg.n_bc,
        cfg,
        device=device,
        dtype=dtype,
    )
    x_f, y_f, t_f = sample_interior(
        cfg.n_f,
        cfg,
        device=device,
        dtype=dtype,
    )

    u_ic_prediction = model(
        torch.cat([x_ic, y_ic, t_ic], dim=1)
    )
    u_bc_prediction = model(
        torch.cat([x_bc, y_bc, t_bc], dim=1)
    )
    _, residual = burgers_residual(
        model,
        x_f,
        y_f,
        t_f,
    )

    (
        gate,
        ratio_max,
        normalized_jump_max,
        jump_mean,
        _,
        valid,
    ) = trace_ratio_gate_2d(
        model,
        x_f,
        y_f,
        t_f,
        h_probe,
        cmin,
        cfg,
    )

    weight = residual_weight(gate, cfg)

    loss_ic = F.mse_loss(
        u_ic_prediction,
        u_ic,
    )
    loss_bc = F.mse_loss(
        u_bc_prediction,
        u_bc,
    )
    loss_pde_raw = residual.pow(2).mean()
    loss_pde_weighted = (
        (weight * residual.pow(2)).sum()
        / (
            weight.sum()
            + cfg.weighted_loss_epsilon
        )
    )

    loss = (
        cfg.w_ic * loss_ic
        + cfg.w_bc * loss_bc
        + cfg.w_pde * loss_pde_weighted
    )

    return loss, {
        "ic": loss_ic.detach(),
        "bc": loss_bc.detach(),
        "pde": loss_pde_raw.detach(),
        "weighted_pde": loss_pde_weighted.detach(),
        "gate_mean": gate.mean().detach(),
        "gate_max": gate.max().detach(),
        "gate_active_gt_0p5": (
            gate > 0.5
        ).to(dtype).mean().detach(),
        "gate_active_gt_0p1": (
            gate > 0.1
        ).to(dtype).mean().detach(),
        "C_mean": ratio_max.mean().detach(),
        "C_max": ratio_max.max().detach(),
        "Jhat_max": normalized_jump_max.max().detach(),
        "Jbar": jump_mean.detach(),
        "W_mean": weight.mean().detach(),
        "W_min": weight.min().detach(),
        "h": torch.tensor(
            h_probe,
            device=device,
            dtype=dtype,
        ),
        "cmin": torch.tensor(
            cmin,
            device=device,
            dtype=dtype,
        ),
        "progress": torch.tensor(
            progress,
            device=device,
            dtype=dtype,
        ),
        "valid_frac": valid.mean().detach(),
    }


def _adamw(
    model: nn.Module,
    learning_rate: float,
    cfg: Burgers2DConfig,
):
    return torch.optim.AdamW(
        model.parameters(),
        lr=float(learning_rate),
        betas=(0.9, 0.999),
        eps=1.0e-8,
        weight_decay=float(cfg.weight_decay),
        amsgrad=False,
    )


def train_warmup(
    model: nn.Module,
    cfg: Burgers2DConfig,
):
    model.train()
    optimizer = _adamw(
        model,
        cfg.lr_warmup,
        cfg,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.warmup_iters,
        eta_min=cfg.lr_warmup * 0.05,
    )

    history = []

    for iteration in range(
        1,
        cfg.warmup_iters + 1,
    ):
        optimizer.zero_grad(set_to_none=True)
        loss, parts = vanilla_loss(model, cfg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            cfg.grad_clip,
        )
        optimizer.step()
        scheduler.step()

        if (
            iteration == 1
            or iteration % cfg.history_every == 0
            or iteration == cfg.warmup_iters
        ):
            history.append(
                {
                    "phase": "warmup",
                    "iter": iteration,
                    "total_iter": iteration,
                    "loss": float(loss.detach().cpu()),
                    "ic": float(parts["ic"].cpu()),
                    "bc": float(parts["bc"].cpu()),
                    "pde": float(parts["pde"].cpu()),
                    "weighted_pde": float(
                        parts["weighted_pde"].cpu()
                    ),
                    "lr": float(
                        optimizer.param_groups[0]["lr"]
                    ),
                }
            )

    return model, pd.DataFrame(history)


def train_pinn_continuation(
    model: nn.Module,
    cfg: Burgers2DConfig,
):
    model.train()
    optimizer = _adamw(
        model,
        cfg.lr_gated,
        cfg,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.gated_iters,
        eta_min=cfg.lr_gated * 0.03,
    )
    history = []

    for iteration in range(
        1,
        cfg.gated_iters + 1,
    ):
        optimizer.zero_grad(set_to_none=True)
        loss, parts = vanilla_loss(model, cfg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            cfg.grad_clip,
        )
        optimizer.step()
        scheduler.step()

        if (
            iteration == 1
            or iteration % cfg.history_every == 0
            or iteration == cfg.gated_iters
        ):
            history.append(
                {
                    "phase": "PINN_continuation",
                    "iter": iteration,
                    "total_iter": cfg.warmup_iters + iteration,
                    "loss": float(loss.detach().cpu()),
                    "ic": float(parts["ic"].cpu()),
                    "bc": float(parts["bc"].cpu()),
                    "pde": float(parts["pde"].cpu()),
                    "weighted_pde": float(
                        parts["weighted_pde"].cpu()
                    ),
                    "lr": float(
                        optimizer.param_groups[0]["lr"]
                    ),
                }
            )

    return model, pd.DataFrame(history)


def train_trg_continuation(
    model: nn.Module,
    cfg: Burgers2DConfig,
):
    model.train()
    optimizer = _adamw(
        model,
        cfg.lr_gated,
        cfg,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.gated_iters,
        eta_min=cfg.lr_gated * 0.03,
    )
    history = []

    for iteration in range(
        1,
        cfg.gated_iters + 1,
    ):
        optimizer.zero_grad(set_to_none=True)
        loss, parts = trg_loss(
            model,
            iteration,
            cfg.gated_iters,
            cfg,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            cfg.grad_clip,
        )
        optimizer.step()
        scheduler.step()

        if (
            iteration == 1
            or iteration % cfg.history_every == 0
            or iteration == cfg.gated_iters
        ):
            history.append(
                {
                    "phase": "TRG_continuation",
                    "iter": iteration,
                    "total_iter": cfg.warmup_iters + iteration,
                    "loss": float(loss.detach().cpu()),
                    "ic": float(parts["ic"].cpu()),
                    "bc": float(parts["bc"].cpu()),
                    "pde": float(parts["pde"].cpu()),
                    "weighted_pde": float(
                        parts["weighted_pde"].cpu()
                    ),
                    "gate_mean": float(
                        parts["gate_mean"].cpu()
                    ),
                    "gate_max": float(
                        parts["gate_max"].cpu()
                    ),
                    "gate_active_gt_0p5": float(
                        parts["gate_active_gt_0p5"].cpu()
                    ),
                    "gate_active_gt_0p1": float(
                        parts["gate_active_gt_0p1"].cpu()
                    ),
                    "C_mean": float(parts["C_mean"].cpu()),
                    "C_max": float(parts["C_max"].cpu()),
                    "Jhat_max": float(
                        parts["Jhat_max"].cpu()
                    ),
                    "Jbar": float(parts["Jbar"].cpu()),
                    "W_mean": float(parts["W_mean"].cpu()),
                    "W_min": float(parts["W_min"].cpu()),
                    "h": float(parts["h"].cpu()),
                    "cmin": float(parts["cmin"].cpu()),
                    "progress": float(
                        parts["progress"].cpu()
                    ),
                    "valid_frac": float(
                        parts["valid_frac"].cpu()
                    ),
                    "lr": float(
                        optimizer.param_groups[0]["lr"]
                    ),
                }
            )

    return model, pd.DataFrame(history)


@torch.no_grad()
def predict_points(
    model: nn.Module,
    x_values,
    y_values,
    t_values,
    batch_size: int = 65536,
):
    x_flat = np.asarray(
        x_values,
        dtype=np.float64,
    ).reshape(-1)
    y_flat = np.asarray(
        y_values,
        dtype=np.float64,
    ).reshape(-1)
    t_flat = np.asarray(
        t_values,
        dtype=np.float64,
    ).reshape(-1)

    if not (
        x_flat.shape
        == y_flat.shape
        == t_flat.shape
    ):
        raise ValueError(
            "x, y, and t arrays must share one shape."
        )

    device, dtype = _model_device_dtype(model)
    output = []

    for start in range(
        0,
        x_flat.size,
        int(batch_size),
    ):
        stop = min(
            start + int(batch_size),
            x_flat.size,
        )
        coordinates = torch.tensor(
            np.stack(
                [
                    x_flat[start:stop],
                    y_flat[start:stop],
                    t_flat[start:stop],
                ],
                axis=1,
            ),
            device=device,
            dtype=dtype,
        )
        output.append(
            model(coordinates)
            .detach()
            .cpu()
            .numpy()
            .reshape(-1)
        )

    return np.concatenate(output).reshape(
        np.asarray(x_values).shape
    )


@torch.no_grad()
def predict_xy_grid(
    model: nn.Module,
    cfg: Burgers2DConfig,
    n: int | None = None,
    t_value: float | None = None,
):
    if n is None:
        n = cfg.eval_nxy
    if t_value is None:
        t_value = cfg.t_max

    x = np.linspace(
        cfg.x_min,
        cfg.x_max,
        int(n),
    )
    y = np.linspace(
        cfg.y_min,
        cfg.y_max,
        int(n),
    )
    X, Y = np.meshgrid(x, y)
    T = np.full_like(X, float(t_value))

    prediction = predict_points(
        model,
        X,
        Y,
        T,
    )
    exact = exact_solution(
        X,
        Y,
        T,
        cfg,
    )
    return x, y, X, Y, prediction, exact


@torch.no_grad()
def predict_space_time_cube(
    model: nn.Module,
    cfg: Burgers2DConfig,
    nxy: int | None = None,
    nt: int | None = None,
):
    if nxy is None:
        nxy = cfg.eval_space_nxy
    if nt is None:
        nt = cfg.eval_nt

    x = np.linspace(
        cfg.x_min,
        cfg.x_max,
        int(nxy),
    )
    y = np.linspace(
        cfg.y_min,
        cfg.y_max,
        int(nxy),
    )
    t = np.linspace(
        cfg.t_min,
        cfg.t_max,
        int(nt),
    )

    predictions = []
    exact_values = []

    for time_value in t:
        X, Y = np.meshgrid(x, y)
        T = np.full_like(X, time_value)
        predictions.append(
            predict_points(model, X, Y, T)
        )
        exact_values.append(
            exact_solution(X, Y, T, cfg)
        )

    return (
        x,
        y,
        t,
        np.stack(predictions, axis=0),
        np.stack(exact_values, axis=0),
    )


@torch.no_grad()
def predict_line_cut(
    model: nn.Module,
    cfg: Burgers2DConfig,
    t_value: float | None = None,
    n: int | None = None,
):
    if t_value is None:
        t_value = cfg.t_max
    if n is None:
        n = cfg.line_n

    eta_min, eta_max = line_eta_limits(cfg)
    eta = np.linspace(
        eta_min,
        eta_max,
        int(n),
    )
    normal = normal_vector(cfg)

    X = normal[0] * eta
    Y = normal[1] * eta
    T = np.full_like(X, float(t_value))

    prediction = predict_points(
        model,
        X,
        Y,
        T,
    )
    exact = exact_solution(
        X,
        Y,
        T,
        cfg,
    )
    return eta, prediction, exact


def crossing_location(
    x,
    u,
    level,
    expected=None,
    window=None,
):
    x = np.asarray(x)
    u = np.asarray(u)
    mask = np.ones_like(x, dtype=bool)

    if expected is not None and window is not None:
        mask = np.abs(x - expected) <= window
        if mask.sum() < 2:
            mask = np.ones_like(x, dtype=bool)

    x_masked = x[mask]
    u_masked = u[mask]
    shifted = u_masked - level

    candidates = np.where(
        shifted[:-1] * shifted[1:] <= 0.0
    )[0]

    if len(candidates) == 0:
        return float(
            x_masked[
                int(np.argmin(np.abs(shifted)))
            ]
        )

    if expected is not None:
        midpoints = 0.5 * (
            x_masked[candidates]
            + x_masked[candidates + 1]
        )
        index = candidates[
            int(
                np.argmin(
                    np.abs(midpoints - expected)
                )
            )
        ]
    else:
        index = candidates[0]

    x0 = x_masked[index]
    x1 = x_masked[index + 1]
    y0 = shifted[index]
    y1 = shifted[index + 1]

    if abs(y1 - y0) < 1.0e-14:
        return float(0.5 * (x0 + x1))

    return float(
        x0 - y0 * (x1 - x0) / (y1 - y0)
    )


def transition_width_from_line(
    eta,
    prediction,
    cfg: Burgers2DConfig,
    time_value: float,
    hi_frac: float = 0.95,
    lo_frac: float = 0.05,
):
    high = (
        cfg.uR
        + hi_frac * (cfg.uL - cfg.uR)
    )
    low = (
        cfg.uR
        + lo_frac * (cfg.uL - cfg.uR)
    )
    expected = float(
        shock_eta(time_value, cfg)
    )

    eta_high = crossing_location(
        eta,
        prediction,
        high,
        expected=expected,
        window=0.35,
    )
    eta_low = crossing_location(
        eta,
        prediction,
        low,
        expected=expected,
        window=0.45,
    )

    return abs(eta_low - eta_high)


def compute_error_shock_metrics(
    model: nn.Module,
    cfg: Burgers2DConfig,
):
    (
        _,
        _,
        t,
        prediction,
        exact,
    ) = predict_space_time_cube(
        model,
        cfg,
        nxy=cfg.eval_space_nxy,
        nt=cfg.eval_nt,
    )

    error = prediction - exact
    space_time_l1 = float(
        np.mean(np.abs(error))
    )
    space_time_l2 = float(
        np.sqrt(np.mean(error ** 2))
    )
    space_time_rel_l2 = float(
        space_time_l2
        / (
            np.sqrt(np.mean(exact ** 2))
            + cfg.relative_error_epsilon
        )
    )

    _, _, _, _, final_prediction, final_exact = (
        predict_xy_grid(
            model,
            cfg,
            n=cfg.eval_nxy,
            t_value=cfg.t_max,
        )
    )
    final_error = (
        final_prediction - final_exact
    )
    final_l1 = float(
        np.mean(np.abs(final_error))
    )
    final_l2 = float(
        np.sqrt(np.mean(final_error ** 2))
    )
    final_rel_l2 = float(
        final_l2
        / (
            np.sqrt(np.mean(final_exact ** 2))
            + cfg.relative_error_epsilon
        )
    )

    midpoint = 0.5 * (cfg.uL + cfg.uR)

    position_errors = []
    predicted_positions = []
    time_used = []
    widths_95 = []
    widths_90 = []
    total_variations = []
    exact_total_variation = state_scale(cfg)

    for time_value in t:
        if time_value < max(
            0.02,
            2.0 * (t[1] - t[0]),
        ):
            continue

        eta, line, _ = predict_line_cut(
            model,
            cfg,
            t_value=float(time_value),
            n=cfg.line_n,
        )
        expected = float(
            shock_eta(time_value, cfg)
        )
        position = crossing_location(
            eta,
            line,
            midpoint,
            expected=expected,
            window=0.35,
        )

        predicted_positions.append(position)
        time_used.append(time_value)
        position_errors.append(
            abs(position - expected)
        )
        widths_95.append(
            transition_width_from_line(
                eta,
                line,
                cfg,
                float(time_value),
                0.95,
                0.05,
            )
        )
        widths_90.append(
            transition_width_from_line(
                eta,
                line,
                cfg,
                float(time_value),
                0.90,
                0.10,
            )
        )
        total_variations.append(
            float(
                np.sum(
                    np.abs(np.diff(line))
                )
            )
        )

    predicted_positions = np.asarray(
        predicted_positions
    )
    time_used = np.asarray(time_used)

    speed_prediction = (
        float(
            np.polyfit(
                time_used,
                predicted_positions,
                1,
            )[0]
        )
        if len(time_used) >= 2
        else np.nan
    )
    speed_exact = shock_speed_eta(cfg)

    eta_final, final_line, _ = predict_line_cut(
        model,
        cfg,
        t_value=cfg.t_max,
        n=cfg.line_n,
    )
    final_position = crossing_location(
        eta_final,
        final_line,
        midpoint,
        expected=float(
            shock_eta(cfg.t_max, cfg)
        ),
        window=0.35,
    )

    overshoot = float(
        max(
            0.0,
            np.max(prediction)
            - max(cfg.uL, cfg.uR),
        )
    )
    undershoot = float(
        max(
            0.0,
            min(cfg.uL, cfg.uR)
            - np.min(prediction),
        )
    )

    total_variations = np.asarray(
        total_variations
    )
    tv_excess = np.maximum(
        total_variations
        - exact_total_variation,
        0.0,
    )

    return {
        "space_time_l1": space_time_l1,
        "space_time_l2": space_time_l2,
        "space_time_rel_l2": space_time_rel_l2,
        "final_l1": final_l1,
        "final_l2": final_l2,
        "final_rel_l2": final_rel_l2,
        "shock_pos_mae": float(
            np.mean(position_errors)
        ),
        "shock_pos_maxe": float(
            np.max(position_errors)
        ),
        "shock_pos_final_error": float(
            abs(
                final_position
                - float(
                    shock_eta(
                        cfg.t_max,
                        cfg,
                    )
                )
            )
        ),
        "shock_pos_final_pred_eta": float(
            final_position
        ),
        "shock_pos_final_exact_eta": float(
            shock_eta(cfg.t_max, cfg)
        ),
        "shock_speed_pred": speed_prediction,
        "shock_speed_exact": speed_exact,
        "shock_speed_error": (
            float(
                abs(
                    speed_prediction
                    - speed_exact
                )
            )
            if not np.isnan(speed_prediction)
            else np.nan
        ),
        "width_95_05_mean": float(
            np.mean(widths_95)
        ),
        "width_95_05_final": float(
            widths_95[-1]
        ),
        "width_90_10_mean": float(
            np.mean(widths_90)
        ),
        "width_90_10_final": float(
            widths_90[-1]
        ),
        "overshoot": overshoot,
        "undershoot": undershoot,
        "min_pred": float(
            np.min(prediction)
        ),
        "max_pred": float(
            np.max(prediction)
        ),
        "tv_line_mean": float(
            np.mean(total_variations)
        ),
        "tv_line_excess_mean": float(
            np.mean(tv_excess)
        ),
    }


def compute_global_conservation_metrics(
    model: nn.Module,
    cfg: Burgers2DConfig,
):
    x, y, t, prediction, exact = (
        predict_space_time_cube(
            model,
            cfg,
            nxy=cfg.cons_nxy,
            nt=cfg.cons_nt,
        )
    )

    mass = np.trapz(
        np.trapz(
            prediction,
            x,
            axis=2,
        ),
        y,
        axis=1,
    )
    mass_exact = np.trapz(
        np.trapz(
            exact,
            x,
            axis=2,
        ),
        y,
        axis=1,
    )

    flux_left = flux_numpy(
        prediction[:, :, 0]
    )
    flux_right = flux_numpy(
        prediction[:, :, -1]
    )
    flux_x_integral = np.trapz(
        flux_right - flux_left,
        y,
        axis=1,
    )

    flux_bottom = flux_numpy(
        prediction[:, 0, :]
    )
    flux_top = flux_numpy(
        prediction[:, -1, :]
    )
    flux_y_integral = np.trapz(
        flux_top - flux_bottom,
        x,
        axis=1,
    )
    total_boundary_flux = (
        flux_x_integral
        + flux_y_integral
    )

    conservation_values = []
    relative_values = []

    for index in range(len(t)):
        flux_integral = (
            np.trapz(
                total_boundary_flux[
                    : index + 1
                ],
                t[: index + 1],
            )
            if index > 0
            else 0.0
        )
        value = (
            mass[index]
            - mass[0]
            + flux_integral
        )
        denominator = (
            abs(mass[index] - mass[0])
            + abs(flux_integral)
            + 1.0e-12
        )
        conservation_values.append(value)
        relative_values.append(
            abs(value) / denominator
        )

    conservation_values = np.asarray(
        conservation_values
    )
    relative_values = np.asarray(
        relative_values
    )
    mass_error = mass - mass_exact

    return {
        "global_cons_mean_abs": float(
            np.mean(
                np.abs(
                    conservation_values
                )
            )
        ),
        "global_cons_max_abs": float(
            np.max(
                np.abs(
                    conservation_values
                )
            )
        ),
        "global_cons_final_abs": float(
            abs(conservation_values[-1])
        ),
        "global_cons_mean_rel": float(
            np.mean(relative_values[1:])
        ),
        "mass_error_mean_abs": float(
            np.mean(np.abs(mass_error))
        ),
        "mass_error_final_abs": float(
            abs(mass_error[-1])
        ),
    }


def compute_local_conservation_metrics(
    model: nn.Module,
    cfg: Burgers2DConfig,
):
    rng = np.random.default_rng(
        int(cfg.seed) + 141421
    )
    absolute_values = []
    relative_values = []

    for _ in range(
        cfg.n_control_volumes
    ):
        for _ in range(100):
            x1, x2 = np.sort(
                rng.uniform(
                    cfg.x_min,
                    cfg.x_max,
                    size=2,
                )
            )
            y1, y2 = np.sort(
                rng.uniform(
                    cfg.y_min,
                    cfg.y_max,
                    size=2,
                )
            )
            t1, t2 = np.sort(
                rng.uniform(
                    cfg.t_min,
                    cfg.t_max,
                    size=2,
                )
            )

            if (
                (x2 - x1) >= cfg.cv_min_width
                and (y2 - y1) >= cfg.cv_min_width
                and (t2 - t1)
                >= cfg.cv_min_duration
            ):
                break

        x_quadrature = np.linspace(
            x1,
            x2,
            cfg.cv_quad_nx,
        )
        y_quadrature = np.linspace(
            y1,
            y2,
            cfg.cv_quad_ny,
        )
        t_quadrature = np.linspace(
            t1,
            t2,
            cfg.cv_quad_nt,
        )

        Xq, Yq = np.meshgrid(
            x_quadrature,
            y_quadrature,
        )

        u_t1 = predict_points(
            model,
            Xq,
            Yq,
            np.full_like(Xq, t1),
        )
        u_t2 = predict_points(
            model,
            Xq,
            Yq,
            np.full_like(Xq, t2),
        )

        integral_t1 = np.trapz(
            np.trapz(
                u_t1,
                x_quadrature,
                axis=1,
            ),
            y_quadrature,
            axis=0,
        )
        integral_t2 = np.trapz(
            np.trapz(
                u_t2,
                x_quadrature,
                axis=1,
            ),
            y_quadrature,
            axis=0,
        )

        Y_face, T_face = np.meshgrid(
            y_quadrature,
            t_quadrature,
        )

        u_x1 = predict_points(
            model,
            np.full_like(Y_face, x1),
            Y_face,
            T_face,
        )
        u_x2 = predict_points(
            model,
            np.full_like(Y_face, x2),
            Y_face,
            T_face,
        )

        flux_x = np.trapz(
            np.trapz(
                flux_numpy(u_x2)
                - flux_numpy(u_x1),
                y_quadrature,
                axis=1,
            ),
            t_quadrature,
            axis=0,
        )

        X_face, T_face_y = np.meshgrid(
            x_quadrature,
            t_quadrature,
        )

        u_y1 = predict_points(
            model,
            X_face,
            np.full_like(X_face, y1),
            T_face_y,
        )
        u_y2 = predict_points(
            model,
            X_face,
            np.full_like(X_face, y2),
            T_face_y,
        )

        flux_y = np.trapz(
            np.trapz(
                flux_numpy(u_y2)
                - flux_numpy(u_y1),
                x_quadrature,
                axis=1,
            ),
            t_quadrature,
            axis=0,
        )

        residual = (
            integral_t2
            - integral_t1
            + flux_x
            + flux_y
        )
        denominator = (
            abs(integral_t2)
            + abs(integral_t1)
            + abs(flux_x)
            + abs(flux_y)
            + 1.0e-12
        )

        absolute_values.append(
            abs(residual)
        )
        relative_values.append(
            abs(residual)
            / denominator
        )

    absolute_values = np.asarray(
        absolute_values
    )
    relative_values = np.asarray(
        relative_values
    )

    return {
        "local_cons_cv_mean_abs": float(
            absolute_values.mean()
        ),
        "local_cons_cv_median_abs": float(
            np.median(absolute_values)
        ),
        "local_cons_cv_max_abs": float(
            absolute_values.max()
        ),
        "local_cons_cv_mean_rel": float(
            relative_values.mean()
        ),
        "local_cons_cv_max_rel": float(
            relative_values.max()
        ),
    }


@torch.no_grad()
def compute_gate_diagnostics(
    model: nn.Module,
    cfg: Burgers2DConfig,
    progress: float = 1.0,
):
    h_probe, cmin, normalized_progress = (
        trace_schedule(progress, cfg)
    )

    x = np.linspace(
        cfg.x_min,
        cfg.x_max,
        cfg.eval_nxy,
    )
    y = np.linspace(
        cfg.y_min,
        cfg.y_max,
        cfg.eval_nxy,
    )
    X, Y = np.meshgrid(x, y)
    T = np.full_like(X, cfg.t_max)

    device, dtype = _model_device_dtype(model)

    x_tensor = torch.tensor(
        X.reshape(-1, 1),
        device=device,
        dtype=dtype,
    )
    y_tensor = torch.tensor(
        Y.reshape(-1, 1),
        device=device,
        dtype=dtype,
    )
    t_tensor = torch.tensor(
        T.reshape(-1, 1),
        device=device,
        dtype=dtype,
    )

    (
        gate,
        ratio_max,
        normalized_jump_max,
        jump_mean,
        _,
        valid,
    ) = trace_ratio_gate_2d(
        model,
        x_tensor,
        y_tensor,
        t_tensor,
        h_probe,
        cmin,
        cfg,
    )

    gate_numpy = (
        gate.detach()
        .cpu()
        .numpy()
        .reshape(
            cfg.eval_nxy,
            cfg.eval_nxy,
        )
    )
    ratio_numpy = (
        ratio_max.detach()
        .cpu()
        .numpy()
        .reshape(
            cfg.eval_nxy,
            cfg.eval_nxy,
        )
    )
    normalized_jump_numpy = (
        normalized_jump_max.detach()
        .cpu()
        .numpy()
        .reshape(
            cfg.eval_nxy,
            cfg.eval_nxy,
        )
    )

    valid_any = (
        valid.max(
            dim=1,
            keepdim=True,
        )
        .values.detach()
        .cpu()
        .numpy()
        .reshape(
            cfg.eval_nxy,
            cfg.eval_nxy,
        )
        .astype(bool)
    )

    if valid_any.sum() == 0:
        valid_any = np.ones_like(
            valid_any,
            dtype=bool,
        )

    shock_band = (
        np.abs(
            eta_numpy(X, Y, cfg)
            - shock_eta(
                cfg.t_max,
                cfg,
            )
        )
        <= 2.0 * h_probe
    )
    active = gate_numpy > 0.5
    active_loose = gate_numpy > 0.1

    active_valid = active & valid_any
    shock_valid = shock_band & valid_any

    precision = (
        float(
            (
                active
                & shock_band
                & valid_any
            ).sum()
            / active_valid.sum()
        )
        if active_valid.sum() > 0
        else np.nan
    )
    recall = (
        float(
            (
                active
                & shock_band
                & valid_any
            ).sum()
            / shock_valid.sum()
        )
        if shock_valid.sum() > 0
        else np.nan
    )

    return {
        "gate_progress": float(
            normalized_progress
        ),
        "gate_h": float(h_probe),
        "gate_cmin": float(cmin),
        "gate_mean": float(
            gate_numpy[valid_any].mean()
        ),
        "gate_max": float(
            gate_numpy[valid_any].max()
        ),
        "gate_active_frac_gt_0p5": float(
            active_valid.sum()
            / valid_any.sum()
        ),
        "gate_active_frac_gt_0p1": float(
            (
                active_loose
                & valid_any
            ).sum()
            / valid_any.sum()
        ),
        "gate_C_mean": float(
            ratio_numpy[valid_any].mean()
        ),
        "gate_C_max": float(
            ratio_numpy[valid_any].max()
        ),
        "gate_Jhat_mean": float(
            normalized_jump_numpy[
                valid_any
            ].mean()
        ),
        "gate_Jhat_max": float(
            normalized_jump_numpy[
                valid_any
            ].max()
        ),
        "gate_Jbar": float(
            jump_mean.detach().cpu()
        ),
        "gate_precision_band_2h": precision,
        "gate_recall_band_2h": recall,
    }


def evaluate_model(
    model: nn.Module,
    model_name: str,
    cfg: Burgers2DConfig,
):
    row = {
        "model": model_name,
        "seed": int(cfg.seed),
    }
    row.update(
        compute_error_shock_metrics(
            model,
            cfg,
        )
    )
    row.update(
        compute_global_conservation_metrics(
            model,
            cfg,
        )
    )
    row.update(
        compute_local_conservation_metrics(
            model,
            cfg,
        )
    )
    row.update(
        compute_gate_diagnostics(
            model,
            cfg,
            progress=1.0,
        )
    )
    return row


def _save_checkpoint(
    model: nn.Module,
    path: Path,
    metrics: dict[str, Any],
) -> None:
    torch.save(
        {
            "model_state_dict": clone_state_dict_cpu(model),
            "extra": dict(metrics),
        },
        path,
    )


def run_paired_experiment(
    cfg: Burgers2DConfig,
    *,
    output_root: str | Path,
    protected_reported_root: str | Path,
    overwrite: bool = False,
) -> pd.DataFrame:
    """Run the canonical shared-warmup 2D Burgers paired experiment.

    This function never writes into immutable reported artifacts.
    """

    device = resolve_device(cfg.device)
    dtype = dtype_from_config(cfg)
    set_seed(cfg.seed)

    root = ensure_unprotected_output(
        output_root,
        protected_reported_root,
    )
    seed_root = (
        root
        / "2d_burgers"
        / f"seed_{cfg.seed}"
    )

    if (
        seed_root.exists()
        and any(seed_root.iterdir())
        and not overwrite
    ):
        raise FileExistsError(
            f"Reproduction output already exists: {seed_root}. "
            "Use overwrite explicitly."
        )

    seed_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_json_atomic(
        {
            "equation": "2d_burgers",
            "seed": int(cfg.seed),
            "config": asdict(cfg),
            "protocol": (
                "shared warm-up; paired PINN and TRG-PINN continuation; "
                "same RNG state restored before each continuation; "
                "post-valid-mask multidirectional gate; no validation; "
                "no early stopping; final checkpoint evaluation"
            ),
            "implementation_revision": "valid_mask_all_probe_bounds_v3",
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(device),
        },
        seed_root / "config.json",
    )

    warmup_model = build_model(cfg).to(
        device=device,
        dtype=dtype,
    )
    warmup_start = time.time()
    warmup_model, warmup_history = train_warmup(
        warmup_model,
        cfg,
    )
    warmup_seconds = time.time() - warmup_start

    warmup_state = clone_state_dict_cpu(
        warmup_model
    )
    continuation_state = (
        capture_rng_state()
    )

    pinn_model = build_model(cfg).to(
        device=device,
        dtype=dtype,
    )
    load_state_dict_to_model(
        pinn_model,
        warmup_state,
    )
    restore_rng_state(
        continuation_state
    )
    pinn_start = time.time()
    pinn_model, pinn_history = (
        train_pinn_continuation(
            pinn_model,
            cfg,
        )
    )
    pinn_seconds = time.time() - pinn_start

    trg_model = build_model(cfg).to(
        device=device,
        dtype=dtype,
    )
    load_state_dict_to_model(
        trg_model,
        warmup_state,
    )
    restore_rng_state(
        continuation_state
    )
    trg_start = time.time()
    trg_model, trg_history = (
        train_trg_continuation(
            trg_model,
            cfg,
        )
    )
    trg_seconds = time.time() - trg_start

    pinn_metrics = evaluate_model(
        pinn_model,
        "PINN",
        cfg,
    )
    trg_metrics = evaluate_model(
        trg_model,
        "TRG-PINN",
        cfg,
    )

    for metrics, continuation_seconds in (
        (pinn_metrics, pinn_seconds),
        (trg_metrics, trg_seconds),
    ):
        metrics.update(
            {
                "equation": "2d_burgers",
                "method": metrics["model"],
                "implementation_revision": (
                    "valid_mask_all_probe_bounds_v3"
                ),
                "warmup_iters": cfg.warmup_iters,
                "continuation_iters": cfg.gated_iters,
                "adam_total_iters": (
                    cfg.warmup_iters
                    + cfg.gated_iters
                ),
                "optimizer_main": "AdamW",
                "lbfgs_used": False,
                "wall_clock_sec_warmup": (
                    warmup_seconds
                ),
                "wall_clock_sec_continuation": (
                    continuation_seconds
                ),
                "wall_clock_sec_total": (
                    warmup_seconds
                    + continuation_seconds
                ),
                "num_parameters": count_parameters(
                    pinn_model
                ),
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

    frame = pd.DataFrame(
        [pinn_metrics, trg_metrics]
    )
    frame.to_csv(
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
            "equation": "2d_burgers",
            "method": "shared_warmup",
            "seed": int(cfg.seed),
            "implementation_revision": (
                "valid_mask_all_probe_bounds_v3"
            ),
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
    (
        seed_root / "_SUCCESS"
    ).write_text(
        "success\\n",
        encoding="utf-8",
    )

    return frame
