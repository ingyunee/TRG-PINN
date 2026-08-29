"""Trace-ratio schedules and nontrainable residual gates."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Protocol

import numpy as np
import torch


class TraceConfig1D(Protocol):
    x_min: float
    x_max: float
    n_f: int
    h_min_factor: float
    h_max_factor: float
    cmin_start: float
    cmin_end: float
    beta: float
    residual_floor: float


@dataclass(frozen=True)
class TraceGateResult:
    gate: torch.Tensor
    ratio: torch.Tensor
    normalized_jump: torch.Tensor
    jump_mean: torch.Tensor
    valid: torch.Tensor


def characteristic_spacing_1d(cfg: TraceConfig1D) -> float:
    return (float(cfg.x_max) - float(cfg.x_min)) / math.sqrt(float(cfg.n_f))


def trace_schedule_1d(progress: float, cfg: TraceConfig1D) -> tuple[float, float, float]:
    normalized = float(np.clip(progress, 0.0, 1.0))
    base = characteristic_spacing_1d(cfg)
    distance = base * (
        float(cfg.h_min_factor)
        + (float(cfg.h_max_factor) - float(cfg.h_min_factor)) * (1.0 - normalized) ** 2
    )
    threshold = float(cfg.cmin_start) + (
        float(cfg.cmin_end) - float(cfg.cmin_start)
    ) * normalized
    return distance, threshold, normalized


def trace_schedule_from_iteration_1d(
    iteration: int,
    total_iterations: int,
    cfg: TraceConfig1D,
) -> tuple[float, float, float]:
    return trace_schedule_1d(float(iteration) / max(1, int(total_iterations)), cfg)


@torch.no_grad()
def centered_trace_ratio_gate_1d(
    model: torch.nn.Module,
    x: torch.Tensor,
    t: torch.Tensor,
    *,
    h_probe: float,
    cmin: float,
    x_min: float,
    x_max: float,
    state_scale: float,
    beta: float,
    ratio_epsilon: float = 1.0e-6,
    mean_epsilon: float = 1.0e-8,
) -> TraceGateResult:
    x_m1 = x - h_probe
    x_p1 = x + h_probe
    x_m2 = x - 2.0 * h_probe
    x_p2 = x + 2.0 * h_probe

    valid = ((x_m2 >= x_min) & (x_p2 <= x_max)).to(dtype=x.dtype)

    x_m1 = x_m1.clamp(x_min, x_max)
    x_p1 = x_p1.clamp(x_min, x_max)
    x_m2 = x_m2.clamp(x_min, x_max)
    x_p2 = x_p2.clamp(x_min, x_max)

    u_m1 = model(torch.cat([x_m1, t], dim=1))
    u_p1 = model(torch.cat([x_p1, t], dim=1))
    u_m2 = model(torch.cat([x_m2, t], dim=1))
    u_p2 = model(torch.cat([x_p2, t], dim=1))

    scale = max(float(state_scale), 1.0e-8)
    jump_h = (u_m1 - u_p1).abs() / scale
    jump_2h = (u_m2 - u_p2).abs() / scale

    ratio = torch.clamp(jump_h / (jump_2h + ratio_epsilon), 0.0, 2.0)
    valid_sum = valid.sum()
    jump_mean_valid = (jump_h * valid).sum() / (valid_sum + mean_epsilon)
    jump_mean_all = jump_h.mean()
    jump_mean = torch.where(valid_sum > 0, jump_mean_valid, jump_mean_all).clamp_min(
        mean_epsilon
    )
    normalized_jump = jump_h / jump_mean

    gate = (
        torch.sigmoid((normalized_jump - 1.0) / float(beta))
        * torch.sigmoid((ratio - float(cmin)) / float(beta))
        * valid
    )

    return TraceGateResult(
        gate=gate,
        ratio=ratio,
        normalized_jump=normalized_jump,
        jump_mean=jump_mean,
        valid=valid,
    )


def residual_weight(gate: torch.Tensor, residual_floor: float) -> torch.Tensor:
    floor = float(residual_floor)
    if not 0.0 < floor <= 1.0:
        raise ValueError("Residual floor must lie in (0, 1].")
    return floor + (1.0 - floor) * (1.0 - gate)


@torch.no_grad()
def centered_euler_trace_ratio_gate_1d(
    model: torch.nn.Module,
    x: torch.Tensor,
    t: torch.Tensor,
    *,
    h_probe: float,
    cmin: float,
    x_min: float,
    x_max: float,
    rho_scale: float,
    velocity_scale: float,
    pressure_scale: float,
    beta: float,
    norm_epsilon: float = 1.0e-12,
    ratio_epsilon: float = 1.0e-6,
    mean_epsilon: float = 1.0e-8,
) -> TraceGateResult:
    """Centered 1D trace-ratio gate for primitive Euler outputs.

    The explicit three-component arithmetic preserves the operation order used
    by the reported 1D Euler notebook.
    """

    x_m1 = x - h_probe
    x_p1 = x + h_probe
    x_m2 = x - 2.0 * h_probe
    x_p2 = x + 2.0 * h_probe
    valid = ((x_m2 >= x_min) & (x_p2 <= x_max)).to(dtype=x.dtype)

    x_m1 = x_m1.clamp(x_min, x_max)
    x_p1 = x_p1.clamp(x_min, x_max)
    x_m2 = x_m2.clamp(x_min, x_max)
    x_p2 = x_p2.clamp(x_min, x_max)

    state_m1 = model(torch.cat([x_m1, t], dim=1))
    state_p1 = model(torch.cat([x_p1, t], dim=1))
    state_m2 = model(torch.cat([x_m2, t], dim=1))
    state_p2 = model(torch.cat([x_p2, t], dim=1))

    d1_rho = (state_m1[:, 0:1] - state_p1[:, 0:1]) / float(rho_scale)
    d1_velocity = (
        state_m1[:, 1:2] - state_p1[:, 1:2]
    ) / float(velocity_scale)
    d1_pressure = (
        state_m1[:, 2:3] - state_p1[:, 2:3]
    ) / float(pressure_scale)

    d2_rho = (state_m2[:, 0:1] - state_p2[:, 0:1]) / float(rho_scale)
    d2_velocity = (
        state_m2[:, 1:2] - state_p2[:, 1:2]
    ) / float(velocity_scale)
    d2_pressure = (
        state_m2[:, 2:3] - state_p2[:, 2:3]
    ) / float(pressure_scale)

    jump_h = torch.sqrt(
        d1_rho.pow(2)
        + d1_velocity.pow(2)
        + d1_pressure.pow(2)
        + float(norm_epsilon)
    )
    jump_2h = torch.sqrt(
        d2_rho.pow(2)
        + d2_velocity.pow(2)
        + d2_pressure.pow(2)
        + float(norm_epsilon)
    )
    ratio = torch.clamp(jump_h / (jump_2h + float(ratio_epsilon)), 0.0, 2.0)

    valid_sum = valid.sum()
    jump_mean_valid = (jump_h * valid).sum() / (valid_sum + float(mean_epsilon))
    jump_mean_all = jump_h.mean()
    jump_mean = torch.where(
        valid_sum > 0, jump_mean_valid, jump_mean_all
    ).clamp_min(float(mean_epsilon))
    normalized_jump = jump_h / jump_mean

    gate = (
        torch.sigmoid((normalized_jump - 1.0) / float(beta))
        * torch.sigmoid((ratio - float(cmin)) / float(beta))
        * valid
    )

    return TraceGateResult(
        gate=gate,
        ratio=ratio,
        normalized_jump=normalized_jump,
        jump_mean=jump_mean,
        valid=valid,
    )

@torch.no_grad()
def centered_shallowwater_trace_ratio_gate_1d(
    model: torch.nn.Module,
    x: torch.Tensor,
    t: torch.Tensor,
    *,
    h_probe: float,
    cmin: float,
    x_min: float,
    x_max: float,
    h_scale: float,
    q_scale: float,
    beta: float,
    norm_epsilon: float = 1.0e-12,
    ratio_epsilon: float = 1.0e-6,
    mean_epsilon: float = 1.0e-8,
) -> TraceGateResult:
    """Centered 1D trace-ratio gate for conservative shallow-water outputs.

    The explicit two-component arithmetic preserves the operation order used by
    the reported 1D shallow-water notebook.
    """

    x_m1 = x - h_probe
    x_p1 = x + h_probe
    x_m2 = x - 2.0 * h_probe
    x_p2 = x + 2.0 * h_probe
    valid = ((x_m2 >= x_min) & (x_p2 <= x_max)).to(dtype=x.dtype)

    x_m1 = x_m1.clamp(x_min, x_max)
    x_p1 = x_p1.clamp(x_min, x_max)
    x_m2 = x_m2.clamp(x_min, x_max)
    x_p2 = x_p2.clamp(x_min, x_max)

    state_m1 = model(torch.cat([x_m1, t], dim=1))
    state_p1 = model(torch.cat([x_p1, t], dim=1))
    state_m2 = model(torch.cat([x_m2, t], dim=1))
    state_p2 = model(torch.cat([x_p2, t], dim=1))

    d1_h = (state_m1[:, 0:1] - state_p1[:, 0:1]) / float(h_scale)
    d1_q = (state_m1[:, 1:2] - state_p1[:, 1:2]) / float(q_scale)
    d2_h = (state_m2[:, 0:1] - state_p2[:, 0:1]) / float(h_scale)
    d2_q = (state_m2[:, 1:2] - state_p2[:, 1:2]) / float(q_scale)

    jump_h = torch.sqrt(
        d1_h.pow(2)
        + d1_q.pow(2)
        + float(norm_epsilon)
    )
    jump_2h = torch.sqrt(
        d2_h.pow(2)
        + d2_q.pow(2)
        + float(norm_epsilon)
    )
    ratio = torch.clamp(
        jump_h / (jump_2h + float(ratio_epsilon)),
        0.0,
        2.0,
    )

    valid_sum = valid.sum()
    jump_mean_valid = (
        (jump_h * valid).sum()
        / (valid_sum + float(mean_epsilon))
    )
    jump_mean_all = jump_h.mean()
    jump_mean = torch.where(
        valid_sum > 0,
        jump_mean_valid,
        jump_mean_all,
    ).clamp_min(float(mean_epsilon))
    normalized_jump = jump_h / jump_mean

    gate = (
        torch.sigmoid((normalized_jump - 1.0) / float(beta))
        * torch.sigmoid((ratio - float(cmin)) / float(beta))
        * valid
    )

    return TraceGateResult(
        gate=gate,
        ratio=ratio,
        normalized_jump=normalized_jump,
        jump_mean=jump_mean,
        valid=valid,
    )

