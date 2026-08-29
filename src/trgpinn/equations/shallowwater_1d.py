
"""Canonical 1D shallow-water implementation used for the manuscript.

The module preserves the reported Stoker benchmark's network parameterization,
sampling order, residual scaling, detached trace-ratio gate, paired RNG
restoration, optimizer/scheduler order, and exact-Stoker final-checkpoint
evaluation.
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
    centered_shallowwater_trace_ratio_gate_1d,
    residual_weight,
    trace_schedule_1d,
    trace_schedule_from_iteration_1d,
)
from trgpinn.models import (
    ConservativeShallowWaterMLP1D,
    trainable_parameter_count,
)
from trgpinn.sampling import (
    sample_shallowwater_1d_boundary,
    sample_shallowwater_1d_initial,
    sample_shallowwater_1d_interior,
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


REFERENCE_TYPE = "exact_stoker_entropy_solution"
PRIMARY_METRIC = "state_scaled_space_time_rel_l2"
FINAL_METRIC = "state_scaled_final_rel_l2"


@dataclass
class ShallowWater1DConfig:
    seed: int = 2026
    device: str = "auto"
    dtype: str = "float32"
    output_dir: str = "runs_shallow_water_trace_ratio_paper"
    experiment_name: str = "shallow_water_dambreak_trace_ratio_original_schedule"
    save_outputs: bool = True

    x_min: float = -1.0
    x_max: float = 1.0
    t_min: float = 0.0
    t_max: float = 0.25
    g_const: float = 1.0
    hL: float = 2.0
    qL: float = 0.0
    hR: float = 1.0
    qR: float = 0.0
    x0: float = 0.0

    width: int = 128
    depth: int = 6
    activation: str = "tanh"
    h_floor: float = 1.0e-5

    warmup_iters: int = 3000
    gated_iters: int = 7000
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

    h_min_factor: float = 1.0
    h_max_factor: float = 5.0
    cmin_start: float = 0.50
    cmin_end: float = 0.80
    beta: float = 0.05
    residual_floor: float = 0.02

    trace_norm_epsilon: float = 1.0e-12
    trace_ratio_epsilon: float = 1.0e-6
    batch_mean_epsilon: float = 1.0e-8
    weighted_loss_epsilon: float = 1.0e-8
    relative_error_epsilon: float = 1.0e-12

    print_every: int = 500
    history_every: int = 50

    eval_nx: int = 500
    eval_nt: int = 260
    slice_nx: int = 1600
    t_final_plot: float = 0.25

    n_control_volumes: int = 64
    cv_quad_nx: int = 64
    cv_quad_nt: int = 64
    cv_min_width: float = 0.15
    cv_min_duration: float = 0.04

    @classmethod
    def from_legacy_mapping(
        cls,
        mapping: dict[str, Any],
        **overrides: Any,
    ) -> "ShallowWater1DConfig":
        valid = {field.name for field in fields(cls)}
        values = {key: value for key, value in mapping.items() if key in valid}
        values.update(overrides)
        return cls(**values)

    def smoke_copy(
        self,
        *,
        seed: int = 2026,
        device: str = "auto",
    ) -> "ShallowWater1DConfig":
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
            cv_min_width=0.15,
            cv_min_duration=0.04,
        )


def build_model(cfg: ShallowWater1DConfig) -> ConservativeShallowWaterMLP1D:
    return ConservativeShallowWaterMLP1D(
        x_min=cfg.x_min,
        x_max=cfg.x_max,
        t_min=cfg.t_min,
        t_max=cfg.t_max,
        hidden_width=cfg.width,
        hidden_layers=cfg.depth,
        activation=cfg.activation,
        h_floor=cfg.h_floor,
        left_state=(cfg.hL, cfg.qL),
        right_state=(cfg.hR, cfg.qR),
    )


def cat_xt(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    return torch.cat([x, t], dim=1)


def state_scales(cfg: ShallowWater1DConfig) -> tuple[float, float]:
    h_scale = max(float(cfg.hL), float(cfg.hR), 1.0)
    c_scale = math.sqrt(float(cfg.g_const) * h_scale)
    q_scale = max(
        h_scale * c_scale,
        abs(float(cfg.qL)),
        abs(float(cfg.qR)),
        1.0,
    )
    return h_scale, q_scale


def h0(cfg: ShallowWater1DConfig) -> float:
    return (cfg.x_max - cfg.x_min) / math.sqrt(cfg.n_f)


def schedule_from_progress(
    progress: float,
    cfg: ShallowWater1DConfig,
) -> tuple[float, float, float]:
    return trace_schedule_1d(progress, cfg)


def schedule_from_iter(
    iteration: int,
    total_iterations: int,
    cfg: ShallowWater1DConfig,
) -> tuple[float, float, float]:
    return trace_schedule_from_iteration_1d(iteration, total_iterations, cfg)


def stoker_star_state(
    cfg: ShallowWater1DConfig,
) -> tuple[float, float]:
    h_left = float(cfg.hL)
    h_right = float(cfg.hR)
    u_left = float(cfg.qL) / h_left
    u_right = float(cfg.qR) / h_right
    gravity = float(cfg.g_const)

    def phi(depth: float, depth_k: float) -> float:
        if depth > depth_k:
            return (
                (depth - depth_k)
                * math.sqrt(
                    0.5
                    * gravity
                    * (depth + depth_k)
                    / (depth * depth_k)
                )
            )
        return 2.0 * (
            math.sqrt(gravity * depth)
            - math.sqrt(gravity * depth_k)
        )

    def residual(depth: float) -> float:
        return (
            phi(depth, h_left)
            + phi(depth, h_right)
            + u_right
            - u_left
        )

    lower = 1.0e-12
    upper = max(h_left, h_right, 1.0)
    while residual(upper) < 0.0:
        upper *= 2.0

    for _ in range(160):
        midpoint = 0.5 * (lower + upper)
        if residual(midpoint) > 0.0:
            upper = midpoint
        else:
            lower = midpoint

    h_star = 0.5 * (lower + upper)
    u_star = 0.5 * (
        u_left
        + u_right
        + phi(h_star, h_right)
        - phi(h_star, h_left)
    )
    return h_star, u_star


def stoker_exact_hq(
    x: np.ndarray | float,
    t: np.ndarray | float,
    cfg: ShallowWater1DConfig,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    x_values, t_values = np.broadcast_arrays(
        np.asarray(x, dtype=np.float64),
        np.asarray(t, dtype=np.float64),
    )

    h_left = float(cfg.hL)
    h_right = float(cfg.hR)
    u_left = float(cfg.qL) / h_left
    u_right = float(cfg.qR) / h_right
    gravity = float(cfg.g_const)
    x0 = float(cfg.x0)

    h_star, u_star = stoker_star_state(cfg)
    c_left = math.sqrt(gravity * h_left)
    c_star = math.sqrt(gravity * h_star)
    speed_head = u_left - c_left
    speed_tail = u_star - c_star
    speed_shock = (
        h_star * u_star - h_right * u_right
    ) / (h_star - h_right)

    depth = np.empty_like(x_values)
    velocity = np.empty_like(x_values)

    initial = t_values <= 1.0e-14
    depth[initial] = np.where(
        x_values[initial] < x0,
        h_left,
        h_right,
    )
    velocity[initial] = np.where(
        x_values[initial] < x0,
        u_left,
        u_right,
    )

    dynamic = ~initial
    if np.any(dynamic):
        xi = (x_values[dynamic] - x0) / t_values[dynamic]
        dynamic_depth = np.empty_like(xi)
        dynamic_velocity = np.empty_like(xi)

        left = xi <= speed_head
        fan = (xi > speed_head) & (xi <= speed_tail)
        star = (xi > speed_tail) & (xi < speed_shock)
        right = xi >= speed_shock

        dynamic_depth[left] = h_left
        dynamic_velocity[left] = u_left

        c_fan = (
            u_left + 2.0 * c_left - xi[fan]
        ) / 3.0
        u_fan = (
            u_left + 2.0 * c_left + 2.0 * xi[fan]
        ) / 3.0
        dynamic_depth[fan] = c_fan**2 / gravity
        dynamic_velocity[fan] = u_fan

        dynamic_depth[star] = h_star
        dynamic_velocity[star] = u_star
        dynamic_depth[right] = h_right
        dynamic_velocity[right] = u_right

        depth[dynamic] = dynamic_depth
        velocity[dynamic] = dynamic_velocity

    discharge = depth * velocity
    info = {
        "h_star": h_star,
        "u_star": u_star,
        "q_star": h_star * u_star,
        "left_rarefaction_head_speed": speed_head,
        "left_rarefaction_tail_speed": speed_tail,
        "right_shock_speed": speed_shock,
    }
    return depth, discharge, info


def shallowwater_flux_np(
    depth: np.ndarray,
    discharge: np.ndarray,
    cfg: ShallowWater1DConfig,
) -> tuple[np.ndarray, np.ndarray]:
    safe_depth = np.maximum(np.asarray(depth), 1.0e-10)
    discharge = np.asarray(discharge)
    return (
        discharge,
        discharge**2 / safe_depth
        + 0.5 * cfg.g_const * safe_depth**2,
    )


def shallowwater_flux_torch(
    depth: torch.Tensor,
    discharge: torch.Tensor,
    cfg: ShallowWater1DConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    safe_depth = torch.clamp(depth, min=1.0e-8)
    return (
        discharge,
        discharge.pow(2) / safe_depth
        + 0.5 * cfg.g_const * safe_depth.pow(2),
    )


def shallowwater_residual(
    model: nn.Module,
    x: torch.Tensor,
    t: torch.Tensor,
    cfg: ShallowWater1DConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    x = x.detach().clone().requires_grad_(True)
    t = t.detach().clone().requires_grad_(True)
    state = model(cat_xt(x, t))
    depth = state[:, 0:1]
    discharge = state[:, 1:2]
    flux_mass, flux_momentum = shallowwater_flux_torch(
        depth, discharge, cfg
    )

    depth_t = torch.autograd.grad(
        depth,
        t,
        grad_outputs=torch.ones_like(depth),
        create_graph=True,
        retain_graph=True,
    )[0]
    discharge_t = torch.autograd.grad(
        discharge,
        t,
        grad_outputs=torch.ones_like(discharge),
        create_graph=True,
        retain_graph=True,
    )[0]
    flux_mass_x = torch.autograd.grad(
        flux_mass,
        x,
        grad_outputs=torch.ones_like(flux_mass),
        create_graph=True,
        retain_graph=True,
    )[0]
    flux_momentum_x = torch.autograd.grad(
        flux_momentum,
        x,
        grad_outputs=torch.ones_like(flux_momentum),
        create_graph=True,
        retain_graph=True,
    )[0]

    residual_mass = depth_t + flux_mass_x
    residual_momentum = discharge_t + flux_momentum_x
    h_scale, q_scale = state_scales(cfg)
    residual_norm_squared = (
        (residual_mass / h_scale).pow(2)
        + (residual_momentum / q_scale).pow(2)
    )
    return (
        state,
        residual_mass,
        residual_momentum,
        residual_norm_squared,
    )


def scaled_state_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    cfg: ShallowWater1DConfig,
) -> torch.Tensor:
    h_scale, q_scale = state_scales(cfg)
    return (
        ((prediction[:, 0:1] - target[:, 0:1]) / h_scale).pow(2)
        + ((prediction[:, 1:2] - target[:, 1:2]) / q_scale).pow(2)
    ).mean()


def _model_runtime(model: nn.Module) -> tuple[torch.device, torch.dtype]:
    parameter = next(model.parameters())
    return parameter.device, parameter.dtype


def pinn_loss(
    model: nn.Module,
    cfg: ShallowWater1DConfig,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    device, dtype = _model_runtime(model)
    x_ic, t_ic, target_ic = sample_shallowwater_1d_initial(
        cfg.n_ic, cfg, device=device, dtype=dtype
    )
    x_bc, t_bc, target_bc = sample_shallowwater_1d_boundary(
        cfg.n_bc, cfg, device=device, dtype=dtype
    )
    x_f, t_f = sample_shallowwater_1d_interior(
        cfg.n_f, cfg, device=device, dtype=dtype
    )

    prediction_ic = model(cat_xt(x_ic, t_ic))
    prediction_bc = model(cat_xt(x_bc, t_bc))
    _, _, _, residual_norm_squared = shallowwater_residual(
        model, x_f, t_f, cfg
    )

    loss_ic = scaled_state_mse(prediction_ic, target_ic, cfg)
    loss_bc = scaled_state_mse(prediction_bc, target_bc, cfg)
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
    cfg: ShallowWater1DConfig,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    device, dtype = _model_runtime(model)
    h_probe, cmin, progress = schedule_from_iter(
        iteration, total_iterations, cfg
    )

    x_ic, t_ic, target_ic = sample_shallowwater_1d_initial(
        cfg.n_ic, cfg, device=device, dtype=dtype
    )
    x_bc, t_bc, target_bc = sample_shallowwater_1d_boundary(
        cfg.n_bc, cfg, device=device, dtype=dtype
    )
    x_f, t_f = sample_shallowwater_1d_interior(
        cfg.n_f, cfg, device=device, dtype=dtype
    )

    prediction_ic = model(cat_xt(x_ic, t_ic))
    prediction_bc = model(cat_xt(x_bc, t_bc))
    _, _, _, residual_norm_squared = shallowwater_residual(
        model, x_f, t_f, cfg
    )

    h_scale, q_scale = state_scales(cfg)
    gate_result = centered_shallowwater_trace_ratio_gate_1d(
        model,
        x_f,
        t_f,
        h_probe=h_probe,
        cmin=cmin,
        x_min=cfg.x_min,
        x_max=cfg.x_max,
        h_scale=h_scale,
        q_scale=q_scale,
        beta=cfg.beta,
        norm_epsilon=cfg.trace_norm_epsilon,
        ratio_epsilon=cfg.trace_ratio_epsilon,
        mean_epsilon=cfg.batch_mean_epsilon,
    )
    weight = residual_weight(gate_result.gate, cfg.residual_floor)

    loss_ic = scaled_state_mse(prediction_ic, target_ic, cfg)
    loss_bc = scaled_state_mse(prediction_bc, target_bc, cfg)
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
    cfg: ShallowWater1DConfig,
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
    cfg: ShallowWater1DConfig,
) -> tuple[nn.Module, pd.DataFrame]:
    return _train_phase(model, cfg, phase="warmup")


def train_pinn_continuation(
    model: nn.Module,
    cfg: ShallowWater1DConfig,
) -> tuple[nn.Module, pd.DataFrame]:
    return _train_phase(model, cfg, phase="pinn")


def train_trg_continuation(
    model: nn.Module,
    cfg: ShallowWater1DConfig,
) -> tuple[nn.Module, pd.DataFrame]:
    return _train_phase(model, cfg, phase="trg_pinn")


@torch.no_grad()
def predict_points(
    model: nn.Module,
    x_values: np.ndarray,
    t_values: np.ndarray,
    *,
    batch_size: int = 65536,
) -> np.ndarray:
    x_flat = np.asarray(x_values, dtype=np.float64).reshape(-1)
    t_flat = np.asarray(t_values, dtype=np.float64).reshape(-1)
    if x_flat.shape != t_flat.shape:
        raise ValueError("x_values and t_values must have the same shape.")

    parameter = next(model.parameters())
    outputs = []
    for start in range(0, x_flat.size, int(batch_size)):
        stop = min(start + int(batch_size), x_flat.size)
        coordinates = torch.tensor(
            np.stack(
                [x_flat[start:stop], t_flat[start:stop]],
                axis=1,
            ),
            device=parameter.device,
            dtype=parameter.dtype,
        )
        outputs.append(model(coordinates).detach().cpu().numpy())

    values = np.concatenate(outputs, axis=0)
    return values.reshape(np.asarray(x_values).shape + (2,))


@torch.no_grad()
def predict_grid(
    model: nn.Module,
    cfg: ShallowWater1DConfig,
    *,
    nx: int | None = None,
    nt: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    nx = cfg.eval_nx if nx is None else int(nx)
    nt = cfg.eval_nt if nt is None else int(nt)
    x = np.linspace(cfg.x_min, cfg.x_max, nx)
    t = np.linspace(cfg.t_min, cfg.t_max, nt)
    X, T = np.meshgrid(x, t)
    state = predict_points(model, X, T)
    return x, t, state[:, :, 0], state[:, :, 1]


@torch.no_grad()
def predict_slice(
    model: nn.Module,
    t_value: float,
    cfg: ShallowWater1DConfig,
    *,
    nx: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nx = cfg.slice_nx if nx is None else int(nx)
    x = np.linspace(cfg.x_min, cfg.x_max, nx)
    t = np.full_like(x, float(t_value))
    state = predict_points(model, x, t)
    return x, state[:, 0], state[:, 1]


def compute_error_metrics(
    model: nn.Module,
    cfg: ShallowWater1DConfig,
    *,
    grid: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> dict[str, Any]:
    if grid is None:
        grid = predict_grid(model, cfg)
    x, t, depth, discharge = grid
    X, T = np.meshgrid(x, t)
    depth_exact, discharge_exact, _ = stoker_exact_hq(X, T, cfg)

    row: dict[str, Any] = {}
    for name, prediction, reference in (
        ("h", depth, depth_exact),
        ("q", discharge, discharge_exact),
    ):
        error = prediction - reference
        l2 = float(np.sqrt(np.mean(error**2)))
        reference_rms = float(np.sqrt(np.mean(reference**2)))
        row[f"{name}_space_time_l1"] = float(np.mean(np.abs(error)))
        row[f"{name}_space_time_l2"] = l2
        row[f"{name}_space_time_rel_l2"] = float(
            l2 / (reference_rms + cfg.relative_error_epsilon)
        )

        final_error = error[-1]
        final_reference = reference[-1]
        final_l2 = float(np.sqrt(np.mean(final_error**2)))
        final_reference_rms = float(np.sqrt(np.mean(final_reference**2)))
        row[f"{name}_final_l1"] = float(np.mean(np.abs(final_error)))
        row[f"{name}_final_l2"] = final_l2
        row[f"{name}_final_rel_l2"] = float(
            final_l2 / (
                final_reference_rms + cfg.relative_error_epsilon
            )
        )

    row.update(
        {
            "h_min": float(depth.min()),
            "h_max": float(depth.max()),
            "q_min": float(discharge.min()),
            "q_max": float(discharge.max()),
            "h_positivity_violation": float(
                max(0.0, cfg.h_floor - float(depth.min()))
            ),
        }
    )

    x_slice, h_slice, q_slice = predict_slice(
        model,
        cfg.t_final_plot,
        cfg,
        nx=cfg.slice_nx,
    )
    t_slice = np.full_like(x_slice, cfg.t_final_plot)
    h_slice_exact, q_slice_exact, _ = stoker_exact_hq(
        x_slice, t_slice, cfg
    )
    for name, prediction, reference in (
        ("h", h_slice, h_slice_exact),
        ("q", q_slice, q_slice_exact),
    ):
        error = prediction - reference
        l2 = float(np.sqrt(np.mean(error**2)))
        reference_rms = float(np.sqrt(np.mean(reference**2)))
        row[f"{name}_slice_l1_t"] = float(np.mean(np.abs(error)))
        row[f"{name}_slice_l2_t"] = l2
        row[f"{name}_slice_rel_l2_t"] = float(
            l2 / (reference_rms + cfg.relative_error_epsilon)
        )
        predicted_tv = float(np.sum(np.abs(np.diff(prediction))))
        reference_tv = float(np.sum(np.abs(np.diff(reference))))
        row[f"{name}_tv_t"] = predicted_tv
        row[f"{name}_tv_ref_t"] = reference_tv
        row[f"{name}_tv_excess_t"] = float(
            max(0.0, predicted_tv - reference_tv)
        )

    h_scale, q_scale = state_scales(cfg)
    state_space_l2 = float(
        math.sqrt(
            (row["h_space_time_l2"] / h_scale) ** 2
            + (row["q_space_time_l2"] / q_scale) ** 2
        )
    )
    state_final_l2 = float(
        math.sqrt(
            (row["h_final_l2"] / h_scale) ** 2
            + (row["q_final_l2"] / q_scale) ** 2
        )
    )
    reference_norm_space_time = float(
        np.sqrt(
            np.mean(
                (depth_exact / h_scale) ** 2
                + (discharge_exact / q_scale) ** 2
            )
        )
    )
    reference_norm_final = float(
        np.sqrt(
            np.mean(
                (depth_exact[-1] / h_scale) ** 2
                + (discharge_exact[-1] / q_scale) ** 2
            )
        )
    )
    row.update(
        {
            "state_scaled_space_time_l2": state_space_l2,
            PRIMARY_METRIC: float(
                state_space_l2
                / (
                    reference_norm_space_time
                    + cfg.relative_error_epsilon
                )
            ),
            "state_scaled_final_l2": state_final_l2,
            FINAL_METRIC: float(
                state_final_l2
                / (
                    reference_norm_final
                    + cfg.relative_error_epsilon
                )
            ),
            "state_scale_h": h_scale,
            "state_scale_q": q_scale,
            "state_reference_norm_space_time": (
                reference_norm_space_time
            ),
            "state_reference_norm_final": reference_norm_final,
            "state_scaled_reference_type": REFERENCE_TYPE,
            "state_scaled_metric_formula": (
                "sqrt(mean((dh/h_scale)^2+(dq/q_scale)^2))/"
                "sqrt(mean((h_ref/h_scale)^2+(q_ref/q_scale)^2))"
            ),
            "primary_metric": PRIMARY_METRIC,
            "reference_type": REFERENCE_TYPE,
            "reference_is_exact": True,
            "reference_name": (
                "Stoker wet-bed dam-break entropy solution"
            ),
            "reference_evaluation_mode": "pointwise_analytic",
            "reference_eval_nx": cfg.eval_nx,
            "reference_eval_nt": cfg.eval_nt,
            "reference_t_final": cfg.t_final_plot,
        }
    )
    return row


def compute_global_conservation_metrics(
    model: nn.Module,
    cfg: ShallowWater1DConfig,
    *,
    grid: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> dict[str, float]:
    if grid is None:
        grid = predict_grid(model, cfg)
    x, t, depth, discharge = grid
    flux_mass_left, flux_momentum_left = shallowwater_flux_np(
        depth[:, 0], discharge[:, 0], cfg
    )
    flux_mass_right, flux_momentum_right = shallowwater_flux_np(
        depth[:, -1], discharge[:, -1], cfg
    )

    mass = np.trapz(depth, x, axis=1)
    momentum = np.trapz(discharge, x, axis=1)

    mass_residual = []
    momentum_residual = []
    mass_relative = []
    momentum_relative = []
    for index in range(len(t)):
        integrated_mass_flux = (
            np.trapz(
                flux_mass_right[: index + 1]
                - flux_mass_left[: index + 1],
                t[: index + 1],
            )
            if index > 0
            else 0.0
        )
        integrated_momentum_flux = (
            np.trapz(
                flux_momentum_right[: index + 1]
                - flux_momentum_left[: index + 1],
                t[: index + 1],
            )
            if index > 0
            else 0.0
        )
        mass_value = mass[index] - mass[0] + integrated_mass_flux
        momentum_value = (
            momentum[index] - momentum[0]
            + integrated_momentum_flux
        )
        mass_residual.append(mass_value)
        momentum_residual.append(momentum_value)
        mass_relative.append(
            abs(mass_value)
            / (
                abs(mass[index] - mass[0])
                + abs(integrated_mass_flux)
                + cfg.relative_error_epsilon
            )
        )
        momentum_relative.append(
            abs(momentum_value)
            / (
                abs(momentum[index] - momentum[0])
                + abs(integrated_momentum_flux)
                + cfg.relative_error_epsilon
            )
        )

    mass_residual = np.asarray(mass_residual)
    momentum_residual = np.asarray(momentum_residual)
    mass_relative = np.asarray(mass_relative)
    momentum_relative = np.asarray(momentum_relative)
    return {
        "global_mass_cons_mean_abs": float(
            np.mean(np.abs(mass_residual))
        ),
        "global_mass_cons_final_abs": float(
            abs(mass_residual[-1])
        ),
        "global_mass_cons_mean_rel": float(
            np.mean(mass_relative[1:])
        ),
        "global_mom_cons_mean_abs": float(
            np.mean(np.abs(momentum_residual))
        ),
        "global_mom_cons_final_abs": float(
            abs(momentum_residual[-1])
        ),
        "global_mom_cons_mean_rel": float(
            np.mean(momentum_relative[1:])
        ),
    }


def compute_local_conservation_metrics(
    model: nn.Module,
    cfg: ShallowWater1DConfig,
) -> dict[str, float]:
    rng = np.random.default_rng(cfg.seed + 271828)
    mass_values: list[float] = []
    momentum_values: list[float] = []
    mass_relative: list[float] = []
    momentum_relative: list[float] = []

    for _ in range(cfg.n_control_volumes):
        for _attempt in range(100):
            left, right = np.sort(
                rng.uniform(cfg.x_min, cfg.x_max, size=2)
            )
            t_start, t_stop = np.sort(
                rng.uniform(cfg.t_min, cfg.t_max, size=2)
            )
            if (
                right - left >= cfg.cv_min_width
                and t_stop - t_start >= cfg.cv_min_duration
            ):
                break

        x_quadrature = np.linspace(
            left, right, cfg.cv_quad_nx
        )
        t_quadrature = np.linspace(
            t_start, t_stop, cfg.cv_quad_nt
        )

        start_state = predict_points(
            model,
            x_quadrature,
            np.full_like(x_quadrature, t_start),
        )
        stop_state = predict_points(
            model,
            x_quadrature,
            np.full_like(x_quadrature, t_stop),
        )
        start_depth, start_discharge = (
            start_state[:, 0],
            start_state[:, 1],
        )
        stop_depth, stop_discharge = (
            stop_state[:, 0],
            stop_state[:, 1],
        )
        mass_start = np.trapz(start_depth, x_quadrature)
        mass_stop = np.trapz(stop_depth, x_quadrature)
        momentum_start = np.trapz(start_discharge, x_quadrature)
        momentum_stop = np.trapz(stop_discharge, x_quadrature)

        left_state = predict_points(
            model,
            np.full_like(t_quadrature, left),
            t_quadrature,
        )
        right_state = predict_points(
            model,
            np.full_like(t_quadrature, right),
            t_quadrature,
        )
        left_mass_flux, left_momentum_flux = shallowwater_flux_np(
            left_state[:, 0], left_state[:, 1], cfg
        )
        right_mass_flux, right_momentum_flux = shallowwater_flux_np(
            right_state[:, 0], right_state[:, 1], cfg
        )
        integrated_mass_flux = np.trapz(
            right_mass_flux - left_mass_flux,
            t_quadrature,
        )
        integrated_momentum_flux = np.trapz(
            right_momentum_flux - left_momentum_flux,
            t_quadrature,
        )

        mass_residual = (
            mass_stop - mass_start + integrated_mass_flux
        )
        momentum_residual = (
            momentum_stop - momentum_start
            + integrated_momentum_flux
        )
        mass_values.append(abs(mass_residual))
        momentum_values.append(abs(momentum_residual))
        mass_relative.append(
            abs(mass_residual)
            / (
                abs(mass_stop)
                + abs(mass_start)
                + abs(integrated_mass_flux)
                + cfg.relative_error_epsilon
            )
        )
        momentum_relative.append(
            abs(momentum_residual)
            / (
                abs(momentum_stop)
                + abs(momentum_start)
                + abs(integrated_momentum_flux)
                + cfg.relative_error_epsilon
            )
        )

    return {
        "local_mass_cons_cv_mean_abs": float(
            np.mean(mass_values)
        ),
        "local_mass_cons_cv_median_abs": float(
            np.median(mass_values)
        ),
        "local_mass_cons_cv_mean_rel": float(
            np.mean(mass_relative)
        ),
        "local_mom_cons_cv_mean_abs": float(
            np.mean(momentum_values)
        ),
        "local_mom_cons_cv_median_abs": float(
            np.median(momentum_values)
        ),
        "local_mom_cons_cv_mean_rel": float(
            np.mean(momentum_relative)
        ),
    }


@torch.no_grad()
def compute_gate_diagnostics(
    model: nn.Module,
    cfg: ShallowWater1DConfig,
    *,
    progress: float = 1.0,
) -> dict[str, float]:
    device, dtype = _model_runtime(model)
    h_probe, cmin, normalized = schedule_from_progress(
        progress, cfg
    )
    x = np.linspace(cfg.x_min, cfg.x_max, cfg.eval_nx)
    t = np.linspace(cfg.t_min, cfg.t_max, cfg.eval_nt)
    X, T = np.meshgrid(x, t)
    coordinates = torch.tensor(
        np.stack([X.reshape(-1), T.reshape(-1)], axis=1),
        device=device,
        dtype=dtype,
    )
    h_scale, q_scale = state_scales(cfg)
    gate_result = centered_shallowwater_trace_ratio_gate_1d(
        model,
        coordinates[:, 0:1],
        coordinates[:, 1:2],
        h_probe=h_probe,
        cmin=cmin,
        x_min=cfg.x_min,
        x_max=cfg.x_max,
        h_scale=h_scale,
        q_scale=q_scale,
        beta=cfg.beta,
        norm_epsilon=cfg.trace_norm_epsilon,
        ratio_epsilon=cfg.trace_ratio_epsilon,
        mean_epsilon=cfg.batch_mean_epsilon,
    )

    gate = gate_result.gate.detach().cpu().numpy().reshape(
        cfg.eval_nt, cfg.eval_nx
    )
    ratio = gate_result.ratio.detach().cpu().numpy().reshape(
        cfg.eval_nt, cfg.eval_nx
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
    if not np.any(valid):
        valid = np.ones_like(valid, dtype=bool)

    active = gate > 0.5
    active_loose = gate > 0.1
    return {
        "gate_progress": float(normalized),
        "gate_h": float(h_probe),
        "gate_cmin": float(cmin),
        "gate_mean": float(gate[valid].mean()),
        "gate_max": float(gate[valid].max()),
        "gate_active_frac_gt_0p5": float(
            np.sum(active & valid) / np.sum(valid)
        ),
        "gate_active_frac_gt_0p1": float(
            np.sum(active_loose & valid) / np.sum(valid)
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
    cfg: ShallowWater1DConfig,
) -> dict[str, Any]:
    model.eval()
    grid = predict_grid(model, cfg)
    return {
        "model": method_name,
        "seed": cfg.seed,
        **compute_error_metrics(model, cfg, grid=grid),
        **compute_global_conservation_metrics(
            model, cfg, grid=grid
        ),
        **compute_local_conservation_metrics(model, cfg),
        **compute_gate_diagnostics(model, cfg, progress=1.0),
    }


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
    cfg: ShallowWater1DConfig,
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
    seed_root = root / "1d_shallowwater" / f"seed_{cfg.seed}"
    if seed_root.exists() and any(seed_root.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Reproduction output already exists: {seed_root}. "
            "Use overwrite explicitly."
        )
    seed_root.mkdir(parents=True, exist_ok=True)

    write_json_atomic(
        {
            "equation": "1d_shallowwater",
            "seed": cfg.seed,
            "config": asdict(cfg),
            "reference": REFERENCE_TYPE,
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
    warmup_model, warmup_history = train_warmup(
        warmup_model, cfg
    )
    warmup_seconds = time.time() - warmup_start
    warmup_state = clone_state_dict_cpu(warmup_model)
    continuation_state = capture_rng_state()

    pinn_model = build_model(cfg).to(device=device, dtype=dtype)
    load_state_dict_to_model(pinn_model, warmup_state)
    restore_rng_state(continuation_state)
    pinn_start = time.time()
    pinn_model, pinn_history = train_pinn_continuation(
        pinn_model, cfg
    )
    pinn_seconds = time.time() - pinn_start

    trg_model = build_model(cfg).to(device=device, dtype=dtype)
    load_state_dict_to_model(trg_model, warmup_state)
    restore_rng_state(continuation_state)
    trg_start = time.time()
    trg_model, trg_history = train_trg_continuation(
        trg_model, cfg
    )
    trg_seconds = time.time() - trg_start

    pinn_metrics = evaluate_model(pinn_model, "PINN", cfg)
    trg_metrics = evaluate_model(trg_model, "TRG-PINN", cfg)

    for metrics, continuation_seconds in (
        (pinn_metrics, pinn_seconds),
        (trg_metrics, trg_seconds),
    ):
        metrics.update(
            {
                "equation": "1d_shallowwater",
                "method": metrics["model"],
                "warmup_iters": cfg.warmup_iters,
                "continuation_iters": cfg.gated_iters,
                "adam_total_iters": (
                    cfg.warmup_iters + cfg.gated_iters
                ),
                "optimizer_main": "AdamW",
                "lbfgs_used": False,
                "wall_clock_sec_warmup": warmup_seconds,
                "wall_clock_sec_continuation": (
                    continuation_seconds
                ),
                "wall_clock_sec_total": (
                    warmup_seconds + continuation_seconds
                ),
                "num_parameters": trainable_parameter_count(
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
            "equation": "1d_shallowwater",
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
