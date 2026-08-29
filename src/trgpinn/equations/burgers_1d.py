"""Canonical 1D inviscid Burgers implementation used for the manuscript.

This module is an extraction of the executed notebook implementation. The
sampling order, optimizer/scheduler order, paired RNG restoration, trace-ratio
stabilizers, and evaluation definitions are intentionally preserved.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
import platform
import time
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F

from trgpinn.gate import (
    centered_trace_ratio_gate_1d,
    residual_weight,
    trace_schedule_1d,
    trace_schedule_from_iteration_1d,
)
from trgpinn.losses import composite_loss, normalized_weighted_residual_loss
from trgpinn.metrics import crossing_location
from trgpinn.models import CoordinateMLP, trainable_parameter_count
from trgpinn.sampling import (
    sample_burgers_1d_boundary,
    sample_burgers_1d_initial,
    sample_burgers_1d_interior,
)
from trgpinn.utils import (
    capture_rng_state,
    clone_state_dict_cpu,
    configure_torch_runtime,
    ensure_unprotected_output,
    load_state_dict_to_model,
    resolve_device,
    resolve_dtype,
    restore_rng_state,
    set_seed,
    write_json_atomic,
)


@dataclass
class Burgers1DConfig:
    seed: int = 2026
    device: str = "auto"
    dtype: str = "float32"

    x_min: float = -1.0
    x_max: float = 1.0
    t_min: float = 0.0
    t_max: float = 1.0
    uL: float = 1.0
    uR: float = 0.0
    x0: float = 0.0

    width: int = 128
    depth: int = 6
    activation: str = "tanh"

    warmup_iters: int = 1000
    gated_iters: int = 9000
    lr_warmup: float = 8.0e-4
    lr_gated: float = 3.0e-4
    weight_decay: float = 1.0e-2
    grad_clip: float = 1.0

    n_f: int = 10000
    n_ic: int = 500
    n_bc: int = 500

    w_ic: float = 100.0
    w_bc: float = 1.0
    w_pde: float = 1.0

    h_min_factor: float = 2.0
    h_max_factor: float = 8.0
    cmin_start: float = 0.45
    cmin_end: float = 0.65
    beta: float = 0.05
    residual_floor: float = 0.02

    trace_ratio_epsilon: float = 1.0e-6
    batch_mean_epsilon: float = 1.0e-8
    weighted_loss_epsilon: float = 1.0e-8
    relative_error_epsilon: float = 1.0e-12

    print_every: int = 500
    history_every: int = 50

    eval_nx: int = 1000
    eval_nt: int = 401
    slice_nx: int = 1600
    plot_nx: int = 700
    plot_nt: int = 300

    n_control_volumes: int = 96
    cv_quad_nx: int = 96
    cv_quad_nt: int = 96
    cv_min_width: float = 0.15
    cv_min_duration: float = 0.08

    @classmethod
    def from_legacy_mapping(cls, mapping: dict[str, Any], **overrides: Any) -> "Burgers1DConfig":
        valid = {field.name for field in fields(cls)}
        values = {key: value for key, value in mapping.items() if key in valid}
        values.update(overrides)
        return cls(**values)

    def smoke_copy(self, *, seed: int = 2026, device: str = "auto") -> "Burgers1DConfig":
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
            eval_nt=20,
            n_control_volumes=2,
            cv_quad_nx=12,
            cv_quad_nt=12,
            cv_min_width=0.10,
            cv_min_duration=0.05,
        )


def build_model(cfg: Burgers1DConfig) -> CoordinateMLP:
    return CoordinateMLP(
        input_dimension=2,
        output_dimension=1,
        hidden_width=cfg.width,
        hidden_layers=cfg.depth,
        activation=cfg.activation,
        lower_bounds=(cfg.x_min, cfg.t_min),
        upper_bounds=(cfg.x_max, cfg.t_max),
        output_bias=0.5 * (cfg.uL + cfg.uR),
    )


def cat_xt(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    return torch.cat([x, t], dim=1)


def flux_np(values: np.ndarray) -> np.ndarray:
    return 0.5 * np.asarray(values) ** 2


def flux_torch(values: torch.Tensor) -> torch.Tensor:
    return 0.5 * values.pow(2)


def shock_speed(cfg: Burgers1DConfig) -> float:
    if abs(cfg.uL - cfg.uR) < 1.0e-14:
        return float(cfg.uL)
    return float((0.5 * cfg.uL**2 - 0.5 * cfg.uR**2) / (cfg.uL - cfg.uR))


def shock_location(time_value: np.ndarray | float, cfg: Burgers1DConfig) -> np.ndarray:
    return cfg.x0 + shock_speed(cfg) * np.asarray(time_value)


def exact_solution(
    x_values: np.ndarray,
    t_values: np.ndarray,
    cfg: Burgers1DConfig,
) -> np.ndarray:
    return np.where(
        np.asarray(x_values) < shock_location(np.asarray(t_values), cfg),
        cfg.uL,
        cfg.uR,
    ).astype(np.float64)


def state_scale(cfg: Burgers1DConfig) -> float:
    return max(abs(cfg.uL - cfg.uR), 1.0e-8)


def burgers_residual(
    model: nn.Module,
    x: torch.Tensor,
    t: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    x = x.detach().clone().requires_grad_(True)
    t = t.detach().clone().requires_grad_(True)
    prediction = model(cat_xt(x, t))
    u_t = torch.autograd.grad(
        prediction,
        t,
        grad_outputs=torch.ones_like(prediction),
        create_graph=True,
        retain_graph=True,
    )[0]
    flux = flux_torch(prediction)
    flux_x = torch.autograd.grad(
        flux,
        x,
        grad_outputs=torch.ones_like(flux),
        create_graph=True,
        retain_graph=True,
    )[0]
    return prediction, u_t + flux_x


def _model_runtime(model: nn.Module) -> tuple[torch.device, torch.dtype]:
    parameter = next(model.parameters())
    return parameter.device, parameter.dtype


def pinn_loss(model: nn.Module, cfg: Burgers1DConfig) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    device, dtype = _model_runtime(model)

    x_ic, t_ic, target_ic = sample_burgers_1d_initial(
        cfg.n_ic, cfg, device=device, dtype=dtype
    )
    x_bc, t_bc, target_bc = sample_burgers_1d_boundary(
        cfg.n_bc, cfg, device=device, dtype=dtype
    )
    x_f, t_f = sample_burgers_1d_interior(cfg.n_f, cfg, device=device, dtype=dtype)

    prediction_ic = model(cat_xt(x_ic, t_ic))
    prediction_bc = model(cat_xt(x_bc, t_bc))
    _, residual = burgers_residual(model, x_f, t_f)

    loss_ic = F.mse_loss(prediction_ic, target_ic)
    loss_bc = F.mse_loss(prediction_bc, target_bc)
    loss_pde = residual.pow(2).mean()
    total = composite_loss(
        initial_loss=loss_ic,
        boundary_loss=loss_bc,
        pde_loss=loss_pde,
        initial_weight=cfg.w_ic,
        boundary_weight=cfg.w_bc,
        pde_weight=cfg.w_pde,
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
    cfg: Burgers1DConfig,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    device, dtype = _model_runtime(model)
    h_probe, cmin, progress = trace_schedule_from_iteration_1d(
        iteration, total_iterations, cfg
    )

    x_ic, t_ic, target_ic = sample_burgers_1d_initial(
        cfg.n_ic, cfg, device=device, dtype=dtype
    )
    x_bc, t_bc, target_bc = sample_burgers_1d_boundary(
        cfg.n_bc, cfg, device=device, dtype=dtype
    )
    x_f, t_f = sample_burgers_1d_interior(cfg.n_f, cfg, device=device, dtype=dtype)

    prediction_ic = model(cat_xt(x_ic, t_ic))
    prediction_bc = model(cat_xt(x_bc, t_bc))
    _, residual = burgers_residual(model, x_f, t_f)

    gate = centered_trace_ratio_gate_1d(
        model,
        x_f,
        t_f,
        h_probe=h_probe,
        cmin=cmin,
        x_min=cfg.x_min,
        x_max=cfg.x_max,
        state_scale=state_scale(cfg),
        beta=cfg.beta,
        ratio_epsilon=cfg.trace_ratio_epsilon,
        mean_epsilon=cfg.batch_mean_epsilon,
    )
    weight = residual_weight(gate.gate, cfg.residual_floor)

    loss_ic = F.mse_loss(prediction_ic, target_ic)
    loss_bc = F.mse_loss(prediction_bc, target_bc)
    loss_pde_raw = residual.pow(2).mean()
    loss_pde_weighted = normalized_weighted_residual_loss(
        residual, weight, epsilon=cfg.weighted_loss_epsilon
    )
    total = composite_loss(
        initial_loss=loss_ic,
        boundary_loss=loss_bc,
        pde_loss=loss_pde_weighted,
        initial_weight=cfg.w_ic,
        boundary_weight=cfg.w_bc,
        pde_weight=cfg.w_pde,
    )

    return total, {
        "ic": loss_ic.detach(),
        "bc": loss_bc.detach(),
        "pde": loss_pde_raw.detach(),
        "weighted_pde": loss_pde_weighted.detach(),
        "gate_mean": gate.gate.mean().detach(),
        "gate_max": gate.gate.max().detach(),
        "gate_active_gt_0p5": (gate.gate > 0.5).to(dtype).mean().detach(),
        "gate_active_gt_0p1": (gate.gate > 0.1).to(dtype).mean().detach(),
        "C_mean": gate.ratio.mean().detach(),
        "C_max": gate.ratio.max().detach(),
        "Jhat_mean": gate.normalized_jump.mean().detach(),
        "Jhat_max": gate.normalized_jump.max().detach(),
        "Jbar": gate.jump_mean.detach(),
        "W_mean": weight.mean().detach(),
        "W_min": weight.min().detach(),
        "h": torch.tensor(h_probe, device=device, dtype=dtype),
        "cmin": torch.tensor(cmin, device=device, dtype=dtype),
        "progress": torch.tensor(progress, device=device, dtype=dtype),
        "valid_frac": gate.valid.mean().detach(),
    }


def _train_phase(
    model: nn.Module,
    cfg: Burgers1DConfig,
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
        loss_function = lambda iteration: trg_loss(model, iteration, cfg.gated_iters, cfg)
        total_offset = cfg.warmup_iters
    else:
        raise ValueError(f"Unknown training phase: {phase}")

    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=cfg.weight_decay
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

        if iteration == 1 or iteration % cfg.history_every == 0 or iteration == iterations:
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

        if iteration == 1 or iteration % cfg.print_every == 0 or iteration == iterations:
            print(
                f"[{phase}] {iteration:6d}/{iterations} "
                f"loss={float(loss.detach().cpu()):.3e} "
                f"ic={parts['ic'].item():.1e} "
                f"bc={parts['bc'].item():.1e} "
                f"pde={parts['pde'].item():.1e}"
            )

    return model, pd.DataFrame(history)


def train_warmup(model: nn.Module, cfg: Burgers1DConfig) -> tuple[nn.Module, pd.DataFrame]:
    return _train_phase(model, cfg, phase="warmup")


def train_pinn_continuation(
    model: nn.Module, cfg: Burgers1DConfig
) -> tuple[nn.Module, pd.DataFrame]:
    return _train_phase(model, cfg, phase="pinn")


def train_trg_continuation(
    model: nn.Module, cfg: Burgers1DConfig
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
    x_array = np.asarray(x_values, dtype=np.float64)
    t_array = np.asarray(t_values, dtype=np.float64)
    x_flat = x_array.reshape(-1)
    t_flat = t_array.reshape(-1)
    if x_flat.shape != t_flat.shape:
        raise ValueError("x and t arrays must have the same shape.")

    parameter = next(model.parameters())
    outputs: list[np.ndarray] = []
    for start in range(0, x_flat.size, int(batch_size)):
        stop = min(start + int(batch_size), x_flat.size)
        coordinates = torch.tensor(
            np.stack([x_flat[start:stop], t_flat[start:stop]], axis=1),
            device=parameter.device,
            dtype=parameter.dtype,
        )
        outputs.append(model(coordinates).detach().cpu().numpy().reshape(-1))
    return np.concatenate(outputs).reshape(x_array.shape)


@torch.no_grad()
def predict_grid(
    model: nn.Module,
    cfg: Burgers1DConfig,
    *,
    nx: int | None = None,
    nt: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    nx = cfg.eval_nx if nx is None else int(nx)
    nt = cfg.eval_nt if nt is None else int(nt)
    x = np.linspace(cfg.x_min, cfg.x_max, nx)
    t = np.linspace(cfg.t_min, cfg.t_max, nt)
    X, T = np.meshgrid(x, t)
    prediction = predict_points(model, X, T)
    reference = exact_solution(X, T, cfg)
    return x, t, X, T, prediction, reference


def _shock_width(
    x: np.ndarray,
    values: np.ndarray,
    cfg: Burgers1DConfig,
    time_value: float,
    high_fraction: float,
    low_fraction: float,
) -> float:
    high = cfg.uR + high_fraction * (cfg.uL - cfg.uR)
    low = cfg.uR + low_fraction * (cfg.uL - cfg.uR)
    expected = float(shock_location(time_value, cfg))
    window = max(0.15, 10.0 * (x[1] - x[0]))
    x_high = crossing_location(x, values, high, expected=expected, window=window)
    x_low = crossing_location(
        x, values, low, expected=expected, window=max(0.35, 2.0 * window)
    )
    return abs(x_low - x_high)


def compute_error_and_shock_metrics(
    model: nn.Module,
    cfg: Burgers1DConfig,
    *,
    grid: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> dict[str, float]:
    x, t, _, _, prediction, reference = grid or predict_grid(model, cfg)
    error = prediction - reference

    st_l1 = float(np.mean(np.abs(error)))
    st_l2 = float(np.sqrt(np.mean(error**2)))
    st_rel_l2 = float(
        st_l2 / (np.sqrt(np.mean(reference**2)) + cfg.relative_error_epsilon)
    )

    final_error = error[-1]
    final_reference = reference[-1]
    final_l1 = float(np.mean(np.abs(final_error)))
    final_l2 = float(np.sqrt(np.mean(final_error**2)))
    final_rel_l2 = float(
        final_l2
        / (np.sqrt(np.mean(final_reference**2)) + cfg.relative_error_epsilon)
    )

    midpoint = 0.5 * (cfg.uL + cfg.uR)
    position_errors: list[float] = []
    predicted_positions: list[float] = []
    used_times: list[float] = []
    widths_95_05: list[float] = []
    widths_90_10: list[float] = []

    for index, time_value in enumerate(t):
        if time_value < max(0.02, 2.0 * (t[1] - t[0])):
            continue
        expected = float(shock_location(time_value, cfg))
        row = prediction[index]
        position = crossing_location(
            x, row, midpoint, expected=expected, window=0.35
        )
        predicted_positions.append(position)
        used_times.append(float(time_value))
        position_errors.append(abs(position - expected))
        widths_95_05.append(_shock_width(x, row, cfg, time_value, 0.95, 0.05))
        widths_90_10.append(_shock_width(x, row, cfg, time_value, 0.90, 0.10))

    predicted_positions_array = np.asarray(predicted_positions)
    used_times_array = np.asarray(used_times)
    speed_prediction = (
        float(np.polyfit(used_times_array, predicted_positions_array, 1)[0])
        if len(used_times_array) >= 2
        else np.nan
    )
    speed_exact = shock_speed(cfg)
    final_position_prediction = crossing_location(
        x,
        prediction[-1],
        midpoint,
        expected=float(shock_location(cfg.t_max, cfg)),
        window=0.35,
    )
    final_position_exact = float(shock_location(cfg.t_max, cfg))

    total_variation = np.sum(np.abs(np.diff(prediction, axis=1)), axis=1)
    variation_excess = np.maximum(total_variation - state_scale(cfg), 0.0)
    overshoot = float(max(0.0, np.max(prediction) - max(cfg.uL, cfg.uR)))
    undershoot = float(max(0.0, min(cfg.uL, cfg.uR) - np.min(prediction)))

    return {
        "space_time_l1": st_l1,
        "space_time_l2": st_l2,
        "space_time_rel_l2": st_rel_l2,
        "final_l1": final_l1,
        "final_l2": final_l2,
        "final_rel_l2": final_rel_l2,
        "shock_pos_mae": float(np.mean(position_errors)),
        "shock_pos_maxe": float(np.max(position_errors)),
        "shock_pos_final_error": float(
            abs(final_position_prediction - final_position_exact)
        ),
        "shock_pos_final_pred": float(final_position_prediction),
        "shock_pos_final_exact": float(final_position_exact),
        "shock_speed_pred": speed_prediction,
        "shock_speed_exact": speed_exact,
        "shock_speed_error": float(abs(speed_prediction - speed_exact))
        if not np.isnan(speed_prediction)
        else np.nan,
        "width_95_05_mean": float(np.mean(widths_95_05)),
        "width_95_05_final": float(widths_95_05[-1]),
        "width_90_10_mean": float(np.mean(widths_90_10)),
        "width_90_10_final": float(widths_90_10[-1]),
        "overshoot": overshoot,
        "undershoot": undershoot,
        "min_pred": float(np.min(prediction)),
        "max_pred": float(np.max(prediction)),
        "tv_mean": float(np.mean(total_variation)),
        "tv_excess_mean": float(np.mean(variation_excess)),
        "dx_eval": float(x[1] - x[0]),
    }


def compute_global_conservation_metrics(
    model: nn.Module,
    cfg: Burgers1DConfig,
    *,
    grid: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> dict[str, float]:
    x, t, _, _, prediction, reference = grid or predict_grid(model, cfg)
    mass = np.trapz(prediction, x, axis=1)
    mass_reference = np.trapz(reference, x, axis=1)
    flux_difference = flux_np(prediction[:, -1]) - flux_np(prediction[:, 0])

    defects: list[float] = []
    relative_defects: list[float] = []
    for index in range(len(t)):
        integrated_flux = (
            np.trapz(flux_difference[: index + 1], t[: index + 1]) if index > 0 else 0.0
        )
        value = mass[index] - mass[0] + integrated_flux
        defects.append(value)
        denominator = abs(mass[index] - mass[0]) + abs(integrated_flux) + 1.0e-12
        relative_defects.append(abs(value) / denominator)

    defects_array = np.asarray(defects)
    relative_array = np.asarray(relative_defects)
    mass_error = mass - mass_reference
    return {
        "global_cons_mean_abs": float(np.mean(np.abs(defects_array))),
        "global_cons_max_abs": float(np.max(np.abs(defects_array))),
        "global_cons_final_abs": float(abs(defects_array[-1])),
        "global_cons_mean_rel": float(np.mean(relative_array[1:])),
        "mass_error_mean_abs": float(np.mean(np.abs(mass_error))),
        "mass_error_final_abs": float(abs(mass_error[-1])),
    }


def compute_local_conservation_metrics(model: nn.Module, cfg: Burgers1DConfig) -> dict[str, float]:
    rng = np.random.default_rng(cfg.seed + 314159)
    absolute_values: list[float] = []
    relative_values: list[float] = []

    for _ in range(cfg.n_control_volumes):
        for _attempt in range(100):
            left, right = np.sort(rng.uniform(cfg.x_min, cfg.x_max, size=2))
            time_left, time_right = np.sort(rng.uniform(cfg.t_min, cfg.t_max, size=2))
            if (
                right - left >= cfg.cv_min_width
                and time_right - time_left >= cfg.cv_min_duration
            ):
                break

        x_quadrature = np.linspace(left, right, cfg.cv_quad_nx)
        t_quadrature = np.linspace(time_left, time_right, cfg.cv_quad_nt)
        u_t1 = predict_points(model, x_quadrature, np.full_like(x_quadrature, time_left))
        u_t2 = predict_points(model, x_quadrature, np.full_like(x_quadrature, time_right))
        mass_t1 = np.trapz(u_t1, x_quadrature)
        mass_t2 = np.trapz(u_t2, x_quadrature)
        u_left = predict_points(model, np.full_like(t_quadrature, left), t_quadrature)
        u_right = predict_points(model, np.full_like(t_quadrature, right), t_quadrature)
        integrated_flux = np.trapz(flux_np(u_right) - flux_np(u_left), t_quadrature)
        defect = mass_t2 - mass_t1 + integrated_flux
        denominator = abs(mass_t2) + abs(mass_t1) + abs(integrated_flux) + 1.0e-12
        absolute_values.append(abs(defect))
        relative_values.append(abs(defect) / denominator)

    absolute_array = np.asarray(absolute_values)
    relative_array = np.asarray(relative_values)
    return {
        "local_cons_cv_mean_abs": float(np.mean(absolute_array)),
        "local_cons_cv_median_abs": float(np.median(absolute_array)),
        "local_cons_cv_max_abs": float(np.max(absolute_array)),
        "local_cons_cv_mean_rel": float(np.mean(relative_array)),
        "local_cons_cv_max_rel": float(np.max(relative_array)),
    }


@torch.no_grad()
def compute_gate_diagnostics(
    model: nn.Module,
    cfg: Burgers1DConfig,
    *,
    progress: float = 1.0,
) -> dict[str, float]:
    h_probe, cmin, normalized_progress = trace_schedule_1d(progress, cfg)
    x = np.linspace(cfg.x_min, cfg.x_max, cfg.eval_nx)
    t = np.linspace(cfg.t_min, cfg.t_max, cfg.eval_nt)
    X, T = np.meshgrid(x, t)

    parameter = next(model.parameters())
    coordinates = torch.tensor(
        np.stack([X.reshape(-1), T.reshape(-1)], axis=1),
        device=parameter.device,
        dtype=parameter.dtype,
    )
    gate = centered_trace_ratio_gate_1d(
        model,
        coordinates[:, 0:1],
        coordinates[:, 1:2],
        h_probe=h_probe,
        cmin=cmin,
        x_min=cfg.x_min,
        x_max=cfg.x_max,
        state_scale=state_scale(cfg),
        beta=cfg.beta,
        ratio_epsilon=cfg.trace_ratio_epsilon,
        mean_epsilon=cfg.batch_mean_epsilon,
    )

    G = gate.gate.detach().cpu().numpy().reshape(cfg.eval_nt, cfg.eval_nx)
    C = gate.ratio.detach().cpu().numpy().reshape(cfg.eval_nt, cfg.eval_nx)
    Jhat = gate.normalized_jump.detach().cpu().numpy().reshape(cfg.eval_nt, cfg.eval_nx)
    valid = gate.valid.detach().cpu().numpy().reshape(cfg.eval_nt, cfg.eval_nx).astype(bool)
    if valid.sum() == 0:
        valid = np.ones_like(valid, dtype=bool)

    shock_band = np.abs(X - shock_location(T, cfg)) <= 2.0 * h_probe
    active = G > 0.5
    active_loose = G > 0.1
    precision = (
        float((active & shock_band & valid).sum() / (active & valid).sum())
        if (active & valid).sum() > 0
        else np.nan
    )
    recall = (
        float((active & shock_band & valid).sum() / (shock_band & valid).sum())
        if (shock_band & valid).sum() > 0
        else np.nan
    )

    return {
        "gate_progress": float(normalized_progress),
        "gate_h": float(h_probe),
        "gate_cmin": float(cmin),
        "gate_mean": float(G[valid].mean()),
        "gate_max": float(G[valid].max()),
        "gate_active_frac_gt_0p5": float((active & valid).sum() / valid.sum()),
        "gate_active_frac_gt_0p1": float((active_loose & valid).sum() / valid.sum()),
        "gate_C_mean": float(C[valid].mean()),
        "gate_C_max": float(C[valid].max()),
        "gate_Jhat_mean": float(Jhat[valid].mean()),
        "gate_Jhat_max": float(Jhat[valid].max()),
        "gate_Jbar": float(gate.jump_mean.detach().cpu()),
        "gate_precision_band_2h": precision,
        "gate_recall_band_2h": recall,
    }


def evaluate_model(model: nn.Module, method_name: str, cfg: Burgers1DConfig) -> dict[str, Any]:
    model.eval()
    grid = predict_grid(model, cfg)
    row: dict[str, Any] = {
        "model": method_name,
        "seed": cfg.seed,
        **compute_error_and_shock_metrics(model, cfg, grid=grid),
        **compute_global_conservation_metrics(model, cfg, grid=grid),
        **compute_local_conservation_metrics(model, cfg),
        **compute_gate_diagnostics(model, cfg, progress=1.0),
        "xshock_tmax_exact": float(shock_location(cfg.t_max, cfg)),
    }
    return row


def _save_checkpoint(model: nn.Module, path: Path, extra: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "extra": extra}, path)


def run_paired_experiment(
    cfg: Burgers1DConfig,
    *,
    output_root: str | Path,
    protected_reported_root: str | Path,
    overwrite: bool = False,
) -> pd.DataFrame:
    """Run the canonical shared-warmup paired experiment.

    Output is always written outside ``artifacts/reported``. This function is
    used for fresh reproduction runs and never mutates manuscript artifacts.
    """

    device = resolve_device(cfg.device)
    dtype = configure_torch_runtime(cfg.dtype)
    set_seed(cfg.seed)

    root = ensure_unprotected_output(output_root, protected_reported_root)
    seed_root = root / "1d_burgers" / f"seed_{cfg.seed}"
    if seed_root.exists() and any(seed_root.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Reproduction output already exists: {seed_root}. Use overwrite explicitly."
        )
    seed_root.mkdir(parents=True, exist_ok=True)

    write_json_atomic(
        {
            "equation": "1d_burgers",
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
                "equation": "1d_burgers",
                "method": metrics["model"],
                "warmup_iters": cfg.warmup_iters,
                "continuation_iters": cfg.gated_iters,
                "adam_total_iters": cfg.warmup_iters + cfg.gated_iters,
                "optimizer_main": "AdamW",
                "lbfgs_used": False,
                "wall_clock_sec_warmup": warmup_seconds,
                "wall_clock_sec_continuation": continuation_seconds,
                "wall_clock_sec_total": warmup_seconds + continuation_seconds,
                "num_parameters": trainable_parameter_count(pinn_model),
            }
        )

    warmup_history.to_csv(seed_root / "history_warmup.csv", index=False)
    pinn_history.to_csv(seed_root / "history_pinn.csv", index=False)
    trg_history.to_csv(seed_root / "history_trg_pinn.csv", index=False)
    pd.DataFrame([pinn_metrics, trg_metrics]).to_csv(seed_root / "metrics_final.csv", index=False)
    write_json_atomic(pinn_metrics, seed_root / "metrics_pinn.json")
    write_json_atomic(trg_metrics, seed_root / "metrics_trg_pinn.json")
    torch.save(
        {
            "model_state_dict": warmup_state,
            "equation": "1d_burgers",
            "method": "shared_warmup",
            "seed": cfg.seed,
        },
        seed_root / "model_warmup.pt",
    )
    _save_checkpoint(pinn_model, seed_root / "model_pinn_final.pt", pinn_metrics)
    _save_checkpoint(trg_model, seed_root / "model_trg_pinn_final.pt", trg_metrics)
    (seed_root / "_SUCCESS").write_text("success\n", encoding="utf-8")
    return pd.DataFrame([pinn_metrics, trg_metrics])
