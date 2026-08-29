# ============================================================
# Paper-ready paired multi-seed Trace-Ratio Gated PINN
# 2D shallow-water equations: circular wet-bed dam-break benchmark
#
# PDE, conservative form:
#   U_t + F(U)_x + G(U)_y = 0,
#   U = [h, m, n]^T = [h, h*u, h*v]^T,
#   F = [m, m^2/h + g*h^2/2, m*n/h]^T,
#   G = [n, m*n/h, n^2/h + g*h^2/2]^T.
#
# Circular wet-bed dam break:
#   r = sqrt((x-xc)^2 + (y-yc)^2)
#   U(x,y,0) = (h_in,0,0) for r<R0, else (h_out,0,0).
#   The fixed exterior state is imposed on all four boundaries.
#
# Network output: conservative variables W=(h,m,n), with h>0 enforced
# by a softplus floor. Residual is computed in conservative form.
#
# Paper protocol:
#   - paired PINN vs tPINN comparison
#   - shared vanilla warm-up checkpoint
#   - same RNG state restored before PINN and tPINN continuation
#   - no validation set, no early stopping, no best-checkpoint selection
#   - final checkpoint evaluation only
#   - 2D finite-volume reference is used only for evaluation/plotting
# ============================================================

from __future__ import annotations

import copy
import json
import math
import platform
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

try:
    from IPython.display import display
except Exception:
    display = print


# ============================================================
# 1. Configuration
# ============================================================

@dataclass
class ShallowWater2DConfig:
    # Runtime / reproducibility
    seed: int = 1234
    device: str = "cuda:4"  # use "auto", "cpu", or another CUDA device if needed
    dtype: str = "float32"
    output_dir: str = "runs_shallow_water2d_trace_ratio_paper"
    experiment_name: str = "shallow_water2d_circular_dambreak_ring_trace_ratio_locked"
    save_outputs: bool = True

    # Circular wet-bed dam-break benchmark
    x_min: float = -1.0
    x_max: float = 1.0
    y_min: float = -1.0
    y_max: float = 1.0
    t_min: float = 0.0
    t_max: float = 0.20
    g_const: float = 1.0
    center_x: float = 0.0
    center_y: float = 0.0
    dam_radius: float = 0.35
    h_inside: float = 2.0
    h_outside: float = 1.0
    m_inside: float = 0.0
    n_inside: float = 0.0
    m_outside: float = 0.0
    n_outside: float = 0.0

    # Network: same width/depth as both 1D SWE and 2D Euler
    width: int = 128
    depth: int = 6
    activation: str = "tanh"
    h_floor: float = 1.0e-5

    # Training: 2D-system budget inherited from 2D Euler
    warmup_iters: int = 3000
    gated_iters: int = 7000
    lr_warmup: float = 8.0e-4
    lr_gated: float = 3.0e-4
    grad_clip: float = 1.0


    # 2D batch budget inherited from 2D Euler
    n_f: int = 20000
    n_ic: int = 6000
    n_bc: int = 2500

    # SWE-specific loss weights inherited from 1D SWE
    w_ic: float = 100.0
    w_bc: float = 30.0
    w_pde: float = 1.0

    # 2D ring trace-ratio detector inherited from 2D Euler
    ring_trace_pairs: int = 6
    gate_tau: float = 1.0
    h_max_factor: float = 5.0
    h_min_factor: float = 2.0
    cmin_start: float = 0.50
    cmin_end: float = 0.70
    beta: float = 0.05
    residual_floor: float = 0.02

    # Logging
    print_every: int = 500
    history_every: int = 50

    # Neural-model evaluation grids
    eval_nxy: int = 220            # final-time spatial grid / stored FV grid
    eval_space_nxy: int = 90       # space-time error grid in x-y
    eval_nt: int = 41              # space-time error time levels
    line_n: int = 1600             # radial-line evaluation
    radial_angles: int = 16        # front metric angles
    gate_eval_nxy: int = 180

    # Finite-volume reference. Use 512 for the paper run; verify with a finer grid.
    fv_nxy: int = 512
    fv_cfl: float = 0.35
    fv_order: int = 2
    fv_depth_floor: float = 1.0e-10
    reference_cache: bool = True

    # Global/local conservation diagnostics
    cons_nxy: int = 70
    cons_nt: int = 31
    n_control_volumes: int = 24
    cv_quad_nxy: int = 32
    cv_quad_nt: int = 32
    cv_min_width: float = 0.25
    cv_min_duration: float = 0.04


cfg = ShallowWater2DConfig()
DEVICE: torch.device
DTYPE: torch.dtype


# ============================================================
# 2. Runtime and reproducibility
# ============================================================

def resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    if device_str.startswith("cuda:2") and not torch.cuda.is_available():
        print("CUDA was requested but is unavailable. Falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device_str)


def resolve_dtype(dtype_str: str) -> torch.dtype:
    if dtype_str == "float32":
        return torch.float32
    if dtype_str == "float64":
        return torch.float64
    raise ValueError(f"Unsupported dtype: {dtype_str}")


def configure_runtime(new_cfg: ShallowWater2DConfig) -> Tuple[torch.device, torch.dtype]:
    global cfg, DEVICE, DTYPE
    cfg = new_cfg
    DEVICE = resolve_device(cfg.device)
    DTYPE = resolve_dtype(cfg.dtype)
    torch.set_default_dtype(DTYPE)
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
    return DEVICE, DTYPE


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_rng_state() -> Dict[str, object]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def set_rng_state(state: Dict[str, object]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state["torch_cuda"] is not None:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


def cat_xyt(x: torch.Tensor, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    return torch.cat([x, y, t], dim=1)


def clone_state(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def make_run_dir(c: ShallowWater2DConfig) -> Path:
    path = Path(c.output_dir) / c.experiment_name / f"seed_{c.seed}"
    if c.save_outputs:
        path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(obj: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


configure_runtime(cfg)


# ============================================================
# 3. Physical scales, fluxes and detector schedule
# ============================================================

def h_scale(c: ShallowWater2DConfig) -> float:
    return max(c.h_inside, c.h_outside, 1.0)


def c_scale(c: ShallowWater2DConfig) -> float:
    return math.sqrt(c.g_const * h_scale(c))


def q_scale(c: ShallowWater2DConfig) -> float:
    return max(
        h_scale(c) * c_scale(c),
        abs(c.m_inside), abs(c.n_inside),
        abs(c.m_outside), abs(c.n_outside),
        1.0,
    )


def h0(c: ShallowWater2DConfig) -> float:
    area = (c.x_max - c.x_min) * (c.y_max - c.y_min)
    return math.sqrt(area) / math.sqrt(c.n_f)


def schedule_from_progress(progress: float, c: ShallowWater2DConfig) -> Tuple[float, float, float]:
    p = float(np.clip(progress, 0.0, 1.0))
    base = h0(c)
    probe = base * (
        c.h_min_factor
        + (c.h_max_factor - c.h_min_factor) * (1.0 - p) ** 2
    )
    cmin = c.cmin_start + (c.cmin_end - c.cmin_start) * p
    return probe, cmin, p


def schedule_from_iter(iteration: int, total_iters: int, c: ShallowWater2DConfig):
    return schedule_from_progress(iteration / max(1, total_iters), c)


def flux_np(U: np.ndarray, c: ShallowWater2DConfig) -> Tuple[np.ndarray, np.ndarray]:
    h = np.maximum(U[0], c.fv_depth_floor)
    m = U[1]
    n = U[2]
    u = m / h
    v = n / h
    Fv = np.stack([
        m,
        m * u + 0.5 * c.g_const * h * h,
        m * v,
    ], axis=0)
    Gv = np.stack([
        n,
        n * u,
        n * v + 0.5 * c.g_const * h * h,
    ], axis=0)
    return Fv, Gv


def flux_torch(h: torch.Tensor, m: torch.Tensor, n: torch.Tensor, c: ShallowWater2DConfig):
    hs = torch.clamp(h, min=1.0e-8)
    u = m / hs
    v = n / hs
    Fv = (
        m,
        m * u + 0.5 * c.g_const * hs.pow(2),
        m * v,
    )
    Gv = (
        n,
        n * u,
        n * v + 0.5 * c.g_const * hs.pow(2),
    )
    return Fv, Gv


# ============================================================
# 4. Neural network
# ============================================================

def inv_softplus(y: float) -> float:
    y = max(float(y), 1.0e-10)
    return math.log(math.expm1(y))


class MLP(nn.Module):
    """Input (x,y,t); output conservative SWE state (h,m=hu,n=hv)."""

    def __init__(self, c: ShallowWater2DConfig):
        super().__init__()
        self.cfg = c
        self.h_floor = c.h_floor

        if c.activation.lower() == "tanh":
            act = nn.Tanh
        elif c.activation.lower() == "silu":
            act = nn.SiLU
        else:
            raise ValueError(f"Unsupported activation: {c.activation}")

        layers: List[nn.Module] = []
        dim = 3
        for _ in range(c.depth):
            layer = nn.Linear(dim, c.width)
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)
            layers += [layer, act()]
            dim = c.width
        self.trunk = nn.Sequential(*layers)
        self.out = nn.Linear(dim, 3)
        nn.init.xavier_normal_(self.out.weight)
        nn.init.zeros_(self.out.bias)

        h_mean = 0.5 * (c.h_inside + c.h_outside)
        m_mean = 0.5 * (c.m_inside + c.m_outside)
        n_mean = 0.5 * (c.n_inside + c.n_outside)
        with torch.no_grad():
            self.out.bias[0].fill_(inv_softplus(h_mean - c.h_floor))
            self.out.bias[1].fill_(m_mean)
            self.out.bias[2].fill_(n_mean)

    def forward(self, xyt: torch.Tensor) -> torch.Tensor:
        x = xyt[:, 0:1]
        y = xyt[:, 1:2]
        t = xyt[:, 2:3]
        xh = 2.0 * (x - self.cfg.x_min) / (self.cfg.x_max - self.cfg.x_min) - 1.0
        yh = 2.0 * (y - self.cfg.y_min) / (self.cfg.y_max - self.cfg.y_min) - 1.0
        th = 2.0 * (t - self.cfg.t_min) / (self.cfg.t_max - self.cfg.t_min) - 1.0
        raw = self.out(self.trunk(torch.cat([xh, yh, th], dim=1)))
        h = self.h_floor + F.softplus(raw[:, 0:1])
        m = raw[:, 1:2]
        n = raw[:, 2:3]
        return torch.cat([h, m, n], dim=1)


# ============================================================
# 5. Sampling
# ============================================================

def state_inside(npts: int, c: ShallowWater2DConfig) -> torch.Tensor:
    return torch.cat([
        torch.full((npts, 1), c.h_inside, device=DEVICE, dtype=DTYPE),
        torch.full((npts, 1), c.m_inside, device=DEVICE, dtype=DTYPE),
        torch.full((npts, 1), c.n_inside, device=DEVICE, dtype=DTYPE),
    ], dim=1)


def state_outside(npts: int, c: ShallowWater2DConfig) -> torch.Tensor:
    return torch.cat([
        torch.full((npts, 1), c.h_outside, device=DEVICE, dtype=DTYPE),
        torch.full((npts, 1), c.m_outside, device=DEVICE, dtype=DTYPE),
        torch.full((npts, 1), c.n_outside, device=DEVICE, dtype=DTYPE),
    ], dim=1)


def sample_uniform_disk(npts: int, c: ShallowWater2DConfig) -> Tuple[torch.Tensor, torch.Tensor]:
    rr = c.dam_radius * torch.sqrt(torch.rand(npts, 1, device=DEVICE, dtype=DTYPE))
    phi = 2.0 * math.pi * torch.rand(npts, 1, device=DEVICE, dtype=DTYPE)
    return c.center_x + rr * torch.cos(phi), c.center_y + rr * torch.sin(phi)


def sample_uniform_outside_disk(npts: int, c: ShallowWater2DConfig) -> Tuple[torch.Tensor, torch.Tensor]:
    xs: List[torch.Tensor] = []
    ys: List[torch.Tensor] = []
    remaining = npts
    while remaining > 0:
        batch = max(1024, 2 * remaining)
        x = c.x_min + (c.x_max - c.x_min) * torch.rand(batch, 1, device=DEVICE, dtype=DTYPE)
        y = c.y_min + (c.y_max - c.y_min) * torch.rand(batch, 1, device=DEVICE, dtype=DTYPE)
        mask = ((x - c.center_x).pow(2) + (y - c.center_y).pow(2) >= c.dam_radius**2).reshape(-1)
        x = x[mask]
        y = y[mask]
        take = min(remaining, x.shape[0])
        if take > 0:
            xs.append(x[:take])
            ys.append(y[:take])
            remaining -= take
    return torch.cat(xs, dim=0), torch.cat(ys, dim=0)


def sample_ic(npts: int, c: ShallowWater2DConfig):
    # Balanced inside/outside sampling, mirroring the balanced 1D Riemann IC protocol.
    ni = npts // 2
    no = npts - ni
    xi, yi = sample_uniform_disk(ni, c)
    xo, yo = sample_uniform_outside_disk(no, c)
    x = torch.cat([xi, xo], dim=0)
    y = torch.cat([yi, yo], dim=0)
    t = torch.zeros_like(x)
    target = torch.cat([state_inside(ni, c), state_outside(no, c)], dim=0)
    idx = torch.randperm(npts, device=DEVICE)
    return x[idx], y[idx], t[idx], target[idx]


def sample_bc(npts: int, c: ShallowWater2DConfig):
    counts = [npts // 4, npts // 4, npts // 4, npts - 3 * (npts // 4)]
    xs: List[torch.Tensor] = []
    ys: List[torch.Tensor] = []
    ts: List[torch.Tensor] = []
    for side, count in enumerate(counts):
        tt = c.t_min + (c.t_max - c.t_min) * torch.rand(count, 1, device=DEVICE, dtype=DTYPE)
        if side == 0:
            xx = torch.full((count, 1), c.x_min, device=DEVICE, dtype=DTYPE)
            yy = c.y_min + (c.y_max - c.y_min) * torch.rand(count, 1, device=DEVICE, dtype=DTYPE)
        elif side == 1:
            xx = torch.full((count, 1), c.x_max, device=DEVICE, dtype=DTYPE)
            yy = c.y_min + (c.y_max - c.y_min) * torch.rand(count, 1, device=DEVICE, dtype=DTYPE)
        elif side == 2:
            xx = c.x_min + (c.x_max - c.x_min) * torch.rand(count, 1, device=DEVICE, dtype=DTYPE)
            yy = torch.full((count, 1), c.y_min, device=DEVICE, dtype=DTYPE)
        else:
            xx = c.x_min + (c.x_max - c.x_min) * torch.rand(count, 1, device=DEVICE, dtype=DTYPE)
            yy = torch.full((count, 1), c.y_max, device=DEVICE, dtype=DTYPE)
        xs.append(xx); ys.append(yy); ts.append(tt)
    x = torch.cat(xs, dim=0)
    y = torch.cat(ys, dim=0)
    t = torch.cat(ts, dim=0)
    target = state_outside(x.shape[0], c)
    idx = torch.randperm(x.shape[0], device=DEVICE)
    return x[idx], y[idx], t[idx], target[idx]


def sample_f(npts: int, c: ShallowWater2DConfig):
    x = c.x_min + (c.x_max - c.x_min) * torch.rand(npts, 1, device=DEVICE, dtype=DTYPE)
    y = c.y_min + (c.y_max - c.y_min) * torch.rand(npts, 1, device=DEVICE, dtype=DTYPE)
    t = c.t_min + (c.t_max - c.t_min) * torch.rand(npts, 1, device=DEVICE, dtype=DTYPE)
    return x, y, t


# ============================================================
# 6. Residual, directional gate and losses
# ============================================================

def scaled_state_mse(pred: torch.Tensor, target: torch.Tensor, c: ShallowWater2DConfig) -> torch.Tensor:
    hs = h_scale(c)
    qs = q_scale(c)
    return (
        ((pred[:, 0:1] - target[:, 0:1]) / hs).pow(2)
        + ((pred[:, 1:2] - target[:, 1:2]) / qs).pow(2)
        + ((pred[:, 2:3] - target[:, 2:3]) / qs).pow(2)
    ).mean()


def swe2d_residual(model: nn.Module, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor, c: ShallowWater2DConfig):
    x = x.detach().clone().requires_grad_(True)
    y = y.detach().clone().requires_grad_(True)
    t = t.detach().clone().requires_grad_(True)
    W = model(cat_xyt(x, y, t))
    h, m, n = W[:, 0:1], W[:, 1:2], W[:, 2:3]
    Fv, Gv = flux_torch(h, m, n, c)
    U = (h, m, n)
    residuals = []
    for Uk, Fk, Gk in zip(U, Fv, Gv):
        Uk_t = torch.autograd.grad(Uk, t, torch.ones_like(Uk), create_graph=True, retain_graph=True)[0]
        Fk_x = torch.autograd.grad(Fk, x, torch.ones_like(Fk), create_graph=True, retain_graph=True)[0]
        Gk_y = torch.autograd.grad(Gk, y, torch.ones_like(Gk), create_graph=True, retain_graph=True)[0]
        residuals.append(Uk_t + Fk_x + Gk_y)
    r_h, r_m, r_n = residuals
    hs = h_scale(c)
    qs = q_scale(c)
    R2 = (r_h / hs).pow(2) + (r_m / qs).pow(2) + (r_n / qs).pow(2)
    return W, r_h, r_m, r_n, R2


def get_ring_directions(c: ShallowWater2DConfig) -> torch.Tensor:
    theta = torch.arange(c.ring_trace_pairs, device=DEVICE, dtype=DTYPE) * (math.pi / c.ring_trace_pairs)
    return torch.stack([torch.cos(theta), torch.sin(theta)], dim=1)


@torch.no_grad()
def trace_ratio_gate_conservative(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    t: torch.Tensor,
    probe: float,
    cmin: float,
    c: ShallowWater2DConfig,
):
    npts = x.shape[0]
    dirs = get_ring_directions(c)
    ndir = dirs.shape[0]
    dx = dirs[:, 0].view(1, ndir)
    dy = dirs[:, 1].view(1, ndir)
    x0 = x.view(npts, 1)
    y0 = y.view(npts, 1)
    t0 = t.view(npts, 1)

    def coords(scale: float):
        xp = x0 + scale * dx
        xm = x0 - scale * dx
        yp = y0 + scale * dy
        ym = y0 - scale * dy
        return xp, xm, yp, ym

    xph, xmh, yph, ymh = coords(probe)
    xp2, xm2, yp2, ym2 = coords(2.0 * probe)
    valid = (
        (xm2 >= c.x_min) & (xm2 <= c.x_max)
        & (xp2 >= c.x_min) & (xp2 <= c.x_max)
        & (ym2 >= c.y_min) & (ym2 <= c.y_max)
        & (yp2 >= c.y_min) & (yp2 <= c.y_max)
    ).to(DTYPE)

    def clamp_xy(xx: torch.Tensor, yy: torch.Tensor):
        return xx.clamp(c.x_min, c.x_max), yy.clamp(c.y_min, c.y_max)

    xph, yph = clamp_xy(xph, yph)
    xmh, ymh = clamp_xy(xmh, ymh)
    xp2, yp2 = clamp_xy(xp2, yp2)
    xm2, ym2 = clamp_xy(xm2, ym2)
    tt = t0.repeat(1, ndir)

    def eval_state(xx: torch.Tensor, yy: torch.Tensor) -> torch.Tensor:
        pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), tt.reshape(-1, 1)], dim=1)
        return model(pts).reshape(npts, ndir, 3)

    Wph = eval_state(xph, yph)
    Wmh = eval_state(xmh, ymh)
    Wp2 = eval_state(xp2, yp2)
    Wm2 = eval_state(xm2, ym2)

    hs = h_scale(c)
    qs = q_scale(c)

    def scaled_jump(Wp: torch.Tensor, Wm: torch.Tensor) -> torch.Tensor:
        d = Wp - Wm
        return torch.sqrt(
            (d[:, :, 0] / hs).pow(2)
            + (d[:, :, 1] / qs).pow(2)
            + (d[:, :, 2] / qs).pow(2)
            + 1.0e-12
        )

    J1 = scaled_jump(Wph, Wmh)
    J2 = scaled_jump(Wp2, Wm2)
    C = torch.clamp(J1 / (J2 + 1.0e-8), 0.0, 2.0)
    valid_sum = valid.sum()
    Jbar_valid = (J1 * valid).sum() / (valid_sum + 1.0e-8)
    Jbar_all = J1.mean()
    Jbar = torch.where(valid_sum > 0, Jbar_valid, Jbar_all).clamp_min(1.0e-8)
    Jhat = J1 / Jbar

    g_jump = torch.sigmoid((Jhat - c.gate_tau) / c.beta)
    g_ratio = torch.sigmoid((C - cmin) / c.beta)
    g_dir = g_jump * g_ratio * valid
    gate = g_dir.max(dim=1, keepdim=True).values
    Cmax = (C * valid).max(dim=1, keepdim=True).values
    Jmax = (Jhat * valid).max(dim=1, keepdim=True).values
    return gate, C, Jhat, Jbar, valid, Cmax, Jmax


def vanilla_loss(model: nn.Module, c: ShallowWater2DConfig):
    x_ic, y_ic, t_ic, target_ic = sample_ic(c.n_ic, c)
    x_bc, y_bc, t_bc, target_bc = sample_bc(c.n_bc, c)
    x_f, y_f, t_f = sample_f(c.n_f, c)
    W_ic = model(cat_xyt(x_ic, y_ic, t_ic))
    W_bc = model(cat_xyt(x_bc, y_bc, t_bc))
    _, _, _, _, R2 = swe2d_residual(model, x_f, y_f, t_f, c)
    L_ic = scaled_state_mse(W_ic, target_ic, c)
    L_bc = scaled_state_mse(W_bc, target_bc, c)
    L_pde = R2.mean()
    loss = c.w_ic * L_ic + c.w_bc * L_bc + c.w_pde * L_pde
    return loss, {"ic": L_ic.detach(), "bc": L_bc.detach(), "pde": L_pde.detach(), "weighted_pde": L_pde.detach()}


def trace_ratio_gated_loss(model: nn.Module, iteration: int, total_iters: int, c: ShallowWater2DConfig):
    probe, cmin, progress = schedule_from_iter(iteration, total_iters, c)
    x_ic, y_ic, t_ic, target_ic = sample_ic(c.n_ic, c)
    x_bc, y_bc, t_bc, target_bc = sample_bc(c.n_bc, c)
    x_f, y_f, t_f = sample_f(c.n_f, c)
    W_ic = model(cat_xyt(x_ic, y_ic, t_ic))
    W_bc = model(cat_xyt(x_bc, y_bc, t_bc))
    _, _, _, _, R2 = swe2d_residual(model, x_f, y_f, t_f, c)
    gate, C, Jhat, Jbar, valid, Cmax, Jmax = trace_ratio_gate_conservative(model, x_f, y_f, t_f, probe, cmin, c)
    weight = c.residual_floor + (1.0 - c.residual_floor) * (1.0 - gate)
    L_ic = scaled_state_mse(W_ic, target_ic, c)
    L_bc = scaled_state_mse(W_bc, target_bc, c)
    L_raw = R2.mean()
    L_weighted = (weight * R2).sum() / (weight.sum() + 1.0e-8)
    loss = c.w_ic * L_ic + c.w_bc * L_bc + c.w_pde * L_weighted
    valid_mask = valid > 0
    Cmean = C[valid_mask].mean() if torch.any(valid_mask) else C.mean()
    Jmean = Jhat[valid_mask].mean() if torch.any(valid_mask) else Jhat.mean()
    return loss, {
        "ic": L_ic.detach(), "bc": L_bc.detach(), "pde": L_raw.detach(),
        "weighted_pde": L_weighted.detach(),
        "gate_mean": gate.mean().detach(), "gate_max": gate.max().detach(),
        "gate_active_gt_0p5": (gate > 0.5).to(DTYPE).mean().detach(),
        "gate_active_gt_0p1": (gate > 0.1).to(DTYPE).mean().detach(),
        "C_mean": Cmean.detach(), "C_max": Cmax.max().detach(),
        "Jhat_mean": Jmean.detach(), "Jhat_max": Jmax.max().detach(),
        "Jbar": Jbar.detach(), "W_mean": weight.mean().detach(), "W_min": weight.min().detach(),
        "h": torch.tensor(probe, device=DEVICE, dtype=DTYPE),
        "cmin": torch.tensor(cmin, device=DEVICE, dtype=DTYPE),
        "progress": torch.tensor(progress, device=DEVICE, dtype=DTYPE),
        "valid_frac": valid.mean().detach(),
    }


# ============================================================
# 7. Training loops
# ============================================================

def train_warmup_fixed(model, cfg: ShallowWater2DConfig):
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr_warmup)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=cfg.warmup_iters, eta_min=cfg.lr_warmup * 0.05
    )
    history = []
    print("\n[Phase 1] Vanilla PINN warm-up")
    print("Protocol: no validation, no early stopping, no best checkpoint; final warm-up state is used.")

    for it in range(1, cfg.warmup_iters + 1):
        opt.zero_grad(set_to_none=True)
        loss, parts = vanilla_loss(model, cfg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step(); sch.step()

        if it == 1 or it % cfg.history_every == 0 or it == cfg.warmup_iters:
            history.append({
                "phase": "warmup", "iter": it, "total_iter": it,
                "loss": float(loss.detach().cpu()),
                "ic": float(parts["ic"].cpu()),
                "bc": float(parts["bc"].cpu()),
                "pde": float(parts["pde"].cpu()),
                "weighted_pde": float(parts["weighted_pde"].cpu()),
                "lr": float(opt.param_groups[0]["lr"]),
            })

        if it == 1 or it % cfg.print_every == 0 or it == cfg.warmup_iters:
            print(
                f"[warmup] {it:6d}/{cfg.warmup_iters} "
                f"loss={float(loss.detach().cpu()):.3e} "
                f"ic={parts['ic'].item():.1e} bc={parts['bc'].item():.1e} "
                f"pde={parts['pde'].item():.1e}"
            )
    return model, pd.DataFrame(history)


def train_vanilla_continuation_fixed(model, cfg: ShallowWater2DConfig):
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr_gated)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=cfg.gated_iters, eta_min=cfg.lr_gated * 0.03
    )
    history = []
    print("\n[Baseline] PINN continuation from the same warm-up checkpoint")
    print("Protocol: no validation, no early stopping, final checkpoint is evaluated.")

    for it in range(1, cfg.gated_iters + 1):
        opt.zero_grad(set_to_none=True)
        loss, parts = vanilla_loss(model, cfg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step(); sch.step()

        if it == 1 or it % cfg.history_every == 0 or it == cfg.gated_iters:
            history.append({
                "phase": "PINN_continuation", "iter": it,
                "total_iter": cfg.warmup_iters + it,
                "loss": float(loss.detach().cpu()),
                "ic": float(parts["ic"].cpu()),
                "bc": float(parts["bc"].cpu()),
                "pde": float(parts["pde"].cpu()),
                "weighted_pde": float(parts["weighted_pde"].cpu()),
                "lr": float(opt.param_groups[0]["lr"]),
            })

        if it == 1 or it % cfg.print_every == 0 or it == cfg.gated_iters:
            print(
                f"[PINN-cont] {it:6d}/{cfg.gated_iters} "
                f"loss={float(loss.detach().cpu()):.3e} "
                f"ic={parts['ic'].item():.1e} bc={parts['bc'].item():.1e} "
                f"pde={parts['pde'].item():.1e}"
            )
    return model, pd.DataFrame(history)


def train_gated_fixed(model, cfg: ShallowWater2DConfig):
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr_gated)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=cfg.gated_iters, eta_min=cfg.lr_gated * 0.03
    )
    history = []
    print("\n[Phase 2] Trace-ratio gated PINN")
    print("Protocol: no validation, no early stopping, no best checkpoint; final checkpoint is evaluated.")

    for it in range(1, cfg.gated_iters + 1):
        opt.zero_grad(set_to_none=True)
        loss, parts = trace_ratio_gated_loss(model, it, cfg.gated_iters, cfg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step(); sch.step()

        if it == 1 or it % cfg.history_every == 0 or it == cfg.gated_iters:
            history.append({
                "phase": "tPINN_trace_ratio", "iter": it,
                "total_iter": cfg.warmup_iters + it,
                "loss": float(loss.detach().cpu()),
                "ic": float(parts["ic"].cpu()),
                "bc": float(parts["bc"].cpu()),
                "pde": float(parts["pde"].cpu()),
                "weighted_pde": float(parts["weighted_pde"].cpu()),
                "gate_mean": float(parts["gate_mean"].cpu()),
                "gate_max": float(parts["gate_max"].cpu()),
                "gate_active_gt_0p5": float(parts["gate_active_gt_0p5"].cpu()),
                "gate_active_gt_0p1": float(parts["gate_active_gt_0p1"].cpu()),
                "C_mean": float(parts["C_mean"].cpu()),
                "C_max": float(parts["C_max"].cpu()),
                "Jhat_mean": float(parts["Jhat_mean"].cpu()),
                "Jhat_max": float(parts["Jhat_max"].cpu()),
                "Jbar": float(parts["Jbar"].cpu()),
                "W_mean": float(parts["W_mean"].cpu()),
                "W_min": float(parts["W_min"].cpu()),
                "h": float(parts["h"].cpu()),
                "cmin": float(parts["cmin"].cpu()),
                "progress": float(parts["progress"].cpu()),
                "valid_frac": float(parts["valid_frac"].cpu()),
                "lr": float(opt.param_groups[0]["lr"]),
            })

        if it == 1 or it % cfg.print_every == 0 or it == cfg.gated_iters:
            print(
                f"[tPINN] {it:6d}/{cfg.gated_iters} "
                f"loss={float(loss.detach().cpu()):.3e} "
                f"ic={parts['ic'].item():.1e} bc={parts['bc'].item():.1e} "
                f"pde={parts['pde'].item():.1e} "
                f"wpde={parts['weighted_pde'].item():.1e} "
                f"g={parts['gate_mean'].item():.3f} "
                f"act>.5={parts['gate_active_gt_0p5'].item():.3f} "
                f"h={parts['h'].item():.5f} cmin={parts['cmin'].item():.2f}"
            )
    return model, pd.DataFrame(history)


# ============================================================
# 8. Second-order finite-volume reference (MUSCL + Rusanov + SSP-RK2)
# ============================================================

def minmod(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    same = a * b > 0.0
    return np.where(same, np.sign(a) * np.minimum(np.abs(a), np.abs(b)), 0.0)


def fixed_pad(U: np.ndarray, c: ShallowWater2DConfig, ng: int = 2) -> np.ndarray:
    out = np.empty((3, U.shape[1] + 2 * ng, U.shape[2] + 2 * ng), dtype=U.dtype)
    out[0].fill(c.h_outside)
    out[1].fill(c.m_outside)
    out[2].fill(c.n_outside)
    out[:, ng:-ng, ng:-ng] = U
    return out


def positivity_limit_slope(Ue: np.ndarray, slope: np.ndarray, c: ShallowWater2DConfig) -> np.ndarray:
    room = np.maximum(Ue[0] - c.fv_depth_floor, 0.0)
    theta = np.minimum(1.0, room / (0.5 * np.abs(slope[0]) + 1.0e-14))
    return slope * theta[None, :, :]


def rusanov_x(UL: np.ndarray, UR: np.ndarray, c: ShallowWater2DConfig) -> np.ndarray:
    FL, _ = flux_np(UL, c)
    FR, _ = flux_np(UR, c)
    hL = np.maximum(UL[0], c.fv_depth_floor)
    hR = np.maximum(UR[0], c.fv_depth_floor)
    a = np.maximum(np.abs(UL[1] / hL) + np.sqrt(c.g_const * hL),
                   np.abs(UR[1] / hR) + np.sqrt(c.g_const * hR))
    return 0.5 * (FL + FR) - 0.5 * a[None, :, :] * (UR - UL)


def rusanov_y(UL: np.ndarray, UR: np.ndarray, c: ShallowWater2DConfig) -> np.ndarray:
    _, GL = flux_np(UL, c)
    _, GR = flux_np(UR, c)
    hL = np.maximum(UL[0], c.fv_depth_floor)
    hR = np.maximum(UR[0], c.fv_depth_floor)
    a = np.maximum(np.abs(UL[2] / hL) + np.sqrt(c.g_const * hL),
                   np.abs(UR[2] / hR) + np.sqrt(c.g_const * hR))
    return 0.5 * (GL + GR) - 0.5 * a[None, :, :] * (UR - UL)


def fv_rhs(U: np.ndarray, dx: float, dy: float, c: ShallowWater2DConfig) -> np.ndarray:
    Ue = fixed_pad(U, c, ng=2)
    sx = np.zeros_like(Ue)
    sy = np.zeros_like(Ue)
    if c.fv_order >= 2:
        sx[:, :, 1:-1] = minmod(Ue[:, :, 1:-1] - Ue[:, :, :-2], Ue[:, :, 2:] - Ue[:, :, 1:-1])
        sy[:, 1:-1, :] = minmod(Ue[:, 1:-1, :] - Ue[:, :-2, :], Ue[:, 2:, :] - Ue[:, 1:-1, :])
        sx = positivity_limit_slope(Ue, sx, c)
        sy = positivity_limit_slope(Ue, sy, c)

    ULx = Ue[:, 2:-2, 1:-2] + 0.5 * sx[:, 2:-2, 1:-2]
    URx = Ue[:, 2:-2, 2:-1] - 0.5 * sx[:, 2:-2, 2:-1]
    ULy = Ue[:, 1:-2, 2:-2] + 0.5 * sy[:, 1:-2, 2:-2]
    URy = Ue[:, 2:-1, 2:-2] - 0.5 * sy[:, 2:-1, 2:-2]
    Fx = rusanov_x(ULx, URx, c)
    Gy = rusanov_y(ULy, URy, c)
    return -(Fx[:, :, 1:] - Fx[:, :, :-1]) / dx - (Gy[:, 1:, :] - Gy[:, :-1, :]) / dy


def enforce_fv_positivity(U: np.ndarray, c: ShallowWater2DConfig) -> np.ndarray:
    U = U.copy()
    bad = U[0] < c.fv_depth_floor
    U[0] = np.maximum(U[0], c.fv_depth_floor)
    U[1][bad] = 0.0
    U[2][bad] = 0.0
    return U


def bilinear_grid_interp(xsrc, ysrc, A, xdst, ydst):
    ix = np.clip(np.searchsorted(xsrc, xdst) - 1, 0, len(xsrc) - 2)
    iy = np.clip(np.searchsorted(ysrc, ydst) - 1, 0, len(ysrc) - 2)
    tx = (xdst - xsrc[ix]) / (xsrc[ix + 1] - xsrc[ix])
    ty = (ydst - ysrc[iy]) / (ysrc[iy + 1] - ysrc[iy])
    A00 = A[iy[:, None], ix[None, :]]
    A01 = A[iy[:, None], (ix + 1)[None, :]]
    A10 = A[(iy + 1)[:, None], ix[None, :]]
    A11 = A[(iy + 1)[:, None], (ix + 1)[None, :]]
    low = (1.0 - tx[None, :]) * A00 + tx[None, :] * A01
    high = (1.0 - tx[None, :]) * A10 + tx[None, :] * A11
    return (1.0 - ty[:, None]) * low + ty[:, None] * high


def reference_cache_path(c: ShallowWater2DConfig) -> Path:
    root = Path(c.output_dir) / c.experiment_name
    return root / f"fv_reference_n{c.fv_nxy}_nt{c.eval_nt}_order{c.fv_order}.npz"


def compute_fv_reference(c: ShallowWater2DConfig):
    cache = reference_cache_path(c)
    if c.reference_cache and cache.exists():
        data = np.load(cache)
        return {k: data[k] for k in data.files}

    nx = ny = c.fv_nxy
    x_edges = np.linspace(c.x_min, c.x_max, nx + 1)
    y_edges = np.linspace(c.y_min, c.y_max, ny + 1)
    dx = x_edges[1] - x_edges[0]
    dy = y_edges[1] - y_edges[0]
    xc = 0.5 * (x_edges[:-1] + x_edges[1:])
    yc = 0.5 * (y_edges[:-1] + y_edges[1:])
    Xc, Yc = np.meshgrid(xc, yc, indexing="xy")
    inside = (Xc - c.center_x) ** 2 + (Yc - c.center_y) ** 2 < c.dam_radius**2
    U = np.zeros((3, ny, nx), dtype=np.float64)
    U[0] = np.where(inside, c.h_inside, c.h_outside)
    U[1] = np.where(inside, c.m_inside, c.m_outside)
    U[2] = np.where(inside, c.n_inside, c.n_outside)

    xeval = np.linspace(c.x_min, c.x_max, c.eval_nxy)
    yeval = np.linspace(c.y_min, c.y_max, c.eval_nxy)
    teval = np.linspace(c.t_min, c.t_max, c.eval_nt)
    H = np.empty((c.eval_nt, c.eval_nxy, c.eval_nxy), dtype=np.float64)
    M = np.empty_like(H)
    N = np.empty_like(H)

    def store(k: int):
        H[k] = bilinear_grid_interp(xc, yc, U[0], xeval, yeval)
        M[k] = bilinear_grid_interp(xc, yc, U[1], xeval, yeval)
        N[k] = bilinear_grid_interp(xc, yc, U[2], xeval, yeval)

    store(0)
    tcur = c.t_min
    next_idx = 1
    print(f"Computing 2D FV reference: {nx}x{ny}, order={c.fv_order}, nt={c.eval_nt}")
    while tcur < c.t_max - 1.0e-14:
        hs = np.maximum(U[0], c.fv_depth_floor)
        ax = np.max(np.abs(U[1] / hs) + np.sqrt(c.g_const * hs))
        ay = np.max(np.abs(U[2] / hs) + np.sqrt(c.g_const * hs))
        dt = c.fv_cfl / (ax / dx + ay / dy + 1.0e-14)
        if next_idx < len(teval):
            dt = min(dt, teval[next_idx] - tcur)
        dt = min(dt, c.t_max - tcur)
        if dt <= 0.0:
            raise RuntimeError("Non-positive FV time step")

        U1 = enforce_fv_positivity(U + dt * fv_rhs(U, dx, dy, c), c)
        U2 = enforce_fv_positivity(0.5 * U + 0.5 * (U1 + dt * fv_rhs(U1, dx, dy, c)), c)
        U = U2
        tcur += dt

        while next_idx < len(teval) and abs(tcur - teval[next_idx]) < 1.0e-11:
            store(next_idx)
            print(f"  stored t={teval[next_idx]:.4f} ({next_idx + 1}/{len(teval)})")
            next_idx += 1

    ref = {"x": xeval, "y": yeval, "t": teval, "H": H, "M": M, "N": N}
    if c.save_outputs or c.reference_cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, **ref)
    return ref


# ============================================================
# 9. Prediction and interpolation
# ============================================================

@torch.no_grad()
def predict_points(model: nn.Module, x_np, y_np, t_np, batch_size: int = 65536) -> np.ndarray:
    shape = np.asarray(x_np).shape
    xf = np.asarray(x_np, dtype=np.float64).reshape(-1)
    yf = np.asarray(y_np, dtype=np.float64).reshape(-1)
    tf = np.asarray(t_np, dtype=np.float64).reshape(-1)
    if not (xf.shape == yf.shape == tf.shape):
        raise ValueError("x, y and t arrays must have identical shapes")
    out = []
    for s in range(0, xf.size, batch_size):
        e = min(s + batch_size, xf.size)
        pts = torch.tensor(np.stack([xf[s:e], yf[s:e], tf[s:e]], axis=1), device=DEVICE, dtype=DTYPE)
        out.append(to_numpy(model(pts)))
    return np.concatenate(out, axis=0).reshape(*shape, 3)


def reference_time_index(ref, t_value: float) -> int:
    return int(np.argmin(np.abs(ref["t"] - t_value)))


def bilinear_points(xgrid, ygrid, A, xq, yq):
    xq = np.asarray(xq); yq = np.asarray(yq)
    ix = np.clip(np.searchsorted(xgrid, xq) - 1, 0, len(xgrid) - 2)
    iy = np.clip(np.searchsorted(ygrid, yq) - 1, 0, len(ygrid) - 2)
    tx = (xq - xgrid[ix]) / (xgrid[ix + 1] - xgrid[ix])
    ty = (yq - ygrid[iy]) / (ygrid[iy + 1] - ygrid[iy])
    return (
        (1 - tx) * (1 - ty) * A[iy, ix]
        + tx * (1 - ty) * A[iy, ix + 1]
        + (1 - tx) * ty * A[iy + 1, ix]
        + tx * ty * A[iy + 1, ix + 1]
    )


def reference_points(ref, xq, yq, t_value: float) -> np.ndarray:
    tvals = ref["t"]
    if t_value <= tvals[0]:
        k0 = k1 = 0; a = 0.0
    elif t_value >= tvals[-1]:
        k0 = k1 = len(tvals) - 1; a = 0.0
    else:
        k1 = int(np.searchsorted(tvals, t_value))
        k0 = k1 - 1
        a = (t_value - tvals[k0]) / (tvals[k1] - tvals[k0])
    vals = []
    for key in ["H", "M", "N"]:
        v0 = bilinear_points(ref["x"], ref["y"], ref[key][k0], xq, yq)
        v1 = bilinear_points(ref["x"], ref["y"], ref[key][k1], xq, yq)
        vals.append((1.0 - a) * v0 + a * v1)
    return np.stack(vals, axis=-1)


def predict_eval_grid(model: nn.Module, c: ShallowWater2DConfig, t_value: float):
    x = np.linspace(c.x_min, c.x_max, c.eval_nxy)
    y = np.linspace(c.y_min, c.y_max, c.eval_nxy)
    X, Y = np.meshgrid(x, y, indexing="xy")
    T = np.full_like(X, t_value)
    return x, y, X, Y, predict_points(model, X, Y, T)


# ============================================================
# 10. Metrics
# ============================================================

def compute_reference_error_metrics(model, ref, cfg: ShallowWater2DConfig):
    """Space-time and final-time errors against the cached 2D FV reference."""
    hs = h_scale(cfg)
    qs = q_scale(cfg)

    # Space-time grid: same reduced evaluation protocol as the 2D Euler notebook.
    x_st = np.linspace(cfg.x_min, cfg.x_max, cfg.eval_space_nxy)
    y_st = np.linspace(cfg.y_min, cfg.y_max, cfg.eval_space_nxy)
    t_st = np.linspace(cfg.t_min, cfg.t_max, cfg.eval_nt)
    Xst, Yst = np.meshgrid(x_st, y_st, indexing="xy")

    err_acc = {name: [] for name in ["h", "m", "n"]}
    exact_acc = {name: [] for name in ["h", "m", "n"]}
    scaled_err_sq = []
    scaled_exact_sq = []

    for tt in t_st:
        Tst = np.full_like(Xst, tt)
        W = predict_points(model, Xst, Yst, Tst)
        We = reference_points(ref, Xst, Yst, float(tt))
        for j, name in enumerate(["h", "m", "n"]):
            err_acc[name].append((W[..., j] - We[..., j]).reshape(-1))
            exact_acc[name].append(We[..., j].reshape(-1))
        se = ((W[..., 0] - We[..., 0]) / hs) ** 2 \
             + ((W[..., 1] - We[..., 1]) / qs) ** 2 \
             + ((W[..., 2] - We[..., 2]) / qs) ** 2
        sx = (We[..., 0] / hs) ** 2 + (We[..., 1] / qs) ** 2 + (We[..., 2] / qs) ** 2
        scaled_err_sq.append(se.reshape(-1))
        scaled_exact_sq.append(sx.reshape(-1))

    out = {}
    for name in ["h", "m", "n"]:
        err = np.concatenate(err_acc[name])
        exact = np.concatenate(exact_acc[name])
        out[f"{name}_space_time_l1"] = float(np.mean(np.abs(err)))
        out[f"{name}_space_time_l2"] = float(np.sqrt(np.mean(err**2)))
        out[f"{name}_space_time_rel_l2"] = float(
            np.sqrt(np.mean(err**2)) / (np.sqrt(np.mean(exact**2)) + 1.0e-12)
        )

    se_all = np.concatenate(scaled_err_sq)
    sx_all = np.concatenate(scaled_exact_sq)
    out["state_scaled_space_time_rel_l2"] = float(
        np.sqrt(np.mean(se_all)) / (np.sqrt(np.mean(sx_all)) + 1.0e-12)
    )

    # Final-time grid: same full spatial resolution as the stored reference.
    x_f = np.linspace(cfg.x_min, cfg.x_max, cfg.eval_nxy)
    y_f = np.linspace(cfg.y_min, cfg.y_max, cfg.eval_nxy)
    Xf, Yf = np.meshgrid(x_f, y_f, indexing="xy")
    Tf = np.full_like(Xf, cfg.t_max)
    Wf = predict_points(model, Xf, Yf, Tf)
    Wef = reference_points(ref, Xf, Yf, cfg.t_max)

    for j, name in enumerate(["h", "m", "n"]):
        err = Wf[..., j] - Wef[..., j]
        exact = Wef[..., j]
        out[f"{name}_final_l1"] = float(np.mean(np.abs(err)))
        out[f"{name}_final_l2"] = float(np.sqrt(np.mean(err**2)))
        out[f"{name}_final_rel_l2"] = float(
            np.sqrt(np.mean(err**2)) / (np.sqrt(np.mean(exact**2)) + 1.0e-12)
        )

    se = ((Wf[..., 0] - Wef[..., 0]) / hs) ** 2 \
         + ((Wf[..., 1] - Wef[..., 1]) / qs) ** 2 \
         + ((Wf[..., 2] - Wef[..., 2]) / qs) ** 2
    sx = (Wef[..., 0] / hs) ** 2 + (Wef[..., 1] / qs) ** 2 + (Wef[..., 2] / qs) ** 2
    out["state_scaled_final_rel_l2"] = float(
        np.sqrt(np.mean(se)) / (np.sqrt(np.mean(sx)) + 1.0e-12)
    )
    out["h_min"] = float(Wf[..., 0].min())
    out["h_max"] = float(Wf[..., 0].max())
    out["h_positivity_violation"] = float(max(0.0, cfg.h_floor - Wf[..., 0].min()))
    return out


def radial_profile(model: nn.Module, ref, c: ShallowWater2DConfig, t_value: float, angle: float, n: Optional[int] = None):
    if n is None:
        n = c.line_n
    # Restrict to a radius that remains inside the square for all angles.
    rmax = 0.98 * min(c.x_max - c.center_x, c.center_x - c.x_min, c.y_max - c.center_y, c.center_y - c.y_min)
    r = np.linspace(0.0, rmax, n)
    x = c.center_x + r * np.cos(angle)
    y = c.center_y + r * np.sin(angle)
    W = predict_points(model, x, y, np.full_like(r, t_value))
    We = reference_points(ref, x, y, t_value)
    qr = W[:, 1] * np.cos(angle) + W[:, 2] * np.sin(angle)
    qre = We[:, 1] * np.cos(angle) + We[:, 2] * np.sin(angle)
    return r, W[:, 0], qr, We[:, 0], qre


def front_radius(r: np.ndarray, h: np.ndarray, c: ShallowWater2DConfig) -> float:
    mask = r >= 0.8 * c.dam_radius
    rr = r[mask]; hh = h[mask]
    grad = np.abs(np.gradient(hh, rr))
    return float(rr[int(np.argmax(grad))])


def compute_front_metrics(model: nn.Module, ref, c: ShallowWater2DConfig) -> Dict[str, float]:
    errs = []
    pred_r = []
    ref_r = []
    for phi in np.linspace(0.0, 2.0 * math.pi, c.radial_angles, endpoint=False):
        r, hp, _, he, _ = radial_profile(model, ref, c, c.t_max, float(phi))
        rp = front_radius(r, hp, c)
        re = front_radius(r, he, c)
        pred_r.append(rp); ref_r.append(re); errs.append(abs(rp - re))
    return {
        "front_radius_mae": float(np.mean(errs)),
        "front_radius_rmse": float(np.sqrt(np.mean(np.asarray(errs) ** 2))),
        "front_radius_pred_angular_std": float(np.std(pred_r, ddof=0)),
        "front_radius_ref_angular_std": float(np.std(ref_r, ddof=0)),
    }


def compute_global_conservation_metrics(model: nn.Module, c: ShallowWater2DConfig) -> Dict[str, float]:
    x = np.linspace(c.x_min, c.x_max, c.cons_nxy)
    y = np.linspace(c.y_min, c.y_max, c.cons_nxy)
    tvals = np.linspace(c.t_min, c.t_max, c.cons_nt)
    X, Y = np.meshgrid(x, y, indexing="xy")
    Uint = []
    Bint = []
    for tt in tvals:
        W = predict_points(model, X, Y, np.full_like(X, tt))
        Uint.append(np.array([np.trapz(np.trapz(W[..., k], x, axis=1), y, axis=0) for k in range(3)]))
        yy = y; xx = x
        Wl = predict_points(model, np.full_like(yy, c.x_min), yy, np.full_like(yy, tt))
        Wr = predict_points(model, np.full_like(yy, c.x_max), yy, np.full_like(yy, tt))
        Fl, _ = flux_np(np.moveaxis(Wl, -1, 0), c)
        Fr, _ = flux_np(np.moveaxis(Wr, -1, 0), c)
        bx = np.array([np.trapz(Fr[k] - Fl[k], yy) for k in range(3)])
        Wb = predict_points(model, xx, np.full_like(xx, c.y_min), np.full_like(xx, tt))
        Wt = predict_points(model, xx, np.full_like(xx, c.y_max), np.full_like(xx, tt))
        _, Gb = flux_np(np.moveaxis(Wb, -1, 0), c)
        _, Gt = flux_np(np.moveaxis(Wt, -1, 0), c)
        by = np.array([np.trapz(Gt[k] - Gb[k], xx) for k in range(3)])
        Bint.append(bx + by)
    Uint = np.asarray(Uint); Bint = np.asarray(Bint)
    residuals = []
    for i in range(len(tvals)):
        btime = np.zeros(3) if i == 0 else np.array([np.trapz(Bint[:i+1, k], tvals[:i+1]) for k in range(3)])
        residuals.append(Uint[i] - Uint[0] + btime)
    residuals = np.asarray(residuals)
    out = {}
    for k, name in enumerate(["mass", "momx", "momy"]):
        out[f"global_{name}_cons_mean_abs"] = float(np.mean(np.abs(residuals[:, k])))
        out[f"global_{name}_cons_final_abs"] = float(abs(residuals[-1, k]))
    return out


def compute_local_conservation_metrics(model: nn.Module, c: ShallowWater2DConfig) -> Dict[str, float]:
    rng = np.random.default_rng(c.seed + 271828)
    abs_vals = []
    rel_vals = []
    for _ in range(c.n_control_volumes):
        for _try in range(200):
            x1, x2 = np.sort(rng.uniform(c.x_min, c.x_max, 2))
            y1, y2 = np.sort(rng.uniform(c.y_min, c.y_max, 2))
            t1, t2 = np.sort(rng.uniform(c.t_min, c.t_max, 2))
            if (x2-x1) >= c.cv_min_width and (y2-y1) >= c.cv_min_width and (t2-t1) >= c.cv_min_duration:
                break
        xq = np.linspace(x1, x2, c.cv_quad_nxy)
        yq = np.linspace(y1, y2, c.cv_quad_nxy)
        tq = np.linspace(t1, t2, c.cv_quad_nt)
        X, Y = np.meshgrid(xq, yq, indexing="xy")
        W1 = predict_points(model, X, Y, np.full_like(X, t1))
        W2 = predict_points(model, X, Y, np.full_like(X, t2))
        V1 = np.array([np.trapz(np.trapz(W1[..., k], xq, axis=1), yq, axis=0) for k in range(3)])
        V2 = np.array([np.trapz(np.trapz(W2[..., k], xq, axis=1), yq, axis=0) for k in range(3)])
        boundary = []
        for tt in tq:
            Wl = predict_points(model, np.full_like(yq, x1), yq, np.full_like(yq, tt))
            Wr = predict_points(model, np.full_like(yq, x2), yq, np.full_like(yq, tt))
            Fl, _ = flux_np(np.moveaxis(Wl, -1, 0), c)
            Fr, _ = flux_np(np.moveaxis(Wr, -1, 0), c)
            bx = np.array([np.trapz(Fr[k] - Fl[k], yq) for k in range(3)])
            Wb = predict_points(model, xq, np.full_like(xq, y1), np.full_like(xq, tt))
            Wt = predict_points(model, xq, np.full_like(xq, y2), np.full_like(xq, tt))
            _, Gb = flux_np(np.moveaxis(Wb, -1, 0), c)
            _, Gt = flux_np(np.moveaxis(Wt, -1, 0), c)
            by = np.array([np.trapz(Gt[k] - Gb[k], xq) for k in range(3)])
            boundary.append(bx + by)
        boundary = np.asarray(boundary)
        B = np.array([np.trapz(boundary[:, k], tq) for k in range(3)])
        res = V2 - V1 + B
        abs_vals.append(np.abs(res))
        rel_vals.append(np.abs(res) / (np.abs(V2) + np.abs(V1) + np.abs(B) + 1.0e-12))
    abs_vals = np.asarray(abs_vals); rel_vals = np.asarray(rel_vals)
    out = {}
    for k, name in enumerate(["mass", "momx", "momy"]):
        out[f"local_{name}_cons_cv_mean_abs"] = float(np.mean(abs_vals[:, k]))
        out[f"local_{name}_cons_cv_median_abs"] = float(np.median(abs_vals[:, k]))
        out[f"local_{name}_cons_cv_mean_rel"] = float(np.mean(rel_vals[:, k]))
    return out


@torch.no_grad()
def compute_gate_diagnostics(model: nn.Module, c: ShallowWater2DConfig, progress: float = 1.0) -> Dict[str, float]:
    probe, cmin, p = schedule_from_progress(progress, c)
    x = np.linspace(c.x_min, c.x_max, c.gate_eval_nxy)
    y = np.linspace(c.y_min, c.y_max, c.gate_eval_nxy)
    X, Y = np.meshgrid(x, y, indexing="xy")
    T = np.full_like(X, c.t_max)
    pts = torch.tensor(np.stack([X.ravel(), Y.ravel(), T.ravel()], axis=1), device=DEVICE, dtype=DTYPE)
    gate, C, Jhat, Jbar, valid, _, _ = trace_ratio_gate_conservative(
        model, pts[:, 0:1], pts[:, 1:2], pts[:, 2:3], probe, cmin, c
    )
    g = to_numpy(gate).reshape(X.shape)
    v = to_numpy(valid).reshape(X.size, c.ring_trace_pairs) > 0
    valid_point = v.any(axis=1).reshape(X.shape)
    if not valid_point.any():
        valid_point[:] = True
    return {
        "gate_progress": float(p), "gate_h": float(probe), "gate_cmin": float(cmin),
        "gate_mean": float(g[valid_point].mean()), "gate_max": float(g[valid_point].max()),
        "gate_active_frac_gt_0p5": float(np.mean(g[valid_point] > 0.5)),
        "gate_active_frac_gt_0p1": float(np.mean(g[valid_point] > 0.1)),
        "gate_Jbar": float(Jbar.detach().cpu()),
    }


def evaluate_model(model: nn.Module, name: str, ref, c: ShallowWater2DConfig) -> Dict[str, float]:
    return {
        "model": name, "seed": c.seed,
        **compute_reference_error_metrics(model, ref, c),
        **compute_front_metrics(model, ref, c),
        **compute_global_conservation_metrics(model, c),
        **compute_local_conservation_metrics(model, c),
        **compute_gate_diagnostics(model, c, progress=1.0),
    }


# ============================================================
# 11. Paired five-seed runner and summary
# ============================================================

def print_header(c: ShallowWater2DConfig, run_dir: Path):
    h_start, c_start, _ = schedule_from_progress(0.0, c)
    h_end, c_end, _ = schedule_from_progress(1.0, c)
    print("=" * 92)
    print("2D shallow-water circular dam break: paired PINN vs ring trace-ratio gated PINN")
    print("=" * 92)
    print(f"seed={c.seed}, device={DEVICE}, dtype={c.dtype}")
    print(f"domain=[{c.x_min},{c.x_max}]x[{c.y_min},{c.y_max}], t=[{c.t_min},{c.t_max}]")
    print(f"IC: h={c.h_inside} inside r<{c.dam_radius}; h={c.h_outside} outside; momenta zero")
    print(f"network: width={c.width}, depth={c.depth}, activation={c.activation}, params={count_params(MLP(c))}")
    print(f"budget: warmup={c.warmup_iters}, continuation={c.gated_iters}")
    print(f"batches: n_f={c.n_f}, n_ic={c.n_ic}, n_bc={c.n_bc}")
    print(f"loss weights: {c.w_ic}, {c.w_bc}, {c.w_pde}")
    print(f"ring directions: projective pairs={c.ring_trace_pairs}; h0={h0(c):.8f}; probe={h_start:.8f}->{h_end:.8f}")
    print(f"cmin={c_start:.2f}->{c_end:.2f}; beta={c.beta}; residual floor={c.residual_floor}")
    print("protocol: shared warm-up, restored RNG, fixed budget, final checkpoint, no validation")
    print(f"run_dir={run_dir}")
    print("=" * 92)


def run_one_seed_paired(seed: int, base_cfg: ShallowWater2DConfig, ref) -> pd.DataFrame:
    c = copy.deepcopy(base_cfg)
    c.seed = seed
    configure_runtime(c)
    set_seed(seed)
    run_dir = make_run_dir(c)
    print_header(c, run_dir)
    if c.save_outputs:
        save_json({
            "config": asdict(c),
            "protocol": "shared warm-up; identical continuation RNG; fixed budget; final checkpoint only",
            "python": platform.python_version(), "torch": torch.__version__,
            "created_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        }, run_dir / "config_and_runtime.json")

    warm = MLP(c).to(DEVICE, dtype=DTYPE)
    warm, hist_warm = train_warmup_fixed(warm, c)
    state = clone_state(warm)
    rng_state = get_rng_state()

    pinn = MLP(c).to(DEVICE, dtype=DTYPE); pinn.load_state_dict(state)
    set_rng_state(rng_state)
    pinn, hist_pinn = train_vanilla_continuation_fixed(pinn, c)

    tpinn = MLP(c).to(DEVICE, dtype=DTYPE); tpinn.load_state_dict(state)
    set_rng_state(rng_state)
    tpinn, hist_tpinn = train_gated_fixed(tpinn, c)

    rows = [evaluate_model(pinn, "PINN", ref, c), evaluate_model(tpinn, "tPINN", ref, c)]
    metrics = pd.DataFrame(rows)
    display(metrics)

    if c.save_outputs:
        hist_warm.to_csv(run_dir / "history_warmup_shared.csv", index=False)
        hist_pinn.to_csv(run_dir / "history_PINN_continuation.csv", index=False)
        hist_tpinn.to_csv(run_dir / "history_tPINN_trace_ratio.csv", index=False)
        metrics.to_csv(run_dir / "metrics_final.csv", index=False)
        torch.save(state, run_dir / "checkpoint_warmup_final.pt")
        torch.save(pinn.state_dict(), run_dir / "checkpoint_final_PINN.pt")
        torch.save(tpinn.state_dict(), run_dir / "checkpoint_final_tPINN.pt")
    return metrics


def summarize_multiseed(all_metrics: pd.DataFrame, save_dir: Path):
    save_dir.mkdir(parents=True, exist_ok=True)
    all_metrics.to_csv(save_dir / "all_seed_metrics_raw.csv", index=False)
    metrics = [c for c in all_metrics.columns if c not in {"model", "seed"} and pd.api.types.is_numeric_dtype(all_metrics[c])]
    summary = all_metrics.groupby("model")[metrics].agg(["mean", "std"])
    summary.to_csv(save_dir / "summary_mean_std_numeric.csv")
    formatted = []
    for model, group in all_metrics.groupby("model"):
        row = {"model": model}
        for m in metrics:
            row[m] = f"{group[m].mean():.4e} ± {group[m].std(ddof=1):.2e}"
        formatted.append(row)
    paper = pd.DataFrame(formatted)
    paper.to_csv(save_dir / "summary_mean_std_formatted.csv", index=False)

    lower_better = [m for m in metrics if not m.startswith("gate_") and m not in {"h_max"}]
    paired = []
    for m in lower_better:
        wide = all_metrics.pivot(index="seed", columns="model", values=m)
        if {"PINN", "tPINN"}.issubset(wide.columns):
            diff = wide["PINN"] - wide["tPINN"]
            pct = 100.0 * diff / (wide["PINN"].abs() + 1.0e-12)
            paired.append({
                "metric": m, "PINN_mean": wide["PINN"].mean(), "tPINN_mean": wide["tPINN"].mean(),
                "diff_mean_PINN_minus_tPINN": diff.mean(), "diff_std": diff.std(ddof=1),
                "improvement_pct_mean": pct.mean(), "improvement_pct_std": pct.std(ddof=1),
                "wins_tPINN_better_out_of_n": int((wide["tPINN"] < wide["PINN"]).sum()), "n": len(wide),
            })
    paired_df = pd.DataFrame(paired)
    paired_df.to_csv(save_dir / "paired_improvement_table.csv", index=False)
    display(paper)
    display(paired_df)
    return {"all_metrics": all_metrics, "summary": summary, "paper_table": paper, "paired": paired_df}


def run_multiseed_shallow_water2d(base_cfg: ShallowWater2DConfig, seeds):
    configure_runtime(base_cfg)
    ref = compute_fv_reference(base_cfg)
    rows = [run_one_seed_paired(seed, base_cfg, ref) for seed in seeds]
    all_metrics = pd.concat(rows, ignore_index=True)
    summary_dir = Path(base_cfg.output_dir) / base_cfg.experiment_name / "multiseed_summary"
    return summarize_multiseed(all_metrics, summary_dir), ref

# ============================================================
# 13. Run 5 seeds
# ============================================================

base_cfg = copy.deepcopy(cfg)
base_cfg.device = "cuda:4"  # change only if this device is unavailable
base_cfg.save_outputs = True

# Locked 2D-system protocol:
# - architecture / budget / directional detector: same as 2D Euler
# - SWE state scaling and loss weights: same as 1D shallow water
base_cfg.warmup_iters = 3000
base_cfg.gated_iters = 7000
base_cfg.n_f = 20000
base_cfg.n_ic = 6000
base_cfg.n_bc = 2500
base_cfg.ring_trace_pairs = 6
base_cfg.w_ic = 100.0
base_cfg.w_bc = 30.0
base_cfg.w_pde = 1.0
base_cfg.h_max_factor = 5.0
base_cfg.h_min_factor = 2.0
base_cfg.cmin_start = 0.50
base_cfg.cmin_end = 0.70
base_cfg.beta = 0.05
base_cfg.residual_floor = 0.02

seeds = [2026, 7, 42, 100, 31415]

# Uncomment for a quick smoke test before full training.
# base_cfg.experiment_name = "shallow_water2d_circular_ring_trace_ratio_smoke_test"
# base_cfg.device = "cpu"
# base_cfg.warmup_iters = 2
# base_cfg.gated_iters = 2
# base_cfg.n_f = 64
# base_cfg.n_ic = 32
# base_cfg.n_bc = 32
# base_cfg.eval_nxy = 40
# base_cfg.eval_space_nxy = 24
# base_cfg.eval_nt = 4
# base_cfg.gate_eval_nxy = 30
# base_cfg.cons_nxy = 16
# base_cfg.cons_nt = 4
# base_cfg.n_control_volumes = 2
# base_cfg.cv_quad_nxy = 8
# base_cfg.cv_quad_nt = 8
# base_cfg.fv_nxy = 48
# base_cfg.fv_order = 1
# seeds = [2026]

#summary_results_shallow_water2d, reference_shallow_water2d = run_multiseed_shallow_water2d(base_cfg, seeds)



from pathlib import Path
import os

RELEASE_REPO_ROOT = Path(__file__).resolve().parents[2]

PROTECTED_ROOTS = (
    (RELEASE_REPO_ROOT / "artifacts").resolve(),
    (RELEASE_REPO_ROOT / "results" / "reported_metrics").resolve(),
    (RELEASE_REPO_ROOT / "results" / "paper_figures").resolve(),
)

RUNS_ROOT = Path(
    os.environ.get(
        "TRGPINN_BASELINE_RUNS_ROOT",
        str(
            RELEASE_REPO_ROOT
            / "runs"
            / "reproduction"
            / "baselines"
        ),
    )
).expanduser().resolve()

for protected_root in PROTECTED_ROOTS:
    try:
        RUNS_ROOT.relative_to(protected_root)
    except ValueError:
        continue
    raise ValueError(
        "Refusing to write specialized-baseline outputs "
        f"inside protected release data: {protected_root}"
    )

RESULT_ROOT = RUNS_ROOT.parent
PROCESSED_ROOT = RESULT_ROOT / "processed"
TABLE_ROOT = RESULT_ROOT / "tables"
FIGURE_ROOT = RESULT_ROOT / "figures"

for directory in (
    RUNS_ROOT,
    PROCESSED_ROOT,
    TABLE_ROOT,
    FIGURE_ROOT,
):
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )


# ============================================================
# Cell B. Common storage utilities
# Paste once into each benchmark notebook.
# ============================================================

import os
import gc
import json
import time
import copy
import random
import platform
import inspect
from pathlib import Path
from dataclasses import asdict, is_dataclass

import numpy as np
import pandas as pd
import torch


SEEDS = [2026, 7, 42, 100, 31415]
ADAM_TOTAL_ITERS = 10_000


# ------------------------------------------------------------
# JSON helpers
# ------------------------------------------------------------

def _jsonable(obj):
    if is_dataclass(obj):
        return _jsonable(asdict(obj))

    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, np.generic):
        return obj.item()

    if torch.is_tensor(obj):
        if obj.numel() == 1:
            return obj.detach().cpu().item()
        return obj.detach().cpu().numpy().tolist()

    if isinstance(obj, Path):
        return str(obj)

    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj

    return str(obj)


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(_jsonable(obj), f, indent=2, ensure_ascii=False)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ------------------------------------------------------------
# Naming and folders
# ------------------------------------------------------------

def safe_name(name: str) -> str:
    return (
        str(name)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
        .replace("__", "_")
    )


def get_run_dir(equation_name: str, method_name: str, seed: int) -> Path:
    return RUNS_ROOT / safe_name(equation_name) / safe_name(method_name) / f"seed_{int(seed)}"


def ensure_run_dirs(run_dir: Path):
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "predictions").mkdir(exist_ok=True)
    (run_dir / "diagnostics").mkdir(exist_ok=True)


def mark_status(run_dir: Path, status: str, extra=None):
    ensure_run_dirs(run_dir)

    payload = {
        "status": status,
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    if extra is not None:
        payload.update(_jsonable(extra))

    save_json(payload, run_dir / "status.json")

    if status == "success":
        (run_dir / "_SUCCESS").write_text("success\n", encoding="utf-8")

    if status == "failed":
        (run_dir / "_FAILED").write_text("failed\n", encoding="utf-8")


def run_is_success(equation_name: str, method_name: str, seed: int) -> bool:
    return (get_run_dir(equation_name, method_name, seed) / "_SUCCESS").exists()


# ------------------------------------------------------------
# Config helpers
# ------------------------------------------------------------

def cfg_to_dict(cfg):
    if is_dataclass(cfg):
        return asdict(cfg)

    if hasattr(cfg, "__dict__"):
        return dict(cfg.__dict__)

    return {"repr": repr(cfg)}


def force_total_adam_iters(cfg, total_iters=ADAM_TOTAL_ITERS):
    """
    warmup_iters는 기존 방정식 설정을 유지.
    gated_iters만 조정해서 warmup + gated = total_iters로 맞춤.
    """
    if not hasattr(cfg, "warmup_iters"):
        raise AttributeError("cfg에 warmup_iters가 없습니다.")

    warmup = int(cfg.warmup_iters)
    continuation = int(total_iters) - warmup

    if continuation <= 0:
        raise ValueError(
            f"total_iters={total_iters}, warmup_iters={warmup}라서 continuation이 0 이하입니다."
        )

    if hasattr(cfg, "gated_iters"):
        cfg.gated_iters = continuation

    if hasattr(cfg, "continuation_iters"):
        cfg.continuation_iters = continuation

    return cfg


def prepare_cfg_for_seed(base_cfg, seed, total_iters=ADAM_TOTAL_ITERS):
    cfg_s = copy.deepcopy(base_cfg)
    cfg_s.seed = int(seed)

    if hasattr(cfg_s, "save_outputs"):
        cfg_s.save_outputs = False

    cfg_s = force_total_adam_iters(cfg_s, total_iters=total_iters)
    return cfg_s


# ------------------------------------------------------------
# Environment
# ------------------------------------------------------------

def collect_env():
    info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }

    if torch.cuda.is_available():
        info["cuda_device_count"] = torch.cuda.device_count()
        info["cuda_current_device"] = torch.cuda.current_device()
        info["cuda_device_name"] = torch.cuda.get_device_name(torch.cuda.current_device())

    return info


# ------------------------------------------------------------
# RNG state helpers
# ------------------------------------------------------------

def get_rng_state_any():
    """
    기존 노트북에 get_rng_state가 있으면 그걸 우선 사용.
    없으면 여기서 직접 저장.
    """
    if "get_rng_state" in globals():
        return get_rng_state()

    state = {
        "python_random": random.getstate(),
        "numpy_random": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }

    if torch.cuda.is_available():
        state["torch_cuda_all"] = torch.cuda.get_rng_state_all()

    return state


def set_rng_state_any(state):
    """
    기존 노트북에 set_rng_state가 있으면 그걸 우선 사용.
    없으면 여기서 직접 복구.
    """
    if "set_rng_state" in globals():
        set_rng_state(state)
        return

    random.setstate(state["python_random"])
    np.random.set_state(state["numpy_random"])
    torch.set_rng_state(state["torch_cpu"])

    if torch.cuda.is_available() and "torch_cuda_all" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda_all"])


# ------------------------------------------------------------
# Model helpers
# ------------------------------------------------------------

def clone_state_dict_cpu(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def load_state_dict_to_model(model, state_dict):
    device = next(model.parameters()).device
    model.load_state_dict({k: v.to(device) for k, v in state_dict.items()})
    return model


def parameter_count(model):
    if "count_params" in globals():
        try:
            return int(count_params(model))
        except Exception:
            pass

    return int(sum(p.numel() for p in model.parameters()))


# ------------------------------------------------------------
# Save helpers
# ------------------------------------------------------------

def save_history(history, run_dir: Path, filename="history.csv"):
    if history is None:
        return

    path = run_dir / filename

    if isinstance(history, pd.DataFrame):
        history.to_csv(path, index=False)
    else:
        pd.DataFrame(history).to_csv(path, index=False)


def save_model_checkpoint(model, run_dir: Path, filename: str, extra=None):
    payload = {
        "model_state_dict": model.state_dict(),
        "extra": extra or {},
    }

    torch.save(payload, run_dir / filename)


def save_run_header(run_dir, equation_name, method_name, cfg, method_config):
    ensure_run_dirs(run_dir)

    save_json(
        {
            "equation": equation_name,
            "method": method_name,
            "seed": int(getattr(cfg, "seed", -1)),
            "config": cfg_to_dict(cfg),
        },
        run_dir / "config.json",
    )

    save_json(method_config, run_dir / "method.json")
    save_json(collect_env(), run_dir / "env.json")

    mark_status(
        run_dir,
        "running",
        {
            "equation": equation_name,
            "method": method_name,
            "seed": int(getattr(cfg, "seed", -1)),
        },
    )


# ------------------------------------------------------------
# Evaluation helper
# ------------------------------------------------------------

def evaluate_model_any(model, method_name: str, cfg, ref=None):
    """
    Exact reference 문제:
        evaluate_model(model, method_name, cfg)

    FV reference 문제:
        evaluate_model(model, method_name, ref, cfg)

    둘 다 자동으로 처리.
    """
    if "evaluate_model" not in globals():
        return {
            "method": method_name,
            "seed": int(getattr(cfg, "seed", -1)),
            "status": "failed",
            "evaluation_error": "evaluate_model is not defined.",
        }

    try:
        row = evaluate_model(model, method_name, cfg)
    except TypeError:
        row = evaluate_model(model, method_name, ref, cfg)

    if isinstance(row, pd.Series):
        row = row.to_dict()

    if not isinstance(row, dict):
        row = dict(row)

    row["method"] = method_name
    row["seed"] = int(getattr(cfg, "seed", row.get("seed", -1)))
    row["warmup_iters"] = int(getattr(cfg, "warmup_iters", 0))
    row["continuation_iters"] = int(getattr(cfg, "gated_iters", 0))
    row["adam_total_iters"] = row["warmup_iters"] + row["continuation_iters"]
    row["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")

    return row


# ------------------------------------------------------------
# Reference helper
# ------------------------------------------------------------

def build_reference_if_needed(cfg):
    """
    Exact solution 문제는 None.
    FV reference 문제는 reference를 한 번 계산해서 반환.
    """
    if "reference_fv_solution" in globals():
        print("[reference] Building 1D FV reference...")
        return reference_fv_solution(cfg)

    if "compute_fv_reference" in globals():
        print("[reference] Building 2D FV reference...")
        return compute_fv_reference(cfg)

    return None


# ------------------------------------------------------------
# Prediction save helper
# ------------------------------------------------------------

def try_save_predictions(model, cfg, run_dir: Path):
    """
    각 노트북마다 prediction 함수 이름이 조금씩 달라서,
    있는 함수 중 하나를 찾아 prediction npz로 저장.
    """
    pred_dir = run_dir / "predictions"
    pred_dir.mkdir(exist_ok=True)

    candidate_names = [
        "predict_grid",          # 1D Burgers, 1D Euler, 1D SWE
        "predict_final_grid",    # 2D Euler
        "predict_eval_grid",     # 2D SWE
        "predict_xy_grid",       # 2D Burgers
        "predict_space_time_cube"
    ]

    for fn_name in candidate_names:
        if fn_name not in globals():
            continue

        fn = globals()[fn_name]

        try:
            sig = inspect.signature(fn)
            params = sig.parameters

            kwargs = {}

            if "cfg" in params:
                kwargs["cfg"] = cfg

            if "c" in params:
                kwargs["c"] = cfg

            if "t_value" in params:
                kwargs["t_value"] = float(getattr(cfg, "t_max", 0.0))

            out = fn(model, **kwargs)

        except Exception as e:
            save_json(
                {
                    "function": fn_name,
                    "error": repr(e),
                },
                pred_dir / f"{fn_name}_save_error.json",
            )
            continue

        arrays = {}

        if isinstance(out, dict):
            for k, v in out.items():
                try:
                    arrays[str(k)] = np.asarray(v)
                except Exception:
                    pass

        elif isinstance(out, tuple):
            for i, v in enumerate(out):
                try:
                    arrays[f"arr_{i}"] = np.asarray(v)
                except Exception:
                    pass

        else:
            try:
                arrays["arr_0"] = np.asarray(out)
            except Exception:
                pass

        if arrays:
            np.savez_compressed(pred_dir / f"{fn_name}.npz", **arrays)
            save_json(
                {"saved_prediction_function": fn_name},
                pred_dir / "prediction_info.json",
            )
            return

    save_json(
        {"warning": "No prediction function was successfully saved."},
        pred_dir / "prediction_info.json",
    )


# ------------------------------------------------------------
# Ours diagnostic save helper
# ------------------------------------------------------------

def try_save_gate_diagnostics(model, cfg, run_dir: Path):
    if "compute_gate_diagnostics" not in globals():
        return

    diag_dir = run_dir / "diagnostics"
    diag_dir.mkdir(exist_ok=True)

    try:
        try:
            diag = compute_gate_diagnostics(model, cfg, progress=1.0)
        except TypeError:
            diag = compute_gate_diagnostics(model, cfg)

        save_json(diag, diag_dir / "gate_diagnostics.json")

    except Exception as e:
        save_json(
            {"gate_diagnostics_error": repr(e)},
            diag_dir / "gate_diagnostics_error.json",
        )


# ------------------------------------------------------------
# Vanilla continuation function helper
# ------------------------------------------------------------

def get_vanilla_continuation_function():
    """
    대부분 노트북:
        train_vanilla_continuation_fixed

    2D Burgers:
        train_pinn_continuation_fixed
    """
    if "train_vanilla_continuation_fixed" in globals():
        return train_vanilla_continuation_fixed

    if "train_pinn_continuation_fixed" in globals():
        return train_pinn_continuation_fixed

    raise RuntimeError(
        "Vanilla continuation 함수가 없습니다. "
        "train_vanilla_continuation_fixed 또는 train_pinn_continuation_fixed가 필요합니다."
    )


def gpinn_residual_scaled_components(model, coordinates, cfg):
    x, y, t = (
        value.detach().clone().to(DEVICE, dtype=DTYPE).requires_grad_(True)
        for value in coordinates
    )
    W = model(cat_xyt(x, y, t))
    h = W[:, 0:1]
    m = W[:, 1:2]
    n = W[:, 2:3]
    flux_x_values, flux_y_values = flux_torch(h, m, n, cfg)
    conserved = (h, m, n)
    residuals = []
    for state, flux_x_value, flux_y_value in zip(
        conserved, flux_x_values, flux_y_values
    ):
        state_t = torch.autograd.grad(
            state, t, torch.ones_like(state), create_graph=True, retain_graph=True
        )[0]
        flux_x = torch.autograd.grad(
            flux_x_value, x, torch.ones_like(flux_x_value),
            create_graph=True, retain_graph=True
        )[0]
        flux_y = torch.autograd.grad(
            flux_y_value, y, torch.ones_like(flux_y_value),
            create_graph=True, retain_graph=True
        )[0]
        residuals.append(state_t + flux_x + flux_y)
    scaled = (
        residuals[0] / h_scale(cfg),
        residuals[1] / q_scale(cfg),
        residuals[2] / q_scale(cfg),
    )
    return scaled, (x, y, t)

def gpinn_gradient_loss_from_components(components, coordinates):
    losses = []
    for component in components:
        for coordinate in coordinates:
            gradient = torch.autograd.grad(
                component,
                coordinate,
                grad_outputs=torch.ones_like(component),
                create_graph=True,
                retain_graph=True,
                allow_unused=True,
            )[0]
            if gradient is not None:
                losses.append(gradient.pow(2).mean())
    if not losses:
        raise RuntimeError("No residual-coordinate gradients were produced.")
    return torch.stack(losses).sum()


# ============================================================
# Cell E. RAD-PINN baseline
# Paste after Cell A/B/C in each benchmark notebook.
# ============================================================

import time
import gc
import numpy as np
import pandas as pd
import torch


def _tuple_to_device_dtype(pts):
    return tuple(p.to(DEVICE, dtype=DTYPE) for p in pts)


def compute_residual_score_for_points(model, pts, cfg):
    """
    Compute pointwise PDE residual score for RAD sampling.
    Works across your 1D/2D notebooks by detecting the residual function.
    """
    pts = _tuple_to_device_dtype(pts)

    with torch.enable_grad():
        if len(pts) == 2:
            x, t = pts

            if "burgers_residual" in globals():
                _, r = burgers_residual(model, x, t)
                score = r.pow(2).reshape(-1)

            elif "euler_residual" in globals():
                *_, R2 = euler_residual(model, x, t, cfg)
                score = R2.reshape(-1)

            elif "shallow_residual" in globals():
                *_, R2 = shallow_residual(model, x, t, cfg)
                score = R2.reshape(-1)

            else:
                raise RuntimeError("No compatible 1D residual function found.")

        elif len(pts) == 3:
            x, y, t = pts

            if "burgers2d_residual" in globals():
                _, r = burgers2d_residual(model, x, y, t)
                score = r.pow(2).reshape(-1)

            elif "euler2d_residual" in globals():
                *_, R2 = euler2d_residual(model, x, y, t, cfg)
                score = R2.reshape(-1)

            elif "swe2d_residual" in globals():
                *_, R2 = swe2d_residual(model, x, y, t, cfg)
                score = R2.reshape(-1)

            else:
                raise RuntimeError("No compatible 2D residual function found.")

        else:
            raise RuntimeError(f"Unsupported residual point tuple length: {len(pts)}")

    score = torch.nan_to_num(score.detach(), nan=0.0, posinf=1.0e12, neginf=0.0)
    score = score.clamp_min(0.0)
    return score


def _slice_tuple(pts, start, end):
    return tuple(p[start:end] for p in pts)


def build_rad_point_pool(
    model,
    original_sample_f,
    cfg,
    k=1.0,
    c=1.0,
    candidate_factor=3,
    residual_batch_size=4096,
):
    """
    RAD sampling:
      1. draw candidate residual points uniformly using original sample_f
      2. evaluate pointwise residual score
      3. sample active residual points according to residual-based probability
    """
    n_active = int(cfg.n_f)
    n_candidates = int(candidate_factor * n_active)

    candidate_pts = original_sample_f(n_candidates, cfg)
    candidate_pts = _tuple_to_device_dtype(candidate_pts)

    scores = []
    for start in range(0, n_candidates, residual_batch_size):
        end = min(start + residual_batch_size, n_candidates)
        pts_batch = _slice_tuple(candidate_pts, start, end)
        sc = compute_residual_score_for_points(model, pts_batch, cfg)
        scores.append(sc.detach().cpu())

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    scores = torch.cat(scores, dim=0).float()
    scores = torch.nan_to_num(scores, nan=0.0, posinf=1.0e12, neginf=0.0).clamp_min(0.0)

    eps = scores + 1.0e-12
    weights = eps.pow(float(k))

    # Normalize by mean to avoid extreme scale sensitivity, then add c.
    mean_w = weights.mean().clamp_min(1.0e-12)
    weights = weights / mean_w + float(c)

    probs = weights / weights.sum().clamp_min(1.0e-12)

    idx = torch.multinomial(probs, n_active, replacement=True)
    idx_device = idx.to(candidate_pts[0].device)

    active_pts = tuple(p[idx_device].detach().clone() for p in candidate_pts)

    stats = {
        "n_active": n_active,
        "n_candidates": n_candidates,
        "k": float(k),
        "c": float(c),
        "score_mean": float(scores.mean().item()),
        "score_max": float(scores.max().item()),
        "score_min": float(scores.min().item()),
        "score_std": float(scores.std(unbiased=False).item()),
        "prob_max": float(probs.max().item()),
        "prob_min": float(probs.min().item()),
    }

    return active_pts, stats


def train_rad_pinn_fixed(
    model,
    cfg,
    k=1.0,
    c=1.0,
    candidate_factor=3,
    resample_period=500,
    residual_batch_size=4096,
):
    """
    RAD-PINN training.
    Uses the same top-level vanilla PINN loss, but replaces sample_f with RAD-selected residual points.
    Total optimizer budget = cfg.warmup_iters + cfg.gated_iters.
    """
    model.train()

    if "sample_f" not in globals():
        raise RuntimeError("sample_f is not defined in this notebook.")

    original_sample_f = globals()["sample_f"]

    rad_state = {
        "active_pts": None,
        "last_resample_iter": None,
    }

    sampling_history = []

    def rad_sample_f(n, cfg_arg=cfg):
        if rad_state["active_pts"] is None:
            # Fallback for safety; normally active_pts is set before each loss call.
            rad_state["active_pts"] = original_sample_f(int(n), cfg_arg)
        return rad_state["active_pts"]

    def maybe_resample(total_iter):
        if (
            rad_state["active_pts"] is None
            or total_iter == 1
            or ((total_iter - 1) % int(resample_period) == 0)
        ):
            t_sample = time.time()
            active_pts, stats = build_rad_point_pool(
                model=model,
                original_sample_f=original_sample_f,
                cfg=cfg,
                k=k,
                c=c,
                candidate_factor=candidate_factor,
                residual_batch_size=residual_batch_size,
            )
            stats["total_iter"] = int(total_iter)
            stats["sampling_wall_clock_sec"] = float(time.time() - t_sample)
            sampling_history.append(stats)

            rad_state["active_pts"] = active_pts
            rad_state["last_resample_iter"] = int(total_iter)

    # Temporarily override sample_f used inside vanilla_loss.
    globals()["sample_f"] = rad_sample_f

    history = []

    try:
        phases = [
            ("rad_warmup_lr", int(cfg.warmup_iters), float(cfg.lr_warmup), float(cfg.lr_warmup) * 0.05),
            ("rad_main_lr", int(cfg.gated_iters), float(cfg.lr_gated), float(cfg.lr_gated) * 0.03),
        ]

        total_iter = 0

        for phase_name, n_iters, lr, eta_min in phases:
            if n_iters <= 0:
                continue

            opt = torch.optim.AdamW(
                model.parameters(),
                lr=lr,
                weight_decay=float(getattr(cfg, "weight_decay", 0.0)),
            )
            sch = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt,
                T_max=n_iters,
                eta_min=eta_min,
            )

            print(f"\n[RAD-PINN] phase={phase_name}, iters={n_iters}, lr={lr:.3e}")

            for local_it in range(1, n_iters + 1):
                total_iter += 1

                maybe_resample(total_iter)

                opt.zero_grad(set_to_none=True)
                loss, parts = vanilla_loss(model, cfg)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                opt.step()
                sch.step()

                if (
                    total_iter == 1
                    or total_iter % int(cfg.history_every) == 0
                    or local_it == n_iters
                ):
                    row = {
                        "phase": phase_name,
                        "iter": int(local_it),
                        "total_iter": int(total_iter),
                        "loss": float(loss.detach().cpu()),
                        "ic": float(parts["ic"].cpu()),
                        "bc": float(parts["bc"].cpu()),
                        "pde": float(parts["pde"].cpu()),
                        "weighted_pde": float(parts["weighted_pde"].cpu()),
                        "lr": float(opt.param_groups[0]["lr"]),
                        "rad_last_resample_iter": rad_state["last_resample_iter"],
                    }
                    history.append(row)

                if (
                    total_iter == 1
                    or total_iter % int(cfg.print_every) == 0
                    or local_it == n_iters
                ):
                    print(
                        f"[RAD-PINN] {total_iter:6d}/{cfg.warmup_iters + cfg.gated_iters} "
                        f"loss={float(loss.detach().cpu()):.3e} "
                        f"ic={parts['ic'].item():.1e} "
                        f"bc={parts['bc'].item():.1e} "
                        f"pde={parts['pde'].item():.1e}"
                    )

        history_df = pd.DataFrame(history)
        sampling_df = pd.DataFrame(sampling_history)

        return model, history_df, sampling_df

    finally:
        globals()["sample_f"] = original_sample_f

# ============================================================
# Cell F. RAD-PINN runner with the same storage schema
# ============================================================

def train_rad_one_seed(
    equation_name: str,
    base_cfg,
    seed: int,
    ref=None,
    total_iters: int = ADAM_TOTAL_ITERS,
    skip_if_done: bool = True,
    k: float = 1.0,
    c: float = 1.0,
    candidate_factor: int = 3,
    resample_period: int = 500,
    residual_batch_size: int = 4096,
):
    """
    Train one RAD-PINN run and save it under:
      results/raw/runs/<equation>/rad_pinn/seed_<seed>/
    """
    cfg_s = prepare_cfg_for_seed(base_cfg, seed, total_iters=total_iters)

    configure_runtime(cfg_s)
    set_seed(int(seed))

    method_name = "RAD-PINN"
    run_dir = get_run_dir(equation_name, method_name, seed)

    if skip_if_done and (run_dir / "_SUCCESS").exists():
        print(f"[skip] RAD-PINN already completed: {equation_name}, seed={seed}")
        return pd.DataFrame([load_json(run_dir / "metrics_final.json")])

    print("=" * 100)
    print(f"Equation          : {equation_name}")
    print(f"Method            : RAD-PINN")
    print(f"Seed              : {seed}")
    print(f"Total Adam iters  : {cfg_s.warmup_iters + cfg_s.gated_iters}")
    print(f"n_f               : {cfg_s.n_f}")
    print(f"RAD k, c          : {k}, {c}")
    print(f"candidate_factor  : {candidate_factor}")
    print(f"resample_period   : {resample_period}")
    print("=" * 100)

    try:
        save_run_header(
            run_dir,
            equation_name,
            method_name,
            cfg_s,
            {
                "method": "RAD-PINN",
                "description": "Residual-based adaptive distribution sampling baseline.",
                "optimizer_main": "AdamW",
                "lbfgs_used": False,
                "adam_total_iters": int(cfg_s.warmup_iters + cfg_s.gated_iters),
                "rad_k": float(k),
                "rad_c": float(c),
                "candidate_factor": int(candidate_factor),
                "resample_period": int(resample_period),
                "residual_batch_size": int(residual_batch_size),
                "final_active_n_f": int(cfg_s.n_f),
                "note": "Same IC/BC/PDE loss weights and active residual point count as PINN/Ours.",
            },
        )

        model = MLP(cfg_s).to(DEVICE, dtype=DTYPE)

        t0 = time.time()
        model, history_df, sampling_df = train_rad_pinn_fixed(
            model=model,
            cfg=cfg_s,
            k=k,
            c=c,
            candidate_factor=candidate_factor,
            resample_period=resample_period,
            residual_batch_size=residual_batch_size,
        )
        train_time = time.time() - t0

        metrics = evaluate_model_any(model, method_name, cfg_s, ref=ref)
        metrics.update({
            "equation": equation_name,
            "method": method_name,
            "seed": int(seed),
            "optimizer_main": "AdamW",
            "lbfgs_used": False,
            "wall_clock_sec_total": float(train_time),
            "num_parameters": parameter_count(model),
            "rad_k": float(k),
            "rad_c": float(c),
            "rad_candidate_factor": int(candidate_factor),
            "rad_resample_period": int(resample_period),
            "rad_num_resamples": int(len(sampling_df)),
        })

        save_history(history_df, run_dir, "history.csv")
        save_json(metrics, run_dir / "metrics_adamw.json")
        save_json(metrics, run_dir / "metrics_final.json")

        save_model_checkpoint(
            model,
            run_dir,
            "model_adamw_final.pt",
            extra=metrics,
        )
        save_model_checkpoint(
            model,
            run_dir,
            "model_final.pt",
            extra=metrics,
        )

        try_save_predictions(model, cfg_s, run_dir)

        diag_dir = run_dir / "diagnostics"
        diag_dir.mkdir(exist_ok=True)
        sampling_df.to_csv(diag_dir / "rad_sampling_history.csv", index=False)

        mark_status(run_dir, "success", {"wall_clock_sec_total": float(train_time)})

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        return pd.DataFrame([metrics])

    except Exception as e:
        mark_status(run_dir, "failed", {"error": repr(e)})
        raise


def run_rad_seeds(
    equation_name: str,
    base_cfg,
    seeds,
    total_iters: int = ADAM_TOTAL_ITERS,
    skip_if_done: bool = True,
    k: float = 1.0,
    c: float = 1.0,
    candidate_factor: int = 3,
    resample_period: int = 500,
    residual_batch_size: int = 4096,
):
    """
    Run RAD-PINN for multiple seeds in the current equation notebook.
    """
    cfg_ref = copy.deepcopy(base_cfg)
    configure_runtime(cfg_ref)

    ref = build_reference_if_needed(cfg_ref)

    rows = []

    for seed in seeds:
        df_seed = train_rad_one_seed(
            equation_name=equation_name,
            base_cfg=base_cfg,
            seed=seed,
            ref=ref,
            total_iters=total_iters,
            skip_if_done=skip_if_done,
            k=k,
            c=c,
            candidate_factor=candidate_factor,
            resample_period=resample_period,
            residual_batch_size=residual_batch_size,
        )
        rows.append(df_seed)

    df = pd.concat(rows, ignore_index=True)

    summary_dir = RUNS_ROOT / safe_name(equation_name) / "_summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    out_path = summary_dir / "rad_pinn_metrics.csv"
    df.to_csv(out_path, index=False)

    print("\nSaved RAD summary:")
    print(out_path)

    return df

# ============================================================
# Cell E-correction. More faithful RAD implementation
# Run this AFTER Cell E/F and BEFORE running run_rad_seeds(...)
# ============================================================

def compute_residual_score_for_points(model, pts, cfg):
    """
    Return epsilon(x), not epsilon(x)^2.

    RAD paper uses a PDF based on residual magnitude:
        p(x) ∝ epsilon(x)^k / E[epsilon(x)^k] + c

    For scalar PDE:
        epsilon = |r|

    For system PDE:
        epsilon = sqrt(sum scaled residual^2)
    """
    pts = _tuple_to_device_dtype(pts)

    with torch.enable_grad():
        if len(pts) == 2:
            x, t = pts

            if "burgers_residual" in globals():
                _, r = burgers_residual(model, x, t)
                eps = r.abs().reshape(-1)

            elif "euler_residual" in globals():
                *_, R2 = euler_residual(model, x, t, cfg)
                eps = torch.sqrt(torch.clamp(R2.reshape(-1), min=0.0))

            elif "shallow_residual" in globals():
                *_, R2 = shallow_residual(model, x, t, cfg)
                eps = torch.sqrt(torch.clamp(R2.reshape(-1), min=0.0))

            else:
                raise RuntimeError("No compatible 1D residual function found.")

        elif len(pts) == 3:
            x, y, t = pts

            if "burgers2d_residual" in globals():
                _, r = burgers2d_residual(model, x, y, t)
                eps = r.abs().reshape(-1)

            elif "euler2d_residual" in globals():
                *_, R2 = euler2d_residual(model, x, y, t, cfg)
                eps = torch.sqrt(torch.clamp(R2.reshape(-1), min=0.0))

            elif "swe2d_residual" in globals():
                *_, R2 = swe2d_residual(model, x, y, t, cfg)
                eps = torch.sqrt(torch.clamp(R2.reshape(-1), min=0.0))

            else:
                raise RuntimeError("No compatible 2D residual function found.")

        else:
            raise RuntimeError(f"Unsupported residual point tuple length: {len(pts)}")

    eps = torch.nan_to_num(eps.detach(), nan=0.0, posinf=1.0e12, neginf=0.0)
    eps = eps.clamp_min(0.0)
    return eps


def build_rad_point_pool(
    model,
    original_sample_f,
    cfg,
    k=1.0,
    c=1.0,
    candidate_factor=3,
    residual_batch_size=4096,
):
    """
    Brute-force RAD sampling.

    1. Draw candidate points S0 from the original uniform sampler.
    2. Compute epsilon(x) on S0.
    3. Define weights:
           w(x) = epsilon(x)^k / mean(epsilon^k) + c
    4. Sample cfg.n_f active residual points from S0 according to w.
    """
    n_active = int(cfg.n_f)
    n_candidates = int(candidate_factor * n_active)

    candidate_pts = original_sample_f(n_candidates, cfg)
    candidate_pts = _tuple_to_device_dtype(candidate_pts)

    eps_list = []

    for start in range(0, n_candidates, residual_batch_size):
        end = min(start + residual_batch_size, n_candidates)
        pts_batch = _slice_tuple(candidate_pts, start, end)
        eps_batch = compute_residual_score_for_points(model, pts_batch, cfg)
        eps_list.append(eps_batch.detach().cpu())

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    eps = torch.cat(eps_list, dim=0).float()
    eps = torch.nan_to_num(eps, nan=0.0, posinf=1.0e12, neginf=0.0).clamp_min(0.0)

    eps_k = (eps + 1.0e-12).pow(float(k))
    mean_eps_k = eps_k.mean().clamp_min(1.0e-12)

    weights = eps_k / mean_eps_k + float(c)
    probs = weights / weights.sum().clamp_min(1.0e-12)

    idx = torch.multinomial(probs, n_active, replacement=True)
    idx_device = idx.to(candidate_pts[0].device)

    active_pts = tuple(p[idx_device].detach().clone() for p in candidate_pts)

    stats = {
        "n_active": int(n_active),
        "n_candidates": int(n_candidates),
        "k": float(k),
        "c": float(c),
        "epsilon_mean": float(eps.mean().item()),
        "epsilon_max": float(eps.max().item()),
        "epsilon_min": float(eps.min().item()),
        "epsilon_std": float(eps.std(unbiased=False).item()),
        "prob_max": float(probs.max().item()),
        "prob_min": float(probs.min().item()),
        "candidate_factor": int(candidate_factor),
    }

    return active_pts, stats


def train_rad_pinn_fixed(
    model,
    cfg,
    k=1.0,
    c=1.0,
    candidate_factor=3,
    resample_period=1000,
    residual_batch_size=4096,
):
    """
    More faithful RAD-PINN training:

      Phase 1:
        uniform residual sampling warm-up using the notebook's train_warmup_fixed

      Phase 2:
        RAD residual resampling every resample_period iterations,
        with p(x) ∝ epsilon(x)^k / E[epsilon(x)^k] + c

    This keeps the total Adam-family budget:
        cfg.warmup_iters + cfg.gated_iters = 10000
    """
    model.train()

    if "sample_f" not in globals():
        raise RuntimeError("sample_f is not defined in this notebook.")

    if "train_warmup_fixed" not in globals():
        raise RuntimeError("train_warmup_fixed is not defined in this notebook.")

    original_sample_f = globals()["sample_f"]

    # --------------------------------------------------------
    # Phase 1. Uniform warm-up, matching RAD Algorithm 2's
    # initial training before residual-based resampling.
    # --------------------------------------------------------
    print(f"\n[RAD-PINN] Phase 1: uniform warm-up, iters={cfg.warmup_iters}")

    t_warm = time.time()
    model, hist_warm = train_warmup_fixed(model, cfg)
    warm_time = time.time() - t_warm

    if isinstance(hist_warm, pd.DataFrame):
        hist_warm_df = hist_warm.copy()
    else:
        hist_warm_df = pd.DataFrame(hist_warm)

    if len(hist_warm_df) > 0:
        hist_warm_df["rad_stage"] = "uniform_warmup"
        hist_warm_df["rad_sampling"] = False

    # --------------------------------------------------------
    # Phase 2. RAD continuation.
    # --------------------------------------------------------
    rad_state = {
        "active_pts": None,
        "last_resample_iter": None,
    }

    sampling_history = []

    def rad_sample_f(n, cfg_arg=cfg):
        if rad_state["active_pts"] is None:
            rad_state["active_pts"] = original_sample_f(int(n), cfg_arg)
        return rad_state["active_pts"]

    def maybe_resample(local_iter):
        if (
            rad_state["active_pts"] is None
            or local_iter == 1
            or ((local_iter - 1) % int(resample_period) == 0)
        ):
            t_sample = time.time()

            active_pts, stats = build_rad_point_pool(
                model=model,
                original_sample_f=original_sample_f,
                cfg=cfg,
                k=k,
                c=c,
                candidate_factor=candidate_factor,
                residual_batch_size=residual_batch_size,
            )

            stats["local_rad_iter"] = int(local_iter)
            stats["global_iter"] = int(cfg.warmup_iters + local_iter)
            stats["sampling_wall_clock_sec"] = float(time.time() - t_sample)

            sampling_history.append(stats)

            rad_state["active_pts"] = active_pts
            rad_state["last_resample_iter"] = int(local_iter)

    globals()["sample_f"] = rad_sample_f

    history = []

    try:
        n_iters = int(cfg.gated_iters)
        lr = float(cfg.lr_gated)
        eta_min = lr * 0.03

        opt = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=float(getattr(cfg, "weight_decay", 0.0)),
        )

        sch = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt,
            T_max=n_iters,
            eta_min=eta_min,
        )

        print(
            f"\n[RAD-PINN] Phase 2: RAD continuation, "
            f"iters={n_iters}, lr={lr:.3e}, k={k}, c={c}, period={resample_period}"
        )

        t_rad = time.time()

        hist_every = int(getattr(cfg, "history_every", 500))
        print_every = int(getattr(cfg, "print_every", 1000))

        for local_it in range(1, n_iters + 1):
            maybe_resample(local_it)

            opt.zero_grad(set_to_none=True)
            loss, parts = vanilla_loss(model, cfg)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

            opt.step()
            sch.step()

            global_iter = int(cfg.warmup_iters + local_it)

            if (
                local_it == 1
                or local_it % hist_every == 0
                or local_it == n_iters
            ):
                history.append({
                    "phase": "rad_continuation",
                    "iter": int(local_it),
                    "global_iter": int(global_iter),
                    "loss": float(loss.detach().cpu()),
                    "ic": float(parts["ic"].cpu()),
                    "bc": float(parts["bc"].cpu()),
                    "pde": float(parts["pde"].cpu()),
                    "weighted_pde": float(parts["weighted_pde"].cpu()),
                    "lr": float(opt.param_groups[0]["lr"]),
                    "rad_last_resample_iter": rad_state["last_resample_iter"],
                    "rad_stage": "rad_continuation",
                    "rad_sampling": True,
                })

            if (
                local_it == 1
                or local_it % print_every == 0
                or local_it == n_iters
            ):
                print(
                    f"[RAD-PINN] {global_iter:6d}/{cfg.warmup_iters + cfg.gated_iters} "
                    f"loss={float(loss.detach().cpu()):.3e} "
                    f"ic={parts['ic'].item():.1e} "
                    f"bc={parts['bc'].item():.1e} "
                    f"pde={parts['pde'].item():.1e}"
                )

        rad_time = time.time() - t_rad

        hist_rad_df = pd.DataFrame(history)
        sampling_df = pd.DataFrame(sampling_history)

        history_df = pd.concat([hist_warm_df, hist_rad_df], ignore_index=True)

        if len(sampling_df) > 0:
            sampling_df["warmup_wall_clock_sec"] = float(warm_time)
            sampling_df["rad_wall_clock_sec"] = float(rad_time)

        return model, history_df, sampling_df

    finally:
        globals()["sample_f"] = original_sample_f

import time
import gc
import copy
import numpy as np
import pandas as pd
import torch

def rard_to_device_dtype(pts):
    return tuple(p.to(DEVICE, dtype=DTYPE) for p in pts)

def rard_to_residual_inputs(pts):
    return tuple(p.detach().clone().to(DEVICE, dtype=DTYPE).requires_grad_(True) for p in pts)

def rard_slice_tuple(pts, start, end):
    return tuple(p[start:end] for p in pts)

def rard_concat_tuple(a, b):
    return tuple(torch.cat([x.detach(), y.detach()], dim=0) for x, y in zip(a, b))

def rard_point_count(pts):
    return int(pts[0].shape[0])

def rard_residual_epsilon(model, pts, cfg):
    pts = rard_to_residual_inputs(pts)

    with torch.enable_grad():
        if len(pts) == 2:
            x, t = pts

            if "burgers_residual" in globals():
                _, r = burgers_residual(model, x, t)
                eps = r.abs().reshape(-1)

            elif "euler_residual" in globals():
                *_, R2 = euler_residual(model, x, t, cfg)
                eps = torch.sqrt(torch.clamp(R2.reshape(-1), min=0.0))

            elif "shallow_residual" in globals():
                *_, R2 = shallow_residual(model, x, t, cfg)
                eps = torch.sqrt(torch.clamp(R2.reshape(-1), min=0.0))

            else:
                raise RuntimeError("No compatible 1D residual function found.")

        elif len(pts) == 3:
            x, y, t = pts

            if "burgers2d_residual" in globals():
                _, r = burgers2d_residual(model, x, y, t)
                eps = r.abs().reshape(-1)

            elif "euler2d_residual" in globals():
                *_, R2 = euler2d_residual(model, x, y, t, cfg)
                eps = torch.sqrt(torch.clamp(R2.reshape(-1), min=0.0))

            elif "swe2d_residual" in globals():
                *_, R2 = swe2d_residual(model, x, y, t, cfg)
                eps = torch.sqrt(torch.clamp(R2.reshape(-1), min=0.0))

            else:
                raise RuntimeError("No compatible 2D residual function found.")

        else:
            raise RuntimeError(f"Unsupported point tuple length: {len(pts)}")

    eps = torch.nan_to_num(eps.detach(), nan=0.0, posinf=1.0e12, neginf=0.0)
    return eps.clamp_min(0.0)

def rard_sample_new_points(
    model,
    original_sample_f,
    cfg,
    n_new,
    final_n_f,
    k=2.0,
    c=0.0,
    candidate_factor=3,
    residual_batch_size=4096,
):
    n_new = int(n_new)
    final_n_f = int(final_n_f)
    n_candidates = max(int(candidate_factor * final_n_f), n_new + 1)

    candidate_pts = original_sample_f(n_candidates, cfg)
    candidate_pts = rard_to_device_dtype(candidate_pts)

    eps_parts = []

    for start in range(0, n_candidates, int(residual_batch_size)):
        end = min(start + int(residual_batch_size), n_candidates)
        pts_batch = rard_slice_tuple(candidate_pts, start, end)
        eps_batch = rard_residual_epsilon(model, pts_batch, cfg)
        eps_parts.append(eps_batch.detach().cpu())

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    eps = torch.cat(eps_parts, dim=0).float()
    eps = torch.nan_to_num(eps, nan=0.0, posinf=1.0e12, neginf=0.0).clamp_min(0.0)

    eps_k = (eps + 1.0e-12).pow(float(k))
    mean_eps_k = eps_k.mean().clamp_min(1.0e-12)
    weights = eps_k / mean_eps_k + float(c)

    if float(weights.sum().item()) <= 0.0 or not torch.isfinite(weights).all():
        weights = torch.ones_like(weights)

    probs = weights / weights.sum().clamp_min(1.0e-12)

    idx = torch.multinomial(probs, n_new, replacement=False)
    idx_device = idx.to(candidate_pts[0].device)

    new_pts = tuple(p[idx_device].detach().clone() for p in candidate_pts)

    stats = {
        "n_new": int(n_new),
        "n_candidates": int(n_candidates),
        "k": float(k),
        "c": float(c),
        "epsilon_mean": float(eps.mean().item()),
        "epsilon_max": float(eps.max().item()),
        "epsilon_min": float(eps.min().item()),
        "epsilon_std": float(eps.std(unbiased=False).item()),
        "prob_max": float(probs.max().item()),
        "prob_min": float(probs.min().item()),
    }

    return new_pts, stats

def rard_add_schedule(final_n, initial_n, n_iters, add_period):
    remaining = int(final_n) - int(initial_n)

    if remaining <= 0:
        return {}

    event_iters = list(range(1, int(n_iters) + 1, int(add_period)))

    if len(event_iters) == 0:
        event_iters = [1]

    n_events = min(len(event_iters), remaining)
    event_iters = event_iters[:n_events]

    base = remaining // n_events
    extra = remaining % n_events

    schedule = {}
    for i, it in enumerate(event_iters):
        schedule[int(it)] = int(base + (1 if i < extra else 0))

    return schedule

def train_rard_pinn_fixed(
    model,
    cfg,
    k=2.0,
    c=0.0,
    initial_fraction=0.5,
    add_period=500,
    candidate_factor=3,
    residual_batch_size=4096,
):
    if "sample_f" not in globals():
        raise RuntimeError("sample_f is not defined.")

    if "vanilla_loss" not in globals():
        raise RuntimeError("vanilla_loss is not defined.")

    original_sample_f = globals()["sample_f"]

    final_n = int(cfg.n_f)
    initial_n = int(round(float(initial_fraction) * final_n))
    initial_n = max(1, min(initial_n, final_n))

    active_pts = original_sample_f(initial_n, cfg)
    active_pts = rard_to_device_dtype(active_pts)

    state = {
        "active_pts": active_pts,
        "active_n": initial_n,
    }

    def rard_sample_f(n, cfg_arg=cfg):
        return state["active_pts"]

    globals()["sample_f"] = rard_sample_f

    history = []
    refinement_history = []

    try:
        total_budget = int(cfg.warmup_iters) + int(cfg.gated_iters)

        phases = [
            ("rard_initial", int(cfg.warmup_iters), float(cfg.lr_warmup), float(cfg.lr_warmup) * 0.05),
            ("rard_refine", int(cfg.gated_iters), float(cfg.lr_gated), float(cfg.lr_gated) * 0.03),
        ]

        global_iter = 0

        for phase_name, n_iters, lr, eta_min in phases:
            if n_iters <= 0:
                continue

            opt = torch.optim.AdamW(
                model.parameters(),
                lr=lr,
                weight_decay=float(getattr(cfg, "weight_decay", 0.0)),
            )

            sch = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt,
                T_max=n_iters,
                eta_min=eta_min,
            )

            if phase_name == "rard_refine":
                schedule = rard_add_schedule(final_n, state["active_n"], n_iters, add_period)
            else:
                schedule = {}

            hist_every = int(getattr(cfg, "history_every", 500))
            print_every = int(getattr(cfg, "print_every", 1000))

            print(f"[RAR-D] phase={phase_name}, iters={n_iters}, active_n={state['active_n']}")

            for local_it in range(1, n_iters + 1):
                global_iter += 1

                if local_it in schedule:
                    t_add = time.time()

                    new_pts, stats = rard_sample_new_points(
                        model=model,
                        original_sample_f=original_sample_f,
                        cfg=cfg,
                        n_new=schedule[local_it],
                        final_n_f=final_n,
                        k=k,
                        c=c,
                        candidate_factor=candidate_factor,
                        residual_batch_size=residual_batch_size,
                    )

                    state["active_pts"] = rard_concat_tuple(state["active_pts"], new_pts)
                    state["active_n"] = rard_point_count(state["active_pts"])

                    stats.update({
                        "phase": phase_name,
                        "local_iter": int(local_it),
                        "global_iter": int(global_iter),
                        "active_n_after": int(state["active_n"]),
                        "add_wall_clock_sec": float(time.time() - t_add),
                    })

                    refinement_history.append(stats)

                opt.zero_grad(set_to_none=True)

                loss, parts = vanilla_loss(model, cfg)

                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    float(getattr(cfg, "grad_clip", 1.0)),
                )

                opt.step()
                sch.step()

                if local_it == 1 or local_it % hist_every == 0 or local_it == n_iters:
                    history.append({
                        "phase": phase_name,
                        "local_iter": int(local_it),
                        "global_iter": int(global_iter),
                        "active_n": int(state["active_n"]),
                        "loss": float(loss.detach().cpu()),
                        "ic": float(parts["ic"].detach().cpu()),
                        "bc": float(parts["bc"].detach().cpu()),
                        "pde": float(parts["pde"].detach().cpu()),
                        "weighted_pde": float(parts["weighted_pde"].detach().cpu()),
                        "lr": float(opt.param_groups[0]["lr"]),
                    })

                if local_it == 1 or local_it % print_every == 0 or local_it == n_iters:
                    print(
                        f"[RAR-D] {global_iter:6d}/{total_budget} "
                        f"active_n={state['active_n']} "
                        f"loss={float(loss.detach().cpu()):.3e} "
                        f"ic={float(parts['ic'].detach().cpu()):.1e} "
                        f"bc={float(parts['bc'].detach().cpu()):.1e} "
                        f"pde={float(parts['pde'].detach().cpu()):.1e}"
                    )

        history_df = pd.DataFrame(history)
        refinement_df = pd.DataFrame(refinement_history)

        final_pts = tuple(p.detach().cpu().numpy() for p in state["active_pts"])

        return model, history_df, refinement_df, final_pts

    finally:
        globals()["sample_f"] = original_sample_f

def train_rard_one_seed(
    equation_name,
    base_cfg,
    seed,
    ref=None,
    total_iters=ADAM_TOTAL_ITERS,
    skip_if_done=True,
    k=2.0,
    c=0.0,
    initial_fraction=0.5,
    add_period=500,
    candidate_factor=3,
    residual_batch_size=4096,
):
    cfg_s = prepare_cfg_for_seed(base_cfg, seed, total_iters=total_iters)

    configure_runtime(cfg_s)
    set_seed(int(seed))

    method_name = "RAR-D-PINN"
    run_dir = get_run_dir(equation_name, method_name, seed)

    if skip_if_done and (run_dir / "_SUCCESS").exists():
        return pd.DataFrame([load_json(run_dir / "metrics_final.json")])

    try:
        save_run_header(
            run_dir,
            equation_name,
            method_name,
            cfg_s,
            {
                "method": "RAR-D-PINN",
                "description": "Residual-based adaptive refinement with distribution.",
                "optimizer_main": "AdamW",
                "lbfgs_used": False,
                "adam_total_iters": int(cfg_s.warmup_iters + cfg_s.gated_iters),
                "rard_k": float(k),
                "rard_c": float(c),
                "initial_fraction": float(initial_fraction),
                "initial_active_n_f": int(round(float(initial_fraction) * int(cfg_s.n_f))),
                "final_active_n_f": int(cfg_s.n_f),
                "add_period": int(add_period),
                "candidate_factor": int(candidate_factor),
                "residual_batch_size": int(residual_batch_size),
            },
        )

        model = MLP(cfg_s).to(DEVICE, dtype=DTYPE)

        t0 = time.time()

        model, history_df, refinement_df, final_pts = train_rard_pinn_fixed(
            model=model,
            cfg=cfg_s,
            k=k,
            c=c,
            initial_fraction=initial_fraction,
            add_period=add_period,
            candidate_factor=candidate_factor,
            residual_batch_size=residual_batch_size,
        )

        train_time = time.time() - t0

        metrics = evaluate_model_any(model, method_name, cfg_s, ref=ref)

        metrics.update({
            "equation": equation_name,
            "method": method_name,
            "seed": int(seed),
            "optimizer_main": "AdamW",
            "lbfgs_used": False,
            "wall_clock_sec_total": float(train_time),
            "num_parameters": parameter_count(model),
            "rard_k": float(k),
            "rard_c": float(c),
            "rard_initial_fraction": float(initial_fraction),
            "rard_initial_active_n_f": int(round(float(initial_fraction) * int(cfg_s.n_f))),
            "rard_final_active_n_f": int(cfg_s.n_f),
            "rard_add_period": int(add_period),
            "rard_candidate_factor": int(candidate_factor),
            "rard_num_additions": int(len(refinement_df)),
        })

        save_history(history_df, run_dir, "history.csv")
        save_json(metrics, run_dir / "metrics_adamw.json")
        save_json(metrics, run_dir / "metrics_final.json")

        save_model_checkpoint(
            model,
            run_dir,
            "model_adamw_final.pt",
            extra=metrics,
        )

        save_model_checkpoint(
            model,
            run_dir,
            "model_final.pt",
            extra=metrics,
        )

        try_save_predictions(model, cfg_s, run_dir)

        diag_dir = run_dir / "diagnostics"
        diag_dir.mkdir(exist_ok=True)

        refinement_df.to_csv(diag_dir / "rard_refinement_history.csv", index=False)

        np.savez_compressed(
            diag_dir / "rard_active_points_final.npz",
            **{f"arr_{i}": arr for i, arr in enumerate(final_pts)}
        )

        mark_status(run_dir, "success", {"wall_clock_sec_total": float(train_time)})

        del model

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        gc.collect()

        return pd.DataFrame([metrics])

    except Exception as e:
        mark_status(run_dir, "failed", {"error": repr(e)})
        raise

def run_rard_seeds(
    equation_name,
    base_cfg,
    seeds,
    total_iters=ADAM_TOTAL_ITERS,
    skip_if_done=True,
    k=2.0,
    c=0.0,
    initial_fraction=0.5,
    add_period=500,
    candidate_factor=3,
    residual_batch_size=4096,
):
    cfg_ref = copy.deepcopy(base_cfg)

    configure_runtime(cfg_ref)

    ref = build_reference_if_needed(cfg_ref)

    rows = []

    for seed in seeds:
        df_seed = train_rard_one_seed(
            equation_name=equation_name,
            base_cfg=base_cfg,
            seed=seed,
            ref=ref,
            total_iters=total_iters,
            skip_if_done=skip_if_done,
            k=k,
            c=c,
            initial_fraction=initial_fraction,
            add_period=add_period,
            candidate_factor=candidate_factor,
            residual_batch_size=residual_batch_size,
        )

        rows.append(df_seed)

    df = pd.concat(rows, ignore_index=True)

    summary_dir = RUNS_ROOT / safe_name(equation_name) / "_summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    out_path = summary_dir / "rar_d_pinn_metrics.csv"
    df.to_csv(out_path, index=False)

    print("saved:", out_path)

    return df

import time
import gc
import math
import numpy as np
import pandas as pd
import torch

def sa_coords_to_input(coords):
    if len(coords) == 2:
        return cat_xt(coords[0], coords[1])
    if len(coords) == 3:
        return cat_xyt(coords[0], coords[1], coords[2])
    raise RuntimeError(f"unsupported coordinate tuple length: {len(coords)}")

def sa_pointwise_data_loss(pred, target, cfg):
    c = pred.shape[1]

    if c == 1:
        return (pred - target).pow(2).reshape(-1)

    if c == 2 and "h_scale" in globals() and "q_scale" in globals():
        hs = h_scale(cfg)
        qs = q_scale(cfg)
        return (
            ((pred[:, 0:1] - target[:, 0:1]) / hs).pow(2)
            + ((pred[:, 1:2] - target[:, 1:2]) / qs).pow(2)
        ).reshape(-1)

    if c == 3 and "h_scale" in globals() and "q_scale" in globals():
        hs = h_scale(cfg)
        qs = q_scale(cfg)
        return (
            ((pred[:, 0:1] - target[:, 0:1]) / hs).pow(2)
            + ((pred[:, 1:2] - target[:, 1:2]) / qs).pow(2)
            + ((pred[:, 2:3] - target[:, 2:3]) / qs).pow(2)
        ).reshape(-1)

    if c == 3 and "scales" in globals():
        rho_s, u_s, p_s, _, _ = scales(cfg)
        return (
            ((pred[:, 0:1] - target[:, 0:1]) / rho_s).pow(2)
            + ((pred[:, 1:2] - target[:, 1:2]) / u_s).pow(2)
            + ((pred[:, 2:3] - target[:, 2:3]) / p_s).pow(2)
        ).reshape(-1)

    if c == 4 and "scales" in globals():
        rho_s, u_s, p_s, _, _ = scales(cfg)
        return (
            ((pred[:, 0:1] - target[:, 0:1]) / rho_s).pow(2)
            + ((pred[:, 1:2] - target[:, 1:2]) / u_s).pow(2)
            + ((pred[:, 2:3] - target[:, 2:3]) / u_s).pow(2)
            + ((pred[:, 3:4] - target[:, 3:4]) / p_s).pow(2)
        ).reshape(-1)

    return (pred - target).pow(2).sum(dim=1)

def sa_pointwise_residual_loss(model, coords, cfg):
    if len(coords) == 2:
        x, t = coords

        if "burgers_residual" in globals():
            _, r = burgers_residual(model, x, t)
            return r.pow(2).reshape(-1)

        if "euler_residual" in globals():
            *_, R2 = euler_residual(model, x, t, cfg)
            return R2.reshape(-1)

        if "shallow_residual" in globals():
            *_, R2 = shallow_residual(model, x, t, cfg)
            return R2.reshape(-1)

    if len(coords) == 3:
        x, y, t = coords

        if "burgers2d_residual" in globals():
            _, r = burgers2d_residual(model, x, y, t)
            return r.pow(2).reshape(-1)

        if "euler2d_residual" in globals():
            *_, R2 = euler2d_residual(model, x, y, t, cfg)
            return R2.reshape(-1)

        if "swe2d_residual" in globals():
            *_, R2 = swe2d_residual(model, x, y, t, cfg)
            return R2.reshape(-1)

    raise RuntimeError("no compatible residual function found")

def sa_sample_bc_points(cfg):
    if "sample_bc" in globals():
        return sample_bc(cfg.n_bc, cfg)
    if "sample_bc_exact" in globals():
        return sample_bc_exact(cfg.n_bc, cfg)
    raise RuntimeError("no boundary sampler found")

def sa_make_fixed_points(cfg):
    ic = sample_ic(cfg.n_ic, cfg)
    bc = sa_sample_bc_points(cfg)
    f = sample_f(cfg.n_f, cfg)

    ic_coords = tuple(v.detach().clone() for v in ic[:-1])
    ic_target = ic[-1].detach().clone()

    bc_coords = tuple(v.detach().clone() for v in bc[:-1])
    bc_target = bc[-1].detach().clone()

    f_coords = tuple(v.detach().clone() for v in f)

    return {
        "ic_coords": ic_coords,
        "ic_target": ic_target,
        "bc_coords": bc_coords,
        "bc_target": bc_target,
        "f_coords": f_coords,
    }

def sa_make_params(fixed_points):
    return torch.nn.ParameterDict({
        "log_w_ic": torch.nn.Parameter(torch.zeros(fixed_points["ic_target"].shape[0], device=DEVICE, dtype=DTYPE)),
        "log_w_bc": torch.nn.Parameter(torch.zeros(fixed_points["bc_target"].shape[0], device=DEVICE, dtype=DTYPE)),
        "log_w_f": torch.nn.Parameter(torch.zeros(fixed_points["f_coords"][0].shape[0], device=DEVICE, dtype=DTYPE)),
    })

def sa_loss(model, cfg, fixed_points, sa_params):
    pred_ic = model(sa_coords_to_input(fixed_points["ic_coords"]))
    pred_bc = model(sa_coords_to_input(fixed_points["bc_coords"]))

    ic_point = sa_pointwise_data_loss(pred_ic, fixed_points["ic_target"], cfg)
    bc_point = sa_pointwise_data_loss(pred_bc, fixed_points["bc_target"], cfg)
    f_point = sa_pointwise_residual_loss(model, fixed_points["f_coords"], cfg)

    w_ic = torch.exp(sa_params["log_w_ic"])
    w_bc = torch.exp(sa_params["log_w_bc"])
    w_f = torch.exp(sa_params["log_w_f"])

    loss_ic = ic_point.mean()
    loss_bc = bc_point.mean()
    loss_pde = f_point.mean()

    loss_ic_sa = (w_ic * ic_point).mean()
    loss_bc_sa = (w_bc * bc_point).mean()
    loss_pde_sa = (w_f * f_point).mean()

    loss = cfg.w_ic * loss_ic_sa + cfg.w_bc * loss_bc_sa + cfg.w_pde * loss_pde_sa

    parts = {
        "ic": loss_ic.detach(),
        "bc": loss_bc.detach(),
        "pde": loss_pde.detach(),
        "weighted_pde": loss_pde_sa.detach(),
        "ic_sa": loss_ic_sa.detach(),
        "bc_sa": loss_bc_sa.detach(),
        "pde_sa": loss_pde_sa.detach(),
        "w_ic_mean": w_ic.detach().mean(),
        "w_bc_mean": w_bc.detach().mean(),
        "w_f_mean": w_f.detach().mean(),
        "w_ic_max": w_ic.detach().max(),
        "w_bc_max": w_bc.detach().max(),
        "w_f_max": w_f.detach().max(),
    }

    return loss, parts

def sa_numpy_tuple(coords):
    return [v.detach().cpu().numpy() for v in coords]

def train_sa_pinn_fixed(
    model,
    cfg,
    sa_weight_lr=5.0e-3,
    sa_weight_max=100.0,
):
    model.train()

    fixed_points = sa_make_fixed_points(cfg)
    sa_params = sa_make_params(fixed_points)

    opt_sa = torch.optim.Adam(sa_params.parameters(), lr=float(sa_weight_lr))

    log_max = float(np.log(sa_weight_max))
    history = []

    phases = [
        ("sa_warmup", int(cfg.warmup_iters), float(cfg.lr_warmup), float(cfg.lr_warmup) * 0.05),
        ("sa_main", int(cfg.gated_iters), float(cfg.lr_gated), float(cfg.lr_gated) * 0.03),
    ]

    global_iter = 0
    total_iters = int(cfg.warmup_iters) + int(cfg.gated_iters)

    for phase_name, n_iters, lr, eta_min in phases:
        if n_iters <= 0:
            continue

        opt_model = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=float(getattr(cfg, "weight_decay", 0.0)),
        )

        sch = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt_model,
            T_max=n_iters,
            eta_min=eta_min,
        )

        hist_every = int(getattr(cfg, "history_every", 500))
        print_every = int(getattr(cfg, "print_every", 1000))

        print(f"[SA-PINN] phase={phase_name}, iters={n_iters}, lr={lr:.3e}, sa_lr={sa_weight_lr:.3e}")

        for local_it in range(1, n_iters + 1):
            global_iter += 1

            opt_model.zero_grad(set_to_none=True)
            opt_sa.zero_grad(set_to_none=True)

            loss, parts = sa_loss(model, cfg, fixed_points, sa_params)
            loss.backward()

            for p in sa_params.parameters():
                if p.grad is not None:
                    p.grad.mul_(-1.0)

            torch.nn.utils.clip_grad_norm_(model.parameters(), float(getattr(cfg, "grad_clip", 1.0)))

            opt_model.step()
            opt_sa.step()
            sch.step()

            with torch.no_grad():
                for p in sa_params.parameters():
                    p.clamp_(0.0, log_max)

            if local_it == 1 or local_it % hist_every == 0 or local_it == n_iters:
                history.append({
                    "phase": phase_name,
                    "local_iter": int(local_it),
                    "global_iter": int(global_iter),
                    "loss": float(loss.detach().cpu()),
                    "ic": float(parts["ic"].cpu()),
                    "bc": float(parts["bc"].cpu()),
                    "pde": float(parts["pde"].cpu()),
                    "ic_sa": float(parts["ic_sa"].cpu()),
                    "bc_sa": float(parts["bc_sa"].cpu()),
                    "pde_sa": float(parts["pde_sa"].cpu()),
                    "weighted_pde": float(parts["weighted_pde"].cpu()),
                    "w_ic_mean": float(parts["w_ic_mean"].cpu()),
                    "w_bc_mean": float(parts["w_bc_mean"].cpu()),
                    "w_f_mean": float(parts["w_f_mean"].cpu()),
                    "w_ic_max": float(parts["w_ic_max"].cpu()),
                    "w_bc_max": float(parts["w_bc_max"].cpu()),
                    "w_f_max": float(parts["w_f_max"].cpu()),
                    "lr": float(opt_model.param_groups[0]["lr"]),
                    "sa_lr": float(opt_sa.param_groups[0]["lr"]),
                })

            if local_it == 1 or local_it % print_every == 0 or local_it == n_iters:
                print(
                    f"[SA-PINN] {global_iter:6d}/{total_iters} "
                    f"loss={float(loss.detach().cpu()):.3e} "
                    f"ic={float(parts['ic'].cpu()):.1e} "
                    f"bc={float(parts['bc'].cpu()):.1e} "
                    f"pde={float(parts['pde'].cpu()):.1e} "
                    f"wf_mean={float(parts['w_f_mean'].cpu()):.2f}"
                )

    history_df = pd.DataFrame(history)

    diagnostics = {
        "sa_weight_lr": float(sa_weight_lr),
        "sa_weight_max": float(sa_weight_max),
        "w_ic_mean": float(torch.exp(sa_params["log_w_ic"]).detach().mean().cpu()),
        "w_bc_mean": float(torch.exp(sa_params["log_w_bc"]).detach().mean().cpu()),
        "w_f_mean": float(torch.exp(sa_params["log_w_f"]).detach().mean().cpu()),
        "w_ic_max": float(torch.exp(sa_params["log_w_ic"]).detach().max().cpu()),
        "w_bc_max": float(torch.exp(sa_params["log_w_bc"]).detach().max().cpu()),
        "w_f_max": float(torch.exp(sa_params["log_w_f"]).detach().max().cpu()),
        "n_ic": int(fixed_points["ic_target"].shape[0]),
        "n_bc": int(fixed_points["bc_target"].shape[0]),
        "n_f": int(fixed_points["f_coords"][0].shape[0]),
    }

    return model, sa_params, fixed_points, history_df, diagnostics

def save_sa_diagnostics(run_dir, sa_params, fixed_points, history_df, diagnostics):
    diag_dir = run_dir / "diagnostics"
    diag_dir.mkdir(exist_ok=True)

    history_df.to_csv(diag_dir / "sa_weight_history.csv", index=False)
    save_json(diagnostics, diag_dir / "sa_weight_summary.json")

    np.savez_compressed(
        diag_dir / "sa_weights_final.npz",
        log_w_ic=sa_params["log_w_ic"].detach().cpu().numpy(),
        log_w_bc=sa_params["log_w_bc"].detach().cpu().numpy(),
        log_w_f=sa_params["log_w_f"].detach().cpu().numpy(),
        w_ic=torch.exp(sa_params["log_w_ic"]).detach().cpu().numpy(),
        w_bc=torch.exp(sa_params["log_w_bc"]).detach().cpu().numpy(),
        w_f=torch.exp(sa_params["log_w_f"]).detach().cpu().numpy(),
    )

    arrays = {}

    for i, arr in enumerate(sa_numpy_tuple(fixed_points["ic_coords"])):
        arrays[f"ic_coord_{i}"] = arr
    arrays["ic_target"] = fixed_points["ic_target"].detach().cpu().numpy()

    for i, arr in enumerate(sa_numpy_tuple(fixed_points["bc_coords"])):
        arrays[f"bc_coord_{i}"] = arr
    arrays["bc_target"] = fixed_points["bc_target"].detach().cpu().numpy()

    for i, arr in enumerate(sa_numpy_tuple(fixed_points["f_coords"])):
        arrays[f"f_coord_{i}"] = arr

    np.savez_compressed(diag_dir / "sa_training_points_fixed.npz", **arrays)

    torch.save(
        {
            "log_w_ic": sa_params["log_w_ic"].detach().cpu(),
            "log_w_bc": sa_params["log_w_bc"].detach().cpu(),
            "log_w_f": sa_params["log_w_f"].detach().cpu(),
            "diagnostics": diagnostics,
        },
        diag_dir / "sa_weight_state_final.pt",
    )

def train_sa_one_seed(
    equation_name,
    base_cfg,
    seed,
    ref=None,
    total_iters=ADAM_TOTAL_ITERS,
    skip_if_done=True,
    sa_weight_lr=5.0e-3,
    sa_weight_max=100.0,
):
    cfg_s = prepare_cfg_for_seed(base_cfg, seed, total_iters=total_iters)

    configure_runtime(cfg_s)
    set_seed(int(seed))

    method_name = "SA-PINN"
    run_dir = get_run_dir(equation_name, method_name, seed)

    if skip_if_done and (run_dir / "_SUCCESS").exists():
        return pd.DataFrame([load_json(run_dir / "metrics_final.json")])

    try:
        save_run_header(
            run_dir,
            equation_name,
            method_name,
            cfg_s,
            {
                "method": "SA-PINN",
                "description": "Self-adaptive PINN with trainable pointwise soft-attention weights.",
                "optimizer_main": "AdamW",
                "optimizer_sa_weights": "Adam",
                "lbfgs_used": False,
                "adam_total_iters": int(cfg_s.warmup_iters + cfg_s.gated_iters),
                "fixed_training_points": True,
                "sa_mask": "exp(log_weight)",
                "sa_weight_init": 1.0,
                "sa_weight_lr": float(sa_weight_lr),
                "sa_weight_max": float(sa_weight_max),
                "n_ic": int(cfg_s.n_ic),
                "n_bc": int(cfg_s.n_bc),
                "n_f": int(cfg_s.n_f),
            },
        )

        model = MLP(cfg_s).to(DEVICE, dtype=DTYPE)

        t0 = time.time()

        model, sa_params, fixed_points, history_df, diagnostics = train_sa_pinn_fixed(
            model=model,
            cfg=cfg_s,
            sa_weight_lr=sa_weight_lr,
            sa_weight_max=sa_weight_max,
        )

        train_time = time.time() - t0

        metrics = evaluate_model_any(model, method_name, cfg_s, ref=ref)

        metrics.update({
            "equation": equation_name,
            "method": method_name,
            "seed": int(seed),
            "optimizer_main": "AdamW",
            "optimizer_sa_weights": "Adam",
            "lbfgs_used": False,
            "wall_clock_sec_total": float(train_time),
            "num_parameters": parameter_count(model),
            "sa_weight_lr": float(sa_weight_lr),
            "sa_weight_max": float(sa_weight_max),
            "sa_w_ic_mean": diagnostics["w_ic_mean"],
            "sa_w_bc_mean": diagnostics["w_bc_mean"],
            "sa_w_f_mean": diagnostics["w_f_mean"],
            "sa_w_ic_max": diagnostics["w_ic_max"],
            "sa_w_bc_max": diagnostics["w_bc_max"],
            "sa_w_f_max": diagnostics["w_f_max"],
        })

        save_history(history_df, run_dir, "history.csv")
        save_json(metrics, run_dir / "metrics_adamw.json")
        save_json(metrics, run_dir / "metrics_final.json")

        save_model_checkpoint(
            model,
            run_dir,
            "model_adamw_final.pt",
            extra=metrics,
        )

        save_model_checkpoint(
            model,
            run_dir,
            "model_final.pt",
            extra=metrics,
        )

        try_save_predictions(model, cfg_s, run_dir)
        save_sa_diagnostics(run_dir, sa_params, fixed_points, history_df, diagnostics)

        mark_status(run_dir, "success", {"wall_clock_sec_total": float(train_time)})

        del model, sa_params, fixed_points

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        gc.collect()

        return pd.DataFrame([metrics])

    except Exception as e:
        mark_status(run_dir, "failed", {"error": repr(e)})
        raise

def run_sa_pinn_seeds(
    equation_name,
    base_cfg,
    seeds,
    total_iters=ADAM_TOTAL_ITERS,
    skip_if_done=True,
    sa_weight_lr=5.0e-3,
    sa_weight_max=100.0,
):
    cfg_ref = copy.deepcopy(base_cfg)

    configure_runtime(cfg_ref)

    ref = build_reference_if_needed(cfg_ref)

    rows = []

    for seed in seeds:
        df_seed = train_sa_one_seed(
            equation_name=equation_name,
            base_cfg=base_cfg,
            seed=seed,
            ref=ref,
            total_iters=total_iters,
            skip_if_done=skip_if_done,
            sa_weight_lr=sa_weight_lr,
            sa_weight_max=sa_weight_max,
        )

        rows.append(df_seed)

    df = pd.concat(rows, ignore_index=True)

    summary_dir = RUNS_ROOT / safe_name(equation_name) / "_summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    out_path = summary_dir / "sa_pinn_metrics.csv"
    df.to_csv(out_path, index=False)

    print("saved:", out_path)

    return df

import time
import gc
import math
import numpy as np
import pandas as pd
import torch

def lra_get_weight_attrs(cfg):
    if not hasattr(cfg, "w_ic"):
        raise AttributeError("cfg.w_ic is missing.")
    if not hasattr(cfg, "w_bc"):
        raise AttributeError("cfg.w_bc is missing.")

    if hasattr(cfg, "w_pde"):
        pde_attr = "w_pde"
    elif hasattr(cfg, "w_f"):
        pde_attr = "w_f"
    else:
        raise AttributeError("cfg.w_pde or cfg.w_f is missing.")

    return "w_ic", "w_bc", pde_attr

def lra_read_loss_weights(cfg):
    ic_attr, bc_attr, pde_attr = lra_get_weight_attrs(cfg)
    return {
        "ic_attr": ic_attr,
        "bc_attr": bc_attr,
        "pde_attr": pde_attr,
        "w_ic": float(getattr(cfg, ic_attr)),
        "w_bc": float(getattr(cfg, bc_attr)),
        "w_pde": float(getattr(cfg, pde_attr)),
    }

def lra_set_loss_weights(cfg, w_ic, w_bc, w_pde):
    ic_attr, bc_attr, pde_attr = lra_get_weight_attrs(cfg)
    setattr(cfg, ic_attr, float(w_ic))
    setattr(cfg, bc_attr, float(w_bc))
    setattr(cfg, pde_attr, float(w_pde))

def lra_loss_with_weights(model, cfg, w_ic, w_bc, w_pde):
    old = lra_read_loss_weights(cfg)
    try:
        lra_set_loss_weights(cfg, w_ic, w_bc, w_pde)
        loss, parts = vanilla_loss(model, cfg)
    finally:
        lra_set_loss_weights(cfg, old["w_ic"], old["w_bc"], old["w_pde"])
    return loss, parts

def lra_grad_abs_stats(loss, params):
    if not torch.is_tensor(loss) or not loss.requires_grad:
        return {"max_abs": 0.0, "mean_abs": 0.0, "numel": 0}

    grads = torch.autograd.grad(
        loss,
        params,
        retain_graph=False,
        create_graph=False,
        allow_unused=True,
    )

    max_abs = 0.0
    sum_abs = 0.0
    numel = 0

    for g in grads:
        if g is None:
            continue
        a = g.detach().abs()
        if a.numel() == 0:
            continue
        max_abs = max(max_abs, float(a.max().cpu()))
        sum_abs += float(a.sum().cpu())
        numel += int(a.numel())

    mean_abs = sum_abs / max(numel, 1)

    return {
        "max_abs": float(max_abs),
        "mean_abs": float(mean_abs),
        "numel": int(numel),
    }

def lra_part_float(parts, key):
    if parts is None or key not in parts:
        return float("nan")
    v = parts[key]
    if torch.is_tensor(v):
        return float(v.detach().cpu())
    try:
        return float(v)
    except Exception:
        return float("nan")

def lra_compute_lambda_update(
    model,
    cfg,
    base_weights,
    lambdas,
    alpha=0.9,
    lambda_min=1.0e-4,
    lambda_max=1.0e4,
):
    params = [p for p in model.parameters() if p.requires_grad]

    loss_pde, _ = lra_loss_with_weights(
        model,
        cfg,
        w_ic=0.0,
        w_bc=0.0,
        w_pde=base_weights["w_pde"],
    )
    g_pde = lra_grad_abs_stats(loss_pde, params)

    loss_ic, _ = lra_loss_with_weights(
        model,
        cfg,
        w_ic=base_weights["w_ic"],
        w_bc=0.0,
        w_pde=0.0,
    )
    g_ic = lra_grad_abs_stats(loss_ic, params)

    loss_bc, _ = lra_loss_with_weights(
        model,
        cfg,
        w_ic=0.0,
        w_bc=base_weights["w_bc"],
        w_pde=0.0,
    )
    g_bc = lra_grad_abs_stats(loss_bc, params)

    eps = 1.0e-12

    hat_ic = g_pde["max_abs"] / max(g_ic["mean_abs"], eps)
    hat_bc = g_pde["max_abs"] / max(g_bc["mean_abs"], eps)

    hat_ic = float(np.clip(hat_ic, lambda_min, lambda_max))
    hat_bc = float(np.clip(hat_bc, lambda_min, lambda_max))

    new_ic = (1.0 - float(alpha)) * float(lambdas["ic"]) + float(alpha) * hat_ic
    new_bc = (1.0 - float(alpha)) * float(lambdas["bc"]) + float(alpha) * hat_bc

    new_ic = float(np.clip(new_ic, lambda_min, lambda_max))
    new_bc = float(np.clip(new_bc, lambda_min, lambda_max))

    stats = {
        "lambda_ic": new_ic,
        "lambda_bc": new_bc,
        "lambda_hat_ic": hat_ic,
        "lambda_hat_bc": hat_bc,
        "grad_pde_max": g_pde["max_abs"],
        "grad_pde_mean": g_pde["mean_abs"],
        "grad_ic_max": g_ic["max_abs"],
        "grad_ic_mean": g_ic["mean_abs"],
        "grad_bc_max": g_bc["max_abs"],
        "grad_bc_mean": g_bc["mean_abs"],
    }

    return {"ic": new_ic, "bc": new_bc}, stats

def train_lra_pinn_fixed(
    model,
    cfg,
    lra_alpha=0.9,
    lra_update_period=500,
    lra_lambda_min=1.0e-4,
    lra_lambda_max=1.0e4,
):
    model.train()

    base_weights = lra_read_loss_weights(cfg)
    lambdas = {"ic": 1.0, "bc": 1.0}

    history = []
    lambda_history = []

    phases = [
        ("lra_warmup", int(cfg.warmup_iters), float(cfg.lr_warmup), float(cfg.lr_warmup) * 0.05),
        ("lra_main", int(cfg.gated_iters), float(cfg.lr_gated), float(cfg.lr_gated) * 0.03),
    ]

    total_iters = int(cfg.warmup_iters) + int(cfg.gated_iters)
    global_iter = 0

    for phase_name, n_iters, lr, eta_min in phases:
        if n_iters <= 0:
            continue

        opt = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=float(getattr(cfg, "weight_decay", 0.0)),
        )

        sch = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt,
            T_max=n_iters,
            eta_min=eta_min,
        )

        hist_every = int(getattr(cfg, "history_every", 500))
        print_every = int(getattr(cfg, "print_every", 1000))

        print(f"[LRA-PINN] phase={phase_name}, iters={n_iters}, lr={lr:.3e}")

        for local_it in range(1, n_iters + 1):
            global_iter += 1

            if global_iter == 1 or global_iter % int(lra_update_period) == 0:
                lambdas, stats = lra_compute_lambda_update(
                    model=model,
                    cfg=cfg,
                    base_weights=base_weights,
                    lambdas=lambdas,
                    alpha=lra_alpha,
                    lambda_min=lra_lambda_min,
                    lambda_max=lra_lambda_max,
                )

                stats.update({
                    "phase": phase_name,
                    "local_iter": int(local_it),
                    "global_iter": int(global_iter),
                    "base_w_ic": float(base_weights["w_ic"]),
                    "base_w_bc": float(base_weights["w_bc"]),
                    "base_w_pde": float(base_weights["w_pde"]),
                    "effective_w_ic": float(base_weights["w_ic"] * lambdas["ic"]),
                    "effective_w_bc": float(base_weights["w_bc"] * lambdas["bc"]),
                    "effective_w_pde": float(base_weights["w_pde"]),
                })

                lambda_history.append(stats)

            opt.zero_grad(set_to_none=True)

            loss, parts = lra_loss_with_weights(
                model,
                cfg,
                w_ic=base_weights["w_ic"] * lambdas["ic"],
                w_bc=base_weights["w_bc"] * lambdas["bc"],
                w_pde=base_weights["w_pde"],
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                float(getattr(cfg, "grad_clip", 1.0)),
            )

            opt.step()
            sch.step()

            if local_it == 1 or local_it % hist_every == 0 or local_it == n_iters:
                history.append({
                    "phase": phase_name,
                    "local_iter": int(local_it),
                    "global_iter": int(global_iter),
                    "loss": float(loss.detach().cpu()),
                    "ic": lra_part_float(parts, "ic"),
                    "bc": lra_part_float(parts, "bc"),
                    "pde": lra_part_float(parts, "pde"),
                    "weighted_pde": lra_part_float(parts, "weighted_pde"),
                    "lambda_ic": float(lambdas["ic"]),
                    "lambda_bc": float(lambdas["bc"]),
                    "base_w_ic": float(base_weights["w_ic"]),
                    "base_w_bc": float(base_weights["w_bc"]),
                    "base_w_pde": float(base_weights["w_pde"]),
                    "effective_w_ic": float(base_weights["w_ic"] * lambdas["ic"]),
                    "effective_w_bc": float(base_weights["w_bc"] * lambdas["bc"]),
                    "effective_w_pde": float(base_weights["w_pde"]),
                    "lr": float(opt.param_groups[0]["lr"]),
                })

            if local_it == 1 or local_it % print_every == 0 or local_it == n_iters:
                print(
                    f"[LRA-PINN] {global_iter:6d}/{total_iters} "
                    f"loss={float(loss.detach().cpu()):.3e} "
                    f"ic={lra_part_float(parts, 'ic'):.1e} "
                    f"bc={lra_part_float(parts, 'bc'):.1e} "
                    f"pde={lra_part_float(parts, 'pde'):.1e} "
                    f"lam_ic={lambdas['ic']:.2e} "
                    f"lam_bc={lambdas['bc']:.2e}"
                )

    history_df = pd.DataFrame(history)
    lambda_df = pd.DataFrame(lambda_history)

    final_info = {
        "lambda_ic": float(lambdas["ic"]),
        "lambda_bc": float(lambdas["bc"]),
        "base_w_ic": float(base_weights["w_ic"]),
        "base_w_bc": float(base_weights["w_bc"]),
        "base_w_pde": float(base_weights["w_pde"]),
        "effective_w_ic": float(base_weights["w_ic"] * lambdas["ic"]),
        "effective_w_bc": float(base_weights["w_bc"] * lambdas["bc"]),
        "effective_w_pde": float(base_weights["w_pde"]),
        "lra_alpha": float(lra_alpha),
        "lra_update_period": int(lra_update_period),
        "lra_lambda_min": float(lra_lambda_min),
        "lra_lambda_max": float(lra_lambda_max),
    }

    return model, history_df, lambda_df, final_info

def save_lra_diagnostics(run_dir, lambda_df, final_info):
    diag_dir = run_dir / "diagnostics"
    diag_dir.mkdir(exist_ok=True)

    lambda_df.to_csv(diag_dir / "lra_lambda_history.csv", index=False)
    lambda_df.to_csv(diag_dir / "lra_gradient_stats.csv", index=False)
    save_json(final_info, diag_dir / "lra_final_lambdas.json")

def train_lra_one_seed(
    equation_name,
    base_cfg,
    seed,
    ref=None,
    total_iters=ADAM_TOTAL_ITERS,
    skip_if_done=True,
    lra_alpha=0.9,
    lra_update_period=500,
    lra_lambda_min=1.0e-4,
    lra_lambda_max=1.0e4,
):
    cfg_s = prepare_cfg_for_seed(base_cfg, seed, total_iters=total_iters)

    configure_runtime(cfg_s)
    set_seed(int(seed))

    method_name = "LRA-PINN"
    run_dir = get_run_dir(equation_name, method_name, seed)

    if skip_if_done and (run_dir / "_SUCCESS").exists():
        return pd.DataFrame([load_json(run_dir / "metrics_final.json")])

    try:
        base_weights = lra_read_loss_weights(cfg_s)

        save_run_header(
            run_dir,
            equation_name,
            method_name,
            cfg_s,
            {
                "method": "LRA-PINN",
                "description": "Learning-rate annealing PINN using gradient-statistics-based adaptive IC/BC loss multipliers.",
                "optimizer_main": "AdamW",
                "lbfgs_used": False,
                "adam_total_iters": int(cfg_s.warmup_iters + cfg_s.gated_iters),
                "base_w_ic": float(base_weights["w_ic"]),
                "base_w_bc": float(base_weights["w_bc"]),
                "base_w_pde": float(base_weights["w_pde"]),
                "lra_alpha": float(lra_alpha),
                "lra_update_period": int(lra_update_period),
                "lra_lambda_min": float(lra_lambda_min),
                "lra_lambda_max": float(lra_lambda_max),
                "adaptive_terms": ["ic", "bc"],
                "fixed_term": "pde",
            },
        )

        model = MLP(cfg_s).to(DEVICE, dtype=DTYPE)

        t0 = time.time()

        model, history_df, lambda_df, final_info = train_lra_pinn_fixed(
            model=model,
            cfg=cfg_s,
            lra_alpha=lra_alpha,
            lra_update_period=lra_update_period,
            lra_lambda_min=lra_lambda_min,
            lra_lambda_max=lra_lambda_max,
        )

        train_time = time.time() - t0

        metrics = evaluate_model_any(model, method_name, cfg_s, ref=ref)

        metrics.update({
            "equation": equation_name,
            "method": method_name,
            "seed": int(seed),
            "optimizer_main": "AdamW",
            "lbfgs_used": False,
            "wall_clock_sec_total": float(train_time),
            "num_parameters": parameter_count(model),
            "lra_alpha": float(lra_alpha),
            "lra_update_period": int(lra_update_period),
            "lra_lambda_ic": float(final_info["lambda_ic"]),
            "lra_lambda_bc": float(final_info["lambda_bc"]),
            "lra_effective_w_ic": float(final_info["effective_w_ic"]),
            "lra_effective_w_bc": float(final_info["effective_w_bc"]),
            "lra_effective_w_pde": float(final_info["effective_w_pde"]),
        })

        save_history(history_df, run_dir, "history.csv")
        save_json(metrics, run_dir / "metrics_adamw.json")
        save_json(metrics, run_dir / "metrics_final.json")

        save_model_checkpoint(
            model,
            run_dir,
            "model_adamw_final.pt",
            extra=metrics,
        )

        save_model_checkpoint(
            model,
            run_dir,
            "model_final.pt",
            extra=metrics,
        )

        try_save_predictions(model, cfg_s, run_dir)
        save_lra_diagnostics(run_dir, lambda_df, final_info)

        mark_status(run_dir, "success", {"wall_clock_sec_total": float(train_time)})

        del model

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        gc.collect()

        return pd.DataFrame([metrics])

    except Exception as e:
        mark_status(run_dir, "failed", {"error": repr(e)})
        raise

def run_lra_pinn_seeds(
    equation_name,
    base_cfg,
    seeds,
    total_iters=ADAM_TOTAL_ITERS,
    skip_if_done=True,
    lra_alpha=0.9,
    lra_update_period=500,
    lra_lambda_min=1.0e-4,
    lra_lambda_max=1.0e4,
):
    cfg_ref = copy.deepcopy(base_cfg)

    configure_runtime(cfg_ref)

    ref = build_reference_if_needed(cfg_ref)

    rows = []

    for seed in seeds:
        df_seed = train_lra_one_seed(
            equation_name=equation_name,
            base_cfg=base_cfg,
            seed=seed,
            ref=ref,
            total_iters=total_iters,
            skip_if_done=skip_if_done,
            lra_alpha=lra_alpha,
            lra_update_period=lra_update_period,
            lra_lambda_min=lra_lambda_min,
            lra_lambda_max=lra_lambda_max,
        )

        rows.append(df_seed)

    df = pd.concat(rows, ignore_index=True)

    summary_dir = RUNS_ROOT / safe_name(equation_name) / "_summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    out_path = summary_dir / "lra_pinn_metrics.csv"
    df.to_csv(out_path, index=False)

    print("saved:", out_path)

    return df

import time
import gc
import numpy as np
import pandas as pd
import torch

def require_gpinn_dependencies():
    required = [
        "vanilla_loss",
        "sample_f",
        "gpinn_residual_scaled_components",
        "gpinn_gradient_loss_from_components",
        "evaluate_model_any",
        "save_run_header",
        "save_model_checkpoint",
        "try_save_predictions",
    ]

    missing = [name for name in required if name not in globals()]
    if missing:
        raise RuntimeError("missing required definitions: " + ", ".join(missing))

def gpinn_sub_slice_coords(coords, start, end):
    return tuple(v[start:end] for v in coords)

def gpinn_sub_gradient_backward(
    model,
    cfg,
    gpinn_weight=1.0e-4,
    gpinn_n_g=1024,
    gpinn_grad_batch_size=512,
):
    coords = sample_f(int(gpinn_n_g), cfg)
    coords = tuple(v.detach().clone().to(DEVICE, dtype=DTYPE) for v in coords)

    n = int(coords[0].shape[0])
    grad_loss_value = 0.0

    for start in range(0, n, int(gpinn_grad_batch_size)):
        end = min(start + int(gpinn_grad_batch_size), n)
        coords_b = gpinn_sub_slice_coords(coords, start, end)

        comps_b, coords_req = gpinn_residual_scaled_components(model, coords_b, cfg)
        grad_loss_b = gpinn_gradient_loss_from_components(comps_b, coords_req)

        weight = float(end - start) / float(n)
        scaled_loss = float(cfg.w_pde) * float(gpinn_weight) * weight * grad_loss_b

        scaled_loss.backward()

        grad_loss_value += weight * float(grad_loss_b.detach().cpu())

        del comps_b, coords_req, grad_loss_b, scaled_loss

    return float(grad_loss_value)

def train_gpinn_subsampled_fixed(
    model,
    cfg,
    gpinn_weight=1.0e-4,
    gpinn_n_g=1024,
    gpinn_grad_batch_size=512,
):
    require_gpinn_dependencies()

    model.train()

    history = []
    total_iters = int(cfg.warmup_iters) + int(cfg.gated_iters)
    global_iter = 0

    phases = [
        ("gpinn_sub_warmup", int(cfg.warmup_iters), float(cfg.lr_warmup), float(cfg.lr_warmup) * 0.05),
        ("gpinn_sub_main", int(cfg.gated_iters), float(cfg.lr_gated), float(cfg.lr_gated) * 0.03),
    ]

    for phase_name, n_iters, lr, eta_min in phases:
        if n_iters <= 0:
            continue

        opt = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=float(getattr(cfg, "weight_decay", 0.0)),
        )

        sch = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt,
            T_max=n_iters,
            eta_min=eta_min,
        )

        hist_every = int(getattr(cfg, "history_every", 500))
        print_every = int(getattr(cfg, "print_every", 1000))

        print(f"[gPINN-sub] phase={phase_name}, iters={n_iters}, lr={lr:.3e}, n_g={gpinn_n_g}, w_g={gpinn_weight:.1e}")

        for local_it in range(1, n_iters + 1):
            global_iter += 1

            opt.zero_grad(set_to_none=True)

            loss_base, parts = vanilla_loss(model, cfg)
            loss_base.backward()

            grad_loss_value = gpinn_sub_gradient_backward(
                model=model,
                cfg=cfg,
                gpinn_weight=gpinn_weight,
                gpinn_n_g=gpinn_n_g,
                gpinn_grad_batch_size=gpinn_grad_batch_size,
            )

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                float(getattr(cfg, "grad_clip", 1.0)),
            )

            opt.step()
            sch.step()

            loss_total_value = float(loss_base.detach().cpu()) + float(cfg.w_pde) * float(gpinn_weight) * grad_loss_value

            if local_it == 1 or local_it % hist_every == 0 or local_it == n_iters:
                history.append({
                    "phase": phase_name,
                    "local_iter": int(local_it),
                    "global_iter": int(global_iter),
                    "loss": float(loss_total_value),
                    "base_loss": float(loss_base.detach().cpu()),
                    "ic": float(parts["ic"].detach().cpu()),
                    "bc": float(parts["bc"].detach().cpu()),
                    "pde": float(parts["pde"].detach().cpu()),
                    "weighted_pde": float(parts["weighted_pde"].detach().cpu()),
                    "gpinn_grad_loss": float(grad_loss_value),
                    "gpinn_weighted_grad_loss": float(cfg.w_pde) * float(gpinn_weight) * float(grad_loss_value),
                    "gpinn_weight": float(gpinn_weight),
                    "gpinn_n_g": int(gpinn_n_g),
                    "gpinn_grad_batch_size": int(gpinn_grad_batch_size),
                    "lr": float(opt.param_groups[0]["lr"]),
                })

            if local_it == 1 or local_it % print_every == 0 or local_it == n_iters:
                print(
                    f"[gPINN-sub] {global_iter:6d}/{total_iters} "
                    f"loss={loss_total_value:.3e} "
                    f"ic={float(parts['ic'].detach().cpu()):.1e} "
                    f"bc={float(parts['bc'].detach().cpu()):.1e} "
                    f"pde={float(parts['pde'].detach().cpu()):.1e} "
                    f"g={grad_loss_value:.1e}"
                )

            del loss_base, parts

    history_df = pd.DataFrame(history)

    final_info = {
        "gpinn_weight": float(gpinn_weight),
        "gpinn_n_g": int(gpinn_n_g),
        "gpinn_grad_batch_size": int(gpinn_grad_batch_size),
        "T_g": "independently sampled residual-gradient points each iteration",
        "T_g_equals_T_f": False,
        "base_loss": "vanilla PINN loss using full configured n_f, n_ic, n_bc",
        "gradient_terms": "first derivatives of scaled PDE residual components with respect to input coordinates",
        "higher_order_residual_gradients": False,
    }

    return model, history_df, final_info

def save_gpinn_subsampled_diagnostics(run_dir, history_df, final_info):
    diag_dir = run_dir / "diagnostics"
    diag_dir.mkdir(exist_ok=True)

    history_df.to_csv(diag_dir / "gpinn_gradient_history.csv", index=False)
    save_json(final_info, diag_dir / "gpinn_settings.json")

    cols = [
        c for c in [
            "global_iter",
            "gpinn_grad_loss",
            "gpinn_weighted_grad_loss",
            "pde",
            "loss",
            "gpinn_n_g",
        ]
        if c in history_df.columns
    ]

    if len(cols) > 0:
        history_df[cols].to_csv(diag_dir / "gpinn_gradient_stats.csv", index=False)

def train_gpinn_subsampled_one_seed(
    equation_name,
    base_cfg,
    seed,
    ref=None,
    total_iters=ADAM_TOTAL_ITERS,
    skip_if_done=True,
    gpinn_weight=1.0e-4,
    gpinn_n_g=1024,
    gpinn_grad_batch_size=512,
):
    cfg_s = prepare_cfg_for_seed(base_cfg, seed, total_iters=total_iters)

    configure_runtime(cfg_s)
    set_seed(int(seed))

    method_name = "gPINN-subsampled"
    run_dir = get_run_dir(equation_name, method_name, seed)

    if skip_if_done and (run_dir / "_SUCCESS").exists():
        return pd.DataFrame([load_json(run_dir / "metrics_final.json")])

    try:
        save_run_header(
            run_dir,
            equation_name,
            method_name,
            cfg_s,
            {
                "method": "gPINN-subsampled",
                "description": "Gradient-enhanced PINN with first derivatives of scaled PDE residuals evaluated on a subsampled Tg set.",
                "optimizer_main": "AdamW",
                "lbfgs_used": False,
                "adam_total_iters": int(cfg_s.warmup_iters + cfg_s.gated_iters),
                "gpinn_weight": float(gpinn_weight),
                "gpinn_n_g": int(gpinn_n_g),
                "gpinn_grad_batch_size": int(gpinn_grad_batch_size),
                "T_g": "independently sampled residual-gradient points each iteration",
                "T_g_equals_T_f": False,
                "base_n_f": int(cfg_s.n_f),
                "n_ic": int(cfg_s.n_ic),
                "n_bc": int(cfg_s.n_bc),
                "n_f": int(cfg_s.n_f),
            },
        )

        model = MLP(cfg_s).to(DEVICE, dtype=DTYPE)

        t0 = time.time()

        model, history_df, final_info = train_gpinn_subsampled_fixed(
            model=model,
            cfg=cfg_s,
            gpinn_weight=gpinn_weight,
            gpinn_n_g=gpinn_n_g,
            gpinn_grad_batch_size=gpinn_grad_batch_size,
        )

        train_time = time.time() - t0

        metrics = evaluate_model_any(model, method_name, cfg_s, ref=ref)

        metrics.update({
            "equation": equation_name,
            "method": method_name,
            "seed": int(seed),
            "optimizer_main": "AdamW",
            "lbfgs_used": False,
            "wall_clock_sec_total": float(train_time),
            "num_parameters": parameter_count(model),
            "gpinn_weight": float(gpinn_weight),
            "gpinn_n_g": int(gpinn_n_g),
            "gpinn_grad_batch_size": int(gpinn_grad_batch_size),
            "T_g_equals_T_f": False,
        })

        if len(history_df) > 0:
            metrics["gpinn_grad_loss_final_logged"] = float(history_df["gpinn_grad_loss"].iloc[-1])
            metrics["gpinn_weighted_grad_loss_final_logged"] = float(history_df["gpinn_weighted_grad_loss"].iloc[-1])

        save_history(history_df, run_dir, "history.csv")
        save_json(metrics, run_dir / "metrics_adamw.json")
        save_json(metrics, run_dir / "metrics_final.json")

        save_model_checkpoint(
            model,
            run_dir,
            "model_adamw_final.pt",
            extra=metrics,
        )

        save_model_checkpoint(
            model,
            run_dir,
            "model_final.pt",
            extra=metrics,
        )

        try_save_predictions(model, cfg_s, run_dir)
        save_gpinn_subsampled_diagnostics(run_dir, history_df, final_info)

        mark_status(run_dir, "success", {"wall_clock_sec_total": float(train_time)})

        del model

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        gc.collect()

        return pd.DataFrame([metrics])

    except Exception as e:
        mark_status(run_dir, "failed", {"error": repr(e)})
        raise

def run_gpinn_subsampled_seeds(
    equation_name,
    base_cfg,
    seeds,
    total_iters=ADAM_TOTAL_ITERS,
    skip_if_done=True,
    gpinn_weight=1.0e-4,
    gpinn_n_g=1024,
    gpinn_grad_batch_size=512,
):
    cfg_ref = copy.deepcopy(base_cfg)

    configure_runtime(cfg_ref)

    ref = build_reference_if_needed(cfg_ref)

    rows = []

    for seed in seeds:
        df_seed = train_gpinn_subsampled_one_seed(
            equation_name=equation_name,
            base_cfg=base_cfg,
            seed=seed,
            ref=ref,
            total_iters=total_iters,
            skip_if_done=skip_if_done,
            gpinn_weight=gpinn_weight,
            gpinn_n_g=gpinn_n_g,
            gpinn_grad_batch_size=gpinn_grad_batch_size,
        )

        rows.append(df_seed)

    df = pd.concat(rows, ignore_index=True)

    summary_dir = RUNS_ROOT / safe_name(equation_name) / "_summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    out_path = summary_dir / "gpinn_subsampled_metrics.csv"
    df.to_csv(out_path, index=False)

    print("saved:", out_path)

    return df

import time
import gc
import math
import shutil
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

class AAFScaledTanh(nn.Module):
    def __init__(self, a_getter, n):
        super().__init__()
        self.a_getter = a_getter
        self.n = float(n)

    def forward(self, x):
        a = torch.clamp(self.a_getter(), min=1.0e-8)
        return torch.tanh(self.n * a * x)

class AAFScaledSiLU(nn.Module):
    def __init__(self, a_getter, n):
        super().__init__()
        self.a_getter = a_getter
        self.n = float(n)

    def forward(self, x):
        a = torch.clamp(self.a_getter(), min=1.0e-8)
        return F.silu(self.n * a * x)

def aaf_replace_activations(module, a_getter, n):
    count = 0

    for name, child in list(module.named_children()):
        if isinstance(child, nn.Tanh):
            setattr(module, name, AAFScaledTanh(a_getter, n))
            count += 1
        elif isinstance(child, nn.SiLU):
            setattr(module, name, AAFScaledSiLU(a_getter, n))
            count += 1
        else:
            count += aaf_replace_activations(child, a_getter, n)

    return count

def make_aaf_model(cfg, aaf_n=5.0, aaf_init_na=1.0):
    model = MLP(cfg).to(DEVICE, dtype=DTYPE)

    a_init = float(aaf_init_na) / float(aaf_n)
    model.aaf_a = nn.Parameter(torch.tensor(a_init, device=DEVICE, dtype=DTYPE))
    model.aaf_n = float(aaf_n)
    model.aaf_init_na = float(aaf_init_na)

    n_replaced = aaf_replace_activations(
        model,
        a_getter=lambda: model.aaf_a,
        n=aaf_n,
    )

    if n_replaced <= 0:
        raise RuntimeError("No Tanh or SiLU activation was replaced by AAF activation.")

    model.aaf_num_activations = int(n_replaced)

    return model

def aaf_current_state(model):
    with torch.no_grad():
        a = float(torch.clamp(model.aaf_a, min=1.0e-8).detach().cpu())
        n = float(model.aaf_n)
        return {
            "a": a,
            "n": n,
            "na": n * a,
            "num_adaptive_activations": int(getattr(model, "aaf_num_activations", -1)),
        }

def aaf_clamp_a(model, aaf_na_min=1.0e-6, aaf_na_max=20.0):
    with torch.no_grad():
        n = float(model.aaf_n)
        a_min = float(aaf_na_min) / n
        a_max = float(aaf_na_max) / n
        model.aaf_a.clamp_(a_min, a_max)

def aaf_optimizer(model, cfg, lr):
    wd = float(getattr(cfg, "weight_decay", 0.0))

    aaf_params = []
    other_params = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name == "aaf_a":
            aaf_params.append(p)
        else:
            other_params.append(p)

    return torch.optim.AdamW(
        [
            {"params": other_params, "weight_decay": wd},
            {"params": aaf_params, "weight_decay": 0.0},
        ],
        lr=lr,
    )

def train_aaf_pinn_fixed(
    model,
    cfg,
    aaf_na_min=1.0e-6,
    aaf_na_max=20.0,
):
    model.train()

    history = []
    total_iters = int(cfg.warmup_iters) + int(cfg.gated_iters)
    global_iter = 0

    phases = [
        ("aaf_warmup", int(cfg.warmup_iters), float(cfg.lr_warmup), float(cfg.lr_warmup) * 0.05),
        ("aaf_main", int(cfg.gated_iters), float(cfg.lr_gated), float(cfg.lr_gated) * 0.03),
    ]

    for phase_name, n_iters, lr, eta_min in phases:
        if n_iters <= 0:
            continue

        opt = aaf_optimizer(model, cfg, lr)

        sch = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt,
            T_max=n_iters,
            eta_min=eta_min,
        )

        hist_every = int(getattr(cfg, "history_every", 500))
        print_every = int(getattr(cfg, "print_every", 1000))

        print(f"[AAF-PINN] phase={phase_name}, iters={n_iters}, lr={lr:.3e}")

        for local_it in range(1, n_iters + 1):
            global_iter += 1

            opt.zero_grad(set_to_none=True)

            loss, parts = vanilla_loss(model, cfg)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                float(getattr(cfg, "grad_clip", 1.0)),
            )

            opt.step()
            aaf_clamp_a(model, aaf_na_min=aaf_na_min, aaf_na_max=aaf_na_max)
            sch.step()

            state = aaf_current_state(model)

            if local_it == 1 or local_it % hist_every == 0 or local_it == n_iters:
                history.append({
                    "phase": phase_name,
                    "local_iter": int(local_it),
                    "global_iter": int(global_iter),
                    "loss": float(loss.detach().cpu()),
                    "ic": float(parts["ic"].detach().cpu()),
                    "bc": float(parts["bc"].detach().cpu()),
                    "pde": float(parts["pde"].detach().cpu()),
                    "weighted_pde": float(parts["weighted_pde"].detach().cpu()),
                    "aaf_a": float(state["a"]),
                    "aaf_n": float(state["n"]),
                    "aaf_na": float(state["na"]),
                    "num_adaptive_activations": int(state["num_adaptive_activations"]),
                    "lr": float(opt.param_groups[0]["lr"]),
                })

            if local_it == 1 or local_it % print_every == 0 or local_it == n_iters:
                print(
                    f"[AAF-PINN] {global_iter:6d}/{total_iters} "
                    f"loss={float(loss.detach().cpu()):.3e} "
                    f"ic={float(parts['ic'].detach().cpu()):.1e} "
                    f"bc={float(parts['bc'].detach().cpu()):.1e} "
                    f"pde={float(parts['pde'].detach().cpu()):.1e} "
                    f"na={state['na']:.3f}"
                )

    history_df = pd.DataFrame(history)

    final_state = aaf_current_state(model)
    final_info = {
        "aaf_n": float(final_state["n"]),
        "aaf_a": float(final_state["a"]),
        "aaf_na": float(final_state["na"]),
        "aaf_init_na": float(getattr(model, "aaf_init_na", 1.0)),
        "aaf_na_min": float(aaf_na_min),
        "aaf_na_max": float(aaf_na_max),
        "num_adaptive_activations": int(final_state["num_adaptive_activations"]),
        "loss_structure_changed": False,
        "slope_recovery_term_used": False,
        "activation_form": "sigma(n*a*preactivation)",
    }

    return model, history_df, final_info

def save_aaf_diagnostics(run_dir, history_df, final_info, model):
    diag_dir = run_dir / "diagnostics"
    diag_dir.mkdir(exist_ok=True)

    history_df.to_csv(diag_dir / "aaf_activation_history.csv", index=False)
    save_json(final_info, diag_dir / "aaf_settings.json")

    torch.save(
        {
            "aaf_a": model.aaf_a.detach().cpu(),
            "aaf_n": float(model.aaf_n),
            "aaf_na": float(model.aaf_n * model.aaf_a.detach().cpu()),
            "final_info": final_info,
        },
        diag_dir / "aaf_final_state.pt",
    )

def train_aaf_one_seed(
    equation_name,
    base_cfg,
    seed,
    ref=None,
    total_iters=ADAM_TOTAL_ITERS,
    skip_if_done=True,
    aaf_n=5.0,
    aaf_init_na=1.0,
    aaf_na_min=1.0e-6,
    aaf_na_max=20.0,
):
    cfg_s = prepare_cfg_for_seed(base_cfg, seed, total_iters=total_iters)

    configure_runtime(cfg_s)
    set_seed(int(seed))

    method_name = "AAF-PINN"
    run_dir = get_run_dir(equation_name, method_name, seed)

    if skip_if_done and (run_dir / "_SUCCESS").exists():
        return pd.DataFrame([load_json(run_dir / "metrics_final.json")])

    try:
        model = make_aaf_model(
            cfg_s,
            aaf_n=aaf_n,
            aaf_init_na=aaf_init_na,
        )

        save_run_header(
            run_dir,
            equation_name,
            method_name,
            cfg_s,
            {
                "method": "AAF-PINN",
                "description": "Adaptive activation function PINN with trainable global activation slope a in sigma(n*a*z).",
                "optimizer_main": "AdamW",
                "lbfgs_used": False,
                "adam_total_iters": int(cfg_s.warmup_iters + cfg_s.gated_iters),
                "loss_structure_changed": False,
                "slope_recovery_term_used": False,
                "aaf_n": float(aaf_n),
                "aaf_init_na": float(aaf_init_na),
                "aaf_initial_a": float(aaf_init_na / aaf_n),
                "aaf_na_min": float(aaf_na_min),
                "aaf_na_max": float(aaf_na_max),
                "num_adaptive_activations": int(model.aaf_num_activations),
                "activation_form": "sigma(n*a*preactivation)",
                "n_ic": int(cfg_s.n_ic),
                "n_bc": int(cfg_s.n_bc),
                "n_f": int(cfg_s.n_f),
            },
        )

        t0 = time.time()

        model, history_df, final_info = train_aaf_pinn_fixed(
            model=model,
            cfg=cfg_s,
            aaf_na_min=aaf_na_min,
            aaf_na_max=aaf_na_max,
        )

        train_time = time.time() - t0

        metrics = evaluate_model_any(model, method_name, cfg_s, ref=ref)

        metrics.update({
            "equation": equation_name,
            "method": method_name,
            "seed": int(seed),
            "optimizer_main": "AdamW",
            "lbfgs_used": False,
            "wall_clock_sec_total": float(train_time),
            "num_parameters": parameter_count(model),
            "aaf_n": float(final_info["aaf_n"]),
            "aaf_a": float(final_info["aaf_a"]),
            "aaf_na": float(final_info["aaf_na"]),
            "aaf_init_na": float(final_info["aaf_init_na"]),
            "aaf_na_min": float(aaf_na_min),
            "aaf_na_max": float(aaf_na_max),
            "num_adaptive_activations": int(final_info["num_adaptive_activations"]),
        })

        save_history(history_df, run_dir, "history.csv")
        save_json(metrics, run_dir / "metrics_adamw.json")
        save_json(metrics, run_dir / "metrics_final.json")

        save_model_checkpoint(
            model,
            run_dir,
            "model_adamw_final.pt",
            extra=metrics,
        )

        save_model_checkpoint(
            model,
            run_dir,
            "model_final.pt",
            extra=metrics,
        )

        try_save_predictions(model, cfg_s, run_dir)
        save_aaf_diagnostics(run_dir, history_df, final_info, model)

        mark_status(run_dir, "success", {"wall_clock_sec_total": float(train_time)})

        del model

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        gc.collect()

        return pd.DataFrame([metrics])

    except Exception as e:
        mark_status(run_dir, "failed", {"error": repr(e)})
        raise

def run_aaf_pinn_seeds(
    equation_name,
    base_cfg,
    seeds,
    total_iters=ADAM_TOTAL_ITERS,
    skip_if_done=True,
    aaf_n=5.0,
    aaf_init_na=1.0,
    aaf_na_min=1.0e-6,
    aaf_na_max=20.0,
):
    cfg_ref = copy.deepcopy(base_cfg)

    configure_runtime(cfg_ref)

    ref = build_reference_if_needed(cfg_ref)

    rows = []

    for seed in seeds:
        df_seed = train_aaf_one_seed(
            equation_name=equation_name,
            base_cfg=base_cfg,
            seed=seed,
            ref=ref,
            total_iters=total_iters,
            skip_if_done=skip_if_done,
            aaf_n=aaf_n,
            aaf_init_na=aaf_init_na,
            aaf_na_min=aaf_na_min,
            aaf_na_max=aaf_na_max,
        )

        rows.append(df_seed)

    df = pd.concat(rows, ignore_index=True)

    summary_dir = RUNS_ROOT / safe_name(equation_name) / "_summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    out_path = summary_dir / "aaf_pinn_metrics.csv"
    df.to_csv(out_path, index=False)

    print("saved:", out_path)

    return df

def reset_aaf_runs(equation_name, seeds):
    for seed in seeds:
        run_dir = get_run_dir(equation_name, "AAF-PINN", seed)
        if run_dir.exists():
            print("remove:", run_dir)
            shutil.rmtree(run_dir)
        else:
            print("not found:", run_dir)

import time
import gc
import math
import shutil
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

class CPINNScaledTanh(nn.Module):
    def __init__(self, a_getter, n):
        super().__init__()
        self.a_getter = a_getter
        self.n = float(n)

    def forward(self, x):
        a = torch.clamp(self.a_getter(), min=1.0e-8)
        return torch.tanh(self.n * a * x)

class CPINNScaledSiLU(nn.Module):
    def __init__(self, a_getter, n):
        super().__init__()
        self.a_getter = a_getter
        self.n = float(n)

    def forward(self, x):
        a = torch.clamp(self.a_getter(), min=1.0e-8)
        return F.silu(self.n * a * x)

def cpinn_replace_activations(module, a_getter, n):
    count = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Tanh):
            setattr(module, name, CPINNScaledTanh(a_getter, n))
            count += 1
        elif isinstance(child, nn.SiLU):
            setattr(module, name, CPINNScaledSiLU(a_getter, n))
            count += 1
        else:
            count += cpinn_replace_activations(child, a_getter, n)
    return count

def cpinn_make_submodel(cfg, use_local_adaptive_activation=True, aaf_n=5.0, aaf_init_na=1.0):
    model = MLP(cfg).to(DEVICE, dtype=DTYPE)
    model.cpinn_local_aaf_used = False
    model.cpinn_aaf_num_activations = 0

    if use_local_adaptive_activation:
        a_init = float(aaf_init_na) / float(aaf_n)
        model.cpinn_aaf_a = nn.Parameter(torch.tensor(a_init, device=DEVICE, dtype=DTYPE))
        model.cpinn_aaf_n = float(aaf_n)
        model.cpinn_aaf_init_na = float(aaf_init_na)
        n_replaced = cpinn_replace_activations(model, lambda: model.cpinn_aaf_a, aaf_n)
        model.cpinn_local_aaf_used = n_replaced > 0
        model.cpinn_aaf_num_activations = int(n_replaced)

    return model

def cpinn_cat(coords):
    if len(coords) == 2:
        return cat_xt(coords[0], coords[1])
    if len(coords) == 3:
        return cat_xyt(coords[0], coords[1], coords[2])
    raise RuntimeError(f"unsupported coordinate dimension: {len(coords)}")

def cpinn_get_attr(cfg, names, default=None):
    for name in names:
        if hasattr(cfg, name):
            return float(getattr(cfg, name))
    return default

def cpinn_get_bounds(cfg):
    x_min = cpinn_get_attr(cfg, ["x_min", "xmin", "x0", "x_left"], None)
    x_max = cpinn_get_attr(cfg, ["x_max", "xmax", "x1", "x_right"], None)
    y_min = cpinn_get_attr(cfg, ["y_min", "ymin", "y0", "y_bottom"], None)
    y_max = cpinn_get_attr(cfg, ["y_max", "ymax", "y1", "y_top"], None)
    t_min = cpinn_get_attr(cfg, ["t_min", "tmin", "t0"], 0.0)
    t_max = cpinn_get_attr(cfg, ["t_max", "tmax", "T", "t_final"], None)

    if x_min is None or x_max is None or t_max is None:
        pts = sample_f(4096, cfg)
        x_min = float(pts[0].detach().min().cpu()) if x_min is None else x_min
        x_max = float(pts[0].detach().max().cpu()) if x_max is None else x_max
        if len(pts) == 2:
            t_max = float(pts[1].detach().max().cpu()) if t_max is None else t_max
        else:
            y_min = float(pts[1].detach().min().cpu()) if y_min is None else y_min
            y_max = float(pts[1].detach().max().cpu()) if y_max is None else y_max
            t_max = float(pts[2].detach().max().cpu()) if t_max is None else t_max

    return {
        "x_min": float(x_min),
        "x_max": float(x_max),
        "y_min": None if y_min is None else float(y_min),
        "y_max": None if y_max is None else float(y_max),
        "t_min": float(t_min),
        "t_max": float(t_max),
    }

class CPINNModel(nn.Module):
    def __init__(self, submodels, spatial_dim, x_split, y_split=None):
        super().__init__()
        self.submodels = nn.ModuleList(submodels)
        self.spatial_dim = int(spatial_dim)
        self.x_split = float(x_split)
        self.y_split = None if y_split is None else float(y_split)

    def subdomain_index(self, z):
        x = z[:, 0:1]

        if self.spatial_dim == 1:
            return (x > self.x_split).long().reshape(-1)

        y = z[:, 1:2]
        ix = (x > self.x_split).long().reshape(-1)
        iy = (y > self.y_split).long().reshape(-1)
        return ix + 2 * iy

    def forward(self, z):
        idx = self.subdomain_index(z)
        out_all = None

        for k, net in enumerate(self.submodels):
            mask = idx == k
            if not torch.any(mask):
                continue
            out_k = net(z[mask])
            if out_all is None:
                out_all = z.new_zeros((z.shape[0], out_k.shape[1]))
            out_all[mask] = out_k

        if out_all is None:
            out_all = self.submodels[0](z)

        return out_all

    def forward_subdomain(self, subdomain_id, coords):
        z = cpinn_cat(coords)
        return self.submodels[int(subdomain_id)](z)

def make_cpinn_model(
    cfg,
    spatial_dim,
    use_local_adaptive_activation=True,
    aaf_n=5.0,
    aaf_init_na=1.0,
):
    bounds = cpinn_get_bounds(cfg)
    x_split = 0.5 * (bounds["x_min"] + bounds["x_max"])

    if spatial_dim == 1:
        submodels = [
            cpinn_make_submodel(cfg, use_local_adaptive_activation, aaf_n, aaf_init_na),
            cpinn_make_submodel(cfg, use_local_adaptive_activation, aaf_n, aaf_init_na),
        ]
        model = CPINNModel(submodels, spatial_dim=1, x_split=x_split)

    else:
        y_split = 0.5 * (bounds["y_min"] + bounds["y_max"])
        submodels = [
            cpinn_make_submodel(cfg, use_local_adaptive_activation, aaf_n, aaf_init_na),
            cpinn_make_submodel(cfg, use_local_adaptive_activation, aaf_n, aaf_init_na),
            cpinn_make_submodel(cfg, use_local_adaptive_activation, aaf_n, aaf_init_na),
            cpinn_make_submodel(cfg, use_local_adaptive_activation, aaf_n, aaf_init_na),
        ]
        model = CPINNModel(submodels, spatial_dim=2, x_split=x_split, y_split=y_split)

    model.cpinn_bounds = bounds
    return model.to(DEVICE, dtype=DTYPE)

def cpinn_rand_uniform(n, lo, hi):
    return lo + (hi - lo) * torch.rand(int(n), 1, device=DEVICE, dtype=DTYPE)

def cpinn_build_interface_points(cfg, spatial_dim, n_per_piece):
    bounds = cpinn_get_bounds(cfg)
    x0, x1 = bounds["x_min"], bounds["x_max"]
    t0, t1 = bounds["t_min"], bounds["t_max"]
    xs = 0.5 * (x0 + x1)

    interfaces = []

    if spatial_dim == 1:
        t = cpinn_rand_uniform(n_per_piece, t0, t1)
        x = torch.full_like(t, xs)
        interfaces.append({
            "name": "x_interface",
            "pair": (0, 1),
            "normal_axis": "x",
            "coords": (x, t),
        })
        return interfaces, bounds

    y0, y1 = bounds["y_min"], bounds["y_max"]
    ys = 0.5 * (y0 + y1)

    n = int(n_per_piece)

    y = cpinn_rand_uniform(n, y0, ys)
    t = cpinn_rand_uniform(n, t0, t1)
    x = torch.full_like(y, xs)
    interfaces.append({
        "name": "vertical_lower",
        "pair": (0, 1),
        "normal_axis": "x",
        "coords": (x, y, t),
    })

    y = cpinn_rand_uniform(n, ys, y1)
    t = cpinn_rand_uniform(n, t0, t1)
    x = torch.full_like(y, xs)
    interfaces.append({
        "name": "vertical_upper",
        "pair": (2, 3),
        "normal_axis": "x",
        "coords": (x, y, t),
    })

    x = cpinn_rand_uniform(n, x0, xs)
    t = cpinn_rand_uniform(n, t0, t1)
    y = torch.full_like(x, ys)
    interfaces.append({
        "name": "horizontal_left",
        "pair": (0, 2),
        "normal_axis": "y",
        "coords": (x, y, t),
    })

    x = cpinn_rand_uniform(n, xs, x1)
    t = cpinn_rand_uniform(n, t0, t1)
    y = torch.full_like(x, ys)
    interfaces.append({
        "name": "horizontal_right",
        "pair": (1, 3),
        "normal_axis": "y",
        "coords": (x, y, t),
    })

    return interfaces, bounds

def cpinn_state_difference_loss(a, b, cfg):
    if "scaled_primitive_mse" in globals():
        try:
            return scaled_primitive_mse(a, b, cfg)
        except Exception:
            pass

    if "scaled_state_mse" in globals():
        try:
            return scaled_state_mse(a, b, cfg)
        except Exception:
            pass

    c = a.shape[1]

    if c == 1:
        return (a - b).pow(2).mean()

    if c == 2 and "h_scale" in globals() and "q_scale" in globals():
        hs = h_scale(cfg)
        qs = q_scale(cfg)
        return (
            ((a[:, 0:1] - b[:, 0:1]) / hs).pow(2)
            + ((a[:, 1:2] - b[:, 1:2]) / qs).pow(2)
        ).mean()

    if c == 3 and "h_scale" in globals() and "q_scale" in globals():
        hs = h_scale(cfg)
        qs = q_scale(cfg)
        return (
            ((a[:, 0:1] - b[:, 0:1]) / hs).pow(2)
            + ((a[:, 1:2] - b[:, 1:2]) / qs).pow(2)
            + ((a[:, 2:3] - b[:, 2:3]) / qs).pow(2)
        ).mean()

    if c == 3 and "scales" in globals():
        rho_s, u_s, p_s, _, _ = scales(cfg)
        return (
            ((a[:, 0:1] - b[:, 0:1]) / rho_s).pow(2)
            + ((a[:, 1:2] - b[:, 1:2]) / u_s).pow(2)
            + ((a[:, 2:3] - b[:, 2:3]) / p_s).pow(2)
        ).mean()

    if c == 4 and "scales" in globals():
        rho_s, u_s, p_s, _, _ = scales(cfg)
        return (
            ((a[:, 0:1] - b[:, 0:1]) / rho_s).pow(2)
            + ((a[:, 1:2] - b[:, 1:2]) / u_s).pow(2)
            + ((a[:, 2:3] - b[:, 2:3]) / u_s).pow(2)
            + ((a[:, 3:4] - b[:, 3:4]) / p_s).pow(2)
        ).mean()

    return (a - b).pow(2).mean()

def cpinn_flux_difference_loss(fa, fb):
    return sum((a - b).pow(2).mean() for a, b in zip(fa, fb))

def cpinn_flux_and_state(model, subdomain_id, coords, cfg, normal_axis):
    W = model.forward_subdomain(subdomain_id, coords)

    if len(coords) == 2:
        if "burgers_residual" in globals():
            f = 0.5 * W.pow(2)
            return [f], W

        if "euler_residual" in globals():
            F1, F2, F3 = flux_torch(W, cfg)
            rho_s, u_s, p_s, mom_s, e_s = scales(cfg)
            return [F1 / rho_s, F2 / mom_s, F3 / e_s], W

        if "shallow_residual" in globals():
            h = W[:, 0:1]
            q = W[:, 1:2]
            F1, F2 = shallow_flux_torch(h, q, cfg)
            return [F1 / h_scale(cfg), F2 / q_scale(cfg)], W

    if len(coords) == 3:
        if "burgers2d_residual" in globals():
            f = 0.5 * W.pow(2)
            return [f], W

        if "euler2d_residual" in globals():
            Fv, Gv = flux_torch(W, cfg)
            rho_s, u_s, p_s, mom_s, e_s = scales(cfg)
            flux = Fv if normal_axis == "x" else Gv
            return [flux[0] / rho_s, flux[1] / mom_s, flux[2] / mom_s, flux[3] / e_s], W

        if "swe2d_residual" in globals():
            h = W[:, 0:1]
            m = W[:, 1:2]
            n = W[:, 2:3]
            Fv, Gv = flux_torch(h, m, n, cfg)
            flux = Fv if normal_axis == "x" else Gv
            return [flux[0] / h_scale(cfg), flux[1] / q_scale(cfg), flux[2] / q_scale(cfg)], W

    raise RuntimeError("No compatible conservation-law flux function found.")

def cpinn_interface_loss(model, cfg, interfaces):
    flux_terms = []
    avg_terms = []
    rows = []

    for item in interfaces:
        p, q = item["pair"]
        coords = item["coords"]
        normal_axis = item["normal_axis"]

        fp, up = cpinn_flux_and_state(model, p, coords, cfg, normal_axis)
        fq, uq = cpinn_flux_and_state(model, q, coords, cfg, normal_axis)

        flux_loss = cpinn_flux_difference_loss(fp, fq)
        avg_loss = cpinn_state_difference_loss(up, uq, cfg)

        flux_terms.append(flux_loss)
        avg_terms.append(avg_loss)

        rows.append({
            "interface": item["name"],
            "pair_left": int(p),
            "pair_right": int(q),
            "normal_axis": normal_axis,
            "flux_loss": float(flux_loss.detach().cpu()),
            "avg_solution_loss": float(avg_loss.detach().cpu()),
            "num_points": int(coords[0].shape[0]),
        })

    flux_total = sum(flux_terms) / max(len(flux_terms), 1)
    avg_total = sum(avg_terms) / max(len(avg_terms), 1)

    return flux_total, avg_total, rows

def cpinn_aaf_state(model):
    out = []
    for i, sub in enumerate(model.submodels):
        if hasattr(sub, "cpinn_aaf_a"):
            a = float(torch.clamp(sub.cpinn_aaf_a, min=1.0e-8).detach().cpu())
            n = float(sub.cpinn_aaf_n)
            out.append({
                "subdomain": int(i),
                "local_adaptive_activation": bool(sub.cpinn_local_aaf_used),
                "a": a,
                "n": n,
                "na": n * a,
                "num_adaptive_activations": int(sub.cpinn_aaf_num_activations),
            })
        else:
            out.append({
                "subdomain": int(i),
                "local_adaptive_activation": False,
                "a": None,
                "n": None,
                "na": None,
                "num_adaptive_activations": 0,
            })
    return out

def cpinn_clamp_aaf(model, na_min=1.0e-6, na_max=20.0):
    with torch.no_grad():
        for sub in model.submodels:
            if hasattr(sub, "cpinn_aaf_a"):
                n = float(sub.cpinn_aaf_n)
                sub.cpinn_aaf_a.clamp_(float(na_min) / n, float(na_max) / n)

def train_cpinn_fixed(
    model,
    cfg,
    interfaces,
    cpinn_w_flux=20.0,
    cpinn_w_avg=20.0,
    cpinn_aaf_na_min=1.0e-6,
    cpinn_aaf_na_max=20.0,
):
    model.train()

    history = []
    interface_history = []

    total_iters = int(cfg.warmup_iters) + int(cfg.gated_iters)
    global_iter = 0

    phases = [
        ("cpinn_warmup", int(cfg.warmup_iters), float(cfg.lr_warmup), float(cfg.lr_warmup) * 0.05),
        ("cpinn_main", int(cfg.gated_iters), float(cfg.lr_gated), float(cfg.lr_gated) * 0.03),
    ]

    for phase_name, n_iters, lr, eta_min in phases:
        if n_iters <= 0:
            continue

        opt = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=float(getattr(cfg, "weight_decay", 0.0)),
        )

        sch = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt,
            T_max=n_iters,
            eta_min=eta_min,
        )

        hist_every = int(getattr(cfg, "history_every", 500))
        print_every = int(getattr(cfg, "print_every", 1000))

        print(f"[cPINN] phase={phase_name}, iters={n_iters}, lr={lr:.3e}")

        for local_it in range(1, n_iters + 1):
            global_iter += 1

            opt.zero_grad(set_to_none=True)

            base_loss, parts = vanilla_loss(model, cfg)
            flux_loss, avg_loss, interface_rows = cpinn_interface_loss(model, cfg, interfaces)

            loss = base_loss + float(cpinn_w_flux) * flux_loss + float(cpinn_w_avg) * avg_loss

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                float(getattr(cfg, "grad_clip", 1.0)),
            )

            opt.step()
            cpinn_clamp_aaf(model, cpinn_aaf_na_min, cpinn_aaf_na_max)
            sch.step()

            if local_it == 1 or local_it % hist_every == 0 or local_it == n_iters:
                aaf_states = cpinn_aaf_state(model)
                mean_na = np.nanmean([s["na"] for s in aaf_states if s["na"] is not None]) if len(aaf_states) > 0 else np.nan

                history.append({
                    "phase": phase_name,
                    "local_iter": int(local_it),
                    "global_iter": int(global_iter),
                    "loss": float(loss.detach().cpu()),
                    "base_loss": float(base_loss.detach().cpu()),
                    "ic": float(parts["ic"].detach().cpu()),
                    "bc": float(parts["bc"].detach().cpu()),
                    "pde": float(parts["pde"].detach().cpu()),
                    "weighted_pde": float(parts["weighted_pde"].detach().cpu()),
                    "interface_flux_loss": float(flux_loss.detach().cpu()),
                    "interface_avg_solution_loss": float(avg_loss.detach().cpu()),
                    "weighted_interface_flux": float(cpinn_w_flux) * float(flux_loss.detach().cpu()),
                    "weighted_interface_avg": float(cpinn_w_avg) * float(avg_loss.detach().cpu()),
                    "cpinn_w_flux": float(cpinn_w_flux),
                    "cpinn_w_avg": float(cpinn_w_avg),
                    "mean_adaptive_na": float(mean_na),
                    "lr": float(opt.param_groups[0]["lr"]),
                })

                for row in interface_rows:
                    row = dict(row)
                    row.update({
                        "phase": phase_name,
                        "local_iter": int(local_it),
                        "global_iter": int(global_iter),
                    })
                    interface_history.append(row)

            if local_it == 1 or local_it % print_every == 0 or local_it == n_iters:
                print(
                    f"[cPINN] {global_iter:6d}/{total_iters} "
                    f"loss={float(loss.detach().cpu()):.3e} "
                    f"base={float(base_loss.detach().cpu()):.3e} "
                    f"fluxI={float(flux_loss.detach().cpu()):.1e} "
                    f"avgI={float(avg_loss.detach().cpu()):.1e}"
                )

    history_df = pd.DataFrame(history)
    interface_df = pd.DataFrame(interface_history)

    final_info = {
        "spatial_dim": int(model.spatial_dim),
        "num_subdomains": int(len(model.submodels)),
        "x_split": float(model.x_split),
        "y_split": None if model.y_split is None else float(model.y_split),
        "bounds": model.cpinn_bounds,
        "cpinn_w_flux": float(cpinn_w_flux),
        "cpinn_w_avg": float(cpinn_w_avg),
        "interface_conditions": ["normal_flux_continuity", "average_solution_continuity"],
        "domain_decomposition": "spatial_midpoint_split_1d_or_quadrant_split_2d",
        "local_adaptive_activation": cpinn_aaf_state(model),
    }

    return model, history_df, interface_df, final_info

def cpinn_save_interface_points(run_dir, interfaces):
    diag_dir = run_dir / "diagnostics"
    diag_dir.mkdir(exist_ok=True)

    arrays = {}
    meta = []

    for i, item in enumerate(interfaces):
        coords = item["coords"]
        for j, arr in enumerate(coords):
            arrays[f"interface_{i}_coord_{j}"] = arr.detach().cpu().numpy()

        meta.append({
            "interface_id": int(i),
            "name": item["name"],
            "pair": [int(item["pair"][0]), int(item["pair"][1])],
            "normal_axis": item["normal_axis"],
            "num_points": int(coords[0].shape[0]),
        })

    np.savez_compressed(diag_dir / "cpinn_interface_points.npz", **arrays)
    save_json(meta, diag_dir / "cpinn_interface_points_meta.json")

def save_cpinn_diagnostics(run_dir, model, history_df, interface_df, final_info, interfaces):
    diag_dir = run_dir / "diagnostics"
    diag_dir.mkdir(exist_ok=True)

    history_df.to_csv(diag_dir / "cpinn_loss_history.csv", index=False)
    interface_df.to_csv(diag_dir / "cpinn_interface_history.csv", index=False)
    save_json(final_info, diag_dir / "cpinn_subdomain_config.json")
    cpinn_save_interface_points(run_dir, interfaces)

    torch.save(
        {
            "submodel_state_dicts": [sub.state_dict() for sub in model.submodels],
            "final_info": final_info,
        },
        diag_dir / "cpinn_submodel_state_dicts.pt",
    )

def train_cpinn_one_seed(
    equation_name,
    base_cfg,
    seed,
    ref=None,
    total_iters=ADAM_TOTAL_ITERS,
    skip_if_done=True,
    cpinn_n_interface_per_piece=1000,
    cpinn_w_flux=20.0,
    cpinn_w_avg=20.0,
    cpinn_use_local_adaptive_activation=True,
    cpinn_aaf_n=5.0,
    cpinn_aaf_init_na=1.0,
    cpinn_aaf_na_min=1.0e-6,
    cpinn_aaf_na_max=20.0,
):
    cfg_s = prepare_cfg_for_seed(base_cfg, seed, total_iters=total_iters)

    configure_runtime(cfg_s)
    set_seed(int(seed))

    method_name = "cPINN"
    run_dir = get_run_dir(equation_name, method_name, seed)

    if skip_if_done and (run_dir / "_SUCCESS").exists():
        return pd.DataFrame([load_json(run_dir / "metrics_final.json")])

    try:
        spatial_dim = 2 if equation_name.startswith("2d_") else 1

        model = make_cpinn_model(
            cfg_s,
            spatial_dim=spatial_dim,
            use_local_adaptive_activation=cpinn_use_local_adaptive_activation,
            aaf_n=cpinn_aaf_n,
            aaf_init_na=cpinn_aaf_init_na,
        )

        interfaces, bounds = cpinn_build_interface_points(
            cfg_s,
            spatial_dim=spatial_dim,
            n_per_piece=cpinn_n_interface_per_piece,
        )

        save_run_header(
            run_dir,
            equation_name,
            method_name,
            cfg_s,
            {
                "method": "cPINN",
                "description": "Conservative PINN with spatial domain decomposition, subdomain networks, normal flux continuity, and average-solution interface continuity.",
                "optimizer_main": "AdamW",
                "lbfgs_used": False,
                "adam_total_iters": int(cfg_s.warmup_iters + cfg_s.gated_iters),
                "spatial_dim": int(spatial_dim),
                "num_subdomains": int(len(model.submodels)),
                "domain_decomposition": "midpoint split in 1D; quadrant split in 2D",
                "x_split": float(model.x_split),
                "y_split": None if model.y_split is None else float(model.y_split),
                "bounds": bounds,
                "cpinn_n_interface_per_piece": int(cpinn_n_interface_per_piece),
                "num_interface_pieces": int(len(interfaces)),
                "total_interface_points": int(sum(item["coords"][0].shape[0] for item in interfaces)),
                "cpinn_w_flux": float(cpinn_w_flux),
                "cpinn_w_avg": float(cpinn_w_avg),
                "interface_conditions": ["normal_flux_continuity", "average_solution_continuity"],
                "local_adaptive_activation": bool(cpinn_use_local_adaptive_activation),
                "cpinn_aaf_n": float(cpinn_aaf_n),
                "cpinn_aaf_init_na": float(cpinn_aaf_init_na),
                "n_ic": int(cfg_s.n_ic),
                "n_bc": int(cfg_s.n_bc),
                "n_f": int(cfg_s.n_f),
            },
        )

        t0 = time.time()

        model, history_df, interface_df, final_info = train_cpinn_fixed(
            model=model,
            cfg=cfg_s,
            interfaces=interfaces,
            cpinn_w_flux=cpinn_w_flux,
            cpinn_w_avg=cpinn_w_avg,
            cpinn_aaf_na_min=cpinn_aaf_na_min,
            cpinn_aaf_na_max=cpinn_aaf_na_max,
        )

        train_time = time.time() - t0

        metrics = evaluate_model_any(model, method_name, cfg_s, ref=ref)

        metrics.update({
            "equation": equation_name,
            "method": method_name,
            "seed": int(seed),
            "optimizer_main": "AdamW",
            "lbfgs_used": False,
            "wall_clock_sec_total": float(train_time),
            "num_parameters": parameter_count(model),
            "cpinn_num_subdomains": int(len(model.submodels)),
            "cpinn_spatial_dim": int(spatial_dim),
            "cpinn_x_split": float(model.x_split),
            "cpinn_y_split": None if model.y_split is None else float(model.y_split),
            "cpinn_n_interface_per_piece": int(cpinn_n_interface_per_piece),
            "cpinn_total_interface_points": int(sum(item["coords"][0].shape[0] for item in interfaces)),
            "cpinn_w_flux": float(cpinn_w_flux),
            "cpinn_w_avg": float(cpinn_w_avg),
            "cpinn_local_adaptive_activation": bool(cpinn_use_local_adaptive_activation),
        })

        if len(history_df) > 0:
            metrics["cpinn_interface_flux_loss_final_logged"] = float(history_df["interface_flux_loss"].iloc[-1])
            metrics["cpinn_interface_avg_loss_final_logged"] = float(history_df["interface_avg_solution_loss"].iloc[-1])

        save_history(history_df, run_dir, "history.csv")
        save_json(metrics, run_dir / "metrics_adamw.json")
        save_json(metrics, run_dir / "metrics_final.json")

        save_model_checkpoint(
            model,
            run_dir,
            "model_adamw_final.pt",
            extra=metrics,
        )

        save_model_checkpoint(
            model,
            run_dir,
            "model_final.pt",
            extra=metrics,
        )

        try_save_predictions(model, cfg_s, run_dir)
        save_cpinn_diagnostics(run_dir, model, history_df, interface_df, final_info, interfaces)

        mark_status(run_dir, "success", {"wall_clock_sec_total": float(train_time)})

        del model

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        gc.collect()

        return pd.DataFrame([metrics])

    except Exception as e:
        mark_status(run_dir, "failed", {"error": repr(e)})
        raise

def run_cpinn_seeds(
    equation_name,
    base_cfg,
    seeds,
    total_iters=ADAM_TOTAL_ITERS,
    skip_if_done=True,
    cpinn_n_interface_per_piece=1000,
    cpinn_w_flux=20.0,
    cpinn_w_avg=20.0,
    cpinn_use_local_adaptive_activation=True,
    cpinn_aaf_n=5.0,
    cpinn_aaf_init_na=1.0,
):
    cfg_ref = copy.deepcopy(base_cfg)

    configure_runtime(cfg_ref)

    ref = build_reference_if_needed(cfg_ref)

    rows = []

    for seed in seeds:
        df_seed = train_cpinn_one_seed(
            equation_name=equation_name,
            base_cfg=base_cfg,
            seed=seed,
            ref=ref,
            total_iters=total_iters,
            skip_if_done=skip_if_done,
            cpinn_n_interface_per_piece=cpinn_n_interface_per_piece,
            cpinn_w_flux=cpinn_w_flux,
            cpinn_w_avg=cpinn_w_avg,
            cpinn_use_local_adaptive_activation=cpinn_use_local_adaptive_activation,
            cpinn_aaf_n=cpinn_aaf_n,
            cpinn_aaf_init_na=cpinn_aaf_init_na,
        )

        rows.append(df_seed)

    df = pd.concat(rows, ignore_index=True)

    summary_dir = RUNS_ROOT / safe_name(equation_name) / "_summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    out_path = summary_dir / "cpinn_metrics.csv"
    df.to_csv(out_path, index=False)

    print("saved:", out_path)

    return df

def reset_cpinn_runs(equation_name, seeds):
    for seed in seeds:
        run_dir = get_run_dir(equation_name, "cPINN", seed)
        if run_dir.exists():
            print("remove:", run_dir)
            shutil.rmtree(run_dir)
        else:
            print("not found:", run_dir)

import time
import gc
import math
import shutil
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

class XPINNScaledTanh(nn.Module):
    def __init__(self, a_getter, n):
        super().__init__()
        self.a_getter = a_getter
        self.n = float(n)

    def forward(self, x):
        a = torch.clamp(self.a_getter(), min=1.0e-8)
        return torch.tanh(self.n * a * x)

class XPINNScaledSiLU(nn.Module):
    def __init__(self, a_getter, n):
        super().__init__()
        self.a_getter = a_getter
        self.n = float(n)

    def forward(self, x):
        a = torch.clamp(self.a_getter(), min=1.0e-8)
        return F.silu(self.n * a * x)

def xpinn_replace_activations(module, a_getter, n):
    count = 0

    for name, child in list(module.named_children()):
        if isinstance(child, nn.Tanh):
            setattr(module, name, XPINNScaledTanh(a_getter, n))
            count += 1
        elif isinstance(child, nn.SiLU):
            setattr(module, name, XPINNScaledSiLU(a_getter, n))
            count += 1
        else:
            count += xpinn_replace_activations(child, a_getter, n)

    return count

def xpinn_make_submodel(cfg, use_local_adaptive_activation=True, aaf_n=5.0, aaf_init_na=1.0):
    model = MLP(cfg).to(DEVICE, dtype=DTYPE)
    model.xpinn_local_aaf_used = False
    model.xpinn_aaf_num_activations = 0

    if use_local_adaptive_activation:
        a_init = float(aaf_init_na) / float(aaf_n)
        model.xpinn_aaf_a = nn.Parameter(torch.tensor(a_init, device=DEVICE, dtype=DTYPE))
        model.xpinn_aaf_n = float(aaf_n)
        model.xpinn_aaf_init_na = float(aaf_init_na)

        n_replaced = xpinn_replace_activations(
            model,
            a_getter=lambda: model.xpinn_aaf_a,
            n=aaf_n,
        )

        model.xpinn_local_aaf_used = n_replaced > 0
        model.xpinn_aaf_num_activations = int(n_replaced)

    return model

def xpinn_get_attr(cfg, names, default=None):
    for name in names:
        if hasattr(cfg, name):
            return float(getattr(cfg, name))
    return default

def xpinn_get_bounds(cfg):
    x_min = xpinn_get_attr(cfg, ["x_min", "xmin", "x0", "x_left"], None)
    x_max = xpinn_get_attr(cfg, ["x_max", "xmax", "x1", "x_right"], None)
    y_min = xpinn_get_attr(cfg, ["y_min", "ymin", "y0", "y_bottom"], None)
    y_max = xpinn_get_attr(cfg, ["y_max", "ymax", "y1", "y_top"], None)
    t_min = xpinn_get_attr(cfg, ["t_min", "tmin", "t0"], 0.0)
    t_max = xpinn_get_attr(cfg, ["t_max", "tmax", "T", "t_final"], None)

    if x_min is None or x_max is None or t_max is None:
        pts = sample_f(4096, cfg)

        if x_min is None:
            x_min = float(pts[0].detach().min().cpu())
        if x_max is None:
            x_max = float(pts[0].detach().max().cpu())

        if len(pts) == 2:
            if t_max is None:
                t_max = float(pts[1].detach().max().cpu())
        else:
            if y_min is None:
                y_min = float(pts[1].detach().min().cpu())
            if y_max is None:
                y_max = float(pts[1].detach().max().cpu())
            if t_max is None:
                t_max = float(pts[2].detach().max().cpu())

    return {
        "x_min": float(x_min),
        "x_max": float(x_max),
        "y_min": None if y_min is None else float(y_min),
        "y_max": None if y_max is None else float(y_max),
        "t_min": float(t_min),
        "t_max": float(t_max),
    }

def xpinn_cat(coords):
    if len(coords) == 2:
        return cat_xt(coords[0], coords[1])
    if len(coords) == 3:
        return cat_xyt(coords[0], coords[1], coords[2])
    raise RuntimeError(f"unsupported coordinate dimension: {len(coords)}")

class XPINNModel(nn.Module):
    def __init__(self, submodels, spatial_dim, x_split, t_split):
        super().__init__()
        self.submodels = nn.ModuleList(submodels)
        self.spatial_dim = int(spatial_dim)
        self.x_split = float(x_split)
        self.t_split = float(t_split)

    def subdomain_index(self, z):
        x = z[:, 0:1]

        if self.spatial_dim == 1:
            t = z[:, 1:2]
        else:
            t = z[:, 2:3]

        ix = (x > self.x_split).long().reshape(-1)
        it = (t > self.t_split).long().reshape(-1)

        return ix + 2 * it

    def forward(self, z):
        idx = self.subdomain_index(z)
        out_all = None

        for k, net in enumerate(self.submodels):
            mask = idx == k
            if not torch.any(mask):
                continue

            out_k = net(z[mask])

            if out_all is None:
                out_all = z.new_zeros((z.shape[0], out_k.shape[1]))

            out_all[mask] = out_k

        if out_all is None:
            out_all = self.submodels[0](z)

        return out_all

    def forward_subdomain(self, subdomain_id, coords):
        z = xpinn_cat(coords)
        return self.submodels[int(subdomain_id)](z)

def make_xpinn_model(
    cfg,
    spatial_dim,
    use_local_adaptive_activation=True,
    aaf_n=5.0,
    aaf_init_na=1.0,
):
    bounds = xpinn_get_bounds(cfg)

    x_split = 0.5 * (bounds["x_min"] + bounds["x_max"])
    t_split = 0.5 * (bounds["t_min"] + bounds["t_max"])

    submodels = [
        xpinn_make_submodel(cfg, use_local_adaptive_activation, aaf_n, aaf_init_na),
        xpinn_make_submodel(cfg, use_local_adaptive_activation, aaf_n, aaf_init_na),
        xpinn_make_submodel(cfg, use_local_adaptive_activation, aaf_n, aaf_init_na),
        xpinn_make_submodel(cfg, use_local_adaptive_activation, aaf_n, aaf_init_na),
    ]

    model = XPINNModel(
        submodels=submodels,
        spatial_dim=spatial_dim,
        x_split=x_split,
        t_split=t_split,
    )

    model.xpinn_bounds = bounds

    return model.to(DEVICE, dtype=DTYPE)

def xpinn_rand_uniform(n, lo, hi):
    return lo + (hi - lo) * torch.rand(int(n), 1, device=DEVICE, dtype=DTYPE)

def xpinn_build_interface_points(cfg, spatial_dim, n_per_piece):
    bounds = xpinn_get_bounds(cfg)

    x0 = bounds["x_min"]
    x1 = bounds["x_max"]
    t0 = bounds["t_min"]
    t1 = bounds["t_max"]
    xs = 0.5 * (x0 + x1)
    ts = 0.5 * (t0 + t1)

    interfaces = []
    n = int(n_per_piece)

    if spatial_dim == 1:
        t = xpinn_rand_uniform(n, t0, ts)
        x = torch.full_like(t, xs)
        interfaces.append({
            "name": "space_interface_early",
            "pair": (0, 1),
            "coords": (x, t),
        })

        t = xpinn_rand_uniform(n, ts, t1)
        x = torch.full_like(t, xs)
        interfaces.append({
            "name": "space_interface_late",
            "pair": (2, 3),
            "coords": (x, t),
        })

        x = xpinn_rand_uniform(n, x0, xs)
        t = torch.full_like(x, ts)
        interfaces.append({
            "name": "time_interface_left",
            "pair": (0, 2),
            "coords": (x, t),
        })

        x = xpinn_rand_uniform(n, xs, x1)
        t = torch.full_like(x, ts)
        interfaces.append({
            "name": "time_interface_right",
            "pair": (1, 3),
            "coords": (x, t),
        })

        return interfaces, bounds

    y0 = bounds["y_min"]
    y1 = bounds["y_max"]

    t = xpinn_rand_uniform(n, t0, ts)
    y = xpinn_rand_uniform(n, y0, y1)
    x = torch.full_like(t, xs)
    interfaces.append({
        "name": "space_interface_early",
        "pair": (0, 1),
        "coords": (x, y, t),
    })

    t = xpinn_rand_uniform(n, ts, t1)
    y = xpinn_rand_uniform(n, y0, y1)
    x = torch.full_like(t, xs)
    interfaces.append({
        "name": "space_interface_late",
        "pair": (2, 3),
        "coords": (x, y, t),
    })

    x = xpinn_rand_uniform(n, x0, xs)
    y = xpinn_rand_uniform(n, y0, y1)
    t = torch.full_like(x, ts)
    interfaces.append({
        "name": "time_interface_left",
        "pair": (0, 2),
        "coords": (x, y, t),
    })

    x = xpinn_rand_uniform(n, xs, x1)
    y = xpinn_rand_uniform(n, y0, y1)
    t = torch.full_like(x, ts)
    interfaces.append({
        "name": "time_interface_right",
        "pair": (1, 3),
        "coords": (x, y, t),
    })

    return interfaces, bounds

def xpinn_coords_require_grad(coords):
    return tuple(
        v.detach().clone().to(DEVICE, dtype=DTYPE).requires_grad_(True)
        for v in coords
    )

def xpinn_state_difference_loss(a, b, cfg):
    if "scaled_primitive_mse" in globals():
        try:
            return scaled_primitive_mse(a, b, cfg)
        except Exception:
            pass

    if "scaled_state_mse" in globals():
        try:
            return scaled_state_mse(a, b, cfg)
        except Exception:
            pass

    c = a.shape[1]

    if c == 1:
        return (a - b).pow(2).mean()

    if c == 2 and "h_scale" in globals() and "q_scale" in globals():
        hs = h_scale(cfg)
        qs = q_scale(cfg)
        return (
            ((a[:, 0:1] - b[:, 0:1]) / hs).pow(2)
            + ((a[:, 1:2] - b[:, 1:2]) / qs).pow(2)
        ).mean()

    if c == 3 and "h_scale" in globals() and "q_scale" in globals():
        hs = h_scale(cfg)
        qs = q_scale(cfg)
        return (
            ((a[:, 0:1] - b[:, 0:1]) / hs).pow(2)
            + ((a[:, 1:2] - b[:, 1:2]) / qs).pow(2)
            + ((a[:, 2:3] - b[:, 2:3]) / qs).pow(2)
        ).mean()

    if c == 3 and "scales" in globals():
        rho_s, u_s, p_s, _, _ = scales(cfg)
        return (
            ((a[:, 0:1] - b[:, 0:1]) / rho_s).pow(2)
            + ((a[:, 1:2] - b[:, 1:2]) / u_s).pow(2)
            + ((a[:, 2:3] - b[:, 2:3]) / p_s).pow(2)
        ).mean()

    if c == 4 and "scales" in globals():
        rho_s, u_s, p_s, _, _ = scales(cfg)
        return (
            ((a[:, 0:1] - b[:, 0:1]) / rho_s).pow(2)
            + ((a[:, 1:2] - b[:, 1:2]) / u_s).pow(2)
            + ((a[:, 2:3] - b[:, 2:3]) / u_s).pow(2)
            + ((a[:, 3:4] - b[:, 3:4]) / p_s).pow(2)
        ).mean()

    return (a - b).pow(2).mean()

def xpinn_residual_components(model, subdomain_id, coords, cfg):
    coords = xpinn_coords_require_grad(coords)

    if len(coords) == 2:
        x, t = coords
        W = model.forward_subdomain(subdomain_id, coords)

        if "burgers_residual" in globals():
            u = W
            u_t = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True, retain_graph=True)[0]
            f = flux_torch(u)
            f_x = torch.autograd.grad(f, x, torch.ones_like(f), create_graph=True, retain_graph=True)[0]
            return [u_t + f_x], W

        if "euler_residual" in globals():
            rho, m, E = primitive_to_conserved_torch(W, cfg)
            F1, F2, F3 = flux_torch(W, cfg)

            rho_t = torch.autograd.grad(rho, t, torch.ones_like(rho), create_graph=True, retain_graph=True)[0]
            m_t = torch.autograd.grad(m, t, torch.ones_like(m), create_graph=True, retain_graph=True)[0]
            E_t = torch.autograd.grad(E, t, torch.ones_like(E), create_graph=True, retain_graph=True)[0]

            F1_x = torch.autograd.grad(F1, x, torch.ones_like(F1), create_graph=True, retain_graph=True)[0]
            F2_x = torch.autograd.grad(F2, x, torch.ones_like(F2), create_graph=True, retain_graph=True)[0]
            F3_x = torch.autograd.grad(F3, x, torch.ones_like(F3), create_graph=True, retain_graph=True)[0]

            rho_s, u_s, p_s, mom_s, e_s = scales(cfg)

            return [
                (rho_t + F1_x) / rho_s,
                (m_t + F2_x) / mom_s,
                (E_t + F3_x) / e_s,
            ], W

        if "shallow_residual" in globals():
            h = W[:, 0:1]
            q = W[:, 1:2]
            F1, F2 = shallow_flux_torch(h, q, cfg)

            h_t = torch.autograd.grad(h, t, torch.ones_like(h), create_graph=True, retain_graph=True)[0]
            q_t = torch.autograd.grad(q, t, torch.ones_like(q), create_graph=True, retain_graph=True)[0]
            F1_x = torch.autograd.grad(F1, x, torch.ones_like(F1), create_graph=True, retain_graph=True)[0]
            F2_x = torch.autograd.grad(F2, x, torch.ones_like(F2), create_graph=True, retain_graph=True)[0]

            return [
                (h_t + F1_x) / h_scale(cfg),
                (q_t + F2_x) / q_scale(cfg),
            ], W

    if len(coords) == 3:
        x, y, t = coords
        W = model.forward_subdomain(subdomain_id, coords)

        if "burgers2d_residual" in globals():
            u = W
            u_t = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True, retain_graph=True)[0]
            fx = flux_torch(u)
            fy = flux_torch(u)
            fx_x = torch.autograd.grad(fx, x, torch.ones_like(fx), create_graph=True, retain_graph=True)[0]
            fy_y = torch.autograd.grad(fy, y, torch.ones_like(fy), create_graph=True, retain_graph=True)[0]
            return [u_t + fx_x + fy_y], W

        if "euler2d_residual" in globals():
            U = prim_to_cons_torch(W, cfg)
            Fv, Gv = flux_torch(W, cfg)

            residuals = []

            for Uk, Fk, Gk in zip(U, Fv, Gv):
                Uk_t = torch.autograd.grad(Uk, t, torch.ones_like(Uk), create_graph=True, retain_graph=True)[0]
                Fk_x = torch.autograd.grad(Fk, x, torch.ones_like(Fk), create_graph=True, retain_graph=True)[0]
                Gk_y = torch.autograd.grad(Gk, y, torch.ones_like(Gk), create_graph=True, retain_graph=True)[0]
                residuals.append(Uk_t + Fk_x + Gk_y)

            r1, r2, r3, r4 = residuals
            rho_s, u_s, p_s, mom_s, e_s = scales(cfg)

            return [
                r1 / rho_s,
                r2 / mom_s,
                r3 / mom_s,
                r4 / e_s,
            ], W

        if "swe2d_residual" in globals():
            h = W[:, 0:1]
            m = W[:, 1:2]
            n = W[:, 2:3]

            Fv, Gv = flux_torch(h, m, n, cfg)
            U = (h, m, n)

            residuals = []

            for Uk, Fk, Gk in zip(U, Fv, Gv):
                Uk_t = torch.autograd.grad(Uk, t, torch.ones_like(Uk), create_graph=True, retain_graph=True)[0]
                Fk_x = torch.autograd.grad(Fk, x, torch.ones_like(Fk), create_graph=True, retain_graph=True)[0]
                Gk_y = torch.autograd.grad(Gk, y, torch.ones_like(Gk), create_graph=True, retain_graph=True)[0]
                residuals.append(Uk_t + Fk_x + Gk_y)

            r_h, r_m, r_n = residuals

            return [
                r_h / h_scale(cfg),
                r_m / q_scale(cfg),
                r_n / q_scale(cfg),
            ], W

    raise RuntimeError("No compatible residual function found for XPINN.")

def xpinn_residual_difference_loss(rp, rq):
    return sum((a - b).pow(2).mean() for a, b in zip(rp, rq))

def xpinn_average_solution_loss(up, uq, cfg):
    uavg = 0.5 * (up + uq)
    return 0.5 * (
        xpinn_state_difference_loss(up, uavg, cfg)
        + xpinn_state_difference_loss(uq, uavg, cfg)
    )

def xpinn_interface_loss(model, cfg, interfaces):
    residual_terms = []
    avg_terms = []
    rows = []

    for item in interfaces:
        p, q = item["pair"]
        coords = item["coords"]

        rp, up = xpinn_residual_components(model, p, coords, cfg)
        rq, uq = xpinn_residual_components(model, q, coords, cfg)

        residual_loss = xpinn_residual_difference_loss(rp, rq)
        avg_loss = xpinn_average_solution_loss(up, uq, cfg)

        residual_terms.append(residual_loss)
        avg_terms.append(avg_loss)

        rows.append({
            "interface": item["name"],
            "pair_left": int(p),
            "pair_right": int(q),
            "residual_continuity_loss": float(residual_loss.detach().cpu()),
            "average_solution_loss": float(avg_loss.detach().cpu()),
            "num_points": int(coords[0].shape[0]),
        })

    residual_total = sum(residual_terms) / max(len(residual_terms), 1)
    avg_total = sum(avg_terms) / max(len(avg_terms), 1)

    return residual_total, avg_total, rows

def xpinn_aaf_state(model):
    out = []

    for i, sub in enumerate(model.submodels):
        if hasattr(sub, "xpinn_aaf_a"):
            a = float(torch.clamp(sub.xpinn_aaf_a, min=1.0e-8).detach().cpu())
            n = float(sub.xpinn_aaf_n)
            out.append({
                "subdomain": int(i),
                "local_adaptive_activation": bool(sub.xpinn_local_aaf_used),
                "a": a,
                "n": n,
                "na": n * a,
                "num_adaptive_activations": int(sub.xpinn_aaf_num_activations),
            })
        else:
            out.append({
                "subdomain": int(i),
                "local_adaptive_activation": False,
                "a": None,
                "n": None,
                "na": None,
                "num_adaptive_activations": 0,
            })

    return out

def xpinn_clamp_aaf(model, na_min=1.0e-6, na_max=20.0):
    with torch.no_grad():
        for sub in model.submodels:
            if hasattr(sub, "xpinn_aaf_a"):
                n = float(sub.xpinn_aaf_n)
                sub.xpinn_aaf_a.clamp_(float(na_min) / n, float(na_max) / n)

def train_xpinn_fixed(
    model,
    cfg,
    interfaces,
    xpinn_w_residual=1.0,
    xpinn_w_avg=20.0,
    xpinn_aaf_na_min=1.0e-6,
    xpinn_aaf_na_max=20.0,
):
    model.train()

    history = []
    interface_history = []

    total_iters = int(cfg.warmup_iters) + int(cfg.gated_iters)
    global_iter = 0

    phases = [
        ("xpinn_warmup", int(cfg.warmup_iters), float(cfg.lr_warmup), float(cfg.lr_warmup) * 0.05),
        ("xpinn_main", int(cfg.gated_iters), float(cfg.lr_gated), float(cfg.lr_gated) * 0.03),
    ]

    for phase_name, n_iters, lr, eta_min in phases:
        if n_iters <= 0:
            continue

        opt = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=float(getattr(cfg, "weight_decay", 0.0)),
        )

        sch = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt,
            T_max=n_iters,
            eta_min=eta_min,
        )

        hist_every = int(getattr(cfg, "history_every", 500))
        print_every = int(getattr(cfg, "print_every", 1000))

        print(f"[XPINN] phase={phase_name}, iters={n_iters}, lr={lr:.3e}")

        for local_it in range(1, n_iters + 1):
            global_iter += 1

            opt.zero_grad(set_to_none=True)

            base_loss, parts = vanilla_loss(model, cfg)
            residual_i_loss, avg_i_loss, interface_rows = xpinn_interface_loss(model, cfg, interfaces)

            loss = base_loss + float(xpinn_w_residual) * residual_i_loss + float(xpinn_w_avg) * avg_i_loss

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                float(getattr(cfg, "grad_clip", 1.0)),
            )

            opt.step()
            xpinn_clamp_aaf(model, xpinn_aaf_na_min, xpinn_aaf_na_max)
            sch.step()

            if local_it == 1 or local_it % hist_every == 0 or local_it == n_iters:
                aaf_states = xpinn_aaf_state(model)
                na_values = [s["na"] for s in aaf_states if s["na"] is not None]
                mean_na = float(np.mean(na_values)) if len(na_values) > 0 else np.nan

                history.append({
                    "phase": phase_name,
                    "local_iter": int(local_it),
                    "global_iter": int(global_iter),
                    "loss": float(loss.detach().cpu()),
                    "base_loss": float(base_loss.detach().cpu()),
                    "ic": float(parts["ic"].detach().cpu()),
                    "bc": float(parts["bc"].detach().cpu()),
                    "pde": float(parts["pde"].detach().cpu()),
                    "weighted_pde": float(parts["weighted_pde"].detach().cpu()),
                    "interface_residual_loss": float(residual_i_loss.detach().cpu()),
                    "interface_avg_solution_loss": float(avg_i_loss.detach().cpu()),
                    "weighted_interface_residual": float(xpinn_w_residual) * float(residual_i_loss.detach().cpu()),
                    "weighted_interface_avg": float(xpinn_w_avg) * float(avg_i_loss.detach().cpu()),
                    "xpinn_w_residual": float(xpinn_w_residual),
                    "xpinn_w_avg": float(xpinn_w_avg),
                    "mean_adaptive_na": mean_na,
                    "lr": float(opt.param_groups[0]["lr"]),
                })

                for row in interface_rows:
                    row = dict(row)
                    row.update({
                        "phase": phase_name,
                        "local_iter": int(local_it),
                        "global_iter": int(global_iter),
                    })
                    interface_history.append(row)

            if local_it == 1 or local_it % print_every == 0 or local_it == n_iters:
                print(
                    f"[XPINN] {global_iter:6d}/{total_iters} "
                    f"loss={float(loss.detach().cpu()):.3e} "
                    f"base={float(base_loss.detach().cpu()):.3e} "
                    f"resI={float(residual_i_loss.detach().cpu()):.1e} "
                    f"avgI={float(avg_i_loss.detach().cpu()):.1e}"
                )

    history_df = pd.DataFrame(history)
    interface_df = pd.DataFrame(interface_history)

    final_info = {
        "spatial_dim": int(model.spatial_dim),
        "num_subdomains": int(len(model.submodels)),
        "x_split": float(model.x_split),
        "t_split": float(model.t_split),
        "bounds": model.xpinn_bounds,
        "xpinn_w_residual": float(xpinn_w_residual),
        "xpinn_w_avg": float(xpinn_w_avg),
        "interface_conditions": ["residual_continuity", "average_solution_continuity"],
        "additional_flux_continuity_used": False,
        "domain_decomposition": "space-time x-t midpoint split into four subdomains; y full range for 2D equations",
        "local_adaptive_activation": xpinn_aaf_state(model),
    }

    return model, history_df, interface_df, final_info

def xpinn_save_interface_points(run_dir, interfaces):
    diag_dir = run_dir / "diagnostics"
    diag_dir.mkdir(exist_ok=True)

    arrays = {}
    meta = []

    for i, item in enumerate(interfaces):
        coords = item["coords"]

        for j, arr in enumerate(coords):
            arrays[f"interface_{i}_coord_{j}"] = arr.detach().cpu().numpy()

        meta.append({
            "interface_id": int(i),
            "name": item["name"],
            "pair": [int(item["pair"][0]), int(item["pair"][1])],
            "num_points": int(coords[0].shape[0]),
        })

    np.savez_compressed(diag_dir / "xpinn_interface_points.npz", **arrays)
    save_json(meta, diag_dir / "xpinn_interface_points_meta.json")

def save_xpinn_diagnostics(run_dir, model, history_df, interface_df, final_info, interfaces):
    diag_dir = run_dir / "diagnostics"
    diag_dir.mkdir(exist_ok=True)

    history_df.to_csv(diag_dir / "xpinn_loss_history.csv", index=False)
    interface_df.to_csv(diag_dir / "xpinn_interface_history.csv", index=False)
    save_json(final_info, diag_dir / "xpinn_subdomain_config.json")
    xpinn_save_interface_points(run_dir, interfaces)

    torch.save(
        {
            "submodel_state_dicts": [sub.state_dict() for sub in model.submodels],
            "final_info": final_info,
        },
        diag_dir / "xpinn_submodel_state_dicts.pt",
    )

def train_xpinn_one_seed(
    equation_name,
    base_cfg,
    seed,
    ref=None,
    total_iters=ADAM_TOTAL_ITERS,
    skip_if_done=True,
    xpinn_n_interface_per_piece=512,
    xpinn_w_residual=1.0,
    xpinn_w_avg=20.0,
    xpinn_use_local_adaptive_activation=True,
    xpinn_aaf_n=5.0,
    xpinn_aaf_init_na=1.0,
    xpinn_aaf_na_min=1.0e-6,
    xpinn_aaf_na_max=20.0,
):
    cfg_s = prepare_cfg_for_seed(base_cfg, seed, total_iters=total_iters)

    configure_runtime(cfg_s)
    set_seed(int(seed))

    method_name = "XPINN"
    run_dir = get_run_dir(equation_name, method_name, seed)

    if skip_if_done and (run_dir / "_SUCCESS").exists():
        return pd.DataFrame([load_json(run_dir / "metrics_final.json")])

    try:
        spatial_dim = 2 if equation_name.startswith("2d_") else 1

        model = make_xpinn_model(
            cfg_s,
            spatial_dim=spatial_dim,
            use_local_adaptive_activation=xpinn_use_local_adaptive_activation,
            aaf_n=xpinn_aaf_n,
            aaf_init_na=xpinn_aaf_init_na,
        )

        interfaces, bounds = xpinn_build_interface_points(
            cfg_s,
            spatial_dim=spatial_dim,
            n_per_piece=xpinn_n_interface_per_piece,
        )

        save_run_header(
            run_dir,
            equation_name,
            method_name,
            cfg_s,
            {
                "method": "XPINN",
                "description": "Extended PINN with generalized space-time domain decomposition, subdomain networks, residual-continuity interface loss, and average-solution interface loss.",
                "optimizer_main": "AdamW",
                "lbfgs_used": False,
                "adam_total_iters": int(cfg_s.warmup_iters + cfg_s.gated_iters),
                "spatial_dim": int(spatial_dim),
                "num_subdomains": int(len(model.submodels)),
                "domain_decomposition": "x-t midpoint split into four subdomains; y is not split for 2D equations",
                "x_split": float(model.x_split),
                "t_split": float(model.t_split),
                "bounds": bounds,
                "xpinn_n_interface_per_piece": int(xpinn_n_interface_per_piece),
                "num_interface_pieces": int(len(interfaces)),
                "total_interface_points": int(sum(item["coords"][0].shape[0] for item in interfaces)),
                "xpinn_w_residual": float(xpinn_w_residual),
                "xpinn_w_avg": float(xpinn_w_avg),
                "interface_conditions": ["residual_continuity", "average_solution_continuity"],
                "additional_flux_continuity_used": False,
                "local_adaptive_activation": bool(xpinn_use_local_adaptive_activation),
                "xpinn_aaf_n": float(xpinn_aaf_n),
                "xpinn_aaf_init_na": float(xpinn_aaf_init_na),
                "n_ic": int(cfg_s.n_ic),
                "n_bc": int(cfg_s.n_bc),
                "n_f": int(cfg_s.n_f),
            },
        )

        t0 = time.time()

        model, history_df, interface_df, final_info = train_xpinn_fixed(
            model=model,
            cfg=cfg_s,
            interfaces=interfaces,
            xpinn_w_residual=xpinn_w_residual,
            xpinn_w_avg=xpinn_w_avg,
            xpinn_aaf_na_min=xpinn_aaf_na_min,
            xpinn_aaf_na_max=xpinn_aaf_na_max,
        )

        train_time = time.time() - t0

        metrics = evaluate_model_any(model, method_name, cfg_s, ref=ref)

        metrics.update({
            "equation": equation_name,
            "method": method_name,
            "seed": int(seed),
            "optimizer_main": "AdamW",
            "lbfgs_used": False,
            "wall_clock_sec_total": float(train_time),
            "num_parameters": parameter_count(model),
            "xpinn_num_subdomains": int(len(model.submodels)),
            "xpinn_spatial_dim": int(spatial_dim),
            "xpinn_x_split": float(model.x_split),
            "xpinn_t_split": float(model.t_split),
            "xpinn_n_interface_per_piece": int(xpinn_n_interface_per_piece),
            "xpinn_total_interface_points": int(sum(item["coords"][0].shape[0] for item in interfaces)),
            "xpinn_w_residual": float(xpinn_w_residual),
            "xpinn_w_avg": float(xpinn_w_avg),
            "xpinn_local_adaptive_activation": bool(xpinn_use_local_adaptive_activation),
        })

        if len(history_df) > 0:
            metrics["xpinn_interface_residual_loss_final_logged"] = float(history_df["interface_residual_loss"].iloc[-1])
            metrics["xpinn_interface_avg_loss_final_logged"] = float(history_df["interface_avg_solution_loss"].iloc[-1])

        save_history(history_df, run_dir, "history.csv")
        save_json(metrics, run_dir / "metrics_adamw.json")
        save_json(metrics, run_dir / "metrics_final.json")

        save_model_checkpoint(
            model,
            run_dir,
            "model_adamw_final.pt",
            extra=metrics,
        )

        save_model_checkpoint(
            model,
            run_dir,
            "model_final.pt",
            extra=metrics,
        )

        try_save_predictions(model, cfg_s, run_dir)
        save_xpinn_diagnostics(run_dir, model, history_df, interface_df, final_info, interfaces)

        mark_status(run_dir, "success", {"wall_clock_sec_total": float(train_time)})

        del model

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        gc.collect()

        return pd.DataFrame([metrics])

    except Exception as e:
        mark_status(run_dir, "failed", {"error": repr(e)})
        raise

def run_xpinn_seeds(
    equation_name,
    base_cfg,
    seeds,
    total_iters=ADAM_TOTAL_ITERS,
    skip_if_done=True,
    xpinn_n_interface_per_piece=512,
    xpinn_w_residual=1.0,
    xpinn_w_avg=20.0,
    xpinn_use_local_adaptive_activation=True,
    xpinn_aaf_n=5.0,
    xpinn_aaf_init_na=1.0,
):
    cfg_ref = copy.deepcopy(base_cfg)

    configure_runtime(cfg_ref)

    ref = build_reference_if_needed(cfg_ref)

    rows = []

    for seed in seeds:
        df_seed = train_xpinn_one_seed(
            equation_name=equation_name,
            base_cfg=base_cfg,
            seed=seed,
            ref=ref,
            total_iters=total_iters,
            skip_if_done=skip_if_done,
            xpinn_n_interface_per_piece=xpinn_n_interface_per_piece,
            xpinn_w_residual=xpinn_w_residual,
            xpinn_w_avg=xpinn_w_avg,
            xpinn_use_local_adaptive_activation=xpinn_use_local_adaptive_activation,
            xpinn_aaf_n=xpinn_aaf_n,
            xpinn_aaf_init_na=xpinn_aaf_init_na,
        )

        rows.append(df_seed)

    df = pd.concat(rows, ignore_index=True)

    summary_dir = RUNS_ROOT / safe_name(equation_name) / "_summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    out_path = summary_dir / "xpinn_metrics.csv"
    df.to_csv(out_path, index=False)

    print("saved:", out_path)

    return df

def reset_xpinn_runs(equation_name, seeds):
    for seed in seeds:
        run_dir = get_run_dir(equation_name, "XPINN", seed)
        if run_dir.exists():
            print("remove:", run_dir)
            shutil.rmtree(run_dir)
        else:
            print("not found:", run_dir)


EQUATION_NAME = '2d_shallowwater'

if "base_cfg" in globals():
    PUBLIC_BASE_CONFIG = copy.deepcopy(base_cfg)
elif "cfg" in globals():
    PUBLIC_BASE_CONFIG = copy.deepcopy(cfg)
else:
    raise RuntimeError(
        "The extracted benchmark runtime does not define cfg or base_cfg."
    )

SPECIALIZED_TRAINERS = {
    "aaf_pinn": train_aaf_one_seed,
    "lra_pinn": train_lra_one_seed,
    "sa_pinn": train_sa_one_seed,
    "rad_pinn": train_rad_one_seed,
    "rar_d_pinn": train_rard_one_seed,
    "gpinn_subsampled": train_gpinn_subsampled_one_seed,
    "cpinn": train_cpinn_one_seed,
    "xpinn": train_xpinn_one_seed,
}

SPECIALIZED_PUBLIC_NAMES = {'aaf_pinn': 'AAF-PINN', 'lra_pinn': 'LRA-PINN', 'sa_pinn': 'SA-PINN', 'rad_pinn': 'RAD-PINN', 'rar_d_pinn': 'RAR-D-PINN', 'gpinn_subsampled': 'gPINN-sub.', 'cpinn': 'cPINN', 'xpinn': 'XPINN'}
EXACT_METHOD_BLOCK_HASHES = {'rad_pinn': '131fa2ce52557a00ddbced3429badb3f26e3691182220333e5395bfa2142bf98', 'rar_d_pinn': 'b3c4a9247585642a4944b979c7f3857ab818e192aa3c57369f75db0ee42444ba', 'sa_pinn': '49fc3426c612ed4a5826b77b01f325519d6aef796ea6f06b2552cf1c828402d2', 'lra_pinn': 'b669557905bf03b94378d670071dc560650bf5e83fec94ee89b9372e0e916146', 'gpinn_subsampled': 'dc7bda62404906b55388ee6ba85eb7c1107b07b9f4e87bb0eca22979cf63a8e1', 'aaf_pinn': 'd682474c797dfd01b04d743ea1899ed03bc3a0d46b78335af76e5bd0eaa8b16f', 'cpinn': 'a392101848c63ef5ee13d5f457dfdd2fd07bb0772625d9e264c8b0d7c35bda0b', 'xpinn': 'cbed0371f0f81e8f2d41dbd7abfd21df548cf51812604d4b1b0654202a2445b4'}
GPINN_HELPERS_RECONSTRUCTED = True
GPINN_HELPER_RECONSTRUCTION_SHA256 = '7fa425dc0cd578c091b73ef2c8db96c3f2a28e6944490cc2e7db9d9f0073fb3c'
