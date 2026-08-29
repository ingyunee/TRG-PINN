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
import hashlib
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
    device: str = "auto"
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


    @classmethod
    def from_legacy_mapping(
        cls,
        mapping: dict,
        *,
        seed: int | None = None,
        device: str | None = None,
    ) -> "ShallowWater2DConfig":
        """Construct the canonical config from a saved manuscript run."""
        if not isinstance(mapping, dict):
            raise TypeError("mapping must be a dictionary")
        field_names = set(cls.__dataclass_fields__)
        values = {
            key: value
            for key, value in mapping.items()
            if key in field_names
        }
        if seed is not None:
            values["seed"] = int(seed)
        if device is not None:
            values["device"] = str(device)
        return cls(**values)

    def smoke_copy(
        self,
        *,
        seed: int | None = None,
        device: str | None = None,
    ) -> "ShallowWater2DConfig":
        clone = copy.deepcopy(self)
        if seed is not None:
            clone.seed = int(seed)
        if device is not None:
            clone.device = str(device)
        clone.save_outputs = True
        clone.warmup_iters = 2
        clone.gated_iters = 2
        clone.n_f = 64
        clone.n_ic = 32
        clone.n_bc = 32
        clone.eval_nxy = 24
        clone.eval_space_nxy = 16
        clone.eval_nt = 4
        clone.line_n = 64
        clone.radial_angles = 4
        clone.gate_eval_nxy = 24
        clone.cons_nxy = 12
        clone.cons_nt = 4
        clone.n_control_volumes = 2
        clone.cv_quad_nxy = 6
        clone.cv_quad_nt = 5
        clone.print_every = 1
        clone.history_every = 1
        return clone


cfg = ShallowWater2DConfig()
DEVICE: torch.device
DTYPE: torch.dtype


# ============================================================
# 2. Runtime and reproducibility
# ============================================================

def resolve_device(device_str: str) -> torch.device:
    text = str(device_str).strip().lower()
    if text == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    requested = torch.device(text)
    if requested.type == "cuda" and not torch.cuda.is_available():
        print("CUDA was requested but is unavailable. Falling back to CPU.")
        return torch.device("cpu")
    return requested


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



# Public release API ---------------------------------------------------------

REFERENCE_TYPE = "finite_volume_1024x1024_order2"
IMPLEMENTATION_REVISION = "valid_mask_all_probe_bounds_v1"


def build_model(config: ShallowWater2DConfig) -> MLP:
    configure_runtime(config)
    return MLP(config)


def count_parameters(model: nn.Module) -> int:
    return count_params(model)


def direction_tensor(
    config: ShallowWater2DConfig,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    resolved_device = DEVICE if device is None else torch.device(device)
    resolved_dtype = DTYPE if dtype is None else dtype
    count = int(config.ring_trace_pairs)
    theta = (
        torch.arange(
            count,
            device=resolved_device,
            dtype=resolved_dtype,
        )
        * (math.pi / count)
    )
    return torch.stack(
        [torch.cos(theta), torch.sin(theta)],
        dim=1,
    )


def residual_weight(
    gate: torch.Tensor,
    config: ShallowWater2DConfig,
) -> torch.Tensor:
    return (
        config.residual_floor
        + (1.0 - config.residual_floor)
        * (1.0 - gate)
    )


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_fv_reference(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
) -> dict[str, np.ndarray]:
    reference_path = Path(path).expanduser().resolve()
    if not reference_path.is_file():
        raise FileNotFoundError(reference_path)
    if expected_sha256 is not None:
        actual = sha256_file(reference_path)
        if actual != str(expected_sha256):
            raise AssertionError(
                f"Reference SHA-256 mismatch: {actual} != {expected_sha256}"
            )
    with np.load(reference_path, allow_pickle=False) as data:
        required = {"x", "y", "t", "H", "M", "N"}
        missing = required - set(data.files)
        if missing:
            raise KeyError(
                f"FV reference is missing fields: {sorted(missing)}"
            )
        reference = {
            key: np.asarray(data[key])
            for key in ("x", "y", "t", "H", "M", "N")
        }
    if reference["H"].shape != (41, 1024, 1024):
        raise AssertionError(
            f"Unexpected canonical FV field shape: {reference['H'].shape}"
        )
    return reference


def evaluate_checkpoint_model(
    model: nn.Module,
    name: str,
    reference: dict[str, np.ndarray],
    config: ShallowWater2DConfig,
) -> dict[str, float]:
    return evaluate_model(model, name, reference, config)


def train_warmup(model, config):
    return train_warmup_fixed(model, config)


def train_pinn_continuation(model, config):
    return train_vanilla_continuation_fixed(model, config)


def train_trg_continuation(model, config):
    return train_gated_fixed(model, config)


def _safe_output_root(
    output_root: str | Path,
    protected_root: str | Path,
) -> Path:
    root = Path(output_root).expanduser().resolve()
    protected = Path(protected_root).expanduser().resolve()
    try:
        root.relative_to(protected)
    except ValueError:
        pass
    else:
        raise ValueError(
            f"Refusing to write reproduction output inside {protected}"
        )
    root.mkdir(parents=True, exist_ok=True)
    return root


def _save_public_checkpoint(
    model: nn.Module,
    path: Path,
    metrics: dict,
) -> None:
    torch.save(
        {
            "model_state_dict": clone_state(model),
            "extra": metrics,
        },
        path,
    )


@torch.no_grad()
def _smoke_metrics(
    model: nn.Module,
    name: str,
    config: ShallowWater2DConfig,
) -> dict[str, float]:
    coordinates = torch.tensor(
        [
            [config.center_x, config.center_y, config.t_min],
            [0.5, 0.0, config.t_max],
            [-0.5, 0.25, config.t_max],
        ],
        device=DEVICE,
        dtype=DTYPE,
    )
    state = model(coordinates)
    if not torch.isfinite(state).all():
        raise FloatingPointError("Non-finite smoke-test prediction.")
    return {
        "model": name,
        "seed": int(config.seed),
        "smoke_test": True,
        "h_min": float(state[:, 0].min().cpu()),
        "h_max": float(state[:, 0].max().cpu()),
        "m_abs_mean": float(state[:, 1].abs().mean().cpu()),
        "n_abs_mean": float(state[:, 2].abs().mean().cpu()),
        "num_parameters": count_parameters(model),
    }


def run_paired_experiment(
    config: ShallowWater2DConfig,
    *,
    output_root: str | Path,
    protected_reported_root: str | Path,
    reference_path: str | Path | None = None,
    reference_sha256: str | None = None,
    overwrite: bool = False,
    smoke_test: bool = False,
) -> pd.DataFrame:
    """Run the canonical paired protocol outside frozen reported artifacts."""
    configure_runtime(config)
    set_seed(config.seed)

    root = _safe_output_root(
        output_root,
        protected_reported_root,
    )
    seed_root = root / "2d_shallowwater" / f"seed_{config.seed}"

    if (
        seed_root.exists()
        and any(seed_root.iterdir())
        and not overwrite
    ):
        raise FileExistsError(
            f"Reproduction output already exists: {seed_root}"
        )
    seed_root.mkdir(parents=True, exist_ok=True)

    save_json(
        {
            "equation": "2d_shallowwater",
            "seed": int(config.seed),
            "config": asdict(config),
            "implementation_revision": IMPLEMENTATION_REVISION,
            "reference_type": (
                None if reference_path is None else REFERENCE_TYPE
            ),
            "protocol": (
                "shared warm-up; paired PINN and TRG-PINN continuation; "
                "same RNG restored; no validation; final checkpoint evaluation"
            ),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(DEVICE),
            "smoke_test": bool(smoke_test),
        },
        seed_root / "config.json",
    )

    warmup_model = build_model(config).to(DEVICE, dtype=DTYPE)
    warmup_model, history_warmup = train_warmup_fixed(
        warmup_model,
        config,
    )
    warmup_state = clone_state(warmup_model)
    continuation_rng = get_rng_state()

    pinn_model = build_model(config).to(DEVICE, dtype=DTYPE)
    pinn_model.load_state_dict(warmup_state, strict=True)
    set_rng_state(continuation_rng)
    pinn_model, history_pinn = train_vanilla_continuation_fixed(
        pinn_model,
        config,
    )

    trg_model = build_model(config).to(DEVICE, dtype=DTYPE)
    trg_model.load_state_dict(warmup_state, strict=True)
    set_rng_state(continuation_rng)
    trg_model, history_trg = train_gated_fixed(
        trg_model,
        config,
    )

    if smoke_test:
        pinn_metrics = _smoke_metrics(
            pinn_model,
            "PINN",
            config,
        )
        trg_metrics = _smoke_metrics(
            trg_model,
            "TRG-PINN",
            config,
        )
    else:
        if reference_path is None:
            raise ValueError(
                "Full 2D shallow-water evaluation requires reference_path."
            )
        reference = load_fv_reference(
            reference_path,
            expected_sha256=reference_sha256,
        )
        pinn_metrics = evaluate_model(
            pinn_model,
            "PINN",
            reference,
            config,
        )
        trg_metrics = evaluate_model(
            trg_model,
            "TRG-PINN",
            reference,
            config,
        )
        del reference

    for item in (pinn_metrics, trg_metrics):
        item.update(
            {
                "equation": "2d_shallowwater",
                "method": item["model"],
                "implementation_revision": IMPLEMENTATION_REVISION,
                "warmup_iters": int(config.warmup_iters),
                "continuation_iters": int(config.gated_iters),
                "adam_total_iters": int(
                    config.warmup_iters + config.gated_iters
                ),
                "optimizer_main": "AdamW",
                "lbfgs_used": False,
                "num_parameters": count_parameters(pinn_model),
            }
        )

    history_warmup.to_csv(
        seed_root / "history_warmup.csv",
        index=False,
    )
    history_pinn.to_csv(
        seed_root / "history_pinn.csv",
        index=False,
    )
    history_trg.to_csv(
        seed_root / "history_trg_pinn.csv",
        index=False,
    )

    frame = pd.DataFrame([pinn_metrics, trg_metrics])
    frame.to_csv(
        seed_root / "metrics_final.csv",
        index=False,
    )
    save_json(
        pinn_metrics,
        seed_root / "metrics_pinn.json",
    )
    save_json(
        trg_metrics,
        seed_root / "metrics_trg_pinn.json",
    )

    torch.save(
        {
            "model_state_dict": warmup_state,
            "equation": "2d_shallowwater",
            "method": "shared_warmup",
            "seed": int(config.seed),
            "implementation_revision": IMPLEMENTATION_REVISION,
        },
        seed_root / "model_warmup.pt",
    )
    _save_public_checkpoint(
        pinn_model,
        seed_root / "model_pinn_final.pt",
        pinn_metrics,
    )
    _save_public_checkpoint(
        trg_model,
        seed_root / "model_trg_pinn_final.pt",
        trg_metrics,
    )
    (seed_root / "_SUCCESS").write_text(
        "success\n",
        encoding="utf-8",
    )
    return frame


__all__ = [
    "ShallowWater2DConfig",
    "build_model",
    "count_parameters",
    "configure_runtime",
    "resolve_device",
    "resolve_dtype",
    "direction_tensor",
    "residual_weight",
    "trace_ratio_gate_conservative",
    "swe2d_residual",
    "predict_points",
    "reference_points",
    "load_fv_reference",
    "evaluate_model",
    "evaluate_checkpoint_model",
    "run_paired_experiment",
]
