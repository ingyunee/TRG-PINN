"""2D compressible Euler rotated-Sod benchmark for TRG-PINN.

The numerical operations are extracted from the final post-valid-mask
manuscript notebook. Reported artifacts remain immutable; fresh runs are
written only to reproduction directories.
"""

from __future__ import annotations

import os
import json
import math
import time
import copy
import random
import platform
from pathlib import Path
from dataclasses import dataclass, asdict, fields, replace

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from IPython.display import display
except Exception:
    display = print


# ============================================================
# 1. Configuration
# ============================================================

@dataclass
class Euler2DRotatedSodConfig:
    # Runtime / reproducibility
    seed: int = 1234
    device: str = "cpu"          # "auto", "cpu", "cuda", "cuda:0", ...
    dtype: str = "float32"
    output_dir: str = "runs_euler2d_rotated_sod_trace_ratio_paper"
    experiment_name: str = "euler2d_rotated_sod_ring_trace_ratio_original_schedule"
    save_outputs: bool = True

    # Domain and rotated Sod setup
    x_min: float = -1.0
    x_max: float = 1.0
    y_min: float = -1.0
    y_max: float = 1.0
    t_min: float = 0.0
    t_max: float = 0.20
    x0: float = 0.0
    y0: float = 0.0
    theta_deg: float = 30.0
    gamma: float = 1.4

    # Left/right states in the normal direction
    rho_L: float = 1.0
    un_L: float = 0.0
    p_L: float = 1.0
    rho_R: float = 0.125
    un_R: float = 0.0
    p_R: float = 0.1

    # Network
    width: int = 128
    depth: int = 6
    activation: str = "tanh"
    rho_floor: float = 1.0e-6
    p_floor: float = 1.0e-8

    # Training budget. These are iterations with newly resampled collocation points.
    warmup_iters: int = 3000
    gated_iters: int = 7000
    lr_warmup: float = 8.0e-4
    lr_gated: float = 3.0e-4
    grad_clip: float = 1.0

    # Batch sizes
    n_f: int = 20000
    n_ic: int = 6000
    n_bc: int = 2500

    # Loss weights
    w_ic: float = 100.0
    w_bc: float = 1.0
    w_pde: float = 1.0

    # Trace-ratio gate schedule
    # h(p) = h0 * [h_min_factor + (h_max_factor-h_min_factor)*(1-p)^2]
    # cmin(p) = cmin_start + (cmin_end-cmin_start)*p
    ring_trace_pairs: int = 6   # centered traces make this equivalent to a 12-direction signed ring
    gate_tau: float = 1.0
    h_max_factor: float = 5.0
    h_min_factor: float = 2.0
    cmin_start: float = 0.50
    cmin_end: float = 0.70
    beta: float = 0.05
    residual_floor: float = 0.02

    # Gate state scaling. Original uploaded code used unit primitive scales.
    gate_rho_scale: float = 1.0
    gate_u_scale: float = 1.0
    gate_p_scale: float = 1.0

    # Logging
    print_every: int = 500
    history_every: int = 50

    # Evaluation
    eval_nxy: int = 220            # final-time spatial grid
    eval_space_nxy: int = 90       # space-time error grid in x-y
    eval_nt: int = 41              # space-time error time levels
    line_n: int = 1600             # normal-line evaluation
    gate_eval_nxy: int = 180

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
    ) -> "Euler2DRotatedSodConfig":
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
        return cls(**values)

    def smoke_copy(
        self,
        *,
        seed: int = 2026,
        device: str = "cpu",
    ) -> "Euler2DRotatedSodConfig":
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
            eval_space_nxy=24,
            eval_nt=4,
            line_n=240,
            gate_eval_nxy=30,
            cons_nxy=16,
            cons_nt=4,
            n_control_volumes=2,
            cv_quad_nxy=8,
            cv_quad_nt=8,
            cv_min_width=0.15,
            cv_min_duration=0.03,
        )


cfg = Euler2DRotatedSodConfig()
DEVICE = None
DTYPE = None


# ============================================================
# 2. Runtime helpers
# ============================================================

def resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if str(device_str).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device {device_str} was requested but CUDA is unavailable.")
    return torch.device(device_str)


def resolve_dtype(dtype_str: str):
    if dtype_str == "float64":
        return torch.float64
    if dtype_str == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_str}")


def configure_runtime(new_cfg: Euler2DRotatedSodConfig):
    global cfg, DEVICE, DTYPE
    cfg = new_cfg
    DEVICE = resolve_device(cfg.device)
    DTYPE = resolve_dtype(cfg.dtype)
    torch.set_default_dtype(DTYPE)
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
    return DEVICE, DTYPE


configure_runtime(cfg)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def set_rng_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state["torch_cuda"] is not None:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def to_numpy(x):
    return x.detach().cpu().numpy()


def cat_xyt(x, y, t):
    return torch.cat([x, y, t], dim=1)


def make_run_dir(cfg: Euler2DRotatedSodConfig) -> Path:
    run_dir = Path(cfg.output_dir) / cfg.experiment_name / f"seed_{cfg.seed}"
    if cfg.save_outputs:
        run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_json(obj, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def clone_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def inv_softplus(y: float) -> float:
    y = max(float(y), 1.0e-10)
    return math.log(math.expm1(y))


# ============================================================
# 3. Problem geometry, scales, and schedule
# ============================================================

def normal_vec(cfg: Euler2DRotatedSodConfig):
    th = math.radians(cfg.theta_deg)
    return np.array([math.cos(th), math.sin(th)], dtype=np.float64)


def normal_components(cfg: Euler2DRotatedSodConfig):
    n = normal_vec(cfg)
    return float(n[0]), float(n[1])


def eta_np(X, Y, cfg: Euler2DRotatedSodConfig):
    nx, ny = normal_components(cfg)
    return (np.asarray(X) - cfg.x0) * nx + (np.asarray(Y) - cfg.y0) * ny


def eta_torch(x, y, cfg: Euler2DRotatedSodConfig):
    nx, ny = normal_components(cfg)
    return (x - cfg.x0) * nx + (y - cfg.y0) * ny


def scales(cfg: Euler2DRotatedSodConfig):
    rho_scale = max(cfg.rho_L, cfg.rho_R, 1.0)
    p_scale = max(cfg.p_L, cfg.p_R, 1.0)
    cL = math.sqrt(cfg.gamma * cfg.p_L / cfg.rho_L)
    cR = math.sqrt(cfg.gamma * cfg.p_R / cfg.rho_R)
    u_scale = max(abs(cfg.un_L), abs(cfg.un_R), cL, cR, 1.0)
    mom_scale = max(rho_scale * u_scale, 1.0)
    e_scale = p_scale / (cfg.gamma - 1.0) + 0.5 * rho_scale * u_scale**2
    return rho_scale, u_scale, p_scale, mom_scale, e_scale


def h0(cfg: Euler2DRotatedSodConfig):
    Lx = cfg.x_max - cfg.x_min
    Ly = cfg.y_max - cfg.y_min
    return math.sqrt(Lx * Ly) / math.sqrt(cfg.n_f)


def schedule_from_progress(progress: float, cfg: Euler2DRotatedSodConfig):
    p = float(np.clip(progress, 0.0, 1.0))
    base = h0(cfg)
    h = base * (cfg.h_min_factor + (cfg.h_max_factor - cfg.h_min_factor) * (1.0 - p) ** 2)
    cmin = cfg.cmin_start + (cfg.cmin_end - cfg.cmin_start) * p
    return h, cmin, p


def schedule_from_iter(iteration: int, total_iters: int, cfg: Euler2DRotatedSodConfig):
    return schedule_from_progress(iteration / max(1, total_iters), cfg)


# ============================================================
# 4. Exact Sod solver and rotated exact solution
# ============================================================

def pressure_function(p, rho_k, p_k, a_k, gamma):
    if p > p_k:  # shock
        A = 2.0 / ((gamma + 1.0) * rho_k)
        B = (gamma - 1.0) / (gamma + 1.0) * p_k
        f = (p - p_k) * math.sqrt(A / (p + B))
        fd = math.sqrt(A / (p + B)) * (1.0 - 0.5 * (p - p_k) / (p + B))
    else:        # rarefaction
        pr = p / p_k
        f = 2.0 * a_k / (gamma - 1.0) * (pr ** ((gamma - 1.0) / (2.0 * gamma)) - 1.0)
        fd = (1.0 / (rho_k * a_k)) * pr ** (-(gamma + 1.0) / (2.0 * gamma))
    return f, fd


def solve_star_region(cfg: Euler2DRotatedSodConfig):
    g = cfg.gamma
    aL = math.sqrt(g * cfg.p_L / cfg.rho_L)
    aR = math.sqrt(g * cfg.p_R / cfg.rho_R)
    p_pv = 0.5 * (cfg.p_L + cfg.p_R) - 0.125 * (cfg.un_R - cfg.un_L) * (cfg.rho_L + cfg.rho_R) * (aL + aR)
    p = max(1.0e-8, p_pv)

    for _ in range(100):
        fL, fdL = pressure_function(p, cfg.rho_L, cfg.p_L, aL, g)
        fR, fdR = pressure_function(p, cfg.rho_R, cfg.p_R, aR, g)
        p_new = p - (fL + fR + cfg.un_R - cfg.un_L) / (fdL + fdR)
        p_new = max(1.0e-10, p_new)
        if abs(p_new - p) / (0.5 * (p_new + p) + 1.0e-12) < 1.0e-12:
            p = p_new
            break
        p = p_new

    fL, _ = pressure_function(p, cfg.rho_L, cfg.p_L, aL, g)
    fR, _ = pressure_function(p, cfg.rho_R, cfg.p_R, aR, g)
    u = 0.5 * (cfg.un_L + cfg.un_R + fR - fL)
    return p, u, aL, aR


def sod_wave_speeds(cfg: Euler2DRotatedSodConfig):
    g = cfg.gamma
    gm1 = g - 1.0
    gp1 = g + 1.0
    pstar, ustar, aL, aR = solve_star_region(cfg)

    if pstar > cfg.p_L:
        left_head = cfg.un_L - aL * math.sqrt((gp1 / (2.0 * g)) * (pstar / cfg.p_L) + gm1 / (2.0 * g))
        left_tail = left_head
    else:
        astarL = aL * (pstar / cfg.p_L) ** (gm1 / (2.0 * g))
        left_head = cfg.un_L - aL
        left_tail = ustar - astarL

    if pstar > cfg.p_R:
        right_head = cfg.un_R + aR * math.sqrt((gp1 / (2.0 * g)) * (pstar / cfg.p_R) + gm1 / (2.0 * g))
        right_tail = right_head
    else:
        astarR = aR * (pstar / cfg.p_R) ** (gm1 / (2.0 * g))
        right_head = cfg.un_R + aR
        right_tail = ustar + astarR

    return {
        "p_star": pstar,
        "un_star": ustar,
        "left_head": left_head,
        "left_tail": left_tail,
        "contact": ustar,
        "right_head": right_head,
        "right_tail": right_tail,
    }


def exact_sod_1d_np(S, T, cfg: Euler2DRotatedSodConfig):
    S = np.asarray(S, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64)
    rho = np.zeros_like(S, dtype=np.float64)
    un = np.zeros_like(S, dtype=np.float64)
    p = np.zeros_like(S, dtype=np.float64)

    init = T <= 1.0e-14
    left0 = S < 0.0
    rho[init] = np.where(left0[init], cfg.rho_L, cfg.rho_R)
    un[init] = np.where(left0[init], cfg.un_L, cfg.un_R)
    p[init] = np.where(left0[init], cfg.p_L, cfg.p_R)

    mask = ~init
    if not np.any(mask):
        return rho, un, p

    xi = np.zeros_like(S, dtype=np.float64)
    xi[mask] = S[mask] / T[mask]

    g = cfg.gamma
    gm1 = g - 1.0
    gp1 = g + 1.0
    pstar, ustar, aL, aR = solve_star_region(cfg)

    left = mask & (xi <= ustar)
    right = mask & (xi > ustar)

    # Left wave
    if pstar > cfg.p_L:
        pr = pstar / cfg.p_L
        SL = cfg.un_L - aL * math.sqrt((gp1 / (2.0 * g)) * pr + gm1 / (2.0 * g))
        state = left & (xi <= SL)
        star = left & (xi > SL)
        rho[state] = cfg.rho_L
        un[state] = cfg.un_L
        p[state] = cfg.p_L
        rho_star = cfg.rho_L * ((pr + gm1 / gp1) / ((gm1 / gp1) * pr + 1.0))
        rho[star] = rho_star
        un[star] = ustar
        p[star] = pstar
    else:
        aStarL = aL * (pstar / cfg.p_L) ** (gm1 / (2.0 * g))
        head = cfg.un_L - aL
        tail = ustar - aStarL
        state = left & (xi <= head)
        star = left & (xi >= tail)
        fan = left & (xi > head) & (xi < tail)
        rho[state] = cfg.rho_L
        un[state] = cfg.un_L
        p[state] = cfg.p_L
        rho_star = cfg.rho_L * (pstar / cfg.p_L) ** (1.0 / g)
        rho[star] = rho_star
        un[star] = ustar
        p[star] = pstar
        xi_f = xi[fan]
        u_f = 2.0 / gp1 * (aL + 0.5 * gm1 * cfg.un_L + xi_f)
        a_f = 2.0 / gp1 * (aL + 0.5 * gm1 * (cfg.un_L - xi_f))
        rho[fan] = cfg.rho_L * (a_f / aL) ** (2.0 / gm1)
        un[fan] = u_f
        p[fan] = cfg.p_L * (a_f / aL) ** (2.0 * g / gm1)

    # Right wave
    if pstar > cfg.p_R:
        pr = pstar / cfg.p_R
        SR = cfg.un_R + aR * math.sqrt((gp1 / (2.0 * g)) * pr + gm1 / (2.0 * g))
        state = right & (xi >= SR)
        star = right & (xi < SR)
        rho[state] = cfg.rho_R
        un[state] = cfg.un_R
        p[state] = cfg.p_R
        rho_star = cfg.rho_R * ((pr + gm1 / gp1) / ((gm1 / gp1) * pr + 1.0))
        rho[star] = rho_star
        un[star] = ustar
        p[star] = pstar
    else:
        aStarR = aR * (pstar / cfg.p_R) ** (gm1 / (2.0 * g))
        head = cfg.un_R + aR
        tail = ustar + aStarR
        state = right & (xi >= head)
        star = right & (xi <= tail)
        fan = right & (xi < head) & (xi > tail)
        rho[state] = cfg.rho_R
        un[state] = cfg.un_R
        p[state] = cfg.p_R
        rho_star = cfg.rho_R * (pstar / cfg.p_R) ** (1.0 / g)
        rho[star] = rho_star
        un[star] = ustar
        p[star] = pstar
        xi_f = xi[fan]
        u_f = 2.0 / gp1 * (-aR + 0.5 * gm1 * cfg.un_R + xi_f)
        a_f = 2.0 / gp1 * (aR - 0.5 * gm1 * (cfg.un_R - xi_f))
        rho[fan] = cfg.rho_R * (a_f / aR) ** (2.0 / gm1)
        un[fan] = u_f
        p[fan] = cfg.p_R * (a_f / aR) ** (2.0 * g / gm1)

    return rho, un, p


def rotated_exact_np(X, Y, T, cfg: Euler2DRotatedSodConfig):
    S = eta_np(X, Y, cfg)
    rho, un, p = exact_sod_1d_np(S, T, cfg)
    nx, ny = normal_components(cfg)
    u = un * nx
    v = un * ny
    return rho, u, v, p


def exact_sod_1d_torch(s, t, cfg: Euler2DRotatedSodConfig):
    g = cfg.gamma
    gm1 = g - 1.0
    gp1 = g + 1.0
    pstar, ustar, aL, aR = solve_star_region(cfg)
    t_safe = torch.clamp(t, min=1.0e-8)
    xi = s / t_safe

    rho = torch.empty_like(s)
    un = torch.empty_like(s)
    p = torch.empty_like(s)

    # Left rarefaction/shock and right shock/rarefaction; standard Sod uses left rarefaction, right shock.
    rho_star_L = cfg.rho_L * (pstar / cfg.p_L) ** (1.0 / g)
    a_star_L = aL * (pstar / cfg.p_L) ** (gm1 / (2.0 * g))
    g1 = gm1 / gp1
    rho_star_R = cfg.rho_R * ((pstar / cfg.p_R + g1) / (g1 * pstar / cfg.p_R + 1.0))
    s_hl = cfg.un_L - aL
    s_tl = ustar - a_star_L
    s_r = cfg.un_R + aR * math.sqrt((gp1 / (2.0 * g)) * (pstar / cfg.p_R) + gm1 / (2.0 * g))

    mask_left = xi <= s_hl
    rho[mask_left] = cfg.rho_L
    un[mask_left] = cfg.un_L
    p[mask_left] = cfg.p_L

    mask_fan = (xi > s_hl) & (xi <= s_tl)
    u_fan = 2.0 / gp1 * (aL + 0.5 * gm1 * cfg.un_L + xi)
    a_fan = 2.0 / gp1 * (aL + 0.5 * gm1 * (cfg.un_L - xi))
    rho[mask_fan] = cfg.rho_L * (a_fan[mask_fan] / aL) ** (2.0 / gm1)
    un[mask_fan] = u_fan[mask_fan]
    p[mask_fan] = cfg.p_L * (a_fan[mask_fan] / aL) ** (2.0 * g / gm1)

    mask_star_l = (xi > s_tl) & (xi <= ustar)
    rho[mask_star_l] = rho_star_L
    un[mask_star_l] = ustar
    p[mask_star_l] = pstar

    mask_star_r = (xi > ustar) & (xi <= s_r)
    rho[mask_star_r] = rho_star_R
    un[mask_star_r] = ustar
    p[mask_star_r] = pstar

    mask_right = xi > s_r
    rho[mask_right] = cfg.rho_R
    un[mask_right] = cfg.un_R
    p[mask_right] = cfg.p_R

    initial_mask = t <= 1.0e-8
    left0 = s < 0.0
    rho0 = torch.where(left0, torch.full_like(s, cfg.rho_L), torch.full_like(s, cfg.rho_R))
    un0 = torch.where(left0, torch.full_like(s, cfg.un_L), torch.full_like(s, cfg.un_R))
    p0 = torch.where(left0, torch.full_like(s, cfg.p_L), torch.full_like(s, cfg.p_R))
    rho = torch.where(initial_mask, rho0, rho)
    un = torch.where(initial_mask, un0, un)
    p = torch.where(initial_mask, p0, p)
    return rho, un, p


def rotated_exact_torch(x, y, t, cfg: Euler2DRotatedSodConfig):
    s = eta_torch(x, y, cfg)
    rho, un, p = exact_sod_1d_torch(s, t, cfg)
    nx, ny = normal_components(cfg)
    u = un * nx
    v = un * ny
    return torch.cat([rho, u, v, p], dim=1)


def rotated_initial_torch(x, y, cfg: Euler2DRotatedSodConfig):
    s = eta_torch(x, y, cfg)
    left = s < 0.0
    rho = torch.where(left, torch.full_like(x, cfg.rho_L), torch.full_like(x, cfg.rho_R))
    un = torch.where(left, torch.full_like(x, cfg.un_L), torch.full_like(x, cfg.un_R))
    p = torch.where(left, torch.full_like(x, cfg.p_L), torch.full_like(x, cfg.p_R))
    nx, ny = normal_components(cfg)
    u = un * nx
    v = un * ny
    return torch.cat([rho, u, v, p], dim=1)


# ============================================================
# 5. Neural network and Euler residual
# ============================================================

class MLP(nn.Module):
    def __init__(self, cfg: Euler2DRotatedSodConfig):
        super().__init__()
        self.cfg = cfg
        self.rho_floor = cfg.rho_floor
        self.p_floor = cfg.p_floor

        if cfg.activation.lower() == "tanh":
            act = nn.Tanh
        elif cfg.activation.lower() == "silu":
            act = nn.SiLU
        else:
            raise ValueError(f"Unsupported activation: {cfg.activation}")

        layers = []
        dim = 3
        for _ in range(cfg.depth):
            layer = nn.Linear(dim, cfg.width)
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)
            layers += [layer, act()]
            dim = cfg.width
        self.trunk = nn.Sequential(*layers)
        self.out = nn.Linear(dim, 4)
        nn.init.xavier_normal_(self.out.weight)
        nn.init.zeros_(self.out.bias)

        rho_mean = 0.5 * (cfg.rho_L + cfg.rho_R)
        p_mean = 0.5 * (cfg.p_L + cfg.p_R)
        with torch.no_grad():
            self.out.bias[0].fill_(inv_softplus(rho_mean - cfg.rho_floor))
            self.out.bias[1].fill_(0.0)
            self.out.bias[2].fill_(0.0)
            self.out.bias[3].fill_(inv_softplus(p_mean - cfg.p_floor))

    def forward(self, xyt):
        x = xyt[:, 0:1]
        y = xyt[:, 1:2]
        t = xyt[:, 2:3]
        xh = 2.0 * (x - self.cfg.x_min) / (self.cfg.x_max - self.cfg.x_min) - 1.0
        yh = 2.0 * (y - self.cfg.y_min) / (self.cfg.y_max - self.cfg.y_min) - 1.0
        th = 2.0 * (t - self.cfg.t_min) / (self.cfg.t_max - self.cfg.t_min) - 1.0
        raw = self.out(self.trunk(torch.cat([xh, yh, th], dim=1)))
        rho = self.rho_floor + F.softplus(raw[:, 0:1])
        u = raw[:, 1:2]
        v = raw[:, 2:3]
        p = self.p_floor + F.softplus(raw[:, 3:4])
        return torch.cat([rho, u, v, p], dim=1)


def prim_to_cons_torch(W, cfg: Euler2DRotatedSodConfig):
    rho = W[:, 0:1]
    u = W[:, 1:2]
    v = W[:, 2:3]
    p = W[:, 3:4]
    m1 = rho * u
    m2 = rho * v
    E = p / (cfg.gamma - 1.0) + 0.5 * rho * (u.pow(2) + v.pow(2))
    return rho, m1, m2, E


def flux_torch(W, cfg: Euler2DRotatedSodConfig):
    rho, m1, m2, E = prim_to_cons_torch(W, cfg)
    u = W[:, 1:2]
    v = W[:, 2:3]
    p = W[:, 3:4]
    Fv = (m1, m1 * u + p, m1 * v, u * (E + p))
    Gv = (m2, m2 * u, m2 * v + p, v * (E + p))
    return Fv, Gv


def euler2d_residual(model, x, y, t, cfg: Euler2DRotatedSodConfig):
    x = x.detach().clone().requires_grad_(True)
    y = y.detach().clone().requires_grad_(True)
    t = t.detach().clone().requires_grad_(True)
    W = model(cat_xyt(x, y, t))
    U = prim_to_cons_torch(W, cfg)
    Fv, Gv = flux_torch(W, cfg)

    residuals = []
    for Uk, Fk, Gk in zip(U, Fv, Gv):
        Uk_t = torch.autograd.grad(Uk, t, grad_outputs=torch.ones_like(Uk), create_graph=True, retain_graph=True)[0]
        Fk_x = torch.autograd.grad(Fk, x, grad_outputs=torch.ones_like(Fk), create_graph=True, retain_graph=True)[0]
        Gk_y = torch.autograd.grad(Gk, y, grad_outputs=torch.ones_like(Gk), create_graph=True, retain_graph=True)[0]
        residuals.append(Uk_t + Fk_x + Gk_y)

    r1, r2, r3, r4 = residuals
    rho_s, u_s, p_s, mom_s, e_s = scales(cfg)
    R2 = (r1 / rho_s).pow(2) + (r2 / mom_s).pow(2) + (r3 / mom_s).pow(2) + (r4 / e_s).pow(2)
    return W, r1, r2, r3, r4, R2


# ============================================================
# 6. Sampling and losses
# ============================================================

def sample_f(n: int, cfg: Euler2DRotatedSodConfig):
    x = cfg.x_min + (cfg.x_max - cfg.x_min) * torch.rand(n, 1, device=DEVICE, dtype=DTYPE)
    y = cfg.y_min + (cfg.y_max - cfg.y_min) * torch.rand(n, 1, device=DEVICE, dtype=DTYPE)
    t = cfg.t_min + (cfg.t_max - cfg.t_min) * torch.rand(n, 1, device=DEVICE, dtype=DTYPE)
    return x, y, t


def sample_ic(n: int, cfg: Euler2DRotatedSodConfig):
    x = cfg.x_min + (cfg.x_max - cfg.x_min) * torch.rand(n, 1, device=DEVICE, dtype=DTYPE)
    y = cfg.y_min + (cfg.y_max - cfg.y_min) * torch.rand(n, 1, device=DEVICE, dtype=DTYPE)
    t = torch.zeros_like(x)
    with torch.no_grad():
        target = rotated_initial_torch(x, y, cfg)
    return x, y, t, target


def sample_bc_exact(n: int, cfg: Euler2DRotatedSodConfig):
    counts = [n // 4, n // 4, n // 4, n - 3 * (n // 4)]
    xs, ys, ts = [], [], []
    for side, c in enumerate(counts):
        if c <= 0:
            continue
        tt = cfg.t_min + (cfg.t_max - cfg.t_min) * torch.rand(c, 1, device=DEVICE, dtype=DTYPE)
        if side == 0:  # x = xmin
            yy = cfg.y_min + (cfg.y_max - cfg.y_min) * torch.rand(c, 1, device=DEVICE, dtype=DTYPE)
            xx = torch.full_like(yy, cfg.x_min)
        elif side == 1:  # x = xmax
            yy = cfg.y_min + (cfg.y_max - cfg.y_min) * torch.rand(c, 1, device=DEVICE, dtype=DTYPE)
            xx = torch.full_like(yy, cfg.x_max)
        elif side == 2:  # y = ymin
            xx = cfg.x_min + (cfg.x_max - cfg.x_min) * torch.rand(c, 1, device=DEVICE, dtype=DTYPE)
            yy = torch.full_like(xx, cfg.y_min)
        else:  # y = ymax
            xx = cfg.x_min + (cfg.x_max - cfg.x_min) * torch.rand(c, 1, device=DEVICE, dtype=DTYPE)
            yy = torch.full_like(xx, cfg.y_max)
        xs.append(xx); ys.append(yy); ts.append(tt)
    x = torch.cat(xs, dim=0)
    y = torch.cat(ys, dim=0)
    t = torch.cat(ts, dim=0)
    with torch.no_grad():
        target = rotated_exact_torch(x, y, t, cfg)
    idx = torch.randperm(x.shape[0], device=DEVICE)
    return x[idx], y[idx], t[idx], target[idx]


def scaled_primitive_mse(W, target, cfg: Euler2DRotatedSodConfig):
    rho_s, u_s, p_s, _, _ = scales(cfg)
    return (
        ((W[:, 0:1] - target[:, 0:1]) / rho_s).pow(2)
        + ((W[:, 1:2] - target[:, 1:2]) / u_s).pow(2)
        + ((W[:, 2:3] - target[:, 2:3]) / u_s).pow(2)
        + ((W[:, 3:4] - target[:, 3:4]) / p_s).pow(2)
    ).mean()


def get_ring_directions(cfg: Euler2DRotatedSodConfig):
    m = cfg.ring_trace_pairs
    theta = torch.arange(m, device=DEVICE, dtype=DTYPE) * (math.pi / m)
    return torch.stack([torch.cos(theta), torch.sin(theta)], dim=1)


@torch.no_grad()
def trace_ratio_gate_primitive(model, x, y, t, h_probe: float, cmin: float, cfg: Euler2DRotatedSodConfig):
    npts = x.shape[0]
    directions = get_ring_directions(cfg)
    m = directions.shape[0]
    dx = directions[:, 0].view(1, m)
    dy = directions[:, 1].view(1, m)

    x0 = x.view(npts, 1)
    y0 = y.view(npts, 1)
    t0 = t.view(npts, 1)

    xph = x0 + h_probe * dx
    xmh = x0 - h_probe * dx
    yph = y0 + h_probe * dy
    ymh = y0 - h_probe * dy
    hp2 = 2.0 * h_probe
    xp2 = x0 + hp2 * dx
    xm2 = x0 - hp2 * dx
    yp2 = y0 + hp2 * dy
    ym2 = y0 - hp2 * dy

    valid = (
        (xm2 >= cfg.x_min) & (xm2 <= cfg.x_max)
        & (xp2 >= cfg.x_min) & (xp2 <= cfg.x_max)
        & (ym2 >= cfg.y_min) & (ym2 <= cfg.y_max)
        & (yp2 >= cfg.y_min) & (yp2 <= cfg.y_max)
    ).to(DTYPE)

    xph = xph.clamp(cfg.x_min, cfg.x_max)
    xmh = xmh.clamp(cfg.x_min, cfg.x_max)
    yph = yph.clamp(cfg.y_min, cfg.y_max)
    ymh = ymh.clamp(cfg.y_min, cfg.y_max)
    xp2 = xp2.clamp(cfg.x_min, cfg.x_max)
    xm2 = xm2.clamp(cfg.x_min, cfg.x_max)
    yp2 = yp2.clamp(cfg.y_min, cfg.y_max)
    ym2 = ym2.clamp(cfg.y_min, cfg.y_max)
    tt = t0.repeat(1, m)

    def eval_state(xx, yy):
        xyt = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), tt.reshape(-1, 1)], dim=1)
        return model(xyt).reshape(npts, m, 4)

    Wph = eval_state(xph, yph)
    Wmh = eval_state(xmh, ymh)
    Wp2 = eval_state(xp2, yp2)
    Wm2 = eval_state(xm2, ym2)

    def scaled_jump(Wp, Wm):
        d = Wp - Wm
        d_scaled = torch.stack([
            d[:, :, 0] / cfg.gate_rho_scale,
            d[:, :, 1] / cfg.gate_u_scale,
            d[:, :, 2] / cfg.gate_u_scale,
            d[:, :, 3] / cfg.gate_p_scale,
        ], dim=2)
        return torch.sqrt(d_scaled.pow(2).sum(dim=2) + 1.0e-12)

    Jh = scaled_jump(Wph, Wmh)
    J2h = scaled_jump(Wp2, Wm2)
    C = torch.clamp(Jh / (J2h + 1.0e-8), 0.0, 2.0)
    valid_sum = valid.sum()
    Jbar_valid = (Jh * valid).sum() / (valid_sum + 1.0e-8)
    Jbar_all = Jh.mean()
    Jbar = torch.where(valid_sum > 0, Jbar_valid, Jbar_all).clamp_min(1.0e-8)
    Jhat = Jh / Jbar

    g_jump = torch.sigmoid((Jhat - cfg.gate_tau) / cfg.beta)
    g_ratio = torch.sigmoid((C - cmin) / cfg.beta)
    g_dir = g_jump * g_ratio * valid
    g = g_dir.max(dim=1, keepdim=True).values
    Cmax = (C * valid).max(dim=1, keepdim=True).values
    Jhatmax = (Jhat * valid).max(dim=1, keepdim=True).values
    return g, C, Jhat, Jbar, valid, Cmax, Jhatmax


def vanilla_loss(model, cfg: Euler2DRotatedSodConfig):
    x_ic, y_ic, t_ic, target_ic = sample_ic(cfg.n_ic, cfg)
    x_bc, y_bc, t_bc, target_bc = sample_bc_exact(cfg.n_bc, cfg)
    x_f, y_f, t_f = sample_f(cfg.n_f, cfg)

    W_ic = model(cat_xyt(x_ic, y_ic, t_ic))
    W_bc = model(cat_xyt(x_bc, y_bc, t_bc))
    _, _, _, _, _, R2 = euler2d_residual(model, x_f, y_f, t_f, cfg)

    loss_ic = scaled_primitive_mse(W_ic, target_ic, cfg)
    loss_bc = scaled_primitive_mse(W_bc, target_bc, cfg)
    loss_pde = R2.mean()
    loss = cfg.w_ic * loss_ic + cfg.w_bc * loss_bc + cfg.w_pde * loss_pde
    return loss, {"ic": loss_ic.detach(), "bc": loss_bc.detach(), "pde": loss_pde.detach(), "weighted_pde": loss_pde.detach()}


def trace_ratio_gated_loss(model, iteration: int, total_iters: int, cfg: Euler2DRotatedSodConfig):
    h_probe, cmin, progress = schedule_from_iter(iteration, total_iters, cfg)
    x_ic, y_ic, t_ic, target_ic = sample_ic(cfg.n_ic, cfg)
    x_bc, y_bc, t_bc, target_bc = sample_bc_exact(cfg.n_bc, cfg)
    x_f, y_f, t_f = sample_f(cfg.n_f, cfg)

    W_ic = model(cat_xyt(x_ic, y_ic, t_ic))
    W_bc = model(cat_xyt(x_bc, y_bc, t_bc))
    _, _, _, _, _, R2 = euler2d_residual(model, x_f, y_f, t_f, cfg)
    gate, C, Jhat, Jbar, valid, Cmax, Jhatmax = trace_ratio_gate_primitive(model, x_f, y_f, t_f, h_probe, cmin, cfg)
    weight = cfg.residual_floor + (1.0 - cfg.residual_floor) * (1.0 - gate)

    loss_ic = scaled_primitive_mse(W_ic, target_ic, cfg)
    loss_bc = scaled_primitive_mse(W_bc, target_bc, cfg)
    loss_pde_raw = R2.mean()
    loss_pde_weighted = (weight * R2).sum() / (weight.sum() + 1.0e-8)
    loss = cfg.w_ic * loss_ic + cfg.w_bc * loss_bc + cfg.w_pde * loss_pde_weighted

    valid_mask = valid > 0
    C_mean = C[valid_mask].mean() if torch.any(valid_mask) else C.mean()
    Jhat_valid_mean = Jhat[valid_mask].mean() if torch.any(valid_mask) else Jhat.mean()

    return loss, {
        "ic": loss_ic.detach(), "bc": loss_bc.detach(), "pde": loss_pde_raw.detach(), "weighted_pde": loss_pde_weighted.detach(),
        "gate_mean": gate.mean().detach(), "gate_max": gate.max().detach(),
        "gate_active_gt_0p5": (gate > 0.5).to(DTYPE).mean().detach(),
        "gate_active_gt_0p1": (gate > 0.1).to(DTYPE).mean().detach(),
        "C_mean": C_mean.detach(), "C_max": C.max().detach(),
        "Jhat_mean": Jhat_valid_mean.detach(), "Jhat_max": Jhat.max().detach(), "Jbar": Jbar.detach(),
        "W_mean": weight.mean().detach(), "W_min": weight.min().detach(),
        "h": torch.tensor(h_probe, device=DEVICE, dtype=DTYPE), "cmin": torch.tensor(cmin, device=DEVICE, dtype=DTYPE),
        "progress": torch.tensor(progress, device=DEVICE, dtype=DTYPE), "valid_frac": valid.mean().detach(),
    }


# ============================================================
# 7. Training loops
# ============================================================

def train_warmup_fixed(model, cfg: Euler2DRotatedSodConfig):
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr_warmup)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.warmup_iters, eta_min=cfg.lr_warmup * 0.05)
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
            history.append({"phase": "warmup", "iter": it, "total_iter": it, "loss": float(loss.detach().cpu()),
                            "ic": float(parts["ic"].cpu()), "bc": float(parts["bc"].cpu()),
                            "pde": float(parts["pde"].cpu()), "weighted_pde": float(parts["weighted_pde"].cpu()),
                            "lr": float(opt.param_groups[0]["lr"])})
        if it == 1 or it % cfg.print_every == 0 or it == cfg.warmup_iters:
            print(f"[warmup] {it:6d}/{cfg.warmup_iters} loss={float(loss.detach().cpu()):.3e} "
                  f"ic={parts['ic'].item():.1e} bc={parts['bc'].item():.1e} pde={parts['pde'].item():.1e}")
    return model, pd.DataFrame(history)


def train_vanilla_continuation_fixed(model, cfg: Euler2DRotatedSodConfig):
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr_gated)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.gated_iters, eta_min=cfg.lr_gated * 0.03)
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
            history.append({"phase": "PINN_continuation", "iter": it, "total_iter": cfg.warmup_iters + it,
                            "loss": float(loss.detach().cpu()), "ic": float(parts["ic"].cpu()),
                            "bc": float(parts["bc"].cpu()), "pde": float(parts["pde"].cpu()),
                            "weighted_pde": float(parts["weighted_pde"].cpu()), "lr": float(opt.param_groups[0]["lr"])})
        if it == 1 or it % cfg.print_every == 0 or it == cfg.gated_iters:
            print(f"[PINN-cont] {it:6d}/{cfg.gated_iters} loss={float(loss.detach().cpu()):.3e} "
                  f"ic={parts['ic'].item():.1e} bc={parts['bc'].item():.1e} pde={parts['pde'].item():.1e}")
    return model, pd.DataFrame(history)


def train_gated_fixed(model, cfg: Euler2DRotatedSodConfig):
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr_gated)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.gated_iters, eta_min=cfg.lr_gated * 0.03)
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
                "phase": "tPINN_trace_ratio", "iter": it, "total_iter": cfg.warmup_iters + it,
                "loss": float(loss.detach().cpu()), "ic": float(parts["ic"].cpu()), "bc": float(parts["bc"].cpu()),
                "pde": float(parts["pde"].cpu()), "weighted_pde": float(parts["weighted_pde"].cpu()),
                "gate_mean": float(parts["gate_mean"].cpu()), "gate_max": float(parts["gate_max"].cpu()),
                "gate_active_gt_0p5": float(parts["gate_active_gt_0p5"].cpu()),
                "gate_active_gt_0p1": float(parts["gate_active_gt_0p1"].cpu()),
                "C_mean": float(parts["C_mean"].cpu()), "C_max": float(parts["C_max"].cpu()),
                "Jhat_mean": float(parts["Jhat_mean"].cpu()), "Jhat_max": float(parts["Jhat_max"].cpu()),
                "Jbar": float(parts["Jbar"].cpu()), "W_mean": float(parts["W_mean"].cpu()), "W_min": float(parts["W_min"].cpu()),
                "h": float(parts["h"].cpu()), "cmin": float(parts["cmin"].cpu()), "progress": float(parts["progress"].cpu()),
                "valid_frac": float(parts["valid_frac"].cpu()), "lr": float(opt.param_groups[0]["lr"]),
            })
        if it == 1 or it % cfg.print_every == 0 or it == cfg.gated_iters:
            print(f"[gated] {it:6d}/{cfg.gated_iters} loss={float(loss.detach().cpu()):.3e} "
                  f"ic={parts['ic'].item():.1e} bc={parts['bc'].item():.1e} "
                  f"pde={parts['pde'].item():.1e} wpde={parts['weighted_pde'].item():.1e} "
                  f"g={parts['gate_mean'].item():.3f} act>.5={parts['gate_active_gt_0p5'].item():.3f} "
                  f"C={parts['C_mean'].item():.3f} Jmax={parts['Jhat_max'].item():.1f} "
                  f"W={parts['W_mean'].item():.3f} h={parts['h'].item():.5f} cmin={parts['cmin'].item():.2f}")
    return model, pd.DataFrame(history)


# ============================================================
# 8. Prediction and metrics
# ============================================================

def primitive_to_conserved_np(rho, u, v, p, cfg: Euler2DRotatedSodConfig):
    m1 = rho * u
    m2 = rho * v
    E = p / (cfg.gamma - 1.0) + 0.5 * rho * (u**2 + v**2)
    return rho, m1, m2, E


def flux_np_2d(rho, u, v, p, cfg: Euler2DRotatedSodConfig):
    r, m1, m2, E = primitive_to_conserved_np(rho, u, v, p, cfg)
    F = (m1, m1 * u + p, m1 * v, u * (E + p))
    G = (m2, m2 * u, m2 * v + p, v * (E + p))
    return F, G


@torch.no_grad()
def predict_primitive_points(model, x_np, y_np, t_np, batch_size=65536):
    x_flat = np.asarray(x_np, dtype=np.float64).reshape(-1)
    y_flat = np.asarray(y_np, dtype=np.float64).reshape(-1)
    t_flat = np.asarray(t_np, dtype=np.float64).reshape(-1)
    assert x_flat.shape == y_flat.shape == t_flat.shape
    out = []
    for s in range(0, x_flat.size, batch_size):
        e = min(s + batch_size, x_flat.size)
        xyt = torch.tensor(np.stack([x_flat[s:e], y_flat[s:e], t_flat[s:e]], axis=1), device=DEVICE, dtype=DTYPE)
        out.append(to_numpy(model(xyt)))
    W = np.concatenate(out, axis=0)
    return W.reshape(*np.asarray(x_np).shape, 4)


def predict_final_grid(model, cfg: Euler2DRotatedSodConfig, n=None, t_value=None):
    if n is None:
        n = cfg.eval_nxy
    if t_value is None:
        t_value = cfg.t_max
    x = np.linspace(cfg.x_min, cfg.x_max, n)
    y = np.linspace(cfg.y_min, cfg.y_max, n)
    X, Y = np.meshgrid(x, y, indexing="xy")
    T = np.full_like(X, t_value)
    W = predict_primitive_points(model, X, Y, T)
    rho_e, u_e, v_e, p_e = rotated_exact_np(X, Y, T, cfg)
    W_exact = np.stack([rho_e, u_e, v_e, p_e], axis=-1)
    return x, y, X, Y, W, W_exact


def primitive_fields(W, cfg: Euler2DRotatedSodConfig):
    rho, u, v, p = W[..., 0], W[..., 1], W[..., 2], W[..., 3]
    nx, ny = normal_components(cfg)
    un = u * nx + v * ny
    speed = np.sqrt(u**2 + v**2)
    return {"rho": rho, "u": u, "v": v, "p": p, "un": un, "speed": speed}


def compute_space_time_error_metrics(model, cfg: Euler2DRotatedSodConfig):
    x = np.linspace(cfg.x_min, cfg.x_max, cfg.eval_space_nxy)
    y = np.linspace(cfg.y_min, cfg.y_max, cfg.eval_space_nxy)
    t_vals = np.linspace(cfg.t_min, cfg.t_max, cfg.eval_nt)

    err_acc = {name: [] for name in ["rho", "u", "v", "p", "un"]}
    exact_acc = {name: [] for name in ["rho", "u", "v", "p", "un"]}
    rho_s, u_s, p_s, _, _ = scales(cfg)
    scaled_err_sq = []
    scaled_exact_sq = []

    X, Y = np.meshgrid(x, y, indexing="xy")
    for tt in t_vals:
        T = np.full_like(X, tt)
        W = predict_primitive_points(model, X, Y, T)
        rho_e, u_e, v_e, p_e = rotated_exact_np(X, Y, T, cfg)
        We = np.stack([rho_e, u_e, v_e, p_e], axis=-1)
        fp = primitive_fields(W, cfg)
        fe = primitive_fields(We, cfg)
        for name in err_acc.keys():
            err_acc[name].append((fp[name] - fe[name]).reshape(-1))
            exact_acc[name].append(fe[name].reshape(-1))
        se = ((W[..., 0] - We[..., 0]) / rho_s) ** 2 + ((W[..., 1] - We[..., 1]) / u_s) ** 2 + ((W[..., 2] - We[..., 2]) / u_s) ** 2 + ((W[..., 3] - We[..., 3]) / p_s) ** 2
        sx = (We[..., 0] / rho_s) ** 2 + (We[..., 1] / u_s) ** 2 + (We[..., 2] / u_s) ** 2 + (We[..., 3] / p_s) ** 2
        scaled_err_sq.append(se.reshape(-1))
        scaled_exact_sq.append(sx.reshape(-1))

    out = {}
    for name in err_acc.keys():
        err = np.concatenate(err_acc[name])
        ex = np.concatenate(exact_acc[name])
        out[f"{name}_space_time_l1"] = float(np.mean(np.abs(err)))
        out[f"{name}_space_time_l2"] = float(np.sqrt(np.mean(err**2)))
        out[f"{name}_space_time_rel_l2"] = float(np.sqrt(np.mean(err**2)) / (np.sqrt(np.mean(ex**2)) + 1.0e-12))
    se_all = np.concatenate(scaled_err_sq)
    sx_all = np.concatenate(scaled_exact_sq)
    out["primitive_scaled_space_time_rel_l2"] = float(np.sqrt(np.mean(se_all)) / (np.sqrt(np.mean(sx_all)) + 1.0e-12))
    return out


def compute_final_error_metrics(model, cfg: Euler2DRotatedSodConfig):
    x, y, X, Y, W, We = predict_final_grid(model, cfg, cfg.eval_nxy, cfg.t_max)
    fp = primitive_fields(W, cfg)
    fe = primitive_fields(We, cfg)
    out = {}
    for name in ["rho", "u", "v", "p", "un"]:
        err = fp[name] - fe[name]
        exact = fe[name]
        out[f"{name}_final_l1"] = float(np.mean(np.abs(err)))
        out[f"{name}_final_l2"] = float(np.sqrt(np.mean(err**2)))
        out[f"{name}_final_rel_l2"] = float(np.sqrt(np.mean(err**2)) / (np.sqrt(np.mean(exact**2)) + 1.0e-12))
    rho_s, u_s, p_s, _, _ = scales(cfg)
    se = ((W[..., 0] - We[..., 0]) / rho_s) ** 2 + ((W[..., 1] - We[..., 1]) / u_s) ** 2 + ((W[..., 2] - We[..., 2]) / u_s) ** 2 + ((W[..., 3] - We[..., 3]) / p_s) ** 2
    sx = (We[..., 0] / rho_s) ** 2 + (We[..., 1] / u_s) ** 2 + (We[..., 2] / u_s) ** 2 + (We[..., 3] / p_s) ** 2
    out["primitive_scaled_final_rel_l2"] = float(np.sqrt(np.mean(se)) / (np.sqrt(np.mean(sx)) + 1.0e-12))
    out["rho_min"] = float(np.min(W[..., 0])); out["rho_max"] = float(np.max(W[..., 0]))
    out["p_min"] = float(np.min(W[..., 3])); out["p_max"] = float(np.max(W[..., 3]))
    out["rho_positivity_violation"] = float(max(0.0, cfg.rho_floor - np.min(W[..., 0])))
    out["p_positivity_violation"] = float(max(0.0, cfg.p_floor - np.min(W[..., 3])))
    return out


def eta_line_limits(cfg: Euler2DRotatedSodConfig, safety=0.98):
    nx, ny = normal_components(cfg)
    lows, highs = [], []
    if abs(nx) > 1.0e-12:
        vals = [(cfg.x_min - cfg.x0) / nx, (cfg.x_max - cfg.x0) / nx]
        lows.append(min(vals)); highs.append(max(vals))
    if abs(ny) > 1.0e-12:
        vals = [(cfg.y_min - cfg.y0) / ny, (cfg.y_max - cfg.y0) / ny]
        lows.append(min(vals)); highs.append(max(vals))
    eta_min = max(lows); eta_max = min(highs)
    mid = 0.5 * (eta_min + eta_max)
    half = 0.5 * (eta_max - eta_min) * safety
    return mid - half, mid + half


def compute_normal_line_metrics(model, cfg: Euler2DRotatedSodConfig):
    eta_min, eta_max = eta_line_limits(cfg)
    eta = np.linspace(eta_min, eta_max, cfg.line_n)
    nx, ny = normal_components(cfg)
    X = cfg.x0 + nx * eta
    Y = cfg.y0 + ny * eta
    T = np.full_like(eta, cfg.t_max)
    W = predict_primitive_points(model, X, Y, T)
    rho_e, u_e, v_e, p_e = rotated_exact_np(X, Y, T, cfg)
    We = np.stack([rho_e, u_e, v_e, p_e], axis=-1)
    fp = primitive_fields(W, cfg); fe = primitive_fields(We, cfg)
    out = {}
    for name in ["rho", "un", "p"]:
        err = fp[name] - fe[name]
        exact = fe[name]
        out[f"{name}_normal_line_final_l1"] = float(np.mean(np.abs(err)))
        out[f"{name}_normal_line_final_rel_l2"] = float(np.sqrt(np.mean(err**2)) / (np.sqrt(np.mean(exact**2)) + 1.0e-12))
    # TV excess along the normal line for rho/un/p
    for name in ["rho", "un", "p"]:
        tv_pred = np.sum(np.abs(np.diff(fp[name])))
        tv_exact = np.sum(np.abs(np.diff(fe[name])))
        out[f"{name}_normal_line_tv_excess"] = float(max(0.0, tv_pred - tv_exact))
    return out


def compute_global_conservation_metrics(model, cfg: Euler2DRotatedSodConfig):
    n = cfg.cons_nxy
    nt = cfg.cons_nt
    x = np.linspace(cfg.x_min, cfg.x_max, n)
    y = np.linspace(cfg.y_min, cfg.y_max, n)
    t_vals = np.linspace(cfg.t_min, cfg.t_max, nt)
    X, Y = np.meshgrid(x, y, indexing="xy")

    U_hist = []
    B_hist = []
    for tt in t_vals:
        T = np.full_like(X, tt)
        W = predict_primitive_points(model, X, Y, T)
        rho, u, v, p = W[..., 0], W[..., 1], W[..., 2], W[..., 3]
        U = primitive_to_conserved_np(rho, u, v, p, cfg)
        U_int = np.array([np.trapz(np.trapz(comp, y, axis=0), x, axis=0) for comp in U])
        U_hist.append(U_int)

        # Boundary flux integral with outward normals over the square boundary.
        yy = y
        xx = x
        # x = xmin/xmax
        Xl = np.full_like(yy, cfg.x_min); Xr = np.full_like(yy, cfg.x_max); Ty = np.full_like(yy, tt)
        Wl = predict_primitive_points(model, Xl, yy, Ty)
        Wr = predict_primitive_points(model, Xr, yy, Ty)
        Fl, Gl = flux_np_2d(Wl[..., 0], Wl[..., 1], Wl[..., 2], Wl[..., 3], cfg)
        Fr, Gr = flux_np_2d(Wr[..., 0], Wr[..., 1], Wr[..., 2], Wr[..., 3], cfg)
        bx = np.array([np.trapz(Fr[k] - Fl[k], yy) for k in range(4)])
        # y = ymin/ymax
        Yb = np.full_like(xx, cfg.y_min); Yt = np.full_like(xx, cfg.y_max); Tx = np.full_like(xx, tt)
        Wb = predict_primitive_points(model, xx, Yb, Tx)
        Wt = predict_primitive_points(model, xx, Yt, Tx)
        Fb, Gb = flux_np_2d(Wb[..., 0], Wb[..., 1], Wb[..., 2], Wb[..., 3], cfg)
        Ft, Gt = flux_np_2d(Wt[..., 0], Wt[..., 1], Wt[..., 2], Wt[..., 3], cfg)
        by = np.array([np.trapz(Gt[k] - Gb[k], xx) for k in range(4)])
        B_hist.append(bx + by)

    U_hist = np.asarray(U_hist)
    B_hist = np.asarray(B_hist)
    residuals = []
    for i in range(nt):
        if i == 0:
            bint = np.zeros(4)
        else:
            bint = np.array([np.trapz(B_hist[: i + 1, k], t_vals[: i + 1]) for k in range(4)])
        residuals.append(U_hist[i] - U_hist[0] + bint)
    residuals = np.asarray(residuals)
    names = ["mass", "momx", "momy", "energy"]
    out = {}
    for k, name in enumerate(names):
        out[f"global_{name}_cons_mean_abs"] = float(np.mean(np.abs(residuals[:, k])))
        out[f"global_{name}_cons_final_abs"] = float(abs(residuals[-1, k]))
    return out


def compute_local_conservation_metrics(model, cfg: Euler2DRotatedSodConfig):
    rng = np.random.default_rng(cfg.seed + 271828)
    vals = []
    for _ in range(cfg.n_control_volumes):
        for _try in range(200):
            x1, x2 = np.sort(rng.uniform(cfg.x_min, cfg.x_max, 2))
            y1, y2 = np.sort(rng.uniform(cfg.y_min, cfg.y_max, 2))
            t1, t2 = np.sort(rng.uniform(cfg.t_min, cfg.t_max, 2))
            if (x2 - x1) >= cfg.cv_min_width and (y2 - y1) >= cfg.cv_min_width and (t2 - t1) >= cfg.cv_min_duration:
                break
        xq = np.linspace(x1, x2, cfg.cv_quad_nxy)
        yq = np.linspace(y1, y2, cfg.cv_quad_nxy)
        tq = np.linspace(t1, t2, cfg.cv_quad_nt)
        Xq, Yq = np.meshgrid(xq, yq, indexing="xy")

        def area_U(tt):
            Tq = np.full_like(Xq, tt)
            W = predict_primitive_points(model, Xq, Yq, Tq)
            U = primitive_to_conserved_np(W[..., 0], W[..., 1], W[..., 2], W[..., 3], cfg)
            return np.array([np.trapz(np.trapz(comp, yq, axis=0), xq, axis=0) for comp in U])

        U1 = area_U(t1); U2 = area_U(t2)
        b_flux = []
        for tt in tq:
            # x sides
            yy = yq; xx = xq
            Wl = predict_primitive_points(model, np.full_like(yy, x1), yy, np.full_like(yy, tt))
            Wr = predict_primitive_points(model, np.full_like(yy, x2), yy, np.full_like(yy, tt))
            Fl, Gl = flux_np_2d(Wl[..., 0], Wl[..., 1], Wl[..., 2], Wl[..., 3], cfg)
            Fr, Gr = flux_np_2d(Wr[..., 0], Wr[..., 1], Wr[..., 2], Wr[..., 3], cfg)
            bx = np.array([np.trapz(Fr[k] - Fl[k], yy) for k in range(4)])
            # y sides
            Wb = predict_primitive_points(model, xx, np.full_like(xx, y1), np.full_like(xx, tt))
            Wt = predict_primitive_points(model, xx, np.full_like(xx, y2), np.full_like(xx, tt))
            Fb, Gb = flux_np_2d(Wb[..., 0], Wb[..., 1], Wb[..., 2], Wb[..., 3], cfg)
            Ft, Gt = flux_np_2d(Wt[..., 0], Wt[..., 1], Wt[..., 2], Wt[..., 3], cfg)
            by = np.array([np.trapz(Gt[k] - Gb[k], xx) for k in range(4)])
            b_flux.append(bx + by)
        b_flux = np.asarray(b_flux)
        bint = np.array([np.trapz(b_flux[:, k], tq) for k in range(4)])
        vals.append(np.abs(U2 - U1 + bint))
    vals = np.asarray(vals)
    names = ["mass", "momx", "momy", "energy"]
    out = {}
    for k, name in enumerate(names):
        out[f"local_{name}_cv_mean_abs"] = float(np.mean(vals[:, k]))
        out[f"local_{name}_cv_median_abs"] = float(np.median(vals[:, k]))
        out[f"local_{name}_cv_max_abs"] = float(np.max(vals[:, k]))
    return out


def compute_gate_diagnostics(model, cfg: Euler2DRotatedSodConfig, progress=1.0):
    n = cfg.gate_eval_nxy
    x = np.linspace(cfg.x_min, cfg.x_max, n)
    y = np.linspace(cfg.y_min, cfg.y_max, n)
    X, Y = np.meshgrid(x, y, indexing="xy")
    T = np.full_like(X, cfg.t_max)
    h_probe, cmin, _ = schedule_from_progress(progress, cfg)
    xt = torch.tensor(X.reshape(-1, 1), device=DEVICE, dtype=DTYPE)
    yt = torch.tensor(Y.reshape(-1, 1), device=DEVICE, dtype=DTYPE)
    tt = torch.tensor(T.reshape(-1, 1), device=DEVICE, dtype=DTYPE)
    gate, C, Jhat, Jbar, valid, Cmax, Jhatmax = trace_ratio_gate_primitive(model, xt, yt, tt, h_probe, cmin, cfg)
    g = to_numpy(gate.reshape(-1))
    valid_np = to_numpy(valid.reshape(-1))
    return {
        "gate_progress": float(progress), "gate_h": float(h_probe), "gate_cmin": float(cmin),
        "gate_mean": float(g.mean()), "gate_max": float(g.max()),
        "gate_active_frac_gt_0p5": float((g > 0.5).mean()),
        "gate_active_frac_gt_0p1": float((g > 0.1).mean()),
        "gate_valid_frac": float(valid_np.mean()),
        "gate_C_mean": float(to_numpy(C[valid > 0]).mean()) if np.any(valid_np > 0) else float(to_numpy(C).mean()),
        "gate_C_max": float(to_numpy(C).max()),
        "gate_Jhat_mean": float(to_numpy(Jhat[valid > 0]).mean()) if np.any(valid_np > 0) else float(to_numpy(Jhat).mean()),
        "gate_Jhat_max": float(to_numpy(Jhat).max()),
        "gate_Jbar": float(Jbar.detach().cpu()),
    }


def evaluate_model(model, model_name: str, cfg: Euler2DRotatedSodConfig):
    row = {"model": model_name, "seed": cfg.seed}
    row.update(compute_space_time_error_metrics(model, cfg))
    row.update(compute_final_error_metrics(model, cfg))
    row.update(compute_normal_line_metrics(model, cfg))
    row.update(compute_global_conservation_metrics(model, cfg))
    row.update(compute_local_conservation_metrics(model, cfg))
    row.update(compute_gate_diagnostics(model, cfg, progress=1.0))
    return row


# ============================================================
# 9. Experiment runner and summary
# ============================================================

def print_header(cfg: Euler2DRotatedSodConfig, run_dir: Path):
    rho_s, u_s, p_s, mom_s, e_s = scales(cfg)
    h_start, c_start, _ = schedule_from_progress(0.0, cfg)
    h_end, c_end, _ = schedule_from_progress(1.0, cfg)
    speeds = sod_wave_speeds(cfg)
    nx, ny = normal_components(cfg)
    print("=" * 100)
    print("2D Euler rotated Sod: paired PINN vs ring trace-ratio gated PINN")
    print("=" * 100)
    print(f"seed                 : {cfg.seed}")
    print(f"device               : {DEVICE}")
    print(f"dtype                : {cfg.dtype}")
    print(f"domain               : x in [{cfg.x_min},{cfg.x_max}], y in [{cfg.y_min},{cfg.y_max}], t in [{cfg.t_min},{cfg.t_max}]")
    print("PDE                  : conservative 2D Euler, U_t + F(U)_x + G(U)_y = 0")
    print("network output       : primitive variables W=(rho,u,v,p), rho/p positivity via softplus floors")
    print(f"states               : L=(rho,un,p)=({cfg.rho_L},{cfg.un_L},{cfg.p_L}), R=({cfg.rho_R},{cfg.un_R},{cfg.p_R})")
    print(f"rotation             : theta={cfg.theta_deg} deg, n=({nx:.6f},{ny:.6f})")
    print(f"Sod star state       : p*={speeds['p_star']:.6f}, un*={speeds['un_star']:.6f}")
    print("boundary condition   : exact time-dependent Dirichlet trace on all four boundaries")
    print("note                 : exact boundary values are used for training in this controlled benchmark")
    print(f"network              : width={cfg.width}, depth={cfg.depth}, activation={cfg.activation}")
    print(f"training budget      : warmup={cfg.warmup_iters}, continuation/gated={cfg.gated_iters}")
    print(f"batch sizes          : n_f={cfg.n_f}, n_ic={cfg.n_ic}, n_bc={cfg.n_bc}")
    print(f"loss weights         : w_ic={cfg.w_ic}, w_bc={cfg.w_bc}, w_pde={cfg.w_pde}")
    print(f"scales               : rho={rho_s:.4g}, u={u_s:.4g}, p={p_s:.4g}, mom={mom_s:.4g}, E={e_s:.4g}")
    print(f"ring directions      : projective pairs={cfg.ring_trace_pairs}; signed ring directions={2*cfg.ring_trace_pairs}")
    print(f"h0                   : {h0(cfg):.8f}")
    print(f"h schedule           : h(0)={h_start:.8f}, h(1)={h_end:.8f}")
    print(f"cmin schedule        : cmin(0)={c_start:.3f}, cmin(1)={c_end:.3f}")
    print(f"beta                 : {cfg.beta}")
    print(f"residual floor       : {cfg.residual_floor}")
    print("protocol             : no validation, no early stopping, final checkpoint evaluation")
    print("loss additions       : no RH loss, no entropy loss, no weak form, no numerical flux")
    print(f"run directory        : {run_dir}")
    print("=" * 100)


def run_one_seed_paired(seed: int, base_cfg: Euler2DRotatedSodConfig):
    cfg_s = copy.deepcopy(base_cfg)
    cfg_s.seed = int(seed)
    configure_runtime(cfg_s)
    set_seed(cfg_s.seed)
    run_dir = make_run_dir(cfg_s)
    print_header(cfg_s, run_dir)

    if cfg_s.save_outputs:
        save_json({
            "config": asdict(cfg_s),
            "platform": platform.platform(),
            "torch_version": torch.__version__,
            "device": str(DEVICE),
            "created_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "protocol": "shared warm-up; paired PINN/tPINN continuation; same RNG state restored; no validation; final checkpoint evaluation",
        }, run_dir / "config_and_runtime.json")

    warmup_model = MLP(cfg_s).to(DEVICE, dtype=DTYPE)
    print("parameter count:", count_params(warmup_model))
    warmup_model, hist_warmup = train_warmup_fixed(warmup_model, cfg_s)
    warmup_state = clone_state(warmup_model)
    continuation_rng = get_rng_state()

    pinn_model = MLP(cfg_s).to(DEVICE, dtype=DTYPE)
    pinn_model.load_state_dict(warmup_state)
    set_rng_state(continuation_rng)
    pinn_model, hist_pinn = train_vanilla_continuation_fixed(pinn_model, cfg_s)

    tpinn_model = MLP(cfg_s).to(DEVICE, dtype=DTYPE)
    tpinn_model.load_state_dict(warmup_state)
    set_rng_state(continuation_rng)
    tpinn_model, hist_tpinn = train_gated_fixed(tpinn_model, cfg_s)

    print("\nEvaluating final checkpoints ...")
    rows = [evaluate_model(pinn_model, "PINN", cfg_s), evaluate_model(tpinn_model, "tPINN", cfg_s)]
    metrics_df = pd.DataFrame(rows)

    if cfg_s.save_outputs:
        hist_warmup.to_csv(run_dir / "history_warmup_shared.csv", index=False)
        hist_pinn.to_csv(run_dir / "history_PINN_continuation.csv", index=False)
        hist_tpinn.to_csv(run_dir / "history_tPINN_trace_ratio.csv", index=False)
        metrics_df.to_csv(run_dir / "metrics_final.csv", index=False)
        torch.save(warmup_state, run_dir / "checkpoint_warmup_final.pt")
        torch.save(pinn_model.state_dict(), run_dir / "checkpoint_final_PINN.pt")
        torch.save(tpinn_model.state_dict(), run_dir / "checkpoint_final_tPINN.pt")

    display(metrics_df)
    return metrics_df


def summarize_multiseed(all_metrics: pd.DataFrame, save_dir: str):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    all_metrics.to_csv(save_dir / "all_seed_metrics_raw.csv", index=False)

    main_metrics = [
        "primitive_scaled_space_time_rel_l2", "primitive_scaled_final_rel_l2",
        "rho_space_time_rel_l2", "un_space_time_rel_l2", "p_space_time_rel_l2",
        "rho_final_rel_l2", "un_final_rel_l2", "p_final_rel_l2",
        "rho_normal_line_final_rel_l2", "un_normal_line_final_rel_l2", "p_normal_line_final_rel_l2",
        "rho_normal_line_tv_excess", "un_normal_line_tv_excess", "p_normal_line_tv_excess",
        "rho_positivity_violation", "p_positivity_violation",
        "global_mass_cons_mean_abs", "global_momx_cons_mean_abs", "global_momy_cons_mean_abs", "global_energy_cons_mean_abs",
        "local_mass_cv_mean_abs", "local_momx_cv_mean_abs", "local_momy_cv_mean_abs", "local_energy_cv_mean_abs",
        "gate_mean", "gate_active_frac_gt_0p5", "gate_active_frac_gt_0p1",
    ]
    main_metrics = [c for c in main_metrics if c in all_metrics.columns]
    summary_numeric = all_metrics.groupby("model")[main_metrics].agg(["mean", "std"])
    summary_numeric.to_csv(save_dir / "summary_mean_std_numeric.csv")

    formatted = []
    for model_name, group in all_metrics.groupby("model"):
        row = {"model": model_name}
        for m in main_metrics:
            row[m] = f"{group[m].mean():.4e} ± {group[m].std(ddof=1):.2e}"
        formatted.append(row)
    paper_table = pd.DataFrame(formatted)
    paper_table.to_csv(save_dir / "summary_mean_std_formatted.csv", index=False)

    paired_rows = []
    for m in main_metrics:
        wide = all_metrics.pivot(index="seed", columns="model", values=m)
        if "PINN" not in wide.columns or "tPINN" not in wide.columns:
            continue
        pinn = wide["PINN"]
        tpinn = wide["tPINN"]
        diff = pinn - tpinn
        imp = 100.0 * diff / (pinn.abs() + 1.0e-12)
        paired_rows.append({
            "metric": m,
            "PINN_mean": float(pinn.mean()),
            "tPINN_mean": float(tpinn.mean()),
            "absolute_diff_mean_PINN_minus_tPINN": float(diff.mean()),
            "absolute_diff_std": float(diff.std(ddof=1)),
            "improvement_pct_mean": float(imp.mean()),
            "improvement_pct_std": float(imp.std(ddof=1)),
            "wins_tPINN_better_out_of_n": int((tpinn < pinn).sum()),
            "n": int(len(wide)),
        })
    paired_table = pd.DataFrame(paired_rows)
    paired_table.to_csv(save_dir / "paired_improvement_table.csv", index=False)

    tpinn_only = all_metrics[all_metrics["model"] == "tPINN"]
    gate_cols = [c for c in ["gate_mean", "gate_active_frac_gt_0p5", "gate_active_frac_gt_0p1", "gate_h", "gate_cmin", "gate_valid_frac"] if c in all_metrics.columns]
    gate_summary = tpinn_only[gate_cols].agg(["mean", "std"]) if len(tpinn_only) and gate_cols else pd.DataFrame()
    if not gate_summary.empty:
        gate_summary.to_csv(save_dir / "tPINN_gate_summary_mean_std.csv")

    print("\nRaw per-seed metrics:")
    display(all_metrics)
    print("\nMain paper table: mean ± std")
    display(paper_table)
    print("\nPaired improvement table:")
    display(paired_table)
    if not gate_summary.empty:
        print("\ntPINN gate diagnostics: mean ± std")
        display(gate_summary)
    return {"all_metrics": all_metrics, "summary_numeric": summary_numeric, "paper_table": paper_table, "paired_table": paired_table, "gate_summary": gate_summary}


def run_multiseed_euler2d_rotated_sod(base_cfg: Euler2DRotatedSodConfig, seeds):
    all_rows = []
    for seed in seeds:
        df_seed = run_one_seed_paired(seed, base_cfg)
        all_rows.append(df_seed)
    all_metrics = pd.concat(all_rows, ignore_index=True)
    summary_dir = Path(base_cfg.output_dir) / base_cfg.experiment_name / "multiseed_summary"
    return summarize_multiseed(all_metrics, str(summary_dir))


# Public API aliases ---------------------------------------------------------
Euler2DConfig = Euler2DRotatedSodConfig


def build_model(cfg: Euler2DRotatedSodConfig) -> MLP:
    configure_runtime(cfg)
    return MLP(cfg)


def count_parameters(model: nn.Module) -> int:
    return count_params(model)


def direction_tensor(
    cfg: Euler2DRotatedSodConfig,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    resolved_device = DEVICE if device is None else torch.device(device)
    resolved_dtype = DTYPE if dtype is None else dtype
    m = int(cfg.ring_trace_pairs)
    theta = (
        torch.arange(m, device=resolved_device, dtype=resolved_dtype)
        * (math.pi / m)
    )
    return torch.stack([torch.cos(theta), torch.sin(theta)], dim=1)


def residual_weight(gate: torch.Tensor, cfg: Euler2DRotatedSodConfig):
    return cfg.residual_floor + (1.0 - cfg.residual_floor) * (1.0 - gate)


def train_warmup(model, cfg):
    return train_warmup_fixed(model, cfg)


def train_pinn_continuation(model, cfg):
    return train_vanilla_continuation_fixed(model, cfg)


def train_trg_continuation(model, cfg):
    return train_gated_fixed(model, cfg)


def _safe_output_root(output_root: str | Path, protected_root: str | Path) -> Path:
    root = Path(output_root).expanduser().resolve()
    protected = Path(protected_root).expanduser().resolve()
    try:
        root.relative_to(protected)
    except ValueError:
        pass
    else:
        raise ValueError(f"Refusing to write reproduction output inside {protected}")
    root.mkdir(parents=True, exist_ok=True)
    return root


def _save_public_checkpoint(model, path: Path, metrics: dict):
    torch.save(
        {
            "model_state_dict": clone_state(model),
            "extra": metrics,
        },
        path,
    )


def run_paired_experiment(
    cfg: Euler2DRotatedSodConfig,
    *,
    output_root: str | Path,
    protected_reported_root: str | Path,
    overwrite: bool = False,
) -> pd.DataFrame:
    """Run the canonical shared-warmup paired experiment outside reported artifacts."""
    configure_runtime(cfg)
    set_seed(cfg.seed)
    root = _safe_output_root(output_root, protected_reported_root)
    seed_root = root / "2d_euler" / f"seed_{cfg.seed}"
    if seed_root.exists() and any(seed_root.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Reproduction output already exists: {seed_root}. Use overwrite explicitly."
        )
    seed_root.mkdir(parents=True, exist_ok=True)

    save_json(
        {
            "equation": "2d_euler",
            "seed": int(cfg.seed),
            "config": asdict(cfg),
            "implementation_revision": "valid_mask_all_probe_bounds_v1",
            "protocol": (
                "shared warm-up; paired PINN and TRG-PINN continuation; "
                "same RNG state restored; no validation; final checkpoint evaluation"
            ),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(DEVICE),
        },
        seed_root / "config.json",
    )

    warmup_model = build_model(cfg).to(DEVICE, dtype=DTYPE)
    warmup_start = time.time()
    warmup_model, history_warmup = train_warmup_fixed(warmup_model, cfg)
    warmup_seconds = time.time() - warmup_start
    warmup_state = clone_state(warmup_model)
    continuation_rng_state = get_rng_state()

    pinn_model = build_model(cfg).to(DEVICE, dtype=DTYPE)
    pinn_model.load_state_dict(warmup_state, strict=True)
    set_rng_state(continuation_rng_state)
    pinn_start = time.time()
    pinn_model, history_pinn = train_vanilla_continuation_fixed(pinn_model, cfg)
    pinn_seconds = time.time() - pinn_start

    trg_model = build_model(cfg).to(DEVICE, dtype=DTYPE)
    trg_model.load_state_dict(warmup_state, strict=True)
    set_rng_state(continuation_rng_state)
    trg_start = time.time()
    trg_model, history_trg = train_gated_fixed(trg_model, cfg)
    trg_seconds = time.time() - trg_start

    pinn_metrics = evaluate_model(pinn_model, "PINN", cfg)
    trg_metrics = evaluate_model(trg_model, "TRG-PINN", cfg)

    for item, continuation_seconds in (
        (pinn_metrics, pinn_seconds),
        (trg_metrics, trg_seconds),
    ):
        item.update(
            {
                "equation": "2d_euler",
                "method": item["model"],
                "implementation_revision": "valid_mask_all_probe_bounds_v1",
                "warmup_iters": cfg.warmup_iters,
                "continuation_iters": cfg.gated_iters,
                "adam_total_iters": cfg.warmup_iters + cfg.gated_iters,
                "optimizer_main": "AdamW",
                "lbfgs_used": False,
                "wall_clock_sec_warmup": warmup_seconds,
                "wall_clock_sec_continuation": continuation_seconds,
                "wall_clock_sec_total": warmup_seconds + continuation_seconds,
                "num_parameters": count_parameters(pinn_model),
            }
        )

    history_warmup.to_csv(seed_root / "history_warmup.csv", index=False)
    history_pinn.to_csv(seed_root / "history_pinn.csv", index=False)
    history_trg.to_csv(seed_root / "history_trg_pinn.csv", index=False)
    frame = pd.DataFrame([pinn_metrics, trg_metrics])
    frame.to_csv(seed_root / "metrics_final.csv", index=False)
    save_json(pinn_metrics, seed_root / "metrics_pinn.json")
    save_json(trg_metrics, seed_root / "metrics_trg_pinn.json")
    torch.save(
        {
            "model_state_dict": warmup_state,
            "equation": "2d_euler",
            "method": "shared_warmup",
            "seed": int(cfg.seed),
            "implementation_revision": "valid_mask_all_probe_bounds_v1",
        },
        seed_root / "model_warmup.pt",
    )
    _save_public_checkpoint(pinn_model, seed_root / "model_pinn_final.pt", pinn_metrics)
    _save_public_checkpoint(trg_model, seed_root / "model_trg_pinn_final.pt", trg_metrics)
    (seed_root / "_SUCCESS").write_text("success\n", encoding="utf-8")
    return frame


__all__ = [
    "Euler2DRotatedSodConfig",
    "Euler2DConfig",
    "build_model",
    "count_parameters",
    "configure_runtime",
    "resolve_device",
    "resolve_dtype",
    "normal_vec",
    "normal_components",
    "rotated_exact_np",
    "rotated_exact_torch",
    "trace_ratio_gate_primitive",
    "direction_tensor",
    "residual_weight",
    "evaluate_model",
    "run_paired_experiment",
]
