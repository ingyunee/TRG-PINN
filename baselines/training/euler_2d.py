# ============================================================
# Paper-ready paired multi-seed Trace-Ratio Gated PINN
# 2D compressible Euler equations: rotated Sod shock-tube benchmark
#
# PDE, conservative form:
#   U_t + F(U)_x + G(U)_y = 0,
#   U = [rho, rho*u, rho*v, E]^T,
#   F = [rho*u, rho*u^2+p, rho*u*v, u(E+p)]^T,
#   G = [rho*v, rho*u*v, rho*v^2+p, v(E+p)]^T,
#   E = p/(gamma-1) + 0.5*rho*(u^2+v^2).
#
# Rotated Sod solution:
#   s = (x-x0) cos(theta) + (y-y0) sin(theta)
#   W(x,y,t) = [rho(s,t), un(s,t) n_x, un(s,t) n_y, p(s,t)]
# where (rho, un, p) is the exact 1D Sod Riemann solution.
#
# Network output: primitive variables W=(rho,u,v,p), with rho>0 and p>0 enforced
# by softplus floors. Residual is computed in conservative variables.
#
# Paper protocol:
#   - paired PINN vs tPINN comparison
#   - shared vanilla warm-up checkpoint
#   - same RNG state restored before PINN continuation and tPINN continuation
#   - no validation set
#   - no early stopping
#   - no best-checkpoint selection
#   - final checkpoint evaluation only
#   - exact rotated Sod solution is used for IC, exact boundary trace, evaluation/plotting
# ============================================================

import os
import json
import math
import time
import copy
import random
import platform
from pathlib import Path
from dataclasses import dataclass, asdict

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
    device: str = "cuda:4"          # "auto", "cpu", "cuda", "cuda:0", ...
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


cfg = Euler2DRotatedSodConfig()
DEVICE = None
DTYPE = None


# ============================================================
# 2. Runtime helpers
# ============================================================

def resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
    if device_str.startswith("cuda:4") and not torch.cuda.is_available():
        print("CUDA was requested but is unavailable. Falling back to CPU.")
        return torch.device("cpu")
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


# ============================================================
# 10. Run 5 seeds
# ============================================================

base_cfg = copy.deepcopy(cfg)
base_cfg.device = "cuda:4"  # For a specific GPU, e.g. base_cfg.device = "cuda:5"
base_cfg.save_outputs = True

# Original rotated-Sod setting from the uploaded code, with paper protocol.
base_cfg.warmup_iters = 3000
base_cfg.gated_iters = 7000
base_cfg.n_f = 20000
base_cfg.n_ic = 6000
base_cfg.n_bc = 2500
base_cfg.ring_trace_pairs = 6
base_cfg.w_ic = 100.0
base_cfg.w_bc = 1.0
base_cfg.w_pde = 1.0
base_cfg.h_max_factor = 5.0
base_cfg.h_min_factor = 2.0
base_cfg.cmin_start = 0.50
base_cfg.cmin_end = 0.70
base_cfg.beta = 0.05
base_cfg.residual_floor = 0.02

seeds = [2026, 7, 42, 100, 31415]

# Uncomment for a quick smoke test before full training.
# base_cfg.experiment_name = "euler2d_rotated_sod_ring_trace_ratio_smoke_test"
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
# seeds = [2026]

# summary_results_euler2d = run_multiseed_euler2d_rotated_sod(base_cfg, seeds)



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
    conserved = prim_to_cons_torch(W, cfg)
    flux_x_values, flux_y_values = flux_torch(W, cfg)
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
    rho_s, _, _, momentum_s, energy_s = scales(cfg)
    scaled = (
        residuals[0] / rho_s,
        residuals[1] / momentum_s,
        residuals[2] / momentum_s,
        residuals[3] / energy_s,
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


EQUATION_NAME = '2d_euler'

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
GPINN_HELPER_RECONSTRUCTION_SHA256 = '0ff7c8888ef564daa0abbdb5cd6db737bdcc981e30d9c8d8dcbcc7bd1fd751f6'
