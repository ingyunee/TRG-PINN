#!/usr/bin/env python
"""Regenerate the canonical 2D Burgers cache arrays and manuscript figures.

No training is performed.

The original manuscript cache was produced inside the training notebook after
``configure_runtime`` had selected PyTorch's ``high`` float32 matmul mode.
This standalone script therefore restores that numerical mode explicitly.
The device is supplied by the release parity notebook after a small
cache-fingerprint provenance check.
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime
import gc
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import torch

FIGURE_MATMUL_PRECISION = os.environ.get(
    "TRGPINN_FLOAT32_MATMUL_PRECISION",
    "high",
).strip().lower()

if FIGURE_MATMUL_PRECISION not in {
    "highest",
    "high",
    "medium",
}:
    raise ValueError(
        "Unsupported TRGPINN_FLOAT32_MATMUL_PRECISION: "
        f"{FIGURE_MATMUL_PRECISION}"
    )

torch.set_float32_matmul_precision(
    FIGURE_MATMUL_PRECISION
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from trgpinn.equations.burgers_2d import (
    Burgers2DConfig,
    build_model,
    exact_solution,
    line_eta_limits,
    normal_vector,
    predict_points,
    shock_speed_eta,
    trace_schedule,
)
from trgpinn.utils import (
    load_checkpoint_into_model,
    read_json,
    sha256_file,
)


EXPECTED_SEEDS = [2026, 7, 42, 100, 31415]
PRIMARY_METRIC = "space_time_rel_l2"
LINE_TIMES = np.asarray(
    [0.125, 0.250, 0.375, 0.500],
    dtype=np.float64,
)
LINE_N = 1600
HEATMAP_N = 300
SPACE_TIME_NX = 34
SPACE_TIME_NY = 34
SPACE_TIME_NT = 28
BATCH_SIZE = 65536


def load_reported_model(
    method: str,
    seed: int,
    *,
    device: str,
):
    run_dir = (
        REPO_ROOT
        / "artifacts"
        / "reported"
        / "2d_burgers"
        / method
        / f"seed_{int(seed)}"
    )
    payload = read_json(run_dir / "config.json")
    cfg = Burgers2DConfig.from_legacy_mapping(
        payload.get("config", payload),
        seed=int(seed),
        device=device,
    )
    torch_device = torch.device(device)
    dtype = (
        torch.float64
        if cfg.dtype == "float64"
        else torch.float32
    )
    model = build_model(cfg).to(
        device=torch_device,
        dtype=dtype,
    )
    checkpoint = run_dir / "model_final.pt"
    load_checkpoint_into_model(
        model,
        checkpoint,
        strict=True,
    )
    model.eval()
    return model, cfg, checkpoint


def select_representative_seed() -> tuple[int, pd.DataFrame]:
    master = pd.read_csv(
        REPO_ROOT
        / "results"
        / "reported_metrics"
        / "all_metrics_final.csv"
    )
    rows = master[
        master["equation"].astype(str).eq("2d_burgers")
        & master["method"].astype(str).eq("Ours")
    ].copy()
    rows["seed"] = pd.to_numeric(
        rows["seed"],
        errors="raise",
    ).astype(int)
    rows[PRIMARY_METRIC] = pd.to_numeric(
        rows[PRIMARY_METRIC],
        errors="raise",
    )
    seed_order = {
        seed: index
        for index, seed in enumerate(EXPECTED_SEEDS)
    }
    rows = rows[
        rows["seed"].isin(EXPECTED_SEEDS)
    ].copy()
    rows["_seed_order"] = rows["seed"].map(
        seed_order
    )
    rows = rows.sort_values(
        [PRIMARY_METRIC, "_seed_order"],
        kind="stable",
    ).reset_index(drop=True)

    if len(rows) != 5:
        raise RuntimeError(
            "Expected five post-valid-mask TRG-PINN seed rows."
        )

    seed = int(
        rows.iloc[len(rows) // 2]["seed"]
    )
    return seed, rows


def predict_flat(
    model,
    x_values,
    y_values,
    t_values,
):
    x_values = np.asarray(
        x_values,
        dtype=np.float64,
    ).reshape(-1)
    y_values = np.asarray(
        y_values,
        dtype=np.float64,
    ).reshape(-1)
    t_values = np.asarray(
        t_values,
        dtype=np.float64,
    ).reshape(-1)

    if not (
        x_values.shape
        == y_values.shape
        == t_values.shape
    ):
        raise ValueError(
            "x, y, and t arrays must have the same shape."
        )

    parameter = next(model.parameters())
    outputs = []

    for start in range(
        0,
        x_values.size,
        BATCH_SIZE,
    ):
        stop = min(
            start + BATCH_SIZE,
            x_values.size,
        )
        coordinates = torch.as_tensor(
            np.column_stack(
                [
                    x_values[start:stop],
                    y_values[start:stop],
                    t_values[start:stop],
                ]
            ),
            device=parameter.device,
            dtype=parameter.dtype,
        )
        outputs.append(
            model(coordinates)[:, 0]
            .detach()
            .cpu()
            .numpy()
        )

    return np.concatenate(outputs)


def exact_normal_lines(
    eta,
    times,
    cfg: Burgers2DConfig,
):
    eta_grid, time_grid = np.meshgrid(
        eta,
        times,
        indexing="xy",
    )
    shock_position = (
        float(cfg.eta0)
        + shock_speed_eta(cfg) * time_grid
    )
    return np.where(
        eta_grid < shock_position,
        float(cfg.uL),
        float(cfg.uR),
    ).astype(np.float64)


def predict_normal_lines(
    model,
    eta,
    times,
    cfg: Burgers2DConfig,
):
    normal = normal_vector(cfg)
    eta_grid, time_grid = np.meshgrid(
        eta,
        times,
        indexing="xy",
    )
    x_grid = normal[0] * eta_grid
    y_grid = normal[1] * eta_grid

    return predict_flat(
        model,
        x_grid.ravel(),
        y_grid.ravel(),
        time_grid.ravel(),
    ).reshape(
        len(times),
        len(eta),
    )


def compute_gate_grid_numpy(
    model,
    cfg: Burgers2DConfig,
    n: int = 300,
    progress: float = 1.0,
):
    x = np.linspace(
        cfg.x_min,
        cfg.x_max,
        n,
    )
    y = np.linspace(
        cfg.y_min,
        cfg.y_max,
        n,
    )
    X, Y = np.meshgrid(
        x,
        y,
        indexing="xy",
    )
    T = np.full_like(
        X,
        cfg.t_max,
    )

    h_probe, cmin, _ = trace_schedule(
        progress,
        cfg,
    )
    scale = max(
        abs(
            float(cfg.uL)
            - float(cfg.uR)
        ),
        1.0e-8,
    )

    directions = np.asarray(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [
                1.0 / np.sqrt(2.0),
                1.0 / np.sqrt(2.0),
            ],
            [
                1.0 / np.sqrt(2.0),
                -1.0 / np.sqrt(2.0),
            ],
        ],
        dtype=np.float64,
    )

    x0 = X.reshape(-1, 1)
    y0 = Y.reshape(-1, 1)
    t0 = T.reshape(-1, 1)

    n_points = x0.shape[0]
    n_directions = directions.shape[0]

    dx = directions[:, 0].reshape(
        1,
        n_directions,
    )
    dy = directions[:, 1].reshape(
        1,
        n_directions,
    )

    xm1 = x0 - h_probe * dx
    xp1 = x0 + h_probe * dx
    ym1 = y0 - h_probe * dy
    yp1 = y0 + h_probe * dy

    xm2 = x0 - 2.0 * h_probe * dx
    xp2 = x0 + 2.0 * h_probe * dx
    ym2 = y0 - 2.0 * h_probe * dy
    yp2 = y0 + 2.0 * h_probe * dy

    valid = (
        (xm2 >= cfg.x_min)
        & (xm2 <= cfg.x_max)
        & (xp2 >= cfg.x_min)
        & (xp2 <= cfg.x_max)
        & (ym2 >= cfg.y_min)
        & (ym2 <= cfg.y_max)
        & (yp2 >= cfg.y_min)
        & (yp2 <= cfg.y_max)
    ).astype(np.float64)

    xm1 = np.clip(
        xm1,
        cfg.x_min,
        cfg.x_max,
    )
    xp1 = np.clip(
        xp1,
        cfg.x_min,
        cfg.x_max,
    )
    ym1 = np.clip(
        ym1,
        cfg.y_min,
        cfg.y_max,
    )
    yp1 = np.clip(
        yp1,
        cfg.y_min,
        cfg.y_max,
    )

    xm2 = np.clip(
        xm2,
        cfg.x_min,
        cfg.x_max,
    )
    xp2 = np.clip(
        xp2,
        cfg.x_min,
        cfg.x_max,
    )
    ym2 = np.clip(
        ym2,
        cfg.y_min,
        cfg.y_max,
    )
    yp2 = np.clip(
        yp2,
        cfg.y_min,
        cfg.y_max,
    )

    repeated_t = np.repeat(
        t0,
        n_directions,
        axis=1,
    )

    def evaluate(xx, yy):
        return predict_flat(
            model,
            xx.reshape(-1),
            yy.reshape(-1),
            repeated_t.reshape(-1),
        ).reshape(
            n_points,
            n_directions,
        )

    u_m1 = evaluate(xm1, ym1)
    u_p1 = evaluate(xp1, yp1)
    u_m2 = evaluate(xm2, ym2)
    u_p2 = evaluate(xp2, yp2)

    jump_h = np.abs(
        u_m1 - u_p1
    ) / scale
    jump_2h = np.abs(
        u_m2 - u_p2
    ) / scale
    ratio = np.clip(
        jump_h
        / (
            jump_2h
            + cfg.trace_ratio_epsilon
        ),
        0.0,
        2.0,
    )

    jump_mean = float(
        np.sum(jump_h * valid)
        / (
            np.sum(valid)
            + cfg.batch_mean_epsilon
        )
    )
    jump_mean = max(
        jump_mean,
        cfg.batch_mean_epsilon,
    )
    normalized_jump = (
        jump_h / jump_mean
    )

    gate_jump = 1.0 / (
        1.0
        + np.exp(
            -np.clip(
                (
                    normalized_jump
                    - 1.0
                )
                / float(cfg.beta),
                -60.0,
                60.0,
            )
        )
    )
    gate_ratio = 1.0 / (
        1.0
        + np.exp(
            -np.clip(
                (
                    ratio
                    - cmin
                )
                / float(cfg.beta),
                -60.0,
                60.0,
            )
        )
    )
    gate_direction = (
        gate_jump
        * gate_ratio
        * valid
    )

    return {
        "x": x,
        "y": y,
        "G": np.max(
            gate_direction,
            axis=1,
        ).reshape(n, n),
        "C": np.max(
            ratio * valid,
            axis=1,
        ).reshape(n, n),
        "Jhat": np.max(
            normalized_jump * valid,
            axis=1,
        ).reshape(n, n),
        "h": float(h_probe),
        "cmin": float(cmin),
        "Jbar": float(jump_mean),
    }


def build_caches(
    output_dir: Path,
    *,
    device: str,
):
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    representative_seed, ours_rank = (
        select_representative_seed()
    )

    _, cfg_reference, _ = (
        load_reported_model(
            "trg_pinn",
            representative_seed,
            device=device,
        )
    )

    eta_min, eta_max = line_eta_limits(
        cfg_reference
    )
    eta = np.linspace(
        eta_min,
        eta_max,
        LINE_N,
        dtype=np.float64,
    )
    exact_line = exact_normal_lines(
        eta,
        LINE_TIMES,
        cfg_reference,
    )

    line_predictions = {
        "pinn": [],
        "trg_pinn": [],
    }
    line_checkpoint_paths = []
    line_checkpoint_hashes = []

    for method in ("pinn", "trg_pinn"):
        for seed in EXPECTED_SEEDS:
            model, cfg, checkpoint = (
                load_reported_model(
                    method,
                    seed,
                    device=device,
                )
            )
            prediction = predict_normal_lines(
                model,
                eta,
                LINE_TIMES,
                cfg,
            )
            if not np.isfinite(
                prediction
            ).all():
                raise RuntimeError(
                    f"Non-finite line prediction: "
                    f"{method}, seed={seed}"
                )

            line_predictions[
                method
            ].append(prediction)

            line_checkpoint_paths.append(
                str(checkpoint.resolve())
            )
            line_checkpoint_hashes.append(
                sha256_file(checkpoint)
            )

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    pinn_all = np.asarray(
        line_predictions["pinn"],
        dtype=np.float64,
    )
    trg_all = np.asarray(
        line_predictions["trg_pinn"],
        dtype=np.float64,
    )

    pinn_median = np.median(
        pinn_all,
        axis=0,
    )
    trg_median = np.median(
        trg_all,
        axis=0,
    )

    pinn_rmse = np.sqrt(
        np.mean(
            (
                pinn_all
                - exact_line[None, ...]
            )
            ** 2,
            axis=0,
        )
    )
    trg_rmse = np.sqrt(
        np.mean(
            (
                trg_all
                - exact_line[None, ...]
            )
            ** 2,
            axis=0,
        )
    )

    line_cache = (
        output_dir
        / "burgers2d_normal_line_benchmark.npz"
    )

    np.savez_compressed(
        line_cache,
        eta=eta,
        times=LINE_TIMES,
        seeds=np.asarray(
            EXPECTED_SEEDS,
            dtype=np.int64,
        ),
        exact=exact_line.astype(np.float32),
        pinn_all=pinn_all.astype(np.float32),
        tpinn_all=trg_all.astype(np.float32),
        pinn_median=pinn_median.astype(np.float32),
        tpinn_median=trg_median.astype(np.float32),
        pinn_rmse=pinn_rmse.astype(np.float32),
        tpinn_rmse=trg_rmse.astype(np.float32),
        pinn_lower=(
            pinn_median
            - pinn_rmse
        ).astype(np.float32),
        pinn_upper=(
            pinn_median
            + pinn_rmse
        ).astype(np.float32),
        tpinn_lower=(
            trg_median
            - trg_rmse
        ).astype(np.float32),
        tpinn_upper=(
            trg_median
            + trg_rmse
        ).astype(np.float32),
        checkpoint_paths=np.asarray(
            line_checkpoint_paths
        ),
        checkpoint_sha256=np.asarray(
            line_checkpoint_hashes
        ),
        source_csv=np.asarray(
            str(
                (
                    REPO_ROOT
                    / "results"
                    / "reported_metrics"
                    / "all_metrics_final.csv"
                ).resolve()
            )
        ),
        median_metric=np.asarray(
            PRIMARY_METRIC
        ),
        median_seed=np.int64(
            representative_seed
        ),
        rank_seed=ours_rank[
            "seed"
        ].to_numpy(dtype=np.int64),
        rank_metric=ours_rank[
            PRIMARY_METRIC
        ].to_numpy(dtype=np.float64),
        cfg_x_min=np.float64(
            cfg_reference.x_min
        ),
        cfg_x_max=np.float64(
            cfg_reference.x_max
        ),
        cfg_y_min=np.float64(
            cfg_reference.y_min
        ),
        cfg_y_max=np.float64(
            cfg_reference.y_max
        ),
        cfg_t_min=np.float64(
            cfg_reference.t_min
        ),
        cfg_t_max=np.float64(
            cfg_reference.t_max
        ),
        cfg_uL=np.float64(
            cfg_reference.uL
        ),
        cfg_uR=np.float64(
            cfg_reference.uR
        ),
        cfg_theta_deg=np.float64(
            cfg_reference.theta_deg
        ),
        cfg_eta0=np.float64(
            cfg_reference.eta0
        ),
        generated_at=np.asarray(
            datetime.now().isoformat(
                timespec="seconds"
            )
        ),
    )

    pinn_model, cfg_selected, pinn_checkpoint = (
        load_reported_model(
            "pinn",
            representative_seed,
            device=device,
        )
    )
    trg_model, _, trg_checkpoint = (
        load_reported_model(
            "trg_pinn",
            representative_seed,
            device=device,
        )
    )

    x_heatmap = np.linspace(
        cfg_selected.x_min,
        cfg_selected.x_max,
        HEATMAP_N,
    )
    y_heatmap = np.linspace(
        cfg_selected.y_min,
        cfg_selected.y_max,
        HEATMAP_N,
    )
    X_heatmap, Y_heatmap = np.meshgrid(
        x_heatmap,
        y_heatmap,
        indexing="xy",
    )
    T_heatmap = np.full_like(
        X_heatmap,
        cfg_selected.t_max,
    )

    exact_heatmap = exact_solution(
        X_heatmap,
        Y_heatmap,
        T_heatmap,
        cfg_selected,
    )
    pinn_heatmap = predict_flat(
        pinn_model,
        X_heatmap.ravel(),
        Y_heatmap.ravel(),
        T_heatmap.ravel(),
    ).reshape(
        HEATMAP_N,
        HEATMAP_N,
    )
    trg_heatmap = predict_flat(
        trg_model,
        X_heatmap.ravel(),
        Y_heatmap.ravel(),
        T_heatmap.ravel(),
    ).reshape(
        HEATMAP_N,
        HEATMAP_N,
    )

    pinn_error = np.abs(
        pinn_heatmap - exact_heatmap
    )
    trg_error = np.abs(
        trg_heatmap - exact_heatmap
    )

    gate_fields = compute_gate_grid_numpy(
        trg_model,
        cfg_selected,
        n=HEATMAP_N,
        progress=1.0,
    )

    error_vmax = float(
        np.percentile(
            np.r_[
                pinn_error.ravel(),
                trg_error.ravel(),
            ],
            99.5,
        )
    )

    heatmap_cache = (
        output_dir
        / "burgers2d_heatmap_median_seed_benchmark.npz"
    )

    np.savez_compressed(
        heatmap_cache,
        x=x_heatmap,
        y=y_heatmap,
        U_exact=exact_heatmap.astype(np.float32),
        U_pinn=pinn_heatmap.astype(np.float32),
        U_tpinn=trg_heatmap.astype(np.float32),
        E_pinn=pinn_error.astype(np.float32),
        E_tpinn=trg_error.astype(np.float32),
        gate=gate_fields["G"].astype(np.float32),
        gate_C=gate_fields["C"].astype(np.float32),
        gate_Jhat=gate_fields["Jhat"].astype(np.float32),
        gate_h=np.float64(
            gate_fields["h"]
        ),
        gate_cmin=np.float64(
            gate_fields["cmin"]
        ),
        gate_Jbar=np.float64(
            gate_fields["Jbar"]
        ),
        err_vmax=np.float64(
            error_vmax
        ),
        selected_seed=np.int64(
            representative_seed
        ),
        median_metric=np.asarray(
            PRIMARY_METRIC
        ),
        rank_seed=ours_rank[
            "seed"
        ].to_numpy(dtype=np.int64),
        rank_metric=ours_rank[
            PRIMARY_METRIC
        ].to_numpy(dtype=np.float64),
        pinn_checkpoint=np.asarray(
            str(pinn_checkpoint.resolve())
        ),
        tpinn_checkpoint=np.asarray(
            str(trg_checkpoint.resolve())
        ),
        pinn_checkpoint_sha256=np.asarray(
            sha256_file(pinn_checkpoint)
        ),
        tpinn_checkpoint_sha256=np.asarray(
            sha256_file(trg_checkpoint)
        ),
        source_csv=np.asarray(
            str(
                (
                    REPO_ROOT
                    / "results"
                    / "reported_metrics"
                    / "all_metrics_final.csv"
                ).resolve()
            )
        ),
        cfg_x_min=np.float64(
            cfg_selected.x_min
        ),
        cfg_x_max=np.float64(
            cfg_selected.x_max
        ),
        cfg_y_min=np.float64(
            cfg_selected.y_min
        ),
        cfg_y_max=np.float64(
            cfg_selected.y_max
        ),
        cfg_t_min=np.float64(
            cfg_selected.t_min
        ),
        cfg_t_max=np.float64(
            cfg_selected.t_max
        ),
        cfg_uL=np.float64(
            cfg_selected.uL
        ),
        cfg_uR=np.float64(
            cfg_selected.uR
        ),
        cfg_theta_deg=np.float64(
            cfg_selected.theta_deg
        ),
        cfg_eta0=np.float64(
            cfg_selected.eta0
        ),
        generated_at=np.asarray(
            datetime.now().isoformat(
                timespec="seconds"
            )
        ),
    )

    x_3d = np.linspace(
        cfg_selected.x_min,
        cfg_selected.x_max,
        SPACE_TIME_NX,
    )
    y_3d = np.linspace(
        cfg_selected.y_min,
        cfg_selected.y_max,
        SPACE_TIME_NY,
    )
    t_3d = np.linspace(
        cfg_selected.t_min,
        cfg_selected.t_max,
        SPACE_TIME_NT,
    )

    X_3d, Y_3d, T_3d = np.meshgrid(
        x_3d,
        y_3d,
        t_3d,
        indexing="ij",
    )

    exact_3d = exact_solution(
        X_3d,
        Y_3d,
        T_3d,
        cfg_selected,
    ).reshape(-1)

    pinn_3d = predict_flat(
        pinn_model,
        X_3d.ravel(),
        Y_3d.ravel(),
        T_3d.ravel(),
    )
    trg_3d = predict_flat(
        trg_model,
        X_3d.ravel(),
        Y_3d.ravel(),
        T_3d.ravel(),
    )

    pinn_error_3d = np.abs(
        exact_3d - pinn_3d
    )
    trg_error_3d = np.abs(
        exact_3d - trg_3d
    )

    error_vmax_3d = max(
        float(
            np.percentile(
                np.r_[
                    pinn_error_3d,
                    trg_error_3d,
                ],
                99.5,
            )
        ),
        1.0e-8,
    )

    cube_cache = (
        output_dir
        / "burgers2d_3d_median_seed_benchmark.npz"
    )

    np.savez_compressed(
        cube_cache,
        x=x_3d,
        y=y_3d,
        t=t_3d,
        U_exact=exact_3d.astype(np.float32),
        U_pinn=pinn_3d.astype(np.float32),
        U_tpinn=trg_3d.astype(np.float32),
        E_pinn=pinn_error_3d.astype(np.float32),
        E_tpinn=trg_error_3d.astype(np.float32),
        err_vmax=np.float64(
            error_vmax_3d
        ),
        selected_seed=np.int64(
            representative_seed
        ),
        median_metric=np.asarray(
            PRIMARY_METRIC
        ),
        pinn_checkpoint=np.asarray(
            str(pinn_checkpoint.resolve())
        ),
        tpinn_checkpoint=np.asarray(
            str(trg_checkpoint.resolve())
        ),
        pinn_checkpoint_sha256=np.asarray(
            sha256_file(pinn_checkpoint)
        ),
        tpinn_checkpoint_sha256=np.asarray(
            sha256_file(trg_checkpoint)
        ),
        source_csv=np.asarray(
            str(
                (
                    REPO_ROOT
                    / "results"
                    / "reported_metrics"
                    / "all_metrics_final.csv"
                ).resolve()
            )
        ),
        cfg_x_min=np.float64(
            cfg_selected.x_min
        ),
        cfg_x_max=np.float64(
            cfg_selected.x_max
        ),
        cfg_y_min=np.float64(
            cfg_selected.y_min
        ),
        cfg_y_max=np.float64(
            cfg_selected.y_max
        ),
        cfg_t_min=np.float64(
            cfg_selected.t_min
        ),
        cfg_t_max=np.float64(
            cfg_selected.t_max
        ),
        cfg_uL=np.float64(
            cfg_selected.uL
        ),
        cfg_uR=np.float64(
            cfg_selected.uR
        ),
        cfg_theta_deg=np.float64(
            cfg_selected.theta_deg
        ),
        cfg_eta0=np.float64(
            cfg_selected.eta0
        ),
        generated_at=np.asarray(
            datetime.now().isoformat(
                timespec="seconds"
            )
        ),
    )

    del pinn_model, trg_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return (
        line_cache,
        heatmap_cache,
        cube_cache,
        representative_seed,
    )


def render_all_figures() -> list[Path]:
    ### B2D-FIG-HEATMAP

    from pathlib import Path
    from types import SimpleNamespace

    import numpy as np
    import matplotlib.pyplot as plt


    PROJECT_ROOT_OVERRIDE = REPO_ROOT
    SAVE_FIGURES = True
    SHOW_SHOCK_LINE = False
    SOLUTION_CMAP = "jet"
    ERROR_CMAP = "magma"
    GATE_CMAP = "jet"


    def resolve_project_root():
        candidates = []

        if PROJECT_ROOT_OVERRIDE is not None:
            candidates.append(Path(PROJECT_ROOT_OVERRIDE).expanduser())

        existing = globals().get("PROJECT_ROOT")
        if existing is not None:
            candidates.append(Path(existing).expanduser())

        cwd = Path.cwd().resolve()
        candidates.extend([cwd, *cwd.parents])
        candidates.append(
            REPO_ROOT
        )

        seen = set()

        for candidate in candidates:
            candidate = candidate.resolve()

            if candidate in seen:
                continue

            seen.add(candidate)

            cache_path = (
                candidate
                / "build"
                / "reproduced_cache"
                / "2d_burgers"
                / "burgers2d_heatmap_median_seed_benchmark.npz"
            )

            if cache_path.is_file():
                return candidate

        raise FileNotFoundError(
            "2D Burgers heatmap cache was not found. "
            "Run B2D-FIG-DATA once."
        )


    def shock_speed_eta(cfg):
        theta = np.deg2rad(float(cfg.theta_deg))
        normal = np.asarray(
            [np.cos(theta), np.sin(theta)],
            dtype=np.float64,
        )

        return (
            0.5
            * (float(cfg.uL) + float(cfg.uR))
            * float(normal.sum())
        )


    def draw_exact_shock_line(
        axis,
        cfg,
        time_value,
        color="white",
        linewidth=1.1,
        alpha=0.85,
    ):
        theta = np.deg2rad(float(cfg.theta_deg))
        normal = np.asarray(
            [np.cos(theta), np.sin(theta)],
            dtype=np.float64,
        )

        eta_shock = (
            float(cfg.eta0)
            + shock_speed_eta(cfg) * float(time_value)
        )

        y_values = np.linspace(
            cfg.y_min,
            cfg.y_max,
            600,
        )

        x_values = (
            eta_shock
            - normal[1] * y_values
        ) / normal[0]

        mask = (
            (x_values >= cfg.x_min)
            & (x_values <= cfg.x_max)
        )

        axis.plot(
            x_values[mask],
            y_values[mask],
            ls="--",
            color=color,
            lw=linewidth,
            alpha=alpha,
        )


    def save_figure(figure, stem, figure_dir):
        if not SAVE_FIGURES:
            return

        figure.savefig(
            figure_dir / f"{stem}.pdf",
            bbox_inches="tight",
        )

        figure.savefig(
            figure_dir / f"{stem}.png",
            dpi=300,
            bbox_inches="tight",
        )


    PROJECT_ROOT = resolve_project_root()

    CACHE_PATH = (
        PROJECT_ROOT
        / "build"
        / "reproduced_cache"
        / "2d_burgers"
        / "burgers2d_heatmap_median_seed_benchmark.npz"
    )

    FIGURE_DIR = (
        PROJECT_ROOT
        / "results"
        / "reproduced_figures"
        / "2d_burgers"
    )

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with np.load(
        CACHE_PATH,
        allow_pickle=False,
    ) as data:
        x = np.asarray(data["x"], dtype=np.float64)
        y = np.asarray(data["y"], dtype=np.float64)

        U_exact = np.asarray(
            data["U_exact"],
            dtype=np.float64,
        )

        U_pinn = np.asarray(
            data["U_pinn"],
            dtype=np.float64,
        )

        U_tpinn = np.asarray(
            data["U_tpinn"],
            dtype=np.float64,
        )

        E_pinn = np.asarray(
            data["E_pinn"],
            dtype=np.float64,
        )

        E_tpinn = np.asarray(
            data["E_tpinn"],
            dtype=np.float64,
        )

        gate = np.asarray(
            data["gate"],
            dtype=np.float64,
        )

        err_vmax = float(data["err_vmax"])
        selected_seed = int(data["selected_seed"])
        median_metric = str(data["median_metric"].item())

        cfg_plot = SimpleNamespace(
            x_min=float(data["cfg_x_min"]),
            x_max=float(data["cfg_x_max"]),
            y_min=float(data["cfg_y_min"]),
            y_max=float(data["cfg_y_max"]),
            t_min=float(data["cfg_t_min"]),
            t_max=float(data["cfg_t_max"]),
            uL=float(data["cfg_uL"]),
            uR=float(data["cfg_uR"]),
            theta_deg=float(data["cfg_theta_deg"]),
            eta0=float(data["cfg_eta0"]),
        )

    expected_shape = (
        len(y),
        len(x),
    )

    for name, values in {
        "U_exact": U_exact,
        "U_pinn": U_pinn,
        "U_tpinn": U_tpinn,
        "E_pinn": E_pinn,
        "E_tpinn": E_tpinn,
        "gate": gate,
    }.items():
        if values.shape != expected_shape:
            raise ValueError(
                f"Unexpected shape for {name}: "
                f"{values.shape}"
            )

        if not np.isfinite(values).all():
            raise ValueError(
                f"Non-finite values in {name}."
            )

    plt.rcParams.update({
        "figure.dpi": 140,
        "savefig.dpi": 300,
        "font.size": 9.5,
        "axes.titlesize": 10.5,
        "axes.labelsize": 9.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.5,
        "axes.linewidth": 0.75,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 3.2,
        "ytick.major.size": 3.2,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
    })

    extent = [
        cfg_plot.x_min,
        cfg_plot.x_max,
        cfg_plot.y_min,
        cfg_plot.y_max,
    ]

    time_label = f"{cfg_plot.t_max:.1f}"

    fig_solution, axes_solution = plt.subplots(
        1,
        3,
        figsize=(9.2, 2.8),
        constrained_layout=True,
        sharex=True,
        sharey=True,
    )

    solution_panels = [
        (U_exact, f"(a) Exact, t = {time_label}"),
        (U_pinn, f"(b) PINN, t = {time_label}"),
        (U_tpinn, f"(c) TRG-PINN, t = {time_label}"),
    ]

    solution_image = None

    for axis, (values, title) in zip(
        axes_solution,
        solution_panels,
    ):
        solution_image = axis.imshow(
            values,
            extent=extent,
            origin="lower",
            aspect="equal",
            cmap=SOLUTION_CMAP,
            vmin=0.0,
            vmax=1.0,
            interpolation="nearest",
        )

        if SHOW_SHOCK_LINE:
            draw_exact_shock_line(
                axis,
                cfg_plot,
                cfg_plot.t_max,
                color="white",
            )

        axis.set_title(title, pad=5)
        axis.set_xlabel(r"$x$")
        axis.tick_params(
            direction="out",
            length=3.2,
            width=0.8,
        )

    axes_solution[0].set_ylabel(r"$y$")
    axes_solution[1].set_ylabel("")
    axes_solution[2].set_ylabel("")

    solution_colorbar = fig_solution.colorbar(
        solution_image,
        ax=axes_solution,
        location="right",
        shrink=0.92,
        pad=0.02,
    )

    solution_colorbar.set_label(
        r"$u(x,y,t_{\max})$"
    )

    save_figure(
        fig_solution,
        "fig_burgers2d_solution_median_seed",
        FIGURE_DIR,
    )

    plt.show()

    err_vmax = max(
        float(err_vmax),
        1.0e-8,
    )

    fig_error, axes_error = plt.subplots(
        1,
        2,
        figsize=(6.35, 2.8),
        constrained_layout=True,
        sharex=True,
        sharey=True,
    )

    error_panels = [
        (
            E_pinn,
            rf"(a) $|u_{{\mathrm{{exact}}}}"
            rf"-u_{{\mathrm{{PINN}}}}|$, t = {time_label}",
        ),
        (
            E_tpinn,
            rf"(b) $|u_{{\mathrm{{exact}}}}"
            rf"-u_{{\mathrm{{TRG-PINN}}}}|$, t = {time_label}",
        ),
    ]

    error_image = None

    for axis, (values, title) in zip(
        axes_error,
        error_panels,
    ):
        error_image = axis.imshow(
            values,
            extent=extent,
            origin="lower",
            aspect="equal",
            cmap=ERROR_CMAP,
            vmin=0.0,
            vmax=err_vmax,
            interpolation="nearest",
        )

        if SHOW_SHOCK_LINE:
            draw_exact_shock_line(
                axis,
                cfg_plot,
                cfg_plot.t_max,
                color="cyan",
                alpha=0.75,
            )

        axis.set_title(title, pad=5)
        axis.set_xlabel(r"$x$")
        axis.tick_params(
            direction="out",
            length=3.2,
            width=0.8,
        )

    axes_error[0].set_ylabel(r"$y$")
    axes_error[1].set_ylabel("")

    error_colorbar = fig_error.colorbar(
        error_image,
        ax=axes_error,
        location="right",
        shrink=0.92,
        pad=0.02,
    )

    error_colorbar.set_label(
        r"$|u-u_{\mathrm{exact}}|$"
    )

    save_figure(
        fig_error,
        "fig_burgers2d_error_median_seed",
        FIGURE_DIR,
    )

    plt.show()

    fig_gate, axis_gate = plt.subplots(
        1,
        1,
        figsize=(3.4, 2.8),
        constrained_layout=True,
    )

    gate_image = axis_gate.imshow(
        gate,
        extent=extent,
        origin="lower",
        aspect="equal",
        cmap=GATE_CMAP,
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
    )

    if SHOW_SHOCK_LINE:
        draw_exact_shock_line(
            axis_gate,
            cfg_plot,
            cfg_plot.t_max,
            color="black",
            alpha=0.65,
        )

    axis_gate.set_title(
        f"(a) Trace-ratio gate, t = {time_label}",
        pad=5,
    )

    axis_gate.set_xlabel(r"$x$")
    axis_gate.set_ylabel(r"$y$")
    axis_gate.tick_params(
        direction="out",
        length=3.2,
        width=0.8,
    )

    gate_colorbar = fig_gate.colorbar(
        gate_image,
        ax=axis_gate,
        location="right",
        shrink=0.92,
        pad=0.03,
    )

    gate_colorbar.set_label(r"$g$")

    save_figure(
        fig_gate,
        "fig_burgers2d_gate_median_seed",
        FIGURE_DIR,
    )

    plt.show()

    print("[OK] 2D Burgers heatmaps saved")
    print("Median metric:", median_metric)
    print("Median seed  :", selected_seed)
    print("Cache        :", CACHE_PATH)
    print("Figure dir   :", FIGURE_DIR)


    ### B2D-FIG-LINE

    from pathlib import Path

    import numpy as np
    import matplotlib.pyplot as plt

    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.ticker import FormatStrFormatter


    PROJECT_ROOT_OVERRIDE = REPO_ROOT


    def resolve_project_root():
        candidates = []

        if PROJECT_ROOT_OVERRIDE is not None:
            candidates.append(Path(PROJECT_ROOT_OVERRIDE).expanduser())

        existing = globals().get("PROJECT_ROOT")
        if existing is not None:
            candidates.append(Path(existing).expanduser())

        cwd = Path.cwd().resolve()
        candidates.extend([cwd, *cwd.parents])
        candidates.append(
            REPO_ROOT
        )

        seen = set()

        for candidate in candidates:
            candidate = candidate.resolve()

            if candidate in seen:
                continue

            seen.add(candidate)

            cache_path = (
                candidate
                / "build"
                / "reproduced_cache"
                / "2d_burgers"
                / "burgers2d_normal_line_benchmark.npz"
            )

            if cache_path.is_file():
                return candidate

        raise FileNotFoundError(
            "2D Burgers line cache was not found. "
            "Run B2D-FIG-DATA once."
        )


    def make_eta_ticks(
        eta_min,
        eta_max,
        step=0.5,
    ):
        start = np.ceil(
            eta_min / step
        ) * step

        stop = np.floor(
            eta_max / step
        ) * step

        ticks = np.arange(
            start,
            stop + 0.5 * step,
            step,
        )

        if ticks.size == 0:
            ticks = np.linspace(
                eta_min,
                eta_max,
                5,
            )

        return ticks


    PROJECT_ROOT = resolve_project_root()

    CACHE_PATH = (
        PROJECT_ROOT
        / "build"
        / "reproduced_cache"
        / "2d_burgers"
        / "burgers2d_normal_line_benchmark.npz"
    )

    FIGURE_DIR = (
        PROJECT_ROOT
        / "results"
        / "reproduced_figures"
        / "2d_burgers"
    )

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with np.load(
        CACHE_PATH,
        allow_pickle=False,
    ) as data:
        eta = np.asarray(
            data["eta"],
            dtype=np.float64,
        )

        times = np.asarray(
            data["times"],
            dtype=np.float64,
        )

        seeds = np.asarray(
            data["seeds"],
            dtype=int,
        )

        exact = np.asarray(
            data["exact"],
            dtype=np.float64,
        )

        pinn_median = np.asarray(
            data["pinn_median"],
            dtype=np.float64,
        )

        tpinn_median = np.asarray(
            data["tpinn_median"],
            dtype=np.float64,
        )

        pinn_lower = np.asarray(
            data["pinn_lower"],
            dtype=np.float64,
        )

        pinn_upper = np.asarray(
            data["pinn_upper"],
            dtype=np.float64,
        )

        tpinn_lower = np.asarray(
            data["tpinn_lower"],
            dtype=np.float64,
        )

        tpinn_upper = np.asarray(
            data["tpinn_upper"],
            dtype=np.float64,
        )

        selected_seed = int(
            data["median_seed"]
        )

        median_metric = str(
            data["median_metric"].item()
        )

    expected_shape = (
        len(times),
        len(eta),
    )

    for name, values in {
        "exact": exact,
        "pinn_median": pinn_median,
        "tpinn_median": tpinn_median,
        "pinn_lower": pinn_lower,
        "pinn_upper": pinn_upper,
        "tpinn_lower": tpinn_lower,
        "tpinn_upper": tpinn_upper,
    }.items():
        if values.shape != expected_shape:
            raise ValueError(
                f"Unexpected shape for {name}: "
                f"{values.shape}"
            )

        if not np.isfinite(values).all():
            raise ValueError(
                f"Non-finite values in {name}."
            )

    plt.rcParams.update({
        "figure.dpi": 160,
        "savefig.dpi": 300,
        "font.family": "serif",
        "mathtext.fontset": "stix",
        "font.size": 9.5,
        "axes.titlesize": 9.5,
        "axes.labelsize": 7.8,
        "legend.fontsize": 7.0,
        "xtick.labelsize": 6.8,
        "ytick.labelsize": 6.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.75,
        "ytick.major.width": 0.75,
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
    })

    EXACT_COLOR = "black"
    PINN_COLOR = "#2C7BB6"
    TPINN_COLOR = "#D7191C"

    PINN_BAND_ALPHA = 0.13
    TPINN_BAND_ALPHA = 0.16

    EXACT_LINE_WIDTH = 1.50
    PINN_LINE_WIDTH = 1.50
    TPINN_LINE_WIDTH = 2.0

    BAND_EDGE_WIDTH = 0.50
    PINN_EDGE_ALPHA = 0.68
    TPINN_EDGE_ALPHA = 0.72

    GRID_ALPHA = 0.13
    GRID_LINE_WIDTH = 0.45

    FIGSIZE_B2D = (7.9, 2.1)
    ETA_TICK_STEP = 0.5
    U_TICKS = np.linspace(0.0, 1.0, 5)

    figure, axes = plt.subplots(
        1,
        len(times),
        figsize=FIGSIZE_B2D,
        sharex=True,
        sharey=True,
        constrained_layout=False,
    )

    axes = np.atleast_1d(axes)
    eta_ticks = make_eta_ticks(
        eta[0],
        eta[-1],
        step=ETA_TICK_STEP,
    )

    for panel_index, (
        axis,
        time_value,
    ) in enumerate(
        zip(
            axes,
            times,
        )
    ):
        axis.fill_between(
            eta,
            pinn_lower[panel_index],
            pinn_upper[panel_index],
            color=PINN_COLOR,
            alpha=PINN_BAND_ALPHA,
            linewidth=0,
            zorder=1,
        )

        axis.fill_between(
            eta,
            tpinn_lower[panel_index],
            tpinn_upper[panel_index],
            color=TPINN_COLOR,
            alpha=TPINN_BAND_ALPHA,
            linewidth=0,
            zorder=2,
        )

        axis.plot(
            eta,
            pinn_lower[panel_index],
            color=PINN_COLOR,
            lw=BAND_EDGE_WIDTH,
            ls=":",
            alpha=PINN_EDGE_ALPHA,
            zorder=3,
        )

        axis.plot(
            eta,
            pinn_upper[panel_index],
            color=PINN_COLOR,
            lw=BAND_EDGE_WIDTH,
            ls=":",
            alpha=PINN_EDGE_ALPHA,
            zorder=3,
        )

        axis.plot(
            eta,
            tpinn_lower[panel_index],
            color=TPINN_COLOR,
            lw=BAND_EDGE_WIDTH,
            ls=":",
            alpha=TPINN_EDGE_ALPHA,
            zorder=4,
        )

        axis.plot(
            eta,
            tpinn_upper[panel_index],
            color=TPINN_COLOR,
            lw=BAND_EDGE_WIDTH,
            ls=":",
            alpha=TPINN_EDGE_ALPHA,
            zorder=4,
        )

        axis.plot(
            eta,
            exact[panel_index],
            color=EXACT_COLOR,
            lw=EXACT_LINE_WIDTH,
            ls="-",
            zorder=7,
        )

        axis.plot(
            eta,
            pinn_median[panel_index],
            color=PINN_COLOR,
            lw=PINN_LINE_WIDTH,
            ls="--",
            zorder=8,
        )

        axis.plot(
            eta,
            tpinn_median[panel_index],
            color=TPINN_COLOR,
            lw=TPINN_LINE_WIDTH,
            ls="--",
            zorder=9,
        )

        axis.set_title(
            rf"$t={time_value:.3f}$",
            pad=6,
        )

        axis.set_xlim(
            eta[0],
            eta[-1],
        )

        axis.set_ylim(
            -0.08,
            1.08,
        )

        axis.set_xticks(
            eta_ticks
        )

        axis.xaxis.set_major_formatter(
            FormatStrFormatter("%.1f")
        )

        axis.set_yticks(
            U_TICKS
        )

        axis.yaxis.set_major_formatter(
            FormatStrFormatter("%.1f")
        )

        axis.grid(
            alpha=GRID_ALPHA,
            linewidth=GRID_LINE_WIDTH,
        )

        axis.tick_params(
            axis="both",
            which="major",
            pad=2,
        )

        if panel_index == 0:
            axis.set_xlabel(
                r"$\eta$",
                labelpad=2,
            )

            axis.set_ylabel(
                r"$u(\eta,t)$",
                labelpad=3,
            )

            axis.tick_params(
                axis="x",
                which="both",
                bottom=True,
                labelbottom=True,
            )

            axis.tick_params(
                axis="y",
                which="both",
                left=True,
                labelleft=True,
            )

        else:
            axis.set_xlabel("")
            axis.set_ylabel("")

            axis.tick_params(
                axis="x",
                which="both",
                bottom=True,
                labelbottom=False,
            )

            axis.tick_params(
                axis="y",
                which="both",
                left=False,
                labelleft=False,
            )

    legend_handles = [
        Patch(
            facecolor=PINN_COLOR,
            alpha=PINN_BAND_ALPHA,
            edgecolor=PINN_COLOR,
            linewidth=0.7,
            label="PINN ± RMSE",
        ),
        Patch(
            facecolor=TPINN_COLOR,
            alpha=TPINN_BAND_ALPHA,
            edgecolor=TPINN_COLOR,
            linewidth=0.7,
            label="TRG-PINN ± RMSE",
        ),
        Line2D(
            [0],
            [0],
            color=EXACT_COLOR,
            lw=EXACT_LINE_WIDTH,
            ls="-",
            label="Exact",
        ),
        Line2D(
            [0],
            [0],
            color=PINN_COLOR,
            lw=PINN_LINE_WIDTH,
            ls="--",
            label="PINN",
        ),
        Line2D(
            [0],
            [0],
            color=TPINN_COLOR,
            lw=TPINN_LINE_WIDTH,
            ls="--",
            label="TRG-PINN",
        ),
    ]

    figure.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.5, 1.025),
        handlelength=2.15,
        columnspacing=1.35,
        handletextpad=0.50,
    )

    plt.subplots_adjust(
        left=0.065,
        right=0.995,
        bottom=0.220,
        top=0.780,
        wspace=0.0,
    )

    pdf_path = (
        FIGURE_DIR
        / "burgers2d_normal_line_multiseed.pdf"
    )

    png_path = (
        FIGURE_DIR
        / "burgers2d_normal_line_multiseed.png"
    )

    figure.savefig(
        pdf_path,
        bbox_inches="tight",
    )

    figure.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()

    print("[OK] 2D Burgers line figure saved")
    print("Seeds        :", seeds.tolist())
    print("Median metric:", median_metric)
    print("Median seed  :", selected_seed)
    print("Cache        :", CACHE_PATH)
    print("Saved        :", pdf_path)
    print("Saved        :", png_path)


    ### B2D-FIG-3D

    from pathlib import Path
    from types import SimpleNamespace

    import numpy as np
    import matplotlib.pyplot as plt

    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    from matplotlib.ticker import NullFormatter


    PROJECT_ROOT_OVERRIDE = REPO_ROOT


    def resolve_project_root():
        candidates = []

        if PROJECT_ROOT_OVERRIDE is not None:
            candidates.append(
                Path(
                    PROJECT_ROOT_OVERRIDE
                ).expanduser()
            )

        existing = globals().get(
            "PROJECT_ROOT"
        )

        if existing is not None:
            candidates.append(
                Path(
                    existing
                ).expanduser()
            )

        cwd = Path.cwd().resolve()

        candidates.extend(
            [
                cwd,
                *cwd.parents,
            ]
        )

        candidates.append(
            REPO_ROOT
        )

        seen = set()

        for candidate in candidates:
            candidate = candidate.resolve()

            if candidate in seen:
                continue

            seen.add(candidate)

            cache_path = (
                candidate
                / "build"
                / "reproduced_cache"
                / "2d_burgers"
                / "burgers2d_3d_median_seed_benchmark.npz"
            )

            if cache_path.is_file():
                return candidate

        raise FileNotFoundError(
            "2D Burgers 3D cache was not found. "
            "Run B2D-FIG-DATA once."
        )


    def clean_ticklabels(values):
        labels = []

        for value in values:
            if abs(value) < 1.0e-12:
                value = 0.0

            labels.append(
                f"{value:.2f}"
                .rstrip("0")
                .rstrip(".")
            )

        return labels


    def clean_colorbar_labels(values):
        labels = []

        for value in values:
            value = float(value)

            if abs(value) < 1.0e-12:
                labels.append(
                    "0"
                )

            elif abs(value) >= 1.0:
                labels.append(
                    f"{value:.2f}"
                    .rstrip("0")
                    .rstrip(".")
                )

            elif abs(value) >= 0.01:
                labels.append(
                    f"{value:.3f}"
                    .rstrip("0")
                    .rstrip(".")
                )

            else:
                labels.append(
                    f"{value:.1e}"
                )

        return labels


    def style_3d_axis_minimal(
        axis,
        cfg,
        show_ticks=False,
        show_ticklabels=False,
        show_labels=False,
        elev=22,
        azim=-55,
    ):
        axis.set_xlim(
            cfg.x_min,
            cfg.x_max,
        )

        axis.set_ylim(
            cfg.y_min,
            cfg.y_max,
        )

        axis.set_zlim(
            cfg.t_min,
            cfg.t_max,
        )

        try:
            axis.set_box_aspect(
                (
                    1.0,
                    1.0,
                    0.65,
                )
            )

        except Exception:
            pass

        try:
            axis.set_proj_type(
                "ortho"
            )

        except Exception:
            pass

        axis.view_init(
            elev=elev,
            azim=azim,
        )

        axis.grid(
            False
        )

        axis.xaxis.pane.set_alpha(
            0.0
        )

        axis.yaxis.pane.set_alpha(
            0.0
        )

        axis.zaxis.pane.set_alpha(
            0.0
        )

        try:
            axis.xaxis.line.set_linewidth(
                0.7
            )

            axis.yaxis.line.set_linewidth(
                0.7
            )

            axis.zaxis.line.set_linewidth(
                0.7
            )

        except Exception:
            pass

        if show_ticks:
            xticks = [
                cfg.x_min,
                0.0,
                cfg.x_max,
            ]

            yticks = [
                cfg.y_min,
                0.0,
                cfg.y_max,
            ]

            zticks = [
                cfg.t_min,
                0.5
                * (
                    cfg.t_min
                    + cfg.t_max
                ),
                cfg.t_max,
            ]

            axis.set_xticks(
                xticks
            )

            axis.set_yticks(
                yticks
            )

            axis.set_zticks(
                zticks
            )

            axis.tick_params(
                axis="both",
                which="major",
                pad=AXIS_TICK_PAD,
                length=AXIS_TICK_LENGTH,
                width=AXIS_TICK_WIDTH,
                labelsize=AXIS_TICK_LABEL_SIZE,
            )

            try:
                axis.zaxis.set_tick_params(
                    pad=AXIS_TICK_PAD,
                    length=AXIS_TICK_LENGTH,
                    width=AXIS_TICK_WIDTH,
                    labelsize=AXIS_TICK_LABEL_SIZE,
                )

            except Exception:
                pass

            if show_ticklabels:
                axis.set_xticklabels(
                    clean_ticklabels(
                        xticks
                    ),
                    fontsize=AXIS_TICK_LABEL_SIZE,
                )

                axis.set_yticklabels(
                    clean_ticklabels(
                        yticks
                    ),
                    fontsize=AXIS_TICK_LABEL_SIZE,
                )

                axis.set_zticklabels(
                    clean_ticklabels(
                        zticks
                    ),
                    fontsize=AXIS_TICK_LABEL_SIZE,
                )

            else:
                axis.xaxis.set_major_formatter(
                    NullFormatter()
                )

                axis.yaxis.set_major_formatter(
                    NullFormatter()
                )

                axis.zaxis.set_major_formatter(
                    NullFormatter()
                )

        else:
            axis.set_xticks(
                []
            )

            axis.set_yticks(
                []
            )

            axis.set_zticks(
                []
            )

        if show_labels:
            axis.set_xlabel(
                r"$x$",
                labelpad=AXIS_LABEL_PAD_X,
                fontsize=AXIS_LABEL_SIZE,
            )

            axis.set_ylabel(
                r"$y$",
                labelpad=AXIS_LABEL_PAD_Y,
                fontsize=AXIS_LABEL_SIZE,
            )

            axis.set_zlabel(
                r"$t$",
                labelpad=AXIS_LABEL_PAD_T,
                fontsize=AXIS_LABEL_SIZE,
            )

        else:
            axis.set_xlabel(
                ""
            )

            axis.set_ylabel(
                ""
            )

            axis.set_zlabel(
                ""
            )


    def error_rgba(
        error_values,
        error_cmap,
        error_norm,
        error_vmax,
    ):
        scaled = np.clip(
            error_values
            / (
                error_vmax
                + 1.0e-12
            ),
            0.0,
            1.0,
        )

        colors = error_cmap(
            error_norm(
                error_values
            )
        )

        colors[:, 3] = (
            0.05
            + 0.62
            * (
                scaled ** 0.75
            )
        )

        return colors


    PROJECT_ROOT = resolve_project_root()


    CACHE_PATH = (
        PROJECT_ROOT
        / "build"
        / "reproduced_cache"
        / "2d_burgers"
        / "burgers2d_3d_median_seed_benchmark.npz"
    )


    FIGURE_DIR = (
        PROJECT_ROOT
        / "results"
        / "reproduced_figures"
        / "2d_burgers"
    )


    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    with np.load(
        CACHE_PATH,
        allow_pickle=False,
    ) as data:

        x = np.asarray(
            data["x"],
            dtype=np.float64,
        )

        y = np.asarray(
            data["y"],
            dtype=np.float64,
        )

        t = np.asarray(
            data["t"],
            dtype=np.float64,
        )

        U_exact = np.asarray(
            data["U_exact"],
            dtype=np.float64,
        )

        U_pinn = np.asarray(
            data["U_pinn"],
            dtype=np.float64,
        )

        U_tpinn = np.asarray(
            data["U_tpinn"],
            dtype=np.float64,
        )

        E_pinn = np.asarray(
            data["E_pinn"],
            dtype=np.float64,
        )

        E_tpinn = np.asarray(
            data["E_tpinn"],
            dtype=np.float64,
        )

        err_vmax = float(
            data["err_vmax"]
        )

        selected_seed = int(
            data["selected_seed"]
        )

        median_metric = str(
            data[
                "median_metric"
            ].item()
        )

        cfg_plot = SimpleNamespace(
            x_min=float(
                data["cfg_x_min"]
            ),

            x_max=float(
                data["cfg_x_max"]
            ),

            y_min=float(
                data["cfg_y_min"]
            ),

            y_max=float(
                data["cfg_y_max"]
            ),

            t_min=float(
                data["cfg_t_min"]
            ),

            t_max=float(
                data["cfg_t_max"]
            ),

            uL=float(
                data["cfg_uL"]
            ),

            uR=float(
                data["cfg_uR"]
            ),

            theta_deg=float(
                data["cfg_theta_deg"]
            ),

            eta0=float(
                data["cfg_eta0"]
            ),
        )


    expected_size = (
        len(x)
        * len(y)
        * len(t)
    )


    for name, values in {
        "U_exact": U_exact,
        "U_pinn": U_pinn,
        "U_tpinn": U_tpinn,
        "E_pinn": E_pinn,
        "E_tpinn": E_tpinn,
    }.items():

        if values.shape != (
            expected_size,
        ):
            raise ValueError(
                f"Unexpected shape for {name}: "
                f"{values.shape}"
            )

        if not np.isfinite(
            values
        ).all():
            raise ValueError(
                f"Non-finite values in {name}."
            )


    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,

        "font.family": "serif",
        "mathtext.fontset": "stix",

        "font.size": 8.8,

        "axes.titlesize": 8.5,
        "axes.labelsize": 8.5,

        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0,

        "axes.linewidth": 0.7,
    })


    FIGSIZE_3D = (
        9.2,
        2.8,
    )


    HEATMAP_LEFT = 0.055
    HEATMAP_RIGHT = 0.935
    HEATMAP_BOTTOM = 0.120
    HEATMAP_TOP = 0.930

    HEATMAP_WSPACE = -0.50 # 0.15
    HEATMAP_HSPACE = -0.15 # 0.15


    PANEL_TITLE_SIZE = 8.0
    PANEL_TITLE_PAD = 3

    COLORBAR_LABEL_SIZE = 7.0
    COLORBAR_TICK_SIZE = 7.0

    AXIS_LABEL_SIZE = 8.0

    AXIS_LABEL_PAD_X = -5
    AXIS_LABEL_PAD_Y = -5
    AXIS_LABEL_PAD_T = -5

    AXIS_TICK_LABEL_SIZE = 7.0
    AXIS_TICK_PAD = -1
    AXIS_TICK_LENGTH = 2.0
    AXIS_TICK_WIDTH = 0.55

    POINT_SIZE_3D = 4.5
    ALPHA_3D = 0.16

    SOLUTION_CMAP_NAME = "jet"
    ERROR_CMAP_NAME = "magma"


    COLUMN_COUNT = 5
    ROW_COUNT = 2


    PANEL_W = (
        HEATMAP_RIGHT
        - HEATMAP_LEFT
    ) / (
        COLUMN_COUNT
        + (
            COLUMN_COUNT
            - 1
        )
        * HEATMAP_WSPACE
    )


    COLUMN_GAP = (
        HEATMAP_WSPACE
        * PANEL_W
    )


    PANEL_H = (
        HEATMAP_TOP
        - HEATMAP_BOTTOM
    ) / (
        ROW_COUNT
        + HEATMAP_HSPACE
    )


    ROW_GAP = (
        HEATMAP_HSPACE
        * PANEL_H
    )


    TOP_Y = (
        HEATMAP_BOTTOM
        + PANEL_H
        + ROW_GAP
    )


    BOT_Y = HEATMAP_BOTTOM


    TOP_X0 = HEATMAP_LEFT


    TOP_X1 = (
        HEATMAP_LEFT
        + 2.0
        * (
            PANEL_W
            + COLUMN_GAP
        )
    )


    TOP_X2 = (
        HEATMAP_LEFT
        + 4.0
        * (
            PANEL_W
            + COLUMN_GAP
        )
    )


    BOT_X0 = (
        HEATMAP_LEFT
        + (
            PANEL_W
            + COLUMN_GAP
        )
    )


    BOT_X1 = (
        HEATMAP_LEFT
        + 3.0
        * (
            PANEL_W
            + COLUMN_GAP
        )
    )


    COLORBAR_W = 0.0038
    COLORBAR_H = 0.20  # 0.120

    COLORBAR_X_RATIO = 1.00

    COLORBAR_SOL_DY = 0.000
    COLORBAR_ERR_DY = 0.000


    COLORBAR_SOL_X = (
        TOP_X2
        + COLORBAR_X_RATIO
        * PANEL_W
    )


    COLORBAR_ERR_X = (
        BOT_X1
        + COLORBAR_X_RATIO
        * PANEL_W
    )


    COLORBAR_SOL_Y = (
        TOP_Y
        + 0.5
        * PANEL_H
        - 0.5
        * COLORBAR_H
        + COLORBAR_SOL_DY
    )


    COLORBAR_ERR_Y = (
        BOT_Y
        + 0.5
        * PANEL_H
        - 0.5
        * COLORBAR_H
        + COLORBAR_ERR_DY
    )


    X, Y, T = np.meshgrid(
        x,
        y,
        t,
        indexing="ij",
    )


    xf = X.reshape(
        -1
    )

    yf = Y.reshape(
        -1
    )

    tf = T.reshape(
        -1
    )


    err_vmax = max(
        float(
            err_vmax
        ),
        1.0e-8,
    )


    solution_cmap = plt.get_cmap(
        SOLUTION_CMAP_NAME
    )


    error_cmap = plt.get_cmap(
        ERROR_CMAP_NAME
    )


    u_min = min(
        float(
            cfg_plot.uL
        ),
        float(
            cfg_plot.uR
        ),
    )


    u_max = max(
        float(
            cfg_plot.uL
        ),
        float(
            cfg_plot.uR
        ),
    )


    u_mid = 0.5 * (
        u_min
        + u_max
    )


    solution_norm = Normalize(
        vmin=u_min,
        vmax=u_max,
    )


    error_norm = Normalize(
        vmin=0.0,
        vmax=err_vmax,
    )


    solution_colors_exact = solution_cmap(
        solution_norm(
            U_exact
        )
    )


    solution_colors_pinn = solution_cmap(
        solution_norm(
            U_pinn
        )
    )


    solution_colors_tpinn = solution_cmap(
        solution_norm(
            U_tpinn
        )
    )


    solution_colors_exact[:, 3] = (
        ALPHA_3D
    )

    solution_colors_pinn[:, 3] = (
        ALPHA_3D
    )

    solution_colors_tpinn[:, 3] = (
        ALPHA_3D
    )


    plt.close(
        "all"
    )


    figure = plt.figure(
        figsize=FIGSIZE_3D
    )


    figure.patch.set_facecolor(
        "white"
    )


    axis_exact = figure.add_axes(
        [
            TOP_X0,
            TOP_Y,
            PANEL_W,
            PANEL_H,
        ],
        projection="3d",
    )


    axis_pinn = figure.add_axes(
        [
            TOP_X1,
            TOP_Y,
            PANEL_W,
            PANEL_H,
        ],
        projection="3d",
    )


    axis_tpinn = figure.add_axes(
        [
            TOP_X2,
            TOP_Y,
            PANEL_W,
            PANEL_H,
        ],
        projection="3d",
    )


    axis_error_pinn = figure.add_axes(
        [
            BOT_X0,
            BOT_Y,
            PANEL_W,
            PANEL_H,
        ],
        projection="3d",
    )


    axis_error_tpinn = figure.add_axes(
        [
            BOT_X1,
            BOT_Y,
            PANEL_W,
            PANEL_H,
        ],
        projection="3d",
    )


    colorbar_axis_solution = figure.add_axes(
        [
            COLORBAR_SOL_X,
            COLORBAR_SOL_Y,
            COLORBAR_W,
            COLORBAR_H,
        ]
    )


    colorbar_axis_error = figure.add_axes(
        [
            COLORBAR_ERR_X,
            COLORBAR_ERR_Y,
            COLORBAR_W,
            COLORBAR_H,
        ]
    )


    scatter_kwargs = {
        "s": POINT_SIZE_3D,
        "linewidths": 0.0,
        "depthshade": False,
    }


    axis_exact.scatter(
        xf,
        yf,
        tf,
        c=solution_colors_exact,
        **scatter_kwargs,
    )


    axis_pinn.scatter(
        xf,
        yf,
        tf,
        c=solution_colors_pinn,
        **scatter_kwargs,
    )


    axis_tpinn.scatter(
        xf,
        yf,
        tf,
        c=solution_colors_tpinn,
        **scatter_kwargs,
    )


    axis_error_pinn.scatter(
        xf,
        yf,
        tf,
        c=error_rgba(
            E_pinn,
            error_cmap,
            error_norm,
            err_vmax,
        ),
        s=POINT_SIZE_3D,
        linewidths=0.0,
        depthshade=False,
    )


    axis_error_tpinn.scatter(
        xf,
        yf,
        tf,
        c=error_rgba(
            E_tpinn,
            error_cmap,
            error_norm,
            err_vmax,
        ),
        s=POINT_SIZE_3D,
        linewidths=0.0,
        depthshade=False,
    )


    axis_exact.set_title(
        "(a) Exact",
        fontsize=PANEL_TITLE_SIZE,
        pad=PANEL_TITLE_PAD,
    )


    axis_pinn.set_title(
        "(b) PINN",
        fontsize=PANEL_TITLE_SIZE,
        pad=PANEL_TITLE_PAD,
    )


    axis_tpinn.set_title(
        "(c) TRG-PINN",
        fontsize=PANEL_TITLE_SIZE,
        pad=PANEL_TITLE_PAD,
    )


    axis_error_pinn.set_title(
        r"(d) PINN error ($u$)",
        fontsize=PANEL_TITLE_SIZE,
        pad=PANEL_TITLE_PAD,
    )


    axis_error_tpinn.set_title(
        r"(e) TRG-PINN error ($u$)",
        fontsize=PANEL_TITLE_SIZE,
        pad=PANEL_TITLE_PAD,
    )


    style_3d_axis_minimal(
        axis_exact,
        cfg_plot,
        show_ticks=True,
        show_ticklabels=True,
        show_labels=True,
        elev=22,
        azim=-55,
    )


    for axis in (
        axis_pinn,
        axis_tpinn,
        axis_error_pinn,
        axis_error_tpinn,
    ):
        style_3d_axis_minimal(
            axis,
            cfg_plot,
            show_ticks=True,
            show_ticklabels=False,
            show_labels=False,
            elev=22,
            azim=-55,
        )


    solution_scalar = ScalarMappable(
        norm=solution_norm,
        cmap=solution_cmap,
    )


    solution_scalar.set_array(
        []
    )


    solution_colorbar = figure.colorbar(
        solution_scalar,
        cax=colorbar_axis_solution,
    )


    solution_ticks = [
        u_min,
        u_mid,
        u_max,
    ]


    solution_colorbar.set_ticks(
        solution_ticks
    )


    solution_colorbar.set_ticklabels(
        clean_colorbar_labels(
            solution_ticks
        )
    )


    solution_colorbar.set_label(
        r"$u$",
        labelpad=4,
        fontsize=COLORBAR_LABEL_SIZE,
    )


    solution_colorbar.ax.tick_params(
        labelsize=COLORBAR_TICK_SIZE,
        length=2.0,
        width=0.6,
    )


    error_scalar = ScalarMappable(
        norm=error_norm,
        cmap=error_cmap,
    )


    error_scalar.set_array(
        []
    )


    error_colorbar = figure.colorbar(
        error_scalar,
        cax=colorbar_axis_error,
    )


    error_ticks = [
        0.0,
        0.5 * err_vmax,
        err_vmax,
    ]

    error_colorbar.set_ticks(
        error_ticks
    )


    error_colorbar.set_ticklabels(
        [
            f"{value:.2f}"
            for value in error_ticks
        ]
    )


    error_colorbar.set_label(
        "absolute error",
        labelpad=4,
        fontsize=COLORBAR_LABEL_SIZE,
    )


    error_colorbar.ax.tick_params(
        labelsize=COLORBAR_TICK_SIZE,
        length=2.0,
        width=0.6,
    )


    pdf_path = (
        FIGURE_DIR
        / "fig_burgers2d_3d_compact.pdf"
    )


    png_path = (
        FIGURE_DIR
        / "fig_burgers2d_3d_compact.png"
    )


    figure.savefig(
        pdf_path
    )


    figure.savefig(
        png_path,
        dpi=300,
    )


    plt.show()


    print(
        "[OK] 2D Burgers 3D figure saved"
    )

    print(
        "Median metric:",
        median_metric,
    )

    print(
        "Median seed  :",
        selected_seed,
    )

    print(
        "Cache        :",
        CACHE_PATH,
    )

    print(
        "Saved        :",
        pdf_path,
    )

    print(
        "Saved        :",
        png_path,
    )
    expected = [
        FIGURE_DIR / "burgers2d_normal_line_multiseed.pdf",
        FIGURE_DIR / "burgers2d_normal_line_multiseed.png",
        FIGURE_DIR / "fig_burgers2d_3d_compact.pdf",
        FIGURE_DIR / "fig_burgers2d_3d_compact.png",
        FIGURE_DIR / "fig_burgers2d_error_median_seed.pdf",
        FIGURE_DIR / "fig_burgers2d_error_median_seed.png",
        FIGURE_DIR / "fig_burgers2d_gate_median_seed.pdf",
        FIGURE_DIR / "fig_burgers2d_gate_median_seed.png",
        FIGURE_DIR / "fig_burgers2d_solution_median_seed.pdf",
        FIGURE_DIR / "fig_burgers2d_solution_median_seed.png",
    ]
    return expected


def reproduce(
    *,
    device: str,
    data_only: bool = False,
) -> list[Path]:
    cache_dir = (
        REPO_ROOT
        / "build"
        / "reproduced_cache"
        / "2d_burgers"
    )
    figure_dir = (
        REPO_ROOT
        / "results"
        / "reproduced_figures"
        / "2d_burgers"
    )
    cache_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    figure_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    line_cache, heatmap_cache, cube_cache, seed = (
        build_caches(
            cache_dir,
            device=device,
        )
    )

    print(
        "[OK] Rebuilt 2D Burgers figure data "
        f"from frozen checkpoints; representative seed={seed}"
    )
    print("Figure device :", device)
    print(
        "Float32 matmul precision :",
        FIGURE_MATMUL_PRECISION,
    )
    print("Line cache   :", line_cache)
    print("Heatmap cache:", heatmap_cache)
    print("3D cache     :", cube_cache)

    if data_only:
        return []

    paths = render_all_figures()

    for path in paths:
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)

    return paths


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device",
        default="cuda:0",
    )
    parser.add_argument(
        "--data-only",
        action="store_true",
    )
    args = parser.parse_args()

    paths = reproduce(
        device=args.device,
        data_only=args.data_only,
    )

    for path in paths:
        print(path)

    print("Training run: NO")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
