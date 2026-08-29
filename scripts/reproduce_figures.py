#!/usr/bin/env python
"""Regenerate canonical 1D benchmark figure data and manuscript figures.

No training is performed. Frozen checkpoints are loaded from immutable
``artifacts/reported`` trees and outputs are written only to ``build`` and
``results/reproduced_figures``.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import gc
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from trgpinn.equations.burgers_1d import (
    Burgers1DConfig,
    build_model,
    exact_solution,
    predict_points,
)
from trgpinn.metrics import pointwise_median_rmse_band
from trgpinn.utils import load_checkpoint_into_model, read_json, sha256_file

EXPECTED_SEEDS = [2026, 7, 42, 100, 31415]
PRIMARY_METRIC = "space_time_rel_l2"
LINE_TIMES = np.asarray([0.25, 0.50, 0.75, 1.00], dtype=np.float64)
LINE_NX = 1001
HEATMAP_NX = 501
HEATMAP_NT = 251
CACHE_RTOL = 2.0e-5
CACHE_ATOL = 2.0e-6


def load_reported_model(method: str, seed: int, *, device: str = "cpu"):
    run_dir = REPO_ROOT / "artifacts" / "reported" / "1d_burgers" / method / f"seed_{seed}"
    payload = read_json(run_dir / "config.json")
    cfg = Burgers1DConfig.from_legacy_mapping(
        payload.get("config", payload), seed=seed, device=device
    )
    torch_device = torch.device(device)
    dtype = torch.float64 if cfg.dtype == "float64" else torch.float32
    model = build_model(cfg).to(device=torch_device, dtype=dtype)
    load_checkpoint_into_model(model, run_dir / "model_final.pt")
    model.eval()
    return model, cfg, run_dir / "model_final.pt"


def select_representative_seed() -> int:
    master = pd.read_csv(REPO_ROOT / "results" / "reported_metrics" / "all_metrics_final.csv")
    rows = master[
        master["equation"].astype(str).eq("1d_burgers")
        & master["method"].astype(str).eq("Ours")
    ].copy()
    rows["seed"] = pd.to_numeric(rows["seed"], errors="raise").astype(int)
    rows[PRIMARY_METRIC] = pd.to_numeric(rows[PRIMARY_METRIC], errors="raise")
    order = {seed: index for index, seed in enumerate(EXPECTED_SEEDS)}
    rows = rows[rows["seed"].isin(EXPECTED_SEEDS)]
    rows["seed_order"] = rows["seed"].map(order)
    rows = rows.sort_values([PRIMARY_METRIC, "seed_order"], kind="mergesort").reset_index(drop=True)
    if len(rows) != 5:
        raise RuntimeError("Expected five TRG-PINN seed rows.")
    return int(rows.iloc[len(rows) // 2]["seed"])


def numpy_gate_grid(model, cfg: Burgers1DConfig, x_values, t_values):
    X, T = np.meshgrid(x_values, t_values, indexing="xy")
    x_flat = X.reshape(-1)
    t_flat = T.reshape(-1)
    h_probe = cfg.h_min_factor * (cfg.x_max - cfg.x_min) / np.sqrt(float(cfg.n_f))
    cmin = float(cfg.cmin_end)
    x_m1 = x_flat - h_probe
    x_p1 = x_flat + h_probe
    x_m2 = x_flat - 2.0 * h_probe
    x_p2 = x_flat + 2.0 * h_probe
    valid = ((x_m2 >= cfg.x_min) & (x_p2 <= cfg.x_max)).astype(np.float64)
    x_m1 = np.clip(x_m1, cfg.x_min, cfg.x_max)
    x_p1 = np.clip(x_p1, cfg.x_min, cfg.x_max)
    x_m2 = np.clip(x_m2, cfg.x_min, cfg.x_max)
    x_p2 = np.clip(x_p2, cfg.x_min, cfg.x_max)
    u_m1 = predict_points(model, x_m1, t_flat)
    u_p1 = predict_points(model, x_p1, t_flat)
    u_m2 = predict_points(model, x_m2, t_flat)
    u_p2 = predict_points(model, x_p2, t_flat)
    scale = max(abs(cfg.uL - cfg.uR), 1.0e-8)
    jump_h = np.abs(u_m1 - u_p1) / scale
    jump_2h = np.abs(u_m2 - u_p2) / scale
    ratio = np.clip(jump_h / (jump_2h + cfg.trace_ratio_epsilon), 0.0, 2.0)
    valid_sum = float(valid.sum())
    if valid_sum > 0.0:
        jump_mean = float((jump_h * valid).sum() / (valid_sum + cfg.batch_mean_epsilon))
    else:
        jump_mean = float(jump_h.mean())
    jump_mean = max(jump_mean, cfg.batch_mean_epsilon)
    normalized_jump = jump_h / jump_mean

    def sigmoid(values):
        return 1.0 / (1.0 + np.exp(-np.clip(values, -60.0, 60.0)))

    gate = (
        sigmoid((normalized_jump - 1.0) / cfg.beta)
        * sigmoid((ratio - cmin) / cfg.beta)
        * valid
    )
    return gate.reshape(len(t_values), len(x_values)), float(h_probe), float(cmin)


def build_caches(output_dir: Path) -> tuple[Path, Path, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    representative_seed = select_representative_seed()
    _, cfg_reference, _ = load_reported_model("trg_pinn", representative_seed, device="cpu")

    x_line = np.linspace(cfg_reference.x_min, cfg_reference.x_max, LINE_NX, dtype=np.float64)
    X_line, T_line = np.meshgrid(x_line, LINE_TIMES, indexing="xy")
    exact_line = exact_solution(X_line, T_line, cfg_reference)
    predictions = {"pinn": [], "trg_pinn": []}
    checkpoint_paths = []
    checkpoint_hashes = []

    for method in ("pinn", "trg_pinn"):
        for seed in EXPECTED_SEEDS:
            model, cfg, checkpoint = load_reported_model(method, seed, device="cpu")
            values = predict_points(model, X_line.reshape(-1), T_line.reshape(-1)).reshape(
                len(LINE_TIMES), LINE_NX
            )
            predictions[method].append(values)
            checkpoint_paths.append(str(checkpoint.resolve()))
            checkpoint_hashes.append(sha256_file(checkpoint))
            del model
            gc.collect()

    pinn_all = np.asarray(predictions["pinn"], dtype=np.float64)
    trg_all = np.asarray(predictions["trg_pinn"], dtype=np.float64)
    pinn_median, _, pinn_lower, pinn_upper = pointwise_median_rmse_band(pinn_all, exact_line)
    trg_median, _, trg_lower, trg_upper = pointwise_median_rmse_band(trg_all, exact_line)

    line_path = output_dir / "burgers_multiseed_paper_cache_benchmark.npz"
    np.savez_compressed(
        line_path,
        x=x_line,
        times=LINE_TIMES,
        seeds=np.asarray(EXPECTED_SEEDS, dtype=np.int64),
        exact=exact_line.astype(np.float32),
        pinn_all=pinn_all.astype(np.float32),
        tpinn_all=trg_all.astype(np.float32),
        pinn_median=pinn_median.astype(np.float32),
        tpinn_median=trg_median.astype(np.float32),
        pinn_lower=pinn_lower.astype(np.float32),
        pinn_upper=pinn_upper.astype(np.float32),
        tpinn_lower=trg_lower.astype(np.float32),
        tpinn_upper=trg_upper.astype(np.float32),
        primary_metric=np.asarray(PRIMARY_METRIC),
        representative_seed=np.int64(representative_seed),
        checkpoint_paths=np.asarray(checkpoint_paths),
        checkpoint_sha256=np.asarray(checkpoint_hashes),
        source_csv=np.asarray(
            str((REPO_ROOT / "results" / "reported_metrics" / "all_metrics_final.csv").resolve())
        ),
        generated_at=np.asarray(datetime.now().isoformat(timespec="seconds")),
    )

    pinn_model, _, pinn_checkpoint = load_reported_model(
        "pinn", representative_seed, device="cpu"
    )
    trg_model, cfg_plot, trg_checkpoint = load_reported_model(
        "trg_pinn", representative_seed, device="cpu"
    )
    x_heat = np.linspace(cfg_plot.x_min, cfg_plot.x_max, HEATMAP_NX, dtype=np.float64)
    t_heat = np.linspace(cfg_plot.t_min, cfg_plot.t_max, HEATMAP_NT, dtype=np.float64)
    X_heat, T_heat = np.meshgrid(x_heat, t_heat, indexing="xy")
    exact_heat = exact_solution(X_heat, T_heat, cfg_plot)
    pinn_heat = predict_points(pinn_model, X_heat.reshape(-1), T_heat.reshape(-1)).reshape(
        HEATMAP_NT, HEATMAP_NX
    )
    trg_heat = predict_points(trg_model, X_heat.reshape(-1), T_heat.reshape(-1)).reshape(
        HEATMAP_NT, HEATMAP_NX
    )
    gate, h_used, cmin_used = numpy_gate_grid(trg_model, cfg_plot, x_heat, t_heat)
    error_pinn = np.abs(exact_heat - pinn_heat)
    error_trg = np.abs(exact_heat - trg_heat)
    error_max = float(np.percentile(np.r_[error_pinn.ravel(), error_trg.ravel()], 99.5))

    heatmap_path = output_dir / "burgers_1d_median_seed_heatmap_benchmark.npz"
    np.savez_compressed(
        heatmap_path,
        x=x_heat,
        t=t_heat,
        U_exact=exact_heat.astype(np.float32),
        U_pinn=pinn_heat.astype(np.float32),
        U_tpinn=trg_heat.astype(np.float32),
        G=gate.astype(np.float32),
        E_pinn=error_pinn.astype(np.float32),
        E_tpinn=error_trg.astype(np.float32),
        err_vmax=np.float64(error_max),
        x_min=np.float64(cfg_plot.x_min),
        x_max=np.float64(cfg_plot.x_max),
        t_max=np.float64(cfg_plot.t_max),
        h_used=np.float64(h_used),
        cmin_used=np.float64(cmin_used),
        nx_plot=np.int64(HEATMAP_NX),
        nt_plot=np.int64(HEATMAP_NT),
        source_seed=np.int64(representative_seed),
        source_metric=np.asarray(PRIMARY_METRIC),
        pinn_checkpoint=np.asarray(str(pinn_checkpoint.resolve())),
        tpinn_checkpoint=np.asarray(str(trg_checkpoint.resolve())),
        pinn_checkpoint_sha256=np.asarray(sha256_file(pinn_checkpoint)),
        tpinn_checkpoint_sha256=np.asarray(sha256_file(trg_checkpoint)),
        source_csv=np.asarray(
            str((REPO_ROOT / "results" / "reported_metrics" / "all_metrics_final.csv").resolve())
        ),
        generated_at=np.asarray(datetime.now().isoformat(timespec="seconds")),
    )
    return line_path, heatmap_path, representative_seed


def _compact_tick(value, _position=None):
    if np.isclose(value, round(value)):
        return f"{int(round(value))}"
    return f"{value:.2f}".rstrip("0").rstrip(".")


def _format_colorbar(colorbar):
    colorbar.outline.set_linewidth(0.65)
    colorbar.ax.tick_params(
        axis="both",
        which="both",
        direction="out",
        length=2.5,
        width=0.7,
        labelsize=8.2,
        pad=2.0,
    )


def render_heatmaps(cache_path: Path, figure_dir: Path) -> list[Path]:
    figure_dir.mkdir(parents=True, exist_ok=True)
    with np.load(cache_path, allow_pickle=False) as data:
        x = np.asarray(data["x"], dtype=np.float64)
        t = np.asarray(data["t"], dtype=np.float64)
        exact = np.asarray(data["U_exact"], dtype=np.float64)
        pinn = np.asarray(data["U_pinn"], dtype=np.float64)
        trg = np.asarray(data["U_tpinn"], dtype=np.float64)
        gate = np.asarray(data["G"], dtype=np.float64)
        error_pinn = np.asarray(data["E_pinn"], dtype=np.float64)
        error_trg = np.asarray(data["E_tpinn"], dtype=np.float64)
        error_max = float(np.asarray(data["err_vmax"]).item())

    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 600,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 9.5,
            "axes.titlesize": 8.0,
            "axes.labelsize": 9.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.0,
            "axes.linewidth": 0.75,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "legend.fontsize": 8.5,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "xtick.major.width": 0.75,
            "ytick.major.width": 0.75,
        }
    )

    extent = [float(x.min()), float(x.max()), float(t.min()), float(t.max())]

    def format_axis(axis, first=False):
        axis.set_xlim(x.min(), x.max())
        axis.set_ylim(t.min(), t.max())
        axis.set_yticks(np.linspace(t.min(), t.max(), 5))
        axis.yaxis.set_major_formatter(FuncFormatter(_compact_tick))
        axis.tick_params(
            axis="both",
            which="both",
            direction="out",
            bottom=True,
            left=True,
            top=False,
            right=False,
            labelbottom=first,
            labelleft=first,
            pad=2.0,
        )
        if first:
            axis.set_xlabel(r"$x$")
            axis.set_ylabel(r"$t$")

    comparison = plt.figure(figsize=(9.2, 2.8), constrained_layout=False)
    grid = comparison.add_gridspec(
        2, 5, left=0.055, right=0.935, bottom=0.120, top=0.930, wspace=0.20, hspace=-0.20
    )
    solution_axes = [
        comparison.add_subplot(grid[0, 0]),
        comparison.add_subplot(grid[0, 2]),
        comparison.add_subplot(grid[0, 4]),
    ]
    error_axes = [comparison.add_subplot(grid[1, 1]), comparison.add_subplot(grid[1, 3])]

    solution_image = None
    for index, (axis, values, title) in enumerate(
        zip(
            solution_axes,
            [exact, pinn, trg],
            [r"(a) Exact ($u$)", r"(b) PINN ($u$)", r"(c) TRG-PINN ($u$)"],
        )
    ):
        solution_image = axis.imshow(
            values,
            extent=extent,
            origin="lower",
            aspect="auto",
            cmap="jet",
            vmin=0.0,
            vmax=1.0,
            interpolation="nearest",
            rasterized=True,
        )
        axis.set_title(title, pad=5)
        format_axis(axis, first=index == 0)

    cax_solution = inset_axes(
        solution_axes[-1],
        width="2.5%",
        height="50%",
        loc="lower left",
        bbox_to_anchor=(1.080, 0.040, 1.0, 1.0),
        bbox_transform=solution_axes[-1].transAxes,
        borderpad=0.0,
    )
    colorbar_solution = comparison.colorbar(solution_image, cax=cax_solution)
    colorbar_solution.set_ticks([0.0, 0.5, 1.0])
    _format_colorbar(colorbar_solution)

    error_image = None
    for axis, values, title in zip(
        error_axes,
        [error_pinn, error_trg],
        [r"(d) PINN error ($u$)", r"(e) TRG-PINN error ($u$)"],
    ):
        error_image = axis.imshow(
            values,
            extent=extent,
            origin="lower",
            aspect="auto",
            cmap="magma",
            vmin=0.0,
            vmax=error_max,
            interpolation="nearest",
            rasterized=True,
        )
        axis.set_title(title, pad=5)
        format_axis(axis, first=False)

    cax_error = inset_axes(
        error_axes[-1],
        width="2.5%",
        height="50%",
        loc="lower left",
        bbox_to_anchor=(1.080, 0.040, 1.0, 1.0),
        bbox_transform=error_axes[-1].transAxes,
        borderpad=0.0,
    )
    colorbar_error = comparison.colorbar(error_image, cax=cax_error)
    colorbar_error.set_ticks(np.linspace(0.0, error_max, 4))
    colorbar_error.formatter = FuncFormatter(lambda value, _position=None: f"{value:.2f}")
    colorbar_error.update_ticks()
    _format_colorbar(colorbar_error)

    outputs = []
    for suffix, kwargs in (("pdf", {}), ("png", {"dpi": 600})):
        path = figure_dir / f"burgers_solution_comparison.{suffix}"
        comparison.savefig(path, **kwargs)
        outputs.append(path)
    plt.close(comparison)

    gate_figure, gate_axis = plt.subplots(1, 1, figsize=(3.4, 2.8), constrained_layout=True)
    gate_image = gate_axis.imshow(
        gate,
        extent=extent,
        origin="lower",
        aspect="auto",
        cmap="jet",
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
        rasterized=True,
    )
    gate_axis.set_title("Trace-ratio gate", pad=5)
    format_axis(gate_axis, first=True)
    gate_colorbar = gate_figure.colorbar(
        gate_image, ax=gate_axis, location="right", shrink=0.92, pad=0.03, fraction=0.055
    )
    gate_colorbar.set_label(r"$g$")
    gate_colorbar.set_ticks([0.0, 0.5, 1.0])
    _format_colorbar(gate_colorbar)
    for suffix, kwargs in (("pdf", {}), ("png", {"dpi": 600})):
        path = figure_dir / f"burgers_trace_ratio_gate.{suffix}"
        gate_figure.savefig(path, **kwargs)
        outputs.append(path)
    plt.close(gate_figure)
    return outputs


def render_line_figure(cache_path: Path, figure_dir: Path) -> list[Path]:
    with np.load(cache_path, allow_pickle=False) as data:
        x = np.asarray(data["x"], dtype=np.float64)
        times = np.asarray(data["times"], dtype=np.float64)
        exact = np.asarray(data["exact"], dtype=np.float64)
        pinn_median = np.asarray(data["pinn_median"], dtype=np.float64)
        trg_median = np.asarray(data["tpinn_median"], dtype=np.float64)
        pinn_lower = np.asarray(data["pinn_lower"], dtype=np.float64)
        pinn_upper = np.asarray(data["pinn_upper"], dtype=np.float64)
        trg_lower = np.asarray(data["tpinn_lower"], dtype=np.float64)
        trg_upper = np.asarray(data["tpinn_upper"], dtype=np.float64)

    plt.rcParams.update(
        {
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
        }
    )
    exact_color = "black"
    pinn_color = "#2C7BB6"
    trg_color = "#D7191C"
    figure, axes = plt.subplots(
        1, len(times), figsize=(7.9, 2.1), sharex=True, sharey=True, constrained_layout=False
    )
    axes = np.atleast_1d(axes)
    for panel, (axis, time_value) in enumerate(zip(axes, times)):
        axis.fill_between(x, pinn_lower[panel], pinn_upper[panel], color=pinn_color, alpha=0.13, linewidth=0, zorder=1)
        axis.fill_between(x, trg_lower[panel], trg_upper[panel], color=trg_color, alpha=0.16, linewidth=0, zorder=2)
        axis.plot(x, pinn_lower[panel], color=pinn_color, lw=0.50, ls=":", alpha=0.68, zorder=3)
        axis.plot(x, pinn_upper[panel], color=pinn_color, lw=0.50, ls=":", alpha=0.68, zorder=3)
        axis.plot(x, trg_lower[panel], color=trg_color, lw=0.50, ls=":", alpha=0.72, zorder=4)
        axis.plot(x, trg_upper[panel], color=trg_color, lw=0.50, ls=":", alpha=0.72, zorder=4)
        axis.plot(x, exact[panel], color=exact_color, lw=1.50, ls="-", zorder=7)
        axis.plot(x, pinn_median[panel], color=pinn_color, lw=1.50, ls="--", zorder=8)
        axis.plot(x, trg_median[panel], color=trg_color, lw=2.00, ls="--", zorder=9)
        axis.set_title(rf"$t={time_value:.2f}$", pad=6)
        axis.set_xlim(float(x.min()), float(x.max()))
        axis.set_ylim(-0.08, 1.08)
        axis.set_xticks(np.linspace(float(x.min()), float(x.max()), 5))
        axis.set_yticks([0.0, 0.25, 0.50, 0.75, 1.0])
        axis.set_yticklabels(["0", "0.25", "0.5", "0.75", "1"])
        axis.grid(True, which="major", alpha=0.13, linewidth=0.45)
        axis.tick_params(axis="both", which="major", pad=2)
        if panel == 0:
            axis.set_xlabel(r"$x$", labelpad=2)
            axis.set_ylabel(r"$u(x,t)$", labelpad=3)
        else:
            axis.tick_params(axis="x", labelbottom=False)
            axis.tick_params(axis="y", labelleft=False, left=False)

    handles = [
        Patch(facecolor=pinn_color, alpha=0.13, edgecolor=pinn_color, linewidth=0.7, label="PINN ± RMSE"),
        Patch(facecolor=trg_color, alpha=0.16, edgecolor=trg_color, linewidth=0.7, label="TRG-PINN ± RMSE"),
        Line2D([0], [0], color=exact_color, lw=1.50, ls="-", label="Exact"),
        Line2D([0], [0], color=pinn_color, lw=1.50, ls="--", label="PINN"),
        Line2D([0], [0], color=trg_color, lw=2.00, ls="--", label="TRG-PINN"),
    ]
    figure.legend(
        handles=handles,
        loc="upper center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.5, 1.025),
        handlelength=2.15,
        columnspacing=1.35,
        handletextpad=0.50,
    )
    plt.subplots_adjust(left=0.065, right=0.995, bottom=0.220, top=0.800, wspace=0.0)

    outputs = []
    for suffix, kwargs in (("pdf", {"bbox_inches": "tight"}), ("png", {"dpi": 300, "bbox_inches": "tight"})):
        path = figure_dir / f"burgers_multiseed_paper.{suffix}"
        figure.savefig(path, **kwargs)
        outputs.append(path)
    plt.close(figure)
    return outputs




EULER_EXPECTED_SEEDS = [2026, 7, 42, 100, 31415]
EULER_PRIMARY_METRIC = "primitive_scaled_space_time_rel_l2"
EULER_LINE_TIMES = np.asarray([0.10, 0.15, 0.20], dtype=np.float64)
EULER_LINE_NX = 1600
EULER_HEATMAP_NX = 601
EULER_HEATMAP_NT = 251


def _load_euler_reported_model(method: str, seed: int, *, device: str):
    from trgpinn.equations.euler_1d import Euler1DConfig, build_model
    from trgpinn.utils import (
        configure_torch_runtime,
        load_checkpoint_into_model,
        read_json,
        resolve_device,
    )

    run_dir = (
        REPO_ROOT
        / "artifacts"
        / "reported"
        / "1d_euler"
        / method
        / f"seed_{seed}"
    )
    payload = read_json(run_dir / "config.json")
    cfg = Euler1DConfig.from_legacy_mapping(
        payload.get("config", payload),
        seed=seed,
        device=device,
        save_outputs=False,
    )
    torch_device = resolve_device(device)
    dtype = configure_torch_runtime(cfg.dtype)
    model = build_model(cfg).to(device=torch_device, dtype=dtype)
    checkpoint = run_dir / "model_final.pt"
    load_checkpoint_into_model(model, checkpoint)
    model.eval()
    return model, cfg, checkpoint.resolve()


def _select_euler_representative_seed() -> int:
    master = pd.read_csv(
        REPO_ROOT / "results" / "reported_metrics" / "all_metrics_final.csv"
    )
    rows = master[
        master["equation"].astype(str).eq("1d_euler")
        & master["method"].astype(str).eq("Ours")
    ].copy()
    rows["seed"] = pd.to_numeric(rows["seed"], errors="raise").astype(int)
    rows[EULER_PRIMARY_METRIC] = pd.to_numeric(
        rows[EULER_PRIMARY_METRIC], errors="raise"
    )
    order = {seed: index for index, seed in enumerate(EULER_EXPECTED_SEEDS)}
    rows = rows[rows["seed"].isin(EULER_EXPECTED_SEEDS)]
    rows["seed_order"] = rows["seed"].map(order)
    rows = rows.sort_values(
        [EULER_PRIMARY_METRIC, "seed_order"], kind="mergesort"
    ).reset_index(drop=True)
    if len(rows) != 5:
        raise RuntimeError("Expected five 1D Euler TRG-PINN seed rows.")
    return int(rows.iloc[len(rows) // 2]["seed"])


def _euler_exact_grid(cfg, x_values, t_values):
    from trgpinn.equations.euler_1d import euler_exact_np

    X, T = np.meshgrid(
        np.asarray(x_values, dtype=np.float64),
        np.asarray(t_values, dtype=np.float64),
        indexing="xy",
    )
    rho, velocity, pressure = euler_exact_np(X, T, cfg)
    return np.stack([rho, velocity, pressure], axis=-1)


@torch.no_grad()
def _euler_gate_grid(model, cfg, x_values, t_values):
    from trgpinn.equations.euler_1d import scales, schedule_from_progress
    from trgpinn.gate import centered_euler_trace_ratio_gate_1d

    X, T = np.meshgrid(x_values, t_values, indexing="xy")
    parameter = next(model.parameters())
    x_tensor = torch.as_tensor(
        X.reshape(-1, 1), device=parameter.device, dtype=parameter.dtype
    )
    t_tensor = torch.as_tensor(
        T.reshape(-1, 1), device=parameter.device, dtype=parameter.dtype
    )
    h_probe, cmin, _ = schedule_from_progress(1.0, cfg)
    rho_scale, velocity_scale, pressure_scale, _, _ = scales(cfg)
    result = centered_euler_trace_ratio_gate_1d(
        model,
        x_tensor,
        t_tensor,
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
    shape = (len(t_values), len(x_values))
    return {
        "G": result.gate.detach().cpu().numpy().reshape(shape),
        "C": result.ratio.detach().cpu().numpy().reshape(shape),
        "Jhat": result.normalized_jump.detach().cpu().numpy().reshape(shape),
        "h": float(h_probe),
        "cmin": float(cmin),
        "Jbar": float(result.jump_mean.detach().cpu()),
    }


def build_euler_caches(cache_dir: Path, *, device: str):
    from trgpinn.equations.euler_1d import predict_primitive_points

    cache_dir.mkdir(parents=True, exist_ok=True)
    representative_seed = _select_euler_representative_seed()

    model0, cfg0, checkpoint0 = _load_euler_reported_model(
        "trg_pinn", representative_seed, device=device
    )
    del model0
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    x_line = np.linspace(cfg0.x_min, cfg0.x_max, EULER_LINE_NX)
    X_line, T_line = np.meshgrid(
        x_line, EULER_LINE_TIMES, indexing="xy"
    )
    exact_line = _euler_exact_grid(cfg0, x_line, EULER_LINE_TIMES)

    predictions = {"pinn": [], "trg_pinn": []}
    checkpoint_paths = []
    checkpoint_hashes = []
    source_rows = []

    for method in ("pinn", "trg_pinn"):
        for seed in EULER_EXPECTED_SEEDS:
            model, cfg, checkpoint = _load_euler_reported_model(
                method, seed, device=device
            )
            prediction = predict_primitive_points(
                model,
                X_line,
                T_line,
            ).reshape(len(EULER_LINE_TIMES), EULER_LINE_NX, 3)
            predictions[method].append(prediction)
            checkpoint_paths.append(str(checkpoint))
            checkpoint_hashes.append(sha256_file(checkpoint))
            source_rows.append(
                {
                    "method": "PINN" if method == "pinn" else "TRG-PINN",
                    "seed": seed,
                    "checkpoint": str(checkpoint),
                    "checkpoint_sha256": sha256_file(checkpoint),
                }
            )
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    pd.DataFrame(source_rows).to_csv(
        cache_dir / "euler1d_figure_source_audit.csv", index=False
    )

    pinn_all = np.asarray(predictions["pinn"], dtype=np.float64)
    trg_all = np.asarray(predictions["trg_pinn"], dtype=np.float64)
    pinn_median = np.median(pinn_all, axis=0)
    trg_median = np.median(trg_all, axis=0)
    pinn_rmse = np.sqrt(
        np.mean((pinn_all - exact_line[None, ...]) ** 2, axis=0)
    )
    trg_rmse = np.sqrt(
        np.mean((trg_all - exact_line[None, ...]) ** 2, axis=0)
    )

    line_cache = cache_dir / "euler1d_multiseed_line_benchmark.npz"
    np.savez_compressed(
        line_cache,
        x_euler=x_line,
        times_euler=EULER_LINE_TIMES,
        seeds=np.asarray(EULER_EXPECTED_SEEDS, dtype=np.int64),
        exact=exact_line.astype(np.float32),
        pinn_all=pinn_all.astype(np.float32),
        tpinn_all=trg_all.astype(np.float32),
        pinn_median=pinn_median.astype(np.float32),
        tpinn_median=trg_median.astype(np.float32),
        pinn_lower=(pinn_median - pinn_rmse).astype(np.float32),
        pinn_upper=(pinn_median + pinn_rmse).astype(np.float32),
        tpinn_lower=(trg_median - trg_rmse).astype(np.float32),
        tpinn_upper=(trg_median + trg_rmse).astype(np.float32),
        cfg_x_min=np.float64(cfg0.x_min),
        cfg_x_max=np.float64(cfg0.x_max),
        cfg_t_min=np.float64(cfg0.t_min),
        cfg_t_max=np.float64(cfg0.t_max),
        median_seed=np.int64(representative_seed),
        median_metric=np.asarray(EULER_PRIMARY_METRIC),
        checkpoint_paths=np.asarray(checkpoint_paths),
        checkpoint_sha256=np.asarray(checkpoint_hashes),
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
        reference_type=np.asarray("analytical_sod_riemann_solution"),
        generated_at=np.asarray(datetime.now().isoformat(timespec="seconds")),
    )

    pinn_model, pinn_cfg, pinn_checkpoint = _load_euler_reported_model(
        "pinn", representative_seed, device=device
    )
    trg_model, trg_cfg, trg_checkpoint = _load_euler_reported_model(
        "trg_pinn", representative_seed, device=device
    )

    x_heat = np.linspace(cfg0.x_min, cfg0.x_max, EULER_HEATMAP_NX)
    t_heat = np.linspace(cfg0.t_min, cfg0.t_max, EULER_HEATMAP_NT)
    X_heat, T_heat = np.meshgrid(x_heat, t_heat, indexing="xy")
    exact_heat = _euler_exact_grid(cfg0, x_heat, t_heat)
    pinn_heat = predict_primitive_points(
        pinn_model, X_heat, T_heat
    ).reshape(EULER_HEATMAP_NT, EULER_HEATMAP_NX, 3)
    trg_heat = predict_primitive_points(
        trg_model, X_heat, T_heat
    ).reshape(EULER_HEATMAP_NT, EULER_HEATMAP_NX, 3)
    gate = _euler_gate_grid(trg_model, trg_cfg, x_heat, t_heat)

    heatmap_cache = cache_dir / "euler1d_median_seed_heatmap_benchmark.npz"
    np.savez_compressed(
        heatmap_cache,
        x=x_heat,
        t=t_heat,
        exact=exact_heat.astype(np.float32),
        pinn=pinn_heat.astype(np.float32),
        tpinn=trg_heat.astype(np.float32),
        gate=gate["G"].astype(np.float32),
        C=gate["C"].astype(np.float32),
        Jhat=gate["Jhat"].astype(np.float32),
        gate_h=np.float64(gate["h"]),
        gate_cmin=np.float64(gate["cmin"]),
        gate_Jbar=np.float64(gate["Jbar"]),
        cfg_x_min=np.float64(cfg0.x_min),
        cfg_x_max=np.float64(cfg0.x_max),
        cfg_t_min=np.float64(cfg0.t_min),
        cfg_t_max=np.float64(cfg0.t_max),
        source_seed=np.int64(representative_seed),
        source_metric=np.asarray(EULER_PRIMARY_METRIC),
        pinn_checkpoint=np.asarray(str(pinn_checkpoint)),
        tpinn_checkpoint=np.asarray(str(trg_checkpoint)),
        pinn_checkpoint_sha256=np.asarray(sha256_file(pinn_checkpoint)),
        tpinn_checkpoint_sha256=np.asarray(sha256_file(trg_checkpoint)),
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
        reference_type=np.asarray("analytical_sod_riemann_solution"),
        generated_at=np.asarray(datetime.now().isoformat(timespec="seconds")),
    )

    del pinn_model
    del trg_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return line_cache, heatmap_cache, representative_seed


def render_euler_heatmaps(cache_path, figure_dir):
    EXPECTED_OUTPUT_NAMES = ['fig_euler_sod_rho_solution_comparison.pdf', 'fig_euler_sod_rho_solution_comparison.png', 'fig_euler_sod_u_solution_comparison.pdf', 'fig_euler_sod_u_solution_comparison.png', 'fig_euler_sod_p_solution_comparison.pdf', 'fig_euler_sod_p_solution_comparison.png', 'fig_euler_sod_trace_ratio_gate.pdf', 'fig_euler_sod_trace_ratio_gate.png']
    ### 1D euler fig heatmap

    from pathlib import Path
    from types import SimpleNamespace

    import numpy as np
    import matplotlib.pyplot as plt

    from matplotlib.ticker import FuncFormatter
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes


    SAVE_FIGURES = True
    HEATMAP_CACHE_PATH = Path(cache_path)
    FIGURE_DIR = Path(figure_dir)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    with np.load(
        HEATMAP_CACHE_PATH,
        allow_pickle=False,
    ) as data:

        x = np.asarray(
            data["x"],
            dtype=np.float64,
        )

        t = np.asarray(
            data["t"],
            dtype=np.float64,
        )

        W_exact = np.asarray(
            data["exact"],
            dtype=np.float64,
        )

        W_pinn = np.asarray(
            data["pinn"],
            dtype=np.float64,
        )

        W_tpinn = np.asarray(
            data["tpinn"],
            dtype=np.float64,
        )

        gate_fields = {
            "G": np.asarray(
                data["gate"],
                dtype=np.float64,
            ),

            "C": np.asarray(
                data["C"],
                dtype=np.float64,
            ),

            "Jhat": np.asarray(
                data["Jhat"],
                dtype=np.float64,
            ),

            "h": float(
                data["gate_h"]
            ),

            "cmin": float(
                data["gate_cmin"]
            ),

            "Jbar": float(
                data["gate_Jbar"]
            ),
        }

        cfg_plot = SimpleNamespace(
            x_min=float(
                data["cfg_x_min"]
            ),

            x_max=float(
                data["cfg_x_max"]
            ),

            t_min=float(
                data["cfg_t_min"]
            ),

            t_max=float(
                data["cfg_t_max"]
            ),
        )

        source_seed = int(
            data["source_seed"]
        )

        source_metric = str(
            data[
                "source_metric"
            ].item()
        )


    if (
        W_exact.shape
        != W_pinn.shape
        or W_exact.shape
        != W_tpinn.shape
    ):
        raise ValueError(
            "Heatmap field shapes are inconsistent."
        )


    if W_exact.shape != (
        len(t),
        len(x),
        3,
    ):
        raise ValueError(
            f"Unexpected heatmap shape: "
            f"{W_exact.shape}"
        )


    if not all(
        np.isfinite(
            array
        ).all()
        for array in (
            W_exact,
            W_pinn,
            W_tpinn,
            gate_fields["G"],
        )
    ):
        raise ValueError(
            "Heatmap cache contains "
            "non-finite values."
        )


    solution_cmap = "jet"
    error_cmap = "magma"
    gate_cmap = "jet"


    FIGSIZE_COMPARISON = (
        9.2,
        2.8,
    )

    FIGSIZE_GATE = (
        3.4,
        2.8,
    )


    HEATMAP_LEFT = 0.055
    HEATMAP_RIGHT = 0.935
    HEATMAP_BOTTOM = 0.120
    HEATMAP_TOP = 0.930

    HEATMAP_WSPACE = 0.20
    HEATMAP_HSPACE = -0.20

    COLORBAR_WIDTH = "2.5%" # 5%
    COLORBAR_HEIGHT = "50%" # 92%
    COLORBAR_Y = 0.040

    SOLUTION_COLORBAR_X = 1.080
    ERROR_COLORBAR_X = 1.080

    SOLUTION_COLORBAR_TICK_COUNT = 3
    ERROR_COLORBAR_TICK_COUNT = 4


    plt.rcParams.update({
        "figure.dpi": 140,
        "savefig.dpi": 600,

        "font.family": "serif",

        "font.serif": [
            "Times New Roman",
            "Times",
            "DejaVu Serif",
        ],

        "mathtext.fontset": "stix",

        "font.size": 9.5,
        "axes.titlesize": 8.0,
        "axes.labelsize": 9.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.0,
        "legend.fontsize": 8.5,

        "axes.linewidth": 0.75,
        "axes.spines.top": False,
        "axes.spines.right": False,

        "xtick.direction": "out",
        "ytick.direction": "out",

        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,

        "xtick.major.width": 0.75,
        "ytick.major.width": 0.75,
    })


    def _compact_tick_formatter(
        value,
        pos=None,
    ):
        if np.isclose(
            value,
            round(value),
        ):
            return f"{int(round(value))}"

        return (
            f"{value:.4f}"
            .rstrip("0")
            .rstrip(".")
        )


    def _solution_colorbar_tick_formatter(
        value,
        pos=None,
    ):
        return f"{value:.1f}"


    def _error_colorbar_tick_formatter(
        value,
        pos=None,
    ):
        return f"{value:.2f}"


    def _set_heatmap_axis(
        ax,
        first_panel=False,
    ):
        ax.set_xlim(
            cfg_plot.x_min,
            cfg_plot.x_max,
        )

        ax.set_ylim(
            cfg_plot.t_min,
            cfg_plot.t_max,
        )

        ax.set_xticks(
            np.linspace(
                cfg_plot.x_min,
                cfg_plot.x_max,
                5,
            )
        )

        ax.set_yticks(
            np.linspace(
                cfg_plot.t_min,
                cfg_plot.t_max,
                3,
            )
        )

        ax.xaxis.set_major_formatter(
            FuncFormatter(
                _compact_tick_formatter
            )
        )

        ax.yaxis.set_major_formatter(
            FuncFormatter(
                _compact_tick_formatter
            )
        )

        ax.xaxis.set_ticks_position(
            "bottom"
        )

        ax.yaxis.set_ticks_position(
            "left"
        )

        ax.tick_params(
            axis="both",
            which="both",
            direction="out",

            bottom=True,
            left=True,
            top=False,
            right=False,

            labelbottom=first_panel,
            labelleft=first_panel,

            pad=2.0,
        )

        if first_panel:
            ax.set_xlabel(
                r"$x$"
            )

            ax.set_ylabel(
                r"$t$"
            )

        else:
            ax.set_xlabel("")
            ax.set_ylabel("")


    def _format_colorbar(
        colorbar,
    ):
        colorbar.outline.set_linewidth(
            0.65
        )

        colorbar.ax.tick_params(
            axis="both",
            which="both",
            direction="out",

            length=2.5,
            width=0.7,
            labelsize=8.2,
            pad=2.0,
        )


    def _safe_field_limits(
        fields,
    ):
        vmin = min(
            float(
                np.nanmin(
                    field
                )
            )
            for field in fields
        )

        vmax = max(
            float(
                np.nanmax(
                    field
                )
            )
            for field in fields
        )

        if (
            not np.isfinite(vmin)
            or not np.isfinite(vmax)
        ):
            raise ValueError(
                "Non-finite value detected "
                "in solution fields."
            )

        if np.isclose(
            vmin,
            vmax,
        ):
            epsilon = (
                1.0e-8
                if np.isclose(
                    vmin,
                    0.0,
                )
                else 1.0e-6
                * abs(vmin)
            )

            vmin -= epsilon
            vmax += epsilon

        return (
            vmin,
            vmax,
        )


    def _safe_error_vmax(
        error_1,
        error_2,
        percentile=99.5,
    ):
        values = np.r_[
            error_1.ravel(),
            error_2.ravel(),
        ]

        vmax = float(
            np.nanpercentile(
                values,
                percentile,
            )
        )

        if (
            not np.isfinite(vmax)
            or vmax <= 0.0
        ):
            vmax = float(
                np.nanmax(
                    values
                )
            )

        if (
            not np.isfinite(vmax)
            or vmax <= 0.0
        ):
            vmax = 1.0

        return vmax


    def _save_figure(
        fig,
        stem,
    ):
        if not SAVE_FIGURES:
            return

        fig.savefig(
            FIGURE_DIR
            / f"{stem}.pdf"
        )

        fig.savefig(
            FIGURE_DIR
            / f"{stem}.png",
            dpi=600,
        )


    def _variable_symbol(
        name,
    ):
        symbols = {
            "rho": r"\rho",
            "u": r"u",
            "p": r"p",
        }

        return symbols[name]


    def plot_euler_state_comparison(
        name,
        exact,
        pinn,
        tpinn,
    ):
        symbol = _variable_symbol(
            name
        )

        exact = np.asarray(
            exact
        )

        pinn = np.asarray(
            pinn
        )

        tpinn = np.asarray(
            tpinn
        )

        solution_fields = [
            exact,
            pinn,
            tpinn,
        ]

        solution_vmin, solution_vmax = (
            _safe_field_limits(
                solution_fields
            )
        )

        error_pinn = np.abs(
            pinn
            - exact
        )

        error_tpinn = np.abs(
            tpinn
            - exact
        )

        error_vmax = _safe_error_vmax(
            error_pinn,
            error_tpinn,
            percentile=99.5,
        )

        extent = [
            cfg_plot.x_min,
            cfg_plot.x_max,
            cfg_plot.t_min,
            cfg_plot.t_max,
        ]

        fig = plt.figure(
            figsize=FIGSIZE_COMPARISON,
            constrained_layout=False,
        )

        heatmap_grid = fig.add_gridspec(
            2,
            5,

            left=HEATMAP_LEFT,
            right=HEATMAP_RIGHT,
            bottom=HEATMAP_BOTTOM,
            top=HEATMAP_TOP,

            wspace=HEATMAP_WSPACE,
            hspace=HEATMAP_HSPACE,
        )

        axes_solution = np.asarray(
            [
                fig.add_subplot(
                    heatmap_grid[0, 0]
                ),

                fig.add_subplot(
                    heatmap_grid[0, 2]
                ),

                fig.add_subplot(
                    heatmap_grid[0, 4]
                ),
            ],
            dtype=object,
        )

        axes_error = np.asarray(
            [
                fig.add_subplot(
                    heatmap_grid[1, 1]
                ),

                fig.add_subplot(
                    heatmap_grid[1, 3]
                ),
            ],
            dtype=object,
        )

        solution_titles = [
            rf"(a) Exact (${symbol}$)",
            rf"(b) PINN (${symbol}$)",
            rf"(c) TRG-PINN (${symbol}$)",
        ]

        solution_image = None

        for index, (
            axis,
            field,
            title,
        ) in enumerate(
            zip(
                axes_solution,
                solution_fields,
                solution_titles,
            )
        ):
            solution_image = axis.imshow(
                field,

                extent=extent,

                origin="lower",
                aspect="auto",

                cmap=solution_cmap,
                vmin=solution_vmin,
                vmax=solution_vmax,

                interpolation="nearest",
                rasterized=True,
            )

            axis.set_title(
                title,
                pad=5,
            )

            _set_heatmap_axis(
                axis,
                first_panel=(
                    index == 0
                ),
            )

        solution_colorbar_axis = inset_axes(
            axes_solution[-1],

            width=COLORBAR_WIDTH,
            height=COLORBAR_HEIGHT,

            loc="lower left",

            bbox_to_anchor=(
                SOLUTION_COLORBAR_X,
                COLORBAR_Y,
                1.0,
                1.0,
            ),

            bbox_transform=(
                axes_solution[-1].transAxes
            ),

            borderpad=0.0,
        )

        solution_colorbar = fig.colorbar(
            solution_image,
            cax=solution_colorbar_axis,
        )

        solution_colorbar.set_ticks(
            np.linspace(
                solution_vmin,
                solution_vmax,
                SOLUTION_COLORBAR_TICK_COUNT,
            )
        )

        solution_colorbar.formatter = (
            FuncFormatter(
                _solution_colorbar_tick_formatter
            )
        )

        solution_colorbar.update_ticks()

        _format_colorbar(
            solution_colorbar
        )

        error_titles = [
            rf"(d) PINN error (${symbol}$)",
            rf"(e) TRG-PINN error (${symbol}$)",
        ]

        error_image = None

        for index, (
            axis,
            field,
            title,
        ) in enumerate(
            zip(
                axes_error,

                [
                    error_pinn,
                    error_tpinn,
                ],

                error_titles,
            )
        ):
            error_image = axis.imshow(
                field,

                extent=extent,

                origin="lower",
                aspect="auto",

                cmap=error_cmap,
                vmin=0.0,
                vmax=error_vmax,

                interpolation="nearest",
                rasterized=True,
            )

            axis.set_title(
                title,
                pad=5,
            )

            _set_heatmap_axis(
                axis,
                first_panel=False,
            )

        error_colorbar_axis = inset_axes(
            axes_error[-1],

            width=COLORBAR_WIDTH,
            height=COLORBAR_HEIGHT,

            loc="lower left",

            bbox_to_anchor=(
                ERROR_COLORBAR_X,
                COLORBAR_Y,
                1.0,
                1.0,
            ),

            bbox_transform=(
                axes_error[-1].transAxes
            ),

            borderpad=0.0,
        )

        error_colorbar = fig.colorbar(
            error_image,
            cax=error_colorbar_axis,
        )

        error_colorbar.set_ticks(
            np.linspace(
                0.0,
                error_vmax,
                ERROR_COLORBAR_TICK_COUNT,
            )
        )

        error_colorbar.formatter = (
            FuncFormatter(
                _error_colorbar_tick_formatter
            )
        )

        error_colorbar.update_ticks()

        _format_colorbar(
            error_colorbar
        )

        _save_figure(
            fig,
            f"fig_euler_sod_"
            f"{name}_solution_comparison",
        )

        plt.close('all')

        return (
            fig,
            axes_solution,
            axes_error,
        )


    def plot_euler_gate_map():
        extent = [
            cfg_plot.x_min,
            cfg_plot.x_max,
            cfg_plot.t_min,
            cfg_plot.t_max,
        ]

        fig, axis = plt.subplots(
            1,
            1,
            figsize=FIGSIZE_GATE,
            constrained_layout=True,
        )

        image = axis.imshow(
            gate_fields["G"],

            extent=extent,
            origin="lower",
            aspect="auto",

            cmap=gate_cmap,
            vmin=0.0,
            vmax=1.0,

            interpolation="nearest",
            rasterized=True,
        )

        axis.set_title(
            "Trace-ratio gate",
            pad=5,
        )

        _set_heatmap_axis(
            axis,
            first_panel=True,
        )

        colorbar = fig.colorbar(
            image,

            ax=axis,

            location="right",
            shrink=0.92,
            pad=0.03,
            fraction=0.055,
        )

        colorbar.set_label(
            r"$g$"
        )

        colorbar.set_ticks(
            [
                0.0,
                0.5,
                1.0,
            ]
        )

        _format_colorbar(
            colorbar
        )

        _save_figure(
            fig,
            "fig_euler_sod_trace_ratio_gate",
        )

        plt.close('all')

        return (
            fig,
            axis,
        )


    fig_rho, axes_rho_solution, axes_rho_error = (
        plot_euler_state_comparison(
            "rho",

            W_exact[:, :, 0],
            W_pinn[:, :, 0],
            W_tpinn[:, :, 0],
        )
    )


    fig_u, axes_u_solution, axes_u_error = (
        plot_euler_state_comparison(
            "u",

            W_exact[:, :, 1],
            W_pinn[:, :, 1],
            W_tpinn[:, :, 1],
        )
    )


    fig_p, axes_p_solution, axes_p_error = (
        plot_euler_state_comparison(
            "p",

            W_exact[:, :, 2],
            W_pinn[:, :, 2],
            W_tpinn[:, :, 2],
        )
    )


    fig_gate, ax_gate = (
        plot_euler_gate_map()
    )


    print(
        "[OK] 1D Euler heatmaps saved"
    )

    print(
        "Source seed  :",
        source_seed,
    )

    print(
        "Source metric:",
        source_metric,
    )

    print(
        "Cache        :",
        HEATMAP_CACHE_PATH,
    )

    print(
        "Figure dir   :",
        FIGURE_DIR,
    )

    return [FIGURE_DIR / name for name in EXPECTED_OUTPUT_NAMES]


def render_euler_line(cache_path, figure_dir):
    EXPECTED_OUTPUT_NAMES = ['euler1d_multiseed_nozoom.pdf', 'euler1d_multiseed_nozoom.png']
    # euler fig line

    from pathlib import Path
    from types import SimpleNamespace

    import numpy as np
    import matplotlib.pyplot as plt

    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch


    SAVE_FIGURES = True
    LINE_CACHE_PATH = Path(cache_path)
    FIGURE_DIR = Path(figure_dir)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    with np.load(
        LINE_CACHE_PATH,
        allow_pickle=False,
    ) as data:

        x_euler = np.asarray(
            data["x_euler"],
            dtype=np.float64,
        )

        times_euler = np.asarray(
            data["times_euler"],
            dtype=np.float64,
        )

        seeds_euler_line = np.asarray(
            data["seeds"],
            dtype=int,
        )

        exact = np.asarray(
            data["exact"],
            dtype=np.float64,
        )

        stats_euler = {
            "exact": exact,
            "reference": exact,

            "pinn_median": np.asarray(
                data["pinn_median"],
                dtype=np.float64,
            ),

            "pinn_lower": np.asarray(
                data["pinn_lower"],
                dtype=np.float64,
            ),

            "pinn_upper": np.asarray(
                data["pinn_upper"],
                dtype=np.float64,
            ),

            "tpinn_median": np.asarray(
                data["tpinn_median"],
                dtype=np.float64,
            ),

            "tpinn_lower": np.asarray(
                data["tpinn_lower"],
                dtype=np.float64,
            ),

            "tpinn_upper": np.asarray(
                data["tpinn_upper"],
                dtype=np.float64,
            ),
        }

        cfg_plot_euler = SimpleNamespace(
            x_min=float(
                data["cfg_x_min"]
            ),

            x_max=float(
                data["cfg_x_max"]
            ),

            t_min=float(
                data["cfg_t_min"]
            ),

            t_max=float(
                data["cfg_t_max"]
            ),
        )

        median_seed_euler = int(
            data["median_seed"]
        )

        median_metric_euler = str(
            data[
                "median_metric"
            ].item()
        )

    expected_shape = (
        len(times_euler),
        len(x_euler),
        3,
    )

    for name, values in (
        stats_euler.items()
    ):
        if (
            np.asarray(
                values
            ).shape
            != expected_shape
        ):
            raise ValueError(
                f"Unexpected shape for "
                f"{name}: "
                f"{np.asarray(values).shape}"
            )

        if not np.isfinite(
            values
        ).all():
            raise ValueError(
                f"Non-finite values in {name}"
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
    PINN_LINE_WIDTH = 1.5
    TPINN_LINE_WIDTH = 2.00

    BAND_EDGE_WIDTH = 0.50

    PINN_EDGE_ALPHA = 0.68
    TPINN_EDGE_ALPHA = 0.72

    GRID_ALPHA = 0.13
    GRID_LINE_WIDTH = 0.45

    EULER_TIMES_TO_SHOW = (
        0.10,
        0.15,
        0.20,
    )

    EULER_FIGSIZE = (
        7.9,
        3.0,
    )


    def euler_reference_array(
        stats,
    ):
        if "reference" in stats:
            return np.asarray(
                stats["reference"]
            )

        if "exact" in stats:
            return np.asarray(
                stats["exact"]
            )

        raise KeyError(
            "stats_euler requires "
            "'reference' or 'exact'."
        )


    def nearest_time_indices(
        available_times,
        target_times,
    ):
        available_times = np.asarray(
            available_times,
            dtype=float,
        )

        return [
            int(
                np.argmin(
                    np.abs(
                        available_times
                        - float(
                            target_time
                        )
                    )
                )
            )
            for target_time
            in target_times
        ]


    def euler_row_limits(
        stats,
        time_indices,
        variable_index,
    ):
        reference = (
            euler_reference_array(
                stats
            )
        )

        values = []

        for time_index in time_indices:
            values.extend([
                reference[
                    time_index,
                    :,
                    variable_index,
                ],

                stats[
                    "pinn_lower"
                ][
                    time_index,
                    :,
                    variable_index,
                ],

                stats[
                    "pinn_upper"
                ][
                    time_index,
                    :,
                    variable_index,
                ],

                stats[
                    "tpinn_lower"
                ][
                    time_index,
                    :,
                    variable_index,
                ],

                stats[
                    "tpinn_upper"
                ][
                    time_index,
                    :,
                    variable_index,
                ],
            ])

        y_min = min(
            np.nanmin(
                value
            )
            for value in values
        )

        y_max = max(
            np.nanmax(
                value
            )
            for value in values
        )

        padding = (
            0.06
            * max(
                y_max
                - y_min,
                1.0e-12,
            )
        )

        return (
            y_min
            - padding,

            y_max
            + padding,
        )


    def draw_euler_panel(
        axis,
        x,
        stats,
        time_index,
        variable_index,
    ):
        reference = (
            euler_reference_array(
                stats
            )[
                time_index,
                :,
                variable_index,
            ]
        )

        pinn_median = stats[
            "pinn_median"
        ][
            time_index,
            :,
            variable_index,
        ]

        tpinn_median = stats[
            "tpinn_median"
        ][
            time_index,
            :,
            variable_index,
        ]

        pinn_lower = stats[
            "pinn_lower"
        ][
            time_index,
            :,
            variable_index,
        ]

        pinn_upper = stats[
            "pinn_upper"
        ][
            time_index,
            :,
            variable_index,
        ]

        tpinn_lower = stats[
            "tpinn_lower"
        ][
            time_index,
            :,
            variable_index,
        ]

        tpinn_upper = stats[
            "tpinn_upper"
        ][
            time_index,
            :,
            variable_index,
        ]

        axis.fill_between(
            x,
            pinn_lower,
            pinn_upper,

            color=PINN_COLOR,
            alpha=PINN_BAND_ALPHA,

            linewidth=0,
            zorder=1,
        )

        axis.fill_between(
            x,
            tpinn_lower,
            tpinn_upper,

            color=TPINN_COLOR,
            alpha=TPINN_BAND_ALPHA,

            linewidth=0,
            zorder=2,
        )

        axis.plot(
            x,
            pinn_lower,

            color=PINN_COLOR,
            lw=BAND_EDGE_WIDTH,
            ls=":",

            alpha=PINN_EDGE_ALPHA,
            zorder=3,
        )

        axis.plot(
            x,
            pinn_upper,

            color=PINN_COLOR,
            lw=BAND_EDGE_WIDTH,
            ls=":",

            alpha=PINN_EDGE_ALPHA,
            zorder=3,
        )

        axis.plot(
            x,
            tpinn_lower,

            color=TPINN_COLOR,
            lw=BAND_EDGE_WIDTH,
            ls=":",

            alpha=TPINN_EDGE_ALPHA,
            zorder=4,
        )

        axis.plot(
            x,
            tpinn_upper,

            color=TPINN_COLOR,
            lw=BAND_EDGE_WIDTH,
            ls=":",

            alpha=TPINN_EDGE_ALPHA,
            zorder=4,
        )

        axis.plot(
            x,
            reference,

            color=EXACT_COLOR,
            lw=EXACT_LINE_WIDTH,
            ls="-",

            zorder=7,
        )

        axis.plot(
            x,
            pinn_median,

            color=PINN_COLOR,
            lw=PINN_LINE_WIDTH,
            ls="--",

            zorder=8,
        )

        axis.plot(
            x,
            tpinn_median,

            color=TPINN_COLOR,
            lw=TPINN_LINE_WIDTH,
            ls="--",

            zorder=9,
        )


    def plot_euler_1d_no_zoom(
        times_to_show=EULER_TIMES_TO_SHOW,
    ):
        x = np.asarray(
            x_euler
        )

        available_times = np.asarray(
            times_euler,
            dtype=float,
        )

        time_indices = nearest_time_indices(
            available_times,
            times_to_show,
        )

        actual_times = available_times[
            time_indices
        ]

        variables = [
            {
                "index": 0,
                "ylabel": r"$\rho(x,t)$",
            },

            {
                "index": 1,
                "ylabel": r"$u(x,t)$",
            },

            {
                "index": 2,
                "ylabel": r"$p(x,t)$",
            },
        ]

        figure, axes = plt.subplots(
            3,
            len(time_indices),

            figsize=EULER_FIGSIZE,

            sharex=True,
            sharey="row",

            constrained_layout=False,
        )

        axes = np.asarray(
            axes
        )

        x_ticks = np.linspace(
            cfg_plot_euler.x_min,
            cfg_plot_euler.x_max,
            5,
        )

        for row_index, variable in enumerate(
            variables
        ):
            y_limits = euler_row_limits(
                stats_euler,
                time_indices,
                variable["index"],
            )

            for column_index, (
                time_index,
                time_value,
            ) in enumerate(
                zip(
                    time_indices,
                    actual_times,
                )
            ):
                axis = axes[
                    row_index,
                    column_index,
                ]

                draw_euler_panel(
                    axis,
                    x,
                    stats_euler,
                    time_index,
                    variable["index"],
                )

                axis.set_xlim(
                    cfg_plot_euler.x_min,
                    cfg_plot_euler.x_max,
                )

                axis.set_ylim(
                    *y_limits
                )

                axis.set_xticks(
                    x_ticks
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

                if row_index == 0:
                    axis.set_title(
                        rf"$t={time_value:.2f}$",
                        pad=6,
                    )

                if column_index == 0:
                    axis.set_ylabel(
                        variable["ylabel"],
                        labelpad=3,
                    )

                else:
                    axis.tick_params(
                        axis="y",
                        which="both",

                        left=False,
                        labelleft=False,
                    )

                show_x_labels = (
                    row_index == 2
                    and column_index == 0
                )

                show_x_tick_marks_only = (
                    row_index == 2
                    and column_index > 0
                )

                if show_x_labels:
                    axis.set_xlabel(
                        r"$x$",
                        labelpad=2,
                    )

                    axis.tick_params(
                        axis="x",
                        which="both",

                        bottom=True,
                        labelbottom=True,
                    )

                elif show_x_tick_marks_only:
                    axis.set_xlabel(
                        ""
                    )

                    axis.tick_params(
                        axis="x",
                        which="both",

                        bottom=True,
                        labelbottom=False,
                    )

                else:
                    axis.set_xlabel(
                        ""
                    )

                    axis.tick_params(
                        axis="x",
                        which="both",

                        bottom=False,
                        labelbottom=False,
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

            bbox_to_anchor=(
                0.5,
                0.995,
            ),

            handlelength=2.15,
            columnspacing=1.35,
            handletextpad=0.50,
        )

        plt.subplots_adjust(
            left=0.075,
            right=0.995,

            bottom=0.090,
            top=0.80,

            wspace=0.0,
            hspace=0.0,
        )

        pdf_path = (
            FIGURE_DIR
            / "euler1d_multiseed_nozoom.pdf"
        )

        png_path = (
            FIGURE_DIR
            / "euler1d_multiseed_nozoom.png"
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

        plt.close('all')

        print(
            "Saved:",
            pdf_path,
        )

        print(
            "Saved:",
            png_path,
        )

        return (
            figure,
            axes,
        )


    EULER_1D_FIGURE, EULER_1D_AXES = (
        plot_euler_1d_no_zoom()
    )

    print(
        "[OK] 1D Euler line figure saved"
    )

    print(
        "Seeds        :",
        seeds_euler_line.tolist(),
    )

    print(
        "Median seed  :",
        median_seed_euler,
    )

    print(
        "Median metric:",
        median_metric_euler,
    )

    print(
        "Cache        :",
        LINE_CACHE_PATH,
    )

    print(
        "Figure dir   :",
        FIGURE_DIR,
    )

    return [FIGURE_DIR / name for name in EXPECTED_OUTPUT_NAMES]

def _load_shallowwater_reported_model(
    method: str,
    seed: int,
    *,
    device: str = "cpu",
):
    from trgpinn.equations.shallowwater_1d import (
        ShallowWater1DConfig,
        build_model,
    )
    from trgpinn.utils import (
        configure_torch_runtime,
        load_checkpoint_into_model,
        read_json,
        resolve_device,
    )

    run_dir = (
        REPO_ROOT
        / "artifacts"
        / "reported"
        / "1d_shallowwater"
        / method
        / f"seed_{int(seed)}"
    )
    payload = read_json(run_dir / "config.json")
    cfg = ShallowWater1DConfig.from_legacy_mapping(
        payload.get("config", payload),
        seed=int(seed),
        device=device,
    )
    torch_device = resolve_device(device)
    dtype = configure_torch_runtime(cfg.dtype)
    model = build_model(cfg).to(
        device=torch_device,
        dtype=dtype,
    )
    checkpoint_path = run_dir / "model_final.pt"
    load_checkpoint_into_model(model, checkpoint_path)
    model.eval()
    return model, cfg, checkpoint_path


@torch.no_grad()
def _predict_shallowwater_figure_points(
    model: torch.nn.Module,
    x_values,
    t_values,
    *,
    batch_size: int = 65536,
) -> np.ndarray:
    """Canonical CPU inference path used by the frozen manuscript caches."""

    x_values = np.asarray(
        x_values,
        dtype=np.float64,
    ).reshape(-1)
    t_values = np.asarray(
        t_values,
        dtype=np.float64,
    ).reshape(-1)
    if x_values.shape != t_values.shape:
        raise ValueError(
            "x_values and t_values must have the same shape."
        )

    parameter = next(model.parameters())
    outputs = []
    for start in range(0, x_values.size, int(batch_size)):
        stop = min(start + int(batch_size), x_values.size)
        coordinates = torch.as_tensor(
            np.column_stack(
                [
                    x_values[start:stop],
                    t_values[start:stop],
                ]
            ),
            device=parameter.device,
            dtype=parameter.dtype,
        )
        outputs.append(
            model(coordinates).detach().cpu().numpy()
        )
    return np.concatenate(outputs, axis=0)


def _select_shallowwater_representative_seed(
    master: pd.DataFrame,
) -> tuple[int, pd.DataFrame]:
    primary_metric = "state_scaled_space_time_rel_l2"
    expected_seeds = [2026, 7, 42, 100, 31415]
    rows = master[
        master["equation"].astype(str).eq("1d_shallowwater")
        & master["method"].astype(str).eq("Ours")
    ].copy()
    rows["seed"] = pd.to_numeric(
        rows["seed"],
        errors="raise",
    ).astype(int)
    rows[primary_metric] = pd.to_numeric(
        rows[primary_metric],
        errors="raise",
    )
    seed_order = {
        seed: index
        for index, seed in enumerate(expected_seeds)
    }
    rows = rows[rows["seed"].isin(expected_seeds)]
    rows["seed_order"] = rows["seed"].map(seed_order)
    rows = rows.sort_values(
        [primary_metric, "seed_order"],
        kind="mergesort",
    ).reset_index(drop=True)
    if len(rows) != 5:
        raise RuntimeError(
            "Expected five TRG-PINN 1D shallow-water rows."
        )
    return int(rows.iloc[len(rows) // 2]["seed"]), rows


def build_shallowwater_caches(
    cache_dir: Path,
    *,
    device: str = "cpu",
) -> tuple[Path, Path, int]:
    """Rebuild exact-Stoker line and field caches from frozen checkpoints."""

    from trgpinn.equations.shallowwater_1d import stoker_exact_hq

    if str(device) != "cpu":
        raise ValueError(
            "The frozen 1D shallow-water manuscript caches were generated "
            "through the canonical CPU inference path."
        )

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    master_path = (
        REPO_ROOT
        / "results"
        / "reported_metrics"
        / "all_metrics_final.csv"
    )
    master = pd.read_csv(master_path)
    expected_seeds = [2026, 7, 42, 100, 31415]
    primary_metric = "state_scaled_space_time_rel_l2"
    final_metric = "state_scaled_final_rel_l2"
    representative_seed, ours_rank = (
        _select_shallowwater_representative_seed(master)
    )

    methods = {
        "PINN": "pinn",
        "Ours": "trg_pinn",
    }
    configs = {}
    checkpoint_paths = []
    checkpoint_hashes = []

    # Validate the same scientific configuration for all paired runs.
    reference_signature = None
    for legacy_method, public_method in methods.items():
        for seed in expected_seeds:
            model, cfg, checkpoint_path = (
                _load_shallowwater_reported_model(
                    public_method,
                    seed,
                    device=device,
                )
            )
            signature = (
                float(cfg.x_min),
                float(cfg.x_max),
                float(cfg.t_min),
                float(cfg.t_max),
                float(cfg.g_const),
                float(cfg.hL),
                float(cfg.qL),
                float(cfg.hR),
                float(cfg.qR),
                float(cfg.x0),
                int(cfg.width),
                int(cfg.depth),
                str(cfg.activation).lower(),
                float(cfg.h_floor),
                int(cfg.warmup_iters),
                int(cfg.gated_iters),
                int(cfg.n_f),
                int(cfg.n_ic),
                int(cfg.n_bc),
                float(cfg.w_ic),
                float(cfg.w_bc),
                float(cfg.w_pde),
                float(cfg.h_min_factor),
                float(cfg.h_max_factor),
                float(cfg.cmin_start),
                float(cfg.cmin_end),
                float(cfg.beta),
                float(cfg.residual_floor),
                int(cfg.eval_nx),
                int(cfg.eval_nt),
                float(cfg.t_final_plot),
            )
            if reference_signature is None:
                reference_signature = signature
            elif signature != reference_signature:
                raise AssertionError(
                    "Scientific configuration mismatch: "
                    f"{legacy_method}, seed={seed}"
                )
            configs[(legacy_method, seed)] = cfg
            checkpoint_paths.append(
                str(checkpoint_path.resolve())
            )
            checkpoint_hashes.append(
                sha256_file(checkpoint_path)
            )
            del model
            gc.collect()

    cfg_plot = configs[("Ours", representative_seed)]
    line_times = np.asarray(
        [0.0625, 0.125, 0.1875, 0.25],
        dtype=np.float64,
    )
    line_nx = 500
    x_line = np.linspace(
        cfg_plot.x_min,
        cfg_plot.x_max,
        line_nx,
        dtype=np.float64,
    )
    X_line, T_line = np.meshgrid(
        x_line,
        line_times,
        indexing="xy",
    )
    H_line, Q_line, stoker_info = stoker_exact_hq(
        X_line,
        T_line,
        cfg_plot,
    )
    reference_line = np.stack(
        [H_line, Q_line],
        axis=-1,
    )

    line_predictions = {
        "PINN": [],
        "Ours": [],
    }
    line_checkpoint_paths = []
    line_checkpoint_hashes = []
    for legacy_method, public_method in methods.items():
        for seed in expected_seeds:
            model, _, checkpoint_path = (
                _load_shallowwater_reported_model(
                    public_method,
                    seed,
                    device=device,
                )
            )
            prediction = _predict_shallowwater_figure_points(
                model,
                X_line.reshape(-1),
                T_line.reshape(-1),
            ).reshape(
                len(line_times),
                line_nx,
                2,
            )
            if not np.isfinite(prediction).all():
                raise RuntimeError(
                    "Non-finite line prediction: "
                    f"{legacy_method}, seed={seed}"
                )
            line_predictions[legacy_method].append(prediction)
            line_checkpoint_paths.append(
                str(checkpoint_path.resolve())
            )
            line_checkpoint_hashes.append(
                sha256_file(checkpoint_path)
            )
            del model
            gc.collect()

    pinn_all = np.asarray(
        line_predictions["PINN"],
        dtype=np.float64,
    )
    trg_all = np.asarray(
        line_predictions["Ours"],
        dtype=np.float64,
    )
    pinn_median = np.median(pinn_all, axis=0)
    trg_median = np.median(trg_all, axis=0)
    pinn_rmse = np.sqrt(
        np.mean(
            (pinn_all - reference_line[None, ...]) ** 2,
            axis=0,
        )
    )
    trg_rmse = np.sqrt(
        np.mean(
            (trg_all - reference_line[None, ...]) ** 2,
            axis=0,
        )
    )

    line_cache_path = (
        cache_dir
        / "shallowwater_multiseed_line_benchmark_exact_stoker.npz"
    )
    np.savez_compressed(
        line_cache_path,
        x_sw=x_line,
        times_sw=line_times,
        seeds=np.asarray(expected_seeds, dtype=np.int64),
        reference=reference_line.astype(np.float32),
        pinn_all=pinn_all.astype(np.float32),
        tpinn_all=trg_all.astype(np.float32),
        pinn_median=pinn_median.astype(np.float32),
        tpinn_median=trg_median.astype(np.float32),
        pinn_rmse=pinn_rmse.astype(np.float32),
        tpinn_rmse=trg_rmse.astype(np.float32),
        pinn_lower=(pinn_median - pinn_rmse).astype(np.float32),
        pinn_upper=(pinn_median + pinn_rmse).astype(np.float32),
        tpinn_lower=(trg_median - trg_rmse).astype(np.float32),
        tpinn_upper=(trg_median + trg_rmse).astype(np.float32),
        selected_seed=np.int64(representative_seed),
        median_mode=np.asarray("primary_metric_median"),
        primary_metric=np.asarray(primary_metric),
        final_metric=np.asarray(final_metric),
        rank_seed=ours_rank["seed"].to_numpy(dtype=np.int64),
        rank_score=ours_rank[primary_metric].to_numpy(
            dtype=np.float64
        ),
        checkpoint_paths=np.asarray(line_checkpoint_paths),
        checkpoint_sha256=np.asarray(line_checkpoint_hashes),
        source_csv=np.asarray(str(master_path.resolve())),
        reference_type=np.asarray(
            "exact_stoker_entropy_solution"
        ),
        cfg_x_min=np.float64(cfg_plot.x_min),
        cfg_x_max=np.float64(cfg_plot.x_max),
        cfg_t_min=np.float64(cfg_plot.t_min),
        cfg_t_max=np.float64(cfg_plot.t_max),
        cfg_hL=np.float64(cfg_plot.hL),
        cfg_qL=np.float64(cfg_plot.qL),
        cfg_hR=np.float64(cfg_plot.hR),
        cfg_qR=np.float64(cfg_plot.qR),
        cfg_x0=np.float64(cfg_plot.x0),
        cfg_g_const=np.float64(cfg_plot.g_const),
        stoker_h_star=np.float64(stoker_info["h_star"]),
        stoker_u_star=np.float64(stoker_info["u_star"]),
        stoker_q_star=np.float64(stoker_info["q_star"]),
        generated_at=np.asarray(
            datetime.now().isoformat(timespec="seconds")
        ),
    )

    pinn_model, _, pinn_checkpoint = (
        _load_shallowwater_reported_model(
            "pinn",
            representative_seed,
            device=device,
        )
    )
    trg_model, _, trg_checkpoint = (
        _load_shallowwater_reported_model(
            "trg_pinn",
            representative_seed,
            device=device,
        )
    )

    heatmap_nx = 501
    heatmap_nt = 251
    x_heatmap = np.linspace(
        cfg_plot.x_min,
        cfg_plot.x_max,
        heatmap_nx,
        dtype=np.float64,
    )
    t_heatmap = np.linspace(
        cfg_plot.t_min,
        cfg_plot.t_max,
        heatmap_nt,
        dtype=np.float64,
    )
    X_heatmap, T_heatmap = np.meshgrid(
        x_heatmap,
        t_heatmap,
        indexing="xy",
    )
    H_heatmap, Q_heatmap, heatmap_info = stoker_exact_hq(
        X_heatmap,
        T_heatmap,
        cfg_plot,
    )
    reference_heatmap = np.stack(
        [H_heatmap, Q_heatmap],
        axis=-1,
    )
    pinn_heatmap = _predict_shallowwater_figure_points(
        pinn_model,
        X_heatmap.reshape(-1),
        T_heatmap.reshape(-1),
    ).reshape(
        heatmap_nt,
        heatmap_nx,
        2,
    )
    trg_heatmap = _predict_shallowwater_figure_points(
        trg_model,
        X_heatmap.reshape(-1),
        T_heatmap.reshape(-1),
    ).reshape(
        heatmap_nt,
        heatmap_nx,
        2,
    )

    heatmap_cache_path = (
        cache_dir
        / "shallowwater_median_seed_heatmap_benchmark_exact_stoker.npz"
    )
    np.savez_compressed(
        heatmap_cache_path,
        x=x_heatmap,
        t=t_heatmap,
        H_exact=reference_heatmap[:, :, 0].astype(np.float32),
        Q_exact=reference_heatmap[:, :, 1].astype(np.float32),
        H_pinn=pinn_heatmap[:, :, 0].astype(np.float32),
        Q_pinn=pinn_heatmap[:, :, 1].astype(np.float32),
        H_tpinn=trg_heatmap[:, :, 0].astype(np.float32),
        Q_tpinn=trg_heatmap[:, :, 1].astype(np.float32),
        EH_pinn=np.abs(
            reference_heatmap[:, :, 0]
            - pinn_heatmap[:, :, 0]
        ).astype(np.float32),
        EH_tpinn=np.abs(
            reference_heatmap[:, :, 0]
            - trg_heatmap[:, :, 0]
        ).astype(np.float32),
        EQ_pinn=np.abs(
            reference_heatmap[:, :, 1]
            - pinn_heatmap[:, :, 1]
        ).astype(np.float32),
        EQ_tpinn=np.abs(
            reference_heatmap[:, :, 1]
            - trg_heatmap[:, :, 1]
        ).astype(np.float32),
        x_min=np.float64(cfg_plot.x_min),
        x_max=np.float64(cfg_plot.x_max),
        t_min=np.float64(cfg_plot.t_min),
        t_max=np.float64(cfg_plot.t_max),
        selected_seed=np.int64(representative_seed),
        median_mode=np.asarray("primary_metric_median"),
        primary_metric=np.asarray(primary_metric),
        final_metric=np.asarray(final_metric),
        rank_seed=ours_rank["seed"].to_numpy(dtype=np.int64),
        rank_score=ours_rank[primary_metric].to_numpy(
            dtype=np.float64
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
        source_csv=np.asarray(str(master_path.resolve())),
        reference_type=np.asarray(
            "exact_stoker_entropy_solution"
        ),
        stoker_h_star=np.float64(heatmap_info["h_star"]),
        stoker_u_star=np.float64(heatmap_info["u_star"]),
        stoker_q_star=np.float64(heatmap_info["q_star"]),
        nx_plot=np.int64(heatmap_nx),
        nt_plot=np.int64(heatmap_nt),
        generated_at=np.asarray(
            datetime.now().isoformat(timespec="seconds")
        ),
    )

    del pinn_model
    del trg_model
    gc.collect()

    return line_cache_path, heatmap_cache_path, representative_seed

def render_shallowwater_heatmaps(
    heatmap_cache_path: Path,
    figure_dir: Path,
) -> list[Path]:
    """Render the canonical 1D shallow-water field/error figures."""

    PROJECT_ROOT = REPO_ROOT
    HEATMAP_CACHE_PATH = Path(heatmap_cache_path)
    ALL_METRICS_PATH = (
        REPO_ROOT / "results" / "reported_metrics" / "all_metrics_final.csv"
    )
    FIGURE_DIR = Path(figure_dir)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    PRIMARY_METRIC = "state_scaled_space_time_rel_l2"
    MEDIAN_MODE = "primary_metric_median"
    EXPECTED_SEEDS = [2026, 7, 42, 100, 31415]
    SAVE_FIGURES = True
    display = lambda value: None

    with np.load(
        HEATMAP_CACHE_PATH,
        allow_pickle=False,
    ) as data:

        required_keys = {
            'x',
            't',
            'H_exact',
            'Q_exact',
            'H_pinn',
            'Q_pinn',
            'H_tpinn',
            'Q_tpinn',
            'EH_pinn',
            'EH_tpinn',
            'EQ_pinn',
            'EQ_tpinn',
            'x_min',
            'x_max',
            't_min',
            't_max',
            'selected_seed',
            'median_mode',
            'primary_metric',
            'rank_seed',
            'rank_score',
            'reference_type',
            'pinn_checkpoint',
            'tpinn_checkpoint',
            'source_csv',
        }

        missing_keys = (
            required_keys
            - set(data.files)
        )

        if missing_keys:
            raise KeyError(
                'Missing heatmap-cache keys: '
                f'{sorted(missing_keys)}'
            )

        x = np.asarray(
            data['x'],
            dtype=np.float64,
        )

        t = np.asarray(
            data['t'],
            dtype=np.float64,
        )

        H_exact = np.asarray(
            data['H_exact'],
            dtype=np.float64,
        )

        Q_exact = np.asarray(
            data['Q_exact'],
            dtype=np.float64,
        )

        H_pinn = np.asarray(
            data['H_pinn'],
            dtype=np.float64,
        )

        Q_pinn = np.asarray(
            data['Q_pinn'],
            dtype=np.float64,
        )

        H_tpinn = np.asarray(
            data['H_tpinn'],
            dtype=np.float64,
        )

        Q_tpinn = np.asarray(
            data['Q_tpinn'],
            dtype=np.float64,
        )

        EH_pinn = np.asarray(
            data['EH_pinn'],
            dtype=np.float64,
        )

        EH_tpinn = np.asarray(
            data['EH_tpinn'],
            dtype=np.float64,
        )

        EQ_pinn = np.asarray(
            data['EQ_pinn'],
            dtype=np.float64,
        )

        EQ_tpinn = np.asarray(
            data['EQ_tpinn'],
            dtype=np.float64,
        )

        x_min = float(
            np.asarray(
                data['x_min']
            ).item()
        )

        x_max = float(
            np.asarray(
                data['x_max']
            ).item()
        )

        t_min = float(
            np.asarray(
                data['t_min']
            ).item()
        )

        t_max = float(
            np.asarray(
                data['t_max']
            ).item()
        )

        selected_seed = int(
            np.asarray(
                data['selected_seed']
            ).item()
        )

        median_mode = str(
            np.asarray(
                data['median_mode']
            ).item()
        )

        primary_metric = str(
            np.asarray(
                data['primary_metric']
            ).item()
        )

        reference_type = str(
            np.asarray(
                data['reference_type']
            ).item()
        )

        pinn_checkpoint = Path(
            str(
                np.asarray(
                    data['pinn_checkpoint']
                ).item()
            )
        )

        tpinn_checkpoint = Path(
            str(
                np.asarray(
                    data['tpinn_checkpoint']
                ).item()
            )
        )

        cached_source_csv = Path(
            str(
                np.asarray(
                    data['source_csv']
                ).item()
            )
        ).resolve()

        rank_table = pd.DataFrame(
            {
                'seed': np.asarray(
                    data['rank_seed'],
                    dtype=int,
                ),

                primary_metric: np.asarray(
                    data['rank_score'],
                    dtype=np.float64,
                ),
            }
        )


    if primary_metric != PRIMARY_METRIC:
        raise AssertionError(
            'Heatmap primary metric mismatch: '
            f'cache={primary_metric}, '
            f'expected={PRIMARY_METRIC}'
        )


    if median_mode != MEDIAN_MODE:
        raise AssertionError(
            'Heatmap median mode mismatch: '
            f'cache={median_mode}, '
            f'expected={MEDIAN_MODE}'
        )


    if reference_type != 'exact_stoker_entropy_solution':
        raise AssertionError(
            'Heatmap reference is not '
            'the exact Stoker solution.'
        )


    if cached_source_csv != ALL_METRICS_PATH.resolve():
        raise AssertionError(
            'Heatmap source CSV mismatch.'
        )


    ranking = pd.read_csv(
        ALL_METRICS_PATH
    )


    ranking = ranking[
        ranking[
            'equation'
        ]
        .astype(str)
        .eq('1d_shallowwater')
        &
        ranking[
            'method'
        ]
        .astype(str)
        .eq('Ours')
    ].copy()


    ranking['seed'] = pd.to_numeric(
        ranking['seed'],
        errors='raise',
    ).astype(int)


    ranking[PRIMARY_METRIC] = pd.to_numeric(
        ranking[PRIMARY_METRIC],
        errors='raise',
    )


    seed_order = {
        seed: index
        for index, seed
        in enumerate(EXPECTED_SEEDS)
    }


    ranking['seed_order'] = (
        ranking['seed']
        .map(seed_order)
    )


    ranking = (
        ranking[
            ranking[
                'seed'
            ].isin(EXPECTED_SEEDS)
        ]
        .drop_duplicates(
            ['seed'],
            keep='last',
        )
        .sort_values(
            [
                PRIMARY_METRIC,
                'seed_order',
            ],
            kind='mergesort',
        )
        .reset_index(drop=True)
    )


    if len(ranking) != 5:
        raise RuntimeError(
            'Expected five Ours seeds, '
            f'found {len(ranking)}.'
        )


    expected_seed = int(
        ranking.iloc[
            len(ranking) // 2
        ]['seed']
    )


    if selected_seed != expected_seed:
        raise AssertionError(
            'Heatmap representative seed mismatch: '
            f'cache={selected_seed}, '
            f'expected={expected_seed}'
        )


    if not pinn_checkpoint.is_file():
        raise FileNotFoundError(
            pinn_checkpoint
        )


    if not tpinn_checkpoint.is_file():
        raise FileNotFoundError(
            tpinn_checkpoint
        )


    expected_shape = (
        len(t),
        len(x),
    )


    fields = {
        'H_exact': H_exact,
        'Q_exact': Q_exact,
        'H_pinn': H_pinn,
        'Q_pinn': Q_pinn,
        'H_tpinn': H_tpinn,
        'Q_tpinn': Q_tpinn,
        'EH_pinn': EH_pinn,
        'EH_tpinn': EH_tpinn,
        'EQ_pinn': EQ_pinn,
        'EQ_tpinn': EQ_tpinn,
    }


    for name, values in fields.items():
        if values.shape != expected_shape:
            raise ValueError(
                f'Unexpected shape for {name}: '
                f'{values.shape}; '
                f'expected {expected_shape}.'
            )

        if not np.isfinite(values).all():
            raise ValueError(
                f'Non-finite values in {name}.'
            )


    if not np.allclose(
        EH_pinn,
        np.abs(
            H_exact
            - H_pinn
        ),
        rtol=2.0e-5,
        atol=2.0e-6,
    ):
        raise AssertionError(
            'PINN h absolute-error cache mismatch.'
        )


    if not np.allclose(
        EH_tpinn,
        np.abs(
            H_exact
            - H_tpinn
        ),
        rtol=2.0e-5,
        atol=2.0e-6,
    ):
        raise AssertionError(
            'tPINN h absolute-error cache mismatch.'
        )


    if not np.allclose(
        EQ_pinn,
        np.abs(
            Q_exact
            - Q_pinn
        ),
        rtol=2.0e-5,
        atol=2.0e-6,
    ):
        raise AssertionError(
            'PINN q absolute-error cache mismatch.'
        )


    if not np.allclose(
        EQ_tpinn,
        np.abs(
            Q_exact
            - Q_tpinn
        ),
        rtol=2.0e-5,
        atol=2.0e-6,
    ):
        raise AssertionError(
            'tPINN q absolute-error cache mismatch.'
        )


    solution_cmap = 'jet'
    error_cmap = 'magma'


    FIGSIZE_COMPARISON = (
        9.2,
        2.8,
    )


    HEATMAP_LEFT = 0.055
    HEATMAP_RIGHT = 0.935
    HEATMAP_BOTTOM = 0.120
    HEATMAP_TOP = 0.930

    HEATMAP_WSPACE = 0.20
    HEATMAP_HSPACE = -0.20

    COLORBAR_WIDTH = '2.5%'
    COLORBAR_HEIGHT = '50%'
    COLORBAR_Y = 0.040

    SOLUTION_COLORBAR_X = 1.080
    ERROR_COLORBAR_X = 1.080

    SOLUTION_COLORBAR_TICK_COUNT = 3
    ERROR_COLORBAR_TICK_COUNT = 4


    plt.rcParams.update(
        {
            'figure.dpi': 140,
            'savefig.dpi': 600,

            'font.family': 'serif',

            'font.serif': [
                'Times New Roman',
                'Times',
                'DejaVu Serif',
            ],

            'mathtext.fontset': 'stix',

            'font.size': 9.5,
            'axes.titlesize': 8.0,
            'axes.labelsize': 9.5,

            'xtick.labelsize': 8.5,
            'ytick.labelsize': 8.0,

            'legend.fontsize': 8.5,

            'axes.linewidth': 0.75,

            'axes.spines.top': False,
            'axes.spines.right': False,

            'xtick.direction': 'out',
            'ytick.direction': 'out',

            'xtick.major.size': 3.0,
            'ytick.major.size': 3.0,

            'xtick.major.width': 0.75,
            'ytick.major.width': 0.75,
        }
    )


    def compact_tick(
        value,
        pos=None,
    ):
        if np.isclose(
            value,
            round(value),
        ):
            return f'{int(round(value))}'

        return (
            f'{value:.4f}'
            .rstrip('0')
            .rstrip('.')
        )


    def solution_colorbar_tick(
        value,
        pos=None,
    ):
        return f'{value:.1f}'


    def error_colorbar_tick(
        value,
        pos=None,
    ):
        return f'{value:.2f}'


    def set_axis(
        axis,
        first_panel=False,
    ):
        axis.set_xlim(
            x_min,
            x_max,
        )

        axis.set_ylim(
            t_min,
            t_max,
        )

        axis.set_xticks(
            np.linspace(
                x_min,
                x_max,
                5,
            )
        )

        axis.set_yticks(
            np.linspace(
                t_min,
                t_max,
                3,
            )
        )

        axis.xaxis.set_major_formatter(
            FuncFormatter(
                compact_tick
            )
        )

        axis.yaxis.set_major_formatter(
            FuncFormatter(
                compact_tick
            )
        )

        axis.xaxis.set_ticks_position(
            'bottom'
        )

        axis.yaxis.set_ticks_position(
            'left'
        )

        axis.tick_params(
            axis='both',
            which='both',

            direction='out',

            bottom=True,
            left=True,
            top=False,
            right=False,

            labelbottom=first_panel,
            labelleft=first_panel,

            pad=2.0,
        )

        if first_panel:
            axis.set_xlabel(
                r'$x$'
            )

            axis.set_ylabel(
                r'$t$'
            )

        else:
            axis.set_xlabel('')
            axis.set_ylabel('')


    def format_colorbar(
        colorbar,
    ):
        colorbar.outline.set_linewidth(
            0.65
        )

        colorbar.ax.tick_params(
            axis='both',
            which='both',

            direction='out',

            length=2.5,
            width=0.7,

            labelsize=8.2,
            pad=2.0,
        )


    def save_figure(
        figure,
        stem,
    ):
        if not SAVE_FIGURES:
            return

        figure.savefig(
            FIGURE_DIR
            / f'{stem}.pdf'
        )

        figure.savefig(
            FIGURE_DIR
            / f'{stem}.png',
            dpi=600,
        )


    def variable_pack(
        variable,
    ):
        if variable == 'h':
            return (
                H_exact,
                H_pinn,
                H_tpinn,
                EH_pinn,
                EH_tpinn,
                r'h',
                'h',
            )

        if variable == 'q':
            return (
                Q_exact,
                Q_pinn,
                Q_tpinn,
                EQ_pinn,
                EQ_tpinn,
                r'q',
                'q',
            )

        raise ValueError(
            "variable must be 'h' or 'q'."
        )


    def plot_state_comparison(
        variable,
    ):
        (
            exact,
            pinn,
            tpinn,
            pinn_error,
            tpinn_error,
            symbol,
            name,
        ) = variable_pack(
            variable
        )

        solution_fields = [
            np.asarray(exact),
            np.asarray(pinn),
            np.asarray(tpinn),
        ]

        solution_vmin = min(
            float(
                np.nanmin(values)
            )
            for values in solution_fields
        )

        solution_vmax = max(
            float(
                np.nanmax(values)
            )
            for values in solution_fields
        )

        if (
            not np.isfinite(solution_vmin)
            or not np.isfinite(solution_vmax)
        ):
            raise ValueError(
                'Non-finite value detected '
                'in solution fields.'
            )

        if np.isclose(
            solution_vmin,
            solution_vmax,
        ):
            epsilon = (
                1.0e-8
                if np.isclose(
                    solution_vmin,
                    0.0,
                )
                else 1.0e-6
                * abs(solution_vmin)
            )

            solution_vmin -= epsilon
            solution_vmax += epsilon

        error_vmax = float(
            np.percentile(
                np.r_[
                    pinn_error.ravel(),
                    tpinn_error.ravel(),
                ],
                99.5,
            )
        )

        if (
            not np.isfinite(error_vmax)
            or error_vmax <= 0.0
        ):
            error_vmax = float(
                max(
                    np.nanmax(pinn_error),
                    np.nanmax(tpinn_error),
                )
            )

        if (
            not np.isfinite(error_vmax)
            or error_vmax <= 0.0
        ):
            error_vmax = 1.0

        extent = [
            x_min,
            x_max,
            t_min,
            t_max,
        ]

        figure = plt.figure(
            figsize=FIGSIZE_COMPARISON,
            constrained_layout=False,
        )

        heatmap_grid = figure.add_gridspec(
            2,
            5,

            left=HEATMAP_LEFT,
            right=HEATMAP_RIGHT,
            bottom=HEATMAP_BOTTOM,
            top=HEATMAP_TOP,

            wspace=HEATMAP_WSPACE,
            hspace=HEATMAP_HSPACE,
        )

        solution_axes = np.asarray(
            [
                figure.add_subplot(
                    heatmap_grid[0, 0]
                ),

                figure.add_subplot(
                    heatmap_grid[0, 2]
                ),

                figure.add_subplot(
                    heatmap_grid[0, 4]
                ),
            ],
            dtype=object,
        )

        error_axes = np.asarray(
            [
                figure.add_subplot(
                    heatmap_grid[1, 1]
                ),

                figure.add_subplot(
                    heatmap_grid[1, 3]
                ),
            ],
            dtype=object,
        )

        solution_titles = [
            rf'(a) Exact (${symbol}$)',
            rf'(b) PINN (${symbol}$)',
            rf'(c) TRG-PINN (${symbol}$)',
        ]

        solution_image = None

        for index, (
            axis,
            values,
            title,
        ) in enumerate(
            zip(
                solution_axes,
                solution_fields,
                solution_titles,
            )
        ):
            solution_image = axis.imshow(
                values,

                extent=extent,

                origin='lower',
                aspect='auto',

                cmap=solution_cmap,
                vmin=solution_vmin,
                vmax=solution_vmax,

                interpolation='nearest',
                rasterized=True,
            )

            axis.set_title(
                title,
                pad=5,
            )

            set_axis(
                axis,
                first_panel=(
                    index == 0
                ),
            )

        solution_colorbar_axis = inset_axes(
            solution_axes[-1],

            width=COLORBAR_WIDTH,
            height=COLORBAR_HEIGHT,

            loc='lower left',

            bbox_to_anchor=(
                SOLUTION_COLORBAR_X,
                COLORBAR_Y,
                1.0,
                1.0,
            ),

            bbox_transform=(
                solution_axes[-1].transAxes
            ),

            borderpad=0.0,
        )

        solution_colorbar = figure.colorbar(
            solution_image,
            cax=solution_colorbar_axis,
        )

        solution_colorbar.set_ticks(
            np.linspace(
                solution_vmin,
                solution_vmax,
                SOLUTION_COLORBAR_TICK_COUNT,
            )
        )

        solution_colorbar.formatter = (
            FuncFormatter(
                solution_colorbar_tick
            )
        )

        solution_colorbar.update_ticks()

        format_colorbar(
            solution_colorbar
        )

        error_titles = [
            rf'(d) PINN error (${symbol}$)',
            rf'(e) TRG-PINN error (${symbol}$)',
        ]

        error_image = None

        for index, (
            axis,
            values,
            title,
        ) in enumerate(
            zip(
                error_axes,

                [
                    pinn_error,
                    tpinn_error,
                ],

                error_titles,
            )
        ):
            error_image = axis.imshow(
                values,

                extent=extent,

                origin='lower',
                aspect='auto',

                cmap=error_cmap,
                vmin=0.0,
                vmax=error_vmax,

                interpolation='nearest',
                rasterized=True,
            )

            axis.set_title(
                title,
                pad=5,
            )

            set_axis(
                axis,
                first_panel=False,
            )

        error_colorbar_axis = inset_axes(
            error_axes[-1],

            width=COLORBAR_WIDTH,
            height=COLORBAR_HEIGHT,

            loc='lower left',

            bbox_to_anchor=(
                ERROR_COLORBAR_X,
                COLORBAR_Y,
                1.0,
                1.0,
            ),

            bbox_transform=(
                error_axes[-1].transAxes
            ),

            borderpad=0.0,
        )

        error_colorbar = figure.colorbar(
            error_image,
            cax=error_colorbar_axis,
        )

        error_colorbar.set_ticks(
            np.linspace(
                0.0,
                error_vmax,
                ERROR_COLORBAR_TICK_COUNT,
            )
        )

        error_colorbar.formatter = (
            FuncFormatter(
                error_colorbar_tick
            )
        )

        error_colorbar.update_ticks()

        format_colorbar(
            error_colorbar
        )

        save_figure(
            figure,
            f'shallow_water_{name}_solution_comparison',
        )

        plt.close(figure)

        return (
            figure,
            solution_axes,
            error_axes,
        )


    fig_h, ax_h_solution, ax_h_error = (
        plot_state_comparison(
            'h'
        )
    )


    fig_q, ax_q_solution, ax_q_error = (
        plot_state_comparison(
            'q'
        )
    )


    expected_files = [
        FIGURE_DIR
        / 'shallow_water_h_solution_comparison.pdf',

        FIGURE_DIR
        / 'shallow_water_h_solution_comparison.png',

        FIGURE_DIR
        / 'shallow_water_q_solution_comparison.pdf',

        FIGURE_DIR
        / 'shallow_water_q_solution_comparison.png',
    ]


    if SAVE_FIGURES:
        missing_files = [
            str(path)

            for path in expected_files

            if (
                not path.is_file()
                or path.stat().st_size == 0
            )
        ]

        if missing_files:
            raise FileNotFoundError(
                'Missing or empty heatmap figures:\n'
                + '\n'.join(
                    missing_files
                )
            )


    print(
        '[OK] 1D shallow-water heatmaps saved'
    )


    print(
        'Reference           :',
        reference_type,
    )


    print(
        'Primary metric      :',
        primary_metric,
    )


    print(
        'Median mode         :',
        median_mode,
    )


    print(
        'Representative seed :',
        selected_seed,
    )


    print(
        'PINN checkpoint     :',
        pinn_checkpoint,
    )


    print(
        'tPINN checkpoint    :',
        tpinn_checkpoint,
    )


    print(
        'Cache               :',
        HEATMAP_CACHE_PATH,
    )


    print(
        'Figure directory    :',
        FIGURE_DIR,
    )


    display(
        rank_table
    )

    return expected_files

def render_shallowwater_line(
    line_cache_path: Path,
    figure_dir: Path,
) -> list[Path]:
    """Render the canonical 1D shallow-water median/RMSE line figure."""

    from types import SimpleNamespace

    PROJECT_ROOT = REPO_ROOT
    LINE_CACHE_PATH = Path(line_cache_path)
    FIGURE_DIR = Path(figure_dir)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    PRIMARY_METRIC = "state_scaled_space_time_rel_l2"
    MEDIAN_MODE = "primary_metric_median"
    EXPECTED_SEEDS = [2026, 7, 42, 100, 31415]
    SAVE_FIGURES = True

    with np.load(
        LINE_CACHE_PATH,
        allow_pickle=False,
    ) as data:
        required_keys = {
            'x_sw',
            'times_sw',
            'seeds',
            'reference',
            'pinn_median',
            'tpinn_median',
            'pinn_rmse',
            'tpinn_rmse',
            'pinn_lower',
            'pinn_upper',
            'tpinn_lower',
            'tpinn_upper',
            'selected_seed',
            'median_mode',
            'primary_metric',
            'cfg_x_min',
            'cfg_x_max',
        }

        missing_keys = (
            required_keys
            - set(data.files)
        )

        if missing_keys:
            raise KeyError(
                'Missing line-cache keys: '
                f'{sorted(missing_keys)}'
            )

        x_sw = np.asarray(
            data['x_sw'],
            dtype=np.float64,
        )

        times_sw = np.asarray(
            data['times_sw'],
            dtype=np.float64,
        )

        seeds_sw = np.asarray(
            data['seeds'],
            dtype=int,
        )

        reference_sw = np.asarray(
            data['reference'],
            dtype=np.float64,
        )

        stats_sw = {
            'reference': reference_sw,
            'exact': reference_sw,

            'pinn_median': np.asarray(
                data['pinn_median'],
                dtype=np.float64,
            ),

            'tpinn_median': np.asarray(
                data['tpinn_median'],
                dtype=np.float64,
            ),

            'pinn_rmse': np.asarray(
                data['pinn_rmse'],
                dtype=np.float64,
            ),

            'tpinn_rmse': np.asarray(
                data['tpinn_rmse'],
                dtype=np.float64,
            ),

            'pinn_lower': np.asarray(
                data['pinn_lower'],
                dtype=np.float64,
            ),

            'pinn_upper': np.asarray(
                data['pinn_upper'],
                dtype=np.float64,
            ),

            'tpinn_lower': np.asarray(
                data['tpinn_lower'],
                dtype=np.float64,
            ),

            'tpinn_upper': np.asarray(
                data['tpinn_upper'],
                dtype=np.float64,
            ),
        }

        cfg_plot_sw = SimpleNamespace(
            x_min=float(
                np.asarray(
                    data['cfg_x_min']
                ).item()
            ),
            x_max=float(
                np.asarray(
                    data['cfg_x_max']
                ).item()
            ),
        )

        selected_seed_sw = int(
            np.asarray(
                data['selected_seed']
            ).item()
        )

        median_mode_sw = str(
            np.asarray(
                data['median_mode']
            ).item()
        )

        primary_metric_sw = str(
            np.asarray(
                data['primary_metric']
            ).item()
        )

    if seeds_sw.tolist() != EXPECTED_SEEDS:
        raise AssertionError(
            'Line seed order mismatch: '
            f'cache={seeds_sw.tolist()}, '
            f'expected={EXPECTED_SEEDS}'
        )

    if primary_metric_sw != PRIMARY_METRIC:
        raise AssertionError(
            'Line primary metric mismatch: '
            f'cache={primary_metric_sw}, '
            f'expected={PRIMARY_METRIC}'
        )

    if median_mode_sw != MEDIAN_MODE:
        raise AssertionError(
            'Line median mode mismatch: '
            f'cache={median_mode_sw}, '
            f'expected={MEDIAN_MODE}'
        )

    expected_shape = (
        len(times_sw),
        len(x_sw),
        2,
    )

    for name, values in stats_sw.items():
        if np.asarray(values).shape != expected_shape:
            raise ValueError(
                f'Unexpected shape for {name}: '
                f'{np.asarray(values).shape}; '
                f'expected {expected_shape}.'
            )

        if not np.isfinite(values).all():
            raise ValueError(
                f'Non-finite values in {name}.'
            )

    plt.rcParams.update(
        {
            'figure.dpi': 160,
            'savefig.dpi': 300,
            'font.family': 'serif',
            'mathtext.fontset': 'stix',
            'font.size': 9.5,
            'axes.titlesize': 9.5,
            'axes.labelsize': 7.8,
            'legend.fontsize': 7.0,
            'xtick.labelsize': 6.8,
            'ytick.labelsize': 6.8,
            'axes.spines.top': False,
            'axes.spines.right': False,
            'axes.linewidth': 0.8,
            'xtick.major.width': 0.75,
            'ytick.major.width': 0.75,
            'xtick.major.size': 3.5,
            'ytick.major.size': 3.5,
        }
    )

    EXACT_COLOR = 'black'
    PINN_COLOR = '#2C7BB6'
    TPINN_COLOR = '#D7191C'

    PINN_BAND_ALPHA = 0.13
    TPINN_BAND_ALPHA = 0.16

    EXACT_LINE_WIDTH = 1.50
    PINN_LINE_WIDTH = 1.50
    TPINN_LINE_WIDTH = 2.00

    BAND_EDGE_WIDTH = 0.50
    PINN_EDGE_ALPHA = 0.68
    TPINN_EDGE_ALPHA = 0.72

    GRID_ALPHA = 0.13
    GRID_LINE_WIDTH = 0.45

    SHALLOW_REFERENCE_LABEL = 'Exact'

    SHALLOW_TIMES_TO_SHOW = (
        0.12,
        0.18,
        0.25,
    )

    SHALLOW_FIGSIZE = (
        7.9,
        3.0,
    )

    SHALLOW_Q_ZERO_FRACTION_FROM_BOTTOM = 0.06

    SHALLOW_ZOOM_SEARCH_WINDOWS = {
        0.12: (
            0.105,
            0.195,
        ),
        0.18: (
            0.190,
            0.310,
        ),
        0.25: (
            0.270,
            0.390,
        ),
    }

    SHALLOW_H_INSET_PAIR_POSITIONS = {
        0.12: {
            'upper': [
                0.67,
                0.58,
                0.35,
                0.35,
            ],
            'lower': [
                0.67,
                0.16,
                0.35,
                0.35,
            ],
        },
        0.18: {
            'upper': [
                0.67,
                0.58,
                0.35,
                0.35,
            ],
            'lower': [
                0.67,
                0.16,
                0.35,
                0.35,
            ],
        },
        0.25: {
            'upper': [
                0.67,
                0.58,
                0.35,
                0.35,
            ],
            'lower': [
                0.67,
                0.16,
                0.35,
                0.35,
            ],
        },
    }

    SHALLOW_Q_INSET_PAIR_POSITIONS = {
        0.12: {
            'upper': [
                0.67,
                0.58,
                0.35,
                0.35,
            ],
            'lower': [
                0.67,
                0.16,
                0.35,
                0.35,
            ],
        },
        0.18: {
            'upper': [
                0.67,
                0.58,
                0.35,
                0.35,
            ],
            'lower': [
                0.67,
                0.16,
                0.35,
                0.35,
            ],
        },
        0.25: {
            'upper': [
                0.67,
                0.58,
                0.35,
                0.35,
            ],
            'lower': [
                0.67,
                0.16,
                0.35,
                0.35,
            ],
        },
    }

    SHALLOW_BOX_CONTROLS = {
        'h': {
            0.12: {
                'upper': {
                    'x_half': 0.05,
                    'x_offset': -0.03,
                    'y_offset': -0.04,
                },
                'lower': {
                    'x_half': 0.05,
                    'x_offset': 0.03,
                    'y_offset': 0.05,
                },
            },
            0.18: {
                'upper': {
                    'x_half': 0.05,
                    'x_offset': -0.03,
                    'y_offset': -0.04,
                },
                'lower': {
                    'x_half': 0.05,
                    'x_offset': 0.026,
                    'y_offset': 0.05,
                },
            },
            0.25: {
                'upper': {
                    'x_half': 0.05,
                    'x_offset': -0.03,
                    'y_offset': -0.04,
                },
                'lower': {
                    'x_half': 0.05,
                    'x_offset': 0.03,
                    'y_offset': 0.05,
                },
            },
        },
        'q': {
            0.12: {
                'upper': {
                    'x_half': 0.05,
                    'x_offset': -0.03,
                    'y_offset': -0.035,
                },
                'lower': {
                    'x_half': 0.05,
                    'x_offset': 0.03,
                    'y_offset': 0.04,
                },
            },
            0.18: {
                'upper': {
                    'x_half': 0.05,
                    'x_offset': -0.03,
                    'y_offset': -0.035,
                },
                'lower': {
                    'x_half': 0.05,
                    'x_offset': 0.03,
                    'y_offset': 0.04,
                },
            },
            0.25: {
                'upper': {
                    'x_half': 0.05,
                    'x_offset': -0.03,
                    'y_offset': -0.035,
                },
                'lower': {
                    'x_half': 0.05,
                    'x_offset': 0.03,
                    'y_offset': 0.04,
                },
            },
        },
    }

    SHALLOW_STATE_SAMPLE_MULTIPLIER_INNER = 1.5
    SHALLOW_STATE_SAMPLE_MULTIPLIER_OUTER = 3.5


    def reference_array(stats):
        if 'reference' in stats:
            return np.asarray(
                stats['reference']
            )

        if 'exact' in stats:
            return np.asarray(
                stats['exact']
            )

        raise KeyError(
            "stats_sw requires 'reference' or 'exact'."
        )


    def nearest_time_indices(
        available_times,
        target_times,
    ):
        available_times = np.asarray(
            available_times,
            dtype=float,
        )

        return [
            int(
                np.argmin(
                    np.abs(
                        available_times
                        - float(target_time)
                    )
                )
            )
            for target_time in target_times
        ]


    def component_statistics(
        stats,
        component_index,
    ):
        reference = reference_array(
            stats
        )

        result = {
            'reference': reference[
                :,
                :,
                component_index,
            ]
        }

        for key in (
            'pinn_median',
            'pinn_lower',
            'pinn_upper',
            'tpinn_median',
            'tpinn_lower',
            'tpinn_upper',
        ):
            result[key] = stats[key][
                :,
                :,
                component_index,
            ]

        return result


    def component_limits(
        component_stats,
        time_indices,
    ):
        values = []

        for index in time_indices:
            values.extend(
                [
                    component_stats[
                        'reference'
                    ][index],

                    component_stats[
                        'pinn_lower'
                    ][index],

                    component_stats[
                        'pinn_upper'
                    ][index],

                    component_stats[
                        'tpinn_lower'
                    ][index],

                    component_stats[
                        'tpinn_upper'
                    ][index],
                ]
            )

        y_min = min(
            np.nanmin(value)
            for value in values
        )

        y_max = max(
            np.nanmax(value)
            for value in values
        )

        padding = (
            0.06
            * max(
                y_max - y_min,
                1.0e-12,
            )
        )

        return (
            y_min - padding,
            y_max + padding,
        )


    def q_display_limits(
        q_limits,
        zero_fraction=SHALLOW_Q_ZERO_FRACTION_FROM_BOTTOM,
    ):
        q_ymax = float(
            q_limits[1]
        )

        if (
            not np.isfinite(q_ymax)
            or q_ymax <= 0.0
        ):
            return q_limits

        zero_fraction = np.clip(
            float(zero_fraction),
            0.001,
            0.35,
        )

        q_ymin = (
            -zero_fraction
            * q_ymax
            / (
                1.0
                - zero_fraction
            )
        )

        return (
            q_ymin,
            q_ymax,
        )


    def draw_curves(
        axis,
        x,
        component_stats,
        time_index,
        inset=False,
    ):
        reference = component_stats[
            'reference'
        ][time_index]

        pinn_median = component_stats[
            'pinn_median'
        ][time_index]

        tpinn_median = component_stats[
            'tpinn_median'
        ][time_index]

        pinn_lower = component_stats[
            'pinn_lower'
        ][time_index]

        pinn_upper = component_stats[
            'pinn_upper'
        ][time_index]

        tpinn_lower = component_stats[
            'tpinn_lower'
        ][time_index]

        tpinn_upper = component_stats[
            'tpinn_upper'
        ][time_index]

        axis.fill_between(
            x,
            pinn_lower,
            pinn_upper,
            color=PINN_COLOR,
            alpha=PINN_BAND_ALPHA,
            linewidth=0,
            zorder=1,
        )

        axis.fill_between(
            x,
            tpinn_lower,
            tpinn_upper,
            color=TPINN_COLOR,
            alpha=TPINN_BAND_ALPHA,
            linewidth=0,
            zorder=2,
        )

        axis.plot(
            x,
            pinn_lower,
            color=PINN_COLOR,
            lw=BAND_EDGE_WIDTH,
            ls=':',
            alpha=PINN_EDGE_ALPHA,
            zorder=3,
        )

        axis.plot(
            x,
            pinn_upper,
            color=PINN_COLOR,
            lw=BAND_EDGE_WIDTH,
            ls=':',
            alpha=PINN_EDGE_ALPHA,
            zorder=3,
        )

        axis.plot(
            x,
            tpinn_lower,
            color=TPINN_COLOR,
            lw=BAND_EDGE_WIDTH,
            ls=':',
            alpha=TPINN_EDGE_ALPHA,
            zorder=4,
        )

        axis.plot(
            x,
            tpinn_upper,
            color=TPINN_COLOR,
            lw=BAND_EDGE_WIDTH,
            ls=':',
            alpha=TPINN_EDGE_ALPHA,
            zorder=4,
        )

        scale = (
            0.72
            if inset
            else 1.0
        )

        axis.plot(
            x,
            reference,
            color=EXACT_COLOR,
            lw=(
                EXACT_LINE_WIDTH
                * scale
            ),
            ls='-',
            zorder=7,
        )

        axis.plot(
            x,
            pinn_median,
            color=PINN_COLOR,
            lw=(
                PINN_LINE_WIDTH
                * scale
            ),
            ls='--',
            zorder=8,
        )

        axis.plot(
            x,
            tpinn_median,
            color=TPINN_COLOR,
            lw=(
                TPINN_LINE_WIDTH
                * scale
            ),
            ls='--',
            zorder=9,
        )


    def detect_shock_x(
        x,
        h_reference,
        search_window,
    ):
        mask = (
            (x >= search_window[0])
            & (x <= search_window[1])
        )

        if not np.any(mask):
            raise ValueError(
                f'Search window {search_window} '
                'does not overlap the x grid.'
            )

        gradient = np.gradient(
            h_reference,
            x,
        )

        return float(
            x[mask][
                np.argmin(
                    gradient[mask]
                )
            ]
        )


    def sample_states(
        x,
        y,
        shock_x,
        x_half,
    ):
        inner = (
            SHALLOW_STATE_SAMPLE_MULTIPLIER_INNER
            * x_half
        )

        outer = (
            SHALLOW_STATE_SAMPLE_MULTIPLIER_OUTER
            * x_half
        )

        left_mask = (
            (x >= shock_x - outer)
            & (x <= shock_x - inner)
        )

        right_mask = (
            (x >= shock_x + inner)
            & (x <= shock_x + outer)
        )

        if np.any(left_mask):
            left_value = float(
                np.median(
                    y[left_mask]
                )
            )

        else:
            left_value = float(
                y[
                    np.argmin(
                        np.abs(
                            x
                            - (
                                shock_x
                                - outer
                            )
                        )
                    )
                ]
            )

        if np.any(right_mask):
            right_value = float(
                np.median(
                    y[right_mask]
                )
            )

        else:
            right_value = float(
                y[
                    np.argmin(
                        np.abs(
                            x
                            - (
                                shock_x
                                + outer
                            )
                        )
                    )
                ]
            )

        return (
            max(
                left_value,
                right_value,
            ),
            min(
                left_value,
                right_value,
            ),
        )


    def square_y_halfwidth(
        parent_axis,
        x_halfwidth,
    ):
        bbox = (
            parent_axis
            .get_window_extent()
        )

        (
            x_min_axis,
            x_max_axis,
        ) = parent_axis.get_xlim()

        (
            y_min_axis,
            y_max_axis,
        ) = parent_axis.get_ylim()

        pixels_per_x = (
            bbox.width
            / max(
                x_max_axis
                - x_min_axis,
                1.0e-12,
            )
        )

        pixels_per_y = (
            bbox.height
            / max(
                y_max_axis
                - y_min_axis,
                1.0e-12,
            )
        )

        return (
            x_halfwidth
            * pixels_per_x
            / pixels_per_y
        )


    def shift_inside(
        center,
        halfwidth,
        lower,
        upper,
    ):
        low = center - halfwidth
        high = center + halfwidth

        if low < lower:
            high += lower - low
            low = lower

        if high > upper:
            low -= high - upper
            high = upper

        return low, high


    def square_box_limits(
        parent_axis,
        x,
        component_stats,
        time_index,
        shock_x,
        time_key,
        variable_key,
        corner_key,
    ):
        control = (
            SHALLOW_BOX_CONTROLS[
                variable_key
            ][
                time_key
            ][
                corner_key
            ]
        )

        x_half = float(
            control['x_half']
        )

        center_x = (
            shock_x
            + float(
                control.get(
                    'x_offset',
                    0.0,
                )
            )
        )

        x_limits = (
            center_x - x_half,
            center_x + x_half,
        )

        (
            upper_state,
            lower_state,
        ) = sample_states(
            x,
            component_stats[
                'reference'
            ][time_index],
            shock_x,
            x_half,
        )

        center_y = (
            upper_state
            if corner_key == 'upper'
            else lower_state
        )

        center_y += float(
            control.get(
                'y_offset',
                0.0,
            )
        )

        y_half = square_y_halfwidth(
            parent_axis,
            x_half,
        )

        y_limits = shift_inside(
            center_y,
            y_half,
            *parent_axis.get_ylim(),
        )

        return x_limits, y_limits


    def add_square_inset(
        parent_axis,
        x,
        component_stats,
        time_index,
        x_limits,
        y_limits,
        position,
    ):
        inset_axis = parent_axis.inset_axes(
            position
        )

        draw_curves(
            inset_axis,
            x,
            component_stats,
            time_index,
            inset=True,
        )

        inset_axis.set_xlim(
            *x_limits
        )

        inset_axis.set_ylim(
            *y_limits
        )

        inset_axis.set_box_aspect(
            1.0
        )

        inset_axis.set_xticks([])
        inset_axis.set_yticks([])

        inset_axis.set_xlabel('')
        inset_axis.set_ylabel('')
        inset_axis.set_title('')

        inset_axis.tick_params(
            axis='both',
            which='both',
            bottom=False,
            top=False,
            left=False,
            right=False,
            labelbottom=False,
            labelleft=False,
        )

        inset_axis.grid(False)

        inset_axis.set_facecolor(
            'white'
        )

        inset_axis.patch.set_alpha(
            0.96
        )

        for spine in (
            inset_axis
            .spines
            .values()
        ):
            spine.set_visible(True)
            spine.set_linewidth(0.75)
            spine.set_edgecolor('0.35')

        try:
            parent_axis.indicate_inset_zoom(
                inset_axis,
                edgecolor='0.55',
                alpha=0.80,
                linewidth=0.75,
            )

        except Exception:
            pass

        return inset_axis


    def plot_shallow_water_1d_two_square_insets(
        times_to_show=SHALLOW_TIMES_TO_SHOW,
        search_windows=SHALLOW_ZOOM_SEARCH_WINDOWS,
        h_inset_pair_positions=SHALLOW_H_INSET_PAIR_POSITIONS,
        q_inset_pair_positions=SHALLOW_Q_INSET_PAIR_POSITIONS,
    ):
        x = np.asarray(
            x_sw
        )

        available_times = np.asarray(
            times_sw,
            dtype=float,
        )

        time_indices = nearest_time_indices(
            available_times,
            times_to_show,
        )

        actual_times = available_times[
            time_indices
        ]

        h_stats = component_statistics(
            stats_sw,
            0,
        )

        q_stats = component_statistics(
            stats_sw,
            1,
        )

        h_limits = component_limits(
            h_stats,
            time_indices,
        )

        q_limits = component_limits(
            q_stats,
            time_indices,
        )

        q_main_limits = q_display_limits(
            q_limits
        )

        figure, axes = plt.subplots(
            2,
            len(time_indices),
            figsize=SHALLOW_FIGSIZE,
            sharex=True,
            sharey='row',
            constrained_layout=False,
        )

        axes = np.asarray(
            axes
        )

        x_ticks = np.linspace(
            cfg_plot_sw.x_min,
            cfg_plot_sw.x_max,
            5,
        )

        inset_jobs = []

        for column, (
            requested_time,
            time_index,
            actual_time,
        ) in enumerate(
            zip(
                times_to_show,
                time_indices,
                actual_times,
            )
        ):
            time_key = float(
                requested_time
            )

            h_axis = axes[
                0,
                column,
            ]

            q_axis = axes[
                1,
                column,
            ]

            draw_curves(
                h_axis,
                x,
                h_stats,
                time_index,
            )

            draw_curves(
                q_axis,
                x,
                q_stats,
                time_index,
            )

            h_axis.set_xlim(
                cfg_plot_sw.x_min,
                cfg_plot_sw.x_max,
            )

            q_axis.set_xlim(
                cfg_plot_sw.x_min,
                cfg_plot_sw.x_max,
            )

            h_axis.set_ylim(
                *h_limits
            )

            q_axis.set_ylim(
                *q_limits
            )

            h_axis.set_xticks(
                x_ticks
            )

            q_axis.set_xticks(
                x_ticks
            )

            h_axis.grid(
                alpha=GRID_ALPHA,
                linewidth=GRID_LINE_WIDTH,
            )

            q_axis.grid(
                alpha=GRID_ALPHA,
                linewidth=GRID_LINE_WIDTH,
            )

            h_axis.tick_params(
                axis='both',
                which='major',
                pad=2,
            )

            q_axis.tick_params(
                axis='both',
                which='major',
                pad=2,
            )

            h_axis.set_title(
                f'$t={actual_time}$', # actual_time:.2f
                pad=6,
            )

            if column == 0:
                h_axis.set_ylabel(
                    '$h(x,t)$',
                    labelpad=3,
                )

                q_axis.set_ylabel(
                    '$q(x,t)$',
                    labelpad=3,
                )

                q_axis.set_xlabel(
                    '$x$',
                    labelpad=2,
                )

                q_axis.tick_params(
                    axis='x',
                    which='both',
                    bottom=True,
                    labelbottom=True,
                )

            else:
                h_axis.tick_params(
                    axis='y',
                    which='both',
                    left=False,
                    labelleft=False,
                )

                q_axis.tick_params(
                    axis='y',
                    which='both',
                    left=False,
                    labelleft=False,
                )

                q_axis.tick_params(
                    axis='x',
                    which='both',
                    bottom=True,
                    labelbottom=False,
                )

            h_axis.set_xlabel('')

            h_axis.tick_params(
                axis='x',
                which='both',
                bottom=True,
                labelbottom=False,
            )

            shock_x = detect_shock_x(
                x,
                h_stats[
                    'reference'
                ][time_index],
                search_windows[
                    time_key
                ],
            )

            inset_jobs.extend(
                [
                    (
                        h_axis,
                        h_stats,
                        time_index,
                        shock_x,
                        time_key,
                        'h',
                        'upper',
                        h_inset_pair_positions[
                            time_key
                        ]['upper'],
                    ),
                    (
                        h_axis,
                        h_stats,
                        time_index,
                        shock_x,
                        time_key,
                        'h',
                        'lower',
                        h_inset_pair_positions[
                            time_key
                        ]['lower'],
                    ),
                    (
                        q_axis,
                        q_stats,
                        time_index,
                        shock_x,
                        time_key,
                        'q',
                        'upper',
                        q_inset_pair_positions[
                            time_key
                        ]['upper'],
                    ),
                    (
                        q_axis,
                        q_stats,
                        time_index,
                        shock_x,
                        time_key,
                        'q',
                        'lower',
                        q_inset_pair_positions[
                            time_key
                        ]['lower'],
                    ),
                ]
            )

        legend_handles = [
            Patch(
                facecolor=PINN_COLOR,
                alpha=PINN_BAND_ALPHA,
                edgecolor=PINN_COLOR,
                linewidth=0.7,
                label='PINN ± RMSE',
            ),

            Patch(
                facecolor=TPINN_COLOR,
                alpha=TPINN_BAND_ALPHA,
                edgecolor=TPINN_COLOR,
                linewidth=0.7,
                label='TRG-PINN ± RMSE',
            ),

            Line2D(
                [0],
                [0],
                color=EXACT_COLOR,
                lw=EXACT_LINE_WIDTH,
                ls='-',
                label=SHALLOW_REFERENCE_LABEL,
            ),

            Line2D(
                [0],
                [0],
                color=PINN_COLOR,
                lw=PINN_LINE_WIDTH,
                ls='--',
                label='PINN',
            ),

            Line2D(
                [0],
                [0],
                color=TPINN_COLOR,
                lw=TPINN_LINE_WIDTH,
                ls='--',
                label='TRG-PINN',
            ),
        ]

        figure.legend(
            handles=legend_handles,
            loc='upper center',
            ncol=5,
            frameon=False,
            bbox_to_anchor=(
                0.5,
                0.995,
            ),
            handlelength=2.15,
            columnspacing=1.35,
            handletextpad=0.50,
        )

        plt.subplots_adjust(
            left=0.075,
            right=0.995,
            bottom=0.105,
            top=0.800,
            wspace=0.0,
            hspace=0.0,
        )

        figure.canvas.draw()

        for (
            parent_axis,
            component_stats,
            time_index,
            shock_x,
            time_key,
            variable_key,
            corner_key,
            position,
        ) in inset_jobs:
            (
                x_limits,
                y_limits,
            ) = square_box_limits(
                parent_axis,
                x,
                component_stats,
                time_index,
                shock_x,
                time_key,
                variable_key,
                corner_key,
            )

            add_square_inset(
                parent_axis,
                x,
                component_stats,
                time_index,
                x_limits,
                y_limits,
                position,
            )

        for q_axis in axes[1, :]:
            q_axis.set_ylim(
                *q_main_limits
            )

        pdf_path = (
            FIGURE_DIR
            / 'shallow_water_1d_2x3_two_square_insets.pdf'
        )

        png_path = (
            FIGURE_DIR
            / 'shallow_water_1d_2x3_two_square_insets.png'
        )

        if SAVE_FIGURES:
            figure.savefig(
                pdf_path,
                bbox_inches='tight',
            )

            figure.savefig(
                png_path,
                dpi=300,
                bbox_inches='tight',
            )

        plt.close(figure)

        if SAVE_FIGURES:
            for path in (
                pdf_path,
                png_path,
            ):
                if (
                    not path.is_file()
                    or path.stat().st_size == 0
                ):
                    raise FileNotFoundError(
                        f'Missing or empty figure: {path}'
                    )

        return figure, axes


    SHALLOW_1D_FIGURE, SHALLOW_1D_AXES = (
        plot_shallow_water_1d_two_square_insets()
    )

    print(
        '[OK] 1D shallow-water '
        'line figure saved'
    )

    print(
        'Reference           : '
        'exact_stoker_entropy_solution'
    )

    print(
        'Primary metric      :',
        primary_metric_sw,
    )

    print(
        'Representative seed :',
        selected_seed_sw,
    )

    print(
        'Median mode         :',
        median_mode_sw,
    )

    print(
        'Line seeds          :',
        seeds_sw.tolist(),
    )

    print(
        'Line statistic      : '
        'pointwise median ± RMSE-to-exact'
    )

    print(
        'Cache               :',
        LINE_CACHE_PATH,
    )

    print(
        'Figure directory    :',
        FIGURE_DIR,
    )

    return [
        FIGURE_DIR / "shallow_water_1d_2x3_two_square_insets.pdf",
        FIGURE_DIR / "shallow_water_1d_2x3_two_square_insets.png",
    ]

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmark",
        choices=["burgers_1d", "euler_1d", "shallowwater_1d", "burgers_2d", "euler_2d", "shallowwater_2d"],
        required=True,
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--data-only", action="store_true")
    parser.add_argument(
        "--reference-path",
        type=Path,
        default=None,
        help="Canonical FV1024 reference required for 2D shallow-water figures.",
    )
    args = parser.parse_args()

    paths: list[Path] = []
    if args.benchmark == "burgers_1d":
        cache_dir = (
            REPO_ROOT / "build" / "reproduced_cache" / "1d_burgers"
        )
        figure_dir = (
            REPO_ROOT
            / "results"
            / "reproduced_figures"
            / "1d_burgers"
        )
        line_cache, heatmap_cache, seed = build_caches(cache_dir)
        print(
            "[OK] Rebuilt 1D Burgers figure data from frozen "
            f"checkpoints; representative seed={seed}"
        )
        if not args.data_only:
            paths.extend(render_heatmaps(heatmap_cache, figure_dir))
            paths.extend(render_line_figure(line_cache, figure_dir))

    elif args.benchmark == "euler_1d":
        cache_dir = (
            REPO_ROOT / "build" / "reproduced_cache" / "1d_euler"
        )
        figure_dir = (
            REPO_ROOT
            / "results"
            / "reproduced_figures"
            / "1d_euler"
        )
        line_cache, heatmap_cache, seed = build_euler_caches(
            cache_dir,
            device=args.device,
        )
        print(
            "[OK] Rebuilt 1D Euler figure data from frozen "
            f"checkpoints; representative seed={seed}"
        )
        if not args.data_only:
            paths.extend(
                render_euler_heatmaps(heatmap_cache, figure_dir)
            )
            paths.extend(
                render_euler_line(line_cache, figure_dir)
            )

    elif args.benchmark == "shallowwater_1d":
        cache_dir = (
            REPO_ROOT
            / "build"
            / "reproduced_cache"
            / "1d_shallowwater"
        )
        figure_dir = (
            REPO_ROOT
            / "results"
            / "reproduced_figures"
            / "1d_shallowwater"
        )
        line_cache, heatmap_cache, seed = (
            build_shallowwater_caches(
                cache_dir,
                device=args.device,
            )
        )
        print(
            "[OK] Rebuilt 1D shallow-water figure data from "
            f"frozen checkpoints; representative seed={seed}"
        )
        if not args.data_only:
            paths.extend(
                render_shallowwater_heatmaps(
                    heatmap_cache,
                    figure_dir,
                )
            )
            paths.extend(
                render_shallowwater_line(
                    line_cache,
                    figure_dir,
                )
            )

    elif args.benchmark == "burgers_2d":
        from reproduce_figures_2d_burgers import reproduce as reproduce_burgers_2d

        paths.extend(
            reproduce_burgers_2d(
                device=args.device,
                data_only=args.data_only,
            )
        )

    elif args.benchmark == "euler_2d":
        from reproduce_figures_2d_euler import reproduce as reproduce_euler_2d

        paths.extend(
            reproduce_euler_2d(
                device=args.device,
                data_only=args.data_only,
            )
        )

    else:
        if args.reference_path is None:
            raise ValueError(
                "--reference-path is required for 2D shallow-water figures."
            )
        from reproduce_figures_2d_shallowwater import (
            reproduce as reproduce_shallowwater_2d,
        )

        paths.extend(
            reproduce_shallowwater_2d(
                args.reference_path,
                data_only=args.data_only,
            )
        )

    for path in paths:
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
        print(path)

    print("Training run: NO")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
