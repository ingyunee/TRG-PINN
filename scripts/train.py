#!/usr/bin/env python
"""Fresh-training entry point.

Reported artifacts under ``artifacts/reported`` are immutable. New runs are
written only under ``runs/reproduction``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from trgpinn.equations.burgers_1d import (
    Burgers1DConfig,
    run_paired_experiment as run_burgers_1d,
)
from trgpinn.equations.euler_1d import (
    Euler1DConfig,
    run_paired_experiment as run_euler_1d,
)
from trgpinn.equations.shallowwater_1d import (
    ShallowWater1DConfig,
    run_paired_experiment as run_shallowwater_1d,
)
from trgpinn.equations.burgers_2d import (
    Burgers2DConfig,
    run_paired_experiment as run_burgers_2d,
)
from trgpinn.equations.euler_2d import (
    Euler2DConfig,
    run_paired_experiment as run_euler_2d,
)
from trgpinn.equations.shallowwater_2d import (
    ShallowWater2DConfig,
    run_paired_experiment as run_shallowwater_2d,
)
from trgpinn.utils import load_yaml


CONFIG_FILES = {
    "burgers_1d": "burgers_1d.yaml",
    "euler_1d": "euler_1d.yaml",
    "shallowwater_1d": "shallowwater_1d.yaml",
    "burgers_2d": "burgers_2d.yaml",
    "euler_2d": "euler_2d.yaml",
    "shallowwater_2d": "shallowwater_2d.yaml",
}


def _common(payload: dict[str, Any], *, seed: int, device: str) -> dict[str, Any]:
    network = payload["network"]
    training = payload["training"]
    sampling = payload["sampling"]
    loss = payload["loss"]
    trg = payload["trg"]
    optimizer = training["optimizer"]
    return {
        "seed": int(seed),
        "device": device,
        "dtype": payload["runtime"]["dtype"],
        "width": network["hidden_width"],
        "depth": network["hidden_layers"],
        "activation": network["activation"],
        "warmup_iters": training["warmup_iterations"],
        "gated_iters": training["continuation_iterations"],
        "lr_warmup": optimizer["warmup_learning_rate"],
        "lr_gated": optimizer["continuation_learning_rate"],
        "weight_decay": optimizer["weight_decay"],
        "grad_clip": training["gradient_clip_norm"],
        "n_f": sampling["interior_points"],
        "n_ic": sampling["initial_points"],
        "n_bc": sampling["boundary_points"],
        "w_ic": loss["initial_weight"],
        "w_bc": loss["boundary_weight"],
        "w_pde": loss["pde_weight"],
        "h_min_factor": trg["h_min_factor"],
        "h_max_factor": trg["h_max_factor"],
        "cmin_start": trg["cmin_start"],
        "cmin_end": trg["cmin_end"],
        "beta": trg["beta"],
        "residual_floor": trg["residual_floor"],
        "trace_ratio_epsilon": trg.get(
            "trace_ratio_epsilon",
            trg.get("ratio_epsilon", 1.0e-6),
        ),
        "batch_mean_epsilon": trg["batch_mean_epsilon"],
        "weighted_loss_epsilon": trg["weighted_loss_epsilon"],
        "relative_error_epsilon": (
            payload.get("reported_evaluation", payload.get("evaluation", {}))
            .get("relative_error_epsilon", trg.get("relative_error_epsilon", 1.0e-12))
        ),
    }


def burgers_config(path: Path, *, seed: int, device: str) -> Burgers1DConfig:
    payload = load_yaml(path)
    problem = payload["problem"]
    evaluation = payload["reported_evaluation"]
    values = _common(payload, seed=seed, device=device)
    values.update(
        x_min=problem["x_min"],
        x_max=problem["x_max"],
        t_min=problem["t_min"],
        t_max=problem["t_max"],
        uL=problem["left_state"],
        uR=problem["right_state"],
        x0=problem["discontinuity_x0"],
        eval_nx=evaluation["space_time_nx"],
        eval_nt=evaluation["space_time_nt"],
    )
    return Burgers1DConfig(**values)


def euler_config(path: Path, *, seed: int, device: str) -> Euler1DConfig:
    payload = load_yaml(path)
    problem = payload["problem"]
    network = payload["network"]
    evaluation = payload["reported_evaluation"]
    left = problem["left_state"]
    right = problem["right_state"]
    values = _common(payload, seed=seed, device=device)
    values.update(
        x_min=problem["x_min"],
        x_max=problem["x_max"],
        t_min=problem["t_min"],
        t_max=problem["t_max"],
        gamma=problem["gamma"],
        rhoL=left[0],
        uL=left[1],
        pL=left[2],
        rhoR=right[0],
        uR=right[1],
        pR=right[2],
        x0=problem["discontinuity_x0"],
        rho_floor=network["density_floor"],
        p_floor=network["pressure_floor"],
        trace_norm_epsilon=payload["trg"]["trace_norm_epsilon"],
        eval_nx=evaluation["space_time_nx"],
        eval_nt=evaluation["space_time_nt"],
        slice_nx=payload["paper_figures"]["line_grid_nx"],
    )
    return Euler1DConfig(**values)


def shallowwater_config(
    path: Path,
    *,
    seed: int,
    device: str,
) -> ShallowWater1DConfig:
    payload = load_yaml(path)
    problem = payload["problem"]
    network = payload["network"]
    evaluation = payload["evaluation"]
    diagnostics = payload["diagnostics"]
    left = problem["left_state"]
    right = problem["right_state"]

    values = _common(payload, seed=seed, device=device)
    values.update(
        x_min=problem["x_min"],
        x_max=problem["x_max"],
        t_min=problem["t_min"],
        t_max=problem["t_max"],
        g_const=problem["gravity"],
        hL=left[0],
        qL=left[1],
        hR=right[0],
        qR=right[1],
        x0=problem["discontinuity_x0"],
        h_floor=network["depth_floor"],
        trace_norm_epsilon=payload["trg"].get(
            "trace_norm_epsilon",
            1.0e-12,
        ),
        eval_nx=evaluation["space_time_grid"][0],
        eval_nt=evaluation["space_time_grid"][1],
        slice_nx=1600,
        t_final_plot=problem["t_max"],
        n_control_volumes=diagnostics["control_volumes"],
        cv_quad_nx=diagnostics["control_volume_quadrature_x"],
        cv_quad_nt=diagnostics["control_volume_quadrature_t"],
        cv_min_width=diagnostics["control_volume_min_width"],
        cv_min_duration=diagnostics["control_volume_min_duration"],
    )
    return ShallowWater1DConfig(**values)


def burgers_2d_config(
    path: Path,
    *,
    seed: int,
    device: str,
) -> Burgers2DConfig:
    payload = load_yaml(path)
    domain = payload["domain"]
    riemann = payload["riemann"]
    network = payload["network"]
    training = payload["training"]
    loss_weights = training["loss_weights"]
    trg = payload["trg"]
    evaluation = payload["evaluation"]

    return Burgers2DConfig(
        seed=int(seed),
        device=str(device),
        dtype="float32",
        x_min=domain["x"][0],
        x_max=domain["x"][1],
        y_min=domain["y"][0],
        y_max=domain["y"][1],
        t_min=domain["t"][0],
        t_max=domain["t"][1],
        uL=riemann["u_left"],
        uR=riemann["u_right"],
        theta_deg=riemann["theta_deg"],
        eta0=riemann["eta0"],
        width=network["hidden_width"],
        depth=network["hidden_layers"],
        activation=network["activation"],
        warmup_iters=training["warmup_iterations"],
        gated_iters=training["continuation_iterations"],
        lr_warmup=training["lr_warmup"],
        lr_gated=training["lr_continuation"],
        weight_decay=training["weight_decay"],
        grad_clip=training["gradient_clip_norm"],
        n_f=training["collocation_per_iteration"],
        n_ic=training["initial_points_per_iteration"],
        n_bc=training["boundary_points_per_iteration"],
        w_ic=loss_weights["ic"],
        w_bc=loss_weights["bc"],
        w_pde=loss_weights["pde"],
        h_max_factor=trg["h_max_factor"],
        h_min_factor=trg["h_min_factor"],
        cmin_start=trg["cmin_start"],
        cmin_end=trg["cmin_end"],
        beta=trg["beta"],
        residual_floor=trg["residual_floor"],
        trace_ratio_epsilon=trg["trace_ratio_stabilizer"],
        batch_mean_epsilon=trg["normalization_stabilizer"],
        weighted_loss_epsilon=trg["normalization_stabilizer"],
        eval_nxy=evaluation["final_time_grid"][0],
        eval_space_nxy=evaluation["space_time_grid"][0],
        eval_nt=evaluation["space_time_grid"][2],
    )


def euler_2d_config(
    path: Path,
    *,
    seed: int,
    device: str,
) -> Euler2DConfig:
    payload = load_yaml(path)
    domain = payload["domain"]
    problem = payload["rotated_sod"]
    network = payload["network"]
    training = payload["training"]
    loss = training["loss_weights"]
    trg = payload["trg"]
    evaluation = payload["evaluation"]
    left = problem["left_primitive"]
    right = problem["right_primitive"]

    return Euler2DConfig(
        seed=int(seed),
        device=str(device),
        dtype="float32",
        x_min=domain["x"][0],
        x_max=domain["x"][1],
        y_min=domain["y"][0],
        y_max=domain["y"][1],
        t_min=domain["t"][0],
        t_max=domain["t"][1],
        x0=problem["x0"],
        y0=problem["y0"],
        theta_deg=problem["theta_deg"],
        gamma=problem["gamma"],
        rho_L=left[0],
        un_L=left[1],
        p_L=left[2],
        rho_R=right[0],
        un_R=right[1],
        p_R=right[2],
        width=network["hidden_width"],
        depth=network["hidden_layers"],
        activation=network["activation"],
        rho_floor=network["rho_floor"],
        p_floor=network["p_floor"],
        warmup_iters=training["warmup_iterations"],
        gated_iters=training["continuation_iterations"],
        lr_warmup=training["lr_warmup"],
        lr_gated=training["lr_continuation"],
        grad_clip=training["gradient_clip_norm"],
        n_f=training["collocation_per_iteration"],
        n_ic=training["initial_points_per_iteration"],
        n_bc=training["boundary_points_per_iteration"],
        w_ic=loss["ic"],
        w_bc=loss["bc"],
        w_pde=loss["pde"],
        ring_trace_pairs=trg["projective_direction_count"],
        gate_tau=trg["gate_tau"],
        h_max_factor=trg["h_max_factor"],
        h_min_factor=trg["h_min_factor"],
        cmin_start=trg["cmin_start"],
        cmin_end=trg["cmin_end"],
        beta=trg["beta"],
        residual_floor=trg["residual_floor"],
        gate_rho_scale=trg["component_scales"]["rho"],
        gate_u_scale=trg["component_scales"]["u"],
        gate_p_scale=trg["component_scales"]["p"],
        eval_nxy=evaluation["final_time_grid"][0],
        eval_space_nxy=evaluation["metric_space_time_grid"][0],
        eval_nt=evaluation["metric_space_time_grid"][2],
        line_n=evaluation["normal_line_points"],
        gate_eval_nxy=220,
        cons_nxy=70,
        cons_nt=31,
        n_control_volumes=24,
        cv_quad_nxy=32,
        cv_quad_nt=32,
        cv_min_width=0.25,
        cv_min_duration=0.04,
    )


def shallowwater_2d_config(
    path: Path,
    *,
    seed: int,
    device: str,
) -> ShallowWater2DConfig:
    payload = load_yaml(path)
    domain = payload["domain"]
    problem = payload["problem"]
    model = payload["model"]
    training = payload["training"]
    sampling = payload["sampling_per_iteration"]
    loss = payload["loss_weights"]
    trg = payload["trace_ratio_gate"]
    evaluation = payload["evaluation"]
    reference = payload["reference"]["canonical_reporting"]

    return ShallowWater2DConfig(
        seed=int(seed),
        device=str(device),
        dtype="float32",
        x_min=domain["x"][0],
        x_max=domain["x"][1],
        y_min=domain["y"][0],
        y_max=domain["y"][1],
        t_min=domain["t"][0],
        t_max=domain["t"][1],
        g_const=problem["gravity"],
        center_x=problem["center"][0],
        center_y=problem["center"][1],
        dam_radius=problem["radius"],
        h_inside=problem["inside_state"][0],
        m_inside=problem["inside_state"][1],
        n_inside=problem["inside_state"][2],
        h_outside=problem["outside_state"][0],
        m_outside=problem["outside_state"][1],
        n_outside=problem["outside_state"][2],
        width=model["width"],
        depth=model["hidden_layers"],
        activation=model["activation"],
        h_floor=model["h_floor"],
        warmup_iters=training["warmup_iterations"],
        gated_iters=training["continuation_iterations"],
        lr_warmup=training["warmup_learning_rate"],
        lr_gated=training["continuation_learning_rate"],
        grad_clip=training["gradient_norm_limit"],
        n_f=sampling["interior"],
        n_ic=sampling["initial"],
        n_bc=sampling["boundary"],
        w_ic=loss["initial"],
        w_bc=loss["boundary"],
        w_pde=loss["pde"],
        ring_trace_pairs=trg["projective_directions"],
        gate_tau=trg["gate_tau"],
        h_max_factor=trg["h_max_factor"],
        h_min_factor=trg["h_min_factor"],
        cmin_start=trg["cmin_start"],
        cmin_end=trg["cmin_end"],
        beta=trg["beta"],
        residual_floor=trg["residual_floor"],
        eval_nxy=evaluation["metric_final_grid"][0],
        eval_space_nxy=evaluation["metric_space_time_grid"][0],
        eval_nt=evaluation["metric_space_time_grid"][2],
        line_n=1600,
        radial_angles=16,
        gate_eval_nxy=180,
        fv_nxy=reference["fv_nxy"],
        fv_cfl=reference["cfl"],
        fv_order=reference["order"],
        cons_nxy=70,
        cons_nt=31,
        n_control_volumes=24,
        cv_quad_nxy=32,
        cv_quad_nt=32,
        cv_min_width=0.25,
        cv_min_duration=0.04,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmark",
        choices=list(CONFIG_FILES),
        required=True,
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--reference-path",
        type=Path,
        default=None,
        help="Canonical FV reference required for full 2D shallow-water evaluation.",
    )
    args = parser.parse_args()

    config_path = args.config or (
        REPO_ROOT / "configs" / CONFIG_FILES[args.benchmark]
    )

    if args.benchmark == "burgers_1d":
        cfg = burgers_config(config_path, seed=args.seed, device=args.device)
        runner = run_burgers_1d
        metric = "space_time_rel_l2"
    elif args.benchmark == "euler_1d":
        cfg = euler_config(config_path, seed=args.seed, device=args.device)
        runner = run_euler_1d
        metric = "primitive_scaled_space_time_rel_l2"
    elif args.benchmark == "shallowwater_1d":
        cfg = shallowwater_config(
            config_path,
            seed=args.seed,
            device=args.device,
        )
        runner = run_shallowwater_1d
        metric = "state_scaled_space_time_rel_l2"
    elif args.benchmark == "burgers_2d":
        cfg = burgers_2d_config(
            config_path,
            seed=args.seed,
            device=args.device,
        )
        runner = run_burgers_2d
        metric = "space_time_rel_l2"
    elif args.benchmark == "euler_2d":
        cfg = euler_2d_config(
            config_path,
            seed=args.seed,
            device=args.device,
        )
        runner = run_euler_2d
        metric = "primitive_scaled_space_time_rel_l2"
    else:
        cfg = shallowwater_2d_config(
            config_path,
            seed=args.seed,
            device=args.device,
        )
        runner = run_shallowwater_2d
        metric = "state_scaled_space_time_rel_l2"

    if args.smoke_test:
        cfg = cfg.smoke_copy(seed=args.seed, device=args.device)
        output = REPO_ROOT / "runs" / "reproduction" / "smoke"
    else:
        output = REPO_ROOT / "runs" / "reproduction" / "full"

    runner_kwargs = {
        "output_root": output,
        "protected_reported_root": REPO_ROOT / "artifacts" / "reported",
        "overwrite": args.overwrite,
    }

    if args.benchmark == "shallowwater_2d":
        payload = load_yaml(config_path)
        reference = payload["reference"]["canonical_reporting"]
        reference_path = args.reference_path
        if reference_path is None and not args.smoke_test:
            pointer = (
                REPO_ROOT
                / "artifacts"
                / "reported"
                / "2d_shallowwater"
                / "reference"
                / "fv1024_reference_pointer.json"
            )
            if pointer.is_file():
                pointer_payload = __import__("json").loads(
                    pointer.read_text(encoding="utf-8")
                )
                candidate = Path(
                    pointer_payload["expected_project_relpath"]
                )
                if not candidate.is_absolute():
                    candidate = REPO_ROOT / candidate
                reference_path = candidate
        runner_kwargs.update(
            reference_path=reference_path,
            reference_sha256=reference["sha256"],
            smoke_test=args.smoke_test,
        )

    frame = runner(
        cfg,
        **runner_kwargs,
    )

    if metric in frame.columns:
        print(frame[["method", "seed", metric]].to_string(index=False))
    else:
        print(frame.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
