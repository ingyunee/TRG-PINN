#!/usr/bin/env python
"""Evaluate frozen reported checkpoints without retraining."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from trgpinn.equations.burgers_1d import (
    Burgers1DConfig,
    build_model as build_burgers_1d,
    evaluate_model as evaluate_burgers_1d,
)
from trgpinn.equations.euler_1d import (
    Euler1DConfig,
    build_model as build_euler_1d,
    evaluate_model as evaluate_euler_1d,
)
from trgpinn.equations.shallowwater_1d import (
    ShallowWater1DConfig,
    build_model as build_shallowwater_1d,
    evaluate_model as evaluate_shallowwater_1d,
)
from trgpinn.equations.burgers_2d import (
    Burgers2DConfig,
    build_model as build_burgers_2d,
    evaluate_model as evaluate_burgers_2d,
)
from trgpinn.equations.euler_2d import (
    Euler2DConfig,
    build_model as build_euler_2d,
    evaluate_model as evaluate_euler_2d,
)
from trgpinn.equations.shallowwater_2d import (
    ShallowWater2DConfig,
    build_model as build_shallowwater_2d,
    evaluate_model as evaluate_shallowwater_2d,
    load_fv_reference,
)
from trgpinn.utils import (
    configure_torch_runtime,
    load_checkpoint_into_model,
    read_json,
    resolve_device,
)


BENCHMARKS = {
    "burgers_1d": {
        "artifact_name": "1d_burgers",
        "config_type": Burgers1DConfig,
        "build_model": build_burgers_1d,
        "evaluate_model": evaluate_burgers_1d,
        "default_seed": 7,
    },
    "euler_1d": {
        "artifact_name": "1d_euler",
        "config_type": Euler1DConfig,
        "build_model": build_euler_1d,
        "evaluate_model": evaluate_euler_1d,
        "default_seed": 2026,
    },
    "shallowwater_1d": {
        "artifact_name": "1d_shallowwater",
        "config_type": ShallowWater1DConfig,
        "build_model": build_shallowwater_1d,
        "evaluate_model": evaluate_shallowwater_1d,
        "default_seed": 7,
    },
    "burgers_2d": {
        "artifact_name": "2d_burgers",
        "config_type": Burgers2DConfig,
        "build_model": build_burgers_2d,
        "evaluate_model": evaluate_burgers_2d,
        "default_seed": 31415,
    },
    "euler_2d": {
        "artifact_name": "2d_euler",
        "config_type": Euler2DConfig,
        "build_model": build_euler_2d,
        "evaluate_model": evaluate_euler_2d,
        "default_seed": 31415,
    },
    "shallowwater_2d": {
        "artifact_name": "2d_shallowwater",
        "config_type": ShallowWater2DConfig,
        "build_model": build_shallowwater_2d,
        "evaluate_model": evaluate_shallowwater_2d,
        "default_seed": 100,
        "requires_reference": True,
    },
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmark",
        choices=list(BENCHMARKS),
        required=True,
    )
    parser.add_argument(
        "--method",
        choices=["pinn", "trg_pinn", "all"],
        default="all",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--reference-path",
        type=Path,
        default=None,
        help="Canonical FV1024 reference for 2D shallow-water evaluation.",
    )
    args = parser.parse_args()

    specification = BENCHMARKS[args.benchmark]
    seed = (
        specification["default_seed"]
        if args.seed is None
        else int(args.seed)
    )
    methods = (
        ["pinn", "trg_pinn"]
        if args.method == "all"
        else [args.method]
    )
    rows = []

    reference = None
    if specification.get("requires_reference"):
        reference_path = args.reference_path
        pointer_path = (
            REPO_ROOT
            / "artifacts"
            / "reported"
            / "2d_shallowwater"
            / "reference"
            / "fv1024_reference_pointer.json"
        )
        expected_sha256 = None
        if pointer_path.is_file():
            pointer = read_json(pointer_path)
            expected_sha256 = pointer["sha256"]
            if reference_path is None:
                candidate = Path(pointer["expected_project_relpath"])
                if not candidate.is_absolute():
                    candidate = REPO_ROOT / candidate
                reference_path = candidate
        if reference_path is None:
            raise FileNotFoundError(
                "Provide --reference-path or install the canonical "
                "FV1024 artifact referenced by fv1024_reference_pointer.json."
            )
        reference = load_fv_reference(
            reference_path,
            expected_sha256=expected_sha256,
        )

    for method in methods:
        run_dir = (
            REPO_ROOT
            / "artifacts"
            / "reported"
            / specification["artifact_name"]
            / method
            / f"seed_{seed}"
        )
        config_payload = read_json(run_dir / "config.json")
        cfg = specification["config_type"].from_legacy_mapping(
            config_payload.get("config", config_payload),
            seed=seed,
            device=args.device,
        )
        device = resolve_device(args.device)
        dtype = configure_torch_runtime(cfg.dtype)
        model = specification["build_model"](cfg).to(
            device=device,
            dtype=dtype,
        )
        load_checkpoint_into_model(model, run_dir / "model_final.pt")
        label = "PINN" if method == "pinn" else "TRG-PINN"
        if specification.get("requires_reference"):
            rows.append(
                specification["evaluate_model"](
                    model,
                    label,
                    reference,
                    cfg,
                )
            )
        else:
            rows.append(
                specification["evaluate_model"](
                    model,
                    label,
                    cfg,
                )
            )

    frame = pd.DataFrame(rows)
    print(frame.to_string(index=False))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(args.output, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
