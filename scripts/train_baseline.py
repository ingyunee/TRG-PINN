#!/usr/bin/env python
"""Run one specialized baseline with the reported training adapter."""

from __future__ import annotations

import argparse
import copy
import hashlib
import os
from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(REPO_ROOT),
    )


from baselines.training import (
    BENCHMARK_MODULES,
    METHODS,
    get_trainer,
)


FV1024_SHA256 = (
    "9df6008a0562c9de764fe48c4653a682"
    "a99912fe95983f1faff9d4bbf759572c"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(8 * 1024 * 1024),
            b"",
        ):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmark",
        choices=tuple(BENCHMARK_MODULES),
        required=True,
    )
    parser.add_argument(
        "--method",
        choices=METHODS,
        required=True,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
    )
    parser.add_argument(
        "--device",
        default="cpu",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=(
            REPO_ROOT
            / "runs"
            / "reproduction"
            / "baselines"
        ),
    )
    parser.add_argument(
        "--reference-path",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
    )
    return parser.parse_args()


def apply_smoke_settings(cfg):
    settings = {
        "warmup_iters": 1,
        "gated_iters": 1,
        "continuation_iters": 1,
        "n_f": 32,
        "n_ic": 16,
        "n_bc": 16,
        "eval_nx": 48,
        "eval_nxy": 24,
        "eval_space_nxy": 16,
        "eval_nt": 4,
        "line_n": 64,
        "gate_eval_nxy": 20,
        "cons_nxy": 12,
        "cons_nt": 4,
        "n_control_volumes": 2,
        "cv_quad_nx": 6,
        "cv_quad_ny": 6,
        "cv_quad_nxy": 6,
        "cv_quad_nt": 5,
        "radial_angles": 4,
        "fv_nx": 64,
        "fv_nxy": 16,
        "fv_order": 1,
        "print_every": 1,
        "history_every": 1,
        "save_outputs": True,
    }

    for name, value in settings.items():
        if hasattr(cfg, name):
            setattr(cfg, name, value)

    return cfg


def load_reference(
    benchmark: str,
    path: Path | None,
):
    if benchmark != "shallowwater_2d":
        if path is not None:
            raise ValueError(
                "--reference-path is used only for shallowwater_2d."
            )
        return None

    if path is None:
        raise ValueError(
            "Full shallowwater_2d baseline evaluation requires "
            "--reference-path. Smoke tests should use burgers_1d."
        )

    path = path.expanduser().resolve()

    if not path.is_file():
        raise FileNotFoundError(path)

    if sha256_file(path) != FV1024_SHA256:
        raise AssertionError(
            "The supplied 2D shallow-water reference is not "
            "the canonical FV1024 artifact."
        )

    with np.load(
        path,
        allow_pickle=False,
    ) as data:
        required = {
            "x",
            "y",
            "t",
            "H",
            "M",
            "N",
        }
        missing = required - set(data.files)
        if missing:
            raise KeyError(
                f"Reference fields missing: {sorted(missing)}"
            )
        return {
            key: np.asarray(data[key])
            for key in required
        }


def main() -> int:
    args = parse_args()

    output_root = (
        args.output_root
        .expanduser()
        .resolve()
    )

    protected_roots = (
        (REPO_ROOT / "artifacts").resolve(),
        (REPO_ROOT / "results" / "reported_metrics").resolve(),
        (REPO_ROOT / "results" / "paper_figures").resolve(),
    )

    for protected_root in protected_roots:
        try:
            output_root.relative_to(protected_root)
        except ValueError:
            continue
        raise ValueError(
            f"Refusing to write inside protected release data: "
            f"{protected_root}"
        )

    os.environ[
        "TRGPINN_BASELINE_RUNS_ROOT"
    ] = str(output_root)

    runtime, trainer = get_trainer(
        args.benchmark,
        args.method,
    )

    runtime.RUNS_ROOT = output_root

    cfg = copy.deepcopy(
        runtime.PUBLIC_BASE_CONFIG
    )

    if hasattr(cfg, "device"):
        cfg.device = str(args.device)

    if args.smoke_test:
        cfg = apply_smoke_settings(cfg)

    reference = (
        None
        if args.smoke_test
        else load_reference(
            args.benchmark,
            args.reference_path,
        )
    )

    kwargs = {
        "equation_name": runtime.EQUATION_NAME,
        "base_cfg": cfg,
        "seed": int(args.seed),
        "ref": reference,
        "total_iters": (
            2
            if args.smoke_test
            else 10_000
        ),
        "skip_if_done": (
            not args.overwrite
        ),
    }

    if args.smoke_test:
        if args.method == "rad_pinn":
            kwargs.update(
                candidate_factor=2,
                resample_period=1,
                residual_batch_size=64,
            )
        elif args.method == "rar_d_pinn":
            kwargs.update(
                initial_fraction=0.5,
                add_period=1,
                candidate_factor=2,
                residual_batch_size=64,
            )
        elif args.method == "gpinn_subsampled":
            kwargs.update(
                gpinn_n_g=8,
                gpinn_grad_batch_size=8,
            )
        elif args.method == "cpinn":
            kwargs.update(
                cpinn_n_interface_per_piece=8,
            )
        elif args.method == "xpinn":
            kwargs.update(
                xpinn_n_interface_per_piece=8,
            )

    frame = trainer(**kwargs)

    if frame is None or len(frame) != 1:
        raise RuntimeError(
            "The baseline trainer did not return one result row."
        )

    print(
        frame.to_string(
            index=False
        )
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
