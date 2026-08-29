
from __future__ import annotations

import importlib


BENCHMARK_MODULES = {
    "burgers_1d": "baselines.training.burgers_1d",
    "euler_1d": "baselines.training.euler_1d",
    "shallowwater_1d": "baselines.training.shallowwater_1d",
    "burgers_2d": "baselines.training.burgers_2d",
    "euler_2d": "baselines.training.euler_2d",
    "shallowwater_2d": "baselines.training.shallowwater_2d",
}

METHODS = (
    "aaf_pinn",
    "lra_pinn",
    "sa_pinn",
    "rad_pinn",
    "rar_d_pinn",
    "gpinn_subsampled",
    "cpinn",
    "xpinn",
)


def load_runtime(benchmark: str):
    if benchmark not in BENCHMARK_MODULES:
        raise KeyError(
            f"Unknown benchmark: {benchmark}"
        )
    return importlib.import_module(
        BENCHMARK_MODULES[benchmark]
    )


def get_trainer(
    benchmark: str,
    method: str,
):
    if method not in METHODS:
        raise KeyError(
            f"Unknown specialized baseline: {method}"
        )
    runtime = load_runtime(benchmark)
    return (
        runtime,
        runtime.SPECIALIZED_TRAINERS[method],
    )
