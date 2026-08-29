#!/usr/bin/env python
"""Reproduce the ten-method benchmark comparison plotted as manuscript Figure 2."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import shutil


REPO_ROOT = Path(__file__).resolve().parents[1]
VERIFIED_GENERATOR = REPO_ROOT / "scripts" / "reproduce_table1.py"


def load_generator():
    if not VERIFIED_GENERATOR.is_file():
        raise FileNotFoundError(VERIFIED_GENERATOR)

    specification = importlib.util.spec_from_file_location(
        "_trgpinn_verified_comparison_generator",
        VERIFIED_GENERATOR,
    )

    if specification is None or specification.loader is None:
        raise ImportError(VERIFIED_GENERATOR)

    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--master",
        type=Path,
        default=(
            REPO_ROOT
            / "results"
            / "reported_metrics"
            / "all_metrics_final.csv"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=(
            REPO_ROOT
            / "results"
            / "benchmark_comparison"
        ),
    )
    args = parser.parse_args()

    output_root = args.output_root.expanduser().resolve()
    temporary_root = output_root / "_verified_generator_output"

    generator = load_generator()
    generator.regenerate(
        args.master.expanduser().resolve(),
        temporary_root,
    )

    mapping = {
        "table1_long.csv": "benchmark_comparison_long.csv",
        "table1_final_numeric.csv": "benchmark_comparison_numeric.csv",
        "table1_final_formatted.csv": "benchmark_comparison_formatted.csv",
        "table1_final_ranks.csv": "benchmark_comparison_ranks.csv",
        "pinn_trg_reductions.csv": "pinn_trg_reductions.csv",
        "table1_final.tex": "benchmark_comparison.tex",
    }

    output_root.mkdir(parents=True, exist_ok=True)

    for source_name, target_name in mapping.items():
        source = temporary_root / source_name
        target = output_root / target_name

        if not source.is_file():
            raise FileNotFoundError(source)

        shutil.copy2(source, target)
        print(target)

    shutil.rmtree(temporary_root)
    print("The current manuscript presents this matrix as Figure 2.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
