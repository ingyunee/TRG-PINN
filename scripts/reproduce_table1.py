#!/usr/bin/env python
"""Regenerate the final ten-method comparison table from the canonical master."""

from __future__ import annotations

import argparse
from pathlib import Path
import json
import os

import numpy as np
import pandas as pd


METHOD_ORDER = [
    "PINN",
    "AAF-PINN",
    "LRA-PINN",
    "SA-PINN",
    "RAD-PINN",
    "RAR-D-PINN",
    "gPINN-subsampled",
    "cPINN",
    "XPINN",
    "Ours",
]

METHOD_LABELS = {
    "PINN": "PINN",
    "AAF-PINN": "AAF-PINN",
    "LRA-PINN": "LRA-PINN",
    "SA-PINN": "SA-PINN",
    "RAD-PINN": "RAD-PINN",
    "RAR-D-PINN": "RAR-D-PINN",
    "gPINN-subsampled": "gPINN-sub.",
    "cPINN": "cPINN",
    "XPINN": "XPINN",
    "Ours": "TRG-PINN",
}

BENCHMARKS = [
    ("1d_burgers", "1D Burgers", "space_time_rel_l2"),
    ("1d_euler", "1D Euler", "primitive_scaled_space_time_rel_l2"),
    ("1d_shallowwater", "1D SW", "state_scaled_space_time_rel_l2"),
    ("2d_burgers", "2D Burgers", "space_time_rel_l2"),
    ("2d_euler", "2D Euler", "primitive_scaled_space_time_rel_l2"),
    ("2d_shallowwater", "2D SW", "state_scaled_space_time_rel_l2"),
]

SEEDS = [2026, 7, 42, 100, 31415]


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    temporary = path.with_name(
        f".{path.name}.tmp"
    )
    temporary.write_text(
        content,
        encoding="utf-8",
    )
    os.replace(
        temporary,
        path,
    )


def cell_text(mean: float, std: float) -> str:
    return (
        f"{mean:.2e}"
        + "\n"
        + f"({std:.2e})"
    )


def latex_cell(
    mean: float,
    std: float,
    rank: float,
) -> str:
    mean_text = f"{mean:.2e}"
    std_text = f"({std:.2e})"

    if rank == 1.0:
        mean_text = (
            r"\textbf{"
            + mean_text
            + "}"
        )
        std_text = (
            r"\textbf{"
            + std_text
            + "}"
        )

    elif rank == 2.0:
        mean_text = (
            r"\underline{"
            + mean_text
            + "}"
        )
        std_text = (
            r"\underline{"
            + std_text
            + "}"
        )

    return (
        r"\shortstack{"
        + mean_text
        + r"\\"
        + std_text
        + "}"
    )


def regenerate(
    master_path: Path,
    output_root: Path,
) -> dict:
    master = pd.read_csv(
        master_path,
        low_memory=False,
    )

    master["seed"] = pd.to_numeric(
        master["seed"],
        errors="raise",
    ).astype(int)

    records = []

    for equation, benchmark, metric in BENCHMARKS:
        frame = master[
            master["equation"]
            .astype(str)
            .eq(equation)
        ].copy()

        frame[metric] = pd.to_numeric(
            frame[metric],
            errors="raise",
        )

        for method in METHOD_ORDER:
            values = (
                frame[
                    frame["method"]
                    .astype(str)
                    .eq(method)
                ]
                .set_index("seed")
                .loc[
                    SEEDS,
                    metric,
                ]
                .to_numpy(dtype=float)
            )

            records.append(
                {
                    "equation": equation,
                    "benchmark": benchmark,
                    "method": method,
                    "public_method": METHOD_LABELS[method],
                    "primary_metric": metric,
                    "seed_count": 5,
                    "mean": float(
                        np.mean(
                            values
                        )
                    ),
                    "sample_std": float(
                        np.std(
                            values,
                            ddof=1,
                        )
                    ),
                    "seed_values_json": json.dumps(
                        {
                            str(seed): float(value)
                            for seed, value in zip(
                                SEEDS,
                                values,
                            )
                        },
                        sort_keys=True,
                    ),
                }
            )

    long_frame = pd.DataFrame(
        records
    )

    long_frame["rank"] = (
        long_frame
        .groupby("benchmark")["mean"]
        .rank(
            method="average",
            ascending=True,
        )
    )

    mean_rank = (
        long_frame
        .groupby("method")["rank"]
        .mean()
    )

    benchmark_labels = [
        label
        for _, label, _ in BENCHMARKS
    ]

    numeric_wide = pd.DataFrame(
        index=[
            METHOD_LABELS[
                method
            ]
            for method in METHOD_ORDER
        ]
    )

    formatted = pd.DataFrame(
        index=numeric_wide.index
    )

    rank_wide = (
        long_frame
        .pivot(
            index="method",
            columns="benchmark",
            values="rank",
        )
        .loc[METHOD_ORDER]
    )

    latex_rows = []

    for method in METHOD_ORDER:
        method_rows = (
            long_frame[
                long_frame["method"].eq(
                    method
                )
            ]
            .set_index("benchmark")
            .loc[
                benchmark_labels
            ]
        )

        for benchmark in benchmark_labels:
            row = method_rows.loc[
                benchmark
            ]

            numeric_wide.loc[
                METHOD_LABELS[method],
                f"{benchmark} mean",
            ] = float(
                row["mean"]
            )

            numeric_wide.loc[
                METHOD_LABELS[method],
                f"{benchmark} std",
            ] = float(
                row["sample_std"]
            )

            formatted.loc[
                METHOD_LABELS[method],
                benchmark,
            ] = cell_text(
                float(
                    row["mean"]
                ),
                float(
                    row["sample_std"]
                ),
            )

        numeric_wide.loc[
            METHOD_LABELS[method],
            "Mean rank",
        ] = float(
            mean_rank.loc[
                method
            ]
        )

        formatted.loc[
            METHOD_LABELS[method],
            "Mean rank",
        ] = (
            f"{float(mean_rank.loc[method]):.2f}"
        )

        latex_cells = [
            latex_cell(
                float(
                    method_rows.loc[
                        benchmark,
                        "mean",
                    ]
                ),
                float(
                    method_rows.loc[
                        benchmark,
                        "sample_std",
                    ]
                ),
                float(
                    method_rows.loc[
                        benchmark,
                        "rank",
                    ]
                ),
            )
            for benchmark in benchmark_labels
        ]

        latex_rows.append(
            METHOD_LABELS[method]
            + " & "
            + " & ".join(
                latex_cells
            )
            + " & "
            + f"{float(mean_rank.loc[method]):.2f}"
            + r" \\"
        )

    numeric_wide.index.name = "Method"
    formatted.index.name = "Method"
    rank_wide.index = [
        METHOD_LABELS[method]
        for method in METHOD_ORDER
    ]
    rank_wide.index.name = "Method"

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    long_path = (
        output_root
        / "table1_long.csv"
    )
    numeric_path = (
        output_root
        / "table1_final_numeric.csv"
    )
    formatted_path = (
        output_root
        / "table1_final_formatted.csv"
    )
    ranks_path = (
        output_root
        / "table1_final_ranks.csv"
    )
    reductions_path = (
        output_root
        / "pinn_trg_reductions.csv"
    )
    latex_path = (
        output_root
        / "table1_final.tex"
    )

    long_frame.to_csv(
        long_path,
        index=False,
        float_format="%.17g",
    )
    numeric_wide.to_csv(
        numeric_path,
        float_format="%.17g",
    )
    formatted.to_csv(
        formatted_path,
    )
    rank_wide.to_csv(
        ranks_path,
        float_format="%.17g",
    )

    reduction_records = []

    for equation, benchmark, metric in BENCHMARKS:
        frame = master[
            master["equation"]
            .astype(str)
            .eq(equation)
        ]

        pinn = (
            frame[
                frame["method"]
                .astype(str)
                .eq("PINN")
            ]
            .set_index("seed")
            .loc[
                SEEDS,
                metric,
            ]
            .to_numpy(dtype=float)
        )

        trg = (
            frame[
                frame["method"]
                .astype(str)
                .eq("Ours")
            ]
            .set_index("seed")
            .loc[
                SEEDS,
                metric,
            ]
            .to_numpy(dtype=float)
        )

        pinn_mean = float(
            np.mean(
                pinn
            )
        )
        trg_mean = float(
            np.mean(
                trg
            )
        )

        reduction_records.append(
            {
                "equation": equation,
                "benchmark": benchmark,
                "primary_metric": metric,
                "pinn_mean": pinn_mean,
                "trg_pinn_mean": trg_mean,
                "reduction_percent": (
                    100.0
                    * (
                        pinn_mean
                        - trg_mean
                    )
                    / pinn_mean
                ),
                "paired_wins": int(
                    np.sum(
                        trg
                        < pinn
                    )
                ),
            }
        )

    pd.DataFrame(
        reduction_records
    ).to_csv(
        reductions_path,
        index=False,
        float_format="%.17g",
    )

    latex = "\n".join(
        [
            r"\begin{table*}[t]",
            r"\centering",
            (
                r"\caption{Five-seed mean and sample standard deviation "
                r"of the primary space--time error.}"
            ),
            r"\label{tab:primary_comparison}",
            r"\begin{tabular}{lccccccc}",
            r"\toprule",
            (
                "Method & "
                + " & ".join(
                    benchmark_labels
                )
                + r" & Mean rank \\"
            ),
            r"\midrule",
            *latex_rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table*}",
            "",
        ]
    )

    atomic_write(
        latex_path,
        latex,
    )

    return {
        "long": str(
            long_path
        ),
        "numeric": str(
            numeric_path
        ),
        "formatted": str(
            formatted_path
        ),
        "ranks": str(
            ranks_path
        ),
        "reductions": str(
            reductions_path
        ),
        "latex": str(
            latex_path
        ),
        "rows": int(
            len(
                long_frame
            )
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()

    repo_root = (
        Path(__file__)
        .resolve()
        .parents[1]
    )

    parser.add_argument(
        "--master",
        type=Path,
        default=(
            repo_root
            / "results"
            / "reported_metrics"
            / "all_metrics_final.csv"
        ),
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=(
            repo_root
            / "results"
            / "table1"
        ),
    )

    args = parser.parse_args()

    report = regenerate(
        args.master
        .expanduser()
        .resolve(),
        args.output_root
        .expanduser()
        .resolve(),
    )

    print(
        json.dumps(
            report,
            indent=2,
        )
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
