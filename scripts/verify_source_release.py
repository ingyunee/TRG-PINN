#!/usr/bin/env python
"""Check source syntax, reporting metadata, and saved comparison tables.

This command does not train models or verify manuscript accuracy. Optional
checkpoint/reference artifacts are checked separately by verify_artifacts.py.
"""

from __future__ import annotations

from pathlib import Path
import ast
import json
import os
import subprocess
import sys
import tempfile

import pandas as pd
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_TITLE = (
    "Trace-Ratio-Gated Physics-Informed Neural Networks for "
    "Hyperbolic Conservation Laws with Discontinuous Solutions"
)
EXPECTED_METHODS = {
    "PINN", "AAF-PINN", "LRA-PINN", "SA-PINN", "RAD-PINN",
    "RAR-D-PINN", "gPINN-subsampled", "cPINN", "XPINN", "Ours",
}
EXPECTED_EQUATIONS = {
    "1d_burgers", "1d_euler", "1d_shallowwater",
    "2d_burgers", "2d_euler", "2d_shallowwater",
}
EXPECTED_SEEDS = {2026, 7, 42, 100, 31415}
PROHIBITED_TOKENS = {
    "/home/" + "jupyter-ingyunk",
    "tpinn_" + "benchmark_project",
    "/mnt/" + "data/",
}
BINARY_SUFFIXES = {".pt", ".pth", ".ckpt", ".npz", ".npy"}
TEXT_SUFFIXES = {
    ".py", ".md", ".txt", ".yaml", ".yml", ".json", ".cff",
    ".toml", ".tex", ".csv",
}


def run(command: list[str]) -> None:
    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {command}")



def iter_source_files(root: Path):
    """Skip local outputs, environments, and optional external artifact trees."""
    ignored_names = {".git", ".venv", "venv", ".conda", ".tox", "__pycache__",
                     ".pytest_cache", ".mypy_cache", ".ipynb_checkpoints"}
    ignored_roots = {"runs", "dist", "artifacts_external"}
    ignored_paths = {"artifacts/reported", "results/raw"}
    for directory, subdirs, files in os.walk(root):
        relative = Path(directory).relative_to(root)
        subdirs[:] = [
            name for name in subdirs
            if name not in ignored_names
            and not (relative == Path(".") and name in ignored_roots)
            and (relative / name).as_posix() not in ignored_paths
        ]
        for name in files:
            path = Path(directory) / name
            # Installed baseline binaries are optional, manifest-checked data.
            if (path.suffix.lower() in BINARY_SUFFIXES
                    and path.relative_to(root).parts[:2] == ("artifacts", "reported_baselines")):
                continue
            yield path


def main() -> int:
    required = [
        "README.md",
        "CITATION.cff",
        ".gitignore",
        "configs",
        "src/trgpinn",
        "baselines",
        "scripts",
        "environment/environment.yml",
        "environment/requirements.txt",
        "results/reported_metrics/all_metrics_final.csv",
        "results/benchmark_comparison/benchmark_comparison_formatted.csv",
        "artifacts/manifests",
    ]
    for relative in required:
        path = REPO_ROOT / relative
        if not path.exists():
            raise FileNotFoundError(path)

    citation = yaml.safe_load(
        (REPO_ROOT / "CITATION.cff").read_text(encoding="utf-8")
    )
    if citation.get("title") != EXPECTED_TITLE:
        raise AssertionError(citation.get("title"))
    authors = citation.get("authors", [])
    author_pairs = [
        (item.get("given-names"), item.get("family-names"))
        for item in authors
    ]
    if author_pairs != [("Ingyun", "Kang"), ("Eunho", "Koo")]:
        raise AssertionError(author_pairs)
    # Release metadata records the original package preparation. Visibility
    # and license selection are not tests of source correctness.
    metadata_path = REPO_ROOT / "release" / "release_metadata.json"
    if metadata_path.is_file():
        json.loads(metadata_path.read_text(encoding="utf-8"))

    master = pd.read_csv(
        REPO_ROOT / "results" / "reported_metrics" / "all_metrics_final.csv",
        low_memory=False,
    )
    if len(master) != 300:
        raise AssertionError(len(master))
    if set(master["method"].astype(str)) != EXPECTED_METHODS:
        raise AssertionError(sorted(set(master["method"].astype(str))))
    if set(master["equation"].astype(str)) != EXPECTED_EQUATIONS:
        raise AssertionError(sorted(set(master["equation"].astype(str))))
    if set(pd.to_numeric(master["seed"], errors="raise").astype(int)) != EXPECTED_SEEDS:
        raise AssertionError(sorted(set(master["seed"])))
    if master.duplicated(["equation", "method", "seed"]).any():
        raise AssertionError("Duplicate canonical rows")

    comparison = pd.read_csv(
        REPO_ROOT
        / "results"
        / "benchmark_comparison"
        / "benchmark_comparison_formatted.csv",
        index_col=0,
    )
    if comparison.shape != (10, 7):
        raise AssertionError(comparison.shape)

    # Recompute comparison in a temporary directory and compare exact products.
    with tempfile.TemporaryDirectory(prefix="trgpinn_source_verify_") as tmp:
        tmp_root = Path(tmp)
        run([
            sys.executable,
            str(REPO_ROOT / "scripts" / "reproduce_benchmark_comparison.py"),
            "--master",
            str(REPO_ROOT / "results" / "reported_metrics" / "all_metrics_final.csv"),
            "--output-root",
            str(tmp_root),
        ])
        for name in (
            "benchmark_comparison_long.csv",
            "benchmark_comparison_numeric.csv",
            "benchmark_comparison_formatted.csv",
            "benchmark_comparison_ranks.csv",
            "pinn_trg_reductions.csv",
        ):
            expected = REPO_ROOT / "results" / "benchmark_comparison" / name
            actual = tmp_root / name
            if not actual.is_file():
                raise FileNotFoundError(actual)
            if expected.read_bytes() != actual.read_bytes():
                raise AssertionError(f"Comparison mismatch: {name}")

    python_files = 0
    for root_name in ("src", "baselines", "scripts"):
        for path in (REPO_ROOT / root_name).rglob("*.py"):
            ast.parse(
                path.read_text(encoding="utf-8"),
                filename=str(path.relative_to(REPO_ROOT)),
            )
            python_files += 1

    for script_name in (
        "train.py",
        "evaluate.py",
        "reproduce_figures.py",
        "train_baseline.py",
        "reproduce_benchmark_comparison.py",
    ):
        run([sys.executable, str(REPO_ROOT / "scripts" / script_name), "--help"])

    source_files = list(iter_source_files(REPO_ROOT))
    binary_files = [
        path.relative_to(REPO_ROOT).as_posix()
        for path in source_files if path.suffix.lower() in BINARY_SUFFIXES
    ]
    if binary_files:
        raise AssertionError(f"Unexpected binaries in source files: {binary_files[:10]}")

    violations = []
    for path in source_files:
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for token in PROHIBITED_TOKENS:
            if token in text:
                violations.append((path.relative_to(REPO_ROOT).as_posix(), token))
    if violations:
        raise AssertionError(violations[:20])

    print("[OK] source and citation metadata    : PASS")
    print("[OK] canonical master rows           : 300/300")
    print("[OK] canonical grid                  : 10 x 6 x 5")
    print("[OK] comparison matrix               : 10 x 7")
    print("[OK] comparison regeneration         : EXACT")
    print("[OK] public Python syntax            :", python_files)
    print("[OK] public CLI help                 : 5/5")
    print("[OK] source binary scan              : PASS (external artifacts excluded)")
    print("[OK] portable path scan              : PASS")
    print("[INFO] runtime and accuracy tests      : NOT RUN by this command")
    if not (REPO_ROOT / "LICENSE").is_file():
        print("[INFO] software license                : not assigned; see docs/LICENSE_PENDING.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
