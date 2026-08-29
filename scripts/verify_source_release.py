#!/usr/bin/env python
"""Verify the private/source-only TRG-PINN package.

This command does not require the separately archived checkpoint, cache, or
FV1024 reference binaries. It verifies the curated source tree, canonical
metrics, comparison products, public Python syntax, CLI entry points, and
private pre-submission metadata.
"""

from __future__ import annotations

from pathlib import Path
import ast
import json
import shutil
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
        "release/release_metadata.json",
        "release/PRIVATE_GITHUB_UPLOAD_GUIDE.md",
    ]
    for relative in required:
        path = REPO_ROOT / relative
        if not path.exists():
            raise FileNotFoundError(path)

    # Private pre-submission package: no license or fabricated public URL yet.
    if (REPO_ROOT / "LICENSE").exists():
        raise AssertionError("Private package unexpectedly contains LICENSE")

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
    if "license" in citation:
        raise AssertionError("Private package claims a finalized license")
    if "repository-code" in citation:
        raise AssertionError("Private package claims a repository URL")

    metadata = json.loads(
        (REPO_ROOT / "release" / "release_metadata.json").read_text(
            encoding="utf-8"
        )
    )
    if metadata.get("planned_repository_visibility") != "private":
        raise AssertionError(metadata)
    if metadata.get("github_upload_performed") is not False:
        raise AssertionError(metadata)
    if metadata.get("software_license") != "PENDING_BEFORE_PUBLIC_RELEASE":
        raise AssertionError(metadata)

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

    binary_files = [
        path.relative_to(REPO_ROOT).as_posix()
        for path in REPO_ROOT.rglob("*")
        if path.is_file() and path.suffix.lower() in BINARY_SUFFIXES
    ]
    if binary_files:
        raise AssertionError(f"External binaries entered source package: {binary_files[:10]}")

    violations = []
    for path in REPO_ROOT.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for token in PROHIBITED_TOKENS:
            if token in text:
                violations.append((path.relative_to(REPO_ROOT).as_posix(), token))
    if violations:
        raise AssertionError(violations[:20])

    print("[OK] private/source package metadata : PASS")
    print("[OK] canonical master rows           : 300/300")
    print("[OK] canonical grid                  : 10 x 6 x 5")
    print("[OK] comparison matrix               : 10 x 7")
    print("[OK] comparison regeneration         : EXACT")
    print("[OK] public Python syntax            :", python_files)
    print("[OK] public CLI help                 : 5/5")
    print("[OK] external binaries               : EXCLUDED")
    print("[OK] portable path scan              : PASS")
    print("[INFO] license/repository URL         : PENDING BY DESIGN")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
