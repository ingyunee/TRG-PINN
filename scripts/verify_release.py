#!/usr/bin/env python
"""Verify the TRG-PINN release at the available artifact level.

The curated source ZIP always receives source-only verification. Full frozen
artifact verification is added automatically when the external artifact tree
has been installed at the manifest-relative paths.
"""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]


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


def external_artifacts_installed() -> bool:
    reported = REPO_ROOT / "artifacts" / "reported"
    if not reported.is_dir():
        return False
    return any(
        path.is_file()
        for path in reported.rglob("*")
    )


def main() -> int:
    run([
        sys.executable,
        str(REPO_ROOT / "scripts" / "verify_source_release.py"),
    ])

    run([
        sys.executable,
        str(REPO_ROOT / "scripts" / "verify_baselines.py"),
        "--repo-root",
        str(REPO_ROOT),
    ])

    if external_artifacts_installed():
        run([
            sys.executable,
            str(REPO_ROOT / "scripts" / "verify_artifacts.py"),
            "--benchmark",
            "all",
        ])
        print("[OK] full frozen-artifact verification : PASS")
    else:
        print(
            "[INFO] full frozen-artifact verification : SKIPPED "
            "(external artifact tree not installed)"
        )

    print("[OK] TRG-PINN release verification at available artifact level")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
