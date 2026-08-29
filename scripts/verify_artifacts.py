#!/usr/bin/env python
"""Verify immutable reported artifacts against benchmark staging manifests."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from trgpinn.utils import sha256_file


MANIFESTS = {
    "burgers_1d": (
        REPO_ROOT
        / "artifacts"
        / "manifests"
        / "1d_burgers_staging_manifest.csv"
    ),
    "euler_1d": (
        REPO_ROOT
        / "artifacts"
        / "manifests"
        / "1d_euler_staging_manifest.csv"
    ),
    "shallowwater_1d": (
        REPO_ROOT
        / "artifacts"
        / "manifests"
        / "1d_shallowwater_staging_manifest.csv"
    ),
    "burgers_2d": (
        REPO_ROOT
        / "artifacts"
        / "manifests"
        / "2d_burgers_staging_manifest.csv"
    ),
    "euler_2d": (
        REPO_ROOT
        / "artifacts"
        / "manifests"
        / "2d_euler_staging_manifest.csv"
    ),
    "shallowwater_2d": (
        REPO_ROOT
        / "artifacts"
        / "manifests"
        / "2d_shallowwater_staging_manifest.csv"
    ),
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmark",
        choices=[*MANIFESTS, "all"],
        default="all",
    )
    parser.add_argument("--manifest", type=Path, default=None)
    args = parser.parse_args()

    if args.manifest is not None:
        manifests = [args.manifest]
    elif args.benchmark == "all":
        manifests = list(MANIFESTS.values())
    else:
        manifests = [MANIFESTS[args.benchmark]]

    failures = []
    verified = 0
    for manifest_path in manifests:
        manifest = pd.read_csv(manifest_path)
        for row in manifest.itertuples(index=False):
            path = REPO_ROOT / str(row.release_relpath)
            if not path.is_file():
                failures.append((str(path), "missing"))
                continue
            actual = sha256_file(path)
            if actual != str(row.sha256):
                failures.append((str(path), "sha256 mismatch"))
                continue
            verified += 1

    if failures:
        for path, reason in failures[:20]:
            print(f"[FAIL] {path}: {reason}")
        print(f"Failed files: {len(failures)}")
        return 1

    print(
        f"[OK] Verified {verified} frozen files "
        f"across {len(manifests)} manifest(s)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
