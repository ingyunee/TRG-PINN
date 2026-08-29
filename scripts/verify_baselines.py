#!/usr/bin/env python
"""Verify the specialized-baseline registry and STEP 7A provenance metadata."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=None,
        help=(
            "Optional external artifact bundle. When supplied, every manifest "
            "entry that is present is hash-verified."
        ),
    )
    parser.add_argument(
        "--require-full-artifacts",
        action="store_true",
    )
    return parser.parse_args()


def read_csv(path: Path):
    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.expanduser().resolve()

    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from baselines import BASELINE_SPECS, validate_registry

    validate_registry()

    if len(BASELINE_SPECS) != 8:
        raise AssertionError(len(BASELINE_SPECS))

    manifest_path = (
        repo_root
        / "artifacts"
        / "manifests"
        / "baseline_artifact_manifest.csv"
    )
    rows_path = (
        repo_root
        / "results"
        / "reported_metrics"
        / "baseline_reported_rows.csv"
    )
    summary_path = (
        repo_root
        / "results"
        / "reported_metrics"
        / "baseline_summary_mean_std.csv"
    )
    provenance_path = (
        repo_root
        / "artifacts"
        / "baseline_method_provenance.json"
    )
    checkpoint_extra_audit_path = (
        repo_root
        / "tests"
        / "parity"
        / "baseline_checkpoint_extra_metric_audit.csv"
    )

    for path in (
        manifest_path,
        rows_path,
        summary_path,
        provenance_path,
        checkpoint_extra_audit_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    manifest = read_csv(manifest_path)
    rows = read_csv(rows_path)
    summary = read_csv(summary_path)
    checkpoint_extra_audit = read_csv(
        checkpoint_extra_audit_path
    )

    if len(rows) != 240:
        raise AssertionError(
            f"Reported rows: {len(rows)} != 240"
        )

    if len(summary) != 48:
        raise AssertionError(
            f"Summary rows: {len(summary)} != 48"
        )

    provenance = json.loads(
        provenance_path.read_text(
            encoding="utf-8"
        )
    )

    if provenance.get("baseline_count") != 8:
        raise AssertionError(
            provenance.get("baseline_count")
        )

    if provenance.get("reported_run_count") != 240:
        raise AssertionError(
            provenance.get("reported_run_count")
        )

    if len(provenance.get("methods", [])) != 8:
        raise AssertionError(
            len(provenance.get("methods", []))
        )

    success_count = int(
        provenance.get(
            "reported_success_run_count",
            -1,
        )
    )
    unavailable_count = int(
        provenance.get(
            "reported_unavailable_run_count",
            -1,
        )
    )

    if success_count + unavailable_count != 240:
        raise AssertionError(
            "Reported-run accounting mismatch: "
            f"{success_count} + {unavailable_count} != 240"
        )

    expected_manifest_rows = int(
        provenance.get(
            "artifact_manifest_rows",
            -1,
        )
    )

    if len(manifest) != expected_manifest_rows:
        raise AssertionError(
            f"Manifest rows: {len(manifest)} "
            f"!= {expected_manifest_rows}"
        )

    expected_checkpoint_extra_rows = int(
        provenance.get(
            "checkpoint_extra_numeric_overlap_fields",
            -1,
        )
    )

    if (
        len(checkpoint_extra_audit)
        != expected_checkpoint_extra_rows
    ):
        raise AssertionError(
            "Checkpoint-extra audit row count: "
            f"{len(checkpoint_extra_audit)} "
            f"!= {expected_checkpoint_extra_rows}"
        )

    if provenance.get(
        "checkpoint_extra_metrics_authoritative"
    ) is not False:
        raise AssertionError(
            "Checkpoint-extra metrics must not be "
            "treated as the canonical reporting source."
        )

    external_verified = 0
    external_missing = 0

    if args.artifact_root is not None:
        artifact_root = (
            args.artifact_root
            .expanduser()
            .resolve()
        )

        for row in manifest:
            path = (
                artifact_root
                / row["artifact_relpath"]
            )

            if not path.is_file():
                external_missing += 1
                continue

            if sha256_file(path) != row["sha256"]:
                raise AssertionError(
                    f"External artifact hash mismatch: {path}"
                )

            external_verified += 1

        if (
            args.require_full_artifacts
            and external_missing
        ):
            raise FileNotFoundError(
                f"Missing {external_missing} external baseline artifact files."
            )

    print("[OK] specialized baseline registry : 8/8")
    print("[OK] canonical baseline rows       : 240/240")
    print(
        "[OK] reported run accounting      : "
        f"success={success_count}, "
        f"unavailable={unavailable_count}"
    )
    print(
        "[OK] artifact manifest rows       : "
        f"{len(manifest)}"
    )
    print(
        "[OK] checkpoint-extra audit rows  : "
        f"{len(checkpoint_extra_audit)}"
    )
    print(
        "[OK] checkpoint-extra authority   : "
        "HISTORICAL / NON-AUTHORITATIVE"
    )

    if args.artifact_root is not None:
        print(
            "[OK] external artifacts           : "
            f"verified={external_verified}, missing={external_missing}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
