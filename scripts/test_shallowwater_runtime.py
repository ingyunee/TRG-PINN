#!/usr/bin/env python
"""Regression tests for shallow-water baseline evaluation and smoke runs.

Uses reduced CPU runs, not the manuscript training budget. The 2D smoke test
checks predictions without generating a reference or reporting error metrics.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
for entry in (ROOT, ROOT / "src"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))


def adapter_checks() -> dict:
    import torch
    from baselines.training import get_trainer
    from scripts.train_baseline import FV1024_SHA256, apply_smoke_settings, load_reference
    from trgpinn.equations.shallowwater_1d import (
        ShallowWater1DConfig, compute_error_metrics, state_scales,
    )

    torch.set_num_threads(1)
    one, _ = get_trainer("shallowwater_1d", "aaf_pinn")
    two, _ = get_trainer("shallowwater_2d", "aaf_pinn")

    class ConstantState(torch.nn.Module):
        def __init__(self, values):
            super().__init__()
            self.values = torch.nn.Parameter(torch.tensor(values, dtype=torch.float32))

        def forward(self, coords):
            return self.values.unsqueeze(0).expand(coords.shape[0], -1)

    cfg = apply_smoke_settings(copy.deepcopy(one.PUBLIC_BASE_CONFIG))
    cfg.device = "cpu"
    cfg.smoke_test = False  # Test the full evaluator with a small evaluation grid.
    model = ConstantState([1.5, 0.1])
    before = {key: value.clone() for key, value in model.state_dict().items()}
    model.train()
    row = one.evaluate_model_any(model, "test", cfg)
    canonical = ShallowWater1DConfig.from_legacy_mapping(one.cfg_to_dict(cfg))
    expected = compute_error_metrics(model, canonical)
    for key in ("state_scaled_space_time_rel_l2", "state_scaled_final_rel_l2"):
        assert math.isclose(row[key], expected[key], rel_tol=1e-12, abs_tol=1e-12), key
    assert row["reference_type"] == "exact_stoker_entropy_solution"
    assert row["smoke_test"] is False
    assert model.training
    assert all(torch.equal(before[key], value) for key, value in model.state_dict().items())
    assert state_scales(canonical) == (2.0, 2.0 * math.sqrt(2.0))
    assert one.build_reference_if_needed(cfg) is None
    try:
        one.evaluate_model_any(model, "test", cfg, ref=object())
    except ValueError:
        pass
    else:
        raise AssertionError("1D adapter silently accepted an FV reference")
    with patch.object(model, "forward", side_effect=TypeError("model-error-sentinel")):
        try:
            one.evaluate_model_any(model, "test", cfg)
        except TypeError as exc:
            assert "model-error-sentinel" in str(exc)
        else:
            raise AssertionError("Internal TypeError was swallowed")

    cfg2 = apply_smoke_settings(copy.deepcopy(two.PUBLIC_BASE_CONFIG))
    cfg2.device = "cpu"
    cfg2.smoke_test = True
    model2 = ConstantState([1.5, 0.0, 0.0])
    result = two.evaluate_model_any(model2, "test", cfg2)
    assert result["smoke_test"] is True
    assert result["prediction_finite"] is True
    assert result["reference_evaluation_skipped"] is True
    assert "state_scaled_space_time_rel_l2" not in result
    assert "state_scaled_final_rel_l2" not in result
    assert result["prediction_points"] == 48
    assert model2.training
    bad = ConstantState([float("nan"), 0.0, 0.0])
    try:
        two.evaluate_model_any(bad, "test", cfg2)
    except FloatingPointError:
        pass
    else:
        raise AssertionError("Non-finite predictions were accepted")
    cfg2.smoke_test = False
    try:
        two.evaluate_model_any(model2, "test", cfg2)
    except ValueError:
        pass
    else:
        raise AssertionError("Full 2D evaluation accepted a missing reference")
    sentinel_ref = object()
    with patch.object(two, "evaluate_model", return_value={"full_path_sentinel": True}) as evaluator:
        result = two.evaluate_model_any(model2, "test", cfg2, ref=sentinel_ref)
        evaluator.assert_called_once_with(model2, "test", sentinel_ref, cfg2)
        assert result["full_path_sentinel"] and not result["reference_evaluation_skipped"]
    # Production evaluation must reject an arbitrary reference file.
    with tempfile.TemporaryDirectory(prefix="trgpinn_reference_check_") as tmp:
        bad_file = Path(tmp) / "wrong_reference.npz"
        bad_file.write_bytes(b"not-the-canonical-reference")
        try:
            load_reference("shallowwater_2d", bad_file)
        except AssertionError:
            pass
        else:
            raise AssertionError("Production FV1024 hash validation was bypassed")
    assert FV1024_SHA256 == "9df6008a0562c9de764fe48c4653a682a99912fe95983f1faff9d4bbf759572c"
    return {"status": "pass", "exact_stoker_metrics_match": True,
            "evaluation_preserves_parameters": True, "model_mode_restored": True,
            "internal_errors_propagate": True, "nonfinite_smoke_predictions_rejected": True,
            "full_reference_path_preserved": True, "fv1024_hash_check_preserved": True}


def main() -> int:
    from baselines.training import METHODS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=("all", "shallowwater_1d", "shallowwater_2d"), default="all")
    parser.add_argument("--method", choices=("all", *METHODS), default="all")
    parser.add_argument("--output-root", type=Path, default=ROOT / "runs" / "runtime_checks")
    parser.add_argument("--timeout", type=float, default=180.0, help="Seconds per reduced training command.")
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    out = args.output_root.expanduser().resolve()
    from trgpinn.utils import ensure_unprotected_output
    for protected in (ROOT / "artifacts", ROOT / "results"):
        ensure_unprotected_output(out, protected)
    out.mkdir(parents=True, exist_ok=True)
    checks = adapter_checks()
    benchmarks = ("shallowwater_1d", "shallowwater_2d") if args.benchmark == "all" else (args.benchmark,)
    methods = METHODS if args.method == "all" else (args.method,)
    env = dict(os.environ)
    env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MPLBACKEND="Agg")
    env.pop("PYTHONPATH", None)  # CLI must find src/ without custom shell setup.
    rows = []
    for benchmark in benchmarks:
        for method in methods:
            log = out / f"{benchmark}_{method}.log"
            start = time.monotonic()
            row = {"benchmark": benchmark, "method": method}
            with tempfile.TemporaryDirectory(prefix="trgpinn_runtime_") as tmp:
                output = Path(tmp) / "outputs"
                command = [sys.executable, str(ROOT / "scripts" / "train_baseline.py"),
                           "--benchmark", benchmark, "--method", method, "--seed", "2026",
                           "--device", "cpu", "--smoke-test", "--output-root", str(output)]
                try:
                    with log.open("w", encoding="utf-8") as handle:
                        proc = subprocess.run(command, cwd=ROOT, env=env, stdout=handle,
                                              stderr=subprocess.STDOUT, timeout=args.timeout)
                    assert proc.returncode == 0, f"Exit {proc.returncode}; see {log.name}"
                    files = list(output.rglob("metrics_final.json"))
                    assert len(files) == 1, f"Expected one saved metric file, found {len(files)}"
                    assert "smoke_tests" in files[0].relative_to(output).parts
                    metric = json.loads(files[0].read_text())
                    assert metric["smoke_test"] is True
                    assert metric["adam_total_iters"] == 2
                    assert (files[0].parent / "_SUCCESS").is_file()
                    assert (files[0].parent / "model_final.pt").is_file()
                    if benchmark == "shallowwater_1d":
                        assert metric["reference_type"] == "exact_stoker_entropy_solution"
                        for key in ("state_scaled_space_time_rel_l2", "state_scaled_final_rel_l2"):
                            assert math.isfinite(metric[key]), key
                    else:
                        assert metric["prediction_finite"] is True
                        assert metric["reference_evaluation_skipped"] is True
                        assert metric["evaluation_mode"] == "prediction_only_no_reference"
                        assert "state_scaled_space_time_rel_l2" not in metric
                    row.update(status="pass", evaluation_mode=metric["evaluation_mode"])
                except (AssertionError, KeyError, subprocess.TimeoutExpired) as exc:
                    row.update(status="fail", error=str(exc))
            row["elapsed_seconds"] = round(time.monotonic() - start, 3)
            rows.append(row)
            print(f"[{row['status'].upper()}] {benchmark} / {method}", flush=True)
    report = {"scope": "Reduced CPU execution and adapter tests; not manuscript-accuracy validation.",
              "adapter_checks": checks, "runs": rows,
              "passed": sum(row["status"] == "pass" for row in rows), "total": len(rows)}
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"{report['passed']}/{report['total']} reduced runs passed. Report: {out / 'report.json'}")
    return 0 if report["passed"] == report["total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
