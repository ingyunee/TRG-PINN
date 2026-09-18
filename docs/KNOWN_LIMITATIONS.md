# Reproducibility notes and limitations

## Reported results and fresh runs

The manuscript results are preserved in `results/reported_metrics/all_metrics_final.csv`
and the recorded per-run metadata. Checkpoint `extra` fields may contain earlier
evaluation values and are not a substitute for the final metrics.

Fresh training runs are separate experiments. Reduced execution tests do not
establish agreement with the reported full-budget accuracy, and results need not
be bitwise identical across hardware and software environments.

## Shallow-water baseline evaluation

The 1D shallow-water baseline adapter uses the exact Stoker evaluator in
`src/trgpinn/equations/shallowwater_1d.py`, including the state-scaled space-time
and final-time relative errors. No finite-volume reference tuple is required.
Legacy FV utilities remain in the extracted baseline module for provenance, but
are not called by the public baseline evaluation adapter.

Full 2D shallow-water baseline commands require the recorded FV1024 reference
through `--reference-path`; its hash is checked before training. A reference-free
2D smoke run checks finite predictions and saves the model without calculating
reference-based errors. These outputs are explicitly marked `smoke_test: true`
and `reference_evaluation_skipped: true`.

Baseline smoke outputs are placed under `OUTPUT_ROOT/smoke_tests/`, separately
from full runs. `scripts/test_shallowwater_runtime.py` checks all eight baseline
methods for both shallow-water benchmarks using short CPU runs. The production
FV1024 calculation and full-budget training are not covered by that test.

## gPINN helper reconstruction

The saved gPINN method blocks are retained. Two helper definitions were absent
from the final notebooks, so the public fresh-run helpers were reconstructed from
the benchmark PDE residuals. The reported gPINN results come from the recorded
experiments, not from these reconstructed helpers. See
`baselines/training/README.md` for the helper names and provenance.

## External artifacts

Reported model checkpoints, binary prediction caches, and the FV1024 reference
array are not included in the source package. Independent checkpoint reevaluation
and checkpoint-based figure regeneration require those artifacts. Paths and
hashes are listed in `artifacts/manifests/`.

The manuscript compares FV512 and FV1024 reference resolutions. This comparison
is a reference-sensitivity check, not a formal grid-convergence study. Improved
state reconstruction also does not imply improvement in every conservation or
directional-uniformity diagnostic.

## Verification scope

`verify_source_release.py` checks the stored comparison tables, citation metadata,
Python syntax, command-line help, and portable source paths. It does not train
models. Local runs and installed external artifacts are outside its source scan;
use `verify_artifacts.py` for available reported artifacts and
`test_shallowwater_runtime.py` for the shallow-water execution paths.

Existing manifests and parity reports document the original package preparation.
They are historical records and do not imply that later documentation changes
were independently retrained.
