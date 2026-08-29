# TRG-PINN

**Trace-Ratio-Gated Physics-Informed Neural Networks for Hyperbolic Conservation Laws with Discontinuous Solutions**

Authors: **Ingyun Kang** (first author) and **Eunho Koo** (corresponding author)

TRG-PINN is a strong-form physics-informed neural-network method for
hyperbolic conservation laws with discontinuous solutions. It compares network
traces at the spatial scales `h` and `2h`, constructs a nontrainable soft gate
from cross-scale persistence and relative variation magnitude, and attenuates
the pointwise PDE residual only near likely discontinuities.

The method does not add a weak formulation, numerical flux, artificial
viscosity, prescribed shock trajectory, Rankine–Hugoniot loss, or a trainable
shock detector.

## Repository status

This package is prepared for a **private pre-submission GitHub repository**.

```text
Owner: ingyunee
Planned repository: TRG-PINN
Visibility: PRIVATE
Public URL: PENDING
```

No GitHub upload is performed by the release notebook. The author uploads the
curated ZIP manually and may change repository visibility to public later.

## Verified scope

The release contains six benchmark implementations:

- 1D inviscid Burgers equation
- 1D Euler Sod shock tube
- 1D shallow-water Stoker problem
- 2D oblique Burgers Riemann problem
- 2D rotated Euler Sod problem
- 2D circular shallow-water dam break

The canonical comparison uses the fixed seeds
`[2026, 7, 42, 100, 31415]`. Across the six benchmarks, TRG-PINN reduces the
paired vanilla-PINN primary space-time error by **26.3%–63.8%**, records
**30/30 paired seed-level wins**, ranks first on four benchmarks and second on
two, and has a mean rank of **1.33** among ten methods.

## Repository layout

```text
configs/                 benchmark configurations
src/trgpinn/             TRG-PINN and benchmark implementations
baselines/               specialized baseline registry and adapters
scripts/                 train, evaluate, reproduce, and verify commands
environment/             portable and reported environments
results/                 metrics, figures, and comparison outputs
artifacts/manifests/     SHA-256 artifact manifests
artifacts/reported_baselines/
                         sanitized specialized-baseline metadata
release/                  private-upload and external-artifact instructions
docs/                    reproducibility and artifact documentation
notebooks/demo.ipynb     lightweight release walkthrough
```

## Installation

```bash
conda env create -f environment/environment.yml
conda activate trgpinn
export PYTHONPATH="$PWD/src:$PWD"
```

or:

```bash
python -m pip install -r environment/requirements.txt
export PYTHONPATH="$PWD/src:$PWD"
```

The reported environment used Python 3.10.20, PyTorch 2.0.1 with CUDA 11.7,
NumPy 1.26.4, pandas 2.3.3, SciPy 1.15.3, and Matplotlib 3.10.9.

## Verification

This private source package is self-contained for source, metadata, and
comparison verification:

```bash
python scripts/verify_source_release.py
```

The broader release command automatically performs source checks and uses the
external-artifact verifier only when the external artifact tree is installed:

```bash
python scripts/verify_release.py
```

Component checks available without external binaries:

```bash
python scripts/verify_baselines.py --repo-root .
python scripts/reproduce_benchmark_comparison.py
```

After installing the external checkpoint/cache/reference archive at its
manifest-relative paths, run:

```bash
python scripts/verify_artifacts.py --benchmark all
```

## Training examples

```bash
python scripts/train.py \
  --benchmark burgers_1d \
  --seed 2026 \
  --device cuda:0
```

```bash
python scripts/train_baseline.py \
  --benchmark burgers_1d \
  --method aaf_pinn \
  --seed 2026 \
  --device cpu \
  --smoke-test \
  --overwrite
```

Fresh runs are written under `runs/reproduction/`. Reported artifacts are
immutable.

## Manuscript mapping

The ten-method comparison matrix is plotted as **Figure 2 in the current
manuscript**. The files under `results/benchmark_comparison/` provide the same
matrix as CSV and LaTeX audit products.

The earlier internal filename `table1_final` is retained only as build
provenance. In the current manuscript, **Table 1 is the principal-notation
table**, not the ten-method performance matrix.

## Reporting authority

Canonical reporting values are:

```text
results/reported_metrics/all_metrics_final.csv
seed-level metrics_final.json files
```

Checkpoint `extra` dictionaries contain historical training-time evaluation
metadata and are not the final reporting authority.

The 2D shallow-water reporting reference is the second-order FV1024 artifact
with SHA-256:

```text
9df6008a0562c9de764fe48c4653a682a99912fe95983f1faff9d4bbf759572c
```

## gPINN disclosure

The frozen gPINN-sub. checkpoints and metrics are authoritative reported
artifacts. The saved gPINN method block is preserved exactly. Two helper
definitions were absent from the final source notebooks; public fresh-run
helpers were reconstructed from the benchmark PDE residuals, smoke-tested, and
explicitly disclosed. These reconstructed helpers are not used to produce the
reported comparison matrix.

## Internal naming

Historical internal variable names, checkpoint keys, or console strings may
retain `tpinn`/`tPINN` for artifact compatibility. Manuscript-facing and
repository-facing method names use **TRG-PINN**. See
`docs/INTERNAL_NAMING.md`.

## Citation

See `CITATION.cff`. Add the article DOI and software/Zenodo DOI after
assignment.

## License status

Software license: **PENDING before public release**. See `docs/LICENSE_PENDING.md`.
