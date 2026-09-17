# TRG-PINN

Code and reported results for **Trace-Ratio-Gated Physics-Informed Neural Networks for Hyperbolic Conservation Laws with Discontinuous Solutions**.

**Ingyun Kang and Eunho Koo** · Chonnam National University  
Corresponding author: Eunho Koo, kooeunho@jnu.ac.kr

TRG-PINN compares neural-network state variations at spatial scales `h` and `2h` to construct a non-trainable gate. The gate locally reweights the strong-form PDE loss; the initial- and boundary-condition losses are unchanged.

## Installation

Run the commands below from the repository root. The supplied environment uses Python 3.10, PyTorch 2.0.1, and CUDA 11.7.

```bash
conda env create -f environment/environment.yml
conda activate trgpinn
```

For an existing Python 3.10 environment, dependencies are also listed in `environment/requirements.txt`. The command-line scripts locate the local source package automatically.

## Quick start

Check the paired PINN/TRG-PINN training path with a short CPU run:

```bash
python scripts/train.py --benchmark burgers_1d --seed 2026 --device cpu --smoke-test
```

Run the full paired experiment:

```bash
python scripts/train.py --benchmark burgers_1d --seed 2026 --device cuda:0
```

Run a specialized baseline:

```bash
python scripts/train_baseline.py --benchmark shallowwater_1d --method aaf_pinn --seed 2026 --device cuda:0
```

Use `--device cpu` for CPU execution and `--help` to list the available options. New runs are stored under `runs/reproduction/`, separately from the reported results. Baseline smoke runs use the `smoke_tests/` subdirectory and reduced training settings; they do not measure manuscript accuracy.

## Benchmarks and methods

| Equations | One-dimensional problem | Two-dimensional problem |
|---|---|---|
| Inviscid Burgers | Riemann shock (`burgers_1d`) | Oblique planar shock (`burgers_2d`) |
| Compressible Euler | Sod shock tube (`euler_1d`) | Rotated Sod problem (`euler_2d`) |
| Shallow water | Stoker wet-bed dam break (`shallowwater_1d`) | Circular wet-bed dam break (`shallowwater_2d`) |

The comparison includes PINN, AAF-PINN, LRA-PINN, SA-PINN, RAD-PINN, RAR-D-PINN, gPINN-sub, cPINN, XPINN, and TRG-PINN. The five reporting seeds are `2026, 7, 42, 100, 31415`. Benchmark configurations are in `configs/`; specialized baseline settings are documented in `baselines/` and encoded in `baselines/training/`.

## Reported results

The seed-level results are stored in [`results/reported_metrics/all_metrics_final.csv`](results/reported_metrics/all_metrics_final.csv). To regenerate the comparison CSV and LaTeX tables:

```bash
python scripts/reproduce_benchmark_comparison.py
```

The outputs are written to `results/benchmark_comparison/`. This command aggregates saved metrics without retraining; it does not redraw Figure 2. Means and sample standard deviations are computed across five seeds, and ranks use unrounded means.

## Reference solutions and artifacts

The Burgers and Euler benchmarks use analytical references. The 1D shallow-water baseline evaluator uses the exact Stoker solution. Full 2D shallow-water evaluation requires the separately supplied second-order FV1024 reference:

```bash
python scripts/train_baseline.py --benchmark shallowwater_2d --method aaf_pinn --seed 2026 --device cuda:0 --reference-path /path/to/fv1024.npz
```

Replace `/path/to/fv1024.npz` with the local reference path. The file is checked against its recorded SHA-256 digest. A 2D shallow-water smoke test needs no reference, but reports prediction checks rather than reference-based errors.

The 300 reported final checkpoints and the FV1024 reference are available in [release v1.0.0](https://github.com/ingyunee/TRG-PINN/releases/tag/v1.0.0). Download the artifact-setup addendum from the same release and extract it into the repository root before checkpoint-based reevaluation. The addendum supplies the original per-run configurations and a hash-verifying installation helper; follow the release notes or `docs/RELEASE_ARTIFACT_SETUP.md` inside the addendum. Intermediate checkpoints and historical binary prediction caches are not included.

## Verification

Check source syntax, reporting metadata, and comparison-table consistency:

```bash
python scripts/verify_source_release.py
```

Test all eight specialized baselines on both shallow-water benchmarks using reduced CPU runs:

```bash
python scripts/test_shallowwater_runtime.py
```

The second command checks training, evaluation routing, and saved outputs. It does not reproduce full-budget accuracy; the 2D checks do not use FV1024. Further qualifications, including the reconstructed fresh-run gPINN helpers and historical checkpoint metadata, are documented in [`docs/KNOWN_LIMITATIONS.md`](docs/KNOWN_LIMITATIONS.md).

## Repository layout

```text
configs/          Benchmark configurations
src/trgpinn/      PINN/TRG-PINN implementations and evaluation
baselines/        Specialized baseline implementations
scripts/          Training, evaluation, reproduction, and checks
environment/      Dependencies and recorded environment
results/          Reported metrics, figures, and comparison tables
artifacts/        Run metadata and artifact manifests
```

Citation metadata is provided in [`CITATION.cff`](CITATION.cff). A software license has not yet been assigned; see [`docs/LICENSE_PENDING.md`](docs/LICENSE_PENDING.md).
