# Reproducibility contract

## Canonical reporting source

```text
all_metrics_final.csv
        ↕
seed-level metrics_final.json
```

The comparison uses benchmark-specific primary metrics and reports the
five-seed mean and sample standard deviation (`ddof=1`). Ranks are computed from
unrounded means.

## Frozen artifacts

The six PINN/TRG-PINN benchmark implementations passed checkpoint, metric,
cache, figure, and smoke-test parity. The eight specialized baselines contain
240 reported runs whose canonical metrics agree with the master table.

## Fresh training

Fresh training is not expected to be bitwise identical across arbitrary CUDA
devices. Frozen reported artifacts are the exact source for the paper numbers.

## Provenance qualifications

- Valid-mask corrections changed historical post-training PINN gate
  diagnostics but did not change vanilla-PINN training.
- Checkpoint `extra` metrics may predate final reference/metric reevaluation.
- The canonical 2D shallow-water reference is FV1024.
- Historical pre-valid-mask 2D snapshots are not used for reporting.
