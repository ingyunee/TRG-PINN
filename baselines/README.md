# Specialized baseline methods

This directory describes the eight specialized comparison methods used in
addition to the vanilla PINN and TRG-PINN implementations provided by the six
benchmark modules.

The eight specialized baselines are:

- AAF-PINN
- LRA-PINN
- SA-PINN
- RAD-PINN
- RAR-D-PINN
- gPINN-sub.
- cPINN
- XPINN

`gPINN-sub.` is the manuscript-facing name. The frozen artifact method label is
`gPINN-subsampled` and the artifact folder is `gpinn_subsampled`.

The canonical reported metric source is each run's `metrics_final.json` together
with `results/reported_metrics/all_metrics_final.csv`. Numerical values embedded
inside `model_final.pt` under `extra` are retained as historical training-time
metadata and are audited separately; they are not required to equal later
canonical re-evaluations.

STEP 7A records frozen-artifact provenance and sanitized metadata without
copying checkpoint binaries into the Git-tracked tree. Canonical rows that
represent an unavailable or failed baseline run are preserved as such rather
than converted into synthetic results. Specialized executable training adapters
are handled separately in STEP 7B.
