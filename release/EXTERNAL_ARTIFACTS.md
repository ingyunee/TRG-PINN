# External artifact policy

The private GitHub source ZIP excludes large binary artifacts.

External materials include reported checkpoint binaries, binary NumPy caches,
specialized-baseline checkpoints, and the FV1024 2D shallow-water reference.

These can remain local until the authors decide how to provide reviewer or
public access. When an archival release is prepared, preserve the
manifest-relative paths and SHA-256 digests.

Canonical reporting remains:

```text
results/reported_metrics/all_metrics_final.csv
seed-level metrics_final.json
```
