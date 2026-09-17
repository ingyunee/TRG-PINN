# Artifact policy

## Git-tracked content

The public repository contains source, configurations, canonical metrics,
paper figures, SHA-256 manifests, and sanitized metadata.

## External artifacts

Specialized-baseline checkpoint binaries are described by:

```text
artifacts/manifests/baseline_artifact_manifest.csv
```

The 2D shallow-water FV1024 reference is described under:

```text
artifacts/reported/2d_shallowwater/reference/
```

External archives must preserve manifest-relative paths and SHA-256 digests.

## Integrity checks

For the Git-tracked source package:

```bash
python scripts/verify_source_release.py
```

After installing the external checkpoint/cache/reference archive:

```bash
python scripts/verify_artifacts.py --benchmark all
python scripts/verify_baselines.py --repo-root .
```


## Local-only environment captures

Raw STEP 0 environment captures may contain machine-local absolute paths.
They are intentionally excluded from the public release. See
`docs/LOCAL_ENVIRONMENT_PROVENANCE.md`.
