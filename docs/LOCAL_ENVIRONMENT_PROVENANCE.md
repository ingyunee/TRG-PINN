# Local environment provenance

STEP 0 captured several raw machine-specific environment files:

```text
environment/reported_runtime.json
environment/reported_pip_freeze.txt
environment/reported_conda_explicit.txt
environment/reported_conda_environment.yml
```

These files may contain absolute server-local paths such as the Python
executable or Conda environment prefix. They are retained only in the local
research workspace for provenance and are not part of the public GitHub
release.

Public portable environment information is provided through:

```text
environment/environment.yml
environment/requirements.txt
environment/reported_environment.txt
```

STEP 9A records SHA-256 hashes of the raw captures without rewriting their
contents. STEP 9B must exclude these raw captures from curated public staging.
