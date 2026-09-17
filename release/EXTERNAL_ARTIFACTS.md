# External artifact policy

Large binary artifacts are distributed separately from the Git-tracked source.

The reported final checkpoints and the FV1024 reference used for the two-dimensional shallow-water benchmark are available as assets of GitHub release v1.0.0:

https://github.com/ingyunee/TRG-PINN/releases/tag/v1.0.0

Checkpoint-based reevaluation requires the artifact-setup addendum provided with the same release. The addendum supplies the original paired-run configuration files and installs the final checkpoints at the paths expected by the public evaluation scripts.

Intermediate checkpoints and historical binary prediction caches are not included. Artifact integrity is tracked using SHA-256 manifests.
