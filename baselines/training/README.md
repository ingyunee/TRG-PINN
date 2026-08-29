
# Executable specialized-baseline adapters

The six benchmark runtime modules contain the final benchmark code and the
exact eight specialized-baseline method blocks recovered from the final
experiment notebooks.

Seven method implementations are source-complete as saved. The gPINN-sub.
method block is also exact, but the saved notebooks reference two helper
functions whose definitions are absent:

- `gpinn_residual_scaled_components`
- `gpinn_gradient_loss_from_components`

The public runtime supplies benchmark-specific reconstructions derived from the
same PDE residual equations and records their SHA-256 values. This affects only
fresh public gPINN training. The frozen gPINN checkpoints and canonical
`metrics_final.json` values remain the paper's authoritative reported results.

Use `scripts/train_baseline.py` for fresh runs. Outputs are written below
`runs/reproduction/baselines` by default and are blocked from the frozen
artifact/result directories.
