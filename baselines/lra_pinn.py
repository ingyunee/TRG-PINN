from .base import BaselineSpec


SPEC = BaselineSpec(
    key="lra_pinn",
    public_name="LRA-PINN",
    artifact_method="LRA-PINN",
    method_folder="lra_pinn",
    category="adaptive_loss_balancing",
    implementation_variant=(
        "Gradient-statistics learning-rate annealing used by the reported artifacts."
    ),
)
