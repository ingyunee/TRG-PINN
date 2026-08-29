from .base import BaselineSpec


SPEC = BaselineSpec(
    key="rad_pinn",
    public_name="RAD-PINN",
    artifact_method="RAD-PINN",
    method_folder="rad_pinn",
    category="adaptive_sampling",
    implementation_variant=(
        "Residual-based adaptive distribution sampling used by the reported artifacts."
    ),
)
