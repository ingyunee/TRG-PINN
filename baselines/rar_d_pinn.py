from .base import BaselineSpec


SPEC = BaselineSpec(
    key="rar_d_pinn",
    public_name="RAR-D-PINN",
    artifact_method="RAR-D-PINN",
    method_folder="rar_d_pinn",
    category="adaptive_sampling",
    implementation_variant=(
        "Residual-based adaptive refinement with distribution used by the reported artifacts."
    ),
)
