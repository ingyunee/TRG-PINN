from .base import BaselineSpec


SPEC = BaselineSpec(
    key="gpinn_subsampled",
    public_name="gPINN-sub.",
    artifact_method="gPINN-subsampled",
    method_folder="gpinn_subsampled",
    category="residual_gradient",
    implementation_variant=(
        "Gradient-enhanced PINN evaluated on the reported residual-gradient subset."
    ),
)
