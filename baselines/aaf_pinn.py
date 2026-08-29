from .base import BaselineSpec


SPEC = BaselineSpec(
    key="aaf_pinn",
    public_name="AAF-PINN",
    artifact_method="AAF-PINN",
    method_folder="aaf_pinn",
    category="adaptive_activation",
    implementation_variant=(
        "Global trainable activation slope used by the reported artifacts."
    ),
)
