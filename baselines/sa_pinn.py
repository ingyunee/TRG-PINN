from .base import BaselineSpec


SPEC = BaselineSpec(
    key="sa_pinn",
    public_name="SA-PINN",
    artifact_method="SA-PINN",
    method_folder="sa_pinn",
    category="self_adaptive_weights",
    implementation_variant=(
        "Trainable pointwise self-adaptive weights used by the reported artifacts."
    ),
)
