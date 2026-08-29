from .base import BaselineSpec


SPEC = BaselineSpec(
    key="cpinn",
    public_name="cPINN",
    artifact_method="cPINN",
    method_folder="cpinn",
    category="spatial_domain_decomposition",
    implementation_variant=(
        "Conservative spatial domain decomposition used by the reported artifacts."
    ),
    single_network=False,
)
