from .base import BaselineSpec


SPEC = BaselineSpec(
    key="xpinn",
    public_name="XPINN",
    artifact_method="XPINN",
    method_folder="xpinn",
    category="space_time_domain_decomposition",
    implementation_variant=(
        "Extended space-time domain decomposition used by the reported artifacts."
    ),
    single_network=False,
)
