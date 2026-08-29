from .aaf_pinn import SPEC as AAF_PINN
from .lra_pinn import SPEC as LRA_PINN
from .sa_pinn import SPEC as SA_PINN
from .rad_pinn import SPEC as RAD_PINN
from .rar_d_pinn import SPEC as RAR_D_PINN
from .gpinn_subsampled import SPEC as GPINN_SUB
from .cpinn import SPEC as CPINN
from .xpinn import SPEC as XPINN


BASELINE_SPECS = (
    AAF_PINN,
    LRA_PINN,
    SA_PINN,
    RAD_PINN,
    RAR_D_PINN,
    GPINN_SUB,
    CPINN,
    XPINN,
)


def validate_registry() -> None:
    if len(BASELINE_SPECS) != 8:
        raise AssertionError(
            f"Expected eight specialized baselines; found {len(BASELINE_SPECS)}"
        )

    for spec in BASELINE_SPECS:
        spec.validate()

    for attribute in (
        "key",
        "public_name",
        "artifact_method",
        "method_folder",
    ):
        values = [
            getattr(spec, attribute)
            for spec in BASELINE_SPECS
        ]
        if len(values) != len(set(values)):
            raise AssertionError(
                f"Duplicate baseline {attribute}: {values}"
            )


BY_KEY = {
    spec.key: spec
    for spec in BASELINE_SPECS
}
BY_PUBLIC_NAME = {
    spec.public_name: spec
    for spec in BASELINE_SPECS
}
BY_ARTIFACT_METHOD = {
    spec.artifact_method: spec
    for spec in BASELINE_SPECS
}
BY_METHOD_FOLDER = {
    spec.method_folder: spec
    for spec in BASELINE_SPECS
}


def get_baseline(identifier: str):
    for mapping in (
        BY_KEY,
        BY_PUBLIC_NAME,
        BY_ARTIFACT_METHOD,
        BY_METHOD_FOLDER,
    ):
        if identifier in mapping:
            return mapping[identifier]
    raise KeyError(
        f"Unknown specialized baseline: {identifier}"
    )


validate_registry()
