from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BaselineSpec:
    key: str
    public_name: str
    artifact_method: str
    method_folder: str
    category: str
    implementation_variant: str
    single_network: bool = True

    def validate(self) -> None:
        for field in (
            "key",
            "public_name",
            "artifact_method",
            "method_folder",
            "category",
            "implementation_variant",
        ):
            value = getattr(self, field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"Invalid {field} for baseline {self.key!r}: {value!r}"
                )
