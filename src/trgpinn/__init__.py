"""TRG-PINN public implementation."""

from .models import (
    ConservativeShallowWaterMLP1D,
    CoordinateMLP,
    PrimitiveEulerMLP1D,
)

__all__ = [
    "CoordinateMLP",
    "PrimitiveEulerMLP1D",
    "ConservativeShallowWaterMLP1D",
]
__version__ = "0.1.0"
PUBLIC_METHOD_NAME = "TRG-PINN"
