"""Benchmark equations exposed by the public package."""

from .burgers_1d import Burgers1DConfig
from .euler_1d import Euler1DConfig
from .shallowwater_1d import ShallowWater1DConfig
from .burgers_2d import Burgers2DConfig
from .euler_2d import Euler2DConfig, Euler2DRotatedSodConfig
from .shallowwater_2d import ShallowWater2DConfig

__all__ = [
    "Burgers1DConfig",
    "Euler1DConfig",
    "ShallowWater1DConfig",
    "Burgers2DConfig",
    "Euler2DConfig",
    "Euler2DRotatedSodConfig",
    "ShallowWater2DConfig",
]
