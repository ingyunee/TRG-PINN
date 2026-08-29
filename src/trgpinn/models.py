"""Neural-network backbones used by TRG-PINN benchmarks."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


def activation_class(name: str) -> type[nn.Module]:
    normalized = str(name).strip().lower()
    if normalized == "tanh":
        return nn.Tanh
    if normalized == "silu":
        return nn.SiLU
    if normalized == "relu":
        return nn.ReLU
    raise ValueError(f"Unsupported activation: {name}")


class CoordinateMLP(nn.Module):
    """Fully connected coordinate network with affine input normalization.

    The sequential layer layout intentionally preserves the state-dict keys used
    by the manuscript checkpoints (``net.0.weight``, ``net.2.weight``, ...).
    """

    def __init__(
        self,
        *,
        input_dimension: int,
        output_dimension: int,
        hidden_width: int,
        hidden_layers: int,
        activation: str,
        lower_bounds: Sequence[float],
        upper_bounds: Sequence[float],
        output_bias: Sequence[float] | float | None = None,
    ) -> None:
        super().__init__()

        if input_dimension <= 0 or output_dimension <= 0:
            raise ValueError("Input and output dimensions must be positive.")
        if hidden_width <= 0 or hidden_layers <= 0:
            raise ValueError("Hidden width and layer count must be positive.")
        if len(lower_bounds) != input_dimension or len(upper_bounds) != input_dimension:
            raise ValueError("Coordinate bounds do not match input dimension.")

        lower = tuple(float(value) for value in lower_bounds)
        upper = tuple(float(value) for value in upper_bounds)
        if any(high <= low for low, high in zip(lower, upper)):
            raise ValueError("Every upper coordinate bound must exceed its lower bound.")

        self.input_dimension = int(input_dimension)
        self.output_dimension = int(output_dimension)
        self.lower_bounds = lower
        self.upper_bounds = upper

        act = activation_class(activation)
        layers: list[nn.Module] = []
        dimension = self.input_dimension

        for _ in range(int(hidden_layers)):
            layer = nn.Linear(dimension, int(hidden_width))
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)
            layers.extend([layer, act()])
            dimension = int(hidden_width)

        output_layer = nn.Linear(dimension, self.output_dimension)
        nn.init.xavier_normal_(output_layer.weight)
        nn.init.zeros_(output_layer.bias)

        if output_bias is not None:
            bias_values = torch.as_tensor(output_bias, dtype=output_layer.bias.dtype).reshape(-1)
            if bias_values.numel() == 1 and self.output_dimension > 1:
                bias_values = bias_values.repeat(self.output_dimension)
            if bias_values.numel() != self.output_dimension:
                raise ValueError("Output-bias dimension mismatch.")
            with torch.no_grad():
                output_layer.bias.copy_(bias_values)

        layers.append(output_layer)
        self.net = nn.Sequential(*layers)

    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        if coordinates.ndim != 2 or coordinates.shape[1] != self.input_dimension:
            raise ValueError(
                f"Expected coordinates with shape (N, {self.input_dimension}); "
                f"received {tuple(coordinates.shape)}."
            )

        normalized_components = []
        for index, (lower, upper) in enumerate(zip(self.lower_bounds, self.upper_bounds)):
            value = coordinates[:, index : index + 1]
            normalized_components.append(2.0 * (value - lower) / (upper - lower) - 1.0)

        return self.net(torch.cat(normalized_components, dim=1))


def trainable_parameter_count(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))


def inverse_softplus_scalar(value: float) -> float:
    """Stable scalar inverse of softplus used for positive-output bias initialization."""

    import math

    positive = max(float(value), 1.0e-10)
    return math.log(math.expm1(positive))


class PrimitiveEulerMLP1D(nn.Module):
    """1D Euler coordinate network with primitive-variable outputs.

    The module names and layer ordering preserve the manuscript checkpoint keys:
    ``trunk.0.weight``, ..., ``trunk.10.bias``, ``out.weight``, and ``out.bias``.
    Density and pressure are mapped through softplus plus fixed floors.
    """

    def __init__(
        self,
        *,
        x_min: float,
        x_max: float,
        t_min: float,
        t_max: float,
        hidden_width: int,
        hidden_layers: int,
        activation: str,
        rho_floor: float,
        p_floor: float,
        left_state: Sequence[float],
        right_state: Sequence[float],
    ) -> None:
        super().__init__()

        if float(x_max) <= float(x_min) or float(t_max) <= float(t_min):
            raise ValueError("Euler coordinate bounds are invalid.")
        if len(left_state) != 3 or len(right_state) != 3:
            raise ValueError("Euler primitive states must be [rho, u, p].")

        self.x_min = float(x_min)
        self.x_max = float(x_max)
        self.t_min = float(t_min)
        self.t_max = float(t_max)
        self.rho_floor = float(rho_floor)
        self.p_floor = float(p_floor)

        act = activation_class(activation)
        layers: list[nn.Module] = []
        dimension = 2

        for _ in range(int(hidden_layers)):
            layer = nn.Linear(dimension, int(hidden_width))
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)
            layers.extend([layer, act()])
            dimension = int(hidden_width)

        self.trunk = nn.Sequential(*layers)
        self.out = nn.Linear(dimension, 3)
        nn.init.xavier_normal_(self.out.weight)
        nn.init.zeros_(self.out.bias)

        rho_left, u_left, p_left = (float(value) for value in left_state)
        rho_right, u_right, p_right = (float(value) for value in right_state)
        rho_mean = 0.5 * (rho_left + rho_right)
        p_mean = 0.5 * (p_left + p_right)

        with torch.no_grad():
            self.out.bias[0].fill_(
                inverse_softplus_scalar(rho_mean - self.rho_floor)
            )
            self.out.bias[1].fill_(0.5 * (u_left + u_right))
            self.out.bias[2].fill_(
                inverse_softplus_scalar(p_mean - self.p_floor)
            )

    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        if coordinates.ndim != 2 or coordinates.shape[1] != 2:
            raise ValueError(
                "PrimitiveEulerMLP1D expects coordinates with shape (N, 2)."
            )

        from torch.nn import functional as F

        x = coordinates[:, 0:1]
        t = coordinates[:, 1:2]
        x_normalized = (
            2.0 * (x - self.x_min) / (self.x_max - self.x_min) - 1.0
        )
        t_normalized = (
            2.0 * (t - self.t_min) / (self.t_max - self.t_min) - 1.0
        )
        raw = self.out(self.trunk(torch.cat([x_normalized, t_normalized], dim=1)))
        rho = self.rho_floor + F.softplus(raw[:, 0:1])
        velocity = raw[:, 1:2]
        pressure = self.p_floor + F.softplus(raw[:, 2:3])
        return torch.cat([rho, velocity, pressure], dim=1)


class ConservativeShallowWaterMLP1D(nn.Module):
    """1D shallow-water coordinate network with conservative outputs ``(h, q)``.

    The module names and layer ordering preserve the manuscript checkpoint keys:
    ``trunk.0.weight``, ..., ``trunk.10.bias``, ``out.weight``, and ``out.bias``.
    Water depth is mapped through softplus plus a fixed floor.
    """

    def __init__(
        self,
        *,
        x_min: float,
        x_max: float,
        t_min: float,
        t_max: float,
        hidden_width: int,
        hidden_layers: int,
        activation: str,
        h_floor: float,
        left_state: Sequence[float],
        right_state: Sequence[float],
    ) -> None:
        super().__init__()

        import math

        if float(x_max) <= float(x_min) or float(t_max) <= float(t_min):
            raise ValueError("Shallow-water coordinate bounds are invalid.")
        if len(left_state) != 2 or len(right_state) != 2:
            raise ValueError("Shallow-water states must be [h, q].")

        self.x_min = float(x_min)
        self.x_max = float(x_max)
        self.t_min = float(t_min)
        self.t_max = float(t_max)
        self.h_floor = float(h_floor)

        act = activation_class(activation)
        layers: list[nn.Module] = []
        dimension = 2

        for _ in range(int(hidden_layers)):
            layer = nn.Linear(dimension, int(hidden_width))
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)
            layers.extend([layer, act()])
            dimension = int(hidden_width)

        self.trunk = nn.Sequential(*layers)
        self.out = nn.Linear(dimension, 2)
        nn.init.xavier_normal_(self.out.weight)
        nn.init.zeros_(self.out.bias)

        h_left, q_left = (float(value) for value in left_state)
        h_right, q_right = (float(value) for value in right_state)
        h_mean = 0.5 * (h_left + h_right)
        q_mean = 0.5 * (q_left + q_right)

        with torch.no_grad():
            raw_h_bias = math.log(
                math.exp(max(h_mean - self.h_floor, 1.0e-8)) - 1.0
            )
            self.out.bias[0].fill_(raw_h_bias)
            self.out.bias[1].fill_(q_mean)

    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        if coordinates.ndim != 2 or coordinates.shape[1] != 2:
            raise ValueError(
                "ConservativeShallowWaterMLP1D expects coordinates with shape (N, 2)."
            )

        from torch.nn import functional as F

        x = coordinates[:, 0:1]
        t = coordinates[:, 1:2]
        x_normalized = (
            2.0 * (x - self.x_min) / (self.x_max - self.x_min) - 1.0
        )
        t_normalized = (
            2.0 * (t - self.t_min) / (self.t_max - self.t_min) - 1.0
        )
        raw = self.out(
            self.trunk(torch.cat([x_normalized, t_normalized], dim=1))
        )
        depth = self.h_floor + F.softplus(raw[:, 0:1])
        discharge = raw[:, 1:2]
        return torch.cat([depth, discharge], dim=1)
