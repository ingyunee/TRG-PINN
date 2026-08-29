"""Metric and parity helpers."""

from __future__ import annotations

from typing import Any

import numpy as np


def crossing_location(
    x: np.ndarray,
    values: np.ndarray,
    level: float,
    *,
    expected: float | None = None,
    window: float | None = None,
) -> float:
    x = np.asarray(x)
    values = np.asarray(values)

    mask = np.ones_like(x, dtype=bool)
    if expected is not None and window is not None:
        mask = np.abs(x - expected) <= window
        if mask.sum() < 2:
            mask = np.ones_like(x, dtype=bool)

    x_masked = x[mask]
    shifted = values[mask] - level
    candidates = np.where(shifted[:-1] * shifted[1:] <= 0.0)[0]

    if len(candidates) == 0:
        return float(x_masked[int(np.argmin(np.abs(shifted)))])

    if expected is not None:
        midpoints = 0.5 * (x_masked[candidates] + x_masked[candidates + 1])
        index = candidates[int(np.argmin(np.abs(midpoints - expected)))]
    else:
        index = candidates[0]

    x0, x1 = x_masked[index], x_masked[index + 1]
    y0, y1 = shifted[index], shifted[index + 1]
    if abs(y1 - y0) < 1.0e-14:
        return float(0.5 * (x0 + x1))
    return float(x0 - y0 * (x1 - x0) / (y1 - y0))


def relative_l2(prediction: np.ndarray, reference: np.ndarray, epsilon: float = 1.0e-12) -> float:
    prediction = np.asarray(prediction, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    error = prediction - reference
    return float(
        np.sqrt(np.mean(error**2)) / (np.sqrt(np.mean(reference**2)) + float(epsilon))
    )


def pointwise_median_rmse_band(
    predictions: np.ndarray,
    reference: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    predictions = np.asarray(predictions, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    median = np.median(predictions, axis=0)
    rmse = np.sqrt(np.mean((predictions - reference[None, ...]) ** 2, axis=0))
    return median, rmse, median - rmse, median + rmse


def compare_numeric_mappings(
    actual: dict[str, Any],
    expected: dict[str, Any],
    *,
    rtol: float,
    atol: float,
    ignored: set[str] | None = None,
) -> list[dict[str, Any]]:
    ignored = ignored or set()
    records: list[dict[str, Any]] = []
    for key, expected_value in expected.items():
        if key in ignored or key not in actual:
            continue
        try:
            left = float(actual[key])
            right = float(expected_value)
        except (TypeError, ValueError):
            continue
        if not (np.isfinite(left) and np.isfinite(right)):
            continue
        absolute_difference = abs(left - right)
        passed = bool(np.isclose(left, right, rtol=rtol, atol=atol))
        records.append(
            {
                "metric": key,
                "actual": left,
                "expected": right,
                "absolute_difference": absolute_difference,
                "passed": passed,
            }
        )
    return records
