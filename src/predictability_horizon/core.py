"""Shared value types and small numerical utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class Trajectory:
    """A discrete state trajectory: states[t] is the state at step t."""

    states: npt.NDArray[np.float64]  # shape (T+1, dim)
    dt: float

    @property
    def horizon(self) -> int:
        """Number of integration steps (rows - 1)."""
        return int(self.states.shape[0]) - 1

    @property
    def dim(self) -> int:
        return int(self.states.shape[1])

    @property
    def duration(self) -> float:
        return self.horizon * self.dt


@dataclass(frozen=True)
class LyapunovResult:
    exponents: npt.NDArray[np.float64]  # per unit time, descending

    @property
    def largest(self) -> float:
        return float(self.exponents[0])


@dataclass(frozen=True)
class GradientLawResult:
    horizons: npt.NDArray[np.float64]
    grad_norms: npt.NDArray[np.float64]
    slope_per_step: float  # slope of log||grad|| vs horizon (steps)
    lambda1_per_step: float  # measured largest Lyapunov exponent * dt


def fit_loglinear_slope(
    x: npt.NDArray[np.float64], y: npt.NDArray[np.float64]
) -> tuple[float, float]:
    """Least-squares slope & intercept of log(y) vs x. Requires y > 0."""
    logy = np.log(np.asarray(y, dtype=float))
    slope, intercept = np.polyfit(np.asarray(x, dtype=float), logy, 1)
    return float(slope), float(intercept)
