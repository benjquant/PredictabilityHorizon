"""Calibration-only dynamical systems with known Lyapunov spectra (pure NumPy)."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import numpy.typing as npt

Vec = npt.NDArray[np.float64]


def harmonic_map(omega: float, dt: float) -> tuple[Callable[[Vec], Vec], Callable[[Vec], Vec]]:
    """Exact rotation map for a harmonic oscillator. All exponents are 0.

    State is normalised: x = [q, p/omega], so the Jacobian is an orthogonal rotation
    and QR decomposition gives R = ±I exactly — log|diag R| = 0 at every step.
    """
    c, s = np.cos(omega * dt), np.sin(omega * dt)
    # In normalised coords (q, p/omega) the map is a pure rotation:
    rot = np.array([[c, s], [-s, c]])

    def step(x: Vec) -> Vec:
        return rot @ x

    def jac(_x: Vec) -> Vec:
        return rot

    return step, jac


def lorenz_rk4_map(
    dt: float, sigma: float = 10.0, rho: float = 28.0, beta: float = 8.0 / 3.0
) -> Callable[[Vec], Vec]:
    def f(s: Vec) -> Vec:
        x, y, z = s
        return np.array([sigma * (y - x), x * (rho - z) - y, x * y - beta * z])

    def step(s: Vec) -> Vec:
        k1 = f(s)
        k2 = f(s + 0.5 * dt * k1)
        k3 = f(s + 0.5 * dt * k2)
        k4 = f(s + dt * k3)
        return s + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    return step


def lorenz_jacobian_fd(step: Callable[[Vec], Vec], eps: float = 1e-6) -> Callable[[Vec], Vec]:
    """Central-difference Jacobian of a discrete map."""

    def jac(s: Vec) -> Vec:
        n = s.shape[0]
        J = np.zeros((n, n))  # noqa: N806
        for j in range(n):
            e = np.zeros(n)
            e[j] = eps
            J[:, j] = (step(s + e) - step(s - e)) / (2.0 * eps)
        return J

    return jac
