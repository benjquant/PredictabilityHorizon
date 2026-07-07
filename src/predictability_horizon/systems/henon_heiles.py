"""S3: Hénon-Heiles — a smooth galactic potential, chaotic (lambda_1 > 0).

A unit-mass particle in the 2D potential V = 1/2 (x^2 + y^2) + lam (x^2 y - y^3/3)
(standard Hénon-Heiles at lam = 1). The Hamiltonian H = 1/2 (px^2 + py^2) + V is
separable, so semi-implicit (symplectic) Euler is *exactly* volume-preserving here
(det J = 1) — like the pendulum, unlike the non-separable acrobot. Canonical state
(x, y, px, py). Originally a model for a star's meridional motion in an axisymmetric
galaxy (Hénon & Heiles 1964, AJ 69, 73); chaotic for E above the KAM transition,
bounded for 0 < E < 1/6 (escape energy).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import warp as wp

from predictability_horizon.systems import System, register


@wp.kernel
def henon_heiles_step(
    states: wp.array2d(dtype=wp.float32),  # type: ignore[valid-type]
    actions: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
    params: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
    dt: float,
    t: int,
) -> None:
    x = states[t, 0]
    y = states[t, 1]
    px = states[t, 2]
    py = states[t, 3]
    lam = params[0]
    # forces = -grad V:  -dV/dx = -(x + 2 lam x y),  -dV/dy = -(y + lam (x^2 - y^2))
    fx = -(x + lam * 2.0 * x * y)
    fy = -(y + lam * (x * x - y * y))
    px_new = px + dt * (fx + actions[t])  # actions = control channel; zero in all experiments
    py_new = py + dt * fy
    states[t + 1, 0] = x + dt * px_new  # semi-implicit: momenta first, then positions
    states[t + 1, 1] = y + dt * py_new
    states[t + 1, 2] = px_new
    states[t + 1, 3] = py_new


def _jacobian(
    state: npt.NDArray[Any], u: float, params: npt.NDArray[Any], dt: float
) -> npt.NDArray[Any]:
    """Exact one-step Jacobian of the symplectic-Euler map (det = 1 identically).

    Rows/cols ordered (x, y, px, py). u enters px_new additively, so d/dstate is 0.
    """
    x, y, _px, _py = state
    lam = float(params[0])
    hxx = 1.0 + 2.0 * lam * y  # d^2 V / dx^2
    hxy = 2.0 * lam * x        # d^2 V / dx dy
    hyy = 1.0 - 2.0 * lam * y  # d^2 V / dy^2
    return np.array(
        [
            [1.0 - dt * dt * hxx, -dt * dt * hxy, dt, 0.0],
            [-dt * dt * hxy, 1.0 - dt * dt * hyy, 0.0, dt],
            [-dt * hxx, -dt * hxy, 1.0, 0.0],
            [-dt * hxy, -dt * hyy, 0.0, 1.0],
        ]
    )


def _energy(state: npt.NDArray[Any], params: npt.NDArray[Any]) -> float:
    x, y, px, py = state
    lam = float(params[0])
    ke = 0.5 * (px * px + py * py)
    pe = 0.5 * (x * x + y * y) + lam * (x * x * y - y * y * y / 3.0)
    return float(ke + pe)


def chaotic_sea_ic(
    energy: float, params: npt.NDArray[Any] | None = None
) -> npt.NDArray[np.float64]:
    """Symmetry-broken IC on the x=0 section at the given energy.

    Fixes (x, y, py) = (0, -0.12, 0.12) and solves px from H = energy. Breaking the
    x -> -x symmetry keeps the orbit off the symmetric periodic orbits, so for E above
    the KAM transition (~1/12) it lands in the chaotic sea (lambda_1 > 0); at small E
    the same IC is near-integrable (lambda_1 ~ 0). Each energy's measured lambda_1 is
    recorded in the verification ledger.
    """
    lam = 1.0 if params is None else float(params[0])
    y, py = -0.12, 0.12
    v = 0.5 * y * y + lam * (-(y**3) / 3.0)  # V(0, y)
    px_sq = 2.0 * (energy - v) - py * py
    if px_sq < 0.0:
        raise ValueError(f"energy {energy} too low for section point (px^2={px_sq:.3g})")
    return np.array([0.0, y, float(np.sqrt(px_sq)), py])


HENON_HEILES = register(
    System(
        name="henon_heiles",
        dim=4,
        default_params=np.array([1.0]),  # coupling lambda = 1 (standard Hénon-Heiles)
        step_kernel=henon_heiles_step,
        jacobian=_jacobian,
        energy=_energy,
        suggested_dt=0.01,
    )
)
