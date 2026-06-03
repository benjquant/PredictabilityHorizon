"""S0: single pendulum (integrable, lambda_1 = 0)."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import warp as wp

from predictability_horizon.systems import System, register


@wp.kernel
def pendulum_step(
    states: wp.array2d(dtype=wp.float32),  # type: ignore[valid-type]
    actions: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
    params: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
    dt: float,
    t: int,
) -> None:
    theta = states[t, 0]
    omega = states[t, 1]
    g = params[0]
    ell = params[1]
    acc = -(g / ell) * wp.sin(theta) + actions[t]
    omega_new = omega + dt * acc  # semi-implicit: update velocity first
    theta_new = theta + dt * omega_new
    states[t + 1, 0] = theta_new
    states[t + 1, 1] = omega_new


def _jacobian(
    state: npt.NDArray[Any], u: float, params: npt.NDArray[Any], dt: float
) -> npt.NDArray[Any]:
    theta, _omega = state
    g, ell = params
    # omega_new = omega + dt*(-(g/ell) sin theta + u);  theta_new = theta + dt*omega_new
    domega_dtheta = -dt * (g / ell) * np.cos(theta)
    domega_domega = 1.0
    dtheta_dtheta = 1.0 + dt * domega_dtheta
    dtheta_domega = dt * domega_domega
    return np.array([[dtheta_dtheta, dtheta_domega], [domega_dtheta, domega_domega]])


def _energy(state: npt.NDArray[Any], params: npt.NDArray[Any]) -> float:
    theta, omega = state
    g, ell = params
    return float(0.5 * ell * ell * omega * omega + g * ell * (1.0 - np.cos(theta)))  # per unit mass


PENDULUM = register(
    System(
        name="pendulum",
        dim=2,
        default_params=np.array([9.81, 1.0]),
        step_kernel=pendulum_step,
        jacobian=_jacobian,
        energy=_energy,
        suggested_dt=0.005,
    )
)
