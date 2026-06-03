"""S2: acrobot / double pendulum (chaotic, lambda_1 > 0)."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import warp as wp

from predictability_horizon.systems import System, register
from predictability_horizon.warpsim import rollout


@wp.func
def _accels(
    th1: float,
    th2: float,
    w1: float,
    w2: float,
    m1: float,
    m2: float,
    l1: float,
    l2: float,
    g: float,
) -> tuple[float, float]:
    d = th2 - th1
    den1 = (m1 + m2) * l1 - m2 * l1 * wp.cos(d) * wp.cos(d)
    a1 = (
        m2 * l1 * w1 * w1 * wp.sin(d) * wp.cos(d)
        + m2 * g * wp.sin(th2) * wp.cos(d)
        + m2 * l2 * w2 * w2 * wp.sin(d)
        - (m1 + m2) * g * wp.sin(th1)
    ) / den1
    den2 = (l2 / l1) * den1
    a2 = (
        -m2 * l2 * w2 * w2 * wp.sin(d) * wp.cos(d)
        + (m1 + m2) * g * wp.sin(th1) * wp.cos(d)
        - (m1 + m2) * l1 * w1 * w1 * wp.sin(d)
        - (m1 + m2) * g * wp.sin(th2)
    ) / den2
    return a1, a2


@wp.kernel
def acrobot_step(
    states: wp.array2d(dtype=wp.float32),  # type: ignore[valid-type]
    actions: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
    params: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
    dt: float,
    t: int,
) -> None:
    th1 = states[t, 0]
    th2 = states[t, 1]
    w1 = states[t, 2]
    w2 = states[t, 3]
    m1 = params[0]
    m2 = params[1]
    l1 = params[2]
    l2 = params[3]
    g = params[4]
    a1, a2 = _accels(th1, th2, w1, w2, m1, m2, l1, l2, g)
    w1n = w1 + dt * a1
    w2n = w2 + dt * (a2 + actions[t])  # optional torque on joint 2
    states[t + 1, 0] = th1 + dt * w1n
    states[t + 1, 1] = th2 + dt * w2n
    states[t + 1, 2] = w1n
    states[t + 1, 3] = w2n


def _jacobian(
    state: npt.NDArray[np.float64],
    u: float,
    params: npt.NDArray[np.float64],
    dt: float,
) -> npt.NDArray[np.float64]:
    """Central finite-difference Jacobian of one acrobot step.

    Uses eps=5e-3 so that the angular-velocity → position coupling (dt * eps)
    stays above float32 resolution (~3e-7 at values ~2.5).
    """
    eps = 5e-3
    n = 4
    J = np.zeros((n, n))  # noqa: N806
    for j in range(n):
        e = np.zeros(n)
        e[j] = eps
        sp = rollout(acrobot_step, state + e, np.array([u]), params, dt, 1)[1]
        sm = rollout(acrobot_step, state - e, np.array([u]), params, dt, 1)[1]
        J[:, j] = (sp - sm) / (2.0 * eps)
    return J


def _energy(state: npt.NDArray[np.float64], params: npt.NDArray[np.float64]) -> float:
    """Total mechanical energy of the double pendulum."""
    th1, th2, w1, w2 = state
    m1, m2, l1, l2, g = params
    ke = 0.5 * m1 * (l1 * w1) ** 2 + 0.5 * m2 * (
        (l1 * w1) ** 2 + (l2 * w2) ** 2 + 2.0 * l1 * l2 * w1 * w2 * np.cos(th1 - th2)
    )
    pe = -(m1 + m2) * g * l1 * np.cos(th1) - m2 * g * l2 * np.cos(th2)
    return float(ke + pe)


ACROBOT = register(
    System(
        name="acrobot",
        dim=4,
        default_params=np.array([1.0, 1.0, 1.0, 1.0, 9.81]),
        step_kernel=acrobot_step,
        jacobian=_jacobian,
        energy=_energy,
        suggested_dt=0.0005,
    )
)
