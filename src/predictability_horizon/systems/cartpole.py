"""S1: cartpole (underactuated; the canonical RL benchmark)."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import warp as wp

from predictability_horizon.systems import System, register
from predictability_horizon.warpsim import rollout


@wp.kernel
def cartpole_step(
    states: wp.array2d(dtype=wp.float32),  # type: ignore[valid-type]
    actions: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
    params: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
    dt: float,
    t: int,
) -> None:
    x = states[t, 0]  # noqa: F841
    th = states[t, 1]
    v = states[t, 2]
    w = states[t, 3]
    mc = params[0]
    mp = params[1]
    l = params[2]  # noqa: E741
    g = params[3]
    f = actions[t]
    s = wp.sin(th)
    c = wp.cos(th)
    total = mc + mp
    temp = (f + mp * l * w * w * s) / total
    th_acc = (g * s - c * temp) / (l * (4.0 / 3.0 - mp * c * c / total))
    x_acc = temp - mp * l * th_acc * c / total
    vn = v + dt * x_acc
    wn = w + dt * th_acc
    states[t + 1, 0] = states[t, 0] + dt * vn
    states[t + 1, 1] = th + dt * wn
    states[t + 1, 2] = vn
    states[t + 1, 3] = wn


def _jacobian(
    state: npt.NDArray[np.float64],
    u: float,
    params: npt.NDArray[np.float64],
    dt: float,
) -> npt.NDArray[np.float64]:
    """Central finite-difference Jacobian of one cartpole step."""
    eps = 1e-4
    n = 4
    J = np.zeros((n, n))  # noqa: N806
    for j in range(n):
        e = np.zeros(n)
        e[j] = eps
        sp = rollout(cartpole_step, state + e, np.array([u]), params, dt, 1)[1]
        sm = rollout(cartpole_step, state - e, np.array([u]), params, dt, 1)[1]
        J[:, j] = (sp - sm) / (2.0 * eps)
    return J


def _energy(state: npt.NDArray[np.float64], params: npt.NDArray[np.float64]) -> float:
    """Mechanical energy consistent with the uniform-rod cartpole kernel.

    The kernel integrates the uniform-rod equations of motion (the
    ``4.0/3.0 - mp*c*c/total`` denominator).  This function computes the
    corresponding total mechanical energy:
      - cart translational KE: 0.5 mc v^2
      - pole CoM translational KE: 0.5 mp |v_cm|^2
      - pole rotational KE about CoM: 0.5 (mp pl^2 / 3) w^2   [uniform rod]
      - pole gravitational PE: mp g pl cos(th)
    """
    _x, th, v, w = state
    mc, mp, pl, g = params
    ke = 0.5 * mc * v * v + 0.5 * mp * ((v + pl * w * np.cos(th)) ** 2 + (pl * w * np.sin(th)) ** 2)
    ke += 0.5 * (mp * pl * pl / 3.0) * w * w
    pe = mp * g * pl * np.cos(th)
    return float(ke + pe)


CARTPOLE = register(
    System(
        name="cartpole",
        dim=4,
        default_params=np.array([1.0, 0.1, 0.5, 9.81]),
        step_kernel=cartpole_step,
        jacobian=_jacobian,
        energy=_energy,
        suggested_dt=0.02,
    )
)
