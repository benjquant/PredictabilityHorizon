"""S2: acrobot / double pendulum (chaotic, lambda_1 > 0).

Canonical state (theta1, theta2, p1, p2): angles and their CONJUGATE MOMENTA p = M(theta)
omega, not angular velocities. The Hamiltonian H = 1/2 p^T M(theta)^-1 p + V(theta) is
non-separable (the kinetic term mixes theta and p through M(theta)^-1), so unlike the
pendulum or Hénon-Heiles, semi-implicit Euler is not an option here -- the step kernel is
implicit midpoint, solved by a FIXED count of _N_FP_ITER Picard passes so the map stays
straight-line (no state-dependent branching) and differentiable end to end under Warp's
reverse-mode AD. The one-step Jacobian has a closed form, a Cayley transform of J4 @ Hess H
evaluated at the converged midpoint, which is symplectic (det J = 1) to machine precision by
construction rather than to O(dt^2) truncation. The retired (theta, omega) explicit scheme
survives as `legacy_acrobot_step`, unregistered, solely to keep its ~21%/10s energy leak
reproducible."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import warp as wp

from predictability_horizon.systems import System, register

_N_FP_ITER = 6
"""Picard passes per implicit-midpoint step.

FIXED, not residual-triggered. A state-dependent trip count would make the map
piecewise-defined and its Jacobian discontinuous across the switching surfaces -- and this
repo measures Jacobian-derived observables (lambda_1, gradient growth). It also keeps the
kernel straight-line, which is what makes Warp reverse-mode AD through it work.

6 rather than 4: the Picard contraction rate is (dt/2) * L(E), so the binding constraint is
dt, not torque. 4 suffices at dt=5e-4 (every figure) but falls ~30x short at dt=5e-3 (the
trajopt tests). Beyond saturation extra passes are no-ops, which
test_fixed_point_saturates_at_six_passes verifies rather than assumes.
"""

_E2 = np.array([[0.0, 1.0], [1.0, 0.0]])  # exchange matrix; dM/dtheta_i is proportional to it
_J4 = np.block([[np.zeros((2, 2)), np.eye(2)], [-np.eye(2), np.zeros((2, 2))]])
"""Canonical structure matrix: _J4 @ grad H = (dH/dp, -dH/dq)."""


def _mass_matrix(q: npt.NDArray[np.float64], params: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """(2,) angles -> (2,2) mass matrix M(q).  p = M(q) omega."""
    m1, m2, l1, l2, _g = params
    c = np.cos(q[0] - q[1])
    return np.array([[(m1 + m2) * l1 * l1, m2 * l1 * l2 * c], [m2 * l1 * l2 * c, m2 * l2 * l2]])


def _dmass(q: npt.NDArray[np.float64], params: npt.NDArray[np.float64]) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """-> (dM/dtheta1, dM/dtheta2). Only the off-diagonal depends on d = theta1 - theta2."""
    _m1, m2, l1, l2, _g = params
    k = m2 * l1 * l2 * np.sin(q[0] - q[1])
    return -k * _E2, k * _E2


def _d2mass(q: npt.NDArray[np.float64], params: npt.NDArray[np.float64]) -> list[list[npt.NDArray[np.float64]]]:
    """-> nested list with d2M[i][j] = d^2 M / dtheta_i dtheta_j."""
    _m1, m2, l1, l2, _g = params
    k = -m2 * l1 * l2 * np.cos(q[0] - q[1])
    return [[k * _E2, -k * _E2], [-k * _E2, k * _E2]]


def _dpotential(q: npt.NDArray[np.float64], params: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    m1, m2, l1, l2, g = params
    return np.array([(m1 + m2) * g * l1 * np.sin(q[0]), m2 * g * l2 * np.sin(q[1])])


def _d2potential(q: npt.NDArray[np.float64], params: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    m1, m2, l1, l2, g = params
    return np.diag([(m1 + m2) * g * l1 * np.cos(q[0]), m2 * g * l2 * np.cos(q[1])])


def _canonical_energy(state: npt.NDArray[np.float64], params: npt.NDArray[np.float64]) -> float:
    """H(theta, p) = 1/2 p^T M(theta)^-1 p + V(theta)."""
    pars = np.asarray(params, dtype=np.float64)
    q = np.asarray(state[:2], dtype=np.float64)
    p = np.asarray(state[2:], dtype=np.float64)
    m1, m2, l1, l2, g = pars
    ke = 0.5 * p @ np.linalg.solve(_mass_matrix(q, pars), p)
    pe = -(m1 + m2) * g * l1 * np.cos(q[0]) - m2 * g * l2 * np.cos(q[1])
    return float(ke + pe)


def _grad_hamiltonian(z: npt.NDArray[np.float64], params: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """-> (4,) = (dH/dq1, dH/dq2, dH/dp1, dH/dp2).

    dH/dq_i = -1/2 u^T (dM/dq_i) u + dV/dq_i  with u = M^-1 p, using
    d(M^-1)/dq_i = -M^-1 (dM/dq_i) M^-1.   dH/dp = M^-1 p = omega.
    """
    pars = np.asarray(params, dtype=np.float64)
    q, p = np.asarray(z[:2], dtype=np.float64), np.asarray(z[2:], dtype=np.float64)
    u = np.linalg.solve(_mass_matrix(q, pars), p)
    dm = _dmass(q, pars)
    dhdq = np.array([-0.5 * u @ dm[i] @ u for i in range(2)]) + _dpotential(q, pars)
    return np.concatenate([dhdq, u])


def _hess_hamiltonian(z: npt.NDArray[np.float64], params: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """-> (4,4) symmetric Hessian of H, ordered (q1, q2, p1, p2).

    d2(M^-1)/dq_i dq_j = M^-1 [ (d_i M) M^-1 (d_j M) + (d_j M) M^-1 (d_i M) - d_ij M ] M^-1.
    """
    pars = np.asarray(params, dtype=np.float64)
    q, p = np.asarray(z[:2], dtype=np.float64), np.asarray(z[2:], dtype=np.float64)
    minv = np.linalg.inv(_mass_matrix(q, pars))
    dm = _dmass(q, pars)
    d2m = _d2mass(q, pars)
    d2v = _d2potential(q, pars)
    out = np.zeros((4, 4))
    out[2:, 2:] = minv
    for i in range(2):
        col = -(minv @ dm[i] @ minv) @ p
        out[i, 2:] = col
        out[2:, i] = col
    for i in range(2):
        for j in range(2):
            inner = dm[i] @ minv @ dm[j] + dm[j] @ minv @ dm[i] - d2m[i][j]
            out[i, j] = 0.5 * p @ (minv @ inner @ minv) @ p + d2v[i, j]
    return out


def _midpoint_np(
    z: npt.NDArray[np.float64],
    u: float,
    params: npt.NDArray[np.float64],
    dt: float,
    n_iter: int = _N_FP_ITER,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """float64 reference of one implicit-midpoint step. -> (z_next, zbar).

    zbar <- z + (dt/2) [J grad H(zbar) + F],  z_next = 2 zbar - z,  F = (0,0,0,u).
    """
    pars = np.asarray(params, dtype=np.float64)
    z0 = np.asarray(z, dtype=np.float64)
    force = np.array([0.0, 0.0, 0.0, float(u)])
    zbar = z0.copy()
    for _ in range(n_iter):
        g = _grad_hamiltonian(zbar, pars)
        zbar = z0 + 0.5 * dt * (np.concatenate([g[2:], -g[:2]]) + force)
    return 2.0 * zbar - z0, zbar


def _cayley_jacobian(
    state: npt.NDArray[np.float64], u: float, params: npt.NDArray[np.float64], dt: float
) -> npt.NDArray[np.float64]:
    """Analytic one-step Jacobian of implicit midpoint -- a Cayley transform.

    Differentiating z_next = z + dt [J grad H(zbar) + F] with zbar = (z + z_next)/2 gives
    (I - (dt/2) A) dz_next = (I + (dt/2) A) dz  with  A = J Hess H(zbar), so

        J_step = (I - (dt/2) A)^-1 (I + (dt/2) A)

    A is Hamiltonian, so this is symplectic identically: det J = 1 to machine precision, not
    to O(dt^2) and not on a finite-difference noise floor. The constant force F drops out of
    the derivative, so u affects the Jacobian only through zbar.
    """
    pars = np.asarray(params, dtype=np.float64)
    _znext, zbar = _midpoint_np(np.asarray(state, dtype=np.float64), u, pars, dt)
    a = _J4 @ _hess_hamiltonian(zbar, pars)
    eye = np.eye(4)
    j_step: npt.NDArray[np.float64] = np.linalg.solve(eye - 0.5 * dt * a, eye + 0.5 * dt * a).astype(
        np.float64
    )
    return j_step


def to_canonical(state: npt.NDArray[np.float64], params: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """(theta, omega) -> (theta, p). ICs with omega = 0 are unchanged, since p = 0 too."""
    pars = np.asarray(params, dtype=np.float64)
    q = np.asarray(state[:2], dtype=np.float64)
    return np.concatenate([q, _mass_matrix(q, pars) @ np.asarray(state[2:], dtype=np.float64)])


def to_legacy(z: npt.NDArray[np.float64], params: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """(theta, p) -> (theta, omega). Kept for anything that must be reported in velocities."""
    pars = np.asarray(params, dtype=np.float64)
    q = np.asarray(z[:2], dtype=np.float64)
    return np.concatenate(
        [q, np.linalg.solve(_mass_matrix(q, pars), np.asarray(z[2:], dtype=np.float64))]
    )


@wp.func
def _legacy_accels(
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
def legacy_acrobot_step(
    states: wp.array2d(dtype=wp.float32),  # type: ignore[valid-type]
    actions: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
    params: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
    dt: float,
    t: int,
) -> None:
    """RETIRED. The pre-2026-08 scheme, in (theta, omega). NOT REGISTERED -- do not use.

    Looked like symplectic Euler and was not one: the state held angular velocities rather
    than conjugate momenta, and the momentum kick used the OLD velocities (the Coriolis
    terms force this). The result was a fully explicit first-order scheme losing ~21% of the
    system's energy over 10 s. Kept solely so that leak stays reproducible -- see
    test_legacy_kernel_leaks_energy_as_documented. `actions[t]` here is a fictitious
    acceleration on omega_2, not a torque.
    """
    th1 = states[t, 0]
    th2 = states[t, 1]
    w1 = states[t, 2]
    w2 = states[t, 3]
    m1 = params[0]
    m2 = params[1]
    l1 = params[2]
    l2 = params[3]
    g = params[4]
    a1, a2 = _legacy_accels(th1, th2, w1, w2, m1, m2, l1, l2, g)
    w1n = w1 + dt * a1
    w2n = w2 + dt * (a2 + actions[t])  # optional torque on joint 2
    states[t + 1, 0] = th1 + dt * w1n
    states[t + 1, 1] = th2 + dt * w2n
    states[t + 1, 2] = w1n
    states[t + 1, 3] = w2n


@wp.func
def _symp_grad(
    q1: float, q2: float, p1: float, p2: float,
    m1: float, m2: float, l1: float, l2: float, g: float,
) -> tuple[float, float, float, float]:
    """J grad H = (dH/dp, -dH/dq) in canonical (theta, p).

    The 2x2 inverse of M is hand-expanded in scalars rather than using wp.mat22 / wp.inverse:
    it keeps the kernel AD-transparent and avoids relying on Warp's adjoints for matrix
    intrinsics. det M = 2 - cos^2 d in [1, 2] at the paper's parameters, so it is
    well-conditioned everywhere.
    """
    c = wp.cos(q1 - q2)
    s = wp.sin(q1 - q2)
    a = (m1 + m2) * l1 * l1
    b = m2 * l1 * l2 * c
    d = m2 * l2 * l2
    det = a * d - b * b
    u1 = (d * p1 - b * p2) / det  # u = M^-1 p  (= omega)
    u2 = (a * p2 - b * p1) / det
    k = m2 * l1 * l2 * s
    dhdq1 = k * u1 * u2 + (m1 + m2) * g * l1 * wp.sin(q1)
    dhdq2 = -k * u1 * u2 + m2 * g * l2 * wp.sin(q2)
    return u1, u2, -dhdq1, -dhdq2


@wp.kernel
def acrobot_step(
    states: wp.array2d(dtype=wp.float32),  # type: ignore[valid-type]
    actions: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
    params: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
    dt: float,
    t: int,
) -> None:
    """Implicit midpoint in canonical (theta, p), _N_FP_ITER unrolled Picard passes.

    `actions[t]` is a GENERALIZED FORCE on p_2 -- a genuine elbow torque. It differs from the
    retired kernel's action, which added to omega_2 alone and omitted the coupling
    (M^-1)_12 tau to the shoulder link entirely; no rescaling reproduces it.
    """
    q1 = states[t, 0]
    q2 = states[t, 1]
    p1 = states[t, 2]
    p2 = states[t, 3]
    m1 = params[0]
    m2 = params[1]
    l1 = params[2]
    l2 = params[3]
    g = params[4]
    tau = actions[t]

    b1 = q1
    b2 = q2
    b3 = p1
    b4 = p2
    for _ in range(_N_FP_ITER):
        f1, f2, f3, f4 = _symp_grad(b1, b2, b3, b4, m1, m2, l1, l2, g)
        b1 = q1 + 0.5 * dt * f1
        b2 = q2 + 0.5 * dt * f2
        b3 = p1 + 0.5 * dt * f3
        b4 = p2 + 0.5 * dt * (f4 + tau)

    states[t + 1, 0] = 2.0 * b1 - q1
    states[t + 1, 1] = 2.0 * b2 - q2
    states[t + 1, 2] = 2.0 * b3 - p1
    states[t + 1, 3] = 2.0 * b4 - p2


def _legacy_energy(state: npt.NDArray[np.float64], params: npt.NDArray[np.float64]) -> float:
    """Total mechanical energy in the retired (theta, omega) coordinates.

    Retained for the legacy-kernel regression test only; the live system uses
    _canonical_energy.
    """
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
        jacobian=_cayley_jacobian,
        energy=_canonical_energy,
        suggested_dt=0.0005,
    )
)
