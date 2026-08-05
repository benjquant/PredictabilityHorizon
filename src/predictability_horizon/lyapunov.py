"""Benettin/QR Lyapunov spectrum from discrete one-step Jacobians."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import numpy.typing as npt

from predictability_horizon.core import LyapunovResult

Vec = npt.NDArray[np.float64]


def lyapunov_spectrum_from_jacobians(
    jacs: npt.NDArray[np.float64],
    dt: float,
    k: int | None = None,
    seed: int = 0,
) -> LyapunovResult:
    """Top-k Lyapunov exponents (per unit time) from a PRECOMPUTED stack of one-step Jacobians.

    jacs: (steps, dim, dim), one Jacobian per step, in trajectory order.

    Separated from ``lyapunov_spectrum`` rather than folded into it: callers that can produce
    every Jacobian in a single batched pass -- the HNN's Cayley transform under torch.func.vmap
    -- would otherwise have to re-enter torch once per state and lose the batching entirely.
    ``lyapunov_spectrum`` is deliberately left alone because Part A depends on it;
    test_batched_entry_point_matches_the_callback_one pins the two together.
    """
    n = jacs.shape[1]
    k = n if k is None else k
    Q = np.linalg.qr(np.random.default_rng(seed).standard_normal((n, k)))[0]  # noqa: N806
    log_sum = np.zeros(k)
    steps = jacs.shape[0]
    for t in range(steps):
        Q, R = np.linalg.qr(jacs[t] @ Q)  # noqa: N806
        log_sum += np.log(np.abs(np.diag(R)))
    exponents = np.sort(log_sum / (steps * dt))[::-1]
    return LyapunovResult(exponents=exponents)


def lyapunov_spectrum(
    jacobian_fn: Callable[[Vec], Vec],
    traj: npt.NDArray[np.float64],
    dt: float,
    k: int | None = None,
    seed: int = 0,
) -> LyapunovResult:
    """Top-k Lyapunov exponents (per unit time) via tangent-space QR.

    traj: (N+1, dim) states. jacobian_fn(state) -> (dim, dim) discrete Jacobian.
    seed: RNG seed for the initial QR orthonormal frame (default 0, backward-compatible).
    """
    n = traj.shape[1]
    k = n if k is None else k
    Q = np.linalg.qr(np.random.default_rng(seed).standard_normal((n, k)))[0]  # noqa: N806
    log_sum = np.zeros(k)
    steps = traj.shape[0] - 1
    for t in range(steps):
        J = jacobian_fn(traj[t])  # noqa: N806
        Q, R = np.linalg.qr(J @ Q)  # noqa: N806
        log_sum += np.log(np.abs(np.diag(R)))
    exponents = np.sort(log_sum / (steps * dt))[::-1]
    return LyapunovResult(exponents=exponents)
