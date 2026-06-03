"""Phase-1 experiments: slope-vs-λ₁ across regimes, and gradient SNR vs horizon."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from predictability_horizon.gradient_law import gradient_law
from predictability_horizon.lyapunov import lyapunov_spectrum
from predictability_horizon.warpsim import grad_loss_wrt_x0

Vec = npt.NDArray[np.float64]


def slope_vs_lambda_points(specs: Sequence[dict[str, Any]]) -> list[tuple[str, float, float]]:
    """For each spec, return (name, λ₁ per unit time, gradient-gain slope per unit time).

    Each spec dict requires the keys ``name`` (str), ``step_kernel``, ``x0``, ``params``,
    ``dt`` (float), ``horizons``, ``dim`` (int), ``jacobian`` — i.e. the arguments of
    ``gradient_law`` plus a display ``name``. Both returned rates are per unit time
    (the per-step values from ``gradient_law`` divided by ``dt``).
    """
    out: list[tuple[str, float, float]] = []
    for s in specs:
        res = gradient_law(
            s["step_kernel"], s["x0"], s["params"], s["dt"], s["horizons"], s["dim"], s["jacobian"]
        )
        out.append((s["name"], res.lambda1_per_step / s["dt"], res.slope_per_step / s["dt"]))
    return out


def lambda_spread(
    jac_fn: Callable[[Vec], Vec], traj: Vec, dt: float, k: int, seeds: Sequence[int]
) -> tuple[float, float]:
    """Mean & std of the largest Lyapunov exponent (per unit time) over QR seeds."""
    if len(seeds) == 0:
        raise ValueError("seeds must be non-empty")
    vals = [lyapunov_spectrum(jac_fn, traj, dt=dt, k=k, seed=sd).largest for sd in seeds]
    return float(np.mean(vals)), float(np.std(vals))


def gradient_snr_vs_horizon(
    step_kernel: Any,
    x0: Vec,
    params: Vec,
    dt: float,
    horizons: npt.NDArray[np.int_],
    dim: int,
    n_ic: int = 20,
    eps: float = 1e-3,
    seed: int = 0,
) -> tuple[Vec, Vec]:
    """Signal-to-noise of the analytic gradient ∇_{x0}||x_T||² across nearby ICs vs horizon.

    For each horizon T (in integration *steps*), perturb x0 by N(0, eps) over n_ic
    samples, take the gradient via grad_loss_wrt_x0, and report
    SNR = ||mean gradient|| / mean(||g - mean||). In a chaotic system the gradient
    directions decorrelate across nearby ICs, so the SNR collapses once the horizon
    reaches the predictability limit — at a step count ≈ 1/(λ₁·dt), i.e. Lyapunov time
    T·dt·λ₁ ≈ 1. The returned x-axis is in steps; callers convert to Lyapunov time by
    multiplying by ``dt`` and the per-time λ₁. (For a *linear* map the e^{λ₁T} factor
    cancels between signal and noise, so the SNR is flat — the collapse is a genuinely
    nonlinear effect.)

    Requires n_ic >= 2 (the noise estimate needs spread across ICs).
    """
    if n_ic < 2:
        raise ValueError("n_ic must be >= 2 (the noise estimate needs spread across ICs)")
    rng = np.random.default_rng(seed)
    snr = []
    for T in horizons:  # noqa: N806
        grads = []
        for _ in range(n_ic):
            xp = x0 + eps * rng.standard_normal(dim)
            grads.append(
                grad_loss_wrt_x0(
                    step_kernel, xp, np.zeros(int(T)), params, dt, int(T), np.zeros(dim)
                )
            )
        g = np.array(grads)
        mean = g.mean(axis=0)
        noise = np.mean(np.linalg.norm(g - mean, axis=1)) + 1e-30
        snr.append(float(np.linalg.norm(mean) / noise))
    return horizons.astype(float), np.array(snr)
