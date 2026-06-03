"""Part A1: the rollout Jacobian norm ‖∂x_T/∂x₀‖₂ grows as e^{λ₁ T}."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import numpy.typing as npt
import warp as wp

from predictability_horizon.core import GradientLawResult, fit_loglinear_slope
from predictability_horizon.lyapunov import lyapunov_spectrum
from predictability_horizon.warpsim import rollout, rollout_jacobian

Vec = npt.NDArray[np.float64]


def gradient_law(
    step_kernel: wp.Kernel,
    x0: Vec,
    params: Vec,
    dt: float,
    horizons: npt.NDArray[np.int_],
    dim: int,
    jacobian: Callable[[Vec, float, Vec, float], Vec],
    device: str = "cpu",
) -> GradientLawResult:
    """Spectral norm of the rollout Jacobian ∂x_T/∂x₀ vs horizon; slope == λ₁.

    ‖∂x_T/∂x₀‖₂ is the worst-case gain a differentiable simulator applies to an input
    perturbation — exactly the sensitivity reverse-mode AD propagates. It grows as
    e^{λ₁ T}, so its log-slope equals the largest Lyapunov exponent (measured here on
    the same map via `lyapunov_spectrum`), and analytic simulator gradients are only
    usable for horizons T ≲ 1/λ₁.
    """
    norms = []
    for T in horizons:  # noqa: N806
        M = rollout_jacobian(  # noqa: N806
            step_kernel,
            x0,
            np.zeros(int(T)),
            params,
            dt,
            int(T),
            device=device,
        )
        norms.append(float(np.linalg.norm(M, 2)))  # spectral norm = top singular value
    norms_arr = np.array(norms)
    slope, _ = fit_loglinear_slope(horizons.astype(float), norms_arr)

    Tmax = int(horizons[-1])  # noqa: N806
    traj = rollout(step_kernel, x0, np.zeros(Tmax), params, dt, Tmax, device=device)

    def jac(s: Vec) -> Vec:
        return jacobian(s, 0.0, params, dt)

    lam = lyapunov_spectrum(jac, traj, dt=dt, k=dim).largest

    return GradientLawResult(
        horizons=horizons.astype(float),
        grad_norms=norms_arr,
        slope_per_step=slope,
        lambda1_per_step=lam * dt,
    )
