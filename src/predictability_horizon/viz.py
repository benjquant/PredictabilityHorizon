"""Figure generators. `fast=True` uses short horizons/iters for the smoke test."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import matplotlib
import numpy as np
import numpy.typing as npt
import warp as wp

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from predictability_horizon.experiments import gradient_snr_vs_horizon, slope_vs_lambda_points
from predictability_horizon.gradient_law import gradient_law
from predictability_horizon.lyapunov import lyapunov_spectrum
from predictability_horizon.swingup import optimize_swingup
from predictability_horizon.systems import SYSTEMS, acrobot, cartpole, pendulum  # noqa: F401
from predictability_horizon.warpsim import rollout
from predictability_horizon.worldmodel import (
    MLP,
    _model_rollout,
    error_growth_rate,
    make_dataset,
    model_lyapunov_on_traj,
    train_world_model,
)


def make_fig1_gradient_law(out: Path, fast: bool = False) -> Path:
    """Two-panel semilogy of rollout-Jacobian norm vs horizon: pendulum (integrable) and acrobot (chaotic).

    Shows the clean integrable-vs-chaotic contrast:
    - Pendulum: sub-exponential growth (λ₁≈0), no fit line drawn.
    - Acrobot: exponential growth tracking e^{λ₁T}, steep slope.
    """
    specs: list[tuple[str, np.ndarray, np.ndarray]] = [
        ("pendulum", np.array([2.0, 0.0]), np.arange(1000, 9000, 1000)),
        ("acrobot", np.array([2.5, 0.0, 0.0, 0.0]), np.arange(1000, 9000, 1000)),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, (name, x0, horizons) in zip(axes, specs, strict=True):
        sys = SYSTEMS[name]
        h = horizons[:3] if fast else horizons
        res = gradient_law(
            cast(wp.Kernel, sys.step_kernel),
            x0,
            sys.default_params,
            sys.suggested_dt,
            h,
            sys.dim,
            sys.jacobian,
        )
        lam_t = res.lambda1_per_step / sys.suggested_dt  # per unit time
        slope_t = res.slope_per_step / sys.suggested_dt  # per unit time

        ax.semilogy(res.horizons, res.grad_norms, "o", label="measured")

        if name == "acrobot":
            # Exponential fit tracks the growth well for chaotic dynamics
            intercept = np.log(res.grad_norms[0]) - res.slope_per_step * res.horizons[0]
            fit = np.exp(np.polyval([res.slope_per_step, intercept], res.horizons))
            ax.semilogy(res.horizons, fit, "-", alpha=0.7, label=r"$e^{\lambda_1 T}$ fit")
            ax.legend(fontsize=8)

        ax.set_title(f"{name}\n" r"$\lambda_1$" f"≈{lam_t:.2f}/s   slope≈{slope_t:.2f}/s")
        ax.set_xlabel("rollout steps T")

    axes[0].set_ylabel(r"$\|\partial x_T/\partial x_0\|_2$")
    fig.suptitle(
        r"Gradient gain $\|\partial x_T/\partial x_0\|$ blows up exponentially at $\lambda_1$"
        " for chaotic dynamics (acrobot), but not for integrable dynamics (pendulum)"
    )
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def make_fig2_swingup(out: Path, fast: bool = False) -> Path:
    """Loss curve for cartpole swing-up optimized via differentiable simulation."""
    hist = optimize_swingup(T=100, iters=30 if fast else 300, lr=1.5)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(hist)
    ax.set_xlabel("optimization iteration")
    ax.set_ylabel("upright-tracking cost")
    ax.set_title("Cartpole swing-up via differentiable simulation (Warp)")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def _trained_model(name: str, fast: bool) -> MLP:
    """Train a world model for the named system; small config when fast=True."""
    sys = SYSTEMS[name]
    n_traj = 10 if fast else 100
    epochs = 5 if fast else 200
    # Physical horizon: ~3 s for pendulum (dt=0.005), ~2 s for acrobot (dt=0.0005)
    T_map = {  # noqa: N806
        "pendulum": 100 if fast else 600,
        "acrobot": 200 if fast else 4000,
    }
    T = T_map.get(name, 200 if fast else 1000)  # noqa: N806
    ds = make_dataset(sys, n_traj=n_traj, T=T, seed=0)
    return train_world_model(ds, epochs=epochs, seed=0)


def make_fig3_error_growth(out: Path, fast: bool = False) -> Path:
    """Prediction error ||pred-gt|| vs physical time for pendulum and acrobot.

    Acrobot (chaotic) error rises visibly faster than pendulum (integrable).
    Each curve is annotated with its error_growth_rate (per unit time).
    """
    configs: list[tuple[str, npt.NDArray[np.float64], int]] = [
        ("pendulum", np.array([2.0, 0.0]), 100 if fast else 600),
        ("acrobot", np.array([2.5, 0.0, 0.0, 0.0]), 200 if fast else 4000),
    ]

    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["tab:blue", "tab:red"]

    for (name, x0, T), color in zip(configs, colors, strict=True):  # noqa: N806
        sys = SYSTEMS[name]
        model = _trained_model(name, fast=fast)

        gt = rollout(
            cast(wp.Kernel, sys.step_kernel),
            x0,
            np.zeros(T),
            sys.default_params,
            sys.suggested_dt,
            T,
        )
        pred = _model_rollout(model, x0, T)
        err = np.linalg.norm(pred - gt, axis=1) + 1e-12

        t_phys = np.arange(T + 1) * sys.suggested_dt
        rate = error_growth_rate(model, sys, x0, T=T)

        ax.semilogy(t_phys, err, color=color, label=f"{name}  (rate≈{rate:.2f}/s)")

    ax.set_xlabel("physical time (s)")
    ax.set_ylabel(r"$\|\hat{x} - x\|_2$")
    ax.set_title("World-model prediction error grows faster for chaotic dynamics")
    ax.legend()
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def make_fig4_lyapunov_scatter(out: Path, fast: bool = False) -> Path:
    """Learned vs true λ₁ scatter for pendulum and acrobot.

    Both true and learned exponents are evaluated along the SAME long true trajectory,
    after dropping a transient — matching the validated test settings so the true λ₁
    converges (pendulum→~0.05, acrobot→~1.1).  This is apples-to-apples and avoids
    model-drift artifacts in the comparison.

    fast=True uses short training (epochs=5, n_traj=10) and short λ₁ horizons
    (T=2000 for both systems) to produce a file quickly for the smoke test.
    """
    np.random.seed(0)

    # Per-system config: (x0, T_full, T_fast, transient)
    # T_full / transient match the validated test settings in test_systems.py.
    configs: dict[
        str,
        tuple[npt.NDArray[np.float64], int, int, int],
    ] = {
        "pendulum": (np.array([2.0, 0.0]), 16000, 2000, 1000),
        "acrobot": (np.array([2.5, 0.0, 0.0, 0.0]), 12000, 2000, 1000),
    }
    systems_to_plot = ["pendulum", "acrobot"]

    true_lams: list[float] = []
    learned_lams: list[float] = []
    labels: list[str] = []

    for name in systems_to_plot:
        sys = SYSTEMS[name]
        x0, T_full, T_fast, transient = configs[name]  # noqa: N806
        T = T_fast if fast else T_full  # noqa: N806

        # 1. Roll out the TRUE system for a long horizon.
        traj_full = rollout(
            cast(wp.Kernel, sys.step_kernel),
            x0,
            np.zeros(T),
            sys.default_params,
            sys.suggested_dt,
            T,
        )
        # Drop the transient (or a proportional amount in fast mode).
        drop = min(transient, T // 4)
        traj = traj_full[drop:]

        # 2. True λ₁: Benettin on the analytic Jacobian along the true trajectory.
        def _jac_analytic(
            s: npt.NDArray[np.float64],
            _sys: object = sys,
        ) -> npt.NDArray[np.float64]:
            from predictability_horizon.systems import System as _System

            _s = cast(_System, _sys)
            return cast(
                npt.NDArray[np.float64],
                _s.jacobian(s, 0.0, _s.default_params, _s.suggested_dt),
            )

        lam_true = lyapunov_spectrum(_jac_analytic, traj, dt=sys.suggested_dt, k=1).largest

        # 3. Learned λ₁: the MODEL's Jacobian evaluated on the SAME true trajectory.
        model = _trained_model(name, fast=fast)
        lam_learned = model_lyapunov_on_traj(model, traj, dt=sys.suggested_dt, k=1)

        true_lams.append(lam_true)
        learned_lams.append(lam_learned)
        labels.append(name)
        print(f"[fig4] {name}: true λ₁={lam_true:.4f}/s  learned λ₁={lam_learned:.4f}/s")

    true_arr = np.array(true_lams)
    learned_arr = np.array(learned_lams)

    # y=x reference spanning the full data range with margin
    margin = 0.15 * max(true_arr.max(), learned_arr.max(), 0.5)
    ref_lo = min(true_arr.min(), learned_arr.min()) - margin
    ref_hi = max(true_arr.max(), learned_arr.max()) + margin
    ref = np.linspace(ref_lo, ref_hi, 100)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(ref, ref, "k--", alpha=0.5, label="y = x (perfect)")
    colors = ["tab:blue", "tab:red"]
    for x_val, y_val, label, color in zip(true_arr, learned_arr, labels, colors, strict=True):
        ax.scatter(x_val, y_val, color=color, s=80, zorder=3)
        ax.annotate(
            label,
            (x_val, y_val),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=9,
        )

    ax.set_xlabel(r"true $\lambda_1$ (Benettin, along true traj, /s)")
    ax.set_ylabel(r"learned $\lambda_1$ (MLP Jacobian, on true traj, /s)")
    ax.set_title(
        "Learned world models capture integrable dynamics\nbut mis-estimate chaotic sensitivity"
    )
    ax.set_xlim(ref_lo, ref_hi)
    ax.set_ylim(ref_lo, ref_hi)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def _finite_time_lambda_std(
    spec: dict[str, object],
    t_steps: int,
    k: int,
    n_windows: int,
) -> float:
    """Std of the largest Lyapunov exponent across contiguous trajectory windows.

    An honest finite-time error bar: lambda1 measured over a finite window fluctuates,
    and this captures that scatter. (The QR-seed spread, by contrast, is ~0 and would
    understate the uncertainty.)
    """
    from collections.abc import Callable as _Callable

    dt = cast(float, spec["dt"])
    traj = rollout(
        cast(wp.Kernel, spec["step_kernel"]),
        cast(npt.NDArray[np.float64], spec["x0"]),
        np.zeros(t_steps),
        cast(npt.NDArray[np.float64], spec["params"]),
        dt,
        t_steps,
    )
    traj = traj[t_steps // 10 :]  # drop transient

    jac_fn = cast(
        _Callable[
            [npt.NDArray[np.float64], float, npt.NDArray[np.float64], float],
            npt.NDArray[np.float64],
        ],
        spec["jacobian"],
    )
    params = cast(npt.NDArray[np.float64], spec["params"])

    def jac(state: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return jac_fn(state, 0.0, params, dt)

    win = len(traj) // n_windows
    vals: list[float] = []
    for i in range(n_windows):
        seg = traj[i * win : (i + 1) * win]
        if len(seg) > 10:
            vals.append(lyapunov_spectrum(jac, seg, dt=dt, k=k).largest)
    return float(np.std(vals)) if vals else 0.0


def make_fig5_slope_vs_lambda(out: Path, fast: bool = False) -> Path:
    """Gradient-gain slope tracks λ₁ across integrable→chaotic regimes.

    Points: pendulum (integrable, λ₁≈0) and acrobot at a sweep of initial energies
    (rising θ₁ → rising λ₁ up to ~1.2/s). Plotted slope-vs-λ₁ clusters on y=x.
    Cartpole is intentionally excluded (λ₁ is regime-dependent/ill-defined for cartpole;
    the acrobot energy sweep already spans the integrable→chaotic transition).
    X-error bars show finite-time window scatter — an honest representation of the
    finite-time Lyapunov uncertainty.
    """
    s = SYSTEMS["acrobot"]
    energies: tuple[float, ...] = (1.5, 2.5) if fast else (0.5, 1.0, 1.5, 2.0, 2.5, 3.0)
    horizons_arr: npt.NDArray[np.int_] = (
        np.arange(500, 2500, 500) if fast else np.arange(2000, 18000, 2000)
    )
    t_win = 2000 if fast else 24000
    n_windows = 2 if fast else 4
    specs: list[dict[str, object]] = [
        {
            "name": f"acrobot θ₁={th}",
            "step_kernel": s.step_kernel,
            "x0": np.array([th, 0.0, 0.0, 0.0]),
            "params": s.default_params,
            "dt": s.suggested_dt,
            "dim": 4,
            "horizons": horizons_arr,
            "jacobian": s.jacobian,
        }
        for th in energies
    ]
    p = SYSTEMS["pendulum"]
    pendulum_horizons: npt.NDArray[np.int_] = (
        np.arange(500, 2500, 500) if fast else np.arange(2000, 18000, 2000)
    )
    specs.append(
        {
            "name": "pendulum",
            "step_kernel": p.step_kernel,
            "x0": np.array([2.0, 0.0]),
            "params": p.default_params,
            "dt": p.suggested_dt,
            "dim": 2,
            "horizons": pendulum_horizons,
            "jacobian": p.jacobian,
        }
    )

    pts = slope_vs_lambda_points(specs)
    # x-error = finite-time window scatter of λ₁. The plotted point (λ₁ from gradient_law's
    # full rollout) and this error bar are independent but consistent estimators of the same
    # IC's λ₁; the bar illustrates finite-time uncertainty, not a CI on the point estimate.
    xerr = [_finite_time_lambda_std(spec, t_win, 1, n_windows) for spec in specs]

    fig, ax = plt.subplots(figsize=(6.0, 6.0))
    lims = max(0.2, max(max(pt[1] for pt in pts), max(pt[2] for pt in pts)) * 1.15)
    ax.plot([0, lims], [0, lims], "k--", alpha=0.5, label="slope = λ₁")
    for (name, lam, slope), xe in zip(pts, xerr, strict=True):
        ax.errorbar(lam, slope, xerr=xe, fmt="o", capsize=3, zorder=3, label=name)
    ax.set_xlabel("measured λ₁ (Benettin, /s)   [x-err = finite-time window scatter]")
    ax.set_ylabel("gradient-gain slope (/s)")
    ax.set_title("Gradient-gain rate tracks λ₁ across regimes")
    ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("FIG5 points (name, lambda1/s, slope/s, lambda_xerr):")
    for (name, lam, slope), xe in zip(pts, xerr, strict=True):
        print(f"  {name:18s} lam={lam:+.3f} slope={slope:+.3f} xerr={xe:.3f}")
    lams_arr = np.array([pt[1] for pt in pts])
    slopes_arr = np.array([pt[2] for pt in pts])
    fit_slope, fit_int = np.polyfit(lams_arr, slopes_arr, 1)
    corr = float(np.corrcoef(lams_arr, slopes_arr)[0, 1])
    print(
        f"FIG5 trend: corr(λ₁,slope)={corr:.3f}; best-fit slope={fit_slope:.3f} "
        f"(ideal 1.0), intercept={fit_int:+.3f}"
    )
    return out


def make_fig6_gradient_horizon(out: Path, fast: bool = False) -> Path:
    """Analytic-gradient SNR collapses past the predictability horizon T ≲ 1/λ₁.

    X-axis is in Lyapunov-time units: steps * dt * lambda1.  The vertical dashed line
    at T*lambda1=1 marks the predictability horizon.
    """
    s = SYSTEMS["acrobot"]
    horizons_arr: npt.NDArray[np.int_] = (
        np.arange(200, 1200, 200) if fast else np.arange(500, 4500, 500)
    )
    x0 = np.array([2.5, 0.0, 0.0, 0.0])
    # A single chaotic orbit's SNR has multiplicative finite-time fluctuations (its
    # gradient/Hessian don't grow perfectly smoothly), so SNR(T) ~ (1/eps)·e^{-λ₁T}
    # times a fluctuating factor. Average over several nearby BASE trajectories, with a
    # geometric mean (the fluctuations are multiplicative), to expose the underlying
    # decay. Averaging over perturbation seeds at a single base does NOT help — the mean
    # gradient is fixed by that one orbit.
    rng_base = np.random.default_rng(0)
    n_base = 1 if fast else 4
    n_ic = 3 if fast else 30
    base_eps = 0.05
    snr_runs = []
    horizons = horizons_arr.astype(float)
    for b in range(n_base):
        xb = x0 if b == 0 else x0 + base_eps * rng_base.standard_normal(4)
        _, snr_b = gradient_snr_vs_horizon(
            s.step_kernel,
            xb,
            s.default_params,
            s.suggested_dt,
            horizons_arr,
            4,
            n_ic=n_ic,
            seed=b,
        )
        snr_runs.append(snr_b)
    snr = np.exp(np.mean(np.log(np.maximum(np.array(snr_runs), 1e-30)), axis=0))
    # λ₁ reference on a dedicated long trajectory (the SNR horizons are too short to
    # converge λ₁; using horizons[-1] here would under-estimate it and mis-scale the axis).
    lam_steps = 2000 if fast else 20000
    traj = rollout(
        cast(wp.Kernel, s.step_kernel),
        x0,
        np.zeros(lam_steps),
        s.default_params,
        s.suggested_dt,
        lam_steps,
    )

    def _jac6(state: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return cast(
            npt.NDArray[np.float64],
            s.jacobian(state, 0.0, s.default_params, s.suggested_dt),
        )

    lam = lyapunov_spectrum(
        _jac6,
        traj[len(traj) // 10 :],
        dt=s.suggested_dt,
        k=4,
    ).largest
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.semilogy(horizons * s.suggested_dt * lam, snr, "o-")
    ax.axvline(1.0, color="r", ls="--", alpha=0.6, label="T·λ₁ = 1  (the 1/λ₁ horizon)")
    ax.set_xlabel("Lyapunov time  T·λ₁  (= steps · dt · λ₁)")
    ax.set_ylabel("analytic-gradient SNR")
    ax.set_title("Analytic-gradient SNR degrades past the predictability horizon", fontsize=11)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(
        f"FIG6: lambda1={lam:.3f}/s; SNR at T*lam=",
        [f"{(h * s.suggested_dt * lam):.2f}:{v:.1f}" for h, v in zip(horizons, snr, strict=True)],
    )
    return out
