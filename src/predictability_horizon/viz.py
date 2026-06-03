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
