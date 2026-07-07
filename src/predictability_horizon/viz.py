"""Figure generators. `fast=True` uses short horizons/iters for the smoke test."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import matplotlib
import numpy as np
import numpy.typing as npt
import warp as wp

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from predictability_horizon.core import GradientLawResult
from predictability_horizon.experiments import gradient_snr_vs_horizon, slope_vs_lambda_points
from predictability_horizon.gradient_law import gradient_law
from predictability_horizon.lyapunov import lyapunov_spectrum
from predictability_horizon.systems import (  # noqa: F401  (register acrobot/cartpole/henon_heiles/pendulum)
    SYSTEMS,
    acrobot,
    cartpole,
    henon_heiles,
    pendulum,
)
from predictability_horizon.trajopt import (
    measure_lambda1,
    reach_horizon_sweep,
    swingup_horizon_sweep,
)
from predictability_horizon.warpsim import rollout
from predictability_horizon.worldmodel import (
    MLP,
    _model_rollout,
    error_growth_rate,
    make_dataset,
    model_lyapunov_on_traj,
    train_world_model,
)


def _grad_law_on(name: str, x0: np.ndarray, t_max: float, n: int) -> tuple[GradientLawResult, float]:
    """Run ``gradient_law`` for ``name`` over physical horizons up to ``t_max`` seconds."""
    sys = SYSTEMS[name]
    dt = sys.suggested_dt
    horizons = np.unique(np.round(np.linspace(t_max / n, t_max, n) / dt).astype(int))
    horizons = horizons[horizons >= 1]
    res = gradient_law(
        cast(wp.Kernel, sys.step_kernel), x0, sys.default_params, dt, horizons, sys.dim, sys.jacobian
    )
    return res, dt


def make_fig1_gradient_law(out: Path, fast: bool = False) -> Path:
    """Two-panel semilog of rollout-Jacobian norm ‖∂x_T/∂x₀‖₂ vs horizon T (seconds).

    The integrable-vs-chaotic contrast is *linear vs exponential* growth (not "flat vs
    blows up"), shown over each system's own physical-time window because the two laws
    live on different timescales:

    - Pendulum (integrable, λ₁=0): ‖M_T‖ grows *linearly* (anharmonic shear); the overlaid
      line is the linear law a·t (a log curve on this semilog axis). The finite-time Benettin
      estimate is a window artifact still decaying toward 0 (~ln T / T).
    - Acrobot (chaotic, λ₁>0): genuine exponential e^{λ₁T}; λ₁ is fit past an initial
      transient (T ≥ cutoff) and matches the Benettin/QR exponent within finite-time scatter.
    """
    pend_tmax, acro_tmax = (8.0, 2.0) if fast else (40.0, 8.0)
    n_p, n_a = (6, 6) if fast else (32, 20)
    fig, (axp, axa) = plt.subplots(1, 2, figsize=(9.5, 4.2))

    # --- pendulum (left): linear law (λ₁=0); inset proves linearity (log-log slope 1) ---
    resp, dtp = _grad_law_on("pendulum", np.array([2.0, 0.0]), pend_tmax, n_p)
    tp = resp.horizons * dtp
    sp = resp.grad_norms
    axp.semilogy(tp, sp, "o", ms=4, color="C0", label="measured")
    cut_p = min(5.0, 0.4 * pend_tmax)  # fit on the post-transient part
    m_p = tp > cut_p
    a, b = np.polyfit(tp[m_p], sp[m_p], 1)  # linear law (the physics)
    lam_sp = np.polyfit(tp[m_p], np.log(sp[m_p]), 1)[0]  # naive slope-method estimate
    tl = np.linspace(tp[0], tp[-1], 200)
    lin = np.clip(a * tl + b, 1e-12, None)
    axp.semilogy(tl[tl >= cut_p], lin[tl >= cut_p], "-", color="darkcyan", lw=1.6,
                 label=r"$\|M_T\|\propto t$  ($\lambda_1=0$)")
    axp.semilogy(tl[tl < cut_p], lin[tl < cut_p], "--", color="darkcyan", lw=1.0, alpha=0.45)
    lam_qr_p = resp.lambda1_per_step / dtp
    axp.set_title("pendulum — integrable")
    axp.set_xlabel(r"rollout horizon $T$ (s)")
    axp.set_ylabel(r"$\|M_T\|_2 \equiv \|\partial x_T/\partial x_0\|_2$")
    axp.text(0.045, 0.035, rf"finite-time $\lambda_1$:  Benettin {lam_qr_p:.2f},  slope {lam_sp:.2f}",
             transform=axp.transAxes, va="bottom", ha="left", fontsize=7.5, color="0.4")
    axp.legend(fontsize=8, loc="upper left")
    axins = axp.inset_axes((0.60, 0.12, 0.37, 0.40))  # log-log view: linear growth -> slope-1 line
    axins.loglog(tp, sp, "o", ms=2.0, color="C0")
    axins.loglog(tp, sp[-1] * (tp / tp[-1]), "--", color="0.4", lw=1.0)
    axins.set_title(r"log-log: slope 1 $\Rightarrow \propto t$", fontsize=6.5)
    axins.tick_params(labelsize=6)

    # --- acrobot (right): exponential law (slope fit), Benettin cross-check ---
    resa, dta = _grad_law_on("acrobot", np.array([2.5, 0.0, 0.0, 0.0]), acro_tmax, n_a)
    ta = resa.horizons * dta
    axa.semilogy(ta, resa.grad_norms, "o", ms=4, color="crimson", label="measured")
    cut = min(2.0, 0.25 * acro_tmax)  # fit past the initial transient
    m_a = ta >= cut
    lam, c = np.polyfit(ta[m_a], np.log(resa.grad_norms[m_a]), 1)  # slope-method fit
    tl = np.linspace(ta[0], ta[-1], 200)
    ex = np.exp(lam * tl + c)
    axa.semilogy(tl[tl >= cut], ex[tl >= cut], "-", color="coral", lw=1.6,
                 label=r"$\|M_T\|\propto e^{\lambda_1 T}$")
    axa.semilogy(tl[tl < cut], ex[tl < cut], "--", color="coral", lw=1.0, alpha=0.45)
    lam_qr_a = resa.lambda1_per_step / dta  # independent Benettin exponent
    axa.set_title("acrobot — chaotic")
    axa.set_xlabel(r"rollout horizon $T$ (s)")
    axa.text(0.045, 0.96,
             rf"Benettin $\lambda_1={lam_qr_a:.2f}\,\mathrm{{s}}^{{-1}}$" "\n"
             rf"slope $\lambda_1={lam:.2f}\,\mathrm{{s}}^{{-1}}$",
             transform=axa.transAxes, va="top", ha="left", fontsize=8, color="0.45", linespacing=1.6)
    axa.legend(fontsize=8, loc="lower right")

    print(f"[fig1] pendulum: a={a:.3f}/s slope={lam_sp:.3f}/s Benettin={lam_qr_p:.3f}/s | "
          f"acrobot: slope={lam:.3f}/s Benettin={lam_qr_a:.3f}/s (fit T>={cut:.0f}s)")
    fig.suptitle(
        r"Rollout-Jacobian gain $\|M_T\|_2$: linear (integrable) vs exponential (chaotic)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def _geomean_band(seeds_per_horizon: list[list[float]]) -> tuple[Any, Any, Any]:
    """Per-horizon geometric mean and 10-90th percentile band of a positive, heavy-tailed quantity.

    Averaging the normalized cost in log space over reach targets smooths the curve naturally and
    keeps its magnitude; the band shows the seed-to-seed reliability spread.
    """
    g, lo, hi = [], [], []
    for s in seeds_per_horizon:
        a = np.clip(np.asarray(s, dtype=float), 1e-12, None)
        g.append(float(np.exp(np.mean(np.log(a)))))
        lo.append(float(np.percentile(a, 10)))
        hi.append(float(np.percentile(a, 90)))
    return np.array(g), np.array(lo), np.array(hi)


def make_fig2_trajopt_horizon(out: Path, fast: bool = False) -> Path:
    """One-panel Fig 2: normalized final cost vs horizon T·λ₁ for three trajectory-optimisation settings.

    Cost = final / do-nothing-baseline (≪1 succeeds; ≥1 no better than doing nothing), geometric-mean
    over reach targets with a 10-90% band. The acrobot's *precise* reach rises through the
    do-nothing baseline at the predictability horizon (the §4.3 horizon with its log factor) and blows
    up beyond it (e^{2λ₁T} gradient gain); the *forgiving* swing-up and the *integrable* pendulum's
    precise reach stay far below the baseline at every horizon. All curves share the acrobot's
    Lyapunov axis at matched step counts.
    """
    acro = SYSTEMS["acrobot"]
    pend = SYSTEMS["pendulum"]
    x0a = np.array([2.5, 0.0, 0.0, 0.0])
    x0p = np.array([2.0, 0.0])
    if fast:
        dt, tlams, n_acro, n_ctrl, ri, si = 5e-3, [1.0, 3.0, 5.0], 4, 3, 80, 80
    else:
        dt, tlams, n_acro, n_ctrl, ri, si = 5e-4, [1.0, 2.0, 3.0, 3.5, 4.0, 4.5, 5.0], 12, 6, 250, 250
    lam1 = measure_lambda1(acro, x0a, dt)
    ra = reach_horizon_sweep(acro, x0a, dt, tlams, lam1=lam1, n_seed=n_acro, iters=ri)
    rp = reach_horizon_sweep(pend, x0p, dt, tlams, lam1=lam1, n_seed=n_ctrl, iters=ri)
    su = swingup_horizon_sweep(acro, x0a, dt, tlams, lam1=lam1, n_seed=n_ctrl, iters=si)

    ag, alo, ahi = _geomean_band(ra["ratio_seeds"])
    pg, plo, phi = _geomean_band(rp["ratio_seeds"])
    sg, slo, shi = _geomean_band(su["cost_seeds"])
    tl = np.asarray(tlams)
    print(
        f"[fig2] lam1={lam1:.3f} | acro_reach_geo={ag.tolist()} | "
        f"pend_reach_geo={pg.tolist()} | swing_cost_geo={sg.tolist()}"
    )

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    for g, lo, hi, c, mk, lab in (
        (sg, slo, shi, "C2", "o", "acrobot — swing-up (forgiving)"),
        (pg, plo, phi, "C0", "s", "pendulum — precise reach (integrable)"),
        (ag, alo, ahi, "C3", "D", "acrobot — precise reach (chaotic)"),
    ):
        ax.fill_between(tl, lo, hi, color=c, alpha=0.15)
        ax.plot(tl, g, mk + "-", color=c, label=lab)
    ax.axhline(1.0, ls=":", color="grey", lw=1)
    ax.text(tl[0], 1.4, "do-nothing baseline", fontsize=8, color="grey")
    ax.set_yscale("log")
    ax.set_xlabel(r"horizon $T\lambda_1$ (Lyapunov times)")
    ax.set_ylabel("final cost / do-nothing baseline")
    ax.set_title(
        rf"Predictability horizon bounds precise diff-sim control ($\lambda_1={lam1:.2f}$/s)"
    )
    ax.legend(loc="upper left", fontsize=9)
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

    Points: pendulum (integrable, λ₁≈0), acrobot at a sweep of initial energies (θ₁ sweep,
    λ₁ up to ~1.3/s, non-monotonic), and Hénon-Heiles at a sweep of energies — a second,
    independent chaotic system (a smooth galactic potential, λ₁ up to ~0.13). All cluster on y=x.
    Units are each system's own inverse time (pendulum/acrobot in s⁻¹; Hénon-Heiles is
    dimensionless), so the y=x line, slope = λ₁, is the unit-free statement of the law.
    Cartpole is intentionally excluded (λ₁ is regime-dependent/ill-defined for cartpole).
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

    # --- Hénon-Heiles: a second, independent chaotic system (a smooth galactic potential) ---
    # lambda_1 is O(0.1) and IC-dependent (mixed phase space): chaotic-sea ICs, long horizons.
    from predictability_horizon.systems.henon_heiles import chaotic_sea_ic

    hh = SYSTEMS["henon_heiles"]
    hh_energies: tuple[float, ...] = (0.13, 0.16) if fast else (0.11, 0.13, 0.145, 0.16)
    hh_horizons: npt.NDArray[np.int_] = (
        np.arange(500, 2500, 500) if fast else np.arange(2000, 14000, 2000)
    )
    for hh_e in hh_energies:
        specs.append(
            {
                "name": f"HH E={hh_e}",
                "step_kernel": hh.step_kernel,
                "x0": chaotic_sea_ic(hh_e, hh.default_params),
                "params": hh.default_params,
                "dt": hh.suggested_dt,
                "dim": 4,
                "horizons": hh_horizons,
                "jacobian": hh.jacobian,
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
        mk = "^" if name.startswith("HH") else "o"
        ax.errorbar(lam, slope, xerr=xe, fmt=mk, capsize=3, zorder=3, label=name)
    ax.set_xlabel("measured λ₁ (Benettin, per unit time)   [x-err = finite-time window scatter]")
    ax.set_ylabel("gradient-gain slope (per unit time)")
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


def make_fig7_structured_spectrum(out: Path, fast: bool = False) -> Path:
    """Learned λ₁ vs the true acrobot exponent for the plain MLP, the volume-penalty MLP,
    and the symplectic HNN.

    λ₁ is coordinate-invariant, so the HNN (a canonical (q,p) map, measured along the
    canonical image p=M(θ)ω of the true orbit) is directly comparable to the (θ,ω)-space
    models and the true system. Only the hard symplectic constraint (HNN) brings λ₁ close
    to true; the unconstrained MLP and the soft volume-penalty MLP over-amplify it. The
    HNN is volume-preserving by construction (canonical det J = 1; the spectrum sum ≈ 0 is
    verified in test_structured_models) — the structural reason it does not over-amplify.
    We compare λ₁ (the coordinate-invariant exponent) rather than the spectrum sum, which
    is coordinate-dependent and finite-time-noisy across these heterogeneous models.
    """
    from predictability_horizon.structured_models import (
        hnn_spectrum_on_traj,
        train_hnn,
        train_volume_penalty_mlp,
    )
    from predictability_horizon.worldmodel import (
        make_dataset,
        model_lyapunov_on_traj,
        train_world_model,
    )

    s = SYSTEMS["acrobot"]
    n_traj = 20 if fast else 80
    t_data = 400 if fast else 2000
    epochs = 15 if fast else 120
    hnn_epochs = 15 if fast else 200
    t_meas = 800 if fast else 12000
    x0 = np.array([2.5, 0.0, 0.0, 0.0])

    ds = make_dataset(s, n_traj=n_traj, T=t_data, seed=0)
    base = train_world_model(ds, epochs=epochs, seed=0)
    pen = train_volume_penalty_mlp(ds, epochs=epochs, penalty=1.0, seed=0)
    hnn = train_hnn(ds, s.default_params, s.suggested_dt, epochs=hnn_epochs, seed=0)

    true_traj = rollout(
        cast(wp.Kernel, s.step_kernel),
        x0,
        np.zeros(t_meas),
        s.default_params,
        s.suggested_dt,
        t_meas,
    )[t_meas // 10 :]

    def true_jac(st: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return cast(
            npt.NDArray[np.float64],
            s.jacobian(st, 0.0, s.default_params, s.suggested_dt),
        )

    true_lam = lyapunov_spectrum(true_jac, true_traj, dt=s.suggested_dt, k=1).largest
    base_lam = model_lyapunov_on_traj(base, true_traj, dt=s.suggested_dt, k=1)
    pen_lam = model_lyapunov_on_traj(pen, true_traj, dt=s.suggested_dt, k=1)
    hnn_lam = hnn_spectrum_on_traj(hnn, true_traj, dt=s.suggested_dt, k=1).largest

    names = ["MLP\n(no structure)", "vol-penalty\nMLP (soft)", "HNN\n(symplectic)"]
    lams = [base_lam, pen_lam, hnn_lam]
    colors = ["#d62728", "#ff7f0e", "#2ca02c"]

    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    ax.bar(names, lams, color=colors)
    ax.axhline(true_lam, color="k", ls="--", lw=1.3, label=rf"true $\lambda_1$ ≈ {true_lam:.2f}/s")
    ax.set_ylabel(r"learned $\lambda_1$  (/s)")
    ax.set_title("Only the symplectic HNN recovers the chaotic acrobot's λ₁", fontsize=11)
    ax.legend(fontsize=9)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(
        f"FIG7 lambda1 (/s): true={true_lam:+.3f}  MLP={base_lam:+.3f}  "
        f"penalty={pen_lam:+.3f}  HNN={hnn_lam:+.3f}"
    )
    return out
