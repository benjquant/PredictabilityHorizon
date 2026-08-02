"""Part A §4.4: trajectory optimisation through the differentiable simulator.

Two objectives, both Adam on an open-loop control sequence backpropagated through a Warp rollout:
  - reach   : hit a feasible target terminal state x*_T (PRECISE)  -> works inside 1/λ₁, fails past.
  - swingup : raise the acrobot's hand to upright            (FORGIVING) -> works at all horizons.
The contrast isolates the predictability horizon: it bounds precise control, not forgiving control.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt
import warp as wp

from predictability_horizon.lyapunov import lyapunov_spectrum
from predictability_horizon.systems import System
from predictability_horizon.warpsim import _alloc_states, _sqdist_loss, init_warp, rollout

Vec = npt.NDArray[np.float64]


def _steps(tl: float, lam1: float, dt: float) -> int:
    """Integration steps for a horizon of ``tl`` Lyapunov times at rate ``lam1`` and step ``dt``."""
    return max(2, round(tl / (lam1 * dt)))


def measure_lambda1(
    system: System, x0: Vec, dt: float, t_phys: float = 8.0, transient_phys: float = 1.0
) -> float:
    """Benettin/QR largest Lyapunov exponent (per unit time) on the passive trajectory from ``x0``.

    Rolls out ``t_phys`` seconds (default 8 s = ``round(t_phys / dt)`` steps), discards the
    first ``transient_phys`` seconds (default 1 s) so the Benettin QR renormalisation has
    settled onto the leading Oseledets direction, then averages the log-growth over the
    remaining ~7 s. ``t_phys`` is honoured exactly -- there is no internal step cap. This is
    not free: at the production ``dt=5e-4`` this is 16000 Jacobian evaluations (vs. 1600 at the
    coarse test ``dt=5e-3``). A previous ``min(8000, ...)`` cap silently halved the window at
    dt=5e-4 (8 s -> 4 s) and understated lambda_1 by 11% (0.8953 vs. the honoured 1.0066 /s).
    """
    n = round(t_phys / dt)
    traj = rollout(system.step_kernel, x0, np.zeros(n), system.default_params, dt, n)  # type: ignore[arg-type]
    jac = lambda s: system.jacobian(s, 0.0, system.default_params, dt)  # noqa: E731
    tr = round(transient_phys / dt)
    return float(lyapunov_spectrum(jac, traj[tr:], dt, k=system.dim).largest)


def _optimize_actions(
    system: System,
    x0: Vec,
    dt: float,
    T: int,  # noqa: N803
    u_init: Vec,
    build_loss: Callable[[Any, Any, Any], None],
    iters: int,
    lr: float,
    device: str,
) -> tuple[Vec, Vec]:
    """Adam on an open-loop action sequence through the diff-sim. Returns (loss_history, final_state).

    ``build_loss(states, actions, loss)`` launches the objective's cost kernel onto the open tape.
    """
    init_warp()
    actions: wp.array[Any] = wp.array(
        u_init.astype(np.float32), dtype=wp.float32, requires_grad=True, device=device
    )
    pars: wp.array[Any] = wp.array(
        system.default_params.astype(np.float32), dtype=wp.float32, device=device
    )
    m = np.zeros(T)
    v = np.zeros(T)
    b1, b2, eps = 0.9, 0.999, 1e-8
    history: list[float] = []
    for it in range(iters):
        states = _alloc_states(x0, T, requires_grad=True, device=device)
        loss = wp.zeros(1, dtype=wp.float32, requires_grad=True, device=device)
        tape = wp.Tape()
        with tape:
            for t in range(T):
                wp.launch(
                    system.step_kernel, dim=1, inputs=[states, actions, pars, float(dt), t], device=device
                )
            build_loss(states, actions, loss)
        tape.backward(loss=loss)
        g = actions.grad.numpy()
        history.append(float(loss.numpy()[0]))
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        mh = m / (1 - b1 ** (it + 1))
        vh = v / (1 - b2 ** (it + 1))
        anew = actions.numpy() - lr * mh / (np.sqrt(vh) + eps)
        actions = wp.array(anew.astype(np.float32), dtype=wp.float32, requires_grad=True, device=device)
        tape.zero()
    final = rollout(system.step_kernel, x0, actions.numpy(), system.default_params, dt, T)[T]  # type: ignore[arg-type]
    return np.array(history), final


def optimize_reach(
    system: System,
    x0: Vec,
    dt: float,
    T: int,  # noqa: N803
    target: Vec,
    iters: int = 150,
    lr: float = 0.3,
    sigma0: float = 0.0,
    seed: int = 0,
    device: str = "cpu",
) -> tuple[Vec, Vec]:
    """Optimise the control to reach terminal ``target`` (minimise ||x_T - target||²)."""
    rng = np.random.default_rng(seed)
    u_init = sigma0 * rng.standard_normal(T)
    tgt: wp.array[Any] = wp.array(np.asarray(target, np.float32), dtype=wp.float32, device=device)

    def build_loss(states: Any, actions: Any, loss: Any) -> None:
        wp.launch(_sqdist_loss, dim=1, inputs=[states, tgt, T, loss], device=device)

    return _optimize_actions(system, x0, dt, T, u_init, build_loss, iters, lr, device)


def reach_ratio(
    system: System,
    x0: Vec,
    dt: float,
    T: int,  # noqa: N803
    sigma_ref: float = 1.0,
    iters: int = 150,
    lr: float = 0.3,
    seed: int = 0,
    device: str = "cpu",
) -> float:
    """final/baseline ratio for a *feasible* target (built from a known control u_ref).

    A zero-cost solution provably exists at every horizon, so any failure is gradient quality.
    Ratio ≪ 1 = recovered; ≳ 1 = no better than doing nothing.
    """
    rng = np.random.default_rng(1000 + seed)
    u_ref = sigma_ref * rng.standard_normal(T)
    target = rollout(system.step_kernel, x0, u_ref, system.default_params, dt, T)[T]  # type: ignore[arg-type]
    passive = rollout(system.step_kernel, x0, np.zeros(T), system.default_params, dt, T)[T]  # type: ignore[arg-type]
    c0 = float(np.sum((passive - target) ** 2)) + 1e-12
    _, final = optimize_reach(
        system, x0, dt, T, target, iters=iters, lr=lr, sigma0=0.0, seed=seed, device=device
    )
    cf = float(np.sum((final - target) ** 2))
    return cf / c0


def reach_horizon_sweep(
    system: System,
    x0: Vec,
    dt: float,
    tlams: list[float],
    lam1: float | None = None,
    n_seed: int = 5,
    iters: int = 150,
    lr: float = 0.3,
    sigma_ref: float = 1.0,
    device: str = "cpu",
) -> dict[str, Any]:
    """reach_ratio across a horizon grid (in Lyapunov times ``tlams`` on ``lam1``'s axis).

    Pass ``lam1`` explicitly (e.g. the acrobot's) to place a second system on a shared Lyapunov axis
    with matched step counts; if ``None`` it is measured for ``system``.
    """
    if lam1 is None:
        lam1 = measure_lambda1(system, x0, dt)
    out: dict[str, Any] = {"lam1": lam1, "tlams": [], "steps": [], "ratio_mean": [], "ratio_seeds": []}
    for tl in tlams:
        T = _steps(tl, lam1, dt)  # noqa: N806
        ratios = [
            reach_ratio(system, x0, dt, T, sigma_ref=sigma_ref, iters=iters, lr=lr, seed=sd, device=device)
            for sd in range(n_seed)
        ]
        out["tlams"].append(tl)
        out["steps"].append(T)
        out["ratio_mean"].append(float(np.mean(ratios)))
        out["ratio_seeds"].append(ratios)
    return out


@wp.kernel
def _terminal_upright_cost(
    states: wp.array2d(dtype=wp.float32),  # type: ignore[valid-type]
    actions: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
    T: int,  # noqa: N803
    loss: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
) -> None:
    """Acrobot hand-up: 0 when both links are vertical (θ=π), 4 hanging. Terminal step only.

    The regulariser is T-INDEPENDENT (per-step 1e-7) so it never suppresses the swing-up.
    """
    c = (wp.float32(1.0) + wp.cos(states[T, 0])) + (wp.float32(1.0) + wp.cos(states[T, 1]))
    for t in range(T):
        c = c + wp.float32(1.0e-7) * actions[t] * actions[t]
    loss[0] = c


def hand_height(state: Vec) -> float:
    """Acrobot hand (link-2 tip) height: -2 hanging, 0 horizontal, +2 straight up."""
    return float(-(np.cos(state[0]) + np.cos(state[1])))


def upright_cost(state: Vec) -> float:
    """Terminal upright cost (1+cos θ₁)+(1+cos θ₂): 0 at upright, 4 hanging (matches the kernel)."""
    return float((1.0 + np.cos(state[0])) + (1.0 + np.cos(state[1])))


def optimize_swingup(
    system: System,
    x0: Vec,
    dt: float,
    T: int,  # noqa: N803
    iters: int = 250,
    lr: float = 1.0,
    sigma0: float = 0.3,
    seed: int = 0,
    device: str = "cpu",
) -> tuple[Vec, Vec]:
    """Raise the acrobot's hand to upright (terminal-upright cost). Returns (loss_history, final_state).

    Acrobot-specific: the cost assumes the first two state components are the two link angles.
    """
    rng = np.random.default_rng(seed)
    u_init = sigma0 * rng.standard_normal(T)

    def build_loss(states: Any, actions: Any, loss: Any) -> None:
        wp.launch(_terminal_upright_cost, dim=1, inputs=[states, actions, T, loss], device=device)

    return _optimize_actions(system, x0, dt, T, u_init, build_loss, iters, lr, device)


def swingup_horizon_sweep(
    system: System,
    x0: Vec,
    dt: float,
    tlams: list[float],
    lam1: float | None = None,
    n_seed: int = 5,
    iters: int = 250,
    lr: float = 1.0,
    device: str = "cpu",
) -> dict[str, Any]:
    """Per-horizon swing-up results: final hand height, #up, and the normalized terminal-upright cost.

    ``cost_seeds`` = final upright cost / passive (u=0) upright cost — the swing-up analogue of the
    reach ratio, so swing-up sits on the same normalized-cost axis as reaching (≪1 = succeeds).
    """
    if lam1 is None:
        lam1 = measure_lambda1(system, x0, dt)
    out: dict[str, Any] = {
        "lam1": lam1, "tlams": [], "steps": [], "height_mean": [], "height_seeds": [],
        "n_up": [], "cost_seeds": [],
    }
    for tl in tlams:
        T = _steps(tl, lam1, dt)  # noqa: N806
        passive = rollout(system.step_kernel, x0, np.zeros(T), system.default_params, dt, T)[T]  # type: ignore[arg-type]
        c0 = upright_cost(passive) + 1e-12
        finals = [optimize_swingup(system, x0, dt, T, iters=iters, lr=lr, seed=sd, device=device)[1] for sd in range(n_seed)]
        hs = [hand_height(f) for f in finals]
        out["tlams"].append(tl)
        out["steps"].append(T)
        out["height_mean"].append(float(np.mean(hs)))
        out["height_seeds"].append(hs)
        out["n_up"].append(int(sum(1 for h in hs if h > 1.5)))
        out["cost_seeds"].append([upright_cost(f) / c0 for f in finals])
    return out
