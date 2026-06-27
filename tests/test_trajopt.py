import numpy as np

from predictability_horizon.systems import (  # noqa: F401  (registers systems)
    SYSTEMS,
    acrobot,
    pendulum,
)
from predictability_horizon.trajopt import reach_ratio

X0A = np.array([2.5, 0.0, 0.0, 0.0])
X0P = np.array([2.0, 0.0])
DT = 5e-3
LAM1 = 0.86  # acrobot λ₁ at dt=5e-3 (spike-measured); sets matched step counts on the Lyapunov axis


def _steps(tl):
    return max(2, round(tl / (LAM1 * DT)))


def test_reach_recovers_inside_horizon():
    # Inside 1/λ₁ the analytic gradient recovers a feasible target almost exactly.
    r = reach_ratio(SYSTEMS["acrobot"], X0A, DT, _steps(0.5), iters=100, seed=0)
    assert r < 1e-2


def test_reach_fails_past_horizon():
    # The wall: precise reaching collapses past the predictability horizon.
    short = reach_ratio(SYSTEMS["acrobot"], X0A, DT, _steps(0.5), iters=100, seed=0)
    long = reach_ratio(SYSTEMS["acrobot"], X0A, DT, _steps(3.0), iters=100, seed=0)
    assert long > 5e-2
    assert long > 20 * short


def test_pendulum_no_wall():
    # Chaos-specificity control: the integrable pendulum recovers at the SAME horizon the acrobot fails.
    r = reach_ratio(SYSTEMS["pendulum"], X0P, DT, _steps(3.0), iters=100, seed=0)
    assert r < 1e-2


def test_acrobot_swings_up_inside_horizon():
    # Forgiving objective: the hand reaches upright even at a short horizon.
    from predictability_horizon.trajopt import hand_height, optimize_swingup

    _, final = optimize_swingup(SYSTEMS["acrobot"], X0A, DT, _steps(1.0), iters=180, lr=1.0, seed=0)
    assert hand_height(final) > 1.0


def test_acrobot_swings_up_past_horizon():
    # The forgiving objective sails through the horizon where precise reaching fails.
    from predictability_horizon.trajopt import hand_height, optimize_swingup

    _, final = optimize_swingup(SYSTEMS["acrobot"], X0A, DT, _steps(3.0), iters=180, lr=1.0, seed=0)
    assert hand_height(final) > 1.0
