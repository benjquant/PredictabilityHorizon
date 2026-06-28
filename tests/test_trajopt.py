import numpy as np

from predictability_horizon.systems import (  # noqa: F401  (registers systems)
    SYSTEMS,
    acrobot,
    pendulum,
)
from predictability_horizon.trajopt import hand_height, optimize_swingup, reach_ratio

X0A = np.array([2.5, 0.0, 0.0, 0.0])
X0P = np.array([2.0, 0.0])
DT = 5e-3  # coarse dt keeps tests fast; the wall (in Lyapunov time) is dt-robust

# At dt=5e-3 the acrobot has lambda1 ~ 1.2 /s (1/lambda1 ~ 160 steps). The PRECISE-reach wall sits
# at the predictability horizon WITH its log factor: T*·lambda1 ~ log(1/tau) ~ 4.6 (tau=1e-2),
# i.e. ~750 steps. So we probe two regimes by explicit step count (no hardcoded lambda1):
STEPS_OK = 320  # ~T·lambda1 = 2: PAST 1/lambda1 but before the wall -> precise reach SUCCEEDS
STEPS_WALL = 810  # ~T·lambda1 = 5: past the log-factor wall -> precise reach FAILS (chaos only)


def test_reach_succeeds_well_past_lyapunov_time():
    # The single-trajectory gradient is EXACT: precise reaching succeeds well past 1/lambda1 given
    # an adequate budget. (The wall is NOT at 1/lambda1.)
    r = reach_ratio(SYSTEMS["acrobot"], X0A, DT, STEPS_OK, iters=150, seed=0)
    assert r < 1e-2


def test_reach_fails_past_log_horizon():
    # The wall: past T·lambda1 ~ log(1/tau) the e^{2 lambda1 T} gain defeats the optimiser; precise
    # reach recovers nothing -- final cost is of order the do-nothing baseline.
    r = reach_ratio(SYSTEMS["acrobot"], X0A, DT, STEPS_WALL, iters=200, seed=0)
    assert r > 5e-1  # recovers nothing: final cost ≈ (or worse than) the do-nothing baseline


def test_pendulum_no_wall_at_same_horizon():
    # Chaos-specificity: the integrable pendulum recovers at the SAME long horizon the acrobot fails.
    r = reach_ratio(SYSTEMS["pendulum"], X0P, DT, STEPS_WALL, iters=200, seed=0)
    assert r < 1e-2


def test_acrobot_swings_up_past_the_wall():
    # The FORGIVING objective (hand anywhere up) sails through the horizon where precise reaching dies.
    _, final = optimize_swingup(SYSTEMS["acrobot"], X0A, DT, STEPS_WALL, iters=200, lr=1.0, seed=0)
    assert hand_height(final) > 1.5
