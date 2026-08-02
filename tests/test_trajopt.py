import numpy as np
import pytest

from predictability_horizon.systems import (  # noqa: F401  (registers systems)
    SYSTEMS,
    acrobot,
    pendulum,
)
from predictability_horizon.trajopt import (
    hand_height,
    measure_lambda1,
    optimize_swingup,
    reach_ratio,
)

X0A = np.array([2.5, 0.0, 0.0, 0.0])
X0P = np.array([2.0, 0.0])
DT = 5e-3  # coarse dt keeps tests fast; the wall (in Lyapunov time) is dt-robust

# STEPS_OK / STEPS_WALL are DERIVED from the measured lambda_1 at DT, not tuned to pass:
#   STEPS_OK   = round(2 / (lambda_1 * DT))  -- past 1/lambda_1, before the wall
#   STEPS_WALL = round(5 / (lambda_1 * DT))  -- past the log-factor wall, log(1/eps) ~ 4.6
# Re-derived 2026-08-01 for the canonical symplectic acrobot (lambda_1 = 1.0519/s at DT; was
# ~1.235/s in the retired velocity coordinates -- STEPS_WALL was 810 there, T*lambda1 = 5.00 in
# that frame but only 4.26 in this one). Recompute, never hand-tune: the constants ARE the claim.
STEPS_OK = 380
STEPS_WALL = 951

# reach_ratio at a single (T, seed) spans FIVE ORDERS OF MAGNITUDE near and past the wall (e.g. at
# STEPS_WALL: 4.65, 1.08, 0.29, 3.55, 10.9, 373 across seeds 0-5) -- a single seed is a coin flip.
# The median is the outlier-robust choice on such heavy-tailed statistics. Note: this deliberately
# differs from reach_horizon_sweep (arithmetic mean) and the figures (geometric mean), so no shared
# aggregation assumption is valid across test, code, and plots.
N_SEED = 6


def _median_reach_ratio(system_name, x0, dt, steps, iters):
    ratios = [
        reach_ratio(SYSTEMS[system_name], x0, dt, steps, iters=iters, seed=sd) for sd in range(N_SEED)
    ]
    return float(np.median(ratios))


def test_reach_succeeds_well_past_lyapunov_time():
    # The single-trajectory gradient is EXACT: precise reaching succeeds well past 1/lambda1 given
    # an adequate budget. (The wall is NOT at 1/lambda1.)
    r = _median_reach_ratio("acrobot", X0A, DT, STEPS_OK, iters=150)
    assert r < 1e-2


def test_reach_fails_past_log_horizon():
    # The wall: past T·lambda1 ~ log(1/tau) the e^{2 lambda1 T} gain defeats the optimiser; precise
    # reach recovers nothing -- median final cost is of order the do-nothing baseline.
    r = _median_reach_ratio("acrobot", X0A, DT, STEPS_WALL, iters=200)
    assert r > 5e-1  # recovers nothing: median final cost ≈ (or worse than) the do-nothing baseline


def test_pendulum_no_wall_at_same_horizon():
    # Chaos-specificity: the integrable pendulum recovers at the SAME long horizon the acrobot fails.
    r = _median_reach_ratio("pendulum", X0P, DT, STEPS_WALL, iters=200)
    assert r < 1e-2


def test_acrobot_swings_up_past_the_wall():
    # The FORGIVING objective (hand anywhere up) sails through the horizon where precise reaching dies.
    heights = []
    for sd in range(N_SEED):
        _, final = optimize_swingup(SYSTEMS["acrobot"], X0A, DT, STEPS_WALL, iters=200, lr=1.0, seed=sd)
        heights.append(hand_height(final))
    assert float(np.median(heights)) > 1.5


DT_FIGURES = 5e-4  # the acrobot's suggested_dt -- what every figure actually runs


@pytest.mark.integration
def test_reach_succeeds_at_the_figures_timestep():
    """The wall claim, exercised at the dt the figures use rather than the tests' coarse one.

    The energy leak this spec fixed survived for years because the test checked a 1 s window
    while the figures ran 5.5-10 s. tests/test_trajopt.py is the only place in the repo not
    using the acrobot's suggested_dt, so at least one case must run at the production dt or
    the same class of blind spot stays open.
    """
    lam = measure_lambda1(SYSTEMS["acrobot"], X0A, DT_FIGURES)
    steps_ok = round(2.0 / (lam * DT_FIGURES))
    r = reach_ratio(SYSTEMS["acrobot"], X0A, DT_FIGURES, steps_ok, iters=150, seed=0)
    assert r < 1e-2
