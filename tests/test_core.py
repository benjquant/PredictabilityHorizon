import numpy as np

from predictability_horizon.core import Trajectory, fit_loglinear_slope


def test_trajectory_holds_states_and_time():
    states = np.zeros((5, 2))
    traj = Trajectory(states=states, dt=0.01)
    assert traj.horizon == 4  # number of steps = T+1 rows -> T steps
    assert traj.dim == 2
    assert np.isclose(traj.duration, 0.04)


def test_fit_loglinear_slope_recovers_known_rate():
    # y = C * exp(rate * t); slope of log y vs t == rate
    t = np.arange(20, dtype=float)
    y = 3.0 * np.exp(0.25 * t)
    slope, _intercept = fit_loglinear_slope(t, y)
    assert np.isclose(slope, 0.25, atol=1e-6)
