import numpy as np

from predictability_horizon.gradient_law import gradient_law
from predictability_horizon.systems import (
    SYSTEMS,
    acrobot,  # noqa: F401
)
from predictability_horizon.warpsim import linear_step_kernel


def test_linear_map_slope_equals_lambda1():
    a = 1.1

    def jac(_s, _u, _p, _dt):
        return np.array([[a]])

    res = gradient_law(
        linear_step_kernel,
        x0=np.array([1.0]),
        params=np.array([a]),
        dt=1.0,
        horizons=np.arange(5, 40, 5),
        dim=1,
        jacobian=jac,
    )
    # ||d x_T / d x_0|| = |a|^T  ->  log-slope per step = log|a| = lambda_1
    assert np.isclose(res.slope_per_step, np.log(a), rtol=1e-2)
    assert np.isclose(res.lambda1_per_step, np.log(a), rtol=1e-2)


def test_chaotic_system_gradient_slope_tracks_lambda1():
    sys = SYSTEMS["acrobot"]
    # Horizons must give a healthy dynamic range: lambda_1 * dt * Tmax >~ 2 so the
    # spectral norm grows ~10x across the sweep and the log-linear slope is clean.
    res = gradient_law(
        sys.step_kernel,
        x0=np.array([2.5, 0.0, 0.0, 0.0]),
        params=sys.default_params,
        dt=sys.suggested_dt,
        horizons=np.arange(1000, 9000, 1000),
        dim=4,
        jacobian=sys.jacobian,
    )
    assert res.lambda1_per_step > 0.0
    assert np.isclose(res.slope_per_step, res.lambda1_per_step, rtol=0.25)
