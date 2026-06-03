import numpy as np

from predictability_horizon.experiments import (
    gradient_snr_vs_horizon,
    lambda_spread,
    slope_vs_lambda_points,
)
from predictability_horizon.systems import SYSTEMS, acrobot, pendulum  # noqa: F401
from predictability_horizon.warpsim import linear_step_kernel, rollout


def test_linear_map_point_on_identity():
    a = 1.1

    def jac(_s, _u, _p, _dt):
        return np.array([[a]])

    pts = slope_vs_lambda_points(
        [
            {
                "name": "linear",
                "step_kernel": linear_step_kernel,
                "x0": np.array([1.0]),
                "params": np.array([a]),
                "dt": 1.0,
                "dim": 1,
                "horizons": np.arange(5, 40, 5),
                "jacobian": jac,
            },
        ]
    )
    _name, lam, slope = pts[0]
    assert abs(slope - np.log(a)) < 1e-2 and abs(lam - np.log(a)) < 1e-2  # on y=x


def test_acrobot_energy_sweep_spans_lambda():
    s = SYSTEMS["acrobot"]
    specs = [
        {
            "name": f"acro_{th}",
            "step_kernel": s.step_kernel,
            "x0": np.array([th, 0.0, 0.0, 0.0]),
            "params": s.default_params,
            "dt": s.suggested_dt,
            "dim": 4,
            "horizons": np.arange(1000, 9000, 1000),
            "jacobian": s.jacobian,
        }
        for th in (0.5, 2.5)
    ]
    pts = slope_vs_lambda_points(specs)
    lams = [p[1] for p in pts]
    assert lams[1] > lams[0]  # higher energy -> larger λ₁ (a continuum)
    assert all(p[2] > 0 for p in pts)  # positive slopes


def test_lambda_spread_returns_mean_std():
    s = SYSTEMS["acrobot"]
    traj = rollout(
        s.step_kernel,
        np.array([2.5, 0, 0, 0.0]),
        np.zeros(6000),
        s.default_params,
        s.suggested_dt,
        6000,
    )

    def jac(state):
        return s.jacobian(state, 0.0, s.default_params, s.suggested_dt)

    mean, std = lambda_spread(jac, traj[500:], s.suggested_dt, k=4, seeds=range(5))
    assert std >= 0 and mean > 0


def test_gradient_snr_shape_and_finite_on_linear_map():
    # On a linear map the e^{λ₁T} factor cancels between signal and noise, so the SNR
    # is flat (not collapsing). Here we only assert shape + finiteness/positivity; the
    # chaotic collapse is demonstrated on the acrobot in the Fig-6 experiment (Task 2).
    horizons = np.arange(5, 30, 5)
    h, snr = gradient_snr_vs_horizon(
        linear_step_kernel,
        np.array([1.0]),
        np.array([1.1]),
        1.0,
        horizons,
        dim=1,
        n_ic=4,
        eps=1e-3,
        seed=0,
    )
    assert h.shape == (len(horizons),)
    assert snr.shape == (len(horizons),)
    assert np.all(np.isfinite(snr)) and np.all(snr > 0)
