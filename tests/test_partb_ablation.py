import itertools

import numpy as np
import pytest
import torch

from predictability_horizon.partb_ablation import (
    AblationResult,
    Run,
    _symplectic_euler_jacobians,
    build_legacy_dataset,
)
from predictability_horizon.structured_models import HNN, hnn_cayley_jacobians
from predictability_horizon.systems import SYSTEMS, acrobot  # noqa: F401
from predictability_horizon.worldmodel import make_dataset


def test_legacy_dataset_is_initial_condition_matched():
    """Corner C must differ from corner A only in whether the dynamics conserved.

    make_dataset now draws momenta where it once drew velocities, so the same U(-2.5, 2.5)^4
    sampling covers a far quieter energy range. Sampling corner C in (theta, omega) would
    measure that accident on top of the effect being studied. Sampling canonically and mapping
    through to_legacy holds the starting energies fixed.
    """
    s = SYSTEMS["acrobot"]
    n_traj, t_steps = 3, 10
    new = make_dataset(s, n_traj=n_traj, T=t_steps, seed=0)
    old = build_legacy_dataset(s, n_traj=n_traj, T=t_steps, seed=0)
    assert old.x.shape == new.x.shape
    assert old.y.shape == new.y.shape
    for i in range(n_traj):
        a, b = new.x[i * t_steps], old.x[i * t_steps]
        assert np.allclose(a, b, atol=1e-6)
        assert np.isclose(
            s.energy(a, s.default_params), s.energy(b, s.default_params), atol=1e-6
        )


def test_legacy_dataset_leaks_energy_and_the_canonical_one_does_not():
    """...and the retired kernel must actually leak, or corner C measures nothing.

    Compared as an absolute worst-case |dE| over one second rather than a percentage: the
    acrobot's energy straddles zero on these orbits, so a relative drift is ill-defined
    (see _canonical_energy's caveat in systems/acrobot.py).

    Measured: canonical |dE| = 0.00019, legacy |dE| = 0.06054 -- a ~319x ratio, well past
    the 20x floor this test asserts.
    """
    s = SYSTEMS["acrobot"]
    n_traj, t_steps = 4, 2000
    new = make_dataset(s, n_traj=n_traj, T=t_steps, seed=0)
    old = build_legacy_dataset(s, n_traj=n_traj, T=t_steps, seed=0)

    def worst_drift(ds):
        return max(
            abs(
                s.energy(ds.y[i * t_steps + t_steps - 1], s.default_params)
                - s.energy(ds.x[i * t_steps], s.default_params)
            )
            for i in range(n_traj)
        )

    d_new, d_old = worst_drift(new), worst_drift(old)
    print(f"worst |dE| over 1 s: canonical={d_new:.5f}  legacy={d_old:.5f}")
    assert d_old > 20.0 * d_new


def test_result_reports_median_spread_and_window():
    """The summary statistics the reporting rule is written in terms of."""
    runs = [Run("A", s, lam, 0.0, 0.0) for s, lam in enumerate([0.90, 0.94, 1.02, 1.06, 1.30])]
    r = AblationResult(
        runs=runs,
        true_lambda1=1.01,
        true_qr_spread=0.03,
        dt=5e-4,
        n_steps=109_200,
        transient_steps=1_200,
        qr_seeds=8,
    )
    assert r.median("A") == pytest.approx(1.02)
    assert r.spread("A") == pytest.approx(0.40)
    assert r.window_s == pytest.approx(54.0)


def test_the_two_evaluation_integrators_actually_differ():
    """Corners A and B must not be the same measurement.

    If _symplectic_euler_jacobians returned the same Jacobians as the Cayley path, the whole
    integrator arm would silently read zero and look like a null result rather than a bug.

    Run in FLOAT64. The two schemes are first- and second-order, so their Jacobians differ at
    O(dt^2) -- 4.44e-10 at dt=5e-4 on an untrained network, which is 268x BELOW float32's
    resolution of 1.19e-7. In float32 this test measures rounding noise and cannot tell the two
    paths apart at all. Third time this floor has bitten in this plan; see also
    test_hnn_step_solves_the_implicit_midpoint_equation.

    Asserts the dt^2 SCALING, not just a magnitude: the scaling is positive evidence that these
    are the two schemes they claim to be, where a bare threshold only says "not identical".
    Measured 4.4419e-10 / 1.7768e-09 / 7.1070e-09 at dt = 5e-4 / 1e-3 / 2e-3 -- exactly 4.00x
    per doubling.
    """
    states = torch.tensor(
        [[2.5, 0.0, 0.0, 0.0], [1.2, -2.0, 4.0, -3.0]], dtype=torch.float64
    )
    gaps = []
    for dt in (5e-4, 1e-3, 2e-3):
        torch.manual_seed(0)
        hnn = HNN(dt).double()
        hnn.eval()
        j_mid = hnn_cayley_jacobians(hnn, states)
        j_eul = _symplectic_euler_jacobians(hnn, states)
        gaps.append(float((j_mid - j_eul).abs().max().item()))
    print(f"[integrator gap] {[f'{g:.4e}' for g in gaps]}")
    assert gaps[0] > 1e-11  # genuinely different maps, ~44x above the measured float64 gap
    for lo, hi in itertools.pairwise(gaps):
        assert 3.6 < hi / lo < 4.4  # O(dt^2): 4x per doubling of dt


def test_penalty_labels_round_trip():
    """Labels are the join key between the sweep and the reporting rule, so pin their format."""
    from predictability_horizon.partb_ablation import _MUS, penalty_label

    labels = [penalty_label(mu) for mu in _MUS]
    assert labels == ["penalty:mu=0.1", "penalty:mu=1", "penalty:mu=10"]
    assert len(set(labels)) == len(_MUS)
