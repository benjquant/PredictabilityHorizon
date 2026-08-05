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


def _fake(label, values):
    from predictability_horizon.partb_ablation import Run

    return [Run(label, i, v, 0.0, 0.0) for i, v in enumerate(values)]


def _fake_result(a, b, c, d, true_lambda1=1.00):
    from predictability_horizon.partb_ablation import AblationResult, penalty_label

    runs = _fake("A", a) + _fake("B", b) + _fake("C", c) + _fake("D", d)
    runs += _fake("plain", [2.4] * 5)
    runs += _fake(penalty_label(0.1), [3.1] * 5)
    runs += _fake(penalty_label(1.0), [2.9] * 5)
    runs += _fake(penalty_label(10.0), [2.6] * 5)
    return AblationResult(
        runs=runs,
        true_lambda1=true_lambda1,
        true_qr_spread=0.03,
        dt=5e-4,
        n_steps=109_200,
        transient_steps=1_200,
        qr_seeds=8,
    )


def test_effect_below_the_seed_spread_is_not_reportable():
    """The rule's whole point: a shift smaller than the noise makes no claim."""
    from predictability_horizon.partb_ablation import summarise

    # A spans 0.90-1.10 (spread 0.20); B's median sits 0.05 away -- inside the noise.
    s = summarise(_fake_result(
        a=[0.90, 0.95, 1.00, 1.05, 1.10],
        b=[0.95, 1.00, 1.05, 1.10, 1.15],
        c=[0.90, 0.95, 1.00, 1.05, 1.10],
        d=[0.95, 1.00, 1.05, 1.10, 1.15],
    ))
    by_name = {e.name: e for e in s.effects}
    assert not by_name["integrator"].reportable
    assert not by_name["data"].reportable
    assert not s.second_panel


def test_effect_above_the_seed_spread_is_reportable_and_earns_a_panel():
    """A shift several times the spread, and a quarter of the shortfall, gets the panel."""
    from predictability_horizon.partb_ablation import summarise

    # A spans 0.60-0.64 (spread 0.04), median 0.62; true 1.00 -> shortfall 0.38.
    # B's median is 0.42 above A -- far outside the spread, and > 0.25 * 0.38.
    s = summarise(_fake_result(
        a=[0.60, 0.61, 0.62, 0.63, 0.64],
        b=[1.02, 1.03, 1.04, 1.05, 1.06],
        c=[0.60, 0.61, 0.62, 0.63, 0.64],
        d=[1.02, 1.03, 1.04, 1.05, 1.06],
    ))
    by_name = {e.name: e for e in s.effects}
    assert by_name["integrator"].reportable
    assert by_name["integrator"].delta == pytest.approx(0.42)
    assert s.second_panel
    assert s.shortfall == pytest.approx(0.38)


def test_non_additive_effects_are_flagged():
    """D exists to catch this: two effects that do not sum are not a decomposition."""
    from predictability_horizon.partb_ablation import summarise

    # integrator +0.20, data +0.20, but D sits +0.80 above A rather than +0.40.
    s = summarise(_fake_result(
        a=[0.60, 0.61, 0.62, 0.63, 0.64],
        b=[0.80, 0.81, 0.82, 0.83, 0.84],
        c=[0.80, 0.81, 0.82, 0.83, 0.84],
        d=[1.40, 1.41, 1.42, 1.43, 1.44],
    ))
    assert not s.additive
    assert s.additivity_residual == pytest.approx(0.40)


def test_best_mu_is_the_one_closest_to_the_true_exponent():
    """The sweep reports the penalty's best case, not an arbitrary setting."""
    from predictability_horizon.partb_ablation import summarise

    s = summarise(_fake_result(
        a=[0.62] * 5, b=[0.62] * 5, c=[0.62] * 5, d=[0.62] * 5
    ))
    assert s.best_mu == 10.0  # medians 3.1 / 2.9 / 2.6 against a true 1.00
