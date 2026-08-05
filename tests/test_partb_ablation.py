import numpy as np

from predictability_horizon.partb_ablation import build_legacy_dataset
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
