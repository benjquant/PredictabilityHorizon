import numpy as np

from predictability_horizon.experiments import slope_vs_lambda_points
from predictability_horizon.lyapunov import lyapunov_spectrum
from predictability_horizon.systems import (
    SYSTEMS,
    acrobot,  # noqa: F401  (registers "acrobot")
    cartpole,  # noqa: F401  (registers "cartpole")
    henon_heiles,  # noqa: F401  (registers "henon_heiles")
    pendulum,  # noqa: F401  (registers "pendulum")
)
from predictability_horizon.systems.henon_heiles import chaotic_sea_ic
from predictability_horizon.warpsim import autodiff_jacobian, rollout


def test_pendulum_conserves_energy():
    sys = SYSTEMS["pendulum"]
    x0 = np.array([2.0, 0.0])  # released from 2 rad, no damping
    T = 4000  # noqa: N806
    states = rollout(sys.step_kernel, x0, np.zeros(T), sys.default_params, sys.suggested_dt, T)
    e = np.array([sys.energy(s, sys.default_params) for s in states])
    drift = (e.max() - e.min()) / abs(e.mean())
    assert drift < 0.02  # symplectic step keeps energy bounded


def test_pendulum_jacobian_autodiff_matches_analytic():
    sys = SYSTEMS["pendulum"]
    x = np.array([0.7, -0.3])
    Ja = sys.jacobian(x, 0.0, sys.default_params, sys.suggested_dt)  # noqa: N806
    Jad = autodiff_jacobian(sys.step_kernel, x, 0.0, sys.default_params, sys.suggested_dt)  # noqa: N806
    assert np.allclose(Ja, Jad, atol=1e-4)


def test_acrobot_energy_bounded_short_horizon():
    sys = SYSTEMS["acrobot"]
    x0 = np.array([2.5, 0.0, 0.0, 0.0])  # high potential energy -> chaotic swing
    T = 2000  # noqa: N806
    states = rollout(sys.step_kernel, x0, np.zeros(T), sys.default_params, sys.suggested_dt, T)
    e = np.array([sys.energy(s, sys.default_params) for s in states])
    assert (e.max() - e.min()) / abs(e.mean()) < 0.05  # bounded drift over short horizon


def test_acrobot_jacobian_autodiff_matches_analytic():
    sys = SYSTEMS["acrobot"]
    x = np.array([2.5, -0.4, 0.3, 0.1])
    Ja = sys.jacobian(x, 0.0, sys.default_params, sys.suggested_dt)  # noqa: N806
    Jad = autodiff_jacobian(sys.step_kernel, x, 0.0, sys.default_params, sys.suggested_dt)  # noqa: N806
    assert np.allclose(Ja, Jad, atol=1e-3)


def test_acrobot_is_chaotic():
    sys = SYSTEMS["acrobot"]
    x0 = np.array([2.5, 0.0, 0.0, 0.0])
    T = 12000  # noqa: N806
    states = rollout(sys.step_kernel, x0, np.zeros(T), sys.default_params, sys.suggested_dt, T)

    def jac(s: np.ndarray) -> np.ndarray:
        return sys.jacobian(s, 0.0, sys.default_params, sys.suggested_dt)

    res = lyapunov_spectrum(jac, states[1000:], dt=sys.suggested_dt, k=4)
    assert res.largest > 0.5  # clearly positive -> chaos


def test_cartpole_conserves_energy():
    sys = SYSTEMS["cartpole"]
    x0 = np.array([0.0, 0.3, 0.0, 0.0])  # passive rollout from small angle
    # The cartpole kernel uses explicit Euler (not symplectic), so energy drift
    # accumulates ~1.8% over 20 steps (0.4 s) and grows beyond 3% by step 25.
    # We test over 20 steps to confirm the rod-corrected energy formula is
    # consistent with the kernel on a short horizon where drift is bounded.
    T = 20  # noqa: N806
    states = rollout(sys.step_kernel, x0, np.zeros(T), sys.default_params, sys.suggested_dt, T)
    e = np.array([sys.energy(s, sys.default_params) for s in states])
    drift = (e.max() - e.min()) / abs(e.mean())
    assert drift < 0.03  # uniform-rod energy: ~1.8% drift at dt=0.02 over 0.4 s


def test_cartpole_jacobian_autodiff_matches_analytic():
    sys = SYSTEMS["cartpole"]
    x = np.array([0.0, 0.5, 0.2, -0.1])
    Ja = sys.jacobian(x, 0.0, sys.default_params, sys.suggested_dt)  # noqa: N806
    Jad = autodiff_jacobian(sys.step_kernel, x, 0.0, sys.default_params, sys.suggested_dt)  # noqa: N806
    assert np.allclose(Ja, Jad, atol=1e-3)


def test_pendulum_lambda1_near_zero():
    sys = SYSTEMS["pendulum"]
    x0 = np.array([2.0, 0.0])
    # T=8000 steps converges to ~0.079 at 2 rad amplitude; 16000 steps < 0.05.
    # Large-amplitude pendulum needs more Benettin iterations to converge from the
    # random initial Q — spec said T=8000/<0.05 but that bound is too tight for
    # this amplitude; 16000 is the minimal honest fix.
    T = 16000  # noqa: N806
    states = rollout(sys.step_kernel, x0, np.zeros(T), sys.default_params, sys.suggested_dt, T)
    jac = lambda s: sys.jacobian(s, 0.0, sys.default_params, sys.suggested_dt)  # noqa: E731
    res = lyapunov_spectrum(jac, states, dt=sys.suggested_dt, k=2)
    assert abs(res.largest) < 0.05


def test_henon_heiles_conserves_energy():
    sys = SYSTEMS["henon_heiles"]
    x0 = chaotic_sea_ic(0.14)
    T = 20000  # noqa: N806
    states = rollout(sys.step_kernel, x0, np.zeros(T), sys.default_params, sys.suggested_dt, T)
    e = np.array([sys.energy(s, sys.default_params) for s in states])
    drift = (e.max() - e.min()) / abs(e.mean())
    assert drift < 0.02  # symplectic Euler: bounded energy, no secular drift


def test_henon_heiles_jacobian_autodiff_matches_analytic():
    sys = SYSTEMS["henon_heiles"]
    x = np.array([0.1, -0.2, 0.3, 0.15])
    Ja = sys.jacobian(x, 0.0, sys.default_params, sys.suggested_dt)  # noqa: N806
    Jad = autodiff_jacobian(sys.step_kernel, x, 0.0, sys.default_params, sys.suggested_dt)  # noqa: N806
    assert np.allclose(Ja, Jad, atol=1e-4)


def test_henon_heiles_is_exactly_symplectic():
    sys = SYSTEMS["henon_heiles"]
    states = (
        np.array([0.1, -0.2, 0.3, 0.15]),
        np.array([-0.3, 0.25, -0.1, 0.4]),
        chaotic_sea_ic(0.16),
    )
    for x in states:
        for dt in (0.005, 0.01, 0.02):
            J = sys.jacobian(x, 0.0, sys.default_params, dt)  # noqa: N806
            assert abs(np.linalg.det(J) - 1.0) < 1e-10  # separable H -> volume-preserving


def test_henon_heiles_is_chaotic():
    sys = SYSTEMS["henon_heiles"]
    x0 = chaotic_sea_ic(0.16)  # near the escape energy 1/6 -> strong chaos (measured lam1 ~ 0.13)
    T = 20000  # noqa: N806
    states = rollout(sys.step_kernel, x0, np.zeros(T), sys.default_params, sys.suggested_dt, T)
    jac = lambda s: sys.jacobian(s, 0.0, sys.default_params, sys.suggested_dt)  # noqa: E731
    res = lyapunov_spectrum(jac, states[2000:], dt=sys.suggested_dt, k=4)
    assert res.largest > 0.05  # clearly positive -> chaos (measured ~0.134; exact value in ledger)


def test_henon_heiles_regular_at_low_energy():
    sys = SYSTEMS["henon_heiles"]
    x0 = chaotic_sea_ic(0.02)  # deep in the KAM regime (E << 1/12) -> tori
    T = 20000  # noqa: N806
    states = rollout(sys.step_kernel, x0, np.zeros(T), sys.default_params, sys.suggested_dt, T)
    jac = lambda s: sys.jacobian(s, 0.0, sys.default_params, sys.suggested_dt)  # noqa: E731
    res = lyapunov_spectrum(jac, states[2000:], dt=sys.suggested_dt, k=4)
    assert abs(res.largest) < 0.02  # near-integrable: lam1 ~ 0 (measured ~0.008; IC-dependence)


def test_henon_heiles_spectrum_is_volume_preserving():
    sys = SYSTEMS["henon_heiles"]
    x0 = chaotic_sea_ic(0.16)  # strong, clean chaos for a well-resolved spectrum
    T = 20000  # noqa: N806
    states = rollout(sys.step_kernel, x0, np.zeros(T), sys.default_params, sys.suggested_dt, T)
    jac = lambda s: sys.jacobian(s, 0.0, sys.default_params, sys.suggested_dt)  # noqa: E731
    exps = lyapunov_spectrum(jac, states[2000:], dt=sys.suggested_dt, k=4).exponents
    assert exps[0] > 0.05 and exps[3] < -0.05  # a clear +/- pair -> chaos
    assert abs(float(np.sum(exps))) < 2e-3  # symplectic: sum lambda_i = 0 (volume-preserving)
    assert abs(exps[0] + exps[3]) < 0.03  # Hamiltonian +/- pairing (finite-time approx)
    assert abs(exps[1]) < 0.03 and abs(exps[2]) < 0.03  # two ~zero exponents (flow + energy)


def test_slope_vs_lambda_on_henon_heiles():
    sys = SYSTEMS["henon_heiles"]
    spec = {
        "name": "HH E=0.16",
        "step_kernel": sys.step_kernel,
        "x0": chaotic_sea_ic(0.16),
        "params": sys.default_params,
        "dim": 4,
        "dt": sys.suggested_dt,
        "horizons": np.arange(2000, 14000, 2000),
        "jacobian": sys.jacobian,
    }
    ((name, lam, slope),) = slope_vs_lambda_points([spec])
    assert name == "HH E=0.16"
    assert lam > 0.05 and slope > 0.05  # chaotic: both clearly positive
    assert abs(slope - lam) / lam < 0.4  # gradient-gain slope tracks lambda_1 (measured ratio ~0.92)


def _rand_canonical_states(n=6, pmax=6.0, seed=0):
    """Random canonical (theta1, theta2, p1, p2) states spanning the operating envelope."""
    rng = np.random.default_rng(seed)
    return [
        np.array([rng.uniform(-3.2, 3.2), rng.uniform(-3.2, 3.2),
                  pmax * rng.normal(), pmax * rng.normal()])
        for _ in range(n)
    ]


def test_canonical_hamiltonian_matches_legacy_energy():
    """H(theta, p) must be the same physical energy the legacy (theta, omega) formula gave."""
    from predictability_horizon.systems.acrobot import (
        _canonical_energy,
        _legacy_energy,
        to_canonical,
    )

    params = np.array([1.0, 1.0, 1.0, 1.0, 9.81])
    rng = np.random.default_rng(1)
    for _ in range(8):
        s = np.array([rng.uniform(-3, 3), rng.uniform(-3, 3), rng.normal(), rng.normal()])
        assert np.isclose(_canonical_energy(to_canonical(s, params), params),
                          _legacy_energy(s, params), rtol=1e-12)


def test_grad_hamiltonian_is_the_gradient_of_the_hamiltonian():
    """grad H must be the actual gradient of H -- checked against central differences."""
    from predictability_horizon.systems.acrobot import _canonical_energy, _grad_hamiltonian

    params = np.array([1.0, 1.0, 1.0, 1.0, 9.81])
    eps = 1e-6
    for z in _rand_canonical_states(seed=2):
        fd = np.zeros(4)
        for j in range(4):
            e = np.zeros(4)
            e[j] = eps
            fd[j] = (_canonical_energy(z + e, params) - _canonical_energy(z - e, params)) / (2 * eps)
        assert np.allclose(_grad_hamiltonian(z, params), fd, rtol=1e-5, atol=1e-7)


def test_hess_hamiltonian_is_the_derivative_of_the_gradient():
    from predictability_horizon.systems.acrobot import _grad_hamiltonian, _hess_hamiltonian

    params = np.array([1.0, 1.0, 1.0, 1.0, 9.81])
    eps = 1e-6
    for z in _rand_canonical_states(seed=3):
        fd = np.zeros((4, 4))
        for j in range(4):
            e = np.zeros(4)
            e[j] = eps
            fd[:, j] = (_grad_hamiltonian(z + e, params) - _grad_hamiltonian(z - e, params)) / (2 * eps)
        H = _hess_hamiltonian(z, params)  # noqa: N806
        # Construction-guaranteed today (same `col` array written into both cross-blocks),
        # not evidence -- this is a regression guard against a future refactor that
        # assembles the Hessian asymmetrically.
        assert np.allclose(H, H.T, atol=1e-12)  # Hessians are symmetric
        assert np.allclose(H, fd, rtol=1e-4, atol=1e-6)


def test_cayley_jacobian_matches_finite_difference_of_the_numpy_step():
    """The closed-form Jacobian must be the Jacobian of the map _midpoint_np actually computes."""
    from predictability_horizon.systems.acrobot import _cayley_jacobian, _midpoint_np

    params = np.array([1.0, 1.0, 1.0, 1.0, 9.81])
    eps = 1e-6  # balances truncation against float64 roundoff (~1e-16 * |z| / eps)
    for dt in (5e-4, 5e-3):
        for z in _rand_canonical_states(pmax=3.0, seed=4):
            fd = np.zeros((4, 4))
            for j in range(4):
                e = np.zeros(4)
                e[j] = eps
                fd[:, j] = (_midpoint_np(z + e, 0.0, params, dt)[0]
                            - _midpoint_np(z - e, 0.0, params, dt)[0]) / (2 * eps)
            assert np.allclose(_cayley_jacobian(z, 0.0, params, dt), fd, rtol=1e-5, atol=1e-8)


def test_cayley_transform_is_algebraically_symplectic():
    """Certifies the Cayley arithmetic and that the Hessian is assembled symmetric.

    det J = 1 holds identically for J = (I - aA)^-1 (I + aA) whenever A = J4 @ S is
    Hamiltonian with S symmetric -- regardless of whether S is the *correct* physics
    Hessian. This test therefore CANNOT certify that the Hessian encodes the right
    physics, and it must not be cited as the paper's symplecticity evidence. That
    evidence comes from `test_acrobot_is_symplectic_under_autodiff` (added later),
    which differentiates the code that actually ran.
    """
    from predictability_horizon.systems.acrobot import _cayley_jacobian

    params = np.array([1.0, 1.0, 1.0, 1.0, 9.81])
    for dt in (5e-4, 5e-3, 2e-2):
        for z in _rand_canonical_states(seed=5):
            J = _cayley_jacobian(z, 0.0, params, dt)  # noqa: N806
            assert abs(np.linalg.det(J) - 1.0) < 1e-12


def test_fixed_point_saturates_at_six_passes():
    """Criterion 1: two extra Picard passes change the one-step map by less than float32 rounding.

    Per-step, NOT per-trajectory: the acrobot is chaotic, so a single-ulp difference at step 1
    is amplified to O(1) within a few Lyapunov times regardless of solver convergence. A
    trajectory comparison would measure chaos, not convergence.
    """
    from predictability_horizon.systems.acrobot import _canonical_energy, _midpoint_np

    params = np.array([1.0, 1.0, 1.0, 1.0, 9.81])
    eps32 = np.finfo(np.float32).eps
    for dt in (5e-4, 5e-3):
        for z in _rand_canonical_states(n=12, pmax=12.0, seed=6):
            if _canonical_energy(z, params) > 150.0:
                continue  # outside the declared operating envelope
            z6 = _midpoint_np(z, 0.0, params, dt, n_iter=6)[0]
            z8 = _midpoint_np(z, 0.0, params, dt, n_iter=8)[0]
            assert np.linalg.norm(z8 - z6) < eps32 * max(np.linalg.norm(z), 1.0)
