import numpy as np
import pytest

from predictability_horizon.calib_systems import harmonic_map, lorenz_jacobian_fd, lorenz_rk4_map
from predictability_horizon.lyapunov import lyapunov_spectrum


@pytest.mark.calibration
def test_harmonic_oscillator_has_zero_exponents():
    # Discrete rotation map: energy-conserving, all Lyapunov exponents = 0.
    dt = 0.01
    step, jac = harmonic_map(omega=2.0, dt=dt)
    x0 = np.array([1.0, 0.0])
    traj = [x0]
    for _ in range(20000):
        traj.append(step(traj[-1]))
    res = lyapunov_spectrum(jac, np.array(traj), dt=dt, k=2)
    assert np.allclose(res.exponents, 0.0, atol=1e-3)


@pytest.mark.calibration
def test_lorenz_largest_exponent_matches_literature():
    # Standard Lorenz (sigma=10, rho=28, beta=8/3): literature lambda_1 = 0.9056.
    #
    # The finite-time largest exponent fluctuates by ~0.02 over a few hundred Lyapunov
    # times, so a SINGLE trajectory can sit ~0.05 from the literature value -- the old
    # one-IC test (x0=[1,1,1]) measured 0.880 and was only ~0.002 inside its own 0.05
    # tolerance, i.e. seed-flaky. We instead average lambda_1 over several initial
    # conditions (deterministic RNG, so the test is reproducible), which puts the mean
    # within ~0.002 of 0.9056, and we additionally assert the EXACT sum rule
    # (sum_i lambda_i = tr Df = -(sigma+1+beta)), which is the sharp regression check:
    # it holds to ~1e-4 at every horizon and would break immediately on a bad Jacobian
    # or a mis-ordered QR.
    dt = 0.005
    step = lorenz_rk4_map(dt=dt)
    jac = lorenz_jacobian_fd(step)
    rng = np.random.default_rng(0)
    largest, spectrum_sums = [], []
    for i in range(5):
        x0 = rng.uniform(-12.0, 12.0, 3)
        traj = [x0]
        for _ in range(40000):
            traj.append(step(traj[-1]))
        traj = np.array(traj)[2000:]  # drop transient onto the attractor
        res = lyapunov_spectrum(jac, traj, dt=dt, k=3, seed=i)
        largest.append(res.largest)
        spectrum_sums.append(float(res.exponents.sum()))
    mean_largest = float(np.mean(largest))
    print(f"\nLorenz lambda_1 (mean of 5 ICs) = {mean_largest:.4f}  per-IC={np.round(largest, 4)}")
    # robust: the IC-averaged estimate is stable near the literature value
    assert abs(mean_largest - 0.9056) < 0.05
    # sharp: every spectrum sums to the exact, constant flow divergence
    assert np.allclose(spectrum_sums, -(10.0 + 1.0 + 8.0 / 3.0), atol=1e-3)
