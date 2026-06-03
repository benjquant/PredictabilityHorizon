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
    # Standard Lorenz (sigma=10, rho=28, beta=8/3): lambda_1 ~ 0.906.
    dt = 0.005
    step = lorenz_rk4_map(dt=dt)
    x0 = np.array([1.0, 1.0, 1.0])
    traj = [x0]
    for _ in range(40000):
        traj.append(step(traj[-1]))
    traj = np.array(traj)[2000:]  # drop transient onto the attractor
    jac = lorenz_jacobian_fd(step)
    res = lyapunov_spectrum(jac, traj, dt=dt, k=3)
    print(f"\nMeasured Lorenz lambda_1 = {res.largest:.4f}")
    assert abs(res.largest - 0.906) < 0.05
