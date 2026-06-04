import numpy as np
import pytest
import torch

from predictability_horizon.structured_models import acrobot_mass_matrix, omega_to_p, p_to_omega


def test_mass_matrix_roundtrip():
    params = torch.tensor([1.0, 1.0, 1.0, 1.0, 9.81])
    theta = torch.tensor([[0.3, -0.7], [1.2, 0.4]])
    omega = torch.tensor([[0.5, -0.2], [1.1, 0.3]])
    p = omega_to_p(theta, omega, params)
    omega2 = p_to_omega(theta, p, params)
    assert torch.allclose(omega, omega2, atol=1e-5)


def test_mass_matrix_matches_kinetic_energy():
    # ½ ωᵀ M ω must equal the acrobot KE used in systems/acrobot.py
    params = torch.tensor([1.0, 1.0, 1.0, 1.0, 9.81])
    th = torch.tensor([[0.3, -0.7]])
    w = torch.tensor([[0.5, -0.2]])
    M = acrobot_mass_matrix(th, params)  # noqa: N806
    ke_M = 0.5 * torch.einsum("bi,bij,bj->b", w, M, w)  # noqa: N806
    m1, m2, l1, l2, _ = params
    th1, th2 = th[0]
    w1, w2 = w[0]
    ke_ref = 0.5 * m1 * (l1 * w1) ** 2 + 0.5 * m2 * (
        (l1 * w1) ** 2 + (l2 * w2) ** 2 + 2 * l1 * l2 * w1 * w2 * torch.cos(th1 - th2)
    )
    assert torch.allclose(ke_M[0], ke_ref, atol=1e-5)


@pytest.mark.integration
def test_volume_penalty_reduces_spectrum_drift():
    from predictability_horizon.structured_models import (
        model_spectrum_sum,
        train_volume_penalty_mlp,
    )
    from predictability_horizon.systems import SYSTEMS, acrobot  # noqa: F401
    from predictability_horizon.worldmodel import make_dataset, train_world_model

    s = SYSTEMS["acrobot"]
    ds = make_dataset(s, n_traj=80, T=2000, seed=0)
    base = train_world_model(ds, epochs=120, seed=0)
    pen = train_volume_penalty_mlp(ds, epochs=120, seed=0, penalty=1.0)
    x0 = np.array([2.5, 0.0, 0.0, 0.0])
    sb = model_spectrum_sum(base, s, x0)
    sp = model_spectrum_sum(pen, s, x0)
    print(f"spectrum_sum base={sb:.3f} pen={sp:.3f}")
    # the penalised model's Lyapunov spectrum should sum closer to 0 (volume-preserving)
    assert abs(sp) < abs(sb)


@pytest.mark.integration
def test_hnn_spectrum_on_traj_is_canonical():
    from predictability_horizon.structured_models import hnn_spectrum_on_traj, train_hnn
    from predictability_horizon.systems import SYSTEMS, acrobot  # noqa: F401
    from predictability_horizon.warpsim import rollout
    from predictability_horizon.worldmodel import make_dataset

    s = SYSTEMS["acrobot"]
    ds = make_dataset(s, n_traj=80, T=2000, seed=0)
    hnn = train_hnn(ds, s.default_params, s.suggested_dt, epochs=120, seed=0)
    traj = rollout(
        s.step_kernel,
        np.array([2.5, 0, 0, 0.0]),
        np.zeros(4000),
        s.default_params,
        s.suggested_dt,
        4000,
    )[400:]
    spec = hnn_spectrum_on_traj(hnn, traj, dt=s.suggested_dt, k=4)
    assert abs(float(np.sum(spec.exponents))) < 0.2  # volume-preserving (canonical, exact)
    assert spec.largest > 0.3  # a sensible positive exponent, not garbage


@pytest.mark.integration
def test_hnn_is_near_volume_preserving():
    from predictability_horizon.structured_models import model_spectrum_sum, train_hnn
    from predictability_horizon.systems import SYSTEMS, acrobot  # noqa: F401
    from predictability_horizon.worldmodel import make_dataset

    s = SYSTEMS["acrobot"]
    ds = make_dataset(s, n_traj=80, T=2000, seed=0)
    hnn = train_hnn(ds, s.default_params, s.suggested_dt, epochs=200, seed=0)
    ss = model_spectrum_sum(hnn, s, np.array([2.5, 0.0, 0.0, 0.0]))
    print(f"HNN spectrum_sum={ss:.4f}")
    # symplectic-by-construction ⇒ spectrum sum near 0 (loose tol; explicit integrator drifts)
    assert abs(ss) < 0.3
