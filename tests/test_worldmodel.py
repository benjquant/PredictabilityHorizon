import numpy as np
import pytest

from predictability_horizon.systems import SYSTEMS, acrobot, pendulum  # noqa: F401
from predictability_horizon.worldmodel import (
    error_growth_rate,
    make_dataset,
    model_lyapunov,
    train_world_model,
)


@pytest.mark.integration
def test_trained_model_errors_grow_and_are_ordered_by_chaos() -> None:
    # Comparable physical time across different dt; chaotic (acrobot) must grow faster.
    # With minibatch training the pendulum model converges well → rate_p drops.
    ds_p = make_dataset(SYSTEMS["pendulum"], n_traj=100, T=600, seed=0)  # ~3 s
    ds_a = make_dataset(SYSTEMS["acrobot"], n_traj=100, T=4000, seed=0)  # ~2 s
    m_p = train_world_model(ds_p, epochs=200, seed=0)
    m_a = train_world_model(ds_a, epochs=200, seed=0)
    rate_p = error_growth_rate(m_p, SYSTEMS["pendulum"], x0=np.array([2.0, 0.0]), T=500)
    rate_a = error_growth_rate(m_a, SYSTEMS["acrobot"], x0=np.array([2.5, 0.0, 0.0, 0.0]), T=3000)
    print(f"\nrate_p = {rate_p:.6f} /s,  rate_a = {rate_a:.6f} /s")
    assert rate_a > rate_p  # chaos forces faster per-unit-time error growth


@pytest.mark.integration
def test_learned_model_reproduces_lyapunov_ordering() -> None:
    # A faithful model reproduces the system's lambda_1: chaotic >> integrable.
    ds_p = make_dataset(SYSTEMS["pendulum"], n_traj=100, T=600, seed=0)
    ds_a = make_dataset(SYSTEMS["acrobot"], n_traj=100, T=4000, seed=0)
    m_p = train_world_model(ds_p, epochs=200, seed=0)
    m_a = train_world_model(ds_a, epochs=200, seed=0)
    lam_p = model_lyapunov(m_p, SYSTEMS["pendulum"], x0=np.array([2.0, 0.0]), T=2000)
    lam_a = model_lyapunov(m_a, SYSTEMS["acrobot"], x0=np.array([2.5, 0.0, 0.0, 0.0]), T=4000)
    print(f"\nlam_p = {lam_p:.4f} /s,  lam_a = {lam_a:.4f} /s  (true: pendulum≈0, acrobot≈1.1)")
    assert lam_a > 0.5  # chaotic model learned a clearly positive lambda_1
    assert lam_a > lam_p + 0.3  # and clearly exceeds the integrable model's
