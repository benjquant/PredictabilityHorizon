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


def _pdot_rel_err(hnn, ds, dt: float, n: int = 256) -> float:
    """Relative error of the trained HNN's momentum-rate vector field against the finite-
    difference pdot of the data it was fit to. This is what moves under training or a
    coordinate regression -- unlike a symplectic-Euler determinant, which is architecture-
    guaranteed and reads the same whether or not the network has been trained at all."""
    x = torch.tensor(ds.x[:n], dtype=torch.float32)
    y = torch.tensor(ds.y[:n], dtype=torch.float32)
    q = x[:, :2].detach().requires_grad_(True)
    p = x[:, 2:].detach().requires_grad_(True)
    _, pd = hnn.vector_field(q, p)
    pdot_ref = (y[:, 2:] - x[:, 2:]) / dt
    return (torch.norm(pd - pdot_ref) / torch.norm(pdot_ref)).item()


@pytest.mark.integration
def test_hnn_spectrum_on_traj_is_canonical():
    """Spectrum sum near 0 is architecture-guaranteed; the pdot fit is what certifies training.

    ``hnn_spectrum_on_traj``'s exponent sum reads near 0 for an HNN because its symplectic-
    Euler one-step map has det J ~= 1 (to O(dt^2)) EVERYWHERE in (theta, p) -- true for a
    randomly-initialised, untrained network exactly as for a trained one. Measured on an
    untrained HNN(params, dt): sum=-0.000005, largest=0.0036 (both comfortably inside the
    bounds below, on pure noise). So ``abs(sum) < 0.2`` alone certifies the architecture, not
    the fit. ``largest > 0.3`` is somewhat discriminating on its own (0.0036 untrained would
    fail it) but is loose: 0.31 and 5.0 both pass it against a true lambda_1 ~ 0.99. The pdot
    relative-error assertion below is what actually ties this test to the trained model:
    measured untrained rel err ~1.00 (fails the < 0.5 bound below) vs. trained (120 epochs)
    0.0024 -- a ~400x margin.
    """
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
    # architecture-only (see docstring): passes on an untrained network by construction.
    assert abs(float(np.sum(spec.exponents))) < 0.2  # volume-preserving (canonical, exact)
    assert spec.largest > 0.3  # a sensible positive exponent, not garbage

    rel_pdot = _pdot_rel_err(hnn, ds, s.suggested_dt)
    print(f"HNN pdot relative error = {rel_pdot:.4f}")
    # discriminates training: untrained ~1.00 (fails this bound), trained 0.0024 (~400x margin)
    assert rel_pdot < 0.5


@pytest.mark.integration
def test_hnn_is_near_volume_preserving():
    """Spectrum sum near 0 is architecture-guaranteed; the pdot fit is what certifies training.

    ``model_spectrum_sum`` reading near 0 is an ARCHITECTURE property of the HNN: its
    symplectic-Euler step has det J ~= 1 (to O(dt^2)) at every (theta, p), true for a
    randomly-initialised, untrained network exactly as for a trained one (measured: untrained
    spectrum_sum=-0.000005, trained (200 epochs)=-0.009869 -- both comfortably inside the
    bound below). So ``abs(ss) < 0.3`` alone certifies the architecture, not training, data,
    or coordinates -- see the caveat in ``structured_models.model_spectrum_sum``. The pdot
    relative-error assertion below is what actually moves under training or a coordinate
    regression: measured untrained pdot rel err ~1.00 (fails the < 0.5 bound below) vs.
    trained 0.0026 -- a ~190x margin.
    """
    from predictability_horizon.structured_models import model_spectrum_sum, train_hnn
    from predictability_horizon.systems import SYSTEMS, acrobot  # noqa: F401
    from predictability_horizon.worldmodel import make_dataset

    s = SYSTEMS["acrobot"]
    ds = make_dataset(s, n_traj=80, T=2000, seed=0)
    hnn = train_hnn(ds, s.default_params, s.suggested_dt, epochs=200, seed=0)
    ss = model_spectrum_sum(hnn, s, np.array([2.5, 0.0, 0.0, 0.0]))
    print(f"HNN spectrum_sum={ss:.4f}")
    # architecture-only (see docstring): passes on an untrained network by construction.
    assert abs(ss) < 0.3

    rel_pdot = _pdot_rel_err(hnn, ds, s.suggested_dt)
    print(f"HNN pdot relative error = {rel_pdot:.4f}")
    # discriminates training: untrained ~1.00 (fails this bound), trained 0.0026 (~190x margin)
    assert rel_pdot < 0.5


@pytest.mark.integration
def test_hnn_trains_directly_on_canonical_data():
    """The simulator now emits (theta, p), so train_hnn must NOT convert again.

    A second application of M(theta) would inflate the momenta and the learned vector field
    would not match the data it was fit to. Checked by requiring the trained Hamiltonian
    vector field to reproduce the finite-difference derivatives of the training data.

    The momentum-rate channel (pdot) is the discriminating one: a reintroduced double
    application of M(theta) lands on the momenta, not the angles, so it is the pdot relative
    error that catches it — measured at 1.078 before the fix vs. 0.006 after, against a
    qdot channel that only moves 0.325 -> 0.016 and would not, by itself, cross the < 0.5
    threshold either way.
    """
    import torch

    from predictability_horizon.structured_models import train_hnn
    from predictability_horizon.systems import SYSTEMS, acrobot  # noqa: F401
    from predictability_horizon.worldmodel import make_dataset

    s = SYSTEMS["acrobot"]
    ds = make_dataset(s, n_traj=40, T=1000, seed=0)
    hnn = train_hnn(ds, s.default_params, s.suggested_dt, epochs=60, seed=0)

    x = torch.tensor(ds.x[:256], dtype=torch.float32)
    y = torch.tensor(ds.y[:256], dtype=torch.float32)
    q = x[:, :2].detach().requires_grad_(True)
    p = x[:, 2:].detach().requires_grad_(True)
    qd, pd = hnn.vector_field(q, p)
    qdot_ref = (y[:, :2] - x[:, :2]) / s.suggested_dt
    pdot_ref = (y[:, 2:] - x[:, 2:]) / s.suggested_dt
    rel = (torch.norm(qd - qdot_ref) / torch.norm(qdot_ref)).item()
    print(f"HNN qdot relative error = {rel:.3f}")
    assert rel < 0.5  # loose: 60 epochs is a smoke fit, not the Fig-7 training run
    rel_pdot = (torch.norm(pd - pdot_ref) / torch.norm(pdot_ref)).item()
    print(f"HNN pdot relative error = {rel_pdot:.3f}")
    assert rel_pdot < 0.5  # discriminates the double-M(theta) bug (1.078 before, 0.006 after)
