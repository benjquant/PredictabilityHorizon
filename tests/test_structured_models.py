import numpy as np
import pytest
import torch

from predictability_horizon.structured_models import HNN


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
    coordinate regression -- unlike the implicit-midpoint step's determinant, which is
    architecture-guaranteed (det J = 1 for any Hamiltonian) and reads the same whether or not
    the network has been trained at all."""
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

    ``hnn_spectrum_on_traj``'s exponent sum reads near 0 for an HNN because implicit midpoint is
    symplectic for ANY Hamiltonian: det J = 1 identically, at every (theta, p), on a randomly
    initialised network exactly as on a trained one. The sum is therefore an ARCHITECTURE
    receipt, not evidence about training, data or coordinates.

    Since the closed-form Cayley Jacobian landed, ``hnn_spectrum_on_traj`` reads its Jacobians off
    that formula rather than off autodiff, so this sum now certifies the Cayley ARITHMETIC -- det
    J = 1 is an algebraic identity for any Hamiltonian A. This test must never be cited in place
    of the executed-map evidence, which is ``test_hnn_is_near_volume_preserving``: that test
    stays on the autodiff path; see the note in ``structured_models.model_spectrum_sum``.

    Measured (120 epochs, seed 0) over the test's 1.8 s window (3600 steps at dt=5e-4, QR seed 0):
    sum=-2.346e-05, largest=1.4754. ``largest`` is unchanged to four decimals from the autodiff
    measurement this replaces (same window, same dt, same QR seed), which is the receipt that
    routing through the closed form moved no lambda_1; only the sum moved, -7.3e-06 -> -2.346e-05,
    both being float32 round-off against an identity. The ``abs(sum) < 1e-4`` bound is ~4x the new
    magnitude, tightened from 0.2 now that the step is exactly symplectic rather than to O(dt^2).
    ``largest > 0.3`` is somewhat discriminating on its own (an untrained network reads ~0.0036
    and would fail it) but is loose: 0.31 and 5.0 both pass it against a true lambda_1 of 1.554
    on this same 1.8 s window (dt=5e-4, QR seed 0). Note: this tripwire's short-window, single-QR
    protocol is a fast test gating and must not be compared to the 54 s, 8-QR-seed averaging
    protocol the paper reports. The pdot relative-error assertion below is what actually ties this
    test to the trained model: measured untrained rel err ~1.00 (fails the < 0.5 bound below) vs.
    trained (120 epochs) 0.0024 -- a ~400x margin.
    """
    from predictability_horizon.structured_models import hnn_spectrum_on_traj, train_hnn
    from predictability_horizon.systems import SYSTEMS, acrobot  # noqa: F401
    from predictability_horizon.warpsim import rollout
    from predictability_horizon.worldmodel import make_dataset

    s = SYSTEMS["acrobot"]
    ds = make_dataset(s, n_traj=80, T=2000, seed=0)
    hnn = train_hnn(ds, s.suggested_dt, epochs=120, seed=0)
    traj = rollout(
        s.step_kernel,
        np.array([2.5, 0, 0, 0.0]),
        np.zeros(4000),
        s.default_params,
        s.suggested_dt,
        4000,
    )[400:]
    spec = hnn_spectrum_on_traj(hnn, traj, dt=s.suggested_dt, k=4)
    print(f"HNN spectrum sum={float(np.sum(spec.exponents)):.3e} largest={spec.largest:.4f}")
    # architecture-only (see docstring): passes on an untrained network by construction.
    assert abs(float(np.sum(spec.exponents))) < 1e-4  # volume-preserving (canonical, exact)
    assert spec.largest > 0.3  # a sensible positive exponent, not garbage

    rel_pdot = _pdot_rel_err(hnn, ds, s.suggested_dt)
    print(f"HNN pdot relative error = {rel_pdot:.4f}")
    # discriminates training: untrained ~1.00 (fails this bound), trained 0.0024 (~400x margin)
    assert rel_pdot < 0.5


@pytest.mark.integration
def test_hnn_is_near_volume_preserving():
    """The executed-map tripwire for symplectic structure; spectrum sum and pdot both move under training.

    ``model_spectrum_sum`` reading near 0 is an ARCHITECTURE property of the HNN: implicit
    midpoint is symplectic for ANY Hamiltonian, so det J = 1 identically, at every (theta, p),
    on a randomly initialised network exactly as on a trained one. The sum is therefore an
    architecture receipt, not evidence about training, data or coordinates -- see the caveat in
    ``structured_models.model_spectrum_sum``. BUT this test's spectrum check stays on the autodiff
    path (differentiating the executed map), so it IS the only tripwire that fires if the symplectic
    structure is removed outright, which the pdot assertion below would not catch, since a plain MLP
    fits pdot perfectly well. See the note in ``structured_models.model_spectrum_sum``.

    Measured (200 epochs, seed 0): spectrum_sum=-1.007e-05, against -0.009869 under the retired
    symplectic-Euler step -- a ~1000x drop, and the reason ``abs(ss) < 1e-4`` (~10x the measured
    magnitude) replaces the old bound of 0.3. The pdot relative error is what actually moves
    under training or a coordinate regression: measured untrained ~1.00 (fails the < 0.5 bound
    below) vs. trained 0.0026 -- a ~190x margin.
    """
    from predictability_horizon.structured_models import model_spectrum_sum, train_hnn
    from predictability_horizon.systems import SYSTEMS, acrobot  # noqa: F401
    from predictability_horizon.worldmodel import make_dataset

    s = SYSTEMS["acrobot"]
    ds = make_dataset(s, n_traj=80, T=2000, seed=0)
    hnn = train_hnn(ds, s.suggested_dt, epochs=200, seed=0)
    ss = model_spectrum_sum(hnn, s, np.array([2.5, 0.0, 0.0, 0.0]))
    print(f"HNN spectrum_sum={ss:.3e}")
    # architecture-only (see docstring): passes on an untrained network by construction.
    assert abs(ss) < 1e-4

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
    hnn = train_hnn(ds, s.suggested_dt, epochs=60, seed=0)

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


def test_hnn_cayley_jacobian_matches_autodiff():
    """The closed form must be the Jacobian of the map that actually ran.

    Autodiff differentiates the executed Picard passes; the Cayley formula describes the exact
    implicit-midpoint map. An under-converged fixed point is what would separate them, and
    nothing else detects it. Direct analogue of
    test_systems.py::test_acrobot_analytic_jacobian_matches_autodiff.

    Measured (torch.manual_seed(0), untrained HNN, this 3-state fixture): max abs diff =
    1.192e-07 at BOTH dt=5e-4 and dt=5e-3 -- a single float32 ulp on entries of this magnitude,
    matching the acrobot ground truth's measured range of 3e-11..1.5e-7 (systems/acrobot.py
    module docstring). The tolerances below are ~10x that measurement.
    """
    from predictability_horizon.structured_models import _model_jac_fn, hnn_cayley_jacobians

    torch.manual_seed(0)
    states = np.array(
        [[2.5, 0.0, 0.0, 0.0], [1.2, -2.0, 4.0, -3.0], [-0.7, 2.2, -9.0, 7.0]]
    )
    for dt, tol in ((5e-4, 1.5e-6), (5e-3, 1.5e-6)):
        hnn = HNN(dt)
        hnn.eval()
        ja = hnn_cayley_jacobians(hnn, torch.tensor(states, dtype=torch.float32))
        ja_np = ja.detach().numpy().astype(np.float64)
        jad = np.stack([_model_jac_fn(hnn)(s) for s in states])
        err = np.abs(ja_np - jad).max()
        print(f"[cayley vs autodiff dt={dt:g}] max abs diff = {err:.3e}")
        assert err < tol


def test_hnn_cayley_transform_is_algebraically_symplectic():
    """det J = 1 for the closed form.

    This holds identically for any Hamiltonian A = J4 @ S with symmetric S, so it certifies the
    Cayley ARITHMETIC, not the model. The physics evidence is
    test_hnn_is_near_volume_preserving, which reads the determinant off the autodiff Jacobian of
    the executed map -- this test must never be cited in its place.

    Measured (torch.manual_seed(0), untrained HNN, this 3-state fixture): max |det J - 1| = 0.0
    at dt=5e-4 and 1.192e-07 at dt=5e-3 -- a single float32 ulp, matching the acrobot ground
    truth's measured range of 3e-11..1.5e-7. The bound below is ~10x the larger measurement.
    """
    from predictability_horizon.structured_models import hnn_cayley_jacobians

    torch.manual_seed(0)
    states = torch.tensor(
        [[2.5, 0.0, 0.0, 0.0], [1.2, -2.0, 4.0, -3.0], [-0.7, 2.2, -9.0, 7.0]],
        dtype=torch.float32,
    )
    for dt in (5e-4, 5e-3):
        hnn = HNN(dt)
        hnn.eval()
        det = torch.linalg.det(hnn_cayley_jacobians(hnn, states))
        err = (det - 1.0).abs().max().item()
        print(f"[cayley det dt={dt:g}] max |det J - 1| = {err:.3e}")
        assert err < 1.5e-6


def test_hnn_step_solves_the_implicit_midpoint_equation():
    """The HNN's one-step map must satisfy implicit midpoint's defining identity.

    (z1 - z0)/dt == J grad H_phi(zbar) with zbar = (z0 + z1)/2, to float64 round-off. This is
    the same scheme the ground-truth kernel uses (systems/acrobot.py), so the learned model and
    the simulator are advanced by the same map -- which is what makes the Part B comparison
    apples-to-apples at the level of the integrator, not just the coordinates.

    Run in FLOAT64. In float32 this test cannot discriminate at all: (z1 - z0) is a difference
    of same-signed numbers of magnitude up to 9, whose rounding error divided by dt floors the
    residual near 1e-3 -- above the O(dt) signal being measured. Measured float32 residuals are
    9.9e-4 (midpoint) against 9.2e-4 (legacy), i.e. no separation whatever. Same reason
    test_systems.py::test_finite_difference_equals_the_vector_field_at_the_midpoint uses the
    float64 reference step rather than the Warp kernel.

    Discriminating, and permanently so: the retired explicit symplectic-Euler step
    (HNN.legacy_euler_step) evaluates the vector field at the endpoints, which differs from the
    midpoint value at O(dt). Measured in float64 -- implicit midpoint 2.8e-12 (dt=5e-4) and
    1.1e-13 (dt=5e-3), against legacy Euler 1.9e-7 and 1.9e-6: factors of 6.7e4 and 1.6e7. The
    legacy residual scales exactly 10x with a 10x dt, which is the O(dt) signature this bound
    is placed against.
    """

    def _identity_residual(hnn: HNN, step, z0: torch.Tensor, dt: float) -> float:
        z1 = step(z0)
        zbar = 0.5 * (z0 + z1)
        q = zbar[..., :2].detach().requires_grad_(True)
        p = zbar[..., 2:].detach().requires_grad_(True)
        qd, pd = hnn.vector_field(q, p)
        lhs = ((z1 - z0) / dt).detach()
        rhs = torch.cat([qd, pd], dim=-1).detach()
        return float((lhs - rhs).abs().max().item())

    states = torch.tensor(
        [[2.5, 0.0, 0.0, 0.0], [1.2, -2.0, 4.0, -3.0], [-0.7, 2.2, -9.0, 7.0]],
        dtype=torch.float64,
    )
    for dt in (5e-4, 5e-3):
        torch.manual_seed(0)
        hnn = HNN(dt).double()
        z0 = states.clone().requires_grad_(True)
        err = _identity_residual(hnn, hnn.forward, z0, dt)
        err_legacy = _identity_residual(hnn, hnn.legacy_euler_step, z0, dt)
        print(
            f"[midpoint identity dt={dt:g}] midpoint={err:.3e}  legacy={err_legacy:.3e}  "
            f"ratio={err_legacy / err:.1e}"
        )
        assert err < 1e-9
        # The retired step is what this bound excludes -- assert it, rather than
        # measuring it once by hand and trusting a docstring to stay true.
        assert err_legacy > 100.0 * err
