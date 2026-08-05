"""Phase 2: structure-preserving world models for the acrobot (canonical coordinates)."""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import numpy as np
import numpy.typing as npt
import torch
import warp as wp
from torch import nn

from predictability_horizon.core import LyapunovResult
from predictability_horizon.lyapunov import lyapunov_spectrum, lyapunov_spectrum_from_jacobians
from predictability_horizon.systems import System
from predictability_horizon.warpsim import rollout
from predictability_horizon.worldmodel import MLP, Dataset

_N_FP_ITER = 6
"""Picard passes per implicit-midpoint step, matching systems/acrobot.py's _N_FP_ITER.

The same count for the same reason: the contraction rate is (dt/2)*L, so the binding
constraint is dt, and 6 saturates at both the production step size and the coarser one used in
tests. FIXED rather than residual-triggered -- a state-dependent trip count would make the map
piecewise-defined and its Jacobian discontinuous across the switching surfaces, and this module
measures Jacobian-derived observables (lambda_1, det J).
"""

_J4 = torch.tensor(
    [[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0], [-1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0]]
)
"""Canonical structure matrix: _J4 @ grad H = (dH/dp, -dH/dq). Mirrors systems/acrobot.py."""


def _logdet_jac(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """log|det ∂model/∂x| for each row of a batch x (small dim -> full Jacobian)."""
    out = []
    for xi in x:
        jac = torch.autograd.functional.jacobian(
            lambda z: model(z.unsqueeze(0)).squeeze(0), xi, create_graph=True
        )
        # nan_to_num guards a (rare) singular Jacobian: slogdet -> -inf would NaN the penalty.
        out.append(torch.nan_to_num(torch.linalg.slogdet(jac)[1], neginf=-50.0, posinf=50.0))
    return torch.stack(out)


def train_volume_penalty_mlp(
    ds: Dataset,
    epochs: int = 200,
    lr: float = 1e-3,
    batch_size: int = 256,
    penalty: float = 1.0,
    seed: int = 0,
) -> MLP:
    """Residual MLP + a soft penalty (log|det J_step|)^2 pushing the one-step map toward
    volume preservation (det J -> 1 ⇒ Lyapunov spectrum sums to 0)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = MLP(ds.x.shape[1])
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    x_all = torch.tensor(ds.x, dtype=torch.float32)
    y_all = torch.tensor(ds.y, dtype=torch.float32)
    n = x_all.shape[0]
    rng = np.random.default_rng(seed)
    for _ in range(epochs):
        perm = rng.permutation(n)
        for start in range(0, n, batch_size):
            idx = torch.from_numpy(perm[start : start + batch_size])
            opt.zero_grad()
            mse = nn.functional.mse_loss(model(x_all[idx]), y_all[idx])
            sub = x_all[idx][: min(16, idx.shape[0])]  # det-J penalty on a small sub-batch
            vol = (_logdet_jac(model, sub) ** 2).mean()
            (mse + penalty * vol).backward()
            opt.step()
    return model


def _model_jac_fn(
    model: nn.Module,
) -> Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]:
    """Returns jac_fn(state)->(dim,dim) using the model's torch autograd Jacobian."""

    def jac_fn(s: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        st = torch.tensor(s, dtype=torch.float32)
        jac = torch.autograd.functional.jacobian(lambda z: model(z.unsqueeze(0)).squeeze(0), st)
        return jac.detach().numpy().astype(np.float64)

    return jac_fn


def _hnn_midpoint(hnn: HNN, states: torch.Tensor) -> torch.Tensor:
    """(N,4) states -> (N,4) converged implicit-midpoint points zbar, detached.

    The same _N_FP_ITER Picard passes HNN.forward runs, with the graph thrown away: the Cayley
    transform needs only the LOCATION of zbar, because differentiating
    z' = z + dt J grad H(zbar) with zbar = (z + z')/2 already accounts analytically for zbar's
    dependence on z.
    """
    zb = states
    for _ in range(_N_FP_ITER):
        q = zb[..., :2].detach().requires_grad_(True)
        p = zb[..., 2:].detach().requires_grad_(True)
        qd, pd = hnn.vector_field(q, p)
        zb = states + 0.5 * hnn.dt * torch.cat([qd, pd], dim=-1).detach()
    return zb.detach()


def hnn_cayley_jacobians(hnn: HNN, states: torch.Tensor) -> torch.Tensor:
    """(N,4) canonical states -> (N,4,4) analytic one-step Jacobians of the HNN's map.

    Differentiating z' = z + dt [J grad H_phi(zbar)] with zbar = (z + z')/2 gives a Cayley
    transform

        J_step = (I - (dt/2) A)^-1 (I + (dt/2) A),   A = J4 @ Hess H_phi(zbar),

    which is symplectic identically because A is Hamiltonian: det J = 1 to machine precision,
    not to O(dt^2). H_phi is the LEARNED network and zbar is the midpoint of the MODEL's own
    step -- the true Hamiltonian appears nowhere in this. Batched with torch.func so an orbit
    costs one call rather than one torch re-entry per state, and pinned to autodiff through the
    executed Picard passes by test_hnn_cayley_jacobian_matches_autodiff.
    """
    zbar = _hnn_midpoint(hnn, states)
    hess = torch.func.vmap(torch.func.hessian(hnn.hamiltonian))(zbar)
    a = _J4.to(hess.dtype) @ hess
    eye = torch.eye(4, dtype=hess.dtype)
    half = 0.5 * hnn.dt
    return torch.linalg.solve(eye - half * a, eye + half * a)


class HNN(nn.Module):
    """Hamiltonian NN as a canonical (θ,p) one-step map via implicit midpoint.

    Learns H_φ(q,p) directly on the canonical coords the simulator emits (q=θ, p already
    the canonical momentum — no ω->p conversion happens here or anywhere upstream); the
    vector field q̇=∂H/∂p, ṗ=-∂H/∂q is Hamiltonian, and one implicit-midpoint step gives a
    (θ,p)->(θ',p') map, symplectic exactly rather than to O(dt²). Angles enter H through a
    (cosθ,sinθ) embedding to respect the S¹ topology.
    """

    def __init__(self, dt: float, hidden: int = 128) -> None:
        super().__init__()
        self.dt = float(dt)
        self.dim = 4
        self.net = nn.Sequential(
            nn.Linear(6, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def _hamiltonian(self, q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        emb = torch.cat(
            [
                torch.cos(q[..., :1]),
                torch.sin(q[..., :1]),
                torch.cos(q[..., 1:2]),
                torch.sin(q[..., 1:2]),
                p,
            ],
            dim=-1,
        )
        return self.net(emb).squeeze(-1)

    def hamiltonian(self, z: torch.Tensor) -> torch.Tensor:
        """H_phi at ONE canonical state z = (q1, q2, p1, p2) -> scalar.

        Single-state and flat so torch.func.hessian composes with vmap over a whole orbit;
        _hamiltonian keeps the batched (q, p) signature the training loop uses.
        """
        return self._hamiltonian(z[:2], z[2:])

    def vector_field(self, q: torch.Tensor, p: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # q, p must already require grad (see the autograd design note). Returns (q̇, ṗ).
        h = self._hamiltonian(q, p).sum()
        dh_dq, dh_dp = torch.autograd.grad(h, (q, p), create_graph=True)
        return dh_dp, -dh_dq

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Interprets input as canonical (q, p) and advances it with IMPLICIT MIDPOINT, the same
        # scheme systems/acrobot.py uses:  zbar = z + (dt/2) J grad H(zbar),  z' = 2 zbar - z.
        # Symplectic for any Hamiltonian, second-order and time-reversible -- so det J = 1
        # exactly, not to O(dt^2). H_phi is non-separable (q and p are jointly embedded), which
        # is precisely why the retired explicit symplectic-Euler step was only approximately
        # volume-preserving here. Solved by _N_FP_ITER Picard passes, a fixed count so the map
        # stays straight-line and differentiable end to end.
        q, p = x[..., :2], x[..., 2:]
        qb, pb = q, p
        for _ in range(_N_FP_ITER):
            qd, pd = self.vector_field(qb, pb)
            qb = q + 0.5 * self.dt * qd
            pb = p + 0.5 * self.dt * pd
        return torch.cat([2.0 * qb - q, 2.0 * pb - p], dim=-1)

    def legacy_euler_step(self, x: torch.Tensor) -> torch.Tensor:
        """RETIRED. The pre-spec-3 explicit symplectic-Euler step. NOT used in production.

        p' = p + dt * pdot(q, p);  q' = q + dt * qdot(q, p'). Symplectic only to O(dt^2) on a
        non-separable H -- and H_phi IS non-separable, because q and p are jointly embedded --
        which is the same defect spec 2 removed from the ground truth. Kept, like
        systems/acrobot.py's legacy_acrobot_step, so the scheme stays executable: corners B and
        D of the Part-B ablation measure one learned Hamiltonian under this step and under
        `forward`, and the difference is what the model's own integrator was costing.
        """
        q, p = x[..., :2], x[..., 2:]
        _, pd = self.vector_field(q, p)
        p_new = p + self.dt * pd
        qd2, _ = self.vector_field(q, p_new)
        return torch.cat([q + self.dt * qd2, p_new], dim=-1)


def train_hnn(
    ds: Dataset,
    dt: float,
    epochs: int = 200,
    lr: float = 1e-3,
    batch_size: int = 256,
    seed: int = 0,
) -> HNN:
    """Train H_phi by matching the Hamiltonian vector field to finite-difference derivatives.

    The dataset is already canonical (theta, p) -- the simulator emits conjugate momenta --
    so no omega -> p conversion happens here. Applying M(theta) again would inflate the
    momenta and fit H_phi to data the simulator never produced.

    The field is matched at the midpoint (z0 + z1)/2 rather than at z0, because that is where
    implicit midpoint's defining identity places it. Applied uniformly to every corner of the
    spec-3 grid, including the retired-kernel ones -- for data no Hamiltonian generated no
    target is strictly correct, and holding the procedure fixed is what makes the comparison
    mean anything.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = HNN(dt)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    x_t = torch.tensor(ds.x, dtype=torch.float32)
    y_t = torch.tensor(ds.y, dtype=torch.float32)
    q0, p0 = x_t[:, :2], x_t[:, 2:]
    q1, p1 = y_t[:, :2], y_t[:, 2:]
    qdot = (q1 - q0) / dt
    pdot = (p1 - p0) / dt
    # Evaluate the field at the MIDPOINT, not the endpoint: implicit midpoint is defined by
    # (z1 - z0)/dt = J grad H((z0 + z1)/2), so this is the exact relation the data satisfies.
    # Fitting at z0 instead is a systematic O(dt) bias in the learned grad H_phi. Certified by
    # test_systems.py::test_finite_difference_equals_the_vector_field_at_the_midpoint.
    qm, pm = 0.5 * (q0 + q1), 0.5 * (p0 + p1)
    n = x_t.shape[0]
    rng = np.random.default_rng(seed)
    for _ in range(epochs):
        perm = rng.permutation(n)
        for start in range(0, n, batch_size):
            idx = torch.from_numpy(perm[start : start + batch_size])
            opt.zero_grad()
            q = qm[idx].detach().requires_grad_(True)
            p = pm[idx].detach().requires_grad_(True)
            qd, pd = model.vector_field(q, p)
            loss = nn.functional.mse_loss(qd, qdot[idx]) + nn.functional.mse_loss(pd, pdot[idx])
            loss.backward()
            opt.step()
    return model


def hnn_spectrum_on_traj(
    hnn: HNN, true_traj: npt.NDArray[np.float64], dt: float, k: int = 4
) -> LyapunovResult:
    """HNN Lyapunov spectrum along a true canonical trajectory.

    The HNN is a canonical (q, p) = (theta, p) map and the simulator emits exactly those
    coordinates, so the model's Jacobian is evaluated directly on the true orbit -- no change of
    frame. This is the same apples-to-apples "model Jacobian along the true orbit" convention
    used for the MLP (model_lyapunov_on_traj): the model's own derivative, taken at the states
    the real system visits rather than wherever the model's own drift ended up.

    Uses the closed-form Cayley Jacobian, computed for the whole orbit in one batched call.
    """
    hnn.eval()
    traj = np.asarray(true_traj, dtype=np.float64)
    z = torch.tensor(traj[:-1], dtype=torch.float32)
    jacs = hnn_cayley_jacobians(hnn, z).detach().numpy().astype(np.float64)
    return lyapunov_spectrum_from_jacobians(jacs, dt=dt, k=k)


def model_spectrum_sum(
    model: nn.Module, sys: System, x0: npt.NDArray[np.float64], t_steps: int = 4000
) -> float:
    """Sum of the model's Lyapunov spectrum along the TRUE trajectory (≈0 ⇔ volume-preserving).

    Uses the model's autograd Jacobian evaluated at TRUE acrobot states (not the model's own
    rollout) — avoids model-drift artifacts, matching the Part-B audit's apples-to-apples
    convention. All models now live in canonical (θ,p), so this is directly comparable across
    the plain MLP, the volume-penalty MLP and the HNN. For a meaningful λ₁ (not just the
    volume sum) on the HNN, use ``hnn_spectrum_on_traj``.

    Deliberately uses the AUTODIFF Jacobian (_model_jac_fn) for every model, the HNN included,
    rather than the HNN's closed-form Cayley Jacobian. Autodiff differentiates the code that
    actually ran, which is what makes a near-zero sum evidence about the executed map; the
    Cayley determinant is an algebraic identity for any Hamiltonian A and would certify the
    arithmetic instead. Same distinction as test_acrobot_is_symplectic_under_autodiff versus
    test_cayley_transform_is_algebraically_symplectic. Note the sum is still
    architecture-guaranteed and reads the same on an untrained network as on a trained one.
    """
    model.eval()
    true_traj = rollout(
        cast(wp.Kernel, sys.step_kernel),
        x0,
        np.zeros(t_steps),
        sys.default_params,
        sys.suggested_dt,
        t_steps,
    )
    spec = lyapunov_spectrum(
        _model_jac_fn(model), true_traj[t_steps // 10 :], dt=sys.suggested_dt, k=sys.dim
    )
    return float(np.sum(spec.exponents))
