"""Phase 2: structure-preserving world models for the acrobot (canonical coordinates,
mass matrix + transforms)."""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import numpy as np
import numpy.typing as npt
import torch
import warp as wp
from torch import nn

from predictability_horizon.core import LyapunovResult
from predictability_horizon.lyapunov import lyapunov_spectrum
from predictability_horizon.systems import System
from predictability_horizon.warpsim import rollout
from predictability_horizon.worldmodel import MLP, Dataset


def acrobot_mass_matrix(theta: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
    """(...,2) angles -> (...,2,2) mass matrix M(θ); p = M(θ) ω. Matches systems/acrobot KE."""
    m1, m2, l1, l2, _g = params
    c = torch.cos(theta[..., 0] - theta[..., 1])
    M = torch.zeros(*theta.shape[:-1], 2, 2, dtype=theta.dtype)  # noqa: N806
    M[..., 0, 0] = (m1 + m2) * l1 * l1
    M[..., 0, 1] = m2 * l1 * l2 * c
    M[..., 1, 0] = m2 * l1 * l2 * c
    M[..., 1, 1] = m2 * l2 * l2
    return M


def omega_to_p(theta: torch.Tensor, omega: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
    """Velocities ω -> conjugate momenta p = M(θ) ω."""
    return torch.einsum("...ij,...j->...i", acrobot_mass_matrix(theta, params), omega)


def p_to_omega(theta: torch.Tensor, p: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
    """Conjugate momenta p -> velocities ω = M(θ)⁻¹ p."""
    M = acrobot_mass_matrix(theta, params)  # noqa: N806
    return torch.linalg.solve(M, p.unsqueeze(-1)).squeeze(-1)


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


class HNN(nn.Module):
    """Hamiltonian NN as a canonical (θ,p) one-step map via symplectic Euler.

    Learns H_φ(q,p) directly on the canonical coords the simulator emits (q=θ, p already
    the canonical momentum — no ω->p conversion happens here or anywhere upstream); the
    vector field q̇=∂H/∂p, ṗ=-∂H/∂q is Hamiltonian, and one symplectic-Euler step gives a
    (θ,p)->(θ',p') map. Angles enter H through a (cosθ,sinθ) embedding to respect the S¹
    topology.
    """

    def __init__(self, params: torch.Tensor, dt: float, hidden: int = 128) -> None:
        super().__init__()
        self.register_buffer("params", params)
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

    def vector_field(self, q: torch.Tensor, p: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # q, p must already require grad (see the autograd design note). Returns (q̇, ṗ).
        h = self._hamiltonian(q, p).sum()
        dh_dq, dh_dp = torch.autograd.grad(h, (q, p), create_graph=True)
        return dh_dp, -dh_dq

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Interprets input as canonical (q, p); symplectic Euler in (q,p) space. H_φ is
        # non-separable (q and p are jointly embedded), so this explicit step is symplectic
        # only to O(dt²): det J = 1 + O(dt²) per step (measured spectrum sum ≈0.001) — i.e.
        # approximately, not exactly, volume-preserving. Training data is already canonical
        # (θ,p).
        q, p = x[..., :2], x[..., 2:]
        _, pd = self.vector_field(q, p)
        p_new = p + self.dt * pd  # symplectic Euler: momentum first
        qd2, _ = self.vector_field(q, p_new)
        q_new = q + self.dt * qd2
        return torch.cat([q_new, p_new], dim=-1)


def train_hnn(
    ds: Dataset,
    params: npt.NDArray[np.float64],
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
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    params_t = torch.as_tensor(np.asarray(params), dtype=torch.float32)
    model = HNN(params_t, dt)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    x_t = torch.tensor(ds.x, dtype=torch.float32)
    y_t = torch.tensor(ds.y, dtype=torch.float32)
    q0, p0 = x_t[:, :2], x_t[:, 2:]
    q1, p1 = y_t[:, :2], y_t[:, 2:]
    qdot = (q1 - q0) / dt
    pdot = (p1 - p0) / dt
    n = x_t.shape[0]
    rng = np.random.default_rng(seed)
    for _ in range(epochs):
        perm = rng.permutation(n)
        for start in range(0, n, batch_size):
            idx = torch.from_numpy(perm[start : start + batch_size])
            opt.zero_grad()
            q = q0[idx].detach().requires_grad_(True)
            p = p0[idx].detach().requires_grad_(True)
            qd, pd = model.vector_field(q, p)
            loss = nn.functional.mse_loss(qd, qdot[idx]) + nn.functional.mse_loss(pd, pdot[idx])
            loss.backward()
            opt.step()
    return model


def hnn_spectrum_on_traj(
    hnn: HNN, true_traj: npt.NDArray[np.float64], dt: float, k: int = 4
) -> LyapunovResult:
    """HNN Lyapunov spectrum along a true canonical trajectory.

    The HNN is a canonical (q, p) = (theta, p) map and the simulator now emits exactly those
    coordinates, so the model's Jacobian is evaluated directly on the true orbit -- no change
    of frame. This is the same apples-to-apples "model Jacobian along the true orbit"
    convention used for the MLP (model_lyapunov_on_traj).
    """
    hnn.eval()
    return lyapunov_spectrum(
        _model_jac_fn(hnn), np.asarray(true_traj, dtype=np.float64), dt=dt, k=k
    )


def model_spectrum_sum(
    model: nn.Module, sys: System, x0: npt.NDArray[np.float64], t_steps: int = 4000
) -> float:
    """Sum of the model's Lyapunov spectrum along the TRUE trajectory (≈0 ⇔ volume-preserving).

    Uses the model's autograd Jacobian evaluated at TRUE acrobot states (not the model's own
    rollout) — avoids model-drift artifacts, matching the Part-B audit's apples-to-apples
    convention. All models now live in canonical (θ,p), so this is directly comparable across
    the plain MLP, the volume-penalty MLP and the HNN. For a meaningful λ₁ (not just the
    volume sum) on the HNN, use ``hnn_spectrum_on_traj``; the volume *sum* still reads ≈0 here
    for an HNN only because its symplectic-Euler map has det J ≈ 1 (to O(dt²)) ANYWHERE in
    (θ,p) space, on an untrained network exactly as on a trained one -- a near-zero sum is
    architecture-guaranteed, not evidence about training or fit quality.
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
