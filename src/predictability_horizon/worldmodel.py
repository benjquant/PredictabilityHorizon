"""Part B: a small MLP dynamics model + the error-growth-vs-lambda_1 diagnostic."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
import numpy.typing as npt
import torch
import warp as wp
from torch import nn

from predictability_horizon.core import fit_loglinear_slope
from predictability_horizon.lyapunov import lyapunov_spectrum
from predictability_horizon.systems import System
from predictability_horizon.warpsim import rollout


@dataclass
class Dataset:
    x: npt.NDArray[np.float64]  # (N, dim) state at t
    y: npt.NDArray[np.float64]  # (N, dim) state at t+1


def make_dataset(sys: System, n_traj: int, T: int, seed: int = 0) -> Dataset:  # noqa: N803
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    for _ in range(n_traj):
        x0 = rng.uniform(-2.5, 2.5, size=sys.dim)
        traj = rollout(
            cast(wp.Kernel, sys.step_kernel),
            x0,
            np.zeros(T),
            sys.default_params,
            sys.suggested_dt,
            T,
        )
        xs.append(traj[:-1])
        ys.append(traj[1:])
    return Dataset(x=np.concatenate(xs), y=np.concatenate(ys))


class MLP(nn.Module):
    def __init__(self, dim: int, hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, dim),
        )
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)  # residual: predict the increment


def train_world_model(
    ds: Dataset,
    epochs: int = 200,
    lr: float = 1e-3,
    batch_size: int = 256,
    seed: int = 0,
) -> MLP:
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
            loss = nn.functional.mse_loss(model(x_all[idx]), y_all[idx])
            loss.backward()
            opt.step()
    return model


def _model_rollout(model: MLP, x0: npt.NDArray[np.float64], T: int) -> npt.NDArray[np.float64]:  # noqa: N803
    model.eval()
    s = torch.tensor(x0, dtype=torch.float32).unsqueeze(0)
    out = [s.squeeze(0).detach().numpy()]
    with torch.no_grad():
        for _ in range(T):
            s = model(s)
            out.append(s.squeeze(0).numpy())
    return np.array(out, dtype=np.float64)


def error_growth_rate(
    model: MLP,
    sys: System,
    x0: npt.NDArray[np.float64],
    T: int,  # noqa: N803
    window: tuple[int, int] | None = None,
) -> float:
    gt = rollout(
        cast(wp.Kernel, sys.step_kernel), x0, np.zeros(T), sys.default_params, sys.suggested_dt, T
    )
    pred = _model_rollout(model, x0, T)
    err = np.linalg.norm(pred - gt, axis=1) + 1e-12
    if window is None:
        lo, hi = max(5, T // 20), T // 2  # early growth regime, pre-saturation
    else:
        lo, hi = window
    slope, _ = fit_loglinear_slope(np.arange(lo, hi, dtype=float), err[lo:hi])
    return slope / sys.suggested_dt  # per unit time


def model_lyapunov(
    model: MLP,
    sys: System,
    x0: npt.NDArray[np.float64],
    T: int,  # noqa: N803
    k: int | None = None,
) -> float:
    """Largest Lyapunov exponent (per unit time) of the LEARNED dynamics g_phi.

    Rolls the model out from x0, then runs the same Benettin/QR estimator used for the
    true systems, with the model's one-step Jacobian d g_phi/d x from torch autograd.
    A faithful model reproduces the true system's lambda_1.
    """
    k = sys.dim if k is None else k
    traj = _model_rollout(model, x0, T)
    model.eval()

    def jac_fn(s: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        st = torch.tensor(s, dtype=torch.float32)
        J = torch.autograd.functional.jacobian(  # noqa: N806
            lambda z: model(z.unsqueeze(0)).squeeze(0), st
        )
        return J.detach().numpy().astype(np.float64)

    # drop a short transient so the tangent vectors align before averaging
    start = max(1, T // 10)
    return lyapunov_spectrum(jac_fn, traj[start:], dt=sys.suggested_dt, k=k).largest


def model_lyapunov_on_traj(
    model: MLP,
    true_traj: npt.NDArray[np.float64],
    dt: float,
    k: int = 1,
) -> float:
    """Largest Lyapunov exponent of the MODEL evaluated along the TRUE trajectory.

    Uses the model's one-step Jacobian (torch autograd) at each state of true_traj
    instead of rolling the model out — avoids model-drift artifacts and makes the
    comparison with the true λ₁ apples-to-apples.

    Args:
        model: Trained MLP dynamics model.
        true_traj: (N+1, dim) true states (transient already dropped by the caller).
        dt: Integration time-step (s).
        k: Number of top exponents to keep; returns largest.
    """
    model.eval()

    def jac_fn(s: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        st = torch.tensor(s, dtype=torch.float32)
        J = torch.autograd.functional.jacobian(  # noqa: N806
            lambda z: model(z.unsqueeze(0)).squeeze(0), st
        )
        return J.detach().numpy().astype(np.float64)

    return lyapunov_spectrum(jac_fn, true_traj, dt=dt, k=k).largest
