"""Spec 3: the Part-B ablation.

How much of the HNN's lambda_1 shortfall is the model's OWN integrator, how much is the
quality of the training simulator, and how much is genuine model error. Two ingredients vary
independently -- which simulator produced the training data, and which integrator steps the
model when its Jacobian is taken -- giving four corners:

    A  corrected data, implicit midpoint   -- the paper's number
    B  corrected data, symplectic Euler    -- B - A is the model's own integrator
    C  retired data,   implicit midpoint   -- C - A is training-data quality
    D  retired data,   symplectic Euler    -- additivity check on the other two

B and D reuse the weights of A and C, so the grid costs two trainings per seed. Experiment
code only: nothing here is imported by the models or the figures.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

import numpy as np
import numpy.typing as npt
import torch
import warp as wp

from predictability_horizon.lyapunov import lyapunov_spectrum_from_jacobians
from predictability_horizon.structured_models import HNN, hnn_cayley_jacobians, train_hnn
from predictability_horizon.systems import System
from predictability_horizon.systems.acrobot import (
    legacy_acrobot_step,
    to_canonical,
    to_legacy,
)
from predictability_horizon.warpsim import rollout
from predictability_horizon.worldmodel import Dataset, make_dataset


def build_legacy_dataset(sys: System, n_traj: int, T: int, seed: int = 0) -> Dataset:  # noqa: N803
    """Training pairs from the RETIRED kernel, in canonical coordinates, IC-matched to
    ``worldmodel.make_dataset``.

    Draws the same U(-2.5, 2.5)^4 canonical initial conditions make_dataset draws, from the
    same RNG and seed, then pushes each to (theta, omega) with to_legacy, rolls it through the
    unregistered legacy kernel, and maps the trajectory back with to_canonical. Both datasets
    therefore start from identical energies and differ only in whether the generating dynamics
    conserved -- which is the entire point of corners C and D.

    The retired kernel stays UNREGISTERED: this calls rollout on it directly. Registering a
    second acrobot would undo spec 2.
    """
    rng = np.random.default_rng(seed)
    pars = np.asarray(sys.default_params)
    xs, ys = [], []
    for _ in range(n_traj):
        x0 = rng.uniform(-2.5, 2.5, size=sys.dim)  # canonical, exactly as make_dataset
        legacy_traj = rollout(
            cast(wp.Kernel, legacy_acrobot_step),
            to_legacy(x0, pars),
            np.zeros(T),
            sys.default_params,
            sys.suggested_dt,
            T,
        )
        traj = np.stack([to_canonical(s, pars) for s in legacy_traj])
        xs.append(traj[:-1])
        ys.append(traj[1:])
    return Dataset(x=np.concatenate(xs), y=np.concatenate(ys))


@dataclass(frozen=True)
class Run:
    """One trained model, measured once.

    label: "A".."D" for the grid corners, "plain" for the unconstrained MLP, or
    "penalty:mu=<value>" for one soft-penalty setting. seed is the TRAINING seed; the dataset
    seed is 0 for every run in the study.
    """

    label: str
    seed: int
    lambda1: float
    spectrum_sum: float
    qr_spread: float


@dataclass(frozen=True)
class AblationResult:
    """Every measurement in the study, plus the protocol every number must be quoted with."""

    runs: list[Run]
    true_lambda1: float
    true_qr_spread: float
    dt: float
    n_steps: int
    transient_steps: int
    qr_seeds: int

    @property
    def window_s(self) -> float:
        """Averaging window in seconds. No lambda_1 from this study is quotable without it."""
        return (self.n_steps - self.transient_steps) * self.dt

    def values(self, label: str) -> list[float]:
        return sorted(r.lambda1 for r in self.runs if r.label == label)

    def median(self, label: str) -> float:
        return float(np.median(self.values(label)))

    def spread(self, label: str) -> float:
        """Full min-max range over seeds -- the uncertainty this study reports."""
        v = self.values(label)
        return float(v[-1] - v[0])


def dump(result: AblationResult, path: Path) -> None:
    """Write the whole study to JSON so the ledger and the figure need no retraining."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "runs": [asdict(r) for r in result.runs],
        "true_lambda1": result.true_lambda1,
        "true_qr_spread": result.true_qr_spread,
        "dt": result.dt,
        "n_steps": result.n_steps,
        "transient_steps": result.transient_steps,
        "qr_seeds": result.qr_seeds,
        "window_s": result.window_s,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _symplectic_euler_jacobians(hnn: HNN, states: torch.Tensor) -> torch.Tensor:
    """(N,4) -> (N,4,4) Jacobians of the RETIRED explicit symplectic-Euler step on H_phi.

    Corners B and D. The step itself is HNN.legacy_euler_step, kept in structured_models.py
    beside the scheme that replaced it -- the same convention systems/acrobot.py uses for
    legacy_acrobot_step. This function only differentiates it.

    The comparison this enables is two integrators applied to ONE learned Hamiltonian, both in
    torch. It has nothing to do with the retired Warp kernel, which this module uses only to
    build corner C's dataset.
    """
    return torch.stack(
        [torch.autograd.functional.jacobian(hnn.legacy_euler_step, zi) for zi in states]
    )


def true_orbit(
    sys: System, x0: npt.NDArray[np.float64], n_steps: int, transient_steps: int
) -> npt.NDArray[np.float64]:
    """The evaluation itinerary: one true trajectory, transient dropped.

    Held FIXED across all four corners and both MLP arms, so only the model varies. This
    preserves the apples-to-apples convention -- the model's own Jacobian, taken at the states
    the real system visits rather than wherever the model's own drift ended up.
    """
    traj = rollout(
        cast(wp.Kernel, sys.step_kernel),
        x0,
        np.zeros(n_steps),
        sys.default_params,
        sys.suggested_dt,
        n_steps,
    )
    return traj[transient_steps:]


_QR_SEEDS = 8
"""QR-frame seeds averaged over for every reported lambda_1.

lyapunov_spectrum draws its initial tangent frame from this seed, and it had never been varied.
At the retired 5.4 s window the TRUE lambda_1 spanned 0.260..1.242 across 8 seeds -- a factor of
4.8 with no model involved. The 54 s window suppresses that to ~0.034; averaging over seeds and
reporting the spread makes what remains visible rather than a matter of luck.

Nearly free: the Jacobians are computed ONCE and the QR sweep re-runs a cheap numpy loop over the
same stack. Never recompute Jacobians per seed.
"""


def _measure_from_jacobians(
    jacs: npt.NDArray[np.float64], dt: float
) -> tuple[float, float, float]:
    """(mean lambda_1, mean spectrum sum, min-max lambda_1 spread) over _QR_SEEDS QR frames."""
    specs = [
        lyapunov_spectrum_from_jacobians(jacs, dt=dt, k=4, seed=sd) for sd in range(_QR_SEEDS)
    ]
    lams = [float(s.largest) for s in specs]
    sums = [float(np.sum(s.exponents)) for s in specs]
    return float(np.mean(lams)), float(np.mean(sums)), float(max(lams) - min(lams))


_GRID = ((False, "A", "B"), (True, "C", "D"))
"""(use_retired_data, midpoint_label, symplectic_euler_label) for the four corners."""


def run_grid(
    sys: System,
    seeds: Sequence[int],
    true_traj: npt.NDArray[np.float64],
    n_traj: int = 80,
    t_data: int = 2000,
    epochs: int = 200,
) -> list[Run]:
    """Train the HNN on both datasets at each seed and measure each under both integrators.

    Seeds vary TRAINING only: both datasets are built once, at dataset seed 0, and shared by
    every run. Data-sampling variance is therefore not measured -- state that beside the
    numbers rather than letting a reader assume otherwise.
    """
    dt = sys.suggested_dt
    datasets = {
        False: make_dataset(sys, n_traj=n_traj, T=t_data, seed=0),
        True: build_legacy_dataset(sys, n_traj=n_traj, T=t_data, seed=0),
    }
    z = torch.tensor(np.asarray(true_traj)[:-1], dtype=torch.float32)
    runs: list[Run] = []
    for seed in seeds:
        for retired, mid_label, euler_label in _GRID:
            hnn = train_hnn(datasets[retired], dt, epochs=epochs, seed=seed)
            hnn.eval()
            for label, jac_fn in (
                (mid_label, hnn_cayley_jacobians),
                (euler_label, _symplectic_euler_jacobians),
            ):
                jacs = jac_fn(hnn, z).detach().numpy().astype(np.float64)
                lam, ssum, qr = _measure_from_jacobians(jacs, dt)
                runs.append(Run(label, seed, lam, ssum, qr))
                print(
                    f"[grid] {label} seed={seed} lambda1={lam:+.4f} (qr spread {qr:.4f}) "
                    f"sum={ssum:+.2e}"
                )
    return runs
