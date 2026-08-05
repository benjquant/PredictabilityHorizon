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

from typing import cast

import numpy as np
import warp as wp

from predictability_horizon.systems import System
from predictability_horizon.systems.acrobot import (
    legacy_acrobot_step,
    to_canonical,
    to_legacy,
)
from predictability_horizon.warpsim import rollout
from predictability_horizon.worldmodel import Dataset


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
