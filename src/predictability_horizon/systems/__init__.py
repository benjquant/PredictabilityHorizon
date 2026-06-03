"""System interface: a differentiable Warp step kernel + analytic helpers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy.typing as npt

Vec = npt.NDArray[Any]


@dataclass(frozen=True)
class System:
    name: str
    dim: int
    default_params: Vec  # float array consumed by the kernel
    step_kernel: object  # a @wp.kernel with the standard signature
    jacobian: Callable[[Vec, float, Vec, float], Vec]  # (state, u, params, dt) -> (dim,dim)
    energy: Callable[[Vec, Vec], float]  # (state, params) -> scalar
    suggested_dt: float


SYSTEMS: dict[str, System] = {}


def register(system: System) -> System:
    SYSTEMS[system.name] = system
    return system
