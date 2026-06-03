"""Part A2: cartpole swing-up by gradient descent through the differentiable simulator."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import warp as wp

from predictability_horizon.systems import SYSTEMS, cartpole  # noqa: F401
from predictability_horizon.warpsim import _alloc_states, init_warp


@wp.kernel
def _upright_cost(
    states: wp.array2d(dtype=wp.float32),  # type: ignore[valid-type]
    actions: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
    T: int,  # noqa: N803
    loss: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
) -> None:
    c = wp.float32(0.0)
    for t in range(T + 1):
        c = c + (wp.float32(1.0) + wp.cos(states[t, 1]))  # 0 at theta=pi, 2 at theta=0
        c = c + wp.float32(0.01) * states[t, 0] * states[t, 0]
    for t in range(T):
        c = c + wp.float32(0.001) * actions[t] * actions[t]
    loss[0] = c


def optimize_swingup(
    T: int = 100,  # noqa: N803
    iters: int = 300,
    lr: float = 1.5,
    seed: int = 0,
    device: str = "cpu",
) -> npt.NDArray[np.float64]:
    """Adam on the control sequence. Returns the loss history (numpy array)."""
    init_warp()
    sys = SYSTEMS["cartpole"]
    rng = np.random.default_rng(seed)
    actions: wp.array[Any] = wp.array(
        0.1 * rng.standard_normal(T).astype(np.float32),
        dtype=wp.float32,
        requires_grad=True,
        device=device,
    )
    pars: wp.array[Any] = wp.array(
        sys.default_params.astype(np.float32), dtype=wp.float32, device=device
    )
    x0 = np.array([0.0, 0.0, 0.0, 0.0])  # hanging down
    dt = sys.suggested_dt

    m = np.zeros(T)
    v = np.zeros(T)
    b1, b2, eps = 0.9, 0.999, 1e-8
    history: list[float] = []
    for it in range(iters):
        states = _alloc_states(x0, T, requires_grad=True, device=device)
        loss = wp.zeros(1, dtype=wp.float32, requires_grad=True, device=device)
        tape = wp.Tape()
        with tape:
            for t in range(T):
                wp.launch(
                    sys.step_kernel,
                    dim=1,
                    inputs=[states, actions, pars, float(dt), t],
                    device=device,
                )
            wp.launch(_upright_cost, dim=1, inputs=[states, actions, T, loss], device=device)
        tape.backward(loss=loss)
        g = actions.grad.numpy()
        history.append(float(loss.numpy()[0]))
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        mh = m / (1 - b1 ** (it + 1))
        vh = v / (1 - b2 ** (it + 1))
        a = actions.numpy() - lr * mh / (np.sqrt(vh) + eps)
        actions = wp.array(
            a.astype(np.float32), dtype=wp.float32, requires_grad=True, device=device
        )
        tape.zero()
    return np.array(history)
