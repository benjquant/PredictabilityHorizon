"""Warp device init + differentiable rollout helpers (CPU, reverse-mode AD).

Every step kernel has the SAME signature so rollout/gradient code is generic::

    kernel(states: wp.array2d(f32), actions: wp.array(f32),
           params: wp.array(f32), dt: float, t: int)

It reads row ``t`` and writes row ``t + 1`` (never overwriting a cell, so reverse-mode
adjoints through the trajectory are correct). Launches are single-threaded (``dim=1``):
these calibration systems are tiny and the point is differentiability, not throughput.

Gradients use ``wp.Tape``. Arrays that need adjoints are allocated with
``requires_grad=True``; the initial state is written into row 0 *before* the tape is
opened so that ``states.grad[0]`` after ``tape.backward`` is exactly ``d loss / d x_0``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import warp as wp

_INITIALIZED = False


def init_warp() -> None:
    """Initialise the Warp runtime exactly once (idempotent)."""
    global _INITIALIZED
    if not _INITIALIZED:
        wp.init()
        _INITIALIZED = True


# Kernel-parameter annotations below use Warp's DSL forms (``wp.array2d(dtype=...)``),
# which are runtime markers the Warp compiler reads from source -- not types mypy can
# parse, hence the targeted ``valid-type`` ignores. The Python wrappers stay fully typed.
@wp.kernel
def linear_step_kernel(
    states: wp.array2d(dtype=wp.float32),  # type: ignore[valid-type]
    actions: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
    params: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
    dt: float,
    t: int,
) -> None:
    """Trivial linear map ``x_{t+1} = a * x_t`` (a = params[0]), for AD verification."""
    a = params[0]
    states[t + 1, 0] = a * states[t, 0]


@wp.kernel
def _sqdist_loss(
    states: wp.array2d(dtype=wp.float32),  # type: ignore[valid-type]
    target: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
    T: int,  # noqa: N803
    loss: wp.array(dtype=wp.float32),  # type: ignore[valid-type]
) -> None:
    """Scalar loss ``||states[T] - target||^2`` written to ``loss[0]``."""
    d = wp.float32(0.0)
    for i in range(states.shape[1]):
        diff = states[T, i] - target[i]
        d = d + diff * diff
    loss[0] = d


def _alloc_states(
    x0: npt.NDArray[Any],
    T: int,  # noqa: N803
    requires_grad: bool,
    device: str,
) -> wp.array[Any, Any]:
    """Allocate a ``(T+1, dim)`` f32 Warp array with ``x0`` in row 0.

    Row 0 is written before any tape is opened, so the assignment is not part of the
    recorded graph and ``states.grad[0]`` is a clean ``d loss / d x_0``.
    """
    dim = int(x0.shape[0])
    arr = wp.zeros((T + 1, dim), dtype=wp.float32, requires_grad=requires_grad, device=device)
    host = arr.numpy()
    host[0] = x0.astype(np.float32)
    arr.assign(host)
    return arr


def rollout(
    step_kernel: wp.Kernel,
    x0: npt.NDArray[Any],
    actions: npt.NDArray[Any],
    params: npt.NDArray[Any],
    dt: float,
    T: int,  # noqa: N803
    device: str = "cpu",
) -> npt.NDArray[np.float64]:
    """Forward rollout (no grad). Returns the ``(T+1, dim)`` trajectory as float64."""
    init_warp()
    states = _alloc_states(np.asarray(x0), T, requires_grad=False, device=device)
    acts: wp.array[Any] = wp.array(
        np.asarray(actions, dtype=np.float32), dtype=wp.float32, device=device
    )
    pars: wp.array[Any] = wp.array(
        np.asarray(params, dtype=np.float32), dtype=wp.float32, device=device
    )
    for t in range(T):
        wp.launch(step_kernel, dim=1, inputs=[states, acts, pars, float(dt), t], device=device)
    return states.numpy().astype(np.float64)


def autodiff_jacobian(
    step_kernel: wp.Kernel,
    state: npt.NDArray[Any],
    u: float,
    params: npt.NDArray[Any],
    dt: float,
    device: str = "cpu",
) -> npt.NDArray[np.float64]:
    """Discrete one-step Jacobian d x_{t+1}/d x_t via reverse-mode AD (n backward passes).

    For each output component i, seeds the adjoint of row 1 with a unit vector at position i
    and reads the adjoint of row 0 = J[i, :].  Uses ``tape.backward(grads={states: seed})``
    (Warp 1.14 API) to inject per-output adjoints without a scalar loss.
    """
    init_warp()
    x0 = np.asarray(state, dtype=np.float32)
    dim = x0.shape[0]
    J = np.zeros((dim, dim), dtype=np.float64)  # noqa: N806
    acts_np = np.array([u], dtype=np.float32)
    pars_np = np.asarray(params, dtype=np.float32)

    for i in range(dim):
        states = _alloc_states(x0, T=1, requires_grad=True, device=device)
        acts: wp.array[Any] = wp.array(acts_np, dtype=wp.float32, requires_grad=True, device=device)
        pars: wp.array[Any] = wp.array(pars_np, dtype=wp.float32, device=device)
        tape = wp.Tape()
        with tape:
            wp.launch(step_kernel, dim=1, inputs=[states, acts, pars, float(dt), 0], device=device)
        # Seed adjoint: dL/d states[1, i] = 1; all others = 0
        seed = np.zeros((2, dim), dtype=np.float32)
        seed[1, i] = 1.0
        seed_arr: wp.array[Any] = wp.array(seed, dtype=wp.float32, device=device)
        tape.backward(grads={states: seed_arr})
        J[i, :] = states.grad.numpy()[0]  # row i: d states[1,i]/d states[0,:]
        tape.zero()
    return J


def rollout_jacobian(
    step_kernel: wp.Kernel,
    x0: npt.NDArray[Any],
    actions: npt.NDArray[Any],
    params: npt.NDArray[Any],
    dt: float,
    T: int,  # noqa: N803
    device: str = "cpu",
) -> npt.NDArray[np.float64]:
    """Rollout Jacobian M = d x_T / d x_0 via reverse-mode AD (dim backward passes).

    One backward pass per output component k of x_T: seed the row-T adjoint with the
    basis vector e_k, backprop through the whole T-step tape, read the row-0 adjoint =
    row k of M. ``actions`` are held fixed (zeros for the passive sensitivity).
    """
    init_warp()
    x0 = np.asarray(x0, dtype=np.float32)
    dim = int(x0.shape[0])
    M = np.zeros((dim, dim))  # noqa: N806
    for k in range(dim):
        states = _alloc_states(x0, T, requires_grad=True, device=device)
        acts: wp.array[Any] = wp.array(
            np.asarray(actions, dtype=np.float32),
            dtype=wp.float32,
            requires_grad=True,
            device=device,
        )
        pars: wp.array[Any] = wp.array(
            np.asarray(params, dtype=np.float32), dtype=wp.float32, device=device
        )
        tape = wp.Tape()
        with tape:
            for t in range(T):
                wp.launch(
                    step_kernel, dim=1, inputs=[states, acts, pars, float(dt), t], device=device
                )
        seed = np.zeros((T + 1, dim), dtype=np.float32)
        seed[T, k] = 1.0
        seed_arr: wp.array[Any] = wp.array(seed, dtype=wp.float32, device=device)
        tape.backward(grads={states: seed_arr})
        M[k, :] = states.grad.numpy()[0]  # row k: d x_T[k] / d x_0
        tape.zero()
    return M


def grad_loss_wrt_x0(
    step_kernel: wp.Kernel,
    x0: npt.NDArray[Any],
    actions: npt.NDArray[Any],
    params: npt.NDArray[Any],
    dt: float,
    T: int,  # noqa: N803
    target: npt.NDArray[Any],
    device: str = "cpu",
) -> npt.NDArray[np.float64]:
    """Gradient of ``||x_T - target||^2`` w.r.t. ``x_0`` via reverse-mode AD.

    The whole rollout plus the loss kernel are recorded on a single ``wp.Tape``; after
    ``tape.backward`` the adjoint of row 0 of the state array is ``d loss / d x_0``.
    """
    init_warp()
    states = _alloc_states(np.asarray(x0), T, requires_grad=True, device=device)
    acts: wp.array[Any] = wp.array(
        np.asarray(actions, dtype=np.float32),
        dtype=wp.float32,
        requires_grad=True,
        device=device,
    )
    pars: wp.array[Any] = wp.array(
        np.asarray(params, dtype=np.float32), dtype=wp.float32, device=device
    )
    tgt: wp.array[Any] = wp.array(
        np.asarray(target, dtype=np.float32), dtype=wp.float32, device=device
    )
    loss = wp.zeros(1, dtype=wp.float32, requires_grad=True, device=device)

    tape = wp.Tape()
    with tape:
        for t in range(T):
            wp.launch(step_kernel, dim=1, inputs=[states, acts, pars, float(dt), t], device=device)
        wp.launch(_sqdist_loss, dim=1, inputs=[states, tgt, T, loss], device=device)
    tape.backward(loss=loss)
    grad: npt.NDArray[np.float64] = states.grad.numpy()[0].astype(np.float64)
    tape.zero()
    return grad
