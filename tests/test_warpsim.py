import numpy as np

from predictability_horizon.warpsim import grad_loss_wrt_x0, init_warp, linear_step_kernel, rollout


def test_linear_map_gradient_matches_closed_form():
    # x_{t+1} = a * x_t (1-D embedded in dim=1). loss = x_T^2.
    # d loss / d x_0 = 2 * a^(2T) * x_0.
    init_warp()
    a, x0, T = 1.1, 1.0, 30  # noqa: N806
    states = rollout(
        linear_step_kernel,
        x0=np.array([x0]),
        actions=np.zeros(T),
        params=np.array([a]),
        dt=1.0,
        T=T,
    )
    assert np.isclose(states[-1, 0], x0 * a**T, rtol=1e-4)

    grad = grad_loss_wrt_x0(
        linear_step_kernel,
        x0=np.array([x0]),
        actions=np.zeros(T),
        params=np.array([a]),
        dt=1.0,
        T=T,
        target=np.array([0.0]),
    )
    # loss = (x_T - 0)^2 ; d/dx0 = 2 * a^T * a^T * x0
    assert np.isclose(grad[0], 2.0 * (a**T) * (a**T) * x0, rtol=1e-3)
