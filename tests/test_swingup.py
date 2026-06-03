from predictability_horizon.swingup import optimize_swingup


def test_swingup_reduces_loss():
    # From hanging (theta=0) toward upright (theta=pi). Loss = upright-tracking cost.
    history = optimize_swingup(T=100, iters=300, lr=1.5, seed=0)
    assert history[-1] < 0.3 * history[0]  # optimizer makes clear progress
    assert history[-1] < history[0]
