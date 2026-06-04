"""`predictability_horizon` CLI: inspect systems and reproduce the figures."""

from __future__ import annotations

from pathlib import Path

import click

from predictability_horizon.systems import (  # noqa: F401  (register)
    SYSTEMS,
    acrobot,
    cartpole,
    pendulum,
)


@click.group()
def main() -> None:
    """Lyapunov-grounded predictability diagnostics."""


@main.command()
def systems() -> None:
    """List registered dynamical systems."""
    for name, sys in SYSTEMS.items():
        click.echo(f"{name}: dim={sys.dim}, dt={sys.suggested_dt}")


@main.command()
@click.option("--out", default="writeup/figures", type=click.Path())
def reproduce(out: str) -> None:
    """Regenerate Fig 1-7 (trains the world models, runs the Part-A sweeps + the
    structure-preserving comparison; ~35 min on CPU)."""
    from predictability_horizon.viz import (
        make_fig1_gradient_law,
        make_fig2_swingup,
        make_fig3_error_growth,
        make_fig4_lyapunov_scatter,
        make_fig5_slope_vs_lambda,
        make_fig6_gradient_horizon,
        make_fig7_structured_spectrum,
    )

    d = Path(out)
    for fn, name in [
        (make_fig1_gradient_law, "fig1.png"),
        (make_fig2_swingup, "fig2.png"),
        (make_fig3_error_growth, "fig3.png"),
        (make_fig4_lyapunov_scatter, "fig4.png"),
        (make_fig5_slope_vs_lambda, "fig5.png"),
        (make_fig6_gradient_horizon, "fig6.png"),
        (make_fig7_structured_spectrum, "fig7.png"),
    ]:
        p = fn(out=d / name)
        click.echo(f"wrote {p}")
