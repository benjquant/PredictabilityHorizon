from pathlib import Path

from predictability_horizon.viz import (
    make_fig1_gradient_law,
    make_fig2_swingup,
    make_fig3_error_growth,
    make_fig4_lyapunov_scatter,
    make_fig5_slope_vs_lambda,
    make_fig6_gradient_horizon,
)


def test_fig1_and_fig2_written(tmp_path: Path):
    p1 = make_fig1_gradient_law(out=tmp_path / "fig1.png", fast=True)
    p2 = make_fig2_swingup(out=tmp_path / "fig2.png", fast=True)
    assert p1.exists() and p1.stat().st_size > 0
    assert p2.exists() and p2.stat().st_size > 0


def test_fig3_and_fig4_written(tmp_path: Path):
    p3 = make_fig3_error_growth(out=tmp_path / "fig3.png", fast=True)
    p4 = make_fig4_lyapunov_scatter(out=tmp_path / "fig4.png", fast=True)
    assert p3.exists() and p3.stat().st_size > 0
    assert p4.exists() and p4.stat().st_size > 0


def test_fig5_fig6_written(tmp_path: Path):
    p5 = make_fig5_slope_vs_lambda(out=tmp_path / "fig5.png", fast=True)
    p6 = make_fig6_gradient_horizon(out=tmp_path / "fig6.png", fast=True)
    assert p5.exists() and p5.stat().st_size > 0
    assert p6.exists() and p6.stat().st_size > 0
