from pathlib import Path

from predictability_horizon.viz import (
    make_fig1_gradient_law,
    make_fig2_trajopt_horizon,
    make_fig3_error_growth,
    make_fig4_lyapunov_scatter,
    make_fig5_slope_vs_lambda,
    make_fig6_gradient_horizon,
)


def test_fig1_and_fig2_written(tmp_path: Path):
    p1 = make_fig1_gradient_law(out=tmp_path / "fig1.png", fast=True)
    p2 = make_fig2_trajopt_horizon(out=tmp_path / "fig2.png", fast=True)
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


def test_fig8_written(tmp_path: Path):
    from predictability_horizon.viz import make_fig8_lambda_convergence

    p8 = make_fig8_lambda_convergence(out=tmp_path / "fig8.png", fast=True)
    assert p8.exists() and p8.stat().st_size > 0


def test_fig7_uses_saved_measurements(tmp_path: Path, monkeypatch):
    import json

    import numpy as np
    from matplotlib.axes import Axes

    from predictability_horizon.viz import make_fig7_structured_spectrum

    source = Path(__file__).resolve().parents[1] / "writeup/figures/partb_ablation.json"
    data = json.loads(source.read_text())
    # Deliberately different numbers catch hard-coded results or a training fallback.
    expected = [[1, 2, 3, 4, 8], [2, 4, 6, 8, 12], [3, 4, 5, 6, 9]]
    for label, samples in zip(["plain", "penalty:mu=10", "A"], expected, strict=True):
        for run in data["runs"]:
            if run["label"] == label:
                run["lambda1"] = samples[run["seed"]]
    data["true_lambda1"] = 7.5
    saved = tmp_path / "study.json"
    saved.write_text(json.dumps(data))
    observed = {}
    bar, axhline = Axes.bar, Axes.axhline

    def capture_bar(self, x, height, **kwargs):
        observed["medians"] = height
        observed["whiskers"] = kwargs["yerr"]
        return bar(self, x, height, **kwargs)

    def capture_reference(self, y, **kwargs):
        observed["reference"] = y
        return axhline(self, y, **kwargs)

    monkeypatch.setattr(Axes, "bar", capture_bar)
    monkeypatch.setattr(Axes, "axhline", capture_reference)
    p7 = make_fig7_structured_spectrum(tmp_path / "elsewhere/fig7.png", fast=True, results=saved)
    assert p7.stat().st_size > 0
    np.testing.assert_allclose(observed["medians"], [3, 6, 5])
    np.testing.assert_allclose(observed["whiskers"], [[2, 4, 2], [5, 6, 4]])
    assert observed["reference"] == 7.5
