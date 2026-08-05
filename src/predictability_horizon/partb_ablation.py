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
from torch import nn

from predictability_horizon.lyapunov import lyapunov_spectrum_from_jacobians
from predictability_horizon.structured_models import (
    HNN,
    hnn_cayley_jacobians,
    model_jac_fn,
    train_hnn,
    train_volume_penalty_mlp,
)
from predictability_horizon.systems import (
    SYSTEMS,
    System,
    acrobot,  # noqa: F401  (registration)
)
from predictability_horizon.systems.acrobot import (
    legacy_acrobot_step,
    to_canonical,
    to_legacy,
)
from predictability_horizon.warpsim import rollout
from predictability_horizon.worldmodel import Dataset, make_dataset, train_world_model


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


_MUS = (0.1, 1.0, 10.0)
"""Soft-penalty weights swept.

One value was never neutral. The coordinate change shrank the data -- momenta are ~5x smaller
than the velocities they replaced -- so the fit term shrank while the volume term, being
scale-free, did not: holding mu fixed silently strengthened the penalty. Sweeping reports the
soft method's negative result against its own best case rather than against one arbitrary
setting.
"""


def penalty_label(mu: float) -> str:
    """Run label for one soft-penalty setting. The join key used by ``summarise``."""
    return f"penalty:mu={mu:g}"


def _measure_torch_model(
    label: str, seed: int, model: nn.Module, true_traj: npt.NDArray[np.float64], dt: float
) -> Run:
    """lambda_1 and spectrum sum of a torch model along the true orbit, per-state autograd.

    The MLPs are not integrators of anything, so they have no closed-form Jacobian; this is the
    same per-state autograd path Part B used before spec 3. The Jacobians are materialised once
    and handed to _measure_from_jacobians, so the QR-seed sweep costs a numpy loop rather than
    108000 more torch calls per seed.
    """
    model.eval()
    jac_fn = model_jac_fn(model)
    traj = np.asarray(true_traj)
    jacs = np.stack([jac_fn(traj[t]) for t in range(traj.shape[0] - 1)])
    lam, ssum, qr = _measure_from_jacobians(jacs, dt)
    return Run(label, seed, lam, ssum, qr)


def run_mlp_arms(
    sys: System,
    seeds: Sequence[int],
    true_traj: npt.NDArray[np.float64],
    n_traj: int = 80,
    t_data: int = 2000,
    epochs: int = 120,
    mus: Sequence[float] = _MUS,
) -> list[Run]:
    """The plain MLP and the soft-penalty sweep, on the same fixed orbit as the grid.

    Both spectrum sum and lambda_1 are recorded at every mu, because the penalty has two
    separable jobs: does it achieve volume preservation, and does achieving it help.
    """
    dt = sys.suggested_dt
    ds = make_dataset(sys, n_traj=n_traj, T=t_data, seed=0)
    runs: list[Run] = []
    for seed in seeds:
        plain = train_world_model(ds, epochs=epochs, seed=seed)
        runs.append(_measure_torch_model("plain", seed, plain, true_traj, dt))
        print(f"[mlp] plain seed={seed} lambda1={runs[-1].lambda1:+.4f}")
        for mu in mus:
            pen = train_volume_penalty_mlp(ds, epochs=epochs, penalty=mu, seed=seed)
            runs.append(_measure_torch_model(penalty_label(mu), seed, pen, true_traj, dt))
            print(
                f"[mlp] mu={mu:g} seed={seed} lambda1={runs[-1].lambda1:+.4f} "
                f"sum={runs[-1].spectrum_sum:+.4f}"
            )
    return runs


_X0 = np.array([2.5, 0.0, 0.0, 0.0])
_N_STEPS = 109_200
_TRANSIENT = 1_200
"""109200 - 1200 = 108000 Jacobians = exactly 54.0 s of averaging at dt = 5e-4.

Do NOT shorten. At the retired 5.4 s window the true lambda_1 spanned 0.260..1.242 across 8 QR
seeds -- an estimator noise floor an order of magnitude above any effect this study attributes.
Spread by window: 0.982 at 5.4 s, 0.117 at 18 s, 0.034 at 54 s.
"""
_SEEDS = (0, 1, 2, 3, 4)
_DEFAULT_OUT = Path("writeup/figures/partb_ablation.json")


def run_ablation(
    out: Path = _DEFAULT_OUT, seeds: Sequence[int] = _SEEDS
) -> AblationResult:
    """Run the whole study and write it to JSON. Roughly five hours on CPU.

    Protocol, fixed and carried on every number: x0 = (2.5, 0, 0, 0), dt = 5e-4, 109200 steps
    rolled with the first 1200 dropped -- 54.0 s of averaging after a 0.6 s transient -- and
    every lambda_1 a mean over _QR_SEEDS QR frames, reported with its spread. The true exponent
    is measured on exactly the same states, with the simulator's Cayley Jacobian and the same
    QR treatment, so truth and models are estimated identically.
    """
    sys = SYSTEMS["acrobot"]
    dt = sys.suggested_dt
    traj = true_orbit(sys, _X0, _N_STEPS, _TRANSIENT)

    true_jacs = np.stack(
        [
            cast(
                npt.NDArray[np.float64],
                sys.jacobian(traj[t], 0.0, sys.default_params, dt),
            )
            for t in range(traj.shape[0] - 1)
        ]
    )
    true_lam, _true_sum, true_qr = _measure_from_jacobians(true_jacs, dt)
    print(
        f"[true] lambda1={true_lam:+.4f} (qr spread {true_qr:.4f}) over "
        f"{(_N_STEPS - _TRANSIENT) * dt:.1f} s at dt={dt:g}, {_QR_SEEDS} QR seeds"
    )
    runs = run_grid(sys, seeds, traj) + run_mlp_arms(sys, seeds, traj)
    result = AblationResult(
        runs=runs,
        true_lambda1=float(true_lam),
        true_qr_spread=float(true_qr),
        dt=dt,
        n_steps=_N_STEPS,
        transient_steps=_TRANSIENT,
        qr_seeds=_QR_SEEDS,
    )
    dump(result, out)
    return result


_PANEL_SHARE = 0.25
"""An effect earns Fig 7's second panel only if it is at least this fraction of the shortfall.

Reportability alone is not enough: with tight spreads a fraction of a percent can clear the
noise and still not be worth a picture.
"""

_SHORTFALL_FLOOR = 0.10
"""Floors the shortfall denominator at this fraction of the true exponent.

Without it the panel rule is unsound exactly where the study hopes to land. Corner A is the
corner expected to sit CLOSEST to truth, so a small shortfall is the good outcome, not an edge
case -- and dividing by it makes `share` diverge, handing a panel to any effect at all. Worse,
at shortfall == 0 exactly a bare `if shortfall` guard flips to the opposite failure and refuses
a panel however large the effect. Flooring the denominator removes both branches and is
continuous across zero, so nothing depends on which side of it the study happens to land.
"""


@dataclass(frozen=True)
class Effect:
    """One attributed cause, with the verdict of the pre-registered rule already applied."""

    name: str
    delta: float
    threshold: float
    reportable: bool
    share_of_shortfall: float


@dataclass(frozen=True)
class Summary:
    """The reporting rule of the spec, applied. Fixed before the numbers existed."""

    shortfall: float
    effects: list[Effect]
    additivity_residual: float
    additive: bool
    second_panel: bool
    best_mu: float


def _effect(result: AblationResult, name: str, cell: str, shortfall: float) -> Effect:
    delta = result.median(cell) - result.median("A")
    threshold = max(result.spread("A"), result.spread(cell))
    # Floored denominator -- see _SHORTFALL_FLOOR. Never divide by the raw shortfall: it is
    # smallest precisely when the model is best, which would inflate every effect into a panel.
    denom = max(abs(shortfall), _SHORTFALL_FLOOR * abs(result.true_lambda1))
    share = abs(delta) / denom
    return Effect(
        name=name,
        delta=delta,
        threshold=threshold,
        reportable=abs(delta) > threshold,
        share_of_shortfall=share,
    )


def summarise(result: AblationResult, mus: Sequence[float] = _MUS) -> Summary:
    """Apply the spec's reporting rule.

    An effect is REPORTABLE iff the shift in medians exceeds the full min-max seed range of
    BOTH cells involved -- below that it is not separable from training noise and makes no
    claim. Fig 7's second panel exists iff some reportable effect is also at least
    _PANEL_SHARE of the total shortfall. Additivity is checked against the same spread: if D
    does not sit where the two effects predict, they interact and the result is two
    observations, not a decomposition.
    """
    shortfall = result.true_lambda1 - result.median("A")
    effects = [
        _effect(result, "integrator", "B", shortfall),
        _effect(result, "data", "C", shortfall),
    ]
    predicted = effects[0].delta + effects[1].delta
    residual = abs((result.median("D") - result.median("A")) - predicted)
    additive = residual <= max(result.spread("A"), result.spread("D"))
    second_panel = any(
        e.reportable and e.share_of_shortfall >= _PANEL_SHARE for e in effects
    )
    best_mu = min(
        mus, key=lambda mu: abs(result.median(penalty_label(mu)) - result.true_lambda1)
    )
    return Summary(
        shortfall=shortfall,
        effects=effects,
        additivity_residual=residual,
        additive=additive,
        second_panel=second_panel,
        best_mu=float(best_mu),
    )


def report_lines(result: AblationResult, summary: Summary) -> list[str]:
    """Human-readable summary, in the form the ledger entry wants.

    Every lambda_1 carries its window and step size: a bare number is not a result, because
    the exponent does not converge at these horizons under any integrator.
    """
    proto = (
        f"window {result.window_s:.1f} s, dt = {result.dt:g}, "
        f"mean over {result.qr_seeds} QR seeds"
    )
    lines = [f"Part-B ablation -- all lambda_1 per second, {proto}", ""]
    for label in ("A", "B", "C", "D", "plain", *(penalty_label(m) for m in _MUS)):
        v = result.values(label)
        qr = max(r.qr_spread for r in result.runs if r.label == label)
        lines.append(
            f"  {label:<16} median {result.median(label):+.4f}  "
            f"range [{v[0]:+.4f}, {v[-1]:+.4f}]  train spread {result.spread(label):.4f}  "
            f"worst QR spread {qr:.4f}"
        )
    lines += [
        "",
        f"  true             {result.true_lambda1:+.4f}  (QR spread {result.true_qr_spread:.4f})",
        f"  shortfall (true - A) {summary.shortfall:+.4f}",
        "",
    ]
    for e in summary.effects:
        verdict = "REPORTABLE" if e.reportable else "below seed spread -- no claim"
        lines.append(
            f"  {e.name:<11} delta {e.delta:+.4f}  threshold {e.threshold:.4f}  "
            f"({e.share_of_shortfall:.0%} of shortfall)  {verdict}"
        )
    lines += [
        "",
        f"  additivity residual {summary.additivity_residual:.4f} -- "
        + ("additive" if summary.additive else "NOT additive: report as two observations"),
        f"  Fig 7 second panel: {'yes' if summary.second_panel else 'no'}",
        f"  best penalty weight: mu = {summary.best_mu:g}",
        "",
        "  Seeds vary training only; dataset seed is 0 throughout, so data-sampling",
        "  variance is not measured.",
    ]
    return lines
