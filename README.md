# PredictabilityHorizon — Lyapunov exponents as predictability diagnostics

The largest Lyapunov exponent λ₁ measures how rapidly nearby trajectories separate,
linking chaotic dynamics to predictability. This project investigates what that
sensitivity means for two pillars of Physical AI: **differentiable simulators** and
**learned world models**, using small dynamical systems built in **NVIDIA Warp on CPU**.
The acrobot, a two-link pendulum, provides a chaotic test case alongside the integrable
single pendulum and the Hénon–Heiles system.

**Part A — differentiable simulators.** The rollout Jacobian ‖∂x_T/∂x₀‖₂ measures
how strongly initial-state perturbations are amplified—the sensitivity propagated by
reverse-mode autodiff. Its asymptotic exponential growth rate is λ₁; finite-window
estimates can differ. Across 11 tested regimes, the gradient-gain slope tracks the
finite-time Lyapunov estimate (correlation 0.969, regression slope 0.986; Figure 5).
Figure 2 explores the optimisation consequence: under a fixed optimiser budget,
precise acrobot reaching has geometric mean cost above the zero-control baseline
from four tested Lyapunov times onward, while pendulum reaching and forgiving swing-up
stay below it. This supports horizon-dependent difficulty, not a universal cutoff
at T = 1/λ₁ beyond which gradients become unusable.

**Part B — learned world models.** Does a model learn the dynamics’ sensitivity as
well as their next-step predictions? We compare Lyapunov estimates from a plain MLP,
a volume-penalty MLP, and a Hamiltonian neural network (HNN). Across five training
seeds, the HNN gives more consistent estimates, while the tuned penalty model has a
slightly closer median to the reference. Lyapunov analysis reveals differences that
prediction error alone does not establish.

![Five-seed sensitivity comparison](writeup/figures/fig7.png)

See the [paper](writeup/paper.pdf) for the background, methods, results, and limitations
of Parts A and B; the Introduction and Abstract remain drafts.

**Evidence status.** The acrobot implementation was revised to use canonical momenta
and implicit midpoint integration because its earlier velocity-state update did not
preserve the claimed symplectic structure. Figures 2, 5, and 7 contain current evidence
(these numbers identify the `figN.png` source files). Figures 1, 3, 4, 6, and 8 remain
explicitly historical and do not validate the revised simulator. These toy-system
results do not establish Cosmos or real-robot performance.

The `acrobot-symplectic` branch is retained for an application link.
[Main](https://github.com/benjquant/PredictabilityHorizon/tree/main) is the destination
for subsequent development after integration.

## Install

```bash
git clone https://github.com/benjquant/PredictabilityHorizon.git
cd PredictabilityHorizon
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Quickstart

List the available systems:

```bash
predictability-horizon systems
```

Render Figure 7 from the [published measurements](writeup/figures/partb_ablation.json),
without rerunning training:

```bash
python -c 'from pathlib import Path; from predictability_horizon.viz import make_fig7_structured_spectrum; make_fig7_structured_spectrum(Path("output/fig7.png"))'
```

For an interactive introduction, see [the quickstart notebook](notebooks/01_quickstart.ipynb).

## Reproduce the experiments

Run the Part B five-seed study, including model training and sensitivity measurements,
then plot the new results:

```bash
predictability-horizon ablation --out output/partb_ablation.json

python -c 'from pathlib import Path; from predictability_horizon.viz import make_fig7_structured_spectrum; make_fig7_structured_spectrum(Path("output/fig7.png"), results=Path("output/partb_ablation.json"))'
```

This is a long CPU computation. Completed measurements are checkpointed; repeating
the command resumes unfinished work. Use a fresh output location to start a fresh study.

Figure 7 shows medians and min–max ranges across five training seeds, with a fixed
dataset seed. The penalty weight was selected from the tested sweep. Measurements
evaluate learned sensitivity along the reference trajectory, not autonomous model
rollouts; see the paper for the full protocol.

To run all eight figure generators:

```bash
predictability-horizon reproduce --out output/figures
```

This runs the current experiments, including optimisation sweeps and model training
for Figures 3 and 4. **Figure 7 uses the committed study measurements**; use the
commands above to retrain that study. Outputs from the current simulator do not
recreate the paper’s historical figures.

## Validation

```bash
make lint
make typecheck
make test
make calibration
```

The suite includes integration tests.
Registered systems are pendulum, acrobot, Hénon–Heiles, and cartpole; cartpole is not
part of the paper's retained evidence.

## Citation and license

See [CITATION.cff](CITATION.cff) and the [MIT license](LICENSE).
