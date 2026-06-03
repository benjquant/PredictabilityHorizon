# Lyapunov Exponents as Predictability Diagnostics for Differentiable Simulators and Learned World Models

**Benjamin Strittmatter** — June 2026

## Abstract

Predictability in a physical system is bounded by its Lyapunov spectrum, and two pillars of the NVIDIA Physical-AI stack inherit that bound. (i) In a **differentiable simulator** (Warp / Newton), the rollout's input–output Jacobian $\partial x_T/\partial x_0$ — the gradient reverse-mode autodiff propagates — has spectral norm growing as $e^{\lambda_1 T}$, so analytic simulator gradients are usable only out to a horizon $T \lesssim 1/\lambda_1$. (ii) In a **learned world model** — the regime of large video models such as Cosmos — a faithful model must reproduce the system's Lyapunov spectrum, not merely predict the next state.

We measure both on small, recognizable robot-learning systems (pendulum, cartpole, acrobot) implemented as hand-written differentiable **Warp** kernels on CPU, with a single Benettin/QR Lyapunov estimator (calibrated on the harmonic oscillator and Lorenz; per-step Jacobians cross-checked against Warp autodiff to $\sim10^{-9}$). For the differentiable-simulator law we confirm the one-factor relation $\text{slope}=\lambda_1$ to machine precision on a linear map, and find it consistent with the independently-measured $\lambda_1$ on the chaotic acrobot to within finite-time scatter. For world models we report an honest, partly-negative result: a small MLP reproduces an integrable system's near-zero exponent but **over-amplifies the chaotic one** (its learned $\lambda_1$ is ill-conditioned), because a next-step-accuracy objective does not constrain the Jacobian/sensitivity structure — a cautionary note for certifying the physical fidelity of large world models from cheap surrogates.

The gradient-explosion phenomenon itself is established [Parmas 2018; Metz 2021; Suh 2022]; the contribution here is the **unified Lyapunov-grounded diagnostic** on a named NVIDIA platform, plus the world-model fidelity result and a small reusable toolkit.

## 1. Setup and prior work

Differentiable simulation underpins much of modern robot learning — gradient-based system identification, trajectory optimization, and first-order policy gradients all backpropagate through a simulated rollout. NVIDIA's **Warp** is a Python framework for this, and **Newton** — an open-source robotics physics engine (a Linux Foundation project co-developed by NVIDIA, Google DeepMind, and Disney Research) — is built on Warp and supersedes Warp's now-removed `warp.sim` module. A known failure mode is that backpropagation through long rollouts of chaotic or stiff dynamics produces exploding, high-variance gradients: Parmas et al. (2018) named the "curse of chaos" in model-based policy gradients; Metz et al. (2021) tie the explosion to the growth of the iterated Jacobian; and Suh et al. (ICML 2022) show the resulting first-order gradients can be *worse* than zeroth-order estimators for stiff or chaotic systems.

This note makes that link **quantitative and constructive**: across regimes we show the gradient-gain growth rate tracks the *independently measured* largest Lyapunov exponent $\lambda_1$, give the resulting usable-horizon heuristic, and turn the same Lyapunov lens onto learned world models.

## 2. Method

### 2.1 Systems
Three low-dimensional systems, each a canonical robot-learning benchmark, implemented as semi-implicit Euler integrators written directly as Warp kernels (no `warp.sim`; the rollout is stored as a `(T+1, dim)` array, one new row per step, so reverse-mode adjoints through the trajectory are correct):

| System | Dynamics | dim | role |
|---|---|---|---|
| pendulum | integrable, $\lambda_1 = 0$ | 2 | non-chaotic baseline |
| cartpole | underactuated | 4 | trajectory-optimization target |
| acrobot (double pendulum) | chaotic | 4 | chaotic headline |

For the pendulum the semi-implicit step is exactly symplectic ($\det J = 1$ to machine precision), so $\lambda_1 = 0$ is preserved and the small positive finite-time estimates ($\approx 0.05$–$0.08$) are pure shear/finite-time artifacts. The acrobot's $\lambda_1$ is genuinely positive but **integrator- and $dt$-dependent in magnitude** under semi-implicit Euler ($\approx 1.1\,\text{s}^{-1}$ at $dt = 5\times10^{-4}$, corroborated by an RK4 run at the same step); the *presence* of chaos is integrator-independent, the precise value is not.

### 2.2 Lyapunov spectrum
A single Benettin/QR estimator evolves an orthonormal tangent frame under the discrete one-step Jacobians and accumulates $\log$ of the QR diagonals. It is decoupled from Warp and **calibrated against known answers** — the harmonic oscillator ($\lambda = 0$, exact) and the Lorenz system ($\lambda_1 = 0.880$ measured at 40k steps vs $0.906$ in the literature; the shortfall is **finite-time convergence of the estimator**, not Jacobian error — extending the trajectory to 80k steps gives $0.904$). As an independent check, the estimator recovers Lorenz's near-zero middle exponent and a spectrum that sums to the flow divergence $\text{tr}(Df)$ to four digits.

Per-step Jacobians are available from Warp reverse-mode autodiff; the autodiff Jacobian matches the analytic pendulum Jacobian to $1.7\times10^{-9}$ and the acrobot's central-difference Jacobian to $2\times10^{-5}$.

### 2.3 Consistency of the two estimators
The gradient-gain slope (a least-squares fit of $\log\|M_T\|$) and the Benettin $\lambda_1$ are two *distinct finite-$T$ functionals* of the same Jacobian sequence; they converge to the same value as $T\to\infty$ (Oseledets), which requires $T \gg 1/(\lambda_1-\lambda_2)$. They are **not** algebraically identical at finite $T$ — the slope is the higher-variance of the two — so the residual finite-time scatter (a few to ~10%) *explains* the gap between them rather than the gap being evidence for the law. We therefore report a consistency relation, not an exact identity, and not a physically exact $\lambda_1$.

## 3. Part A — gradient gain in differentiable simulators

The rollout Jacobian $M_T = \partial x_T/\partial x_0 = \prod_{t<T} J_t$ is obtained by $\dim$ reverse-mode passes (seeding each output component of $x_T$, reading the adjoint of $x_0$). Its spectral norm is the worst-case factor by which the simulator amplifies an input perturbation — the gradient gain — and by Oseledets it grows as $e^{\lambda_1 T}$: **one factor** of $\lambda_1$. (The squared-loss gradient $\|\nabla\|x_T\|^2\|$ would carry $2\lambda_1$, but only because $\|x_T\|$ itself grows for an *unbounded* state; we measure the Jacobian norm directly to avoid that confound.)

**Results (Fig. 1).** On the linear map $x_{t+1} = a\,x_t$ the measured slope of $\log\|M_T\|$ equals $\log|a| = \lambda_1$ to machine precision. On the **chaotic acrobot** the measured gradient-gain rate ($\approx 1.07\,\text{s}^{-1}$) is consistent with the independently-measured $\lambda_1$ ($\approx 1.0$–$1.2\,\text{s}^{-1}$, itself varying $\pm\sim20\%$ with horizon and QR seed) — agreement to within the finite-time scatter, not a precision match. The **integrable pendulum** shows sub-exponential Jacobian growth ($\lambda_1 \approx 0$, no exponential blow-up). We illustrate the law on one integrable and one chaotic system rather than claiming generality across all regimes. The practical consequence is a horizon heuristic: **analytic simulator gradients are trustworthy only for $T \lesssim 1/\lambda_1$**; past that the Lyapunov-driven growth dominates and one should truncate the horizon or fall back to a zeroth-order estimator.

**Trajectory optimization (Fig. 2).** As a positive demonstration, cartpole swing-up is solved by Adam descending an upright-tracking cost *through* the differentiable Warp rollout (gradient w.r.t. the control sequence); the cost falls steadily over 300 iterations. The same Jacobian growth quantified in Fig. 1 is what bounds the usable planning horizon for this kind of gradient-based control.

## 4. Part B — predictability of learned world models

A small residual MLP one-step dynamics model is trained (minibatch Adam) on simulator rollouts of each system; we then ask the world-model analog of Part A.

**Prediction error (Fig. 3).** Rolled out autoregressively against the simulator, the model's prediction error grows faster for the chaotic acrobot ($\approx 1.7\,\text{s}^{-1}$) than for the integrable pendulum ($\approx 1.0\,\text{s}^{-1}$). We read this *qualitatively only*: the ordering is partly a **structural** effect (the acrobot's smaller $dt$, higher dimension, and larger velocity scales — an *untrained* network reproduces the same ordering), and these error-growth rates are model-divergence speeds, **not** Lyapunov exponents (they sit far from $\lambda_1$).

**Lyapunov fidelity (Fig. 4).** The sharper test is whether the learned dynamics $g_\phi$ reproduce the true Lyapunov exponent, measured from $g_\phi$'s autograd Jacobians:

| system | true $\lambda_1$ | learned $\lambda_1$ |
|---|---|---|
| pendulum (integrable) | $\approx 0.06\,\text{s}^{-1}$ | $\approx 0.05\,\text{s}^{-1}$ — reproduced |
| acrobot (chaotic) | $\approx 1.0\,\text{s}^{-1}$ | over-amplified and **ill-conditioned**: $\approx 0.5$–$3.5\,\text{s}^{-1}$ depending on horizon and initial condition |

The integrable system's near-zero sensitivity is captured; the chaotic exponent is not. The **mechanism** is concrete: despite a one-step MSE of $\sim10^{-7}$, the learned Jacobian field does not preserve the dynamics' volume (symplectic) structure — the true acrobot's full Lyapunov spectrum sums to $\approx 0$ (volume-preserving), whereas the learned spectrum sums to $\approx -0.44$ and is stretched *outward* (top exponent $\sim1.1 \to \sim3.5$). A next-step-accuracy objective constrains the *increment* but places no constraint on the *local sensitivity*, so the learned $\lambda_1$ is wrong. This is **not** under-training (the gap persists when trained to near-machine-precision MSE, $\sim10^{-8}$) and **not** a measurement artifact (the *true* Jacobian, evaluated along the model's own orbit, does not inflate — $0.98 \approx 1.04$ — so the inflation is genuinely a property of the model's Jacobians).

**Honest reading.** Low next-step loss does not certify physical predictability: it leaves the Lyapunov spectrum — the very thing that *sets* the predictability horizon — unconstrained. This is exactly the question to put to a large video world model such as Cosmos: does it preserve the physical Lyapunov spectrum, or wash it out / hallucinate sensitivity? Our surrogate *hallucinates* sensitivity (over-amplifies). The error-growth diagnostic (Fig. 3) transfers to such models as a qualitative test; the Lyapunov-fidelity diagnostic (Fig. 4) requires recovering physical state from generated video — an open problem, not a drop-in.

## 5. Extensions

- **Cosmos / world models.** The error-growth diagnostic transfers as a qualitative pixel-space predictability test; the Lyapunov-fidelity diagnostic needs a perception model to recover physical state $(\theta, \omega)$ from video — or a latent-space tangent-vector adaptation — both research problems rather than a front-end swap.
- **GR00T / action models.** The natural robustness diagnostic is the finite-time Lyapunov exponent of the closed-loop observation→action→next-state map; this needs the policy in a simulation loop (Isaac Lab / Newton) and numerical (perturbed-rollout) sensitivities, since the policy is a large model with a finite context window (a non-Markovian, non-autodiff-friendly map).
- **Contact and stiff dynamics.** A distinct gradient-pathology regime (non-smooth rather than chaotic), deliberately outside this CPU study.

## 6. Limitations

The world-model results use a small MLP on CPU; the learned-$\lambda_1$ failure is **structural** (a one-step-MSE objective does not constrain the Lyapunov spectrum) rather than a budget artifact — it persists to near-machine-precision loss — so it is the cautionary finding, not a bug. The headline learned acrobot $\lambda_1$ is horizon- and IC-dependent ($\approx 0.5$–$3.5$) and is reported as a range. Lyapunov estimates for near-integrable systems are finite-time limited (pendulum $\approx 0.05$–$0.08$, true value $0$). Finite-time Lyapunov exponents are trajectory- and horizon-dependent, so the acrobot $\lambda_1$ is quoted as $\approx 1.0$–$1.2\,\text{s}^{-1}$ and is integrator/$dt$-specific under semi-implicit Euler. The gradient-gain/$\lambda_1$ consistency holds by construction on the same discrete map regardless of integrator accuracy (§2.3). Most figures use a single seed; Part A illustrates the law on two systems (integrable + chaotic), not a broad sweep.

## References

1. T. Parmas, C. E. Rasmussen, J. Peters, K. Doya. *PIPPS: Flexible Model-Based Policy Search Robust to the Curse of Chaos.* ICML 2018.
2. L. Metz, C. D. Freeman, S. S. Schoenholz, T. Kachman. *Gradients Are Not All You Need.* arXiv:2111.05803, 2021.
3. H. J. T. Suh, M. Simchowitz, K. Zhang, R. Tedrake. *Do Differentiable Simulators Give Better Policy Gradients?* ICML 2022.
4. V. I. Oseledets. *A Multiplicative Ergodic Theorem; Lyapunov Characteristic Numbers for Dynamical Systems.* Trans. Moscow Math. Soc. 19 (1968) 197–231.
5. G. Benettin, L. Galgani, A. Giorgilli, J.-M. Strelcyn. *Lyapunov Characteristic Exponents for Smooth Dynamical Systems; a Method for Computing All of Them.* Meccanica 15 (1980) 9–30.
6. R. Engelken, F. Wolf, L. F. Abbott. *Lyapunov Spectra of Chaotic Recurrent Neural Networks.* Physical Review Research 5 (2023) 043044.
