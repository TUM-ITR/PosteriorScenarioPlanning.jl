# [Validation and Certification](@id method-validation)

The validation step assesses how the uncertainty remaining after learning affects the robustness of a given control input. Robustness is measured by the probability that the resulting trajectory satisfies the constraints, where the probability is taken with respect to the posterior predictive distribution, i.e., the uncertainty over the dynamics, the latent states, and future disturbances given the data. This probability cannot be computed in closed form. Instead, it is bounded from below using sampled scenarios, with a rigorous statistical guarantee.

## Procedure

Given a candidate input sequence and ``K_\mathrm{val}`` validation scenarios that were generated independently of it, the system is simulated for each scenario over the horizon, and the constraint value ``Z^{[k]} = h(\cdot)`` is evaluated along the resulting trajectory. The number of violations is

```math
V = \sum_{k=1}^{K_\mathrm{val}} \mathbf{1}\left\{ Z^{[k]} > 0 \right\}.
```

This is implemented in [`validate_candidate`](@ref), which also returns the sample-average cost over the validation scenarios.

## Guarantee

Let ``\nu`` denote the probability of constraint violation under the posterior predictive distribution given the data. Since the validation scenarios are independent draws from this distribution and independent of the candidate, the violation count is binomially distributed, ``V \sim \mathrm{Binomial}(K_\mathrm{val}, \nu)``. The Clopper–Pearson construction yields an upper bound ``\delta``, the unique solution of

```math
\sum_{i=0}^{V} \binom{K_\mathrm{val}}{i} \delta^{i} (1 - \delta)^{K_\mathrm{val} - i} = \beta
```

for ``V < K_\mathrm{val}`` (and ``\delta = 1`` otherwise), such that

```math
\mathbb{P}\left[ \nu \leq \delta \right] \geq 1 - \beta.
```

In words: with confidence at least ``1 - \beta``, the candidate satisfies the constraints with probability at least ``1 - \delta``. The bound is computed by [`clopper_pearson_bound`](@ref).

## What the guarantee does and does not depend on

The bound holds for **any** candidate input sequence, regardless of how it was obtained. In particular, it is unaffected by the sample-average approximation used in the synthesis step, by local optimality of the non-convex scenario program, and by the calibration of the risk weight. These aspects determine how good the candidate is and how likely it is to certify, not whether the certificate can be trusted.

The guarantee does rely on:

- **A well-specified model and prior.** The bound refers to the posterior predictive distribution. It transfers to the true system under the assumption that the true system is described by the parametric model, i.e., by the functions ``f_\theta`` and ``g_\theta`` and the noise distributions ``\mathcal{V}_\theta`` and ``\mathcal{W}_\theta`` for some parameter value ``\theta``, and that this parameter value and the initial state are drawn from the prior. Errors in the model structure or a prior that does not reflect the true variability are not accounted for by the bound.
- **Independent posterior samples.** The PMCMC procedure must produce samples that can be treated as independent draws from the posterior, which in practice depends on adequate burn-in and thinning.
- **Independence of the validation set and the candidate.** The validation scenarios must not influence the computation of the candidate they are used to assess. In the calibration loop, each candidate depends on the outcomes of the previous attempts, so every attempt uses a fresh validation set.

## Interpreting the parameters

- For a fixed empirical violation rate ``V / K_\mathrm{val}``, the bound tightens as ``K_\mathrm{val}`` grows. For a fixed ``K_\mathrm{val}``, fewer violations yield a tighter bound.
- The confidence parameter ``\beta`` has comparatively little influence on the bound and can be chosen very small, e.g. ``\beta = 10^{-3}``.
- When the validation is repeated over several attempts, ``\beta`` is split across them (see [Synthesis and Calibration](@ref method-synthesis)).

Since validation only requires forward simulation, ``K_\mathrm{val}`` can be chosen much larger than the number of training scenarios. The main cost of a large validation set is therefore not the validation itself but generating the posterior samples.

!!! note "Scope of the implementation"
    The guarantee in the paper covers arbitrary causal control laws, including output feedback. The current implementation synthesizes and validates open-loop input sequences.
