# [Choosing the Parameters](@id parameters)

The framework has a fair number of settings, but they play very different roles. This page groups them by **what they affect**, in the order in which they are typically chosen:

1. The **problem specification** defines what is certified.
2. The **model and prior** and the **posterior sampling** determine whether the certificate, which refers to the posterior predictive distribution, is meaningful for the true system.
3. The **statistical budget** determines how tight the bound can be.
4. The **synthesis** and **calibration** settings determine how good the candidates are and how efficiently a certified candidate is found. They cannot invalidate the certificate.

The values used in the [vertical-farm case study](@ref case-study-overview) are given as examples. They are set in `experiments/setup.jl`.

## Problem specification

These parameters are design decisions that define the control problem. They are not tuned.

| Parameter | Meaning | Case study |
|:----------|:--------|:-----------|
| `alpha` | Admissible risk level ``\alpha``: the planned trajectory must satisfy the constraints with probability at least ``1 - \alpha``. | 0.05 |
| `beta` | Confidence parameter ``\beta``: the bound holds with confidence at least ``1 - \beta`` over the whole calibration. It has little influence on the bound and can be chosen small. | ``10^{-3}`` |
| `H` | Prediction horizon. | 25 days |
| cost, constraints | Cost function ``J`` and constraint functions ``h_i``, defined through an [`OptimalControlProblem`](@ref). Input bounds are enforced as hard constraints; state and output constraints enter the chance constraint. | |

## Model and prior

The guarantee refers to the posterior predictive distribution and transfers to the true system only if the model and prior are adequate (see [Validation and Certification](@ref method-validation)). In practice, these choices typically require the most effort. They should be based on knowledge that is available independently of the data used for inference. For example, the model structure can be derived from first principles, and the prior can be constructed from expert knowledge, the literature, or data from comparable systems. The [crop model and prior](@ref case-study-model) page of the case study shows how these sources were used for the crop model.

| Choice | Meaning and guidance | Case study |
|:-------|:---------------------|:-----------|
| model structure | State-transition and observation functions ``f_\theta`` and ``g_\theta`` of a [`DiscreteTimeStochasticSSM`](@ref). Structural knowledge, e.g., from first principles, enters here. Errors in the structure are not covered by the guarantee. | SIMPLE crop model |
| noise distributions | Process and measurement noise ``\mathcal{V}_\theta`` and ``\mathcal{W}_\theta``. The measurement noise characterizes the sensor and can often be determined by sensor calibration. Noise distributions may also depend on ``\theta``, so unknown noise levels can be inferred along with the other parameters. | Gaussian, known. Process noise: standard deviations of about 5 % of the average daily state increments under nominal conditions. Measurement noise: standard deviation of about 5 % of the biomass after 50 days under nominal conditions. |
| uncertain parameters | Which parameters are treated as uncertain and inferred from data. These form ``\theta``, and the guarantee accounts for the uncertainty in them, together with the latent states and the noise. All remaining parameters are part of the model structure and are assumed to be known. Including more parameters in ``\theta`` widens the scope of the guarantee but increases the sampling effort, in particular for parameters that are barely identifiable from the data. A sensitivity analysis can indicate which parameters most influence the constrained quantities and should therefore be included. | 6 parameters |
| prior ``p(\theta)`` | Should reflect the actual variability of the parameters across the systems that may be encountered. A prior that is too narrow makes the guarantee overconfident; a prior that is too broad requires more informative data and makes sampling harder. The prior must not be tuned on the data ``\mathcal{D}`` used for inference, since this would use the data twice. | independent truncated Gaussians from published cultivars |
| prior ``p(x_0 \mid \theta)`` | Prior over the initial state. Physically known components can be fixed. | biomass: lognormal with small mean; cumulative temperature: zero (known); senescence variable: truncated Gaussian from published cultivars |

## Posterior sampling

The certificate assumes independent samples from the posterior. The sampler settings therefore matter for validity, and their adequacy must be checked with diagnostics such as trace plots, autocorrelation, effective sample size, and the Gelman–Rubin statistic (see the [sampler tuning](@ref case-study-sampler) of the case study). Since the sampler is tuned for the posterior at hand, it should be revisited whenever the model, the prior, or the amount of data changes substantially.

| Parameter | Meaning | Case study |
|:----------|:--------|:-----------|
| burn-in | Number of discarded iterations per stage ([`MCMCConfig`](@ref)). It must be long enough for each chain to forget its initialization. | 500 |
| thinning | Number of iterations between successive draws of the same chain. Since each scenario comes from its own chain, thinning only matters when chains are continued, e.g., for fresh validation sets between attempts or in the receding-horizon setting. It should be chosen such that successive draws are approximately independent. As a practical check, the empirical autocorrelation of the parameters at this lag should be negligible and the chains should agree with each other (``\hat{R}`` close to 1). | 300 |
| staging (`T_chunk`, `K_intermediate`) | Number of data points added per stage and number of samples per intermediate stage ([`StagedMCMCConfig`](@ref)). Smaller stages keep successive posteriors close, which makes adaptation and warm-starting more reliable. | 5, 20 000 |
| particle count ([`NAdaptation`](@ref)) | Bounds on the number of particles and the target variance of the log-likelihood estimate. Too few particles cause sticky chains; too many waste computation. A target variance of roughly 1–2 is a common choice. | 20–200, target 1.2 |
| proposal ([`GaussianProposal`](@ref), [`RobbinsMonro`](@ref), [`GaussianCovAdaptation`](@ref)) | Initial proposal covariance (e.g., the prior variances) and its adaptation. The shape is learned between stages, the scale within stages. | prior variances, target acceptance 0.234 |
| mode | `Standard()` or `Blocked()`; see [Posterior Inference](@ref method-inference). | `Standard()` |

## Statistical budget

These parameters determine how tight the bound can be and how many attempts the calibration has.

| Parameter | Meaning | Case study |
|:----------|:--------|:-----------|
| `K_val` | Number of validation scenarios per attempt. A larger validation set yields a tighter bound for the same violation rate. It must be large enough that certification at level ``\alpha`` is possible with at least one violation; otherwise, [`AdaptiveCalibrationConfig`](@ref) throws an error. Validation itself is cheap (forward simulation only), but each validation scenario requires its own posterior sample, so in practice ``K_\mathrm{val}`` is limited by the sampling cost. | 5000 |
| `n_max` | Maximum number of synthesis–validation attempts. More attempts give the calibration more room but split the confidence ``\beta`` into smaller parts, ``\beta / n_\mathrm{max}``. Since ``\beta`` has little influence on the bound, this cost is small. | 8 |

## Synthesis

These parameters affect how well the candidates generalize from the training to the validation scenarios and how long each attempt takes.

| Parameter | Meaning | Case study |
|:----------|:--------|:-----------|
| `K_train` | Number of training scenarios in the scenario program. More scenarios approximate the posterior predictive distribution better and thus reduce the approximation error of both the expected cost and the CVaR in the objective. This matters in particular for the CVaR, which is estimated from the comparatively few scenarios in the upper tail. A smaller approximation error typically makes the candidates generalize better to the validation set. The size of the nonlinear program and the solve time grow with it. | 400 |
| `tau` | CVaR tail level ``\tau``. The default ``\tau = \alpha`` aligns the penalized tail with the desired risk level. A larger value can be preferable if only a few training scenarios fall into the tail; see below. | ``\tau = \alpha`` (about 20 scenarios in the tail) |
| solver settings | Passed through [`JuMPSolverConfig`](@ref), with separate attributes for cold and warm starts. The iteration limit of the calibration loop can be kept moderate: an attempt that stalls is better abandoned than allowed to consume the computation time. | Ipopt with HSL |

!!! note "Choice of the tail level"
    While ``\tau = \alpha`` is the natural choice for the population-level objective, the CVaR term of the scenario program is determined by the empirical upper tail, which contains only about ``\tau K_\mathrm{train}`` training scenarios. Since ``K_\mathrm{train}`` is limited by the computational budget, this number can be small, especially for small ``\alpha``; for example, ``\alpha = 0.01`` with ``K_\mathrm{train} = 400`` leaves only about four scenarios in the tail. The empirical tail estimate then has a high variance, and the synthesized input sequence tends to overfit these particular scenarios. In this case, it can be beneficial to increase ``\tau`` beyond ``\alpha`` until enough scenarios enter the tail, at the price of a penalty that is less tightly targeted at the ``\alpha``-fraction of worst outcomes. This affects only the synthesis step; the certificate is computed on the independent validation set, where ``\tau`` plays no role.

## Calibration

These parameters of [`AdaptiveCalibrationConfig`](@ref) only affect the choice of the next risk weight. They change how many attempts are needed and how close the returned bound is to ``\alpha``, but **not the validity of the certificate**.

| Parameter | Meaning | Case study |
|:----------|:--------|:-----------|
| `alpha_tol` | Width of the acceptance band ``[\alpha - \alpha_\mathrm{tol}, \alpha]``. A narrower band returns less conservative candidates but needs more attempts. It should be wider than the sampling noise of the empirical violation rate. | 0.015 |
| `omega_init` | Risk weight of the first attempt. A good choice lands close to the acceptance band; it can be taken from a preliminary sweep of ``\omega``. | 0.8 |
| `s_init` | Initial elasticity ``s = d \log \nu / d \log \lambda``, used until enough attempts are available for a fit. It can be estimated from the same preliminary sweep. | −2.0 |
| `s_flat` | Upper limit on the fitted elasticity. It keeps the estimate negative under sampling noise and thereby bounds the step size from the slope side. | −0.5 |
| `step_bounds` | Minimum and maximum change of ``\log \lambda`` per attempt. The lower bound keeps successive attempts distinguishable above the validation noise; the upper bound limits the damage of a poor slope estimate. | (log 1.15, 2.0) |
| `margin_fraction` | Relative margin ``m`` by which the target violation rate is set below the largest certifiable one, so that validation noise alone does not frequently lead to rejection. | 0.1 |
| `n_recovery` | Number of final attempts reserved for the recovery phase, which walks ``\omega`` to 1 if nothing has certified yet. Zero disables it. | 3 |
| `gamma_decay` | Recency weighting of the elasticity fit. Smaller values favor recent attempts, which better reflect the local slope; larger values average out more noise. | 0.4 |
| `omega_min`, `min_spread` | Lower floor on ``\omega`` and minimum spread of the attempts for the elasticity fit to be used. These are safeguards and rarely need adjustment. | ``10^{-3}``, 0.15 |

A practical way to choose `omega_init` and `s_init` is to solve the scenario program for a grid of risk weights on a representative instance and record the empirical violation rate, as done in the [risk-weight sweep](@ref case-study-calibration) of the case study. In the receding-horizon setting, both are taken from the previous cycle after the first one.
