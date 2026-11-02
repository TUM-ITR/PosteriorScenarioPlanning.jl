# [Posterior Inference](@id method-inference)

## Why sampling is needed

To plan under uncertainty, the information contained in the data ``\mathcal{D}`` must be incorporated. If decisions were based solely on the prior, the uncertainty over future state trajectories, obtained by propagating the prior over ``(\theta, x_0)`` through the stochastic dynamics, would typically be so large that the chance constraints become infeasible or can only be satisfied by overly conservative inputs. In the Bayesian framework, this information is captured by the posterior ``p(\theta, x_{0:t_0-1} \mid \mathcal{D})``.

For nonlinear state-space models with latent states, this posterior is analytically intractable: its normalizing constant, the evidence ``p(\mathcal{D})``, requires integrating over the parameters and all latent state trajectories, and the nonlinear dynamics preclude a closed-form solution of this integral. It is, however, possible to draw samples from it, which is sufficient for the control framework: the samples provide access to the uncertainty over the dynamics and the latent states without requiring a closed-form expression of the posterior.

## Particle marginal Metropolis–Hastings

Particle Markov chain Monte Carlo (PMCMC) methods combine an MCMC sampler with a particle filter. The particle filter provides an unbiased estimate of the marginal likelihood ``p(\mathcal{D} \mid \theta)``, which cannot be evaluated exactly but is needed in the Metropolis–Hastings acceptance ratio. Replacing it by this estimate still yields a valid MCMC kernel that targets the exact joint posterior, even with a finite number of particles, so the resulting samples are asymptotically exact. This is what the validation step relies on: its guarantees assume samples from the exact posterior. Variational approximations, in contrast, generally incur an approximation error that does not vanish with increasing computational effort and is difficult to characterize.

The package implements a particle marginal Metropolis–Hastings (PMMH) sampler. In each iteration, a candidate ``\theta'`` is drawn from a Gaussian random-walk proposal, a bootstrap particle filter estimates its likelihood, and the candidate is accepted or rejected via the usual Metropolis–Hastings ratio with the estimated likelihoods. Along with the likelihood estimate, a latent state trajectory is drawn from the candidate's particle system. If the candidate is accepted, both are adopted; otherwise, the previous parameter and trajectory are retained. Each iteration thus yields a joint sample ``(\theta, x_{0:t_0-1})``.

Two modes are available:

- `Standard()`: proposes ``\theta`` only; the initial state is marginalized by the particle filter. This is the recommended default.
- `Blocked()`: proposes ``(\theta, x_0)`` jointly. This can help when the posterior of ``x_0`` is much tighter than its prior or strongly correlated with ``\theta``.

## Making PMMH efficient

The efficiency of PMMH depends mainly on two factors: the proposal should match the geometry of the posterior, and the variance of the log-likelihood estimate should be small enough that the chain does not get stuck after an overestimated likelihood. Both depend on the posterior, which is unknown in advance. The package therefore incorporates the data gradually in **stages**:

1. The chains are first run on a short initial segment of the data.
2. Between stages, further data points are added, and the sampler is adapted using the samples of the previous stage:
   - [`GaussianCovAdaptation`](@ref) sets the shape of the proposal covariance to the empirical covariance of the current samples.
   - [`NAdaptation`](@ref) chooses the number of particles such that the variance of the log-likelihood estimate on the extended data is close to a target value.
3. Within each stage, [`RobbinsMonro`](@ref) adjusts the overall proposal scale to drive the acceptance rate toward a target value (by default 0.234).

Since the posterior changes only slightly from one stage to the next, each stage is warm-started from the final state of the previous one. This is implemented by [`run_staged_mcmc!`](@ref) and configured through [`StagedMCMCConfig`](@ref).

## Independent samples

The validation guarantee requires independent posterior samples. The package obtains them by running **one chain per scenario**: an [`MCMCEnsemble`](@ref) holds ``K_\mathrm{train} + K_\mathrm{val}`` independent chains, each with its own random number generator seeded from a parent generator, and runs them in parallel threads. After burn-in, each chain contributes one sample. Compared with thinning a single long chain, this is embarrassingly parallel and, in the receding-horizon setting, allows every chain to be continued from its own previous sample.

Convergence and mixing can be assessed with [`compute_ess`](@ref), [`compute_gelman_rubin`](@ref), and [`compute_autocorrelation`](@ref). Burn-in and thinning cannot be chosen with rigorous guarantees; they are guided by these diagnostics, as illustrated in the [sampler tuning](@ref case-study-sampler) of the case study.

## Scenarios

A scenario collects all uncertain quantities that, together with the control inputs, determine a future trajectory: a posterior sample of ``\theta``, the state ``x_{t_0}`` obtained by propagating the sampled trajectory one step forward, and sampled process and measurement noise over the horizon. For any given input sequence, a scenario thus yields one realization of the future trajectory under the posterior predictive distribution.

Since the posterior can be sampled arbitrarily often, training scenarios for synthesis and validation scenarios for certification can always be generated independently. Both are drawn from the same posterior, conditioned on all available data. Unlike validation in classical system identification, no part of the data has to be held out for validation, so the full information in the data is available for both synthesis and certification.
