# =============================================================================
# prior_planning.jl — Certified open-loop planning from the prior (baseline)
# =============================================================================
# Prior-based baseline. It runs the same synthesis–validation
# procedure as the proposed method (`run_certified_planner` in `setup.jl`), but its
# scenarios are drawn from the prior instead of the posterior:
#
#     θ ~ p(θ),   x₀ ~ p(x₀ | θ),   propagated to t₀ under the applied inputs
#                                   with fresh process noise,
#
# i.e., the measurements are ignored. The prior is itself a valid Bayesian
# description of the unknown system, so the guarantees of the Theorem on
# robustness apply to this baseline as well. The comparison with
# `posterior_planning.jl` isolates the benefit of using the measurements to reduce
# the uncertainty.
#
# Prior scenarios are exact, independent draws (`PriorPredictiveSampler`), so the
# training scenarios and every validation set are independent by construction and
# no burn-in or thinning is needed. Sampling is cheap; the cost of this baseline is
# in the synthesis, since the wide prior makes the scenario OCP harder and the
# calibration typically needs more attempts. It is therefore kept separate from
# `posterior_planning.jl`, so that it can be run (or skipped) independently.
#
# For the same `seed`, this script faces exactly the same realization — true
# system, data-collection inputs, and noise — as `posterior_planning.jl` (see
# `draw_realization` in `setup.jl`).
# =============================================================================

include(joinpath(@__DIR__, "setup.jl"))

# Settings of this script
seed = 7 # Monte-Carlo run to simulate when executed as a script
enable_plots = true # whether to plot the planned trajectories and predictions

# Number of independent prior "chains". Each chain has its own RNG seeded from the
# method RNG.
const n_chains_prior = 1

# Exact sampling: every step is an independent draw, so there is no burn-in and no
# thinning, and the validation sets of consecutive attempts are independent.
prior_sampler = PriorPredictiveSampler()
prior_sampler_config = MCMCConfig(0, 0; verbose=false)

# Risk weight of the first attempt.
const omega_init_prior = omega_init

# =============================================================================
# Prior sampling
# =============================================================================

"""
    generate_prior_scenarios(realization, rng)

Draw the prior scenarios of one Monte-Carlo run: K_train training scenarios,
propagated to t₀ under the inputs of the data-collection phase, and the ensemble
that continues to supply fresh validation sets.

# Arguments
- `realization::NamedTuple`: realization from [`draw_realization`](@ref); only its
  inputs are used.
- `rng::AbstractRNG`: method RNG for seeding the chains.

# Returns
- `NamedTuple` with the training scenarios (`samples_train`), the live
  `validation_ensemble`, and the sampling time (`runtime_sampling`).
"""
function generate_prior_scenarios(realization, rng)
    data = realization.data

    runtime_sampling = @elapsed begin
        ensemble = initialize_ensemble(rng, prior_sampler, model, data, n_chains_prior)
        chain_results = run_mcmc!(ensemble, prior_sampler_config, model, data, K_train)
        samples_train = reduce(vcat, (c.samples for c in chain_results))[1:K_train]
    end
    @printf("Prior sampling finished in %.2f seconds.\n", runtime_sampling)

    return (
        samples_train=samples_train,
        validation_ensemble=ensemble,
        runtime_sampling=runtime_sampling,
    )
end

# =============================================================================
# Experiment
# =============================================================================

"""
    run_prior_planning(seed)

Run the prior-based baseline of the open-loop experiment for Monte-Carlo run
`seed` and evaluate its plan on the realization of `seed`.

# Arguments
- `seed::Integer`: index of the Monte-Carlo run. Fixes the realization (via
  [`draw_realization`](@ref)) and the internal randomness of the method (via
  [`method_rng`](@ref)).

# Returns
- `NamedTuple` with
  - `seed`, `method = :prior`, the true system (`truth`) and state at t₀ (`x_t0`),
  - the sampling time `runtime_sampling`,
  - `prior_based`: result of [`run_certified_planner`](@ref).
"""
function run_prior_planning(seed::Integer)
    # The realization is drawn first and independently of every method.
    realization = draw_realization(seed)
    rng = method_rng(seed)

    scenarios = generate_prior_scenarios(realization, rng)

    prior_based = run_certified_planner(scenarios.samples_train, scenarios.validation_ensemble,
        prior_sampler_config, realization, rng; omega_start=omega_init_prior)

    return (
        seed=seed,
        method=:prior,
        truth=realization.truth,
        x_t0=realization.x_t0,
        runtime_sampling=scenarios.runtime_sampling,
        prior_based=prior_based,
    )
end

# =============================================================================
# Entry point for interactive / script execution
# =============================================================================
# Only runs when this file is executed directly, not when it is included by the
# Monte-Carlo runner, which sets RUN_PRIOR_PLANNING_ENTRYPOINT = false.
if !@isdefined(RUN_PRIOR_PLANNING_ENTRYPOINT) || RUN_PRIOR_PLANNING_ENTRYPOINT
    result = run_prior_planning(seed)

    if enable_plots
        for i in 1:n_x
            display(plot_state_prediction(result.prior_based, i; title_suffix=" (prior-based)"))
        end
        for i in 1:n_u
            display(plot_input_trajectories(["prior-based" => result.prior_based.U], i))
        end
    end
end
