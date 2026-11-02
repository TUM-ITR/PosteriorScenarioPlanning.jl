# =============================================================================
# posterior_planning.jl — Certified open-loop planning from the posterior
# =============================================================================
# Open-loop experiment for the posterior-based methods and the hindsight oracle:
#
#   • Proposed method       Scenarios are drawn via PMMH from the posterior
#                           conditioned on the biomass measurements of the
#                           data-collection phase, and the adaptive CVaR planner
#                           (`plan!`) runs the synthesis–validation loop.
#   • Certainty equivalent  θ is fixed to its posterior mean, computed from the
#                           same PMMH samples; the state at t₀ is estimated by an
#                           additional particle-filter run with this θ, conditioned
#                           on the same measurements. The yield constraint is
#                           enforced on the deterministic prediction without
#                           process noise.
#   • Hindsight oracle      True θ, true state at t₀, and the realized future
#                           process noise; gives a lower bound on the cost of
#                           reaching the target in this run.
#
# All three are evaluated on the same realization (see `draw_realization` in
# `setup.jl`). The prior-based baseline is run separately in `prior_planning.jl`
# and faces the same realization for the same seed.
#
# Scenarios come from a single warm-up of K_train + K_val MCMC chains, one chain
# per scenario, so the scenarios are independent up to the between-stage
# adaptation shared during warm-up. After warm-up, the ensemble is split:
#   • Training scenarios: one frozen sample from each of the first K_train chains,
#                         baked into the OCP workspace (only ω changes between solves).
#   • Validation chains:  the remaining K_val chains, kept live and handed to the
#                         planner. Their first draws after the warm-up validate the
#                         first attempt; every later attempt continues the chains to
#                         draw a fresh, independent validation set.
#
# The experiment lives in `run_posterior_planning(seed)`, which is called by the
# Monte-Carlo runner (see `experiments/montecarlo/`) and by the entry point below.
# =============================================================================

include(joinpath(@__DIR__, "setup.jl"))

# Settings of this script
seed = 7 # Monte-Carlo run to simulate when executed as a script
enable_data_plots = false # whether to plot the generated training data
enable_plots = true # whether to plot the planned trajectories and predictions

# Particles of the particle filter that estimates the state at t₀ for the
# certainty-equivalent planner. A single run, so it can be generous.
const N_particles_ce = 500

# =============================================================================
# Posterior sampling
# =============================================================================

"""
    generate_posterior_scenarios(realization, rng)

Draw the posterior scenarios of one Monte-Carlo run.

Warms up the joint MCMC ensemble via staged PMMH on the data of `realization` and
splits it into the frozen training scenarios and the live validation ensemble.
Factored out of [`run_posterior_planning`](@ref) so that offline studies (e.g.
`risk_weight_sweep.jl`) can work on the same posterior without rerunning the planner.

# Arguments
- `realization::NamedTuple`: realization from [`draw_realization`](@ref).
- `rng::AbstractRNG`: method RNG for seeding the chains.

# Returns
- `NamedTuple` with the data set (`data`), the frozen training scenarios
  (`samples_train`), the live `validation_ensemble`, its latest draws
  (`samples_val_init`, the first validation set of the planner), and the sampling
  diagnostics (`acceptance_ratio`, `runtime_sampling`).
"""
function generate_posterior_scenarios(realization, rng)
    data = realization.data

    # A single ensemble of K_train + K_val independent chains is warmed up jointly via
    # staged PMMH, then one sample is taken per chain. Because each scenario comes from
    # its own chain, the scenarios are mutually independent by construction (the only
    # coupling is the shared between-stage adaptation). The ensemble is split afterwards:
    # the first K_train chains supply the frozen training scenarios baked into the OCP,
    # while the remaining K_val chains stay live and are handed to the planner, which
    # continues them to draw a fresh validation set each attempt.
    runtime_sampling = @elapsed begin
        ensemble = initialize_ensemble(rng, sampler, stage_adaptations, model, data, n_chains)
        results = run_staged_mcmc!(
            ensemble,
            staged_config,
            stage_adaptations,
            model,
            data,
        )
    end

    acceptance_ratio = results.mean_acceptance

    @printf("Sampling finished in %.2f seconds.\n", runtime_sampling)
    @printf("Final acceptance ratio: %.2f %%\n", acceptance_ratio * 100)

    # Split the ensemble. results.samples holds one sample per chain in chain order, so the
    # first K_train entries are the training scenarios, which are frozen into the OCP. The
    # remaining K_val chains form the live validation ensemble. Their first draws after
    # the warm-up are the chains' current states and independent of the first candidate,
    # so the planner validates its first attempt on them and continues the chains for
    # later attempts.
    samples_train = results.samples[1:K_train]
    samples_val_init = results.samples[(K_train+1):end]
    validation_ensemble = MCMCEnsemble(
        ensemble.workspaces[(K_train+1):end],
        ensemble.samplers[(K_train+1):end],
    )

    return (
        data=data,
        samples_train=samples_train,
        validation_ensemble=validation_ensemble,
        samples_val_init=samples_val_init,
        acceptance_ratio=acceptance_ratio,
        runtime_sampling=runtime_sampling,
    )
end

# =============================================================================
# Certainty-equivalent baseline
# =============================================================================

"""
    estimate_planning_state(rng, data, theta)

Certainty-equivalent state estimate at the planning time t₀ = T + 1.

Runs one bootstrap particle filter conditioned on `theta`, takes the filtering
mean ``\\hat{x}_T = \\sum_n w_T^{(n)} x_T^{(n)}`` at the last measurement, and
predicts it one day ahead with the process noise omitted,
``\\hat{x}_{t_0} = \\varphi_\\theta(\\hat{x}_T, u_T)``, consistent with the
deterministic prediction used for planning.

# Arguments
- `rng::AbstractRNG`: RNG of the particle filter.
- `data::DiscreteTimeIOData`: data of the data-collection phase.
- `theta::AbstractVector`: parameters the filter is conditioned on.

# Returns
- `x_hat::Vector`: state estimate at t₀.
"""
function estimate_planning_state(rng, data, theta)
    FT = eltype(theta)
    T = size(data.y, 2)

    buffer = allocate_pf_buffer(model, T, N_particles_ce, FT)
    particle_filter!(rng, buffer, data, model, N_particles_ce, theta)

    # Filtering mean at the last measurement (weights at T are normalized).
    x_hat_T = zeros(FT, n_x)
    for n in 1:N_particles_ce
        x_hat_T .+= buffer.w_pf[T, n] .* PSP.get_x(buffer, T, n)
    end

    # One-day prediction to t₀ without process noise.
    x_hat = model.f(x_hat_T, data.u[:, T], zeros(FT, n_v), theta)
    return Vector{FT}(x_hat)
end

"""
    run_certainty_equivalent(realization, samples, rng)

Certainty-equivalent planner: plan for the posterior-mean parameters and the
particle-filter state estimate as if they were exact, predicting deterministically
without process noise and enforcing the yield constraint on this prediction.

# Arguments
- `realization::NamedTuple`: realization from [`draw_realization`](@ref).
- `samples::Vector{<:DiscreteTimeSSMSample}`: PMMH samples the posterior mean is
  estimated from.
- `rng::AbstractRNG`: RNG of the particle filter.

# Returns
- `NamedTuple` with the solver status (`success`; `false` if the target is
  unreachable for the estimated system), the fields of [`solve_outcome`](@ref),
  the estimates (`theta_hat`, `x_hat_t0`),
  the plan `U`, its nominal prediction `X_nominal` of size `(n_x, H + 1)`, the runtime
  (`runtime`), and the fields of [`evaluate_plan`](@ref).
"""
function run_certainty_equivalent(realization, samples, rng)
    println("━ Certainty-equivalent planning " * "━"^48)
    runtime = @elapsed begin
        theta_hat = mean(s.theta for s in samples)
        x_hat_t0 = estimate_planning_state(rng, realization.data, theta_hat)
        solution = solve_deterministic_ocp(theta_hat, x_hat_t0, zeros(n_v, H + 1))
    end
    evaluation = evaluate_plan(realization, solution.U)
    outcome = solve_outcome(solution)

    @printf("  Success: %s (%s) | cost: %.3f € m⁻² | yield: %.2f kg m⁻² (constraint satisfied: %s)\n",
        solution.diagnostics.success, outcome.termination_status, evaluation.cost, evaluation.yield,
        evaluation.feasible)

    return merge(
        (
            success=solution.diagnostics.success,
            theta_hat=theta_hat,
            x_hat_t0=x_hat_t0,
            U=solution.U,
            X_nominal=solution.X[:, :, 1],
            runtime=runtime,
        ),
        outcome,
        evaluation,
    )
end

# =============================================================================
# Experiment
# =============================================================================

"""
    run_posterior_planning(seed)

Run the posterior-based part of the open-loop experiment for Monte-Carlo run
`seed`: PMMH inference, the proposed certified planner, the certainty-equivalent
baseline, and the hindsight oracle, all evaluated on the realization of `seed`.

All invariant definitions (model, priors, cost, constraints, solver, calibration
config) are read as read-only globals from `setup.jl`.

# Arguments
- `seed::Integer`: index of the Monte-Carlo run. Fixes the realization (via
  [`draw_realization`](@ref)) and the internal randomness of the methods (via
  [`method_rng`](@ref)).

# Returns
- `NamedTuple` with
  - `seed`, `method = :posterior`, the true system (`truth`) and state at t₀ (`x_t0`),
  - the inference diagnostics `runtime_sampling` and `acceptance_ratio`,
  - `proposed`: result of [`run_certified_planner`](@ref),
  - `certainty_equivalent`: result of [`run_certainty_equivalent`](@ref),
  - `oracle`: result of [`run_hindsight_oracle`](@ref).
"""
function run_posterior_planning(seed::Integer)
    # The realization is drawn first and independently of every method.
    realization = draw_realization(seed)
    rng = method_rng(seed)

    scenarios = generate_posterior_scenarios(realization, rng)

    proposed = run_certified_planner(scenarios.samples_train, scenarios.validation_ensemble,
        validation_ensemble_config, realization, rng;
        initial_validation_scenarios=scenarios.samples_val_init)

    certainty_equivalent = run_certainty_equivalent(realization, scenarios.samples_train, rng)

    oracle = run_hindsight_oracle(realization)

    return (
        seed=seed,
        method=:posterior,
        truth=realization.truth,
        x_t0=realization.x_t0,
        runtime_sampling=scenarios.runtime_sampling,
        acceptance_ratio=scenarios.acceptance_ratio,
        proposed=proposed,
        certainty_equivalent=certainty_equivalent,
        oracle=oracle,
    )
end

# =============================================================================
# Analysis and plotting of a single run
# =============================================================================
# Summary printout and biomass figure of one run. `montecarlo/analyze_open_loop.jl`
# uses the same functions for the representative run of the Monte-Carlo study.

"""
    print_open_loop_run(result; prior_based=nothing)

Print the certification of the proposed planner (and of the prior-based planner,
if given) and the realized yields of one open-loop run.

# Arguments
- `result::NamedTuple`: result of [`run_posterior_planning`](@ref).

# Keywords
- `prior_based`: `prior_based` field of the `prior_planning.jl` result of the
  same seed, or `nothing` to omit the prior-based planner.
"""
function print_open_loop_run(result; prior_based=nothing)
    planners = [("Proposed", result.proposed)]
    if prior_based !== nothing
        push!(planners, ("Prior-based", prior_based))
    end
    for (label, planner) in planners
        if planner.certified
            certified_text = "certified"
        else
            certified_text = "not certified"
        end
        @printf("  %-12s %d attempts, δ = %.2f %%, ω = %.4f (%s)\n", label * ":",
            planner.attempts, 100 * planner.delta, planner.omega, certified_text)
    end

    yields = [("proposed", result.proposed.yield), ("CE", result.certainty_equivalent.yield)]
    if prior_based !== nothing
        push!(yields, ("prior-based", prior_based.yield))
    end
    push!(yields, ("oracle", result.oracle.yield))
    @printf("  Realized yield [kg m⁻²] (target %.2f): %s\n", minimum_yield,
        join([@sprintf("%s %.2f", label, y) for (label, y) in yields], " | "))
    return nothing
end

"""
    plot_open_loop_run(result)

Biomass over the planning window of one open-loop run: the realized trajectories
of the proposed, CE and oracle plans, the 5–95 % predictive interval and the
predictive mean under the plan of the proposed planner, the predictive mean of
its initial attempt, the CE prediction, and the terminal biomass required by the
yield target.

# Arguments
- `result::NamedTuple`: result of [`run_posterior_planning`](@ref).

# Returns
- `Plots.Plot`: the figure.
"""
function plot_open_loop_run(result)
    proposed = result.proposed
    ce = result.certainty_equivalent
    day = t0 .+ (0:H)
    target = minimum_yield / HI  # terminal biomass that meets the yield target

    # Legend order and line styles follow the figure in the paper: solid lines are
    # realized on the true system, dashed lines are predictions, dotted is the
    # initial attempt. One default palette color per method.
    # The x-range extends one day past harvest so the constraint marker is not clipped.
    p = plot(; legend=:topleft, title="Biomass trajectories, seed $(result.seed)",
        xlabel=L"t \ \mathrm{in} \ \mathrm{d}", ylabel=L"m_B \ \mathrm{in} \ \mathrm{kg}\,\mathrm{m}^{-2}",
        xlims=(day[1], day[end] + 1))
    plot!(p, day, proposed.X_true[1, :]; color=1, lw=2, label="Realized (proposed)")
    plot!(p, day, proposed.pred_final.q05[1, :]; fillrange=proposed.pred_final.q95[1, :],
        color=1, fillalpha=0.2, lw=0, label="Pred. interval (5–95 %)")
    plot!(p, day, proposed.pred_final.mean[1, :]; color=1, lw=2, ls=:dash, label="Pred. mean (proposed)")
    plot!(p, day, proposed.pred_initial.mean[1, :]; color=1, lw=2, ls=:dot, label="Pred. mean (initial)")
    plot!(p, day, ce.X_true[1, :]; color=2, lw=2, label="Realized (CE)")
    plot!(p, day, ce.X_nominal[1, :]; color=2, lw=2, ls=:dash, label="Pred. (CE)")
    plot!(p, day, result.oracle.X_true[1, :]; color=3, lw=2, label="Oracle")
    scatter!(p, [day[end]], [target]; color=:black, marker=constraint_marker(), ms=12, label="Constraint")
    return p
end

# =============================================================================
# Entry point for interactive / script execution
# =============================================================================
# Only runs when this file is executed directly, not when it is included by
# another script (e.g., the Monte-Carlo runner or `risk_weight_sweep.jl`), which set
# RUN_POSTERIOR_PLANNING_ENTRYPOINT = false before including it.
if !@isdefined(RUN_POSTERIOR_PLANNING_ENTRYPOINT) || RUN_POSTERIOR_PLANNING_ENTRYPOINT
    result = run_posterior_planning(seed)

    if enable_data_plots
        realization = draw_realization(seed)
        for i in 1:n_u
            name, unit = input_info[i]
            p0 = plot(0:t_h, realization.U[i, :]; lw=2, legend=false,
                title=name, xlabel=time_info[2], ylabel=unit)
            display(p0)
        end
        for i in 1:n_y
            name, unit = output_info[i]
            p0 = plot(0:t_h, realization.Y[i, :]; lw=2, legend=false,
                title=name, xlabel=time_info[2], ylabel=unit)
            display(p0)
        end
    end

    println("\nSeed $seed")
    print_open_loop_run(result)

    if enable_plots
        display(plot_open_loop_run(result))

        plans = [
            "proposed" => result.proposed.U,
            "certainty equivalent" => result.certainty_equivalent.U,
            "oracle" => result.oracle.U,
        ]
        for i in 1:n_u
            display(plot_input_trajectories(plans, i))
        end
    end
end
