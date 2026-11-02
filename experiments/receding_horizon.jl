# =============================================================================
# receding_horizon.jl — Certified receding-horizon control from the posterior
# =============================================================================
# Receding-horizon counterpart of `posterior_planning.jl`. Data are collected
# until t₀ as in the open-loop experiment; from t₀ on, the plan is recomputed
# every L = `L_rh` days until harvest. Cycle i starts at t_i = t₀ + iL and
#
#   1. updates the posterior on the data u_{0:t_i−1}, y_{0:t_i−1},
#   2. plans over the remaining horizon H_i = t_h − t_i,
#   3. applies the first L inputs to the true system.
#
# Methods:
#   • Proposed method       `run_certified_closed_loop`. Synthesis–validation loop
#                           (`plan!`) on posterior scenarios, once per cycle
#                           (see "Uncertified and infeasible cycles" below).
#   • Certainty equivalent  `run_certainty_equivalent_closed_loop`. The open-loop
#                           baseline in the same loop: posterior-mean θ and
#                           filtered state at t_i, deterministic prediction.
#   • First plan            `evaluate_first_plan`. The certified plan of the
#                           proposed method from the first cycle at t₀, applied
#                           open-loop until harvest. Isolates the value of
#                           replanning.
#   • Hindsight oracle      As in `posterior_planning.jl`.
#
# Posterior update: the K_train + K_val chains of `posterior_planning.jl` are
# warmed up once on the data until t₀. Each later cycle adapts the sampler to the
# previous cycle's samples, re-evaluates every chain's likelihood on the extended
# data, and continues the chain for a short burn-in before taking one sample.
#
# The two closed loops apply different inputs, so their data differ after t₀.
# The certainty-equivalent loop therefore keeps its own posterior: K_train chains
# copied from the proposed method's warm-up and updated on its own data.
#
# Warm starts between receding-horizon cycles:
# - OCP: initialized with the previous cycle's plan, shifted by L days.
# - Calibration: initialized with the previous elasticity estimate and the
#   previous certified ω, rescaled to the shorter horizon (`carry_over_omega`).
#
# The rescaling is needed because the expected cost sums over the remaining
# H days, whereas the CVaR term only concerns the terminal shortfall and does
# not scale with H. Without it, the same ω would put increasingly more weight
# on risk as the horizon shrinks. To keep the relative weight per day constant,
# the odds are scaled with the horizon:
#     ω_new / (1 - ω_new) = ω_old / (1 - ω_old) * H_new / H_old
#
# Uncertified and infeasible cycles: only seeds with a certified plan at t₀ are
# evaluated. If the cycle at t₀ is uncertified, the seed is stopped right there:
# no inputs are applied, and neither the first plan, the certainty-equivalent
# loop, nor the oracle is run. Afterwards, both closed loops keep replanning in
# every cycle:
# - Proposed method: if a cycle is uncertified, the candidate with the smallest
#   bound δ returned by `plan!` is applied until the next cycle.
# - Certainty equivalent: if a cycle's OCP is infeasible, the remaining inputs of
#   its last feasible plan are applied until the next cycle. Without any feasible
#   plan so far (infeasible at t₀), the solver's last iterate is applied.
# Status of a seed:
#   :all_certified        every cycle certified,
#   :partially_certified  certified at t₀, first uncertified cycle `cycle_fail`
#                         (day `t_fail`),
#   :uncertified_at_t0    not certified at t₀; the seed is not evaluated.
# =============================================================================

# Includes `setup.jl` and `estimate_planning_state` without running the
# open-loop experiment.
RUN_POSTERIOR_PLANNING_ENTRYPOINT = false
include(joinpath(@__DIR__, "posterior_planning.jl"))

# Settings of this script
seed = 7 # Monte-Carlo run to simulate when executed as a script
enable_plots = true # whether to plot the closed-loop trajectories and predictions

const L_rh = 5 # inputs applied per cycle L (1 ≤ L ≤ H)
@assert 1 <= L_rh <= H "L must lie in [1, H]"

# Burn-in K_b' between cycles. The posterior changes little from one cycle to the
# next, so the decorrelation length between validation attempts suffices.
const burnin_cycle = thinning

# Cap on the warm-started ω₁. A cycle may end at ω = 1 (end of the recovery
# ramp), whose logit is infinite; starting just below lets the secant step down.
const omega_start_max = 0.999

# Posterior update between cycles: burn-in K_b', then one sample per chain. The
# Robbins–Monro adaptation retunes the step size during the burn-in only.
continuation_config = MCMCConfig(burnin_cycle, 0; step_adaptations=step_adaptations, verbose=false)

# =============================================================================
# Closed loop
# =============================================================================

"""
    cycle_starts()

Start days ``t_i = t_0 + iL < t_h`` of the control cycles.

# Returns
- `Vector{Int}`: cycle start days.
"""
cycle_starts() = collect(t0:L_rh:(t_h-1))

"""
    initialize_closed_loop(realization)

Closed-loop trajectories over days 0, …, t_h, holding the data-collection phase
of `realization` until t₀ and `NaN` afterwards.

# Arguments
- `realization::NamedTuple`: realization from [`draw_realization`](@ref).

# Returns
- `NamedTuple` of matrices `U`, `X`, `Y` with `t_h + 1` columns (column t is day t − 1).
"""
function initialize_closed_loop(realization)
    closed_loop = (
        U=fill(NaN, n_u, t_h + 1),
        X=fill(NaN, n_x, t_h + 1),
        Y=fill(NaN, n_y, t_h + 1),
    )
    closed_loop.U[:, 1:t0] .= realization.U[:, 1:t0]
    closed_loop.X[:, 1:(t0+1)] .= realization.X[:, 1:(t0+1)]
    closed_loop.Y[:, 1:t0] .= realization.Y[:, 1:t0]
    return closed_loop
end

"""
    apply_inputs!(closed_loop, realization, U_plan, t_i, L_i)

Apply the first `L_i` inputs of `U_plan` from day `t_i` to the true system and
record the states and measurements.

# Arguments
- `closed_loop::NamedTuple`: trajectories from [`initialize_closed_loop`](@ref),
  mutated in place.
- `realization::NamedTuple`: realization from [`draw_realization`](@ref).
- `U_plan::AbstractMatrix`: plan of the cycle; column 1 is day `t_i`.
- `t_i::Int`: start day of the cycle.
- `L_i::Int`: number of inputs applied.

# Returns
- `nothing`
"""
function apply_inputs!(closed_loop, realization, U_plan::AbstractMatrix, t_i::Int, L_i::Int)
    theta = realization.truth.theta
    U, X, Y = closed_loop.U, closed_loop.X, closed_loop.Y
    for s in 1:L_i
        col = t_i + s # day t_i + s − 1
        U[:, col] .= U_plan[:, s]
        # y_t depends on u_t, so it is recorded once u_t is known.
        Y[:, col] .= model.g(X[:, col], U[:, col], theta) .+ realization.W[:, col]
        X[:, col+1] .= model.f(X[:, col], U[:, col], realization.V[:, col], theta)
    end
    return nothing
end

"""
    complete_closed_loop!(closed_loop, realization)

Fill in harvest day t_h (input held at that of the previous day, as in the OCP) and evaluate the
inputs applied from t₀ on.

# Arguments
- `closed_loop::NamedTuple`: trajectories after the last cycle, mutated in place.
- `realization::NamedTuple`: realization from [`draw_realization`](@ref).

# Returns
- `NamedTuple`: fields of [`evaluate_plan`](@ref); its `X_true` equals the
  closed-loop states from t₀ on.
"""
function complete_closed_loop!(closed_loop, realization)
    theta = realization.truth.theta
    closed_loop.U[:, t_h+1] .= closed_loop.U[:, t_h]
    closed_loop.Y[:, t_h+1] .= model.g(closed_loop.X[:, t_h+1], closed_loop.U[:, t_h+1], theta) .+
        realization.W[:, t_h+1]
    return evaluate_plan(realization, closed_loop.U[:, (t0+1):(t_h+1)])
end

# =============================================================================
# Posterior sampling
# =============================================================================

"""
    warm_up_posterior!(ensemble, data)

Staged PMMH from the prior on the data until t₀, one sample per chain, as in
`generate_posterior_scenarios`.

# Arguments
- `ensemble::MCMCEnsemble`: chains initialized from the prior.
- `data::DiscreteTimeIOData`: data until t₀.

# Returns
- `NamedTuple` with one sample per chain (`samples`) and the diagnostics
  `acceptance_ratio` and `runtime_sampling`.
"""
function warm_up_posterior!(ensemble, data)
    runtime_sampling = @elapsed results = run_staged_mcmc!(ensemble, staged_config, stage_adaptations, model, data)
    @printf("Warm-up finished in %.2f seconds (acceptance %.2f %%).\n",
        runtime_sampling, results.mean_acceptance * 100)
    flush(stdout)
    return (samples=results.samples, acceptance_ratio=results.mean_acceptance, runtime_sampling=runtime_sampling)
end

"""
    advance_posterior!(ensemble, samples_prev, data, cycle)

Update the posterior to the extended data of a new cycle by continuing every
chain from its current state.

The proposal covariance and particle count are adapted to `samples_prev` (with
the particle count tuned on `data`), each chain's likelihood is re-evaluated on
`data`, and each chain runs `burnin_cycle` steps before taking one sample.

# Arguments
- `ensemble::MCMCEnsemble`: chains to continue, mutated in place.
- `samples_prev::Vector{<:DiscreteTimeSSMSample}`: samples of the previous cycle.
- `data::DiscreteTimeIOData`: data until t_i.
- `cycle::Int`: cycle index, passed to the stage adaptations.

# Returns
- `NamedTuple`, as [`warm_up_posterior!`](@ref).
"""
function advance_posterior!(ensemble, samples_prev, data, cycle::Int)
    T = size(data.y, 2)
    runtime_sampling = @elapsed begin
        for ws in ensemble.workspaces
            ws.T = T
        end
        PSP.adapt_stage!(ensemble, stage_adaptations, samples_prev, data, model, cycle)
        Threads.@threads for k in 1:length(ensemble)
            PSP.reevaluate_state!(ensemble.samplers[k], ensemble.workspaces[k], model, data)
        end
        chain_results = run_mcmc!(ensemble, continuation_config, model, data, length(ensemble))
    end
    samples = reduce(vcat, (c.samples for c in chain_results))
    acceptance_ratio = mean(c.acceptance_rate for c in chain_results)
    @printf("Posterior update finished in %.2f seconds (acceptance %.2f %%).\n",
        runtime_sampling, acceptance_ratio * 100)
    flush(stdout)
    return (samples=samples, acceptance_ratio=acceptance_ratio, runtime_sampling=runtime_sampling)
end

"""
    split_scenarios(ensemble, update)

Split a cycle's posterior samples into the frozen training scenarios (first
K_train chains) and the live validation ensemble (remaining K_val chains), as in
`generate_posterior_scenarios`.

# Arguments
- `ensemble::MCMCEnsemble`: all K_train + K_val chains.
- `update::NamedTuple`: output of [`warm_up_posterior!`](@ref) or
  [`advance_posterior!`](@ref).

# Returns
- `NamedTuple` with `samples_train`, `validation_ensemble` (shares its chains
  with `ensemble`), its latest draws `samples_val_init` (the first validation
  set of the cycle), and the diagnostics of `update`.
"""
function split_scenarios(ensemble, update)
    validation_ensemble = MCMCEnsemble(
        ensemble.workspaces[(K_train+1):end],
        ensemble.samplers[(K_train+1):end],
    )
    return (
        samples_train=update.samples[1:K_train],
        validation_ensemble=validation_ensemble,
        samples_val_init=update.samples[(K_train+1):end],
        acceptance_ratio=update.acceptance_ratio,
        runtime_sampling=update.runtime_sampling,
    )
end

# =============================================================================
# Proposed method
# =============================================================================

"""
    carry_over_omega(omega, H_i, H_next)

Initial risk weight for the first attempt of the next cycle, carried over from
the final risk weight `omega` of a cycle with horizon `H_i` to the next cycle
with the shorter horizon `H_next`.

The objective (1-ω)·J̄ + ω·CVaR ranks plans as J̄ + λ·CVaR with λ = ω/(1-ω).
The expected cost J̄ sums the daily cost over the remaining horizon, whereas the
CVaR of the terminal yield shortfall does not scale with it. Let λ' denote the
weight on CVaR relative to the average daily cost. Then

    argmin J̄/H + λ'·CVaR = argmin J̄ + (λ'·H)·CVaR.

Keeping the trade-off between daily cost and CVaR fixed therefore means scaling
λ with the horizon, λ_next = λ·H_next/H_i, which is a shift of log(H_next/H_i)
in logit(ω) = log λ. This is the part of the change between cycles that is known
in advance. It assumes a roughly constant daily cost and leaves the remaining
effects (fewer days to recover a yield deficit, the narrowing posterior) to the
calibration. Since the shift is additive in log λ, the elasticity estimated by
the calibration, a slope in log λ, is unaffected and carried over unchanged.

# Arguments
- `omega::Real`: final risk weight of the cycle, in [0, 1].
- `H_i::Int`: horizon of that cycle.
- `H_next::Int`: horizon of the next cycle, `0 < H_next ≤ H_i`.

# Returns
- `Float64`: the rescaled risk weight, clamped to [`omega_min`, `omega_start_max`].
"""
function carry_over_omega(omega::Real, H_i::Int, H_next::Int)
    # A cycle may end at ω = 1 (end of the recovery ramp), where λ is infinite;
    # clamp first so that λ stays finite.
    omega_clamped = clamp(Float64(omega), omega_min, omega_start_max)
    lambda = omega_clamped / (1 - omega_clamped)
    lambda_next = lambda * H_next / H_i
    omega_next = lambda_next / (1 + lambda_next)
    return clamp(omega_next, omega_min, omega_start_max)
end

"""
    plan_certified_cycle(scenarios, data, H_i, U_guess, omega_start, s_start, rng)

Synthesis–validation loop of one cycle over the remaining horizon `H_i`. Returns
the candidate of [`plan!`](@ref): the certified plan, or otherwise the one with
the smallest bound δ.

# Arguments
- `scenarios::NamedTuple`: output of [`split_scenarios`](@ref).
- `data::DiscreteTimeIOData`: data until t_i, from which validation draws.
- `H_i::Int`: remaining horizon t_h − t_i.
- `U_guess::AbstractMatrix`: initial guess of size `(n_u, H_i + 1)`.
- `omega_start::Real`: risk weight ω₁ of the first attempt.
- `s_start::Real`: initial elasticity.
- `rng::AbstractRNG`: method RNG for the scenario noise.

# Returns
- `NamedTuple` with `certified`, `termination`, `delta`, `omega`, `attempts`, the plan `U`, the
  calibration histories, the latest elasticity `last_slope` (`NaN` if none), the
  predictive summary `pred` under `U` (over the last validation set), and the runtimes.
"""
function plan_certified_cycle(scenarios, data, H_i::Int, U_guess, omega_start, s_start, rng)
    problem_i = build_problem(H_i)
    formulation_i = CVaRScenarioFormulation(problem_i)

    runtime_build = @elapsed workspace = build_ocp_workspace(
        solver, formulation_i, model, scenarios.samples_train, omega_start, tau; rng=rng)

    planner = AdaptiveCVaRPlanner(workspace, scenarios.validation_ensemble,
        validation_ensemble_config, calibration_config)
    runtime_plan = @elapsed solution = plan!(planner, model, data;
        U_init=U_guess, omega_init=omega_start, s_init=s_start,
        initial_validation_scenarios=scenarios.samples_val_init)

    U = solution.solution.U
    if !solution.certified
        @warn "Cycle not certified; applying the candidate with the smallest bound δ." solution.delta solution.omega
    end

    # Predictive distribution under U over the validation scenarios of the last
    # attempt, which, unlike the training scenarios, were not optimized against.
    X_pred, _ = simulate_trajectories(model, solution.validation_scenarios, U; rng=rng)

    return (
        certified=solution.certified,
        termination=solution.termination,
        delta=solution.delta,
        omega=solution.omega,
        attempts=solution.attempts,
        U=U,
        delta_history=solution.delta_history,
        violation_history=solution.violation_history,
        omega_history=solution.omega_history,
        update_history=solution.update_history,
        last_slope=planner.history.last_slope,
        pred=predictive_summary(X_pred),
        runtime_build=runtime_build,
        runtime_plan=runtime_plan,
        runtime_optimization=solution.runtime_optimization,
        runtime_validation=solution.runtime_validation,
    )
end

"""
    run_certified_closed_loop(realization, ensemble, warm_up, t_cycles, rng)

Closed loop of the proposed method. Every cycle applies the candidate of
[`plan_certified_cycle`](@ref) until the next cycle: the certified plan, or, if
the cycle is uncertified, the candidate with the smallest bound δ. If the first
cycle is uncertified, the loop stops there, no inputs are applied, and the
method is not evaluated.

# Arguments
- `realization::NamedTuple`: realization from [`draw_realization`](@ref).
- `ensemble::MCMCEnsemble`: K_train + K_val warmed-up chains, mutated in place.
- `warm_up::NamedTuple`: output of [`warm_up_posterior!`](@ref) for `ensemble`.
- `t_cycles::Vector{Int}`: cycle start days.
- `rng::AbstractRNG`: method RNG.

# Returns
- `NamedTuple` with
  - the outcome of the closed loop:
    - `status::Symbol`: `:all_certified`, `:partially_certified`, or
      `:uncertified_at_t0` (see the file header),
    - `cycle_fail::Int`: index of the first uncertified cycle, `0` if every
      cycle certified,
    - `t_fail::Int`: start day of that cycle, `0` if every cycle certified,
    - `n_certified::Int`: number of certified cycles,
    - `all_certified::Bool`: whether every cycle until harvest certified,
  - the per-cycle results `cycles` of the cycles run (fields of
    [`plan_certified_cycle`](@ref) plus `t`, `H`, warm starts,
    and sampling diagnostics), and the total runtimes,
  - unless `status == :uncertified_at_t0`: the trajectories `U`, `X`, `Y` over
    days 0, …, t_h and the fields of [`evaluate_plan`](@ref).
"""
function run_certified_closed_loop(realization, ensemble, warm_up, t_cycles, rng)
    n_cycles = length(t_cycles)
    closed_loop = initialize_closed_loop(realization)

    # First cycle: generic guess and configured calibration.
    U_guess = U_init
    omega_start = omega_init
    s_start = s_init
    update = warm_up

    cycles = NamedTuple[]
    cycle_fail = 0
    t_fail = 0
    for (i, t_i) in enumerate(t_cycles)
        H_i = t_h - t_i
        L_i = min(L_rh, H_i)
        println("━"^80)
        @printf(" CONTROL CYCLE %d/%d: t = %d, H = %d\n", i, n_cycles, t_i, H_i)
        println("━"^80)
        flush(stdout)

        data = DiscreteTimeIOData(closed_loop.U[:, 1:t_i], closed_loop.Y[:, 1:t_i])
        if i > 1
            update = advance_posterior!(ensemble, update.samples, data, i)
        end
        scenarios = split_scenarios(ensemble, update)

        result = plan_certified_cycle(scenarios, data, H_i, U_guess, omega_start, s_start, rng)

        push!(cycles, merge(
            (
                t=t_i,
                H=H_i,
                omega_start=Float64(omega_start),
                s_start=Float64(s_start),
                acceptance_ratio=scenarios.acceptance_ratio,
                runtime_sampling=scenarios.runtime_sampling,
                N_particles=ensemble.samplers[1].N,
            ),
            result,
        ))
        @printf("  Cycle %d: certified: %s (%s) | δ = %.2f %% | ω = %.4f | %d attempts\n",
            i, result.certified ? "yes" : "no", termination_note(result), result.delta * 100,
            result.omega, result.attempts)
        flush(stdout)

        if !result.certified && cycle_fail == 0
            cycle_fail = i
            t_fail = t_i
        end
        if !result.certified && i == 1
            # Not certified at t₀: the seed is not evaluated.
            break
        end

        # Certified plan, or the candidate with the smallest δ (see
        # `plan_certified_cycle`), applied until the next cycle.
        apply_inputs!(closed_loop, realization, result.U, t_i, L_i)

        # Warm starts of the next cycle. The shifted plan already has the
        # H_{i+1} + 1 columns of the next OCP, since the harvest day is fixed.
        U_guess = result.U[:, (L_i+1):end]
        omega_start = carry_over_omega(result.omega, H_i, H_i - L_i)
        if isfinite(result.last_slope)
            s_start = result.last_slope
        end
    end

    if cycle_fail == 0
        status = :all_certified
    elseif cycle_fail == 1
        status = :uncertified_at_t0
    else
        status = :partially_certified
    end

    n_certified = count(c -> c.certified, cycles)
    summary = (
        status=status,
        cycle_fail=cycle_fail,
        t_fail=t_fail,
        n_certified=n_certified,
        all_certified=(n_certified == n_cycles),
        cycles=cycles,
        runtime_sampling=sum(c.runtime_sampling for c in cycles),
        runtime_plan=sum(c.runtime_plan for c in cycles),
    )
    if status === :uncertified_at_t0
        @printf("  Not certified at t₀; proposed method stopped for this seed.\n")
        flush(stdout)
        return summary
    end
    return merge(
        summary,
        (U=closed_loop.U, X=closed_loop.X, Y=closed_loop.Y),
        complete_closed_loop!(closed_loop, realization),
    )
end

"""
    evaluate_first_plan(realization, certified_loop)

Certified plan of the proposed method from the first cycle (at t₀), applied
open-loop until harvest. Only defined if the first cycle certified.

# Arguments
- `realization::NamedTuple`: realization from [`draw_realization`](@ref).
- `certified_loop::NamedTuple`: result of [`run_certified_closed_loop`](@ref).

# Returns
- `NamedTuple` with whether the plan was `certified`, its bound `delta`, the
  inputs `U` from t₀, and the fields of [`evaluate_plan`](@ref).
"""
function evaluate_first_plan(realization, certified_loop)
    first_cycle = certified_loop.cycles[1]
    U = first_cycle.U
    return merge((certified=first_cycle.certified, delta=first_cycle.delta, U=U),
        evaluate_plan(realization, U))
end

# =============================================================================
# Certainty-equivalent baseline
# =============================================================================

"""
    run_certainty_equivalent_closed_loop(realization, ensemble, samples_t0, t_cycles, rng)

Closed loop of the certainty-equivalent planner: each cycle plans for the
posterior-mean θ and the filtered state at t_i as if they were exact (as
[`run_certainty_equivalent`](@ref), over the remaining horizon). The posterior is
updated on this loop's own data.

If a cycle's OCP is infeasible, the remaining inputs of the last feasible plan
are applied until the next cycle. Without any feasible plan so far (infeasible
at t₀), the solver's last iterate is applied instead.

# Arguments
- `realization::NamedTuple`: realization from [`draw_realization`](@ref).
- `ensemble::MCMCEnsemble`: K_train chains warmed up on the data until t₀,
  mutated in place.
- `samples_t0::Vector{<:DiscreteTimeSSMSample}`: their samples after the warm-up.
- `t_cycles::Vector{Int}`: cycle start days.
- `rng::AbstractRNG`: method RNG.

# Returns
- `NamedTuple` with the trajectories `U`, `X`, `Y` over days 0, …, t_h, the
  number of cycles whose OCP was feasible (`n_success`), the number of cycles
  that applied the last feasible plan (`n_fallback`) or the solver's last iterate
  (`n_last_iterate`), the per-cycle results `cycles` (`t`, `H`, `success`,
  `fallback`, `theta_hat`, `x_hat`, the OCP solution `U`, runtimes), the total
  runtimes, and the fields of [`evaluate_plan`](@ref). `fallback` is `:none`
  (feasible), `:last_feasible`, or `:last_iterate`.
"""
function run_certainty_equivalent_closed_loop(realization, ensemble, samples_t0, t_cycles, rng)
    println("━ Certainty-equivalent planning " * "━"^48)
    n_cycles = length(t_cycles)
    closed_loop = initialize_closed_loop(realization)

    samples = samples_t0

    # Remaining inputs of the last feasible plan from the start of the current
    # cycle on (H_i + 1 columns); `nothing` until a cycle is feasible.
    U_feasible = nothing

    cycles = Vector{NamedTuple}(undef, n_cycles)
    for (i, t_i) in enumerate(t_cycles)
        H_i = t_h - t_i
        L_i = min(L_rh, H_i)
        @printf(" Cycle %d/%d: t = %d, H = %d\n", i, n_cycles, t_i, H_i)
        flush(stdout)

        data = DiscreteTimeIOData(closed_loop.U[:, 1:t_i], closed_loop.Y[:, 1:t_i])
        if i == 1
            runtime_sampling = 0.0 # shares the warm-up of the proposed method
        else
            update = advance_posterior!(ensemble, samples, data, i)
            samples = update.samples
            runtime_sampling = update.runtime_sampling
        end

        runtime_plan = @elapsed begin
            theta_hat = mean(s.theta for s in samples)
            x_hat = estimate_planning_state(rng, data, theta_hat)
            solution = solve_deterministic_ocp(theta_hat, x_hat, zeros(n_v, H_i + 1); horizon=H_i)
        end
        success = solution.diagnostics.success
        if success
            fallback = :none
            U_apply = solution.U
        elseif U_feasible !== nothing
            fallback = :last_feasible
            U_apply = U_feasible
            @warn "Certainty-equivalent OCP infeasible; applying the last feasible plan." t_i
        else
            fallback = :last_iterate
            U_apply = solution.U
            @warn "Certainty-equivalent OCP infeasible without a previous feasible plan; applying the solver's last iterate." t_i
        end

        apply_inputs!(closed_loop, realization, U_apply, t_i, L_i)

        # Shift the applied plan to the next cycle, unless it is a last iterate.
        if fallback === :last_iterate
            U_feasible = nothing
        else
            U_feasible = U_apply[:, (L_i+1):end]
        end

        cycles[i] = (
            t=t_i,
            H=H_i,
            success=success,
            fallback=fallback,
            theta_hat=theta_hat,
            x_hat=x_hat,
            U=solution.U,
            runtime_sampling=runtime_sampling,
            runtime_plan=runtime_plan,
        )
    end

    evaluation = complete_closed_loop!(closed_loop, realization)
    n_success = count(c -> c.success, cycles)
    n_fallback = count(c -> c.fallback === :last_feasible, cycles)
    n_last_iterate = count(c -> c.fallback === :last_iterate, cycles)
    @printf("  Feasible cycles: %d / %d (last feasible plan applied: %d, last iterate: %d) | cost: %.3f € m⁻² | yield: %.2f kg m⁻² (constraint satisfied: %s)\n",
        n_success, n_cycles, n_fallback, n_last_iterate, evaluation.cost, evaluation.yield, evaluation.feasible)
    flush(stdout)

    return merge(
        (
            U=closed_loop.U,
            X=closed_loop.X,
            Y=closed_loop.Y,
            n_success=n_success,
            n_fallback=n_fallback,
            n_last_iterate=n_last_iterate,
            cycles=cycles,
            runtime_sampling=sum(c.runtime_sampling for c in cycles),
            runtime_plan=sum(c.runtime_plan for c in cycles),
        ),
        evaluation,
    )
end

# =============================================================================
# Experiment
# =============================================================================

"""
    run_receding_horizon(seed)

Run the receding-horizon experiment for Monte-Carlo run `seed`: the proposed
method and the certainty-equivalent baseline in closed loop, the first plan
applied open-loop, and the hindsight oracle, all evaluated on the
realization of `seed`. If the proposed method does not certify at t₀, the seed
is not evaluated and only the first cycle of the proposed method is run.

# Arguments
- `seed::Integer`: index of the Monte-Carlo run. Fixes the realization (via
  [`draw_realization`](@ref)) and the internal randomness of the methods (via
  [`method_rng`](@ref)).

# Returns
- `NamedTuple` with
  - `seed`, `method = :receding_horizon`, the true system (`truth`) and state at
    t₀ (`x_t0`), the setting `L`, the cycle starts (`t_cycles`), and
    `fallback_policy = :replan` (replanning continues after an uncertified or
    infeasible cycle; see the file header),
  - `proposed`: result of [`run_certified_closed_loop`](@ref),
  - `first_plan`: result of [`evaluate_first_plan`](@ref),
  - `certainty_equivalent`: result of [`run_certainty_equivalent_closed_loop`](@ref),
  - `oracle`: result of [`run_hindsight_oracle`](@ref).
  `first_plan`, `certainty_equivalent`, and `oracle` are `nothing` if the first
  cycle did not certify.
"""
function run_receding_horizon(seed::Integer)
    # The realization is drawn first and independently of every method.
    realization = draw_realization(seed)
    rng = method_rng(seed)
    t_cycles = cycle_starts()

    @printf("Receding horizon: t₀ = %d, L = %d, %d cycles until harvest at t_h = %d\n",
        t0, L_rh, length(t_cycles), t_h)
    flush(stdout)

    # Warm-up on the data until t₀, shared by both closed loops. The
    # certainty-equivalent loop continues copies of the K_train training chains.
    # Buffers are sized for the data of the last cycle, so that the chains can be
    # continued on the growing data set.
    ensemble = initialize_ensemble(rng, sampler, stage_adaptations, model, realization.data, n_chains;
        T_max=last(t_cycles))
    warm_up = warm_up_posterior!(ensemble, realization.data)
    ensemble_ce = MCMCEnsemble(
        deepcopy(ensemble.workspaces[1:K_train]),
        deepcopy(ensemble.samplers[1:K_train]),
    )

    proposed = run_certified_closed_loop(realization, ensemble, warm_up, t_cycles, rng)

    # Only seeds certified at t₀ are evaluated; skip the other methods otherwise.
    if proposed.status === :uncertified_at_t0
        first_plan = nothing
        certainty_equivalent = nothing
        oracle = nothing
    else
        first_plan = evaluate_first_plan(realization, proposed)
        certainty_equivalent = run_certainty_equivalent_closed_loop(
            realization, ensemble_ce, warm_up.samples[1:K_train], t_cycles, rng)
        oracle = run_hindsight_oracle(realization)
    end

    println("┏━ Receding-Horizon Control Complete " * "━"^43)
    @printf("┃  Status: %s | certified cycles: %d / %d\n", proposed.status, proposed.n_certified, length(t_cycles))
    if proposed.status === :uncertified_at_t0
        println("┃  Not certified at t₀; seed not evaluated.")
    else
        if proposed.status === :partially_certified
            @printf("┃  First uncertified cycle %d (t = %d); smallest-δ candidates applied in uncertified cycles\n",
                proposed.cycle_fail, proposed.t_fail)
        end
        @printf("┃  Proposed:                 cost %.3f € m⁻², yield %.2f kg m⁻² (constraint satisfied: %s)\n",
            proposed.cost, proposed.yield, proposed.feasible)
        @printf("┃  First plan (open loop):   cost %.3f € m⁻², yield %.2f kg m⁻² (constraint satisfied: %s)\n",
            first_plan.cost, first_plan.yield, first_plan.feasible)
        @printf("┃  Certainty equivalent:     cost %.3f € m⁻², yield %.2f kg m⁻² (constraint satisfied: %s)\n",
            certainty_equivalent.cost, certainty_equivalent.yield, certainty_equivalent.feasible)
        @printf("┃  Hindsight oracle:         cost %.3f € m⁻², yield %.2f kg m⁻² (success: %s)\n",
            oracle.cost, oracle.yield, oracle.success)
    end
    println("┗" * "━"^79)
    flush(stdout)

    return (
        seed=seed,
        method=:receding_horizon,
        truth=realization.truth,
        x_t0=realization.x_t0,
        L=L_rh,
        t_cycles=t_cycles,
        fallback_policy=:replan,
        proposed=proposed,
        first_plan=first_plan,
        certainty_equivalent=certainty_equivalent,
        oracle=oracle,
    )
end

# =============================================================================
# Analysis and plotting of a single run
# =============================================================================
# Summary printout and biomass figure of one run. `montecarlo/analyze_receding_horizon.jl`
# uses the same functions for the representative run of the Monte-Carlo study.

"""
    print_receding_horizon_run(result)

Print the certified cycles of the proposed planner, the realized yields (with
their excess over the target for the proposed planner and its first plan), and
the first cycle whose CE OCP is infeasible.

# Arguments
- `result::NamedTuple`: result of [`run_receding_horizon`](@ref).
"""
function print_receding_horizon_run(result)
    if result.proposed.status === :uncertified_at_t0
        println("  Not certified at t₀; seed not evaluated.")
        return nothing
    end
    @printf("  Proposed: certified cycles %d/%d\n", result.proposed.n_certified, length(result.t_cycles))
    if !result.proposed.cycles[end].certified
        @warn "Seed $(result.seed): last cycle not certified; its predictive interval is under the smallest-δ candidate."
    end
    excess(y) = 100 * (y / minimum_yield - 1)
    @printf("  Realized yield [kg m⁻²] (target %.2f): receding horizon %.2f (%+.1f %%) | open loop %.2f (%+.1f %%) | CE %.2f | oracle %.2f\n",
        minimum_yield, result.proposed.yield, excess(result.proposed.yield), result.first_plan.yield,
        excess(result.first_plan.yield), result.certainty_equivalent.yield, result.oracle.yield)
    i_infeasible = findfirst(c -> !c.success, result.certainty_equivalent.cycles)
    if i_infeasible === nothing
        println("  CE: OCP feasible in every cycle.")
    else
        t_infeasible = result.certainty_equivalent.cycles[i_infeasible].t
        @printf("  CE: OCP first infeasible at t = %d (H = %d).\n", t_infeasible, t_h - t_infeasible)
    end
    return nothing
end

# I-shaped marker (vertical bar with caps) for the legend entry of the error bar.
function error_bar_marker()
    x = [-0.5, 0.5, 0.5, 0.1, 0.1, 0.5, 0.5, -0.5, -0.5, -0.1, -0.1, -0.5]
    y = [1.0, 1.0, 0.8, 0.8, -0.8, -0.8, -1.0, -1.0, -0.8, -0.8, 0.8, 0.8]
    return Shape(x, y)
end

"""
    plot_receding_horizon_run(result)

Biomass from t₀ to harvest of one receding-horizon run: the realized trajectories
of the proposed planner, of its first plan applied open-loop, of CE and of the
oracle, the 5–95 % predictive interval under the plan at t₀, the corresponding
interval of the terminal biomass under the plan of the last cycle (as an error
bar), and the terminal biomass required by the yield target. Only defined for
runs certified at t₀.

# Arguments
- `result::NamedTuple`: result of [`run_receding_horizon`](@ref).

# Returns
- `Plots.Plot`: the figure.
"""
function plot_receding_horizon_run(result)
    day = t0 .+ (0:H)
    d = day[end]
    target = minimum_yield / HI  # terminal biomass that meets the yield target
    first_cycle = result.proposed.cycles[1]
    pred_t0 = first_cycle.pred
    last_cycle = result.proposed.cycles[end]
    lower_last = last_cycle.pred.q05[1, end]
    upper_last = last_cycle.pred.q95[1, end]

    # Legend order follows the figure in the paper; one default palette color per
    # method. The interval of the last cycle is drawn as an error bar at harvest,
    # with caps of ±0.3 days; its legend entry is a separate, empty series with an
    # I-shaped marker, since the legend would otherwise show a plain line.
    # The x-range extends one day past harvest so the constraint marker is not clipped.
    error_bar_x = [d - 0.3, d + 0.3, NaN, d, d, NaN, d - 0.3, d + 0.3]
    error_bar_y = [lower_last, lower_last, NaN, lower_last, upper_last, NaN, upper_last, upper_last]
    p = plot(; legend=:topleft, title="Biomass trajectories, seed $(result.seed) (receding horizon)",
        xlabel=L"t \ \mathrm{in} \ \mathrm{d}", ylabel=L"m_B \ \mathrm{in} \ \mathrm{kg}\,\mathrm{m}^{-2}",
        xlims=(day[1], d + 1))
    plot!(p, day, result.proposed.X_true[1, :]; color=1, lw=2, label="Receding horizon")
    plot!(p, day, result.first_plan.X_true[1, :]; color=1, lw=2, linealpha=0.5, label="Open loop")
    plot!(p, day, pred_t0.q05[1, :]; fillrange=pred_t0.q95[1, :], color=1, fillalpha=0.2, lw=0,
        label="Pred. interval (t = $(first_cycle.t), 5–95 %)")
    plot!(p, error_bar_x, error_bar_y; color=1, lw=2, label=false)
    scatter!(p, Float64[], Float64[]; color=1, marker=error_bar_marker(), ms=8,
        label="Pred. interval (t = $(last_cycle.t), 5–95 %)")
    plot!(p, day, result.certainty_equivalent.X_true[1, :]; color=2, lw=2, label="CE")
    plot!(p, day, result.oracle.X_true[1, :]; color=3, lw=2, label="Oracle")
    scatter!(p, [d], [target]; color=:black, marker=constraint_marker(), ms=12, label="Constraint")
    return p
end

# =============================================================================
# Entry point for interactive / script execution
# =============================================================================
# Only runs when this file is executed directly, not when it is included by the
# Monte-Carlo runner, which sets RUN_RECEDING_HORIZON_ENTRYPOINT = false.
if !@isdefined(RUN_RECEDING_HORIZON_ENTRYPOINT) || RUN_RECEDING_HORIZON_ENTRYPOINT
    result = run_receding_horizon(seed)

    println("\nSeed $seed")
    print_receding_horizon_run(result)

    if enable_plots && result.proposed.status !== :uncertified_at_t0
        display(plot_receding_horizon_run(result))

        window = (t0+1):(t_h+1)
        plans = Pair{String,AbstractMatrix}[
            "proposed"=>result.proposed.U[:, window],
            "first plan"=>result.first_plan.U,
            "certainty equivalent"=>result.certainty_equivalent.U[:, window],
            "oracle"=>result.oracle.U,
        ]
        t_planned = [c.t for c in result.proposed.cycles]
        t_uncertified = [c.t for c in result.proposed.cycles if !c.certified]
        for i in 1:n_u
            p = plot_input_trajectories(plans, i; title_suffix=" (seed $(result.seed))")
            vline!(p, t_planned .- t0; color=:gray, alpha=0.3, label=false)
            if !isempty(t_uncertified)
                vline!(p, t_uncertified .- t0; color=:red, ls=:dash, label="uncertified cycle")
            end
            display(p)
        end
    end
end
