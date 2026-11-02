# =============================================================================
# risk_weight_sweep.jl: how the violation rate responds to the risk weight
# =============================================================================
# Offline diagnostic for the calibration of the risk weight ω in
# `src/adaptive_CVaR_planner.jl`. The calibration parametrizes the trade-off by
# the relative CVaR penalty weight λ = ω/(1-ω) and, within the transition
# region, approximates the violation probability ν by a power law,
#
#     log ν ≈ a + s · log λ = a + s · logit(ω),
#
# where s < 0 is the local elasticity of ν with respect to λ. Since s is not
# known a priori, it is estimated from the preceding attempts by exponentially
# weighted least squares in log λ and the log of the empirical violation rate.
#
# This script shows the shape of the response that this local model is meant to
# describe. It fixes one posterior (generated from the nominal ground truth, the
# SunnySD cultivar) and one initial condition, sweeps ω on a grid equally spaced
# in log λ, and evaluates every solution on the same large diagnostic set.
#
# The response saturates at both ends and forms a decreasing sigmoid in log λ.
# For small λ the CVaR term has little influence on the solution and ν stays on
# a plateau; for large λ further increases produce diminishing changes and ν
# approaches a floor induced by the remaining uncertainty. In between lies the
# transition region, in which log ν is approximately affine in log λ. A single
# power law cannot describe the whole curve, but within the transition it is a
# good local model, and because s is re-estimated as the calibration proceeds,
# this local linearity is all the calibration relies on.
# =============================================================================

# Reuse the shared setup (model, priors, cost, constraints, solver and sampler
# settings) and the posterior scenario generation of `posterior_planning.jl`
# (which includes `setup.jl`) without running its entry block.
RUN_POSTERIOR_PLANNING_ENTRYPOINT = false
include(joinpath(@__DIR__, "posterior_planning.jl"))

using Printf
using Random
using Plots
using LaTeXStrings

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
const diagnostic_seed = 1   # fixes the noise realizations of the diagnostic set
const K_diag = 5000         # size of the diagnostic set

# Grid in x = log λ = logit(ω), wide enough to show both plateaus of the
# sigmoid: the plateau at small ω and the approach to the floor at large ω.
const x_grid = collect(range(4.5, 0.5; length=41))

# -----------------------------------------------------------------------------
# Sweep
# -----------------------------------------------------------------------------
"""
    sweep_risk_weight()

Solve the OCP along `x_grid` and count violations of every solution on one fixed
diagnostic set.

Solves are warm-started from the solution at the previous grid point, which is
usually close to the next one, so each solve takes fewer iterations than a cold
start. Violations are counted under a re-seeded RNG at every point, so all
solutions face the same noise realizations of the same scenarios (common
random numbers) and the differences between grid points reflect the input
trajectory alone.

The posterior is that of the nominal configuration (see
[`draw_nominal_realization`](@ref)).

# Returns
- `NamedTuple` with the grid (`x`, `omega`), the violation counts (`V`), the
  count at the endpoint ω = 1 (`V_floor`), and the diagnostic set size (`K_diag`).
"""
function sweep_risk_weight()
    realization = draw_nominal_realization()
    rng = method_rng(seed_nominal)
    scenarios = generate_posterior_scenarios(realization, rng)

    # ---- Diagnostic set -----------------------------------------------------
    # Drawn once by continuing the validation chains, then held fixed for the
    # whole sweep. This is the deliberate difference from certification, where
    # each attempt needs a fresh, independent set.
    print("Drawing $(K_diag) diagnostic scenarios... ")
    runtime_sampling = @elapsed begin
        chain_results = run_mcmc!(scenarios.validation_ensemble, validation_ensemble_config,
            model, scenarios.data, K_diag)
        diag_scenarios = reduce(vcat, (c.samples for c in chain_results))[1:K_diag]
    end
    @printf("done (%.1f s)\n", runtime_sampling)

    # ---- Workspace ----------------------------------------------------------
    print("Building workspace... ")
    runtime_build = @elapsed workspace = build_ocp_workspace(
        solver, formulation, model, scenarios.samples_train, omega_init, tau; rng=rng)
    @printf("done (%.1f s)\n", runtime_build)

    # Count violations of `U` on the diagnostic set under fixed noise.
    function count_diagnostic_violations(U)
        validation = validate_candidate(model, diag_scenarios, U, scenario_constraints, problem.cost;
            output_mode=problem.output_mode, rng=Xoshiro(diagnostic_seed))
        return validation.V
    end

    # Order the grid by decreasing ω so the continuation path runs monotonically
    # away from the pure CVaR problem.
    x_sorted = sort(x_grid; rev=true)
    omega_grid = PSP._logistic.(x_sorted)
    V = Vector{Int}(undef, length(x_sorted))

    println("━ Risk weight sweep " * "━"^60)

    # ---- Floor --------------------------------------------------------------
    # ω = 1 is the pure CVaR problem, the most conservative member of the family.
    update_omega!(workspace, 1.0)
    solution_floor = solve_ocp!(workspace; U_init=U_init, warm_start=false)
    V_floor = count_diagnostic_violations(solution_floor.U)
    @printf("  ω = 1 (floor):        V = %d/%d (%.4f%%)\n",
        V_floor, K_diag, 100 * V_floor / K_diag)

    U_warm = solution_floor.U

    # ---- Grid ---------------------------------------------------------------
    for i in eachindex(x_sorted)
        update_omega!(workspace, omega_grid[i])
        solution = solve_ocp!(workspace; U_init=U_warm, warm_start=true)
        U_warm = solution.U

        V[i] = count_diagnostic_violations(solution.U)
        @printf("  x = %+5.2f, ω = %.6f: V = %d/%d (%.4f%%)\n",
            x_sorted[i], omega_grid[i], V[i], K_diag, 100 * V[i] / K_diag)
    end
    println("┗" * "━"^79)

    return (x=x_sorted, omega=omega_grid, V=V, V_floor=V_floor, K_diag=K_diag)
end

# -----------------------------------------------------------------------------
# Figure
# -----------------------------------------------------------------------------
"""
    plot_risk_weight_sweep(result)

Plot the measured response `log ν̂` against `log λ`, together with the floor at ω = 1.

# Arguments
- `result`: the `NamedTuple` returned by [`sweep_risk_weight`](@ref).

# Returns
- The `Plots.Plot` object.
"""
function plot_risk_weight_sweep(result)
    # Grid points without any violation have no logarithm and are left out of
    # the figure; with K_diag draws this only affects rates below about 1/K_diag.
    observed = result.V .> 0
    x = result.x[observed]
    log_nu = log.(result.V[observed] ./ result.K_diag)

    # A zero count at the floor is shown at the resolution limit 1/K_diag.
    log_nu_floor = log(max(result.V_floor, 1) / result.K_diag)

    # ---- Figure -------------------------------------------------------------
    color_main = "#1F5F9E"
    x_lims = (minimum(x_grid) - 0.1, maximum(x_grid) + 0.1)

    p = plot(; legend=false, framestyle=:box, fontfamily="Computer Modern",
        xlims=x_lims, tick_direction=:out,
        grid=true, gridstyle=:dot, gridalpha=0.3,
        guidefontsize=11, tickfontsize=9,
        xlabel=L"\log \lambda = \mathrm{logit}\,\omega", ylabel=L"\log \hat{\nu}")

    # Shade the gap between the response and the floor, which makes the
    # saturation at both ends visible at a glance.
    plot!(p, x, log_nu; fillrange=log_nu_floor, fillalpha=0.12,
        fillcolor=color_main, lw=0)

    # Floor at ω = 1, labelled directly on the line instead of in a legend.
    hline!(p, [log_nu_floor]; lw=1.2, ls=:dash, color=:gray45)
    annotate!(p, minimum(x) + 0.1, log_nu_floor,
        text("floor (ω = 1)", 9, :left, :bottom, :gray35, "Computer Modern"))

    # Measured response.
    plot!(p, x, log_nu; lw=2, color=color_main,
        marker=:circle, ms=5, markercolor=:white,
        markerstrokecolor=color_main, markerstrokewidth=1.5)

    # Second axis at the top showing the corresponding values of ω.
    omega_ticks = [0.7, 0.8, 0.85, 0.9, 0.95, 0.97, 0.98]
    ax_top = twiny(p)
    plot!(ax_top; xlims=x_lims, framestyle=:box, fontfamily="Computer Modern",
        tick_direction=:out, grid=false, guidefontsize=11, tickfontsize=9,
        xticks=(log.(omega_ticks ./ (1 .- omega_ticks)), string.(omega_ticks)),
        xlabel=L"\omega")

    return p
end

# =============================================================================
# Entry point
# =============================================================================
if !@isdefined(RUN_SWEEP_ENTRYPOINT) || RUN_SWEEP_ENTRYPOINT
    result = sweep_risk_weight()
    display(plot_risk_weight_sweep(result))
end