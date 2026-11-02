# =============================================================================
# yield_threshold.jl — free vs. achievable terminal yield under the prior
# =============================================================================
# Preliminary analysis for choosing `minimum_yield` (Y_min) in `setup.jl`.
#
# The planning problem is only informative if the yield target is neither met for
# free nor out of reach. This script shows both ends: it draws N_draws
# realizations from the prior predictive exactly as the Monte-Carlo study does
# (`draw_realization` in `setup.jl`): true θ and x₀ from the prior, the random
# data-collection inputs, and the process noise. Each system is simulated over
# the data-collection phase to t₀, and then over the remaining H days under two
# constant input profiles, driven by the same noise realization (common random
# numbers):
#
#   • minimal growth   u_min = (T_ambient, D = 1, R = 0, CO₂_ambient),
#                      the lower input bounds except for full drought. With R = 0
#                      there is no growth, so this is also the plan with the
#                      smallest linear cost; the terminal yield is what the crop
#                      carries "for free".
#   • maximal growth   u_max = (T_opt(θ), D = 0, R = R_max, CO₂ = 700 ppm),
#                      with the temperature at the optimum of the respective
#                      system. This is a heuristic upper bound on the achievable
#                      yield, not the exact maximum: a higher temperature also
#                      advances the thermal time and hence senescence.
#
# The result is two distributions of the terminal yield HI·m_B(t_h), reported as
# quantiles and a histogram.
#
# The draws use seeds disjoint from the Monte-Carlo seeds 1, …, R, so that the
# threshold is not tuned on the realizations it is later evaluated on. Since both
# are i.i.d. draws from the same prior predictive, the distributions coincide.
# =============================================================================

include(joinpath(@__DIR__, "setup.jl"))

# Settings of this script
const N_draws = 10000 # number of prior-predictive realizations
const seed_offset = 2_000_000 # draws use seeds seed_offset + 1, …, seed_offset + N_draws
const report_quantiles = [0.05, 0.25, 0.50, 0.75, 0.95]

@assert truth_from_prior "The analysis uses the prior predictive; set `truth_from_prior = true` in `setup.jl`."

# Input profiles of the planning window (H + 1 columns, as in the OCP). The last
# column has no effect on the terminal state.

# Minimal growth: no radiation, no heating, no CO₂ enrichment, full drought.
const u_min_growth = [u_lb[1], u_ub[2], u_lb[3], u_lb[4]]
const U_min_growth = repeat(u_min_growth, 1, H + 1)

"""
    max_growth_inputs(theta)

Constant growth-maximizing input trajectory for the system with parameters
`theta`: temperature at the optimum T_opt, no drought, maximal radiation, and
maximal CO₂ concentration.

# Arguments
- `theta::AbstractVector`: model parameters; `theta[4]` is T_opt.

# Returns
- `Matrix{Float64}`: input trajectory of size `(n_u, H + 1)`.

# Notes
T_opt is clipped to the admissible temperature range. Beyond T_opt the growth
factor `f_temp` saturates, while the thermal time, and with it senescence, keeps
advancing; T_opt is therefore a natural, but not provably optimal, choice.
"""
function max_growth_inputs(theta::AbstractVector)
    T_best = clamp(theta[4], u_lb[1], u_ub[1])
    u_max_growth = [T_best, u_lb[2], u_ub[3], u_ub[4]]
    return repeat(u_max_growth, 1, H + 1)
end

# =============================================================================
# Simulation
# =============================================================================

"""
    terminal_yields(seeds)

Simulate the prior-predictive realizations `seeds` under the minimal and the
maximal growth inputs over the planning window.

# Arguments
- `seeds::AbstractVector{<:Integer}`: seeds passed to [`draw_realization`](@ref).

# Returns
- `NamedTuple` of vectors (one entry per seed) with the yield at t₀ (`yield_t0`),
  the terminal yield under minimal growth (`yield_min`), and the terminal yield
  under maximal growth (`yield_max`), all as HI·m_B [kg m⁻²].
"""
function terminal_yields(seeds::AbstractVector{<:Integer})
    n = length(seeds)
    yield_t0 = Vector{Float64}(undef, n)
    yield_min = Vector{Float64}(undef, n)
    yield_max = Vector{Float64}(undef, n)

    for (k, seed) in enumerate(seeds)
        # True system, excitation, and noise, as in the Monte-Carlo study. Both
        # input profiles see the same process noise V_plan.
        realization = draw_realization(seed)

        X_free = simulate_true_system(realization, U_min_growth)
        X_max = simulate_true_system(realization, max_growth_inputs(realization.truth.theta))

        yield_t0[k] = HI * realization.x_t0[1]
        yield_min[k] = HI * X_free[1, end]
        yield_max[k] = HI * X_max[1, end]
    end

    return (yield_t0=yield_t0, yield_min=yield_min, yield_max=yield_max)
end

seeds = (seed_offset+1):(seed_offset+N_draws)
runtime = @elapsed yields = terminal_yields(seeds)
@printf("Simulated %d prior-predictive realizations in %.2f s.\n", N_draws, runtime)

# =============================================================================
# Quantiles
# =============================================================================

println("━ Terminal yield HI·m_B(t_h) [kg m⁻²] " * "━"^41)
@printf("  %-22s", "quantile")
for p in report_quantiles
    @printf("%9.0f %%", 100 * p)
end
println()
for (label, y) in [("at t₀", yields.yield_t0),
    ("minimal growth (free)", yields.yield_min),
    ("maximal growth", yields.yield_max)]
    @printf("  %-22s", label)
    for p in report_quantiles
        @printf("%11.2f", quantile(y, p))
    end
    println()
end

# =============================================================================
# Plot
# =============================================================================

edges = range(0.0, maximum(yields.yield_max); length=60)
p = histogram(yields.yield_min; bins=edges, alpha=0.5, normalize=:pdf,
    label="minimal growth (free)")
histogram!(p, yields.yield_max; bins=edges, alpha=0.5, normalize=:pdf,
    label="maximal growth")
title!(p, "Terminal yield over $(N_draws) prior-predictive draws")
xlabel!(p, "yield HI·m_B(t_h) [kg m⁻²]")
ylabel!(p, "density")
display(p)
