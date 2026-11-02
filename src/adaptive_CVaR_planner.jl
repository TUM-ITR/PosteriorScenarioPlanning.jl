# =============================================================================
# adaptive_CVaR_planner.jl — Adaptive calibration of the CVaR risk weight
# =============================================================================
# This file implements a risk-sensitive optimal control with adaptive
# calibration of the CVaR risk weight ω.
#
# The risk weight ω trades off nominal performance against robustness, but the
# value that achieves the desired robustness level 1-α is not known a priori. The
# calibration proceeds via a synthesis–validation loop:
#
#   1. Solve the CVaR-penalized scenario OCP for the current ω (synthesis).
#   2. Evaluate the candidate input on a fresh, independent validation set and
#      certify it via the Clopper–Pearson bound (validation).
#   3. If the certified bound δ lies in the acceptance band, stop and return the
#      cheapest certified candidate so far; otherwise move ω toward the target
#      violation rate and repeat.
#
# The training scenarios stored in the OCP workspace are held fixed across all
# attempts so that the approximated objective stays consistent; only ω changes
# between solves, which allows efficient warm-starting. A fresh validation set is
# drawn each attempt to preserve the independence required by the Clopper–Pearson
# bound.
#
# Repeated testing inflates the probability of a spurious certification, so the
# confidence parameter is split uniformly via a Bonferroni correction,
# β_att = β / n_max, and at most n_max attempts are permitted.
#
# The update law
# --------------
# Calibration searches in the unconstrained coordinate
#
#     x = logit(ω) = log(ω/(1-ω)) = log λ,
#
# on which the violation probability is assumed to respond approximately as a
# power law, p ∝ λ^s with elasticity s = dlog(p)/dlog(λ) < 0. The approximation
# is used for step sizing and degrades as ω → 1, where p approaches a floor set
# by the posterior spread and the relation flattens. The elasticity estimate is 
# therefore weighted toward recent attempts, and every step is magnitude-guarded 
# so that a poor local fit costs at most one attempt.
#
# The search stops when an attempt's certified bound δ falls inside the
# acceptance band [α - α_tol, α], the returned candidate is then chosen by the
# selection rule below. Rejection is thus two-sided: too many violations, or a
# solution so conservative that it overshoots the band. After a rejected attempt:
#
#   • Secant. The elasticity is estimated from the within-cycle history of
#     (x_j, y_j = log(V_j/K_val)) by exponentially weighted least squares, and a
#     step Δ = (y* - y_j)/ŝ is taken toward the target rate y*. The step is
#     signed by y* - y_j: it raises ω after too many violations and lowers it
#     after an overshoot. Only its magnitude is guarded, by `step_bounds`: the
#     floor keeps the next attempt distinguishable from the current one above the
#     sampling noise of the validation set, the ceiling keeps a single bad slope
#     estimate from consuming the remaining attempts. Both arguments are
#     direction-symmetric, hence the two-sided guard.
#   • Bracketing. Once the cycle has seen an attempt on either side of the
#     target, one with too many violations (y > y*) at x_lo and one with too few
#     (y < y*) at x_hi > x_lo, the target lies between them. A secant step that
#     would not land at least Δ_min inside the bracket (x_lo, x_hi) is replaced
#     by the bracket midpoint. 
#     This matters where the elasticity is uninformative: across a steep transition
#     the attempts saturate at V = 0 and V = K_val, neither of which refines the
#     fit, and the clipped steps of equal magnitude would alternate between the
#     same two risk weights indefinitely. The bracket is rebuilt from the history
#     at every update. If validation noise has made the extremes inconsistent
#     (x_lo ≥ x_hi), the most recent attempt on each side is used instead: it
#     reflects the current solver branch and the latest validation draws, so a
#     single contradictory point does not disable the guard for the rest of the
#     cycle. Only if those are inconsistent too is the secant used unguarded.
#   • Recovery phase. If no attempt has certified yet, the last `n_recovery`
#     attempts abandon the feedback search and walk ω up to 1 (the pure CVaR
#     problem, i.e., the most conservative member of the family) in equal steps,
#     arriving exactly at attempt n_max. Each recovery step covers the same
#     fraction 1/n_recovery of the distance remaining on entry, and the more
#     conservative of the recovery and secant candidates is taken. The phase is
#     defined in ω rather than in x because ω = 1 sits at x = ∞ and no finite
#     sequence of Δ-steps reaches it. Recovery attempts are recorded and fitted
#     like any other: they measure the flattened region near the floor, which no
#     earlier point covers. Once some attempt has certified, the phase switches
#     off — its purpose is to establish whether the instance certifies at all,
#     and the remaining attempts are then better spent recovering performance.
#     Setting n_recovery = 0 removes the phase altogether, leaving every attempt
#     to the secant; ω = 1 is then never reached, since the secant candidate is
#     capped strictly below it to keep its logit finite.
#   • Lower floor. Downward steps are floored at ω_min. On instances where the
#     cost-only solution (ω → 0) already overshoots into the band's conservative
#     side, the response is flat near ω = 0 and further decreases change neither
#     the solution nor δ. An attempt at ω ≤ ω_min that falls below the band is
#     therefore terminal: it certifies, and no smaller ω is worth trying.
#
# Selection. Any attempt with δ ≤ α certifies. Among those, the candidate with
# the lowest sample-average cost on its validation set is returned. If every
# solve returned a global minimizer, the nominal cost would be nondecreasing in
# ω, and the smallest certified ω would be the best choice. The scenario OCP is
# non-convex, however, and candidates may come from local optima or from solves
# stopped at an iteration or time limit (which are validated like any other),
# so this ordering can fail; the measured cost does not rely on it.
# Certifying without landing in the band is still success — the band expresses 
# which certified solution is preferred, not whether one was obtained. 
# Only if no attempt certified is failure reported, together with the
# smallest-δ candidate. Selecting freely among attempts needs no correction: the
# union bound behind β_att = β/n_max already covers all n_max of them.
#
# The floor on |Δ| is also the resolution limit of the secant: consecutive
# secant attempts differ by at least a factor exp(Δ_min) in λ, so with a narrow
# band the secant alone straddles the band rather than landing inside it.
# Bisection is not subject to the floor, since it relies only on the sign of the
# miss, not on a slope estimated from the difference of two noisy points; it
# therefore narrows such a bracket further.
#
# Layering: an `AbstractOCPWorkspace` (e.g. `CVaRScenarioWorkspace`) is the pure
# optimization engine and exposes `solve_ocp!`. The `AdaptiveCVaRPlanner` defined
# here is the closed-loop object: it bundles that engine, a sampler for drawing
# fresh validation scenarios, and the calibration configuration, and exposes
# `plan!`.
# =============================================================================

# -----------------------------------------------------------------------------
# Calibration configuration
# -----------------------------------------------------------------------------

"""
    AdaptiveCalibrationConfig

Configuration of the adaptive CVaR calibration procedure.

This struct holds the parameters of the adaptive calibration procedure — the
risk and confidence targets, the ``\\omega`` update, and the validation loop.
The OCP engine and sampler are held by [`AdaptiveCVaRPlanner`](@ref).

The procedure adjusts a candidate's risk weight until its certified risk bound
``\\delta`` — an upper bound on the constraint-violation probability, obtained
from the validation step — falls into the acceptance band below the target risk
level ``\\alpha``.

# Fields
## Risk and confidence targets
- `alpha::FT`: risk level ``\\alpha``. A solution is certified when
  ``\\delta \\leq \\alpha``, i.e., the realized trajectory satisfies the
  constraints with probability at least ``1 - \\alpha``.
- `alpha_tol::FT`: width of the acceptance band below ``\\alpha``. The search
  stops when ``\\delta \\in [\\alpha - \\alpha_{\\mathrm{tol}}, \\alpha]``. Solutions with
  ``\\delta < \\alpha - \\alpha_{\\mathrm{tol}}`` are considered too conservative.
- `beta::FT`: overall confidence level ``\\beta``. The certification holds with
  confidence at least ``1 - \\beta`` across the whole procedure.
- `n_max::Int`: maximum number of synthesis–validation attempts. Also acts as an
  infeasibility detector: if no candidate is certified within `n_max` attempts,
  the posterior uncertainty is likely too large for the desired robustness level.
- `K_val::Int`: number of validation scenarios drawn per attempt

## Update law
- `omega_init::FT`: risk weight ``\\omega_1`` of the first attempt of every cycle,
  unless overridden by the `omega_init` keyword of [`plan!`](@ref)
- `omega_min::FT`: lower floor ``\\omega_{\\mathrm{min}} \\in (0, \\omega_1)`` on the
  risk weight. The secant never proposes a smaller value, and an attempt at the
  floor that is certified but below the acceptance band terminates the cycle,
  since lowering ``\\omega`` further cannot reduce conservatism.
- `s_init::FT`: initial elasticity ``s_{\\mathrm{init}} < 0``, used for the first step and
  whenever the cycle has not yet produced a well-conditioned fit, unless
  overridden by the `s_init` keyword of [`plan!`](@ref)
- `s_flat::FT`: upper limit ``s_{\\mathrm{flat}} < 0`` on the elasticity. Every fitted
  slope above it (including any positive one) is capped there, which bounds 
  the resulting step from the slope side.
- `step_bounds::Tuple{FT,FT}`: minimum and maximum magnitude of the step in
  ``x = \\log\\lambda``, i.e. the multiplicative bounds on the factor by which
  ``\\lambda`` may change per attempt. The direction is set by the sign of the
  miss, so the same bounds guard steps in both directions.
- `margin_fraction::FT`: fraction ``m \\in [0, 1)`` by which the target violation
  rate is set below the largest certifiable one, ``p^* = (1 - m) \\, p^*_\\alpha``
- `n_recovery::Int`: number of final attempts given to the recovery phase, active
  only while no candidate has certified. Zero disables the phase, leaving the
  search entirely to the secant.
- `gamma_decay::FT`: recency weighting ``\\gamma \\in (0, 1]`` of the elasticity
  fit; ``\\gamma = 1`` is an unweighted fit over the non-recovery attempts of 
  the cycle.
- `min_spread::FT`: minimum weighted spread of the abscissae ``x`` below which
  the elasticity fit is considered ill-conditioned

# Notes
- The confidence is split uniformly via a Bonferroni correction,
  ``\\beta_{\\mathrm{att}} = \\beta / n_{\\mathrm{max}}``, to maintain the overall confidence
  ``1 - \\beta`` under repeated testing.
- The acceptance band must be non-empty and contained in ``[0, \\alpha]``; this is
  enforced by the inner constructor.
"""
struct AdaptiveCalibrationConfig{FT<:AbstractFloat}
    alpha::FT
    alpha_tol::FT
    beta::FT
    n_max::Int
    K_val::Int

    omega_init::FT
    omega_min::FT
    s_init::FT
    s_flat::FT
    step_bounds::Tuple{FT,FT}
    margin_fraction::FT
    n_recovery::Int
    gamma_decay::FT
    min_spread::FT

    function AdaptiveCalibrationConfig{FT}(
        alpha, alpha_tol, beta, n_max, K_val,
        omega_init, omega_min, s_init, s_flat, step_bounds, margin_fraction, n_recovery, gamma_decay, min_spread,
    ) where {FT<:AbstractFloat}
        0 < alpha < 1 || throw(ArgumentError("alpha must lie in (0, 1), got $alpha"))
        0 <= alpha_tol <= alpha ||
            throw(ArgumentError("alpha_tol must lie in [0, alpha], got alpha_tol=$alpha_tol, alpha=$alpha"))
        0 < beta < 1 || throw(ArgumentError("beta must lie in (0, 1), got $beta"))
        n_max >= 1 || throw(ArgumentError("n_max must be at least 1, got $n_max"))
        K_val >= 1 || throw(ArgumentError("K_val must be at least 1, got $K_val"))

        0 < omega_init < 1 ||
            throw(ArgumentError("omega_init must lie in (0, 1), got $omega_init"))
        0 < omega_min < omega_init || throw(ArgumentError(
            "omega_min must lie in (0, omega_init), got omega_min=$omega_min, omega_init=$omega_init"))
        s_init < 0 || throw(ArgumentError(
            "s_init must be negative — a larger risk weight cannot increase the " *
                "violation probability — got $s_init"))
        s_flat < 0 || throw(ArgumentError(
            "s_flat must be negative, got $s_flat"))
        s_init <= s_flat || throw(ArgumentError(
            "s_init must be at least as steep as the cap s_flat, got s_init=$s_init " *
                "and s_flat=$s_flat."))
        0 < step_bounds[1] < step_bounds[2] || throw(ArgumentError(
            "step_bounds must satisfy 0 < lower < upper, got $step_bounds"))
        0 <= margin_fraction < 1 || throw(ArgumentError(
            "margin_fraction must lie in [0, 1) — it is a fraction of the largest " *
                "certifiable violation rate, not a violation count — got $margin_fraction"))
        0 <= n_recovery < n_max || throw(ArgumentError(
            "n_recovery must lie in [0, n_max), got n_recovery=$n_recovery, n_max=$n_max"))
        0 < gamma_decay <= 1 ||
            throw(ArgumentError("gamma_decay must lie in (0, 1], got $gamma_decay"))
        min_spread > 0 || throw(ArgumentError("min_spread must be positive, got $min_spread"))

        # The update steers toward the rate (1 - margin_fraction)·p*_α, where
        # p*_α = V*_α/K_val is the largest empirical violation rate that still
        # certifies at level α. The target rate must be strictly positive, as
        # log(0) is not defined; being relative, it is positive for every
        # admissible margin_fraction as long as V*_α is.
        beta_att = beta / n_max
        V_alpha_star = _largest_violation_count(Int(K_val), beta_att, alpha)
        V_alpha_star >= 0 || throw(ArgumentError(
            "Certification at level α=$alpha is impossible with K_val=$K_val " *
                "validation scenarios at confidence β_att=$beta_att even with zero violations. " *
                "Increase K_val, increase α, or relax β."))
        V_alpha_star >= 1 || throw(ArgumentError(
            "Certification at level α=$alpha with K_val=$K_val validation scenarios at " *
                "confidence β_att=$beta_att admits zero violations only, so there is no " *
                "positive violation rate for the calibration to steer toward. " *
                "Increase K_val, increase α, or relax β."))

        # Acceptance band in violation counts: δ is strictly increasing in V, so
        # the band [α - α_tol, α] is the count interval [V_low, V*_α], where V*_α
        # is the largest count whose bound stays at or below α and V_low the
        # smallest whose bound reaches α - α_tol.
        V_low = _largest_violation_count(Int(K_val), beta_att, alpha - alpha_tol) + 1
        V_low <= V_alpha_star || throw(ArgumentError(
            "The acceptance band [α - α_tol, α] = [$(alpha - alpha_tol), $alpha] contains no " *
                "admissible violation count with K_val=$K_val at β_att=$beta_att: the smallest " *
                "count that is not over-conservative is $V_low, but the largest that still " *
                "certifies is $V_alpha_star. Widen alpha_tol or increase K_val."))

        # The count the law steers toward must lie in that band, otherwise the
        # update aims at a rate the accept test rejects and the search can only
        # land in the band through the noise of the validation draw. The target is
        # kept real-valued: only its logarithm enters the update.
        V_target = (1 - margin_fraction) * V_alpha_star
        if V_target < V_low
            @warn "The update target lies below the acceptance band; the calibration " *
                "aims at a violation rate the accept test rejects. Reduce " *
                "margin_fraction or widen alpha_tol." margin_fraction V_target V_low
        end

        return new{FT}(alpha, alpha_tol, beta, n_max, K_val,
            omega_init, omega_min, s_init, s_flat, step_bounds, margin_fraction, n_recovery,
            gamma_decay, min_spread)
    end
end

"""
    AdaptiveCalibrationConfig(; alpha, alpha_tol = alpha, beta, n_max, K_val,
                                omega_init = 0.5, omega_min = 1e-3,
                                s_init = -1.0, s_flat = -0.1,
                                step_bounds = (log(1.5), log(100.0)),
                                margin_fraction = 0.1, n_recovery = 3,
                                gamma_decay = 0.5, min_spread = 0.1)

Keyword constructor for [`AdaptiveCalibrationConfig`](@ref).

# Keywords
- `alpha::Real`: risk level ``\\alpha`` (required)
- `alpha_tol::Real`: acceptance-band width below ``\\alpha``. Defaults to
   ``\\alpha``, so every level in ``[0, \\alpha]`` is accepted.
- `beta::Real`: overall confidence level ``\\beta`` (required)
- `n_max::Integer`: maximum number of attempts (required)
- `K_val::Integer`: number of validation scenarios per attempt (required)
- `omega_init::Real`: risk weight of the first attempt, in ``(0, 1)``. Defaults
  to `0.5`.
- `omega_min::Real`: lower floor on the risk weight, in ``(0, \\omega_1)``.
  Defaults to `1e-3`.
- `s_init::Real`: initial elasticity ``s_{\\mathrm{init}} < 0``. Defaults to `-1.0`. Must be
  at least as steep as `s_flat`.
- `s_flat::Real`: upper limit on the elasticity, ``s_{\\mathrm{flat}} < 0``. Defaults to
  `-0.1`.
- `step_bounds::Tuple{Real,Real}`: bounds on the magnitude of the step in
  ``x = \\log\\lambda``. Defaults to ``(\\log 1.5, \\log 100)``.
- `margin_fraction::Real`: conservative bias of the update, as a fraction of the
  largest certifiable violation rate ``p^*_\\alpha = V^*_\\alpha / K_{\\mathrm{val}}``.
  Defaults to `0.1`.
- `n_recovery::Integer`: number of final attempts reserved for the recovery
  phase, in ``[0, n_{\\mathrm{max}})``. Defaults to `3`. Set it to `0` to disable the
  phase.
- `gamma_decay::Real`: recency weighting of the elasticity fit. Defaults to
  `0.5`.
- `min_spread::Real`: minimum weighted spread of the abscissae. Defaults to
  `0.1`.

# Returns
- `config::AdaptiveCalibrationConfig`: validated configuration
"""
function AdaptiveCalibrationConfig(;
    alpha::Real,
    alpha_tol::Real=alpha,
    beta::Real,
    n_max::Integer,
    K_val::Integer,
    omega_init::Real=0.5,
    omega_min::Real=1e-3,
    s_init::Real=-1.0,
    s_flat::Real=-0.1,
    step_bounds::Tuple{Real,Real}=(log(1.5), log(100.0)),
    margin_fraction::Real=0.1,
    n_recovery::Integer=3,
    gamma_decay::Real=0.5,
    min_spread::Real=0.1,
)
    FT = float(promote_type(
        typeof(alpha), typeof(alpha_tol), typeof(beta),
        typeof(omega_init), typeof(omega_min), typeof(s_init), typeof(s_flat),
        typeof(step_bounds[1]), typeof(step_bounds[2]),
        typeof(margin_fraction), typeof(gamma_decay), typeof(min_spread),
    ))

    return AdaptiveCalibrationConfig{FT}(
        alpha, alpha_tol, beta, n_max, K_val,
        omega_init, omega_min, s_init, s_flat, (FT(step_bounds[1]), FT(step_bounds[2])),
        margin_fraction, n_recovery, gamma_decay, min_spread,
    )
end

# Float type the calibration arithmetic is carried out in.
_float_type(::AdaptiveCalibrationConfig{FT}) where {FT} = FT

# Float type of the OCP solutions, which may differ from the calibration's.
_ocp_float_type(::CVaRScenarioWorkspace{FT}) where {FT} = FT

# -----------------------------------------------------------------------------
# Per-cycle history
# -----------------------------------------------------------------------------

"""
    CalibrationHistory{FT}

Within-cycle record of the failed calibration attempts, in the coordinates the
elasticity is fitted in.

One calibration cycle is one call to [`plan!`](@ref). The history is preallocated
to `n_max` entries at planner construction and reset at the start of every cycle,
so a receding-horizon loop reuses the same storage without accumulating points
across time steps — the response curve moves as the posterior and the initial
condition change, and stale points would bias the fit.

# Fields
- `x::Vector{FT}`: abscissae ``x_j = \\mathrm{logit}(\\omega_j) = \\log\\lambda_j``
- `y::Vector{FT}`: ordinates ``y_j = \\log(V_j / K_{\\mathrm{val}})``, the log empirical
  violation rate
- `n::Int`: number of valid entries; `x` and `y` are only meaningful up to `n`
- `s_init::FT`: initial elasticity of the current cycle, used until the cycle
  has produced a well-conditioned fit; set by [`plan!`](@ref)
- `last_slope::FT`: elasticity applied by the most recent update of this cycle,
  or `NaN` before the first one. An ill-conditioned fit is not evidence that the
  response changed, so the update reuses this value rather than discarding what
  the cycle has already measured.
"""
mutable struct CalibrationHistory{FT<:AbstractFloat}
    x::Vector{FT}
    y::Vector{FT}
    n::Int
    s_init::FT
    last_slope::FT
end

"""
    CalibrationHistory{FT}(capacity::Int, s_init::Real)

Allocate an empty history with room for `capacity` attempts and initial
elasticity `s_init`.
"""
function CalibrationHistory{FT}(capacity::Int, s_init::Real) where {FT<:AbstractFloat}
    capacity >= 1 || throw(ArgumentError("capacity must be at least 1, got $capacity"))
    return CalibrationHistory{FT}(
        Vector{FT}(undef, capacity), Vector{FT}(undef, capacity), 0, FT(s_init), FT(NaN))
end

"""
    reset!(history::CalibrationHistory)

Discard all recorded attempts without freeing the storage.
"""
function reset!(history::CalibrationHistory{FT}) where {FT}
    history.n = 0
    history.last_slope = FT(NaN)
    return nothing
end

"""
    record!(history::CalibrationHistory, x, y)

Append one attempt to `history`.

# Arguments
- `history::CalibrationHistory`: history of the current cycle
- `x::Real`: abscissa ``\\mathrm{logit}(\\omega_j)`` of the attempt
- `y::Real`: ordinate ``\\log(V_j / K_{\\mathrm{val}})`` of the attempt
"""
function record!(history::CalibrationHistory{FT}, x::Real, y::Real) where {FT}
    n = history.n + 1
    n <= length(history.x) || throw(BoundsError(history.x, n))
    @inbounds history.x[n] = x
    @inbounds history.y[n] = y
    history.n = n
    return nothing
end

# -----------------------------------------------------------------------------
# Planner
# -----------------------------------------------------------------------------

"""
    AdaptiveCVaRPlanner{P,E,FT} <: AbstractPlanner

Closed-loop planner that produces a certified input trajectory by adaptively
calibrating the CVaR risk weight ``\\omega``.

The planner bundles the components of the synthesis–validation loop:

- a CVaR-penalized scenario OCP,
- a warm MCMC ensemble that supplies fresh validation sets,
- the calibration configuration, and
- the per-cycle history the elasticity is estimated from.

Calling [`plan!`](@ref) repeatedly solves the OCP for the current ``\\omega``,
certifies the candidate on a freshly drawn validation set via [`clopper_pearson_bound`](@ref),
and accepts as soon as the certified bound ``\\delta`` enters the acceptance band
``[\\alpha - \\alpha_{\\mathrm{tol}}, \\alpha]``; otherwise ``\\omega`` is moved toward the
target violation rate and the OCP is resolved.

# Fields
- `ocp_workspace::P`: CVaR scenario OCP workspace ([`CVaRScenarioWorkspace`](@ref)) holding
  the training scenarios. Its ``\\omega`` parameter is updated in place across
  attempts while the training scenarios are held fixed.
- `ensemble::E`: warm MCMC chains ([`MCMCEnsemble`](@ref)) continued in place
  to draw a fresh, independent validation set of `config.K_val` scenarios
  each attempt
- `sampler_config::MCMCConfig`: chain dynamics (thinning, burn-in, …) used when
  drawing validation samples
- `config::AdaptiveCalibrationConfig{FT}`: calibration configuration
- `history::CalibrationHistory{FT}`: preallocated per-cycle history, reset by
  every call to [`plan!`](@ref)

# Notes
- The Clopper–Pearson certification requires the validation scenarios to be drawn
  independently of the training scenarios baked into `ocp_workspace`; the validation set must
  not reuse or overlap with them. This statistical independence is unrelated to the
  number of chains in `ensemble`, which does not need to match `config.K_val`.
"""
struct AdaptiveCVaRPlanner{P<:CVaRScenarioWorkspace,E<:MCMCEnsemble,FT<:AbstractFloat} <: AbstractPlanner
    ocp_workspace::P
    ensemble::E
    sampler_config::MCMCConfig
    config::AdaptiveCalibrationConfig{FT}
    history::CalibrationHistory{FT}
end

"""
    AdaptiveCVaRPlanner(ocp_workspace, ensemble, sampler_config, config)

Construct an [`AdaptiveCVaRPlanner`](@ref), allocating the per-cycle history at
the capacity implied by `config.n_max`.
"""
function AdaptiveCVaRPlanner(
    ocp_workspace::CVaRScenarioWorkspace,
    ensemble::MCMCEnsemble,
    sampler_config::MCMCConfig,
    config::AdaptiveCalibrationConfig{FT},
) where {FT}
    return AdaptiveCVaRPlanner(
        ocp_workspace, ensemble, sampler_config, config,
        CalibrationHistory{FT}(config.n_max, config.s_init),
    )
end

# -----------------------------------------------------------------------------
# Result
# -----------------------------------------------------------------------------

"""
    CalibrationStep

Diagnostics of one risk-weight update, recorded per attempt by [`plan!`](@ref).

Both the raw and the guarded quantities are kept: a run in which the ceiling
binds often indicates that the elasticity estimate is unreliable or that `s_init`
is badly off, which cannot be seen from the post-clip values alone.

# Fields
- `omega::Float64`: risk weight ``\\omega_j`` the attempt was solved at
- `slope_raw::Float64`: elasticity from the weighted fit, `NaN` when fewer than
  two points were available
- `slope::Float64`: elasticity actually used — the fit, its cap at `s_flat`, the
  one last applied, or `s_init` (see `slope_source`)
- `spread::Float64`: weighted spread of the abscissae the fit was based on
- `step_raw::Float64`: secant step ``\\Delta`` before clipping; negative when the
  attempt overshot into over-conservatism
- `step::Float64`: step actually taken in ``x``: the secant step with its
  magnitude clipped to `step_bounds` and the sign of `step_raw` preserved, the
  largest admissible step after a saturated attempt, or the step to the bracket
  midpoint (see `clip`)
- `omega_lo::Float64`: lower end of the bracket, a risk weight with too many
  violations (see `bracket_source`); `NaN` if no usable bracket exists
- `omega_hi::Float64`: upper end of the bracket, a risk weight with too few
  violations (see `bracket_source`); `NaN` if no usable bracket exists
- `bracket_source::Symbol`: which attempts the bracket was built from —
  `:extremes` (the largest ω with too many and the smallest with too few
  violations), `:recent` (the most recent attempt on each side, used when the
  extremes are inconsistent), or `:none` (no usable bracket)
- `omega_secant::Float64`: risk weight proposed by the secant step, after
  bracketing
- `omega_recovery::Float64`: risk weight proposed by the recovery phase; equal to
  `omega` while the phase is inactive
- `recovery_active::Bool`: whether the recovery phase was in force
- `omega_next::Float64`: risk weight of the following attempt
- `slope_source::Symbol`: where the applied elasticity came from — `:measured`,
  `:capped`, `:reused`, or `:init`
- `clip::Symbol`: which rule set the step — `:none` (the plain secant step),
  `:floor` or `:ceiling` (its magnitude was clipped), `:saturated` (the attempt
  was saturated and the largest admissible step was taken), or `:bisected` (the
  step would not have landed at least `step_bounds[1]` inside the bracket, and
  its midpoint was taken instead)
"""
struct CalibrationStep
    omega::Float64
    slope_raw::Float64
    slope::Float64
    spread::Float64
    step_raw::Float64
    step::Float64
    omega_lo::Float64
    omega_hi::Float64
    bracket_source::Symbol
    omega_secant::Float64
    omega_recovery::Float64
    recovery_active::Bool
    omega_next::Float64
    slope_source::Symbol
    clip::Symbol
end

"""
    AdaptiveCVaRResult{FT,S}

Result of an adaptive calibration run produced by [`plan!`](@ref).

# Fields
- `solution::OCPSolution{FT}`: the returned candidate solution, however the
  calibration ended: the certified candidate with the lowest validation cost if
  any attempt met ``\\delta \\leq \\alpha``, otherwise the one with the smallest
  observed ``\\delta``.
- `delta::Float64`: certified violation-probability bound ``\\delta`` of `solution`
  (computed at confidence ``\\beta_{\\mathrm{att}} = \\beta / n_{\\mathrm{max}}``)
- `beta::Float64`: overall confidence level ``\\beta`` the bound was computed
  against
- `omega::Float64`: risk weight ``\\omega`` that produced `solution`
- `cost::Float64`: sample-average cost of `solution` on its validation set
- `certified::Bool`: `true` if the returned solution meets the hard requirement
  ``\\delta \\leq \\alpha``; `false` otherwise.
- `attempts::Int`: number of synthesis–validation attempts performed
- `termination::Symbol`: how the calibration ended — `:in_band` (an attempt
  landed in the acceptance band), `:floor` (an attempt at
  ``\\omega \\leq \\omega_{\\mathrm{min}}`` was below the band), or `:exhausted` (all
  `n_max` attempts were used without landing in the band; `certified` tells
  whether the returned candidate meets ``\\delta \\leq \\alpha``)
- `runtime_optimization::Float64`: total wall-clock time (s) spent in the OCP solves
  (`solve_ocp!`) across all attempts
- `runtime_validation::Float64`: total wall-clock time (s) spent drawing validation
  scenarios, counting violations, and evaluating the validation cost across all
  attempts
- `delta_history::Vector{Float64}`: the bound ``\\delta`` observed at each attempt
- `violation_history::Vector{Int}`: the validation violation count ``V`` at each
  attempt
- `omega_history::Vector{Float64}`: the risk weight ``\\omega`` used at each
  attempt
- `cost_history::Vector{Float64}`: the sample-average validation cost of the
  candidate at each attempt
- `update_history::Vector{CalibrationStep}`: diagnostics of each risk-weight
  update
- `U_history::Vector{Matrix{FT}}`: the candidate input trajectory synthesized at
  each attempt, e.g., to compare the first and the returned candidate
- `validation_scenarios::Vector{S}`: the `K_val` validation scenarios drawn in the
  last attempt. They are drawn independently of every candidate, unlike the
  training scenarios the candidates were optimized on, and so give an unbiased
  view of a candidate's predictive distribution.
"""
struct AdaptiveCVaRResult{FT<:AbstractFloat,S<:AbstractPosteriorSample}
    solution::OCPSolution{FT}
    delta::Float64
    beta::Float64
    omega::Float64
    cost::Float64
    certified::Bool
    attempts::Int
    termination::Symbol
    runtime_optimization::Float64
    runtime_validation::Float64
    delta_history::Vector{Float64}
    violation_history::Vector{Int}
    omega_history::Vector{Float64}
    cost_history::Vector{Float64}
    update_history::Vector{CalibrationStep}
    U_history::Vector{Matrix{FT}}
    validation_scenarios::Vector{S}
end

# -----------------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------------

"""
    _logit(omega)

Map a risk weight ``\\omega \\in [0, 1]`` to the search coordinate
``x = \\log(\\omega / (1 - \\omega)) = \\log\\lambda``.

Evaluated as ``\\log\\omega - \\log(1-\\omega)`` with `log1p`, which stays
accurate as either endpoint is approached and returns ``\\mp\\infty`` there.
"""
@inline function _logit(omega::FT) where {FT<:AbstractFloat}
    return log(omega) - log1p(-omega)
end

"""
    _logistic(x)

Inverse of [`_logit`](@ref): map ``x = \\log\\lambda`` back to
``\\omega = 1/(1 + e^{-x})``.
"""
@inline function _logistic(x::FT) where {FT<:AbstractFloat}
    if x >= zero(FT)
        return one(FT) / (one(FT) + exp(-x))
    else
        e = exp(x)
        return e / (one(FT) + e)
    end
end

"""
    _largest_violation_count(K_val, beta, level)

Return the largest violation count ``v \\in \\{0, \\dots, K_{\\mathrm{val}}\\}`` for which the
Clopper–Pearson bound satisfies ``\\delta_v \\leq \\mathrm{level}``, or `-1` if even
``v = 0`` exceeds `level`.
"""
function _largest_violation_count(K_val::Int, beta::Real, level::Real)
    # If even v = 0 exceeds level, no v satisfies the bound.
    clopper_pearson_bound(0, K_val, beta) <= level || return -1

    # Binary search for the largest v with δ_v ≤ level.
    lo, hi = 0, K_val
    while lo < hi
        mid = lo + (hi - lo + 1) ÷ 2
        if clopper_pearson_bound(mid, K_val, beta) <= level
            lo = mid
        else
            hi = mid - 1
        end
    end
    return lo
end

# -----------------------------------------------------------------------------
# Update law
# -----------------------------------------------------------------------------

"""
    _is_informative(y)

Return `true` if an attempt with log violation rate `y` carries information about
the elasticity. A saturated attempt (V = K_val, y = 0) lies on the upper plateau
of the response, and an attempt without violations (V = 0, y = -Inf) has no finite
log rate; neither constrains the local slope.
"""
@inline function _is_informative(y::Real)
    return isfinite(y) && y < zero(y)
end

"""
    _fit_elasticity(history, gamma_decay)

Estimate the elasticity ``s = \\mathrm{d}\\log p / \\mathrm{d}\\log\\lambda`` by
exponentially weighted least squares over `history`, with weights
``\\gamma^{n-i}`` that put unit weight on the most recent attempt.

The log–log response flattens as ``\\omega \\to 1``, because the violation rate
approaches a floor set by the posterior spread. Old points therefore describe a
steeper regime than the current one, which argues for locality. Against that, the
sampling noise of a single point is ``\\mathrm{sd}(\\log\\hat{p}) \\approx
1/\\sqrt{V}``, and a two-point difference divides that noise by whatever step
happened to be taken, so short windows are unreliable. Exponential weights nest
both extremes — ``\\gamma = 1`` is a global fit, ``\\gamma \\to 0`` the plain
secant — and interpolate smoothly between them.

# Arguments
- `history::CalibrationHistory`: recorded attempts of the current cycle
- `gamma_decay::FT`: recency weighting ``\\gamma \\in (0, 1]``

# Returns
- `(slope, spread)`: the fitted elasticity and the weighted standard deviation of
  the abscissae. With fewer than two points, `(NaN, 0)` is returned.

# Notes
- The moments are accumulated in two passes about the weighted mean; `x` spans
  several units, so the uncentred form would lose precision to cancellation.
"""
function _fit_elasticity(history::CalibrationHistory{FT}, gamma_decay::FT) where {FT<:AbstractFloat}
    n = history.n
    x = history.x
    y = history.y

    # Pass 1: weighted means over the informative attempts. Weights keep their
    # position in the history, so recency is measured in attempts, including the
    # skipped ones.
    W = zero(FT)
    sum_x = zero(FT)
    sum_y = zero(FT)
    n_valid = 0
    @inbounds for i in 1:n
        if !_is_informative(y[i])
            continue
        end
        w = gamma_decay^(n - i)
        W += w
        sum_x += w * x[i]
        sum_y += w * y[i]
        n_valid += 1
    end
    if n_valid < 2
        return (slope=FT(NaN), spread=zero(FT))
    end
    x_bar = sum_x / W
    y_bar = sum_y / W

    # Pass 2: mean-centred weighted second moments over the same attempts.
    S_xx = zero(FT)
    S_xy = zero(FT)
    @inbounds for i in 1:n
        if !_is_informative(y[i])
            continue
        end
        w = gamma_decay^(n - i)
        dx = x[i] - x_bar
        S_xx += w * dx * dx
        S_xy += w * dx * (y[i] - y_bar)
    end

    return (slope=S_xy / S_xx, spread=sqrt(S_xx / W))
end

"""
    _is_conditioned(slope, spread, config)

Return `true` if a fit carries information about the elasticity, i.e., if it is
finite and rests on abscissae spread widely enough to identify a slope.
"""
function _is_conditioned(slope::Real, spread::Real, config::AdaptiveCalibrationConfig)
    return isfinite(slope) && spread >= config.min_spread
end

"""
    _select_elasticity(slope, spread, last_slope, s_init, config)

Return the elasticity to step with, given a fit `(slope, spread)`, the
`last_slope` measured earlier in this cycle (`NaN` if there is none), and the
initial elasticity `s_init` of this cycle.
"""
function _select_elasticity(slope::FT, spread::FT, last_slope::FT, s_init::FT, config::AdaptiveCalibrationConfig{FT}) where {FT}
    if !_is_conditioned(slope, spread, config)
        if isnan(last_slope)
            return s_init
        else
            return last_slope
        end
    else
        return min(slope, config.s_flat)
    end
end

"""
    _slope_source(slope_raw, spread, last_slope, config)

Classify where the elasticity [`_select_elasticity`](@ref) returned came from,
for logging:

- `:measured` — a well-conditioned fit, used as it came out
- `:capped` — a well-conditioned fit, shallower than `s_flat` and capped there
- `:reused` — an ill-conditioned fit; the elasticity last applied was kept
- `:init` — an ill-conditioned fit on the first update, so `s_init` was used
"""
function _slope_source(slope_raw::Real, spread::Real, last_slope::Real, config::AdaptiveCalibrationConfig)
    if !_is_conditioned(slope_raw, spread, config)
        if isnan(last_slope)
            return :init
        else
            return :reused
        end
    elseif slope_raw > config.s_flat
        return :capped
    else
        return :measured
    end
end

"""
    _clip_reason(step_raw, config)

Classify which bound limited the magnitude of the secant step, for logging:
`:none`, `:floor`, or `:ceiling`.
"""
function _clip_reason(step_raw::Real, config::AdaptiveCalibrationConfig)
    if abs(step_raw) < config.step_bounds[1]
        return :floor
    elseif abs(step_raw) > config.step_bounds[2]
        return :ceiling
    else
        return :none
    end
end

"""
    _bracket(history, y_star)

Bracket of the target in ``x = \\log\\lambda`` from the attempts recorded in
`history`.

The lower end ``x_{\\mathrm{lo}}`` is the largest abscissa of an attempt with too many
violations (``y_i > y^*``), the upper end ``x_{\\mathrm{hi}}`` the smallest abscissa of an
attempt with too few (``y_i < y^*``). Saturated attempts (``V = K_{\\mathrm{val}}``) and
attempts without violations (recorded at ``V = 1/2``) take part: they say on which
side of the target they lie, which is all the bracket needs.

# Arguments
- `history::CalibrationHistory`: recorded attempts of the current cycle
- `y_star::Real`: target log violation rate

# Returns
- `(x_lo, x_hi, valid, source)`: the bracket ends, whether the bracket is usable,
  i.e., two-sided and consistent (``x_{\\mathrm{lo}} < x_{\\mathrm{hi}}``), and which attempts it was
  built from — `:extremes`, `:recent`, or `:none` if it is not usable.

# Behavior
Validation noise near the target can produce an attempt with too many violations
above one with too few, which makes the extremes inconsistent. The bracket then
falls back to the most recent attempt on each side: these reflect the current
solver branch and the latest validation draws, so after a single contradictory
point they usually form a consistent bracket again. Only if they are
inconsistent too is the bracket reported as unusable.
"""
function _bracket(history::CalibrationHistory{FT}, y_star::FT) where {FT<:AbstractFloat}
    # Extremes: the tightest bracket the history supports.
    x_lo = FT(-Inf)
    x_hi = FT(Inf)
    # Most recent attempt on each side; later entries overwrite earlier ones.
    x_lo_recent = FT(-Inf)
    x_hi_recent = FT(Inf)
    @inbounds for i in 1:history.n
        if history.y[i] > y_star
            x_lo = max(x_lo, history.x[i])
            x_lo_recent = history.x[i]
        elseif history.y[i] < y_star
            x_hi = min(x_hi, history.x[i])
            x_hi_recent = history.x[i]
        end
    end

    # A side without any attempt leaves the bracket open; the recent ends are
    # then open on the same side, so no fallback can help.
    if !(isfinite(x_lo) && isfinite(x_hi))
        return (x_lo=x_lo, x_hi=x_hi, valid=false, source=:none)
    end
    if x_lo < x_hi
        return (x_lo=x_lo, x_hi=x_hi, valid=true, source=:extremes)
    end
    if x_lo_recent < x_hi_recent
        return (x_lo=x_lo_recent, x_hi=x_hi_recent, valid=true, source=:recent)
    end
    return (x_lo=x_lo, x_hi=x_hi, valid=false, source=:none)
end

"""
    _update_risk_weight(history, omega, y_star, attempt, config, certified)

Compute the risk weight of the next attempt from the recorded history.

The secant candidate is ``\\omega = \\sigma(x_j + \\Delta)`` with
``\\Delta = (y^* - y_j)/\\hat{s}``, whose magnitude is clipped to
`config.step_bounds` while its sign is left alone: the step raises ``\\omega``
after too many violations (``y_j > y^*``) and lowers it after an overshoot into
over-conservatism (``y_j < y^*``).

Once the history holds attempts on both sides of the target, the secant
candidate is confined to the bracket between them: a step that would not land
at least the step floor ``\\Delta_{\\mathrm{min}}`` inside ``(x_{\\mathrm{lo}}, x_{\\mathrm{hi}})`` is
replaced by the step to the bracket midpoint (see [`_bracket`](@ref)).

While no candidate has certified, a recovery candidate
``\\omega_j + (1 - \\omega_j)/r_j`` with ``r_j = n_{\\mathrm{max}} - j`` remaining attempts
is formed over the last `config.n_recovery` attempts, and the more conservative of
the two is taken. Once `certified` is true the phase is switched off, so that an
overshoot can still be corrected downward in the endgame; `n_recovery = 0`
switches it off.

# Arguments
- `history::CalibrationHistory`: recorded attempts, the last of which is the
  attempt being updated from
- `omega::Real`: risk weight ``\\omega_j`` of that attempt
- `y_star::Real`: target log violation rate ``\\log((1 - m)V^*_\\alpha/K_{\\mathrm{val}})``
- `attempt::Int`: index ``j`` of that attempt, which must be below `n_max`
- `config::AdaptiveCalibrationConfig`: calibration configuration
- `certified::Bool`: whether any attempt of this cycle has met
  ``\\delta \\leq \\alpha`` so far

# Returns
- `step::CalibrationStep`: the risk weight of the next attempt together with
  everything the decision rested on — both candidates, the raw and guarded
  elasticity and step, and the labels [`_slope_source`](@ref) and
  [`_clip_reason`](@ref) classifying which rule fired.

# Notes
- The secant candidate is floored at `config.omega_min` and capped strictly
  below 1, so that its logit stays finite and the next attempt can be recorded. Only the recovery phase
  reaches ``\\omega = 1`` exactly, and only on its last step, which is followed by
  attempt `n_max`, after which no update follows.
- The elasticity applied is written back to `history.last_slope`, so the caller
  passes the same history to the next update and nothing else needs carrying.
- Allocation-free: the returned fields are floats, booleans, and interned
  symbols, and the fit reads the preallocated history in place.
- The bracket never interacts with the recovery phase: an attempt with too few
  violations (``y_j < y^*``) has ``V_j`` below the target, hence certifies and
  switches the phase off, so a two-sided bracket implies an inactive phase.
"""
function _update_risk_weight(
    history::CalibrationHistory{FT},
    omega::FT,
    y_star::FT,
    attempt::Int,
    config::AdaptiveCalibrationConfig{FT},
    certified::Bool=false,
) where {FT<:AbstractFloat}
    @assert history.n >= 1 "the failed attempt must be recorded before its update"
    @assert zero(FT) < omega < one(FT) "ω must lie strictly inside (0, 1) to be updated"
    @assert attempt < config.n_max "no update follows the final attempt"

    fit = _fit_elasticity(history, config.gamma_decay)
    previous_slope = history.last_slope
    slope = _select_elasticity(fit.slope, fit.spread, previous_slope, history.s_init, config)
    history.last_slope = slope

    @inbounds x_j = history.x[history.n]
    @inbounds y_j = history.y[history.n]

    # Secant step in x = log λ toward the target rate y*. Its sign is the sign of
    # the miss: with ŝ < 0 guaranteed by the s_flat cap, too many violations (y_j > y*)
    # give Δ > 0 and an overshoot into over-conservatism (y_j < y*) gives Δ < 0.
    # Only the magnitude is guarded as both the noise argument behind the floor and
    # the runaway argument behind the ceiling are direction-symmetric. 
    # A saturated attempt (V = K_val, y_j = 0) is censored: it shows that the transition 
    # lies further up but not how far, so the model step has no meaning there and the 
    # search moves up by the largest admissible step instead.
    step_raw = (y_star - y_j) / slope
    saturated = y_j >= zero(FT)
    if saturated
        step = config.step_bounds[2]
        clip = :saturated
    else
        magnitude = clamp(abs(step_raw), config.step_bounds[1], config.step_bounds[2])
        if step_raw > zero(FT)
            step = magnitude
        else
            step = -magnitude
        end
        clip = _clip_reason(step_raw, config)
    end

    # Bracketing: once attempts on both sides of the target are known, the target
    # lies between them (the response is decreasing in x). A step that would not
    # land at least Δ_min inside the bracket is replaced by the step to its
    # midpoint. A point closer to an end than the step floor is indistinguishable
    # from that end, and the margin also absorbs the round-off of the logit/logistic
    # round trip, which would otherwise let a step back onto an end pass a strict
    # test. This breaks the two-point cycle across a steep transition, where V = 0
    # below and V = K_val above leave the fit unchanged and steps of equal magnitude
    # would alternate between the same two risk weights.
    bracket = _bracket(history, y_star)
    if bracket.valid
        x_secant = x_j + step
        margin = config.step_bounds[1]
        if !(bracket.x_lo + margin <= x_secant <= bracket.x_hi - margin)
            step = (bracket.x_lo + bracket.x_hi) / 2 - x_j
            clip = :bisected
        end
        omega_lo = Float64(_logistic(bracket.x_lo))
        omega_hi = Float64(_logistic(bracket.x_hi))
    else
        omega_lo = NaN
        omega_hi = NaN
    end

    # Keep the secant inside [ω_min, 1): logit(1) is infinite, and an attempt at
    # ω = 1 could then not be recorded. The upper bound is unreachable in practice
    # and only guards against a run of clipped steps. The floor stops the search
    # from chasing ω → 0 where the response is flat; `plan!` terminates on an
    # over-conservative attempt made at the floor.
    omega_secant = clamp(_logistic(x_j + step), config.omega_min, prevfloat(one(FT)))

    # Recovery ramp over the last n_recovery attempts, while nothing has certified.
    # The residual contracts as (1-ω)·r_{j+1}/r_j, so ω = 1 is reached exactly at
    # attempt n_max. It is defined in ω because ω = 1 sits at x = ∞, which no
    # finite step reaches.
    remaining = config.n_max - attempt
    recovery_active = !certified && remaining <= config.n_recovery
    if recovery_active
        omega_recovery = omega + (one(FT) - omega) / remaining
        omega_next = max(omega_secant, omega_recovery)
    else
        # Reported for diagnostics, not applied.
        omega_recovery = omega
        omega_next = omega_secant
    end

    # ω = 1 is admissible only on the ramp's final step, which lands on it exactly
    # and is followed by attempt n_max, where no update is performed. Everywhere
    # else ω stays strictly inside (0, 1) so that the attempt can be recorded.
    if recovery_active && remaining == 1
        omega_max = one(FT)
    else
        omega_max = prevfloat(one(FT))
    end

    return CalibrationStep(
        omega,
        fit.slope,
        slope,
        fit.spread,
        step_raw,
        step,
        omega_lo,
        omega_hi,
        bracket.source,
        omega_secant,
        omega_recovery,
        recovery_active,
        clamp(omega_next, config.omega_min, omega_max),
        _slope_source(fit.slope, fit.spread, previous_slope, config),
        clip,
    )
end

"""
    _is_better_candidate(delta, cost, best_delta, best_cost, alpha)

Return `true` if the candidate with bound `delta` and validation cost `cost`
should replace the incumbent `(best_delta, best_cost)`.

Certification comes first: any candidate meeting ``\\delta \\leq \\alpha`` beats
one that does not, whatever their costs. Among certified candidates the lowest
validation cost wins. Among uncertified candidates the smallest ``\\delta`` wins, 
as the one closest to meeting the requirement.

# Notes
- Selecting freely among the attempts needs no further correction: the Bonferroni
  split ``\\beta_{\\mathrm{att}} = \\beta / n_{\\mathrm{max}}`` already covers all `n_max` of them.
- Each candidate's cost is the sample average over its own validation set. If
  the cost depends on states or outputs, these averages carry Monte-Carlo
  error of order ``\\mathrm{sd}(J)/\\sqrt{K_{\\mathrm{val}}}``, so two candidates whose
  expected costs differ by less than that are ranked essentially at random.
"""
function _is_better_candidate(delta::Real, cost::Real, best_delta::Real, best_cost::Real, alpha::Real)
    new_certified = delta <= alpha
    best_certified = best_delta <= alpha
    if new_certified != best_certified
        return new_certified
    elseif new_certified
        return cost < best_cost
    else
        return delta < best_delta
    end
end

# -----------------------------------------------------------------------------
# Adaptive calibration loop
# -----------------------------------------------------------------------------

"""
    plan!(planner::AdaptiveCVaRPlanner, model, data;
          U_init=nothing, omega_init=nothing, s_init=nothing,
          initial_validation_scenarios=nothing)

Run the adaptive calibration of the CVaR risk weight ``\\omega`` and
return an [`AdaptiveCVaRResult`](@ref).

Each attempt solves the planner's CVaR-penalized scenario OCP for the current
``\\omega``, draws a fresh validation set by continuing the planner's MCMC ensemble,
counts constraint violations, and certifies the candidate via the Clopper–Pearson
bound at confidence ``\\beta_{\\mathrm{att}} = \\beta / n_{\\mathrm{max}}``. The loop stops as soon as
a bound ``\\delta`` enters the acceptance band ``[\\alpha - \\alpha_{\\mathrm{tol}}, \\alpha]``;
otherwise ``\\omega`` is moved toward the target violation rate — up after too many
violations, down after an overshoot into over-conservatism — and the OCP is resolved.

The training scenarios stored in `planner.ocp_workspace` are not redrawn between
attempts, only ``\\omega`` changes. Solves are warm-started from the best
candidate so far, using the warm-start solver options of the workspace's
[`JuMPSolverConfig`](@ref) from the second attempt on. The first attempt is
cold-started from `U_init`. The planner's MCMC ensemble is advanced in place to
produce a fresh, independent validation set of `config.K_val` scenarios each attempt,
except that the first attempt may reuse `initial_validation_scenarios` (see below).

# Arguments
- `planner::AdaptiveCVaRPlanner`: planner bundling the OCP engine, the validation
  sampler, and the calibration configuration. Its OCP ``\\omega``, its validation
  chains, and its calibration history are updated in place.
- `model::AbstractModel`: prediction/inference model.
- `data::AbstractData`: data defining the posterior the validation ensemble draws
  from (typically the training data).

# Keywords
- `U_init::Union{Nothing,AbstractMatrix}`: optional initial guess for the input
  trajectory of the first solve, size `(n_u, H + 1)`. This solve uses the
  cold-start solver options.
- `omega_init::Union{Nothing,Real}`: optional override of `config.omega_init` for
  the risk weight ``\\omega_1`` of the first attempt, in ``(0, 1)``.
- `s_init::Union{Nothing,Real}`: optional override of `config.s_init` for the
  initial elasticity of this cycle, with ``s_{\\mathrm{init}} \\leq s_{\\mathrm{flat}} < 0``.
- `initial_validation_scenarios::Union{Nothing,AbstractVector{<:AbstractPosteriorSample}}`:
  optional validation set of the first attempt, of at least `config.K_val`
  scenarios (the first `config.K_val` are used). If `nothing`, the first attempt
  draws its set by continuing the ensemble like every later attempt.

`U_init`, `omega_init`, and `s_init` are per-cycle warm starts, e.g., carried over
from the previous cycle of a receding-horizon loop; the configuration supplies
the cold-start defaults.

# Notes
`initial_validation_scenarios` lets the first attempt use valid samples that are
already available, typically the first draws after the warm-up, instead of drawing
a new set. These samples must be the most recent draws of `planner.ensemble`,
since later attempts continue the chains from their current states.

# Returns
- `result::AdaptiveCVaRResult`: the best candidate (see
  [`_is_better_candidate`](@ref)), whether it was successfully certified, and
  further diagnostic data.

# Behavior
The loop runs for at most `n_max` attempts and terminates in one of three ways:

- **In band.** As soon as an attempt's bound ``\\delta`` lands in the acceptance
  band ``[\\alpha - \\alpha_{\\mathrm{tol}}, \\alpha]``, the loop stops and the best
  candidate is returned with `certified = true`. This is the certified candidate
  with the lowest validation cost, which is the in-band attempt itself unless an
  earlier certified attempt (below the band) was cheaper.
- **Floor reached.** If an attempt at ``\\omega \\leq \\omega_{\\mathrm{min}}`` is below the
  band (certified, but more conservative than requested), the cycle stops: the
  cost-dominated end of the family already over-satisfies the constraints and
  no smaller ``\\omega`` is admissible. The best candidate is returned with
  `certified = true`.
- **Exhausted.** If no attempt lands in the band within `n_max` tries, the best
  candidate seen so far is returned as a fallback, where "best" is decided in two
  tiers:
    - If at least one attempt satisfied the hard requirement ``\\delta \\leq
      \\alpha`` (i.e., is certified but fell short of the band, being more
      conservative than required), the returned candidate is the certified one
      with the lowest validation cost, and `certified = true` (see
      [`_is_better_candidate`](@ref)).
    - If no attempt was certified (every ``\\delta > \\alpha``), the returned
      candidate is the one with the smallest ``\\delta``, i.e., closest to meeting the
      requirement, and `certified = false`.
"""
function plan!(
    planner::AdaptiveCVaRPlanner,
    model::AbstractModel,
    data::AbstractData;
    U_init::Union{Nothing,AbstractMatrix}=nothing,
    omega_init::Union{Nothing,Real}=nothing,
    s_init::Union{Nothing,Real}=nothing,
    initial_validation_scenarios::Union{Nothing,AbstractVector{<:AbstractPosteriorSample}}=nothing,
)
    config = planner.config
    workspace = planner.ocp_workspace
    ensemble = planner.ensemble
    sampler_config = planner.sampler_config
    history = planner.history
    FT = _float_type(config)

    problem = workspace.formulation.problem
    scenario_constraints = problem.scenario_constraints
    output_mode = problem.output_mode

    # Bonferroni correction: split the confidence uniformly across attempts.
    beta_att = config.beta / config.n_max

    # Acceptance band in violation counts, [V_low, V*_α], and the count the update
    # steers toward. The margin is the conservative bias of the law: it aims a
    # fraction inside the acceptance region rather than at its boundary. The
    # configuration guarantees a non-empty band and a positive target.
    V_alpha_star = _largest_violation_count(config.K_val, beta_att, config.alpha)
    V_low = _largest_violation_count(config.K_val, beta_att, config.alpha - config.alpha_tol) + 1
    V_target = (one(FT) - config.margin_fraction) * V_alpha_star
    y_star = log(V_target / FT(config.K_val))

    # ω₁: the configured start, or the supplied override.
    if omega_init === nothing
        omega = FT(config.omega_init)
    else
        omega = FT(omega_init)
        0 < omega < 1 || throw(ArgumentError("omega_init must lie in (0, 1), got $omega_init"))
    end

    # Initial elasticity: the configured value, or the supplied override.
    if s_init === nothing
        s_start = FT(config.s_init)
    else
        s_start = FT(s_init)
        s_start <= config.s_flat || throw(ArgumentError(
            "s_init must be at least as steep as s_flat < 0, got s_init=$s_init, s_flat=$(config.s_flat)"))
    end

    if initial_validation_scenarios !== nothing
        length(initial_validation_scenarios) >= config.K_val || throw(ArgumentError(
            "initial_validation_scenarios must hold at least K_val = $(config.K_val) scenarios, " *
            "got $(length(initial_validation_scenarios))"))
    end

    # Every call is an independent calibration cycle.
    reset!(history)
    history.s_init = s_start

    # Best candidate observed so far; returned if no attempt lands in the acceptance band.
    best_solution = nothing
    best_delta = 1.0
    best_omega = Float64(omega)
    best_cost = Inf

    # Per-attempt history
    delta_history = Float64[]
    violation_history = Int[]
    omega_history = Float64[]
    cost_history = Float64[]
    update_history = CalibrationStep[]
    U_history = Matrix{_ocp_float_type(workspace)}[]

    # Wall-clock time
    runtime_optimization = 0.0
    runtime_validation = 0.0

    U_warm = U_init
    attempts = 0

    # Validation set of the latest attempt, returned for out-of-sample prediction.
    last_val_scenarios = nothing

    println("━ Adaptive CVaR Calibration " * "━"^51)
    @printf("  α = %.2f%%, band = [%.2f%%, %.2f%%], β = %.2f%% (β_att = %.4f%%)\n",
        config.alpha * 100, (config.alpha - config.alpha_tol) * 100, config.alpha * 100,
        config.beta * 100, beta_att * 100)
    @printf("  K_val = %d, accepted V ∈ [%d, %d], margin = %.1f%% → target V = %.1f (log p* = %.3f)\n",
        config.K_val, V_low, V_alpha_star, config.margin_fraction * 100, V_target, y_star)
    @printf("  ω₁ = %.4f, ω_min = %.1e, s_init = %.2f, |Δ| ∈ [%.3f, %.3f], γ = %.2f, n_recovery = %d\n",
        omega, config.omega_min, history.s_init, config.step_bounds[1], config.step_bounds[2],
        config.gamma_decay, config.n_recovery)
    flush(stdout)

    update_omega!(workspace, omega)

    for j in 1:config.n_max
        attempts = j

        # a. Solve the scenario OCP for the current ω, warm-started from the
        #    best candidate so far (problems differ only through ω). The first
        #    attempt starts from the generic guess `U_init` and uses the
        #    cold-start solver options; every later one starts from a previous
        #    optimum and uses the warm-start options.
        warm_start = j > 1
        runtime_optimization += @elapsed solution = solve_ocp!(workspace;
            U_init=U_warm, warm_start=warm_start)
        push!(U_history, solution.U)

        # The candidate is validated whatever the solver status. Certification
        # rests only on the validation draw, which is independent of how U was
        # obtained, so the bound δ is valid for any input trajectory, optimal or
        # not. A failed solve can only cost performance, which the calibration
        # sees through V like any other miss. This is also why iteration and
        # time limits are accepted outcomes of `solve_ocp!` in the first place:
        # the last iterate is a usable plan even though optimality was not
        # established.

        if j == 1 && initial_validation_scenarios !== nothing
            print("Validating on the supplied scenarios... ")
        else
            print("Drawing validation scenarios... ")
        end
        flush(stdout)
        runtime_validation += @elapsed begin
            # b. Draw K_val fresh validation scenarios by continuing the ensemble.
            #    run_mcmc! distributes K_val across the chains; flatten and take exactly K_val.
            #    The first attempt reuses the supplied set if there is one: it holds the
            #    chains' latest draws, which are already independent of the first
            #    candidate, so a redraw would only cost time.
            if j == 1 && initial_validation_scenarios !== nothing
                val_scenarios = initial_validation_scenarios[1:config.K_val]
            else
                chain_results = run_mcmc!(ensemble, sampler_config, model, data, config.K_val)
                val_scenarios = reduce(vcat, (c.samples for c in chain_results))[1:config.K_val]
            end
            last_val_scenarios = val_scenarios

            # c. Count scenarios with at least one constraint violation.
            #    The sample-average cost is evaluated on the same trajectories; it
            #    ranks the certified candidates.
            validation = validate_candidate(model, val_scenarios, solution.U, scenario_constraints,
                problem.cost; output_mode=output_mode, rng=workspace.rng)
            V = validation.V
            cost = Float64(validation.cost)
        end
        @printf("done\n")
        flush(stdout)

        # d. Certify via the Clopper–Pearson bound at confidence β_att.
        delta = clopper_pearson_bound(V, config.K_val, beta_att)

        push!(delta_history, delta)
        push!(violation_history, V)
        push!(omega_history, Float64(omega))
        push!(cost_history, cost)

        # e. Track the best candidate (see `_is_better_candidate`): certified
        #    beats uncertified, then the lowest validation cost among the
        #    certified and the smallest δ among the rest.
        if best_solution === nothing ||
            _is_better_candidate(delta, cost, best_delta, best_cost, config.alpha)
            best_delta = delta
            best_solution = solution
            U_warm = solution.U
            best_omega = Float64(omega)
            best_cost = cost
        end

        # f. Log the attempt.
        @printf("  Attempt %d/%d: V = %d/%d, δ = %.2f%%, ω = %.4f, validation cost = %.4g\n",
            j, config.n_max, V, config.K_val, delta * 100, omega, cost)
        flush(stdout)

        # g. Termination: stop once δ lies in the acceptance band [α - α_tol, α].
        #    Landing in the band ends the search, but the returned candidate is
        #    still the best certified one so far (step e), which already includes
        #    this attempt: an earlier certified attempt below the band may be
        #    cheaper, e.g., if this one came from a truncated or failed solve.
        if config.alpha - config.alpha_tol <= delta <= config.alpha
            @printf("    → δ within band, stopping; certified at (1-δ)=%.2f%% (confidence %.2f%%)\n",
                (1 - delta) * 100, (1 - config.beta) * 100)
            if best_solution !== solution
                @printf("    → returning a cheaper certified earlier candidate: ω = %.4f, δ = %.2f%%, validation cost = %.4g\n",
                    best_omega, best_delta * 100, best_cost)
            end
            println("┗" * "━"^79)
            flush(stdout)
            return AdaptiveCVaRResult(best_solution, best_delta, config.beta, best_omega, best_cost, true, j, :in_band,
                runtime_optimization, runtime_validation,
                delta_history, violation_history, omega_history, cost_history, update_history, U_history,
                last_val_scenarios)
        end

        # h. Termination at the floor: the attempt is below the band, hence
        #    certified, but ω cannot be lowered any further. Near ω = 0 the
        #    objective is dominated by the expected cost and the solution no
        #    longer responds to ω, so further attempts would only repeat it.
        if delta < config.alpha - config.alpha_tol && omega <= config.omega_min
            @printf("    → below band at ω ≤ ω_min = %.1e; cost-dominated solution already certifies, stopping\n",
                config.omega_min)
            println("┗" * "━"^79)
            flush(stdout)
            return AdaptiveCVaRResult(best_solution, best_delta, config.beta, best_omega, best_cost, true, j, :floor,
                runtime_optimization, runtime_validation,
                delta_history, violation_history, omega_history, cost_history, update_history, U_history,
                last_val_scenarios)
        end

        # i. Diagnose the direction of the miss before updating ω.
        if delta > config.alpha
            @printf("    → too many violations (δ above α); raising ω\n")
            flush(stdout)
        else
            @printf("    → below band (more conservative than requested); lowering ω\n")
            flush(stdout)
        end

        # j. Record the failed attempt and update ω. Nothing is recorded at
        #    j = n_max: no update follows it, and ω may be 1 there, whose logit is
        #    infinite.
        if j < config.n_max
            # A count of zero has no logarithm and is replaced by 0.5.
            if V == 0
                y_observed = log(FT(0.5) / FT(config.K_val))
            else
                y_observed = log(FT(V) / FT(config.K_val))
            end
            record!(history, _logit(omega), y_observed)

            # The recovery ramp is only there to establish whether the instance
            # certifies at all; once something has, the remaining attempts belong
            # to the secant, which can still recover performance.
            update = _update_risk_weight(history, omega, y_star, j, config,
                best_delta <= config.alpha)
            push!(update_history, update)

            if update.clip === :none
                step_note = "unclipped"
            elseif update.clip === :bisected
                step_note = @sprintf("bisected in [%.4f, %.4f] (%s)", update.omega_lo, update.omega_hi,
                    update.bracket_source)
            else
                step_note = "clipped at $(update.clip)"
            end
            if update.recovery_active && update.omega_recovery >= update.omega_secant
                branch_note = " [recovery phase]"
            else
                branch_note = ""
            end
            @printf("      ŝ = %.3f (%s), Δ = %.3f (%s), ω → %.4f %s\n",
                update.slope, update.slope_source, update.step, step_note,
                update.omega_next, branch_note)
            flush(stdout)

            # k. Apply the new ω to the OCP for the next solve.
            omega = FT(update.omega_next)
            update_omega!(workspace, omega)
        end
    end

    # Loop exhausted without acceptance: return the best fallback candidate (the
    # certified one with the lowest validation cost if any attempt met the hard
    # requirement, else the smallest δ). It meets the requirement iff its bound satisfies δ ≤ α.
    certified = best_delta <= config.alpha
    if certified
        @printf("  No in-band candidate, but hard requirement met: δ = %.2f%% ≤ α = %.2f%% for the cheapest certified candidate (ω = %.4f, validation cost = %.4g)\n",
            best_delta * 100, config.alpha * 100, best_omega, best_cost)
    else
        @printf("  Calibration failed: best δ = %.2f%% > α = %.2f%% after %d attempts\n",
            best_delta * 100, config.alpha * 100, attempts)
    end
    println("┗" * "━"^79)
    flush(stdout)
    return AdaptiveCVaRResult(best_solution, best_delta, config.beta, best_omega, best_cost, certified, attempts,
        :exhausted,
        runtime_optimization, runtime_validation,
        delta_history, violation_history, omega_history, cost_history, update_history, U_history,
        last_val_scenarios)
end
