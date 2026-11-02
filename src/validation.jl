# =============================================================================
# validation.jl — Validation of a synthesized input
# =============================================================================
# Provides the tools to validate a candidate input trajectory on a fresh,
# independent validation set of posterior samples. The file contains:
#   - the evaluation of a candidate on the validation set, counting of 
#    constraint violations across validation scenarios and evaluating
#    its sample-average cost, and
#   - the Clopper–Pearson confidence bound on the true violation probability.
#
# These routines turn an empirical violation count into a probabilistic
# guarantee and are used by the planner during calibration, where the cost
# ranks the certified candidates.
# =============================================================================

"""
    validate_candidate(dynamics_model, validation_scenarios, U, constraints, cost;
        output_mode=WithoutOutputs(), rng=Random.default_rng())

Evaluate a candidate input trajectory on a set of validation scenarios: count
the scenarios in which at least one constraint is violated, and average the
cost over all scenarios.

Both quantities are computed from one simulation of the validation set.

# Arguments
- `dynamics_model::AbstractModel`: prediction model
- `validation_scenarios::Vector{<:AbstractPosteriorSample}`: validation
  scenarios drawn independently from the training scenarios
- `U::AbstractMatrix`: input trajectory of size `(n_u, H + 1)`
- `constraints::Vector{<:AbstractScenarioConstraint}`: scenario constraints
- `cost::AbstractCost`: trajectory cost of the OCP

# Keyword Arguments
- `output_mode::AbstractOutputMode`: controls whether outputs are simulated
- `rng::AbstractRNG`: random number generator for noise draws

# Returns
- `(V, cost)`: the number of scenarios with at least one constraint violation,
  and the cost averaged over all validation scenarios, violating or not
"""
function validate_candidate(
    dynamics_model::AbstractModel,
    validation_scenarios::Vector{<:AbstractPosteriorSample},
    U::AbstractMatrix,
    constraints::Vector{<:AbstractScenarioConstraint},
    cost::AbstractCost;
    output_mode::AbstractOutputMode=WithoutOutputs(),
    rng::AbstractRNG=Random.default_rng(),
)
    K_val = length(validation_scenarios)
    x, y = simulate_trajectories(dynamics_model, validation_scenarios, U; rng=rng)

    V = 0
    total_cost = zero(eltype(x))
    for k in 1:K_val
        X_k = view(x, :, :, k)
        if output_mode isa WithOutputs
            Y_k = view(y, :, :, k)
        else
            Y_k = nothing
        end

        if _is_violated(constraints, U, X_k, Y_k)
            V += 1
        end

        # Sample-average cost: the out-of-sample counterpart of the
        # expected-cost term J̄ of the OCP objective.
        total_cost += evaluate_cost(cost, U, X_k, Y_k)
    end

    return (V=V, cost=total_cost / K_val)
end

# `true` if the scenario trajectory `X_k`, `Y_k` violates at least one of the
# `constraints` under the inputs `U`.
function _is_violated(constraints::Vector{<:AbstractScenarioConstraint}, U::AbstractMatrix, X_k, Y_k)
    for c in constraints
        h = evaluate_constraint(c, U, X_k, Y_k)
        if h isa AbstractArray
            if any(>(0), h)
                return true
            end
        else
            if h > 0
                return true
            end
        end
    end
    return false
end

"""
    clopper_pearson_bound(V, K_val, beta)

Compute the one-sided Clopper-Pearson upper confidence bound on the constraint
violation probability.

Given ``V`` observed violations among ``K_{\\mathrm{val}}`` validation scenarios and a
confidence level ``\\beta \\in (0, 1)``, returns ``\\delta`` such that with confidence at least
``1 - \\beta``, the true violation probability satisfies ``\\nu \\leq \\delta``. The probability of
constraint satisfaction is then at least ``1 - \\delta``.

# Arguments
- `V::Int`: number of observed constraint violations
- `K_val::Int`: total number of validation scenarios
- `beta::Real`: confidence parameter `β ∈ (0, 1)`

# Returns
- `delta::Float64`: upper bound on the violation probability
"""
function clopper_pearson_bound(V::Int, K_val::Int, beta::Real)
    0 <= V <= K_val || throw(ArgumentError("V must satisfy 0 ≤ V ≤ K_val, got V=$V, K_val=$K_val"))
    0 < beta < 1 || throw(ArgumentError("beta must lie in (0, 1), got $beta"))

    if V == K_val
        return 1.0
    end

    delta = quantile(Beta(V + 1, K_val - V), 1 - beta)
    return delta
end