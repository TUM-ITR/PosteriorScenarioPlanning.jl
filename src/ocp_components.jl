# =============================================================================
# ocp_components.jl — Generic optimal-control problem (OCP) building blocks
# =============================================================================
# This file provides the formulation-agnostic components from which a
# posterior-sample-based optimal-control problem (OCP) is assembled.
#
# It contains lightweight wrappers for user-defined trajectory costs and
# constraints, concrete box constraints for inputs, states, and outputs, and
# helper routines for translating these constraints into JuMP bounds or JuMP
# inequality constraints.
#
# The file also defines solver configuration objects.
# =============================================================================

# -----------------------------------------------------------------------------
# Output-Mode Compatibility
# -----------------------------------------------------------------------------

"""
    requires_outputs(::AbstractCost) -> Bool
    requires_outputs(::AbstractConstraint) -> Bool
 
Return `true` if the cost or constraint requires output variables `Y`.
 
Defaults to `false`. Concrete types that depend on outputs (e.g.
[`FunctionalCost`](@ref), [`FunctionalScenarioConstraint`](@ref),
[`BoxOutputConstraint`](@ref)) override this to return `true`.
"""
requires_outputs(::AbstractCost) = false
requires_outputs(::AbstractConstraint) = false

"""
    validate_output_mode(problem::OptimalControlProblem)
 
Check that all cost and constraint types are compatible with the problem's
output mode. Throws an `ArgumentError` if a [`WithoutOutputs`](@ref) problem
contains a cost or constraint that requires outputs.
 
This function is called during workspace construction to catch incompatible
configurations early.
"""
function validate_output_mode(problem::OptimalControlProblem)
    if problem.output_mode isa WithoutOutputs
        if requires_outputs(problem.cost)
            throw(ArgumentError(
                "Cost type $(typeof(problem.cost)) requires outputs, " *
                    "but the problem uses WithoutOutputs. " *
                    "Use an output-free cost type such as FunctionalStateCost, " *
                    "or change the problem to use WithOutputs."
            ))
        end
        for c in problem.scenario_constraints
            if requires_outputs(c)
                throw(ArgumentError(
                    "Scenario constraint type $(typeof(c)) requires outputs, " *
                        "but the problem uses WithoutOutputs. " *
                        "Use an output-free constraint type such as " *
                        "FunctionalStateConstraint or BoxStateConstraint, " *
                        "or change the problem to use WithOutputs."
                ))
            end
        end
    end
    return nothing
end

# -----------------------------------------------------------------------------
# Cost Types
# -----------------------------------------------------------------------------

"""
    FunctionalCost(f) <: AbstractCost
 
Cost wrapper for a user-supplied objective function that depends on inputs,
states, and outputs.
 
The function `f` is evaluated as `f(U, X_k, Y_k)` and must return a scalar
JuMP expression or a scalar numeric value. Here, `U` is the shared input
trajectory, while `X_k` and `Y_k` are the state and output trajectories of one
scenario.
 
This cost type requires [`WithOutputs`](@ref) mode.
 
# Fields
- `f`: callable with signature `f(U, X_k, Y_k)`.
"""
struct FunctionalCost{F} <: AbstractCost
    f::F
end

requires_outputs(::FunctionalCost) = true
evaluate_cost(cost::FunctionalCost, u, x, y) = cost.f(u, x, y)

"""
    FunctionalStateCost(f) <: AbstractCost
 
Cost wrapper for a user-supplied objective function that depends on inputs and
states only.
 
The function `f` is evaluated as `f(U, X_k)` and must return a scalar JuMP
expression or a scalar numeric value. Compatible with both [`WithOutputs`](@ref)
and [`WithoutOutputs`](@ref) modes.
 
# Fields
- `f`: callable with signature `f(U, X_k)`.
"""
struct FunctionalStateCost{F} <: AbstractCost
    f::F
end

evaluate_cost(cost::FunctionalStateCost, u, x, ::Nothing) = cost.f(u, x)
evaluate_cost(cost::FunctionalStateCost, u, x, y) = cost.f(u, x)

# -----------------------------------------------------------------------------
# Constraint Types
# -----------------------------------------------------------------------------

# ---- Box input constraints --------------------------------------------------

"""
    BoxInputConstraint{FT} <: AbstractInputConstraint

Box constraint applied to selected input dimensions over a range of time steps.

For each constrained dimension `dims[i]` and time step `t` in `time_range`, the
constraint enforces `lower[i] ≤ U[dims[i], t] ≤ upper[i]`. 

Non-finite bounds disable the corresponding side of the constraint.

# Fields
- `dims::Vector{Int}`: indices of the constrained input dimensions
- `lower::Vector{FT}`: lower bounds for the selected dimensions
- `upper::Vector{FT}`: upper bounds for the selected dimensions
- `time_range::UnitRange{Int}`: time steps on which the bounds are imposed

# Notes
- The bounds are applied directly to the JuMP variables via `set_lower_bound` 
  and `set_upper_bound`
"""
struct BoxInputConstraint{FT<:AbstractFloat} <: AbstractInputConstraint
    dims::Vector{Int}
    lower::Vector{FT}
    upper::Vector{FT}
    time_range::UnitRange{Int}
end

"""
    BoxInputConstraint(dims; lower=-Inf, upper=Inf, time_range=1:typemax(Int))

Convenience constructor for a [`BoxInputConstraint`](@ref).

The bounds `lower` and `upper` may be specified either as scalars or as vectors
of the same length as `dims`. Scalar values are broadcast to all selected input
dimensions.

A lower bound of `-Inf` or an upper bound of `Inf` disables the corresponding
side of the constraint.

`time_range` indexes the columns of the trajectory, i.e. the `H + 1` time points
``0, \\dots, H`` of an OCP with horizon `H` (column `t` is time point `t - 1`). It
is intersected with the available columns when the OCP is constructed, so the
default value applies the constraint over the entire horizon.

# Arguments
- `dims::AbstractVector{Int}`: indices of the input dimensions to constrain

# Keywords
- `lower`: scalar or vector of lower bounds
- `upper`: scalar or vector of upper bounds
- `time_range::UnitRange{Int}`: time steps on which the bounds are imposed
"""
function BoxInputConstraint(dims::AbstractVector{Int};
    lower::Union{Real,AbstractVector{<:Real}}=(-Inf),
    upper::Union{Real,AbstractVector{<:Real}}=Inf,
    time_range::UnitRange{Int}=1:typemax(Int))
    lower_vec = lower isa AbstractVector ? collect(lower) : fill(lower, length(dims))
    upper_vec = upper isa AbstractVector ? collect(upper) : fill(upper, length(dims))
    FT = promote_type(eltype(lower_vec), eltype(upper_vec), Float64)
    return BoxInputConstraint{FT}(collect(dims), FT.(lower_vec), FT.(upper_vec), time_range)
end

"""
    evaluate_constraint(c::BoxInputConstraint, U::AbstractMatrix)

Return a vector of constraint values `lower[i] - U[d, t]` and
`U[d, t] - upper[i]` for each constrained dimension `d` and time step `t`.
Infinite bounds are skipped.
"""
function evaluate_constraint(c::BoxInputConstraint, U::AbstractMatrix)
    n_t = size(U, 2)
    range = intersect(c.time_range, 1:n_t)
    T = typeof(U[1, 1] - zero(eltype(c.lower)))
    exprs = T[]
    @inbounds for (i, d) in enumerate(c.dims)
        for t in range
            if isfinite(c.lower[i])
                push!(exprs, c.lower[i] - U[d, t])
            end
            if isfinite(c.upper[i])
                push!(exprs, U[d, t] - c.upper[i])
            end
        end
    end
    return exprs
end

# ---- Box state constraints --------------------------------------------------

"""
    BoxStateConstraint{FT} <: AbstractScenarioConstraint

Box constraint applied to selected state dimensions over a range of time steps,
enforced separately for each scenario.

For each constrained dimension `dims[i]` and time step `t` in `time_range`, the
constraint enforces `lower[i] ≤ X[dims[i], t] ≤ upper[i]`. 

Non-finite bounds disable the corresponding side of the constraint.

# Fields
- `dims::Vector{Int}`: indices of the constrained state dimensions
- `lower::Vector{FT}`: lower bounds for the selected dimensions
- `upper::Vector{FT}`: upper bounds for the selected dimensions
- `time_range::UnitRange{Int}`: time steps on which the bounds are imposed

# Notes
- The bounds are applied directly to the JuMP variables via `set_lower_bound` 
  and `set_upper_bound`
"""
struct BoxStateConstraint{FT<:AbstractFloat} <: AbstractScenarioConstraint
    dims::Vector{Int}
    lower::Vector{FT}
    upper::Vector{FT}
    time_range::UnitRange{Int}
end

"""
    BoxStateConstraint(dims; lower=-Inf, upper=Inf, time_range=1:typemax(Int))

Convenience constructor for a [`BoxStateConstraint`](@ref).

The bounds `lower` and `upper` may be specified either as scalars or as vectors
of the same length as `dims`. Scalar values are broadcast to all selected state
dimensions.

A lower bound of `-Inf` or an upper bound of `Inf` disables the corresponding
side of the constraint.

`time_range` indexes the columns of the trajectory, i.e. the `H + 1` time points
``0, \\dots, H`` of an OCP with horizon `H` (column `t` is time point `t - 1`). It
is intersected with the available columns when the OCP is constructed, so the
default value applies the constraint over the entire horizon.

# Arguments
- `dims::AbstractVector{Int}`: indices of the state dimensions to constrain

# Keywords
- `lower`: scalar or vector of lower bounds
- `upper`: scalar or vector of upper bounds
- `time_range::UnitRange{Int}`: time steps on which the bounds are imposed
"""
function BoxStateConstraint(dims::AbstractVector{Int};
    lower::Union{Real,AbstractVector{<:Real}}=(-Inf),
    upper::Union{Real,AbstractVector{<:Real}}=Inf,
    time_range::UnitRange{Int}=1:typemax(Int))
    lower_vec = lower isa AbstractVector ? collect(lower) : fill(lower, length(dims))
    upper_vec = upper isa AbstractVector ? collect(upper) : fill(upper, length(dims))
    FT = promote_type(eltype(lower_vec), eltype(upper_vec), Float64)
    return BoxStateConstraint{FT}(collect(dims), FT.(lower_vec), FT.(upper_vec), time_range)
end

"""
    evaluate_constraint(c::BoxStateConstraint, U::AbstractMatrix,
        X_k::AbstractMatrix, ::Any)

Return a vector of constraint values `lower[i] - X_k[d, t]` and
`X_k[d, t] - upper[i]` for each constrained dimension `d` and time step `t`.
Infinite bounds are skipped. The output argument is unused.
"""
function evaluate_constraint(c::BoxStateConstraint, ::AbstractMatrix,
    X_k::AbstractMatrix, ::Any)
    n_t = size(X_k, 2)
    range = intersect(c.time_range, 1:n_t)
    T = typeof(X_k[1, 1] - zero(eltype(c.lower)))
    exprs = T[]
    @inbounds for (i, d) in enumerate(c.dims)
        for t in range
            if isfinite(c.lower[i])
                push!(exprs, c.lower[i] - X_k[d, t])
            end
            if isfinite(c.upper[i])
                push!(exprs, X_k[d, t] - c.upper[i])
            end
        end
    end
    return exprs
end

# ---- Box output constraints -------------------------------------------------

"""
    BoxOutputConstraint{FT} <: AbstractScenarioConstraint

Box constraint applied to selected output dimensions over a range of time steps.

For each constrained dimension `dims[i]` and time step `t` in `time_range`, the
constraint enforces `lower[i] ≤ Y[dims[i], t] ≤ upper[i]`. 

Non-finite bounds disable the corresponding side of the constraint.

This constraint type requires [`WithOutputs`](@ref) mode.

# Fields
- `dims::Vector{Int}`: indices of the constrained output dimensions
- `lower::Vector{FT}`: lower bounds for the selected dimensions
- `upper::Vector{FT}`: upper bounds for the selected dimensions
- `time_range::UnitRange{Int}`: time steps on which the bounds are imposed

# Notes
- The bounds are applied directly to the JuMP variables via `set_lower_bound` 
  and `set_upper_bound`
"""
struct BoxOutputConstraint{FT<:AbstractFloat} <: AbstractScenarioConstraint
    dims::Vector{Int}
    lower::Vector{FT}
    upper::Vector{FT}
    time_range::UnitRange{Int}
end

requires_outputs(::BoxOutputConstraint) = true

"""
    BoxOutputConstraint(dims; lower=-Inf, upper=Inf, time_range=1:typemax(Int))

Convenience constructor for a [`BoxOutputConstraint`](@ref).

The bounds `lower` and `upper` may be specified either as scalars or as vectors
of the same length as `dims`. Scalar values are broadcast to all selected output
dimensions.

A lower bound of `-Inf` or an upper bound of `Inf` disables the corresponding
side of the constraint.

`time_range` indexes the columns of the trajectory, i.e. the `H + 1` time points
``0, \\dots, H`` of an OCP with horizon `H` (column `t` is time point `t - 1`). It
is intersected with the available columns when the OCP is constructed, so the
default value applies the constraint over the entire horizon.

# Arguments
- `dims::AbstractVector{Int}`: indices of the output dimensions to constrain

# Keywords
- `lower`: scalar or vector of lower bounds
- `upper`: scalar or vector of upper bounds
- `time_range::UnitRange{Int}`: time steps on which the bounds are imposed
"""
function BoxOutputConstraint(dims::AbstractVector{Int};
    lower::Union{Real,AbstractVector{<:Real}}=(-Inf),
    upper::Union{Real,AbstractVector{<:Real}}=Inf,
    time_range::UnitRange{Int}=1:typemax(Int))
    lower_vec = lower isa AbstractVector ? collect(lower) : fill(lower, length(dims))
    upper_vec = upper isa AbstractVector ? collect(upper) : fill(upper, length(dims))
    FT = promote_type(eltype(lower_vec), eltype(upper_vec), Float64)
    return BoxOutputConstraint{FT}(collect(dims), FT.(lower_vec), FT.(upper_vec), time_range)
end

"""
    evaluate_constraint(c::BoxOutputConstraint, U::AbstractMatrix,
        X_k::AbstractMatrix, Y_k::AbstractMatrix)

Return a vector of constraint values `lower[i] - Y_k[d, t]` and
`Y_k[d, t] - upper[i]` for each constrained dimension `d` and time step `t`.
Infinite bounds are skipped.
"""
function evaluate_constraint(c::BoxOutputConstraint, ::AbstractMatrix,
    ::AbstractMatrix, Y_k::AbstractMatrix)
    n_t = size(Y_k, 2)
    range = intersect(c.time_range, 1:n_t)
    T = typeof(Y_k[1, 1] - zero(eltype(c.lower)))
    exprs = T[]
    @inbounds for (i, d) in enumerate(c.dims)
        for t in range
            if isfinite(c.lower[i])
                push!(exprs, c.lower[i] - Y_k[d, t])
            end
            if isfinite(c.upper[i])
                push!(exprs, Y_k[d, t] - c.upper[i])
            end
        end
    end
    return exprs
end

# ---- Functional constraints -------------------------------------------------

"""
    FunctionalInputConstraint{F} <: AbstractInputConstraint

Wrap a user-defined input constraint function.

The function `h` is evaluated as `h(U)` and must return a scalar or vector
representing inequality constraints in standard form

```math
h(U) \\leq 0.
```

Here, `U` denotes the full input trajectory (typically a matrix of JuMP
variables, but may also be a numeric matrix during simulation or validation).
The return type should match the input: JuMP expressions when called with JuMP
variables, numeric values when called with numeric arrays.

The returned constraints are added once and shared across all scenarios.

# Fields
- `h::F`: callable with signature `h(U)`

# Notes
- The function `h` must be compatible with JuMP variables, i.e., it should return
  expressions that can be used inside `@constraint`.
"""
struct FunctionalInputConstraint{F} <: AbstractInputConstraint
    h::F
end

evaluate_constraint(c::FunctionalInputConstraint, U) = c.h(U)

"""
    FunctionalScenarioConstraint{F} <: AbstractScenarioConstraint

Wrap a user-defined scenario-dependent constraint function.

The function `h` is evaluated for each scenario as `h(U, X_k, Y_k)` and must
return a scalar or vector representing inequality constraints in standard form

```math
h(U, X_k, Y_k) \\leq 0.
```

Here, `U` is the shared input trajectory, while `X_k` and `Y_k` are the state
and output trajectories for a single scenario. The return type should match the
inputs: JuMP expressions when called with JuMP variables, numeric values when
called with numeric arrays.

This constraint type requires [`WithOutputs`](@ref) mode.

# Fields
- `h::F`: callable with signature `h(U, X_k, Y_k)`

# Notes
- The function `h` must be compatible with JuMP variables, i.e., it should return
  expressions that can be used inside `@constraint`.
"""
struct FunctionalScenarioConstraint{F} <: AbstractScenarioConstraint
    h::F
end

requires_outputs(::FunctionalScenarioConstraint) = true
evaluate_constraint(c::FunctionalScenarioConstraint, U, X_k, Y_k) = c.h(U, X_k, Y_k)


"""
    FunctionalStateConstraint{F} <: AbstractScenarioConstraint
 
Wrap a user-defined scenario-dependent constraint function that depends on
inputs and states only.

The function `h` is evaluated for each scenario as `h(U, X_k)` and must return
a scalar or vector representing inequality constraints in standard form

```math
h(U, X_k) \\leq 0.
```

The return type should match the inputs: JuMP expressions when called with JuMP
variables, numeric values when called with numeric arrays.

Compatible with both [`WithOutputs`](@ref) and [`WithoutOutputs`](@ref) modes.
 
# Fields
- `h::F`: callable with signature `h(U, X_k)`
 
# Notes
- The function `h` must be compatible with JuMP variables, i.e., it should
  return expressions that can be used inside `@constraint`.
"""
struct FunctionalStateConstraint{F} <: AbstractScenarioConstraint
    h::F
end

evaluate_constraint(c::FunctionalStateConstraint, U, X_k, ::Nothing) = c.h(U, X_k)
evaluate_constraint(c::FunctionalStateConstraint, U, X_k, Y_k) = c.h(U, X_k)

# -----------------------------------------------------------------------------
# Internal Helpers — Terminal Input
# -----------------------------------------------------------------------------
# The H transitions use the inputs u_1, …, u_H only, so the input u_{H+1} in the
# last column of U can affect the problem only through the output equation
# y_{H+1} = g(x_{H+1}, u_{H+1}). Without outputs, it is a free variable without
# curvature, which degrades the convergence of interior-point solvers. It is
# therefore held at u_H. With outputs, it is left free for the cost and
# constraints on y_{H+1}.

_add_terminal_input_constraint!(::JuMP.Model, ::WithOutputs, ::AbstractMatrix) = nothing

function _add_terminal_input_constraint!(jump_model::JuMP.Model, ::WithoutOutputs, U::AbstractMatrix)
    n_t = size(U, 2) # H + 1 time points
    @constraint(jump_model, view(U, :, n_t) .== view(U, :, n_t - 1))
    return nothing
end

# -----------------------------------------------------------------------------
# Internal Helpers — Constraint Application
# -----------------------------------------------------------------------------
# These helpers translate high-level constraint objects into JuMP constraints.
#
# Box constraints are handled specially by setting JuMP variable bounds directly,
# which is more efficient and produces cleaner optimization models. General
# functional constraints fall back to `evaluate_constraint` and are added in
# standard inequality form `h <= 0`.
#
# Input constraints are added once because the input trajectory is shared across
# all scenarios. Scenario constraints are added separately for each scenario.
#
# The `Y_k` argument may be `nothing` when the problem uses `WithoutOutputs`.
# Box state constraints and `FunctionalStateConstraint` ignore `Y_k`.
# `BoxOutputConstraint` and `FunctionalScenarioConstraint` require non-nothing
# `Y_k`; this is enforced by `validate_output_mode` at construction time.

# Input constraints (added once; shared across scenarios)
function _add_input_constraint!(::JuMP.Model, c::BoxInputConstraint, U::AbstractMatrix)
    n_t = size(U, 2)
    range = intersect(c.time_range, 1:n_t)
    @inbounds for (i, d) in enumerate(c.dims)
        for t in range
            isfinite(c.lower[i]) && set_lower_bound(U[d, t], c.lower[i])
            isfinite(c.upper[i]) && set_upper_bound(U[d, t], c.upper[i])
        end
    end
    return nothing
end

function _add_input_constraint!(jump_model::JuMP.Model, c::AbstractInputConstraint, U::AbstractMatrix)
    h = evaluate_constraint(c, U)
    @constraint(jump_model, h .<= 0)
    return nothing
end

# Scenario constraints (added once per scenario `k`)
# Y_k may be `nothing` when the problem uses `WithoutOutputs`.
function _add_scenario_constraint!(::JuMP.Model, c::BoxStateConstraint,
    ::AbstractMatrix, X_k::AbstractMatrix, Y_k)
    n_t = size(X_k, 2)
    range = intersect(c.time_range, 1:n_t)
    @inbounds for (i, d) in enumerate(c.dims)
        for t in range
            isfinite(c.lower[i]) && set_lower_bound(X_k[d, t], c.lower[i])
            isfinite(c.upper[i]) && set_upper_bound(X_k[d, t], c.upper[i])
        end
    end
    return nothing
end

function _add_scenario_constraint!(::JuMP.Model, c::BoxOutputConstraint,
    ::AbstractMatrix, ::AbstractMatrix, Y_k::AbstractMatrix)
    n_t = size(Y_k, 2)
    range = intersect(c.time_range, 1:n_t)
    @inbounds for (i, d) in enumerate(c.dims)
        for t in range
            isfinite(c.lower[i]) && set_lower_bound(Y_k[d, t], c.lower[i])
            isfinite(c.upper[i]) && set_upper_bound(Y_k[d, t], c.upper[i])
        end
    end
    return nothing
end

function _add_scenario_constraint!(jump_model::JuMP.Model, c::AbstractScenarioConstraint,
    U::AbstractMatrix, X_k::AbstractMatrix, Y_k)
    h = evaluate_constraint(c, U, X_k, Y_k)
    @constraint(jump_model, h .<= 0)
    return nothing
end

# -----------------------------------------------------------------------------
# JuMP
# -----------------------------------------------------------------------------

"""
    JuMPSolverConfig <: AbstractSolverConfig

Configuration for using a JuMP-compatible optimizer backend.

The configuration stores the optimizer, the solver attributes, and output settings
used when constructing the JuMP model. Attributes are split into a common set,
applied once when the model is built, and two start-dependent sets, one of which
is applied before every solve: the cold-start set when the solve starts from a
generic initial guess, and the warm-start set when it starts from the optimum
of a closely related problem (e.g., the same scenarios at a different risk
weight ω). Typical start-dependent options are the interior-point parameters
`mu_init`, `bound_push` and `bound_frac`.

# Fields
- `optimizer`: optimizer constructor, e.g., `Ipopt.Optimizer` or `MadNLP.Optimizer`
- `attributes::Dict{String,Any}`: common optimizer attributes, applied once when
  the JuMP model is built
- `cold_start_attributes::Dict{String,Any}`: attributes applied before a cold-started
  solve
- `warm_start_attributes::Dict{String,Any}`: attributes applied before a warm-started
  solve
- `silent::Bool`: if `true`, suppress solver output

# Notes
- The cold- and warm-start sets must have the same keys, and these keys must not
  appear in the common set. Attributes persist in the JuMP model between solves,
  so every key set by one start mode must be reset by the other.
- The selected optimizer must support the problem class generated by the OCP
  formulation
- For nonlinear dynamics or nonlinear constraints, use a nonlinear optimizer such
  as Ipopt, MadNLP, or another compatible JuMP backend
"""
struct JuMPSolverConfig{O} <: AbstractSolverConfig
    optimizer::O
    attributes::Dict{String,Any}
    cold_start_attributes::Dict{String,Any}
    warm_start_attributes::Dict{String,Any}
    silent::Bool

    function JuMPSolverConfig{O}(
        optimizer::O,
        attributes::Dict{String,Any},
        cold_start_attributes::Dict{String,Any},
        warm_start_attributes::Dict{String,Any},
        silent::Bool,
    ) where {O}
        cold_keys = Set(keys(cold_start_attributes))
        warm_keys = Set(keys(warm_start_attributes))
        if cold_keys != warm_keys
            throw(ArgumentError("cold- and warm-start attributes must have the same keys; " *
                "only cold: $(collect(setdiff(cold_keys, warm_keys))), " *
                "only warm: $(collect(setdiff(warm_keys, cold_keys)))"))
        end
        shared_keys = intersect(cold_keys, Set(keys(attributes)))
        if !isempty(shared_keys)
            throw(ArgumentError("start-dependent attributes must not also appear in the " *
                "common attributes: $(collect(shared_keys))"))
        end
        return new{O}(optimizer, attributes, cold_start_attributes, warm_start_attributes, silent)
    end
end

"""
    JuMPSolverConfig(optimizer; attributes=Dict(), cold_start_attributes=Dict(),
        warm_start_attributes=Dict(), silent=false)

Construct a JuMP solver configuration.

The `optimizer` argument is the optimizer constructor passed to `JuMP.Model`.
Keys in all attribute dictionaries are converted to strings before being passed
to JuMP.

# Arguments
- `optimizer`: optimizer constructor, e.g. `Ipopt.Optimizer` or `MadNLP.Optimizer`

# Keywords
- `attributes::AbstractDict`: common solver attributes; keys are converted with
  `string`
- `cold_start_attributes::AbstractDict`: attributes for cold-started solves
- `warm_start_attributes::AbstractDict`: attributes for warm-started solves; must
  have the same keys as `cold_start_attributes`
- `silent::Bool`: if `true`, suppress solver output

# Returns
- `config::JuMPSolverConfig`: JuMP solver configuration
"""
function JuMPSolverConfig(
    optimizer;
    attributes::AbstractDict=Dict{String,Any}(),
    cold_start_attributes::AbstractDict=Dict{String,Any}(),
    warm_start_attributes::AbstractDict=Dict{String,Any}(),
    silent::Bool=false,
)
    return JuMPSolverConfig{typeof(optimizer)}(
        optimizer,
        _string_keyed(attributes),
        _string_keyed(cold_start_attributes),
        _string_keyed(warm_start_attributes),
        silent,
    )
end

_string_keyed(d::AbstractDict) = Dict{String,Any}(string(k) => v for (k, v) in pairs(d))

"""
    _build_jump_model(config::JuMPSolverConfig)

Construct a JuMP model from a solver configuration.

This helper creates the JuMP model, applies the common optimizer attributes and
the cold-start attributes, and optionally suppresses solver output. Before each
solve, [`_apply_start_attributes!`](@ref) selects the start-dependent set.
"""
function _build_jump_model(config::JuMPSolverConfig)
    jump_model = Model(config.optimizer)

    for (key, value) in config.attributes
        set_attribute(jump_model, key, value)
    end
    _apply_start_attributes!(jump_model, config, false)

    if config.silent
        set_silent(jump_model)
    end

    return jump_model
end

"""
    _apply_start_attributes!(jump_model::JuMP.Model, config::JuMPSolverConfig, warm_start::Bool)

Set the start-dependent optimizer attributes of `config` on `jump_model`: the
warm-start set if `warm_start` is `true`, the cold-start set otherwise.
"""
function _apply_start_attributes!(jump_model::JuMP.Model, config::JuMPSolverConfig,
    warm_start::Bool)
    if warm_start
        start_attributes = config.warm_start_attributes
    else
        start_attributes = config.cold_start_attributes
    end

    for (key, value) in start_attributes
        set_attribute(jump_model, key, value)
    end
    return nothing
end

"""
    _acceptable_constraint_violation(config::JuMPSolverConfig)

Return the absolute constraint-violation tolerance below which an OCP solve is
accepted, read from the solver attribute `"acceptable_constr_viol_tol"`.

Both Ipopt and MadNLP use this option, with default `1e-2`, as the bound on the
unscaled constraint violation for their own acceptable termination. Using the
same tolerance for runs stopped by an iteration or time limit treats them on
par with `ALMOST_LOCALLY_SOLVED`.
"""
function _acceptable_constraint_violation(config::JuMPSolverConfig)
    return Float64(get(config.attributes, "acceptable_constr_viol_tol", 1e-2))
end

# Termination statuses whose final iterate is accepted, provided its constraint
# violation is within tolerance (see `_assess_solve`). Iteration and time limits
# are included because the solver returns its last iterate, which is usable as
# long as it is feasible.
const _ACCEPTED_TERMINATION_STATUSES = (
    MOI.LOCALLY_SOLVED,
    MOI.ALMOST_LOCALLY_SOLVED,
    MOI.ITERATION_LIMIT,
    MOI.TIME_LIMIT,
)

"""
    _max_constraint_violation(jump_model::JuMP.Model)

Return the largest absolute constraint violation of the current primal
solution of `jump_model`, over all constraints including variable bounds.

Parameters (`VariableRef in Parameter(...)`) are fixed values rather than
constraints and are skipped.
"""
function _max_constraint_violation(jump_model::JuMP.Model)
    max_violation = 0.0
    for (F, S) in list_of_constraint_types(jump_model)
        # Function barrier: the loop body is compiled per constraint type.
        max_violation = max(max_violation, _max_constraint_violation(jump_model, F, S))
    end
    return max_violation
end

function _max_constraint_violation(jump_model::JuMP.Model, ::Type{F}, ::Type{S}) where {F,S}
    max_violation = 0.0
    for con in all_constraints(jump_model, F, S)
        violation = MOI.Utilities.distance_to_set(value(con), constraint_object(con).set)
        max_violation = max(max_violation, violation)
    end
    return max_violation
end

_max_constraint_violation(::JuMP.Model, ::Type, ::Type{<:MOI.Parameter}) = 0.0

"""
    _assess_solve(jump_model::JuMP.Model, tol::Real)

Classify the outcome of the latest `optimize!` on `jump_model`.

A solve counts as successful if its termination status is one of
`LOCALLY_SOLVED`, `ALMOST_LOCALLY_SOLVED`, `ITERATION_LIMIT` or `TIME_LIMIT`
and the maximum constraint violation of the returned iterate is at most
`tol`. Every other status counts as a failed solve.

# Arguments
- `jump_model::JuMP.Model`: model on which `optimize!` has been called
- `tol::Real`: absolute constraint-violation tolerance, see
  [`_acceptable_constraint_violation`](@ref)

# Returns
- `(success, status, max_violation)`: success flag, the termination status, and
  the maximum constraint violation (`Inf` if the solver returned no primal
  solution)
"""
function _assess_solve(jump_model::JuMP.Model, tol::Real)
    status = termination_status(jump_model)

    if has_values(jump_model)
        max_violation = _max_constraint_violation(jump_model)
    else
        max_violation = Inf
    end

    success = (status in _ACCEPTED_TERMINATION_STATUSES) && max_violation <= tol
    return success, status, max_violation
end

"""
    _log_solve(status, iterations, solve_time, max_violation, tol, success, warm_start)

Print a one-line summary of an OCP solve, and warn if it failed.
"""
function _log_solve(status, iterations::Integer, solve_time::Real,
    max_violation::Real, tol::Real, success::Bool, warm_start::Bool)
    if warm_start
        start_note = "warm"
    else
        start_note = "cold"
    end
    if success
        outcome_note = "accepted"
    else
        outcome_note = "FAILED"
    end
    @printf("  OCP solve (%s start): status = %s, iterations = %d, wall time = %.1f s, max violation = %.2e (tol %.1e) → %s\n",
        start_note, status, iterations, solve_time, max_violation, tol, outcome_note)
    flush(stdout)

    if !success
        @warn "OCP solve failed: the termination status is not accepted or the final iterate violates the constraints." status max_violation tol
    end
    return nothing
end
