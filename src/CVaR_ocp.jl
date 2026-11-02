# =============================================================================
# CVaR_ocp.jl — CVaR-penalized scenario optimal-control problem
# =============================================================================
# Implements the CVaR scenario OCP, which synthesizes a single input trajectory
# minimizing a combination of the sample-average cost and a Conditional
# Value-at-Risk penalty on the constraint values. Scenario constraints are
# penalized rather than enforced as hard constraints, so the problem stays
# feasible for any risk weight ω.
# The file defines the formulation type, its preallocated JuMP workspace, the
# `build_ocp_workspace`/`solve_ocp!` entry points, and the `update_omega!` /
# `update_tau!` helpers used by the adaptive planner.
#
# Risk weight ω versus penalty weight λ
# -------------------------------------
# The objective is parametrized as (1-ω)·J̄ + ω·CVaR with ω ∈ [0, 1] rather than
# as J̄ + λ·CVaR with λ ≥ 0. The two families describe the same set of optimizers
# under λ = ω/(1-ω) — scaling an objective by the positive constant 1/(1-ω) does
# not change its argmin — but ω is a bounded coordinate: pure CVaR minimization
# is the attainable member ω = 1, whereas it corresponds to λ = ∞. The adaptive
# calibration in `adaptive_CVaR_planner.jl` relies on being able to reach it.
# =============================================================================

# -----------------------------------------------------------------------------
# OCP Formulation
# -----------------------------------------------------------------------------

"""
    CVaRScenarioFormulation <: AbstractOCPFormulation

CVaR-penalized scenario optimal-control formulation.

This formulation computes a single input trajectory `U` that minimizes a
combination of the sample-average cost across a set of scenarios and a
Conditional Value-at-Risk (CVaR) penalty on the constraint values. Scenario
constraints are not enforced as hard constraints; instead, their violations are
penalized through the CVaR, so the problem remains feasible for any choice of
the risk weight ``\\omega``.

## Multi-dimensional constraints

For notational simplicity, the paper presents the CVaR formulation for a scalar
constraint value ``Z = h(\\cdot)``. This implementation is the natural extension
to ``M`` scalar constraints ``Z_1, \\dots, Z_M`` (e.g., upper and lower bounds
on states or outputs): the CVaR penalty is applied independently to each
constraint dimension ``Z_i``, introducing one auxiliary variable ``\\eta_i`` per
dimension. This is preferred over aggregating constraints via
``h = \\max(h_1, \\dots, h_M)`` because it avoids introducing a non-smooth
operator into the NLP and allows the solver to penalize violations of individual
constraints separately.

## Mathematical formulation

For each scenario ``k \\in \\{1, \\dots, K\\}`` the trajectory cost ``J^{[k]}``
and the constraint value vector ``Z^{[k]} \\in \\mathbb{R}^M`` are obtained by
simulating the dynamics under the sampled parameters and disturbances. Following
Rockafellar & Uryasev's CVaR reformulation, the resulting optimization problem
is:

```math
\\begin{aligned}
\\min_{u_{0:H},\\, \\eta,\\, s} \\quad
& (1 - \\omega) \\frac{1}{K} \\sum_{k=1}^{K} J^{[k]}
  + \\omega \\sum_{i=1}^{M} \\left( \\eta_i + \\frac{1}{\\tau K} \\sum_{k=1}^{K} s_{i,k} \\right) \\\\
\\mathrm{s.t.} \\quad
& x_{t+1}^{[k]} = f(x_t^{[k]}, u_t, v_t^{[k]}, \\theta^{[k]}), \\\\
& y_t^{[k]} = g(x_t^{[k]}, u_t, \\theta^{[k]}) + w_t^{[k]}, \\\\
& s_{i,k} \\geq Z_i^{[k]} - \\eta_i, \\quad s_{i,k} \\geq 0.
\\end{aligned}
```

The slack variables ``s_{i,k}`` linearize the positive part
``(Z_i^{[k]} - \\eta_i)_+``.

Setting ``\\tau = \\alpha`` aligns the CVaR risk measure with the desired
robustness level; ``\\omega`` then trades off nominal performance against
constraint satisfaction. It is equivalent to the penalty parametrization
``\\bar{J} + \\lambda \\, \\mathrm{CVaR}`` under ``\\lambda = \\omega/(1-\\omega)``,
but reaches pure CVaR minimization at the finite endpoint ``\\omega = 1``.

# Fields
- `problem::OptimalControlProblem`: problem definition containing cost,
  constraints, and prediction horizon

# Notes
- Input constraints are kept as hard constraints because they do not depend on
  the uncertainty and are always shared across scenarios.
- Scenario constraints (both `Box*` and `Functional*` types) are converted to
  inequality expressions ``Z_i^{[k]} \\leq 0`` and CVaR-penalized; they are
  not added as hard constraints in this formulation.
- ``\\omega = 0`` removes the CVaR term from the objective and leaves the
  auxiliaries ``\\eta_i`` unbounded along a ray of constant objective value, on
  which the interior-point barrier is unbounded below. Solve at a small positive
  ``\\omega`` instead of exactly zero if a pure-performance solution is needed.
"""
struct CVaRScenarioFormulation <: AbstractOCPFormulation
    problem::OptimalControlProblem
end

# -----------------------------------------------------------------------------
# Workspace
# -----------------------------------------------------------------------------

"""
    CVaRScenarioWorkspace{FT,R,S,M,F,RNG,SC} <: AbstractOCPWorkspace

Persistent JuMP representation of a CVaR-penalized scenario-based OCP.

# Fields
## JuMP model
- `jump_model::JuMP.Model`: underlying JuMP optimization model
- `solver::SC`: solver configuration the model was built with; supplies the
  cold- and warm-start attributes applied before each solve
- `acceptable_violation::Float64`: absolute constraint-violation tolerance
  for accepting a solve, see `_assess_solve`

## Decision variables
- `U::Matrix{VariableRef}`: input trajectory, size `(n_u, H + 1)`
- `X::Array{VariableRef,3}`: state trajectories, size `(n_x, H + 1, K)`
- `Y::Union{Array{VariableRef,3},Nothing}`: output trajectories, size
  `(n_y, H + 1, K)`, or `nothing` if the problem uses [`WithoutOutputs`](@ref)
- `eta::Vector{VariableRef}`: CVaR auxiliary variables ``\\eta_i``, one per
  scalar constraint element (length `M`)
- `s::Matrix{VariableRef}`: non-negative slacks ``s_{i,k}``, size `(M, K)`

## JuMP parameter references
- `scenario_refs::R`: references to scenario-dependent parameters
- `omega_ref::VariableRef`: reference to the risk weight ``\\omega \\in [0, 1]``
- `tau_ref::VariableRef`: reference to tail level ``\\tau \\in (0, 1)``

## Scenario data
- `samples::Vector{S}`: posterior samples currently bound to the JuMP
  parameters

## Problem dimensions
- `K::Int`: number of scenarios
- `H::Int`: prediction horizon
- `M::Int`: number of scalar constraint elements per scenario

## Metadata
- `dynamics_model::M_dyn`: model of the dynamics
- `formulation::F`: OCP formulation used to construct the workspace
- `rng::RNG`: random number generator used for noise sampling during updates
"""
mutable struct CVaRScenarioWorkspace{
    FT<:AbstractFloat,
    R<:AbstractScenarioRefs,
    S<:AbstractPosteriorSample,
    M_dyn<:AbstractModel,
    F<:AbstractOCPFormulation,
    RNG<:AbstractRNG,
    SC<:AbstractSolverConfig
} <: AbstractOCPWorkspace
    jump_model::JuMP.Model
    solver::SC
    acceptable_violation::Float64

    U::Matrix{VariableRef}
    X::Array{VariableRef,3}
    Y::Union{Array{VariableRef,3},Nothing}
    eta::Vector{VariableRef}
    s::Matrix{VariableRef}

    scenario_refs::R
    omega_ref::VariableRef
    tau_ref::VariableRef

    samples::Vector{S}

    K::Int
    H::Int
    M::Int

    dynamics_model::M_dyn
    formulation::F
    rng::RNG
end

# -----------------------------------------------------------------------------
# Build Workspace
# -----------------------------------------------------------------------------

"""
    build_ocp_workspace(solver, formulation::CVaRScenarioFormulation,
                        dynamics_model, samples; rng)

Construct a JuMP representation of the CVaR-penalized scenario OCP and return a
`CVaRScenarioWorkspace` ready to be solved with [`solve_ocp!`](@ref).

A single input trajectory ``u_{0:H}`` is optimized against `K` scenarios drawn
from the posterior samples. Scenario-specific dynamics constraints, the mean
cost, and the CVaR penalty on the constraint values are added once. Subsequent
solves only require updating the numerical parameter values via
[`update_scenarios!`](@ref).

When the problem uses [`WithoutOutputs`](@ref), output decision variables,
observation constraints, and measurement-noise parameters are omitted. The
cost and scenario constraints then receive `nothing` in place of `Y_k`.

# Arguments
- `solver::JuMPSolverConfig`: solver configuration (attributes are applied
  once at construction)
- `formulation::CVaRScenarioFormulation`: formulation carrying cost,
  constraints, and horizon `H`
- `dynamics_model::AbstractModel`: prediction model whose concrete type
  determines the dynamics constraints that couple decision variables and
  the scenario-dependent parameters added to the JuMP model
- `samples::Vector{<:AbstractPosteriorSample}`: posterior samples used as
  scenarios. The number of scenarios `K = length(samples)` is fixed at
  construction; rebuild the workspace if `K` changes
- `omega::Real`: risk weight ``\\omega \\in [0, 1]``
- `tau::Real`: tail level ``\\tau \\in (0, 1)``

# Keyword Arguments
- `rng::AbstractRNG`: random number generator for noise draws
  (default: `Random.default_rng()`)

# Returns
- `workspace::CVaRScenarioWorkspace`: persistent, resolvable workspace.
"""
function build_ocp_workspace(
    solver::JuMPSolverConfig,
    formulation::CVaRScenarioFormulation,
    dynamics_model::AbstractModel,
    samples::Vector{<:AbstractPosteriorSample},
    omega::Real,
    tau::Real;
    rng::AbstractRNG=Random.default_rng(),
)

    0 <= omega <= 1 || throw(ArgumentError("omega must lie in [0, 1], got $omega"))
    0 < tau < 1 || throw(ArgumentError("tau must lie in (0, 1), got $tau"))

    problem = formulation.problem

    # Validate that cost and constraints are compatible with output mode
    validate_output_mode(problem)

    K = length(samples)
    H = problem.H
    n_u = dynamics_model.n_u
    n_x = dynamics_model.n_x
    output_mode = problem.output_mode

    FT = eltype(samples[1].theta)

    # JuMP model with solver attributes applied once
    jump_model = _build_jump_model(solver)

    # Create decision variables for the input and state trajectories across all scenarios.
    @variable(jump_model, U[1:n_u, 1:(H+1)])
    @variable(jump_model, X[1:n_x, 1:(H+1), 1:K])

    # Output variables: only created when the problem uses WithOutputs
    Y = if output_mode isa WithOutputs
        n_y = dynamics_model.n_y
        @variable(jump_model, Y[1:n_y, 1:(H+1), 1:K])
    else
        nothing
    end

    # Add references to scenario-dependent parameters and set values from
    # samples. The output mode controls whether measurement-noise references
    # are created.
    scenario_refs = _add_scenario_refs!(jump_model, dynamics_model, output_mode, H, K; FT=FT)
    _set_scenario_values!(scenario_refs, dynamics_model, output_mode, samples, rng)

    # State-transition constraints (always added)
    _add_dynamics_constraints!(jump_model, dynamics_model, scenario_refs, X, U, H, K)

    # Observation constraints (only when outputs are included)
    if output_mode isa WithOutputs
        _add_observation_constraints!(jump_model, dynamics_model, scenario_refs, X, U, Y, H, K)
    end

    # Input constraints (shared across scenarios)
    for c in problem.input_constraints
        _add_input_constraint!(jump_model, c, U)
    end

    # Hold the terminal input u_{H+1} at u_H when it affects nothing (WithoutOutputs)
    _add_terminal_input_constraint!(jump_model, output_mode, U)

    # Add references for the risk weight ω and the tail level τ, and set their values.
    @variable(jump_model, omega_ref in Parameter(omega))
    @variable(jump_model, tau_ref in Parameter(tau))

    # Collect scenario-constraint expressions Z_i^{[k]} for all scenarios. The
    # number of scalar entries M is determined from scenario k = 1 and assumed
    # identical across scenarios.
    Z = Vector{Vector{AffExpr}}(undef, K)
    for k in 1:K
        X_k = view(X, :, :, k)
        Y_k = output_mode isa WithOutputs ? view(Y, :, :, k) : nothing
        Z_k = AffExpr[]
        for c in problem.scenario_constraints
            append!(Z_k, evaluate_constraint(c, U, X_k, Y_k))
        end
        Z[k] = Z_k
    end
    M = isempty(Z) ? 0 : length(Z[1])
    @assert all(length(Z[k]) == M for k in 1:K) "scenario-constraint vector length must be the same across scenarios"

    # CVaR auxiliary variables and slacks: s_{i,k} >= (Z_i^{[k]} - η_i)_+
    @variable(jump_model, eta[1:M])
    @variable(jump_model, s[1:M, 1:K] >= 0)
    for k in 1:K, i in 1:M
        @constraint(jump_model, s[i, k] >= Z[k][i] - eta[i])
    end

    # Mean cost across scenarios
    mean_cost = sum(
        evaluate_cost(problem.cost, U,
            view(X, :, :, k),
            output_mode isa WithOutputs ? view(Y, :, :, k) : nothing)
        for k in 1:K
    ) / K

    # CVaR term: Σ_i ( η_i + 1/(τK) Σ_k s_{i,k} )
    cvar_sum = if M == 0
        zero(FT)
    else
        sum(
            eta[i] + sum(s[i, k] for k in 1:K) / (tau_ref * K)
            for i in 1:M
        )
    end

    # Objective: (1-ω)·J̄ + ω·CVaR
    @objective(jump_model, Min, (1 - omega_ref) * mean_cost + omega_ref * cvar_sum)

    workspace = CVaRScenarioWorkspace{FT,typeof(scenario_refs),typeof(samples[1]),typeof(dynamics_model),CVaRScenarioFormulation,typeof(rng),typeof(solver)}(
        jump_model, solver, _acceptable_constraint_violation(solver), U, X, Y, eta, s, scenario_refs, omega_ref, tau_ref,
        samples, K, H, M, dynamics_model, formulation, rng
    )

    return workspace
end

# -----------------------------------------------------------------------------
# Update Workspace
# -----------------------------------------------------------------------------

"""
    update_scenarios!(workspace::CVaRScenarioWorkspace,
                      samples::Vector{<:AbstractPosteriorSample})

Replace the scenario data in `workspace` with fresh posterior samples and new
noise realizations.

This is the intended entry point between receding-horizon solves: it swaps
numerical values without rebuilding the JuMP model, preserving the solver's
sparsity analysis.

# Arguments
- `workspace::RobustScenarioWorkCVaRScenarioWorkspacespace`: persistent OCP workspace created by
  [`build_ocp_workspace`](@ref).
- `samples::Vector{<:AbstractPosteriorSample}`: new posterior samples. The
  length must equal `workspace.K`; if the number of scenarios changes, rebuild
  the workspace via [`build_ocp_workspace`](@ref).
"""
function update_scenarios!(
    workspace::CVaRScenarioWorkspace,
    samples::Vector{<:AbstractPosteriorSample},
)
    output_mode = workspace.formulation.problem.output_mode
    _set_scenario_values!(workspace.scenario_refs, workspace.dynamics_model, output_mode, samples, workspace.rng)
    return nothing
end

"""
    update_omega!(workspace::CVaRScenarioWorkspace, omega::Real)

Update the risk weight ``\\omega`` without rebuilding the JuMP model.

# Arguments
- `workspace::CVaRScenarioWorkspace`: persistent OCP workspace
- `omega::Real`: new risk weight in ``[0, 1]``. ``\\omega = 1`` is the pure CVaR
  problem; ``\\omega = 0`` drops the CVaR term and is degenerate (see
  [`CVaRScenarioFormulation`](@ref)).
"""
function update_omega!(workspace::CVaRScenarioWorkspace, omega::Real)
    0 <= omega <= 1 || throw(ArgumentError("omega must lie in [0, 1], got $omega"))
    set_parameter_value(workspace.omega_ref, omega)
    return nothing
end

"""
    update_tau!(workspace::CVaRScenarioWorkspace, tau::Real)

Update the CVaR tail level ``\\tau`` without rebuilding the JuMP model.

# Arguments
- `workspace::CVaRScenarioWorkspace`: persistent OCP workspace
- `tau::Real`: new tail level in ``(0, 1)``
"""
function update_tau!(workspace::CVaRScenarioWorkspace, tau::Real)
    0 < tau < 1 || throw(ArgumentError("tau must lie in (0, 1), got $tau"))
    set_parameter_value(workspace.tau_ref, tau)
    return nothing
end

# -----------------------------------------------------------------------------
# Constraint values at a given trajectory
# -----------------------------------------------------------------------------

"""
    _scenario_constraint_values(problem, U, X, Y, K, FT)

Evaluate the scenario-constraint vector ``Z^{[k]} \\in \\mathbb{R}^M`` for every
scenario at numerical trajectories, returning `Z[k][i] = Z_i^{[k]}`.

Mirrors the expression-level assembly in [`build_ocp_workspace`](@ref), so the
entries appear in the same order as the JuMP constraint expressions.
"""
function _scenario_constraint_values(
    problem::OptimalControlProblem,
    U::AbstractMatrix,
    X::AbstractArray{<:Real,3},
    Y::Union{AbstractArray{<:Real,3},Nothing},
    K::Int,
    ::Type{FT},
) where {FT<:AbstractFloat}
    output_mode = problem.output_mode
    Z = Vector{Vector{FT}}(undef, K)
    for k in 1:K
        X_k = view(X, :, :, k)
        Y_k = output_mode isa WithOutputs ? view(Y, :, :, k) : nothing
        Z_k = FT[]
        for c in problem.scenario_constraints
            h = evaluate_constraint(c, U, X_k, Y_k)
            if h isa AbstractArray
                append!(Z_k, vec(h))
            else
                push!(Z_k, h)
            end
        end
        Z[k] = Z_k
    end
    return Z
end

# -----------------------------------------------------------------------------
# Solve OCP
# -----------------------------------------------------------------------------

"""
    solve_ocp!(workspace::CVaRScenarioWorkspace; U_init=nothing, warm_start=false)

Solve the CVaR-penalized scenario OCP encoded in `workspace` and return an
[`OCPSolution`](@ref).

If `U_init` is provided, it is used as a warm-start for the input trajectory;
states (and outputs) are forward-simulated under each scenario for consistent
initial values. The CVaR auxiliaries `η` and slacks `s` are warm-started from
the resulting constraint values: `η_i` is the empirical `(1-τ)`-quantile of
`Z_i` across scenarios, which is optimal at that input, and
`s_{i,k} = max(Z_i^{[k]} - η_i, 0)`.

# Arguments
- `workspace::CVaRScenarioWorkspace`: workspace with up-to-date parameter
  values.

# Keywords
- `U_init::Union{Nothing,AbstractMatrix}`: optional initial guess for the
  input trajectory of size `(n_u, H + 1)`. When `nothing`, the solver starts from
  whatever values are currently stored in the JuMP variables.
- `warm_start::Bool`: if `true`, the solve is treated as a warm start from the
  optimum of a closely related problem and uses the warm-start solver
  attributes of the workspace's [`JuMPSolverConfig`](@ref); otherwise the
  cold-start attributes are used. This only selects solver options; the
  initial point is set by `U_init` either way.

# Returns
- `solution::OCPSolution`: optimized trajectories and solver diagnostics.
"""
function solve_ocp!(
    workspace::CVaRScenarioWorkspace{FT};
    U_init::Union{Nothing,AbstractMatrix}=nothing,
    warm_start::Bool=false,
) where {FT}
    jump_model = workspace.jump_model
    problem = workspace.formulation.problem
    output_mode = problem.output_mode

    # Warm-start
    if U_init !== nothing
        # Set initial guess for U, and forward-simulate X (and Y)
        set_start_value.(workspace.U, U_init)

        noise = _extract_noise_values(workspace.scenario_refs, output_mode)
        X_init, Y_init = simulate_trajectories(
            workspace.dynamics_model, workspace.samples, U_init;
            noise...
        )

        set_start_value.(workspace.X, X_init)
        if output_mode isa WithOutputs
            set_start_value.(workspace.Y, Y_init)
        end

        # Warm-start CVaR auxiliaries from simulated constraint values.
        if workspace.M > 0
            Y_init = output_mode isa WithOutputs ? Y_init : nothing
            Z = _scenario_constraint_values(problem, U_init, X_init, Y_init, workspace.K, FT)

            # Set η_i to the empirical (1-τ)-quantile of Z_i across scenarios,
            # which is the optimal η at this input.
            for i in 1:workspace.M
                Z_i = [Z[k][i] for k in 1:workspace.K]
                eta_init = quantile(Z_i, 1 - parameter_value(workspace.tau_ref))
                set_start_value(workspace.eta[i], eta_init)
                for k in 1:workspace.K
                    set_start_value(workspace.s[i, k], max(Z[k][i] - eta_init, zero(FT)))
                end
            end
        end
    end

    # Select the cold- or warm-start solver options (e.g., barrier parameter and
    # bound push) for this solve; they persist in the model until the next call.
    _apply_start_attributes!(jump_model, workspace.solver, warm_start)

    t0 = time()
    optimize!(jump_model)
    solve_time = time() - t0

    # Accept a solved, almost solved, or iteration-/time-limited run whose final
    # iterate is feasible within the solver's acceptable tolerance.
    success, status, max_violation = _assess_solve(jump_model, workspace.acceptable_violation)
    iter = MOI.get(jump_model, MOI.BarrierIterations())
    _log_solve(status, iter, solve_time, max_violation,
        workspace.acceptable_violation, success, warm_start)

    U_opt = FT.(value.(workspace.U))
    X_opt = FT.(value.(workspace.X))

    Y_opt = if output_mode isa WithOutputs
        FT.(value.(workspace.Y))
    else
        nothing
    end

    obj = if success
        FT(objective_value(jump_model))
    else
        FT(NaN)
    end

    # Sample-average cost at the optimized input, without the CVaR penalty and
    # without any weighting. This is the `mean_cost` term J̄ of the objective (see
    # `build_ocp_workspace`) and is reported separately because, unlike `obj`, it
    # does not depend on ω.
    cost = if success
        FT(sum(
            evaluate_cost(problem.cost, U_opt,
                view(X_opt, :, :, k),
                output_mode isa WithOutputs ? view(Y_opt, :, :, k) : nothing)
            for k in 1:workspace.K
        ) / workspace.K)
    else
        FT(NaN)
    end

    diagnostics = SolverDiagnostics(
        solve_time,
        Int(iter),
        success,
        Dict{Symbol,Any}(
            :termination_status => status,
            :max_constraint_violation => max_violation,
        ),
    )

    return OCPSolution{FT}(U_opt, X_opt, Y_opt, obj, cost, diagnostics)
end