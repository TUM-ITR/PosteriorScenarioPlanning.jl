# =============================================================================
# robust_scenario_ocp.jl — Worst-case scenario optimal-control problem
# =============================================================================
# Implements the robust scenario OCP, which synthesizes a single input
# trajectory minimizing the worst-case cost over a set of posterior-sample
# scenarios while enforcing dynamics and constraints for every scenario. The
# file defines the formulation type, its preallocated JuMP workspace, and the
# `build_ocp_workspace`/`solve_ocp!` entry points used by the planning layer.
# =============================================================================

# -----------------------------------------------------------------------------
# OCP Formulation
# -----------------------------------------------------------------------------

"""
    RobustScenarioFormulation <: AbstractOCPFormulation

Robust scenario optimal-control formulation.

This formulation computes a single input trajectory `U` that minimizes the
worst-case cost across a set of scenarios (posterior samples), while enforcing
dynamics and constraints for every scenario.

An epigraph reformulation is used, introducing an auxiliary variable ``J_{\\max}``:

```math
\\min_{u_{0:H},\\; J_{\\max}} \\; J_{\\max}
```

subject to, for all scenarios ``k \\in 1{:}K``:

```math
\\begin{aligned}
x_0^{[k]} &= x_{\\mathrm{current}}^{[k]}, \\\\
x_{t+1}^{[k]} &= f(x_t^{[k]}, u_t, v_t^{[k]}, \\theta^{[k]}),
    \\quad t \\in 0{:}H{-}1, \\\\
y_t^{[k]} &= g(x_t^{[k]}, u_t, \\theta^{[k]}) + w_t^{[k]},
    \\quad t \\in 0{:}H, \\\\
J^{[k]} &= J(u_{0:H},\\, x_{0:H}^{[k]},\\, y_{0:H}^{[k]}) \\leq J_{\\max}, \\\\
h(&u_{0:H},\\, x_{0:H}^{[k]},\\, y_{0:H}^{[k]}) \\leq 0.
\\end{aligned}
```

The input trajectory `U` is shared across all scenarios, while states
``X^{[k]}``, outputs ``Y^{[k]}``, and disturbances are scenario-specific.

When the problem uses [`WithoutOutputs`](@ref), the output variables, observation
constraints, and measurement-noise references are omitted entirely. In that case,
cost functions and scenario constraints receive `nothing` in place of `Y_k`.

# Fields
- `problem::OptimalControlProblem`: problem definition containing cost,
  constraints, and prediction horizon.
"""
struct RobustScenarioFormulation <: AbstractOCPFormulation
    problem::OptimalControlProblem
end

# -----------------------------------------------------------------------------
# Workspace
# -----------------------------------------------------------------------------

"""
    RobustScenarioWorkspace{FT,R,S,M,F,RNG,SC} <: AbstractOCPWorkspace

Persistent JuMP representation of a robust scenario-based OCP.

This workspace encodes a worst-case formulation using an epigraph variable
``J_max``. It stores the JuMP model, decision variables, and JuMP parameter
references that can be updated without rebuilding the optimization problem.

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
- `J_max::VariableRef`: epigraph variable for the worst-case cost

## JuMP parameter references
- `scenario_refs::R`: references to scenario-dependent parameters

## Scenario data
- `samples::Vector{S}`: posterior samples currently bound to the JuMP
  parameters. Updated together with the parameter values by
  `update_ocp_workspace!`.

## Problem dimensions
- `K::Int`: number of scenarios
- `H::Int`: prediction horizon

## Metadata
- `dynamics_model::M`: model of the dynamics
- `formulation::F`: OCP formulation used to construct the workspace
- `rng::RNG`: random number generator used for noise sampling during updates

# Notes
- The optimization structure is fixed after construction.
- The workspace is designed for repeated solves, such as in model predictive
  control (MPC), without rebuilding the JuMP model.
"""
mutable struct RobustScenarioWorkspace{
    FT<:AbstractFloat,
    R<:AbstractScenarioRefs,
    S<:AbstractPosteriorSample,
    M<:AbstractModel,
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
    J_max::VariableRef

    scenario_refs::R

    samples::Vector{S}

    K::Int
    H::Int

    dynamics_model::M
    formulation::F
    rng::RNG
end

# -----------------------------------------------------------------------------
# Build Workspace
# -----------------------------------------------------------------------------

"""
    build_ocp_workspace(solver, formulation::RobustScenarioFormulation, 
                        dynamics_model, samples; rng)

Construct a JuMP representation of the robust scenario OCP and return a
`RobustScenarioWorkspace` ready to be solved with [`solve_ocp!`](@ref).

The workspace encodes the epigraph formulation described in
[`RobustScenarioFormulation`](@ref): a single input trajectory ``u_{0:H}``
is optimized against `K` scenarios drawn from the posterior samples. Decision
variables, dynamics constraints, scenario constraints, and the epigraph
objective are all constructed here once. Subsequent solves only require
updating the numerical parameter values via `update_ocp_workspace!`.

When the problem uses [`WithoutOutputs`](@ref), output decision variables,
observation constraints, and measurement-noise parameters are omitted. The
cost and scenario constraints then receive `nothing` in place of `Y_k`.

# Arguments
- `solver::JuMPSolverConfig`: solver configuration (attributes are applied
  once at construction)
- `formulation::RobustScenarioFormulation`: formulation carrying cost,
  constraints, and prediction horizon `H`
- `dynamics_model::AbstractModel`: prediction model whose concrete type
  determines the dynamics constraints that couple decision variables and
  the scenario-dependent parameters added to the JuMP model.
- `samples::Vector{<:AbstractPosteriorSample}`: posterior samples used as
  scenarios. The number of scenarios `K = length(samples)` is fixed at
  construction; rebuild the workspace if `K` changes.

# Keyword Arguments
- `rng::AbstractRNG`: random number generator for noise draws
  (default: `Random.default_rng()`)

# Returns
- `workspace::RobustScenarioWorkspace`: persistent, resolvable workspace
"""
function build_ocp_workspace(
    solver::JuMPSolverConfig,
    formulation::RobustScenarioFormulation,
    dynamics_model::AbstractModel,
    samples::Vector{<:AbstractPosteriorSample};
    rng::AbstractRNG=Random.default_rng(),
)

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

    # Decision variables: inputs shared across scenarios, states per scenario,
    # and the epigraph variable for the worst-case cost.
    @variable(jump_model, U[1:n_u, 1:(H+1)])
    @variable(jump_model, X[1:n_x, 1:(H+1), 1:K])
    @variable(jump_model, J_max)

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

    # Create workspace.
    workspace = RobustScenarioWorkspace{FT,typeof(scenario_refs),typeof(samples[1]),typeof(dynamics_model),RobustScenarioFormulation,typeof(rng),typeof(solver)}(
        jump_model, solver, _acceptable_constraint_violation(solver), U, X, Y, J_max, scenario_refs, samples, K, H, dynamics_model, formulation, rng,
    )

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

    # Scenario-specific constraints.
    # Y_k is `nothing` when the problem uses WithoutOutputs. The uniform loop
    # passes Y_k to all constraints; dispatch on the constraint type handles
    # the `nothing` case (see `evaluate_constraint` methods).
    for k in 1:K
        X_k = view(X, :, :, k)
        Y_k = output_mode isa WithOutputs ? view(Y, :, :, k) : nothing
        for c in problem.scenario_constraints
            _add_scenario_constraint!(jump_model, c, U, X_k, Y_k)
        end
    end

    # Epigraph: J(u, x^{[k]}, y^{[k]}) <= J_max for all k
    for k in 1:K
        X_k = view(X, :, :, k)
        Y_k = output_mode isa WithOutputs ? view(Y, :, :, k) : nothing
        cost_k = evaluate_cost(problem.cost, U, X_k, Y_k)
        @constraint(jump_model, cost_k - J_max <= 0)
    end

    @objective(jump_model, Min, J_max)

    return workspace
end

# -----------------------------------------------------------------------------
# Update Workspace
# -----------------------------------------------------------------------------

"""
    update_scenarios!(workspace::RobustScenarioWorkspace,
                      samples::Vector{<:AbstractPosteriorSample})

Replace the scenario data in `workspace` with fresh posterior samples and new
noise realizations.

This is the intended entry point between receding-horizon solves: it swaps
numerical values without rebuilding the JuMP model, preserving the solver's
sparsity analysis.

# Arguments
- `workspace::RobustScenarioWorkspace`: persistent OCP workspace created by
  [`build_ocp_workspace`](@ref)
- `samples::Vector{<:AbstractPosteriorSample}`: new posterior samples. The
  length must equal `workspace.K`; if the number of scenarios changes, rebuild
  the workspace via [`build_ocp_workspace`](@ref).
"""
function update_scenarios!(
    workspace::RobustScenarioWorkspace,
    samples::Vector{<:AbstractPosteriorSample},
)
    output_mode = workspace.formulation.problem.output_mode
    _set_scenario_values!(workspace.scenario_refs, workspace.dynamics_model, output_mode, samples, workspace.rng)
    return nothing
end

# -----------------------------------------------------------------------------
# Solve OCP
# -----------------------------------------------------------------------------

"""
    solve_ocp!(workspace::RobustScenarioWorkspace; U_init=nothing, warm_start=false)

Solve the robust scenario OCP encoded in `workspace` and return an
[`OCPSolution`](@ref).

If `U_init` is provided, it is used as a warm-start for the input trajectory.
Consistent initial values for the state and output trajectories are then
obtained by forward-simulating the dynamics model under each scenario.

# Arguments
- `workspace::RobustScenarioWorkspace`: workspace with up-to-date parameter
  values

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
- `solution::OCPSolution`: optimized trajectories and solver diagnostics
"""
function solve_ocp!(
    workspace::RobustScenarioWorkspace{FT};
    U_init::Union{Nothing,AbstractMatrix}=nothing,
    warm_start::Bool=false,
) where {FT}
    jump_model = workspace.jump_model
    output_mode = workspace.formulation.problem.output_mode

    # Warm-start: set initial guess for U, and forward-simulate X (and Y)
    if U_init !== nothing
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

        J_max_init = maximum(
            evaluate_cost(workspace.formulation.problem.cost,
                U_init, view(X_init, :, :, k),
                output_mode isa WithOutputs ? view(Y_init, :, :, k) : nothing)
            for k in 1:workspace.K
        )
        set_start_value(workspace.J_max, J_max_init)
    end

    # Select the cold- or warm-start solver options (e.g. barrier parameter and
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

    # Sample-average cost at the optimized input. The robust objective minimizes
    # the worst-case cost `J_max`; `cost` reports the mean cost over the scenarios
    # instead, so it is directly comparable to the CVaR formulation's `cost`.
    cost = if success
        FT(sum(
            evaluate_cost(workspace.formulation.problem.cost, U_opt,
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