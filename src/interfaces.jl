# =============================================================================
# interfaces.jl — Core interfaces for inference, optimal control, and planning
# =============================================================================
#
# This file defines the main abstractions for inference, optimal control, and
# planning with uncertain dynamical systems. The interfaces below separate
# three conceptual layers:
#
#   1. Inference:
#      Generate posterior samples of model parameters and latent states from
#      observed data, typically using MCMC.
#
#   2. Optimal control (synthesis):
#      Use posterior samples as scenarios in an optimal-control problem (OCP) to
#      compute an input trajectory and predicted state and output trajectories
#      under model uncertainty.
#
#   3. Planning (synthesis–validation loop):
#      Wrap an OCP in a synthesis–validation loop that certifies each candidate
#      input on a freshly drawn validation set and adaptively recalibrates the
#      formulation until a robustness guarantee is met.
#
# ── Inference layer ───────────────────────────────────────────────────────────
# - Data (`AbstractData`): representation of observed input-output data.
# - Models (`AbstractModel`): state-space models defining dynamics, outputs,
#   parameters, and noise distributions.
# - Posterior samples (`AbstractPosteriorSample`): sampled parameters and latent
#   states used by both inference and planning.
# - MCMC samplers, states, and workspaces: machinery for constructing Markov
#   chains over parameters and latent variables.
# - Proposals and adaptations: mechanisms for generating and tuning MCMC moves.
#
# ── Optimal-control layer ─────────────────────────────────────────────────────
# - Costs and constraints (`AbstractCost`, `AbstractConstraint`): trajectory-level
#   objectives and feasibility requirements.
# - Optimal-control problems (`OptimalControlProblem`): horizon, cost, and
#   constraints before scenario aggregation.
# - Formulations (`AbstractOCPFormulation`): rules for combining posterior
#   samples into an optimization problem, e.g. robust, expected-cost, or
#   risk-sensitive formulations.
# - Solver configurations (`AbstractSolverConfig`): backend-specific solver
#   options.
# - OCP workspaces (`AbstractOCPWorkspace`): persistent optimization
#   representations that separate problem structure from numerical data.
# - Solutions (`OCPSolution`, `SolverDiagnostics`): optimized trajectories,
#   objective values, and solver diagnostics.
#
# The optimal-control layer is designed for repeated solves, as in model
# predictive control (MPC): the optimization problem structure is built once,
# while scenario-dependent data such as posterior samples, initial states, and
# disturbance realizations can be updated between solves.
#
# ── Planning layer ────────────────────────────────────────────────────────────
# - Planners (`AbstractPlanner`): procedures that wrap an optimal-control 
#   formulation with a certification procedure and a scenario source.
#   A planner repeatedly synthesizes and certifies candidate inputs and 
#   adapts the formulation until a robustness guarantee is met.
#
# Whereas the optimal-control layer performs a single optimization (`solve_ocp!`),
# the planning layer closes the loop around it: synthesis, validation, and
# recalibration, yielding a certified input trajectory.
#
# Concrete implementations specialize these interfaces to define specific
# models, MCMC samplers, OCP formulations, solver backends, and planners while
# keeping inference, optimal-control, and planning components composable.
# =============================================================================

# -----------------------------------------------------------------------------
# Models and Data Interface
# -----------------------------------------------------------------------------

"""
    AbstractData

Abstract supertype for all data containers.
"""
abstract type AbstractData end

"""
    subset_data(data, range)

Return a subset of `data` restricted to the index set `range`.

# Arguments
- `data::AbstractData`: input data container
- `range::UnitRange{Int}`: index range to extract

# Returns
- `subset::AbstractData`: new data container containing only the selected entries
"""
function subset_data end

"""
    AbstractModel

Abstract supertype for all state-space models.
"""
abstract type AbstractModel end

"""
    AbstractPosteriorSample

Abstract supertype for posterior samples.

A posterior sample may include model parameters, latent states, or
additional auxiliary variables.
"""
abstract type AbstractPosteriorSample end

"""
    simulate_trajectories(model, samples, u; rng, kwargs...)
 
Generate predictive state and output trajectories from posterior samples.
 
For each posterior sample, the model is simulated forward over the input
trajectory `u`, producing one trajectory per sample. The result is an
ensemble of trajectories representing posterior predictive uncertainty.
 
# Arguments
- `model::AbstractModel`: dynamics model
- `samples::Vector{<:AbstractPosteriorSample}`: posterior samples
- `u::AbstractMatrix`: input trajectory of size `(n_u, n_t)`
 
# Keywords
- `rng::AbstractRNG`: random number generator (default: `Random.default_rng()`)
- Model-specific noise realizations may be passed as additional keyword
  arguments (e.g. `V`, `W` for `DiscreteTimeStochasticSSM`). When provided, 
  these are used directly instead of drawing fresh realizations from `rng`. 
  When omitted or `nothing`, fresh noise is drawn. This allows callers such as
  [`solve_ocp!`](@ref) to produce initial guesses that are consistent with
  the noise realizations encoded in the OCP (see
  `_extract_noise_values`).
 
# Returns
- `x::Array`: state trajectories of size `(n_x, n_t, K)`
- `y::Array`: output trajectories of size `(n_y, n_t, K)`
 
# Notes
- `n_t` is the number of time points; for an OCP with horizon `H`, `n_t = H + 1`.
- `K = length(samples)` is the number of posterior samples.
- Each slice `x[:, :, k]` and `y[:, :, k]` corresponds to posterior sample `k`.
- Concrete model implementations must support passing pre-drawn noise
  realizations as keyword arguments. Any noise keyword that is `nothing` or
  omitted must be drawn internally using `rng`.
"""
function simulate_trajectories end

"""
    extract_parameters(sample)

Extract the parameter vector from a posterior sample.
The returned parameters should correspond to the variables 
that are part of the sampler's proposal distribution.

# Arguments
- `sample::AbstractPosteriorSample`: posterior sample

# Returns
- `parameters::Vector{<:AbstractFloat}`: vector of model parameters
"""
function extract_parameters end

"""
    AbstractScenarioRefs
 
Abstract supertype for model-specific JuMP parameter references.
 
Concrete subtypes store references to scenario-dependent quantities, such as
posterior parameters, initial states, process noise, or measurement noise.
 
Each dynamics model must define a concrete subtype and implement:
 
- [`_add_scenario_refs!`](@ref): create references during workspace construction
- [`_set_scenario_values!`](@ref): update references before each solve
- [`_add_dynamics_constraints!`](@ref): add state-transition constraints
- [`_add_observation_constraints!`](@ref): add observation-equation constraints
  (only required if the model supports [`WithOutputs`](@ref))
"""
abstract type AbstractScenarioRefs end

"""
    _add_scenario_refs!(jump_model, dynamics_model, output_mode, H, K; FT)
 
Add JuMP parameter references for scenario-dependent quantities.
 
This function is called once during OCP workspace construction. It adds the
parameters required by `dynamics_model` to `jump_model` and returns a refs
object that stores handles to them.
 
The `output_mode` argument determines whether measurement-noise references
are created. Models that do not use measurement noise may ignore this argument.
 
# Arguments
- `jump_model::JuMP.Model`: JuMP model to add parameters to
- `dynamics_model::AbstractModel`: dynamics model
- `output_mode::AbstractOutputMode`: [`WithOutputs`](@ref) or
  [`WithoutOutputs`](@ref)
- `H::Int`: prediction horizon; references cover the `H + 1` time points
- `K::Int`: number of scenarios
 
# Keywords
- `FT::Type`: floating-point type for parameter initialization
 
# Returns
- `refs::AbstractScenarioRefs`: model-specific JuMP parameter references
"""
function _add_scenario_refs! end

"""
    _set_scenario_values!(refs, dynamics_model, output_mode, samples, rng)
 
Update scenario-dependent parameter values.
 
This function is called before each solve to refresh the numerical values stored
in `refs`. For stochastic models, this may include drawing fresh process-noise
realizations using `rng`. Measurement-noise values are only drawn when
`output_mode` is [`WithOutputs`](@ref).
 
# Arguments
- `refs::AbstractScenarioRefs`: model-specific parameter references created by
  [`_add_scenario_refs!`](@ref)
- `dynamics_model::AbstractModel`: dynamics model
- `output_mode::AbstractOutputMode`: [`WithOutputs`](@ref) or
  [`WithoutOutputs`](@ref)
- `samples::Vector{<:AbstractPosteriorSample}`: new posterior samples
- `rng::AbstractRNG`: random number generator
 
# Notes
- The number of samples must match the number of scenarios used during
  workspace construction.
- Implementations should update values in-place without modifying the
  optimization structure.
"""
function _set_scenario_values! end
"""
    _add_dynamics_constraints!(jump_model, dynamics_model, refs, X, U, H, K)
 
Add state-transition constraints (initial conditions and state propagation).
 
This function is called for all problems regardless of output mode. It encodes:
- Initial-state constraints pinning `X[:, 1, k]` to references in `refs`.
- State-transition constraints for `t ∈ 0:H-1`, i.e. `H` transitions.
 
# Arguments
- `jump_model::JuMP.Model`: JuMP model
- `dynamics_model::AbstractModel`: dynamics model
- `refs::AbstractScenarioRefs`: model-specific parameter references
- `X::Array{JuMP.VariableRef,3}`: state variables of size `(n_x, H + 1, K)`
- `U::Matrix{JuMP.VariableRef}`: input variables of size `(n_u, H + 1)`
- `H::Int`: prediction horizon
- `K::Int`: number of scenarios
"""
function _add_dynamics_constraints! end

"""
    _add_observation_constraints!(jump_model, dynamics_model, refs, X, U, Y, H, K)
 
Add observation-equation constraints coupling `Y` to `X` and `U`.
 
This function is called only when the problem uses [`WithOutputs`](@ref).
 
# Arguments
- `jump_model::JuMP.Model`: JuMP model
- `dynamics_model::AbstractModel`: dynamics model
- `refs::AbstractScenarioRefs`: model-specific parameter references
- `X::Array{JuMP.VariableRef,3}`: state variables of size `(n_x, H + 1, K)`
- `U::Matrix{JuMP.VariableRef}`: input variables of size `(n_u, H + 1)`
- `Y::Array{JuMP.VariableRef,3}`: output variables of size `(n_y, H + 1, K)`
- `H::Int`: prediction horizon
- `K::Int`: number of scenarios
"""
function _add_observation_constraints! end

# -----------------------------------------------------------------------------
# Adaptation Interface
# -----------------------------------------------------------------------------

"""
    AbstractAdaptation

Abstract supertype for mechanisms that adapt sampler parameters during MCMC.

Adaptations observe the sampling process and propose parameter updates.
"""
abstract type AbstractAdaptation end

"""
    AbstractStepAdaptation <: AbstractAdaptation

Adaptations that are invoked at every MCMC iteration.

These adaptations operate online and influence the sampler within the
current stage.
"""
abstract type AbstractStepAdaptation <: AbstractAdaptation end

"""
    AbstractStageAdaptation <: AbstractAdaptation

Adaptations that are invoked between stages in a staged MCMC procedure.

These adaptations operate offline (between stages) and typically perform
coarser adjustments based on global properties of the chain.
"""
abstract type AbstractStageAdaptation <: AbstractAdaptation end

"""
    compute_update(adaptation::AbstractStepAdaptation, sampler, accepted::Bool, iteration::Int)
    compute_update(adaptation::AbstractStageAdaptation, sampler, workspace, samples, data, model, iteration::Int)

Compute parameter updates proposed by an adaptation.

The exact inputs depend on the adaptation type:

# AbstractStepAdaptation
Invoked at every MCMC iteration. Receives local information from the current step:
- `sampler`: the MCMC sampler being adapted
- `accepted::Bool`: whether the current proposal was accepted
- `iteration::Int`: the current iteration index

# AbstractStageAdaptation
Invoked between stages in a staged MCMC procedure. Receives aggregated information
from the previous stage:
- `sampler`: the MCMC sampler being adapted
- `workspace`: the MCMC workspace containing state and buffers
- `samples`: posterior samples from the previous stage
- `data`: observed data
- `model`: the model used for inference
- `iteration::Int`: the current stage index (starting from 1)

# Returns
- `update::NamedTuple`: proposed updates to sampler parameters (e.g. `scale`, `N`, `Sigma`).
  An empty `NamedTuple()` indicates that no update is applied.
"""
function compute_update end

"""
    reset!(adaptation::AbstractAdaptation)

Clear the adaptation's internal state. Default is a no-op, 
which is correct for stateless adaptations.
"""
reset!(::AbstractAdaptation) = nothing

# -----------------------------------------------------------------------------
# MCMC Interface
# -----------------------------------------------------------------------------

"""
    AbstractMCMCSampler

Abstract supertype for all MCMC sampling algorithms.
"""
abstract type AbstractMCMCSampler end

"""
    allocate_samples(sampler::AbstractMCMCSampler, model::AbstractModel, K::Int, T::Int)

Allocate storage for `K` posterior samples.

# Arguments
- `sampler::AbstractMCMCSampler`: sampler instance
- `model::AbstractModel`: model under consideration
- `K::Int`: number of samples to allocate
- `T::Int`: data length

# Returns
- `samples::Vector{<:AbstractPosteriorSample}`: preallocated sample container of length `K`
"""
function allocate_samples end

"""
    AbstractMCMCState

Abstract supertype for the internal state of an MCMC sampler.
"""
abstract type AbstractMCMCState end

"""
    initialize_state(rng, buffer, sampler, model, data; theta_start)

Initialize the internal state of an MCMC sampler.

# Arguments
- `rng::AbstractRNG`: random number generator
- `buffer::AbstractBuffer`: auxiliary buffer for intermediate computations
- `sampler::AbstractMCMCSampler`: sampler instance
- `model::AbstractModel`: model under consideration
- `data::AbstractData`: observed data

# Keywords
- `theta_start`: initial parameter vector. If not provided, it is sampled from
  the prior.
"""
function initialize_state end

"""
    AbstractBuffer

Abstract supertype for computation buffers used in samplers and filters.
"""
abstract type AbstractBuffer end

"""
    MCMCWorkspace{S,B,R}

Mutable container for MCMC sampling, containing the current state and a buffer.

# Fields
- `state::S`: current sampler state
- `buffer::B`: auxiliary buffer for intermediate computations
- `rng::R`: random number generator
- `T::Int`: data length
"""
mutable struct MCMCWorkspace{S<:AbstractMCMCState,B<:AbstractBuffer,R<:AbstractRNG}
    state::S
    buffer::B
    rng::R
    T::Int
end

"""
    build_mcmc_workspace(rng, sampler, model, data; theta_start)
    build_mcmc_workspace(rng, sampler, stage_adaptations, model, data; theta_start)

Construct an MCMC workspace for a given sampler.

The workspace stores the internal state, buffers, and auxiliary data required
to run the sampler efficiently. It is reused across iterations of the MCMC
algorithm.

If `stage_adaptations` are provided, the workspace is sized to accommodate the
largest buffer requirements across all adaptation stages.

# Arguments
- `rng::AbstractRNG`: random number generator
- `sampler::AbstractMCMCSampler`: sampler instance
- `stage_adaptations::AbstractVector{<:AbstractStageAdaptation}`: optional stage adaptations for buffer sizing
- `model::AbstractModel`: model under consideration
- `data::AbstractData`: observed data

# Keywords
- `theta_start`: initial parameter vector. If not provided, it is sampled from
  the prior.
- `T_max::Int` (with `stage_adaptations` only): data length the buffers are
  sized for; defaults to the length of `data`.
"""
function build_mcmc_workspace end

"""
    mcmc_step!(sampler, workspace, model, data)

Perform a single MCMC transition step.

# Arguments
- `sampler::AbstractMHSampler`: sampler instance
- `workspace::MCMCWorkspace`: current workspace (modified in-place)
- `model::AbstractModel`: model under consideration
- `data::AbstractData`: observed data

# Returns
- `(success::Bool, accepted::Bool)`:
    - `success`: indicates whether the proposal was valid
    - `accepted`: indicates whether the proposal was accepted
"""
function mcmc_step! end

"""
    create_sample(sampler::AbstractMCMCSampler,model::AbstractModel,data::AbstractData,workspace::MCMCWorkspace)

Generates a posterior sample from the current state of the MCMC sampler.

# Arguments
- `sampler::AbstractMCMCSampler`: sampler instance
- `model::AbstractModel`: model under consideration
- `data::AbstractData`: observed data
- `workspace::MCMCWorkspace`: MCMC workspace

# Returns
- `sample::AbstractPosteriorSample`: posterior sample
"""
function create_sample end

"""
    MCMCConfig

Configuration of an MCMC chain's dynamics.

This object describes how a chain runs (burn-in, thinning, step adaptations,
verbosity).

# Fields
- `burnin::Int`: length of the burn-in period
- `thinning::Int`: number of samples to be skipped to decrease correlation (thinning)
- `step_adaptations::Vector{AbstractStepAdaptation}`: step-level adaptation mechanisms (empty vector = no step adaptation)
- `verbose::Bool`: if `true`, print progress during sampling
"""
struct MCMCConfig
    burnin::Int
    thinning::Int
    step_adaptations::Vector{AbstractStepAdaptation}
    verbose::Bool
end

function MCMCConfig(burnin::Int, thinning::Int;
    step_adaptations::AbstractVector{<:AbstractStepAdaptation}=AbstractStepAdaptation[],
    verbose::Bool=true)
    return MCMCConfig(burnin, thinning,
        convert(Vector{AbstractStepAdaptation}, step_adaptations), verbose)
end

"""
    StagedMCMCConfig

General configuration for staged MCMC inference.

# Fields
- `intermediate_stages::MCMCConfig`: chain dynamics for intermediate stages
- `final_stage::MCMCConfig`: chain dynamics for the final stage
- `K_intermediate::Int`: total number of samples (across all chains) per intermediate stage
- `K_final::Int`: total number of samples (across all chains) in the final stage
- `T_chunk::Int`: number of data points added at each stage
- `verbose::Bool`: if `true`, print progress during sampling
"""
struct StagedMCMCConfig
    intermediate_stages::MCMCConfig
    final_stage::MCMCConfig
    K_intermediate::Int
    K_final::Int
    T_chunk::Int
    verbose::Bool
end

function StagedMCMCConfig(intermediate_stages::MCMCConfig, final_stage::MCMCConfig, T_chunk::Int;
    K_intermediate::Int, K_final::Int, verbose::Bool=true)
    return StagedMCMCConfig(intermediate_stages, final_stage, K_intermediate, K_final, T_chunk, verbose)
end

"""
    ChainResult{S}

Result of running a single MCMC chain (see [`run_mcmc!`](@ref)).

# Fields
- `samples::S`: collected posterior samples (a `Vector{<:AbstractPosteriorSample}`)
- `acceptance_rate::Float64`: ratio of accepted proposals over all iterations
"""
struct ChainResult{S}
    samples::S
    acceptance_rate::Float64
end

# -----------------------------------------------------------------------------
# Metropolis-Hastings Interface
# -----------------------------------------------------------------------------

"""
    AbstractMHSampler <: AbstractMCMCSampler

Abstract supertype for Metropolis-Hastings based samplers.
"""
abstract type AbstractMHSampler <: AbstractMCMCSampler end

"""
    MCMCMode

Abstract supertype for distinguishing between different MCMC exploration strategies (e.g., standard vs. blocked).
"""
abstract type MCMCMode end

"""
    Standard <: MCMCMode

Trait representing a standard parameter-only MCMC exploration.
"""
struct Standard <: MCMCMode end

"""
    Blocked <: MCMCMode

Trait representing a blocked MCMC exploration where the initial latent state ``x_0`` is proposed alongside model parameters ``\\theta``.
"""
struct Blocked <: MCMCMode end

"""
    evaluate_state!(state, sampler, workspace, model, data)

Evaluates the log-prior and log-likelihood of a MCMC state.

# Arguments
- `state::AbstractMCMCState`: MCMC state
- `sampler::AbstractMHSampler`: sampler instance
- `workspace::MCMCWorkspace`: MCMC workspace
- `model::AbstractModel`: model under consideration
- `data::AbstractData`: observed data

# Notes
- Intended for candidate states, whose log-densities enter the MH ratio. To
  re-evaluate the current state of the chain (`workspace.state`), e.g., after the
  data changed, use [`reevaluate_state!`](@ref) instead, which also keeps
  sampler-specific quantities such as the PMMH latent trajectory consistent.
"""
function evaluate_state! end

"""
    accept_state!(sampler, workspace, proposed_state)

Make the evaluated `proposed_state` the current state of the chain. Called by
`mcmc_step!` on acceptance, directly after the proposal was evaluated.
The default only replaces `workspace.state`. Samplers that sample additional
quantities from the evaluation output extend it, e.g., PMMH draws the latent
trajectory from the particle filter that produced the accepted likelihood
estimate.

# Arguments
- `sampler::AbstractMHSampler`: sampler instance
- `workspace::MCMCWorkspace`: MCMC workspace (modified in-place)
- `proposed_state::AbstractMCMCState`: accepted proposal
"""
function accept_state! end

"""
    reevaluate_state!(sampler, workspace, model, data)

Re-evaluate the log-prior and log-likelihood of the current state of the chain
on `data`, e.g., when the data set has changed between stages. The default
calls [`evaluate_state!`](@ref) on `workspace.state`. Samplers that sample
additional quantities from the evaluation output extend it, e.g., PMMH redraws
the latent trajectory from the new particle filter run so that it is consistent
with `data`.

# Arguments
- `sampler::AbstractMHSampler`: sampler instance
- `workspace::MCMCWorkspace`: MCMC workspace (modified in-place)
- `model::AbstractModel`: model under consideration
- `data::AbstractData`: data the state is evaluated on
"""
function reevaluate_state! end

# -----------------------------------------------------------------------------
# Proposal Interface
# -----------------------------------------------------------------------------

"""
    AbstractProposal

Abstract supertype for proposal distributions used in MCMC.
"""
abstract type AbstractProposal end

"""
    propose(proposal, current_state)

Generate a candidate state from the proposal distribution.

# Arguments
- `rng::AbstractRNG`: random number generator
- `proposal::AbstractProposal`: proposal distribution
- `current_state::AbstractMCMCState`: current state

# Returns
- `candidate_state`: candidate state
"""
function propose end

"""
    log_ratio(proposal, current_state, proposed_state)

Compute the log proposal density ratio for Metropolis–Hastings.

# Arguments
- `proposal::AbstractProposal`: proposal distribution
- `current_state::AbstractMCMCState`: current state
- `proposed_state::AbstractMCMCState`: candidate state

# Returns
- `log_ratio::FT`: value of ``\\log q(\\theta'|\\theta) - \\log q(\\theta|\\theta')``
"""
function log_ratio end

# -----------------------------------------------------------------------------
# Planning Interface
# -----------------------------------------------------------------------------

# ---- Output Mode Trait ------------------------------------------------------

"""
    AbstractOutputMode

Trait controlling whether output variables `Y` are included in the OCP.

The output mode is stored in [`OptimalControlProblem`](@ref) and propagated
through workspace construction. It determines whether output decision
variables, observation constraints, and measurement-noise references are
created.
"""
abstract type AbstractOutputMode end

"""
    WithOutputs <: AbstractOutputMode

Include output variables `Y` and observation constraints in the OCP.

Use this mode when the cost or constraints depend on predicted outputs, or
when observation noise is part of the problem formulation.
"""
struct WithOutputs <: AbstractOutputMode end

"""
    WithoutOutputs <: AbstractOutputMode

Exclude output variables and observation constraints from the OCP.

The OCP operates on inputs `U` and states `X` only. Cost functions and
scenario constraints receive `nothing` in place of `Y`.

# Notes
- The terminal input ``u_{H+1}`` affects neither a state nor an output and is
  held at ``u_H`` by the OCP, so that the solver does not see a free variable
  without curvature.
"""
struct WithoutOutputs <: AbstractOutputMode end

# ---- Cost -------------------------------------------------------------------

"""
    AbstractCost

Abstract supertype for cost functions.

Concrete subtypes define how the cost of a single trajectory is evaluated
from input, state, and output trajectories.
"""
abstract type AbstractCost end

"""
    evaluate_cost(cost, u, x, y)

Evaluate the cost of a single trajectory.

# Arguments
- `cost::AbstractCost`: cost specification
- `u::AbstractMatrix`: input trajectory of size `(n_u, H + 1)`
- `x::AbstractMatrix`: state trajectory of size `(n_x, H + 1)`
- `y`: output trajectory of size `(n_y, H + 1)`, or `nothing` if the problem uses
  [`WithoutOutputs`](@ref)

# Returns
- `J`: scalar cost value

# Notes
- The arguments may contain numerical values or JuMP scalar expressions.
- When the problem uses [`WithoutOutputs`](@ref), `y` is `nothing`. Cost
  implementations that do not depend on outputs should accept this gracefully.
  Prefer [`FunctionalStateCost`](@ref) for costs that depend only on `u` and
  `x`.
- Scenario aggregation, such as mean, worst-case, or CVaR, is handled by the
  OCP formulation, not by this function.
"""
function evaluate_cost end

# ---- Constraints ------------------------------------------------------------

"""
    AbstractConstraint

Abstract supertype for OCP constraints.
"""
abstract type AbstractConstraint end

"""
    AbstractInputConstraint <: AbstractConstraint

Constraint depending only on the control input trajectory.

Input constraints are shared across scenarios and are therefore added only once
to the optimization problem.
"""
abstract type AbstractInputConstraint <: AbstractConstraint end

"""
    AbstractScenarioConstraint <: AbstractConstraint
 
Constraint depending on scenario trajectories.
 
Scenario constraints may involve states, outputs, and inputs, and are enforced
separately for each scenario.
 
When the problem uses [`WithoutOutputs`](@ref), scenario constraints receive
`nothing` in place of `Y_k`. Implementations that do not depend on outputs
should accept this; prefer [`FunctionalStateConstraint`](@ref) for constraints
that depend only on `u` and `x`.
"""
abstract type AbstractScenarioConstraint <: AbstractConstraint end

"""
    evaluate_constraint(constraint::AbstractInputConstraint, u)
    evaluate_constraint(constraint::AbstractScenarioConstraint, u, x, y)

Evaluate the constraint function ``h(\\cdot)`` and return its value(s) in
standard inequality form
```math
h \\leq 0.
```

# Arguments
- `constraint::AbstractConstraint`: constraint specification
- `u::AbstractMatrix`: input trajectory of size `(n_u, H + 1)`
- `x::AbstractMatrix`: state trajectory of size `(n_x, H + 1)`
- `y`: output trajectory of size `(n_y, H + 1)`, or `nothing` if the problem uses
  [`WithoutOutputs`](@ref)

# Returns
- `h`: constraint value(s); feasible iff all entries satisfy `h .<= 0`. May be
  a scalar for single constraints or a vector for multiple constraints.

# Notes
- For `AbstractInputConstraint`, only `u` is provided.
- For `AbstractScenarioConstraint`, `u`, `x`, and `y` are provided. When the
  problem uses [`WithoutOutputs`](@ref), `y` is `nothing`.
- Inputs and outputs may contain numerical values or JuMP scalar expressions.
"""
function evaluate_constraint end

# ---- Problem ----------------------------------------------------------------

"""
    OptimalControlProblem{OM}
 
Container for the horizon, cost, constraints, and output mode of an
optimal-control problem.

# Fields
- `cost::AbstractCost`: trajectory cost
- `input_constraints::Vector{AbstractInputConstraint}`: constraints depending
  only on the input trajectory
- `scenario_constraints::Vector{AbstractScenarioConstraint}`: constraints
  enforced separately for each scenario trajectory
- `H::Int`: prediction horizon ``H``. Trajectories span the ``H + 1`` time points
  ``t_0, \\dots, t_0 + H``, so inputs, states, and outputs have `H + 1` columns
  and the dynamics impose `H` transitions.
- `output_mode::OM`: [`WithOutputs`](@ref) or [`WithoutOutputs`](@ref)

# Notes
- The terminal input ``u_{H+1}`` (last column of `U`) does not enter the dynamics.
  It can only affect the problem through the output ``y_{H+1} = g(x_{H+1}, u_{H+1})``.
  Under [`WithoutOutputs`](@ref), the OCP therefore holds it at ``u_H``, and the cost and
  input constraints should not depend on it.
"""
struct OptimalControlProblem{OM<:AbstractOutputMode}
    cost::AbstractCost
    input_constraints::Vector{AbstractInputConstraint}
    scenario_constraints::Vector{AbstractScenarioConstraint}
    H::Int
    output_mode::OM
end

"""
    OptimalControlProblem(cost, input_constraints, scenario_constraints, H)
 
Construct an [`OptimalControlProblem`](@ref) with [`WithOutputs`](@ref) mode
(default).
"""
function OptimalControlProblem(
    cost::AbstractCost,
    input_constraints::Vector{AbstractInputConstraint},
    scenario_constraints::Vector{AbstractScenarioConstraint},
    H::Int,
)
    return OptimalControlProblem(cost, input_constraints, scenario_constraints, H, WithOutputs())
end

# ---- Formulation ------------------------------------------------------------

"""
    AbstractOCPFormulation

Abstract supertype for mathematical OCP formulations based on posterior samples.

A formulation specifies how the cost and constraints of an `OptimalControlProblem` are
combined across posterior samples (scenarios), for example using a 
worst-case or CVaR formulation.

Concrete subtypes are expected to store the underlying problem in a field
`problem::OptimalControlProblem`.
"""
abstract type AbstractOCPFormulation end

# ---- Solver -----------------------------------------------------------------

"""
    AbstractSolverConfig

Abstract supertype for OCP solver configurations.

Concrete subtypes specify which JuMP-compatible optimizer should be used and
which optimizer attributes should be applied when the JuMP model is built.
"""
abstract type AbstractSolverConfig end

# ---- Workspace --------------------------------------------------------------

"""
    AbstractOCPWorkspace

Abstract supertype for persistent JuMP-based optimal-control problem (OCP)
workspaces.

A workspace stores a JuMP representation of an OCP, including the JuMP model,
decision variables, constraints, and references to JuMP parameters.

The key idea is to separate:

- **problem structure** (variables, constraints), which is built once, and
- **numerical data** (e.g. posterior samples, disturbances), which can be 
  updated between solves.

This enables repeated solves (e.g. in MPC) without rebuilding the optimization
problem.
"""
abstract type AbstractOCPWorkspace end

"""
    build_ocp_workspace(solver, formulation, model, samples)

Construct a persistent JuMP workspace for an OCP.

This function builds the JuMP model for a given OCP formulation and set of
posterior samples. The resulting workspace stores the JuMP model, decision
variables, and parameter references needed for repeated solves.

Scenario-dependent data can be updated later with
[`update_scenarios!`](@ref) without rebuilding the JuMP model.

# Arguments
- `solver::AbstractSolverConfig`: solver configuration
- `formulation::AbstractOCPFormulation`: OCP formulation 
- `prediction_model::AbstractModel`: prediction model
- `samples::Vector{<:AbstractPosteriorSample}`: posterior samples defining the
  scenarios

# Returns
- `workspace::AbstractOCPWorkspace`: persistent, solver-specific representation
  of the OCP

# Notes
- The number of scenarios is fixed at construction time. If it changes, a new
  workspace must be created.
- Implementations should encode scenario-dependent quantities as parameters
  (e.g. JuMP `Parameter`s) to enable efficient updates.
"""
function build_ocp_workspace end

"""
    update_scenarios!(workspace, samples)

Replace the scenario data in an existing OCP workspace with fresh posterior
samples and new noise realizations.

This is the intended entry point between receding-horizon solves: it refreshes
numerical values without modifying the JuMP model structure, preserving the
solver's sparsity analysis.

# Arguments
- `workspace::AbstractOCPWorkspace`: existing OCP workspace
- `samples::Vector{<:AbstractPosteriorSample}`: new posterior samples

# Notes
- The length of `samples` must match the number of scenarios used when the
  workspace was constructed.
- Implementations should update parameter values in-place and avoid rebuilding
  the optimization model.
"""
function update_scenarios! end

# ---- Solution ---------------------------------------------------------------

"""
    SolverDiagnostics

Diagnostic information from an OCP solve.

Stores common solver diagnostics explicitly and additional solver- or
formulation-specific information in `info`.

# Fields
- `solve_time::Float64`: wall-clock time spent in the solver, in seconds
- `iterations::Int`: number of solver iterations
- `success::Bool`: `true` if the solver terminated with `LOCALLY_SOLVED`,
  `ALMOST_LOCALLY_SOLVED`, `ITERATION_LIMIT` or `TIME_LIMIT` and the returned
  iterate violates no constraint by more than the solver's
  `acceptable_constr_viol_tol`; `false` otherwise
- `info::Dict{Symbol,Any}`: additional diagnostic information, such as
  termination status codes, primal/dual statuses, or formulation-specific 
  quantities
"""
struct SolverDiagnostics
    solve_time::Float64
    iterations::Int
    success::Bool
    info::Dict{Symbol,Any}
end

"""
    OCPSolution{FT}

Result of one OCP solve.

Stores the optimized input trajectory, predicted state trajectories, optional
output trajectories, the objective value, the sample-average cost, and solver
diagnostics.

# Fields
- `U::Matrix{FT}`: optimized input trajectory, size `(n_u, H + 1)`
- `X::Array{FT,3}`: predicted state trajectories, size `(n_x, H + 1, K)`
- `Y::Union{Array{FT,3},Nothing}`: predicted output trajectories, size
  `(n_y, H + 1, K)`, or `nothing` if the problem uses [`WithoutOutputs`](@ref)
- `objective::FT`: value of the objective that was minimized. Its exact meaning
  is formulation-specific — the worst-case cost for the robust formulation, the
  sample-average cost plus the CVaR penalty for the CVaR formulation. May be
  `NaN` if no valid objective value is available
- `cost::FT`: sample-average cost over the scenarios, evaluated at the optimized
  input trajectory (i.e. the plain cost with no worst-case aggregation and no
  risk penalty). Computed identically across formulations, so it is directly
  comparable between e.g. robust and CVaR solutions. May be `NaN` if no valid
  solution is available
- `diagnostics::SolverDiagnostics`: solver and termination information
"""
struct OCPSolution{FT<:AbstractFloat}
    U::Matrix{FT}
    X::Array{FT,3}
    Y::Union{Array{FT,3},Nothing}
    objective::FT
    cost::FT
    diagnostics::SolverDiagnostics
end

"""
    solve_ocp!(workspace; U_init=nothing, warm_start=false)

Solve the OCP represented by `workspace`.

The solver is applied to the optimization problem stored in the workspace using
the currently stored numerical data, such as posterior samples and disturbance 
realizations.

If `U_init` is provided, it is used as an initial guess for the input trajectory.
Implementations may use this input guess to construct consistent initial values
for dependent variables, such as state and output trajectories.

# Arguments
- `workspace::AbstractOCPWorkspace`: OCP workspace with up-to-date numerical
  data

# Keywords
- `U_init`: optional initial guess for the input trajectory, typically of size
  `(n_u, H + 1)`
- `warm_start::Bool`: whether the solve warm-starts from the optimum of a
  closely related problem (e.g., after changing only the risk weight), in which
  case solver options tuned for warm starts are used

# Returns
- `solution::OCPSolution`: solution object containing optimized
  trajectories and solver diagnostics

# Notes
- Call [`update_scenarios!`](@ref) before solving if the scenario-dependent
  data has changed since the previous solve.
"""
function solve_ocp! end

# ---- Planner ----------------------------------------------------------------

"""
    AbstractPlanner

Abstract supertype for planners that synthesize a certified input trajectory.

Whereas [`solve_ocp!`](@ref) performs a single optimization, a planner
couples synthesis with a validation procedure and a scenario source. 
It repeatedly synthesizes and validates candidate inputs until the desired 
robustness guarantee is met or a retry budget is exhausted.

Concrete subtypes bundle the components they orchestrate (e.g. an OCP workspace,
a sampler for drawing fresh validation scenarios, and a calibration
configuration) and implement [`plan!`](@ref).
"""
abstract type AbstractPlanner end

"""
    plan!(planner::AbstractPlanner, model, data; kwargs...)

Run a planner's synthesis–certification procedure and return its result.

In contrast to [`solve_ocp!`](@ref), which solves a single optimization problem,
`plan!` drives the full adaptive procedure: synthesizing candidate inputs,
validating them on freshly drawn scenarios, and adjusting the formulation
parameters until a robustness guarantee is achieved or the retry budget is
exhausted.

# Arguments
- `planner::AbstractPlanner`: planner bundling an optimization engine, a
  validation procedure, and a scenario source
- `model::AbstractModel`: prediction/inference model
- `data::AbstractData`: current data

# Returns
- a planner-specific result describing the validated candidate (or the best
  candidate found) together with the achieved guarantee
"""
function plan! end