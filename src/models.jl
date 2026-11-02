# =============================================================================
# models.jl — Discrete-time stochastic state-space models
# =============================================================================
# Concrete implementations of the shared model interfaces for discrete-time
# stochastic state-space models. This file provides data containers, model
# definitions, posterior sample types, and JuMP scenario references.
#
# These types serve two roles:
# - Inference: the model evaluates likelihoods and samples latent trajectories
#   via PMCMC.
# - Planning: the same model adds dynamics constraints and scenario-dependent
#   parameters to JuMP-based OCP workspaces.
# =============================================================================

# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------

"""
    DiscreteTimeIOData{FT<:AbstractFloat} <: AbstractData

Container for input-output trajectories of a discrete-time system.

The inputs `u` and outputs `y` are stored as matrices with a shared
time dimension `T`, where each column corresponds to one time step.

# Fields
- `u::Matrix{FT}`: input trajectory of size `(n_u, T)`
- `y::Matrix{FT}`: output trajectory of size `(n_y, T)`
"""
struct DiscreteTimeIOData{FT<:AbstractFloat} <: AbstractData
    u::Matrix{FT}
    y::Matrix{FT}

    function DiscreteTimeIOData(u::Matrix{FT}, y::Matrix{FT}) where FT<:AbstractFloat
        if size(u, 2) != size(y, 2)
            throw(DimensionMismatch("Input 'u' and Output 'y' must have the same number of time steps (columns)."))
        end
        return new{FT}(u, y)
    end
end

subset_data(data::DiscreteTimeIOData, range::UnitRange{Int}) = DiscreteTimeIOData(data.u[:, range], data.y[:, range])

# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------

"""
    DiscreteTimeStochasticSSM <: AbstractModel

Discrete-time stochastic state-space model with parametric dynamics.

The model is defined by
```math
\\begin{aligned}
x_{t+1} &= f(x_t, u_t, v_t, \\theta), \\\\
y_t &= g(x_t, u_t, \\theta) + w_t,
\\end{aligned}
```
where ``x_t`` is the latent state, ``u_t`` is the input, 
``v_t`` is the process noise, ``w_t`` is the measurement noise, 
and ``\\theta`` are the model parameters.

# Fields
## Dimensions
- `n_u::Int`: input dimension
- `n_x::Int`: state dimension
- `n_y::Int`: output dimension
- `n_theta::Int`: parameter dimension
- `n_v::Int`: process noise dimension

## Dynamics
- `f`: state transition function
- `g`: measurement function

## Distributions (parameter-dependent)
Each of the following fields is a function that takes a parameter vector
`theta` and returns the corresponding distribution, so that the distributions
may depend on ``\\theta``:
- `get_dist_x0`: distribution of the initial state ``x_0``, i.e., ``p(x_0 \\mid \\theta)``
- `get_dist_v`: distribution of the process noise ``v_t``
- `get_dist_w`: distribution of the measurement noise ``w_t``

## Prior
- `dist_theta`: prior distribution over the parameters ``\\theta``

# Note
For any parameter vector ``\\theta``, the functions
`get_dist_x0(theta)`, `get_dist_v(theta)`, and `get_dist_w(theta)`
must return objects compatible with the `Distributions.jl` interface
(i.e., supporting `rand` and `logpdf`).
"""
struct DiscreteTimeStochasticSSM{F,G,Gx0,Gv,Gw,Dtheta} <: AbstractModel
    # Dimensions
    n_u::Int
    n_x::Int
    n_y::Int
    n_theta::Int
    n_v::Int

    # System Dynamics
    f::F
    g::G

    # Distribution getters (functions of theta)
    get_dist_x0::Gx0
    get_dist_v::Gv
    get_dist_w::Gw

    # Global parameter prior
    dist_theta::Dtheta

    # Inner constructor in case users want to pass fixed distributions instead of functions.
    # Fixed distributions are copied on access so parallel chains never share
    # distribution objects that may carry mutable sampler/cache state.
    function DiscreteTimeStochasticSSM(n_u, n_x, n_y, n_theta, n_v, f, g, dx0, dv, dw, dtheta)
        gx0 = dx0 isa Function ? dx0 : (theta) -> deepcopy(dx0)
        gv = dv isa Function ? dv : (theta) -> deepcopy(dv)
        gw = dw isa Function ? dw : (theta) -> deepcopy(dw)

        return new{typeof(f),typeof(g),typeof(gx0),typeof(gv),typeof(gw),typeof(dtheta)}(
            n_u, n_x, n_y, n_theta, n_v, f, g, gx0, gv, gw, dtheta
        )
    end
end

# -----------------------------------------------------------------------------
# Posterior samples
# -----------------------------------------------------------------------------

"""
    DiscreteTimeSSMSample{FT,Mode} <: AbstractPosteriorSample

Posterior sample for a discrete-time state-space model.

Represents a single draw from the joint posterior over model parameters
and latent states.

# Fields
- `theta::Vector{FT}`: model parameters ``\\theta``
- `x_current::Vector{FT}`: state at time `T+1` (prediction after the observed data)
- `x_history::Matrix{FT}`: latent state trajectory over the data window; of size `(n_x, T)`
- `log_likelihood::FT`: log-likelihood of the sample
- `log_prior::FT`: log-prior value of the sample
"""
struct DiscreteTimeSSMSample{FT<:AbstractFloat,Mode<:MCMCMode} <: AbstractPosteriorSample
    theta::Vector{FT}
    x_current::Vector{FT}
    x_history::Matrix{FT}
    log_likelihood::FT
    log_prior::FT
end

extract_parameters(s::DiscreteTimeSSMSample{FT,Standard}) where FT = s.theta
extract_parameters(s::DiscreteTimeSSMSample{FT,Blocked}) where FT = vcat(s.theta, s.x_history[:, 1])

# -----------------------------------------------------------------------------
# Simulation
# -----------------------------------------------------------------------------


"""
    _draw_process_noise(rng, model::DiscreteTimeStochasticSSM, samples, n_t)
 
Draw process-noise realizations for all scenarios and time steps.
 
# Arguments
- `rng::AbstractRNG`: random number generator
- `model::DiscreteTimeStochasticSSM`: dynamics model
- `samples::Vector{<:DiscreteTimeSSMSample}`: posterior samples
- `n_t::Int`: number of time points
 
# Returns
- `V::Array{FT,3}`: process-noise realizations of size `(n_v, n_t, K)`, where
  `FT` is the element type of the posterior parameter vectors
"""
function _draw_process_noise(
    rng::AbstractRNG,
    model::DiscreteTimeStochasticSSM,
    samples::Vector{<:DiscreteTimeSSMSample},
    n_t::Int,
)
    K = length(samples)
    FT = eltype(samples[1].theta)
    V = Array{FT}(undef, model.n_v, n_t, K)
    v_t = Vector{FT}(undef, model.n_v)

    @inbounds for k in 1:K
        dist_v = model.get_dist_v(samples[k].theta)
        for t in 1:n_t
            rand!(rng, dist_v, v_t)
            V[:, t, k] .= v_t
        end
    end

    return V
end

"""
    _draw_measurement_noise(rng, model::DiscreteTimeStochasticSSM, samples, n_t)
 
Draw measurement-noise realizations for all scenarios and time steps.
 
# Arguments
- `rng::AbstractRNG`: random number generator
- `model::DiscreteTimeStochasticSSM`: dynamics model
- `samples::Vector{<:DiscreteTimeSSMSample}`: posterior samples
- `n_t::Int`: number of time points
 
# Returns
- `W::Array{FT,3}`: measurement-noise realizations of size `(n_y, n_t, K)`,
  where `FT` is the element type of the posterior parameter vectors
"""
function _draw_measurement_noise(
    rng::AbstractRNG,
    model::DiscreteTimeStochasticSSM,
    samples::Vector{<:DiscreteTimeSSMSample},
    n_t::Int,
)
    K = length(samples)
    FT = eltype(samples[1].theta)
    W = Array{FT}(undef, model.n_y, n_t, K)
    w_t = Vector{FT}(undef, model.n_y)

    @inbounds for k in 1:K
        dist_w = model.get_dist_w(samples[k].theta)
        for t in 1:n_t
            rand!(rng, dist_w, w_t)
            W[:, t, k] .= w_t
        end
    end

    return W
end

"""
    simulate_trajectories(model::DiscreteTimeStochasticSSM, samples, u;
                          rng=Random.default_rng(), V=nothing, W=nothing)

Simulate posterior predictive trajectories for a discrete-time stochastic
state-space model.

Each trajectory is initialized from `samples[k].x_current` and propagated
forward using the sampled parameter vector `samples[k].theta` and the input
sequence `u`.

Pre-drawn noise realizations can be passed via the keyword arguments `V` and
`W`. Any noise keyword that is `nothing` (the default) is drawn internally
using `rng`. This allows callers to either:
- draw fresh noise (omit `V` and `W`), or
- replay specific realizations (pass concrete arrays), e.g. for warm-starting
  an OCP solve with noise consistent with the JuMP parameters.

# Arguments
- `model::DiscreteTimeStochasticSSM`: dynamics model
- `samples::Vector{<:DiscreteTimeSSMSample}`: posterior samples
- `u::AbstractMatrix{T}`: input trajectory of size `(n_u, n_t)` over `n_t` time
  points, e.g. `n_t = H + 1` for an OCP with horizon `H`

# Keywords
- `rng::AbstractRNG`: random number generator (default: `Random.default_rng()`)
- `V::Union{AbstractArray{<:Real,3},Nothing}`: process-noise realizations of
  size `(n_v, n_t, K)`. Drawn from `model.get_dist_v` when `nothing`.
- `W::Union{AbstractArray{<:Real,3},Nothing}`: measurement-noise realizations
  of size `(n_y, n_t, K)`. Drawn from `model.get_dist_w` when `nothing`.

# Returns
- `x::Array{T,3}`: state trajectories of size `(n_x, n_t, K)`
- `y::Array{T,3}`: output trajectories of size `(n_y, n_t, K)`
"""
function simulate_trajectories(
    model::DiscreteTimeStochasticSSM,
    samples::Vector{<:DiscreteTimeSSMSample},
    u::AbstractMatrix{T};
    rng::AbstractRNG=Random.default_rng(),
    V::Union{AbstractArray{<:Real,3},Nothing}=nothing,
    W::Union{AbstractArray{<:Real,3},Nothing}=nothing,
) where {T}
    K = length(samples)
    n_t = size(u, 2)

    # Draw missing noise up front
    if V === nothing
        V = _draw_process_noise(rng, model, samples, n_t)
    end
    if W === nothing
        W = _draw_measurement_noise(rng, model, samples, n_t)
    end

    x = Array{T}(undef, model.n_x, n_t, K)
    y = Array{T}(undef, model.n_y, n_t, K)

    @inbounds for k in 1:K
        theta = samples[k].theta

        # Initial state and observation
        x[:, 1, k] .= samples[k].x_current
        y[:, 1, k] .= model.g(x[:, 1, k], u[:, 1], theta) + view(W, :, 1, k)

        # Simulate forward
        for t in 2:n_t
            x[:, t, k] .= model.f(x[:, t-1, k], u[:, t-1], view(V, :, t - 1, k), theta)
            y[:, t, k] .= model.g(x[:, t, k], u[:, t], theta) + view(W, :, t, k)
        end
    end

    return x, y
end

# -----------------------------------------------------------------------------
# Planning
# -----------------------------------------------------------------------------

"""
    SSMScenarioRefs{W} <: AbstractScenarioRefs

JuMP parameter references for a discrete-time stochastic state-space model.

Stores handles to the scenario-dependent JuMP parameters used in optimal
control formulations based on posterior samples. Their numerical values are
updated between solves via [`_set_scenario_values!`](@ref) without rebuilding
the optimization problem.

# Fields
- `theta_refs::Matrix{VariableRef}`: parameter references, size `(n_theta, K)`
- `x_current_refs::Matrix{VariableRef}`: initial-state references, size `(n_x, K)`
- `V_refs::Array{VariableRef,3}`: process-noise references, size `(n_v, H + 1, K)`;
  the last time point is unused, as no transition follows it
- `W_refs::W`: measurement-noise references of size `(n_y, H + 1, K)`, or
  `nothing` if the problem uses [`WithoutOutputs`](@ref)

# Notes
- JuMP parameters created with `in Parameter(...)` have type `VariableRef`.
  Despite the name, these behave as fixed numerical parameters in the
  optimization problem, not as decision variables.
"""
struct SSMScenarioRefs{W<:Union{Array{VariableRef,3},Nothing}} <: AbstractScenarioRefs
    theta_refs::Matrix{VariableRef}
    x_current_refs::Matrix{VariableRef}
    V_refs::Array{VariableRef,3}
    W_refs::W
end

"""
    _add_scenario_refs!(jump_model, model::DiscreteTimeStochasticSSM,
                        output_mode, H, K; FT=Float64)
 
Add JuMP parameter references for a discrete-time stochastic state-space model.

Creates parameters for posterior model parameters, initial states, and process
noise for each scenario. Measurement-noise parameters are only created when
`output_mode` is [`WithOutputs`](@ref). All parameters are initialized to zero.

This is an internal function called by [`build_ocp_workspace`](@ref) during
workspace construction.

# Arguments
- `jump_model::JuMP.Model`: JuMP model to add parameters to
- `model::DiscreteTimeStochasticSSM`: dynamics model
- `output_mode::AbstractOutputMode`: controls whether measurement-noise
  parameters are created
- `H::Int`: prediction horizon; references are created for the `H + 1` time
  points of the horizon
- `K::Int`: number of scenarios

# Keyword Arguments
- `FT::Type{<:AbstractFloat}`: floating-point type for parameter
  initialization (default: `Float64`)

# Returns
- `refs::SSMScenarioRefs`: JuMP parameter references
"""
function _add_scenario_refs!(
    jump_model::JuMP.Model,
    model::DiscreteTimeStochasticSSM,
    output_mode::AbstractOutputMode,
    H::Int,
    K::Int;
    FT::Type{<:AbstractFloat}=Float64,
)
    @variable(jump_model, theta_refs[1:model.n_theta, 1:K] in Parameter(zero(FT)))
    @variable(jump_model, x_current_refs[1:model.n_x, 1:K] in Parameter(zero(FT)))
    @variable(jump_model, V_refs[1:model.n_v, 1:(H+1), 1:K] in Parameter(zero(FT)))

    W_refs = if output_mode isa WithOutputs
        @variable(jump_model, W_refs[1:model.n_y, 1:(H+1), 1:K] in Parameter(zero(FT)))
    else
        nothing
    end

    return SSMScenarioRefs(theta_refs, x_current_refs, V_refs, W_refs)
end

"""
    _set_scenario_values!(refs::SSMScenarioRefs,
                          model::DiscreteTimeStochasticSSM,
                          output_mode, samples, rng)

Update scenario-dependent JuMP parameter values for a stochastic state-space
model.

Sets posterior parameters and initial states from `samples`, and draws fresh
process-noise realizations for each scenario and time step. Measurement-noise
realizations are only drawn when `output_mode` is [`WithOutputs`](@ref).

This is an internal function called by [`update_ocp!`](@ref) before each solve.

# Arguments
- `refs::SSMScenarioRefs`: JuMP parameter references created by
  [`_add_scenario_refs!`](@ref)
- `model::DiscreteTimeStochasticSSM`: dynamics model
- `output_mode::AbstractOutputMode`: controls whether measurement noise is
  drawn
- `samples::Vector{<:DiscreteTimeSSMSample}`: posterior samples
- `rng::AbstractRNG`: random number generator

# Notes
- Only JuMP parameter values are updated; the optimization structure is not
  modified.
"""
function _set_scenario_values!(
    refs::SSMScenarioRefs,
    model::DiscreteTimeStochasticSSM,
    output_mode::AbstractOutputMode,
    samples::Vector{<:DiscreteTimeSSMSample},
    rng::AbstractRNG,
)
    K = size(refs.theta_refs, 2)
    n_t = size(refs.V_refs, 2) # H + 1 time points

    if length(samples) != K
        throw(ArgumentError(
            "Expected $K posterior samples to match the number of scenarios " *
                "in the workspace, but received $(length(samples)). The number " *
                "of scenarios is fixed at construction; build a new workspace to " *
                "change it."
        ))
    end

    FT = eltype(samples[1].theta)
    v_t = Vector{FT}(undef, model.n_v)

    has_outputs = output_mode isa WithOutputs
    w_t = if has_outputs
        Vector{FT}(undef, model.n_y)
    else
        nothing
    end

    @inbounds for k in 1:K
        theta_k = samples[k].theta

        # Posterior parameters
        for i in 1:model.n_theta
            set_parameter_value(refs.theta_refs[i, k], theta_k[i])
        end

        # Initial state
        for i in 1:model.n_x
            set_parameter_value(refs.x_current_refs[i, k], samples[k].x_current[i])
        end

        # Draw process noise realizations
        dist_v = model.get_dist_v(theta_k)
        for t in 1:n_t
            rand!(rng, dist_v, v_t)
            for i in 1:model.n_v
                set_parameter_value(refs.V_refs[i, t, k], v_t[i])
            end
        end

        # Draw measurement noise realizations (only with outputs)
        if has_outputs
            dist_w = model.get_dist_w(theta_k)
            for t in 1:n_t
                rand!(rng, dist_w, w_t)
                for i in 1:model.n_y
                    set_parameter_value(refs.W_refs[i, t, k], w_t[i])
                end
            end
        end
    end

    return nothing
end


"""
    _extract_noise_values(refs::SSMScenarioRefs, output_mode::AbstractOutputMode)

Extract current noise realizations from the JuMP parameter references.

Returns the process-noise and measurement-noise arrays with the numerical
values currently stored in the JuMP parameters. These can be passed to
[`simulate_trajectories`](@ref) to produce initial guesses that are consistent
with the noise realizations encoded in the OCP.

# Arguments
- `refs::SSMScenarioRefs`: parameter references with current values
- `output_mode::AbstractOutputMode`: controls whether measurement noise is
  extracted

# Returns
A `NamedTuple` with fields:
- `V::Array{Float64,3}`: process-noise values, size `(n_v, H + 1, K)`
- `W::Union{Array{Float64,3},Nothing}`: measurement-noise values, size
  `(n_y, H + 1, K)`, or `nothing` if `output_mode` is [`WithoutOutputs`](@ref)
"""
function _extract_noise_values(
    refs::SSMScenarioRefs,
    output_mode::AbstractOutputMode,
)
    V = parameter_value.(refs.V_refs)
    W = if output_mode isa WithOutputs && refs.W_refs !== nothing
        parameter_value.(refs.W_refs)
    else
        nothing
    end
    return (V=V, W=W)
end

"""
    _add_dynamics_constraints!(jump_model, model::DiscreteTimeStochasticSSM,
                               refs, X, U, H, K)
 
Add initial-state and state-transition constraints for a discrete-time
stochastic state-space model.

The constraints encode
```math
x_{0}^{[k]}  = x_{\\mathrm{current}}^{[k]}, \\\\
x_{t+1}^{[k]} = f(x_t^{[k]}, u_t, v_t^{[k]}, \\theta^{[k]}),
    \\quad t \\in 0{:}H{-}1,
```
for all scenarios ``k \\in 1{:}K``.

# Arguments
- `jump_model::JuMP.Model`: JuMP model to add constraints to
- `model::DiscreteTimeStochasticSSM`: dynamics model providing `f` and `n_x`
- `refs::SSMScenarioRefs`: parameter references from
  [`_add_scenario_refs!`](@ref)
- `X::Array{JuMP.VariableRef,3}`: state variables of size `(n_x, H + 1, K)`
- `U::Matrix{JuMP.VariableRef}`: input variables of size `(n_u, H + 1)`
- `H::Int`: prediction horizon
- `K::Int`: number of scenarios

# Notes
- Initial-state constraints pin `X[:, 1, k]` to the references in
  `refs.x_current_refs[:, k]`.
- `H` transitions are imposed, so the terminal state ``x_H`` is determined by
  ``x_{H-1}`` and ``u_{H-1}`` but not propagated further; ``u_H`` and ``v_H``
  do not enter the dynamics.
- Constraints use parameter references rather than numerical values, so they
  remain valid across parameter updates.
"""
function _add_dynamics_constraints!(
    jump_model::JuMP.Model,
    model::DiscreteTimeStochasticSSM,
    refs::SSMScenarioRefs,
    X::Array{JuMP.VariableRef,3},
    U::Matrix{JuMP.VariableRef},
    H::Int,
    K::Int,
)
    # Initial-state constraints
    @constraint(
        jump_model,
        [i = 1:model.n_x, k = 1:K],
        X[i, 1, k] == refs.x_current_refs[i, k]
    )

    # Transition constraints
    for k in 1:K
        theta_k = view(refs.theta_refs, :, k)
        for t in 1:H
            x_t = view(X, :, t, k)
            u_t = view(U, :, t)
            v_t = view(refs.V_refs, :, t, k)

            x_next = model.f(x_t, u_t, v_t, theta_k)
            @constraint(
                jump_model,
                [i = 1:model.n_x],
                X[i, t+1, k] == x_next[i]
            )
        end
    end

    return nothing
end

"""
    _add_observation_constraints!(jump_model, model::DiscreteTimeStochasticSSM,
                                  refs, X, U, Y, H, K)

Add observation-equation constraints for a discrete-time stochastic SSM.

The constraints encode
```math
y_t^{[k]} = g(x_t^{[k]}, u_t, \\theta^{[k]}) + w_t^{[k]},
    \\quad t \\in 0{:}H,
```
for all scenarios ``k \\in 1{:}K``.

This function is only called when the problem uses [`WithOutputs`](@ref).

# Arguments
- `jump_model::JuMP.Model`: JuMP model to add constraints to
- `model::DiscreteTimeStochasticSSM`: dynamics model providing `g` and `n_y`
- `refs::SSMScenarioRefs`: parameter references (must include `W_refs`)
- `X::Array{JuMP.VariableRef,3}`: state variables of size `(n_x, H + 1, K)`
- `U::Matrix{JuMP.VariableRef}`: input variables of size `(n_u, H + 1)`
- `Y::Array{JuMP.VariableRef,3}`: output variables of size `(n_y, H + 1, K)`
- `H::Int`: prediction horizon
- `K::Int`: number of scenarios
"""
function _add_observation_constraints!(
    jump_model::JuMP.Model,
    model::DiscreteTimeStochasticSSM,
    refs::SSMScenarioRefs,
    X::Array{JuMP.VariableRef,3},
    U::Matrix{JuMP.VariableRef},
    Y::Array{JuMP.VariableRef,3},
    H::Int,
    K::Int,
)
    for k in 1:K
        theta_k = view(refs.theta_refs, :, k)
        for t in 1:(H+1)
            x_t = view(X, :, t, k)
            u_t = view(U, :, t)
            w_t = view(refs.W_refs, :, t, k)

            y_t = model.g(x_t, u_t, theta_k)
            @constraint(
                jump_model,
                [i = 1:model.n_y],
                Y[i, t, k] == y_t[i] + w_t[i]
            )
        end
    end

    return nothing
end