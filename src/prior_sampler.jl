# =============================================================================
# prior_sampler.jl — Exact sampling from the prior, propagated under the inputs
# =============================================================================
# Implements a sampler that ignores the measurements and draws independent
# samples of the parameters and latent states from the prior, propagated through
# the stochastic dynamics under the applied inputs:
#
#   θ ~ p(θ),  x_0 ~ p(x_0 | θ),  x_{t+1} = f(x_t, u_t, v_t, θ),  v_t ~ p(v | θ).
#
# The result is a draw from p(θ, x_{0:T} | u_{0:T-1}), i.e., the "posterior"
# conditioned on the inputs only. It is the scenario source of the prior-based
# baseline, which runs the same synthesis–validation procedure as the proposed
# method but does not use the measurements to reduce the uncertainty.
#
# The sampler plugs into the MCMC interface (`build_mcmc_workspace`,
# `mcmc_step!`, `create_sample`, `allocate_samples`) so that it can be wrapped in
# an `MCMCEnsemble` and handed to the `AdaptiveCVaRPlanner` unchanged. Every step
# is an exact, independent draw that is always accepted, so the chain has no
# burn-in and no autocorrelation; run it with `MCMCConfig(0, 0)`.
# =============================================================================

"""
    PriorPredictiveSampler <: AbstractMCMCSampler

Exact sampler for the prior of the parameters and latent states, propagated
through the stochastic dynamics under the observed inputs.

Each call to `mcmc_step!` draws ``\\theta \\sim p(\\theta)`` and
``x_0 \\sim p(x_0 \\mid \\theta)`` and simulates the latent trajectory under the
inputs of the data set with fresh process noise. The measurements are ignored.
Consecutive draws are therefore independent and every step is accepted, so the
sampler should be run without burn-in or thinning (`MCMCConfig(0, 0)`).

# Notes
- The samples are of type `DiscreteTimeSSMSample{FT,Standard}` and can be used
  wherever posterior samples of a [`ParticleMMHSampler`](@ref) are expected, e.g.
  as OCP scenarios or as validation scenarios of an [`AdaptiveCVaRPlanner`](@ref).
- Their `log_likelihood` is zero, as no measurements are conditioned on.
"""
struct PriorPredictiveSampler <: AbstractMCMCSampler end

"""
    PriorPredictiveState{FT} <: AbstractMCMCState

Most recent draw of a [`PriorPredictiveSampler`](@ref).

# Fields
- `theta::Vector{FT}`: parameter vector ``\\theta``
- `x_history::Matrix{FT}`: latent state trajectory ``x_{0:T-1}`` of size `(n_x, T)`
- `log_prior::FT`: log-prior value ``\\log p(\\theta)``
"""
mutable struct PriorPredictiveState{FT<:AbstractFloat} <: AbstractMCMCState
    theta::Vector{FT}
    x_history::Matrix{FT}
    log_prior::FT
end

"""
    PriorPredictiveBuffer{FT} <: AbstractBuffer

Work storage of a [`PriorPredictiveSampler`](@ref).

# Fields
- `v::Vector{FT}`: storage for one process-noise draw
"""
struct PriorPredictiveBuffer{FT<:AbstractFloat} <: AbstractBuffer
    v::Vector{FT}
end

"""
    _draw_prior_trajectory!(state, buffer, rng, model, data)

Overwrite `state` with a fresh, independent draw of ``\\theta`` and ``x_{0:T-1}``.

# Arguments
- `state::PriorPredictiveState`: state to overwrite
- `buffer::PriorPredictiveBuffer`: work storage for the process noise
- `rng::AbstractRNG`: random number generator
- `model::DiscreteTimeStochasticSSM`: state-space model
- `data::DiscreteTimeIOData`: data whose inputs drive the simulation; the
  measurements are not used

# Returns
- `nothing`
"""
function _draw_prior_trajectory!(
    state::PriorPredictiveState{FT},
    buffer::PriorPredictiveBuffer{FT},
    rng::AbstractRNG,
    model::DiscreteTimeStochasticSSM,
    data::DiscreteTimeIOData{FT},
) where {FT}
    T = size(state.x_history, 2)
    u = data.u

    # θ ~ p(θ)
    rand!(rng, model.dist_theta, state.theta)
    state.log_prior = logpdf(model.dist_theta, state.theta)

    # Without data (T = 0) there is no trajectory to draw; the initial
    # state is drawn in `create_sample` as the state after the data.
    if T == 0
        return nothing
    end

    # x_0 ~ p(x_0 | θ)
    rand!(rng, model.get_dist_x0(state.theta), view(state.x_history, :, 1))

    # x_{t+1} = f(x_t, u_t, v_t, θ) with v_t ~ p(v | θ)
    dist_v = model.get_dist_v(state.theta)
    @inbounds for t in 2:T
        rand!(rng, dist_v, buffer.v)
        x_next = model.f(view(state.x_history, :, t - 1), view(u, :, t - 1), buffer.v, state.theta)
        state.x_history[:, t] .= x_next
    end

    return nothing
end

"""
    build_mcmc_workspace(rng, sampler::PriorPredictiveSampler, model, data; theta_start=nothing)

Initialize a workspace for a [`PriorPredictiveSampler`](@ref) and fill it with a
first draw.

# Arguments
- `rng::AbstractRNG`: random number generator owned by the workspace
- `sampler::PriorPredictiveSampler`: sampler instance
- `model::DiscreteTimeStochasticSSM`: state-space model
- `data::DiscreteTimeIOData`: data whose inputs drive the simulation

# Keywords
- `theta_start`: unused; accepted for interface compatibility. The sampler draws
  independent samples, so there is no chain state to initialize.

# Returns
- `workspace::MCMCWorkspace`: workspace holding the first draw
"""
function build_mcmc_workspace(
    rng::AbstractRNG,
    sampler::PriorPredictiveSampler,
    model::DiscreteTimeStochasticSSM,
    data::DiscreteTimeIOData{FT};
    theta_start=nothing
) where {FT}
    T = size(data.y, 2)
    state = PriorPredictiveState{FT}(
        Vector{FT}(undef, model.n_theta),
        Matrix{FT}(undef, model.n_x, T),
        zero(FT),
    )
    buffer = PriorPredictiveBuffer{FT}(Vector{FT}(undef, model.n_v))
    _draw_prior_trajectory!(state, buffer, rng, model, data)
    return MCMCWorkspace(state, buffer, rng, T)
end

"""
    mcmc_step!(sampler::PriorPredictiveSampler, workspace, model, data)

Replace the current state by a fresh, independent draw.

# Arguments
- `sampler::PriorPredictiveSampler`: sampler instance
- `workspace::MCMCWorkspace`: workspace (modified in place)
- `model::DiscreteTimeStochasticSSM`: state-space model
- `data::DiscreteTimeIOData`: data whose inputs drive the simulation

# Returns
- `(success, accepted) = (true, true)`: an exact draw is always valid and accepted
"""
function mcmc_step!(
    sampler::PriorPredictiveSampler,
    workspace::MCMCWorkspace,
    model::DiscreteTimeStochasticSSM,
    data::DiscreteTimeIOData,
)
    _draw_prior_trajectory!(workspace.state, workspace.buffer, workspace.rng, model, data)
    return true, true
end

"""
    allocate_samples(sampler::PriorPredictiveSampler, model, K, T)

Allocate an (uninitialized) container for `K` samples of a
[`PriorPredictiveSampler`](@ref). The element type follows the prior of the
parameters.

# Returns
- `samples::Vector{DiscreteTimeSSMSample{FT,Standard}}` of length `K`
"""
function allocate_samples(sampler::PriorPredictiveSampler, model::DiscreteTimeStochasticSSM, K::Int, T::Int)
    FT = float(eltype(model.dist_theta))
    return Vector{DiscreteTimeSSMSample{FT,Standard}}(undef, K)
end

"""
    create_sample(sampler::PriorPredictiveSampler, model, data, workspace)

Construct a sample from the current draw, propagating the final state one step
forward to the time ``T`` after the data, as for the PMMH samples. For an empty
data set (``T = 0``), the sample's current state is a fresh draw
``x_0 \\sim p(x_0 \\mid \\theta)``.
# Arguments
- `sampler::PriorPredictiveSampler`: sampler instance
- `model::DiscreteTimeStochasticSSM`: state-space model
- `data::DiscreteTimeIOData`: data whose inputs drive the simulation
- `workspace::MCMCWorkspace`: workspace holding the current draw

# Returns
- `sample::DiscreteTimeSSMSample{FT,Standard}`: sample with zero log-likelihood
"""
function create_sample(
    sampler::PriorPredictiveSampler,
    model::DiscreteTimeStochasticSSM,
    data::DiscreteTimeIOData{FT},
    workspace::MCMCWorkspace,
) where {FT}
    state = workspace.state
    T = size(state.x_history, 2)

    if T == 0
        # Without data, the state after the data is the initial state itself,
        # x_0 ~ p(x_0 | θ).
        x_next = rand(workspace.rng, model.get_dist_x0(state.theta))
    else
        # Propagate to x_T, the state at the time after the data.
        v = rand(workspace.rng, model.get_dist_v(state.theta))
        x_next = model.f(view(state.x_history, :, T), view(data.u, :, T), v, state.theta)
    end

    return DiscreteTimeSSMSample{FT,Standard}(
        copy(state.theta),
        Vector{FT}(x_next),
        copy(state.x_history),
        zero(FT),
        state.log_prior,
    )
end
