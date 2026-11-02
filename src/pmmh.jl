# =============================================================================
# pmmh.jl — Particle Marginal Metropolis-Hastings (PMMH) sampler
# =============================================================================
# This file implements a Particle Marginal Metropolis–Hastings (PMMH) sampler
# for discrete-time state-space models and plugs into the generic MH transition
# defined in `mh_step.jl`.
#
# The intractable likelihood p(y | θ) is replaced by an unbiased estimate
# obtained from a particle filter (see `particle_filter.jl`). This ensures that
# the resulting Markov chain targets the correct posterior distribution
# (Andrieu et al., 2010).
#
# Two sampling modes are supported via the `MCMCMode` trait:
#   Standard  — the proposal updates θ only; x₀ is sampled from p(x₀ | θ)
#   Blocked   — the proposal updates both θ and x₀
# =============================================================================

# -----------------------------------------------------------------------------
# PMMH Sampler
# -----------------------------------------------------------------------------

"""
    ParticleMMHSampler{Mode,P} <: AbstractMHSampler

Particle Marginal Metropolis–Hastings (PMMH) sampler.

This sampler performs Bayesian inference for state-space models by
embedding a particle filter within a Metropolis–Hastings scheme.
The intractable likelihood ``p(y \\mid \\theta)`` is replaced by an unbiased
estimate obtained from the particle filter, ensuring that the Markov
chain targets the correct posterior distribution.

# Fields
- `proposal::P`: proposal distribution used to generate candidate states
- `N::Int`: number of particles used in the particle filter
- `mode::Mode`: sampling mode (`Standard` or `Blocked`) determining the
  parameterization of the Markov chain

# Modes
- `Standard`: the proposal updates ``\\theta`` only; the initial state ``x_0`` is
  sampled internally from ``p(x_0 \\mid \\theta)``
- `Blocked`: the proposal updates both ``\\theta`` and ``x_0``
"""
mutable struct ParticleMMHSampler{Mode<:MCMCMode,P<:AbstractProposal} <: AbstractMHSampler
    proposal::P
    N::Int
    mode::Mode
end

function allocate_samples(sampler::ParticleMMHSampler{Mode}, model::DiscreteTimeStochasticSSM, K::Int, T::Int) where Mode
    FT = Float64

    # Initialize the vector
    samples = Vector{DiscreteTimeSSMSample{FT,Mode}}(undef, K)

    # Pre-allocate every single sample's internal arrays
    for k in 1:K
        samples[k] = DiscreteTimeSSMSample{FT,Mode}(
            Vector{FT}(undef, model.n_theta),
            Vector{FT}(undef, model.n_x),
            Matrix{FT}(undef, model.n_x, T),
            zero(FT),
            zero(FT)
        )
    end

    return samples
end

# -----------------------------------------------------------------------------
# Sampler States
# -----------------------------------------------------------------------------

"""
    ParticleMMHState{FT} <: AbstractMCMCState

Internal state of the PMMH sampler in standard mode.

# Fields
- `theta::Vector{FT}`: current parameter vector ``\\theta``
- `log_likelihood::FT`: estimated log-likelihood ``\\log p(y \\mid \\theta)``
- `log_prior::FT`: log-prior value ``\\log p(\\theta)``
- `x_history::Matrix{FT}`: latent state trajectory ``x_{0:T-1}`` of size
  `(n_x, T)`, drawn from the particle filter run that produced `log_likelihood`

# Notes
- In PMMH, the trajectory is part of the chain state (Andrieu et al., 2010): it
  is drawn when the state is accepted and kept while subsequent proposals are
  rejected. It is stored here rather than read from the particle-filter buffer,
  which is overwritten by every proposal, accepted or not.
- Proposed states carry an empty `(n_x, 0)` placeholder until they are accepted
  (see `accept_state!`).
"""
mutable struct ParticleMMHState{FT<:AbstractFloat} <: AbstractMCMCState
    theta::Vector{FT}
    log_likelihood::FT
    log_prior::FT
    x_history::Matrix{FT}
end

"""
    initialize_state(rng, buffer::ParticleFilterBuffer{FT}, sampler::ParticleMMHSampler{Standard}, model, data; theta_start)

Initialize the PMMH state in standard mode.

# Behavior
- Samples ``\\theta`` from the prior (or uses `theta_start` if provided)
- Estimates the log-likelihood via a particle filter
- Draws the latent trajectory from the same particle filter run
- Computes the log-prior ``\\log p(\\theta)``

# Returns
- `ParticleMMHState`
"""
function initialize_state(
    rng::AbstractRNG,
    buffer::ParticleFilterBuffer{FT},
    sampler::ParticleMMHSampler{Standard},
    model::DiscreteTimeStochasticSSM,
    data::DiscreteTimeIOData{FT};
    theta_start::Union{AbstractVector{FT},Nothing}=nothing
) where FT
    # Sample theta.
    theta = isnothing(theta_start) ? rand(rng, model.dist_theta) : copy(theta_start)

    # Estimate likelihood.
    log_likelihood = particle_filter!(rng, buffer, data, model, sampler.N, theta)
    x_history = extract_trajectory(rng, buffer, sampler.N, size(data.y, 2))

    log_prior = logpdf(model.dist_theta, theta)

    return ParticleMMHState{FT}(theta, log_likelihood, log_prior, x_history)
end

"""
    BlockedParticleMMHState{FT} <: AbstractMCMCState

Internal state of the PMMH sampler in blocked mode.

In addition to parameters ``\\theta``, the initial state ``x_0`` is 
included in the Markov chain

# Fields
- `theta::Vector{FT}`: current parameter vector ``\\theta``
- `x0::Vector{FT}`: current initial state ``x_0``
- `log_likelihood::FT`: estimated log-likelihood ``\\log p(y \\mid \\theta, x_0)``
- `log_prior::FT`: log-prior value ``\\log p(\\theta) + \\log p(x_0 \\mid \\theta)``
- `x_history::Matrix{FT}`: latent state trajectory ``x_{0:T-1}`` of size
  `(n_x, T)`, drawn from the particle filter run that produced `log_likelihood`;
  see [`ParticleMMHState`](@ref)
"""
mutable struct BlockedParticleMMHState{FT<:AbstractFloat} <: AbstractMCMCState
    theta::Vector{FT}
    x0::Vector{FT}
    log_likelihood::FT
    log_prior::FT
    x_history::Matrix{FT}
end

"""
    initialize_state(rng, buffer::ParticleFilterBuffer{FT}, sampler::ParticleMMHSampler{Blocked}, model, data; theta_start)

Initialize the PMMH state in blocked mode.

# Behavior
- Samples ``\\theta`` from the prior (or uses `theta_start`)
- Samples initial state ``x_0 \\sim p(x_0 \\mid \\theta)``
- Estimates the log-likelihood via a particle filter
- Draws the latent trajectory from the same particle filter run
- Computes the joint log-prior ``\\log p(\\theta) + \\log p(x_0 \\mid \\theta)``

# Returns
- `BlockedParticleMMHState`
"""
function initialize_state(
    rng::AbstractRNG,
    buffer::ParticleFilterBuffer{FT},
    sampler::ParticleMMHSampler{Blocked},
    model::DiscreteTimeStochasticSSM,
    data::DiscreteTimeIOData{FT};
    theta_start::Union{AbstractVector{FT},Nothing}=nothing
) where FT
    # Sample theta.
    theta = isnothing(theta_start) ? rand(rng, model.dist_theta) : copy(theta_start)

    # Sample x0.
    dist_x0 = model.get_dist_x0(theta)
    x0 = rand(rng, dist_x0)

    # Estimate likelihood.
    log_likelihood = particle_filter!(rng, buffer, data, model, sampler.N, theta, x0)
    x_history = extract_trajectory(rng, buffer, sampler.N, size(data.y, 2))

    log_prior_theta = logpdf(model.dist_theta, theta)
    log_prior_x0 = logpdf(dist_x0, x0)

    return BlockedParticleMMHState{FT}(theta, x0, log_likelihood, log_prior_theta + log_prior_x0, x_history)
end

# -----------------------------------------------------------------------------
# Workspace Initialization
# -----------------------------------------------------------------------------

"""
    build_mcmc_workspace(rng, sampler, stage_rules, model, data; theta_start)

Initialize an MCMC workspace with a particle filter buffer.
"""
function build_mcmc_workspace(
    rng::AbstractRNG,
    sampler::ParticleMMHSampler,
    model::DiscreteTimeStochasticSSM,
    data::DiscreteTimeIOData{FT};
    theta_start=nothing
) where FT
    buffer = allocate_pf_buffer(model, size(data.y, 2), sampler.N, FT)
    state = initialize_state(rng, buffer, sampler, model, data; theta_start=theta_start)
    return MCMCWorkspace(state, buffer, rng, size(data.y, 2))
end

function build_mcmc_workspace(
    sampler::ParticleMMHSampler,
    model::DiscreteTimeStochasticSSM,
    data::DiscreteTimeIOData{FT};
    theta_start=nothing
) where FT
    return build_mcmc_workspace(Random.default_rng(), sampler, model, data; theta_start=theta_start)
end

# Helper function to determine the maximum number of particles needed across all stage adaptations.
max_particles(stage_rules::AbstractVector{<:AbstractStageAdaptation}, sampler::ParticleMMHSampler) = max([adaptation.N_max for adaptation in stage_rules if adaptation isa NAdaptation]..., sampler.N)

"""
    build_mcmc_workspace(rng, sampler, stage_rules, model, data; theta_start, T_max)

Initialize an MCMC workspace with a particle filter buffer sized for
stage-wise adaptations.

The buffer is allocated using the maximum number of particles that any
`AbstractStageAdaptation` in `stage_rules` may request, and for `T_max` data
points.

# Notes
Use this method when particle count `N` may increase during staged
adaptation (e.g., `NAdaptation`). Pass `T_max` larger than the current data
length when the chain is later continued on a longer data set (e.g., in
receding-horizon control).
"""
function build_mcmc_workspace(
    rng::AbstractRNG,
    sampler::ParticleMMHSampler,
    stage_rules::AbstractVector{<:AbstractStageAdaptation},
    model::DiscreteTimeStochasticSSM,
    data::DiscreteTimeIOData{FT};
    theta_start=nothing,
    T_max::Int=size(data.y, 2)
) where FT
    @assert T_max >= size(data.y, 2) "T_max must be at least the current data length"
    N_max = max_particles(stage_rules, sampler)
    buffer = allocate_pf_buffer(model, T_max, N_max, FT)
    state = initialize_state(rng, buffer, sampler, model, data; theta_start=theta_start)
    return MCMCWorkspace(state, buffer, rng, size(data.y, 2))
end

function build_mcmc_workspace(
    sampler::ParticleMMHSampler,
    stage_rules::AbstractVector{<:AbstractStageAdaptation},
    model::DiscreteTimeStochasticSSM,
    data::DiscreteTimeIOData{FT};
    theta_start=nothing,
    T_max::Int=size(data.y, 2)
) where FT
    return build_mcmc_workspace(Random.default_rng(), sampler, stage_rules, model, data;
        theta_start=theta_start, T_max=T_max)
end

# -----------------------------------------------------------------------------
# Posterior Sample Construction
# -----------------------------------------------------------------------------

"""
    create_sample(sampler, model, data, workspace)

Construct a posterior sample from the current PMMH state.

# Behavior
- Copies the parameters and the latent state trajectory stored in the 
  current state of the Markov chain
- Propagates the final state one step forward to obtain ``x_{T+1}``

# Returns
- `DiscreteTimeSSMSample`
"""
function create_sample(sampler::ParticleMMHSampler{Mode}, model::DiscreteTimeStochasticSSM, data::DiscreteTimeIOData{FT}, workspace::MCMCWorkspace) where {FT,Mode}
    # Copy parameters and latent state trajectory.
    theta = copy(workspace.state.theta)
    x_history = copy(workspace.state.x_history)

    # Propagate to current time t.
    x_T = x_history[:, end]
    u_T = data.u[:, end]
    v = rand(workspace.rng, model.get_dist_v(theta))
    x_next = copy(model.f(x_T, u_T, v, theta))

    return DiscreteTimeSSMSample{FT,Mode}(
        theta,
        x_next,
        x_history,
        workspace.state.log_likelihood,
        workspace.state.log_prior
    )
end

# -----------------------------------------------------------------------------
# Target Evaluation
# -----------------------------------------------------------------------------

"""
    evaluate_state!(state, sampler::ParticleMMHSampler{Standard}, ...)

Evaluate log-prior and log-likelihood for a proposed state.

# Behavior
- Computes ``\\log p(\\theta)``
- If finite, estimates the log-likelihood via particle filtering
"""
function evaluate_state!(state::ParticleMMHState, sampler::ParticleMMHSampler{Standard}, workspace, model, data)
    state.log_prior = logpdf(model.dist_theta, state.theta)

    if state.log_prior > -Inf
        state.log_likelihood = particle_filter!(workspace.rng, workspace.buffer, data, model, sampler.N, state.theta)
    end
end

"""
    evaluate_state!(state, sampler::ParticleMMHSampler{Blocked}, ...)

Evaluate log-prior and log-likelihood for a proposed blocked state.

# Behavior
- Computes joint prior ``\\log p(\\theta) + \\log p(x_0 \\mid \\theta)``
- If finite, estimates the log-likelihood via particle filtering
"""
function evaluate_state!(state::BlockedParticleMMHState, sampler::ParticleMMHSampler{Blocked}, workspace, model, data)
    log_prior_theta = logpdf(model.dist_theta, state.theta)
    log_prior_x0 = logpdf(model.get_dist_x0(state.theta), state.x0)
    state.log_prior = log_prior_theta + log_prior_x0

    if state.log_prior > -Inf
        state.log_likelihood = particle_filter!(workspace.rng, workspace.buffer, data, model, sampler.N, state.theta, state.x0)
    end
end

"""
    accept_state!(sampler::ParticleMMHSampler, workspace, proposed_state)

Accept `proposed_state` as the new state of the chain and draw its latent
trajectory from the particle filter run that produced its likelihood estimate.

# Notes
- Must be called directly after [`evaluate_state!`](@ref) on `proposed_state`,
  while `workspace.buffer` still holds that run.
"""
function accept_state!(sampler::ParticleMMHSampler, workspace::MCMCWorkspace, proposed_state::AbstractMCMCState)
    proposed_state.x_history = extract_trajectory(workspace.rng, workspace.buffer, sampler.N, workspace.T)
    workspace.state = proposed_state
    return nothing
end

"""
    reevaluate_state!(sampler::ParticleMMHSampler, workspace, model, data)

Re-estimate the log-likelihood of the current state under `data`, e.g., after the
data set was extended, and redraw its latent trajectory from the same particle
filter run, so that trajectory and likelihood estimate stay consistent and the
trajectory covers the new data length.

# Arguments
- `sampler::ParticleMMHSampler`: sampler instance
- `workspace::MCMCWorkspace`: workspace whose current state is re-evaluated;
  `workspace.T` must already equal the length of `data`
- `model::DiscreteTimeStochasticSSM`: state-space model
- `data::DiscreteTimeIOData`: data the state is evaluated on

# Returns
- `nothing`
"""
function reevaluate_state!(sampler::ParticleMMHSampler, workspace::MCMCWorkspace, model::DiscreteTimeStochasticSSM, data::DiscreteTimeIOData)
    state = workspace.state
    evaluate_state!(state, sampler, workspace, model, data)
    state.x_history = extract_trajectory(workspace.rng, workspace.buffer, sampler.N, workspace.T)
    return nothing
end

"""
    estimate_log_likelihood(rng, buffer, data, model, N, sample)

Compute the log-likelihood estimate for a given posterior sample.

# Notes
Used by adaptation mechanisms (e.g., particle count adaptation)
to reassess likelihood estimates under different settings.
"""
function estimate_log_likelihood(rng::AbstractRNG, buffer::ParticleFilterBuffer{FT}, data::DiscreteTimeIOData{FT}, model::DiscreteTimeStochasticSSM, N::Int, sample::DiscreteTimeSSMSample{FT}) where FT
    return particle_filter!(rng, buffer, data, model, N, sample.theta)
end

function estimate_log_likelihood(rng::AbstractRNG, buffer::ParticleFilterBuffer{FT}, data::DiscreteTimeIOData{FT}, model::DiscreteTimeStochasticSSM, N::Int, sample::DiscreteTimeSSMSample{FT,Blocked}) where FT
    return particle_filter!(rng, buffer, data, model, N, sample.theta, sample.x_history[:, 1])
end