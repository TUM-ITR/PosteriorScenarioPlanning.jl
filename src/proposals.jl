# =============================================================================
# proposals.jl — Gaussian proposal and scale adaptation
# =============================================================================
# This file provides a Gaussian random-walk proposal for Metropolis–Hastings
# sampling together with adaptation mechanisms for tuning the proposal:
#
# - A stage-wise covariance adaptation based on empirical sample statistics
# - A Robbins–Monro step adaptation for adjusting the proposal scale to
#   achieve a desired acceptance rate
# =============================================================================

# -----------------------------------------------------------------------------
# Proposal
# -----------------------------------------------------------------------------
"""
    GaussianProposal{FT} <: AbstractProposal

Symmetric Gaussian random-walk proposal for Metropolis–Hastings sampling.

Given a current parameter vector `z`, the proposal generates a candidate
according to
```math
z' = z + \\varepsilon, \\qquad \\varepsilon \\sim \\mathcal{N}(0, \\texttt{scale} \\cdot \\Sigma).
```
For blocked samplers, ``z`` denotes the concatenated vector containing all
proposed components, i.e., ``z = [\\theta; x_0]``.

# Fields
- `Sigma::AbstractMatrix{FT}`: base covariance matrix of the proposal
- `scale::FT`: scalar multiplier applied to `Sigma`
"""
mutable struct GaussianProposal{FT,MT<:AbstractMatrix{FT}} <: AbstractProposal
    Sigma::MT
    scale::FT
end

"""
    propose(rng::AbstractRNG, p::GaussianProposal, current_state::ParticleMMHState{FT}) where FT

Generate a Gaussian random-walk proposal for a standard PMMH state.

The proposal perturbs ``\\theta`` and resets cached log-density values to `-Inf`.
"""
function propose(rng::AbstractRNG, p::GaussianProposal, current_state::ParticleMMHState{FT}) where FT
    theta_prop = current_state.theta + rand(rng, MvNormal(p.Sigma * p.scale))

    # The trajectory is drawn only if the proposal is accepted (see `accept_state!`).
    return ParticleMMHState{FT}(
        theta_prop,
        -Inf,
        -Inf,
        Matrix{FT}(undef, size(current_state.x_history, 1), 0)
    )
end

function propose(p::GaussianProposal, current_state::ParticleMMHState{FT}) where FT
    return propose(Random.default_rng(), p, current_state)
end

"""
    propose(rng::AbstractRNG, p::GaussianProposal, current_state::BlockedParticleMMHState{FT}) where FT

Generate a joint Gaussian random-walk proposal for a blocked PMMH state.

The proposal perturbs the concatenated vector ``(\\theta, x_0)`` and resets
cached log-density values to `-Inf`.
"""
function propose(rng::AbstractRNG, p::GaussianProposal, current_state::BlockedParticleMMHState{FT}) where FT
    # Proposes both theta and x0 together using a joint Gaussian proposal.
    n_theta = length(current_state.theta)
    combined_current = vcat(current_state.theta, current_state.x0)
    combined_prop = combined_current + rand(rng, MvNormal(p.Sigma * p.scale))
    theta_prop = combined_prop[1:n_theta]
    x0_prop = combined_prop[(n_theta+1):end]

    # The trajectory is drawn only if the proposal is accepted (see `accept_state!`).
    return BlockedParticleMMHState{FT}(
        theta_prop,
        x0_prop,
        -Inf,
        -Inf,
        Matrix{FT}(undef, size(current_state.x_history, 1), 0)
    )
end

function propose(p::GaussianProposal, current_state::BlockedParticleMMHState{FT}) where FT
    return propose(Random.default_rng(), p, current_state)
end

# Gaussian proposals are symmetric, so the log ratio is zero.
log_ratio(p::GaussianProposal{FT}, current_state::AbstractMCMCState, proposed_state::AbstractMCMCState) where {FT} = zero(FT)

# -----------------------------------------------------------------------------
# Covariance Adaptation
# -----------------------------------------------------------------------------
"""
    GaussianCovAdaptation
    
Stage-wise adaptation of the covariance matrix of a Gaussian proposal.

This adaptation updates the proposal covariance based on the empirical
covariance of collected samples.

# Behavior
- Computes the empirical covariance of the parameter samples
- Adds a diagonal regularization term for numerical stability
- Returns an updated `GaussianProposal`

# Fields
- `regularizer::FT`: diagonal regularization added to the covariance
- `verbose`: if `true`, print adaptation details

# Notes
- The covariance is estimated from `extract_parameters(samples)`, so its
  dimensionality depends on the sampling mode (e.g. blocked vs standard).
- This is a stage adaptation and should be used in conjunction with
  `adapt_stage!`.
"""
struct GaussianCovAdaptation{FT} <: AbstractStageAdaptation
    regularizer::FT
    verbose::Bool
end

"""
    compute_update(adaptation::GaussianCovAdaptation, sampler, workspace, samples, data, model, iteration)

Compute the stage-wise update of the Gaussian proposal.

# Returns
- `NamedTuple` with field `proposal`, containing the updated `GaussianProposal`

# Notes
- Uses the current `samples` to form the update.
- The scale is selected according to the current `iteration`.
"""
function compute_update(adaptation::GaussianCovAdaptation{FT}, sampler::AbstractMCMCSampler, workspace::MCMCWorkspace, samples::Vector{<:AbstractPosteriorSample}, data::AbstractData, model::AbstractModel, iteration::Int) where FT
    # Extract parameters from the MCMC samples.
    parameter_samples = hcat(extract_parameters.(samples)...)

    # Compute the empirical covariance of the parameter samples.
    emp_cov = cov(parameter_samples, dims=2; corrected=false)

    # Add regularization to ensure numerical stability.
    Sigma_new = emp_cov + adaptation.regularizer * I

    if adaptation.verbose
        avg_diag_old = mean(diag(sampler.proposal.Sigma))
        avg_diag_new = mean(diag(Sigma_new))
        @printf("  Covariance Update: Avg. Marginal Var. %5.2e → %5.2e\n", avg_diag_old, avg_diag_new)
        flush(stdout)
    end

    return (proposal=GaussianProposal(Sigma_new, sampler.proposal.scale),)
end

# -----------------------------------------------------------------------------
# Robbins–Monro Adaptation
# -----------------------------------------------------------------------------
"""
    RobbinsMonro{FT} <: AbstractStepAdaptation

Robbins–Monro step-size adaptation for the scale of a Gaussian proposal.

The adaptation updates the proposal scale in log-space at fixed batch
intervals in order to steer the empirical acceptance rate toward a target
value. If ``\\bar{\\alpha}_k`` denotes the acceptance rate in batch ``k``, the
update is
```math
\\log(\\sigma_{k+1}) = \\log(\\sigma_k) + \\gamma_k (\\bar{\\alpha}_k - \\alpha^*)
```
with gain sequence
```math
\\gamma_k = \\frac{C}{k^{\\kappa}}.
```

# Fields
- `target_accept::FT`: target acceptance rate ``\\alpha^*``
- `sub_chain_length::Int`: number of MCMC iterations per adaptation batch
- `step_size_scale::FT`: gain constant ``C``
- `step_size_decay::FT`: decay exponent ``\\kappa \\in (0.5, 1]``
- `accepted_in_batch::Int`: number of accepted proposals in the current batch
- `adaptation_count::Int`: number of adaptation steps performed so far
- `verbose::Bool`: if `true`, print diagnostics whenever an update is applied
"""
mutable struct RobbinsMonro{FT<:AbstractFloat} <: AbstractStepAdaptation
    target_accept::FT
    sub_chain_length::Int
    step_size_scale::FT
    step_size_decay::FT
    accepted_in_batch::Int
    adaptation_count::Int
    verbose::Bool
end

function RobbinsMonro(FT::Type{<:AbstractFloat}=Float64;
    target_accept=FT(0.234), sub_chain_length=100,
    step_size_scale=one(FT), step_size_decay=FT(0.5), verbose=false)
    return RobbinsMonro{FT}(target_accept, sub_chain_length, step_size_scale, step_size_decay, 0, 0, verbose)
end

"""
    compute_update(adaptation::RobbinsMonro, sampler::AbstractMCMCSampler, accepted::Bool, iteration::Int)

Compute a Robbins–Monro scale update from the current acceptance history.

An update is returned only at batch boundaries; otherwise an empty `NamedTuple`
is returned.
"""
function compute_update(adaptation::RobbinsMonro{FT}, sampler::AbstractMCMCSampler, accepted::Bool, iteration::Int) where {FT<:AbstractFloat}
    if accepted
        adaptation.accepted_in_batch += 1
    end

    # Execute only at batch boundaries.
    if iteration % adaptation.sub_chain_length != 0
        return NamedTuple()
    end

    current_accept = adaptation.accepted_in_batch / adaptation.sub_chain_length
    adaptation.accepted_in_batch = 0
    adaptation.adaptation_count += 1

    gamma = adaptation.step_size_scale / (FT(adaptation.adaptation_count)^adaptation.step_size_decay)

    old_scale = sampler.proposal.scale
    new_scale = exp(log(old_scale) + gamma * (FT(current_accept) - adaptation.target_accept))

    if adaptation.verbose
        @printf("\r\033[K  Robbins–Monro Update: %6.3f → %-6.3f | Batch Accept: %5.2f %% | Target: %5.2f %%\n",
            old_scale, new_scale, current_accept * 100, adaptation.target_accept * 100)
        flush(stdout)
    end

    return (proposal=GaussianProposal(sampler.proposal.Sigma, new_scale),)
end

"""
    reset!(adaptation::RobbinsMonro)

Reset the internal counters of the Robbins–Monro adaptation.
"""
function reset!(adaptation::RobbinsMonro)
    adaptation.accepted_in_batch = 0
    adaptation.adaptation_count = 0
    return nothing
end