# =============================================================================
# mh_step.jl — Implementation of the Metropolis–Hastings transition kernel
# =============================================================================
# This file defines the `mcmc_step!` method for `AbstractMHSampler`,
# including proposal generation, target density evaluation, and
# accept/reject logic for advancing the Markov chain.
# =============================================================================

"""
Implementation of a Metropolis–Hastings MCMC transition step.

This function provides the concrete realization of `mcmc_step!` for
`AbstractMHSampler`, including proposal generation, evaluation of the
target density, and acceptance/rejection based on the Metropolis–Hastings
criterion.
"""
function mcmc_step!(
    sampler::AbstractMHSampler,
    workspace::MCMCWorkspace,
    model::AbstractModel,
    data::AbstractData
)
    # Propose new state.
    proposed_state = propose(workspace.rng, sampler.proposal, workspace.state)

    # Evaluate log prior and log-likelihood of the proposed state.
    evaluate_state!(proposed_state, sampler, workspace, model, data)

    # Exit if out of support.
    if proposed_state.log_prior == -Inf
        return true, false
    end

    # Acceptance probability.
    log_acc_ratio = (proposed_state.log_likelihood + proposed_state.log_prior) -
        (workspace.state.log_likelihood + workspace.state.log_prior) +
        log_ratio(sampler.proposal, workspace.state, proposed_state)

    if log(rand(workspace.rng)) < log_acc_ratio
        accept_state!(sampler, workspace, proposed_state)
        return true, true
    end
    return true, false
end

# Default acceptance: the proposal becomes the current state.
function accept_state!(sampler::AbstractMHSampler, workspace::MCMCWorkspace, proposed_state::AbstractMCMCState)
    workspace.state = proposed_state
    return nothing
end

# Default re-evaluation: recompute the log-densities of the current state.
function reevaluate_state!(sampler::AbstractMHSampler, workspace::MCMCWorkspace, model::AbstractModel, data::AbstractData)
    evaluate_state!(workspace.state, sampler, workspace, model, data)
    return nothing
end
