# =============================================================================
# adaptations.jl — Adaptation mechanisms
# =============================================================================
# This file provides utilities for applying parameter updates to samplers
# and for orchestrating stage-wise adaptation in ensemble MCMC.
#
# Adaptations produce parameter updates as `NamedTuple`s, which are applied
# to samplers via `apply_update!`.
#
# The `adapt_stage!` routine aggregates updates from multiple
# `AbstractStageAdaptation`s based on collected samples and broadcasts them
# across the ensemble.
# =============================================================================

"""
    apply_update!(sampler, update::NamedTuple)

Apply a set of parameter updates to a sampler in-place.

Each entry of `update` is interpreted as a field assignment, i.e.,
for `(name = value,)` the function performs `sampler.name = value`.

# Arguments
- `sampler`: MCMC sampler to be updated
- `update::NamedTuple`: mapping from field names to new values

# Returns
- `sampler`: the updated sampler (mutated in-place)
"""
function apply_update!(sampler, update::NamedTuple)
    for name in keys(update)
        hasproperty(sampler, name) || error("Sampler has no property $(name)")
        setproperty!(sampler, name, getproperty(update, name))
    end
    return sampler
end

"""
    adapt_stage!(ensemble, stage_adaptations, samples, data, model, iteration)

Apply stage-wise adaptations to an ensemble of samplers.

Each adaptation is evaluated once using the aggregated `samples` from the
previous stage. The resulting updates are merged into a single `NamedTuple`
and applied to all samplers in the ensemble.

# Arguments
- `ensemble::MCMCEnsemble`: ensemble of samplers and workspaces
- `stage_adaptations::AbstractVector{<:AbstractStageAdaptation}`: adaptations to apply
- `samples::Vector{<:AbstractPosteriorSample}`: samples from the previous stage
- `data::AbstractData`: observed data
- `model::AbstractModel`: model under consideration
- `iteration::Int`: current stage index

# Behavior
- Each adaptation produces a `NamedTuple` of parameter updates via `compute_update`
- Updates are merged (later entries overwrite earlier ones on key conflicts)
- The merged update is applied to all samplers via `apply_update!`

# Notes
- The first sampler and workspace in the ensemble are used for any
  stochastic computations within adaptations
"""
function adapt_stage!(
    ensemble::MCMCEnsemble,
    stage_adaptations::AbstractVector{<:AbstractStageAdaptation},
    samples::Vector{<:AbstractPosteriorSample},
    data::AbstractData,
    model::AbstractModel,
    iteration::Int
)

    sampler = ensemble.samplers[1]
    workspace = ensemble.workspaces[1]

    update = merge(map(adaptation ->
            compute_update(adaptation, sampler, workspace, samples, data, model, iteration),
        stage_adaptations
    )...)

    for s in ensemble.samplers
        apply_update!(s, deepcopy(update))
    end
    return nothing
end
