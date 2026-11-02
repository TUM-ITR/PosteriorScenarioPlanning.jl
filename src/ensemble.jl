# =============================================================================
# ensemble.jl — Parallel and staged MCMC execution
# =============================================================================
# This file provides two complementary execution modes for MCMC:
#
# 1. Parallel multi-chain sampling:
#    Run multiple independent MCMC chains in parallel using separate
#    workspaces, sampler copies, and RNGs. This is useful for improving
#    sampling efficiency and for convergence diagnostics (e.g. R̂).
#
# 2. Staged inference:
#    Perform MCMC on progressively increasing subsets of the data.
#    After each stage, samples are aggregated across chains and used to
#    adapt shared sampler parameters before continuing.
#
# Both mechanisms can be combined to perform parallel staged MCMC.
# =============================================================================

# -----------------------------------------------------------------------------
# Parallel Multi-Chain Sampling
# -----------------------------------------------------------------------------
"""
    MCMCEnsemble{W<:MCMCWorkspace, S<:AbstractMCMCSampler}

Collection of independent MCMC chains.

Each chain owns its own workspace, sampler copy, and random number generator,
while sharing the same model, data, and configuration during execution.
Chains can therefore be run in parallel without sharing mutable sampler or
workspace state.

# Fields
- `workspaces::Vector{W}`: per-chain workspaces containing state, buffers, RNG, and data length
- `samplers::Vector{S}`: per-chain sampler copies
"""
struct MCMCEnsemble{W<:MCMCWorkspace,S<:AbstractMCMCSampler}
    workspaces::Vector{W}
    samplers::Vector{S}
end

Base.length(ensemble::MCMCEnsemble) = length(ensemble.workspaces)

"""
    initialize_ensemble(rng, sampler, model, data, n_chains=1; theta_start=nothing)
    initialize_ensemble(rng, sampler, stage_adaptations, model, data, n_chains=1; theta_start=nothing, T_max)

Initialize an ensemble of independent MCMC chains.

Each chain receives a separate RNG seeded from `rng`, a `deepcopy` of the
prototype `sampler`, and a freshly initialized workspace. The stage-adaptation
variant sizes workspaces for the largest resources required by the provided
stage adaptations, and for `T_max` data points.

# Arguments
- `rng::AbstractRNG`: parent RNG used to generate per-chain seeds
- `sampler::AbstractMCMCSampler`: prototype sampler (deep-copied per chain)
- `stage_adaptations::AbstractVector{<:AbstractStageAdaptation}` (optional): stage adaptations
- `model::AbstractModel`: model under consideration
- `data::AbstractData`: observed data
- `n_chains::Int`: number of independent chains
- `theta_start`: optional initial parameter vector (used for all chains)
- `T_max::Int` (stage-adaptation variant only): data length the buffers are sized
  for; defaults to the length of `data`. Larger values allow continuing the
  chains on a longer data set without reallocation.

# Returns
- `MCMCEnsemble`: initialized ensemble of chains
"""
function initialize_ensemble(
    rng::AbstractRNG,
    sampler::AbstractMCMCSampler,
    model::AbstractModel,
    data::AbstractData,
    n_chains::Int=1;
    theta_start=nothing
)
    seeds = rand(rng, UInt64, n_chains)
    rngs = [Xoshiro(seeds[i]) for i in 1:n_chains]
    samplers = [deepcopy(sampler) for _ in 1:n_chains]
    workspaces = Vector{MCMCWorkspace}(undef, n_chains)
    if n_chains == 1
        workspaces[1] = build_mcmc_workspace(rngs[1], samplers[1], model, data; theta_start=theta_start)
    else
        Threads.@threads for i in 1:n_chains
            workspaces[i] = build_mcmc_workspace(rngs[i], samplers[i], model, data; theta_start=theta_start)
        end
    end
    return MCMCEnsemble(workspaces, samplers)
end

function initialize_ensemble(
    rng::AbstractRNG,
    sampler::AbstractMCMCSampler,
    stage_adaptations::AbstractVector{<:AbstractStageAdaptation},
    model::AbstractModel,
    data::AbstractData,
    n_chains::Int=1;
    theta_start=nothing,
    T_max::Int=size(data.y, 2)
)
    seeds = rand(rng, UInt64, n_chains)
    rngs = [Xoshiro(seeds[i]) for i in 1:n_chains]
    samplers = [deepcopy(sampler) for _ in 1:n_chains]
    workspaces = Vector{MCMCWorkspace}(undef, n_chains)
    if n_chains == 1
        workspaces[1] = build_mcmc_workspace(rngs[1], samplers[1], stage_adaptations, model, data;
            theta_start=theta_start, T_max=T_max)
    else
        Threads.@threads for i in 1:n_chains
            workspaces[i] = build_mcmc_workspace(rngs[i], samplers[i], stage_adaptations, model, data;
                theta_start=theta_start, T_max=T_max)
        end
    end
    return MCMCEnsemble(workspaces, samplers)
end

"""
    run_mcmc!(ensemble, config, model, data, K)

Run one MCMC chain per ensemble member, collecting `K` samples in total across all
chains.

The requested count `K` is split as evenly as possible over the chains, with each
chain collecting `cld(K, n_chains)` samples (ceiling division). When `K` is not a
multiple of `n_chains` this rounds the total up, so slightly more than `K` samples
may be returned. With more than one chain, the chains are run thread-parallel.

# Arguments
- `ensemble::MCMCEnsemble`: collection of chains to execute
- `config::MCMCConfig`: chain dynamics (burn-in, thinning, step adaptations, verbosity)
- `model::AbstractModel`: model under consideration
- `data::AbstractData`: observed data
- `K::Int`: total number of samples to collect across all chains

# Returns
- `Vector{ChainResult}`: one [`ChainResult`](@ref) per chain, each containing:
  - `samples`: posterior samples (`cld(K, n_chains)` per chain)
  - `acceptance_rate`: acceptance rate of the chain

# Notes
- The grand total `cld(K, n_chains) * n_chains` is `≥ K`. Callers that need exactly
  `K` samples should flatten the per-chain results and take the first `K`.
"""
function run_mcmc!(
    ensemble::MCMCEnsemble,
    config::MCMCConfig,
    model::AbstractModel,
    data::AbstractData,
    K::Int
)
    n_chains = length(ensemble)
    K_per_chain = cld(K, n_chains)
    chain = Vector{ChainResult}(undef, n_chains)
    chain_configs = [deepcopy(config) for _ in 1:n_chains]

    # Run all chains thread parallel.
    if n_chains == 1
        chain[1] = run_mcmc!(ensemble.workspaces[1], ensemble.samplers[1], chain_configs[1], model, data, K_per_chain)
    else
        Threads.@threads for k in 1:n_chains
            chain[k] = run_mcmc!(ensemble.workspaces[k], ensemble.samplers[k], chain_configs[k], model, data, K_per_chain)
        end
    end

    return chain
end

# -----------------------------------------------------------------------------
# Staged Sampling
# -----------------------------------------------------------------------------

"""
    run_staged_mcmc!(ensemble, staged_config, stage_adaptations, model, data_full)

Run staged MCMC inference on an ensemble of chains.

The data window is increased by `staged_config.T_chunk` at each stage.
Within each stage, all chains run independently. Between stages, samples
are aggregated across chains and used to compute stage adaptations, which
are then applied to all samplers. Data-dependent adaptations (e.g. the particle
count) are evaluated on the data of the next stage, on which the adapted
sampler runs.

# Arguments
- `ensemble::MCMCEnsemble`: ensemble of chains
- `staged_config::StagedMCMCConfig`: configuration specifying:
  - intermediate-stage settings
  - final-stage settings
  - chunk size `T_chunk`
- `stage_adaptations::AbstractVector{<:AbstractStageAdaptation}`:
  adaptations applied between stages
- `model::AbstractModel`: model under consideration
- `data_full::AbstractData`: observed data (full length)

# Returns
- `NamedTuple` with fields:
  - `samples`: final-stage samples concatenated across all chains
  - `stage`: stage-wise results (`stage[i].chain[k]` is chain `k` at stage `i`)
  - `mean_acceptance`: mean acceptance rate across all chains and stages

# Notes
- Stage adaptations are shared across chains, while step adaptations remain
  chain-local.
"""
function run_staged_mcmc!(
    ensemble::MCMCEnsemble,
    staged_config::StagedMCMCConfig,
    stage_adaptations::AbstractVector{<:AbstractStageAdaptation},
    model::AbstractModel,
    data_full::AbstractData
)
    T_total = size(data_full.y, 2)
    N_stages = ceil(Int, T_total / staged_config.T_chunk)
    n_chains = length(ensemble)
    stage = Vector{NamedTuple}(undef, N_stages)

    if staged_config.verbose
        chain_label = n_chains == 1 ? "chain" : "chains"
        println("┏━ STAGED MCMC INFERENCE " * "━"^55)
        @printf("┃ %d stages | %d %s | +%d data points/stage | Total T: %d\n",
            N_stages, n_chains, chain_label, staged_config.T_chunk, T_total)
        println("┗" * "━"^79)
    end

    for i in 1:N_stages
        T_curr = min(i * staged_config.T_chunk, T_total)
        data_stage = subset_data(data_full, 1:T_curr)
        for ws in ensemble.workspaces
            ws.T = T_curr
        end

        if staged_config.verbose
            println("-"^80)
            @printf(" STAGE %02d/%02d (T = %d)\n", i, N_stages, T_curr)
            println("-"^80)
            flush(stdout)
        end

        # Re-evaluate the likelihood under the new data length for each chain.
        Threads.@threads for k in 1:n_chains
            ws = ensemble.workspaces[k]
            reevaluate_state!(ensemble.samplers[k], ws, model, data_stage)
        end

        cfg = i == N_stages ? staged_config.final_stage : staged_config.intermediate_stages
        K_stage = i == N_stages ? staged_config.K_final : staged_config.K_intermediate

        chain = run_mcmc!(ensemble, cfg, model, data_stage, K_stage)
        stage[i] = (chain=chain,)

        if i < N_stages
            # Update the sampler when stage is finished. 
            # The samples across chains are aggregated so the adaptation is informed by the collective posterior; 
            # the resulting update is broadcast to every chain.
            # Data-dependent adaptations are evaluated on the data of the next stage, on
            # which the adapted sampler runs: Var(log L̂) grows with the data length, so
            # a particle count tuned on the current stage would be too small for the next.
            all_samples = reduce(vcat, (c.samples for c in chain))
            T_next = min((i + 1) * staged_config.T_chunk, T_total)
            data_next = subset_data(data_full, 1:T_next)
            adapt_stage!(ensemble, stage_adaptations, all_samples, data_next, model, i)

            # Reset the step adapters.
            for adaptation in cfg.step_adaptations
                reset!(adaptation)
            end
        end
    end

    final_samples = reduce(vcat, (c.samples for c in stage[end].chain))
    mean_acceptance = mean(c.acceptance_rate for s in stage for c in s.chain)

    if staged_config.verbose
        println("┏━ Staged MCMC Inference Complete " * "━"^46)
        @printf("┃  Final Samples: %d\n", length(final_samples))
        @printf("┃  Mean Acc. Rate: %.2f %%\n", mean_acceptance * 100)
        println("┗" * "━"^79)
        flush(stdout)
    end

    return (samples=final_samples, stage=stage, mean_acceptance=mean_acceptance)
end

"""
    run_staged_mcmc!(rng, sampler, staged_config, stage_adaptations, model, data_full, n_chains=1; theta_start=nothing)
    run_staged_mcmc!(sampler, staged_config, stage_adaptations, model, data_full, n_chains=1; theta_start=nothing)

Convenience entry points for staged MCMC inference.

These methods:
- initialize an `MCMCEnsemble` of `n_chains` independent chains
- size internal buffers according to `stage_adaptations`
- dispatch to `run_staged_mcmc!(ensemble, ...)`

# Arguments
- `rng::AbstractRNG`: RNG used to seed per-chain RNGs (optional variant uses `Random.default_rng()`)
- `sampler::AbstractMCMCSampler`: prototype sampler (deep-copied per chain)
- `staged_config::StagedMCMCConfig`: configuration of staged inference
- `stage_adaptations::AbstractVector{<:AbstractStageAdaptation}`: between-stage adaptations
- `model::AbstractModel`: model used for inference
- `data_full::AbstractData`: full dataset
- `n_chains::Int`: number of parallel chains
- `theta_start`: optional initial parameter vector

# Returns
- Same as `run_staged_mcmc!(ensemble, ...)`
"""
function run_staged_mcmc!(
    rng::AbstractRNG,
    sampler::AbstractMCMCSampler,
    staged_config::StagedMCMCConfig,
    stage_adaptations::AbstractVector{<:AbstractStageAdaptation},
    model::AbstractModel,
    data_full::AbstractData,
    n_chains::Int=1;
    theta_start=nothing
)
    ensemble = initialize_ensemble(rng, sampler, stage_adaptations, model, data_full, n_chains; theta_start=theta_start)

    return run_staged_mcmc!(ensemble, staged_config, stage_adaptations, model, data_full)
end

function run_staged_mcmc!(
    sampler::AbstractMCMCSampler,
    staged_config::StagedMCMCConfig,
    stage_adaptations::AbstractVector{<:AbstractStageAdaptation},
    model::AbstractModel,
    data_full::AbstractData,
    n_chains::Int=1;
    theta_start=nothing
)
    return run_staged_mcmc!(
        Random.default_rng(),
        sampler,
        staged_config,
        stage_adaptations,
        model,
        data_full,
        n_chains;
        theta_start=theta_start
    )
end

