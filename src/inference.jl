# =============================================================================
# inference.jl — Execution routines for running MCMC chains
# =============================================================================
# This file provides the core implementation of a single MCMC chain loop,
# including burn-in, thinning, sample collection, and optional step-wise
# adaptation of sampler parameters.
# =============================================================================

"""
    run_mcmc!(workspace, sampler, config, model, data, K)

Run a single MCMC chain using a pre-initialized workspace, collecting `K` samples.

The function advances the Markov chain in-place, performing burn-in,
thinning, and sample collection. The chain dynamics (burn-in, thinning, step
adaptations, verbosity) are taken from `config`; the number of samples `K` is a
separate argument. Optional step-wise adaptations are applied after each iteration.

# Arguments
- `workspace::MCMCWorkspace`: mutable workspace containing the current
  sampler state and buffers (updated in-place)
- `sampler::AbstractMCMCSampler`: MCMC sampler
- `config::MCMCConfig`: chain dynamics (burn-in length, thinning interval,
  step adaptations, verbosity)
- `model::AbstractModel`: model under consideration
- `data::AbstractData`: observed data
- `K::Int`: number of samples to collect

# Returns
- `ChainResult` with fields:
  - `samples`: vector of `K` posterior samples
  - `acceptance_rate::Float64`: ratio of accepted proposals over all iterations

# Behavior
- Performs MCMC transitions via `mcmc_step!`
- Discards the first `burnin` iterations
- Saves every `(thinning + 1)`-th sample thereafter
- Applies `StepAdaptation`s after each iteration
"""
function run_mcmc!(
    workspace::MCMCWorkspace,
    sampler::AbstractMCMCSampler,
    config::MCMCConfig,
    model::AbstractModel,
    data::AbstractData,
    K::Int
)

    if size(data.y, 2) != workspace.T
        @warn "Data length does not match Workspace T!"
        workspace.T = size(data.y, 2)
    end

    samples = allocate_samples(sampler, model, K, workspace.T)

    total_iters = 0
    accepted_steps = 0
    saved_samples = 0

    # Determine how often to print progress updates.
    print_interval = max(1, (K + config.burnin) ÷ 20)

    if config.verbose
        println("  ▶ Starting MCMC chain with K=$(K), burn-in=$(config.burnin), thinning=$(config.thinning)")
        flush(stdout)
    end

    while saved_samples < K
        total_iters += 1

        # Perform one MCMC step. The `mcmc_step!` function should return a tuple indicating whether the step was successful and whether it was accepted.
        success, accepted = mcmc_step!(sampler, workspace, model, data)

        if success && accepted
            accepted_steps += 1
        end

        # Burn-in and thinning logic
        if total_iters > config.burnin
            steps_since_burn = total_iters - config.burnin
            if (steps_since_burn - 1) % (config.thinning + 1) == 0
                saved_samples += 1

                # Record sample.
                samples[saved_samples] = create_sample(sampler, model, data, workspace)
            end
        end

        # Safety break
        if total_iters > K * 1000
            @error "Chain appears stuck. Total iterations exceeded K * 1000."
            break
        end

        for adaptation in config.step_adaptations
            update = compute_update(adaptation, sampler, accepted, total_iters)
            apply_update!(sampler, update)
        end

        if config.verbose && (total_iters % print_interval == 0)
            current_acc_rate = (accepted_steps / total_iters)
            status = total_iters <= config.burnin ? "Burn-in" : "Sampling"

            acc_color = if 0.20 <= current_acc_rate <= 0.35
                "\e[32m" # Green (healthy)
            elseif current_acc_rate < 0.10
                "\e[31m" # Red (stuck)
            else
                "\e[33m" # Yellow (sub-optimal)
            end

            reset = "\e[0m"

            @printf("\r\033[K  [%-8s] Iters: %7d | Saved: %5d/%d | Acc: %s%6.2f %%%s",
                status, total_iters, saved_samples, K, acc_color, current_acc_rate * 100, reset)
            flush(stdout)
        end
    end

    final_acceptance_rate = (accepted_steps / total_iters)

    # Print summary.
    if config.verbose
        @printf("\n  ✔ MCMC Chain Complete\n")
        flush(stdout)
    end

    return ChainResult(samples, final_acceptance_rate)
end
