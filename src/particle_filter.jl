# =============================================================================
# particle_filter.jl — Sequential importance resampling particle filter
# =============================================================================
# Implements likelihood estimation and trajectory sampling for discrete-time
# stochastic state-space models using a sequential importance resampling (SIR)
# particle filter. The file contains:
#   - buffer allocation and storage-layout helpers,
#   - systematic resampling,
#   - particle filtering routines, and
#   - adaptive selection of the particle count for particle MCMC.
# =============================================================================

# -----------------------------------------------------------------------------
# Buffer and utilities for the particle filter
# -----------------------------------------------------------------------------

"""
    ParticleFilterBuffer{FT,TX} <: AbstractBuffer

Pre-allocated storage used by the particle filter.

The buffer stores the particle states, normalized weights, ancestry indices,
and temporary work arrays needed during propagation and weighting. It is
designed to be reused across repeated particle filter evaluations in order
to avoid repeated memory allocation.

Importantly, the buffer may be over-allocated: its capacity can exceed the
number of particles `N` and time steps `T` used in a particular run. This allows
reusing the same buffer without reallocation, e.g., when `N` is adapted.

# Fields
- `x_pf::TX`: Particle states over time; this is either a `(T_max, N_max)` matrix of `SVector`, 
    or a `(n_x, T_max, N_max)` array of `FT` for the general case.
- `w_pf::Matrix{FT}`: Normalized particle weights of size `(T_max, N_max)`.
- `a_pf::Matrix{Int}`: Ancestor indices of size `(T_max, N_max)`.
- `log_w::Vector{FT}`: Temporary storage for log weights at one time step.
- `v::Vector{FT}`: Temporary storage for one sample of process noise.
- `N_max::Int`: Maximum number of particles supported by this buffer.
"""
struct ParticleFilterBuffer{FT<:AbstractFloat,TX} <: AbstractBuffer
    x_pf::TX
    w_pf::Matrix{FT}
    a_pf::Matrix{Int}
    log_w::Vector{FT}
    v::Vector{FT}
    N_max::Int
end

"""
    allocate_pf_buffer(model, T, N, FT=Float64; use_static=...)

Allocate a `ParticleFilterBuffer` for a trajectory of length `T` and at most
`N` particles.

If `use_static=true`, particle states are stored as `SVector`s, which is often
faster for small state dimensions. Otherwise, states are stored in a dense
3-dimensional array.

# Arguments
- `model::DiscreteTimeStochasticSSM`: state-space model
- `T::Int`: number of time steps
- `N::Int`: maximum number of particles
- `FT::Type`: floating-point type (default: `Float64`)
- `use_static::Bool`: whether to store state vectors as `SVector`s.

# Returns
- `buffer::ParticleFilterBuffer`: a `ParticleFilterBuffer` with all required arrays pre-allocated
"""
function allocate_pf_buffer(
    model::DiscreteTimeStochasticSSM,
    T::Int,
    N::Int,
    ::Type{FT}=Float64;
    use_static::Bool=(model.n_x <= 10)
) where FT
    n_x = model.n_x

    # Allocate weight and ancestor buffers.
    w_pf = Matrix{FT}(undef, T, N)
    a_pf = Matrix{Int}(undef, T, N)
    log_w = Vector{FT}(undef, N)
    v = Vector{FT}(undef, model.n_v)

    # Allocate state buffer.
    if use_static
        # Static mode: Matrix of SVectors with dimension (T, N)
        x_pf = Matrix{SVector{n_x,FT}}(undef, T, N)
    else
        # General mode: 3D Array{FT} with dimension (n_x, T, N)
        x_pf = Array{FT,3}(undef, n_x, T, N)
    end

    return ParticleFilterBuffer{FT,typeof(x_pf)}(x_pf, w_pf, a_pf, log_w, v, N)
end

# Access helpers that hide the particle-state storage layout.
# In static mode, each state is stored directly as `x_pf[t, n]`.
# In dynamic mode, each state is stored as the slice `x_pf[:, t, n]`.
@inline get_x(buffer::ParticleFilterBuffer{FT,<:Array}, t, n) where FT = @view buffer.x_pf[:, t, n]
@inline get_x(buffer::ParticleFilterBuffer{FT,<:Matrix}, t, n) where FT = buffer.x_pf[t, n]

@inline set_x!(buffer::ParticleFilterBuffer{FT,<:Array}, val, t, n) where FT = (buffer.x_pf[:, t, n] .= val)
@inline set_x!(buffer::ParticleFilterBuffer{FT,<:Matrix}, val, t, n) where FT = (buffer.x_pf[t, n] = val)

"""
    systematic_resampling!(rng, a, w, N)

Fill `a[1:N]` with ancestor indices drawn using systematic resampling from the
discrete distribution defined by the normalized weights `w[1:N]`.

This method runs in `O(N)` time, performs no heap allocation, and typically
has lower Monte Carlo variance than multinomial resampling.

# Arguments
- `rng::AbstractRNG`: random number generator
- `a::AbstractVector{Int}`: output vector that receives the sampled ancestor indices
- `w::AbstractVector{FT}`: normalized nonnegative weights
- `N::Int`: number of ancestors to sample
"""
function systematic_resampling!(rng::AbstractRNG, a::AbstractVector{Int}, w::AbstractVector{FT}, N::Int) where FT
    # Pre-calculate the inverse to replace division with faster multiplication.
    inv_N = one(FT) / N

    # Generate a single random starting point in [0, 1/N].
    u = rand(rng, FT) * inv_N

    # Cumulative weight tracker
    c = w[1]
    i = 1

    @inbounds for n in 1:N
        # Calculate the fixed-interval target weight.
        target = u + (n - 1) * inv_N

        # Move through the weights until the cumulative sum hits the target.
        while c < target && i < N
            i += 1
            c += w[i]
        end

        # Assign the ancestor index.
        a[n] = i
    end
end

"""
    systematic_resampling(rng, w)

Sample a single index from the discrete distribution defined by the normalized
weights `w`.

This is a convenience method used, for example, when selecting the terminal
particle index for trajectory extraction.
"""
function systematic_resampling(rng::AbstractRNG, w::AbstractVector{FT}) where FT
    idx_buffer = Vector{Int}(undef, 1)

    systematic_resampling!(rng, idx_buffer, w, 1)

    return idx_buffer[1]
end

"""
    extract_trajectory(buffer, N, T)

Sample one latent state trajectory from a completed particle filter run by
drawing a terminal particle index from the final weights and tracing its
ancestral lineage backward in time.

# Arguments
- `rng::AbstractRNG`: random number generator
- `buffer::ParticleFilterBuffer`: buffer containing particle states, weights, and ancestry
- `N::Int`: number of particles used in the filter run
- `T::Int`: number of time steps

# Returns
- `trajectory::Matrix{FT}`: sampled state trajectory of size `(n_x, T)`
"""
function extract_trajectory(rng::AbstractRNG, buffer::ParticleFilterBuffer{FT}, N::Int, T::Int) where FT
    n_x = length(get_x(buffer, 1, 1))

    # Pick the final particle index based on weights at time T.
    idx = systematic_resampling(rng, view(buffer.w_pf, T, 1:N))

    trajectory = Matrix{FT}(undef, n_x, T)
    trajectory[:, T] .= get_x(buffer, T, idx)

    # Trace back through ancestors.
    for t in T-1:-1:1
        idx = buffer.a_pf[t+1, idx]
        trajectory[:, t] .= get_x(buffer, t, idx)
    end

    return trajectory
end

# -----------------------------------------------------------------------------
# Core particle filter algorithm
# -----------------------------------------------------------------------------

"""
    _particle_filter_inner!(rng, buffer, data, model, N, theta)

Internal implementation of the bootstrap particle filter.

This function assumes that the initial particle states at time `t = 1` have
already been written to `buffer`. It then iterates over the observations,
performing resampling, propagation, measurement weighting, and likelihood
accumulation.

Returns an estimate of the log marginal likelihood.
"""
function _particle_filter_inner!(
    rng::AbstractRNG,
    buffer::ParticleFilterBuffer{FT},
    data::DiscreteTimeIOData{FT},
    model::DiscreteTimeStochasticSSM,
    N::Int,
    theta::AbstractVector{FT}
) where FT

    # Initialize.
    T = size(data.y, 2)
    u, y = data.u, data.y
    log_likelihood = 0
    dist_v = model.get_dist_v(theta)
    dist_w = model.get_dist_w(theta)

    for t in 1:T
        if t >= 2
            # Resample.
            systematic_resampling!(rng, view(buffer.a_pf, t, 1:N), view(buffer.w_pf, t - 1, 1:N), N)

            @inbounds for n in 1:N
                parent_idx = buffer.a_pf[t, n]
                x_prev = get_x(buffer, t - 1, parent_idx)
                u_prev = @view u[:, t-1]

                # Sample noise.
                rand!(rng, dist_v, buffer.v)

                # Propagate.
                x_next = model.f(x_prev, u_prev, buffer.v, theta)
                set_x!(buffer, x_next, t, n)
            end
        end

        # Update weights based on measurement model.
        @inbounds for n in 1:N
            x_curr = get_x(buffer, t, n)
            u_curr = @view u[:, t]
            y_curr = @view y[:, t]
            y_hat = model.g(x_curr, u_curr, theta)

            # log[p(y_t | x_t)]
            buffer.log_w[n] = logpdf(dist_w, y_curr .- y_hat)
        end

        # Convert to linear weights and sum.
        max_log_w = maximum(view(buffer.log_w, 1:N))
        sum_w = zero(FT)
        @inbounds for n in 1:N
            w_val = exp(buffer.log_w[n] - max_log_w)
            buffer.w_pf[t, n] = w_val
            sum_w += w_val
        end

        @inbounds for n in 1:N
            buffer.w_pf[t, n] /= sum_w
        end

        # Accumulate log-likelihood log[p(y_1:T | theta)]
        log_likelihood += log(sum_w) + max_log_w - log(N)
    end

    return log_likelihood
end

"""
    particle_filter!(rng, buffer, data, model, N, theta, x0)

Run a bootstrap particle filter with a deterministic initial state.

All particles are initialized to the same state `x0` at time `t = 1`. The
function overwrites the contents of `buffer` and returns an estimate of the
log marginal likelihood.

# Arguments
- `rng::AbstractRNG`: random number generator
- `buffer::ParticleFilterBuffer`: pre-allocated particle filter storage
- `data::DiscreteTimeIOData`: observed data
- `model::DiscreteTimeStochasticSSM`: state-space model
- `N::Int`: number of particles; must satisfy `N <= buffer.N_max`
- `theta::AbstractVector`: model parameters
- `x0::AbstractVector{FT}`: deterministic initial state shared by all particles

# Returns
- `log_likelihood::FT`: estimated log-marginal likelihood
"""
function particle_filter!(
    rng::AbstractRNG,
    buffer::ParticleFilterBuffer{FT},
    data::DiscreteTimeIOData{FT},
    model::DiscreteTimeStochasticSSM,
    N::Int,
    theta::AbstractVector{FT},
    x0::AbstractVector{FT}
) where FT

    # Check if the workspace buffer can accommodate the requested number of particles.
    if N > buffer.N_max
        throw(ArgumentError("Requested N ($N) exceeds workspace capacity ($(buffer.N_max))."))
    end

    for n in 1:N
        set_x!(buffer, x0, 1, n)
    end

    return _particle_filter_inner!(rng, buffer, data, model, N, theta)
end

particle_filter!(buffer, data, model, N, theta, x0) =
    particle_filter!(Random.default_rng(), buffer, data, model, N, theta, x0)

"""
    particle_filter!(rng, buffer, data, model, N, theta; x0)

Run a bootstrap particle filter with initial particles sampled from the model's
initial-state distribution `model.get_dist_x0(theta)`.

The function overwrites the contents of `buffer` and returns an estimate of the
log marginal likelihood.

# Arguments
- `rng::AbstractRNG`: random number generator
- `buffer::ParticleFilterBuffer`: pre-allocated particle filter storage
- `data::DiscreteTimeIOData`: observed data
- `model::DiscreteTimeStochasticSSM`: state-space model
- `N::Int`: number of particles; must satisfy `N <= buffer.N_max`
- `theta::AbstractVector`: model parameters

# Returns
- `log_likelihood::FT`: estimated log-marginal likelihood
"""
function particle_filter!(
    rng::AbstractRNG,
    buffer::ParticleFilterBuffer{FT},
    data::DiscreteTimeIOData{FT},
    model::DiscreteTimeStochasticSSM,
    N::Int,
    theta::AbstractVector{FT}
) where FT

    # Check if the workspace buffer can accommodate the requested number of particles.
    if N > buffer.N_max
        throw(ArgumentError("Requested N ($N) exceeds workspace capacity ($(buffer.N_max))."))
    end

    dist_x0 = model.get_dist_x0(theta)
    for n in 1:N
        set_x!(buffer, rand(rng, dist_x0), 1, n)
    end

    return _particle_filter_inner!(rng, buffer, data, model, N, theta)
end

particle_filter!(buffer, data, model, N, theta) =
    particle_filter!(Random.default_rng(), buffer, data, model, N, theta)

"""
    particle_filter(rng, data, model, N, theta, args...)

Allocate a new particle filter buffer, run `particle_filter!`, and return the
stored particle states, normalized weights, and estimated log marginal likelihood.

This is a convenience wrapper around `allocate_pf_buffer` and `particle_filter!`.
"""
function particle_filter(rng::AbstractRNG, data, model, N, theta, args...)
    FT = eltype(theta)
    T = size(data.y, 2)
    buffer = allocate_pf_buffer(model, T, N, FT)

    # Run the particle filter
    log_likelihood = particle_filter!(rng, buffer, data, model, N, theta, args...)

    return buffer.x_pf, buffer.w_pf, log_likelihood
end

particle_filter(data, model, N, theta, args...) =
    particle_filter(Random.default_rng(), data, model, N, theta, args...)

# -----------------------------------------------------------------------------
# Mechanism for adapting the particle count N based on log-likelihood variance
# -----------------------------------------------------------------------------

"""
    NAdaptation <: StageAdaptation

Stage adaptation that adjusts the number of particles `N` in a particle filter
based on an empirical estimate of the variance of the log-likelihood estimator.

The adaptation assumes the scaling relation
```math
\\mathrm{Var}(\\log L) \\approx C / N,
```
and selects `N` such that the variance matches a desired target level.

At each stage, the variance is estimated by repeatedly evaluating the
log-likelihood at a subset of posterior samples using a fixed particle count.
The resulting estimate is used to propose a new particle count, which is then
clamped to `[N_min, N_max]`.

# Fields
- `N_min::Int`: lower bound used when adapting `N`
- `N_max::Int`: upper bound used when adapting `N`; also the maximum particle count that must be supported by the PF buffer
- `K_adapt::Int`: number of posterior samples to evaluate variance at
- `N_adapt::Int`: number of particles used in the variance-estimation PF runs
- `num_runs::Int`: number of PF repeats per sample index
- `target_var::FT`: target log-likelihood variance
- `verbose::Bool`: if `true`, print adaptation details
"""
struct NAdaptation{FT<:AbstractFloat} <: AbstractStageAdaptation
    N_min::Int
    N_max::Int
    K_adapt::Int
    N_adapt::Int
    num_runs::Int
    target_var::FT
    verbose::Bool
end

"""
    estimate_N!(workspace, stage_adaptation, samples, data, model)

Estimate the variance of the particle-filter log-likelihood estimator across a
subset of posterior samples and use it to propose a new particle count.

# Arguments
- `workspace::MCMCWorkspace`: workspace providing RNG and PF buffers
- `adaptation::NAdaptation{FT}`: adaptation configuration
- `samples::Vector{<:AbstractPosteriorSample}`: posterior samples from the previous stage
- `data::DiscreteTimeIOData{FT}`: observed data
- `model::DiscreteTimeStochasticSSM`: model definition

# Returns
- `N_suggested::Int`: proposed particle count (before clamping)
- `avg_var::FT`: estimated average variance of the log-likelihood
"""
function estimate_N!(
    workspace::MCMCWorkspace,
    adaptation::NAdaptation{FT},
    samples::Vector{<:AbstractPosteriorSample},
    data::DiscreteTimeIOData{FT},
    model::DiscreteTimeStochasticSSM,
) where FT

    # Check if K_adapt is larger than available samples.
    K = length(samples)
    K_adapt = min(adaptation.K_adapt, K)
    if adaptation.K_adapt > K
        @warn "K_adapt ($(adaptation.K_adapt)) is larger than available samples ($K). Using all samples."
    end

    log_likelihood_vars = FT[]
    indices = shuffle(workspace.rng, 1:K)[1:K_adapt]

    for k in indices
        # Compute variance of log-likelihood.
        log_likelihoods = zeros(adaptation.num_runs)
        theta_k = samples[k].theta
        for i in 1:adaptation.num_runs
            log_likelihoods[i] = estimate_log_likelihood(workspace.rng, workspace.buffer, data, model, adaptation.N_adapt, samples[k])
        end
        push!(log_likelihood_vars, var(log_likelihoods))
    end
    avg_var = mean(log_likelihood_vars)

    # Linear scaling rule: Var(log L) ≈ C / N
    # New N = Current N * (Current Var / Target Var)
    N_suggested = max(1, ceil(Int, adaptation.N_adapt * avg_var / adaptation.target_var))

    return N_suggested, avg_var
end

"""
    compute_update(adaptation::NAdaptation, sampler, workspace, samples, data, model, iteration)

Stage adaptation that updates the particle count `N`.

This method estimates the variance of the log-likelihood estimator and
proposes a new particle count such that the variance is close to
`adaptation.target_var`. The proposed value is clamped to
`[N_min, N_max]`.

# Returns
- `(N = N_new,)`: particle count to use in the next stage
"""
function compute_update(
    adaptation::NAdaptation,
    sampler::AbstractMCMCSampler,
    workspace::MCMCWorkspace,
    samples::Vector{<:AbstractPosteriorSample},
    data::AbstractData,
    model::AbstractModel,
    iteration::Int
)
    N_suggested, avg_var = estimate_N!(workspace, adaptation, samples, data, model)
    N_new = clamp(N_suggested, adaptation.N_min, adaptation.N_max)

    N_old = sampler.N

    if adaptation.verbose
        # Determine status for better feedback.
        status = if N_suggested > adaptation.N_max
            "▲ CAPPED (MAX)"
        elseif N_suggested < adaptation.N_min
            "▼ CAPPED (MIN)"
        elseif N_suggested == N_old
            "STABLE"
        else
            N_suggested > N_old ? "↗ INCREASING" : "↘ DECREASING"
        end
        @printf("  Particle Count Update: %4d → %-4d | %-15s | Var.: %6.3f (Target: %.1f)\n", N_old, N_new, status, avg_var, adaptation.target_var)
        flush(stdout)
    end

    return (N=N_new,)
end