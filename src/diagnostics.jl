# =============================================================================
# diagnostics.jl — MCMC sample analysis and diagnostics
# =============================================================================
# This file provides utilities for analyzing MCMC output.
#
# Samples are converted into matrix form for downstream processing, and standard
# diagnostics are computed, including autocorrelation, effective sample size (ESS),
# and Gelman–Rubin convergence statistics (R̂).
# =============================================================================

"""
    get_sample_matrix(samples::Vector{<:DiscreteTimeSSMSample})

Convert a vector of posterior samples into a matrix representation.

Each row corresponds to one sample, and columns correspond to the
concatenation of:
1. parameters returned by `extract_parameters`
2. the predicted state `x_t`

# Arguments
`samples::Vector{<:DiscreteTimeSSMSample}`: posterior samples

# Returns
- `sample_matrix::Matrix{FT}`: matrix of size `(K, n_vars)`

# Notes
- The dimensionality depends on the sampling mode (e.g. blocked vs standard).
"""
function get_sample_matrix(samples::Vector{<:DiscreteTimeSSMSample{FT}}) where {FT<:AbstractFloat}
    K = length(samples)
    # n_variables includes theta and x0 (if in blocked mode)
    n_params = length(extract_parameters(samples[1]))
    n_x_current = length(samples[1].x_current)

    sample_matrix = Matrix{FT}(undef, K, n_params + n_x_current)

    for k in 1:K
        sample_matrix[k, :] .= vcat(extract_parameters(samples[k]), samples[k].x_current)
    end
    return sample_matrix
end

"""
    compute_autocorrelation(samples; max_lag=100)

Compute the autocorrelation function (ACF) for each variable in the samples.

# Arguments
- `samples::Vector{<:DiscreteTimeSSMSample}`: posterior samples
- `max_lag::Int`: maximum lag to evaluate

# Returns
- `acf::Matrix{FT}`: matrix of size `(max_lag+1, n_vars)` where
  `acf[l, i]` is the autocorrelation at lag `l` for variable `i`.
"""
function compute_autocorrelation(samples::Vector{<:DiscreteTimeSSMSample{FT}}; max_lag::Int=100) where {FT<:AbstractFloat}
    sample_matrix = get_sample_matrix(samples)

    return autocor(sample_matrix, 0:max_lag; demean=true)
end

"""
    compute_ess(samples; max_lag=100)

Estimate the effective sample size (ESS) for each variable.

The ESS is computed as
```math
ESS = \\frac{K}{1 + 2 \\sum_{l=1}^{\\infty} \\rho_l},
```
where ``K`` is the total number of samples and ``\\rho_l`` 
is the autocorrelation at lag ``l``. The sum is truncated 
at the first negative autocorrelation or at max_lag.

# Arguments
- `samples::Vector{<:DiscreteTimeSSMSample}`: posterior samples
- `max_lag::Int`: maximum lag used in the autocorrelation sum

# Returns
- `ess::Vector{FT}`: ESS estimate for each variable
"""
function compute_ess(samples::Vector{<:DiscreteTimeSSMSample{FT}}; max_lag::Int=100) where {FT<:AbstractFloat}
    K = length(samples)
    acf = compute_autocorrelation(samples; max_lag=max_lag)
    n_vars = size(acf, 2)
    ess = zeros(FT, n_vars)

    for i in 1:n_vars
        # Sum autocorrelation until first negative or max_lag
        acf_sum = 0.0
        for lag in 1:max_lag
            if acf[lag+1, i] < 0
                break
            end
            acf_sum += acf[lag+1, i]
        end
        ess[i] = K / (1 + 2 * acf_sum)
    end

    return ess
end

"""
    compute_gelman_rubin(chains)

Compute the Gelman–Rubin convergence diagnostic (R̂) for each variable.

The statistic compares between-chain and within-chain variance across
multiple MCMC chains.

# Arguments
- `chains::Vector{<:Vector{<:DiscreteTimeSSMSample}}`: collection of MCMC chains

# Returns
- `R_hat::Vector{FT}`: R̂ values for each variable

# Notes
- R̂ values close to 1 (typically R̂ < 1.05) indicate convergence.
- Values significantly above 1 suggest lack of convergence.
"""
function compute_gelman_rubin(chains::Vector{<:Vector{<:DiscreteTimeSSMSample{FT}}}) where {FT<:AbstractFloat}
    M = length(chains)      # Number of chains
    K = length(chains[1])   # Number of samples per chain

    sample_matrix = get_sample_matrix(chains[1])
    n_vars = size(sample_matrix, 2)

    # Matrices to store means and variances per chain.
    chain_means = Matrix{FT}(undef, M, n_vars)
    chain_vars = Matrix{FT}(undef, M, n_vars)

    for m in 1:M
        mat = get_sample_matrix(chains[m])
        chain_means[m, :] .= vec(mean(mat, dims=1))
        chain_vars[m, :] .= vec(var(mat, dims=1, corrected=true))
    end

    # Between-chain variance (B)
    grand_mean = mean(chain_means, dims=1)
    B = K / (M - 1) * sum((chain_means .- grand_mean) .^ 2, dims=1)

    # Within-chain variance (W)
    W = mean(chain_vars, dims=1)

    # Pooled variance estimate (V_hat)
    V_hat = ((K - 1) / K) * W + (1 / K) * B

    # R-hat calculation
    R_hat = sqrt.(V_hat ./ W)

    return vec(max.(R_hat, 1.0))
end