# =============================================================================
# sampler_tuning.jl: Tuning and diagnostics of the PMMH sampler
# =============================================================================
# Runs a few long PMMH chains on the posterior of the planning experiments and
# evaluates posterior predictive performance and convergence diagnostics. Model,
# priors, data generation and sampler settings are taken from `setup.jl`, so the
# sampler is tuned for exactly the configuration used for planning. Only the
# run length differs: instead of one sample from each of many chains, a few
# long chains are run so that mixing becomes visible.
# =============================================================================

# Reuse the shared setup (model, priors, data generation, sampler and adaptation
# settings).
include(joinpath(@__DIR__, "setup.jl"))

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
const n_chains_tuning = 10    # number of parallel chains
const K_tuning = Int(1e6)     # number of samples across all chains
const thinning_tuning = 0     # no thinning, so that the autocorrelation is visible
n_pred = 1000                 # number of posterior samples used for the predictive check
max_lag = 500                 # maximum lag of the autocorrelation plot
bins = 50                     # number of histogram bins of the posterior marginals
n_trace_points = 2000         # approximate number of plotted iterations per chain in the trace plots
plot_data = true              # whether to plot the generated training data

final_stage_config_tuning = MCMCConfig(burnin, thinning_tuning;
    step_adaptations=step_adaptations, verbose=false)
staged_config_tuning = StagedMCMCConfig(intermediate_stage_config, final_stage_config_tuning, T_chunk;
    K_intermediate=K_intermediate, K_final=K_tuning, verbose=true)

# Number of inferred variables: theta and x0 for the blocked sampler, theta otherwise.
if use_blocked_sampler
    n_inferred = n_theta + n_x
    prior_marginals = vcat(theta_marginals, x0_marginals)
else
    n_inferred = n_theta
    prior_marginals = theta_marginals
end

"""
    variable_info(i)

Name and unit of the `i`-th column of the sample matrix: the parameters, then the
initial state (blocked sampler only), then the state after the observed data.

# Arguments
- `i::Int`: column index of the sample matrix.

# Returns
- `Tuple{String,String}`: name and unit of the variable.
"""
function variable_info(i)
    if i <= n_theta
        return parameter_info[i]
    elseif i <= n_inferred
        name, unit = state_info[i-n_theta]
        return ("$name at \$t=0\$", unit)
    else
        name, unit = state_info[i-n_inferred]
        return ("$name at \$t=t_0\$", unit)
    end
end

"""
    grid_figure(panels, heading)

Combine `panels` into one figure on an approximately square grid with a common
heading. Unused grid cells are filled with empty panels.

# Arguments
- `panels::Vector`: subplots, in row-major order.
- `heading::String`: common heading of the figure.

# Returns
- `Plots.Plot`: the combined figure.
"""
function grid_figure(panels, heading)
    n_panels = length(panels)
    n_cols = ceil(Int, sqrt(n_panels))
    n_rows = ceil(Int, n_panels / n_cols)
    all_panels = copy(panels)
    for _ in (n_panels+1):(n_rows*n_cols)
        push!(all_panels, plot(framestyle=:none))
    end
    return plot(all_panels...; layout=(n_rows, n_cols), size=(300 * n_cols, 250 * n_rows),
        margin=5Plots.mm, plot_title=heading)
end

# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------
# Nominal configuration (see `draw_nominal_realization` in `setup.jl`).
realization = draw_nominal_realization()
truth = realization.truth
rng = method_rng(seed_nominal)

if plot_data
    for i in 1:n_u
        name, unit = input_info[i]
        p0 = plot(0:t_h, realization.U[i, :]; lw=2, legend=false,
            title=name, xlabel=time_info[2], ylabel=unit)
        display(p0)
    end
    for i in 1:n_y
        name, unit = output_info[i]
        p0 = plot(0:t_h, realization.Y[i, :]; lw=2, legend=false,
            title=name, xlabel=time_info[2], ylabel=unit)
        display(p0)
    end
end

# -----------------------------------------------------------------------------
# Posterior Sampling
# -----------------------------------------------------------------------------
ensemble = initialize_ensemble(rng, sampler, stage_adaptations, model, realization.data, n_chains_tuning)

runtime_sampling = @elapsed begin
    results = run_staged_mcmc!(ensemble, staged_config_tuning, stage_adaptations, model, realization.data)
end

samples = results.samples

@printf("Sampling finished in %.2f seconds.\n", runtime_sampling)
@printf("Final acceptance ratio: %.2f %%\n", results.mean_acceptance * 100)

# -----------------------------------------------------------------------------
# Posterior Predictive Check
# -----------------------------------------------------------------------------
# Simulate a random subset of the posterior samples forward over the test period.
# The predicted trajectories should track the true states well.
samples_pred = samples[rand(rng, 1:length(samples), n_pred)]
X_pred, _ = simulate_trajectories(model, samples_pred, realization.U_plan; rng=rng)

# All states are shown as subplots of one figure; the legend is only drawn once.
x_pred_mean_all = mean(X_pred, dims=3)
predictive_plots = []
for i in 1:n_x
    name, unit = state_info[i]
    x_pred_mean = x_pred_mean_all[i, :, 1]
    x_pred_upper = mapslices(v -> quantile(v, 0.95), X_pred[i, :, :], dims=2)[:, 1]
    x_pred_lower = mapslices(v -> quantile(v, 0.05), X_pred[i, :, :], dims=2)[:, 1]

    if i == 1
        legend = :best
    else
        legend = false
    end
    p_i = plot(0:H, x_pred_lower; fillrange=x_pred_upper, alpha=0.35,
        label="90% prediction interval", legend=legend,
        title=name, xlabel=time_info[2], ylabel=unit,
        titlefontsize=9, guidefontsize=8, tickfontsize=6, legendfontsize=6)
    plot!(p_i, 0:H, realization.X_plan[i, :], label="true trajectory", lw=2)
    plot!(p_i, 0:H, x_pred_mean, label="mean prediction", lw=2)
    push!(predictive_plots, p_i)
end
p1 = grid_figure(predictive_plots, "Posterior predictive vs. true trajectories")
display(p1)

# -----------------------------------------------------------------------------
# MCMC Diagnostics
# -----------------------------------------------------------------------------
per_chain_samples = [results.stage[end].chain[k].samples for k in 1:n_chains_tuning]
per_chain_matrices = [get_sample_matrix(s) for s in per_chain_samples]

# Autocorrelation
# A rapidly decaying autocorrelation function (ACF) indicates good mixing, a slowly
# decaying one indicates strongly correlated samples.
p2 = plot(yticks=-1:0.1:1)
for k in 1:n_chains_tuning
    acf_k = compute_autocorrelation(per_chain_samples[k]; max_lag=max_lag)
    for i in 1:n_inferred
        if i == 1
            label = "Chain $k"
        else
            label = ""
        end
        plot!(0:max_lag, acf_k[:, i], lw=2, alpha=0.5, label=label, color=k)
    end
end
title!("Autocorrelation function (ACF)")
xlabel!("Lag")
ylabel!("ACF")
display(p2)

# Effective sample size (ESS)
# The ESS approximates the number of independent posterior samples.
ess_per_chain = [compute_ess(s; max_lag=100) for s in per_chain_samples]
ess = reduce(+, ess_per_chain)
@printf("Minimum ESS: %.1f (= %.2f / s)\n", minimum(ess), minimum(ess) / runtime_sampling)

# Trace plots
# Traces should look stationary and should not show long-term trends. All
# variables are shown as subplots of one figure; chains are distinguished by
# color only, without a legend. Only every `trace_stride`-th iteration is plotted
# to keep the figure (and its SVG file) small; long-term trends remain visible.
K_chain = length(per_chain_samples[1])
trace_stride = max(1, cld(K_chain, n_trace_points))
trace_iterations = 1:trace_stride:K_chain
trace_plots = []
for i in axes(per_chain_matrices[1], 2)
    name, unit = variable_info(i)
    p_i = plot(title=name, xlabel="Iteration", ylabel=unit, legend=false,
        titlefontsize=9, guidefontsize=8, tickfontsize=6)
    for k in 1:n_chains_tuning
        plot!(p_i, trace_iterations .- 1, per_chain_matrices[k][trace_iterations, i],
            lw=1, alpha=0.7, color=k)
    end
    push!(trace_plots, p_i)
end
p3 = grid_figure(trace_plots, "Trace plots of the MCMC samples")
display(p3)

# Posterior marginals with priors and true values
# If the data is informative, the posterior should be tighter than the prior and
# centered near the true value. The marginals do not reveal posterior correlations,
# so they should be interpreted together with the prediction quality.
if use_blocked_sampler
    true_values = [truth.theta; truth.x0; realization.x_t0]
else
    true_values = [truth.theta; realization.x_t0]
end

# All variables are shown as subplots of one figure; the legend is only drawn once.
sample_matrix = get_sample_matrix(samples)
marginal_plots = []
for i in axes(sample_matrix, 2)
    name, unit = variable_info(i)
    if i == 1
        legend = :best
    else
        legend = false
    end
    p_i = histogram(sample_matrix[:, i], bins=bins, normalize=:pdf, label="Posterior",
        title=name, xlabel=unit, ylabel="Density", legend=legend,
        titlefontsize=9, guidefontsize=8, tickfontsize=6, legendfontsize=6)
    if i <= n_inferred
        prior = prior_marginals[i]
        values = range(quantile(prior, 0.01), quantile(prior, 0.99), length=500)
        plot!(p_i, values, pdf.(prior, values), label="Prior", linestyle=:dash, lw=2)
    end
    vline!(p_i, [true_values[i]], label="True", lw=2)
    push!(marginal_plots, p_i)
end
p4 = grid_figure(marginal_plots, "Posterior marginals vs. prior and true values")
display(p4)

# Gelman-Rubin R-hat
# R-hat compares within-chain to between-chain variance; values close to 1
# (typically below 1.05) indicate convergence across chains.
if n_chains_tuning > 1
    R_hat = compute_gelman_rubin(per_chain_samples)
    max_r_hat = maximum(R_hat)
    @printf("Maximum R-hat: %.4f\n", max_r_hat)
    if max_r_hat < 1.05
        println("Chains seem to have converged (R-hat < 1.05).")
    else
        println("Warning: Chains might not have fully converged (R-hat ≥ 1.05).")
    end
end