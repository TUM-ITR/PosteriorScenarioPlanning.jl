# =============================================================================
# compute_priors.jl: Gaussian priors for the SIMPLE model parameters
# =============================================================================
# Computes independent Gaussian priors for all SIMPLE model parameters from
# published parameter sets of physiologically similar crops.
#
# Of the 22 published parameter sets in Zhao et al. (2019), cool-season crops
# (wheat, potato, carrot), C4 plants (maize, sweetcorn), and tropical crops
# (rice, cassava, banana, cotton) are excluded. The remaining 7 reference
# cultivars are tomato (2), soybean (2), drybean (1), greenbean (1), and
# peanut (1).
#
# The published RUE values are given in [g MJ⁻¹ m⁻²], whereas the SIMPLE
# implementation uses [kg MJ⁻¹ m⁻²], so the prior mean and standard deviation
# of RUE must be multiplied by 1e-3 before use.
#
# Reference: Zhao et al. (2019), "A SIMPLE crop model", European Journal of
# Agronomy.
# =============================================================================

using Printf
using Statistics

# -----------------------------------------------------------------------------
# Parameter Data
# -----------------------------------------------------------------------------

param_names = ["T_sum", "HI", "I_50A", "T_base", "T_opt", "RUE (×1e3)", "I_50maxH", "I_50maxW", "T_max", "T_ext", "S_CO2", "S_water", "I_50B_init"]

# Each row: [T_sum, HI, I_50A, T_base, T_opt, RUE, I_50maxH, I_50maxW, T_max, T_ext, S_CO2, S_water, I_50B_init]
data = [
    2500.0 0.35 680.0 6.0 27.0 0.86 120.0 20.0 36.0 50.0 0.07 0.9 300.0 # Soybean – Bragg
    2350.0 0.40 600.0 6.0 27.0 0.86 120.0 20.0 36.0 50.0 0.07 0.9 200.0 # Soybean – Williams82
    2700.0 0.40 450.0 5.0 27.0 0.80 90.0 20.0 32.0 45.0 0.07 0.9 600.0  # Drybean – Porrillo Sintetico
    3100.0 0.35 520.0 10.0 28.0 1.20 100.0 5.0 36.0 50.0 0.07 2.0 550.0 # Peanut – FLORUNNER
    2800.0 0.68 520.0 6.0 26.0 1.00 100.0 5.0 32.0 45.0 0.07 2.5 400.0  # Tomato – SunnySD
    2300.0 0.50 550.0 6.0 26.0 1.00 100.0 5.0 32.0 45.0 0.07 2.5 300.0  # Tomato – Agriset761
    1600.0 0.45 370.0 5.0 27.0 0.86 100.0 10.0 32.0 45.0 0.07 0.4 300.0 # Greenbean – Bronco Habit 1
]

# -----------------------------------------------------------------------------
# Compute Gaussian Priors
# -----------------------------------------------------------------------------

n_crops, n_params = size(data)

μ = vec(mean(data, dims=1))
σ = vec(std(data, dims=1))

# -----------------------------------------------------------------------------
# Display Results
# -----------------------------------------------------------------------------

@printf("Independent Gaussian priors from %d selected cultivars\n", n_crops)
println("="^60)
println()
@printf("%-12s%-12s%-12s%s\n", "Parameter", "Mean", "Std", "Range")
println("-"^60)
for i in 1:n_params
    @printf("%-12s%-12.2f%-12.2f[%g, %g]\n",
        param_names[i], μ[i], σ[i], minimum(data[:, i]), maximum(data[:, i]))
end

# -----------------------------------------------------------------------------
# Sanity Check
# -----------------------------------------------------------------------------

# The following checks whether the true parameter values for the SunnySD tomato
# cultivar are within 2 standard deviations of the computed priors.
true_tomato = [2800.0, 0.68, 520.0, 6.0, 26.0, 1.00, 100.0, 5.0, 32.0, 45.0, 0.07, 2.5, 400.0]

println()
println("Sanity check: tomato (SunnySD) within 2σ of prior?")
println("-"^60)
for i in 1:n_params
    if σ[i] == 0.0
        @printf("  %-12s  zero variance, prior is a point mass\n", param_names[i])
        continue
    end

    dist = abs(true_tomato[i] - μ[i]) / σ[i]
    if dist < 2.0
        status = "✓"
    else
        status = "✗"
    end
    @printf("  %-12s  |true - μ|/σ = %.2f  %s\n", param_names[i], dist, status)
end