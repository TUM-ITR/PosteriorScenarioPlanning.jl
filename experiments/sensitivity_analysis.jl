"""
This script performs a one-at-a-time sensitivity analysis of the SIMPLE crop 
model parameters with respect to the terminal yield after 100 days. Each 
parameter is perturbed by ±5% under nominal growing conditions, and the 
normalized sensitivity (elasticity) is computed. Parameters are ranked by 
absolute elasticity to identify the most influential ones for estimation.
"""

using Printf
include("SIMPLE/SIMPLE.jl")
using .SIMPLE

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

# Percentage by which each parameter is increased and decreased
perturbation_percent = 5.0

# Simulation length in days
days = 100

# Parameters for a tomato crop (SunnySD cultivar)
tomato_parameters = SIMPLE.SimpleParameters(
    2800.0,     # T_sum
    0.68,       # HI
    520.0,      # I_50A
    6.0,        # T_base
    26.0,       # T_opt
    1.00e-3,    # RUE
    100.0,      # I_50maxH
    5.0,        # I_50maxW
    32.0,       # T_heat
    45.0,       # T_ext
    0.07,       # S_CO2
    2.5,        # S_water
    0.95,       # f_solar_max
)

# Initial state of the simulation
initial_state = SIMPLE.SimpleState(
    0.001,      # m_B
    0.0,        # T_c
    400.0,      # I_50B
)

# Constant input data for the simulation
inputs = [SIMPLE.SimpleInput(
    22.0,       # T
    0.15,        # D
    20.0,       # R
    600.0,      # CO2
) for _ in 1:days]

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

"""
    simulate_yield(p)

Run the SIMPLE model forward for `days` steps from `initial_state` using the
constant `inputs` defined above and return the final yield.
"""
function simulate_yield(p::SIMPLE.SimpleParameters)
    state = initial_state
    for d in 2:days
        state = SIMPLE.next_state(state, inputs[d-1], p)
    end
    return SIMPLE.get_yield(state, p)
end

"""
    perturb_parameter(p, field, factor)

Return a copy of `p` with the value of `field` multiplied by `factor`.
"""
function perturb_parameter(p::SIMPLE.SimpleParameters, field::Symbol, factor::Real)
    values = [getfield(p, f) for f in propertynames(p)]
    idx = findfirst(==(field), propertynames(p))
    values[idx] = values[idx] * factor
    return SIMPLE.SimpleParameters(values...)
end

# -----------------------------------------------------------------------------
# One-at-a-time sensitivity analysis
# -----------------------------------------------------------------------------

delta = perturbation_percent / 100.0
yield_baseline = simulate_yield(tomato_parameters)

results = Tuple{Symbol,Float64,Float64,Float64,Float64}[]
for name in propertynames(tomato_parameters)
    p_plus = perturb_parameter(tomato_parameters, name, 1.0 + delta)
    p_minus = perturb_parameter(tomato_parameters, name, 1.0 - delta)

    yield_plus = simulate_yield(p_plus)
    yield_minus = simulate_yield(p_minus)

    # Normalized sensitivity (elasticity): (Δy/y) / (Δp/p)
    elasticity = (yield_plus - yield_minus) / (2.0 * delta * yield_baseline)

    pct_plus = 100.0 * (yield_plus - yield_baseline) / yield_baseline
    pct_minus = 100.0 * (yield_minus - yield_baseline) / yield_baseline

    push!(results, (name, elasticity, pct_plus, pct_minus, yield_plus - yield_minus))
end

# Rank parameters by influence on yield (highest absolute elasticity first)
sort!(results, by=r -> abs(r[2]), rev=true)

# -----------------------------------------------------------------------------
# Output
# -----------------------------------------------------------------------------

@printf("Baseline yield after %d days: %.2f kg/m²\n", days, yield_baseline)
@printf("Perturbation:  ±%.2f %% (one-at-a-time)\n\n", perturbation_percent)

@printf("%-13s  %12s  %12s  %12s\n",
    "Parameter", "Elasticity", "Δy (+δ) [%]", "Δy (-δ) [%]")
println("-"^55)
for (name, elasticity, pct_plus, pct_minus, _) in results
    @printf("%-13s  %12.5f  %11.4f  %11.4f\n",
        string(name), elasticity, pct_plus, pct_minus)
end