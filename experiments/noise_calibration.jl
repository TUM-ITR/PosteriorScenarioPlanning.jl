# =============================================================================
# noise_calibration.jl: Nominal trajectory and noise levels of the SIMPLE model
# =============================================================================
# Simulates the SunnySD tomato cultivar over 100 days under constant nominal
# growing conditions and derives the noise standard deviations used in
# `planning.jl` from the resulting trajectory:
#   • process noise: 5% of the average daily increment of each state,
#   • measurement noise: 5% of the biomass at day 50.
# =============================================================================

using Plots
using Printf
using Statistics
include("SIMPLE/SIMPLE.jl")
using .SIMPLE

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

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
    32.0,       # T_max
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

# Names and units of the states, inputs, and parameters for plotting.
const input_info = [
    ("Temperature", "°C"),
    ("Drought", "relative level"),
    ("Radiation", "MJ m⁻² d⁻¹"),
    ("CO₂ concentration", "ppm"),
]

const state_info = [
    ("Biomass", "kg m⁻²"),
    ("Cumulative temperature", "°C d"),
    ("Leaf senescence", "°C d"),
]

const time_info = ("Time", "day")

# -----------------------------------------------------------------------------
# Simulation
# -----------------------------------------------------------------------------

states = Vector{SIMPLE.SimpleState}(undef, days)
states[1] = initial_state

for d in 2:days
    states[d] = SIMPLE.next_state(states[d-1], inputs[d-1], tomato_parameters)
end

yield_SIMPLE = SIMPLE.get_yield(states[end], tomato_parameters)
@printf("Yield (fresh biomass in kg m⁻²): %.2f\n", yield_SIMPLE)

# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------

t = 1:days

state_fields = fieldnames(typeof(states[1]))
input_fields = fieldnames(typeof(inputs[1]))

state_plots = Plots.Plot[]
for i in eachindex(state_fields)
    (name, unit) = state_info[i]
    data = [getfield(s, state_fields[i]) for s in states]
    push!(state_plots, plot(t, data; xlabel="$(time_info[2])", ylabel="$unit",
        title=name, lw=2, legend=false))
end

input_plots = Plots.Plot[]
for i in eachindex(input_fields)
    (name, unit) = input_info[i]
    data = [getfield(inp, input_fields[i]) for inp in inputs]
    push!(input_plots, plot(t, data; xlabel="$(time_info[2])", ylabel="$unit",
        title=name, lw=2, legend=false))
end

# State trajectories (shown in the documentation); the fourth grid cell stays empty.
p_states = plot(state_plots..., plot(framestyle=:none);
    layout=(2, 2), size=(900, 800), margin=5Plots.mm)
display(p_states)

# Constant nominal inputs (for reference only).
p_inputs = plot(input_plots...;
    layout=(2, 2), size=(900, 800), margin=5Plots.mm)
display(p_inputs)

# -----------------------------------------------------------------------------
# Noise Calibration
# -----------------------------------------------------------------------------

# State trajectory as a matrix, one row per day and one column per state.
X = [getfield(s, f) for s in states, f in fieldnames(SIMPLE.SimpleState)]

# Process noise
# Standard deviations are set to 5% of the average daily increment of each state.
mean_increment = vec(mean(diff(X, dims=1), dims=1))
σ_v = 0.05 .* mean_increment

# Measurement noise
# Standard deviation is set to 5% of the biomass at day 50.
σ_w = 0.05 * X[50, 1]

println()
println("Noise standard deviations")
println("-"^60)
for (i, (name, unit)) in enumerate(state_info)
    @printf("  %-39s%8.4f %s\n", "Process noise, " * name, σ_v[i], unit)
end
@printf("  %-39s%8.4f %s\n", "Measurement noise, biomass", σ_w, "kg m⁻²")