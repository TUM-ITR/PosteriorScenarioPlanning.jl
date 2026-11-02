# =============================================================================
# setup.jl — Shared setup of the vertical-farm experiments
# =============================================================================
# Single source of truth for everything the experiments have in common, so that
# the settings are defined exactly once:
#
#   1. Model          SIMPLE crop model, process and measurement noise 
#                     distributions
#   2. Prior          prior over θ and x₀
#   3. Realizations   true system, data-collection inputs, and noise sequences of
#                     one Monte-Carlo run, drawn from a dedicated RNG stream
#   4. Inference      PMMH sampler, adaptations, and scenario counts
#   5. OCP            horizon, cost, input and yield constraints, solver
#   6. Calibration    settings of the synthesis–validation loop
#   7. Evaluation     simulation of a plan on the true system, the deterministic
#                     OCP used by the certainty-equivalent planner and the
#                     hindsight oracle, and the certified planner shared by the
#                     proposed method and the prior-based baseline
#   8. Plot helpers
#
# The file only defines settings and functions; it does not run anything. It is
# included by
#   - `posterior_planning.jl`  proposed method, certainty-equivalent baseline,
#                              and hindsight oracle,
#   - `prior_planning.jl`      prior-based baseline,
#   - `receding_horizon.jl`    receding-horizon experiment,
#   - `sampler_tuning.jl`      preliminary analysis of the sampler,
#   - `risk_weight_sweep.jl`   preliminary analysis of the weight update,
#   - `montecarlo/`            Monte-Carlo runner and analysis.
#
# Reproducibility and common random numbers
# -----------------------------------------
# Every Monte-Carlo run is identified by an integer `seed`. The realization of the
# run — true parameters, initial state, data-collection inputs, and the process
# and measurement noise over the whole cultivation cycle — is drawn up front from
# `realization_rng(seed)`, which no method touches. All methods are therefore
# evaluated on identical realizations, even when they run in separate processes.
# The internal randomness of the methods (MCMC, scenario noise, particle filter)
# is drawn from a second stream, `method_rng(seed)`.
# =============================================================================

using LinearAlgebra
using Random
using Statistics
using Distributions
using Plots
using LaTeXStrings
using Printf
using StaticArrays
using JuMP
using Ipopt
using MadNLP
using PosteriorScenarioPlanning

const PSP = PosteriorScenarioPlanning

# Optional dependency: HSL linear solvers (MA97) for Ipopt and MadNLP.
# Falls back to MUMPS (Ipopt) or UMFPACK (MadNLP) if unavailable, which may result
# in increased runtimes or reduced numerical stability.
hsl_available = false
try
    import HSL_jll
    import MadNLPHSL
    global hsl_available = true
    @info "HSL is available."
catch e
    @warn "Optional HSL dependencies not available. Falling back to the default linear solvers." exception = e
end

include("SIMPLE/SIMPLE.jl")

# Global settings
truth_from_prior = true # whether to draw the true system from the prior (true) or use the nominal SunnySD cultivar (false)
use_blocked_sampler = false # whether to use the blocked PMMH sampler (jointly proposing theta and x0) or the standard PMMH sampler (proposing theta only)
use_MadNLP = false # whether to use MadNLP (true) or Ipopt (false) as the optimization backend

@info "Running on $(Threads.nthreads()) Julia threads."

# =============================================================================
# 1. Model
# =============================================================================
# The SIMPLE model is a dynamic plant growth model with 6 unknown parameters
# (T_sum, I_50A, T_base, T_opt, RUE, S_water) and 3 states (biomass, cumulative 
# temperature, leaf senescence). It is implemented in `SIMPLE/SIMPLE.jl`.

# Model dimensions
const n_u = 4 # control inputs: air temperature T [°C], relative drought level D, radiation R [MJ m⁻² d⁻¹], atmospheric CO₂ concentration CO2 [ppm]
const n_x = 3 # states: above-ground biomass m_B [kg m⁻²], cumulative temperature T_c [°C d], leaf senescence I_50B [°C d]
const n_y = 1 # measured output: biomass m_B [kg m⁻²]
const n_theta = 6 # unknown model parameters: T_sum, I_50A, T_base, T_opt, RUE, S_water
const n_v = n_x # additive process noise

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

const output_info = [
    ("Biomass", "kg m⁻²"),
]

const parameter_info = [
    ("\$T_{\\mathrm{sum}}\$", "°C d"),
    ("\$I_{50\\mathrm{A}}\$", "°C d"),
    ("\$T_{\\mathrm{base}}\$", "°C"),
    ("\$T_{\\mathrm{opt}}\$", "°C"),
    ("\$RUE\$", "kg MJ⁻¹ m⁻²"),
    ("\$S_{\\mathrm{water}}\$", "unitless"),
]

const time_info = ("Time", "day")

# Harvest index (fraction of the total biomass that is harvestable)
const HI = 0.68

# Return the state transition function f(x, u, v, θ) for the SIMPLE crop model.
# Of the 13 SIMPLE parameters, the 6 in θ are estimated and the remaining ones are
# fixed at the SunnySD tomato values and hardcoded in the closure. This keeps the
# returned function generic: it only depends on the state, input, noise, and θ,
# as expected by the sampler.
#
# θ contains the cultivar-specific parameters that affect the biomass dynamics
# (T_sum, I_50A; Zhao et al., 2019) and the species-specific parameters that
# significantly influence the terminal yield (T_base, T_opt, RUE, S_water),
# identified by the one-at-a-time sensitivity analysis in `sensitivity_analysis.jl`.
# Their elasticities range from 1.30 to 0.26, whereas all remaining species-specific
# parameters are below 1e-3 and therefore fixed. Three sensitive parameters are
# nevertheless fixed as well: HI only scales the yield and is not identifiable from
# biomass data, f_solar_max depends on the planting density (a management choice),
# and S_CO2 is identical across the reference cultivars, so that its prior has no
# variance. T_sum is included despite its small local elasticity, as it is
# cultivar-specific and becomes influential for cultivars whose senescence starts
# within the cultivation period. The cultivar-specific I_50B is an initial
# condition and is therefore part of x0 rather than θ.
function get_f_theta()
    function f_simple(x, u, v, theta)
        input = SIMPLE.SimpleInput(u[1], u[2], u[3], u[4])
        state = SIMPLE.SimpleState(x[1], x[2], x[3])

        params = SIMPLE.SimpleParameters(
            theta[1],  # T_sum
            HI,        # harvest index
            theta[2],  # I_50A
            theta[3],  # T_base
            theta[4],  # T_opt
            theta[5],  # RUE
            100.0,     # I_50maxH
            5.0,       # I_50maxW
            32.0,      # T_max
            45.0,      # T_ext
            0.07,      # S_CO2
            theta[6],  # S_water
            0.95,      # f_solar_max
        )

        next_state = SIMPLE.next_state(state, input, params)

        # Biomass, cumulative temperature, and leaf senescence are all non-negative.
        # A smoothed max keeps the dynamics twice continuously differentiable, as
        # required by Ipopt.
        # eps_smax is intentionally very small, as the smoothing only matters close to
        # zero, i.e. in the early trajectories where m_B is small, and a larger value
        # would bias them. The resulting sharp kink does not affect Ipopt, since the
        # OCP starts at t = 75, when the states are far away from zero.
        eps_smax = 1.0e-10
        return SVector(
            SIMPLE.smax(next_state.m_B + v[1], 0.0, eps_smax),
            SIMPLE.smax(next_state.T_c + v[2], 0.0, eps_smax),
            SIMPLE.smax(next_state.I_50B + v[3], 0.0, eps_smax),
        )
    end
    return f_simple
end

f_theta = get_f_theta()

# Process noise model
# Standard deviations are set to approximately 5% of the average daily
# state increment under nominal growing conditions (see `noise_calibration.jl`)
# and treated as known.
const Q = Diagonal([0.0058^2,    # m_B [kg m⁻²]
    0.8000^2,                    # T_c [°C d]
    0.0938^2])                   # I_50B [°C d]
dist_v = MvNormal(zeros(n_x), Q)

# Measurement function
# Only biomass m_B is measured; cumulative temperature T_c and leaf senescence I_50B are latent.
const C = [1.0 0.0 0.0]
g_theta(x, u, theta) = C * x

# Measurement noise model
# Standard deviation is set to approximately 5% of the biomass at
# day 50 under nominal growing conditions (see `noise_calibration.jl`)
# and treated as known.
# MvNormal is used (rather than Normal) to keep observations as vectors
# throughout the particle filter.
const R = Diagonal([0.1444^2])  # m_B [kg m⁻²]
dist_w = MvNormal(zeros(n_y), R)

# =============================================================================
# 2. Prior
# =============================================================================

# Prior distribution for the unknown parameters
# Independent Gaussians truncated at zero, since all parameters are physically
# nonnegative. Means and standard deviations are computed from published values
# of 7 cultivars (see `compute_priors.jl`).
const theta_mean = [
    2478.57,  # T_sum [°C d]
    527.14,   # I_50A [°C d]
    6.29,     # T_base [°C]
    26.86,    # T_opt [°C]
    0.94e-3,  # RUE [kg MJ⁻¹ m⁻²]
    1.44      # S_water
]

const theta_std = [
    475.97,   # T_sum
    99.95,    # I_50A
    1.70,     # T_base
    0.69,     # T_opt
    0.14e-3,  # RUE
    0.87      # S_water
]

theta_marginals = [truncated(Normal(μ, σ), 0.0, Inf) for (μ, σ) in zip(theta_mean, theta_std)]
dist_theta = product_distribution(theta_marginals)

# Prior distribution for the initial state (latent)
# Biomass is near zero at sowing with small uncertainty. Cumulative
# temperature is zero at sowing. The initial leaf senescence I_50B
# varies across cultivars, with its prior estimated from the
# literature (see compute_priors.jl).
# For the blocked sampler, zero variances are not allowed, so all states are given
# small positive variances.
if !use_blocked_sampler
    x0_marginals = [
        LogNormal(log(0.001), 0.5),                   # m_B: small positive seed weight
        Dirac(0.0),                                   # T_c: exactly zero at sowing
        truncated(Normal(378.57, 146.79), 0.0, Inf),  # I_50B: from published values
    ]
else
    x0_marginals = [
        LogNormal(log(0.001), 0.5),
        truncated(Normal(0.0, 1e-3), 0.0, Inf),
        truncated(Normal(378.57, 146.79), 0.0, Inf),
    ]
end
dist_x0 = product_distribution(x0_marginals)

# Model container
model = DiscreteTimeStochasticSSM(
    n_u,
    n_x,
    n_y,
    n_theta,
    n_v,
    f_theta,
    g_theta,
    dist_x0,
    dist_v,
    dist_w,
    dist_theta
)

# =============================================================================
# 3. Realizations
# =============================================================================
# One Monte-Carlo run is one realization of the uncertain crop: true parameters and
# initial state (drawn from the prior), random inputs during the data-collection 
# phase, and the process and measurement noise over the whole cultivation cycle. 
# All of it is drawn up front, so that every method faces exactly the same system 
# and the same future noise.

# Cultivation period (in days)
# During the first t₀ days, the farm is operated with random inputs to collect
# data (days 0, …, t₀ − 1). At t₀, the inputs of the remaining H = t_h − t₀ days
# until harvest at t_h are planned.
#
# Time convention: column t of every trajectory is day t − 1, and inputs, states,
# outputs, and noise are defined on the same time points, so that g(x_t, u_t) is
# always available. The cycle has t_h + 1 time points, days 0, …, t_h, and the
# planning window has H + 1 time points, days t₀, …, t₀ + H = t_h. The input and the
# process noise at the final time point have no effect on the state; they only
# exist to keep all trajectories the same length.
const t0 = 75           # planning time t₀
const t_h = 100         # harvest day t_h
const H = t_h - t0      # prediction horizon H of the OCP

# RNG streams: one for the system (`realization_rng`) and one for the methods
# (`method_rng`). Seeding Xoshiro with an integer hashes it, so seeds that differ
# by a constant offset yield statistically independent streams.
const RNG_METHOD_OFFSET = 1_000_000

"""
    realization_rng(seed)

RNG from which the realization of Monte-Carlo run `seed` is drawn. It is never
used by any method, so the realization does not depend on which methods are run.

# Arguments
- `seed::Integer`: index of the Monte-Carlo run.

# Returns
- `Xoshiro`: freshly seeded RNG.
"""
realization_rng(seed::Integer) = Xoshiro(seed)

"""
    method_rng(seed)

RNG for the internal randomness of the methods in Monte-Carlo run `seed`.

# Arguments
- `seed::Integer`: index of the Monte-Carlo run.

# Returns
- `Xoshiro`: freshly seeded RNG, independent of [`realization_rng`](@ref).
"""
method_rng(seed::Integer) = Xoshiro(RNG_METHOD_OFFSET + seed)

# Ground truth used for data generation
# The SunnySD tomato cultivar serves as the nominal ground truth. 
# It is used for the sensitivity analysis, the sampler tuning, and the
# risk weight sweep.
# Alternatively, `sample_truth` draws the true system from the prior.
const truth_nominal = (
    theta=[2800.0, 520.0, 6.0, 26.0, 1.00e-3, 2.5], # T_sum, I_50A, T_base, T_opt, RUE, S_water
    x0=[0.001, 0.0, 400.0], # m_B, T_c, I_50B
)

# Nominal configuration of the preliminary analyses (sampler tuning, risk weight
# sweep): the nominal cultivar with one fixed realization of the data-collection
# inputs and the noise. The seed only makes this realization reproducible.
const seed_nominal = 0

"""
    sample_truth(rng)

Draw the true parameters and initial state from the prior.

# Arguments
- `rng::AbstractRNG`: RNG for the draw.

# Returns
- `NamedTuple` with the parameters (`theta`) and the initial state (`x0`).
"""
function sample_truth(rng)
    theta = rand(rng, model.dist_theta)
    x0 = rand(rng, model.get_dist_x0(theta))
    return (theta=theta, x0=x0)
end

"""
    draw_truth(rng)

Ground truth of one replicate: drawn from the prior if `truth_from_prior` is set,
otherwise the nominal SunnySD cultivar.

# Arguments
- `rng::AbstractRNG`: RNG for the draw; unused for the nominal truth.

# Returns
- `NamedTuple` with the parameters (`theta`) and the initial state (`x0`).
"""
function draw_truth(rng)
    if truth_from_prior
        return sample_truth(rng)
    else
        return truth_nominal
    end
end

# Input trajectory used for data generation
# Independent random inputs around the nominal operating point.

# Nominal values
const T_nom = 22.0      # T [°C]
const D_nom = 0.15      # D [unitless between 0 and 1]
const R_nom = 20.0      # R [MJ m⁻² d⁻¹]
const CO2_nom = 600.0   # CO2 [ppm]
const U_nom = [T_nom, D_nom, R_nom, CO2_nom] # nominal operating

# Standard deviations
T_std = 0.1 * T_nom
D_std = 0.05
R_std = 0.1 * R_nom
CO2_std = 0.05 * CO2_nom

# Since D is a relative drought level between 0 and 1, it is modelled as a Beta
# distribution with mean D_nom and standard deviation D_std.
# See https://en.wikipedia.org/wiki/Beta_distribution#Mean_and_variance for formulas.
function beta_from_mean_std(μ, σ)
    @assert 0.0 < μ < 1.0 "Mean must be strictly between 0 and 1."
    @assert 0.0 < σ < sqrt(μ * (1 - μ)) "Std too large for a Beta distribution."

    ν = μ * (1 - μ) / σ^2 - 1
    α = μ * ν
    β = (1 - μ) * ν

    return Beta(α, β)
end

T_dist = Normal(T_nom, T_std)
D_dist = beta_from_mean_std(D_nom, D_std)
R_dist = Normal(R_nom, R_std)
CO2_dist = Normal(CO2_nom, CO2_std)

u_dist = product_distribution([T_dist, D_dist, R_dist, CO2_dist])

"""
    draw_realization(seed; truth=nothing)

Draw the realization of Monte-Carlo run `seed`: the true system, the random
inputs, and the process and measurement noise over the whole cultivation cycle,
and simulate the true system under the random inputs to generate the data set.

Everything is drawn from [`realization_rng`](@ref)`(seed)` in a fixed order, so
the realization depends on `seed` only. The noise of the planning window
(`V_plan`) is part of the realization; it is the noise under which every plan is
evaluated on the true system and the noise the hindsight oracle is given.

All trajectories cover the time points of days 0, …, t_h (`t_h + 1` columns).
The data set holds days 0, …, t₀ − 1; the planning window holds days t₀, …, t_h
(`H + 1` columns), matching the OCP. Inputs are drawn for the full
cycle; those after t₀ are not used for planning (the plans replace them), they
only drive the predictive check in `sampler_tuning.jl`.

# Arguments
- `seed::Integer`: index of the Monte-Carlo run.

# Keywords
- `truth::Union{Nothing,NamedTuple}`: fixed true parameters (`theta`) and initial
  state (`x0`). By default, drawn via [`draw_truth`](@ref).

# Returns
- `NamedTuple` with
  - `seed` and `truth`,
  - the full random input, state, and output trajectories (`U`, `X`, `Y`) and the
    noise sequences (`V`, `W`) over days 0, …, t_h,
  - the data set of the data-collection phase (`data`),
  - the true state at the planning time t₀ (`x_t0`) and the process noise of the
    planning window (`V_plan`, size `(n_v, H + 1)`; its last column is unused),
  - the planning window under the random inputs (`U_plan`, `X_plan`, `Y_plan`,
    `H + 1` columns each).
"""
function draw_realization(seed::Integer; truth=nothing)
    rng = realization_rng(seed)

    if truth === nothing
        truth = draw_truth(rng)
    end

    dist_v_true = model.get_dist_v(truth.theta)
    dist_w_true = model.get_dist_w(truth.theta)

    # All random quantities are drawn up front, in a fixed order.
    # Column t is day t − 1; the last column is harvest day t_h.
    U = rand(rng, u_dist, t_h + 1)
    V = rand(rng, dist_v_true, t_h + 1)
    W = rand(rng, dist_w_true, t_h + 1)

    # Simulate the true system under the random inputs.
    X = Array{Float64}(undef, n_x, t_h + 1)
    Y = Array{Float64}(undef, n_y, t_h + 1)
    X[:, 1] = truth.x0
    for t in 2:(t_h+1)
        X[:, t] .= model.f(X[:, t-1], U[:, t-1], V[:, t-1], truth.theta)
    end
    for t in 1:(t_h+1)
        Y[:, t] .= model.g(X[:, t], U[:, t], truth.theta) + W[:, t]
    end

    # Data: days 0, …, t₀ − 1. Planning window: days t₀, …, t_h.
    training_window = 1:t0
    planning_window = (t0+1):(t_h+1)
    return (
        seed=seed,
        truth=truth,
        U=U, X=X, Y=Y, V=V, W=W,
        data=DiscreteTimeIOData(U[:, training_window], Y[:, training_window]),
        x_t0=X[:, t0+1],
        V_plan=V[:, planning_window],
        U_plan=U[:, planning_window], X_plan=X[:, planning_window], Y_plan=Y[:, planning_window],
    )
end

"""
    draw_nominal_realization()

Realization of the nominal configuration used by the preliminary analyses: the
nominal cultivar (`truth_nominal`) with the inputs and noise of `seed_nominal`.

# Returns
- `NamedTuple`: see [`draw_realization`](@ref).
"""
draw_nominal_realization() = draw_realization(seed_nominal; truth=truth_nominal)

# =============================================================================
# 4. Inference
# =============================================================================

# Scenario counts. K_train posterior samples are frozen as the OCP scenarios. K_val is
# the number of validation scenarios drawn per attempt; it must be large enough that
# certification at level α is achievable (see V*_α in the planner output).
const K_train = 400
const K_val = 5000

# PMMH settings
n_chains = K_train + K_val # one independent chain per scenario

# Thinning
# Number of MCMC steps between two consecutive draws of the same chain. Since each
# scenario comes from its own chain, the scenarios within one set are independent
# by construction; the thinning only matters when a chain is continued. Between
# validation attempts, each validation chain advances by `thinning` steps. Because
# each candidate depends on the previous validation outcome through the ω update,
# the new validation set is only independent of the candidate if the chains have
# decorrelated. The thinning is therefore chosen such that the empirical
# autocorrelation at this lag is zero within its sampling noise (see
# `sampler_tuning.jl`).
thinning = 300

burnin = 500 # length of burn-in period for each stage
N_init = 100 # initial number of particles of the particle filter - will be adjusted during sampling
T_chunk = 5 # number of datapoints added at each stage
K_intermediate = Int(2e4) # number of samples per intermediate stage

# Sampler and proposal
# The proposal covariance is initialized with the marginal prior variances and
# subsequently refined by the step and stage adaptations.
theta_var = var(dist_theta)

if !use_blocked_sampler
    # Standard PMMH sampler (recommended default)
    # Proposes theta only while marginalizing over latent states via the particle filter.
    proposal = GaussianProposal(Matrix(Diagonal(theta_var)), 1.0)
    sampler = ParticleMMHSampler(proposal, N_init, Standard())
else
    # Blocked PMMH sampler
    # This sampler jointly proposes (theta, x0).
    # This can be beneficial when there is strong posterior correlation between
    # parameters and the initial state, or when the posterior of x_0 is much tighter
    # than its prior.
    x0_var = var(dist_x0)
    proposal = GaussianProposal(Matrix(Diagonal(vcat(theta_var, x0_var))), 1.0)
    sampler = ParticleMMHSampler(proposal, N_init, Blocked())
end

# Step adaptation
# RobbinsMonro adapts the MCMC step size to target an optimal acceptance ratio (e.g., around 0.234 for high-dimensional problems).
# The step size is adjusted based on the observed acceptance rate during sampling.
step_adaptations = AbstractStepAdaptation[RobbinsMonro(; verbose=false)]

# Stage adaptation
# NAdaptation adjusts the number of particles in the particle filter across stages.
# GaussianCovAdaptation adapts the proposal covariance across stages.
stage_adaptations = AbstractStageAdaptation[
    NAdaptation(20, 200, 10, N_init, 100, 1.2, true),
    GaussianCovAdaptation(1e-8, true),
]

# MCMC configuration for intermediate and final stages
intermediate_stage_config = MCMCConfig(burnin, 0; step_adaptations=step_adaptations, verbose=false)
final_stage_config = MCMCConfig(burnin, thinning; step_adaptations=step_adaptations, verbose=false)
staged_config = StagedMCMCConfig(intermediate_stage_config, final_stage_config, T_chunk;
    K_intermediate=K_intermediate, K_final=K_train + K_val, verbose=true)

# =============================================================================
# 5. Optimal Control Problem
# =============================================================================

# The OCP with horizon H spans the H + 1 time points t₀, …, t₀ + H of the planning
# window and imposes H transitions, so the terminal state x_{t_h} lies in the last 
# column, H + 1. The input in that column, i.e., on harvest day, enters no
# transition, so it can only act through the output y_{t_h} = g(x_{t_h}, u_{t_h}).
# In this example, the OCP does not use outputs (WithoutOutputs), and the cost
# charges only the inputs applied before harvest. The input on harvest day thus
# has no effect, and the OCP holds it at the input of the previous day, so that
# the solver does not see a free variable without curvature.

# ---- Input constraints ------------------------------------------------------
# Temperature: 10–30 °C, (ambient temperature is 10 °C, above which heating is required)
# Drought: 0–1,
# Radiation: 0–35 MJ m⁻² d⁻¹
# CO₂: 420–700 ppm (ambient CO₂ is 420 ppm, and growth response saturates above 700 ppm)

const T_ambient = 10.0      # outside ambient temperature [°C]
const CO2_ambient = 420.0   # ambient CO₂ concentration [ppm]

const u_lb = [T_ambient, 0.0, 0.0, CO2_ambient]
const u_ub = [30.0, 1.0, 35.0, 700.0]

input_constraints = AbstractInputConstraint[
    BoxInputConstraint([1, 2, 3, 4]; lower=u_lb, upper=u_ub),
]

# ---- Scenario constraints ---------------------------------------------------
# Terminal production constraint
# The crop must achieve a minimum yield by the end of the growing period.
# The yield is defined as the harvest index (HI) times the terminal biomass.
# The target is chosen from the prior predictive such that it is rarely 
# met without effort and rarely out of reach.
const minimum_yield = 8.0 # minimum required yield [kg m⁻²]
const t_terminal = (H+1):(H+1)  # column of X, on which the terminal yield constraint is imposed
scenario_constraints = AbstractScenarioConstraint[
    BoxStateConstraint([1]; lower=minimum_yield / HI, time_range=t_terminal)
]

# ---- Cost ------------------------------------------------------------------
# Daily production cost per m² of growing area in a vertical farm.
#
# Each input has a linear cost representing the actual resource expenditure,
# plus a quadratic penalty centered at the nominal operating point to improve
# the numerical conditioning of the optimization. The weights are calibrated so
# that, at the largest feasible deviation from the nominal value, the penalty
# equals `penalty_factor` times the largest linear cost of that input.

const penalty_factor = 0.2 # penalty at the largest deviation as fraction of the largest linear cost
const U_ref = U_nom # center of the quadratic penalty: nominal operating point
const u_dev_max = (u_ub .- u_lb) ./ 2 # largest feasible deviation from the nominal point

# Energy prices
const price_kWh = 0.2                       # electricity price [€ kWh⁻¹]
const price_MJ = (1.0 / 3.6) * price_kWh    # electricity price [€ MJ⁻¹]

# Heating: energy required to maintain the indoor temperature above the
# outside ambient temperature. The model accounts for heat losses through the
# building envelope using an effective heat-transfer coefficient. It assumes a
# large farm for which the roof area is approximately equal to the growing area
# and side-wall losses are negligible.
const U_heat = 0.7 # effective heat-transfer coefficient [W m⁻² K⁻¹]
const c_heat = U_heat * 24.0 / 1000.0 * price_kWh # heating cost coefficient [€ m⁻² K⁻¹ day⁻¹]
const c_heat_quad = penalty_factor * c_heat * (u_ub[1] - T_ambient) / u_dev_max[1]^2
cost_heating(T) = c_heat .* (T .- T_ambient) .+ c_heat_quad .* (T .- U_ref[1]) .^ 2

# Irrigation: water cost for recirculating hydroponics.
# Full irrigation is calibrated as approximately 5 L m⁻² day⁻¹ at a water
# cost of approximately 5 € m⁻³, resulting in 0.025 € m⁻² day⁻¹.
const c_irr = 0.025 # full-irrigation cost [€ m⁻² day⁻¹]
const c_irr_quad = penalty_factor * c_irr * (1.0 - u_lb[2]) / u_dev_max[2]^2
cost_irrigation(D) = c_irr .* (1.0 .- D) .+ c_irr_quad .* (D .- U_ref[2]) .^ 2

# Radiation: electrical energy supplied to the LED panels.
# LED waste heat is not modelled separately as it offsets heating demand
# by a similar amount.
const c_rad = price_MJ # radiation cost coefficient [€ m⁻² MJ⁻¹ day⁻¹]
const c_rad_quad = penalty_factor * c_rad * u_ub[3] / u_dev_max[3]^2
cost_radiation(R) = c_rad .* R .+ c_rad_quad .* (R .- U_ref[3]) .^ 2

# CO₂ enrichment: injection cost for maintaining a target concentration above
# ambient in a sealed vertical farm. The coefficient is calibrated such that
# maintaining 820 ppm, i.e. +400 ppm above ambient, requires approximately
# 50 g CO₂ m⁻² day⁻¹ of replenishment, mainly due to plant uptake, at a liquid
# CO₂ price of approximately 0.2 € kg⁻¹.
const c_CO2 = 50e-3 * 0.2 / 400.0   # CO₂ cost coefficient [€ m⁻² day⁻¹ ppm⁻¹]
const c_CO2_quad = penalty_factor * c_CO2 * (u_ub[4] - CO2_ambient) / u_dev_max[4]^2
cost_CO2(CO2) = c_CO2 .* (CO2 .- CO2_ambient) .+ c_CO2_quad .* (CO2 .- U_ref[4]) .^ 2

# Total production cost over the planning horizon. Only the applied inputs
# u_{t₀}, …, u_{t_h − 1} are charged; the input at harvest (last column) has no
# effect and is excluded. The horizon is read off the size of U, so that the same
# cost applies to the shrinking horizons of the receding-horizon experiment.
function production_cost(U, X)
    n_applied = size(U, 2) - 1 # H applied inputs of the H + 1 columns
    return sum(cost_heating(U[1, 1:n_applied])) +
        sum(cost_irrigation(U[2, 1:n_applied])) +
        sum(cost_radiation(U[3, 1:n_applied])) +
        sum(cost_CO2(U[4, 1:n_applied]))
end

cost = FunctionalStateCost((U, X) -> production_cost(U, X))

# ---- OCP formulations -------------------------------------------------------
"""
    build_problem(H_ocp)

OCP of the cultivation problem for a planning window of `H_ocp` days until harvest:
production cost, input bounds, and the terminal yield constraint on the last
column. The input at harvest is held at the input of the previous day by the OCP
(see [`WithoutOutputs`](@ref)).

The open-loop experiment uses the horizon `H = t_h − t₀` (the global `problem`);
the receding-horizon experiment builds one problem per control cycle as the
horizon shrinks toward harvest.

# Arguments
- `H_ocp::Int`: prediction horizon, i.e., the number of days until harvest.

# Returns
- `OptimalControlProblem`: problem with `H_ocp + 1` time points.
"""
function build_problem(H_ocp::Int)
    input_constraints_H = AbstractInputConstraint[
        BoxInputConstraint([1, 2, 3, 4]; lower=u_lb, upper=u_ub),
    ]
    scenario_constraints_H = AbstractScenarioConstraint[
        BoxStateConstraint([1]; lower=minimum_yield / HI, time_range=(H_ocp+1):(H_ocp+1))
    ]
    return OptimalControlProblem(cost, input_constraints_H, scenario_constraints_H, H_ocp, WithoutOutputs())
end

problem = OptimalControlProblem(cost, input_constraints, scenario_constraints, H, WithoutOutputs())

# CVaR-penalized scenario OCP: used by the certified planners
# (proposed method and prior-based baseline).
formulation = CVaRScenarioFormulation(problem)

# Robust scenario OCP with hard constraints. With a single scenario, it is the
# deterministic OCP of the certainty-equivalent planner and the hindsight oracle.
deterministic_formulation = RobustScenarioFormulation(problem)

# ---- Solver settings --------------------------------------------------------
# Each backend has common options plus two start-dependent sets with the same keys.
# The cold-start set is used when a solve starts from the generic guess U_init. The
# warm-start set is used when it starts from the optimum of a previous solve that
# differed only in ω (all calibration attempts after the first). There the 
# iterate is already close to the solution, so a small initial barrier parameter 
# and little push away from the bounds keep the solver from leaving its neighborhood.
if use_MadNLP
    madnlp_options = Dict{String,Any}(
        "max_iter" => 1500,
        "tol" => 1e-6,
        "acceptable_tol" => 1e-4,
        "print_level" => MadNLP.INFO,
    )

    # MadNLP calls Ipopt's `bound_frac` option `bound_fac`.
    madnlp_cold_start_options = Dict{String,Any}(
        "mu_init" => 1e-2,
        "bound_push" => 1e-2,
        "bound_fac" => 1e-2,
    )
    madnlp_warm_start_options = Dict{String,Any}(
        "mu_init" => 1e-4,
        "bound_push" => 1e-6,
        "bound_fac" => 1e-6,
    )

    linear_solver = if hsl_available
        MadNLPHSL.Ma57Solver
    else
        MadNLP.UmfpackSolver
    end

    solver = JuMPSolverConfig(
        () -> MadNLP.Optimizer(kkt_system=MadNLP.SparseKKTSystem, linear_solver=linear_solver);
        attributes=madnlp_options,
        cold_start_attributes=madnlp_cold_start_options,
        warm_start_attributes=madnlp_warm_start_options,
        silent=false,
    )
else
    # Solves are capped by iterations and not by wall-clock time: a time limit
    # would make the returned iterate depend on the hardware and the load of the
    # machine, and hence the results irreproducible.
    ipopt_options = Dict{String,Any}(
        "max_iter" => 1500,
        "tol" => 1e-6,
        "constr_viol_tol" => 1e-4,
        "acceptable_tol" => 1e-4,
        "acceptable_constr_viol_tol" => 1e-4,
        "acceptable_iter" => 10,
        "linear_solver" => "mumps",
        "hessian_approximation" => "exact",
        "print_level" => 5,
    )

    if hsl_available
        ipopt_options["hsllib"] = HSL_jll.libhsl_path
        ipopt_options["linear_solver"] = "ma97"
        ipopt_options["ma97_order"] = "metis"
        ipopt_options["ma97_nemin"] = 32
    end

    ipopt_cold_start_options = Dict{String,Any}(
        "mu_init" => 1e-2,
        "bound_push" => 1e-2,
        "bound_frac" => 1e-2,
    )
    ipopt_warm_start_options = Dict{String,Any}(
        "mu_init" => 1e-4,
        "bound_push" => 1e-6,
        "bound_frac" => 1e-6,
    )

    solver = JuMPSolverConfig(
        Ipopt.Optimizer;
        attributes=ipopt_options,
        cold_start_attributes=ipopt_cold_start_options,
        warm_start_attributes=ipopt_warm_start_options,
        silent=false,
    )
end

# Solver of the deterministic OCP (certainty-equivalent planner and hindsight
# oracle). Same settings, but a larger iteration cap: these single-scenario solves
# are cheap and always cold-started, and the comparison should only fail when the
# yield target is unreachable (LOCALLY_INFEASIBLE), not because the solver ran out
# of iterations. The tighter cap above stays for the calibration loop of the
# certified planners, where a stalled attempt is better abandoned early.
#
# Note: In the open-loop study, one CE run terminates with ITERATION_LIMIT even
# with max_iter = 100000, i.e., Ipopt does not converge rather than converging
# slowly. Like LOCALLY_INFEASIBLE, this is treated as "OCP not solved" (no plan).
const max_iter_deterministic = 100000

deterministic_solver = JuMPSolverConfig(
    solver.optimizer;
    attributes=merge(solver.attributes, Dict{String,Any}("max_iter" => max_iter_deterministic)),
    cold_start_attributes=solver.cold_start_attributes,
    warm_start_attributes=solver.warm_start_attributes,
    silent=solver.silent,
)

# Initial guess for the input trajectory
# The inputs are chosen to maximize growth.
initial_input_guess(H_ocp::Int) = repeat([theta_mean[4], u_lb[2], u_ub[3], u_ub[4]], 1, H_ocp + 1)
const U_init = initial_input_guess(H)

# =============================================================================
# 6. Calibration
# =============================================================================
# Settings for the adaptive calibration of the risk weight ω.
# Risk level α: certify P(constraint satisfaction) ≥ 1 − α. The acceptance band
# [α − α_tol, α] avoids over-conservative solutions. β is the overall confidence;
# it is split across attempts via a Bonferroni correction (β_att = β / n_max).
const alpha = 0.05              # risk level (hard requirement: δ ≤ α)
const alpha_tol = 0.015         # acceptance-band width below α
const beta = 0.001              # overall confidence level (1 − β = 99.9%)
const n_max = 8                 # maximum synthesis–validation attempts
const thinning_val = thinning   # thinning between validation samples
const tau = alpha               # CVaR tail level; τ = α aligns the CVaR with the risk level

# Update law. ω ∈ [0, 1] weights the objective as (1−ω)·J̄ + ω·CVaR; the search
# runs in x = logit(ω) = log λ, where the violation rate is assumed to
# follow a power law locally: dlog(ν)/dlog(λ) = s.
const omega_init = 0.80         # initial risk weight
const omega_min = 1e-3          # floor on ω; an over-conservative attempt at the floor ends the cycle
const s_init = -2.0             # initial elasticity dlog(ν)/dlog(λ)
const s_flat = -0.5             # upper cap on the fitted elasticity; keeps ŝ < 0 under sampling noise
const margin_fraction = 0.10    # aim this fraction below the largest certifiable violation rate
const n_recovery = 3            # final attempts reserved for the recovery phase
const step_bounds = (log(1.15), 2.0) # bounds on the log-step size for ω updates
const gamma_decay = 0.4         # recency weighting of the elasticity fit
const min_spread = 0.15         # minimum spread of the logit(ω) samples for the elasticity fit to be considered valid

calibration_config = AdaptiveCalibrationConfig(;
    alpha=alpha, alpha_tol=alpha_tol, beta=beta, n_max=n_max, K_val=K_val,
    omega_init=omega_init, omega_min=omega_min, s_init=s_init, s_flat=s_flat, margin_fraction=margin_fraction,
    n_recovery=n_recovery, gamma_decay=gamma_decay,
    step_bounds=step_bounds, min_spread=min_spread,
)

# Chain dynamics for the posterior validation draws.
# Since each chain generates a single sample per attempt and the chains are already warmed up,
# the burn-in is set to the thinning interval and the thinning parameter is effectively inert.
# The planner validates the first attempt on each chain's first draw after the warm-up
# and continues the chains to draw a fresh validation set for every later attempt.
validation_ensemble_config = MCMCConfig(thinning_val, thinning_val; verbose=false)

# =============================================================================
# 7. Evaluation
# =============================================================================

"""
    simulate_true_system(realization, U)

Simulate the true system of `realization` over the planning window under the
input trajectory `U`, starting from the true state at t₀ and driven by the
realized process noise `realization.V_plan`.

The indexing matches the OCP: column `t` of the result is the state `t − 1` days
after t₀, and `V_plan[:, t]` drives the transition from column `t` to `t + 1`.

# Arguments
- `realization::NamedTuple`: realization from [`draw_realization`](@ref).
- `U::AbstractMatrix`: input trajectory of size `(n_u, H + 1)`.

# Returns
- `X::Matrix{Float64}`: true state trajectory of size `(n_x, H + 1)`, from x_{t₀}
  to x_{t_h}.
"""
function simulate_true_system(realization, U::AbstractMatrix)
    theta = realization.truth.theta
    V = realization.V_plan
    X = Matrix{Float64}(undef, n_x, H + 1)
    X[:, 1] = realization.x_t0
    for t in 2:(H+1)
        X[:, t] .= model.f(X[:, t-1], U[:, t-1], V[:, t-1], theta)
    end
    return X
end

"""
    evaluate_plan(realization, U)

Apply the input trajectory `U` to the true system of `realization` and evaluate
the production cost and the terminal yield constraint.

# Arguments
- `realization::NamedTuple`: realization from [`draw_realization`](@ref).
- `U::AbstractMatrix`: input trajectory of size `(n_u, H + 1)`.

# Returns
- `NamedTuple` with the realized state trajectory (`X_true`), the production
  cost (`cost`), the terminal yield (`yield`), and whether the yield
  constraint holds on the true system (`feasible`).
"""
function evaluate_plan(realization, U::AbstractMatrix)
    X_true = simulate_true_system(realization, U)
    return (
        X_true=X_true,
        cost=production_cost(U, X_true),
        yield=HI * X_true[1, end],
        feasible=all(X_true[1, t_terminal] .>= minimum_yield / HI),
    )
end

"""
    solve_deterministic_ocp(theta, x_start, V)

Solve the OCP for a single, fully specified system: parameters `theta`, state
`x_start` at t₀, and process-noise sequence `V`. The yield constraint is imposed
as a hard constraint on this single prediction.

This is the synthesis step of the certainty-equivalent planner (posterior-mean
parameters, state estimate, `V = 0`) and of the hindsight oracle (true
parameters, true state, realized noise).

# Arguments
- `theta::AbstractVector`: model parameters.
- `x_start::AbstractVector`: state at the planning time t₀.
- `V::AbstractMatrix`: process-noise sequence of size `(n_v, horizon + 1)`; the
  last column is unused, as no transition follows x_{t_h}.

# Keywords
- `horizon::Int`: prediction horizon of the OCP. Defaults to the open-loop
  horizon `H`; the receding-horizon experiment passes the remaining horizon
  t_h − t_i of each cycle.

# Returns
- `solution::OCPSolution`: solution of the deterministic OCP. Its
  `diagnostics.success` is `false` if no feasible input exists, e.g. when the
  yield target is unreachable for this system.

# Notes
- Solved with `deterministic_solver` (iteration cap `max_iter_deterministic`),
  not with the solver of the certified planners. Use
  [`solve_outcome`](@ref) to tell local infeasibility apart from other failures.
"""
function solve_deterministic_ocp(theta::AbstractVector, x_start::AbstractVector, V::AbstractMatrix;
    horizon::Int=H)
    FT = eltype(theta)
    scenario = DiscreteTimeSSMSample{FT,Standard}(
        Vector{FT}(theta), Vector{FT}(x_start), Matrix{FT}(undef, n_x, 0), zero(FT), zero(FT))

    if horizon == H
        formulation_H = deterministic_formulation
    else
        formulation_H = RobustScenarioFormulation(build_problem(horizon))
    end

    # The noise drawn at construction is overwritten below; a local RNG keeps the
    # global stream untouched.
    workspace = build_ocp_workspace(deterministic_solver, formulation_H, model, [scenario]; rng=Xoshiro(0))

    # Replace the process noise drawn at construction by the prescribed sequence.
    V_refs = workspace.scenario_refs.V_refs
    for t in 1:(horizon+1), i in 1:n_v
        set_parameter_value(V_refs[i, t, 1], V[i, t])
    end

    return solve_ocp!(workspace; U_init=initial_input_guess(horizon))
end

"""
    solve_outcome(solution)

Termination status and final constraint violation of a solve, for telling a
locally infeasible problem (`:LOCALLY_INFEASIBLE`, the yield target is out of
reach) apart from other failures (e.g. `:ITERATION_LIMIT` with an infeasible
last iterate, or numerical errors).

# Arguments
- `solution::OCPSolution`: solution returned by [`solve_ocp!`](@ref).

# Returns
- `NamedTuple` with the termination status as a `Symbol` (`termination_status`)
  and the maximum constraint violation of the returned iterate (`max_violation`).
"""
function solve_outcome(solution)
    info = solution.diagnostics.info
    return (
        termination_status=Symbol(info[:termination_status]),
        max_violation=info[:max_constraint_violation],
    )
end

"""
    run_hindsight_oracle(realization)

Hindsight oracle: the input trajectory that reaches the target yield at minimum
cost given the true parameters, the true state at t₀, and the future process
noise. Its cost is a lower bound on the cost of reaching the target in this run.

# Arguments
- `realization::NamedTuple`: realization from [`draw_realization`](@ref).

# Returns
- `NamedTuple` with the solver status (`success`), the fields of
  [`solve_outcome`](@ref), the input trajectory (`U`), the runtime (`runtime`),
  and the fields of [`evaluate_plan`](@ref).
"""
function run_hindsight_oracle(realization)
    println("━ Hindsight oracle " * "━"^61)
    runtime = @elapsed solution = solve_deterministic_ocp(
        realization.truth.theta, realization.x_t0, realization.V_plan)
    evaluation = evaluate_plan(realization, solution.U)

    outcome = solve_outcome(solution)

    @printf("  Success: %s (%s) | cost: %.3f € m⁻² | yield: %.2f kg m⁻²\n",
        solution.diagnostics.success, outcome.termination_status, evaluation.cost, evaluation.yield)

    return merge(
        (success=solution.diagnostics.success, U=solution.U, runtime=runtime),
        outcome,
        evaluation,
    )
end

"""
    predictive_summary(X)

Pointwise summary of scenario trajectories: mean, 5 % and 95 % quantiles, and
range.

# Arguments
- `X::AbstractArray{<:Real,3}`: trajectories of size `(n_x, H + 1, K)`.

# Returns
- `NamedTuple` of `(n_x, H + 1)` matrices `mean`, `q05`, `q95`, `min`, `max`.
"""
function predictive_summary(X::AbstractArray{<:Real,3})
    return (
        mean=dropdims(mean(X; dims=3); dims=3),
        q05=dropdims(mapslices(v -> quantile(v, 0.05), X; dims=3); dims=3),
        q95=dropdims(mapslices(v -> quantile(v, 0.95), X; dims=3); dims=3),
        min=dropdims(minimum(X; dims=3); dims=3),
        max=dropdims(maximum(X; dims=3); dims=3),
    )
end

"""
    termination_note(result)

Human-readable outcome of a calibration run: how it ended and whether the
returned candidate is certified.

# Arguments
- `result`: an [`AdaptiveCVaRResult`](@ref), or any object with the fields
  `termination` and `certified`.

# Returns
- `String`: `"in band"`, `"floor reached"`, `"fallback, certified (δ ≤ α)"`,
  or `"fallback, NOT certified"`.
"""
function termination_note(result)
    if result.termination === :in_band
        return "in band"
    elseif result.termination === :floor
        return "floor reached"
    elseif result.certified
        return "fallback, certified (δ ≤ α)"
    else
        return "fallback, NOT certified"
    end
end

"""
    run_certified_planner(scenarios_train, validation_ensemble, validation_config,
                          realization, rng; omega_start=omega_init,
                          initial_validation_scenarios=nothing)

Run the synthesis–validation loop on the given scenarios and evaluate the 
returned plan on the true system.

Shared by the proposed method (posterior scenarios) and the prior-based baseline
(prior scenarios), so that both use exactly the same OCP, calibration, and
evaluation; they differ only in where the scenarios come from.

# Arguments
- `scenarios_train::Vector{<:DiscreteTimeSSMSample}`: K_train training scenarios,
  frozen into the OCP.
- `validation_ensemble::MCMCEnsemble`: live ensemble the planner continues to
  draw a fresh, independent validation set of K_val scenarios each attempt.
- `validation_config::MCMCConfig`: chain dynamics of the validation draws.
- `realization::NamedTuple`: realization from [`draw_realization`](@ref).
- `rng::AbstractRNG`: method RNG for the scenario noise of the OCP and of the
  violation counts.

# Keywords
- `omega_start::Real`: risk weight ω₁ of the first attempt; defaults to the
  configured `omega_init`.
- `initial_validation_scenarios`: the latest draws of `validation_ensemble`,
  reused as the validation set of the first attempt instead of redrawing it
  (see [`plan!`](@ref)); `nothing` draws it.

# Returns
- `NamedTuple` with
  - the certification summary: `certified`, `termination` (see
    [`AdaptiveCVaRResult`](@ref)), `delta`, `beta`, `omega`, `attempts`,
  - the calibration histories: `delta_history`, `violation_history`,
    `omega_history`, `update_history`, `U_history`,
  - the plan `U` and the predictive summaries (see [`predictive_summary`](@ref))
    over the last attempt's validation scenarios under the first (`pred_initial`)
    and the returned (`pred_final`) candidate,
  - runtimes: `runtime_build`, `runtime_plan`, `runtime_optimization`,
    `runtime_validation`,
  - the evaluation on the true system (fields of [`evaluate_plan`](@ref)).
"""
function run_certified_planner(scenarios_train, validation_ensemble, validation_config,
    realization, rng; omega_start=omega_init, initial_validation_scenarios=nothing)

    println("━ Certified planning " * "━"^59)
    print("Building workspace...")
    runtime_build = @elapsed workspace = build_ocp_workspace(
        solver, formulation, model, scenarios_train, omega_start, tau; rng=rng)
    @printf("done (%.2f s)\n", runtime_build)

    planner = AdaptiveCVaRPlanner(workspace, validation_ensemble, validation_config, calibration_config)
    runtime_plan = @elapsed solution = plan!(planner, model, realization.data;
        U_init=U_init, omega_init=omega_start,
        initial_validation_scenarios=initial_validation_scenarios)

    println("┏━ Certified Planning Complete " * "━"^49)
    @printf("┃  Certified: %s | outcome: %s\n", solution.certified ? "yes" : "no", termination_note(solution))
    @printf("┃  Achieved bound δ: %.2f %%  ⇒  P(satisfaction) ≥ %.2f %% (confidence %.1f %%)\n",
        solution.delta * 100, (1 - solution.delta) * 100, (1 - solution.beta) * 100)
    @printf("┃  Risk level α: %.2f %% | acceptance band: [%.2f %%, %.2f %%]\n",
        alpha * 100, (alpha - alpha_tol) * 100, alpha * 100)
    @printf("┃  Final ω: %.4f (λ = %.4g)\n", solution.omega, solution.omega / (1 - solution.omega))
    @printf("┃  Attempts used: %d / %d\n", solution.attempts, n_max)
    @printf("┃  Build time: %.2f s | planning time: %.2f s\n", runtime_build, runtime_plan)
    println("┗" * "━"^79)

    if !solution.certified
        @warn "No candidate met the hard requirement δ ≤ α within n_max attempts. " *
            "The scenario uncertainty may be too large for the requested robustness level; " *
            "returned the candidate with the smallest observed bound δ."
    end

    U = solution.solution.U

    # Predictive distribution under the first and the returned candidate, over the
    # validation scenarios of the last attempt. Unlike the training scenarios, these
    # were not optimized against, so the summary is not biased toward the plan.
    # Both candidates are simulated under the same process noise.
    scenarios_pred = solution.validation_scenarios
    V_pred = PSP._draw_process_noise(rng, model, scenarios_pred, size(U, 2))
    X_initial, _ = simulate_trajectories(model, scenarios_pred, solution.U_history[1]; V=V_pred, rng=rng)
    X_final, _ = simulate_trajectories(model, scenarios_pred, U; V=V_pred, rng=rng)

    evaluation = evaluate_plan(realization, U)
    @printf("  On the true system: cost %.3f € m⁻², yield %.2f kg m⁻² (constraint satisfied: %s)\n",
        evaluation.cost, evaluation.yield, evaluation.feasible)

    return merge(
        (
            certified=solution.certified,
            termination=solution.termination,
            delta=solution.delta,
            beta=solution.beta,
            omega=solution.omega,
            attempts=solution.attempts,
            delta_history=solution.delta_history,
            violation_history=solution.violation_history,
            omega_history=solution.omega_history,
            update_history=solution.update_history,
            U_history=solution.U_history,
            U=U,
            pred_initial=predictive_summary(X_initial),
            pred_final=predictive_summary(X_final),
            runtime_build=runtime_build,
            runtime_plan=runtime_plan,
            runtime_optimization=solution.runtime_optimization,
            runtime_validation=solution.runtime_validation,
        ),
        evaluation,
    )
end

# =============================================================================
# 8. Plot helpers
# =============================================================================

# Overlay forbidden regions as red-shaded bands for any box constraints on dimension `dim`
# over the columns 1:T. Column t is drawn at x = t + t_shift, so that the bands line up
# with the trajectories, e.g., t_shift = -1 when these are plotted against days since t₀.
function plot_box_constraints!(p, dim::Int,
    constraints::Vector{<:Union{AbstractInputConstraint,AbstractScenarioConstraint}},
    T::Int; t_shift::Int=0)

    first_label = true

    for c in constraints
        if !(c isa Union{BoxInputConstraint,BoxStateConstraint})
            continue
        end
        idx = findfirst(==(dim), c.dims)
        if isnothing(idx)
            continue
        end
        t_range = intersect(c.time_range, 1:T)
        if isempty(t_range)
            continue
        end

        # Expand y-limits to include constraint bounds with some margin.
        yl, yu = ylims(p)
        margin = 0.05 * (yu - yl)
        new_yl = isfinite(c.lower[idx]) ? min(yl, c.lower[idx] - margin) : yl
        new_yu = isfinite(c.upper[idx]) ? max(yu, c.upper[idx] + margin) : yu
        ylims!(p, (new_yl, new_yu))
        yl, yu = new_yl, new_yu

        # Only label the first constraint region to avoid legend clutter.
        label = first_label ? "constraint" : false
        first_label = false

        # Point constraint: draw a vertical line in the forbidden region.
        if length(t_range) == 1
            t_pt = first(t_range) + t_shift
            if isfinite(c.lower[idx])
                plot!(p, [t_pt, t_pt], [yl, c.lower[idx]];
                    color=:red, lw=2, label=label)
            end
            if isfinite(c.upper[idx])
                plot!(p, [t_pt, t_pt], [c.upper[idx], yu];
                    color=:red, lw=2, label=label)
            end
            continue
        end

        # Range constraint: shade the forbidden region.
        ts = [first(t_range), last(t_range)] .+ t_shift

        if isfinite(c.lower[idx])
            plot!(p, ts, fill(c.lower[idx], 2);
                fillrange=fill(yl, 2),
                fillalpha=0.15, fillcolor=:red,
                lw=0, label=label)
        end

        if isfinite(c.upper[idx])
            plot!(p, ts, fill(c.upper[idx], 2);
                fillrange=fill(yu, 2),
                fillalpha=0.15, fillcolor=:red,
                lw=0, label=label)
        end
    end
end

"""
    plot_state_prediction(planner_result, i; title_suffix="")

Plot the predictive distribution of state `i` under the plan of a certified
planner together with the realized trajectory on the true system, over the days
since t₀ (0, …, H).

# Arguments
- `planner_result::NamedTuple`: result of [`run_certified_planner`](@ref).
- `i::Int`: state index.

# Keywords
- `title_suffix::AbstractString`: appended to the title, e.g. the seed.

# Returns
- `Plots.Plot`: the figure.
"""
function plot_state_prediction(planner_result, i::Int; title_suffix="")
    name, unit = state_info[i]
    pred = planner_result.pred_final
    t = 0:H # days since t₀

    p = plot(; legend=:topleft)
    plot!(p, t, pred.min[i, :]; fillrange=pred.max[i, :], alpha=0.15, lw=0, label="prediction range")
    plot!(p, t, pred.q05[i, :]; fillrange=pred.q95[i, :], alpha=0.35, lw=0, label="90% prediction interval")
    plot!(p, t, planner_result.pred_initial.mean[i, :]; lw=2, ls=:dash, label="mean prediction (first attempt)")
    plot!(p, t, pred.mean[i, :]; lw=2, label="mean prediction (returned plan)")
    plot!(p, t, planner_result.X_true[i, :]; lw=2, label="true trajectory")
    title!(p, "$name: predicted vs. true trajectory$title_suffix")
    ylabel!(p, unit)
    xlabel!(p, "days since t₀")
    plot_box_constraints!(p, i, scenario_constraints, H + 1; t_shift=-1)
    return p
end

"""
    plot_input_trajectories(plans, i; title_suffix="")

Plot input `i` of several plans over the H applied inputs u_{t₀}, …, u_{t_h − 1},
i.e., without the unused input at harvest.

# Arguments
- `plans::AbstractVector{<:Pair{<:AbstractString,<:AbstractMatrix}}`: label ⇒
  input trajectory of size `(n_u, H + 1)`.
- `i::Int`: input index.

# Keywords
- `title_suffix::AbstractString`: appended to the title, e.g. the seed.

# Returns
- `Plots.Plot`: the figure.
"""
function plot_input_trajectories(plans, i::Int; title_suffix="")
    name, unit = input_info[i]
    p = plot(; legend=:best)
    for (label, U) in plans
        plot!(p, 0:(H-1), U[i, 1:H]; lw=2, label=label, linetype=:steppost)
    end
    title!(p, "$name$title_suffix")
    ylabel!(p, unit)
    xlabel!(p, "days since t₀")
    plot_box_constraints!(p, i, input_constraints, H; t_shift=-1)
    return p
end

"""
    constraint_marker()

Up-pointing triangle whose apex, not its center, lies on the data point, so that
a trajectory ending above the apex satisfies the yield constraint (GR draws the
vertices of a `Shape` marker as offsets from the point).

# Returns
- `Shape`: marker for `scatter!(...; marker=constraint_marker())`.
"""
function constraint_marker()
    triangle = Shape(:utriangle)
    return Shape(triangle.x, triangle.y .- maximum(triangle.y))
end
