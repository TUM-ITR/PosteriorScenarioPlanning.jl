"""
    module PosteriorScenarioPlanning

Tools for inference, optimal control, and planning with uncertain stochastic
state-space models.

The package supports workflows in which posterior samples of model parameters
and latent states are inferred from observed input-output data and then used as
scenarios for control. It separates three layers:

- **Inference** — draw posterior samples of parameters and latent states from
  data, typically via particle MCMC.
- **Optimal control (synthesis)** — combine posterior samples as scenarios in an
  optimal-control problem (OCP), e.g. a robust or risk-sensitive formulation, and
  solve it once to obtain an input trajectory (`solve_ocp!`).
- **Planning (synthesis + certification)** — wrap an OCP in a synthesis–validation
  loop that certifies each candidate input on a freshly drawn validation set and
  adaptively recalibrates the formulation until a robustness guarantee holds
  (`plan!`).

The module provides:
- State-space model and data abstractions
- Particle filtering for likelihood estimation and trajectory sampling
- Particle marginal Metropolis--Hastings (PMMH) samplers
- Adaptive MCMC mechanisms, such as step-size and particle-count adaptation
- Diagnostics for assessing MCMC convergence and mixing
- Trajectory cost and constraint abstractions
- Posterior-sample-based OCP formulations
- Solver interfaces for repeated OCP solves, such as in model predictive
  control (MPC)
- Validation tools (violation counting and Clopper--Pearson bounds) and
  planners that certify a control law against a robustness level

A typical workflow is:
1. Define a stochastic state-space model and observed data.
2. Build an MCMC workspace and configure a sampler.
3. Run particle MCMC to obtain posterior samples of parameters and latent states.
4. Define a control problem through trajectory costs and constraints.
5. Build an OCP workspace using posterior samples as scenarios.
6. Solve the OCP directly (`solve_ocp!`) for an input trajectory, or run a
   planner (`plan!`) to obtain a certified input trajectory with an associated
   robustness guarantee.
7. Update posterior samples or scenario data and repeat as needed (e.g. in a
   receding-horizon loop).

The implementation is organized into modular components for models, data,
particle filtering, MCMC sampling, adaptation, diagnostics, optimal-control
formulations, validation, planning, and solver backends.
"""
module PosteriorScenarioPlanning

using LinearAlgebra
using Statistics
using Distributions
using Random
using StatsBase
using StaticArrays
using Printf
using JuMP
using Base.Threads

# Models and data
export AbstractModel
export DiscreteTimeIOData
export DiscreteTimeStochasticSSM
export DiscreteTimeSSMSample
export extract_parameters
export simulate_trajectories

# MCMC
export MCMCWorkspace
export initialize_workspace
export MCMCConfig
export StagedMCMCConfig
export ChainResult
export run_mcmc!
export run_staged_mcmc!
export MCMCEnsemble
export initialize_ensemble

# Particle filter
export particle_filter!
export particle_filter
export ParticleFilterBuffer
export allocate_pf_buffer

# Particle MMH
export ParticleMMHSampler
export Standard
export Blocked
export ParticleMMHState
export BlockedParticleMMHState

# Prior sampler
export PriorPredictiveSampler

# Proposals
export GaussianProposal

# Adaptations
export AbstractAdaptation
export AbstractStepAdaptation
export AbstractStageAdaptation
export RobbinsMonro
export NAdaptation
export GaussianCovAdaptation

# Diagnostics
export get_sample_matrix
export compute_ess
export compute_gelman_rubin
export compute_autocorrelation

# Optimal Control Components
export AbstractOutputMode
export WithoutOutputs
export WithOutputs
export AbstractCost
export evaluate_cost
export AbstractConstraint
export AbstractInputConstraint
export AbstractScenarioConstraint
export evaluate_constraint
export OptimalControlProblem
export AbstractOCPFormulation
export AbstractSolverConfig
export AbstractOCPWorkspace
export build_ocp_workspace
export update_scenarios!
export SolverDiagnostics
export OCPSolution
export solve_ocp!
export FunctionalCost
export FunctionalStateCost
export BoxInputConstraint
export BoxStateConstraint
export BoxOutputConstraint
export FunctionalInputConstraint
export FunctionalScenarioConstraint
export FunctionalStateConstraint
export JuMPSolverConfig

# Robust Scenario Program
export RobustScenarioFormulation
export RobustScenarioWorkspace

# CVaR-Penalized Scenario Program
export CVaRScenarioFormulation
export CVaRScenarioWorkspace
export update_omega!
export update_tau!

# Validation
export validate_candidate
export clopper_pearson_bound

# Planner
export AbstractPlanner
export plan!
export AdaptiveCalibrationConfig
export AdaptiveCVaRPlanner
export AdaptiveCVaRResult
export CalibrationHistory
export CalibrationStep

include("interfaces.jl")
include("models.jl")
include("inference.jl")
include("mh_step.jl")
include("particle_filter.jl")
include("pmmh.jl")
include("prior_sampler.jl")
include("proposals.jl")
include("ensemble.jl")
include("adaptations.jl")
include("diagnostics.jl")
include("ocp_components.jl")
include("robust_scenario_ocp.jl")
include("CVaR_ocp.jl")
include("validation.jl")
include("adaptive_CVaR_planner.jl")

end