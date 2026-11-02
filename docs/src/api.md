# [API Reference](@id api)

This page documents all exported types and functions of `PosteriorScenarioPlanning`, grouped by the three layers of the package.

```@docs
PosteriorScenarioPlanning
```

## Models and data

```@docs
AbstractModel
DiscreteTimeIOData
DiscreteTimeStochasticSSM
DiscreteTimeSSMSample
extract_parameters
simulate_trajectories
```

## Inference

### Particle filter

```@docs
ParticleFilterBuffer
allocate_pf_buffer
particle_filter!
particle_filter
```

### Samplers and proposals

```@docs
ParticleMMHSampler
Standard
Blocked
ParticleMMHState
BlockedParticleMMHState
GaussianProposal
PriorPredictiveSampler
```

### Configuration and execution

```@docs
MCMCWorkspace
MCMCConfig
StagedMCMCConfig
ChainResult
run_mcmc!
run_staged_mcmc!
MCMCEnsemble
initialize_ensemble
```

### Adaptation

```@docs
AbstractAdaptation
AbstractStepAdaptation
AbstractStageAdaptation
RobbinsMonro
NAdaptation
GaussianCovAdaptation
```

### Diagnostics

```@docs
get_sample_matrix
compute_ess
compute_gelman_rubin
compute_autocorrelation
```

## Optimal control

### Problem definition

```@docs
OptimalControlProblem
AbstractOutputMode
WithoutOutputs
WithOutputs
AbstractCost
evaluate_cost
FunctionalCost
FunctionalStateCost
AbstractConstraint
AbstractInputConstraint
AbstractScenarioConstraint
evaluate_constraint
BoxInputConstraint
BoxStateConstraint
BoxOutputConstraint
FunctionalInputConstraint
FunctionalScenarioConstraint
FunctionalStateConstraint
```

### Formulations and solvers

```@docs
AbstractOCPFormulation
AbstractOCPWorkspace
AbstractSolverConfig
JuMPSolverConfig
build_ocp_workspace
update_scenarios!
solve_ocp!
OCPSolution
SolverDiagnostics
CVaRScenarioFormulation
CVaRScenarioWorkspace
update_omega!
update_tau!
RobustScenarioFormulation
RobustScenarioWorkspace
```

## Planning

### Validation

```@docs
validate_candidate
clopper_pearson_bound
```

### Adaptive CVaR planner

```@docs
AbstractPlanner
plan!
AdaptiveCalibrationConfig
AdaptiveCVaRPlanner
AdaptiveCVaRResult
CalibrationHistory
CalibrationStep
```
