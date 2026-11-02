# [Overview](@id case-study-overview)

This section demonstrates the framework on the optimal control of a vertical farm. It follows the numerical example of the paper but describes the setup in more detail, including the preliminary analyses used to choose the model, the prior, the sampler, and the calibration settings, and it points to the code that implements each step.

## The control problem

Unlike in field or greenhouse cultivation, crops in vertical farms are grown in stacked layers inside closed buildings, where temperature, lighting, and CO₂ concentration can be regulated independently of the weather. Since maintaining this controlled environment is costly, adapting it to the changing needs of the crop through optimal control is particularly promising.

We consider a farm cultivating tomato, in which the environmental inputs are set once per day to steer crop growth over a fixed cultivation period. The objective is to reach a target yield at harvest at minimum resource cost. Meeting this target matters economically, since production may be tied to delivery commitments made before the cultivation cycle begins: falling short of the target can lead to direct financial losses, whereas using more resources than necessary reduces the profit margin. The operator thus seeks an input plan that minimizes resource cost while ensuring that the target yield is reached with a prescribed probability.

## Why this is a suitable test case

Crop growth can be described by parametric state-space models derived from plant physiology. Their parameters, however, vary across cultivars and, due to natural biological variability, even across individual plants of the same cultivar, so that they must be learned from data. The problem combines all challenges the framework addresses:

- **Nonlinear and stochastic dynamics.** The growth models are nonlinear. Plant growth is stochastic, e.g., due to unmodeled fluctuations of the growing conditions within the farm, and biomass measurements are subject to considerable noise.
- **Latent states.** Only the biomass can be measured. The states that govern the development of the crop cannot be measured directly.
- **Scarce data.** Measurements are available only once per day, so that only a few data points are available over the entire cultivation cycle. This necessitates the incorporation of prior knowledge, e.g., from plant physiology.
- **Safety requirements.** The target yield must be reached with a prescribed probability. Since both the parameters and the latent plant state remain uncertain throughout the cultivation period, and this uncertainty propagates to the predicted yield, this requires input plans that rigorously account for it.

## Experimental protocol

Each experiment simulates one cultivation cycle of ``t_\mathrm{h} = 100`` days:

1. **Data collection.** During the first ``t_0 = 75`` days, the farm is operated with inputs that vary randomly around a nominal operating point. These inputs and the resulting daily biomass measurements form the dataset ``\mathcal{D}``.
2. **Planning.** At day ``t_0``, the inputs for the remaining ``H = 25`` days until harvest are planned, either once (open loop) or repeatedly every ``L = 5`` days as new measurements arrive (receding horizon).
3. **Evaluation.** The plan is applied to the true system, and the realized yield and production cost are recorded.

To evaluate the probabilistic guarantee empirically, the experiments are repeated for ``R = 200`` Monte Carlo runs. Each run represents a different crop, whose true parameters and initial state are drawn from the prior and kept fixed throughout the cultivation cycle. The frequency of yield shortfalls across these runs provides an empirical check of the guarantee.

## Baselines

The proposed method is compared with two baselines that isolate different aspects of the method and with a hindsight oracle:

- **Certainty-equivalent (CE) planner.** Uses the posterior mean of the parameters, estimated from the same PMMH samples as the proposed method, together with a state estimate at ``t_0`` from a particle filter with 500 particles conditioned on these parameters. It predicts the system deterministically, omitting the zero-mean process noise, and enforces the yield constraint on this prediction. It thus uses the same information as the proposed method but neglects the remaining uncertainty.
- **Prior-based planner.** Runs the same synthesis–validation procedure, but with scenarios drawn from the prior instead of the posterior, propagated to ``t_0`` under the applied inputs while ignoring the measurements. This uses [`PriorPredictiveSampler`](@ref) in place of the PMMH sampler. Since the prior is itself a valid Bayesian description of the unknown system, its plans carry the same guarantee, but the measurements are not used to reduce uncertainty.
- **Hindsight oracle.** Knows the true parameters, the true state at ``t_0``, and the future process noise, and computes the input sequence that reaches the target at minimum cost. It serves as a reference for the cost of reaching the target in each run.

The CE planner and the oracle solve a deterministic OCP with hard constraints, implemented as a [`RobustScenarioFormulation`](@ref) with a single scenario. All methods are evaluated on the same realizations of the true parameters, initial states, and noise sequences.

## Structure of this section

The following pages first describe the setup and the preliminary analyses used to choose the settings, and then the results:

- **[Crop Model and Prior](@ref case-study-model):** the SIMPLE crop model, the noise model, the selection of uncertain parameters, the prior, and the choice of the yield target.
- **[Control Problem](@ref case-study-control):** the cost and the constraints.
- **[Sampler Tuning](@ref case-study-sampler):** the PMMH settings and the diagnostics used to choose them.
- **[Calibration](@ref case-study-calibration):** the settings of the synthesis–validation loop and the risk-weight sweep used to initialize it.
- **[Open-Loop Results](@ref case-study-open-loop):** a representative run and the Monte Carlo results for planning once at ``t_0``.
- **[Receding-Horizon Results](@ref case-study-receding-horizon):** the Monte Carlo results for replanning every 5 days.

## Code structure

All experiment code is located in `experiments/`, which has its own Julia environment (`julia --project=experiments`).

| File | Purpose |
|:-----|:--------|
| `setup.jl` | Single source of all shared settings: model, prior, data generation, sampler, control problem, calibration, solver, and the shared evaluation functions |
| `SIMPLE/` | Implementation of the SIMPLE crop model |
| `noise_calibration.jl` | Definition of the process and measurement noise levels from a nominal simulation |
| `sensitivity_analysis.jl` | Selection of the uncertain parameters |
| `compute_priors.jl` | Prior construction from published cultivars |
| `yield_threshold.jl` | Choice of the yield target |
| `sampler_tuning.jl` | PMMH diagnostics |
| `risk_weight_sweep.jl` | Response of the violation rate to the risk weight |
| `posterior_planning.jl` | Open loop: proposed method, certainty-equivalent planner, and hindsight oracle for one run |
| `prior_planning.jl` | Open loop: prior-based planner for one run |
| `receding_horizon.jl` | Receding-horizon experiment for one run |
| `montecarlo/` | Running the Monte Carlo study on a cluster and analyzing the results |

`setup.jl` only defines settings and functions and is included by all other scripts, so every setting is defined exactly once.

## Reproducibility

Every Monte Carlo run is identified by an integer seed. The realization of a run, consisting of the true parameters and initial state, the data-collection inputs, and the process and measurement noise over the whole cultivation cycle, is drawn up front from a dedicated random number stream (`realization_rng` in `setup.jl`) that no method touches. All methods are therefore evaluated on identical realizations, even when they run in separate processes. The internal randomness of the methods, such as MCMC proposals and scenario noise, is drawn from a second, independent stream (`method_rng`).

The preliminary analyses use a fixed *nominal configuration* instead (`draw_nominal_realization`): the published SunnySD tomato cultivar as the true system, with the inputs and noise of seed 0.

## Running the experiments

A single run of an experiment is started by executing the respective script, e.g.,

```bash
julia --project=experiments -t auto experiments/posterior_planning.jl
```

The seed is set at the top of each script (seed 7 by default, which is the run shown in the figures). `posterior_planning.jl` is the recommended entry point to the code (see [Open-Loop Results](@ref case-study-open-loop)).

The Monte Carlo study runs one process per seed and method:

```bash
julia --project=experiments experiments/montecarlo/run_montecarlo_single.jl \
    --method posterior --seed 1 --results-dir experiments/montecarlo/results/open_loop
```

where `--method` is one of

| Method | Contents | Script |
|:-------|:---------|:-------|
| `posterior` | Open loop: proposed method, CE planner, and oracle | `posterior_planning.jl` |
| `prior` | Open loop: prior-based planner | `prior_planning.jl` |
| `receding` | Receding horizon: proposed method, its first plan applied in open loop, CE planner, and oracle | `receding_horizon.jl` |

The results are stored in `<results-dir>/<method>/seed_<seed>.jld2` and aggregated by `montecarlo/analyze_open_loop.jl` and `montecarlo/analyze_receding_horizon.jl`, after setting `RESULTS_DIR` at the top of the respective script.

For the full study with 200 runs per method, a containerized workflow for HPC clusters is provided. `experiments/montecarlo/README.md` describes how to build the Docker image, convert it into an Apptainer image on the cluster, configure the host paths, and submit all seeds as a SLURM job array via `experiments/montecarlo/run_montecarlo_slurm.sh`.
