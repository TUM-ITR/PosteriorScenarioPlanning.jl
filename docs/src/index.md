# PosteriorScenarioPlanning.jl

**PosteriorScenarioPlanning.jl** provides a framework for safe optimal control of unknown nonlinear systems from input–output data, with probabilistic guarantees for constraint satisfaction.

It implements the method described in:

> **A Bayesian Framework for Safe Optimal Control of Unknown Systems from Input–Output Data**
>
> *Robert Lefringhausen, Annalena Daniels, Michael Fink, Sandra Hirche*
>
> [arXiv:2611.03466](https://arxiv.org/abs/2611.03466) (2026)

## Scope

Learning-based optimal control aims to operate systems with unknown dynamics by learning models from data. In many applications, several challenges arise jointly:

- **Nonlinear and stochastic dynamics**, affected by both process noise and measurement noise.
- **Latent states**, which cannot be measured directly, so that learning and control must rely on input–output data alone.
- **Scarce data**, which makes the incorporation of structural prior knowledge essential.
- **Safety requirements**, which demand that constraints are satisfied with high confidence despite the uncertainty about the dynamics and the latent states.

PosteriorScenarioPlanning addresses these challenges jointly. Prior knowledge is encoded in a parametric state-space model and a prior distribution over its unknown parameters and initial state. This prior is updated with measured inputs and outputs using particle Markov chain Monte Carlo (PMCMC) sampling. The resulting posterior samples are used to synthesize control inputs via a risk-sensitive scenario program and to certify them on independent samples, which yields a high-confidence bound on the probability of constraint violation.

## How these pages are organized

- **[Method](@ref method-overview)** summarizes the theory behind the framework: the problem setting, posterior inference, the validation procedure and its guarantee, the risk-sensitive synthesis with adaptive calibration, and the receding-horizon extension. Each page names the types and functions that implement the respective step.
- **[Choosing the Parameters](@ref parameters)** explains the most important settings, grouped by what they affect, and how to choose them.
- **[Case Study: Vertical Farm](@ref case-study-overview)** describes the numerical example of the paper in more detail than the paper itself, including the preliminary analyses used to choose the model, prior, sampler, and calibration settings. The experiment scripts in `experiments/` serve as the reference implementation of a complete workflow.
- **[API Reference](@ref api)** lists all exported types and functions.

!!! tip "Where to start"
    These pages give a broad overview of the method and its implementation, but they do not replace the paper. For a thorough understanding of the method, including its assumptions, guarantees, and analysis, we recommend reading the paper first. For applying the package, the [vertical-farm case study](@ref case-study-overview) is the best starting point, as it walks through a complete workflow from model and prior to certified control inputs.

## Installation

The package is not registered. Clone the repository and instantiate the environment locally (Julia 1.12 or later):

```bash
git clone https://github.com/TUM-ITR/PosteriorScenarioPlanning.jl.git
cd PosteriorScenarioPlanning.jl
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

The experiments use a separate environment:

```bash
julia --project=experiments -e 'using Pkg; Pkg.instantiate()'
```

MCMC chains run in parallel across threads, so starting Julia with multiple threads (e.g. `julia -t auto`) is recommended.

## Solvers

The results reported in the paper were obtained using Ipopt with the proprietary **HSL linear solvers** (in particular `MA97`). HSL solvers are available from [https://www.hsl.rl.ac.uk/](https://www.hsl.rl.ac.uk/) under a free academic license.

The package itself does not depend on HSL, and the optimal control problems can in principle be solved with open-source linear solvers such as MUMPS. This configuration has not been tested, however, and may be slower or numerically less robust.

## Citation

If you use this code in your research, please cite:

```bibtex
@article{lefringhausen2026bayesian,
  title={A Bayesian Framework for Safe Optimal Control of Unknown Systems from Input--Output Data},
  author={Lefringhausen, Robert and Daniels, Annalena and Fink, Michael and Hirche, Sandra},
  journal={arXiv preprint arXiv:2611.03466},
  year={2026}
}
```
