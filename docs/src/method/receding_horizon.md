# [Receding-Horizon Planning](@id method-receding-horizon)

A single certified input sequence is computed from the data available at the planning time and cannot exploit measurements collected during operation. Inputs late in the horizon are therefore chosen under more uncertainty than necessary. In the receding-horizon setting, the synthesis–validation loop is repeated as new data arrive:

1. At the start of cycle ``i``, at time ``t_i = t_0 + iL``, the posterior is updated with all data collected up to ``t_i - 1``.
2. The calibration loop computes a certified input sequence for the remaining horizon.
3. The first ``L`` inputs are applied, and the next cycle begins.

Since each plan incorporates all data observed so far, the posterior is progressively refined, which reduces conservatism and cost over time.

## Efficient implementation

Two computations dominate each cycle: the posterior update and the solution of the scenario program, which is repeated for each attempt. Since only ``L`` new measurements are added to an already large dataset, the posterior and the scenario program typically change only gradually. Both are therefore warm-started:

- **Posterior.** Each chain of the [`MCMCEnsemble`](@ref) is continued from the sample it produced in the previous cycle, now targeting the posterior on the extended data. The particle filter is rerun to obtain the likelihood estimate under the new data, and the proposal and particle count are adapted. Since each chain starts close to the updated posterior, a short additional burn-in suffices.
- **Validation sets.** Within a cycle, the validation chains are advanced by a short burn-in only when an attempt does not terminate the loop, and the new samples form the validation set of the next attempt. This keeps successive validation sets independent of the candidates adapted to previous outcomes.
- **Optimization.** The input sequence of the previous cycle, shifted by ``L`` steps and padded, initializes the next solve. Within a cycle, consecutive attempts share the training scenarios and differ only in ``\omega``, so the previous solution is used directly. The calibration starts from the risk weight and elasticity estimate of the previous cycle (keywords `omega_init` and `s_init` of [`plan!`](@ref)).

These measures are purely computational and do not affect the per-cycle guarantees.

## Guarantees in closed loop

The guarantee of the validation step applies to the planned trajectory of each individual cycle and is re-established after every replanning step. It does not directly provide a joint guarantee for the complete closed-loop trajectory, which is consistent with standard stochastic model predictive control. If a joint guarantee over a fixed number of cycles is required, the risk ``\alpha`` and the confidence ``\beta`` can be allocated across the cycles accordingly.

The closed-loop scheme of the case study, including the handling of cycles in which no candidate certifies, is implemented in `experiments/receding_horizon.jl`.
