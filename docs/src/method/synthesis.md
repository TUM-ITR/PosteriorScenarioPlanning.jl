# [Synthesis and Calibration](@id method-synthesis)

The validation procedure certifies a given input sequence but does not construct one. This page describes how candidates are synthesized from training scenarios and how the conservativeness of the synthesis is calibrated until the desired risk level is certified.

## CVaR-penalized scenario program

Enforcing the constraints for all training scenarios is overly conservative, since a few unlikely scenarios can severely restrict the feasible set. Explicitly selecting which scenarios may be violated leads to a combinatorial problem. Instead, violations are penalized in the objective through the Conditional Value-at-Risk (CVaR) of the constraint value ``Z = h(\cdot)``, which is a random variable due to the uncertainty, with positive values corresponding to violations. The CVaR at tail level ``\tau \in (0, 1)`` is the expected value of ``Z`` over the worst ``\tau``-fraction of outcomes. Minimizing it suppresses both the magnitude and the frequency of violations. The tail level is set to ``\tau = \alpha`` by default, which aligns the penalized tail with the desired risk level; when only a few training scenarios fall into this tail, a larger ``\tau`` can be preferable (see [Choosing the Parameters](@ref parameters)). The CVaR can be written as

```math
\mathrm{CVaR}_\tau(Z) = \min_{\eta \in \mathbb{R}} \; \eta + \frac{1}{\tau} \mathbb{E}\left[ (Z - \eta)_+ \right],
```

where ``(\cdot)_+ = \max(\cdot, 0)``. This representation allows the CVaR to be minimized jointly with the inputs by treating ``\eta`` as an additional decision variable.

Replacing expectations by sample averages over ``K_\mathrm{train}`` training scenarios yields the scenario program

```math
\begin{aligned}
\min_{u_{t_0:t_0+H} \in \mathcal{U},\, \eta_1, \ldots, \eta_M} \quad
& \frac{1-\omega}{K_\mathrm{train}} \sum_{k} J^{[k]}
+ \omega \sum_{i=1}^{M} \left( \eta_i + \frac{1}{\tau K_\mathrm{train}} \sum_{k} \left( Z_i^{[k]} - \eta_i \right)_+ \right) \\
\mathrm{s.t.} \quad
& x^{[k]}_{t+1} = f_{\theta^{[k]}}\big(x^{[k]}_t, u_t, v^{[k]}_t\big), \\
& y^{[k]}_t = g_{\theta^{[k]}}\big(x^{[k]}_t, u_t\big) + w^{[k]}_t, \\
& J^{[k]} = J\big(u_{t_0:t_0+H}, x^{[k]}_{t_0:t_0+H}, y^{[k]}_{t_0:t_0+H}\big), \\
& Z_i^{[k]} = h_i\big(u_{t_0:t_0+H}, x^{[k]}_{t_0:t_0+H}, y^{[k]}_{t_0:t_0+H}\big),
\end{aligned}
```

where the dynamics hold for all training scenarios ``k`` and time steps ``t``.

For a single constraint, this is the sample-average approximation of ``(1-\omega)\,\mathbb{E}[J] + \omega\, \mathrm{CVaR}_\tau(Z)``, as presented in the paper. For multiple constraints, the implementation applies the CVaR penalty to each constraint ``Z_i`` separately, with its own auxiliary variable ``\eta_i``, and sums the terms under the common weight ``\omega``. This avoids introducing the non-smooth ``\max`` of the joint constraint into the nonlinear program and lets the solver penalize violations of individual constraints separately. Since the certificate is computed for the joint constraint in the validation step, this choice affects only how candidates are constructed, not what is certified.

The **risk weight** ``\omega \in [0, 1]`` balances nominal performance against robustness. At ``\omega = 0``, only the expected cost is minimized; at ``\omega = 1``, only the tail risk. Since the only hard constraints are the input bounds, the problem is feasible for every ``\omega``. It is a deterministic nonlinear program, generally non-convex, and is solved locally with interior-point methods via JuMP.

In the package, the program is defined by a [`CVaRScenarioFormulation`](@ref) wrapping an [`OptimalControlProblem`](@ref), built into a [`CVaRScenarioWorkspace`](@ref) by [`build_ocp_workspace`](@ref), and solved by [`solve_ocp!`](@ref). The risk weight can be changed in place with [`update_omega!`](@ref), so that consecutive solves can be warm-started.

!!! tip "Worst-case formulation"
    The package also provides [`RobustScenarioFormulation`](@ref), which enforces the constraints for all scenarios. It is not used by the calibration loop but can serve as a reference.

## Adaptive calibration of the risk weight

The risk weight that achieves the desired robustness level is not known in advance. It is therefore calibrated in a **synthesis–validation loop**:

1. Solve the scenario program for the current ``\omega`` to obtain a candidate.
2. Validate the candidate on a fresh validation set to obtain ``V`` and ``\delta``.
3. If ``\delta`` lies in the acceptance band ``[\alpha - \alpha_\mathrm{tol}, \alpha]``, stop. Otherwise, update ``\omega`` and repeat.

The training scenarios are kept fixed across attempts so that the approximated objective stays consistent, while each attempt uses a fresh validation set. The acceptance band is two-sided: a candidate with ``\delta`` well below ``\alpha`` is certified but unnecessarily conservative, so the search continues to recover performance.

**Multiple testing.** Each attempt is a separate statistical test. To maintain the overall confidence ``1 - \beta``, at most ``n_\mathrm{max}`` attempts are allowed and the confidence parameter of each attempt is ``\beta / n_\mathrm{max}``. A union bound then makes all per-attempt bounds simultaneously valid, so the certificate holds regardless of which certified candidate is returned.

**Update law.** For ``\omega < 1``, the objective is equivalent to minimizing ``\mathbb{E}[J] + \lambda \, \mathrm{CVaR}_\tau(Z)`` with ``\lambda = \omega / (1 - \omega)``. In the region where the violation probability responds to the risk weight, it is approximated locally by a power law, ``\nu \approx c \lambda^{s}``, with negative elasticity ``s``. The next ``\lambda`` is chosen such that the predicted violation rate hits a target slightly below the largest certifiable rate. The elasticity is estimated from the preceding attempts by exponentially weighted least squares in ``(\log \lambda, \log \hat{\nu})`` and replaced by an initial value ``s_\mathrm{init}`` until enough attempts are available. The step size is bounded, and once attempts on both sides of the target are available, the search is restricted to the interval between them.

**Recovery phase.** If no candidate has certified, the final ``n_\mathrm{rec}`` attempts move ``\omega`` toward ``1`` in equal steps, reaching ``\omega = 1`` on the last attempt. This guarantees that the most conservative member of the family is tried before the search gives up.

**Result.** Among all certified candidates, the one with the smallest sample-average cost on its validation set is returned, together with its bound ``\delta``. If no candidate certifies, the candidate with the smallest ``\delta`` is reported as uncertified.

The loop is implemented by [`plan!`](@ref) for an [`AdaptiveCVaRPlanner`](@ref), configured through [`AdaptiveCalibrationConfig`](@ref). The full calibration trace of each call is recorded in a [`CalibrationHistory`](@ref).

## When no candidate certifies

If no candidate certifies within `n_max` attempts, the recovery phase ensures that the most risk-averse member of the family, ``\omega = 1``, has been tried. Failure at this point usually indicates that the remaining uncertainty is too large for the requested risk level, rather than a poor choice of the calibration parameters: the calibration can find an appropriate risk weight, but it cannot create robustness that the problem does not admit. Possible remedies are collecting more informative data, relaxing the risk level ``\alpha`` or the constraints, or increasing ``K_\mathrm{val}`` to tighten the bound.
