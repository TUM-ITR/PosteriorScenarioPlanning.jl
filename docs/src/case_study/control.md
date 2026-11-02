# [Control Problem](@id case-study-control)

This page specifies the cost and the constraints of the control problem, which are defined in `experiments/setup.jl`.

## Cost

The objective is to minimize the total production cost until harvest at ``t_\mathrm{h} = 100`` days. At planning time ``t_0``, the inputs are optimized over the remaining ``H = t_\mathrm{h} - t_0`` days with cost

```math
J = \sum_{t=t_0}^{t_\mathrm{h}-1} \Big[ c_T (T_t - T_\mathrm{amb}) + c_D (1 - D_t) + c_R R_t + c_C (C_t - C_\mathrm{amb}) + \lVert u_t - \bar{u} \rVert_W^2 \Big].
```

Since the crop is harvested at ``t_\mathrm{h}``, no input is applied on that day. The linear terms are the costs of the resources, based on representative prices:

| Term | Resource | Calibration of the coefficient |
|:-----|:---------|:-------------------------------|
| ``c_T (T_t - T_\mathrm{amb})`` | Heating above the ambient temperature ``T_\mathrm{amb} = 10`` °C | Heat losses through the building envelope with an effective heat-transfer coefficient of 0.7 W m⁻² K⁻¹, at 0.20 € kWh⁻¹ |
| ``c_D (1 - D_t)`` | Irrigation | About 5 L m⁻² d⁻¹ at full irrigation, at about 5 € m⁻³ |
| ``c_R R_t`` | LED lighting | Electrical energy of the supplied radiation, at 0.20 € kWh⁻¹ |
| ``c_C (C_t - C_\mathrm{amb})`` | CO₂ enrichment above the ambient ``C_\mathrm{amb} = 420`` ppm | About 50 g m⁻² d⁻¹ to maintain +400 ppm, at about 0.20 € kg⁻¹ |

The quadratic term regularizes the inputs around the nominal operating point ``\bar{u}`` (22 °C, drought level 0.15, 20 MJ m⁻² d⁻¹, 600 ppm), which corresponds to typical growing conditions for tomato. It improves the numerical conditioning of the optimization. Its diagonal weights are set such that, at the largest feasible deviation from ``\bar{u}``, the penalty equals 20 % of the largest linear cost of the respective input.

## Constraints

The inputs are subject to box constraints that represent the operating limits of the farm:

| Input | Lower bound | Upper bound |
|:------|:------------|:------------|
| Temperature ``T_t`` | 10 °C (ambient) | 30 °C |
| Drought level ``D_t`` | 0 | 1 |
| Radiation ``R_t`` | 0 | 35 MJ m⁻² d⁻¹ |
| CO₂ concentration ``C_t`` | 420 ppm (ambient) | 700 ppm (growth response saturates above) |

At harvest, a minimum yield of ``Y_\mathrm{min} = 8`` kg m⁻² must be reached (see [Crop Model and Prior](@ref case-study-model) for the choice of this value). The resources needed to reach the target depend on the uncertain parameters: a cultivar with low radiation use efficiency, for instance, needs more lighting for the same yield. Since only the fraction ``\mathit{HI} = 0.68`` of the above-ground biomass is harvestable, the requirement is

```math
\mathbb{P}\left[ \frac{Y_\mathrm{min}}{\mathit{HI}} - m_{\mathrm{B},t_\mathrm{h}} \leq 0 \right] \geq 1 - \alpha,
```

with ``\alpha = 5\,\%``. The risk level trades off production cost against the reliability of reaching the target.

In the code, the yield requirement is a [`BoxStateConstraint`](@ref) on the terminal biomass, the cost a [`FunctionalStateCost`](@ref), and the input bounds [`BoxInputConstraint`](@ref)s, combined into an [`OptimalControlProblem`](@ref) by `build_problem` in `setup.jl`.

!!! note "Implementation details"
    - The OCP has ``H + 1`` time points, ``t_0, \ldots, t_\mathrm{h}``. The input in the last column affects neither the state nor the cost and is pinned to the previous input, so that the solver does not see a free variable without curvature.
    - The first solve of each planning step is initialized with constant growth-maximizing inputs: temperature at the prior mean of ``T_\mathrm{opt}``, no drought, and maximal radiation and CO₂. Subsequent attempts are warm-started from the previous solution.
    - The scenario programs are solved with Ipopt using the HSL linear solver MA97, with separate options for cold and warm starts (`JuMPSolverConfig`). The iteration limit in the calibration loop is 1500: a stalled attempt is better abandoned than allowed to consume the computation time.
