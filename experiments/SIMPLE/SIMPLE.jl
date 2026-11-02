"""
    SIMPLE
 
Simulates plant growth based on environmental factors such as temperature,
radiation, drought, and CO₂ concentration.
 
# References
- Zhao (2019) *A SIMPLE crop model*
- Daniels (2023) *Optimal control for indoor vertical farms based on crop growth*
- Woli (2012) *Agricultural reference index for drought (ARID)*
"""
module SIMPLE

export SimpleParameters, SimpleState, SimpleInput
export next_state, get_yield

# -----------------------------------------------------------------------------
# Parameters
# -----------------------------------------------------------------------------

"""
    SimpleParameters{T1,T2,T3,T4,T5,T6,T7,T8,T9,T10,T11,T12,T13}

Parameters of the SIMPLE crop model.

Every field carries its own type parameter so that any subset can be a plain
`Float64`, or a a JuMP parameter.

Compared to the original paper, `I50B` is not included here because it
describes the initial state of the simulation, not a property of the dynamics.

# Fields
- `T_sum::T1`:          cumulative temperature requirement from sowing to maturity [°C d]
- `HI::T2`:             potential harvest index
- `I_50A::T3`:          cumulative temperature requirement for leaf area development
                        to intercept 50 % of radiation [°C d]
- `T_base::T4`:         base temperature for phenology development and growth [°C]
- `T_opt::T5`:          optimal temperature for biomass growth [°C]
- `RUE::T6`:            radiation use efficiency [kg MJ⁻¹ m⁻²]
- `I_50maxH::T7`:       maximum daily reduction in `I50B` due to heat stress [°C d]
- `I_50maxW::T8`:       maximum daily reduction in `I50B` due to drought stress [°C d]
- `T_max::T9`:         threshold temperature to start accelerating senescence from heat stress [°C]
- `T_ext::T10`:         extreme temperature threshold when RUE becomes 0 due to heat stress [°C]
- `S_CO2::T11`:         relative  increase in RUE per ppm CO₂ above 350 ppm
- `S_water::T12`:       sensitivity of RUE to drought stress (ARID index)
- `f_solar_max::T13`:   maximum fraction of radiation interception (management parameter)
"""
struct SimpleParameters{T1,T2,T3,T4,T5,T6,T7,T8,T9,T10,T11,T12,T13}
    T_sum::T1
    HI::T2
    I_50A::T3
    T_base::T4
    T_opt::T5
    RUE::T6
    I_50maxH::T7
    I_50maxW::T8
    T_max::T9
    T_ext::T10
    S_CO2::T11
    S_water::T12
    f_solar_max::T13
end

# -----------------------------------------------------------------------------
# State
# -----------------------------------------------------------------------------

"""
    SimpleState{T1, T2, T3}
 
State of the SIMPLE simulation at a single point in time.
 
# Fields
- `m_B::T1`:    above-ground biomass [kg m⁻²]
- `T_c::T2`:    cumulative temperature [°C d]
- `I_50B::T3`:  leaf senescence [°C d]
"""
struct SimpleState{T1,T2,T3}
    m_B::T1
    T_c::T2
    I_50B::T3
end

# -----------------------------------------------------------------------------
# Input
# -----------------------------------------------------------------------------

"""
    SimpleInput{T1, T2, T3, T4}
 
Environmental inputs to the SIMPLE model for a single time step.
 
# Fields
- `T::T1`:      air temperature [°C]
- `D::T2`:      relative drought level (ARID index; see Woli 2012)
- `R::T3`:      radiation [MJ m⁻² d⁻¹]
- `CO2::T4`:    atmospheric CO₂ concentration [ppm]
"""
mutable struct SimpleInput{T1,T2,T3,T4}
    T::T1
    D::T2
    R::T3
    CO2::T4
end

# -----------------------------------------------------------------------------
# Dynamics
# -----------------------------------------------------------------------------

include("dynamics.jl")

end