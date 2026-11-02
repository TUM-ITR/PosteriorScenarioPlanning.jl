# -----------------------------------------------------------------------------
# Smoothed Operators
# -----------------------------------------------------------------------------

const _EPS = 1.0e-4

"""
    smax(a, b, eps=_EPS)
 
Smoothed maximum operator `max(a, b)`.
 
# Arguments
- `a`:   first operand
- `b`:   second operand
- `eps`: smoothing parameter (in squared units of `a` and `b`)
 
# Returns
- smoothed maximum of `a` and `b`
"""
@inline function smax(a, b, eps=_EPS)
    return (a + b + sqrt((a - b)^2 + eps)) / 2
end

"""
    smin(a, b, eps=_EPS)
 
Smoothed minimum operator `min(a, b)`.
 
# Arguments
- `a`:   first operand
- `b`:   second operand
- `eps`: smoothing parameter, see [`smax`](@ref)
 
# Returns
- smoothed minimum of `a` and `b`
"""
@inline function smin(a, b, eps=_EPS)
    return -smax(-a, -b, eps)
end

"""
    sclip(a, lb, ub, eps=_EPS)
 
Smoothed clipping operator `max(lb, min(a, ub))`.
 
# Arguments
- `a`:   value to clip
- `lb`:  lower bound
- `ub`:  upper bound
- `eps`: smoothing parameter, see [`smax`](@ref)
 
# Returns
- smoothed clipping of `a` to `[lb, ub]`
"""
@inline function sclip(a, lb, ub, eps=_EPS)
    return smax(lb, smin(a, ub, eps), eps)
end

# -----------------------------------------------------------------------------
# Sub-Functions (see Zhao 2019) 
# -----------------------------------------------------------------------------

"""
    f_solar(T_c, I_50B, p)
 
Fraction of solar radiation intercepted by the crop canopy.
 
# Arguments
- `T_c`:    cumulative temperature [°C d]
- `I_50B`:  leaf senescence [°C d]
- `p`:      [`SimpleParameters`](@ref)
 
# Returns
- intercepted radiation fraction
"""
@inline function f_solar(T_c, I_50B, p::SimpleParameters)
    f_solar1 = p.f_solar_max / (1.0 + exp(-0.01 * (T_c - p.I_50A)))
    f_solar2 = p.f_solar_max / (1.0 + exp(0.01 * (T_c - (p.T_sum - I_50B))))
    return smin(f_solar1, f_solar2)
end

"""
    f_temp(T, p)
 
Impact of temperature on biomass growth rate.
 
# Arguments
- `T`:  air temperature [°C]
- `p`:  [`SimpleParameters`](@ref)
 
# Returns
- temperature growth factor in [0, 1]
"""
@inline function f_temp(T, p::SimpleParameters)
    return sclip((T - p.T_base) / (p.T_opt - p.T_base), 0.0, 1.0)
end

"""
    f_heat(T, p)
 
Impact of heat stress on biomass growth rate.
 
# Arguments
- `T`:  air temperature [°C] (assumed constant over the day)
- `p`:  [`SimpleParameters`](@ref)
 
# Returns
- heat stress factor in [0, 1]
"""
@inline function f_heat(T, p::SimpleParameters)
    return sclip(1.0 - (T - p.T_max) / (p.T_ext - p.T_max), 0.0, 1.0)
end

"""
    f_CO2(CO2, p)
 
Impact of CO₂ concentration on radiation use efficiency (RUE).
 
# Arguments
- `CO2`:    atmospheric CO₂ concentration [ppm]
- `p`:      [`SimpleParameters`](@ref)
 
# Returns
- CO₂ enhancement factor in [1, 1 + S_CO2 × 350]
"""
@inline function f_CO2(CO2, p::SimpleParameters)
    return sclip(1.0 + p.S_CO2 * (CO2 - 350.0), 1.0, 1.0 + p.S_CO2 * 350.0)
end

"""
    f_water(D, p)
 
Drought stress impact on radiation use efficiency (RUE).
 
# Arguments
- `D`:  relative drought level (ARID index; see Woli 2012)
- `p`:  [`SimpleParameters`](@ref)
 
# Returns
- drought stress factor in [0, 1]
"""
@inline function f_water(D, p::SimpleParameters)
    return sclip(1.0 - p.S_water * D, 0.0, 1.0)
end

"""
    f_solar_water(D, p)
 
Drought stress impact on radiation interception.
 
# Arguments
- `D`:  relative drought level (ARID index; see Woli 2012)
- `p`:  [`SimpleParameters`](@ref)
 
# Returns
- drought interception factor in [0.9, 1]
"""
@inline function f_solar_water(D, p::SimpleParameters)
    return sclip(1.9 - p.S_water * D, 0.9, 1.0)
end

# -----------------------------------------------------------------------------
# State Transition
# -----------------------------------------------------------------------------

"""
    next_state(state, input, p)
 
Compute the next state of the SIMPLE model given the current state,
environmental input, and parameters.
 
# Arguments
- `state`:  current [`SimpleState`](@ref)
- `input`:  current [`SimpleInput`](@ref)
- `p`:      [`SimpleParameters`](@ref)
 
# Returns
- [`SimpleState`](@ref) at the next time step
"""
@inline function next_state(state::SimpleState, input::SimpleInput, p::SimpleParameters)
    f_solar_ = f_solar(state.T_c, state.I_50B, p)
    f_temp_ = f_temp(input.T, p)
    f_heat_ = f_heat(input.T, p)
    f_CO2_ = f_CO2(input.CO2, p)
    f_water_ = f_water(input.D, p)
    f_solar_water_ = f_solar_water(input.D, p)

    m_B_next = state.m_B + input.R * f_solar_ * p.RUE * f_CO2_ * f_temp_ * f_solar_water_ * smin(f_heat_, f_water_)
    T_c_next = state.T_c + smax(input.T - p.T_base, 0.0)
    I_50B_next = state.I_50B + smax(p.I_50maxW * (1.0 - f_water_), p.I_50maxH * (1.0 - f_heat_))

    return SimpleState(m_B_next, T_c_next, I_50B_next)
end

# -----------------------------------------------------------------------------
# Yield
# -----------------------------------------------------------------------------

"""
    get_yield(state, p)
 
Compute the crop yield from the terminal state.
 
# Arguments
- `state`:  terminal [`SimpleState`](@ref)
- `p`:      [`SimpleParameters`](@ref)
 
# Returns
- yield [kg m⁻²]
"""
function get_yield(state::SimpleState, p::SimpleParameters)
    return state.m_B * p.HI
end