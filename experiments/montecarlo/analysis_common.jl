# =============================================================================
# analysis_common.jl — helpers shared by the Monte-Carlo analysis scripts
# =============================================================================
# Loading of the per-seed result files and the formatting of counts, quartiles,
# and text tables. Included by `analyze_open_loop.jl` and
# `analyze_receding_horizon.jl` after `setup.jl` (for `clopper_pearson_bound`).
# =============================================================================

using JLD2
using LaTeXStrings
using Logging

"""
    load_results(dir)

Load all `.jld2` result files in `dir`. Files that cannot be read (e.g., saved
by an older version of the code whose types no longer match) are skipped with a
warning that lists them.

# Arguments
- `dir::AbstractString`: result directory of one job array.

# Returns
- `Dict{Int,Any}`: seed ⇒ result; empty if `dir` does not exist.
"""
function load_results(dir::AbstractString)
    results = Dict{Int,Any}()
    if !isdir(dir)
        @warn "No results in $dir."
        return results
    end
    unreadable = String[]
    # JLD2 warns about every reconstructed type; silence it while loading.
    with_logger(NullLogger()) do
        for file in filter(f -> endswith(f, ".jld2"), readdir(dir; join=true))
            try
                r = load(file, "result")
                results[r.seed] = r
            catch
                push!(unreadable, basename(file))
            end
        end
    end
    if !isempty(unreadable)
        @warn "Skipped $(length(unreadable)) unreadable result files in $dir." files = join(sort(unreadable), ", ")
    end
    return results
end

"""
    method_result(results, key, seed)

Result of method `key` in run `seed`.

# Arguments
- `results::Dict{Int,Any}`: output of [`load_results`](@ref).
- `key::Symbol`: method, e.g., `:proposed`.
- `seed::Integer`: run.

# Returns
- The method's result NamedTuple, or `nothing` if the run is missing or failed.
"""
function method_result(results, key::Symbol, seed::Integer)
    r = get(results, seed, nothing)
    if r === nothing || r.status != :ok
        return nothing
    end
    return getproperty(r, key)
end

"""
    violation_upper_bound(k, n; confidence=0.95)

One-sided upper Clopper–Pearson bound on the violation rate across runs, from
`k` violating runs among `n` runs with a plan. Same formula as the certified
bound δ (`clopper_pearson_bound`), but applied to Monte-Carlo runs instead of
validation scenarios.

# Arguments
- `k::Integer`: number of violating runs.
- `n::Integer`: number of runs with a plan.

# Keywords
- `confidence::Real`: confidence level.

# Returns
- `Float64`: upper bound; `NaN` if `n == 0`.
"""
function violation_upper_bound(k::Integer, n::Integer; confidence::Real=0.95)
    if n == 0
        return NaN
    end
    return clopper_pearson_bound(k, n, 1 - confidence)
end

function percent(k, n)
    if n == 0
        return NaN
    else
        return 100 * k / n
    end
end

# "k/n (p %)", or "--" if there is nothing to count
function count_text(k, n)
    if n == 0
        return "--"
    end
    return @sprintf("%d/%d (%.1f %%)", k, n, percent(k, n))
end

# "median [q25, q75] unit" of `scale .* v`, or "--" if `v` is empty
function median_iqr(v; digits=2, scale=1, unit="")
    if isempty(v)
        return "--"
    end
    f = Printf.Format("%.$(digits)f [%.$(digits)f, %.$(digits)f]")
    return Printf.format(f, scale * median(v), scale * quantile(v, 0.25), scale * quantile(v, 0.75)) * unit
end

function section(title)
    println("\n", title)
    println("-"^length(title))
end

# Print rows of a label column followed by one column per planner.
function print_columns(labels, columns; width=24)
    for (i, label) in enumerate(labels)
        @printf("  %-34s", label)
        for column in columns
            print(rpad(column[i], width))
        end
        println()
    end
end

# (median, q25, q75) of `v`, or `nothing` if `v` is empty
function quartiles(v)
    if isempty(v)
        return nothing
    end
    return (median(v), quantile(v, 0.25), quantile(v, 0.75))
end

# "median [q25, q75] unit", optionally in parentheses; "--" if there are no values
function format_quartiles(q; digits=2, scale=1, unit="", parentheses=false)
    if q === nothing
        return "--"
    end
    f = Printf.Format("%.$(digits)f")
    med, lo, hi = (Printf.format(f, scale * x) for x in q)
    text = "$(med) [$(lo), $(hi)]"
    if !isempty(unit)
        text *= " " * unit
    end
    if parentheses
        text = "(" * text * ")"
    end
    return text
end
