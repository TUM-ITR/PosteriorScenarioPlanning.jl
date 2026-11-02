# =============================================================================
# run_montecarlo_single.jl — one Monte-Carlo run of the open-loop experiment
# =============================================================================
# Runs one method on one seed and saves the result to
#
#     <results-dir>/<method>/seed_<seed>.jld2
#
# with <method> one of
#   posterior  proposed method + certainty-equivalent baseline + hindsight oracle
#              (`posterior_planning.jl`)
#   prior      prior-based baseline (`prior_planning.jl`)
#   receding   receding-horizon control + first plan open-loop + hindsight
#              oracle (`receding_horizon.jl`)
#
# All methods face the same realization for the same seed (see
# `draw_realization` in `setup.jl`), so their results can be paired by seed in
# `analyze_open_loop.jl`; the receding runs are evaluated by
# `analyze_receding_horizon.jl`. Intended to be launched as one task of a SLURM array
# (see `run_montecarlo_slurm.sh`), one process per seed and method:
#
#     julia --project=experiments experiments/montecarlo/run_montecarlo_single.jl \
#         --method posterior --seed 1 --results-dir experiments/montecarlo/results/open_loop
# =============================================================================

using Printf
using Random
using JLD2

const METHODS = ("posterior", "prior", "receding")

# Parse command-line arguments: --method <posterior|prior|receding> --seed <Int> --results-dir <path>.
function parse_args(args::Vector{String})
    method = nothing
    seed = nothing
    results_dir = nothing

    i = 1
    while i <= length(args)
        arg = args[i]
        if arg == "--method"
            i += 1
            method = args[i]
        elseif arg == "--seed"
            i += 1
            seed = parse(Int, args[i])
        elseif arg == "--results-dir"
            i += 1
            results_dir = args[i]
        else
            error("Unknown argument: $arg")
        end
        i += 1
    end

    if method === nothing || seed === nothing || results_dir === nothing
        error("Usage: julia run_montecarlo_single.jl --method <posterior|prior|receding> --seed <Int> --results-dir <path>")
    end
    if !(method in METHODS)
        error("Unknown method '$method'; expected one of $(METHODS).")
    end

    return method, seed, results_dir
end

# Return the current git commit hash for provenance, or "unknown" if unavailable
# (e.g. detached checkout inside a container without git).
function get_git_commit()
    try
        return readchomp(`git -C $(@__DIR__) rev-parse HEAD`)
    catch
        return "unknown"
    end
end

# The names avoid `seed`, which the experiment scripts define as a script setting.
const mc_method, mc_seed, mc_results_dir = parse_args(ARGS)

# Load only the experiment of the requested method, without executing its entry
# block (the single-seed run + plotting).
if mc_method == "posterior"
    RUN_POSTERIOR_PLANNING_ENTRYPOINT = false
    include(joinpath(@__DIR__, "..", "posterior_planning.jl"))
elseif mc_method == "prior"
    RUN_PRIOR_PLANNING_ENTRYPOINT = false
    include(joinpath(@__DIR__, "..", "prior_planning.jl"))
else
    RUN_RECEDING_HORIZON_ENTRYPOINT = false
    include(joinpath(@__DIR__, "..", "receding_horizon.jl"))
end

# Run the requested method on the requested seed.
function run_method(method::AbstractString, seed::Integer)
    if method == "posterior"
        return run_posterior_planning(seed)
    elseif method == "prior"
        return run_prior_planning(seed)
    else
        return run_receding_horizon(seed)
    end
end

"""
    main(method, seed, results_dir)

Run `method` on `seed` and save the result to `<results_dir>/<method>/seed_<seed>.jld2`.

Always saves a `result` NamedTuple — even on failure — so a failed seed leaves a
visible record instead of a missing file that cannot be told apart from a job
that is still running. The hardware is recorded because the paper reports
wall-clock times.

# Arguments
- `method::AbstractString`: `"posterior"`, `"prior"`, or `"receding"`.
- `seed::Integer`: index of the Monte-Carlo run.
- `results_dir::AbstractString`: root directory of the results of this study.

# Returns
- `nothing`
"""
function main(method::AbstractString, seed::Integer, results_dir::AbstractString)
    method_dir = joinpath(results_dir, method)
    isdir(method_dir) || mkpath(method_dir)
    results_file = joinpath(method_dir, "seed_$(seed).jld2")

    @printf("Running method = %s with seed = %d\n", method, seed)
    flush(stdout)

    provenance = (
        seed=seed,
        method=Symbol(method),
        julia_version=string(VERSION),
        git_commit=get_git_commit(),
        nthreads=Threads.nthreads(),
        hostname=gethostname(),
        cpu_model=Sys.cpu_info()[1].model,
    )

    t_start = time()
    result = try
        sim = run_method(method, seed)
        merge((status=:ok, error=nothing), provenance, sim)
    catch e
        bt = catch_backtrace()
        @error "Simulation failed" method seed exception = (e, bt)
        merge((status=:failed, error=sprint(showerror, e)), provenance)
    end
    runtime_total = time() - t_start
    result = merge(result, (runtime_total=runtime_total,))

    @printf("Finished method = %s with seed = %d (status = %s, %.1f s)\n",
        method, seed, result.status, runtime_total)
    flush(stdout)

    jldsave(results_file; result=result)
    println("Results saved to $results_file")
    return nothing
end

main(mc_method, mc_seed, mc_results_dir)
