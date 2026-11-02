# =============================================================================
# analyze_open_loop.jl — evaluation of the open-loop Monte-Carlo study
# =============================================================================
# Loads the per-seed results written by the two open-loop job arrays of
# `run_montecarlo_slurm.sh`:
#
#     <RESULTS_DIR>/posterior/seed_<s>.jld2   proposed, certainty equivalent, oracle
#     <RESULTS_DIR>/prior/seed_<s>.jld2       prior-based
#
# and reports the results of the open-loop experiment:
#   1. Method comparison: plans found, violations, δ, and cost (absolute and
#      relative to the oracle), plus the violation-rate bound across runs and
#      the OCP failures of the deterministic planners.
#   2. Certification statistics of the two certification-based planners
#      (attempts, acceptance band, recovery).
#   3. Computation times per step: sampling (PMCMC warm-up and validation draws)
#      and optimization (OCP build and solves).
#   4. Representative run: summary and figure (plot and .txt data for pgfplots).
#
# Two Clopper–Pearson bounds appear, and they answer different questions:
#   (a) δ is the certificate the planner produces for a single run: a one-sided
#       upper bound on the violation probability under the posterior, computed
#       from the K_val validation scenarios of that run.
#   (b) The violation-rate bound in section 1 is a one-sided upper bound on
#       the fraction of Monte Carlo runs whose plan violates the yield target on
#       the true system. It quantifies only the Monte Carlo uncertainty of that
#       fraction and is an empirical check of the guarantee, not a certificate.
# =============================================================================

# Includes `setup.jl` and the printout and figure of a single run
# (`print_open_loop_run`, `plot_open_loop_run`) without running the experiment.
RUN_POSTERIOR_PLANNING_ENTRYPOINT = false
include(joinpath(@__DIR__, "..", "posterior_planning.jl"))
include(joinpath(@__DIR__, "..", "data2txt.jl"))
include(joinpath(@__DIR__, "analysis_common.jl"))

# -----------------------------------------------------------------------------
# Settings
# -----------------------------------------------------------------------------
RESULTS_DIR = joinpath(@__DIR__, "results", "open_loop")
PLOT_DIR = joinpath(RESULTS_DIR, "analysis")  # figures and .txt exports
seeds_to_plot = [7] # seeds of the representative run
save_plots = true
show_plots = true
export_txt = true
confidence_violations = 0.95 # confidence of the violation-rate bound across runs

# Methods compared: result key, label, and the result set it is stored in.
METHODS = [
    (:proposed, "Proposed", :posterior),
    (:prior_based, "Prior-based", :prior),
    (:certainty_equivalent, "CE", :posterior),
    (:oracle, "Oracle", :posterior),
]

# -----------------------------------------------------------------------------
# Helpers (shared ones in `analysis_common.jl`)
# -----------------------------------------------------------------------------
# A plan is found if a certification-based planner certifies (δ ≤ α), or if the
# deterministic OCP of the certainty-equivalent planner or the oracle is solved.
function plan_found(m)
    if m === nothing
        return false
    elseif haskey(m, :certified)
        return m.certified
    else
        return m.success
    end
end

# -----------------------------------------------------------------------------
# 1. Method comparison
# -----------------------------------------------------------------------------
# Methods whose cost is compared: they find a plan in (almost) every run. The
# prior-based planner is excluded, as it may certify in (almost) no run.
COST_METHODS = (:proposed, :certainty_equivalent, :oracle)

"""
    table_rows(results, seeds)

Entries of the method comparison, one row per method in `METHODS`.

# Arguments
- `results::Dict{Symbol,Dict{Int,Any}}`: `:posterior` and `:prior` results.
- `seeds::Vector{Int}`: all runs.

# Returns
- `rows`: NamedTuples with fields `key`, `label`, `n_found`, `n_violations`,
  `delta` (quartiles), `delta_certified`, `ratio` and `cost` (quartiles).
- `common::Vector{Int}`: runs in which all methods of `COST_METHODS` found a plan.

# Notes
- Violations are counted among the runs in which the method found a plan.
- δ is taken over the certified runs. If a planner certifies in no run, the
  smallest δ reached in each run is reported instead (`delta_certified = false`).
- J/J_oracle and J are taken over the common subset `common`; `nothing` for
  methods outside `COST_METHODS`.
"""
function table_rows(results, seeds)
    m(key, source, s) = method_result(results[source], key, s)

    common = filter(s -> all(key -> plan_found(m(key, :posterior, s)), COST_METHODS), seeds)

    rows = []
    for (key, label, source) in METHODS
        found = filter(s -> plan_found(m(key, source, s)), seeds)
        n_violations = count(s -> !m(key, source, s).feasible, found)

        # δ of the certification-based planners
        planners = filter(!isnothing, [m(key, source, s) for s in seeds])
        delta = nothing
        delta_certified = true
        if !isempty(planners) && haskey(first(planners), :delta)
            certified = filter(p -> p.certified, planners)
            if !isempty(certified)
                delta = quartiles([p.delta for p in certified])
            else
                delta = quartiles([minimum(p.delta_history) for p in planners])
                delta_certified = false
            end
        end

        if key in COST_METHODS
            ratio = quartiles([m(key, source, s).cost / m(:oracle, :posterior, s).cost for s in common])
            cost = quartiles([m(key, source, s).cost for s in common])
        else
            ratio = nothing
            cost = nothing
        end

        push!(rows, (key=key, label=label, n_found=length(found), n_violations=n_violations,
            delta=delta, delta_certified=delta_certified, ratio=ratio, cost=cost))
    end
    return rows, common
end

# "k/n" of violating runs, or "--" for the oracle and methods without a plan
function violations_text(row)
    if row.key == :oracle || row.n_found == 0
        return "--"
    end
    return "$(row.n_violations)/$(row.n_found)"
end

function ratio_text(row)
    if row.key == :oracle
        return "1"
    end
    return format_quartiles(row.ratio; digits=3)
end

function delta_text(row)
    return format_quartiles(row.delta; digits=1, scale=100, unit="%",
        parentheses=(!row.delta_certified))
end

"""
    print_table(results, seeds)

Print the method comparison of the open-loop experiment and the violation-rate
bound across runs. See [`table_rows`](@ref) for the entries.

# Arguments
- `results::Dict{Symbol,Dict{Int,Any}}`: `:posterior` and `:prior` results.
- `seeds::Vector{Int}`: all runs.
"""
function print_table(results, seeds)
    R = length(seeds)
    rows, common = table_rows(results, seeds)

    section("1. Method comparison (R = $R runs)")
    @printf("  %-14s %-12s %-12s %-26s %-26s %-24s\n", "Method", "Plan found", "Violations", "δ",
        "J [€/m²]", "J/J_oracle")
    for row in rows
        @printf("  %-14s %-12s %-12s %-26s %-26s %-24s\n", row.label, "$(row.n_found)/$R",
            violations_text(row), delta_text(row), format_quartiles(row.cost), ratio_text(row))
    end

    println()
    println("  Plan found: certified (δ ≤ α) for Proposed/Prior-based; OCP solved for CE/Oracle.")
    println("  Violations: runs missing the yield target on the true system / runs with a plan.")
    println("  δ:          median [IQR] over the certified runs; in parentheses if no run is")
    println("              certified: median [IQR] of the smallest δ reached per run.")
    @printf("  J:          median [IQR] over the %d runs in which %s all found a plan.\n",
        length(common), join([label for (key, label, _) in METHODS if key in COST_METHODS], ", "))

    # One-sided Clopper–Pearson upper bound on the violation rate across runs
    # (Monte-Carlo uncertainty of the violation fraction, not δ)
    println()
    @printf("  Violation rate on the true system, %d %% upper confidence bound over the runs with a plan:\n",
        round(Int, 100 * confidence_violations))
    for row in rows
        if row.key == :oracle
            continue
        end
        if row.n_found == 0
            upper_text = "-- (no plan)"
        else
            upper = violation_upper_bound(row.n_violations, row.n_found; confidence=confidence_violations)
            upper_text = @sprintf("%.1f %%", 100 * upper)
        end
        @printf("  %-14s %s\n", row.label, upper_text)
    end

    print_solve_failures(results, seeds)
end

"""
    print_solve_failures(results, seeds)

Break down the runs in which the deterministic OCP of the certainty-equivalent
planner or the oracle was not solved by termination status. `LOCALLY_INFEASIBLE`
means the local solver converged to a point of locally minimal constraint
violation without meeting the yield target. As the OCP is nonconvex, this does
not prove that the target is unattainable; a different initialization might find
a feasible plan. Every other status (e.g. `ITERATION_LIMIT` with an infeasible
last iterate) indicates that the solver stopped without converging.

# Arguments
- `results::Dict{Symbol,Dict{Int,Any}}`: `:posterior` and `:prior` results.
- `seeds::Vector{Int}`: all runs.

# Notes
- Results saved before the termination status was recorded show `unknown`.
"""
function print_solve_failures(results, seeds)
    println()
    println("  OCP not solved (CE/Oracle), by termination status:")
    for (key, label, source) in METHODS
        runs = filter(!isnothing, [method_result(results[source], key, s) for s in seeds])
        if isempty(runs) || haskey(first(runs), :certified)
            continue
        end

        failed = filter(m -> !m.success, runs)
        counts = Dict{Symbol,Int}()
        for m in failed
            status = get(m, :termination_status, :unknown)
            counts[status] = get(counts, status, 0) + 1
        end

        if isempty(counts)
            breakdown = "none"
        else
            breakdown = join(["$status: $n" for (status, n) in sort(collect(counts); by=last, rev=true)], ", ")
        end
        @printf("  %-14s %d/%d   %s\n", label, length(failed), length(runs), breakdown)
    end
end

# -----------------------------------------------------------------------------
# 2. Certification and 3. computation times
# -----------------------------------------------------------------------------
# Successful runs of one job array and the certification-based planner `key` in them.
function planner_runs(results, key::Symbol)
    runs = sort([r for r in values(results) if r.status == :ok]; by=r -> r.seed)
    return runs, [getproperty(r, key) for r in runs]
end

entered_recovery(p) = any(s -> s.recovery_active, p.update_history)

# Column of the certification statistics for one planner (rows: see `print_certification`).
function certification_column(planners)
    certified = filter(p -> p.certified, planners)
    uncertified = filter(p -> !p.certified, planners)
    deltas = [p.delta for p in certified]
    n_in_band = count(d -> alpha - alpha_tol <= d <= alpha, deltas)

    return [
        count_text(length(certified), length(planners)),
        median_iqr(deltas; scale=100, unit=" %"),
        count_text(n_in_band, length(certified)),
        median_iqr(Float64[p.attempts for p in certified]; digits=1),
        count_text(count(entered_recovery, planners), length(planners)),
        median_iqr([minimum(p.delta_history) for p in uncertified]; scale=100, unit=" %"),
    ]
end

"""
    print_certification(results)

Print the certification statistics of the proposed and the prior-based planner
side by side. δ is the certified violation bound of a run: the one-sided upper
Clopper–Pearson bound from its K_val validation scenarios at confidence 1 − β/n_max.

# Arguments
- `results::Dict{Symbol,Dict{Int,Any}}`: `:posterior` and `:prior` results.
"""
function print_certification(results)
    section(@sprintf("2. Certification (α = %g %%, band [%g, %g] %%, 1 − β = %g %%, n_max = %d, K_val = %d)",
        100 * alpha, round(100 * (alpha - alpha_tol); digits=6), 100 * alpha, 100 * (1 - beta), n_max, K_val))
    labels = [
        "Certified",
        "δ (certified runs)",
        "δ in acceptance band",
        "Attempts (certified runs)",
        "Entered recovery",
        "Not certified: smallest δ reached",
    ]
    _, proposed = planner_runs(results[:posterior], :proposed)
    _, prior_based = planner_runs(results[:prior], :prior_based)
    print_columns([""; labels], [["Proposed"; certification_column(proposed)],
        ["Prior-based"; certification_column(prior_based)]])
    println("\n  δ, attempts: median [IQR].")
end

# Mean runtimes for one planner, split into the two computational steps:
#   sampling:     scenario generation (PMCMC warm-up for the proposed planner, prior
#                 draws for the prior-based one) plus the validation draws of all
#                 attempts, which continue the chains. `runtime_validation` also
#                 contains counting violations and evaluating the validation cost
#                 on the drawn scenarios; this is a forward simulation and small
#                 next to the draws, so it is attributed to sampling.
#   optimization: building the scenario OCP and solving it in all attempts.
# The total is the wall-clock time of the run; it differs from the sum of both
# steps only by bookkeeping (calibration updates, printing).
function runtime_column(runs, planners)
    if isempty(runs)
        return fill("n/a", 3)
    end
    sampling = mean(r.runtime_sampling + p.runtime_validation for (r, p) in zip(runs, planners))
    optimization = mean(p.runtime_build + p.runtime_optimization for p in planners)
    total = mean(r.runtime_sampling + p.runtime_build + p.runtime_plan for (r, p) in zip(runs, planners))
    return [
        @sprintf("%.1f s (%.1f min)", sampling, sampling / 60),
        @sprintf("%.1f s (%.1f min)", optimization, optimization / 60),
        @sprintf("%.1f s (%.1f min)", total, total / 60),
    ]
end

"""
    print_computation(results)

Print the mean runtimes per run, split into sampling and optimization (see
`runtime_column`), and the hardware they were measured on.

# Arguments
- `results::Dict{Symbol,Dict{Int,Any}}`: `:posterior` and `:prior` results.
"""
function print_computation(results)
    section("3. Computation (mean per run)")
    posterior_runs, proposed = planner_runs(results[:posterior], :proposed)
    prior_runs, prior_based = planner_runs(results[:prior], :prior_based)
    print_columns(["", "Sampling", "Optimization", "Total"],
        [["Proposed"; runtime_column(posterior_runs, proposed)],
            ["Prior-based"; runtime_column(prior_runs, prior_based)]])
    println("\n  Sampling:     PMCMC warm-up (Proposed) or prior draws (Prior-based), plus the")
    println("                validation draws of all attempts.")
    println("  Optimization: OCP build and solves of all attempts.")

    hardware = unique([(get(r, :cpu_model, "unknown"), r.nthreads) for r in [posterior_runs; prior_runs]])
    for (cpu, nthreads) in hardware
        @printf("\n  Hardware: %s, %d Julia threads", cpu, nthreads)
    end
    println()
end

# -----------------------------------------------------------------------------
# 4. Representative run
# -----------------------------------------------------------------------------
function finalize_plot(p, name)
    if save_plots
        mkpath(PLOT_DIR)
        savefig(p, joinpath(PLOT_DIR, name * ".pdf"))
    end
    if show_plots
        display(p)
    end
    return p
end

"""
    single_run_figure(post, pri)

Representative run: the printout and figure of `posterior_planning.jl`
([`print_open_loop_run`](@ref), [`plot_open_loop_run`](@ref)), with the
prior-based planner added to the printout. The data (including the realized
prior-based trajectory, which is not plotted) are also written to
`seed_<s>_single_run.txt` for pgfplots.

# Arguments
- `post`: `posterior_planning.jl` result of the seed.
- `pri`: `prior_planning.jl` result of the seed, or `nothing`.
"""
function single_run_figure(post, pri)
    s = post.seed
    proposed = post.proposed
    ce = post.certainty_equivalent
    t = collect(0:H)
    day = t0 .+ t
    target = fill(minimum_yield / HI, H + 1)  # terminal biomass that meets the yield target
    if pri !== nothing
        prior_based = pri.prior_based
        prior_realized = prior_based.X_true[1, :]
    else
        prior_based = nothing
        prior_realized = fill(NaN, H + 1)
    end

    section("4. Representative run: Seed $s")
    if pri === nothing
        println("  Prior-based: no result")
    end
    print_open_loop_run(post; prior_based=prior_based)
    finalize_plot(plot_open_loop_run(post), "seed_$(s)_single_run")

    if export_txt
        mkpath(PLOT_DIR)
        file = joinpath(PLOT_DIR, "seed_$(s)_single_run.txt")
        data2txt(file,
            "t", t,
            "day", day,
            "lower", proposed.pred_final.q05[1, :],
            "upper", proposed.pred_final.q95[1, :],
            "mean_initial", proposed.pred_initial.mean[1, :],
            "mean_final", proposed.pred_final.mean[1, :],
            "proposed", proposed.X_true[1, :],
            "ce", ce.X_true[1, :],
            "ce_nominal", ce.X_nominal[1, :],
            "prior", prior_realized,
            "oracle", post.oracle.X_true[1, :],
            "target", target)
        println("  Wrote $file")
    end
end

# -----------------------------------------------------------------------------
# Run
# -----------------------------------------------------------------------------
isdir(RESULTS_DIR) || error("RESULTS_DIR does not exist: $RESULTS_DIR")

results = Dict(
    :posterior => load_results(joinpath(RESULTS_DIR, "posterior")),
    :prior => load_results(joinpath(RESULTS_DIR, "prior")),
)
seeds = sort(collect(union(keys(results[:posterior]), keys(results[:prior]))))

println("="^80)
println("Open-loop Monte-Carlo study: $RESULTS_DIR")
for (source, rs) in results
    failed = sort([r for r in values(rs) if r.status != :ok]; by=r -> r.seed)
    @printf("  %-9s %d runs loaded, %d failed\n", source, length(rs), length(failed))
    for r in failed
        @printf("    seed %d: %s\n", r.seed, first(split(string(r.error), '\n')))
    end
end
println("="^80)

print_table(results, seeds)
print_certification(results)
print_computation(results)

for s in seeds_to_plot
    post = get(results[:posterior], s, nothing)
    if post === nothing || post.status != :ok
        @warn "No successful posterior result for seed $s; skipping its figure."
        continue
    end
    pri = get(results[:prior], s, nothing)
    if pri !== nothing && pri.status != :ok
        pri = nothing
    end
    single_run_figure(post, pri)
end
