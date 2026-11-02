# =============================================================================
# analyze_receding_horizon.jl — evaluation of the receding-horizon Monte-Carlo study
# =============================================================================
# Loads the per-seed results written by the `receding` job array of
# `run_montecarlo_slurm.sh`:
#
#     <RESULTS_DIR>/receding/seed_<s>.jld2   proposed (receding horizon), its
#                                            first plan (open loop), CE
#                                            (receding horizon), oracle
#
# and reports the results of the receding-horizon experiment:
#   1. Method comparison: plan in all cycles, violations, and cost (absolute and
#      relative to the oracle), plus the violation-rate bound across runs.
#   2. Per cycle: certification and calibration of the proposed planner and
#      feasibility of the CE planner.
#   3. Value of replanning: the proposed planner against its first plan applied
#      open-loop, paired by seed.
#   4. Computation: the first cycle (the open-loop planning step) against the
#      later, warm-started cycles.
#   5. Representative run: summary and figure (plot and .txt data for pgfplots),
#      plus one .txt per planning cycle for an animation (`export_cycle_frames`).
#
# The key numbers of sections 1 and 4 are also written to
# `<PLOT_DIR>/receding_horizon_results.txt`.
#
# Only runs certified at t₀ are evaluated. Afterwards both planners keep
# replanning: the proposed one applies the smallest-δ candidate in uncertified
# cycles, the CE one the last feasible plan in infeasible cycles (see
# `receding_horizon.jl`).
# =============================================================================

# Includes `setup.jl` and the printout and figure of a single run
# (`print_receding_horizon_run`, `plot_receding_horizon_run`) without running
# the experiment.
RUN_RECEDING_HORIZON_ENTRYPOINT = false
include(joinpath(@__DIR__, "..", "receding_horizon.jl"))
include(joinpath(@__DIR__, "..", "data2txt.jl"))
include(joinpath(@__DIR__, "analysis_common.jl"))

# -----------------------------------------------------------------------------
# Settings
# -----------------------------------------------------------------------------
RESULTS_DIR = joinpath(@__DIR__, "results", "receding_horizon")
PLOT_DIR = joinpath(RESULTS_DIR, "analysis")  # figures and .txt exports
seeds_to_plot = [7] # seeds of the representative run (same run as in the open-loop analysis)
save_plots = true
show_plots = true
export_txt = true
confidence_violations = 0.95 # confidence of the violation-rate bound across runs
omega_max_logit = 0.999 # cap on ω before taking logit(ω), as `omega_start_max` in `receding_horizon.jl`

# -----------------------------------------------------------------------------
# Helpers (shared ones in `analysis_common.jl`)
# -----------------------------------------------------------------------------
# Outcome of the proposed planner: `:all_certified`, `:partially_certified`, or
# `:uncertified_at_t0` (see `receding_horizon.jl`).
closed_loop_status(run) = run.proposed.status

runs_with(runs, status::Symbol) = filter(r -> closed_loop_status(r) === status, runs)

# Runs certified at t₀, i.e., the evaluated runs.
evaluated_runs(runs) = filter(r -> closed_loop_status(r) !== :uncertified_at_t0, runs)

# Whether the CE OCP was solved at t₀ and in every cycle, respectively.
ce_plan_at_t0(run) = run.certainty_equivalent.cycles[1].success
ce_plan_in_all_cycles(run) = run.certainty_equivalent.n_success == length(run.t_cycles)

# logit(ω) = log λ, with ω clamped to [ω_min, `omega_max_logit`] so that it is
# finite at the end of the recovery ramp (ω = 1).
function logit_omega(omega)
    w = clamp(omega, omega_min, omega_max_logit)
    return log(w / (1 - w))
end

# Relative reduction from `before` to `after`, in %.
reduction(before, after) = 100 * (1 - after / before)

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

# -----------------------------------------------------------------------------
# 1. Method comparison
# -----------------------------------------------------------------------------
"""
    table_entries(runs)

Entries of the method comparison, for the runs certified at t₀.

# Arguments
- `runs::Vector`: successful results, sorted by seed.

# Returns
- `NamedTuple` with
  - `n_evaluated`: runs certified at t₀,
  - `n_ce_t0`: of those, runs whose CE OCP is solved at t₀,
  - `n_common`: of those, runs with an oracle solution; costs are taken over them,
  - `proposed_all_cycles`, `proposed_violations`, `first_plan_violations`: among
    the `n_evaluated` runs,
  - `ce_all_cycles`, `ce_violations`: among the `n_ce_t0` runs,
  - `cost` and `ratio`: quartiles of J and J/J_oracle per method key
    (`:proposed`, `:first_plan`, `:certainty_equivalent`, `:oracle`).

# Notes
- Violations include runs in which a later cycle finds no new plan.
"""
function table_entries(runs)
    evaluated = evaluated_runs(runs)
    ce_t0 = filter(ce_plan_at_t0, evaluated)
    common = filter(r -> r.oracle.success, ce_t0)
    methods = (:proposed, :first_plan, :certainty_equivalent, :oracle)
    cost = Dict(key => quartiles([getproperty(r, key).cost for r in common]) for key in methods)
    ratio = Dict(key => quartiles([getproperty(r, key).cost / r.oracle.cost for r in common]) for key in methods)
    return (
        n_evaluated=length(evaluated),
        n_ce_t0=length(ce_t0),
        n_common=length(common),
        proposed_all_cycles=length(runs_with(evaluated, :all_certified)),
        proposed_violations=count(r -> !r.proposed.feasible, evaluated),
        first_plan_violations=count(r -> !r.first_plan.feasible, evaluated),
        ce_all_cycles=count(ce_plan_in_all_cycles, ce_t0),
        ce_violations=count(r -> !r.certainty_equivalent.feasible, ce_t0),
        cost=cost,
        ratio=ratio,
    )
end

"""
    print_table(runs)

Print the method comparison of the receding-horizon experiment (see
[`table_entries`](@ref)) and the violation-rate bound across runs.

# Arguments
- `runs::Vector`: successful results, sorted by seed.

# Returns
- `NamedTuple`: output of [`table_entries`](@ref).
"""
function print_table(runs)
    e = table_entries(runs)
    n = e.n_evaluated
    n_ce = e.n_ce_t0

    # (label, key, plan in all cycles, violations, number of violations, runs with a plan)
    rows = [
        ("Proposed", :proposed, "$(e.proposed_all_cycles)/$n", "$(e.proposed_violations)/$n",
            e.proposed_violations, n),
        ("Open loop", :first_plan, "--", "$(e.first_plan_violations)/$n", e.first_plan_violations, n),
        ("CE", :certainty_equivalent, "$(e.ce_all_cycles)/$n_ce", "$(e.ce_violations)/$n_ce",
            e.ce_violations, n_ce),
        ("Oracle", :oracle, "--", "--", 0, 0),
    ]

    section("1. Method comparison ($n of $(length(runs)) runs certified at t₀)")
    @printf("  %-12s %-20s %-12s %-26s %-24s\n", "Method", "Plan in all cycles", "Violations", "J [€/m²]",
        "J/J_oracle")
    for (label, key, plan_text, violations_text, _, _) in rows
        if key === :oracle
            ratio_text = "1"
        else
            ratio_text = format_quartiles(e.ratio[key]; digits=3)
        end
        @printf("  %-12s %-20s %-12s %-26s %-24s\n", label, plan_text, violations_text,
            format_quartiles(e.cost[key]), ratio_text)
    end

    println()
    println("  Open loop:          first plan of Proposed (at t₀), applied without replanning.")
    println("  Plan in all cycles: certified (Proposed) or OCP solved (CE) in every cycle.")
    println("  Violations:         runs missing the yield target, incl. runs without a new plan")
    println("                      in a later cycle.")
    @printf("  J:                  median [IQR] over the %d runs in which CE and Oracle also find\n", e.n_common)
    println("                      a plan at t₀.")

    # One-sided Clopper–Pearson upper bound on the violation rate across runs
    # (Monte-Carlo uncertainty of the violation fraction, not δ)
    println()
    @printf("  Violation rate on the true system, %d %% upper confidence bound over the runs with a plan:\n",
        round(Int, 100 * confidence_violations))
    for (label, key, _, _, k, n_plan) in rows
        if key === :oracle
            continue
        end
        if n_plan == 0
            upper_text = "-- (no plan)"
        else
            upper = violation_upper_bound(k, n_plan; confidence=confidence_violations)
            upper_text = @sprintf("%.1f %%", 100 * upper)
        end
        @printf("  %-12s %s\n", label, upper_text)
    end
    return e
end

# -----------------------------------------------------------------------------
# 2. Per cycle
# -----------------------------------------------------------------------------
"""
    print_cycles(runs)

Print, per cycle and over the runs certified at t₀, how many runs the proposed
planner certifies, its mean number of attempts, the distance |Δ logit ω| between
the warm-started and the final risk weight, and how many runs the CE OCP solves.

# Arguments
- `runs::Vector`: successful results, sorted by seed.

# Notes
- |Δ logit ω| = |log λ_final − log λ_start| measures how well the risk weight
  carried over from the previous cycle predicted the calibrated one.
"""
function print_cycles(runs)
    evaluated = evaluated_runs(runs)
    section("2. Per cycle (runs certified at t₀)")
    if isempty(evaluated)
        println("  no runs")
        return nothing
    end
    n = length(evaluated)
    ce_t0 = filter(ce_plan_at_t0, evaluated)

    @printf("  %-6s %-5s %-5s %-20s %-10s %-22s %s\n", "Cycle", "t", "H", "Proposed certified", "Attempts",
        "|Δ logit ω| (cert.)", "CE OCP solved")
    for (i, t_i) in enumerate(first(evaluated).t_cycles)
        cycles = [r.proposed.cycles[i] for r in evaluated]
        certified = filter(c -> c.certified, cycles)
        distance = [abs(logit_omega(c.omega) - logit_omega(c.omega_start)) for c in certified]
        n_ce_solved = count(r -> r.certainty_equivalent.cycles[i].success, ce_t0)
        @printf("  %-6d %-5d %-5d %-20s %-10.2f %-22s %s\n", i, t_i, t_h - t_i,
            count_text(length(certified), n), mean(c.attempts for c in cycles), median_iqr(distance),
            count_text(n_ce_solved, length(ce_t0)))
    end
    println("\n  Attempts: mean. |Δ logit ω|: median [IQR] of |log λ_final − log λ_start|.")
    println("  CE OCP solved: among the runs with a CE plan at t₀.")
    return nothing
end

# -----------------------------------------------------------------------------
# 3. Value of replanning
# -----------------------------------------------------------------------------
"""
    print_replanning_value(runs)

Compare the proposed planner under receding-horizon planning with its first plan
applied open-loop, paired by seed over the runs certified at t₀.

# Arguments
- `runs::Vector`: successful results, sorted by seed.
"""
function print_replanning_value(runs)
    evaluated = evaluated_runs(runs)
    n = length(evaluated)
    ratios = [r.proposed.cost / r.first_plan.cost for r in evaluated]
    yield_gain = [r.proposed.yield - r.first_plan.yield for r in evaluated]
    repaired = count(r -> !r.first_plan.feasible && r.proposed.feasible, evaluated)
    caused = count(r -> r.first_plan.feasible && !r.proposed.feasible, evaluated)

    section("3. Value of replanning (Proposed vs. its open-loop first plan, paired by seed)")
    @printf("  %-36s %s, cheaper in %s\n", "J_receding / J_open-loop", median_iqr(ratios; digits=3),
        count_text(count(<(1), ratios), n))
    @printf("  %-36s %s\n", "Yield receding − open loop [kg m⁻²]", median_iqr(yield_gain; digits=3))
    @printf("  %-36s repaired %d, caused %d\n", "Violations by replanning", repaired, caused)
    println("\n  Median [IQR].")
    return nothing
end

# -----------------------------------------------------------------------------
# 4. Computation
# -----------------------------------------------------------------------------
"""
    computation_entries(runs)

Mean runtimes (in minutes) and attempts per cycle of the proposed planner, for
the first cycle (at t₀) and pooled over the later, warm-started cycles, on the
runs certified at t₀. The split follows `analyze_open_loop.jl`.

# Arguments
- `runs::Vector`: successful results, sorted by seed.

# Returns
- `NamedTuple` with fields `first` and `later`, each with the number of cycles
  `n` and the means `sampling`, `optimization`, `total`, `attempts`.

# Notes
- Sampling: PMCMC warm-up (first cycle) or posterior update (later cycles), plus
  the validation draws of all attempts, which continue the chains.
- Optimization: building the scenario OCP and solving it in all attempts.
- Total: wall-clock time of the cycle; it differs from the sum of both steps only
  by bookkeeping (calibration updates, printing).
"""
function computation_entries(runs)
    evaluated = evaluated_runs(runs)

    function summarize(cycles)
        return (
            n=length(cycles),
            sampling=mean(c.runtime_sampling + c.runtime_validation for c in cycles) / 60,
            optimization=mean(c.runtime_build + c.runtime_optimization for c in cycles) / 60,
            total=mean(c.runtime_sampling + c.runtime_build + c.runtime_plan for c in cycles) / 60,
            attempts=mean(Float64(c.attempts) for c in cycles),
        )
    end

    first_cycles = [r.proposed.cycles[1] for r in evaluated]
    later_cycles = reduce(vcat, [r.proposed.cycles[2:end] for r in evaluated]; init=NamedTuple[])
    return (first=summarize(first_cycles), later=summarize(later_cycles))
end

"""
    print_computation(runs)

Print the computation per cycle of [`computation_entries`](@ref), its relative
reduction from the first to the later cycles, and the hardware.

# Arguments
- `runs::Vector`: successful results, sorted by seed.

# Returns
- `NamedTuple`: output of [`computation_entries`](@ref), or `nothing` if no run
  is certified at t₀.
"""
function print_computation(runs)
    section("4. Computation (mean per cycle, runs certified at t₀)")
    if isempty(evaluated_runs(runs))
        println("  no runs")
        return nothing
    end
    e = computation_entries(runs)

    rows = [
        ("Sampling", :sampling, " min"),
        ("Optimization", :optimization, " min"),
        ("Total", :total, " min"),
        ("Attempts", :attempts, ""),
    ]
    @printf("  %-16s %-20s %-20s %s\n", "", "First cycle (t₀)", "Later cycles", "Reduction")
    @printf("  %-16s %-20d %-20d\n", "Cycles", e.first.n, e.later.n)
    for (label, field, unit) in rows
        before = getproperty(e.first, field)
        after = getproperty(e.later, field)
        @printf("  %-16s %-20s %-20s %.1f %%\n", label, @sprintf("%.2f%s", before, unit),
            @sprintf("%.2f%s", after, unit), reduction(before, after))
    end
    println("\n  Sampling:     PMCMC warm-up (first) or posterior update (later), plus the")
    println("                validation draws of all attempts.")
    println("  Optimization: OCP build and solves of all attempts.")

    hardware = unique([(get(r, :cpu_model, "unknown"), r.nthreads) for r in runs])
    for (cpu, nthreads) in hardware
        @printf("\n  Hardware: %s, %d Julia threads", cpu, nthreads)
    end
    println()
    return e
end

# -----------------------------------------------------------------------------
# 5. Representative run
# -----------------------------------------------------------------------------
# Pad a prediction over days t_i, ..., t_h with `NaN` to the window t₀, ..., t_h.
pad_to_window(v::AbstractVector, t_i::Int) = vcat(fill(NaN, t_i - t0), v)

"""
    single_run_figure(run)

Representative run: the printout and figure of `receding_horizon.jl`
([`print_receding_horizon_run`](@ref), [`plot_receding_horizon_run`](@ref)). The
data are also written to `seed_<s>_receding_horizon.txt` for pgfplots.

# Arguments
- `run::NamedTuple`: successful result of one seed.
"""
function single_run_figure(run)
    s = run.seed
    section("5. Representative run: Seed $s")
    print_receding_horizon_run(run)
    if closed_loop_status(run) === :uncertified_at_t0
        return nothing
    end
    finalize_plot(plot_receding_horizon_run(run), "seed_$(s)_receding_horizon")

    t = collect(0:H)
    day = t0 .+ t
    target = fill(minimum_yield / HI, H + 1)  # terminal biomass that meets the yield target
    first_cycle = run.proposed.cycles[1]
    last_cycle = run.proposed.cycles[end]
    proposed_realized = run.proposed.X_true[1, :]
    open_loop_realized = run.first_plan.X_true[1, :]
    ce_realized = run.certainty_equivalent.X_true[1, :]
    oracle_realized = run.oracle.X_true[1, :]
    lower_t0 = first_cycle.pred.q05[1, :]
    upper_t0 = first_cycle.pred.q95[1, :]
    lower_last = pad_to_window(last_cycle.pred.q05[1, :], last_cycle.t)
    upper_last = pad_to_window(last_cycle.pred.q95[1, :], last_cycle.t)

    if export_txt
        mkpath(PLOT_DIR)
        file = joinpath(PLOT_DIR, "seed_$(s)_receding_horizon.txt")
        data2txt(file,
            "t", t,
            "day", day,
            "proposed", proposed_realized,
            "open_loop", open_loop_realized,
            "ce", ce_realized,
            "oracle", oracle_realized,
            "lower_t0", lower_t0,
            "upper_t0", upper_t0,
            "lower_last", lower_last,
            "upper_last", upper_last,
            "target", target)
        println("  Wrote $file")
        export_cycle_frames(run)
    end
    return nothing
end

"""
    ce_prediction(cycle)

Deterministic prediction of a certainty-equivalent cycle: its plan simulated from
the filtered state at t_i under the posterior-mean θ, without process noise.

# Arguments
- `cycle::NamedTuple`: element of `certainty_equivalent.cycles`.

# Returns
- `Matrix{Float64}`: state trajectory of size `(n_x, H_i + 1)` over days t_i, ..., t_h.

# Notes
- Reconstructed here, since the cycles do not store it. It is the state
  trajectory of the CE OCP (`solve_deterministic_ocp` with zero process noise),
  up to the solver tolerance on the dynamics constraints.
"""
function ce_prediction(cycle)
    X = Matrix{Float64}(undef, n_x, cycle.H + 1)
    v_zero = zeros(n_v)
    X[:, 1] = cycle.x_hat
    for t in 2:(cycle.H+1)
        X[:, t] .= model.f(X[:, t-1], cycle.U[:, t-1], v_zero, cycle.theta_hat)
    end
    return X
end

# Realized trajectory until day `t_i` (column t_i − t₀ + 1), `NaN` afterwards.
function realized_until(x::AbstractVector, t_i::Int)
    n_realized = t_i - t0 + 1
    x_so_far = fill(NaN, H + 1)
    x_so_far[1:n_realized] = x[1:n_realized]
    return x_so_far
end

"""
    export_cycle_frames(run)

Per-cycle data of the proposed and the certainty-equivalent (CE) planner for an
animation of the shrinking uncertainty, one frame per planning cycle. Writes, to
`<PLOT_DIR>/seed_<s>_frames/`,

- `cycle_<i>.txt` on the full window t₀, ..., t_h: per method the realized
  trajectory until the cycle start t_i (`NaN` afterwards); from t_i on (`NaN`
  before) the mean prediction and the 90 % prediction interval of the proposed
  planner and the deterministic prediction of the CE planner (`NaN` if its OCP is
  infeasible in this cycle); and the terminal-biomass target;
- `cycle_<N+1>.txt` at harvest t_h: the full realized trajectories, no prediction;
- `cycles.txt`: one row per frame with its index, start, certification, ω, and δ
  of the proposed planner and whether the CE OCP is solved (`NaN` for the harvest
  frame), e.g. for frame titles.

# Arguments
- `run::NamedTuple`: successful result of one seed.

# Notes
- The proposed prediction is that of the applied plan over the validation
  scenarios of the cycle's last attempt (see `plan_certified_cycle` in
  `receding_horizon.jl`); the CE prediction is reconstructed by
  [`ce_prediction`](@ref).
- Both closed loops replan on the same days `run.t_cycles`.
- Use `unbounded coords=jump` in pgfplots so that the `NaN` entries are skipped.
"""
function export_cycle_frames(run)
    s = run.seed
    dir = joinpath(PLOT_DIR, "seed_$(s)_frames")
    mkpath(dir)

    t = collect(0:H)
    day = t0 .+ t
    target = fill(minimum_yield / HI, H + 1)  # terminal biomass that meets the yield target
    no_prediction = fill(NaN, H + 1)
    realized = run.proposed.X_true[1, :]
    ce_realized = run.certainty_equivalent.X_true[1, :]
    cycles = run.proposed.cycles
    ce_cycles = run.certainty_equivalent.cycles

    for (i, cycle) in enumerate(cycles)
        ce_cycle = ce_cycles[i]
        if ce_cycle.success
            ce_mean = pad_to_window(ce_prediction(ce_cycle)[1, :], ce_cycle.t)
        else
            ce_mean = no_prediction
        end

        data2txt(joinpath(dir, "cycle_$(i).txt"),
            "t", t,
            "day", day,
            "realized", realized_until(realized, cycle.t),
            "mean", pad_to_window(cycle.pred.mean[1, :], cycle.t),
            "lower", pad_to_window(cycle.pred.q05[1, :], cycle.t),
            "upper", pad_to_window(cycle.pred.q95[1, :], cycle.t),
            "ce_realized", realized_until(ce_realized, ce_cycle.t),
            "ce_mean", ce_mean,
            "target", target)
    end

    # Final frame at harvest: the full realized trajectories, no prediction
    n_frames = length(cycles) + 1
    data2txt(joinpath(dir, "cycle_$(n_frames).txt"),
        "t", t,
        "day", day,
        "realized", realized,
        "mean", no_prediction,
        "lower", no_prediction,
        "upper", no_prediction,
        "ce_realized", ce_realized,
        "ce_mean", no_prediction,
        "target", target)

    # One row per frame; the harvest frame has no certification, ω, δ, or CE solve (NaN)
    data2txt(joinpath(dir, "cycles.txt"),
        "cycle", collect(1:n_frames),
        "t_start", vcat([c.t - t0 for c in cycles], H),
        "day_start", vcat([c.t for c in cycles], t_h),
        "certified", vcat([Float64(c.certified) for c in cycles], NaN),
        "omega", vcat([c.omega for c in cycles], NaN),
        "delta", vcat([c.delta for c in cycles], NaN),
        "ce_success", vcat([Float64(c.success) for c in ce_cycles], NaN))
    println("  Wrote $n_frames frames ($(length(cycles)) cycles + harvest) to $dir")
    return nothing
end

"""
    export_results(table, computation)

Write the key numbers of the method comparison and the computation to
`<PLOT_DIR>/receding_horizon_results.txt`, one `key value` pair per line.

# Arguments
- `table::NamedTuple`: output of [`table_entries`](@ref).
- `computation::NamedTuple`: output of [`computation_entries`](@ref).
"""
function export_results(table, computation)
    mkpath(PLOT_DIR)
    file = joinpath(PLOT_DIR, "receding_horizon_results.txt")
    # Median of the quartiles, or NaN if there are no values.
    function med(q)
        if q === nothing
            return NaN
        end
        return q[1]
    end
    first_cycle = computation.first
    later = computation.later
    entries = [
        "n_evaluated" => table.n_evaluated,
        "n_ce_plan_t0" => table.n_ce_t0,
        "n_cost_common" => table.n_common,
        "proposed_plan_all_cycles" => table.proposed_all_cycles,
        "proposed_violations" => table.proposed_violations,
        "first_plan_violations" => table.first_plan_violations,
        "ce_plan_all_cycles" => table.ce_all_cycles,
        "ce_violations" => table.ce_violations,
        "ratio_proposed_median" => round(med(table.ratio[:proposed]); digits=3),
        "ratio_first_plan_median" => round(med(table.ratio[:first_plan]); digits=3),
        "ratio_ce_median" => round(med(table.ratio[:certainty_equivalent]); digits=3),
        "time_first_cycle_min" => round(first_cycle.total; digits=2),
        "time_first_cycle_sampling_min" => round(first_cycle.sampling; digits=2),
        "time_first_cycle_optimization_min" => round(first_cycle.optimization; digits=2),
        "time_later_cycles_min" => round(later.total; digits=2),
        "time_later_cycles_sampling_min" => round(later.sampling; digits=2),
        "time_later_cycles_optimization_min" => round(later.optimization; digits=2),
        "time_reduction_percent" => round(reduction(first_cycle.total, later.total); digits=1),
        "attempts_first_cycle_mean" => round(first_cycle.attempts; digits=2),
        "attempts_later_cycles_mean" => round(later.attempts; digits=2),
    ]
    open(file, "w") do io
        for (key, value) in entries
            println(io, key, " ", value)
        end
    end
    println("\n  Wrote $file")
    return nothing
end

# -----------------------------------------------------------------------------
# Run
# -----------------------------------------------------------------------------
isdir(RESULTS_DIR) || error("RESULTS_DIR does not exist: $RESULTS_DIR")

results = load_results(joinpath(RESULTS_DIR, "receding"))
runs = sort([r for r in values(results) if r.status == :ok]; by=r -> r.seed)
failed = sort([r for r in values(results) if r.status != :ok]; by=r -> r.seed)

println("="^80)
println("Receding-horizon Monte-Carlo study: $RESULTS_DIR")
@printf("  %d runs loaded, %d failed\n", length(results), length(failed))
for r in failed
    @printf("    seed %d: %s\n", r.seed, first(split(string(r.error), '\n')))
end
if !isempty(runs)
    @printf("  L = %d, %d cycles at t = %s\n", first(runs).L, length(first(runs).t_cycles),
        join(first(runs).t_cycles, ", "))
end
println("="^80)

table = print_table(runs)
print_cycles(runs)
print_replanning_value(runs)
computation = print_computation(runs)

if export_txt && computation !== nothing
    export_results(table, computation)
end

for s in seeds_to_plot
    run = get(results, s, nothing)
    if run === nothing || run.status != :ok
        @warn "No successful result for seed $s; skipping its figure."
        continue
    end
    single_run_figure(run)
end
