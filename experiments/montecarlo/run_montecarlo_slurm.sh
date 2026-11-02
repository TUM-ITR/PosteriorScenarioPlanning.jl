#!/bin/bash
#SBATCH --job-name=mc_study
#SBATCH --output=experiments/montecarlo/logs/out_%A_%a.out
#SBATCH --error=experiments/montecarlo/logs/err_%A_%a.err
#SBATCH --array=1-200
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

# Monte-Carlo study (R = 200 runs, one array task per seed) for the open-loop
# and receding-horizon (closed-loop) experiments.
#
# Usage (from the repository root):
#   sbatch experiments/montecarlo/run_montecarlo_slurm.sh <posterior|prior|receding> [RUN_NAME]
#
#   posterior  open loop: proposed method + certainty-equivalent baseline + hindsight oracle
#   prior      open loop: prior-based baseline
#   receding   closed loop: receding-horizon experiment
#
# Results go to experiments/montecarlo/results/<RUN_NAME>/<method>/seed_<seed>.jld2.
# RUN_NAME defaults to "open_loop" for posterior/prior and "receding_horizon" for receding.
# analyze_open_loop.jl pairs the posterior and prior runs within a RUN_NAME by seed;
# analyze_receding_horizon.jl evaluates the receding runs.

set -euo pipefail

# . /etc/profile.d/modules.sh
# module load apptainer

METHOD=${1:?"Usage: sbatch run_montecarlo_slurm.sh <posterior|prior|receding> [RUN_NAME]"}
case "${METHOD}" in
    posterior | prior) DEFAULT_RUN_NAME=open_loop ;;
    receding) DEFAULT_RUN_NAME=receding_horizon ;;
    *)
        echo "Unknown method '${METHOD}'; expected 'posterior', 'prior', or 'receding'." >&2
        exit 1
        ;;
esac
RUN_NAME=${2:-${DEFAULT_RUN_NAME}}

SEED=${SLURM_ARRAY_TASK_ID}
HOST_PROJECT_DIR="/path/to/PosteriorScenarioPlanning.jl"
CONTAINER_PROJECT_DIR="/home/developer/workspace"
CONTAINER_IMAGE="$HOME/.apptainer/containers/julia_container.sif"
JULIA_DEPOT_HOST="$HOME/.julia-hpc"

HOST_RESULTS_DIR="${HOST_PROJECT_DIR}/experiments/montecarlo/results/${RUN_NAME}"
CONTAINER_RESULTS_DIR="${CONTAINER_PROJECT_DIR}/experiments/montecarlo/results/${RUN_NAME}"

mkdir -p "$HOST_RESULTS_DIR" "$JULIA_DEPOT_HOST"

echo "Method: ${METHOD} | seed: ${SEED} | run: ${RUN_NAME} | job: ${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}"

apptainer exec \
    --cleanenv \
    --bind "${HOST_PROJECT_DIR}:${CONTAINER_PROJECT_DIR}" \
    --bind "${JULIA_DEPOT_HOST}:/depot" \
    --env JULIA_DEPOT_PATH="/depot:/home/developer/.julia" \
    --env JULIA_NUM_THREADS="${SLURM_CPUS_PER_TASK}" \
    --env JULIA_PKG_PRECOMPILE_AUTO=0 \
    --env OPENBLAS_NUM_THREADS=1 \
    --env OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}" \
    "${CONTAINER_IMAGE}" \
    julia --heap-size-hint=24G \
        --project="${CONTAINER_PROJECT_DIR}/experiments" \
        "${CONTAINER_PROJECT_DIR}/experiments/montecarlo/run_montecarlo_single.jl" \
        --method "${METHOD}" \
        --seed "${SEED}" \
        --results-dir "${CONTAINER_RESULTS_DIR}"
