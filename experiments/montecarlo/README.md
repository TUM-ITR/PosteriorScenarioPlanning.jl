# Monte Carlo Simulation

This directory contains the code for the Monte Carlo simulation study reported in the paper.

## Monte Carlo Study on HPC (Docker + Apptainer + Slurm)

For the full Monte Carlo study consisting of 200 independent runs per method, we recommend using the provided containerized workflow. This setup ensures reproducibility and reliable execution on HPC clusters.

### 1. Build the Docker image locally

From the **repository root** (not from `.devcontainer/`), build the Docker image:

```bash
docker build \
  -f .devcontainer/Dockerfile \
  -t julia_container:latest \
  .
```

The trailing `.` sets the build context to the repository root; the `COPY` instructions in the Dockerfile are relative to it.

#### Note on linear solvers

The container is configured to use the proprietary HSL linear solvers for improved numerical performance.
To include them, place `HSL_jll.jl.v2025.7.21.tar.gz` into the `.devcontainer/` directory next to the Dockerfile before building the image.

If HSL is not available, remove the corresponding lines from the Dockerfile (the `COPY`/`tar` block and the `Pkg.develop` call for `HSL_PATH`). The code will still run using open-source solvers (e.g., MUMPS), but execution may be slower and numerical results may differ slightly.

The results reported in the paper were produced with the HSL `MA97` solver (used through Ipopt).

### 2. Save and transfer the Docker image to the cluster

Create a tar archive of the image:

```bash
docker save julia_container:latest -o julia_container.tar
```

Transfer it to the cluster:

```bash
scp julia_container.tar <cluster_user>@<cluster_address>:~
```

### 3. Build the Apptainer image on the cluster

On the cluster, load Apptainer and build the `.sif` image into a dedicated container directory:

```bash
. /etc/profile.d/modules.sh
module load apptainer

mkdir -p ~/.apptainer/containers

apptainer build ~/.apptainer/containers/julia_container.sif \
    docker-archive://$HOME/julia_container.tar
```

Once the `.sif` exists, the tar archive can be deleted.

### 4. Configure the host paths in the Slurm script

The Slurm script uses bind mounts to map host directories into the Apptainer container.
Since the container filesystem is read-only, Julia is given a **writable depot on the host**, which is layered in front of the depot baked into the image.

Open `experiments/montecarlo/run_montecarlo_slurm.sh` and set

```bash
HOST_PROJECT_DIR="/your/absolute/path/to/repository/root"
CONTAINER_IMAGE="$HOME/.apptainer/containers/julia_container.sif"
JULIA_DEPOT_HOST="$HOME/.julia-hpc"
```

All bind mounts and output paths are derived from these variables automatically. The relevant runtime configuration is:

| Host path                 | Container path                | Purpose                              |
| ------------------------- | ----------------------------- | ------------------------------------ |
| `$HOST_PROJECT_DIR`       | `/home/developer/workspace`   | Live source checkout and results     |
| `$JULIA_DEPOT_HOST`       | `/depot`                      | Writable Julia depot (persistent)    |
| *(baked into image)*      | `/home/developer/.julia`      | Read-only precompiled depot          |

with `JULIA_DEPOT_PATH=/depot:/home/developer/.julia`, so that packages precompiled at image build time are reused, and anything new is written to the host depot.

Create the required host directories (only needed once):

```bash
mkdir -p ~/.julia-hpc
mkdir -p /your/absolute/path/to/repository/root/experiments/montecarlo/logs
```

The log directory must exist **before** submission: Slurm opens the log files before the job script runs, so a missing directory causes the job to fail without producing any output. Results directories are created automatically by the script for each job.

### 5. Warm the Julia depot (once)

Before submitting the job array for the first time, precompile the experiment environment once on a compute node. This populates the writable depot and prevents the array tasks from precompiling concurrently:

```bash
srun --pty -t 00:30:00 -c 8 --mem=32G \
apptainer exec \
    --cleanenv \
    --bind /your/absolute/path/to/repository/root:/home/developer/workspace \
    --bind $HOME/.julia-hpc:/depot \
    --env JULIA_DEPOT_PATH=/depot:/home/developer/.julia \
    $HOME/.apptainer/containers/julia_container.sif \
    julia --project=/home/developer/workspace/experiments \
          -e 'using Pkg; if isdir(get(ENV, "HSL_PATH", "")); Pkg.develop(path=ENV["HSL_PATH"]); Pkg.add("MadNLPHSL"); end; Pkg.instantiate(); Pkg.precompile()'
```

This step only needs to be performed once per depot. The HSL packages are not part of the committed experiment environment; if the image contains HSL, the command above adds `HSL_jll` and `MadNLPHSL` to the bind-mounted `experiments/Project.toml` and `Manifest.toml`.

### 6. Run the Monte Carlo simulation via Slurm

From the repository root on the cluster, submit one job array per method:

```bash
sbatch experiments/montecarlo/run_montecarlo_slurm.sh posterior # proposed + certainty equivalent + oracle
sbatch experiments/montecarlo/run_montecarlo_slurm.sh prior # prior-based baseline
```

An optional second argument sets the run name (default `open_loop`; `receding_horizon` for the receding-horizon experiment, see below). The prior-based baseline is only needed to reproduce the comparison.
Output is written to:

- `experiments/montecarlo/logs/`
- `experiments/montecarlo/results/<run name>/posterior/` and `.../prior/`

Progress can be monitored with `squeue --me`; a job array can be cancelled with `scancel <jobid>`.

#### Node selection and reported runtimes

Runtimes are only comparable across runs if all jobs execute on identical hardware. The script therefore restricts execution to a homogeneous set of nodes via `#SBATCH --exclude=...`. Adjust this list to your cluster.

The runtimes reported in the paper were obtained on nodes with 2× AMD EPYC 7402 (Zen 2, 24 cores per socket) and 512 GB RAM, using 32 CPUs per task.

#### Threading

Each task uses `JULIA_NUM_THREADS` equal to `--cpus-per-task` for the sampling stage, and pins BLAS to a single thread (`OPENBLAS_NUM_THREADS=1`) to avoid oversubscription from nested parallelism.

## Analyzing Results

Once all simulations have completed, the results are analyzed with one script per experiment. Each script prints the summary statistics reported in the paper and exports the figures; see the script itself for details.

### Open-loop experiment

Set `RESULTS_DIR` in `experiments/montecarlo/analyze_open_loop.jl` to the results directory of the run, for example:

```julia
RESULTS_DIR = joinpath(@__DIR__, "results", "open_loop")
```

and run

```bash
julia --project=experiments experiments/montecarlo/analyze_open_loop.jl
```

### Receding-horizon experiment

Submit the job array with

```bash
sbatch experiments/montecarlo/run_montecarlo_slurm.sh receding # results/receding_horizon/receding/
```

then set `RESULTS_DIR` in `experiments/montecarlo/analyze_receding_horizon.jl` (default `results/receding_horizon`) and run

```bash
julia --project=experiments experiments/montecarlo/analyze_receding_horizon.jl
```
