FROM ubuntu:24.04

ARG DEBIAN_FRONTEND=noninteractive

# Update package lists and install necessary dependencies.
RUN apt-get update && apt-get install -y \
    gfortran \
    build-essential \
    wget \
    curl \
    libblas3 \
    libblas-dev \
    liblapack3 \
    liblapack-dev \
    libmetis-dev \
    git \
    openssh-server \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /run/sshd

# Install Julia.
ARG JULIA_VERSION=1.12.3
RUN JULIA_MAJOR=${JULIA_VERSION%.*} \
    && wget -q https://julialang-s3.julialang.org/bin/linux/x64/${JULIA_MAJOR}/julia-${JULIA_VERSION}-linux-x86_64.tar.gz \
    && tar -xzf julia-${JULIA_VERSION}-linux-x86_64.tar.gz -C /opt \
    && ln -s /opt/julia-${JULIA_VERSION}/bin/julia /usr/local/bin/julia \
    && rm julia-${JULIA_VERSION}-linux-x86_64.tar.gz

# Add user.
ARG USER_NAME=developer
ARG USER_ID=1001
ARG USER_HOME="/home/$USER_NAME"
RUN adduser \
    --home "$USER_HOME" \
    --uid $USER_ID \
    --disabled-password \
    "$USER_NAME"

# Set the default user.
USER ${USER_NAME}

# Set the working directory to the user's home directory.
WORKDIR ${USER_HOME}

# Create a folder to store the claude credentials (optional).
RUN mkdir -p ${USER_HOME}/.claude

# Create environment variables.
ENV WORKSPACE="${USER_HOME}/workspace"
ENV EXPERIMENT_ENV="${WORKSPACE}/experiments"
ENV HSL_PATH="${USER_HOME}/HSL"
ENV JULIA_CPU_TARGET="generic;native"

# --- Optional: HSL linear solvers for Ipopt / MadNLP. -----------------------
# Expects .devcontainer/HSL_jll.jl.v2025.7.21.tar.gz in the build context.
# Comment out this block (and the Pkg.develop call further down) if unavailable;
# open-source solvers will be used instead.
COPY --chown=${USER_NAME}:${USER_NAME} .devcontainer/HSL_jll.jl.v2025.7.21.tar.gz ${USER_HOME}/
RUN mkdir -p ${HSL_PATH} \
    && tar -xzf HSL_jll.jl.v2025.7.21.tar.gz -C ${HSL_PATH} --strip-components=1 \
    && rm HSL_jll.jl.v2025.7.21.tar.gz
# ----------------------------------------------------------------------------

# Copy manifests + package source for dependency caching.
COPY --chown=${USER_NAME}:${USER_NAME} Project.toml Manifest.toml ${WORKSPACE}/
COPY --chown=${USER_NAME}:${USER_NAME} src ${WORKSPACE}/src
COPY --chown=${USER_NAME}:${USER_NAME} experiments/Project.toml experiments/Manifest.toml ${EXPERIMENT_ENV}/

# Instantiate the main package environment.
RUN julia --project=${WORKSPACE} -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'

# Instantiate the experiment environment.
RUN julia --project=${EXPERIMENT_ENV} -e ' \
    using Pkg; \
    Pkg.develop(path=ENV["WORKSPACE"]); \
    if haskey(ENV, "HSL_PATH") && isdir(ENV["HSL_PATH"]); \
        Pkg.develop(path=ENV["HSL_PATH"]); \
        Pkg.add("MadNLPHSL"); \
    end; \
    Pkg.instantiate(); \
    Pkg.precompile()'