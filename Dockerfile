# =============================================================================
# rustims – full-stack Docker image (CUDA 12.4, Python 3.12)
#
# Multi-stage build:
#   builder  – compiles imspy_connector wheel with Rust + maturin
#   runtime  – lean image with all Python packages and pre-cached models
#
# Build:   docker build -t rustims .
# Run:     docker run --rm --gpus all rustims python -c "import torch; print(torch.cuda.is_available())"
# =============================================================================

# ---------------------------------------------------------------------------
# Stage 1: builder – compile the Rust/PyO3 wheel
# ---------------------------------------------------------------------------
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y --no-install-recommends \
        python3.12 \
        python3.12-dev \
        python3.12-venv \
        curl \
        build-essential \
        pkg-config \
        libssl-dev && \
    rm -rf /var/lib/apt/lists/*

# Rust toolchain (stable)
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable
ENV PATH="/root/.cargo/bin:${PATH}"

# Create venv and install maturin
RUN python3.12 -m venv /opt/builder-venv
ENV PATH="/opt/builder-venv/bin:${PATH}"
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir "maturin>=1.2,<2.0"

# Copy the full repo (need workspace Cargo.toml + crate sources)
COPY Cargo.toml /src/
COPY mscore /src/mscore
COPY rustdf /src/rustdf
COPY rustms /src/rustms
COPY imspy_connector /src/imspy_connector
# imsjl_connector is referenced in workspace but we still need a stub
COPY imsjl_connector /src/imsjl_connector

WORKDIR /src/imspy_connector
RUN maturin build --release --interpreter python3.12 && \
    cp /src/target/wheels/*.whl /tmp/

# ---------------------------------------------------------------------------
# Stage 2: runtime – Python packages + pre-cached models
# ---------------------------------------------------------------------------
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y --no-install-recommends \
        python3.12 \
        python3.12-venv \
        libgomp1 \
        ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# Create application venv
RUN python3.12 -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"
RUN pip install --no-cache-dir --upgrade pip

# Install the connector wheel built in stage 1
COPY --from=builder /tmp/*.whl /tmp/wheels/
RUN pip install --no-cache-dir /tmp/wheels/*.whl && \
    rm -rf /tmp/wheels

# Copy Python packages
COPY packages /tmp/packages

# Install packages in dependency order
RUN pip install --no-cache-dir /tmp/packages/imspy-core && \
    pip install --no-cache-dir "/tmp/packages/imspy-predictors[koina]" && \
    pip install --no-cache-dir /tmp/packages/imspy-dia && \
    pip install --no-cache-dir /tmp/packages/imspy-search && \
    pip install --no-cache-dir "/tmp/packages/imspy-simulation[search,gui]" && \
    pip install --no-cache-dir /tmp/packages/imspy-vis && \
    rm -rf /tmp/packages

# Pre-cache pretrained models from the repo tree so users don't download on
# first run.  The hub module (ensure_model) looks in ~/.cache/imspy/models/v<VER>/.
COPY packages/imspy-predictors/src/imspy_predictors/pretrained/ccs/best_model.pt \
     /root/.cache/imspy/models/v0.5.0/ccs/best_model.pt
COPY packages/imspy-predictors/src/imspy_predictors/pretrained/rt/best_model.pt \
     /root/.cache/imspy/models/v0.5.0/rt/best_model.pt
COPY packages/imspy-predictors/src/imspy_predictors/pretrained/charge/best_model.pt \
     /root/.cache/imspy/models/v0.5.0/charge/best_model.pt
COPY packages/imspy-predictors/src/imspy_predictors/pretrained/intensity/best_model.pt \
     /root/.cache/imspy/models/v0.5.0/intensity/best_model.pt
COPY packages/imspy-predictors/src/imspy_predictors/pretrained/pretrained_encoder.pt \
     /root/.cache/imspy/models/v0.5.0/pretrained_encoder.pt

# Verify models are loadable by the hub module
RUN python -c "\
from imspy_predictors.pretrained.hub import ensure_model, MODELS; \
paths = [ensure_model(m) for m in MODELS]; \
print('Cached', len(paths), 'models:', [str(p) for p in paths])"

# Non-root user — move model cache into its home directory
RUN groupadd -g 1000 rustims && \
    useradd -m -u 1000 -g 1000 rustims && \
    mkdir -p /home/rustims/.cache && \
    cp -r /root/.cache/imspy /home/rustims/.cache/imspy && \
    chown -R rustims:rustims /home/rustims/.cache && \
    rm -rf /root/.cache/imspy
USER rustims

# NVIDIA runtime env
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

WORKDIR /workspace

# Ensure venv is on PATH for all commands (python, timsim, etc.)
ENV PATH="/opt/venv/bin:${PATH}"
CMD ["python", "--version"]
