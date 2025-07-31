# Multi-stage Dockerfile for PNO Physics Bench
# Optimized for ML workloads with PyTorch and CUDA support

ARG PYTHON_VERSION=3.9
ARG PYTORCH_VERSION=2.1.0
ARG CUDA_VERSION=11.8

# Base stage - Common dependencies
FROM python:${PYTHON_VERSION}-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TORCH_HOME=/opt/torch \
    PYTHONPATH=/app/src

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r pno && useradd -r -g pno -d /app -s /bin/bash pno
RUN mkdir -p /app /opt/torch && chown -R pno:pno /app /opt/torch

# Development stage - Full development environment
FROM base as development

# Install additional development tools
RUN apt-get update && apt-get install -y \
    vim \
    htop \
    tmux \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency files
COPY --chown=pno:pno requirements.txt pyproject.toml ./

# Install PyTorch CPU version (development default)
RUN pip install --upgrade pip setuptools wheel
RUN pip install torch==${PYTORCH_VERSION} torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install development dependencies
RUN pip install -e ".[dev,benchmark,all]"

# Copy source code
COPY --chown=pno:pno . .

USER pno

# Default command for development
CMD ["bash"]

# Production stage - Optimized for deployment
FROM base as production

WORKDIR /app

# Copy only necessary files
COPY --chown=pno:pno requirements.txt pyproject.toml ./
COPY --chown=pno:pno src/ ./src/
COPY --chown=pno:pno README.md LICENSE ./

# Install production dependencies only
RUN pip install --upgrade pip setuptools wheel
RUN pip install torch==${PYTORCH_VERSION} torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN pip install -e . --no-dev

# Run as non-root user
USER pno

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import pno_physics_bench; print('OK')" || exit 1

# Default command
CMD ["pno-train", "--help"]

# CUDA-enabled stage - For GPU training
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu20.04 as cuda

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    TORCH_HOME=/opt/torch \
    PYTHONPATH=/app/src

# Install Python and system dependencies
RUN apt-get update && apt-get install -y \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-dev \
    python3-pip \
    build-essential \
    git \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic links for python
RUN ln -s /usr/bin/python${PYTHON_VERSION} /usr/bin/python
RUN ln -s /usr/bin/python${PYTHON_VERSION} /usr/bin/python3

# Create non-root user
RUN groupadd -r pno && useradd -r -g pno -d /app -s /bin/bash pno
RUN mkdir -p /app /opt/torch && chown -R pno:pno /app /opt/torch

WORKDIR /app

# Copy dependency files
COPY --chown=pno:pno requirements.txt pyproject.toml ./

# Install PyTorch with CUDA support
RUN pip install --upgrade pip setuptools wheel
RUN pip install torch==${PYTORCH_VERSION} torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install application
COPY --chown=pno:pno src/ ./src/
COPY --chown=pno:pno README.md LICENSE ./
RUN pip install -e .

USER pno

# Verify CUDA installation
RUN python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA devices: {torch.cuda.device_count()}')"

CMD ["pno-train", "--help"]