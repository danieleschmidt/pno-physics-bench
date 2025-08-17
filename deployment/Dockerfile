# Production Dockerfile for PNO Physics Bench
FROM nvidia/cuda:12.1-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6+PTX"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    git \
    curl \
    wget \
    build-essential \
    cmake \
    libnuma1 \
    libnuma-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
COPY pyproject.toml .

# Install Python dependencies
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel
RUN pip3 install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
RUN pip3 install --no-cache-dir -r requirements.txt
RUN pip3 install --no-cache-dir redis lz4 cryptography

# Copy source code
COPY src/ src/
COPY examples/ examples/
COPY tests/ tests/

# Install package in production mode
RUN pip3 install --no-cache-dir -e .

# Create non-root user for security
RUN useradd -m -u 1000 pnouser && \
    chown -R pnouser:pnouser /app
USER pnouser

# Expose application port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python3 -c "import pno_physics_bench; print('OK')" || exit 1

# Default command
CMD ["python3", "-m", "pno_physics_bench.cli", "serve", "--host", "0.0.0.0", "--port", "8080"]
