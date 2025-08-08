FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc g++ \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ src/
COPY deployment/ deployment/

# Set environment variables
ENV PYTHONPATH=/app/src
ENV CUDA_VISIBLE_DEVICES=""

# Expose ports
EXPOSE 8000 8001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s \
  CMD python -c "import torch; import src.pno_physics_bench; print('OK')"

# Run application
CMD ["python", "-m", "pno_physics_bench.server"]
