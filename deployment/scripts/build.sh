#!/bin/bash
set -e

echo "Building PNO Physics Bench for production..."

# Build Docker image
docker build -t pno-physics-bench:latest .

# Tag for registry
docker tag pno-physics-bench:latest registry.example.com/pno-physics-bench:latest

# Run tests in container
docker run --rm pno-physics-bench:latest python test_basic_functionality.py
docker run --rm pno-physics-bench:latest python test_robust_functionality.py
docker run --rm pno-physics-bench:latest python test_scaling_functionality.py

echo "Build completed successfully!"
