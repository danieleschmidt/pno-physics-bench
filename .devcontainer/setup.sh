#!/bin/bash
set -e

echo "🚀 Setting up PNO Physics Bench development environment..."

# Update package lists
sudo apt-get update

# Install system dependencies
sudo apt-get install -y \
    build-essential \
    cmake \
    git \
    curl \
    wget \
    vim \
    htop \
    tree \
    jq \
    graphviz \
    ffmpeg \
    libsm6 \
    libxext6 \
    libfontconfig1 \
    libxrender1 \
    libgl1-mesa-glx

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install --upgrade pip setuptools wheel

# Install project in development mode
pip install -e ".[dev,jax,benchmark]"

# Install additional dev tools
pip install \
    jupyterlab \
    jupyterlab-git \
    jupyterlab-lsp \
    python-lsp-server[all] \
    notebook \
    ipywidgets \
    panel \
    bokeh

# Setup pre-commit hooks
echo "🔧 Setting up pre-commit hooks..."
pre-commit install
pre-commit install --hook-type commit-msg

# Setup Jupyter Lab extensions
echo "🔬 Configuring Jupyter Lab..."
jupyter lab --generate-config --allow-root || true

# Create useful aliases
echo "⚙️ Setting up shell aliases..."
cat >> ~/.bashrc << 'EOF'

# PNO Physics Bench aliases
alias pno-test='pytest tests/ -v --cov=src/pno_physics_bench'
alias pno-lint='flake8 src/ tests/ && mypy src/ && black --check src/ tests/'
alias pno-format='black src/ tests/ && isort src/ tests/'
alias pno-train='python -m pno_physics_bench.cli train'
alias pno-benchmark='python -m pno_physics_bench.cli benchmark'
alias pno-eval='python -m pno_physics_bench.cli evaluate'
alias pno-jupyter='jupyter lab --ip=0.0.0.0 --allow-root --no-browser'
alias pno-tensorboard='tensorboard --logdir=./logs --bind_all'

# Git aliases for efficient development
alias gs='git status'
alias ga='git add'
alias gc='git commit'
alias gp='git push'
alias gl='git log --oneline'
alias gd='git diff'

# Docker aliases
alias dc='docker-compose'
alias dcu='docker-compose up'
alias dcd='docker-compose down'
alias dcl='docker-compose logs'

EOF

# Setup git configuration if not exists
if [ ! -f ~/.gitconfig ]; then
    echo "🔧 Setting up git configuration..."
    git config --global user.name "PNO Developer"
    git config --global user.email "dev@pno-physics-bench.local"
    git config --global init.defaultBranch main
    git config --global pull.rebase false
fi

# Create useful directories
mkdir -p ~/notebooks
mkdir -p ~/experiments
mkdir -p ~/datasets

# Download sample datasets (small ones for development)
echo "📊 Setting up sample datasets..."
mkdir -p data/samples
cat > data/samples/README.md << 'EOF'
# Sample Datasets

This directory contains small sample datasets for development and testing.

## Available Datasets

- `burgers_1d_sample.h5`: Small Burgers equation dataset (100 samples)
- `darcy_2d_sample.h5`: Small Darcy flow dataset (50 samples)

## Usage

```python
from pno_physics_bench.datasets import load_sample_dataset

# Load development dataset
train_loader, val_loader, test_loader = load_sample_dataset('burgers_1d')
```

For full datasets, see the main documentation.
EOF

# Setup monitoring configuration
echo "📊 Setting up monitoring..."
mkdir -p logs/tensorboard
mkdir -p logs/wandb

# Create development configuration
cat > .env.dev << 'EOF'
# Development Environment Configuration
PYTHONPATH=/app/src
WANDB_MODE=offline
CUDA_VISIBLE_DEVICES=0
OMP_NUM_THREADS=4
MKL_NUM_THREADS=4

# Development flags
DEBUG=1
LOG_LEVEL=DEBUG
DEVELOPMENT_MODE=1

# Monitoring
TENSORBOARD_LOG_DIR=./logs/tensorboard
WANDB_PROJECT=pno-physics-bench-dev

# Testing
PYTEST_CURRENT_TEST=""
COVERAGE_REPORT=1
EOF

# Setup Jupyter configuration
mkdir -p ~/.jupyter
cat > ~/.jupyter/jupyter_lab_config.py << 'EOF'
c.ServerApp.ip = '0.0.0.0'
c.ServerApp.port = 8888
c.ServerApp.open_browser = False
c.ServerApp.allow_root = True
c.ServerApp.token = ''
c.ServerApp.password = ''
c.LabApp.default_url = '/lab'
EOF

echo "✅ Development environment setup complete!"
echo ""
echo "🎯 Quick start commands:"
echo "  pno-test          - Run test suite"
echo "  pno-lint          - Check code quality"
echo "  pno-format        - Format code"
echo "  pno-jupyter       - Start Jupyter Lab"
echo "  pno-tensorboard   - Start TensorBoard"
echo ""
echo "📚 Documentation: https://pno-physics-bench.readthedocs.io"
echo "🐛 Issues: https://github.com/yourusername/pno-physics-bench/issues"