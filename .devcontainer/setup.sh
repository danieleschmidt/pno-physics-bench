#!/bin/bash
# Development container setup script

set -e

echo "🚀 Setting up PNO Physics Bench development environment..."

# Update system
echo "📦 Updating system packages..."
sudo apt-get update -qq

# Install additional development tools
echo "🛠️ Installing development tools..."
sudo apt-get install -y \
    htop \
    tree \
    jq \
    curl \
    wget \
    unzip \
    graphviz \
    pandoc \
    texlive-latex-base \
    texlive-latex-extra

# Install Python development dependencies
echo "🐍 Installing Python dependencies..."
pip install --upgrade pip setuptools wheel
pip install -e ".[dev,benchmark,all]"

# Install additional ML tools
echo "🧠 Installing additional ML tools..."
pip install \
    jupyterlab-git \
    jupyterlab-lsp \
    jupyter-ai \
    tensorboard-plugin-profile \
    wandb \
    mlflow

# Setup pre-commit hooks
echo "🔧 Setting up pre-commit hooks..."
pre-commit install
pre-commit install --hook-type commit-msg

# Create necessary directories
echo "📁 Creating project directories..."
mkdir -p \
    data \
    results \
    checkpoints \
    logs \
    experiments \
    notebooks

# Download sample datasets (if they exist)
echo "📊 Setting up sample data..."
if [ -f "scripts/download_data.py" ]; then
    python scripts/download_data.py --sample
fi

# Setup Jupyter extensions
echo "🪐 Configuring Jupyter..."
jupyter lab build --dev-build=False --minimize=False

# Create useful aliases
echo "🔗 Setting up aliases..."
cat >> ~/.bashrc << 'EOF'

# PNO Physics Bench aliases
alias pno-train="python -m pno_physics_bench.cli train"
alias pno-eval="python -m pno_physics_bench.cli evaluate"
alias pno-benchmark="python -m pno_physics_bench.cli benchmark"
alias pno-test="pytest tests/ -v"
alias pno-test-fast="pytest tests/ -v -m 'not slow'"
alias pno-lint="make lint"
alias pno-format="make format"
alias pno-docs="make docs && make serve-docs"
alias pno-clean="make clean"
alias tensorboard-pno="tensorboard --logdir=results/tensorboard_logs"

# Git aliases
alias gs="git status"
alias ga="git add"
alias gc="git commit"
alias gp="git push"
alias gl="git log --oneline -10"

# Python aliases
alias py="python"
alias ipy="ipython"
alias jlab="jupyter lab --ip=0.0.0.0 --allow-root"

# Docker aliases
alias dc="docker-compose"
alias dcup="docker-compose up -d"
alias dcdown="docker-compose down"
alias dclogs="docker-compose logs -f"
EOF

# Setup git configuration helpers
echo "📝 Setting up git helpers..."
cat > ~/.gitconfig_devcontainer << 'EOF'
[core]
    editor = code --wait
    autocrlf = input
[merge]
    tool = vscode
[mergetool "vscode"]
    cmd = code --wait $MERGED
[diff]
    tool = vscode
[difftool "vscode"]
    cmd = code --wait --diff $LOCAL $REMOTE
[pull]
    rebase = false
[init]
    defaultBranch = main
EOF

# Create development configuration
echo "⚙️ Creating development configuration..."
cat > config/dev_config.yaml << 'EOF'
# Development configuration for PNO Physics Bench
development:
  data_dir: "/app/data"
  results_dir: "/app/results"
  checkpoints_dir: "/app/checkpoints"
  
  # Reduced sizes for development
  default_batch_size: 4
  default_epochs: 10
  default_dataset_size: 1000
  
  # Debugging options
  debug: true
  log_level: "DEBUG"
  profile: false
  
  # Visualization
  plot_during_training: true
  save_plots: true
  
  # Experiment tracking
  wandb:
    mode: "offline"  # Use offline mode for development
    project: "pno-physics-bench-dev"
  
  # Testing
  run_slow_tests: false
  run_gpu_tests: false
EOF

# Create useful notebooks
echo "📓 Creating sample notebooks..."
mkdir -p notebooks/examples

cat > notebooks/examples/getting_started.ipynb << 'EOF'
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Getting Started with PNO Physics Bench\n",
    "\n",
    "This notebook demonstrates the basic usage of the PNO Physics Bench library."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import torch\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "# Import PNO components (when implemented)\n",
    "# from pno_physics_bench import ProbabilisticNeuralOperator\n",
    "# from pno_physics_bench.datasets import PDEDataset\n",
    "\n",
    "print(\"PNO Physics Bench development environment is ready!\")\n",
    "print(f\"PyTorch version: {torch.__version__}\")\n",
    "print(f\"CUDA available: {torch.cuda.is_available()}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Next Steps\n",
    "\n",
    "1. Implement your PNO models in `src/pno_physics_bench/models/`\n",
    "2. Add datasets in `src/pno_physics_bench/datasets/`\n",
    "3. Write tests in `tests/`\n",
    "4. Run experiments and track results"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
EOF

# Setup VS Code workspace settings
echo "🎨 Configuring VS Code workspace..."
mkdir -p .vscode
cat > .vscode/settings.json << 'EOF'
{
    "python.defaultInterpreterPath": "/usr/local/bin/python",
    "python.terminal.activateEnvironment": false,
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "python.linting.mypyEnabled": true,
    "python.formatting.provider": "black",
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": [
        "tests/",
        "-v",
        "--tb=short"
    ],
    "jupyter.jupyterServerType": "local",
    "files.watcherExclude": {
        "**/wandb/**": true,
        "**/lightning_logs/**": true,
        "**/.git/**": true,
        "**/node_modules/**": true,
        "**/__pycache__/**": true,
        "**/.pytest_cache/**": true,
        "**/.mypy_cache/**": true
    },
    "search.exclude": {
        "**/wandb": true,
        "**/lightning_logs": true,
        "**/.git": true,
        "**/node_modules": true,
        "**/__pycache__": true,
        "**/.pytest_cache": true,
        "**/.mypy_cache": true,
        "**/data": true,
        "**/results": true
    }
}
EOF

# Create launch configurations
cat > .vscode/launch.json << 'EOF'
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: Current File",
            "type": "python",
            "request": "launch",
            "program": "${file}",
            "console": "integratedTerminal",
            "cwd": "${workspaceFolder}",
            "env": {
                "PYTHONPATH": "${workspaceFolder}/src"
            }
        },
        {
            "name": "PNO: Train Model",
            "type": "python",
            "request": "launch",
            "module": "pno_physics_bench.cli",
            "args": ["train", "--config", "configs/development.yaml"],
            "console": "integratedTerminal",
            "cwd": "${workspaceFolder}",
            "env": {
                "PYTHONPATH": "${workspaceFolder}/src"
            }
        },
        {
            "name": "pytest: Current File",
            "type": "python",
            "request": "launch",
            "module": "pytest",
            "args": ["${file}", "-v"],
            "console": "integratedTerminal",
            "cwd": "${workspaceFolder}",
            "env": {
                "PYTHONPATH": "${workspaceFolder}/src"
            }
        }
    ]
}
EOF

# Create tasks for common operations
cat > .vscode/tasks.json << 'EOF'
{
    "version": "2.0.0",
    "tasks": [
        {
            "label": "Run Tests",
            "type": "shell",
            "command": "pytest",
            "args": ["tests/", "-v"],
            "group": "test",
            "presentation": {
                "echo": true,
                "reveal": "always",
                "focus": false,
                "panel": "shared"
            }
        },
        {
            "label": "Format Code",
            "type": "shell",
            "command": "make",
            "args": ["format"],
            "group": "build",
            "presentation": {
                "echo": true,
                "reveal": "always",
                "focus": false,
                "panel": "shared"
            }
        },
        {
            "label": "Lint Code",
            "type": "shell",
            "command": "make",
            "args": ["lint"],
            "group": "build",
            "presentation": {
                "echo": true,
                "reveal": "always",
                "focus": false,
                "panel": "shared"
            }
        },
        {
            "label": "Build Documentation",
            "type": "shell",
            "command": "make",
            "args": ["docs"],
            "group": "build",
            "presentation": {
                "echo": true,
                "reveal": "always",
                "focus": false,
                "panel": "shared"
            }
        }
    ]
}
EOF

# Setup completion
echo "✅ Development environment setup complete!"
echo ""
echo "🎉 You can now:"
echo "   • Use 'make install-dev' to install development dependencies"
echo "   • Use 'make test' to run tests"
echo "   • Use 'make format' to format code"
echo "   • Use 'make lint' to check code quality"
echo "   • Use 'jupyter lab' to start Jupyter Lab"
echo "   • Access TensorBoard at http://localhost:6006"
echo ""
echo "📚 Check out the examples in notebooks/examples/"
echo "🔧 VS Code is configured with useful extensions and settings"
echo ""

# Source the new aliases
source ~/.bashrc

echo "🚀 Happy coding!"