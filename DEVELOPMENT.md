# Development Guide

This guide covers the development setup and workflow for contributing to pno-physics-bench.

## Quick Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/pno-physics-bench.git
cd pno-physics-bench

# Create development environment
conda create -n pno-dev python=3.9
conda activate pno-dev

# Install in development mode with all dependencies
pip install -e ".[dev,jax,benchmark]"

# Install pre-commit hooks
pre-commit install
```

## Development Environment

### Requirements

- Python 3.9+
- PyTorch 2.0+
- CUDA-capable GPU (recommended for training)
- 16GB+ RAM (for larger datasets)

### Optional Dependencies

- **JAX Backend**: For alternative implementations
  ```bash
  pip install -e ".[jax]"
  ```

- **Benchmarking Tools**: For performance analysis
  ```bash
  pip install -e ".[benchmark]"
  ```

## Code Quality

We maintain high code quality standards:

### Formatting
- **Black**: Code formatting (88 character line limit)
- **isort**: Import sorting
- Run: `make format`

### Linting
- **flake8**: Style and error checking
- **mypy**: Static type checking
- Run: `make lint type-check`

### Testing
- **pytest**: Unit and integration tests
- **pytest-cov**: Coverage reporting (target: 80%+)
- Run: `make test-cov`

### Pre-commit Hooks
All code is automatically checked before commits:
```bash
# Manual run on all files
pre-commit run --all-files

# Skip hooks (emergency only)
git commit --no-verify
```

## Project Structure

```
pno-physics-bench/
├── src/pno_physics_bench/     # Main package code
│   ├── models/                # Neural operator models
│   ├── training/              # Training loops and utilities
│   ├── datasets/              # Data loading and preprocessing
│   ├── uncertainty/           # Uncertainty quantification
│   ├── metrics/               # Evaluation metrics
│   ├── visualization/         # Plotting and analysis
│   └── cli/                   # Command-line interface
├── tests/                     # Test suite
│   ├── unit/                  # Unit tests
│   ├── integration/           # Integration tests
│   └── benchmarks/            # Performance benchmarks
├── docs/                      # Documentation
├── configs/                   # Configuration files
├── scripts/                   # Utility scripts
└── notebooks/                 # Jupyter notebooks
```

## Testing

### Running Tests

```bash
# All tests
make test

# With coverage
make test-cov

# Specific test file
pytest tests/test_models.py -v

# Skip slow tests
pytest -m "not slow"

# GPU tests only (if available)
pytest -m gpu
```

### Test Categories

- **Unit tests**: Fast, isolated component tests
- **Integration tests**: Multi-component interaction tests  
- **GPU tests**: Require CUDA-capable hardware
- **Slow tests**: Long-running benchmarks and training tests

### Writing Tests

```python
import pytest
import torch
from pno_physics_bench.models import ProbabilisticNeuralOperator

class TestPNO:
    def test_forward_pass(self):
        """Test basic forward pass."""
        model = ProbabilisticNeuralOperator(
            input_dim=3, hidden_dim=64, num_layers=2
        )
        x = torch.randn(2, 3, 32, 32)
        output = model(x)
        assert output.shape == (2, 3, 32, 32)
    
    @pytest.mark.gpu
    def test_gpu_training(self):
        """Test GPU training functionality."""
        if not torch.cuda.is_available():
            pytest.skip("GPU not available")
        # GPU-specific test code
    
    @pytest.mark.slow
    def test_full_training_loop(self):
        """Test complete training workflow."""
        # Long-running integration test
```

## Documentation

### Building Docs

```bash
# Build documentation
make docs

# Serve locally
make serve-docs
# Open http://localhost:8000

# Live reload during development
cd docs && sphinx-autobuild . _build/html
```

### Documentation Standards

- Use NumPy-style docstrings
- Include type hints for all public APIs
- Add usage examples for complex functions
- Keep docstrings concise but complete

```python
def predict_with_uncertainty(
    self, 
    x: torch.Tensor, 
    num_samples: int = 100
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Predict with uncertainty quantification.
    
    Parameters
    ----------
    x : torch.Tensor
        Input tensor of shape (batch_size, input_dim, height, width).
    num_samples : int, optional
        Number of Monte Carlo samples for uncertainty estimation.
        Default is 100.
    
    Returns
    -------
    mean : torch.Tensor
        Predictive mean of shape (batch_size, output_dim, height, width).
    std : torch.Tensor  
        Predictive standard deviation with same shape as mean.
    
    Examples
    --------
    >>> model = ProbabilisticNeuralOperator(...)
    >>> x = torch.randn(8, 3, 64, 64)
    >>> mean, std = model.predict_with_uncertainty(x, num_samples=50)
    >>> print(f"Mean shape: {mean.shape}, Std shape: {std.shape}")
    """
```

## Research Workflow

### Experiment Management

1. **Configuration**: Use Hydra configs in `configs/`
2. **Logging**: Weights & Biases integration
3. **Reproducibility**: Set seeds, save configs
4. **Results**: Structured output in `results/`

### Running Experiments

```bash
# Train single model
python -m pno_physics_bench.cli train \
    --config configs/navier_stokes.yaml \
    --gpus 1 \
    --batch_size 32

# Hyperparameter sweep
python scripts/sweep.py --config configs/sweep_pno.yaml

# Benchmark all methods
python -m pno_physics_bench.cli benchmark \
    --all-pdes \
    --methods FNO,TNO,PNO \
    --output results/benchmark_$(date +%Y%m%d)
```

### Reproducibility Checklist

- [ ] Set random seeds (`torch.manual_seed(42)`)
- [ ] Save exact configuration used
- [ ] Log environment info (PyTorch version, CUDA, etc.)
- [ ] Track data preprocessing steps
- [ ] Save model checkpoints at key intervals
- [ ] Document hardware used (GPU model, memory)

## Performance Optimization

### Profiling

```bash
# Memory profiling
python -m memory_profiler train_script.py

# CPU profiling with py-spy
py-spy record -o profile.svg -- python train_script.py

# PyTorch profiler
python -m torch.profiler train_script.py
```

### Optimization Tips

1. **Data Loading**: Use `num_workers > 0`, `pin_memory=True`
2. **Mixed Precision**: Enable AMP with `torch.cuda.amp`
3. **Batch Size**: Maximize GPU utilization without OOM
4. **Gradient Accumulation**: For effective large batch training
5. **Model Parallelism**: For very large models

## Debugging

### Common Issues

1. **CUDA OOM**: Reduce batch size, enable gradient checkpointing
2. **NaN Gradients**: Check learning rate, add gradient clipping
3. **Slow Training**: Profile data loading, check GPU utilization
4. **Import Errors**: Verify installation, check PYTHONPATH

### Debug Mode

```bash
# Enable debug logging
export PYTHONPATH=.
export PNO_DEBUG=1
python -m pdb train_script.py

# PyTorch debugging
export TORCH_SHOW_CPP_STACKTRACES=1
export CUDA_LAUNCH_BLOCKING=1
```

## Contributing Workflow

1. **Issue**: Open issue for bugs/features
2. **Branch**: Create feature branch from main
3. **Develop**: Make changes, add tests
4. **Quality**: Run `make check-all`
5. **PR**: Open pull request with description
6. **Review**: Address reviewer feedback
7. **Merge**: Squash and merge after approval

### Commit Messages

Follow conventional commits:

```
feat: add uncertainty decomposition for PNO layers
fix: resolve CUDA memory leak in training loop
docs: update API documentation for metrics module
test: add integration tests for benchmark suite
refactor: simplify data loading pipeline
```

## Release Process

1. Update version in `pyproject.toml` and `__init__.py`
2. Update `CHANGELOG.md` with new features/fixes
3. Create release branch: `git checkout -b release/v0.2.0`
4. Run full test suite: `make check-all`
5. Build package: `make build`
6. Create GitHub release with tag
7. Upload to PyPI: `twine upload dist/*`

## Getting Help

- **Documentation**: [pno-physics-bench.readthedocs.io](https://pno-physics-bench.readthedocs.io)
- **Issues**: [GitHub Issues](https://github.com/yourusername/pno-physics-bench/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/pno-physics-bench/discussions)
- **Email**: daniel@terragonlabs.com