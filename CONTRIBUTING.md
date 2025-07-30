# Contributing to PNO Physics Bench

Thank you for your interest in contributing to PNO Physics Bench! This document provides guidelines for contributing to our project.

## Table of Contents

- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Contributing Process](#contributing-process)
- [Code Standards](#code-standards)
- [Testing Guidelines](#testing-guidelines)
- [Documentation](#documentation)
- [Research Contributions](#research-contributions)
- [Community Guidelines](#community-guidelines)

## Getting Started

### Ways to Contribute

We welcome contributions in several forms:

- **Bug Reports**: Found a bug? Please report it!
- **Feature Requests**: Have ideas for new features?
- **Code Contributions**: Implement new features or fix bugs
- **Documentation**: Improve our docs and examples
- **Research**: Add new PDE benchmarks or uncertainty methods
- **Performance**: Optimize existing implementations
- **Testing**: Improve test coverage and quality

### Before You Start

1. **Check existing issues** to avoid duplicate work
2. **Open an issue** to discuss major changes before implementing
3. **Read our [Code of Conduct](CODE_OF_CONDUCT.md)**
4. **Review our [Development Guide](DEVELOPMENT.md)** for technical details

## Development Setup

### Quick Setup

```bash
# Fork and clone your fork
git clone https://github.com/yourusername/pno-physics-bench.git
cd pno-physics-bench

# Add upstream remote
git remote add upstream https://github.com/originaluser/pno-physics-bench.git

# Create development environment
conda create -n pno-dev python=3.9
conda activate pno-dev

# Install in development mode
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Verify Setup

```bash
# Run tests to ensure everything works
make test

# Check code quality
make check-all
```

## Contributing Process

### 1. Create an Issue

For significant changes, open an issue first:

- **Bug Report**: Use the bug report template
- **Feature Request**: Use the feature request template
- **Research Addition**: Use the research proposal template

### 2. Fork and Branch

```bash
# Create a feature branch
git checkout -b feature/your-feature-name

# Or for bug fixes
git checkout -b fix/issue-description
```

### 3. Make Changes

- **Write tests** for new functionality
- **Update documentation** as needed
- **Follow our code standards** (see below)
- **Keep commits focused** and write good commit messages

### 4. Test Your Changes

```bash
# Run the full test suite
make test-cov

# Check code quality
make lint type-check

# Run specific tests
pytest tests/test_your_feature.py -v
```

### 5. Update Documentation

- Update docstrings for any changed APIs
- Add examples for new features
- Update relevant sections in docs/
- Consider adding a tutorial if appropriate

### 6. Submit Pull Request

- **Push to your fork**: `git push origin feature/your-feature-name`
- **Open a PR** against the main branch
- **Fill out the PR template** completely
- **Link related issues** using keywords (e.g., "Fixes #123")

### 7. Review Process

- Maintainers will review your PR
- Address any feedback promptly
- Keep your branch up to date with main
- Once approved, we'll merge your changes

## Code Standards

### Style Guidelines

- **Python Style**: Follow PEP 8, enforced by Black (88 char line limit)
- **Import Order**: Use isort with Black profile
- **Type Hints**: Required for all public APIs
- **Docstrings**: NumPy style for all public functions/classes

### Code Quality

```bash
# Auto-format code
make format

# Check style and types
make lint type-check

# All quality checks
make check-all
```

### Naming Conventions

- **Classes**: PascalCase (e.g., `ProbabilisticNeuralOperator`)
- **Functions/Variables**: snake_case (e.g., `compute_uncertainty`)
- **Constants**: UPPER_SNAKE_CASE (e.g., `DEFAULT_BATCH_SIZE`)
- **Private**: Leading underscore (e.g., `_internal_method`)

### Documentation Standards

```python
def compute_uncertainty(
    predictions: torch.Tensor,
    num_samples: int = 100,
    method: str = "monte_carlo"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute predictive uncertainty from model outputs.
    
    Parameters
    ----------
    predictions : torch.Tensor
        Model predictions of shape (num_samples, batch_size, ...).
    num_samples : int, optional
        Number of Monte Carlo samples used. Default is 100.
    method : str, optional
        Uncertainty computation method. Options are 'monte_carlo',
        'variational', or 'ensemble'. Default is 'monte_carlo'.
    
    Returns
    -------
    mean : torch.Tensor
        Predictive mean with shape (batch_size, ...).
    std : torch.Tensor
        Predictive standard deviation with same shape as mean.
    
    Raises
    ------
    ValueError
        If method is not supported or inputs have invalid shapes.
    
    Examples
    --------
    >>> predictions = torch.randn(100, 8, 64, 64)
    >>> mean, std = compute_uncertainty(predictions, num_samples=100)
    >>> print(f"Uncertainty range: {std.min():.3f} - {std.max():.3f}")
    """
```

## Testing Guidelines

### Test Categories

- **Unit Tests**: Fast, isolated component tests
- **Integration Tests**: Multi-component workflows
- **GPU Tests**: Require CUDA hardware
- **Slow Tests**: Long-running benchmarks

### Writing Tests

```python
import pytest
import torch
from pno_physics_bench.models import ProbabilisticNeuralOperator

class TestProbabilisticNeuralOperator:
    """Test suite for PNO model."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.model = ProbabilisticNeuralOperator(
            input_dim=3,
            hidden_dim=64,
            num_layers=2,
            modes=8
        )
        self.batch_size = 4
        self.resolution = 32
    
    def test_forward_pass(self):
        """Test basic forward pass functionality."""
        x = torch.randn(self.batch_size, 3, self.resolution, self.resolution)
        output = self.model(x)
        
        assert output.shape == (self.batch_size, 3, self.resolution, self.resolution)
        assert not torch.isnan(output).any()
        assert torch.isfinite(output).all()
    
    @pytest.mark.gpu
    def test_gpu_computation(self):
        """Test GPU functionality if available."""
        if not torch.cuda.is_available():
            pytest.skip("GPU not available")
        
        device = torch.device("cuda")
        model = self.model.to(device)
        x = torch.randn(2, 3, 16, 16, device=device)
        
        output = model(x)
        assert output.device == device
    
    @pytest.mark.slow
    def test_training_convergence(self):
        """Test that model can overfit small dataset."""
        # Long-running test to verify training works
        pass
    
    @pytest.mark.parametrize("uncertainty_type", ["diagonal", "full"])
    def test_uncertainty_types(self, uncertainty_type):
        """Test different uncertainty parameterizations."""
        model = ProbabilisticNeuralOperator(
            input_dim=3,
            hidden_dim=32,
            uncertainty_type=uncertainty_type
        )
        x = torch.randn(2, 3, 16, 16)
        mean, std = model.predict_with_uncertainty(x)
        
        assert mean.shape == (2, 3, 16, 16)
        assert std.shape == mean.shape
        assert (std > 0).all()
```

### Test Coverage

- **Target**: 80%+ overall coverage
- **Critical Paths**: 95%+ coverage for core functionality
- **New Code**: All new features must include tests

```bash
# Generate coverage report
make test-cov
open htmlcov/index.html  # View detailed coverage
```

## Documentation

### Types of Documentation

1. **API Documentation**: Auto-generated from docstrings
2. **Tutorials**: Step-by-step guides for common tasks
3. **Examples**: Jupyter notebooks with real use cases
4. **Theory**: Mathematical background and algorithm details

### Building Documentation

```bash
# Build docs
make docs

# Serve locally with live reload
cd docs && sphinx-autobuild . _build/html
```

### Writing Good Documentation

- **Start with why**: Explain the purpose before the how
- **Include examples**: Show real usage patterns
- **Be concise**: Respect the reader's time
- **Link extensively**: Connect related concepts
- **Test examples**: Ensure code examples actually work

## Research Contributions

### Adding New PDE Benchmarks

1. **Dataset**: Provide data generation or loading code
2. **Baseline**: Implement at least one baseline method
3. **Metrics**: Define appropriate evaluation metrics
4. **Documentation**: Write theory background and usage guide
5. **Tests**: Ensure reproducible results

### Implementing New Methods

1. **Literature**: Reference original papers
2. **Interface**: Follow existing API patterns
3. **Validation**: Compare against published results
4. **Performance**: Profile and optimize if needed
5. **Examples**: Provide tutorial notebook

### Benchmark Structure

```python
class NewPDEBenchmark(BaseBenchmark):
    """Benchmark for [PDE Name] equation.
    
    References
    ----------
    .. [1] Author et al. "Paper Title", Journal, Year.
    """
    
    def generate_data(self, num_samples: int, resolution: int) -> Dataset:
        """Generate synthetic data for this PDE."""
        pass
    
    def load_data(self, split: str) -> Dataset:
        """Load real-world data if available."""
        pass
    
    def evaluate_method(self, method: str, **kwargs) -> Dict[str, float]:
        """Evaluate a method on this benchmark."""
        pass
```

## Community Guidelines

### Communication

- **Be respectful**: Treat all contributors with respect
- **Be constructive**: Provide helpful feedback
- **Be patient**: Everyone has different experience levels
- **Ask questions**: Don't hesitate to ask for clarification

### Issue and PR Etiquette

- **Search first**: Check for existing issues/PRs
- **Be specific**: Provide minimal reproducible examples
- **Stay on topic**: Keep discussions focused
- **Update status**: Close issues when resolved

### Code Review Guidelines

#### For Contributors

- **Test thoroughly**: Ensure your code works as expected
- **Write clear descriptions**: Explain what and why
- **Respond promptly**: Address reviewer feedback quickly
- **Be open to feedback**: View reviews as learning opportunities

#### For Reviewers

- **Be thorough**: Check code, tests, and documentation
- **Be constructive**: Suggest improvements, don't just criticize
- **Be specific**: Point to exact lines and explain issues
- **Acknowledge good work**: Positive feedback motivates contributors

### Recognition

We recognize contributors in several ways:

- **Changelog**: Major contributions noted in release notes
- **Contributors File**: All contributors listed in CONTRIBUTORS.md
- **Social Media**: Significant contributions highlighted on Twitter/LinkedIn
- **Conference Presentations**: Contributors invited to present work

## Getting Help

### Resources

- **Documentation**: [pno-physics-bench.readthedocs.io](https://pno-physics-bench.readthedocs.io)
- **Development Guide**: [DEVELOPMENT.md](DEVELOPMENT.md)
- **GitHub Discussions**: For questions and general discussion
- **GitHub Issues**: For bug reports and feature requests

### Contact

- **Project Lead**: Daniel Schmidt (daniel@terragonlabs.com)
- **GitHub**: [@yourusername](https://github.com/yourusername)
- **Twitter**: [@terragonlabs](https://twitter.com/terragonlabs)

## License

By contributing to PNO Physics Bench, you agree that your contributions will be licensed under the same [MIT License](LICENSE) that covers the project.

Thank you for contributing to advancing uncertainty quantification in neural PDE solvers! 🚀