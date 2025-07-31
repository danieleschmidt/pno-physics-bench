"""Test configuration and fixtures for PNO Physics Bench."""

import pytest
import torch
import numpy as np
import tempfile
import shutil
from pathlib import Path
from typing import Generator, Dict, Any

# Test configuration
pytest_plugins = ["pytest_benchmark"]


# Device fixtures
@pytest.fixture(scope="session")
def device() -> torch.device:
    """Get the appropriate device for testing."""
    if torch.cuda.is_available() and not pytest.config.getoption("--cpu-only", default=False):
        return torch.device("cuda:0")
    return torch.device("cpu")


@pytest.fixture(scope="session")
def cpu_device() -> torch.device:
    """Force CPU device for certain tests."""
    return torch.device("cpu")


# Data fixture
@pytest.fixture(scope="session")
def synthetic_dataset() -> Dict[str, torch.Tensor]:
    """Generate synthetic PDE dataset for testing."""
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Generate Burgers equation-like data
    batch_size = 16
    spatial_dim = 32
    time_steps = 10
    
    # Input: initial condition + forcing
    x_coords = torch.linspace(0, 2*np.pi, spatial_dim)
    inputs = torch.stack([
        torch.sin(x_coords).unsqueeze(0).expand(batch_size, -1),  # initial condition
        torch.cos(2*x_coords).unsqueeze(0).expand(batch_size, -1) * 0.1,  # forcing
    ], dim=1)  # [batch, 2, spatial]
    
    # Output: solution at later time
    targets = torch.sin(x_coords * 0.9).unsqueeze(0).expand(batch_size, -1)  # [batch, spatial]
    
    return {
        "inputs": inputs,
        "targets": targets,
        "spatial_coords": x_coords,
        "batch_size": batch_size,
        "spatial_dim": spatial_dim,
        "time_steps": time_steps
    }


@pytest.fixture(scope="session")
def navier_stokes_data() -> Dict[str, torch.Tensor]:
    """Generate synthetic 2D Navier-Stokes data."""
    torch.manual_seed(42)
    
    batch_size = 8
    height, width = 32, 32
    
    # Velocity field (u, v) and pressure
    u = torch.randn(batch_size, height, width) * 0.1
    v = torch.randn(batch_size, height, width) * 0.1
    p = torch.randn(batch_size, height, width) * 0.05
    
    inputs = torch.stack([u, v, p], dim=1)  # [batch, 3, height, width]
    
    # Future state (simplified)
    u_next = u * 0.98 + torch.randn_like(u) * 0.01
    v_next = v * 0.98 + torch.randn_like(v) * 0.01
    p_next = p * 0.95 + torch.randn_like(p) * 0.005
    
    targets = torch.stack([u_next, v_next, p_next], dim=1)
    
    return {
        "inputs": inputs,
        "targets": targets,
        "batch_size": batch_size,
        "height": height,
        "width": width
    }


# Model fixtures
@pytest.fixture
def simple_pno_config() -> Dict[str, Any]:
    """Configuration for a simple PNO model for testing."""
    return {
        "input_dim": 2,
        "output_dim": 1,
        "hidden_dim": 32,
        "num_layers": 2,
        "modes": 8,
        "uncertainty_type": "diagonal",
        "posterior": "variational"
    }


@pytest.fixture
def model_checkpoint_path(tmp_path) -> Generator[Path, None, None]:
    """Temporary path for model checkpoints."""
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    yield checkpoint_dir
    # Cleanup is automatic with tmp_path


# Training fixtures
@pytest.fixture
def training_config() -> Dict[str, Any]:
    """Configuration for training tests."""
    return {
        "learning_rate": 1e-3,
        "batch_size": 4,
        "max_epochs": 2,  # Small for testing
        "kl_weight": 1e-4,
        "num_mc_samples": 3,
        "early_stopping_patience": 10,
        "gradient_clip_val": 1.0,
        "scheduler": "cosine",
        "warmup_epochs": 1
    }


# Environment fixtures
@pytest.fixture(scope="session")
def test_data_dir() -> Generator[Path, None, None]:
    """Temporary directory for test data."""
    temp_dir = tempfile.mkdtemp(prefix="pno_test_")
    data_dir = Path(temp_dir)
    yield data_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def wandb_disabled(monkeypatch):
    """Disable W&B for testing."""
    monkeypatch.setenv("WANDB_MODE", "disabled")


# Performance fixtures
@pytest.fixture
def benchmark_config():
    """Configuration for benchmark tests."""
    return {
        "min_time": 0.01,  # Minimum time per benchmark
        "max_time": 10.0,  # Maximum time per benchmark
        "min_rounds": 3,   # Minimum number of rounds
        "warmup": True,    # Enable warmup rounds
    }


# Test data generation utilities
def generate_pde_solution(pde_type: str, size: tuple, **kwargs) -> torch.Tensor:
    """Generate synthetic PDE solutions for testing."""
    torch.manual_seed(42)
    
    if pde_type == "burgers":
        # 1D Burgers equation
        x = torch.linspace(0, 2*np.pi, size[0])
        return torch.sin(x).unsqueeze(0)
    
    elif pde_type == "heat":
        # 2D heat equation
        h, w = size
        x = torch.linspace(0, 1, w)
        y = torch.linspace(0, 1, h)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        return torch.exp(-(X-0.5)**2 - (Y-0.5)**2).unsqueeze(0)
    
    elif pde_type == "wave":
        # 2D wave equation
        h, w = size
        return torch.randn(1, h, w) * 0.1
    
    else:
        raise ValueError(f"Unknown PDE type: {pde_type}")


# Pytest configuration
def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--cpu-only",
        action="store_true",
        default=False,
        help="Force CPU-only testing even if CUDA is available"
    )
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="Run slow tests that are normally skipped"
    )
    parser.addoption(
        "--run-gpu",
        action="store_true",
        default=False,
        help="Run GPU-specific tests (requires CUDA)"
    )


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests that require GPU (deselect with '-m \"not gpu\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "benchmark: marks tests as benchmarks"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection based on command line options."""
    # Skip slow tests unless --run-slow is given
    if not config.getoption("--run-slow"):
        skip_slow = pytest.mark.skip(reason="need --run-slow option to run")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)
    
    # Skip GPU tests unless --run-gpu is given or GPU is available
    if not config.getoption("--run-gpu") and not torch.cuda.is_available():
        skip_gpu = pytest.mark.skip(reason="need --run-gpu option or CUDA to run")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)


# Test utilities
class TestUtils:
    """Utility class for common test operations."""
    
    @staticmethod
    def assert_tensor_shape(tensor: torch.Tensor, expected_shape: tuple, name: str = "tensor"):
        """Assert tensor has expected shape."""
        assert tensor.shape == expected_shape, \
            f"{name} shape mismatch: got {tensor.shape}, expected {expected_shape}"
    
    @staticmethod
    def assert_tensor_finite(tensor: torch.Tensor, name: str = "tensor"):
        """Assert tensor contains only finite values."""
        assert torch.isfinite(tensor).all(), f"{name} contains non-finite values"
    
    @staticmethod
    def assert_tensor_range(tensor: torch.Tensor, min_val: float, max_val: float, name: str = "tensor"):
        """Assert tensor values are within expected range."""
        assert tensor.min() >= min_val and tensor.max() <= max_val, \
            f"{name} values out of range [{min_val}, {max_val}]: got [{tensor.min():.4f}, {tensor.max():.4f}]"
    
    @staticmethod
    def assert_model_output_shape(model, input_tensor: torch.Tensor, expected_output_shape: tuple):
        """Assert model produces output with expected shape."""
        with torch.no_grad():
            output = model(input_tensor)
            if isinstance(output, tuple):
                output = output[0]  # Take mean if tuple (mean, std)
            TestUtils.assert_tensor_shape(output, expected_output_shape, "model output")


@pytest.fixture
def test_utils():
    """Provide test utilities."""
    return TestUtils