# PNO Physics Bench API Documentation

## Overview

PNO Physics Bench provides a comprehensive API for Probabilistic Neural Operators with advanced uncertainty quantification capabilities. This documentation covers all public APIs for the research modules, robustness frameworks, security systems, and scaling components developed during the autonomous SDLC execution.

## Table of Contents

1. [Core Models](#core-models)
2. [Research Modules](#research-modules)
3. [Robustness Framework](#robustness-framework)
4. [Security System](#security-system)
5. [Scaling Components](#scaling-components)
6. [Deployment APIs](#deployment-apis)

## Core Models

### ProbabilisticNeuralOperator

The main model class for uncertainty-aware neural PDE solving.

```python
class ProbabilisticNeuralOperator(nn.Module):
    def __init__(
        self,
        input_dim: int = 3,
        hidden_dim: int = 256,
        num_layers: int = 4,
        modes: int = 20,
        uncertainty_type: str = "full",
        posterior: str = "variational",
        activation: str = "gelu"
    )
```

**Parameters:**
- `input_dim` (int): Number of input channels (e.g., 3 for velocity components + pressure)
- `hidden_dim` (int): Hidden layer dimension
- `num_layers` (int): Number of Fourier layers
- `modes` (int): Number of Fourier modes to keep
- `uncertainty_type` (str): Type of uncertainty ("full", "diagonal", "scalar")
- `posterior` (str): Posterior approximation ("variational", "ensemble")
- `activation` (str): Activation function ("gelu", "relu", "swish")

**Methods:**

#### predict_with_uncertainty
```python
def predict_with_uncertainty(
    self, 
    x: torch.Tensor, 
    num_samples: int = 100
) -> Tuple[torch.Tensor, torch.Tensor]
```
Predict with uncertainty quantification using Monte Carlo sampling.

**Parameters:**
- `x` (Tensor): Input tensor [batch, channels, height, width]
- `num_samples` (int): Number of Monte Carlo samples

**Returns:**
- `prediction` (Tensor): Mean prediction [batch, output_channels, height, width]
- `uncertainty` (Tensor): Predictive uncertainty [batch, output_channels, height, width]

**Example:**
```python
model = ProbabilisticNeuralOperator(input_dim=3, hidden_dim=128)
prediction, uncertainty = model.predict_with_uncertainty(
    input_tensor, num_samples=100
)
```

## Research Modules

### Temporal Uncertainty Dynamics

#### TemporalUncertaintyKernel

Kernel for modeling temporal uncertainty correlations.

```python
class TemporalUncertaintyKernel(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        temporal_horizon: int = 10,
        kernel_type: str = "matern",
        correlation_decay: float = 0.9
    )
```

**Methods:**

##### compute_temporal_kernel
```python
def compute_temporal_kernel(
    self, 
    t1: torch.Tensor, 
    t2: torch.Tensor
) -> torch.Tensor
```
Compute temporal correlation kernel between time points.

#### AdaptiveTemporalPNO

Adaptive temporal probabilistic neural operator.

```python
class AdaptiveTemporalPNO(nn.Module):
    def __init__(
        self,
        base_pno: ProbabilisticNeuralOperator,
        temporal_horizon: int = 10,
        adaptation_rate: float = 0.01,
        uncertainty_threshold: float = 0.1
    )
```

##### predict_temporal_sequence
```python
def predict_temporal_sequence(
    self,
    initial_condition: torch.Tensor,
    time_points: torch.Tensor,
    num_samples: int = 100
) -> Tuple[torch.Tensor, torch.Tensor, List[TemporalUncertaintyState]]
```

**Example:**
```python
temporal_pno = AdaptiveTemporalPNO(base_pno, temporal_horizon=20)
predictions, uncertainties, states = temporal_pno.predict_temporal_sequence(
    initial_condition, time_points, num_samples=50
)
```

### Causal Uncertainty Inference

#### CausalUncertaintyInference

Neural network for learning causal relationships in uncertainty.

```python
class CausalUncertaintyInference(nn.Module):
    def __init__(
        self,
        spatial_shape: Tuple[int, int],
        hidden_dim: int = 128,
        num_causal_layers: int = 3,
        attention_heads: int = 8
    )
```

##### predict_intervention_effect
```python
def predict_intervention_effect(
    self,
    uncertainty_field: torch.Tensor,
    intervention_location: Tuple[int, int],
    intervention_value: float
) -> torch.Tensor
```

**Example:**
```python
causal_model = CausalUncertaintyInference(spatial_shape=(64, 64))
intervention_effect = causal_model.predict_intervention_effect(
    uncertainty_field, intervention_location=(32, 32), intervention_value=2.0
)
```

### Quantum Uncertainty Principles

#### QuantumUncertaintyPrinciple

Implementation of quantum-inspired uncertainty principles.

```python
class QuantumUncertaintyPrinciple:
    def __init__(
        self,
        spatial_shape: Tuple[int, int],
        hbar_effective: float = 1.0,
        principle_type: str = "heisenberg"
    )
```

##### compute_heisenberg_bound
```python
def compute_heisenberg_bound(
    self,
    field: torch.Tensor,
    observable1: QuantumObservable,
    observable2: QuantumObservable,
    op1_name: str,
    op2_name: str
) -> Dict[str, torch.Tensor]
```

**Example:**
```python
principle = QuantumUncertaintyPrinciple(spatial_shape=(32, 32))
result = principle.compute_heisenberg_bound(
    field, principle.position, principle.momentum, 'x', 'px'
)
print(f"Uncertainty product: {result['uncertainty_product']}")
print(f"Quantum bound: {result['quantum_bound']}")
```

## Robustness Framework

### ComprehensiveValidator

Main validator that orchestrates all validation tests.

```python
class ComprehensiveValidator:
    def __init__(
        self,
        pde_type: str = "navier_stokes",
        physics_tolerance: float = 1e-3,
        uncertainty_confidence_levels: List[float] = None,
        robustness_noise_levels: List[float] = None
    )
```

##### run_comprehensive_validation
```python
def run_comprehensive_validation(
    self,
    model: ProbabilisticNeuralOperator,
    test_data: Dict[str, torch.Tensor],
    boundary_type: str = "periodic"
) -> Dict[str, ValidationResult]
```

**Example:**
```python
validator = ComprehensiveValidator(pde_type="navier_stokes")
test_data = {
    'input': torch.randn(8, 3, 64, 64),
    'target': torch.randn(8, 1, 64, 64),
    'ood_input': torch.randn(8, 3, 64, 64) * 2.0
}
results = validator.run_comprehensive_validation(model, test_data)
```

### PhysicsConsistencyValidator

Validator for physics consistency in PDE solutions.

```python
class PhysicsConsistencyValidator:
    def __init__(
        self,
        pde_type: str = "navier_stokes",
        tolerance: float = 1e-3,
        conservation_weight: float = 1.0
    )
```

##### validate_conservation_laws
```python
def validate_conservation_laws(
    self,
    prediction: torch.Tensor,
    input_field: torch.Tensor
) -> ValidationResult
```

## Security System

### SecureInference

Secure inference protocols for production deployment.

```python
class SecureInference:
    def __init__(
        self,
        model: ProbabilisticNeuralOperator,
        encryption_key: Optional[bytes] = None,
        require_authentication: bool = True,
        audit_logging: bool = True
    )
```

##### secure_inference
```python
def secure_inference(
    self,
    encrypted_input: bytes,
    input_shape: Tuple[int, ...],
    auth_token: Optional[str] = None,
    client_id: str = "anonymous"
) -> Tuple[bytes, List[SecurityEvent]]
```

**Example:**
```python
secure_inference = SecureInference(model, require_authentication=True)
token = secure_inference.generate_auth_token("user123", expiry_hours=24)
encrypted_input = secure_inference.encrypt_data(input_tensor)
encrypted_output, events = secure_inference.secure_inference(
    encrypted_input, input_tensor.shape, auth_token=token
)
```

### InputSanitizer

Advanced input sanitization for neural operator inputs.

```python
class InputSanitizer:
    def __init__(
        self,
        expected_shape: Tuple[int, ...],
        value_range: Tuple[float, float] = (-10.0, 10.0),
        distribution_params: Optional[Dict[str, float]] = None,
        anomaly_threshold: float = 3.0
    )
```

##### validate_and_sanitize
```python
def validate_and_sanitize(
    self,
    input_tensor: torch.Tensor,
    strict_mode: bool = False,
    raise_on_failure: bool = False
) -> Tuple[torch.Tensor, List[SecurityEvent]]
```

## Scaling Components

### CachedPNOInference

PNO inference with intelligent caching.

```python
class CachedPNOInference:
    def __init__(
        self,
        model: ProbabilisticNeuralOperator,
        cache_system: Union[LocalCache, DistributedCache],
        similarity_threshold: float = 0.95,
        enable_semantic_caching: bool = True
    )
```

##### predict_with_uncertainty
```python
def predict_with_uncertainty(
    self,
    input_tensor: torch.Tensor,
    num_samples: int = 100,
    use_cache: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]
```

**Example:**
```python
cache = LocalCache(max_size_bytes=1024*1024*1024)  # 1GB
cached_inference = CachedPNOInference(model, cache)
prediction, uncertainty = cached_inference.predict_with_uncertainty(input_tensor)
stats = cached_inference.get_performance_stats()
print(f"Cache hit rate: {stats['cache_performance']['hit_rate']:.2%}")
```

### DistributedPNOTrainer

High-level distributed PNO trainer with advanced optimization.

```python
class DistributedPNOTrainer:
    def __init__(
        self,
        model: ProbabilisticNeuralOperator,
        config: DistributedConfig,
        checkpoint_dir: str = "./checkpoints"
    )
```

##### train
```python
def train(
    self,
    train_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    num_epochs: int,
    val_loader: Optional[torch.utils.data.DataLoader] = None
) -> Dict[str, List]
```

## Deployment APIs

### ProductionDeploymentGenerator

Generator for production deployment artifacts.

```python
class ProductionDeploymentGenerator:
    def __init__(self, config: DeploymentConfig)
```

##### generate_all_manifests
```python
def generate_all_manifests(self, output_dir: str = "./deployment") -> str
```

**Example:**
```python
config = DeploymentConfig(
    environment="production",
    replicas=3,
    gpu_limit=1,
    auto_scaling=True
)
generator = ProductionDeploymentGenerator(config)
deployment_dir = generator.generate_all_manifests()
```

## Error Handling

All APIs use consistent error handling patterns:

```python
from pno_physics_bench.exceptions import (
    PNOValidationError,
    PNOSecurityError,
    PNOComputationError
)

try:
    prediction, uncertainty = model.predict_with_uncertainty(input_tensor)
except PNOValidationError as e:
    print(f"Validation failed: {e}")
except PNOComputationError as e:
    print(f"Computation failed: {e}")
```

## Configuration

### Global Configuration

```python
from pno_physics_bench.config import PNOConfig

config = PNOConfig(
    device="cuda",
    precision="float32",
    cache_enabled=True,
    logging_level="INFO"
)

# Apply configuration globally
config.apply()
```

## Performance Optimization

### Best Practices

1. **Model Optimization:**
```python
# Use compiled models for faster inference
model = torch.compile(model)

# Enable mixed precision training
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()
```

2. **Caching Optimization:**
```python
# Warm up cache for common inputs
cached_inference.warm_cache(sample_inputs, max_workers=4)

# Use semantic caching for similar inputs
cached_inference = CachedPNOInference(
    model, cache, enable_semantic_caching=True
)
```

3. **Distributed Training:**
```python
# Use gradient compression for bandwidth efficiency
config = DistributedConfig(
    gradient_compression="quantization",
    compression_ratio=0.1
)
```

## Monitoring and Metrics

### Performance Metrics

```python
# Get comprehensive performance statistics
perf_stats = cached_inference.get_performance_stats()
print(f"Hit rate: {perf_stats['cache_performance']['hit_rate']:.2%}")
print(f"Speedup: {perf_stats['timing']['speedup_factor']:.2f}x")

# Validation metrics
validation_stats = validator.get_validation_summary()
print(f"Physics consistency: {validation_stats['physics_score']:.3f}")
print(f"Uncertainty calibration: {validation_stats['calibration_score']:.3f}")
```

### Research Metrics

```python
# Temporal uncertainty analysis
analyzer = TemporalUncertaintyAnalyzer()
temporal_results = analyzer.analyze_uncertainty_propagation(
    temporal_model, test_sequence, time_points
)

# Causal discovery metrics
causal_analyzer = CausalUncertaintyAnalyzer()
causal_results = causal_analyzer.test_causal_assumptions(
    causal_model, uncertainty_data
)

# Quantum principle validation
quantum_analyzer = QuantumUncertaintyAnalyzer()
quantum_results = quantum_analyzer.validate_uncertainty_principles(
    test_data, quantum_principle
)
```

## Examples and Tutorials

Complete examples are available in the `/examples` directory:

- **Basic Usage**: `examples/basic_pno_usage.py`
- **Temporal Uncertainty**: `examples/temporal_dynamics_example.py`
- **Causal Inference**: `examples/causal_analysis_example.py`
- **Quantum Principles**: `examples/quantum_uncertainty_example.py`
- **Production Deployment**: `examples/production_deployment_example.py`

## Support and Community

- **Documentation**: [https://pno-physics-bench.readthedocs.io](https://pno-physics-bench.readthedocs.io)
- **Issues**: [GitHub Issues](https://github.com/yourusername/pno-physics-bench/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/pno-physics-bench/discussions)
- **Papers**: See `RESEARCH_METHODOLOGY_REPORT.md` for research papers and citations

---

**Last Updated**: August 17, 2025  
**Version**: 1.0.0  
**License**: MIT