# ADR-0002: PyTorch as Primary Deep Learning Framework

## Status

Accepted

## Context

The project requires a deep learning framework that supports:
- Complex tensor operations for spectral methods
- Automatic differentiation for variational inference
- GPU acceleration for large-scale training
- Flexible model architectures for research
- Strong ecosystem for scientific computing

## Decision

We will use PyTorch as the primary deep learning framework with optional JAX support for specific computational kernels.

## Consequences

### Positive

- Mature ecosystem with extensive community support
- Excellent debugging capabilities with eager execution
- Strong integration with scientific Python stack
- Flexible architecture for research experimentation
- Native support for variational inference patterns
- Good performance on GPU clusters

### Negative

- Slightly slower than JAX for pure numerical computations
- More verbose syntax compared to JAX
- Requires careful memory management for large models

### Neutral

- Industry standard choice for research projects
- Compatible with most CI/CD and deployment tools

## Alternatives Considered

1. **JAX**: Faster compilation but steeper learning curve and smaller ecosystem
2. **TensorFlow**: More deployment-focused, less research-friendly
3. **Julia**: Excellent performance but smaller community

## Implementation Notes

- Use PyTorch ≥2.0 for improved compilation features
- Implement critical numerical kernels in JAX as optional dependency
- Use torch.jit for deployment optimization
- Leverage torch.distributions for probabilistic components
- Use DDP for multi-GPU training

## References

- PyTorch 2.0 Documentation
- Neural Operator implementations in PyTorch ecosystem
- Variational Inference with PyTorch