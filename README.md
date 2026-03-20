# pno-physics-bench

Benchmarking uncertainty quantification for Physics-informed Neural Operators (PNOs) on canonical PDE problems.

## Overview

This library provides:

- **Gaussian Process baseline** — RBF kernel, Cholesky-based GP regression (numpy only)
- **PNO uncertainty head** — two-head MLP (mean + log_var) for heteroscedastic uncertainty
- **Calibration metrics** — NLL, CRPS, coverage, calibration error
- **Benchmark runner** — 1D heat equation and Burgers equation benchmarks with analytical solutions

## Install

```bash
pip install -e .
```

## Quick start

```python
from pno_bench import BenchmarkRunner, GaussianProcessBaseline, PNOLayer

# Run a full benchmark
runner = BenchmarkRunner()
heat_results = runner.run_heat_equation(n_points=50)
print(heat_results)
# {'gp_nll': ..., 'pno_nll': ..., 'gp_crps': ..., ...}

runner.export_calibration(heat_results, "heat_calibration.json")

burgers_results = runner.run_burgers_equation(n_points=50)
print(burgers_results)
```

## Components

### `GaussianProcessBaseline`

RBF kernel GP regression:

```
k(x, x') = σ² exp(-||x - x'||² / (2 l²))
```

Cholesky decomposition for numerical stability.

```python
gp = GaussianProcessBaseline(length_scale=1.0, sigma=1.0, noise=1e-3)
gp.fit(X_train, y_train)
mean, std = gp.predict(X_test)
```

### `PNOLayer`

Physics-informed Neural Operator with a heteroscedastic uncertainty head:

```
input → hidden (ReLU) → mean head
                       → log_var head
std = exp(0.5 * log_var)
```

```python
pno = PNOLayer(input_dim=1, hidden_dim=32, output_dim=1)
pno.fit(X_train, y_train, epochs=200, lr=0.01)
mean, std = pno.predict_with_uncertainty(X_test)
```

### `UncertaintyMetrics`

```python
from pno_bench import UncertaintyMetrics

nll   = UncertaintyMetrics.nll(y_true, mean, std)
crps  = UncertaintyMetrics.crps(y_true, mean, std)
cov   = UncertaintyMetrics.coverage(y_true, mean, std, alpha=0.95)
cal_e = UncertaintyMetrics.calibration_error(y_true, mean, std)
```

## PDE benchmarks

| Equation | Analytical solution |
|---|---|
| Heat: `u_t = α u_xx` | `exp(-α π² t) sin(π x)` |
| Burgers: `u_t + u u_x = ν u_xx` | Cole-Hopf transformation |

## Tests

```bash
pytest tests/ -v
```

## License

MIT
