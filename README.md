# pno-physics-bench

**Physics Neural Operator Benchmark** — uncertainty quantification for 1D PDE problems using Gaussian Process baselines and neural operator layers.

## Overview

`pno-physics-bench` provides:

- **GaussianProcessBaseline** — RBF kernel GP with posterior mean + variance
- **PNOLayer** — simple feedforward neural operator outputting mean + log-variance
- **UncertaintyMetrics** — NLL, CRPS, and coverage probability
- **BenchmarkRunner** — runs 1D heat and Burgers equation benchmarks
- **calibration_export** — saves results to JSON and CSV

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```python
import numpy as np
from pno_physics_bench import BenchmarkRunner, calibration_export

runner = BenchmarkRunner(n_train=30, n_test=20)
results = runner.run_all()

# Print GP metrics on heat equation
print("Heat / GP:", results["heat"]["gp"])
print("Heat / PNO:", results["heat"]["pno"])

# Export calibration data
calibration_export(results, "output/calibration")
```

## Modules

### `GaussianProcessBaseline`

RBF (squared exponential) kernel: `k(x1, x2) = exp(-||x1-x2||² / (2l²))`

```python
from pno_physics_bench import GaussianProcessBaseline
import numpy as np

gp = GaussianProcessBaseline(length_scale=0.3, noise=1e-3)
X_train = np.linspace(0, 1, 20)
y_train = np.sin(2 * np.pi * X_train)
gp.fit(X_train, y_train)

X_test = np.linspace(0, 1, 10)
mean, variance = gp.predict(X_test)
```

### `PNOLayer`

Two-layer MLP outputting `[mean, log_var]` per input point:

```python
from pno_physics_bench import PNOLayer
import numpy as np

pno = PNOLayer(input_dim=1, hidden_dim=64)
x = np.linspace(0, 1, 50)
mean, log_var = pno.forward(x)
sigma = np.sqrt(np.exp(log_var))
```

### `UncertaintyMetrics`

```python
from pno_physics_bench import UncertaintyMetrics
import numpy as np

y = np.array([1.0, 2.0, 3.0])
mu = np.array([1.1, 1.9, 3.1])
sigma = np.array([0.2, 0.3, 0.2])

nll = UncertaintyMetrics.nll(y, mu, sigma)
crps = UncertaintyMetrics.crps(y, mu, sigma)
coverage = UncertaintyMetrics.coverage(y, mu, sigma)
```

### `BenchmarkRunner`

Solves 1D PDEs using finite differences and evaluates GP vs PNO:

- **Heat equation**: `u_t = ν u_xx`  (IC: sin(πx), Dirichlet BCs)
- **Burgers equation**: `u_t + u u_x = ν u_xx`  (IC: -sin(x), periodic BCs)

### `calibration_export`

```python
from pno_physics_bench import calibration_export
calibration_export(results, "output/calibration")
# Creates: output/calibration.json, output/calibration.csv
```

## Running Tests

```bash
python -m pytest tests/ -v
```

## Dependencies

- `numpy >= 1.21`
- `scipy >= 1.7`
- `pytest >= 7.0` (for tests)

## License

MIT
