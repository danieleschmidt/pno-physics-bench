# pno-physics-bench

Benchmark for uncertainty quantification in neural PDE solvers. Compares Gaussian Process baselines against Probabilistic Neural Operators (PNO) on 1D heat and Burgers equations.

## Components

- **GaussianProcessBaseline** — RBF kernel GP with full posterior uncertainty
- **PNOLayer** — Neural operator with mean + log_var uncertainty head
- **UncertaintyMetrics** — NLL, CRPS, coverage probability, calibration data
- **BenchmarkRunner** — Systematic comparison on 1D heat and Burgers PDEs

## Usage

```python
from pno_physics_bench.benchmark import BenchmarkRunner
runner = BenchmarkRunner(n_train=50, n_test=200)
results = runner.run_all()
print(runner.summary())
```

## Install

```bash
pip install -r requirements.txt
pytest tests/
```
