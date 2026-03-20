"""Tests for pno-physics-bench."""
import numpy as np
import pytest
from pno_physics_bench.gp_baseline import GaussianProcessBaseline
from pno_physics_bench.pno_layer import PNOLayer
from pno_physics_bench.metrics import UncertaintyMetrics
from pno_physics_bench.benchmark import BenchmarkRunner, _heat_solution, _burgers_approx


class TestGPBaseline:
    def test_fit_predict_shape(self):
        gp = GaussianProcessBaseline()
        X = np.linspace(0, 1, 20)
        y = np.sin(X)
        gp.fit(X, y)
        mean, var = gp.predict(np.linspace(0, 1, 10))
        assert mean.shape == (10,)
        assert var.shape == (10,)

    def test_variance_positive(self):
        gp = GaussianProcessBaseline()
        X = np.linspace(0, 1, 20)
        gp.fit(X, np.sin(X))
        _, var = gp.predict(np.array([0.5, 0.7]))
        assert np.all(var > 0)

    def test_interpolation_quality(self):
        gp = GaussianProcessBaseline(length_scale=0.3, noise_variance=1e-5)
        X = np.linspace(0, 1, 30)
        y = np.sin(np.pi * X)
        gp.fit(X, y)
        mean, _ = gp.predict(X)
        assert np.mean((mean - y) ** 2) < 1e-3

    def test_rbf_kernel_symmetric(self):
        gp = GaussianProcessBaseline()
        X = np.array([0.1, 0.5, 0.9])
        K = gp._rbf_kernel(X, X)
        assert K.shape == (3, 3)
        np.testing.assert_allclose(K, K.T, atol=1e-12)


class TestPNOLayer:
    def test_forward_shapes(self):
        pno = PNOLayer(input_dim=4, hidden_dim=16, output_dim=1)
        x = np.random.randn(10, 4)
        mean, log_var = pno.forward(x)
        assert mean.shape == (10, 1)
        assert log_var.shape == (10, 1)

    def test_predict_variance_positive(self):
        pno = PNOLayer(input_dim=2, hidden_dim=8, output_dim=1)
        x = np.random.randn(5, 2)
        _, var = pno.predict(x)
        assert np.all(var > 0)

    def test_fit_reduces_loss(self):
        rng = np.random.default_rng(0)
        X = np.column_stack([np.linspace(0, 1, 50), np.linspace(0, 1, 50)**2])
        y = np.sin(np.pi * X[:, 0:1])
        pno = PNOLayer(input_dim=2, hidden_dim=32, output_dim=1, seed=0)
        mean_before, var_before = pno.predict(X[:5])
        pno.fit(X, y, lr=0.05, epochs=100)
        mean_after, _ = pno.predict(X)
        rmse = np.sqrt(np.mean((mean_after - y)**2))
        assert rmse < 0.5  # should learn something

    def test_log_var_clipped(self):
        pno = PNOLayer(input_dim=2, hidden_dim=8, output_dim=1)
        x = np.random.randn(10, 2) * 100
        _, log_var = pno.forward(x)
        assert np.all(log_var <= 5)
        assert np.all(log_var >= -10)


class TestUncertaintyMetrics:
    def setup_method(self):
        rng = np.random.default_rng(1)
        self.mean = rng.standard_normal(100)
        self.var = np.ones(100) * 0.25
        self.y = self.mean + rng.normal(0, 0.5, 100)

    def test_nll_finite(self):
        nll = UncertaintyMetrics.nll_gaussian(self.y, self.mean, self.var)
        assert np.isfinite(nll)

    def test_crps_positive(self):
        crps = UncertaintyMetrics.crps_gaussian(self.y, self.mean, self.var)
        assert crps > 0

    def test_coverage_range(self):
        cov = UncertaintyMetrics.coverage_probability(self.y, self.mean, self.var, alpha=0.9)
        assert 0 <= cov <= 1

    def test_well_calibrated_coverage(self):
        # Perfect predictions: coverage should be ~90%
        rng = np.random.default_rng(42)
        mean = rng.standard_normal(1000)
        var = np.ones(1000)
        y = mean + rng.standard_normal(1000)
        cov = UncertaintyMetrics.coverage_probability(y, mean, var, alpha=0.9)
        assert 0.85 < cov < 0.95

    def test_calibration_data_shape(self):
        exp, obs = UncertaintyMetrics.calibration_data(self.y, self.mean, self.var, n_bins=5)
        assert len(exp) == 5
        assert len(obs) == 5


class TestBenchmarkRunner:
    def test_run_heat(self):
        runner = BenchmarkRunner(n_train=20, n_test=30, seed=0)
        runner.run_heat()
        assert "heat_gp" in runner.results
        assert "heat_pno" in runner.results
        assert "nll" in runner.results["heat_gp"]

    def test_run_burgers(self):
        runner = BenchmarkRunner(n_train=20, n_test=30, seed=0)
        runner.run_burgers()
        assert "burgers_gp" in runner.results
        assert "burgers_pno" in runner.results

    def test_run_all_keys(self):
        runner = BenchmarkRunner(n_train=15, n_test=20, seed=1)
        results = runner.run_all()
        assert len(results) >= 4

    def test_heat_solution(self):
        x = np.linspace(0, 1, 10)
        u = _heat_solution(x, t=0.1)
        assert u.shape == (10,)
        assert np.isfinite(u).all()
        assert abs(u[0]) < 1e-10  # boundary at x=0
        assert abs(u[-1]) < 1e-10  # boundary at x=1

    def test_summary_string(self):
        runner = BenchmarkRunner(n_train=10, n_test=15, seed=0)
        runner.run_all()
        s = runner.summary()
        assert "NLL=" in s
        assert "RMSE=" in s
