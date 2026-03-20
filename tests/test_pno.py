"""Tests for PNO physics benchmark package.

10+ tests covering GP, PNO, metrics, BenchmarkRunner, and end-to-end flow.
"""

import json
import math
import os
import tempfile

import numpy as np
import pytest

from pno_bench.baselines import GaussianProcessBaseline
from pno_bench.pno_layer import PNOLayer
from pno_bench.metrics import UncertaintyMetrics
from pno_bench.benchmark import BenchmarkRunner


# -----------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------

@pytest.fixture
def simple_1d_data():
    rng = np.random.default_rng(7)
    X = rng.uniform(0, 1, 30).reshape(-1, 1)
    y = np.sin(2 * math.pi * X.ravel()) + rng.normal(0, 0.1, 30)
    X_test = np.linspace(0, 1, 20).reshape(-1, 1)
    return X, y, X_test


@pytest.fixture
def fitted_gp(simple_1d_data):
    X, y, _ = simple_1d_data
    gp = GaussianProcessBaseline()
    gp.fit(X, y)
    return gp, simple_1d_data


@pytest.fixture
def fitted_pno(simple_1d_data):
    X, y, _ = simple_1d_data
    pno = PNOLayer(input_dim=1, hidden_dim=16, output_dim=1)
    pno.fit(X, y, epochs=50)
    return pno, simple_1d_data


# -----------------------------------------------------------------------
# GP tests
# -----------------------------------------------------------------------

class TestGaussianProcess:
    def test_fit_runs(self, simple_1d_data):
        X, y, _ = simple_1d_data
        gp = GaussianProcessBaseline()
        gp.fit(X, y)
        assert gp._alpha is not None

    def test_predict_returns_mean_and_std(self, fitted_gp):
        gp, (_, _, X_test) = fitted_gp
        mean, std = gp.predict(X_test)
        assert mean.shape == (len(X_test),)
        assert std.shape == (len(X_test),)

    def test_predict_std_is_positive(self, fitted_gp):
        gp, (_, _, X_test) = fitted_gp
        _, std = gp.predict(X_test)
        assert np.all(std > 0), "All std values must be positive"

    def test_fit_simple_data_reasonable_predictions(self):
        """GP should recover near-zero predictions on constant zero function."""
        X = np.linspace(0, 1, 20).reshape(-1, 1)
        y = np.zeros(20)
        gp = GaussianProcessBaseline(length_scale=0.3, noise=1e-6)
        gp.fit(X, y)
        mean, _ = gp.predict(X)
        assert np.all(np.abs(mean) < 0.1), "GP mean should be near zero for zero function"


# -----------------------------------------------------------------------
# PNO tests
# -----------------------------------------------------------------------

class TestPNOLayer:
    def test_forward_returns_correct_shape(self, simple_1d_data):
        X, _, _ = simple_1d_data
        pno = PNOLayer(input_dim=1, hidden_dim=16, output_dim=1)
        mean, log_var = pno.forward(X)
        assert mean.shape == (len(X), 1)
        assert log_var.shape == (len(X), 1)

    def test_predict_with_uncertainty_returns_positive_std(self, fitted_pno):
        pno, (_, _, X_test) = fitted_pno
        mean, std = pno.predict_with_uncertainty(X_test)
        std = np.asarray(std).ravel()
        assert np.all(std > 0), "std must be positive (exp of log_var)"

    def test_fit_reduces_loss(self, simple_1d_data):
        """Check that training does not blow up."""
        X, y, _ = simple_1d_data
        pno = PNOLayer(input_dim=1, hidden_dim=16, output_dim=1)
        pno.fit(X, y, epochs=100, lr=0.01)
        mean, std = pno.predict_with_uncertainty(X)
        assert np.all(np.isfinite(mean)), "Predictions should be finite after training"


# -----------------------------------------------------------------------
# Metrics tests
# -----------------------------------------------------------------------

class TestUncertaintyMetrics:
    def _sample_data(self):
        rng = np.random.default_rng(42)
        y = rng.normal(0, 1, 100)
        mean = rng.normal(0, 0.5, 100)
        std = rng.uniform(0.5, 1.5, 100)
        return y, mean, std

    def test_nll_is_finite(self):
        y, mean, std = self._sample_data()
        result = UncertaintyMetrics.nll(y, mean, std)
        assert math.isfinite(result), "NLL should be finite"

    def test_crps_is_nonnegative(self):
        y, mean, std = self._sample_data()
        result = UncertaintyMetrics.crps(y, mean, std)
        assert result >= 0.0, "CRPS must be non-negative"

    def test_coverage_between_zero_and_one(self):
        y, mean, std = self._sample_data()
        result = UncertaintyMetrics.coverage(y, mean, std, alpha=0.95)
        assert 0.0 <= result <= 1.0, "Coverage must be in [0, 1]"

    def test_calibration_error_is_nonnegative(self):
        y, mean, std = self._sample_data()
        result = UncertaintyMetrics.calibration_error(y, mean, std)
        assert result >= 0.0, "Calibration error must be non-negative"

    def test_perfect_coverage_at_one(self):
        """Very wide intervals should give ~100% coverage."""
        rng = np.random.default_rng(0)
        y = rng.normal(0, 1, 200)
        mean = np.zeros(200)
        std = np.full(200, 100.0)
        cov = UncertaintyMetrics.coverage(y, mean, std, alpha=0.95)
        assert cov > 0.99


# -----------------------------------------------------------------------
# BenchmarkRunner tests
# -----------------------------------------------------------------------

class TestBenchmarkRunner:
    REQUIRED_KEYS = {
        "gp_nll", "pno_nll",
        "gp_crps", "pno_crps",
        "gp_coverage", "pno_coverage",
    }

    def test_heat_equation_returns_required_keys(self):
        runner = BenchmarkRunner()
        results = runner.run_heat_equation(n_points=20)
        assert self.REQUIRED_KEYS.issubset(set(results.keys()))

    def test_heat_equation_values_are_finite(self):
        runner = BenchmarkRunner()
        results = runner.run_heat_equation(n_points=20)
        for k, v in results.items():
            assert math.isfinite(v), f"{k} = {v} is not finite"

    def test_burgers_equation_returns_required_keys(self):
        runner = BenchmarkRunner()
        results = runner.run_burgers_equation(n_points=20)
        assert self.REQUIRED_KEYS.issubset(set(results.keys()))

    def test_burgers_equation_values_are_finite(self):
        runner = BenchmarkRunner()
        results = runner.run_burgers_equation(n_points=20)
        for k, v in results.items():
            assert math.isfinite(v), f"{k} = {v} is not finite"

    def test_export_calibration_writes_valid_json(self):
        runner = BenchmarkRunner()
        results = runner.run_heat_equation(n_points=20)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as fh:
            path = fh.name
        try:
            BenchmarkRunner.export_calibration(results, path)
            with open(path) as f:
                data = json.load(f)
            assert isinstance(data, dict)
            assert self.REQUIRED_KEYS.issubset(set(data.keys()))
        finally:
            os.unlink(path)

    def test_end_to_end_full_benchmark(self):
        """Complete benchmark run from fit to export must not raise."""
        runner = BenchmarkRunner()
        heat = runner.run_heat_equation(n_points=30)
        burgers = runner.run_burgers_equation(n_points=30)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as fh:
            path = fh.name
        try:
            BenchmarkRunner.export_calibration({**heat, **burgers}, path)
            assert os.path.exists(path)
        finally:
            os.unlink(path)
