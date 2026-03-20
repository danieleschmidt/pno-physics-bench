"""Tests for pno-physics-bench: 12+ tests covering all modules."""

import json
import os
import tempfile

import numpy as np
import pytest

from pno_physics_bench import (
    GaussianProcessBaseline,
    PNOLayer,
    UncertaintyMetrics,
    BenchmarkRunner,
    calibration_export,
)


# ---------------------------------------------------------------------------
# GaussianProcessBaseline tests
# ---------------------------------------------------------------------------

class TestGaussianProcessBaseline:

    def test_init_defaults(self):
        gp = GaussianProcessBaseline()
        assert gp.length_scale == 1.0
        assert gp.noise == 1e-3

    def test_init_custom(self):
        gp = GaussianProcessBaseline(length_scale=0.5, noise=0.01)
        assert gp.length_scale == 0.5
        assert gp.noise == 0.01

    def test_rbf_kernel_shape(self):
        gp = GaussianProcessBaseline()
        X1 = np.array([[0.0], [1.0], [2.0]])
        X2 = np.array([[0.5], [1.5]])
        K = gp._rbf_kernel(X1, X2)
        assert K.shape == (3, 2)

    def test_rbf_kernel_symmetry(self):
        gp = GaussianProcessBaseline(length_scale=0.5)
        X = np.linspace(0, 1, 5)[:, np.newaxis]
        K = gp._rbf_kernel(X, X)
        np.testing.assert_allclose(K, K.T, atol=1e-12)

    def test_rbf_kernel_diagonal_is_one(self):
        gp = GaussianProcessBaseline()
        X = np.array([[0.0], [1.0], [2.0]])
        K = gp._rbf_kernel(X, X)
        np.testing.assert_allclose(np.diag(K), np.ones(3), atol=1e-12)

    def test_fit_and_predict_shape(self):
        gp = GaussianProcessBaseline()
        X_train = np.linspace(0, 1, 10)
        y_train = np.sin(X_train)
        gp.fit(X_train, y_train)

        X_test = np.linspace(0.1, 0.9, 5)
        mean, var = gp.predict(X_test)
        assert mean.shape == (5,)
        assert var.shape == (5,)

    def test_predict_variance_non_negative(self):
        gp = GaussianProcessBaseline(length_scale=0.3)
        X_train = np.linspace(0, 1, 15)
        y_train = np.cos(np.pi * X_train)
        gp.fit(X_train, y_train)

        X_test = np.linspace(0, 1, 20)
        _, var = gp.predict(X_test)
        assert np.all(var >= 0.0)

    def test_predict_low_variance_near_training(self):
        gp = GaussianProcessBaseline(length_scale=0.3, noise=1e-6)
        X_train = np.array([0.0, 0.5, 1.0])
        y_train = np.array([0.0, 1.0, 0.0])
        gp.fit(X_train, y_train)

        # Variance at training points should be near zero
        mean_train, var_train = gp.predict(X_train)
        assert np.all(var_train < 1e-3)

    def test_predict_without_fit_raises(self):
        gp = GaussianProcessBaseline()
        with pytest.raises(RuntimeError):
            gp.predict(np.array([0.5]))


# ---------------------------------------------------------------------------
# PNOLayer tests
# ---------------------------------------------------------------------------

class TestPNOLayer:

    def test_init(self):
        pno = PNOLayer(input_dim=1, hidden_dim=32, output_dim=1)
        assert pno.W1.shape == (32, 1)
        assert pno.W2.shape == (2, 32)

    def test_forward_output_shape(self):
        pno = PNOLayer(input_dim=1, hidden_dim=64)
        x = np.linspace(0, 1, 10)
        mean, log_var = pno.forward(x)
        assert mean.shape == (10,)
        assert log_var.shape == (10,)

    def test_call_alias(self):
        pno = PNOLayer()
        x = np.linspace(0, 1, 5)
        mean1, lv1 = pno(x)
        mean2, lv2 = pno.forward(x)
        np.testing.assert_array_equal(mean1, mean2)
        np.testing.assert_array_equal(lv1, lv2)

    def test_deterministic_with_same_seed(self):
        x = np.linspace(0, 1, 8)
        pno1 = PNOLayer(seed=123)
        pno2 = PNOLayer(seed=123)
        m1, lv1 = pno1(x)
        m2, lv2 = pno2(x)
        np.testing.assert_array_equal(m1, m2)
        np.testing.assert_array_equal(lv1, lv2)


# ---------------------------------------------------------------------------
# UncertaintyMetrics tests
# ---------------------------------------------------------------------------

class TestUncertaintyMetrics:

    def test_nll_perfect_prediction_is_finite(self):
        y = np.array([0.0, 1.0, 2.0])
        mu = y.copy()
        sigma = np.ones(3)
        nll = UncertaintyMetrics.nll(y, mu, sigma)
        assert np.isfinite(nll)

    def test_nll_decreases_with_better_fit(self):
        y = np.array([1.0, 2.0, 3.0])
        sigma = np.ones(3)
        nll_good = UncertaintyMetrics.nll(y, y, sigma)
        nll_bad = UncertaintyMetrics.nll(y, y + 10.0, sigma)
        assert nll_good < nll_bad

    def test_crps_non_negative(self):
        rng = np.random.default_rng(0)
        y = rng.normal(size=50)
        mu = rng.normal(size=50)
        sigma = np.abs(rng.normal(size=50)) + 0.1
        crps = UncertaintyMetrics.crps(y, mu, sigma)
        assert crps >= 0.0

    def test_crps_perfect_is_zero(self):
        """For a perfect point prediction (sigma→0), CRPS → 0."""
        y = np.array([1.0, 2.0, 3.0])
        mu = y.copy()
        sigma = np.full(3, 1e-6)
        crps = UncertaintyMetrics.crps(y, mu, sigma)
        assert crps < 1e-3

    def test_coverage_all_inside(self):
        y = np.array([0.0, 1.0, 2.0])
        mu = y.copy()
        sigma = np.full(3, 100.0)  # very wide intervals
        cov = UncertaintyMetrics.coverage(y, mu, sigma)
        assert cov == 1.0

    def test_coverage_all_outside(self):
        y = np.array([0.0, 1.0, 2.0])
        mu = y + 1000.0  # far off
        sigma = np.ones(3)
        cov = UncertaintyMetrics.coverage(y, mu, sigma)
        assert cov == 0.0

    def test_coverage_nominal_95(self):
        rng = np.random.default_rng(42)
        mu = np.zeros(10000)
        sigma = np.ones(10000)
        y = rng.normal(mu, sigma)
        cov = UncertaintyMetrics.coverage(y, mu, sigma)
        assert abs(cov - 0.95) < 0.02


# ---------------------------------------------------------------------------
# BenchmarkRunner tests
# ---------------------------------------------------------------------------

class TestBenchmarkRunner:

    def test_run_heat_returns_expected_keys(self):
        runner = BenchmarkRunner(n_train=20, n_test=10)
        result = runner.run_heat()
        assert "x" in result
        assert "u" in result
        assert "gp" in result
        assert "pno" in result
        for model in ("gp", "pno"):
            for metric in ("nll", "crps", "coverage"):
                assert metric in result[model]

    def test_run_burgers_returns_expected_keys(self):
        runner = BenchmarkRunner(n_train=20, n_test=10)
        result = runner.run_burgers()
        assert "gp" in result
        assert "pno" in result

    def test_run_all_has_both_benchmarks(self):
        runner = BenchmarkRunner(n_train=15, n_test=10)
        results = runner.run_all()
        assert "heat" in results
        assert "burgers" in results

    def test_gp_coverage_in_range(self):
        runner = BenchmarkRunner(n_train=20, n_test=10)
        result = runner.run_heat()
        cov = result["gp"]["coverage"]
        assert 0.0 <= cov <= 1.0

    def test_metrics_are_finite(self):
        runner = BenchmarkRunner(n_train=20, n_test=10)
        results = runner.run_all()
        for bench in ("heat", "burgers"):
            for model in ("gp", "pno"):
                for metric in ("nll", "crps", "coverage"):
                    val = results[bench][model][metric]
                    assert np.isfinite(val), (
                        f"{bench}/{model}/{metric} = {val} is not finite"
                    )


# ---------------------------------------------------------------------------
# Export tests
# ---------------------------------------------------------------------------

class TestCalibrationExport:

    def _make_results(self):
        return {
            "heat": {
                "gp": {"nll": 1.23, "crps": 0.45, "coverage": 0.90},
                "pno": {"nll": 2.34, "crps": 0.67, "coverage": 0.80},
            },
            "burgers": {
                "gp": {"nll": 1.11, "crps": 0.33, "coverage": 0.95},
                "pno": {"nll": 3.21, "crps": 0.88, "coverage": 0.70},
            },
        }

    def test_export_creates_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "calibration")
            calibration_export(self._make_results(), out_path)
            assert os.path.exists(out_path + ".json")

    def test_export_creates_csv(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "calibration")
            calibration_export(self._make_results(), out_path)
            assert os.path.exists(out_path + ".csv")

    def test_json_content_valid(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "calibration")
            calibration_export(self._make_results(), out_path)
            with open(out_path + ".json") as f:
                data = json.load(f)
            assert "heat" in data
            assert "burgers" in data

    def test_csv_row_count(self):
        import csv as csv_mod

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "calibration")
            calibration_export(self._make_results(), out_path)
            with open(out_path + ".csv") as f:
                reader = csv_mod.DictReader(f)
                rows = list(reader)
            # 2 benchmarks * 2 models = 4 rows
            assert len(rows) == 4

    def test_export_creates_parent_dirs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "sub", "dir", "calibration")
            calibration_export(self._make_results(), out_path)
            assert os.path.exists(out_path + ".json")
