"""Uncertainty quantification metrics."""
import numpy as np
from typing import Tuple


class UncertaintyMetrics:
    """Compute NLL, CRPS, and coverage probability."""

    @staticmethod
    def nll_gaussian(y_true: np.ndarray, mean: np.ndarray, var: np.ndarray) -> float:
        """Negative log-likelihood under Gaussian."""
        var = np.maximum(var, 1e-10)
        nll = 0.5 * (np.log(2 * np.pi * var) + (y_true - mean) ** 2 / var)
        return float(nll.mean())

    @staticmethod
    def crps_gaussian(y_true: np.ndarray, mean: np.ndarray, var: np.ndarray) -> float:
        """Continuous Ranked Probability Score for Gaussian predictive distribution."""
        from scipy.stats import norm as scipy_norm
        std = np.sqrt(np.maximum(var, 1e-10))
        z = (y_true - mean) / std
        crps = std * (z * (2 * scipy_norm.cdf(z) - 1) + 2 * scipy_norm.pdf(z) - 1.0 / np.sqrt(np.pi))
        return float(crps.mean())

    @staticmethod
    def coverage_probability(y_true: np.ndarray, mean: np.ndarray, var: np.ndarray, alpha: float = 0.9) -> float:
        """Empirical coverage of (1-alpha) central prediction interval."""
        from scipy.stats import norm as scipy_norm
        std = np.sqrt(np.maximum(var, 1e-10))
        z = scipy_norm.ppf(0.5 + alpha / 2)
        lower = mean - z * std
        upper = mean + z * std
        covered = np.mean((y_true >= lower) & (y_true <= upper))
        return float(covered)

    @staticmethod
    def calibration_data(y_true: np.ndarray, mean: np.ndarray, var: np.ndarray, n_bins: int = 10):
        """Return (expected_coverage, observed_coverage) arrays for calibration plot."""
        from scipy.stats import norm as scipy_norm
        std = np.sqrt(np.maximum(var, 1e-10))
        alphas = np.linspace(0.1, 0.99, n_bins)
        observed = []
        for a in alphas:
            z = scipy_norm.ppf(0.5 + a / 2)
            cov = np.mean(np.abs(y_true - mean) <= z * std)
            observed.append(float(cov))
        return alphas.tolist(), observed
