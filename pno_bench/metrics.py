"""Uncertainty calibration metrics."""

import numpy as np
from scipy import stats


class UncertaintyMetrics:
    """Static methods for evaluating probabilistic predictions."""

    @staticmethod
    def nll(y_true: np.ndarray, mean: np.ndarray, std: np.ndarray) -> float:
        """Negative log-likelihood under a Gaussian predictive distribution.

        NLL = 0.5 * mean[ log(2π σ²) + (y - μ)² / σ² ]
        """
        y_true = np.asarray(y_true, dtype=float).ravel()
        mean = np.asarray(mean, dtype=float).ravel()
        std = np.asarray(std, dtype=float).ravel()
        std = np.maximum(std, 1e-12)
        nll = 0.5 * np.mean(np.log(2.0 * np.pi * std ** 2) + ((y_true - mean) / std) ** 2)
        return float(nll)

    @staticmethod
    def crps(y_true: np.ndarray, mean: np.ndarray, std: np.ndarray) -> float:
        """Continuous Ranked Probability Score (analytical Gaussian formula).

        CRPS(N(μ,σ²), y) = σ * [ (y-μ)/σ * (2Φ((y-μ)/σ) - 1)
                                  + 2φ((y-μ)/σ) - 1/√π ]
        """
        y_true = np.asarray(y_true, dtype=float).ravel()
        mean = np.asarray(mean, dtype=float).ravel()
        std = np.asarray(std, dtype=float).ravel()
        std = np.maximum(std, 1e-12)

        z = (y_true - mean) / std
        phi = stats.norm.pdf(z)
        Phi = stats.norm.cdf(z)
        crps_vals = std * (z * (2.0 * Phi - 1.0) + 2.0 * phi - 1.0 / np.sqrt(np.pi))
        return float(np.mean(crps_vals))

    @staticmethod
    def coverage(
        y_true: np.ndarray,
        mean: np.ndarray,
        std: np.ndarray,
        alpha: float = 0.95,
    ) -> float:
        """Fraction of true values within the α-confidence interval.

        Uses the Gaussian quantile for the interval width.
        """
        y_true = np.asarray(y_true, dtype=float).ravel()
        mean = np.asarray(mean, dtype=float).ravel()
        std = np.asarray(std, dtype=float).ravel()
        std = np.maximum(std, 1e-12)

        z = stats.norm.ppf((1.0 + alpha) / 2.0)
        lower = mean - z * std
        upper = mean + z * std
        inside = np.logical_and(y_true >= lower, y_true <= upper)
        return float(np.mean(inside))

    @staticmethod
    def calibration_error(
        y_true: np.ndarray,
        mean: np.ndarray,
        std: np.ndarray,
        n_bins: int = 10,
    ) -> float:
        """Mean calibration error across confidence levels.

        For each confidence level p in [0.1, 0.2, …, 1.0],
        compute |expected_coverage(p) - observed_coverage(p)|
        and return the mean.
        """
        y_true = np.asarray(y_true, dtype=float).ravel()
        mean = np.asarray(mean, dtype=float).ravel()
        std = np.asarray(std, dtype=float).ravel()

        levels = np.linspace(1.0 / n_bins, 1.0, n_bins)
        errors = []
        for p in levels:
            obs = UncertaintyMetrics.coverage(y_true, mean, std, alpha=p)
            errors.append(abs(p - obs))
        return float(np.mean(errors))
