"""Uncertainty quantification metrics for probabilistic predictions."""

import numpy as np
from scipy.special import erf


class UncertaintyMetrics:
    """Collection of uncertainty quantification metrics.

    All metrics assume Gaussian predictive distributions parameterized
    by mean (mu) and standard deviation (sigma).
    """

    @staticmethod
    def nll(y: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> float:
        """Negative log-likelihood under a Gaussian predictive distribution.

        NLL = sum_i [ 0.5*log(2*pi*sigma_i^2) + (y_i - mu_i)^2 / (2*sigma_i^2) ]

        Args:
            y: True targets of shape (n,).
            mu: Predicted means of shape (n,).
            sigma: Predicted standard deviations of shape (n,).

        Returns:
            Scalar NLL value.
        """
        y = np.asarray(y, dtype=float)
        mu = np.asarray(mu, dtype=float)
        sigma = np.asarray(sigma, dtype=float)
        sigma = np.maximum(sigma, 1e-8)  # numerical stability

        return float(np.sum(
            0.5 * np.log(2 * np.pi * sigma ** 2)
            + (y - mu) ** 2 / (2 * sigma ** 2)
        ))

    @staticmethod
    def crps(y: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> float:
        """Continuous Ranked Probability Score (CRPS) for Gaussian forecasts.

        CRPS = mean_i [ sigma_i * (z_i * erf(z_i/sqrt(2))
                        + sqrt(2/pi) * exp(-z_i^2/2)
                        - |y_i - mu_i| / sigma_i) ]

        where z_i = (y_i - mu_i) / sigma_i.

        Lower is better. CRPS == 0 means perfect forecast.

        Args:
            y: True targets of shape (n,).
            mu: Predicted means of shape (n,).
            sigma: Predicted standard deviations of shape (n,).

        Returns:
            Scalar mean CRPS value.
        """
        y = np.asarray(y, dtype=float)
        mu = np.asarray(mu, dtype=float)
        sigma = np.asarray(sigma, dtype=float)
        sigma = np.maximum(sigma, 1e-8)

        z = (y - mu) / sigma
        score = sigma * (
            z * erf(z / np.sqrt(2))
            + np.sqrt(2.0 / np.pi) * np.exp(-0.5 * z ** 2)
            - np.abs(y - mu) / sigma
        )
        return float(np.mean(score))

    @staticmethod
    def coverage(
        y: np.ndarray,
        mu: np.ndarray,
        sigma: np.ndarray,
        z_score: float = 1.96,
    ) -> float:
        """Prediction interval coverage probability.

        Computes the fraction of true values that fall within
        [mu - z_score * sigma, mu + z_score * sigma].

        For z_score=1.96, the nominal coverage is 95%.

        Args:
            y: True targets of shape (n,).
            mu: Predicted means of shape (n,).
            sigma: Predicted standard deviations of shape (n,).
            z_score: Number of standard deviations for the interval (default 1.96).

        Returns:
            Scalar coverage fraction in [0, 1].
        """
        y = np.asarray(y, dtype=float)
        mu = np.asarray(mu, dtype=float)
        sigma = np.asarray(sigma, dtype=float)
        sigma = np.maximum(sigma, 1e-8)

        lower = mu - z_score * sigma
        upper = mu + z_score * sigma
        within = (y >= lower) & (y <= upper)
        return float(np.mean(within))
