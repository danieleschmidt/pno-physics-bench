"""Gaussian Process Baseline with RBF kernel for uncertainty quantification."""

import numpy as np


class GaussianProcessBaseline:
    """Gaussian Process regressor with RBF (squared exponential) kernel.

    The kernel is k(x1, x2) = exp(-||x1-x2||^2 / (2*l^2)).
    Fitting solves (K + noise*I) @ alpha = y.
    Prediction returns posterior mean and variance.
    """

    def __init__(self, length_scale: float = 1.0, noise: float = 1e-3):
        """Initialize GP baseline.

        Args:
            length_scale: RBF kernel length scale (l).
            noise: Observation noise variance added to diagonal.
        """
        self.length_scale = length_scale
        self.noise = noise
        self._X_train = None
        self._alpha = None
        self._K_inv = None

    def _rbf_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Compute RBF kernel matrix between X1 and X2.

        Args:
            X1: Array of shape (n, d) or (n,).
            X2: Array of shape (m, d) or (m,).

        Returns:
            Kernel matrix of shape (n, m).
        """
        X1 = np.atleast_2d(X1)
        X2 = np.atleast_2d(X2)
        # Squared Euclidean distances: ||x1 - x2||^2
        diff = X1[:, np.newaxis, :] - X2[np.newaxis, :, :]  # (n, m, d)
        sq_dist = np.sum(diff ** 2, axis=-1)  # (n, m)
        return np.exp(-sq_dist / (2.0 * self.length_scale ** 2))

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GaussianProcessBaseline":
        """Fit the GP to training data.

        Solves (K + noise*I) @ alpha = y where K = k(X, X).

        Args:
            X: Training inputs of shape (n,) or (n, d).
            y: Training targets of shape (n,).

        Returns:
            self
        """
        X = np.atleast_1d(X)
        y = np.atleast_1d(y)
        if X.ndim == 1:
            X = X[:, np.newaxis]

        self._X_train = X.copy()
        n = X.shape[0]

        K = self._rbf_kernel(X, X)
        K_noise = K + self.noise * np.eye(n)

        # Solve for alpha: (K + noise*I) alpha = y
        self._alpha = np.linalg.solve(K_noise, y)
        self._K_inv = np.linalg.inv(K_noise)
        return self

    def predict(self, X: np.ndarray):
        """Predict mean and variance at test points.

        Args:
            X: Test inputs of shape (m,) or (m, d).

        Returns:
            Tuple of (mean, variance) each of shape (m,).
        """
        if self._X_train is None:
            raise RuntimeError("Call fit() before predict().")

        X = np.atleast_1d(X)
        if X.ndim == 1:
            X = X[:, np.newaxis]

        K_star = self._rbf_kernel(X, self._X_train)  # (m, n)
        K_star_star = self._rbf_kernel(X, X)  # (m, m)

        mean = K_star @ self._alpha  # (m,)
        variance = np.diag(K_star_star - K_star @ self._K_inv @ K_star.T)  # (m,)
        variance = np.maximum(variance, 0.0)  # clip numerical negatives

        return mean, variance
