"""Gaussian Process baseline for PDE uncertainty benchmarking."""

import numpy as np


class GaussianProcessBaseline:
    """GP regression with RBF kernel and Cholesky-based inference.

    Kernel: k(x, x') = sigma^2 * exp(-||x - x'||^2 / (2 * l^2))

    Parameters
    ----------
    length_scale : float
        RBF kernel length scale (l).
    sigma : float
        Signal standard deviation.
    noise : float
        Observation noise variance added to diagonal.
    """

    def __init__(self, length_scale: float = 1.0, sigma: float = 1.0, noise: float = 1e-3):
        self.length_scale = length_scale
        self.sigma = sigma
        self.noise = noise
        self._X_train = None
        self._alpha = None
        self._L = None

    # ------------------------------------------------------------------
    # Kernel
    # ------------------------------------------------------------------

    def _rbf_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Compute RBF kernel matrix between X1 and X2."""
        X1 = np.atleast_2d(X1)
        X2 = np.atleast_2d(X2)
        # Squared Euclidean distances
        diff = X1[:, None, :] - X2[None, :, :]  # (n1, n2, d)
        sq_dist = np.sum(diff ** 2, axis=-1)     # (n1, n2)
        return self.sigma ** 2 * np.exp(-sq_dist / (2.0 * self.length_scale ** 2))

    # ------------------------------------------------------------------
    # Fit / predict
    # ------------------------------------------------------------------

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "GaussianProcessBaseline":
        """Fit the GP to training data.

        Parameters
        ----------
        X_train : (n, d) or (n,) array
        y_train : (n,) array
        """
        X_train = np.atleast_2d(X_train) if X_train.ndim > 1 else X_train.reshape(-1, 1)
        y_train = np.asarray(y_train, dtype=float).ravel()

        self._X_train = X_train
        K = self._rbf_kernel(X_train, X_train)
        K += self.noise * np.eye(len(X_train))

        # Cholesky decomposition for numerical stability
        try:
            self._L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            # Jitter if not PD
            K += 1e-6 * np.eye(len(X_train))
            self._L = np.linalg.cholesky(K)

        # alpha = K^{-1} y  via Cholesky
        self._alpha = np.linalg.solve(self._L.T, np.linalg.solve(self._L, y_train))
        return self

    def predict(self, X_test: np.ndarray):
        """Predict mean and std at test points.

        Returns
        -------
        mean : (m,) array
        std  : (m,) array  (always positive)
        """
        if self._X_train is None:
            raise RuntimeError("Call fit() before predict().")

        X_test = np.atleast_2d(X_test) if X_test.ndim > 1 else X_test.reshape(-1, 1)

        K_s = self._rbf_kernel(self._X_train, X_test)   # (n, m)
        K_ss = self._rbf_kernel(X_test, X_test)          # (m, m)

        mean = K_s.T @ self._alpha

        # v = L^{-1} K_s
        v = np.linalg.solve(self._L, K_s)
        cov = K_ss - v.T @ v
        var = np.diag(cov)
        var = np.maximum(var, 0.0)  # clip numerical negatives
        std = np.sqrt(var + 1e-12)

        return mean, std
