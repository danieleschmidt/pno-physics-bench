"""Gaussian Process Baseline for 1D PDE uncertainty quantification."""
import numpy as np


class GaussianProcessBaseline:
    """RBF kernel GP for 1D PDE regression with uncertainty."""

    def __init__(self, length_scale: float = 1.0, signal_variance: float = 1.0, noise_variance: float = 1e-3):
        self.length_scale = length_scale
        self.signal_variance = signal_variance
        self.noise_variance = noise_variance
        self.X_train = None
        self.y_train = None
        self.K_inv = None

    def _rbf_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        diff = X1[:, None] - X2[None, :]
        return self.signal_variance * np.exp(-0.5 * diff**2 / self.length_scale**2)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GaussianProcessBaseline":
        self.X_train = X.ravel()
        self.y_train = y.ravel()
        K = self._rbf_kernel(self.X_train, self.X_train)
        K += self.noise_variance * np.eye(len(self.X_train))
        self.K_inv = np.linalg.inv(K)
        return self

    def predict(self, X: np.ndarray):
        X = X.ravel()
        K_s = self._rbf_kernel(X, self.X_train)
        K_ss = self._rbf_kernel(X, X)
        mean = K_s @ self.K_inv @ self.y_train
        var = np.diag(K_ss - K_s @ self.K_inv @ K_s.T)
        var = np.maximum(var, 1e-10)
        return mean, var
