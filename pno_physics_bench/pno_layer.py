"""Probabilistic Neural Operator layer with uncertainty head."""
import numpy as np


class PNOLayer:
    """Neural operator layer producing mean + log_var (Gaussian output)."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, seed: int = 42):
        rng = np.random.default_rng(seed)
        scale = np.sqrt(2.0 / input_dim)
        self.W1 = rng.standard_normal((input_dim, hidden_dim)) * scale
        self.b1 = np.zeros(hidden_dim)
        self.W_mean = rng.standard_normal((hidden_dim, output_dim)) * 0.1
        self.b_mean = np.zeros(output_dim)
        self.W_logvar = rng.standard_normal((hidden_dim, output_dim)) * 0.1
        self.b_logvar = np.full(output_dim, -2.0)  # init small variance

    def _relu(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    def forward(self, x: np.ndarray):
        """Forward pass. x shape: (batch, input_dim). Returns (mean, log_var)."""
        h = self._relu(x @ self.W1 + self.b1)
        mean = h @ self.W_mean + self.b_mean
        log_var = h @ self.W_logvar + self.b_logvar
        log_var = np.clip(log_var, -10, 5)
        return mean, log_var

    def predict(self, x: np.ndarray):
        mean, log_var = self.forward(x)
        var = np.exp(log_var)
        return mean, var

    def fit(self, X: np.ndarray, y: np.ndarray, lr: float = 0.01, epochs: int = 200):
        """Simple gradient-free (finite diff) training for demo purposes."""
        best_loss = float("inf")
        clip = 1.0  # gradient clip norm
        for epoch in range(epochs):
            mean, log_var = self.forward(X)
            var = np.exp(log_var) + 1e-6
            nll = 0.5 * (np.log(var) + (y - mean) ** 2 / var).mean()
            if nll < best_loss:
                best_loss = nll
            # Gradient step (manual backprop for NLL)
            dy_mean = -(y - mean) / var / len(X)
            dy_logvar = 0.5 * (1 - (y - mean) ** 2 / var) / len(X)
            # Clip gradients to prevent explosion
            dy_mean = np.clip(dy_mean, -clip, clip)
            dy_logvar = np.clip(dy_logvar, -clip, clip)
            h = self._relu(X @ self.W1 + self.b1)
            dW_mean = h.T @ dy_mean
            dW_logvar = h.T @ dy_logvar
            # Clip weight gradients
            dW_mean = np.clip(dW_mean, -clip, clip)
            dW_logvar = np.clip(dW_logvar, -clip, clip)
            self.W_mean -= lr * dW_mean
            self.b_mean -= lr * np.clip(dy_mean.mean(0), -clip, clip)
            self.W_logvar -= lr * dW_logvar
            self.b_logvar -= lr * np.clip(dy_logvar.mean(0), -clip, clip)
        return self
