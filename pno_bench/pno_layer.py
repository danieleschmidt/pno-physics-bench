"""Physics-informed Neural Operator uncertainty head (numpy only)."""

import numpy as np


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def _relu_grad(x: np.ndarray) -> np.ndarray:
    return (x > 0).astype(float)


class PNOLayer:
    """Two-head MLP: shared trunk → mean head + log_var head.

    Architecture:
        input (input_dim)
        → hidden layer (hidden_dim, ReLU)
        → mean head  (output_dim, linear)
        → log_var head (output_dim, linear)

    Parameters
    ----------
    input_dim  : int
    hidden_dim : int
    output_dim : int
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        scale = 0.1
        rng = np.random.default_rng(42)

        # Shared trunk
        self.W1 = rng.normal(0, scale, (hidden_dim, input_dim))
        self.b1 = np.zeros(hidden_dim)

        # Mean head
        self.W_mean = rng.normal(0, scale, (output_dim, hidden_dim))
        self.b_mean = np.zeros(output_dim)

        # Log-var head
        self.W_lv = rng.normal(0, scale, (output_dim, hidden_dim))
        self.b_lv = np.zeros(output_dim)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, x: np.ndarray):
        """Forward pass.

        Parameters
        ----------
        x : (n, input_dim) or (input_dim,)

        Returns
        -------
        mean    : (n, output_dim)
        log_var : (n, output_dim)
        """
        x = np.atleast_2d(x)
        h = _relu(x @ self.W1.T + self.b1)          # (n, hidden_dim)
        mean = h @ self.W_mean.T + self.b_mean       # (n, output_dim)
        log_var = h @ self.W_lv.T + self.b_lv        # (n, output_dim)
        return mean, log_var

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 100, lr: float = 0.01):
        """Train with heteroscedastic NLL via gradient descent.

        Parameters
        ----------
        X : (n, input_dim)
        y : (n, output_dim) or (n,)
        """
        X = np.atleast_2d(X)
        y = np.atleast_2d(y) if y.ndim > 1 else y.reshape(-1, 1)
        n = X.shape[0]

        for _ in range(epochs):
            # ---- forward ----
            h_pre = X @ self.W1.T + self.b1          # (n, hidden)
            h = _relu(h_pre)
            mean = h @ self.W_mean.T + self.b_mean   # (n, out)
            log_var = h @ self.W_lv.T + self.b_lv    # (n, out)
            var = np.exp(log_var) + 1e-6

            # ---- NLL loss ----
            # L = 0.5 * [ log(var) + (y - mean)^2 / var ]
            diff = y - mean
            # d L / d mean   = -(y - mean) / var
            # d L / d log_var = 0.5 * (1 - (y-mean)^2 / var)
            d_mean = -diff / var / n
            d_lv = 0.5 * (1.0 - diff ** 2 / var) / n

            # ---- gradients mean head ----
            dW_mean = d_mean.T @ h
            db_mean = d_mean.sum(axis=0)

            # ---- gradients log_var head ----
            dW_lv = d_lv.T @ h
            db_lv = d_lv.sum(axis=0)

            # ---- gradients shared trunk ----
            d_h = d_mean @ self.W_mean + d_lv @ self.W_lv   # (n, hidden)
            d_h *= _relu_grad(h_pre)
            dW1 = d_h.T @ X
            db1 = d_h.sum(axis=0)

            # ---- gradient descent update ----
            self.W_mean -= lr * dW_mean
            self.b_mean -= lr * db_mean
            self.W_lv -= lr * dW_lv
            self.b_lv -= lr * db_lv
            self.W1 -= lr * dW1
            self.b1 -= lr * db1

        return self

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict_with_uncertainty(self, x: np.ndarray):
        """Return (mean, std).

        std = exp(0.5 * log_var)
        """
        mean, log_var = self.forward(x)
        std = np.exp(0.5 * log_var)
        return mean.squeeze(), std.squeeze()
