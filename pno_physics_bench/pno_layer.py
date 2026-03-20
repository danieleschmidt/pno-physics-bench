"""Physics-informed Neural Operator (PNO) layer for uncertainty quantification."""

import numpy as np


class PNOLayer:
    """Simple feedforward neural operator layer with uncertainty output.

    Outputs [mean, log_var] for each output point, enabling probabilistic
    predictions via the reparameterization of a Gaussian.
    """

    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 64,
        output_dim: int = 1,
        seed: int = 42,
    ):
        """Initialize PNO layer with random weights.

        Args:
            input_dim: Dimension of each input point.
            hidden_dim: Number of hidden units.
            output_dim: Number of output points.
            seed: Random seed for reproducibility.
        """
        rng = np.random.default_rng(seed)
        scale = np.sqrt(2.0 / input_dim)

        # Two-layer MLP: input_dim -> hidden_dim -> 2 (mean + log_var)
        self.W1 = rng.normal(0, scale, (hidden_dim, input_dim))
        self.b1 = np.zeros(hidden_dim)
        self.W2 = rng.normal(0, np.sqrt(2.0 / hidden_dim), (2, hidden_dim))
        self.b2 = np.zeros(2)

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

    def _relu(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, x)

    def forward(self, x: np.ndarray):
        """Forward pass through the PNO layer.

        Args:
            x: Input array of shape (n,) or (n, input_dim).

        Returns:
            Tuple of (mean, log_var) each of shape (n,).
        """
        x = np.atleast_1d(x)
        if x.ndim == 1:
            x = x[:, np.newaxis]

        # Hidden layer
        h = self._relu(x @ self.W1.T + self.b1)  # (n, hidden_dim)
        # Output layer: [mean, log_var]
        out = h @ self.W2.T + self.b2  # (n, 2)

        mean = out[:, 0]      # (n,)
        log_var = out[:, 1]   # (n,)

        return mean, log_var

    def __call__(self, x: np.ndarray):
        """Alias for forward()."""
        return self.forward(x)
