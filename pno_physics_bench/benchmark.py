"""Benchmark runner comparing GP and PNO on 1D PDE problems."""

import numpy as np
from typing import Dict, Any, Optional

from .gp_baseline import GaussianProcessBaseline
from .pno_layer import PNOLayer
from .metrics import UncertaintyMetrics


def _solve_heat_1d(
    n_x: int = 50,
    n_t: int = 100,
    L: float = 1.0,
    T: float = 0.1,
    nu: float = 0.01,
) -> tuple:
    """Solve 1D heat equation u_t = nu * u_xx using explicit finite differences.

    IC: u(x, 0) = sin(pi * x / L)
    BC: u(0, t) = u(L, t) = 0

    Args:
        n_x: Number of spatial grid points.
        n_t: Number of time steps.
        L: Domain length.
        T: Final time.
        nu: Diffusivity coefficient.

    Returns:
        Tuple (x, u_final) of spatial grid and solution at t=T.
    """
    dx = L / (n_x + 1)
    dt = T / n_t
    # Stability: r = nu * dt / dx^2 <= 0.5
    r = nu * dt / dx ** 2

    x = np.linspace(dx, L - dx, n_x)
    u = np.sin(np.pi * x / L)

    for _ in range(n_t):
        u_new = u.copy()
        u_new[1:-1] = u[1:-1] + r * (u[2:] - 2 * u[1:-1] + u[:-2])
        u = u_new

    return x, u


def _solve_burgers_1d(
    n_x: int = 50,
    n_t: int = 200,
    L: float = 2.0 * np.pi,
    T: float = 0.5,
    nu: float = 0.1,
) -> tuple:
    """Solve 1D viscous Burgers equation u_t + u*u_x = nu*u_xx.

    IC: u(x, 0) = -sin(x)
    BC: periodic

    Uses explicit upwind scheme for the advection term and central differences
    for the diffusion term.

    Args:
        n_x: Number of spatial grid points.
        n_t: Number of time steps.
        L: Domain length (2*pi for periodic).
        T: Final time.
        nu: Viscosity coefficient.

    Returns:
        Tuple (x, u_final) of spatial grid and solution at t=T.
    """
    dx = L / n_x
    dt = T / n_t
    r = nu * dt / dx ** 2

    x = np.linspace(0, L - dx, n_x)
    u = -np.sin(x)

    for _ in range(n_t):
        # Advection: upwind scheme
        u_pos = np.maximum(u, 0.0)
        u_neg = np.minimum(u, 0.0)
        adv = (u_pos * (u - np.roll(u, 1)) + u_neg * (np.roll(u, -1) - u)) / dx

        # Diffusion: central differences (periodic)
        diff = (np.roll(u, -1) - 2 * u + np.roll(u, 1)) / dx ** 2

        u = u - dt * adv + nu * dt * diff

    return x, u


class BenchmarkRunner:
    """Runs 1D PDE benchmarks comparing GP baseline vs PNO.

    Generates (input, output) pairs from PDE solvers, then evaluates
    each model's predictive uncertainty using NLL, CRPS, and coverage.
    """

    def __init__(
        self,
        n_train: int = 30,
        n_test: int = 20,
        gp_length_scale: float = 0.3,
        gp_noise: float = 1e-3,
        pno_hidden_dim: int = 64,
        seed: int = 0,
    ):
        """Initialize benchmark runner.

        Args:
            n_train: Number of training points.
            n_test: Number of test points.
            gp_length_scale: GP kernel length scale.
            gp_noise: GP observation noise.
            pno_hidden_dim: Hidden units in PNO layer.
            seed: Random seed.
        """
        self.n_train = n_train
        self.n_test = n_test
        self.gp_length_scale = gp_length_scale
        self.gp_noise = gp_noise
        self.pno_hidden_dim = pno_hidden_dim
        self.seed = seed
        self.metrics = UncertaintyMetrics()

    def _eval_model_gp(
        self, X_train, y_train, X_test, y_test
    ) -> Dict[str, float]:
        gp = GaussianProcessBaseline(
            length_scale=self.gp_length_scale, noise=self.gp_noise
        )
        gp.fit(X_train, y_train)
        mu, var = gp.predict(X_test)
        sigma = np.sqrt(np.maximum(var, 1e-12))
        return {
            "nll": UncertaintyMetrics.nll(y_test, mu, sigma),
            "crps": UncertaintyMetrics.crps(y_test, mu, sigma),
            "coverage": UncertaintyMetrics.coverage(y_test, mu, sigma),
        }

    def _eval_model_pno(
        self, X_train, y_train, X_test, y_test
    ) -> Dict[str, float]:
        pno = PNOLayer(
            input_dim=1,
            hidden_dim=self.pno_hidden_dim,
            output_dim=1,
            seed=self.seed,
        )
        mu, log_var = pno.forward(X_test)
        sigma = np.sqrt(np.exp(np.clip(log_var, -10, 10)))
        return {
            "nll": UncertaintyMetrics.nll(y_test, mu, sigma),
            "crps": UncertaintyMetrics.crps(y_test, mu, sigma),
            "coverage": UncertaintyMetrics.coverage(y_test, mu, sigma),
        }

    def run_heat(self) -> Dict[str, Any]:
        """Run heat equation benchmark.

        Returns:
            Dict with keys 'x', 'u', 'train_idx', 'test_idx', 'gp', 'pno'.
        """
        x, u = _solve_heat_1d()
        rng = np.random.default_rng(self.seed)
        n = len(x)
        idx = rng.permutation(n)
        train_idx = idx[: self.n_train]
        test_idx = idx[self.n_train: self.n_train + self.n_test]

        X_train = x[train_idx]
        y_train = u[train_idx]
        X_test = x[test_idx]
        y_test = u[test_idx]

        return {
            "x": x,
            "u": u,
            "train_idx": train_idx,
            "test_idx": test_idx,
            "gp": self._eval_model_gp(X_train, y_train, X_test, y_test),
            "pno": self._eval_model_pno(X_train, y_train, X_test, y_test),
        }

    def run_burgers(self) -> Dict[str, Any]:
        """Run Burgers equation benchmark.

        Returns:
            Dict with keys 'x', 'u', 'train_idx', 'test_idx', 'gp', 'pno'.
        """
        x, u = _solve_burgers_1d()
        rng = np.random.default_rng(self.seed + 1)
        n = len(x)
        idx = rng.permutation(n)
        train_idx = idx[: self.n_train]
        test_idx = idx[self.n_train: self.n_train + self.n_test]

        X_train = x[train_idx]
        y_train = u[train_idx]
        X_test = x[test_idx]
        y_test = u[test_idx]

        return {
            "x": x,
            "u": u,
            "train_idx": train_idx,
            "test_idx": test_idx,
            "gp": self._eval_model_gp(X_train, y_train, X_test, y_test),
            "pno": self._eval_model_pno(X_train, y_train, X_test, y_test),
        }

    def run_all(self) -> Dict[str, Any]:
        """Run all benchmarks and return consolidated results.

        Returns:
            Dict with 'heat' and 'burgers' sub-dicts, each containing
            GP and PNO metric results.
        """
        return {
            "heat": self.run_heat(),
            "burgers": self.run_burgers(),
        }
