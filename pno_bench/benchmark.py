"""Benchmark runner for PDE uncertainty estimation."""

import json
import math
import numpy as np

from .baselines import GaussianProcessBaseline
from .pno_layer import PNOLayer
from .metrics import UncertaintyMetrics


class BenchmarkRunner:
    """Run GP and PNO benchmarks on canonical PDE problems."""

    def __init__(self):
        pass

    # ------------------------------------------------------------------
    # Heat equation  u_t = α u_xx
    # Analytical: u(x, t) = exp(-α π² t) sin(π x)
    # ------------------------------------------------------------------

    def run_heat_equation(self, n_points: int = 50) -> dict:
        """Benchmark on the 1D heat equation.

        Returns
        -------
        dict with keys: gp_nll, pno_nll, gp_crps, pno_crps,
                        gp_coverage, pno_coverage
        """
        alpha = 0.1
        t = 0.1

        rng = np.random.default_rng(0)
        x_train = rng.uniform(0, 1, n_points)
        u_train = np.exp(-alpha * math.pi ** 2 * t) * np.sin(math.pi * x_train)
        u_train += rng.normal(0, 0.01, n_points)  # small noise

        x_test = np.linspace(0, 1, n_points)
        u_test = np.exp(-alpha * math.pi ** 2 * t) * np.sin(math.pi * x_test)

        # GP
        gp = GaussianProcessBaseline(length_scale=0.3, sigma=1.0, noise=1e-3)
        gp.fit(x_train, u_train)
        gp_mean, gp_std = gp.predict(x_test)

        # PNO
        pno = PNOLayer(input_dim=1, hidden_dim=32, output_dim=1)
        pno.fit(x_train.reshape(-1, 1), u_train, epochs=200, lr=0.01)
        pno_mean, pno_std = pno.predict_with_uncertainty(x_test.reshape(-1, 1))
        pno_mean = pno_mean.ravel()
        pno_std = pno_std.ravel()

        m = UncertaintyMetrics
        return {
            "gp_nll": m.nll(u_test, gp_mean, gp_std),
            "pno_nll": m.nll(u_test, pno_mean, pno_std),
            "gp_crps": m.crps(u_test, gp_mean, gp_std),
            "pno_crps": m.crps(u_test, pno_mean, pno_std),
            "gp_coverage": m.coverage(u_test, gp_mean, gp_std),
            "pno_coverage": m.coverage(u_test, pno_mean, pno_std),
        }

    # ------------------------------------------------------------------
    # Burgers equation  u_t + u u_x = ν u_xx
    # Cole-Hopf analytical solution (simplified initial condition)
    # ------------------------------------------------------------------

    def run_burgers_equation(self, n_points: int = 50) -> dict:
        """Benchmark on the viscous Burgers equation.

        Returns
        -------
        dict with keys: gp_nll, pno_nll, gp_crps, pno_crps,
                        gp_coverage, pno_coverage
        """
        nu = 0.1
        t = 0.1

        rng = np.random.default_rng(1)
        x_vals = np.linspace(-1.0, 1.0, n_points)

        # Cole-Hopf: u(x,t) = -2ν φ_x / φ
        # with φ(x,t) = 1 + exp(-x/(2ν) - t/(4ν))
        # This is one of the exact solutions to Burgers.
        def burgers_exact(x, t, nu=0.1):
            denom = 4.0 * nu
            phi = 1.0 + np.exp(-x / (2.0 * nu) - t / denom)
            dphi_dx = -(1.0 / (2.0 * nu)) * np.exp(-x / (2.0 * nu) - t / denom)
            return -2.0 * nu * dphi_dx / phi

        x_train = rng.uniform(-1, 1, n_points)
        u_train = burgers_exact(x_train, t, nu)
        u_train += rng.normal(0, 0.01, n_points)

        u_test = burgers_exact(x_vals, t, nu)

        # GP
        gp = GaussianProcessBaseline(length_scale=0.3, sigma=1.0, noise=1e-3)
        gp.fit(x_train, u_train)
        gp_mean, gp_std = gp.predict(x_vals)

        # PNO
        pno = PNOLayer(input_dim=1, hidden_dim=32, output_dim=1)
        pno.fit(x_train.reshape(-1, 1), u_train, epochs=200, lr=0.01)
        pno_mean, pno_std = pno.predict_with_uncertainty(x_vals.reshape(-1, 1))
        pno_mean = pno_mean.ravel()
        pno_std = pno_std.ravel()

        m = UncertaintyMetrics
        return {
            "gp_nll": m.nll(u_test, gp_mean, gp_std),
            "pno_nll": m.nll(u_test, pno_mean, pno_std),
            "gp_crps": m.crps(u_test, gp_mean, gp_std),
            "pno_crps": m.crps(u_test, pno_mean, pno_std),
            "gp_coverage": m.coverage(u_test, gp_mean, gp_std),
            "pno_coverage": m.coverage(u_test, pno_mean, pno_std),
        }

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    @staticmethod
    def export_calibration(results: dict, path: str) -> None:
        """Write benchmark results to a JSON file.

        Parameters
        ----------
        results : dict   (from run_heat_equation / run_burgers_equation)
        path    : str    destination file path
        """
        # Convert numpy scalars to plain Python floats
        serializable = {k: float(v) for k, v in results.items()}
        with open(path, "w") as fh:
            json.dump(serializable, fh, indent=2)
