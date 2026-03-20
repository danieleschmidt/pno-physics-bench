"""Benchmark runner comparing GP baseline vs PNO on 1D PDEs."""
import numpy as np
from typing import Dict, Any
from pno_physics_bench.gp_baseline import GaussianProcessBaseline
from pno_physics_bench.pno_layer import PNOLayer
from pno_physics_bench.metrics import UncertaintyMetrics


def _heat_solution(x: np.ndarray, t: float = 0.1, n_terms: int = 20) -> np.ndarray:
    """Analytical solution to 1D heat eq u_t = u_xx, u(x,0)=sin(pi*x), x in [0,1]."""
    return np.sin(np.pi * x) * np.exp(-(np.pi**2) * t)


def _burgers_approx(x: np.ndarray, t: float = 0.1) -> np.ndarray:
    """Simple approximation to Burgers equation solution."""
    return np.sin(np.pi * x) * np.exp(-t) / (1 + t * np.abs(np.sin(np.pi * x)))


class BenchmarkRunner:
    """Compare GP baseline vs PNO on 1D heat and Burgers equations."""

    def __init__(self, n_train: int = 30, n_test: int = 100, seed: int = 0):
        self.n_train = n_train
        self.n_test = n_test
        self.rng = np.random.default_rng(seed)
        self.metrics = UncertaintyMetrics()
        self.results: Dict[str, Any] = {}

    def _make_dataset(self, fn, noise: float = 0.02):
        X_train = np.sort(self.rng.uniform(0, 1, self.n_train))
        y_train = fn(X_train) + self.rng.normal(0, noise, self.n_train)
        X_test = np.linspace(0, 1, self.n_test)
        y_test = fn(X_test)
        return X_train, y_train, X_test, y_test

    def _eval_model(self, name: str, mean: np.ndarray, var: np.ndarray, y_test: np.ndarray):
        return {
            "model": name,
            "nll": self.metrics.nll_gaussian(y_test, mean, var),
            "crps": self.metrics.crps_gaussian(y_test, mean, var),
            "coverage_90": self.metrics.coverage_probability(y_test, mean, var, alpha=0.9),
            "rmse": float(np.sqrt(np.mean((y_test - mean) ** 2))),
        }

    def run_heat(self):
        X_tr, y_tr, X_te, y_te = self._make_dataset(_heat_solution)
        # GP
        gp = GaussianProcessBaseline(length_scale=0.3)
        gp.fit(X_tr, y_tr)
        gp_mean, gp_var = gp.predict(X_te)
        # PNO
        X_feat = np.column_stack([X_te, X_te**2])
        X_tr_feat = np.column_stack([X_tr, X_tr**2])
        pno = PNOLayer(input_dim=2, hidden_dim=32, output_dim=1)
        pno.fit(X_tr_feat, y_tr[:, None], lr=0.05, epochs=300)
        pno_mean, pno_var = pno.predict(X_feat)
        pno_mean = pno_mean.ravel()
        pno_var = pno_var.ravel()
        # Calibration
        cal_exp, cal_obs = self.metrics.calibration_data(y_te, gp_mean, gp_var)
        self.results["heat_gp"] = self._eval_model("GP-heat", gp_mean, gp_var, y_te)
        self.results["heat_pno"] = self._eval_model("PNO-heat", pno_mean, pno_var, y_te)
        self.results["heat_calibration"] = {"expected": cal_exp, "observed": cal_obs}

    def run_burgers(self):
        X_tr, y_tr, X_te, y_te = self._make_dataset(_burgers_approx)
        gp = GaussianProcessBaseline(length_scale=0.3)
        gp.fit(X_tr, y_tr)
        gp_mean, gp_var = gp.predict(X_te)
        X_feat = np.column_stack([X_te, np.sin(np.pi * X_te)])
        X_tr_feat = np.column_stack([X_tr, np.sin(np.pi * X_tr)])
        pno = PNOLayer(input_dim=2, hidden_dim=32, output_dim=1)
        pno.fit(X_tr_feat, y_tr[:, None], lr=0.05, epochs=300)
        pno_mean, pno_var = pno.predict(X_feat)
        pno_mean = pno_mean.ravel()
        pno_var = pno_var.ravel()
        self.results["burgers_gp"] = self._eval_model("GP-burgers", gp_mean, gp_var, y_te)
        self.results["burgers_pno"] = self._eval_model("PNO-burgers", pno_mean, pno_var, y_te)

    def run_all(self) -> Dict[str, Any]:
        self.run_heat()
        self.run_burgers()
        return self.results

    def summary(self) -> str:
        lines = ["=== PNO Physics Bench Results ==="]
        for k, v in self.results.items():
            if k.endswith("_calibration"):
                continue
            lines.append(f"{k}: NLL={v['nll']:.4f} CRPS={v['crps']:.4f} Cov90={v['coverage_90']:.2f} RMSE={v['rmse']:.4f}")
        return "\n".join(lines)
