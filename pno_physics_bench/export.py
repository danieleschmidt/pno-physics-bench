"""Export calibration data from benchmark results to JSON and CSV."""

import csv
import json
import os
from pathlib import Path
from typing import Any, Dict


def calibration_export(results: Dict[str, Any], path: str) -> None:
    """Export benchmark calibration data to JSON and CSV.

    Writes two files:
      - <path>.json  — full nested results dict
      - <path>.csv   — flat table: benchmark, model, nll, crps, coverage

    Args:
        results: Dict returned by BenchmarkRunner.run_all() with
                 keys like 'heat' and 'burgers', each containing
                 'gp' and 'pno' sub-dicts with metrics.
        path: Output path prefix (without extension). Parent directory
              is created automatically if it does not exist.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # --- JSON export ---
    json_path = path.with_suffix(".json")
    serializable = _make_serializable(results)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)

    # --- CSV export ---
    csv_path = path.with_suffix(".csv")
    rows = []
    for benchmark_name, bench_results in results.items():
        for model_name in ("gp", "pno"):
            if model_name not in bench_results:
                continue
            metrics = bench_results[model_name]
            rows.append(
                {
                    "benchmark": benchmark_name,
                    "model": model_name,
                    "nll": metrics.get("nll", float("nan")),
                    "crps": metrics.get("crps", float("nan")),
                    "coverage": metrics.get("coverage", float("nan")),
                }
            )

    fieldnames = ["benchmark", "model", "nll", "crps", "coverage"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _make_serializable(obj: Any) -> Any:
    """Recursively convert numpy types to Python native types for JSON."""
    try:
        import numpy as np

        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
    except ImportError:
        pass

    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_serializable(v) for v in obj]
    return obj
