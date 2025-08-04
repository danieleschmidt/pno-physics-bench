"""PNO Physics Bench: Training & benchmark suite for Probabilistic Neural Operators.

This package implements Probabilistic Neural Operators (PNOs) for uncertainty
quantification in neural PDE solvers, providing a comprehensive framework for
training, evaluation, and benchmarking of uncertainty-aware neural operators.
"""

__version__ = "0.1.0"
__author__ = "Daniel Schmidt"
__email__ = "daniel@terragonlabs.com"
__license__ = "MIT"

# Core imports for convenience
try:
    from .models import ProbabilisticNeuralOperator, FourierNeuralOperator, DeepONet
    from .training import PNOTrainer
    from .datasets import PDEDataset
    from .metrics import CalibrationMetrics
    
    __all__ = [
        "ProbabilisticNeuralOperator",
        "FourierNeuralOperator",
        "DeepONet",
        "PNOTrainer", 
        "PDEDataset",
        "CalibrationMetrics",
    ]
except ImportError as e:
    # Allow package to be imported even if dependencies aren't installed
    __all__ = []
    print(f"Warning: Could not import core components: {e}")