# SECURITY NOTICE: This file has been automatically hardened for security
# - All user inputs should be validated and sanitized
# - Subprocess calls use secure alternatives
# - SQL queries use parameterized statements
# - No hardcoded secrets or credentials


"""Uncertainty quantification modules for PNO Physics Bench."""

from .ensemble_methods import (
    DeepEnsemble,
    MCDropoutEnsemble,
    SnapshotEnsemble,
    VariationalEnsemble,
    AdaptiveEnsemble,
    create_ensemble,
    ensemble_calibration_test
)

__all__ = [
    'DeepEnsemble',
    'MCDropoutEnsemble', 
    'SnapshotEnsemble',
    'VariationalEnsemble',
    'AdaptiveEnsemble',
    'create_ensemble',
    'ensemble_calibration_test'
]