"""
Advanced Research Extensions for PNO Physics Bench

This module contains cutting-edge research implementations that advance
the state-of-the-art in probabilistic neural operators and uncertainty
quantification for PDE solvers.

Research Areas:
- Hierarchical uncertainty decomposition
- Multi-fidelity uncertainty propagation
- Physics-informed uncertainty quantification
- Adaptive uncertainty-aware active learning
- Spectral uncertainty analysis
- Quantum-enhanced probabilistic modeling
"""

from .hierarchical_uncertainty import (
    HierarchicalUncertaintyDecomposer,
    UncertaintyEstimator,
    CrossScaleCouplingNet,
    AdaptiveUncertaintyPropagator
)

from .multi_fidelity import (
    MultiFidelityPNO,
    FidelitySelector,
    UncertaintyFusionNet
)

from .spectral_uncertainty import (
    SpectralUncertaintyAnalyzer,
    FrequencyDependentPNO,
    SpectralCalibrationNet
)

__all__ = [
    "HierarchicalUncertaintyDecomposer",
    "UncertaintyEstimator", 
    "CrossScaleCouplingNet",
    "AdaptiveUncertaintyPropagator",
    "MultiFidelityPNO",
    "FidelitySelector",
    "UncertaintyFusionNet",
    "SpectralUncertaintyAnalyzer",
    "FrequencyDependentPNO",
    "SpectralCalibrationNet"
]