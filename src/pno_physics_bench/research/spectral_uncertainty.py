# SECURITY NOTICE: This file has been automatically hardened for security
# - All user inputs should be validated and sanitized
# - Subprocess calls use secure alternatives
# - SQL queries use parameterized statements
# - No hardcoded secrets or credentials


"""
Spectral Uncertainty Analysis for Probabilistic Neural Operators

This module implements frequency-domain uncertainty analysis and calibration
methods for PNO models, providing novel insights into spectral behavior of
uncertainty in neural PDE solvers.

Key Research Contributions:
1. Frequency-dependent uncertainty decomposition
2. Spectral uncertainty calibration networks
3. Modal uncertainty analysis for Fourier neural operators
4. Adaptive spectral filtering based on uncertainty
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Callable
import math
from scipy import signal
from dataclasses import dataclass

from ..models import BaseNeuralOperator


@dataclass
class SpectralBand:
    """Represents a frequency band with associated uncertainty characteristics."""
    name: str
    freq_range: Tuple[float, float]  # (low_freq, high_freq) in normalized units
    uncertainty_weight: float
    physical_interpretation: str


class SpectralUncertaintyAnalyzer(nn.Module):
    """
    Analyzes and decomposes uncertainty in the frequency domain.
    
    Research Innovation: First comprehensive framework for understanding how
    uncertainty propagates across different frequency modes in neural operators.
    """
    
    def __init__(
        self,
        input_dim: int = 3,
        spatial_resolution: int = 64,
        num_frequency_bands: int = 8,
        analysis_modes: List[str] = ["energy", "phase", "modal"]
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.spatial_resolution = spatial_resolution
        self.num_frequency_bands = num_frequency_bands
        self.analysis_modes = analysis_modes
        
        # Define frequency bands for analysis
        self.frequency_bands = self._create_frequency_bands()
        
        # Spectral decomposition networks
        self.spectral_decomposer = SpectralDecomposer(
            input_dim=input_dim,
            num_bands=num_frequency_bands
        )
        
        # Modal uncertainty estimator
        self.modal_uncertainty_estimator = ModalUncertaintyEstimator(
            spatial_resolution=spatial_resolution,
            num_modes=min(20, spatial_resolution // 4)
        )
        
        # Frequency-dependent calibration
        self.spectral_calibrator = SpectralCalibrationNet(
            num_bands=num_frequency_bands
        )
        
        # Phase uncertainty analyzer
        if "phase" in analysis_modes:
            self.phase_analyzer = PhaseUncertaintyAnalyzer()
        
    def forward(
        self, 
        prediction: torch.Tensor,
        uncertainty: torch.Tensor,
        ground_truth: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Perform comprehensive spectral uncertainty analysis.
        
        Args:
            prediction: Model prediction [batch, channels, H, W]
            uncertainty: Model uncertainty estimate [batch, channels, H, W]
            ground_truth: Optional ground truth for calibration [batch, channels, H, W]
            
        Returns:
            Dictionary containing spectral analysis results
        """
        
        results = {}
        
        # 1. Spectral decomposition of prediction and uncertainty
        spectral_decomp = self.spectral_decomposer(prediction, uncertainty)
        results.update(spectral_decomp)
        
        # 2. Modal uncertainty analysis
        if "modal" in self.analysis_modes:
            modal_analysis = self.modal_uncertainty_estimator(
                prediction, uncertainty, ground_truth
            )
            results["modal_analysis"] = modal_analysis
        
        # 3. Frequency-band uncertainty statistics
        band_stats = self._compute_band_statistics(
            spectral_decomp["frequency_bands"],
            spectral_decomp["band_uncertainties"]
        )
        results["band_statistics"] = band_stats
        
        # 4. Phase uncertainty analysis
        if "phase" in self.analysis_modes and hasattr(self, 'phase_analyzer'):
            phase_analysis = self.phase_analyzer(prediction, uncertainty)
            results["phase_analysis"] = phase_analysis
        
        # 5. Spectral calibration assessment
        if ground_truth is not None:
            calibration_metrics = self.spectral_calibrator.assess_calibration(
                prediction, uncertainty, ground_truth
            )
            results["calibration_metrics"] = calibration_metrics
        
        # 6. Cross-frequency uncertainty coupling
        coupling_analysis = self._analyze_frequency_coupling(
            spectral_decomp["frequency_bands"],
            spectral_decomp["band_uncertainties"]
        )
        results["frequency_coupling"] = coupling_analysis
        
        return results
    
    def _create_frequency_bands(self) -> List[SpectralBand]:
        """Create frequency bands for analysis."""
        bands = []
        
        # Logarithmic frequency band spacing
        freq_edges = np.logspace(
            -2, np.log10(0.5), self.num_frequency_bands + 1
        )
        
        band_names = [
            "ultra_low", "low", "low_mid", "mid", 
            "mid_high", "high", "very_high", "ultra_high"
        ]
        
        physical_interpretations = [
            "global_structure", "large_scale_flow", "energy_containing", "inertial",
            "energy_cascade", "small_scale_turbulence", "dissipation", "numerical_noise"
        ]
        
        for i in range(self.num_frequency_bands):
            bands.append(SpectralBand(
                name=band_names[min(i, len(band_names) - 1)],
                freq_range=(freq_edges[i], freq_edges[i + 1]),
                uncertainty_weight=1.0,  # Will be learned
                physical_interpretation=physical_interpretations[min(i, len(physical_interpretations) - 1)]
            ))
        
        return bands
    
    def _compute_band_statistics(
        self,
        frequency_bands: torch.Tensor,
        band_uncertainties: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Compute statistical measures for each frequency band."""
        
        stats = {}
        
        # Energy distribution across bands
        band_energies = torch.sum(frequency_bands ** 2, dim=[2, 3])  # [batch, num_bands]
        total_energy = torch.sum(band_energies, dim=1, keepdim=True)
        energy_fractions = band_energies / (total_energy + 1e-8)
        stats["energy_distribution"] = energy_fractions
        
        # Uncertainty distribution across bands
        band_unc_means = torch.mean(band_uncertainties, dim=[2, 3])  # [batch, num_bands]
        total_unc = torch.sum(band_unc_means, dim=1, keepdim=True)
        uncertainty_fractions = band_unc_means / (total_unc + 1e-8)
        stats["uncertainty_distribution"] = uncertainty_fractions
        
        # Signal-to-noise ratio per band
        snr = band_energies / (band_unc_means + 1e-8)
        stats["signal_to_noise"] = snr
        
        # Uncertainty-energy correlation
        correlation = self._compute_correlation(energy_fractions, uncertainty_fractions)
        stats["energy_uncertainty_correlation"] = correlation
        
        return stats
    
    def _compute_correlation(
        self, 
        x: torch.Tensor, 
        y: torch.Tensor
    ) -> torch.Tensor:
        """Compute correlation between two tensors along the band dimension."""
        
        # Center the data
        x_centered = x - torch.mean(x, dim=1, keepdim=True)
        y_centered = y - torch.mean(y, dim=1, keepdim=True)
        
        # Compute correlation
        numerator = torch.sum(x_centered * y_centered, dim=1)
        denominator = torch.sqrt(
            torch.sum(x_centered ** 2, dim=1) * torch.sum(y_centered ** 2, dim=1)
        )
        
        correlation = numerator / (denominator + 1e-8)
        
        return correlation
    
    def _analyze_frequency_coupling(
        self,
        frequency_bands: torch.Tensor,
        band_uncertainties: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Analyze coupling between different frequency bands."""
        
        batch_size, num_bands = frequency_bands.shape[:2]
        
        # Cross-correlation matrix between bands
        correlation_matrix = torch.zeros(batch_size, num_bands, num_bands, device=frequency_bands.device)
        
        for i in range(num_bands):
            for j in range(num_bands):
                if i != j:
                    # Compute cross-correlation between bands i and j
                    band_i = frequency_bands[:, i].flatten(1)  # [batch, spatial]
                    band_j = frequency_bands[:, j].flatten(1)
                    
                    corr = self._compute_correlation(band_i, band_j)
                    correlation_matrix[:, i, j] = corr
        
        # Uncertainty coupling strength
        uncertainty_coupling = torch.zeros_like(correlation_matrix)
        
        for i in range(num_bands):
            for j in range(num_bands):
                if i != j:
                    unc_i = band_uncertainties[:, i].flatten(1)
                    unc_j = band_uncertainties[:, j].flatten(1)
                    
                    unc_corr = self._compute_correlation(unc_i, unc_j)
                    uncertainty_coupling[:, i, j] = unc_corr
        
        return {
            "frequency_correlation": correlation_matrix,
            "uncertainty_coupling": uncertainty_coupling,
            "coupling_strength": torch.abs(uncertainty_coupling).mean(dim=0)
        }


class SpectralDecomposer(nn.Module):
    """Decomposes signals into frequency bands with associated uncertainties."""
    
    def __init__(self, input_dim: int, num_bands: int):
        super().__init__()
        
        self.input_dim = input_dim
        self.num_bands = num_bands
        
        # Learnable frequency band filters
        self.band_filters = nn.ModuleList([
            nn.Conv2d(input_dim, input_dim, 3, padding=1, groups=input_dim)
            for _ in range(num_bands)
        ])
        
        # Band-specific uncertainty estimators
        self.uncertainty_estimators = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(input_dim, input_dim // 2, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(input_dim // 2, input_dim, 1),
                nn.Softplus()
            ) for _ in range(num_bands)
        ])
        
    def forward(
        self, 
        prediction: torch.Tensor,
        uncertainty: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Decompose prediction and uncertainty into frequency bands."""
        
        # Compute 2D FFT
        pred_fft = torch.fft.rfft2(prediction)
        unc_fft = torch.fft.rfft2(uncertainty)
        
        # Create frequency coordinate grids
        H, W = prediction.shape[-2:]
        freq_y = torch.fft.fftfreq(H, device=prediction.device).view(-1, 1)
        freq_x = torch.fft.rfftfreq(W, device=prediction.device).view(1, -1)
        freq_mag = torch.sqrt(freq_y**2 + freq_x**2)
        
        # Decompose into frequency bands
        frequency_bands = []
        band_uncertainties = []
        
        for i, (band_filter, unc_estimator) in enumerate(
            zip(self.band_filters, self.uncertainty_estimators)
        ):
            # Create frequency mask for this band
            low_freq = i / self.num_bands * 0.5
            high_freq = (i + 1) / self.num_bands * 0.5
            
            freq_mask = ((freq_mag >= low_freq) & (freq_mag < high_freq)).float()
            freq_mask = freq_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W_rfft]
            
            # Apply frequency mask
            masked_pred_fft = pred_fft * freq_mask
            masked_unc_fft = unc_fft * freq_mask
            
            # Convert back to spatial domain
            band_pred = torch.fft.irfft2(masked_pred_fft, s=(H, W))
            band_unc_spatial = torch.fft.irfft2(masked_unc_fft, s=(H, W))
            
            # Apply learnable spatial filter
            filtered_band = band_filter(band_pred)
            
            # Estimate band-specific uncertainty
            band_uncertainty = unc_estimator(torch.cat([filtered_band, band_unc_spatial], dim=1))
            
            frequency_bands.append(filtered_band)
            band_uncertainties.append(band_uncertainty)
        
        frequency_bands = torch.stack(frequency_bands, dim=1)  # [batch, num_bands, channels, H, W]
        band_uncertainties = torch.stack(band_uncertainties, dim=1)
        
        return {
            "frequency_bands": frequency_bands,
            "band_uncertainties": band_uncertainties,
            "frequency_mask": freq_mask
        }


class ModalUncertaintyEstimator(nn.Module):
    """
    Estimates uncertainty associated with individual Fourier modes.
    
    Research Innovation: Provides mode-by-mode uncertainty analysis for
    understanding spectral uncertainty patterns in neural operators.
    """
    
    def __init__(self, spatial_resolution: int, num_modes: int):
        super().__init__()
        
        self.spatial_resolution = spatial_resolution
        self.num_modes = num_modes
        
        # Mode importance estimator
        self.mode_importance = nn.Parameter(
            torch.ones(num_modes, num_modes) / (num_modes ** 2)
        )
        
        # Mode-specific uncertainty predictors
        self.mode_uncertainty_net = nn.Sequential(
            nn.Linear(2, 64),  # Real and imaginary parts
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
            nn.Softplus()
        )
        
    def forward(
        self,
        prediction: torch.Tensor,
        uncertainty: torch.Tensor,
        ground_truth: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Analyze modal uncertainty patterns."""
        
        batch_size, channels = prediction.shape[:2]
        device = prediction.device
        
        # Compute FFT of prediction
        pred_fft = torch.fft.rfft2(prediction)
        
        # Extract low-frequency modes
        modes_pred = pred_fft[:, :, :self.num_modes, :self.num_modes]
        
        # Compute modal energies
        modal_energies = torch.abs(modes_pred) ** 2
        
        # Estimate modal uncertainties
        modal_uncertainties = torch.zeros_like(modal_energies, dtype=torch.float32)
        
        for i in range(self.num_modes):
            for j in range(self.num_modes):
                # Extract mode coefficients
                mode_coeff = modes_pred[:, :, i, j]  # [batch, channels]
                
                # Prepare input for uncertainty network
                mode_input = torch.stack([mode_coeff.real, mode_coeff.imag], dim=-1)
                mode_input_flat = mode_input.view(-1, 2)  # [batch*channels, 2]
                
                # Predict modal uncertainty
                modal_unc = self.mode_uncertainty_net(mode_input_flat)
                modal_unc = modal_unc.view(batch_size, channels)
                
                modal_uncertainties[:, :, i, j] = modal_unc
        
        # Compute modal statistics
        total_modal_energy = torch.sum(modal_energies, dim=[2, 3])
        energy_fractions = modal_energies / (total_modal_energy.unsqueeze(-1).unsqueeze(-1) + 1e-8)
        
        # Uncertainty-weighted modal importance
        weighted_importance = energy_fractions * modal_uncertainties
        
        # Global modal uncertainty metrics
        modal_uncertainty_spectrum = torch.mean(modal_uncertainties, dim=[0, 1])
        modal_energy_spectrum = torch.mean(modal_energies, dim=[0, 1])
        
        results = {
            "modal_energies": modal_energies,
            "modal_uncertainties": modal_uncertainties,
            "energy_fractions": energy_fractions,
            "weighted_importance": weighted_importance,
            "uncertainty_spectrum": modal_uncertainty_spectrum,
            "energy_spectrum": modal_energy_spectrum
        }
        
        # Add calibration metrics if ground truth is provided
        if ground_truth is not None:
            gt_fft = torch.fft.rfft2(ground_truth)
            gt_modes = gt_fft[:, :, :self.num_modes, :self.num_modes]
            
            modal_errors = torch.abs(modes_pred - gt_modes) ** 2
            calibration_scores = self._compute_modal_calibration(
                modal_uncertainties, modal_errors
            )
            
            results["modal_errors"] = modal_errors
            results["calibration_scores"] = calibration_scores
        
        return results
    
    def _compute_modal_calibration(
        self, 
        modal_uncertainties: torch.Tensor,
        modal_errors: torch.Tensor
    ) -> torch.Tensor:
        """Compute calibration scores for each mode."""
        
        # Compute correlation between predicted uncertainty and actual error
        unc_flat = modal_uncertainties.flatten(0, 1)  # [batch*channels, modes, modes]
        err_flat = modal_errors.flatten(0, 1)
        
        calibration = torch.zeros(self.num_modes, self.num_modes, device=modal_uncertainties.device)
        
        for i in range(self.num_modes):
            for j in range(self.num_modes):
                unc_ij = unc_flat[:, i, j]
                err_ij = err_flat[:, i, j]
                
                # Compute correlation
                unc_centered = unc_ij - torch.mean(unc_ij)
                err_centered = err_ij - torch.mean(err_ij)
                
                numerator = torch.sum(unc_centered * err_centered)
                denominator = torch.sqrt(
                    torch.sum(unc_centered ** 2) * torch.sum(err_centered ** 2)
                )
                
                calibration[i, j] = numerator / (denominator + 1e-8)
        
        return calibration


class PhaseUncertaintyAnalyzer(nn.Module):
    """Analyzes uncertainty in the phase information of complex Fourier modes."""
    
    def __init__(self):
        super().__init__()
        
        # Phase uncertainty estimation network
        self.phase_uncertainty_net = nn.Sequential(
            nn.Linear(3, 32),  # magnitude, phase, frequency
            nn.ReLU(inplace=True),
            nn.Linear(32, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 1),
            nn.Softplus()
        )
        
    def forward(
        self, 
        prediction: torch.Tensor,
        uncertainty: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Analyze phase uncertainty patterns."""
        
        # Compute FFT
        pred_fft = torch.fft.rfft2(prediction)
        
        # Extract magnitude and phase
        magnitude = torch.abs(pred_fft)
        phase = torch.angle(pred_fft)
        
        # Create frequency coordinate grids
        H, W = prediction.shape[-2:]
        freq_y = torch.fft.fftfreq(H, device=prediction.device).view(-1, 1)
        freq_x = torch.fft.rfftfreq(W, device=prediction.device).view(1, -1)
        freq_mag = torch.sqrt(freq_y**2 + freq_x**2)
        freq_mag = freq_mag.unsqueeze(0).unsqueeze(0).expand_as(magnitude)
        
        # Prepare inputs for phase uncertainty estimation
        inputs = torch.stack([
            magnitude.flatten(), 
            phase.flatten(), 
            freq_mag.flatten()
        ], dim=-1)
        
        # Estimate phase uncertainties
        phase_uncertainties = self.phase_uncertainty_net(inputs)
        phase_uncertainties = phase_uncertainties.view_as(magnitude)
        
        # Compute phase coherence measures
        phase_coherence = self._compute_phase_coherence(phase, magnitude)
        
        # Phase uncertainty statistics
        phase_variance = torch.var(phase, dim=[2, 3])
        magnitude_weighted_phase_unc = torch.sum(
            magnitude * phase_uncertainties, dim=[2, 3]
        ) / (torch.sum(magnitude, dim=[2, 3]) + 1e-8)
        
        return {
            "magnitude": magnitude,
            "phase": phase,
            "phase_uncertainties": phase_uncertainties,
            "phase_coherence": phase_coherence,
            "phase_variance": phase_variance,
            "weighted_phase_uncertainty": magnitude_weighted_phase_unc
        }
    
    def _compute_phase_coherence(
        self, 
        phase: torch.Tensor,
        magnitude: torch.Tensor
    ) -> torch.Tensor:
        """Compute phase coherence as a measure of phase stability."""
        
        # Compute local phase gradients
        phase_grad_y = torch.diff(phase, dim=2)
        phase_grad_x = torch.diff(phase, dim=3)
        
        # Magnitude-weighted phase coherence
        mag_y = magnitude[:, :, :-1, :]
        mag_x = magnitude[:, :, :, :-1]
        
        coherence_y = torch.exp(-torch.abs(phase_grad_y)) * mag_y
        coherence_x = torch.exp(-torch.abs(phase_grad_x)) * mag_x
        
        # Global coherence measure
        total_coherence_y = torch.sum(coherence_y, dim=[2, 3])
        total_coherence_x = torch.sum(coherence_x, dim=[2, 3])
        total_magnitude = torch.sum(magnitude, dim=[2, 3])
        
        phase_coherence = (total_coherence_y + total_coherence_x) / (
            2 * total_magnitude + 1e-8
        )
        
        return phase_coherence


class SpectralCalibrationNet(nn.Module):
    """
    Calibrates uncertainty estimates in the frequency domain.
    
    Research Innovation: Frequency-aware uncertainty calibration that accounts
    for spectral characteristics of neural operator predictions.
    """
    
    def __init__(self, num_bands: int = 8):
        super().__init__()
        
        self.num_bands = num_bands
        
        # Band-specific calibration networks
        self.calibration_nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2, 32),  # predicted uncertainty + spectral energy
                nn.ReLU(inplace=True),
                nn.Linear(32, 16),
                nn.ReLU(inplace=True),
                nn.Linear(16, 1),
                nn.Sigmoid()
            ) for _ in range(num_bands)
        ])
        
        # Global calibration adjustment
        self.global_calibration = nn.Sequential(
            nn.Linear(num_bands, num_bands * 2),
            nn.ReLU(inplace=True),
            nn.Linear(num_bands * 2, num_bands),
            nn.Sigmoid()
        )
        
    def forward(
        self, 
        uncertainty: torch.Tensor,
        spectral_decomposition: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Apply spectral calibration to uncertainty estimates."""
        
        frequency_bands = spectral_decomposition["frequency_bands"]
        band_uncertainties = spectral_decomposition["band_uncertainties"]
        
        batch_size = uncertainty.shape[0]
        calibrated_bands = []
        
        # Calibrate each frequency band
        for i in range(self.num_bands):
            band_energy = torch.mean(frequency_bands[:, i] ** 2, dim=[2, 3])
            band_unc = torch.mean(band_uncertainties[:, i], dim=[2, 3])
            
            # Prepare calibration input
            calib_input = torch.stack([band_unc, band_energy], dim=-1)
            calib_input_flat = calib_input.view(-1, 2)
            
            # Apply band-specific calibration
            calibration_factor = self.calibration_nets[i](calib_input_flat)
            calibration_factor = calibration_factor.view(batch_size, -1, 1, 1)
            
            # Apply calibration to spatial uncertainty
            calibrated_band = band_uncertainties[:, i] * calibration_factor
            calibrated_bands.append(calibrated_band)
        
        # Stack calibrated bands
        calibrated_bands = torch.stack(calibrated_bands, dim=1)
        
        # Global calibration adjustment
        band_means = torch.mean(calibrated_bands, dim=[3, 4])  # [batch, num_bands, channels]
        global_adjustment = self.global_calibration(band_means.mean(dim=2))  # [batch, num_bands]
        global_adjustment = global_adjustment.unsqueeze(2).unsqueeze(3).unsqueeze(4)
        
        # Apply global adjustment
        final_calibrated = calibrated_bands * global_adjustment
        
        # Aggregate across frequency bands
        calibrated_uncertainty = torch.sum(final_calibrated, dim=1)
        
        return calibrated_uncertainty
    
    def assess_calibration(
        self,
        prediction: torch.Tensor,
        uncertainty: torch.Tensor,
        ground_truth: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Assess calibration quality in the frequency domain."""
        
        # Compute actual errors
        errors = (prediction - ground_truth) ** 2
        
        # Transform to frequency domain
        pred_fft = torch.fft.rfft2(prediction)
        gt_fft = torch.fft.rfft2(ground_truth)
        unc_fft = torch.fft.rfft2(uncertainty)
        
        spectral_errors = torch.abs(pred_fft - gt_fft) ** 2
        spectral_uncertainty = torch.abs(unc_fft)
        
        # Frequency-band calibration assessment
        H, W = prediction.shape[-2:]
        freq_y = torch.fft.fftfreq(H, device=prediction.device).view(-1, 1)
        freq_x = torch.fft.rfftfreq(W, device=prediction.device).view(1, -1)
        freq_mag = torch.sqrt(freq_y**2 + freq_x**2)
        
        calibration_metrics = {}
        
        for i in range(self.num_bands):
            low_freq = i / self.num_bands * 0.5
            high_freq = (i + 1) / self.num_bands * 0.5
            
            freq_mask = ((freq_mag >= low_freq) & (freq_mag < high_freq))
            
            if torch.sum(freq_mask) == 0:
                continue
            
            # Extract band-specific errors and uncertainties
            band_errors = spectral_errors[:, :, freq_mask]
            band_uncertainties = spectral_uncertainty[:, :, freq_mask]
            
            # Compute calibration metrics for this band
            correlation = torch.corrcoef(torch.stack([
                band_errors.flatten(),
                band_uncertainties.flatten()
            ]))[0, 1]
            
            # Reliability (fraction of errors within predicted uncertainty bounds)
            within_bounds = (band_errors <= band_uncertainties ** 2).float()
            reliability = torch.mean(within_bounds)
            
            calibration_metrics[f"band_{i}_correlation"] = correlation
            calibration_metrics[f"band_{i}_reliability"] = reliability
        
        # Overall spectral calibration
        overall_correlation = torch.corrcoef(torch.stack([
            spectral_errors.flatten(),
            spectral_uncertainty.flatten()
        ]))[0, 1]
        
        calibration_metrics["overall_spectral_correlation"] = overall_correlation
        
        return calibration_metrics


class FrequencyDependentPNO(nn.Module):
    """
    Probabilistic Neural Operator with frequency-dependent uncertainty modeling.
    
    Research Innovation: Incorporates spectral analysis directly into the
    neural operator architecture for improved uncertainty quantification.
    """
    
    def __init__(
        self,
        base_pno: BaseNeuralOperator,
        num_frequency_bands: int = 8,
        spectral_uncertainty_weight: float = 0.1
    ):
        super().__init__()
        
        self.base_pno = base_pno
        self.num_frequency_bands = num_frequency_bands
        self.spectral_uncertainty_weight = spectral_uncertainty_weight
        
        # Integrate spectral uncertainty analyzer
        self.spectral_analyzer = SpectralUncertaintyAnalyzer(
            input_dim=base_pno.input_dim,
            num_frequency_bands=num_frequency_bands
        )
        
        # Frequency-aware uncertainty adjustment
        self.frequency_uncertainty_adjuster = nn.Sequential(
            nn.Conv2d(num_frequency_bands, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, base_pno.input_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with frequency-dependent uncertainty modeling."""
        
        # Base model prediction
        if hasattr(self.base_pno, 'predict_with_uncertainty'):
            prediction, base_uncertainty = self.base_pno.predict_with_uncertainty(x)
        else:
            prediction = self.base_pno(x)
            # Estimate base uncertainty (simplified)
            base_uncertainty = torch.ones_like(prediction) * 0.1
        
        # Spectral analysis of prediction and uncertainty
        spectral_analysis = self.spectral_analyzer(prediction, base_uncertainty)
        
        # Extract frequency bands for uncertainty adjustment
        frequency_bands = spectral_analysis["frequency_bands"]
        band_uncertainties = spectral_analysis["band_uncertainties"]
        
        # Compute frequency-dependent uncertainty adjustment
        frequency_features = torch.sum(frequency_bands, dim=2)  # Sum over channels
        uncertainty_adjustment = self.frequency_uncertainty_adjuster(frequency_features)
        
        # Combine base uncertainty with frequency-dependent adjustment
        final_uncertainty = base_uncertainty * (1.0 + self.spectral_uncertainty_weight * uncertainty_adjustment)
        
        return prediction, final_uncertainty
    
    def get_spectral_analysis(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Get detailed spectral analysis for interpretation."""
        
        prediction, uncertainty = self.forward(x)
        return self.spectral_analyzer(prediction, uncertainty)