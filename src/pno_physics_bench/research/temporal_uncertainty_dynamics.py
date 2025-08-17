"""
Temporal Uncertainty Dynamics for Probabilistic Neural Operators.

This module implements novel algorithms for modeling how uncertainty evolves
over time in PDE solutions, incorporating temporal correlations and dynamics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import math

from ..models import ProbabilisticNeuralOperator


@dataclass
class TemporalUncertaintyState:
    """State representation for temporal uncertainty evolution."""
    mean: torch.Tensor
    covariance: torch.Tensor
    temporal_correlation: torch.Tensor
    dynamics_parameters: Dict[str, torch.Tensor]


class TemporalUncertaintyKernel(nn.Module):
    """
    Kernel for modeling temporal uncertainty correlations in PDE solutions.
    
    Implements a novel approach that captures how uncertainties propagate
    and correlate across time steps in neural PDE solvers.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        temporal_horizon: int = 10,
        kernel_type: str = "matern",
        correlation_decay: float = 0.9
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.temporal_horizon = temporal_horizon
        self.kernel_type = kernel_type
        self.correlation_decay = correlation_decay
        
        # Learnable temporal dynamics parameters
        self.temporal_embedding = nn.Parameter(torch.randn(temporal_horizon, hidden_dim))
        self.correlation_matrix = nn.Parameter(torch.eye(temporal_horizon) * 0.1)
        
        # Neural networks for uncertainty dynamics
        self.uncertainty_evolution = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Softplus()
        )
        
        # Attention mechanism for temporal dependencies
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            batch_first=True
        )
        
    def compute_temporal_kernel(self, t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
        """Compute temporal correlation kernel between time points."""
        if self.kernel_type == "matern":
            # Matérn kernel with nu=3/2
            distance = torch.abs(t1.unsqueeze(-1) - t2.unsqueeze(-2))
            sqrt3_dist = math.sqrt(3) * distance
            return (1 + sqrt3_dist) * torch.exp(-sqrt3_dist)
        
        elif self.kernel_type == "rbf":
            # RBF kernel
            distance_sq = (t1.unsqueeze(-1) - t2.unsqueeze(-2)).pow(2)
            return torch.exp(-0.5 * distance_sq)
        
        else:
            # Exponential decay
            distance = torch.abs(t1.unsqueeze(-1) - t2.unsqueeze(-2))
            return torch.exp(-distance / self.correlation_decay)
    
    def forward(
        self,
        uncertainty_sequence: torch.Tensor,
        time_points: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through temporal uncertainty kernel.
        
        Args:
            uncertainty_sequence: [batch, time, spatial_dims, uncertainty_dim]
            time_points: [time]
            
        Returns:
            evolved_uncertainty: [batch, time, spatial_dims, uncertainty_dim]
            correlation_matrix: [time, time]
        """
        batch_size, seq_len, *spatial_dims, unc_dim = uncertainty_sequence.shape
        
        # Compute temporal correlation matrix
        correlation_matrix = self.compute_temporal_kernel(time_points, time_points)
        
        # Apply temporal attention
        unc_flat = uncertainty_sequence.view(batch_size * np.prod(spatial_dims), seq_len, unc_dim)
        attended_unc, attention_weights = self.temporal_attention(
            unc_flat, unc_flat, unc_flat
        )
        
        # Evolve uncertainties using learned dynamics
        evolved_unc = self.uncertainty_evolution(attended_unc)
        
        # Apply temporal correlation
        correlation_weight = torch.softmax(self.correlation_matrix, dim=-1)
        correlated_unc = torch.matmul(correlation_weight, evolved_unc.transpose(0, 1)).transpose(0, 1)
        
        # Reshape back
        evolved_uncertainty = correlated_unc.view(batch_size, seq_len, *spatial_dims, unc_dim)
        
        return evolved_uncertainty, correlation_matrix


class AdaptiveTemporalPNO(nn.Module):
    """
    Adaptive Temporal Probabilistic Neural Operator.
    
    Novel architecture that adapts its uncertainty quantification based on
    temporal dynamics and learned uncertainty patterns.
    """
    
    def __init__(
        self,
        base_pno: ProbabilisticNeuralOperator,
        temporal_horizon: int = 10,
        adaptation_rate: float = 0.01,
        uncertainty_threshold: float = 0.1
    ):
        super().__init__()
        self.base_pno = base_pno
        self.temporal_horizon = temporal_horizon
        self.adaptation_rate = adaptation_rate
        self.uncertainty_threshold = uncertainty_threshold
        
        # Temporal uncertainty kernel
        self.temporal_kernel = TemporalUncertaintyKernel(
            hidden_dim=base_pno.hidden_dim,
            temporal_horizon=temporal_horizon
        )
        
        # Adaptive uncertainty controller
        self.uncertainty_controller = nn.Sequential(
            nn.Linear(base_pno.hidden_dim, base_pno.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(base_pno.hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Memory for temporal state
        self.register_buffer('temporal_state_mean', torch.zeros(1))
        self.register_buffer('temporal_state_cov', torch.eye(1))
        
    def update_temporal_state(
        self,
        new_uncertainty: torch.Tensor,
        time_step: float
    ) -> TemporalUncertaintyState:
        """Update internal temporal uncertainty state."""
        # Exponential moving average for mean
        if self.temporal_state_mean.numel() == 1:
            self.temporal_state_mean = new_uncertainty.mean()
        else:
            self.temporal_state_mean = (
                (1 - self.adaptation_rate) * self.temporal_state_mean +
                self.adaptation_rate * new_uncertainty.mean()
            )
        
        # Update covariance estimate
        uncertainty_centered = new_uncertainty - self.temporal_state_mean
        new_cov = torch.outer(uncertainty_centered.flatten(), uncertainty_centered.flatten())
        
        if self.temporal_state_cov.numel() == 1:
            self.temporal_state_cov = new_cov
        else:
            self.temporal_state_cov = (
                (1 - self.adaptation_rate) * self.temporal_state_cov +
                self.adaptation_rate * new_cov
            )
        
        return TemporalUncertaintyState(
            mean=self.temporal_state_mean,
            covariance=self.temporal_state_cov,
            temporal_correlation=torch.tensor(time_step),
            dynamics_parameters={"adaptation_rate": torch.tensor(self.adaptation_rate)}
        )
    
    def predict_temporal_sequence(
        self,
        initial_condition: torch.Tensor,
        time_points: torch.Tensor,
        num_samples: int = 100
    ) -> Tuple[torch.Tensor, torch.Tensor, List[TemporalUncertaintyState]]:
        """
        Predict temporal sequence with adaptive uncertainty.
        
        Args:
            initial_condition: [batch, channels, height, width]
            time_points: [num_time_steps]
            num_samples: Number of Monte Carlo samples
            
        Returns:
            predictions: [batch, time, channels, height, width]
            uncertainties: [batch, time, channels, height, width]
            temporal_states: List of temporal uncertainty states
        """
        batch_size = initial_condition.size(0)
        num_time_steps = len(time_points)
        
        predictions = []
        uncertainties = []
        temporal_states = []
        
        current_input = initial_condition
        
        for t_idx, t in enumerate(time_points):
            # Get prediction and uncertainty from base PNO
            pred_mean, pred_std = self.base_pno.predict_with_uncertainty(
                current_input, num_samples=num_samples
            )
            
            # Update temporal state
            temporal_state = self.update_temporal_state(pred_std, t.item())
            
            # Adaptive uncertainty modulation
            uncertainty_features = self.base_pno.lift(current_input).mean(dim=[2, 3])
            adaptation_factor = self.uncertainty_controller(uncertainty_features)
            
            # Modulate uncertainty based on adaptation
            adapted_std = pred_std * adaptation_factor.unsqueeze(-1).unsqueeze(-1)
            
            predictions.append(pred_mean)
            uncertainties.append(adapted_std)
            temporal_states.append(temporal_state)
            
            # Use prediction as input for next time step
            current_input = pred_mean
        
        # Stack results
        predictions = torch.stack(predictions, dim=1)
        uncertainties = torch.stack(uncertainties, dim=1)
        
        # Apply temporal correlation
        correlated_uncertainties, correlation_matrix = self.temporal_kernel(
            uncertainties.unsqueeze(-1), time_points
        )
        
        return predictions, correlated_uncertainties.squeeze(-1), temporal_states
    
    def compute_temporal_uncertainty_metrics(
        self,
        predictions: torch.Tensor,
        uncertainties: torch.Tensor,
        ground_truth: torch.Tensor
    ) -> Dict[str, float]:
        """Compute novel temporal uncertainty metrics."""
        metrics = {}
        
        # Temporal uncertainty consistency
        uncertainty_diff = torch.diff(uncertainties, dim=1)
        metrics['temporal_uncertainty_smoothness'] = (-uncertainty_diff.abs().mean()).item()
        
        # Uncertainty-error correlation over time
        errors = (predictions - ground_truth).abs()
        correlation_per_timestep = []
        
        for t in range(predictions.size(1)):
            err_t = errors[:, t].flatten()
            unc_t = uncertainties[:, t].flatten()
            corr = torch.corrcoef(torch.stack([err_t, unc_t]))[0, 1]
            if not torch.isnan(corr):
                correlation_per_timestep.append(corr.item())
        
        metrics['temporal_error_uncertainty_correlation'] = np.mean(correlation_per_timestep)
        
        # Uncertainty growth rate
        initial_unc = uncertainties[:, 0].mean()
        final_unc = uncertainties[:, -1].mean()
        metrics['uncertainty_growth_rate'] = (final_unc / initial_unc).item()
        
        # Temporal calibration consistency
        coverage_per_timestep = []
        for t in range(predictions.size(1)):
            pred_t = predictions[:, t]
            unc_t = uncertainties[:, t]
            target_t = ground_truth[:, t]
            
            # 90% confidence intervals
            lower = pred_t - 1.645 * unc_t
            upper = pred_t + 1.645 * unc_t
            coverage = ((target_t >= lower) & (target_t <= upper)).float().mean()
            coverage_per_timestep.append(coverage.item())
        
        metrics['temporal_calibration_consistency'] = np.std(coverage_per_timestep)
        
        return metrics


class TemporalUncertaintyAnalyzer:
    """Advanced analyzer for temporal uncertainty dynamics."""
    
    def __init__(self):
        self.analysis_cache = {}
    
    def analyze_uncertainty_propagation(
        self,
        model: AdaptiveTemporalPNO,
        test_sequence: torch.Tensor,
        time_points: torch.Tensor
    ) -> Dict[str, Any]:
        """
        Comprehensive analysis of uncertainty propagation over time.
        
        Args:
            model: Adaptive temporal PNO model
            test_sequence: [batch, time, channels, height, width]
            time_points: [time]
            
        Returns:
            Analysis results dictionary
        """
        results = {}
        
        # Get predictions with temporal uncertainty
        initial_condition = test_sequence[:, 0]
        predictions, uncertainties, temporal_states = model.predict_temporal_sequence(
            initial_condition, time_points
        )
        
        # Basic temporal metrics
        results['temporal_metrics'] = model.compute_temporal_uncertainty_metrics(
            predictions, uncertainties, test_sequence
        )
        
        # Uncertainty decomposition over time
        aleatoric_over_time = []
        epistemic_over_time = []
        
        for t in range(len(time_points)):
            # Run multiple forward passes
            predictions_samples = []
            for _ in range(50):
                pred, _ = model.base_pno.predict_with_uncertainty(
                    initial_condition if t == 0 else predictions[:, t-1], num_samples=1
                )
                predictions_samples.append(pred)
            
            predictions_samples = torch.stack(predictions_samples, dim=0)
            
            # Epistemic uncertainty (variance across samples)
            epistemic = predictions_samples.var(dim=0)
            # Aleatoric uncertainty (mean uncertainty within samples)
            aleatoric = uncertainties[:, t] - epistemic
            
            aleatoric_over_time.append(aleatoric.mean().item())
            epistemic_over_time.append(epistemic.mean().item())
        
        results['uncertainty_decomposition'] = {
            'aleatoric_over_time': aleatoric_over_time,
            'epistemic_over_time': epistemic_over_time,
            'total_uncertainty_over_time': [a + e for a, e in zip(aleatoric_over_time, epistemic_over_time)]
        }
        
        # Temporal correlation analysis
        uncertainty_matrix = uncertainties.view(uncertainties.size(0), uncertainties.size(1), -1)
        temporal_correlations = []
        
        for i in range(len(time_points)):
            for j in range(i+1, len(time_points)):
                unc_i = uncertainty_matrix[:, i].flatten()
                unc_j = uncertainty_matrix[:, j].flatten()
                corr = torch.corrcoef(torch.stack([unc_i, unc_j]))[0, 1]
                if not torch.isnan(corr):
                    temporal_correlations.append({
                        'time_lag': time_points[j] - time_points[i],
                        'correlation': corr.item()
                    })
        
        results['temporal_correlations'] = temporal_correlations
        
        # Lyapunov-like uncertainty exponent
        uncertainty_norm_over_time = [uncertainties[:, t].norm().item() for t in range(len(time_points))]
        log_unc_growth = np.diff(np.log(np.array(uncertainty_norm_over_time) + 1e-8))
        lyapunov_exponent = np.mean(log_unc_growth)
        
        results['lyapunov_uncertainty_exponent'] = lyapunov_exponent
        
        return results
    
    def generate_temporal_uncertainty_report(
        self,
        analysis_results: Dict[str, Any],
        save_path: str = "temporal_uncertainty_report.json"
    ) -> str:
        """Generate comprehensive report of temporal uncertainty analysis."""
        import json
        
        report = {
            "temporal_uncertainty_analysis": {
                "summary": {
                    "uncertainty_growth_rate": analysis_results['temporal_metrics']['uncertainty_growth_rate'],
                    "temporal_calibration_consistency": analysis_results['temporal_metrics']['temporal_calibration_consistency'],
                    "lyapunov_uncertainty_exponent": analysis_results['lyapunov_uncertainty_exponent']
                },
                "detailed_metrics": analysis_results['temporal_metrics'],
                "uncertainty_decomposition": analysis_results['uncertainty_decomposition'],
                "temporal_correlations": analysis_results['temporal_correlations'][:10],  # First 10 for brevity
                "interpretation": {
                    "stability": "stable" if analysis_results['lyapunov_uncertainty_exponent'] < 0 else "unstable",
                    "calibration_quality": "good" if analysis_results['temporal_metrics']['temporal_calibration_consistency'] < 0.1 else "needs_improvement",
                    "uncertainty_behavior": "well_controlled" if analysis_results['temporal_metrics']['uncertainty_growth_rate'] < 2.0 else "rapidly_growing"
                }
            }
        }
        
        with open(save_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        return f"Temporal uncertainty analysis report saved to {save_path}"


def create_temporal_uncertainty_experiment():
    """Create a complete temporal uncertainty experiment setup."""
    from ..models import ProbabilisticNeuralOperator
    
    # Create base PNO
    base_pno = ProbabilisticNeuralOperator(
        input_dim=3,
        hidden_dim=128,
        num_layers=4,
        modes=16,
        uncertainty_type="full"
    )
    
    # Create adaptive temporal PNO
    temporal_pno = AdaptiveTemporalPNO(
        base_pno=base_pno,
        temporal_horizon=20,
        adaptation_rate=0.01
    )
    
    # Create analyzer
    analyzer = TemporalUncertaintyAnalyzer()
    
    return temporal_pno, analyzer


# Research validation functions
def validate_temporal_uncertainty_theory():
    """Validate theoretical properties of temporal uncertainty dynamics."""
    print("🔬 Validating Temporal Uncertainty Theory...")
    
    # Test temporal correlation properties
    kernel = TemporalUncertaintyKernel(hidden_dim=64, temporal_horizon=10)
    time_points = torch.linspace(0, 1, 10)
    
    # Generate random uncertainty sequence
    uncertainty_seq = torch.randn(2, 10, 32, 32, 64)
    
    # Apply temporal kernel
    evolved_unc, corr_matrix = kernel(uncertainty_seq, time_points)
    
    # Validate properties
    assert evolved_unc.shape == uncertainty_seq.shape, "Shape preservation failed"
    assert torch.allclose(corr_matrix, corr_matrix.T, atol=1e-6), "Correlation matrix not symmetric"
    assert (torch.diag(corr_matrix) >= 0.9).all(), "Diagonal correlation too low"
    
    print("✅ Temporal uncertainty theory validation passed")
    return True


if __name__ == "__main__":
    # Run validation
    validate_temporal_uncertainty_theory()
    
    # Create experiment
    model, analyzer = create_temporal_uncertainty_experiment()
    print("🚀 Temporal Uncertainty Dynamics module ready for research!")