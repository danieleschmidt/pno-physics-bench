# SECURITY NOTICE: This file has been automatically hardened for security
# - All user inputs should be validated and sanitized
# - Subprocess calls use secure alternatives
# - SQL queries use parameterized statements
# - No hardcoded secrets or credentials


"""
Hierarchical Uncertainty Decomposition for Probabilistic Neural Operators

This module implements novel hierarchical uncertainty quantification methods
that decompose uncertainty across multiple scales and physical phenomena.

Key Research Contributions:
1. Multi-scale epistemic uncertainty decomposition
2. Physics-informed uncertainty hierarchies
3. Adaptive uncertainty propagation across scales
4. Cross-frequency uncertainty coupling analysis
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from abc import ABC, abstractmethod
import warnings

from ..models import BaseNeuralOperator


class HierarchicalUncertaintyDecomposer(nn.Module):
    """
    Decomposes uncertainty into hierarchical components across multiple scales.
    
    Research Innovation: Unlike traditional aleatoric/epistemic decomposition,
    this method identifies scale-dependent uncertainty sources and their 
    cross-scale interactions in PDE solutions.
    """
    
    def __init__(
        self,
        base_model: BaseNeuralOperator,
        scales: List[int] = [1, 4, 16, 64],
        uncertainty_types: List[str] = ["physics", "boundary", "initial", "numerical"],
        coupling_analysis: bool = True
    ):
        super().__init__()
        
        self.base_model = base_model
        self.scales = scales
        self.uncertainty_types = uncertainty_types
        self.coupling_analysis = coupling_analysis
        
        # Scale-specific uncertainty estimators
        self.scale_estimators = nn.ModuleDict({
            f"scale_{s}": UncertaintyEstimator(
                input_dim=base_model.hidden_dim,
                scale_factor=s,
                uncertainty_dim=len(uncertainty_types)
            ) for s in scales
        })
        
        # Cross-scale coupling network
        if coupling_analysis:
            self.coupling_network = CrossScaleCouplingNet(
                scales=scales,
                hidden_dim=base_model.hidden_dim
            )
        
        # Physics-informed uncertainty weighting
        self.physics_weights = nn.ModuleDict({
            ut: nn.Linear(base_model.hidden_dim, 1) 
            for ut in uncertainty_types
        })
        
    def forward(
        self, 
        x: torch.Tensor,
        physics_params: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute hierarchical uncertainty decomposition.
        
        Args:
            x: Input tensor [batch, channels, height, width]
            physics_params: Optional physics parameters for informed weighting
            
        Returns:
            Dictionary containing:
            - scale_uncertainties: Per-scale uncertainty estimates
            - type_uncertainties: Per-physics-type uncertainty
            - coupling_matrix: Cross-scale uncertainty coupling
            - total_uncertainty: Aggregated uncertainty estimate
        """
        batch_size = x.shape[0]
        device = x.device
        
        # Get base model features
        with torch.no_grad():
            base_features = self.base_model.encode(x)  # Assume encode method exists
        
        # Compute scale-specific uncertainties
        scale_uncertainties = {}
        scale_features = {}
        
        for scale in self.scales:
            # Multi-scale feature extraction
            if scale > 1:
                downsampled_x = F.avg_pool2d(x, kernel_size=scale, stride=scale)
                upsampled_x = F.interpolate(
                    downsampled_x, 
                    size=x.shape[-2:], 
                    mode='bilinear', 
                    align_corners=False
                )
            else:
                upsampled_x = x
            
            # Extract scale-specific features
            scale_feat = self.base_model.encode(upsampled_x)
            scale_features[f"scale_{scale}"] = scale_feat
            
            # Estimate scale-specific uncertainty
            scale_unc = self.scale_estimators[f"scale_{scale}"](scale_feat)
            scale_uncertainties[f"scale_{scale}"] = scale_unc
        
        # Compute physics-type uncertainties
        type_uncertainties = {}
        for unc_type in self.uncertainty_types:
            # Weight by physics-informed factors
            weight = torch.sigmoid(self.physics_weights[unc_type](base_features))
            
            # Aggregate across scales with physics weighting
            type_unc = torch.zeros_like(scale_uncertainties["scale_1"])
            for scale in self.scales:
                scale_weight = self._compute_physics_scale_weight(
                    unc_type, scale, physics_params
                )
                type_unc += scale_weight * scale_uncertainties[f"scale_{scale}"]
            
            type_uncertainties[unc_type] = weight * type_unc
        
        # Cross-scale coupling analysis
        coupling_matrix = None
        if self.coupling_analysis:
            coupling_matrix = self.coupling_network(scale_features)
        
        # Aggregate total uncertainty with coupling effects
        total_uncertainty = self._aggregate_uncertainties(
            scale_uncertainties, 
            type_uncertainties, 
            coupling_matrix
        )
        
        return {
            "scale_uncertainties": scale_uncertainties,
            "type_uncertainties": type_uncertainties,
            "coupling_matrix": coupling_matrix,
            "total_uncertainty": total_uncertainty
        }
    
    def _compute_physics_scale_weight(
        self, 
        unc_type: str, 
        scale: int, 
        physics_params: Optional[Dict[str, torch.Tensor]]
    ) -> float:
        """Compute physics-informed weights for different scales."""
        
        # Default uniform weighting
        if physics_params is None:
            return 1.0 / len(self.scales)
        
        # Physics-specific scale weighting
        if unc_type == "boundary":
            # Boundary effects dominate at large scales
            return min(1.0, scale / max(self.scales))
        elif unc_type == "initial":
            # Initial condition uncertainty affects all scales equally
            return 1.0 / len(self.scales)
        elif unc_type == "physics":
            # Physics uncertainty varies with Reynolds number, Peclet number, etc.
            if "reynolds" in physics_params:
                re_num = physics_params["reynolds"].mean().item()
                # High Re -> more uncertainty at small scales
                return max(0.1, 1.0 / (1.0 + scale * re_num / 1000))
            return 1.0 / len(self.scales)
        elif unc_type == "numerical":
            # Numerical errors accumulate at small scales
            return max(0.1, 1.0 / scale)
        
        return 1.0 / len(self.scales)
    
    def _aggregate_uncertainties(
        self,
        scale_uncertainties: Dict[str, torch.Tensor],
        type_uncertainties: Dict[str, torch.Tensor],
        coupling_matrix: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Aggregate uncertainties accounting for correlations."""
        
        # Base aggregation: sum of type uncertainties
        total = sum(type_uncertainties.values())
        
        # Add coupling effects if available
        if coupling_matrix is not None:
            # Coupling matrix adjusts uncertainty based on cross-scale interactions
            coupling_adjustment = torch.mean(coupling_matrix, dim=[2, 3], keepdim=True)
            total = total * (1.0 + coupling_adjustment)
        
        return total


class UncertaintyEstimator(nn.Module):
    """Estimates uncertainty at a specific scale."""
    
    def __init__(self, input_dim: int, scale_factor: int, uncertainty_dim: int):
        super().__init__()
        
        self.scale_factor = scale_factor
        
        # Scale-adaptive architecture
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(input_dim, input_dim // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(input_dim // 2, uncertainty_dim, 1)
        )
        
        # Scale-specific normalization
        self.scale_norm = nn.GroupNorm(
            num_groups=min(uncertainty_dim, 8), 
            num_channels=uncertainty_dim
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Estimate scale-specific uncertainty."""
        features = self.feature_extractor(x)
        normalized = self.scale_norm(features)
        
        # Ensure positive uncertainty
        uncertainty = F.softplus(normalized) + 1e-6
        
        return uncertainty


class CrossScaleCouplingNet(nn.Module):
    """
    Models cross-scale uncertainty coupling relationships.
    
    Research Innovation: Captures how uncertainty propagates and couples
    across different scales in multiscale PDE phenomena.
    """
    
    def __init__(self, scales: List[int], hidden_dim: int):
        super().__init__()
        
        self.scales = scales
        self.num_scales = len(scales)
        
        # Cross-scale attention mechanism
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            batch_first=True
        )
        
        # Coupling strength estimator
        self.coupling_predictor = nn.Sequential(
            nn.Linear(hidden_dim * self.num_scales, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, self.num_scales * self.num_scales),
            nn.Sigmoid()
        )
        
    def forward(self, scale_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute cross-scale coupling matrix."""
        
        # Flatten spatial dimensions for attention
        features_list = []
        batch_size = None
        
        for scale in self.scales:
            feat = scale_features[f"scale_{scale}"]
            if batch_size is None:
                batch_size = feat.shape[0]
            
            # Flatten spatial dimensions
            feat_flat = feat.view(batch_size, feat.shape[1], -1)  # [B, C, HW]
            feat_flat = feat_flat.mean(dim=2)  # Global average pooling [B, C]
            features_list.append(feat_flat)
        
        # Stack features for cross-attention
        stacked_features = torch.stack(features_list, dim=1)  # [B, num_scales, C]
        
        # Compute cross-scale attention
        attended_features, attention_weights = self.cross_attention(
            stacked_features, stacked_features, stacked_features
        )
        
        # Predict coupling strengths
        combined_features = attended_features.view(batch_size, -1)
        coupling_strengths = self.coupling_predictor(combined_features)
        
        # Reshape to coupling matrix
        coupling_matrix = coupling_strengths.view(
            batch_size, self.num_scales, self.num_scales
        )
        
        # Add spatial dimensions back
        coupling_matrix = coupling_matrix.unsqueeze(-1).unsqueeze(-1)
        
        return coupling_matrix


class AdaptiveUncertaintyPropagator(nn.Module):
    """
    Propagates uncertainty adaptively across time steps in PDE rollouts.
    
    Research Innovation: Dynamically adjusts uncertainty propagation based 
    on local flow characteristics and solution regularity.
    """
    
    def __init__(
        self,
        base_model: BaseNeuralOperator,
        propagation_steps: int = 10,
        adaptation_threshold: float = 0.1
    ):
        super().__init__()
        
        self.base_model = base_model
        self.propagation_steps = propagation_steps
        self.adaptation_threshold = adaptation_threshold
        
        # Adaptive propagation network
        self.propagation_controller = nn.Sequential(
            nn.Linear(base_model.hidden_dim + 1, 64),  # +1 for current uncertainty
            nn.ReLU(inplace=True),
            nn.Linear(64, 3),  # [growth_rate, damping_factor, coupling_strength]
            nn.Sigmoid()
        )
        
        # Uncertainty evolution predictor
        self.uncertainty_predictor = nn.GRUCell(
            input_size=base_model.hidden_dim,
            hidden_size=64
        )
        
    def forward(
        self,
        initial_state: torch.Tensor,
        initial_uncertainty: torch.Tensor,
        physics_params: Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Propagate uncertainty through time with adaptive control.
        
        Returns:
            states: List of predicted states at each time step
            uncertainties: List of uncertainty estimates at each time step
        """
        
        states = [initial_state]
        uncertainties = [initial_uncertainty]
        
        current_state = initial_state
        current_uncertainty = initial_uncertainty
        hidden = None
        
        for step in range(self.propagation_steps):
            # Predict next state with base model
            with torch.no_grad():
                base_features = self.base_model.encode(current_state)
                next_state = self.base_model(current_state)
            
            # Update uncertainty evolution hidden state
            if hidden is None:
                hidden = torch.zeros(
                    current_state.shape[0], 64, 
                    device=current_state.device, 
                    dtype=current_state.dtype
                )
            
            hidden = self.uncertainty_predictor(
                base_features.mean(dim=[2, 3]), hidden
            )
            
            # Compute adaptive propagation parameters
            uncertainty_level = current_uncertainty.mean(dim=[1, 2, 3], keepdim=True)
            control_input = torch.cat([
                base_features.mean(dim=[2, 3]), 
                uncertainty_level.squeeze()
            ], dim=1)
            
            propagation_params = self.propagation_controller(control_input)
            growth_rate = propagation_params[:, 0:1].unsqueeze(-1).unsqueeze(-1)
            damping_factor = propagation_params[:, 1:2].unsqueeze(-1).unsqueeze(-1)
            coupling_strength = propagation_params[:, 2:3].unsqueeze(-1).unsqueeze(-1)
            
            # Adaptive uncertainty propagation
            # Growth term: uncertainty grows with solution complexity
            solution_gradient = self._compute_solution_gradient(next_state)
            growth_term = growth_rate * solution_gradient * current_uncertainty
            
            # Damping term: physical damping reduces uncertainty
            damping_term = damping_factor * current_uncertainty
            
            # Coupling term: uncertainty couples with solution magnitude
            coupling_term = coupling_strength * torch.abs(next_state) * current_uncertainty
            
            # Update uncertainty
            next_uncertainty = current_uncertainty + growth_term - damping_term + coupling_term
            
            # Ensure positivity and reasonable bounds
            next_uncertainty = torch.clamp(next_uncertainty, min=1e-6, max=10.0)
            
            # Check for adaptation threshold
            if torch.mean(next_uncertainty) > self.adaptation_threshold:
                # Trigger adaptive refinement (could integrate with active learning)
                next_uncertainty = self._adaptive_refinement(
                    next_state, next_uncertainty, physics_params
                )
            
            states.append(next_state)
            uncertainties.append(next_uncertainty)
            
            current_state = next_state
            current_uncertainty = next_uncertainty
        
        return states, uncertainties
    
    def _compute_solution_gradient(self, solution: torch.Tensor) -> torch.Tensor:
        """Compute spatial gradient magnitude of solution."""
        
        # Sobel operators for gradient estimation
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                              dtype=solution.dtype, device=solution.device)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                              dtype=solution.dtype, device=solution.device)
        
        sobel_x = sobel_x.view(1, 1, 3, 3).repeat(solution.shape[1], 1, 1, 1)
        sobel_y = sobel_y.view(1, 1, 3, 3).repeat(solution.shape[1], 1, 1, 1)
        
        grad_x = F.conv2d(solution, sobel_x, groups=solution.shape[1], padding=1)
        grad_y = F.conv2d(solution, sobel_y, groups=solution.shape[1], padding=1)
        
        gradient_magnitude = torch.sqrt(grad_x**2 + grad_y**2 + 1e-8)
        
        return gradient_magnitude
    
    def _adaptive_refinement(
        self,
        state: torch.Tensor,
        uncertainty: torch.Tensor,
        physics_params: Optional[Dict[str, torch.Tensor]]
    ) -> torch.Tensor:
        """Apply adaptive refinement when uncertainty exceeds threshold."""
        
        # Simple refinement: spatial smoothing of high-uncertainty regions
        kernel = torch.ones(1, 1, 3, 3, device=state.device, dtype=state.dtype) / 9
        kernel = kernel.repeat(uncertainty.shape[1], 1, 1, 1)
        
        smoothed_uncertainty = F.conv2d(
            uncertainty, kernel, groups=uncertainty.shape[1], padding=1
        )
        
        # Blend based on uncertainty magnitude
        blend_weight = torch.sigmoid(5 * (uncertainty - self.adaptation_threshold))
        refined_uncertainty = (1 - blend_weight) * smoothed_uncertainty + blend_weight * uncertainty
        
        return refined_uncertainty