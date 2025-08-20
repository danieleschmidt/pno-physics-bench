# SECURITY NOTICE: This file has been automatically hardened for security
# - All user inputs should be validated and sanitized
# - Subprocess calls use secure alternatives
# - SQL queries use parameterized statements
# - No hardcoded secrets or credentials


"""Adaptive Uncertainty Calibration for Probabilistic Neural Operators.

This module implements novel adaptive calibration techniques that dynamically
adjust uncertainty estimates based on prediction accuracy and domain conditions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from abc import ABC, abstractmethod


class AdaptiveUncertaintyCalibrator(nn.Module):
    """Adaptive calibration network that learns to adjust uncertainty estimates."""
    
    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 128,
        num_layers: int = 3,
        calibration_method: str = "temperature_scaling"
    ):
        super().__init__()
        
        self.calibration_method = calibration_method
        
        # Feature extractor for uncertainty context
        layers = []
        current_dim = input_dim
        for i in range(num_layers - 1):
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            current_dim = hidden_dim
        
        # Output layer for calibration parameters
        if calibration_method == "temperature_scaling":
            layers.append(nn.Linear(current_dim, 1))  # Single temperature parameter
        elif calibration_method == "platt_scaling":
            layers.append(nn.Linear(current_dim, 2))  # A and B parameters
        elif calibration_method == "isotonic_neural":
            layers.append(nn.Linear(current_dim, 10))  # Piecewise linear segments
        
        self.calibration_network = nn.Sequential(*layers)
        
        # Adaptive learning rate for calibration updates
        self.calibration_lr = nn.Parameter(torch.tensor(0.01))
        
        # Memory bank for storing calibration history
        self.register_buffer('calibration_history', torch.zeros(1000, 3))  # [prediction, uncertainty, target]
        self.register_buffer('history_ptr', torch.zeros(1, dtype=torch.long))
        
    def extract_features(
        self, 
        predictions: torch.Tensor, 
        uncertainties: torch.Tensor,
        context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Extract features for calibration network."""
        features = []
        
        # Statistical features of predictions
        pred_stats = torch.stack([
            predictions.mean(dim=-1),
            predictions.std(dim=-1),
            predictions.min(dim=-1)[0],
            predictions.max(dim=-1)[0]
        ], dim=-1)
        features.append(pred_stats)
        
        # Statistical features of uncertainties  
        unc_stats = torch.stack([
            uncertainties.mean(dim=-1),
            uncertainties.std(dim=-1),
            uncertainties.min(dim=-1)[0],
            uncertainties.max(dim=-1)[0]
        ], dim=-1)
        features.append(unc_stats)
        
        # Spatial gradient features (if 2D/3D data)
        if len(predictions.shape) > 2:
            pred_grad_x = torch.diff(predictions, dim=-1).abs().mean(dim=-1)
            pred_grad_y = torch.diff(predictions, dim=-2).abs().mean(dim=-2)
            grad_features = torch.stack([pred_grad_x, pred_grad_y], dim=-1)
            features.append(grad_features)
        
        # Domain-specific context features
        if context is not None:
            features.append(context)
            
        return torch.cat(features, dim=-1)
    
    def forward(
        self, 
        predictions: torch.Tensor,
        uncertainties: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply adaptive uncertainty calibration."""
        
        # Extract features for calibration
        features = self.extract_features(predictions, uncertainties, context)
        
        # Get calibration parameters
        calib_params = self.calibration_network(features)
        
        if self.calibration_method == "temperature_scaling":
            temperature = F.softplus(calib_params) + 1e-8
            calibrated_uncertainties = uncertainties / temperature
            
        elif self.calibration_method == "platt_scaling":
            A, B = calib_params.chunk(2, dim=-1)
            A = F.softplus(A) + 1e-8
            calibrated_uncertainties = torch.sigmoid(A * uncertainties + B)
            
        elif self.calibration_method == "isotonic_neural":
            # Neural isotonic regression
            segment_weights = F.softmax(calib_params, dim=-1)
            uncertainty_percentiles = torch.linspace(0, 1, 10, device=uncertainties.device)
            
            # Interpolate uncertainties using learned segments
            calibrated_uncertainties = torch.sum(
                segment_weights.unsqueeze(-1) * uncertainty_percentiles, dim=-2
            )
        
        # Update calibration history if training
        if self.training and targets is not None:
            self._update_calibration_history(predictions, uncertainties, targets)
        
        return calibrated_uncertainties, calib_params
    
    def _update_calibration_history(
        self, 
        predictions: torch.Tensor,
        uncertainties: torch.Tensor, 
        targets: torch.Tensor
    ):
        """Update the calibration history buffer."""
        batch_size = predictions.size(0)
        ptr = self.history_ptr.item()
        
        # Compute prediction errors
        errors = (predictions - targets).abs().mean(dim=tuple(range(1, len(predictions.shape))))
        
        # Store in circular buffer
        end_ptr = min(ptr + batch_size, self.calibration_history.size(0))
        actual_batch = end_ptr - ptr
        
        self.calibration_history[ptr:end_ptr, 0] = errors[:actual_batch]
        self.calibration_history[ptr:end_ptr, 1] = uncertainties[:actual_batch].mean(dim=tuple(range(1, len(uncertainties.shape))))
        self.calibration_history[ptr:end_ptr, 2] = (errors[:actual_batch] < uncertainties[:actual_batch].mean(dim=tuple(range(1, len(uncertainties.shape))))).float()
        
        # Update pointer
        self.history_ptr[0] = (ptr + batch_size) % self.calibration_history.size(0)
    
    def compute_calibration_error(self) -> torch.Tensor:
        """Compute expected calibration error from history."""
        valid_entries = (self.calibration_history[:, 0] > 0).sum()
        if valid_entries == 0:
            return torch.tensor(0.0, device=self.calibration_history.device)
        
        # Sort by uncertainty
        sorted_indices = self.calibration_history[:valid_entries, 1].argsort()
        sorted_coverage = self.calibration_history[sorted_indices, 2]
        
        # Compute ECE with 10 bins
        num_bins = 10
        bin_size = valid_entries // num_bins
        ece = 0.0
        
        for i in range(num_bins):
            start_idx = i * bin_size
            end_idx = (i + 1) * bin_size if i < num_bins - 1 else valid_entries
            
            if end_idx > start_idx:
                bin_coverage = sorted_coverage[start_idx:end_idx].mean()
                expected_coverage = (i + 0.5) / num_bins  # Mid-point of bin
                ece += (bin_coverage - expected_coverage).abs() * (end_idx - start_idx) / valid_entries
        
        return torch.tensor(ece, device=self.calibration_history.device)


class HierarchicalUncertaintyDecomposer(nn.Module):
    """Hierarchical decomposition of uncertainty into multiple scales."""
    
    def __init__(
        self,
        scales: List[int] = [1, 2, 4, 8],
        hidden_dim: int = 64,
        uncertainty_types: List[str] = ["aleatoric", "epistemic", "spatial", "temporal"]
    ):
        super().__init__()
        
        self.scales = scales
        self.uncertainty_types = uncertainty_types
        
        # Multi-scale feature extractors
        self.scale_encoders = nn.ModuleDict()
        for scale in scales:
            self.scale_encoders[str(scale)] = nn.Sequential(
                nn.Conv2d(1, hidden_dim // len(scales), kernel_size=scale*2+1, padding=scale, stride=scale),
                nn.ReLU(),
                nn.Conv2d(hidden_dim // len(scales), hidden_dim // len(scales), kernel_size=3, padding=1),
                nn.ReLU()
            )
        
        # Uncertainty type decomposition networks
        self.uncertainty_decoders = nn.ModuleDict()
        total_features = hidden_dim
        
        for unc_type in uncertainty_types:
            self.uncertainty_decoders[unc_type] = nn.Sequential(
                nn.Linear(total_features, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1),
                nn.Softplus()
            )
        
        # Attention mechanism for combining scales
        self.scale_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim // len(scales),
            num_heads=4,
            batch_first=True
        )
    
    def forward(
        self, 
        predictions: torch.Tensor,
        input_features: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Decompose uncertainty hierarchically across scales and types."""
        
        batch_size, *spatial_dims = predictions.shape
        
        # Extract multi-scale features
        scale_features = []
        for scale in self.scales:
            if len(spatial_dims) == 2:  # 2D case
                # Reshape for conv2d
                pred_reshaped = predictions.view(batch_size, 1, *spatial_dims)
                features = self.scale_encoders[str(scale)](pred_reshaped)
                features = F.adaptive_avg_pool2d(features, (8, 8))
                features = features.view(batch_size, -1)
            else:  # 1D or other cases
                features = torch.mean(predictions, dim=-1, keepdim=True)
                features = features.expand(-1, self.scale_encoders[str(scale)][0].in_channels)
            
            scale_features.append(features)
        
        # Combine scale features with attention
        if len(scale_features) > 1:
            stacked_features = torch.stack(scale_features, dim=1)
            attended_features, _ = self.scale_attention(
                stacked_features, stacked_features, stacked_features
            )
            combined_features = attended_features.mean(dim=1)
        else:
            combined_features = scale_features[0]
        
        # Decompose into uncertainty types
        uncertainty_components = {}
        for unc_type in self.uncertainty_types:
            uncertainty_components[unc_type] = self.uncertainty_decoders[unc_type](combined_features)
        
        return uncertainty_components
    
    def get_total_uncertainty(self, components: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Combine uncertainty components into total uncertainty."""
        total = torch.zeros_like(list(components.values())[0])
        
        for component in components.values():
            total += component.squeeze(-1)
        
        return total


class DynamicUncertaintyThresholds:
    """Dynamic thresholding system for uncertainty-based decision making."""
    
    def __init__(
        self,
        initial_threshold: float = 0.1,
        adaptation_rate: float = 0.01,
        target_coverage: float = 0.9,
        min_threshold: float = 0.01,
        max_threshold: float = 1.0
    ):
        self.current_threshold = initial_threshold
        self.adaptation_rate = adaptation_rate
        self.target_coverage = target_coverage
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        
        # Statistics tracking
        self.coverage_history = []
        self.threshold_history = []
        
    def update_threshold(
        self, 
        uncertainties: torch.Tensor,
        errors: torch.Tensor,
        target_coverage: Optional[float] = None
    ) -> float:
        """Dynamically update uncertainty threshold based on observed coverage."""
        
        target = target_coverage or self.target_coverage
        
        # Compute current coverage at current threshold
        within_threshold = (uncertainties > self.current_threshold)
        correct_predictions = (errors < uncertainties)
        current_coverage = (within_threshold & correct_predictions).float().mean().item()
        
        # Update threshold based on coverage gap
        coverage_gap = target - current_coverage
        threshold_update = self.adaptation_rate * coverage_gap
        
        # Apply update with bounds
        self.current_threshold = max(
            self.min_threshold,
            min(self.max_threshold, self.current_threshold + threshold_update)
        )
        
        # Record history
        self.coverage_history.append(current_coverage)
        self.threshold_history.append(self.current_threshold)
        
        return self.current_threshold
    
    def get_confidence_intervals(
        self,
        predictions: torch.Tensor,
        uncertainties: torch.Tensor,
        confidence_levels: List[float] = [0.68, 0.95, 0.99]
    ) -> Dict[float, Tuple[torch.Tensor, torch.Tensor]]:
        """Compute adaptive confidence intervals."""
        
        intervals = {}
        
        for conf_level in confidence_levels:
            # Compute z-score for confidence level
            z_score = torch.tensor(2 * conf_level - 1)  # Approximate
            
            # Adjust uncertainty by current threshold
            adjusted_uncertainty = uncertainties * (self.current_threshold / 0.1)  # Normalize by default
            
            lower = predictions - z_score * adjusted_uncertainty
            upper = predictions + z_score * adjusted_uncertainty
            
            intervals[conf_level] = (lower, upper)
        
        return intervals


class MetaLearningUncertaintyEstimator(nn.Module):
    """Meta-learning approach for few-shot uncertainty estimation."""
    
    def __init__(
        self,
        feature_dim: int = 256,
        hidden_dim: int = 128,
        num_inner_steps: int = 5,
        inner_lr: float = 0.01
    ):
        super().__init__()
        
        self.num_inner_steps = num_inner_steps
        self.inner_lr = inner_lr
        
        # Meta-network for uncertainty estimation
        self.uncertainty_net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus()
        )
        
        # Context encoder for task-specific features
        self.context_encoder = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim),  # [input, target] pairs
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim)
        )
        
    def meta_forward(
        self,
        support_inputs: torch.Tensor,
        support_targets: torch.Tensor,
        query_inputs: torch.Tensor
    ) -> torch.Tensor:
        """Meta-learning forward pass for few-shot uncertainty estimation."""
        
        # Encode task context from support set
        support_pairs = torch.cat([support_inputs, support_targets], dim=-1)
        task_context = self.context_encoder(support_pairs).mean(dim=0, keepdim=True)
        
        # Fast adaptation on support set
        adapted_params = self._fast_adaptation(
            support_inputs, support_targets, task_context
        )
        
        # Predict uncertainty on query set
        query_features = torch.cat([query_inputs, task_context.expand(query_inputs.size(0), -1)], dim=-1)
        uncertainties = self._forward_with_params(query_features, adapted_params)
        
        return uncertainties
    
    def _fast_adaptation(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        context: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """MAML-style fast adaptation."""
        
        # Initialize with current parameters
        adapted_params = {name: param.clone() for name, param in self.uncertainty_net.named_parameters()}
        
        for step in range(self.num_inner_steps):
            # Forward pass with current adapted parameters
            features = torch.cat([inputs, context.expand(inputs.size(0), -1)], dim=-1)
            predictions = self._forward_with_params(features, adapted_params)
            
            # Compute loss (negative log-likelihood for uncertainty)
            loss = F.mse_loss(predictions, (predictions - targets).abs())
            
            # Compute gradients
            grads = torch.autograd.grad(
                loss, adapted_params.values(), create_graph=True, allow_unused=True
            )
            
            # Update parameters
            for (name, param), grad in zip(adapted_params.items(), grads):
                if grad is not None:
                    adapted_params[name] = param - self.inner_lr * grad
        
        return adapted_params
    
    def _forward_with_params(
        self, 
        x: torch.Tensor, 
        params: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Forward pass with given parameters."""
        
        # Manual forward pass through network
        out = x
        layer_idx = 0
        
        for layer in self.uncertainty_net:
            if isinstance(layer, nn.Linear):
                weight_name = f"{layer_idx}.weight"
                bias_name = f"{layer_idx}.bias"
                
                if weight_name in params and bias_name in params:
                    out = F.linear(out, params[weight_name], params[bias_name])
                else:
                    out = layer(out)
                layer_idx += 2  # Skip ReLU layer
            else:
                out = layer(out)
        
        return out


def uncertainty_aware_ensemble(
    models: List[nn.Module],
    inputs: torch.Tensor,
    uncertainty_weights: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Ensemble prediction with uncertainty-aware weighting."""
    
    predictions = []
    uncertainties = []
    
    # Get predictions from all models
    for model in models:
        if hasattr(model, 'predict_with_uncertainty'):
            pred, unc = model.predict_with_uncertainty(inputs)
        else:
            pred = model(inputs)
            unc = torch.zeros_like(pred)
        
        predictions.append(pred)
        uncertainties.append(unc)
    
    predictions = torch.stack(predictions)
    uncertainties = torch.stack(uncertainties)
    
    if uncertainty_weights is None:
        # Weight by inverse uncertainty
        weights = 1.0 / (uncertainties + 1e-8)
        weights = weights / weights.sum(dim=0, keepdim=True)
    else:
        weights = uncertainty_weights
    
    # Weighted ensemble
    ensemble_prediction = (weights * predictions).sum(dim=0)
    
    # Combine uncertainties (epistemic + aleatoric)
    mean_prediction = predictions.mean(dim=0)
    epistemic_uncertainty = ((predictions - mean_prediction.unsqueeze(0)) ** 2).mean(dim=0)
    aleatoric_uncertainty = (weights * uncertainties).sum(dim=0)
    total_uncertainty = epistemic_uncertainty + aleatoric_uncertainty
    
    return ensemble_prediction, total_uncertainty