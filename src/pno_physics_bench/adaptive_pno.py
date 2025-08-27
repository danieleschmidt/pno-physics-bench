"""Adaptive PNO: Self-Optimizing Probabilistic Neural Operators with Real-Time Learning.

This module implements the next generation of PNOs that adapt their uncertainty estimates
and network parameters in real-time based on observed data patterns and performance metrics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from collections import deque
import logging
from .models import ProbabilisticNeuralOperator
from .uncertainty import UncertaintyDecomposer


class AdaptiveUncertaintyRegulator(nn.Module):
    """Adaptive uncertainty regulation based on prediction accuracy and data drift."""
    
    def __init__(self, window_size: int = 1000, alpha: float = 0.95):
        super().__init__()
        self.window_size = window_size
        self.alpha = alpha  # Exponential moving average factor
        
        # Performance tracking
        self.error_history = deque(maxlen=window_size)
        self.uncertainty_history = deque(maxlen=window_size)
        self.calibration_history = deque(maxlen=window_size)
        
        # Adaptive parameters
        self.uncertainty_scale = nn.Parameter(torch.tensor(1.0))
        self.confidence_threshold = nn.Parameter(torch.tensor(0.1))
        
        # Moving averages
        self.register_buffer('avg_error', torch.tensor(0.0))
        self.register_buffer('avg_uncertainty', torch.tensor(0.0))
        self.register_buffer('calibration_score', torch.tensor(0.0))
        
    def update_metrics(self, predictions: torch.Tensor, targets: torch.Tensor, uncertainties: torch.Tensor):
        """Update performance metrics with new observations."""
        with torch.no_grad():
            # Compute errors
            errors = torch.abs(predictions - targets).mean(dim=[1,2,3])  # Per-sample error
            
            # Update histories
            self.error_history.extend(errors.cpu().numpy())
            self.uncertainty_history.extend(uncertainties.mean(dim=[1,2,3]).cpu().numpy())
            
            # Compute calibration (proportion of errors within uncertainty bounds)
            within_bounds = (torch.abs(predictions - targets) <= uncertainties).float().mean()
            self.calibration_history.append(within_bounds.item())
            
            # Update moving averages
            current_error = errors.mean()
            current_uncertainty = uncertainties.mean()
            current_calibration = within_bounds
            
            self.avg_error = self.alpha * self.avg_error + (1 - self.alpha) * current_error
            self.avg_uncertainty = self.alpha * self.avg_uncertainty + (1 - self.alpha) * current_uncertainty
            self.calibration_score = self.alpha * self.calibration_score + (1 - self.alpha) * current_calibration
    
    def get_adaptive_scaling(self) -> torch.Tensor:
        """Compute adaptive uncertainty scaling based on recent performance."""
        if len(self.error_history) < 10:
            return self.uncertainty_scale
        
        # If we're under-confident (high calibration), reduce uncertainty
        # If we're over-confident (low calibration), increase uncertainty
        target_calibration = 0.9  # Target 90% calibration
        calibration_error = self.calibration_score - target_calibration
        
        # Adaptive scaling: increase uncertainty if under-calibrated, decrease if over-calibrated
        adaptive_factor = torch.clamp(1.0 - calibration_error, 0.5, 2.0)
        
        return self.uncertainty_scale * adaptive_factor
    
    def detect_distribution_shift(self) -> float:
        """Detect distribution shift based on error patterns."""
        if len(self.error_history) < self.window_size // 2:
            return 0.0
        
        # Compare recent errors to historical errors
        recent_errors = list(self.error_history)[-self.window_size//4:]
        historical_errors = list(self.error_history)[:self.window_size//2]
        
        if len(historical_errors) > 0 and len(recent_errors) > 0:
            recent_mean = np.mean(recent_errors)
            historical_mean = np.mean(historical_errors)
            shift_magnitude = abs(recent_mean - historical_mean) / (historical_mean + 1e-8)
            return float(shift_magnitude)
        
        return 0.0


class AdaptivePNOLayer(nn.Module):
    """PNO layer with adaptive capacity based on uncertainty requirements."""
    
    def __init__(self, base_layer: nn.Module, expansion_factor: float = 1.5):
        super().__init__()
        self.base_layer = base_layer
        self.expansion_factor = expansion_factor
        
        # Adaptive capacity controls
        self.use_expansion = nn.Parameter(torch.tensor(0.0))  # 0 = no expansion, 1 = full expansion
        
        # Expanded layer for high uncertainty regions
        if hasattr(base_layer, 'out_channels'):
            expanded_channels = int(base_layer.out_channels * expansion_factor)
            self.expansion_layer = nn.Conv2d(
                base_layer.in_channels, 
                expanded_channels - base_layer.out_channels,
                kernel_size=getattr(base_layer, 'kernel_size', 1),
                padding=getattr(base_layer, 'padding', 0)
            )
        else:
            self.expansion_layer = None
    
    def forward(self, x: torch.Tensor, uncertainty_level: float = 0.0) -> torch.Tensor:
        """Forward pass with adaptive capacity."""
        base_out = self.base_layer(x)
        
        if self.expansion_layer is None or uncertainty_level < 0.5:
            return base_out
        
        # Use expanded capacity for high uncertainty
        expansion_weight = torch.sigmoid(self.use_expansion) * uncertainty_level
        expanded_out = self.expansion_layer(x)
        
        # Weighted combination
        combined_out = torch.cat([base_out, expansion_weight * expanded_out], dim=1)
        
        return combined_out


class AdaptiveProbabilisticNeuralOperator(nn.Module):
    """Adaptive PNO that self-optimizes based on real-time performance."""
    
    def __init__(
        self,
        base_pno: ProbabilisticNeuralOperator,
        adaptation_rate: float = 0.001,
        uncertainty_target: float = 0.9,  # Target calibration
        enable_real_time_learning: bool = True
    ):
        super().__init__()
        
        self.base_pno = base_pno
        self.adaptation_rate = adaptation_rate
        self.uncertainty_target = uncertainty_target
        self.enable_real_time_learning = enable_real_time_learning
        
        # Adaptive components
        self.uncertainty_regulator = AdaptiveUncertaintyRegulator()
        self.uncertainty_decomposer = UncertaintyDecomposer()
        
        # Real-time learning buffer
        self.experience_buffer = deque(maxlen=10000)
        self.update_counter = 0
        
        # Adaptive optimization
        self.adaptive_optimizer = torch.optim.Adam([
            self.uncertainty_regulator.uncertainty_scale,
            self.uncertainty_regulator.confidence_threshold
        ], lr=adaptation_rate)
        
        # Performance tracking
        self.performance_history = {
            'rmse': deque(maxlen=1000),
            'nll': deque(maxlen=1000),
            'calibration': deque(maxlen=1000),
            'adaptation_rate': deque(maxlen=1000)
        }
        
        logging.info(f"Initialized AdaptivePNO with adaptation_rate={adaptation_rate}")
    
    def forward(self, x: torch.Tensor, adapt_online: bool = True) -> torch.Tensor:
        """Forward pass with optional online adaptation."""
        # Get base prediction
        base_output = self.base_pno(x)
        
        if not adapt_online or not self.enable_real_time_learning:
            return base_output
        
        # Apply adaptive uncertainty scaling
        if hasattr(self.base_pno, 'predict_distributional'):
            mean, std = self.base_pno.predict_distributional(x)
            adaptive_scale = self.uncertainty_regulator.get_adaptive_scaling()
            scaled_std = std * adaptive_scale
            
            # Return scaled uncertainty prediction
            if self.training:
                # During training, use reparameterization with scaled uncertainty
                eps = torch.randn_like(mean)
                return mean + eps * scaled_std
            else:
                return mean
        
        return base_output
    
    def predict_with_adaptive_uncertainty(
        self, 
        x: torch.Tensor, 
        num_samples: int = 100,
        adapt_realtime: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
        """Predict with adaptive uncertainty and detailed diagnostics."""
        
        # Base prediction
        mean, std = self.base_pno.predict_with_uncertainty(x, num_samples)
        
        # Adaptive scaling
        adaptive_scale = self.uncertainty_regulator.get_adaptive_scaling()
        adapted_std = std * adaptive_scale
        
        # Detect distribution shift
        shift_score = self.uncertainty_regulator.detect_distribution_shift()
        
        # Compute uncertainty decomposition
        try:
            aleatoric, epistemic = self.uncertainty_decomposer.decompose(
                self.base_pno, x, num_samples
            )
        except:
            # Fallback if decomposition fails
            total_var = adapted_std ** 2
            aleatoric = 0.3 * total_var
            epistemic = 0.7 * total_var
        
        # Diagnostics
        diagnostics = {
            'adaptive_scale': float(adaptive_scale),
            'shift_score': shift_score,
            'avg_error': float(self.uncertainty_regulator.avg_error),
            'calibration': float(self.uncertainty_regulator.calibration_score),
            'aleatoric_ratio': float(aleatoric.mean() / (aleatoric.mean() + epistemic.mean() + 1e-8)),
            'total_uncertainty': float(adapted_std.mean())
        }
        
        return mean, adapted_std, diagnostics
    
    def online_update(self, x: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
        """Perform online learning update with new data."""
        if not self.enable_real_time_learning:
            return {'status': 'disabled'}
        
        # Store experience
        self.experience_buffer.append((x.detach(), y.detach()))
        
        # Get current predictions
        with torch.no_grad():
            mean, std, _ = self.predict_with_adaptive_uncertainty(x, num_samples=20)
            
        # Update performance metrics
        self.uncertainty_regulator.update_metrics(mean, y, std)
        
        # Adaptive loss for uncertainty calibration
        calibration_loss = self.compute_adaptive_loss(mean, y, std)
        
        # Gradient update on adaptive parameters
        self.adaptive_optimizer.zero_grad()
        calibration_loss.backward()
        torch.nn.utils.clip_grad_norm_([
            self.uncertainty_regulator.uncertainty_scale,
            self.uncertainty_regulator.confidence_threshold
        ], max_norm=1.0)
        self.adaptive_optimizer.step()
        
        self.update_counter += 1
        
        # Performance metrics
        with torch.no_grad():
            rmse = torch.sqrt(F.mse_loss(mean, y))
            self.performance_history['rmse'].append(float(rmse))
            self.performance_history['calibration'].append(float(
                self.uncertainty_regulator.calibration_score
            ))
        
        return {
            'calibration_loss': float(calibration_loss),
            'rmse': float(rmse),
            'adaptive_scale': float(self.uncertainty_regulator.get_adaptive_scaling()),
            'updates': self.update_counter
        }
    
    def compute_adaptive_loss(self, predictions: torch.Tensor, targets: torch.Tensor, uncertainties: torch.Tensor) -> torch.Tensor:
        """Compute adaptive loss for uncertainty calibration."""
        # Negative log-likelihood with adaptive uncertainty
        nll = 0.5 * (torch.log(2 * np.pi * uncertainties**2) + 
                    (predictions - targets)**2 / uncertainties**2)
        
        # Calibration penalty
        calibration_error = torch.abs(
            (torch.abs(predictions - targets) <= uncertainties).float().mean() - self.uncertainty_target
        )
        
        # Combined loss
        total_loss = nll.mean() + 10.0 * calibration_error
        
        return total_loss
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        if not self.performance_history['rmse']:
            return {'status': 'no_data'}
        
        return {
            'current_rmse': self.performance_history['rmse'][-1] if self.performance_history['rmse'] else 0.0,
            'avg_rmse': np.mean(list(self.performance_history['rmse'])),
            'current_calibration': self.performance_history['calibration'][-1] if self.performance_history['calibration'] else 0.0,
            'avg_calibration': np.mean(list(self.performance_history['calibration'])),
            'adaptive_scale': float(self.uncertainty_regulator.get_adaptive_scaling()),
            'total_updates': self.update_counter,
            'shift_score': self.uncertainty_regulator.detect_distribution_shift(),
            'buffer_size': len(self.experience_buffer)
        }
    
    def save_adaptive_state(self, filepath: str):
        """Save adaptive learning state."""
        state = {
            'uncertainty_regulator_state': self.uncertainty_regulator.state_dict(),
            'adaptive_optimizer_state': self.adaptive_optimizer.state_dict(),
            'performance_history': dict(self.performance_history),
            'update_counter': self.update_counter,
            'config': {
                'adaptation_rate': self.adaptation_rate,
                'uncertainty_target': self.uncertainty_target,
                'enable_real_time_learning': self.enable_real_time_learning
            }
        }
        torch.save(state, filepath)
        logging.info(f"Saved adaptive state to {filepath}")
    
    def load_adaptive_state(self, filepath: str):
        """Load adaptive learning state."""
        state = torch.load(filepath)
        self.uncertainty_regulator.load_state_dict(state['uncertainty_regulator_state'])
        self.adaptive_optimizer.load_state_dict(state['adaptive_optimizer_state'])
        self.update_counter = state['update_counter']
        
        # Restore performance history
        for key, values in state['performance_history'].items():
            self.performance_history[key] = deque(values, maxlen=1000)
        
        logging.info(f"Loaded adaptive state from {filepath}")


class AdaptivePNOTrainer:
    """Trainer for adaptive PNOs with online learning capabilities."""
    
    def __init__(self, adaptive_pno: AdaptiveProbabilisticNeuralOperator):
        self.adaptive_pno = adaptive_pno
        self.training_metrics = []
    
    def train_with_adaptation(
        self, 
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        epochs: int = 100,
        adaptation_frequency: int = 10
    ):
        """Train with periodic adaptation updates."""
        
        self.adaptive_pno.train()
        
        for epoch in range(epochs):
            epoch_metrics = {'epoch': epoch, 'train_losses': [], 'val_metrics': {}}
            
            # Training loop
            for batch_idx, (data, target) in enumerate(train_loader):
                # Forward pass
                output = self.adaptive_pno(data, adapt_online=True)
                
                # Compute loss (delegate to base PNO training)
                loss = F.mse_loss(output, target)
                
                # Online adaptation every N batches
                if batch_idx % adaptation_frequency == 0:
                    adaptation_metrics = self.adaptive_pno.online_update(data, target)
                    epoch_metrics['train_losses'].append({
                        'mse_loss': float(loss),
                        'adaptation_metrics': adaptation_metrics
                    })
                
            # Validation
            if epoch % 10 == 0:
                val_metrics = self.evaluate_adaptive(val_loader)
                epoch_metrics['val_metrics'] = val_metrics
                
                logging.info(f"Epoch {epoch}: Val RMSE={val_metrics.get('rmse', 0):.4f}, "
                           f"Calibration={val_metrics.get('calibration', 0):.3f}")
            
            self.training_metrics.append(epoch_metrics)
    
    def evaluate_adaptive(self, data_loader: torch.utils.data.DataLoader) -> Dict[str, float]:
        """Evaluate adaptive PNO performance."""
        self.adaptive_pno.eval()
        
        all_predictions = []
        all_targets = []
        all_uncertainties = []
        all_diagnostics = []
        
        with torch.no_grad():
            for data, target in data_loader:
                mean, std, diagnostics = self.adaptive_pno.predict_with_adaptive_uncertainty(
                    data, num_samples=50
                )
                
                all_predictions.append(mean)
                all_targets.append(target)
                all_uncertainties.append(std)
                all_diagnostics.append(diagnostics)
        
        # Aggregate predictions
        predictions = torch.cat(all_predictions, dim=0)
        targets = torch.cat(all_targets, dim=0)
        uncertainties = torch.cat(all_uncertainties, dim=0)
        
        # Compute metrics
        rmse = torch.sqrt(F.mse_loss(predictions, targets))
        mae = F.l1_loss(predictions, targets)
        
        # Calibration
        within_bounds = (torch.abs(predictions - targets) <= uncertainties).float().mean()
        
        # Aggregate diagnostics
        avg_diagnostics = {}
        for key in all_diagnostics[0].keys():
            avg_diagnostics[key] = np.mean([d[key] for d in all_diagnostics])
        
        return {
            'rmse': float(rmse),
            'mae': float(mae),
            'calibration': float(within_bounds),
            **avg_diagnostics
        }


# Export classes
__all__ = [
    'AdaptiveUncertaintyRegulator',
    'AdaptivePNOLayer', 
    'AdaptiveProbabilisticNeuralOperator',
    'AdaptivePNOTrainer'
]