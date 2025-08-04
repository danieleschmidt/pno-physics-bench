"""Loss functions for training Probabilistic Neural Operators."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import math


class PNOLoss(nn.Module):
    """Base loss class for Probabilistic Neural Operators."""
    
    def __init__(
        self,
        reconstruction_weight: float = 1.0,
        kl_weight: float = 1e-4,
        calibration_weight: float = 0.1,
    ):
        super().__init__()
        self.reconstruction_weight = reconstruction_weight
        self.kl_weight = kl_weight
        self.calibration_weight = calibration_weight
    
    def forward(self, predictions, targets, model=None) -> Dict[str, torch.Tensor]:
        """Compute loss components.
        
        Args:
            predictions: Model predictions (can be tuple for mean/var or single tensor)
            targets: Ground truth targets
            model: Model instance for computing KL divergence
            
        Returns:
            Dictionary of loss components
        """
        losses = {}
        
        # Handle different prediction formats
        if isinstance(predictions, tuple):
            if len(predictions) == 2:
                pred_mean, pred_log_var = predictions
                pred_std = torch.exp(0.5 * pred_log_var)
            elif len(predictions) == 3:
                pred_mean, pred_log_var, kl_div = predictions
                pred_std = torch.exp(0.5 * pred_log_var)
                losses['kl'] = kl_div
            else:
                raise ValueError(f"Invalid prediction format: {len(predictions)} elements")
        else:
            pred_mean = predictions
            pred_std = None
        
        # Reconstruction loss
        mse_loss = F.mse_loss(pred_mean, targets, reduction='mean')
        losses['reconstruction'] = mse_loss
        
        # Negative log-likelihood (if uncertainty is predicted)
        if pred_std is not None:
            nll = self._negative_log_likelihood(pred_mean, pred_std, targets)
            losses['nll'] = nll
            
            # Calibration loss
            if self.calibration_weight > 0:
                cal_loss = self._calibration_loss(pred_mean, pred_std, targets)
                losses['calibration'] = cal_loss
        
        # KL divergence (if not already computed)
        if 'kl' not in losses and model is not None:
            if hasattr(model, 'compute_kl_divergence'):
                losses['kl'] = model.compute_kl_divergence()
            else:
                losses['kl'] = torch.tensor(0.0, device=pred_mean.device)
        
        # Total loss
        total_loss = self.reconstruction_weight * losses['reconstruction']
        
        if 'nll' in losses:
            total_loss += losses['nll']  # NLL already includes reconstruction
            total_loss -= self.reconstruction_weight * losses['reconstruction']  # Avoid double counting
        
        if 'kl' in losses:
            total_loss += self.kl_weight * losses['kl']
        
        if 'calibration' in losses:
            total_loss += self.calibration_weight * losses['calibration']
        
        losses['total'] = total_loss
        
        return losses
    
    def _negative_log_likelihood(
        self, 
        pred_mean: torch.Tensor, 
        pred_std: torch.Tensor, 
        targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute negative log-likelihood assuming Gaussian distribution."""
        # Ensure numerical stability
        pred_std = torch.clamp(pred_std, min=1e-6)
        
        # Gaussian NLL: 0.5 * (log(2π) + 2*log(σ) + (y-μ)²/σ²)
        squared_error = (targets - pred_mean) ** 2
        variance = pred_std ** 2
        
        nll = 0.5 * (
            math.log(2 * math.pi) + 
            2 * torch.log(pred_std) + 
            squared_error / variance
        )
        
        return nll.mean()
    
    def _calibration_loss(
        self,
        pred_mean: torch.Tensor,
        pred_std: torch.Tensor, 
        targets: torch.Tensor,
        confidence_levels: Optional[Tuple[float, ...]] = None
    ) -> torch.Tensor:
        """Compute calibration loss to encourage well-calibrated uncertainties."""
        if confidence_levels is None:
            confidence_levels = (0.5, 0.9, 0.95)
        
        cal_loss = 0.0
        
        for conf_level in confidence_levels:
            # Z-score for confidence level
            z_score = torch.distributions.Normal(0, 1).icdf(torch.tensor(0.5 + conf_level / 2))
            
            # Expected coverage
            expected_coverage = conf_level
            
            # Actual coverage
            residuals = torch.abs(targets - pred_mean)
            prediction_intervals = z_score * pred_std
            within_interval = (residuals <= prediction_intervals).float()
            actual_coverage = within_interval.mean()
            
            # Calibration error
            cal_error = (actual_coverage - expected_coverage) ** 2
            cal_loss += cal_error
        
        return cal_loss / len(confidence_levels)


class ELBOLoss(PNOLoss):
    """Evidence Lower Bound (ELBO) loss for variational inference."""
    
    def __init__(
        self,
        beta: float = 1.0,
        num_samples: int = 1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.beta = beta  # β-VAE parameter
        self.num_samples = num_samples
    
    def forward(self, predictions, targets, model=None) -> Dict[str, torch.Tensor]:
        """Compute ELBO loss."""
        losses = super().forward(predictions, targets, model)
        
        # Scale KL divergence by β
        if 'kl' in losses:
            losses['kl'] = self.beta * losses['kl']
            
            # Recompute total loss
            total_loss = losses['reconstruction']
            if 'nll' in losses:
                total_loss = losses['nll']  # NLL includes reconstruction
            total_loss += self.kl_weight * losses['kl']
            if 'calibration' in losses:
                total_loss += self.calibration_weight * losses['calibration']
            
            losses['total'] = total_loss
        
        return losses


class CalibrationLoss(nn.Module):
    """Standalone calibration loss for uncertainty quality."""
    
    def __init__(
        self,
        confidence_levels: Tuple[float, ...] = (0.5, 0.8, 0.9, 0.95),
        sharpness_weight: float = 0.1,
    ):
        super().__init__()
        self.confidence_levels = confidence_levels
        self.sharpness_weight = sharpness_weight
    
    def forward(
        self,
        pred_mean: torch.Tensor,
        pred_std: torch.Tensor,
        targets: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Compute calibration-related losses."""
        losses = {}
        
        # Coverage calibration
        cal_loss = 0.0
        for conf_level in self.confidence_levels:
            z_score = torch.distributions.Normal(0, 1).icdf(
                torch.tensor(0.5 + conf_level / 2, device=pred_mean.device)
            )
            
            expected_coverage = conf_level
            residuals = torch.abs(targets - pred_mean)
            prediction_intervals = z_score * pred_std
            within_interval = (residuals <= prediction_intervals).float()
            actual_coverage = within_interval.mean()
            
            cal_error = (actual_coverage - expected_coverage) ** 2
            cal_loss += cal_error
        
        losses['calibration'] = cal_loss / len(self.confidence_levels)
        
        # Sharpness penalty (encourage tight predictions when certain)
        if self.sharpness_weight > 0:
            sharpness = pred_std.mean()
            losses['sharpness'] = self.sharpness_weight * sharpness
        
        # Interval score (proper scoring rule)
        interval_score = self._interval_score(pred_mean, pred_std, targets)
        losses['interval_score'] = interval_score
        
        # Total
        total_loss = losses['calibration']
        if 'sharpness' in losses:
            total_loss += losses['sharpness']
        total_loss += losses['interval_score']
        
        losses['total'] = total_loss
        
        return losses
    
    def _interval_score(
        self,
        pred_mean: torch.Tensor,
        pred_std: torch.Tensor,
        targets: torch.Tensor,
        alpha: float = 0.1  # For 90% prediction intervals
    ) -> torch.Tensor:
        """Compute interval score (lower is better)."""
        z_alpha = torch.distributions.Normal(0, 1).icdf(
            torch.tensor(1 - alpha / 2, device=pred_mean.device)
        )
        
        # Prediction interval bounds
        lower = pred_mean - z_alpha * pred_std
        upper = pred_mean + z_alpha * pred_std
        
        # Interval score components
        interval_width = upper - lower
        lower_penalty = 2 * alpha * torch.clamp(lower - targets, min=0)
        upper_penalty = 2 * alpha * torch.clamp(targets - upper, min=0)
        
        interval_score = interval_width + lower_penalty + upper_penalty
        return interval_score.mean()


class ContrastivePredictionLoss(nn.Module):
    """Contrastive loss for better uncertainty estimation."""
    
    def __init__(
        self,
        temperature: float = 0.1,
        margin: float = 1.0,
    ):
        super().__init__()
        self.temperature = temperature
        self.margin = margin
    
    def forward(
        self,
        pred_mean: torch.Tensor,
        pred_std: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """Contrastive loss to separate good/bad predictions by uncertainty."""
        batch_size = pred_mean.shape[0]
        
        # Compute prediction errors
        errors = torch.abs(pred_mean - targets).mean(dim=(1, 2, 3))  # Mean over spatial dims
        uncertainties = pred_std.mean(dim=(1, 2, 3))
        
        # Create positive/negative pairs
        loss = 0.0
        num_pairs = 0
        
        for i in range(batch_size):
            for j in range(i + 1, batch_size):
                error_i, error_j = errors[i], errors[j]
                unc_i, unc_j = uncertainties[i], uncertainties[j]
                
                # If error_i > error_j, then unc_i should be > unc_j
                if error_i > error_j:
                    # Positive pair: higher error should have higher uncertainty
                    diff = unc_j - unc_i + self.margin
                    loss += torch.clamp(diff, min=0)
                else:
                    # Negative pair: lower error should have lower uncertainty  
                    diff = unc_i - unc_j + self.margin
                    loss += torch.clamp(diff, min=0)
                
                num_pairs += 1
        
        return loss / max(num_pairs, 1)


class AdaptiveLossWeighting(nn.Module):
    """Adaptive loss weighting based on uncertainty quality."""
    
    def __init__(self, base_loss: nn.Module, adaptation_rate: float = 0.01):
        super().__init__()
        self.base_loss = base_loss
        self.adaptation_rate = adaptation_rate
        
        # Learnable weights
        self.log_weight_recon = nn.Parameter(torch.tensor(0.0))
        self.log_weight_kl = nn.Parameter(torch.tensor(-5.0))  # Start with low KL weight
        self.log_weight_cal = nn.Parameter(torch.tensor(-2.0))
    
    def forward(self, predictions, targets, model=None) -> Dict[str, torch.Tensor]:
        """Compute loss with adaptive weighting."""
        # Get base losses
        losses = self.base_loss.forward(predictions, targets, model)
        
        # Convert log weights to actual weights
        weight_recon = torch.exp(self.log_weight_recon)
        weight_kl = torch.exp(self.log_weight_kl)
        weight_cal = torch.exp(self.log_weight_cal)
        
        # Recompute total loss with adaptive weights
        total_loss = weight_recon * losses['reconstruction']
        
        if 'nll' in losses:
            total_loss = losses['nll']  # NLL includes reconstruction
            total_loss -= weight_recon * losses['reconstruction']  # Adjust for double counting
        
        if 'kl' in losses:
            total_loss += weight_kl * losses['kl']
        
        if 'calibration' in losses:
            total_loss += weight_cal * losses['calibration']
        
        losses['total'] = total_loss
        losses['weight_recon'] = weight_recon
        losses['weight_kl'] = weight_kl
        losses['weight_cal'] = weight_cal
        
        return losses