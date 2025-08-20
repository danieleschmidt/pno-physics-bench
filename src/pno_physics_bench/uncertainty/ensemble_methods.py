# SECURITY NOTICE: This file has been automatically hardened for security
# - All user inputs should be validated and sanitized
# - Subprocess calls use secure alternatives
# - SQL queries use parameterized statements
# - No hardcoded secrets or credentials


"""Ensemble methods for improved uncertainty quantification in neural operators."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional, Any, Callable
import numpy as np
from abc import ABC, abstractmethod
from collections import defaultdict
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class EnsembleBase(ABC):
    """Base class for ensemble uncertainty methods."""
    
    def __init__(self, models: List[nn.Module], device: Optional[str] = None):
        """Initialize ensemble base.
        
        Args:
            models: List of trained models
            device: Device for inference
        """
        self.models = models
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_models = len(models)
        
        # Move models to device and set to eval mode
        for model in self.models:
            model.to(self.device)
            model.eval()
    
    @abstractmethod
    def predict(self, x: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict with ensemble uncertainty.
        
        Args:
            x: Input tensor
            
        Returns:
            Tuple of (mean_prediction, uncertainty)
        """
        pass
    
    def save_ensemble(self, save_dir: str):
        """Save ensemble models."""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        for i, model in enumerate(self.models):
            torch.save(model.state_dict(), save_path / f"model_{i}.pt")
            
        # Save ensemble metadata
        metadata = {
            'num_models': self.num_models,
            'model_type': type(self.models[0]).__name__
        }
        torch.save(metadata, save_path / "ensemble_metadata.pt")
        
    @classmethod
    def load_ensemble(cls, load_dir: str, model_class: nn.Module, **kwargs):
        """Load ensemble models."""
        load_path = Path(load_dir)
        metadata = torch.load(load_path / "ensemble_metadata.pt")
        
        models = []
        for i in range(metadata['num_models']):
            model = model_class(**kwargs)
            model.load_state_dict(torch.load(load_path / f"model_{i}.pt"))
            models.append(model)
            
        return cls(models)


class DeepEnsemble(EnsembleBase):
    """Deep ensemble for uncertainty quantification."""
    
    def __init__(
        self,
        models: List[nn.Module],
        device: Optional[str] = None,
        aggregation: str = "mean"
    ):
        """Initialize deep ensemble.
        
        Args:
            models: List of independently trained models
            device: Device for inference
            aggregation: Aggregation method ('mean', 'median', 'trimmed_mean')
        """
        super().__init__(models, device)
        self.aggregation = aggregation
    
    def predict(
        self,
        x: torch.Tensor,
        return_individual: bool = False,
        confidence_level: float = 0.95
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Ensemble prediction with uncertainty.
        
        Args:
            x: Input tensor
            return_individual: Whether to return individual predictions
            confidence_level: Confidence level for uncertainty quantification
            
        Returns:
            Tuple of (ensemble_mean, ensemble_uncertainty)
        """
        x = x.to(self.device)
        predictions = []
        
        with torch.no_grad():
            for model in self.models:
                pred = model(x)
                predictions.append(pred)
        
        predictions = torch.stack(predictions, dim=0)  # [num_models, batch, ...]
        
        # Aggregate predictions
        if self.aggregation == "mean":
            ensemble_mean = torch.mean(predictions, dim=0)
        elif self.aggregation == "median":
            ensemble_mean = torch.median(predictions, dim=0)[0]
        elif self.aggregation == "trimmed_mean":
            # Remove top and bottom 10% of predictions
            trim_size = max(1, self.num_models // 10)
            sorted_preds = torch.sort(predictions, dim=0)[0]
            trimmed = sorted_preds[trim_size:-trim_size] if trim_size > 0 else sorted_preds
            ensemble_mean = torch.mean(trimmed, dim=0)
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation}")
        
        # Calculate uncertainty as variance across models
        ensemble_var = torch.var(predictions, dim=0)
        ensemble_std = torch.sqrt(ensemble_var + 1e-8)
        
        if return_individual:
            return ensemble_mean, ensemble_std, predictions
        else:
            return ensemble_mean, ensemble_std
    
    def predict_quantiles(
        self,
        x: torch.Tensor,
        quantiles: List[float] = [0.05, 0.25, 0.5, 0.75, 0.95]
    ) -> Dict[float, torch.Tensor]:
        """Predict quantiles from ensemble.
        
        Args:
            x: Input tensor
            quantiles: List of quantiles to compute
            
        Returns:
            Dictionary mapping quantiles to predictions
        """
        x = x.to(self.device)
        predictions = []
        
        with torch.no_grad():
            for model in self.models:
                pred = model(x)
                predictions.append(pred)
        
        predictions = torch.stack(predictions, dim=0)
        
        quantile_predictions = {}
        for q in quantiles:
            quantile_predictions[q] = torch.quantile(predictions, q, dim=0)
            
        return quantile_predictions


class MCDropoutEnsemble(EnsembleBase):
    """Monte Carlo Dropout ensemble for uncertainty quantification."""
    
    def __init__(
        self,
        model: nn.Module,
        num_samples: int = 100,
        dropout_rate: float = 0.1,
        device: Optional[str] = None
    ):
        """Initialize MC Dropout ensemble.
        
        Args:
            model: Model with dropout layers
            num_samples: Number of MC samples
            dropout_rate: Dropout rate for inference
            device: Device for inference
        """
        self.model = model
        self.num_samples = num_samples
        self.dropout_rate = dropout_rate
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model.to(self.device)
        self._enable_dropout()
    
    def _enable_dropout(self):
        """Enable dropout during inference."""
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                module.p = self.dropout_rate
                module.train()
    
    def predict(
        self,
        x: torch.Tensor,
        return_samples: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """MC Dropout prediction with uncertainty.
        
        Args:
            x: Input tensor
            return_samples: Whether to return all samples
            
        Returns:
            Tuple of (mean_prediction, uncertainty)
        """
        x = x.to(self.device)
        self._enable_dropout()
        
        samples = []
        with torch.no_grad():
            for _ in range(self.num_samples):
                sample = self.model(x)
                samples.append(sample)
        
        samples = torch.stack(samples, dim=0)
        
        # Calculate mean and uncertainty
        mean_pred = torch.mean(samples, dim=0)
        uncertainty = torch.std(samples, dim=0)
        
        if return_samples:
            return mean_pred, uncertainty, samples
        else:
            return mean_pred, uncertainty


class SnapshotEnsemble(EnsembleBase):
    """Snapshot ensemble using cyclic learning rate scheduling."""
    
    def __init__(
        self,
        model_class: nn.Module,
        snapshots: List[str],
        device: Optional[str] = None,
        **model_kwargs
    ):
        """Initialize snapshot ensemble.
        
        Args:
            model_class: Model class for initialization
            snapshots: List of paths to snapshot checkpoints
            device: Device for inference
            **model_kwargs: Arguments for model initialization
        """
        models = []
        for snapshot_path in snapshots:
            model = model_class(**model_kwargs)
            model.load_state_dict(torch.load(snapshot_path, map_location='cpu'))
            models.append(model)
            
        super().__init__(models, device)
    
    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Snapshot ensemble prediction."""
        x = x.to(self.device)
        predictions = []
        
        with torch.no_grad():
            for model in self.models:
                pred = model(x)
                predictions.append(pred)
        
        predictions = torch.stack(predictions, dim=0)
        
        # Weighted average based on snapshot ordering (later snapshots get higher weight)
        weights = torch.softmax(torch.arange(len(self.models), dtype=torch.float), dim=0)
        weights = weights.view(-1, *[1] * (predictions.dim() - 1)).to(self.device)
        
        ensemble_mean = torch.sum(weights * predictions, dim=0)
        ensemble_std = torch.sqrt(torch.sum(weights * (predictions - ensemble_mean.unsqueeze(0))**2, dim=0))
        
        return ensemble_mean, ensemble_std


class VariationalEnsemble(EnsembleBase):
    """Ensemble of variational models for enhanced uncertainty quantification."""
    
    def __init__(
        self,
        models: List[nn.Module],
        num_samples_per_model: int = 10,
        device: Optional[str] = None
    ):
        """Initialize variational ensemble.
        
        Args:
            models: List of variational models
            num_samples_per_model: MC samples per model
            device: Device for inference
        """
        super().__init__(models, device)
        self.num_samples_per_model = num_samples_per_model
    
    def predict(
        self,
        x: torch.Tensor,
        decompose_uncertainty: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """Variational ensemble prediction with uncertainty decomposition.
        
        Args:
            x: Input tensor
            decompose_uncertainty: Whether to decompose aleatoric/epistemic
            
        Returns:
            Tuple of (mean_prediction, total_uncertainty, decomposition)
        """
        x = x.to(self.device)
        
        all_samples = []
        model_means = []
        model_stds = []
        
        with torch.no_grad():
            for model in self.models:
                # Get samples from each variational model
                model_samples = []
                for _ in range(self.num_samples_per_model):
                    sample = model(x, sample=True)
                    model_samples.append(sample)
                    all_samples.append(sample)
                
                model_samples = torch.stack(model_samples, dim=0)
                model_mean = torch.mean(model_samples, dim=0)
                model_std = torch.std(model_samples, dim=0)
                
                model_means.append(model_mean)
                model_stds.append(model_std)
        
        # Convert to tensors
        all_samples = torch.stack(all_samples, dim=0)
        model_means = torch.stack(model_means, dim=0)
        model_stds = torch.stack(model_stds, dim=0)
        
        # Overall ensemble mean
        ensemble_mean = torch.mean(all_samples, dim=0)
        
        if decompose_uncertainty:
            # Aleatoric uncertainty: average of within-model variances
            aleatoric = torch.mean(model_stds**2, dim=0)
            
            # Epistemic uncertainty: variance of model means
            epistemic = torch.var(model_means, dim=0)
            
            # Total uncertainty
            total_uncertainty = torch.sqrt(aleatoric + epistemic)
            
            decomposition = {
                'aleatoric': torch.sqrt(aleatoric),
                'epistemic': torch.sqrt(epistemic),
                'total': total_uncertainty
            }
            
            return ensemble_mean, total_uncertainty, decomposition
        else:
            total_uncertainty = torch.std(all_samples, dim=0)
            return ensemble_mean, total_uncertainty, None


class AdaptiveEnsemble:
    """Adaptive ensemble that dynamically weights models based on performance."""
    
    def __init__(
        self,
        models: List[nn.Module],
        adaptation_window: int = 100,
        device: Optional[str] = None
    ):
        """Initialize adaptive ensemble.
        
        Args:
            models: List of models
            adaptation_window: Window for performance tracking
            device: Device for inference
        """
        self.models = models
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.adaptation_window = adaptation_window
        
        # Performance tracking
        self.model_performances = defaultdict(list)
        self.model_weights = torch.ones(len(models)) / len(models)
        
        # Move models to device
        for model in self.models:
            model.to(self.device)
            model.eval()
    
    def update_weights(self, predictions: List[torch.Tensor], targets: torch.Tensor):
        """Update model weights based on recent performance.
        
        Args:
            predictions: List of model predictions
            targets: Ground truth targets
        """
        # Calculate per-model losses
        for i, pred in enumerate(predictions):
            loss = F.mse_loss(pred, targets).item()
            self.model_performances[i].append(loss)
            
            # Keep only recent performance
            if len(self.model_performances[i]) > self.adaptation_window:
                self.model_performances[i] = self.model_performances[i][-self.adaptation_window:]
        
        # Update weights based on inverse of average loss
        avg_losses = []
        for i in range(len(self.models)):
            if self.model_performances[i]:
                avg_loss = np.mean(self.model_performances[i])
                avg_losses.append(avg_loss)
            else:
                avg_losses.append(1.0)  # Default loss
        
        # Inverse weighting (lower loss = higher weight)
        inv_losses = [1.0 / (loss + 1e-8) for loss in avg_losses]
        total = sum(inv_losses)
        self.model_weights = torch.tensor([w / total for w in inv_losses])
        
    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Adaptive ensemble prediction.
        
        Args:
            x: Input tensor
            
        Returns:
            Tuple of (weighted_mean, uncertainty)
        """
        x = x.to(self.device)
        predictions = []
        
        with torch.no_grad():
            for model in self.models:
                pred = model(x)
                predictions.append(pred)
        
        predictions = torch.stack(predictions, dim=0)
        weights = self.model_weights.view(-1, *[1] * (predictions.dim() - 1)).to(self.device)
        
        # Weighted ensemble mean
        weighted_mean = torch.sum(weights * predictions, dim=0)
        
        # Weighted uncertainty
        weighted_var = torch.sum(weights * (predictions - weighted_mean.unsqueeze(0))**2, dim=0)
        uncertainty = torch.sqrt(weighted_var + 1e-8)
        
        return weighted_mean, uncertainty


def create_ensemble(
    ensemble_type: str,
    models_or_paths: List,
    **kwargs
) -> EnsembleBase:
    """Factory function for creating ensemble methods.
    
    Args:
        ensemble_type: Type of ensemble ('deep', 'snapshot', 'mc_dropout', 'variational')
        models_or_paths: List of models or model paths
        **kwargs: Additional arguments for ensemble
        
    Returns:
        Configured ensemble instance
    """
    if ensemble_type == "deep":
        return DeepEnsemble(models_or_paths, **kwargs)
    elif ensemble_type == "snapshot":
        return SnapshotEnsemble(models_or_paths, **kwargs)
    elif ensemble_type == "mc_dropout":
        if len(models_or_paths) != 1:
            raise ValueError("MC Dropout ensemble requires exactly one model")
        return MCDropoutEnsemble(models_or_paths[0], **kwargs)
    elif ensemble_type == "variational":
        return VariationalEnsemble(models_or_paths, **kwargs)
    elif ensemble_type == "adaptive":
        return AdaptiveEnsemble(models_or_paths, **kwargs)
    else:
        raise ValueError(f"Unknown ensemble type: {ensemble_type}")


def ensemble_calibration_test(
    ensemble: EnsembleBase,
    test_loader,
    num_bins: int = 20
) -> Dict[str, float]:
    """Test ensemble calibration.
    
    Args:
        ensemble: Trained ensemble
        test_loader: Test data loader
        num_bins: Number of bins for calibration
        
    Returns:
        Calibration metrics
    """
    all_predictions = []
    all_uncertainties = []
    all_targets = []
    
    with torch.no_grad():
        for batch in test_loader:
            x, y = batch
            pred, unc = ensemble.predict(x)
            
            all_predictions.append(pred)
            all_uncertainties.append(unc)
            all_targets.append(y)
    
    predictions = torch.cat(all_predictions, dim=0)
    uncertainties = torch.cat(all_uncertainties, dim=0)
    targets = torch.cat(all_targets, dim=0)
    
    # Calculate calibration metrics
    errors = torch.abs(predictions - targets)
    
    # Bin by uncertainty
    uncertainty_flat = uncertainties.flatten()
    errors_flat = errors.flatten()
    
    bins = torch.linspace(0, uncertainty_flat.max(), num_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    calibration_errors = []
    for i in range(num_bins):
        mask = (uncertainty_flat >= bins[i]) & (uncertainty_flat < bins[i + 1])
        if mask.sum() > 0:
            bin_uncertainty = uncertainty_flat[mask].mean()
            bin_error = errors_flat[mask].mean()
            calibration_errors.append(abs(bin_uncertainty - bin_error))
    
    ece = np.mean(calibration_errors) if calibration_errors else 0.0
    
    return {
        'expected_calibration_error': ece,
        'mean_uncertainty': uncertainty_flat.mean().item(),
        'mean_error': errors_flat.mean().item(),
        'rmse': torch.sqrt(torch.mean(errors_flat**2)).item()
    }