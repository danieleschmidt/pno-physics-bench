"""Uncertainty quantification and decomposition for PNO models."""

import torch
import numpy as np
from typing import Dict, Tuple, List, Optional, Union
import matplotlib.pyplot as plt
from scipy import stats
import warnings


class UncertaintyDecomposer:
    """Decompose and analyze uncertainty in probabilistic neural operators."""
    
    def __init__(self, device: Optional[str] = None):
        """Initialize uncertainty decomposer.
        
        Args:
            device: Device for computation (defaults to model device)
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    
    def decompose(
        self,
        model: torch.nn.Module,
        input_data: torch.Tensor,
        num_forward_passes: int = 100,
        return_samples: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Decompose total uncertainty into aleatoric and epistemic components.
        
        Args:
            model: Trained PNO model
            input_data: Input tensor for prediction
            num_forward_passes: Number of forward passes for epistemic uncertainty
            return_samples: Whether to return individual prediction samples
            
        Returns:
            Tuple of (aleatoric_uncertainty, epistemic_uncertainty) or 
            (aleatoric_uncertainty, epistemic_uncertainty, samples) if return_samples=True
        """
        model.eval()
        input_data = input_data.to(self.device)
        
        predictions = []
        aleatoric_variances = []
        
        with torch.no_grad():
            for _ in range(num_forward_passes):
                # Forward pass with parameter sampling
                if hasattr(model, 'forward'):
                    output = model(input_data, sample=True)
                else:
                    output = model(input_data)
                
                if isinstance(output, tuple) and len(output) >= 2:
                    # Model returns mean and log variance
                    pred_mean, pred_log_var = output[:2]
                    pred_var = torch.exp(pred_log_var)
                    
                    # Sample from aleatoric distribution
                    pred_std = torch.sqrt(pred_var + 1e-8)
                    epsilon = torch.randn_like(pred_mean)
                    prediction = pred_mean + pred_std * epsilon
                    
                    predictions.append(prediction)
                    aleatoric_variances.append(pred_var)
                else:
                    # Model only returns mean (no aleatoric uncertainty)
                    predictions.append(output)
                    aleatoric_variances.append(torch.zeros_like(output))
        
        # Stack predictions
        predictions = torch.stack(predictions, dim=0)  # (num_samples, batch, ...)
        aleatoric_variances = torch.stack(aleatoric_variances, dim=0)
        
        # Epistemic uncertainty: variance of means
        epistemic_variance = predictions.var(dim=0)
        epistemic_uncertainty = torch.sqrt(epistemic_variance)
        
        # Aleatoric uncertainty: average of individual variances
        aleatoric_variance = aleatoric_variances.mean(dim=0)
        aleatoric_uncertainty = torch.sqrt(aleatoric_variance)
        
        if return_samples:
            return aleatoric_uncertainty, epistemic_uncertainty, predictions
        else:
            return aleatoric_uncertainty, epistemic_uncertainty
    
    def analyze_by_frequency(
        self,
        model: torch.nn.Module,
        dataset: torch.utils.data.Dataset,
        max_frequency: int = 20,
        num_samples: int = 50,
    ) -> Dict[str, np.ndarray]:
        """Analyze uncertainty as a function of spatial frequency.
        
        Args:
            model: Trained PNO model
            dataset: Dataset to analyze
            max_frequency: Maximum frequency mode to analyze
            num_samples: Number of samples to analyze
            
        Returns:
            Dictionary with frequency analysis results
        """
        model.eval()
        
        # Collect predictions and uncertainties
        total_uncertainties = []
        aleatoric_uncertainties = []
        epistemic_uncertainties = []
        
        with torch.no_grad():
            for i in range(min(num_samples, len(dataset))):
                input_data, _ = dataset[i]
                input_data = input_data.unsqueeze(0).to(self.device)
                
                # Get uncertainty decomposition
                aleatoric, epistemic = self.decompose(
                    model, input_data, num_forward_passes=20
                )
                
                total_uncertainties.append(aleatoric[0] + epistemic[0])
                aleatoric_uncertainties.append(aleatoric[0])
                epistemic_uncertainties.append(epistemic[0])
        
        # Stack all uncertainties
        total_unc = torch.stack(total_uncertainties).cpu().numpy()
        aleatoric_unc = torch.stack(aleatoric_uncertainties).cpu().numpy()
        epistemic_unc = torch.stack(epistemic_uncertainties).cpu().numpy()
        
        # Analyze by frequency
        frequencies = np.arange(1, max_frequency + 1)
        total_by_freq = np.zeros(len(frequencies))
        aleatoric_by_freq = np.zeros(len(frequencies))
        epistemic_by_freq = np.zeros(len(frequencies))
        
        for i, freq in enumerate(frequencies):
            # Extract frequency components using FFT
            total_fft = np.fft.fft2(total_unc, axes=(-2, -1))
            aleatoric_fft = np.fft.fft2(aleatoric_unc, axes=(-2, -1))
            epistemic_fft = np.fft.fft2(epistemic_unc, axes=(-2, -1))
            
            # Get magnitude at this frequency
            total_by_freq[i] = np.mean(np.abs(total_fft[:, :, :freq, :freq]))
            aleatoric_by_freq[i] = np.mean(np.abs(aleatoric_fft[:, :, :freq, :freq]))
            epistemic_by_freq[i] = np.mean(np.abs(epistemic_fft[:, :, :freq, :freq]))
        
        return {
            'frequencies': frequencies,
            'total_uncertainty': total_by_freq,
            'aleatoric_uncertainty': aleatoric_by_freq,
            'epistemic_uncertainty': epistemic_by_freq,
        }
    
    def uncertainty_correlation_analysis(
        self,
        model: torch.nn.Module,
        test_loader: torch.utils.data.DataLoader,
        num_mc_samples: int = 50,
    ) -> Dict[str, float]:
        """Analyze correlation between predicted uncertainty and actual errors.
        
        Args:
            model: Trained PNO model
            test_loader: Test data loader
            num_mc_samples: Number of MC samples for uncertainty estimation
            
        Returns:
            Dictionary with correlation analysis results
        """
        model.eval()
        
        predicted_uncertainties = []
        actual_errors = []
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # Get predictions with uncertainty
                if hasattr(model, 'predict_with_uncertainty'):
                    pred_mean, pred_std = model.predict_with_uncertainty(
                        inputs, num_samples=num_mc_samples
                    )
                else:
                    # Fallback for models without built-in uncertainty
                    predictions = []
                    for _ in range(num_mc_samples):
                        pred = model(inputs, sample=True)
                        if isinstance(pred, tuple):
                            pred = pred[0]
                        predictions.append(pred)
                    
                    predictions = torch.stack(predictions, dim=0)
                    pred_mean = predictions.mean(dim=0)
                    pred_std = predictions.std(dim=0)
                
                # Compute actual errors
                errors = torch.abs(pred_mean - targets)
                
                # Flatten and collect
                predicted_uncertainties.extend(pred_std.cpu().flatten().numpy())
                actual_errors.extend(errors.cpu().flatten().numpy())
        
        # Convert to numpy arrays
        pred_unc = np.array(predicted_uncertainties)
        actual_err = np.array(actual_errors)
        
        # Compute correlations
        pearson_corr, pearson_p = stats.pearsonr(pred_unc, actual_err)
        spearman_corr, spearman_p = stats.spearmanr(pred_unc, actual_err)
        
        # Compute binned statistics for calibration analysis
        n_bins = 20
        bin_edges = np.percentile(pred_unc, np.linspace(0, 100, n_bins + 1))
        bin_indices = np.digitize(pred_unc, bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)
        
        bin_means_unc = []
        bin_means_err = []
        bin_stds_err = []
        
        for i in range(n_bins):
            mask = bin_indices == i
            if np.sum(mask) > 0:
                bin_means_unc.append(np.mean(pred_unc[mask]))
                bin_means_err.append(np.mean(actual_err[mask]))
                bin_stds_err.append(np.std(actual_err[mask]))
            else:
                bin_means_unc.append(0)
                bin_means_err.append(0)
                bin_stds_err.append(0)
        
        return {
            'pearson_correlation': pearson_corr,
            'pearson_p_value': pearson_p,
            'spearman_correlation': spearman_corr,
            'spearman_p_value': spearman_p,
            'bin_uncertainties': np.array(bin_means_unc),
            'bin_errors': np.array(bin_means_err),
            'bin_error_stds': np.array(bin_stds_err),
        }
    
    def plot_frequency_analysis(
        self,
        analysis_results: Dict[str, np.ndarray],
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Plot uncertainty analysis by frequency.
        
        Args:
            analysis_results: Results from analyze_by_frequency
            save_path: Path to save plot
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        frequencies = analysis_results['frequencies']
        
        ax.plot(frequencies, analysis_results['total_uncertainty'], 
                'k-', linewidth=2, label='Total Uncertainty')
        ax.plot(frequencies, analysis_results['aleatoric_uncertainty'], 
                'r--', linewidth=2, label='Aleatoric Uncertainty')
        ax.plot(frequencies, analysis_results['epistemic_uncertainty'], 
                'b:', linewidth=2, label='Epistemic Uncertainty')
        
        ax.set_xlabel('Spatial Frequency')
        ax.set_ylabel('Uncertainty Magnitude')
        ax.set_title('Uncertainty vs Spatial Frequency')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_uncertainty_correlation(
        self,
        correlation_results: Dict[str, Union[float, np.ndarray]],
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Plot uncertainty vs error correlation analysis.
        
        Args:
            correlation_results: Results from uncertainty_correlation_analysis
            save_path: Path to save plot
            
        Returns:
            Matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Calibration plot
        bin_unc = correlation_results['bin_uncertainties']
        bin_err = correlation_results['bin_errors']
        bin_err_std = correlation_results['bin_error_stds']
        
        ax1.errorbar(bin_unc, bin_err, yerr=bin_err_std, 
                    fmt='o-', capsize=3, capthick=1, label='Observed')
        
        # Perfect calibration line
        max_val = max(bin_unc.max(), bin_err.max())
        ax1.plot([0, max_val], [0, max_val], 'r--', alpha=0.7, label='Perfect Calibration')
        
        ax1.set_xlabel('Predicted Uncertainty')
        ax1.set_ylabel('Actual Error')
        ax1.set_title('Uncertainty Calibration')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add correlation info
        pearson_r = correlation_results['pearson_correlation']
        spearman_r = correlation_results['spearman_correlation']
        ax1.text(0.05, 0.95, f'Pearson r: {pearson_r:.3f}\nSpearman ρ: {spearman_r:.3f}',
                transform=ax1.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Residual plot
        residuals = bin_err - bin_unc
        ax2.bar(range(len(residuals)), residuals, alpha=0.7)
        ax2.axhline(y=0, color='r', linestyle='--', alpha=0.7)
        ax2.set_xlabel('Uncertainty Bin')
        ax2.set_ylabel('Error - Uncertainty')
        ax2.set_title('Calibration Residuals')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def compute_uncertainty_quality_metrics(
        self,
        model: torch.nn.Module,
        test_loader: torch.utils.data.DataLoader,
        confidence_levels: List[float] = [0.5, 0.8, 0.9, 0.95],
        num_mc_samples: int = 100,
    ) -> Dict[str, float]:
        """Compute comprehensive uncertainty quality metrics.
        
        Args:
            model: Trained PNO model
            test_loader: Test data loader
            confidence_levels: Confidence levels for coverage analysis
            num_mc_samples: Number of MC samples
            
        Returns:
            Dictionary with uncertainty quality metrics
        """
        model.eval()
        
        all_predictions = []
        all_uncertainties = []
        all_targets = []
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # Get predictions with uncertainty
                if hasattr(model, 'predict_with_uncertainty'):
                    pred_mean, pred_std = model.predict_with_uncertainty(
                        inputs, num_samples=num_mc_samples
                    )
                else:
                    # Fallback
                    predictions = []
                    for _ in range(num_mc_samples):
                        pred = model(inputs, sample=True)
                        if isinstance(pred, tuple):
                            pred = pred[0]
                        predictions.append(pred)
                    
                    predictions = torch.stack(predictions, dim=0)
                    pred_mean = predictions.mean(dim=0)
                    pred_std = predictions.std(dim=0)
                
                all_predictions.append(pred_mean.cpu())
                all_uncertainties.append(pred_std.cpu())
                all_targets.append(targets.cpu())
        
        # Concatenate all results
        predictions = torch.cat(all_predictions, dim=0)
        uncertainties = torch.cat(all_uncertainties, dim=0)
        targets = torch.cat(all_targets, dim=0)
        
        metrics = {}
        
        # Basic error metrics
        errors = torch.abs(predictions - targets)
        metrics['mae'] = errors.mean().item()
        metrics['rmse'] = torch.sqrt(torch.mean((predictions - targets) ** 2)).item()
        
        # Coverage metrics
        for conf_level in confidence_levels:
            z_score = stats.norm.ppf(0.5 + conf_level / 2)  # For normal distribution
            coverage = (errors <= z_score * uncertainties).float().mean().item()
            metrics[f'coverage_{int(conf_level*100)}'] = coverage
        
        # Sharpness (average uncertainty)
        metrics['sharpness'] = uncertainties.mean().item()
        
        # Calibration error (simplified ECE)
        n_bins = 20
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Find samples in this confidence bin
            in_bin = (uncertainties > bin_lower) & (uncertainties <= bin_upper)
            prop_in_bin = in_bin.float().mean()
            
            if prop_in_bin > 0:
                # Average confidence and accuracy in this bin
                confidence_in_bin = uncertainties[in_bin].mean()
                accuracy_in_bin = (errors[in_bin] <= confidence_in_bin).float().mean()
                ece += torch.abs(accuracy_in_bin - confidence_in_bin) * prop_in_bin
        
        metrics['ece'] = ece.item()
        
        # Correlation between uncertainty and error
        unc_flat = uncertainties.flatten()
        err_flat = errors.flatten()
        
        # Convert to numpy for scipy
        unc_np = unc_flat.numpy()
        err_np = err_flat.numpy()
        
        try:
            correlation, _ = stats.pearsonr(unc_np, err_np)
            metrics['uncertainty_error_correlation'] = correlation
        except:
            metrics['uncertainty_error_correlation'] = 0.0
        
        return metrics