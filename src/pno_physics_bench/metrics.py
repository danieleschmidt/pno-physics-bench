"""Calibration and uncertainty quality metrics for PNO evaluation."""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from scipy import stats
import warnings
import matplotlib.pyplot as plt


class CalibrationMetrics:
    """Comprehensive calibration metrics for uncertainty quantification."""
    
    def __init__(self, device: Optional[str] = None):
        """Initialize calibration metrics.
        
        Args:
            device: Device for computation
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    
    def expected_calibration_error(
        self,
        predictions: torch.Tensor,
        uncertainties: torch.Tensor,
        targets: torch.Tensor,
        num_bins: int = 20,
        norm: str = 'l1',
    ) -> float:
        """Compute Expected Calibration Error (ECE).
        
        Args:
            predictions: Model predictions
            uncertainties: Predicted uncertainties (standard deviations)
            targets: Ground truth targets
            num_bins: Number of bins for calibration histogram
            norm: Norm to use ('l1', 'l2', or 'max')
            
        Returns:
            ECE value
        """
        # Flatten tensors
        predictions = predictions.flatten()
        uncertainties = uncertainties.flatten()
        targets = targets.flatten()
        
        # Compute absolute errors
        errors = torch.abs(predictions - targets)
        
        # Create confidence bins
        confidence_scores = 1 - 2 * torch.distributions.Normal(0, 1).log_prob(
            errors / (uncertainties + 1e-8)
        ).exp()
        confidence_scores = torch.clamp(confidence_scores, 0, 1)
        
        bin_boundaries = torch.linspace(0, 1, num_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0.0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Identify samples in this bin
            in_bin = (confidence_scores > bin_lower) & (confidence_scores <= bin_upper)
            prop_in_bin = in_bin.float().mean()
            
            if prop_in_bin > 0:
                # Average confidence and accuracy in this bin
                confidence_in_bin = confidence_scores[in_bin].mean()
                
                # For regression, "accuracy" is 1 - normalized error
                normalized_errors = errors[in_bin] / (uncertainties[in_bin] + 1e-8)
                accuracy_in_bin = (normalized_errors <= 1.0).float().mean()
                
                # Calibration error in this bin
                if norm == 'l1':
                    bin_error = torch.abs(accuracy_in_bin - confidence_in_bin)
                elif norm == 'l2':
                    bin_error = (accuracy_in_bin - confidence_in_bin) ** 2
                elif norm == 'max':
                    bin_error = torch.abs(accuracy_in_bin - confidence_in_bin)
                
                ece += bin_error * prop_in_bin
        
        if norm == 'l2':
            ece = torch.sqrt(ece)
        
        return ece.item()
    
    def coverage_at_confidence(
        self,
        predictions: torch.Tensor,
        uncertainties: torch.Tensor,
        targets: torch.Tensor,
        confidence: float = 0.9,
    ) -> float:
        """Compute coverage at a specific confidence level.
        
        Args:
            predictions: Model predictions
            uncertainties: Predicted uncertainties
            targets: Ground truth targets
            confidence: Confidence level (e.g., 0.9 for 90%)
            
        Returns:
            Actual coverage fraction
        """
        # Z-score for given confidence level
        z_score = stats.norm.ppf(0.5 + confidence / 2)
        
        # Compute prediction intervals
        errors = torch.abs(predictions - targets)
        intervals = z_score * uncertainties
        
        # Count samples within intervals
        within_interval = (errors <= intervals).float()
        coverage = within_interval.mean().item()
        
        return coverage
    
    def interval_score(
        self,
        predictions: torch.Tensor,
        uncertainties: torch.Tensor,
        targets: torch.Tensor,
        alpha: float = 0.1,  # For 90% prediction intervals
    ) -> float:
        """Compute interval score (proper scoring rule).
        
        Args:
            predictions: Model predictions (mean)
            uncertainties: Predicted uncertainties (std)
            targets: Ground truth targets
            alpha: Miscoverage level (1-alpha is the coverage level)
            
        Returns:
            Average interval score (lower is better)
        """
        # Compute prediction interval bounds
        z_alpha = stats.norm.ppf(1 - alpha / 2)
        lower = predictions - z_alpha * uncertainties
        upper = predictions + z_alpha * uncertainties
        
        # Interval score components
        interval_width = upper - lower
        lower_penalty = (2 / alpha) * torch.clamp(lower - targets, min=0)
        upper_penalty = (2 / alpha) * torch.clamp(targets - upper, min=0)
        
        interval_scores = interval_width + lower_penalty + upper_penalty
        return interval_scores.mean().item()
    
    def sharpness(self, uncertainties: torch.Tensor) -> float:
        """Compute sharpness (average predicted uncertainty).
        
        Args:
            uncertainties: Predicted uncertainties
            
        Returns:
            Average uncertainty (lower is sharper)
        """
        return uncertainties.mean().item()
    
    def reliability_diagram_data(
        self,
        predictions: torch.Tensor,
        uncertainties: torch.Tensor,
        targets: torch.Tensor,
        num_bins: int = 20,
    ) -> Dict[str, np.ndarray]:
        """Compute data for reliability diagram.
        
        Args:
            predictions: Model predictions
            uncertainties: Predicted uncertainties
            targets: Ground truth targets
            num_bins: Number of bins
            
        Returns:
            Dictionary with bin data for plotting
        """
        # Flatten tensors
        predictions = predictions.flatten()
        uncertainties = uncertainties.flatten()
        targets = targets.flatten()
        
        # Compute confidence scores (1 - normalized error probability)
        errors = torch.abs(predictions - targets)
        normalized_errors = errors / (uncertainties + 1e-8)
        
        # Use CDF of standard normal for confidence
        confidence_scores = 2 * torch.distributions.Normal(0, 1).cdf(
            -normalized_errors
        )  # P(|Z| <= |error|/uncertainty)
        confidence_scores = torch.clamp(confidence_scores, 0, 1)
        
        # Create bins
        bin_boundaries = torch.linspace(0, 1, num_bins + 1)
        bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2
        
        bin_confidences = []
        bin_accuracies = []
        bin_counts = []
        
        for i in range(num_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]
            
            # Find samples in this bin
            in_bin = (confidence_scores > bin_lower) & (confidence_scores <= bin_upper)
            
            if in_bin.sum() > 0:
                # Average confidence in bin
                avg_confidence = confidence_scores[in_bin].mean().item()
                
                # Average accuracy in bin (for regression: fraction with small normalized error)
                normalized_errors_in_bin = normalized_errors[in_bin]
                avg_accuracy = (normalized_errors_in_bin <= 1.0).float().mean().item()
                
                bin_confidences.append(avg_confidence)
                bin_accuracies.append(avg_accuracy)
                bin_counts.append(in_bin.sum().item())
            else:
                bin_confidences.append(bin_centers[i].item())
                bin_accuracies.append(0.0)
                bin_counts.append(0)
        
        return {
            'bin_centers': bin_centers.numpy(),
            'bin_confidences': np.array(bin_confidences),
            'bin_accuracies': np.array(bin_accuracies),
            'bin_counts': np.array(bin_counts),
        }
    
    def plot_reliability_diagram(
        self,
        predictions: torch.Tensor,
        uncertainties: torch.Tensor,
        targets: torch.Tensor,
        num_bins: int = 20,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """Plot reliability diagram for calibration visualization.
        
        Args:
            predictions: Model predictions
            uncertainties: Predicted uncertainties
            targets: Ground truth targets
            num_bins: Number of bins
            save_path: Path to save plot
            
        Returns:
            Matplotlib figure
        """
        # Get reliability data
        data = self.reliability_diagram_data(predictions, uncertainties, targets, num_bins)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Reliability diagram
        ax1.bar(data['bin_centers'], data['bin_accuracies'], 
               width=1/num_bins, alpha=0.7, edgecolor='black', linewidth=0.5)
        ax1.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect Calibration')
        ax1.set_xlabel('Confidence')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Reliability Diagram')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        
        # Histogram of confidence scores
        ax2.bar(data['bin_centers'], data['bin_counts'], 
               width=1/num_bins, alpha=0.7, edgecolor='black', linewidth=0.5)
        ax2.set_xlabel('Confidence')
        ax2.set_ylabel('Count')
        ax2.set_title('Confidence Distribution')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def negative_log_likelihood(
        self,
        predictions: torch.Tensor,
        uncertainties: torch.Tensor,
        targets: torch.Tensor,
    ) -> float:
        """Compute negative log-likelihood assuming Gaussian predictive distribution.
        
        Args:
            predictions: Model predictions (mean)
            uncertainties: Predicted uncertainties (std)
            targets: Ground truth targets
            
        Returns:
            Average negative log-likelihood
        """
        # Ensure numerical stability
        uncertainties = torch.clamp(uncertainties, min=1e-6)
        
        # Gaussian NLL: 0.5 * (log(2π) + 2*log(σ) + (y-μ)²/σ²)
        squared_error = (targets - predictions) ** 2
        variance = uncertainties ** 2
        
        nll = 0.5 * (
            np.log(2 * np.pi) + 
            2 * torch.log(uncertainties) + 
            squared_error / variance
        )
        
        return nll.mean().item()
    
    def compute_all_metrics(
        self,
        predictions: torch.Tensor,
        uncertainties: torch.Tensor,
        targets: torch.Tensor,
        confidence_levels: List[float] = [0.5, 0.8, 0.9, 0.95, 0.99],
    ) -> Dict[str, float]:
        """Compute all calibration and uncertainty metrics.
        
        Args:
            predictions: Model predictions
            uncertainties: Predicted uncertainties
            targets: Ground truth targets
            confidence_levels: List of confidence levels for coverage
            
        Returns:
            Dictionary with all metrics
        """
        metrics = {}
        
        try:
            # Basic error metrics
            errors = torch.abs(predictions - targets)
            metrics['mae'] = errors.mean().item()
            metrics['rmse'] = torch.sqrt(torch.mean((predictions - targets) ** 2)).item()
            
            # Calibration metrics
            metrics['ece'] = self.expected_calibration_error(predictions, uncertainties, targets)
            metrics['nll'] = self.negative_log_likelihood(predictions, uncertainties, targets)
            metrics['sharpness'] = self.sharpness(uncertainties)
            
            # Coverage at different confidence levels
            for conf_level in confidence_levels:
                coverage = self.coverage_at_confidence(
                    predictions, uncertainties, targets, conf_level
                )
                metrics[f'coverage_{int(conf_level*100)}'] = coverage
            
            # Interval scores at different levels
            for alpha in [0.1, 0.05, 0.01]:  # 90%, 95%, 99% intervals
                score = self.interval_score(predictions, uncertainties, targets, alpha)
                metrics[f'interval_score_{int((1-alpha)*100)}'] = score
            
            # Correlation between uncertainty and error
            unc_flat = uncertainties.flatten()
            err_flat = errors.flatten()
            
            if len(unc_flat) > 1 and len(err_flat) > 1:
                # Convert to numpy for scipy
                unc_np = unc_flat.cpu().numpy() if hasattr(unc_flat, 'cpu') else unc_flat.numpy()
                err_np = err_flat.cpu().numpy() if hasattr(err_flat, 'cpu') else err_flat.numpy()
                
                correlation, _ = stats.pearsonr(unc_np, err_np)
                metrics['uncertainty_error_correlation'] = correlation
            else:
                metrics['uncertainty_error_correlation'] = 0.0
                
        except Exception as e:
            warnings.warn(f"Error computing metrics: {e}")
            # Return basic metrics only
            errors = torch.abs(predictions - targets)
            metrics = {
                'mae': errors.mean().item(),
                'rmse': torch.sqrt(torch.mean((predictions - targets) ** 2)).item(),
            }
        
        return metrics
    
    def miscalibration_area(
        self,
        predictions: torch.Tensor,
        uncertainties: torch.Tensor,
        targets: torch.Tensor,
        num_bins: int = 20,
    ) -> float:
        """Compute Miscalibration Area (area between reliability curve and diagonal).
        
        Args:
            predictions: Model predictions
            uncertainties: Predicted uncertainties
            targets: Ground truth targets
            num_bins: Number of bins
            
        Returns:
            Miscalibration area
        """
        data = self.reliability_diagram_data(predictions, uncertainties, targets, num_bins)
        
        # Compute area between reliability curve and perfect calibration
        confidences = data['bin_confidences']
        accuracies = data['bin_accuracies']
        counts = data['bin_counts']
        
        # Weight by number of samples in each bin
        total_samples = counts.sum()
        if total_samples == 0:
            return 0.0
        
        weights = counts / total_samples
        miscalibration = np.abs(accuracies - confidences)
        
        return np.sum(weights * miscalibration)
    
    def prediction_interval_coverage_probability(
        self,
        predictions: torch.Tensor,
        uncertainties: torch.Tensor,
        targets: torch.Tensor,
        alpha_levels: List[float] = [0.1, 0.05, 0.01],
    ) -> Dict[str, float]:
        """Compute Prediction Interval Coverage Probability (PICP).
        
        Args:
            predictions: Model predictions
            uncertainties: Predicted uncertainties
            targets: Ground truth targets
            alpha_levels: Miscoverage levels (1-alpha is nominal coverage)
            
        Returns:
            Dictionary with PICP for each level
        """
        picp_results = {}
        
        for alpha in alpha_levels:
            coverage_level = 1 - alpha
            actual_coverage = self.coverage_at_confidence(
                predictions, uncertainties, targets, coverage_level
            )
            picp_results[f'picp_{int(coverage_level*100)}'] = actual_coverage
        
        return picp_results
    
    def mean_prediction_interval_width(
        self,
        uncertainties: torch.Tensor,
        alpha_levels: List[float] = [0.1, 0.05, 0.01],
    ) -> Dict[str, float]:
        """Compute Mean Prediction Interval Width (MPIW).
        
        Args:
            uncertainties: Predicted uncertainties
            alpha_levels: Miscoverage levels
            
        Returns:
            Dictionary with MPIW for each level
        """
        mpiw_results = {}
        
        for alpha in alpha_levels:
            z_alpha = stats.norm.ppf(1 - alpha / 2)
            interval_width = 2 * z_alpha * uncertainties
            mpiw_results[f'mpiw_{int((1-alpha)*100)}'] = interval_width.mean().item()
        
        return mpiw_results