"""Calibration and uncertainty quality metrics for PNO evaluation."""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from scipy import stats
import warnings
import matplotlib.pyplot as plt
from collections import defaultdict
import seaborn as sns


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
    
    def uncertainty_quality_index(
        self,
        predictions: torch.Tensor,
        uncertainties: torch.Tensor,
        targets: torch.Tensor,
        alpha: float = 0.95
    ) -> float:
        """Compute Uncertainty Quality Index (UQI) combining coverage and sharpness.
        
        Args:
            predictions: Model predictions
            uncertainties: Predicted uncertainties
            targets: Ground truth targets
            alpha: Confidence level
            
        Returns:
            UQI score (higher is better)
        """
        # Coverage component
        coverage = self.prediction_interval_coverage_probability(
            predictions, uncertainties, targets, [1 - alpha]
        )[f'picp_{int(alpha*100)}']
        
        # Sharpness component (inverted and normalized)
        sharpness_raw = self.sharpness(uncertainties)
        max_possible_uncertainty = targets.std().item()
        sharpness_normalized = 1 - (sharpness_raw / max_possible_uncertainty)
        sharpness_normalized = max(0, min(1, sharpness_normalized))
        
        # Combined UQI (weighted harmonic mean of coverage and sharpness)
        uqi = 2 * coverage * sharpness_normalized / (coverage + sharpness_normalized + 1e-8)
        
        return uqi
    
    def adaptive_calibration_error(
        self,
        predictions: torch.Tensor,
        uncertainties: torch.Tensor,
        targets: torch.Tensor,
        adaptive_bins: bool = True
    ) -> Dict[str, float]:
        """Compute Adaptive Calibration Error with variable bin sizes.
        
        Args:
            predictions: Model predictions
            uncertainties: Predicted uncertainties
            targets: Ground truth targets
            adaptive_bins: Whether to use adaptive binning
            
        Returns:
            Dictionary with ACE metrics
        """
        errors = torch.abs(predictions - targets)
        uncertainty_flat = uncertainties.flatten()
        errors_flat = errors.flatten()
        
        if adaptive_bins:
            # Use quantile-based adaptive binning
            quantiles = torch.linspace(0, 1, 21)  # 20 adaptive bins
            bin_edges = torch.quantile(uncertainty_flat, quantiles)
        else:
            # Use uniform binning
            bin_edges = torch.linspace(uncertainty_flat.min(), uncertainty_flat.max(), 21)
        
        adaptive_errors = []
        bin_sizes = []
        
        for i in range(len(bin_edges) - 1):
            mask = (uncertainty_flat >= bin_edges[i]) & (uncertainty_flat < bin_edges[i + 1])
            if i == len(bin_edges) - 2:  # Include last edge
                mask |= (uncertainty_flat == bin_edges[i + 1])
            
            if mask.sum() > 0:
                bin_uncertainty = uncertainty_flat[mask].mean()
                bin_error = errors_flat[mask].mean()
                bin_size = mask.sum().item()
                
                adaptive_errors.append(abs(bin_uncertainty - bin_error))
                bin_sizes.append(bin_size)
        
        # Weighted ACE by bin sizes
        total_samples = len(uncertainty_flat)
        weights = [size / total_samples for size in bin_sizes]
        ace = sum(w * err for w, err in zip(weights, adaptive_errors))
        
        return {
            'adaptive_calibration_error': ace,
            'num_adaptive_bins': len(adaptive_errors),
            'avg_bin_size': np.mean(bin_sizes) if bin_sizes else 0
        }
    
    def uncertainty_decomposition_metrics(
        self,
        aleatoric_uncertainty: torch.Tensor,
        epistemic_uncertainty: torch.Tensor,
        total_uncertainty: torch.Tensor
    ) -> Dict[str, float]:
        """Compute metrics for uncertainty decomposition quality.
        
        Args:
            aleatoric_uncertainty: Aleatoric (data) uncertainty
            epistemic_uncertainty: Epistemic (model) uncertainty
            total_uncertainty: Total uncertainty
            
        Returns:
            Dictionary with decomposition metrics
        """
        # Flatten tensors
        aleatoric = aleatoric_uncertainty.flatten()
        epistemic = epistemic_uncertainty.flatten()
        total = total_uncertainty.flatten()
        
        # Verify decomposition consistency
        reconstructed_total = torch.sqrt(aleatoric**2 + epistemic**2)
        decomposition_error = torch.mean(torch.abs(total - reconstructed_total)).item()
        
        # Uncertainty ratios
        aleatoric_ratio = torch.mean(aleatoric / (total + 1e-8)).item()
        epistemic_ratio = torch.mean(epistemic / (total + 1e-8)).item()
        
        # Correlation between uncertainty types
        corr_coef = torch.corrcoef(torch.stack([aleatoric, epistemic]))[0, 1].item()
        
        # Dominance analysis
        aleatoric_dominant = (aleatoric > epistemic).float().mean().item()
        epistemic_dominant = (epistemic > aleatoric).float().mean().item()
        
        return {
            'decomposition_error': decomposition_error,
            'aleatoric_ratio': aleatoric_ratio,
            'epistemic_ratio': epistemic_ratio,
            'uncertainty_correlation': corr_coef,
            'aleatoric_dominance': aleatoric_dominant,
            'epistemic_dominance': epistemic_dominant
        }
    
    def frequency_domain_calibration(
        self,
        predictions: torch.Tensor,
        uncertainties: torch.Tensor,
        targets: torch.Tensor,
        max_frequency: int = 50
    ) -> Dict[str, List[float]]:
        """Analyze calibration in frequency domain (for spatial/temporal data).
        
        Args:
            predictions: Model predictions [batch, channels, height, width]
            uncertainties: Predicted uncertainties
            targets: Ground truth targets
            max_frequency: Maximum frequency component to analyze
            
        Returns:
            Dictionary with frequency-wise calibration metrics
        """
        if len(predictions.shape) < 3:
            raise ValueError("Frequency domain analysis requires spatial/temporal data")
        
        # Take 2D FFT
        pred_fft = torch.fft.fft2(predictions)
        unc_fft = torch.fft.fft2(uncertainties)
        target_fft = torch.fft.fft2(targets)
        
        # Get frequency magnitudes
        pred_mag = torch.abs(pred_fft)
        unc_mag = torch.abs(unc_fft)
        target_mag = torch.abs(target_fft)
        
        h, w = pred_fft.shape[-2:]
        freq_errors = []
        freq_uncertainties = []
        
        # Analyze by frequency rings
        center_h, center_w = h // 2, w // 2
        max_freq = min(max_frequency, min(center_h, center_w))
        
        for freq in range(1, max_freq):
            # Create frequency mask for current ring
            y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
            dist = torch.sqrt((y - center_h)**2 + (x - center_w)**2)
            mask = (dist >= freq - 0.5) & (dist < freq + 0.5)
            
            if mask.sum() > 0:
                freq_pred = pred_mag[..., mask].mean()
                freq_unc = unc_mag[..., mask].mean()
                freq_target = target_mag[..., mask].mean()
                
                freq_error = torch.abs(freq_pred - freq_target)
                freq_errors.append(freq_error.item())
                freq_uncertainties.append(freq_unc.item())
        
        return {
            'frequency_errors': freq_errors,
            'frequency_uncertainties': freq_uncertainties,
            'frequency_calibration_error': np.mean([abs(e - u) for e, u in zip(freq_errors, freq_uncertainties)])
        }
    
    def temporal_calibration_drift(
        self,
        predictions_sequence: List[torch.Tensor],
        uncertainties_sequence: List[torch.Tensor],
        targets_sequence: List[torch.Tensor]
    ) -> Dict[str, float]:
        """Analyze calibration drift over time/iteration sequences.
        
        Args:
            predictions_sequence: List of predictions over time
            uncertainties_sequence: List of uncertainties over time
            targets_sequence: List of targets over time
            
        Returns:
            Dictionary with temporal drift metrics
        """
        calibration_errors = []
        
        for pred, unc, target in zip(predictions_sequence, uncertainties_sequence, targets_sequence):
            ece = self.expected_calibration_error(pred, unc, target, num_bins=10)
            calibration_errors.append(ece)
        
        # Analyze drift
        calibration_trend = np.polyfit(range(len(calibration_errors)), calibration_errors, 1)[0]
        calibration_variance = np.var(calibration_errors)
        calibration_stability = 1 / (1 + calibration_variance)
        
        return {
            'calibration_drift_rate': calibration_trend,
            'calibration_variance': calibration_variance,
            'calibration_stability': calibration_stability,
            'initial_calibration': calibration_errors[0] if calibration_errors else 0,
            'final_calibration': calibration_errors[-1] if calibration_errors else 0
        }
    
    def multi_scale_calibration(
        self,
        predictions: torch.Tensor,
        uncertainties: torch.Tensor,
        targets: torch.Tensor,
        scales: List[int] = [1, 2, 4, 8]
    ) -> Dict[str, Dict[str, float]]:
        """Analyze calibration at multiple spatial scales.
        
        Args:
            predictions: Model predictions [batch, channels, height, width]
            uncertainties: Predicted uncertainties
            targets: Ground truth targets
            scales: List of downsampling scales
            
        Returns:
            Dictionary with scale-wise calibration metrics
        """
        scale_metrics = {}
        
        for scale in scales:
            if scale == 1:
                # Original scale
                scale_pred, scale_unc, scale_target = predictions, uncertainties, targets
            else:
                # Downsample
                scale_pred = F.avg_pool2d(predictions, kernel_size=scale, stride=scale)
                scale_unc = F.avg_pool2d(uncertainties, kernel_size=scale, stride=scale)
                scale_target = F.avg_pool2d(targets, kernel_size=scale, stride=scale)
            
            # Compute calibration at this scale
            ece = self.expected_calibration_error(scale_pred, scale_unc, scale_target)
            coverage = self.prediction_interval_coverage_probability(
                scale_pred, scale_unc, scale_target, [0.1]
            )['picp_90']
            
            scale_metrics[f'scale_{scale}'] = {
                'ece': ece,
                'coverage_90': coverage,
                'spatial_resolution': f"{scale_pred.shape[-2]}x{scale_pred.shape[-1]}"
            }
        
        return scale_metrics
    
    def comprehensive_uncertainty_report(
        self,
        predictions: torch.Tensor,
        uncertainties: torch.Tensor,
        targets: torch.Tensor,
        aleatoric_uncertainty: Optional[torch.Tensor] = None,
        epistemic_uncertainty: Optional[torch.Tensor] = None,
        save_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate comprehensive uncertainty quality report.
        
        Args:
            predictions: Model predictions
            uncertainties: Total predicted uncertainties
            targets: Ground truth targets
            aleatoric_uncertainty: Aleatoric uncertainty component
            epistemic_uncertainty: Epistemic uncertainty component
            save_path: Path to save report
            
        Returns:
            Comprehensive metrics dictionary
        """
        report = {}
        
        # Basic calibration metrics
        report['calibration'] = {
            'ece': self.expected_calibration_error(predictions, uncertainties, targets),
            'ace': self.adaptive_calibration_error(predictions, uncertainties, targets),
            'mce': self.miscalibration_area(predictions, uncertainties, targets)
        }
        
        # Coverage and interval metrics
        report['coverage'] = self.prediction_interval_coverage_probability(
            predictions, uncertainties, targets
        )
        
        # Quality indices
        report['quality'] = {
            'uqi': self.uncertainty_quality_index(predictions, uncertainties, targets),
            'sharpness': self.sharpness(uncertainties),
            'nll': self.negative_log_likelihood(predictions, uncertainties, targets)
        }
        
        # Uncertainty decomposition (if available)
        if aleatoric_uncertainty is not None and epistemic_uncertainty is not None:
            report['decomposition'] = self.uncertainty_decomposition_metrics(
                aleatoric_uncertainty, epistemic_uncertainty, uncertainties
            )
        
        # Frequency domain analysis (if spatial data)
        if len(predictions.shape) >= 3:
            try:
                report['frequency'] = self.frequency_domain_calibration(
                    predictions, uncertainties, targets
                )
            except Exception as e:
                warnings.warn(f"Frequency domain analysis failed: {e}")
        
        # Save report if requested
        if save_path:
            torch.save(report, save_path)
        
        return report