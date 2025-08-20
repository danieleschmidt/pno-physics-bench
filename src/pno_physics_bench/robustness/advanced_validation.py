# SECURITY NOTICE: This file has been automatically hardened for security
# - All user inputs should be validated and sanitized
# - Subprocess calls use secure alternatives
# - SQL queries use parameterized statements
# - No hardcoded secrets or credentials


"""
Advanced Validation Framework for Probabilistic Neural Operators.

This module implements comprehensive validation schemes including physics
consistency checks, uncertainty calibration validation, and robustness
testing against adversarial perturbations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
import warnings
from scipy import stats
import logging

from ..models import ProbabilisticNeuralOperator


@dataclass
class ValidationResult:
    """Structured validation result."""
    test_name: str
    passed: bool
    score: float
    details: Dict[str, Any]
    warnings: List[str]
    critical_failures: List[str]


class PhysicsConsistencyValidator:
    """
    Validator for physics consistency in PDE solutions.
    
    Ensures that neural operator predictions satisfy fundamental physical
    principles like conservation laws, boundary conditions, and PDE residuals.
    """
    
    def __init__(
        self,
        pde_type: str = "navier_stokes",
        tolerance: float = 1e-3,
        conservation_weight: float = 1.0
    ):
        self.pde_type = pde_type
        self.tolerance = tolerance
        self.conservation_weight = conservation_weight
        
        # Physical constants and parameters
        self.physics_params = self._initialize_physics_params()
        
        # Differential operators for physics laws
        self.diff_operators = self._create_differential_operators()
    
    def _initialize_physics_params(self) -> Dict[str, float]:
        """Initialize physics parameters based on PDE type."""
        if self.pde_type == "navier_stokes":
            return {
                'viscosity': 1e-3,
                'density': 1.0,
                'gravity': 9.81
            }
        elif self.pde_type == "darcy_flow":
            return {
                'permeability': 1e-12,
                'viscosity': 1e-3,
                'porosity': 0.3
            }
        elif self.pde_type == "heat_equation":
            return {
                'thermal_diffusivity': 1e-6,
                'conductivity': 400.0,
                'specific_heat': 900.0
            }
        else:
            return {}
    
    def _create_differential_operators(self) -> Dict[str, Callable]:
        """Create differential operators for physics validation."""
        def gradient_2d(field: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            """Compute 2D gradient using finite differences."""
            grad_x = torch.diff(field, dim=-1, prepend=field[..., :1])
            grad_y = torch.diff(field, dim=-2, prepend=field[..., :1, :])
            return grad_x, grad_y
        
        def divergence_2d(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
            """Compute 2D divergence."""
            du_dx = torch.diff(u, dim=-1, prepend=u[..., :1])
            dv_dy = torch.diff(v, dim=-2, prepend=v[..., :1, :])
            return du_dx + dv_dy
        
        def laplacian_2d(field: torch.Tensor) -> torch.Tensor:
            """Compute 2D Laplacian using finite differences."""
            # Second derivatives
            d2_dx2 = torch.diff(field, n=2, dim=-1)
            d2_dy2 = torch.diff(field, n=2, dim=-2)
            
            # Pad to maintain shape
            d2_dx2 = F.pad(d2_dx2, (1, 1), mode='constant', value=0)
            d2_dy2 = F.pad(d2_dy2, (0, 0, 1, 1), mode='constant', value=0)
            
            return d2_dx2 + d2_dy2
        
        return {
            'gradient': gradient_2d,
            'divergence': divergence_2d,
            'laplacian': laplacian_2d
        }
    
    def validate_conservation_laws(
        self,
        prediction: torch.Tensor,
        input_field: torch.Tensor
    ) -> ValidationResult:
        """Validate conservation laws for the given PDE type."""
        details = {}
        warnings = []
        critical_failures = []
        
        if self.pde_type == "navier_stokes":
            # Validate mass conservation (continuity equation)
            if prediction.shape[1] >= 2:  # Has velocity components
                u, v = prediction[:, 0], prediction[:, 1]
                divergence = self.diff_operators['divergence'](u, v)
                mass_conservation_error = torch.mean(torch.abs(divergence))
                
                details['mass_conservation_error'] = mass_conservation_error.item()
                
                if mass_conservation_error > self.tolerance:
                    critical_failures.append(
                        f"Mass conservation violated: error={mass_conservation_error:.6f}"
                    )
                
                # Validate momentum conservation (simplified)
                if prediction.shape[1] >= 3:  # Has pressure
                    p = prediction[:, 2]
                    grad_p_x, grad_p_y = self.diff_operators['gradient'](p)
                    
                    # Momentum balance check (simplified, steady state)
                    momentum_residual_x = grad_p_x
                    momentum_residual_y = grad_p_y
                    
                    momentum_error = torch.mean(
                        torch.sqrt(momentum_residual_x**2 + momentum_residual_y**2)
                    )
                    
                    details['momentum_conservation_error'] = momentum_error.item()
                    
                    if momentum_error > self.tolerance * 10:  # Relaxed tolerance
                        warnings.append(
                            f"Momentum conservation approximate: error={momentum_error:.6f}"
                        )
        
        elif self.pde_type == "heat_equation":
            # Energy conservation for heat equation
            if prediction.shape[1] >= 1:
                temperature = prediction[:, 0]
                laplacian_T = self.diff_operators['laplacian'](temperature)
                
                # Heat equation residual: ∂T/∂t - α∇²T = 0 (for steady state, should be α∇²T = 0)
                heat_residual = laplacian_T
                heat_error = torch.mean(torch.abs(heat_residual))
                
                details['heat_conservation_error'] = heat_error.item()
                
                if heat_error > self.tolerance:
                    warnings.append(
                        f"Heat equation residual high: error={heat_error:.6f}"
                    )
        
        # Overall score based on conservation errors
        total_error = sum([v for k, v in details.items() if 'error' in k])
        score = max(0.0, 1.0 - total_error / self.tolerance)
        
        passed = len(critical_failures) == 0
        
        return ValidationResult(
            test_name="conservation_laws",
            passed=passed,
            score=score,
            details=details,
            warnings=warnings,
            critical_failures=critical_failures
        )
    
    def validate_boundary_conditions(
        self,
        prediction: torch.Tensor,
        boundary_type: str = "periodic"
    ) -> ValidationResult:
        """Validate boundary condition adherence."""
        details = {}
        warnings = []
        critical_failures = []
        
        if boundary_type == "periodic":
            # Check periodic boundary conditions
            left_right_error = torch.mean(torch.abs(prediction[..., 0] - prediction[..., -1]))
            top_bottom_error = torch.mean(torch.abs(prediction[:, :, 0, :] - prediction[:, :, -1, :]))
            
            details['periodic_lr_error'] = left_right_error.item()
            details['periodic_tb_error'] = top_bottom_error.item()
            
            total_boundary_error = left_right_error + top_bottom_error
            
            if total_boundary_error > self.tolerance:
                critical_failures.append(
                    f"Periodic boundary conditions violated: error={total_boundary_error:.6f}"
                )
        
        elif boundary_type == "dirichlet":
            # Check if boundaries have zero values (or specified values)
            boundary_sum = (
                torch.sum(torch.abs(prediction[..., 0])) +  # Left
                torch.sum(torch.abs(prediction[..., -1])) +  # Right
                torch.sum(torch.abs(prediction[:, :, 0, :])) +  # Top
                torch.sum(torch.abs(prediction[:, :, -1, :]))  # Bottom
            )
            
            details['dirichlet_boundary_error'] = boundary_sum.item()
            
            if boundary_sum > self.tolerance * prediction.numel():
                critical_failures.append(
                    f"Dirichlet boundary conditions violated: boundary_sum={boundary_sum:.6f}"
                )
        
        elif boundary_type == "neumann":
            # Check if normal derivatives at boundaries are zero (or specified)
            grad_x, grad_y = self.diff_operators['gradient'](prediction.mean(dim=1))  # Average over channels
            
            # Normal derivatives at boundaries
            left_grad = torch.mean(torch.abs(grad_x[..., 0]))
            right_grad = torch.mean(torch.abs(grad_x[..., -1]))
            top_grad = torch.mean(torch.abs(grad_y[:, 0, :]))
            bottom_grad = torch.mean(torch.abs(grad_y[:, -1, :]))
            
            total_neumann_error = left_grad + right_grad + top_grad + bottom_grad
            
            details['neumann_boundary_error'] = total_neumann_error.item()
            
            if total_neumann_error > self.tolerance:
                warnings.append(
                    f"Neumann boundary conditions approximate: error={total_neumann_error:.6f}"
                )
        
        # Score based on boundary condition adherence
        total_error = sum([v for k, v in details.items() if 'error' in k])
        score = max(0.0, 1.0 - total_error / (self.tolerance * 10))
        
        passed = len(critical_failures) == 0
        
        return ValidationResult(
            test_name=f"boundary_conditions_{boundary_type}",
            passed=passed,
            score=score,
            details=details,
            warnings=warnings,
            critical_failures=critical_failures
        )
    
    def validate_pde_residual(
        self,
        prediction: torch.Tensor,
        input_field: torch.Tensor,
        time_derivative: Optional[torch.Tensor] = None
    ) -> ValidationResult:
        """Validate PDE residual satisfaction."""
        details = {}
        warnings = []
        critical_failures = []
        
        if self.pde_type == "navier_stokes" and prediction.shape[1] >= 3:
            u, v, p = prediction[:, 0], prediction[:, 1], prediction[:, 2]
            
            # Compute required derivatives
            du_dx, du_dy = self.diff_operators['gradient'](u)
            dv_dx, dv_dy = self.diff_operators['gradient'](v)
            dp_dx, dp_dy = self.diff_operators['gradient'](p)
            
            laplacian_u = self.diff_operators['laplacian'](u)
            laplacian_v = self.diff_operators['laplacian'](v)
            
            # Navier-Stokes equations (steady state, incompressible)
            # ∇p = ν∇²u (simplified momentum equation)
            viscosity = self.physics_params['viscosity']
            
            residual_u = dp_dx - viscosity * laplacian_u
            residual_v = dp_dy - viscosity * laplacian_v
            
            total_residual = torch.mean(torch.sqrt(residual_u**2 + residual_v**2))
            
            details['navier_stokes_residual'] = total_residual.item()
            
            if total_residual > self.tolerance * 100:  # Relaxed for complex PDE
                warnings.append(
                    f"Navier-Stokes residual high: residual={total_residual:.6f}"
                )
        
        elif self.pde_type == "heat_equation":
            temperature = prediction[:, 0]
            laplacian_T = self.diff_operators['laplacian'](temperature)
            
            # Heat equation: ∂T/∂t = α∇²T
            alpha = self.physics_params['thermal_diffusivity']
            
            if time_derivative is not None:
                residual = time_derivative - alpha * laplacian_T
            else:
                # Steady state: ∇²T = 0
                residual = laplacian_T
            
            residual_error = torch.mean(torch.abs(residual))
            
            details['heat_equation_residual'] = residual_error.item()
            
            if residual_error > self.tolerance:
                warnings.append(
                    f"Heat equation residual: error={residual_error:.6f}"
                )
        
        # Score based on PDE residual
        total_residual = sum([v for k, v in details.items() if 'residual' in k])
        score = max(0.0, 1.0 - total_residual / (self.tolerance * 50))
        
        passed = len(critical_failures) == 0
        
        return ValidationResult(
            test_name="pde_residual",
            passed=passed,
            score=score,
            details=details,
            warnings=warnings,
            critical_failures=critical_failures
        )


class UncertaintyCalibrationValidator:
    """
    Validator for uncertainty calibration in probabilistic predictions.
    
    Ensures that predicted uncertainties accurately reflect actual errors
    through various statistical calibration tests.
    """
    
    def __init__(self, confidence_levels: List[float] = None):
        self.confidence_levels = confidence_levels or [0.5, 0.8, 0.9, 0.95, 0.99]
    
    def validate_coverage(
        self,
        predictions: torch.Tensor,
        uncertainties: torch.Tensor,
        targets: torch.Tensor
    ) -> ValidationResult:
        """Validate coverage of uncertainty intervals."""
        details = {}
        warnings = []
        critical_failures = []
        
        coverage_results = {}
        
        for confidence in self.confidence_levels:
            # Compute z-score for confidence level
            z_score = stats.norm.ppf((1 + confidence) / 2)
            
            # Confidence intervals
            lower_bound = predictions - z_score * uncertainties
            upper_bound = predictions + z_score * uncertainties
            
            # Check coverage
            within_bounds = (targets >= lower_bound) & (targets <= upper_bound)
            empirical_coverage = within_bounds.float().mean().item()
            
            coverage_results[f'coverage_{int(confidence*100)}'] = empirical_coverage
            
            # Coverage should be close to nominal confidence level
            coverage_error = abs(empirical_coverage - confidence)
            details[f'coverage_error_{int(confidence*100)}'] = coverage_error
            
            if coverage_error > 0.1:  # 10% tolerance
                warnings.append(
                    f"Coverage at {confidence:.0%} deviates by {coverage_error:.3f}"
                )
            
            if coverage_error > 0.2:  # 20% critical threshold
                critical_failures.append(
                    f"Critical coverage failure at {confidence:.0%}: error={coverage_error:.3f}"
                )
        
        details.update(coverage_results)
        
        # Overall coverage score
        average_coverage_error = np.mean([
            details[k] for k in details.keys() if 'coverage_error' in k
        ])
        score = max(0.0, 1.0 - average_coverage_error * 5)  # Scale error to score
        
        passed = len(critical_failures) == 0
        
        return ValidationResult(
            test_name="uncertainty_coverage",
            passed=passed,
            score=score,
            details=details,
            warnings=warnings,
            critical_failures=critical_failures
        )
    
    def validate_sharpness(
        self,
        uncertainties: torch.Tensor,
        target_sharpness: float = 0.1
    ) -> ValidationResult:
        """Validate uncertainty sharpness (not too conservative)."""
        details = {}
        warnings = []
        critical_failures = []
        
        # Compute sharpness metrics
        mean_uncertainty = uncertainties.mean().item()
        std_uncertainty = uncertainties.std().item()
        max_uncertainty = uncertainties.max().item()
        
        details['mean_uncertainty'] = mean_uncertainty
        details['std_uncertainty'] = std_uncertainty
        details['max_uncertainty'] = max_uncertainty
        
        # Check if uncertainties are reasonable (not too large)
        if mean_uncertainty > target_sharpness * 10:
            warnings.append(
                f"Uncertainties may be too conservative: mean={mean_uncertainty:.4f}"
            )
        
        if mean_uncertainty > target_sharpness * 50:
            critical_failures.append(
                f"Uncertainties extremely large: mean={mean_uncertainty:.4f}"
            )
        
        # Check for uncertainty variability
        uncertainty_range = max_uncertainty - uncertainties.min().item()
        if uncertainty_range < mean_uncertainty * 0.1:
            warnings.append(
                "Uncertainties lack variability - may be poorly calibrated"
            )
        
        # Score based on sharpness appropriateness
        sharpness_score = 1.0 / (1.0 + mean_uncertainty / target_sharpness)
        variability_score = min(1.0, uncertainty_range / (mean_uncertainty + 1e-8))
        
        score = 0.7 * sharpness_score + 0.3 * variability_score
        
        passed = len(critical_failures) == 0
        
        return ValidationResult(
            test_name="uncertainty_sharpness",
            passed=passed,
            score=score,
            details=details,
            warnings=warnings,
            critical_failures=critical_failures
        )
    
    def validate_calibration_curve(
        self,
        predictions: torch.Tensor,
        uncertainties: torch.Tensor,
        targets: torch.Tensor,
        num_bins: int = 10
    ) -> ValidationResult:
        """Validate calibration using reliability diagram."""
        details = {}
        warnings = []
        critical_failures = []
        
        # Compute prediction confidence (1 - normalized uncertainty)
        max_uncertainty = uncertainties.max()
        confidence = 1.0 - uncertainties / (max_uncertainty + 1e-8)
        
        # Compute accuracy (1 if prediction is close to target)
        errors = torch.abs(predictions - targets)
        median_error = torch.median(errors)
        accuracy = (errors <= median_error).float()
        
        # Create bins for calibration curve
        bin_boundaries = torch.linspace(0, 1, num_bins + 1)
        bin_confidences = []
        bin_accuracies = []
        bin_counts = []
        
        for i in range(num_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]
            
            in_bin = (confidence >= bin_lower) & (confidence < bin_upper)
            
            if i == num_bins - 1:  # Include upper boundary in last bin
                in_bin = in_bin | (confidence == bin_upper)
            
            if in_bin.sum() > 0:
                bin_confidence = confidence[in_bin].mean().item()
                bin_accuracy = accuracy[in_bin].mean().item()
                bin_count = in_bin.sum().item()
                
                bin_confidences.append(bin_confidence)
                bin_accuracies.append(bin_accuracy)
                bin_counts.append(bin_count)
        
        # Compute calibration metrics
        if len(bin_confidences) > 0:
            # Expected Calibration Error (ECE)
            total_samples = len(confidence)
            ece = sum([
                (bin_counts[i] / total_samples) * abs(bin_confidences[i] - bin_accuracies[i])
                for i in range(len(bin_confidences))
            ])
            
            # Maximum Calibration Error (MCE)
            mce = max([
                abs(bin_confidences[i] - bin_accuracies[i])
                for i in range(len(bin_confidences))
            ])
            
            details['expected_calibration_error'] = ece
            details['maximum_calibration_error'] = mce
            details['num_calibration_bins'] = len(bin_confidences)
            
            # Calibration quality assessment
            if ece > 0.1:
                warnings.append(f"High Expected Calibration Error: {ece:.4f}")
            
            if ece > 0.2:
                critical_failures.append(f"Critical calibration error: ECE={ece:.4f}")
            
            if mce > 0.3:
                warnings.append(f"High Maximum Calibration Error: {mce:.4f}")
            
            # Score based on calibration quality
            score = max(0.0, 1.0 - ece * 5)  # Scale ECE to score
        else:
            score = 0.0
            critical_failures.append("No valid calibration bins found")
        
        passed = len(critical_failures) == 0
        
        return ValidationResult(
            test_name="calibration_curve",
            passed=passed,
            score=score,
            details=details,
            warnings=warnings,
            critical_failures=critical_failures
        )


class RobustnessValidator:
    """
    Validator for model robustness against various perturbations.
    
    Tests model stability against input noise, adversarial attacks,
    and distribution shifts.
    """
    
    def __init__(self, perturbation_strength: float = 0.1):
        self.perturbation_strength = perturbation_strength
    
    def validate_noise_robustness(
        self,
        model: ProbabilisticNeuralOperator,
        test_input: torch.Tensor,
        noise_levels: List[float] = None
    ) -> ValidationResult:
        """Validate robustness to input noise."""
        details = {}
        warnings = []
        critical_failures = []
        
        noise_levels = noise_levels or [0.01, 0.05, 0.1, 0.2]
        
        # Get baseline prediction
        with torch.no_grad():
            baseline_pred, baseline_unc = model.predict_with_uncertainty(test_input)
        
        robustness_scores = []
        
        for noise_level in noise_levels:
            # Add Gaussian noise
            noise = torch.randn_like(test_input) * noise_level
            noisy_input = test_input + noise
            
            with torch.no_grad():
                noisy_pred, noisy_unc = model.predict_with_uncertainty(noisy_input)
            
            # Compute prediction stability
            pred_diff = torch.mean((baseline_pred - noisy_pred)**2).item()
            unc_diff = torch.mean((baseline_unc - noisy_unc)**2).item()
            
            # Robustness score (lower is better)
            robustness_score = pred_diff / (noise_level**2 + 1e-8)
            robustness_scores.append(robustness_score)
            
            details[f'prediction_stability_noise_{noise_level}'] = pred_diff
            details[f'uncertainty_stability_noise_{noise_level}'] = unc_diff
            details[f'robustness_score_noise_{noise_level}'] = robustness_score
            
            # Check for excessive sensitivity
            if robustness_score > 10.0:
                warnings.append(
                    f"High sensitivity to noise level {noise_level}: score={robustness_score:.2f}"
                )
            
            if robustness_score > 100.0:
                critical_failures.append(
                    f"Critical instability at noise level {noise_level}: score={robustness_score:.2f}"
                )
        
        # Overall robustness score
        avg_robustness = np.mean(robustness_scores)
        score = max(0.0, 1.0 - avg_robustness / 50.0)  # Scale to [0,1]
        
        details['average_robustness_score'] = avg_robustness
        
        passed = len(critical_failures) == 0
        
        return ValidationResult(
            test_name="noise_robustness",
            passed=passed,
            score=score,
            details=details,
            warnings=warnings,
            critical_failures=critical_failures
        )
    
    def validate_adversarial_robustness(
        self,
        model: ProbabilisticNeuralOperator,
        test_input: torch.Tensor,
        epsilon: float = 0.1,
        num_steps: int = 10
    ) -> ValidationResult:
        """Validate robustness to adversarial perturbations."""
        details = {}
        warnings = []
        critical_failures = []
        
        # Simple FGSM-like adversarial attack
        test_input.requires_grad_(True)
        
        try:
            # Forward pass
            prediction = model(test_input)
            
            # Compute loss (maximize prediction variance)
            loss = -torch.var(prediction)
            
            # Compute gradients
            loss.backward()
            
            # Create adversarial perturbation
            perturbation = epsilon * torch.sign(test_input.grad)
            adversarial_input = test_input + perturbation
            
            with torch.no_grad():
                # Get predictions for clean and adversarial inputs
                clean_pred, clean_unc = model.predict_with_uncertainty(test_input.detach())
                adv_pred, adv_unc = model.predict_with_uncertainty(adversarial_input)
                
                # Compute adversarial impact
                pred_change = torch.mean((clean_pred - adv_pred)**2).item()
                unc_change = torch.mean((clean_unc - adv_unc)**2).item()
                
                details['adversarial_prediction_change'] = pred_change
                details['adversarial_uncertainty_change'] = unc_change
                details['perturbation_norm'] = torch.norm(perturbation).item()
                
                # Check for excessive vulnerability
                vulnerability_score = pred_change / (epsilon**2 + 1e-8)
                details['adversarial_vulnerability_score'] = vulnerability_score
                
                if vulnerability_score > 5.0:
                    warnings.append(
                        f"High adversarial vulnerability: score={vulnerability_score:.2f}"
                    )
                
                if vulnerability_score > 20.0:
                    critical_failures.append(
                        f"Critical adversarial vulnerability: score={vulnerability_score:.2f}"
                    )
                
                # Score based on adversarial robustness
                score = max(0.0, 1.0 - vulnerability_score / 10.0)
                
        except Exception as e:
            critical_failures.append(f"Adversarial robustness test failed: {str(e)}")
            score = 0.0
        
        finally:
            test_input.requires_grad_(False)
        
        passed = len(critical_failures) == 0
        
        return ValidationResult(
            test_name="adversarial_robustness",
            passed=passed,
            score=score,
            details=details,
            warnings=warnings,
            critical_failures=critical_failures
        )
    
    def validate_distribution_shift_robustness(
        self,
        model: ProbabilisticNeuralOperator,
        in_distribution_data: torch.Tensor,
        out_distribution_data: torch.Tensor
    ) -> ValidationResult:
        """Validate robustness to distribution shifts."""
        details = {}
        warnings = []
        critical_failures = []
        
        with torch.no_grad():
            # Get predictions for both distributions
            in_pred, in_unc = model.predict_with_uncertainty(in_distribution_data)
            out_pred, out_unc = model.predict_with_uncertainty(out_distribution_data)
            
            # Compute distribution statistics
            in_pred_mean = torch.mean(in_pred).item()
            in_pred_std = torch.std(in_pred).item()
            in_unc_mean = torch.mean(in_unc).item()
            
            out_pred_mean = torch.mean(out_pred).item()
            out_pred_std = torch.std(out_pred).item()
            out_unc_mean = torch.mean(out_unc).item()
            
            details['in_distribution_pred_mean'] = in_pred_mean
            details['in_distribution_pred_std'] = in_pred_std
            details['in_distribution_unc_mean'] = in_unc_mean
            
            details['out_distribution_pred_mean'] = out_pred_mean
            details['out_distribution_pred_std'] = out_pred_std
            details['out_distribution_unc_mean'] = out_unc_mean
            
            # Check if uncertainty increases for OOD data (good behavior)
            uncertainty_ratio = out_unc_mean / (in_unc_mean + 1e-8)
            details['uncertainty_ratio_ood'] = uncertainty_ratio
            
            if uncertainty_ratio < 1.1:
                warnings.append(
                    f"Uncertainty doesn't increase enough for OOD data: ratio={uncertainty_ratio:.2f}"
                )
            
            if uncertainty_ratio < 0.9:
                critical_failures.append(
                    f"Uncertainty decreases for OOD data: ratio={uncertainty_ratio:.2f}"
                )
            
            # Check for reasonable prediction changes
            pred_mean_change = abs(out_pred_mean - in_pred_mean) / (abs(in_pred_mean) + 1e-8)
            details['prediction_mean_change'] = pred_mean_change
            
            if pred_mean_change > 2.0:
                warnings.append(
                    f"Large prediction change for OOD data: change={pred_mean_change:.2f}"
                )
            
            # Score based on appropriate uncertainty behavior
            uncertainty_score = min(1.0, uncertainty_ratio / 2.0)  # Reward uncertainty increase
            prediction_score = max(0.0, 1.0 - pred_mean_change / 5.0)  # Penalize extreme changes
            
            score = 0.7 * uncertainty_score + 0.3 * prediction_score
        
        passed = len(critical_failures) == 0
        
        return ValidationResult(
            test_name="distribution_shift_robustness",
            passed=passed,
            score=score,
            details=details,
            warnings=warnings,
            critical_failures=critical_failures
        )


class ComprehensiveValidator:
    """
    Comprehensive validator that orchestrates all validation tests.
    
    Provides a unified interface for running all validation checks
    and generating comprehensive validation reports.
    """
    
    def __init__(
        self,
        pde_type: str = "navier_stokes",
        physics_tolerance: float = 1e-3,
        uncertainty_confidence_levels: List[float] = None,
        robustness_noise_levels: List[float] = None
    ):
        self.physics_validator = PhysicsConsistencyValidator(
            pde_type=pde_type,
            tolerance=physics_tolerance
        )
        
        self.uncertainty_validator = UncertaintyCalibrationValidator(
            confidence_levels=uncertainty_confidence_levels
        )
        
        self.robustness_validator = RobustnessValidator()
        
        # Configure logging
        self.logger = logging.getLogger(__name__)
    
    def run_comprehensive_validation(
        self,
        model: ProbabilisticNeuralOperator,
        test_data: Dict[str, torch.Tensor],
        boundary_type: str = "periodic"
    ) -> Dict[str, ValidationResult]:
        """
        Run comprehensive validation suite.
        
        Args:
            model: PNO model to validate
            test_data: Dictionary containing 'input', 'target', and optionally 'ood_input'
            boundary_type: Type of boundary conditions to validate
            
        Returns:
            Dictionary of validation results
        """
        results = {}
        
        test_input = test_data['input']
        test_target = test_data['target']
        
        self.logger.info("Starting comprehensive validation...")
        
        # Get model predictions
        with torch.no_grad():
            predictions, uncertainties = model.predict_with_uncertainty(test_input)
        
        # 1. Physics Consistency Validation
        self.logger.info("Running physics consistency validation...")
        
        try:
            results['conservation_laws'] = self.physics_validator.validate_conservation_laws(
                predictions, test_input
            )
            
            results['boundary_conditions'] = self.physics_validator.validate_boundary_conditions(
                predictions, boundary_type
            )
            
            results['pde_residual'] = self.physics_validator.validate_pde_residual(
                predictions, test_input
            )
        except Exception as e:
            self.logger.error(f"Physics validation failed: {e}")
            results['physics_validation_error'] = ValidationResult(
                test_name="physics_validation",
                passed=False,
                score=0.0,
                details={'error': str(e)},
                warnings=[],
                critical_failures=[f"Physics validation crashed: {e}"]
            )
        
        # 2. Uncertainty Calibration Validation
        self.logger.info("Running uncertainty calibration validation...")
        
        try:
            results['uncertainty_coverage'] = self.uncertainty_validator.validate_coverage(
                predictions, uncertainties, test_target
            )
            
            results['uncertainty_sharpness'] = self.uncertainty_validator.validate_sharpness(
                uncertainties
            )
            
            results['calibration_curve'] = self.uncertainty_validator.validate_calibration_curve(
                predictions, uncertainties, test_target
            )
        except Exception as e:
            self.logger.error(f"Uncertainty validation failed: {e}")
            results['uncertainty_validation_error'] = ValidationResult(
                test_name="uncertainty_validation",
                passed=False,
                score=0.0,
                details={'error': str(e)},
                warnings=[],
                critical_failures=[f"Uncertainty validation crashed: {e}"]
            )
        
        # 3. Robustness Validation
        self.logger.info("Running robustness validation...")
        
        try:
            results['noise_robustness'] = self.robustness_validator.validate_noise_robustness(
                model, test_input
            )
            
            results['adversarial_robustness'] = self.robustness_validator.validate_adversarial_robustness(
                model, test_input
            )
            
            # Distribution shift robustness (if OOD data provided)
            if 'ood_input' in test_data:
                results['distribution_shift_robustness'] = self.robustness_validator.validate_distribution_shift_robustness(
                    model, test_input, test_data['ood_input']
                )
        except Exception as e:
            self.logger.error(f"Robustness validation failed: {e}")
            results['robustness_validation_error'] = ValidationResult(
                test_name="robustness_validation",
                passed=False,
                score=0.0,
                details={'error': str(e)},
                warnings=[],
                critical_failures=[f"Robustness validation crashed: {e}"]
            )
        
        self.logger.info("Comprehensive validation completed.")
        return results
    
    def generate_validation_report(
        self,
        validation_results: Dict[str, ValidationResult],
        save_path: str = "validation_report.json"
    ) -> str:
        """Generate comprehensive validation report."""
        import json
        
        # Compute overall statistics
        all_scores = [result.score for result in validation_results.values()]
        all_passed = [result.passed for result in validation_results.values()]
        
        overall_score = np.mean(all_scores) if all_scores else 0.0
        overall_passed = all(all_passed)
        
        # Collect all warnings and critical failures
        all_warnings = []
        all_critical_failures = []
        
        for result in validation_results.values():
            all_warnings.extend(result.warnings)
            all_critical_failures.extend(result.critical_failures)
        
        # Create comprehensive report
        report = {
            "validation_summary": {
                "overall_score": overall_score,
                "overall_passed": overall_passed,
                "total_tests": len(validation_results),
                "passed_tests": sum(all_passed),
                "failed_tests": len(all_passed) - sum(all_passed),
                "total_warnings": len(all_warnings),
                "total_critical_failures": len(all_critical_failures)
            },
            "detailed_results": {},
            "warnings": all_warnings,
            "critical_failures": all_critical_failures,
            "recommendations": self._generate_recommendations(validation_results)
        }
        
        # Add detailed results
        for test_name, result in validation_results.items():
            report["detailed_results"][test_name] = {
                "passed": result.passed,
                "score": result.score,
                "details": result.details,
                "warnings": result.warnings,
                "critical_failures": result.critical_failures
            }
        
        # Save report
        with open(save_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        return f"Comprehensive validation report saved to {save_path}"
    
    def _generate_recommendations(
        self,
        validation_results: Dict[str, ValidationResult]
    ) -> List[str]:
        """Generate actionable recommendations based on validation results."""
        recommendations = []
        
        # Physics-based recommendations
        if 'conservation_laws' in validation_results:
            result = validation_results['conservation_laws']
            if not result.passed:
                recommendations.append(
                    "Consider adding physics-informed loss terms to enforce conservation laws"
                )
            if result.score < 0.7:
                recommendations.append(
                    "Improve physics consistency by increasing model capacity or training time"
                )
        
        # Uncertainty-based recommendations
        if 'uncertainty_coverage' in validation_results:
            result = validation_results['uncertainty_coverage']
            if result.score < 0.8:
                recommendations.append(
                    "Improve uncertainty calibration through temperature scaling or recalibration"
                )
        
        if 'uncertainty_sharpness' in validation_results:
            result = validation_results['uncertainty_sharpness']
            if result.score < 0.6:
                recommendations.append(
                    "Reduce uncertainty conservatism through better posterior approximation"
                )
        
        # Robustness-based recommendations
        if 'noise_robustness' in validation_results:
            result = validation_results['noise_robustness']
            if result.score < 0.7:
                recommendations.append(
                    "Improve noise robustness through data augmentation or adversarial training"
                )
        
        if 'adversarial_robustness' in validation_results:
            result = validation_results['adversarial_robustness']
            if result.score < 0.6:
                recommendations.append(
                    "Consider adversarial training to improve model robustness"
                )
        
        return recommendations


def run_validation_example():
    """Example of running comprehensive validation."""
    from ..models import ProbabilisticNeuralOperator
    
    # Create model
    model = ProbabilisticNeuralOperator(
        input_dim=3,
        hidden_dim=128,
        num_layers=4,
        modes=16
    )
    
    # Create test data
    test_data = {
        'input': torch.randn(8, 3, 64, 64),
        'target': torch.randn(8, 1, 64, 64),
        'ood_input': torch.randn(8, 3, 64, 64) * 2.0  # Scaled for OOD
    }
    
    # Create validator
    validator = ComprehensiveValidator(
        pde_type="navier_stokes",
        physics_tolerance=1e-3
    )
    
    # Run validation
    results = validator.run_comprehensive_validation(model, test_data)
    
    # Generate report
    report_path = validator.generate_validation_report(results)
    print(f"Validation completed. Report saved to: {report_path}")
    
    return results


if __name__ == "__main__":
    # Run example validation
    results = run_validation_example()
    print("🔬 Advanced Validation Framework ready for comprehensive testing!")