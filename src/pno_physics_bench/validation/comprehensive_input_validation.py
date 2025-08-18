"""Comprehensive Input Validation for Probabilistic Neural Operators.

This module provides extensive input validation, sanitization, and preprocessing
to ensure robust and secure operation of PNO models.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
import numpy as np
from dataclasses import dataclass
from enum import Enum
import warnings
import logging
from abc import ABC, abstractmethod


class ValidationSeverity(Enum):
    """Validation issue severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationResult:
    """Result of input validation."""
    is_valid: bool
    sanitized_input: Optional[torch.Tensor] = None
    issues: List[Dict[str, Any]] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.issues is None:
            self.issues = []
        if self.metadata is None:
            self.metadata = {}


class InputValidator(ABC):
    """Abstract base class for input validators."""
    
    @abstractmethod
    def validate(self, input_tensor: torch.Tensor) -> ValidationResult:
        """Validate input tensor."""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Get validator name."""
        pass


class ShapeValidator(InputValidator):
    """Validate tensor shapes and dimensions."""
    
    def __init__(
        self,
        expected_shape: Optional[Tuple[int, ...]] = None,
        min_dims: Optional[int] = None,
        max_dims: Optional[int] = None,
        spatial_dims: Optional[Tuple[int, ...]] = None,
        allow_batch_dimension: bool = True
    ):
        self.expected_shape = expected_shape
        self.min_dims = min_dims
        self.max_dims = max_dims
        self.spatial_dims = spatial_dims
        self.allow_batch_dimension = allow_batch_dimension
    
    def validate(self, input_tensor: torch.Tensor) -> ValidationResult:
        issues = []
        metadata = {'original_shape': list(input_tensor.shape)}
        
        # Check number of dimensions
        if self.min_dims is not None and len(input_tensor.shape) < self.min_dims:
            issues.append({
                'type': 'insufficient_dimensions',
                'severity': ValidationSeverity.ERROR,
                'message': f"Input has {len(input_tensor.shape)} dimensions, minimum {self.min_dims} required",
                'current_dims': len(input_tensor.shape),
                'required_dims': self.min_dims
            })
        
        if self.max_dims is not None and len(input_tensor.shape) > self.max_dims:
            issues.append({
                'type': 'excessive_dimensions',
                'severity': ValidationSeverity.ERROR,
                'message': f"Input has {len(input_tensor.shape)} dimensions, maximum {self.max_dims} allowed",
                'current_dims': len(input_tensor.shape),
                'max_dims': self.max_dims
            })
        
        # Check expected shape
        if self.expected_shape is not None:
            expected = self.expected_shape
            actual = input_tensor.shape
            
            # Handle batch dimension
            if self.allow_batch_dimension and len(actual) == len(expected) + 1:
                actual = actual[1:]  # Remove batch dimension for comparison
            
            if len(actual) != len(expected):
                issues.append({
                    'type': 'shape_dimension_mismatch',
                    'severity': ValidationSeverity.ERROR,
                    'message': f"Shape dimension mismatch: expected {len(expected)}, got {len(actual)}",
                    'expected_shape': expected,
                    'actual_shape': actual
                })
            else:
                # Check individual dimensions
                for i, (exp, act) in enumerate(zip(expected, actual)):
                    if exp != -1 and exp != act:  # -1 means any size allowed
                        issues.append({
                            'type': 'shape_size_mismatch',
                            'severity': ValidationSeverity.ERROR,
                            'message': f"Shape mismatch at dimension {i}: expected {exp}, got {act}",
                            'dimension': i,
                            'expected_size': exp,
                            'actual_size': act
                        })
        
        # Check spatial dimensions
        if self.spatial_dims is not None:
            spatial_shape = input_tensor.shape[-len(self.spatial_dims):]
            for i, (exp, act) in enumerate(zip(self.spatial_dims, spatial_shape)):
                if exp != -1 and exp != act:
                    issues.append({
                        'type': 'spatial_dimension_mismatch',
                        'severity': ValidationSeverity.WARNING,
                        'message': f"Spatial dimension {i} mismatch: expected {exp}, got {act}",
                        'spatial_dim': i,
                        'expected_size': exp,
                        'actual_size': act
                    })
        
        is_valid = not any(issue['severity'] == ValidationSeverity.ERROR for issue in issues)
        
        return ValidationResult(
            is_valid=is_valid,
            sanitized_input=input_tensor if is_valid else None,
            issues=issues,
            metadata=metadata
        )
    
    def get_name(self) -> str:
        return "ShapeValidator"


class NumericalValidator(InputValidator):
    """Validate numerical properties of tensors."""
    
    def __init__(
        self,
        check_finite: bool = True,
        check_range: bool = True,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        allow_zero: bool = True,
        check_distribution: bool = True,
        max_std_threshold: float = 100.0,
        min_std_threshold: float = 1e-8
    ):
        self.check_finite = check_finite
        self.check_range = check_range
        self.min_value = min_value
        self.max_value = max_value
        self.allow_zero = allow_zero
        self.check_distribution = check_distribution
        self.max_std_threshold = max_std_threshold
        self.min_std_threshold = min_std_threshold
    
    def validate(self, input_tensor: torch.Tensor) -> ValidationResult:
        issues = []
        sanitized_input = input_tensor.clone()
        
        # Check for NaN values
        if self.check_finite:
            nan_mask = torch.isnan(input_tensor)
            if nan_mask.any():
                nan_count = nan_mask.sum().item()
                issues.append({
                    'type': 'nan_values',
                    'severity': ValidationSeverity.ERROR,
                    'message': f"Found {nan_count} NaN values in input",
                    'nan_count': nan_count,
                    'total_elements': input_tensor.numel()
                })
                
                # Sanitize: replace NaN with zeros
                sanitized_input = torch.where(nan_mask, torch.zeros_like(input_tensor), sanitized_input)
        
        # Check for infinite values
        if self.check_finite:
            inf_mask = torch.isinf(input_tensor)
            if inf_mask.any():
                inf_count = inf_mask.sum().item()
                issues.append({
                    'type': 'infinite_values',
                    'severity': ValidationSeverity.ERROR,
                    'message': f"Found {inf_count} infinite values in input",
                    'inf_count': inf_count,
                    'total_elements': input_tensor.numel()
                })
                
                # Sanitize: replace inf with large but finite values
                pos_inf_mask = torch.isposinf(input_tensor)
                neg_inf_mask = torch.isneginf(input_tensor)
                
                if self.max_value is not None:
                    sanitized_input = torch.where(pos_inf_mask, torch.full_like(input_tensor, self.max_value), sanitized_input)
                else:
                    sanitized_input = torch.where(pos_inf_mask, torch.full_like(input_tensor, 1e6), sanitized_input)
                
                if self.min_value is not None:
                    sanitized_input = torch.where(neg_inf_mask, torch.full_like(input_tensor, self.min_value), sanitized_input)
                else:
                    sanitized_input = torch.where(neg_inf_mask, torch.full_like(input_tensor, -1e6), sanitized_input)
        
        # Check value ranges
        if self.check_range:
            if self.min_value is not None:
                below_min = (sanitized_input < self.min_value)
                if below_min.any():
                    issues.append({
                        'type': 'values_below_minimum',
                        'severity': ValidationSeverity.WARNING,
                        'message': f"Found {below_min.sum().item()} values below minimum {self.min_value}",
                        'count': below_min.sum().item(),
                        'min_threshold': self.min_value,
                        'actual_min': sanitized_input.min().item()
                    })
                    
                    # Sanitize: clamp to minimum
                    sanitized_input = torch.clamp(sanitized_input, min=self.min_value)
            
            if self.max_value is not None:
                above_max = (sanitized_input > self.max_value)
                if above_max.any():
                    issues.append({
                        'type': 'values_above_maximum',
                        'severity': ValidationSeverity.WARNING,
                        'message': f"Found {above_max.sum().item()} values above maximum {self.max_value}",
                        'count': above_max.sum().item(),
                        'max_threshold': self.max_value,
                        'actual_max': sanitized_input.max().item()
                    })
                    
                    # Sanitize: clamp to maximum
                    sanitized_input = torch.clamp(sanitized_input, max=self.max_value)
        
        # Check for zero values
        if not self.allow_zero:
            zero_mask = (sanitized_input == 0)
            if zero_mask.any():
                issues.append({
                    'type': 'zero_values',
                    'severity': ValidationSeverity.WARNING,
                    'message': f"Found {zero_mask.sum().item()} zero values (not allowed)",
                    'zero_count': zero_mask.sum().item()
                })
                
                # Sanitize: replace zeros with small non-zero values
                sanitized_input = torch.where(zero_mask, torch.full_like(sanitized_input, 1e-8), sanitized_input)
        
        # Check distribution properties
        if self.check_distribution and sanitized_input.numel() > 1:
            mean_val = sanitized_input.mean().item()
            std_val = sanitized_input.std().item()
            
            if std_val > self.max_std_threshold:
                issues.append({
                    'type': 'high_variance',
                    'severity': ValidationSeverity.WARNING,
                    'message': f"Input has very high standard deviation: {std_val:.4f}",
                    'std_value': std_val,
                    'threshold': self.max_std_threshold
                })
            
            if std_val < self.min_std_threshold:
                issues.append({
                    'type': 'low_variance',
                    'severity': ValidationSeverity.INFO,
                    'message': f"Input has very low standard deviation: {std_val:.4f}",
                    'std_value': std_val,
                    'threshold': self.min_std_threshold
                })
        
        # Metadata
        metadata = {
            'min_value': sanitized_input.min().item() if sanitized_input.numel() > 0 else 0,
            'max_value': sanitized_input.max().item() if sanitized_input.numel() > 0 else 0,
            'mean_value': sanitized_input.mean().item() if sanitized_input.numel() > 0 else 0,
            'std_value': sanitized_input.std().item() if sanitized_input.numel() > 1 else 0
        }
        
        is_valid = not any(issue['severity'] == ValidationSeverity.ERROR for issue in issues)
        
        return ValidationResult(
            is_valid=is_valid,
            sanitized_input=sanitized_input,
            issues=issues,
            metadata=metadata
        )
    
    def get_name(self) -> str:
        return "NumericalValidator"


class TypeValidator(InputValidator):
    """Validate tensor data types and device placement."""
    
    def __init__(
        self,
        expected_dtype: Optional[torch.dtype] = None,
        allowed_dtypes: Optional[List[torch.dtype]] = None,
        expected_device: Optional[Union[str, torch.device]] = None,
        auto_convert_dtype: bool = True,
        auto_transfer_device: bool = True
    ):
        self.expected_dtype = expected_dtype
        self.allowed_dtypes = allowed_dtypes or []
        self.expected_device = expected_device
        self.auto_convert_dtype = auto_convert_dtype
        self.auto_transfer_device = auto_transfer_device
        
        if expected_dtype and expected_dtype not in self.allowed_dtypes:
            self.allowed_dtypes.append(expected_dtype)
    
    def validate(self, input_tensor: torch.Tensor) -> ValidationResult:
        issues = []
        sanitized_input = input_tensor
        
        # Check data type
        if self.allowed_dtypes and input_tensor.dtype not in self.allowed_dtypes:
            issues.append({
                'type': 'invalid_dtype',
                'severity': ValidationSeverity.WARNING,
                'message': f"Input dtype {input_tensor.dtype} not in allowed types {self.allowed_dtypes}",
                'current_dtype': str(input_tensor.dtype),
                'allowed_dtypes': [str(dt) for dt in self.allowed_dtypes]
            })
            
            if self.auto_convert_dtype and self.expected_dtype:
                try:
                    sanitized_input = sanitized_input.to(dtype=self.expected_dtype)
                    issues[-1]['auto_converted'] = True
                    issues[-1]['new_dtype'] = str(self.expected_dtype)
                except Exception as e:
                    issues[-1]['conversion_failed'] = str(e)
                    issues[-1]['severity'] = ValidationSeverity.ERROR
        
        # Check device
        if self.expected_device is not None:
            expected_device = torch.device(self.expected_device)
            if sanitized_input.device != expected_device:
                issues.append({
                    'type': 'device_mismatch',
                    'severity': ValidationSeverity.INFO,
                    'message': f"Input on device {sanitized_input.device}, expected {expected_device}",
                    'current_device': str(sanitized_input.device),
                    'expected_device': str(expected_device)
                })
                
                if self.auto_transfer_device:
                    try:
                        sanitized_input = sanitized_input.to(device=expected_device)
                        issues[-1]['auto_transferred'] = True
                    except Exception as e:
                        issues[-1]['transfer_failed'] = str(e)
                        issues[-1]['severity'] = ValidationSeverity.ERROR
        
        # Metadata
        metadata = {
            'original_dtype': str(input_tensor.dtype),
            'original_device': str(input_tensor.device),
            'final_dtype': str(sanitized_input.dtype),
            'final_device': str(sanitized_input.device)
        }
        
        is_valid = not any(issue['severity'] == ValidationSeverity.ERROR for issue in issues)
        
        return ValidationResult(
            is_valid=is_valid,
            sanitized_input=sanitized_input,
            issues=issues,
            metadata=metadata
        )
    
    def get_name(self) -> str:
        return "TypeValidator"


class PhysicsValidator(InputValidator):
    """Validate physical properties and constraints."""
    
    def __init__(
        self,
        check_conservation_laws: bool = True,
        check_boundary_conditions: bool = True,
        expected_physics_properties: Optional[Dict[str, Any]] = None,
        tolerance: float = 1e-6
    ):
        self.check_conservation_laws = check_conservation_laws
        self.check_boundary_conditions = check_boundary_conditions
        self.expected_physics_properties = expected_physics_properties or {}
        self.tolerance = tolerance
    
    def validate(self, input_tensor: torch.Tensor) -> ValidationResult:
        issues = []
        metadata = {}
        
        # Check conservation laws (simplified)
        if self.check_conservation_laws:
            # Mass/energy conservation (total sum should be preserved)
            total_quantity = input_tensor.sum().item()
            metadata['total_quantity'] = total_quantity
            
            if 'expected_total' in self.expected_physics_properties:
                expected_total = self.expected_physics_properties['expected_total']
                if abs(total_quantity - expected_total) > self.tolerance:
                    issues.append({
                        'type': 'conservation_violation',
                        'severity': ValidationSeverity.WARNING,
                        'message': f"Total quantity {total_quantity:.6f} deviates from expected {expected_total:.6f}",
                        'actual_total': total_quantity,
                        'expected_total': expected_total,
                        'deviation': abs(total_quantity - expected_total)
                    })
        
        # Check boundary conditions
        if self.check_boundary_conditions and len(input_tensor.shape) >= 2:
            # Check if boundary values are reasonable
            if len(input_tensor.shape) == 2:
                # 2D case
                boundary_values = torch.cat([
                    input_tensor[0, :],   # top
                    input_tensor[-1, :],  # bottom
                    input_tensor[:, 0],   # left
                    input_tensor[:, -1]   # right
                ])
            elif len(input_tensor.shape) >= 3:
                # Multi-dimensional case (sample from boundaries)
                boundary_values = torch.cat([
                    input_tensor[0, ...].flatten(),
                    input_tensor[-1, ...].flatten()
                ])
            
            boundary_std = boundary_values.std().item()
            interior_std = input_tensor[1:-1, 1:-1].std().item() if input_tensor.shape[0] > 2 and input_tensor.shape[1] > 2 else 0
            
            metadata['boundary_std'] = boundary_std
            metadata['interior_std'] = interior_std
            
            # Boundary should be smoother than interior for most physics problems
            if boundary_std > interior_std * 2:
                issues.append({
                    'type': 'irregular_boundary',
                    'severity': ValidationSeverity.INFO,
                    'message': f"Boundary variation ({boundary_std:.6f}) much larger than interior ({interior_std:.6f})",
                    'boundary_std': boundary_std,
                    'interior_std': interior_std
                })
        
        # Check gradient properties (smoothness)
        if len(input_tensor.shape) >= 2:
            # Compute gradients
            if len(input_tensor.shape) == 2:
                grad_x = torch.diff(input_tensor, dim=1)
                grad_y = torch.diff(input_tensor, dim=0)
                
                grad_magnitude = torch.sqrt(grad_x[:-1, :]**2 + grad_y[:, :-1]**2)
                
            else:
                # For higher dimensions, just use the first two spatial dimensions
                grad_x = torch.diff(input_tensor, dim=-1)
                grad_y = torch.diff(input_tensor, dim=-2)
                
                grad_magnitude = torch.sqrt(grad_x[..., :-1]**2 + grad_y[..., :, :-1]**2)
            
            max_gradient = grad_magnitude.max().item()
            mean_gradient = grad_magnitude.mean().item()
            
            metadata['max_gradient'] = max_gradient
            metadata['mean_gradient'] = mean_gradient
            
            # Check for extreme gradients
            if max_gradient > mean_gradient * 100:
                issues.append({
                    'type': 'extreme_gradients',
                    'severity': ValidationSeverity.WARNING,
                    'message': f"Maximum gradient ({max_gradient:.6f}) is {max_gradient/mean_gradient:.1f}x the mean",
                    'max_gradient': max_gradient,
                    'mean_gradient': mean_gradient
                })
        
        is_valid = not any(issue['severity'] == ValidationSeverity.ERROR for issue in issues)
        
        return ValidationResult(
            is_valid=is_valid,
            sanitized_input=input_tensor,
            issues=issues,
            metadata=metadata
        )
    
    def get_name(self) -> str:
        return "PhysicsValidator"


class SecurityValidator(InputValidator):
    """Validate inputs for security concerns."""
    
    def __init__(
        self,
        max_tensor_size: int = 1024**3,  # 1GB limit
        check_adversarial_patterns: bool = True,
        allowed_memory_gb: float = 8.0
    ):
        self.max_tensor_size = max_tensor_size
        self.check_adversarial_patterns = check_adversarial_patterns
        self.allowed_memory_gb = allowed_memory_gb
    
    def validate(self, input_tensor: torch.Tensor) -> ValidationResult:
        issues = []
        metadata = {}
        
        # Check tensor size for memory attacks
        tensor_size = input_tensor.numel() * input_tensor.element_size()
        memory_gb = tensor_size / (1024**3)
        
        metadata['tensor_size_bytes'] = tensor_size
        metadata['memory_gb'] = memory_gb
        
        if tensor_size > self.max_tensor_size:
            issues.append({
                'type': 'excessive_memory_usage',
                'severity': ValidationSeverity.ERROR,
                'message': f"Tensor requires {memory_gb:.2f}GB, limit is {self.allowed_memory_gb}GB",
                'tensor_size_gb': memory_gb,
                'limit_gb': self.allowed_memory_gb
            })
        
        # Check for potential adversarial patterns
        if self.check_adversarial_patterns:
            # High-frequency noise pattern (common in adversarial attacks)
            if len(input_tensor.shape) >= 2:
                # Compute high-frequency content
                if len(input_tensor.shape) == 2:
                    laplacian = (
                        input_tensor[:-2, 1:-1] + 
                        input_tensor[2:, 1:-1] + 
                        input_tensor[1:-1, :-2] + 
                        input_tensor[1:-1, 2:] - 
                        4 * input_tensor[1:-1, 1:-1]
                    )
                else:
                    # Simplified for higher dimensions
                    diff_x = torch.diff(input_tensor, dim=-1)
                    diff_y = torch.diff(input_tensor, dim=-2)
                    laplacian = torch.diff(diff_x, dim=-1)[..., :-1] + torch.diff(diff_y, dim=-2)[..., :-1, :]
                
                high_freq_energy = laplacian.abs().mean().item()
                total_energy = input_tensor.abs().mean().item()
                
                metadata['high_freq_ratio'] = high_freq_energy / (total_energy + 1e-8)
                
                if high_freq_energy / (total_energy + 1e-8) > 0.5:
                    issues.append({
                        'type': 'potential_adversarial_noise',
                        'severity': ValidationSeverity.WARNING,
                        'message': f"High-frequency content ratio: {high_freq_energy / (total_energy + 1e-8):.3f}",
                        'high_freq_ratio': high_freq_energy / (total_energy + 1e-8)
                    })
        
        # Check for uniform patterns (potential DoS)
        if input_tensor.numel() > 100:  # Only for reasonably sized tensors
            unique_values = torch.unique(input_tensor)
            uniqueness_ratio = len(unique_values) / input_tensor.numel()
            
            metadata['uniqueness_ratio'] = uniqueness_ratio
            
            if uniqueness_ratio < 0.01:  # Less than 1% unique values
                issues.append({
                    'type': 'low_entropy_input',
                    'severity': ValidationSeverity.INFO,
                    'message': f"Input has low entropy (uniqueness ratio: {uniqueness_ratio:.4f})",
                    'uniqueness_ratio': uniqueness_ratio,
                    'unique_values': len(unique_values)
                })
        
        is_valid = not any(issue['severity'] == ValidationSeverity.ERROR for issue in issues)
        
        return ValidationResult(
            is_valid=is_valid,
            sanitized_input=input_tensor,
            issues=issues,
            metadata=metadata
        )
    
    def get_name(self) -> str:
        return "SecurityValidator"


class ComprehensiveInputValidator:
    """Comprehensive validation system combining multiple validators."""
    
    def __init__(
        self,
        validators: Optional[List[InputValidator]] = None,
        enable_sanitization: bool = True,
        log_validation_results: bool = True,
        fail_on_error: bool = True
    ):
        if validators is None:
            # Default validator set
            validators = [
                ShapeValidator(min_dims=1, max_dims=6),
                NumericalValidator(),
                TypeValidator(allowed_dtypes=[torch.float32, torch.float64]),
                SecurityValidator(),
                PhysicsValidator()
            ]
        
        self.validators = validators
        self.enable_sanitization = enable_sanitization
        self.log_validation_results = log_validation_results
        self.fail_on_error = fail_on_error
        self.logger = logging.getLogger(__name__)
        
        # Validation statistics
        self.validation_stats = {
            'total_validations': 0,
            'successful_validations': 0,
            'failed_validations': 0,
            'sanitization_applied': 0,
            'issue_counts': {}
        }
    
    def validate(self, input_tensor: torch.Tensor) -> ValidationResult:
        """Comprehensive validation of input tensor."""
        
        self.validation_stats['total_validations'] += 1
        
        all_issues = []
        all_metadata = {}
        current_tensor = input_tensor
        
        overall_valid = True
        
        # Run all validators
        for validator in self.validators:
            try:
                result = validator.validate(current_tensor)
                
                # Collect issues
                all_issues.extend(result.issues)
                
                # Merge metadata
                validator_name = validator.get_name()
                all_metadata[validator_name] = result.metadata
                
                # Update tensor if sanitization is enabled and successful
                if (self.enable_sanitization and 
                    result.sanitized_input is not None and 
                    result.is_valid):
                    current_tensor = result.sanitized_input
                
                # Track overall validity
                if not result.is_valid:
                    overall_valid = False
                
            except Exception as e:
                all_issues.append({
                    'type': 'validator_error',
                    'severity': ValidationSeverity.ERROR,
                    'message': f"Validator {validator.get_name()} failed: {str(e)}",
                    'validator': validator.get_name(),
                    'error': str(e)
                })
                overall_valid = False
        
        # Update statistics
        if overall_valid:
            self.validation_stats['successful_validations'] += 1
        else:
            self.validation_stats['failed_validations'] += 1
        
        if current_tensor is not input_tensor:
            self.validation_stats['sanitization_applied'] += 1
        
        # Count issues by type
        for issue in all_issues:
            issue_type = issue.get('type', 'unknown')
            self.validation_stats['issue_counts'][issue_type] = (
                self.validation_stats['issue_counts'].get(issue_type, 0) + 1
            )
        
        # Log results
        if self.log_validation_results:
            self._log_validation_results(all_issues, overall_valid)
        
        # Create comprehensive result
        result = ValidationResult(
            is_valid=overall_valid,
            sanitized_input=current_tensor if self.enable_sanitization else input_tensor,
            issues=all_issues,
            metadata=all_metadata
        )
        
        # Raise exception if failing on errors
        if self.fail_on_error and not overall_valid:
            error_issues = [issue for issue in all_issues if issue['severity'] == ValidationSeverity.ERROR]
            if error_issues:
                error_msg = f"Validation failed with {len(error_issues)} errors"
                raise ValueError(error_msg)
        
        return result
    
    def _log_validation_results(self, issues: List[Dict], overall_valid: bool):
        """Log validation results."""
        
        if not issues:
            self.logger.debug("Input validation passed with no issues")
            return
        
        # Group issues by severity
        issues_by_severity = {}
        for issue in issues:
            severity = issue['severity']
            if severity not in issues_by_severity:
                issues_by_severity[severity] = []
            issues_by_severity[severity].append(issue)
        
        # Log based on highest severity
        if ValidationSeverity.ERROR in issues_by_severity:
            self.logger.error(f"Input validation failed with {len(issues_by_severity[ValidationSeverity.ERROR])} errors")
        elif ValidationSeverity.WARNING in issues_by_severity:
            self.logger.warning(f"Input validation passed with {len(issues_by_severity[ValidationSeverity.WARNING])} warnings")
        else:
            self.logger.info(f"Input validation passed with {len(issues)} informational issues")
    
    def add_validator(self, validator: InputValidator):
        """Add a new validator to the validation pipeline."""
        self.validators.append(validator)
    
    def remove_validator(self, validator_name: str):
        """Remove a validator by name."""
        self.validators = [v for v in self.validators if v.get_name() != validator_name]
    
    def get_validation_statistics(self) -> Dict[str, Any]:
        """Get validation statistics."""
        stats = self.validation_stats.copy()
        
        if stats['total_validations'] > 0:
            stats['success_rate'] = stats['successful_validations'] / stats['total_validations']
            stats['sanitization_rate'] = stats['sanitization_applied'] / stats['total_validations']
        else:
            stats['success_rate'] = 0.0
            stats['sanitization_rate'] = 0.0
        
        return stats
    
    def reset_statistics(self):
        """Reset validation statistics."""
        self.validation_stats = {
            'total_validations': 0,
            'successful_validations': 0,
            'failed_validations': 0,
            'sanitization_applied': 0,
            'issue_counts': {}
        }


# Convenience function for quick validation
def validate_pno_input(
    input_tensor: torch.Tensor,
    expected_shape: Optional[Tuple[int, ...]] = None,
    expected_dtype: Optional[torch.dtype] = None,
    enable_sanitization: bool = True
) -> ValidationResult:
    """Quick validation function for PNO inputs."""
    
    validators = []
    
    if expected_shape is not None:
        validators.append(ShapeValidator(expected_shape=expected_shape))
    
    validators.extend([
        NumericalValidator(),
        TypeValidator(expected_dtype=expected_dtype) if expected_dtype else TypeValidator(),
        SecurityValidator()
    ])
    
    validator_system = ComprehensiveInputValidator(
        validators=validators,
        enable_sanitization=enable_sanitization,
        log_validation_results=False,  # Don't log for quick validation
        fail_on_error=False
    )
    
    return validator_system.validate(input_tensor)