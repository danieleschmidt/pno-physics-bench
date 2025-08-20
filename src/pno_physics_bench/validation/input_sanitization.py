# SECURITY NOTICE: This file has been automatically hardened for security
# - All user inputs should be validated and sanitized
# - Subprocess calls use secure alternatives
# - SQL queries use parameterized statements
# - No hardcoded secrets or credentials


"""Comprehensive input validation and sanitization for PNO systems."""

import re
import os
import torch
import numpy as np
from typing import Any, Dict, List, Optional, Union, Tuple, Callable
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
import logging
import warnings
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Base exception for validation errors."""
    pass


class SecurityValidationError(ValidationError):
    """Exception for security-related validation failures."""
    pass


class DataValidationError(ValidationError):
    """Exception for data-related validation failures."""
    pass


class ValidationSeverity(Enum):
    """Severity levels for validation issues."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationResult:
    """Result of a validation check."""
    is_valid: bool
    severity: ValidationSeverity
    message: str
    suggestion: Optional[str] = None
    sanitized_value: Optional[Any] = None


class BaseValidator(ABC):
    """Base class for validators."""
    
    def __init__(self, name: str, description: str = ""):
        """Initialize validator.
        
        Args:
            name: Validator name
            description: Validator description
        """
        self.name = name
        self.description = description
    
    @abstractmethod
    def validate(self, value: Any, context: Optional[Dict] = None) -> ValidationResult:
        """Validate a value.
        
        Args:
            value: Value to validate
            context: Additional context for validation
            
        Returns:
            ValidationResult
        """
        pass
    
    def __call__(self, value: Any, context: Optional[Dict] = None) -> ValidationResult:
        """Callable interface for validator."""
        return self.validate(value, context)


class TensorValidator(BaseValidator):
    """Validator for PyTorch tensors."""
    
    def __init__(
        self,
        name: str = "tensor_validator",
        allowed_dtypes: Optional[List[torch.dtype]] = None,
        min_dims: Optional[int] = None,
        max_dims: Optional[int] = None,
        shape_constraints: Optional[Dict[int, Union[int, Tuple[int, int]]]] = None,
        value_range: Optional[Tuple[float, float]] = None,
        allow_nan: bool = False,
        allow_inf: bool = False,
        **kwargs
    ):
        """Initialize tensor validator.
        
        Args:
            name: Validator name
            allowed_dtypes: Allowed tensor data types
            min_dims: Minimum number of dimensions
            max_dims: Maximum number of dimensions  
            shape_constraints: Constraints on specific dimensions {dim: (min, max) or exact}
            value_range: Allowed value range (min, max)
            allow_nan: Whether to allow NaN values
            allow_inf: Whether to allow infinite values
        """
        super().__init__(name, **kwargs)
        self.allowed_dtypes = allowed_dtypes or [torch.float32, torch.float64]
        self.min_dims = min_dims
        self.max_dims = max_dims
        self.shape_constraints = shape_constraints or {}
        self.value_range = value_range
        self.allow_nan = allow_nan
        self.allow_inf = allow_inf
    
    def validate(self, value: Any, context: Optional[Dict] = None) -> ValidationResult:
        """Validate tensor."""
        if not isinstance(value, torch.Tensor):
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                message=f"Expected torch.Tensor, got {type(value)}"
            )
        
        # Check data type
        if value.dtype not in self.allowed_dtypes:
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                message=f"Invalid dtype {value.dtype}. Allowed: {self.allowed_dtypes}",
                suggestion=f"Convert to one of {self.allowed_dtypes}"
            )
        
        # Check dimensions
        if self.min_dims is not None and value.dim() < self.min_dims:
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                message=f"Tensor has {value.dim()} dims, minimum required: {self.min_dims}"
            )
        
        if self.max_dims is not None and value.dim() > self.max_dims:
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                message=f"Tensor has {value.dim()} dims, maximum allowed: {self.max_dims}"
            )
        
        # Check shape constraints
        for dim, constraint in self.shape_constraints.items():
            if dim >= value.dim():
                continue
                
            size = value.shape[dim]
            if isinstance(constraint, int):
                if size != constraint:
                    return ValidationResult(
                        is_valid=False,
                        severity=ValidationSeverity.ERROR,
                        message=f"Dimension {dim} has size {size}, expected {constraint}"
                    )
            elif isinstance(constraint, tuple):
                min_size, max_size = constraint
                if size < min_size or size > max_size:
                    return ValidationResult(
                        is_valid=False,
                        severity=ValidationSeverity.ERROR,
                        message=f"Dimension {dim} size {size} outside range [{min_size}, {max_size}]"
                    )
        
        # Check for NaN and Inf
        if not self.allow_nan and torch.isnan(value).any():
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.CRITICAL,
                message="Tensor contains NaN values",
                suggestion="Clean data or set allow_nan=True"
            )
        
        if not self.allow_inf and torch.isinf(value).any():
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.CRITICAL,
                message="Tensor contains infinite values",
                suggestion="Check for numerical overflow or set allow_inf=True"
            )
        
        # Check value range
        if self.value_range is not None:
            min_val, max_val = self.value_range
            if value.min() < min_val or value.max() > max_val:
                return ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.WARNING,
                    message=f"Values outside range [{min_val}, {max_val}]",
                    suggestion=f"Clip values to range or adjust validation range"
                )
        
        return ValidationResult(
            is_valid=True,
            severity=ValidationSeverity.INFO,
            message="Tensor validation passed"
        )


class ParameterValidator(BaseValidator):
    """Validator for model parameters and hyperparameters."""
    
    def __init__(
        self,
        name: str = "parameter_validator",
        parameter_specs: Optional[Dict[str, Dict]] = None,
        **kwargs
    ):
        """Initialize parameter validator.
        
        Args:
            name: Validator name
            parameter_specs: Specifications for parameters {param_name: spec_dict}
        """
        super().__init__(name, **kwargs)
        self.parameter_specs = parameter_specs or {}
    
    def validate(self, value: Any, context: Optional[Dict] = None) -> ValidationResult:
        """Validate parameters."""
        if not isinstance(value, dict):
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                message=f"Expected dict, got {type(value)}"
            )
        
        param_name = context.get('parameter_name', 'unknown') if context else 'unknown'
        
        # Check if parameter has specifications
        if param_name in self.parameter_specs:
            spec = self.parameter_specs[param_name]
            return self._validate_against_spec(value, spec, param_name)
        
        # Generic parameter validation
        return self._validate_generic_parameters(value)
    
    def _validate_against_spec(self, value: Dict, spec: Dict, param_name: str) -> ValidationResult:
        """Validate against specific parameter specification."""
        required_keys = spec.get('required', [])
        optional_keys = spec.get('optional', [])
        value_ranges = spec.get('ranges', {})
        types = spec.get('types', {})
        
        # Check required keys
        missing_keys = set(required_keys) - set(value.keys())
        if missing_keys:
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                message=f"Missing required parameters for {param_name}: {missing_keys}"
            )
        
        # Check unknown keys
        allowed_keys = set(required_keys) | set(optional_keys)
        unknown_keys = set(value.keys()) - allowed_keys
        if unknown_keys:
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.WARNING,
                message=f"Unknown parameters for {param_name}: {unknown_keys}",
                suggestion="Remove unknown parameters or update specification"
            )
        
        # Check types and ranges
        for key, val in value.items():
            if key in types:
                expected_type = types[key]
                if not isinstance(val, expected_type):
                    return ValidationResult(
                        is_valid=False,
                        severity=ValidationSeverity.ERROR,
                        message=f"Parameter {key} has type {type(val)}, expected {expected_type}"
                    )
            
            if key in value_ranges:
                min_val, max_val = value_ranges[key]
                if not (min_val <= val <= max_val):
                    return ValidationResult(
                        is_valid=False,
                        severity=ValidationSeverity.ERROR,
                        message=f"Parameter {key}={val} outside range [{min_val}, {max_val}]"
                    )
        
        return ValidationResult(
            is_valid=True,
            severity=ValidationSeverity.INFO,
            message=f"Parameter validation passed for {param_name}"
        )
    
    def _validate_generic_parameters(self, params: Dict) -> ValidationResult:
        """Generic parameter validation."""
        issues = []
        
        # Common suspicious patterns
        for key, value in params.items():
            # Check for extremely large learning rates
            if 'learning_rate' in key.lower() or 'lr' in key.lower():
                if isinstance(value, (int, float)) and value > 1.0:
                    issues.append(f"Suspicious learning rate: {key}={value}")
            
            # Check for negative values where positive expected
            positive_params = ['batch_size', 'epochs', 'hidden_dim', 'num_layers']
            if any(p in key.lower() for p in positive_params):
                if isinstance(value, (int, float)) and value <= 0:
                    issues.append(f"Non-positive value for {key}: {value}")
        
        if issues:
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.WARNING,
                message=f"Parameter issues: {'; '.join(issues)}",
                suggestion="Review parameter values"
            )
        
        return ValidationResult(
            is_valid=True,
            severity=ValidationSeverity.INFO,
            message="Generic parameter validation passed"
        )


class PathValidator(BaseValidator):
    """Validator for file and directory paths."""
    
    def __init__(
        self,
        name: str = "path_validator",
        allowed_extensions: Optional[List[str]] = None,
        max_path_length: int = 4096,
        check_existence: bool = False,
        require_readable: bool = False,
        require_writable: bool = False,
        blocked_patterns: Optional[List[str]] = None,
        **kwargs
    ):
        """Initialize path validator.
        
        Args:
            name: Validator name
            allowed_extensions: Allowed file extensions
            max_path_length: Maximum path length
            check_existence: Whether to check if path exists
            require_readable: Whether path must be readable
            require_writable: Whether path must be writable
            blocked_patterns: Regex patterns to block
        """
        super().__init__(name, **kwargs)
        self.allowed_extensions = allowed_extensions
        self.max_path_length = max_path_length
        self.check_existence = check_existence
        self.require_readable = require_readable
        self.require_writable = require_writable
        self.blocked_patterns = [re.compile(p) for p in (blocked_patterns or [])]
        
        # Security patterns to block
        self.security_patterns = [
            re.compile(r'\.\.'),  # Directory traversal
            re.compile(r'/etc/'),  # System directories
            re.compile(r'/proc/'),
            re.compile(r'/sys/'),
            re.compile(r'\\\\'),   # Windows UNC paths
        ]
    
    def validate(self, value: Any, context: Optional[Dict] = None) -> ValidationResult:
        """Validate file path."""
        if not isinstance(value, (str, Path)):
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                message=f"Expected str or Path, got {type(value)}"
            )
        
        path_str = str(value)
        path_obj = Path(value)
        
        # Check path length
        if len(path_str) > self.max_path_length:
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                message=f"Path too long: {len(path_str)} > {self.max_path_length}"
            )
        
        # Security checks
        for pattern in self.security_patterns:
            if pattern.search(path_str):
                return ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.CRITICAL,
                    message=f"Path contains security risk pattern: {path_str}",
                    suggestion="Use absolute paths and avoid directory traversal"
                )
        
        # Check blocked patterns
        for pattern in self.blocked_patterns:
            if pattern.search(path_str):
                return ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    message=f"Path matches blocked pattern: {path_str}"
                )
        
        # Check file extension
        if self.allowed_extensions and path_obj.suffix:
            if path_obj.suffix.lower() not in self.allowed_extensions:
                return ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    message=f"Invalid extension {path_obj.suffix}. Allowed: {self.allowed_extensions}"
                )
        
        # Existence and permission checks
        if self.check_existence and not path_obj.exists():
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                message=f"Path does not exist: {path_str}"
            )
        
        if self.require_readable and not os.access(path_str, os.R_OK):
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                message=f"Path is not readable: {path_str}"
            )
        
        if self.require_writable and not os.access(path_str, os.W_OK):
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                message=f"Path is not writable: {path_str}"
            )
        
        return ValidationResult(
            is_valid=True,
            severity=ValidationSeverity.INFO,
            message="Path validation passed"
        )


class JSONValidator(BaseValidator):
    """Validator for JSON data."""
    
    def __init__(
        self,
        name: str = "json_validator",
        schema: Optional[Dict] = None,
        max_size: int = 1024 * 1024,  # 1MB
        max_depth: int = 10,
        **kwargs
    ):
        """Initialize JSON validator.
        
        Args:
            name: Validator name
            schema: JSON schema for validation
            max_size: Maximum JSON size in bytes
            max_depth: Maximum nesting depth
        """
        super().__init__(name, **kwargs)
        self.schema = schema
        self.max_size = max_size
        self.max_depth = max_depth
    
    def validate(self, value: Any, context: Optional[Dict] = None) -> ValidationResult:
        """Validate JSON data."""
        try:
            # Convert to JSON string if needed
            if isinstance(value, (dict, list)):
                json_str = json.dumps(value)
            elif isinstance(value, str):
                json_str = value
                value = json.loads(json_str)  # Parse to validate
            else:
                return ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    message=f"Invalid JSON input type: {type(value)}"
                )
            
            # Check size
            if len(json_str) > self.max_size:
                return ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    message=f"JSON too large: {len(json_str)} > {self.max_size} bytes"
                )
            
            # Check depth
            depth = self._calculate_depth(value)
            if depth > self.max_depth:
                return ValidationResult(
                    is_valid=False,
                    severity=ValidationSeverity.ERROR,
                    message=f"JSON too deeply nested: {depth} > {self.max_depth}"
                )
            
            # Schema validation (basic)
            if self.schema:
                validation_errors = self._validate_schema(value, self.schema)
                if validation_errors:
                    return ValidationResult(
                        is_valid=False,
                        severity=ValidationSeverity.ERROR,
                        message=f"Schema validation failed: {validation_errors}"
                    )
            
            return ValidationResult(
                is_valid=True,
                severity=ValidationSeverity.INFO,
                message="JSON validation passed"
            )
            
        except json.JSONDecodeError as e:
            return ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                message=f"Invalid JSON: {e}"
            )
    
    def _calculate_depth(self, obj: Any, current_depth: int = 0) -> int:
        """Calculate nesting depth of JSON object."""
        if current_depth > self.max_depth:
            return current_depth
        
        if isinstance(obj, dict):
            return max(
                self._calculate_depth(v, current_depth + 1) 
                for v in obj.values()
            ) if obj else current_depth
        elif isinstance(obj, list):
            return max(
                self._calculate_depth(item, current_depth + 1) 
                for item in obj
            ) if obj else current_depth
        else:
            return current_depth
    
    def _validate_schema(self, data: Any, schema: Dict) -> List[str]:
        """Basic schema validation."""
        errors = []
        
        if 'type' in schema:
            expected_type = schema['type']
            type_map = {
                'object': dict,
                'array': list,
                'string': str,
                'number': (int, float),
                'boolean': bool,
                'null': type(None)
            }
            
            if expected_type in type_map:
                if not isinstance(data, type_map[expected_type]):
                    errors.append(f"Expected {expected_type}, got {type(data).__name__}")
        
        # Additional schema validation can be added here
        return errors


class InputSanitizer:
    """Comprehensive input sanitization system."""
    
    def __init__(self):
        """Initialize sanitizer."""
        self.validators: Dict[str, BaseValidator] = {}
        self.default_validators = self._create_default_validators()
    
    def _create_default_validators(self) -> Dict[str, BaseValidator]:
        """Create default validators."""
        return {
            'tensor': TensorValidator(),
            'parameter': ParameterValidator(),
            'path': PathValidator(),
            'json': JSONValidator()
        }
    
    def register_validator(self, name: str, validator: BaseValidator):
        """Register a custom validator."""
        self.validators[name] = validator
        logger.info(f"Registered validator: {name}")
    
    def validate(
        self,
        value: Any,
        validator_name: str,
        context: Optional[Dict] = None,
        raise_on_error: bool = True
    ) -> ValidationResult:
        """Validate input using specified validator.
        
        Args:
            value: Value to validate
            validator_name: Name of validator to use
            context: Additional context
            raise_on_error: Whether to raise exception on validation failure
            
        Returns:
            ValidationResult
            
        Raises:
            ValidationError: If validation fails and raise_on_error=True
        """
        # Get validator
        validator = self.validators.get(validator_name) or self.default_validators.get(validator_name)
        
        if validator is None:
            raise ValueError(f"Unknown validator: {validator_name}")
        
        # Perform validation
        result = validator.validate(value, context)
        
        # Handle errors
        if not result.is_valid and raise_on_error:
            if result.severity == ValidationSeverity.CRITICAL:
                raise SecurityValidationError(result.message)
            else:
                raise DataValidationError(result.message)
        
        # Log issues
        if not result.is_valid:
            log_level = {
                ValidationSeverity.INFO: logging.INFO,
                ValidationSeverity.WARNING: logging.WARNING,
                ValidationSeverity.ERROR: logging.ERROR,
                ValidationSeverity.CRITICAL: logging.CRITICAL
            }[result.severity]
            
            logger.log(log_level, f"Validation {result.severity.value}: {result.message}")
        
        return result
    
    def sanitize_tensor(
        self,
        tensor: torch.Tensor,
        clip_range: Optional[Tuple[float, float]] = None,
        replace_nan: Optional[float] = None,
        replace_inf: Optional[float] = None
    ) -> torch.Tensor:
        """Sanitize tensor data.
        
        Args:
            tensor: Input tensor
            clip_range: Range to clip values
            replace_nan: Value to replace NaN with
            replace_inf: Value to replace Inf with
            
        Returns:
            Sanitized tensor
        """
        sanitized = tensor.clone()
        
        # Replace NaN values
        if replace_nan is not None:
            sanitized = torch.where(torch.isnan(sanitized), replace_nan, sanitized)
        
        # Replace Inf values
        if replace_inf is not None:
            sanitized = torch.where(torch.isinf(sanitized), replace_inf, sanitized)
        
        # Clip values
        if clip_range is not None:
            sanitized = torch.clamp(sanitized, clip_range[0], clip_range[1])
        
        return sanitized
    
    def sanitize_path(self, path: Union[str, Path]) -> Path:
        """Sanitize file path.
        
        Args:
            path: Input path
            
        Returns:
            Sanitized path
        """
        # Convert to Path object
        path_obj = Path(path)
        
        # Resolve to absolute path (prevents traversal)
        try:
            sanitized = path_obj.resolve()
        except (OSError, ValueError):
            # If resolution fails, use the original path
            sanitized = path_obj
        
        return sanitized


# Global sanitizer instance
input_sanitizer = InputSanitizer()


def validate_input(
    value: Any,
    validator_name: str,
    context: Optional[Dict] = None,
    **kwargs
) -> ValidationResult:
    """Convenience function for input validation."""
    return input_sanitizer.validate(value, validator_name, context, **kwargs)


def sanitize_tensor(tensor: torch.Tensor, **kwargs) -> torch.Tensor:
    """Convenience function for tensor sanitization."""
    return input_sanitizer.sanitize_tensor(tensor, **kwargs)