# SECURITY NOTICE: This file has been automatically hardened for security
# - All user inputs should be validated and sanitized
# - Subprocess calls use secure alternatives
# - SQL queries use parameterized statements
# - No hardcoded secrets or credentials


"""Input validation and sanitization utilities."""

import torch
import numpy as np
from typing import Any, Dict, List, Optional, Union, Tuple
from pathlib import Path
import re

from .exceptions import (
    ValidationError, 
    InvalidParameterError, 
    DimensionMismatchError,
    ConfigurationError,
    InvalidConfigError,
)


class InputValidator:
    """Comprehensive input validation and sanitization."""
    
    @staticmethod
    def validate_tensor(
        tensor: torch.Tensor,
        name: str,
        expected_shape: Optional[Tuple[int, ...]] = None,
        expected_dtype: Optional[torch.dtype] = None,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        allow_nan: bool = False,
        allow_inf: bool = False,
    ) -> torch.Tensor:
        """Validate and sanitize tensor inputs.
        
        Args:
            tensor: Input tensor
            name: Name for error messages
            expected_shape: Expected tensor shape (None for any)
            expected_dtype: Expected data type
            min_value: Minimum allowed value
            max_value: Maximum allowed value
            allow_nan: Whether to allow NaN values
            allow_inf: Whether to allow infinite values
            
        Returns:
            Validated tensor
            
        Raises:
            ValidationError: If validation fails
        """
        if not isinstance(tensor, torch.Tensor):
            raise InvalidParameterError(
                f"{name} must be a torch.Tensor, got {type(tensor)}",
                error_code="INVALID_TENSOR_TYPE"
            )
        
        # Check shape
        if expected_shape is not None:
            if tensor.shape != expected_shape:
                raise DimensionMismatchError(
                    f"{name} shape mismatch: expected {expected_shape}, got {tensor.shape}",
                    error_code="SHAPE_MISMATCH",
                    details={"expected": expected_shape, "actual": tensor.shape}
                )
        
        # Check dtype
        if expected_dtype is not None and tensor.dtype != expected_dtype:
            raise InvalidParameterError(
                f"{name} dtype mismatch: expected {expected_dtype}, got {tensor.dtype}",
                error_code="DTYPE_MISMATCH"
            )
        
        # Check for NaN values
        if not allow_nan and torch.isnan(tensor).any():
            raise ValidationError(
                f"{name} contains NaN values",
                error_code="NAN_VALUES_DETECTED"
            )
        
        # Check for infinite values
        if not allow_inf and torch.isinf(tensor).any():
            raise ValidationError(
                f"{name} contains infinite values",
                error_code="INF_VALUES_DETECTED"
            )
        
        # Check value range
        if min_value is not None and tensor.min() < min_value:
            raise ValidationError(
                f"{name} contains values below minimum {min_value}",
                error_code="VALUE_BELOW_MINIMUM",
                details={"min_allowed": min_value, "actual_min": tensor.min().item()}
            )
        
        if max_value is not None and tensor.max() > max_value:
            raise ValidationError(
                f"{name} contains values above maximum {max_value}",
                error_code="VALUE_ABOVE_MAXIMUM",
                details={"max_allowed": max_value, "actual_max": tensor.max().item()}
            )
        
        return tensor
    
    @staticmethod
    def validate_positive_number(
        value: Union[int, float],
        name: str,
        allow_zero: bool = False,
    ) -> Union[int, float]:
        """Validate positive number.
        
        Args:
            value: Number to validate
            name: Name for error messages
            allow_zero: Whether to allow zero
            
        Returns:
            Validated number
        """
        if not isinstance(value, (int, float)):
            raise InvalidParameterError(
                f"{name} must be a number, got {type(value)}",
                error_code="INVALID_NUMBER_TYPE"
            )
        
        if np.isnan(value) or np.isinf(value):
            raise ValidationError(
                f"{name} must be finite, got {value}",
                error_code="NON_FINITE_VALUE"
            )
        
        min_allowed = 0 if allow_zero else 1e-10
        if value < min_allowed:
            raise ValidationError(
                f"{name} must be {'non-negative' if allow_zero else 'positive'}, got {value}",
                error_code="NEGATIVE_VALUE"
            )
        
        return value
    
    @staticmethod
    def validate_probability(value: float, name: str) -> float:
        """Validate probability value (0 <= p <= 1).
        
        Args:
            value: Probability value
            name: Name for error messages
            
        Returns:
            Validated probability
        """
        if not isinstance(value, (int, float)):
            raise InvalidParameterError(
                f"{name} must be a number, got {type(value)}",
                error_code="INVALID_PROBABILITY_TYPE"
            )
        
        if not (0 <= value <= 1):
            raise ValidationError(
                f"{name} must be between 0 and 1, got {value}",
                error_code="INVALID_PROBABILITY_RANGE"
            )
        
        return float(value)
    
    @staticmethod
    def validate_string(
        value: str,
        name: str,
        allowed_values: Optional[List[str]] = None,
        pattern: Optional[str] = None,
        min_length: int = 0,
        max_length: Optional[int] = None,
    ) -> str:
        """Validate string input.
        
        Args:
            value: String to validate
            name: Name for error messages
            allowed_values: List of allowed values
            pattern: Regex pattern to match
            min_length: Minimum string length
            max_length: Maximum string length
            
        Returns:
            Validated string
        """
        if not isinstance(value, str):
            raise InvalidParameterError(
                f"{name} must be a string, got {type(value)}",
                error_code="INVALID_STRING_TYPE"
            )
        
        # Check length
        if len(value) < min_length:
            raise ValidationError(
                f"{name} too short: minimum {min_length} characters, got {len(value)}",
                error_code="STRING_TOO_SHORT"
            )
        
        if max_length is not None and len(value) > max_length:
            raise ValidationError(
                f"{name} too long: maximum {max_length} characters, got {len(value)}",
                error_code="STRING_TOO_LONG"
            )
        
        # Check allowed values
        if allowed_values is not None and value not in allowed_values:
            raise ValidationError(
                f"{name} must be one of {allowed_values}, got '{value}'",
                error_code="INVALID_STRING_VALUE",
                details={"allowed": allowed_values, "actual": value}
            )
        
        # Check pattern
        if pattern is not None and not re.match(pattern, value):
            raise ValidationError(
                f"{name} does not match required pattern '{pattern}', got '{value}'",
                error_code="INVALID_STRING_PATTERN"
            )
        
        return value
    
    @staticmethod
    def validate_file_path(
        path: Union[str, Path],
        name: str,
        must_exist: bool = False,
        must_be_file: bool = False,
        must_be_dir: bool = False,
        allowed_extensions: Optional[List[str]] = None,
    ) -> Path:
        """Validate file path.
        
        Args:
            path: File path to validate
            name: Name for error messages
            must_exist: Whether path must exist
            must_be_file: Whether path must be a file
            must_be_dir: Whether path must be a directory
            allowed_extensions: List of allowed file extensions
            
        Returns:
            Validated Path object
        """
        if isinstance(path, str):
            path = Path(path)
        elif not isinstance(path, Path):
            raise InvalidParameterError(
                f"{name} must be a string or Path, got {type(path)}",
                error_code="INVALID_PATH_TYPE"
            )
        
        # Check existence
        if must_exist and not path.exists():
            raise ValidationError(
                f"{name} does not exist: {path}",
                error_code="PATH_DOES_NOT_EXIST"
            )
        
        if path.exists():
            # Check if it's a file when required
            if must_be_file and not path.is_file():
                raise ValidationError(
                    f"{name} must be a file, got directory: {path}",
                    error_code="PATH_NOT_FILE"
                )
            
            # Check if it's a directory when required
            if must_be_dir and not path.is_dir():
                raise ValidationError(
                    f"{name} must be a directory, got file: {path}",
                    error_code="PATH_NOT_DIRECTORY"
                )
        
        # Check extension
        if allowed_extensions is not None and path.suffix.lower() not in allowed_extensions:
            raise ValidationError(
                f"{name} must have extension {allowed_extensions}, got '{path.suffix}'",
                error_code="INVALID_FILE_EXTENSION",
                details={"allowed": allowed_extensions, "actual": path.suffix}
            )
        
        return path
    
    @staticmethod
    def validate_device(device: Union[str, torch.device], name: str = "device") -> torch.device:
        """Validate PyTorch device.
        
        Args:
            device: Device specification
            name: Name for error messages
            
        Returns:
            Validated torch.device
        """
        if isinstance(device, str):
            try:
                device = torch.device(device)
            except Exception as e:
                raise ValidationError(
                    f"Invalid {name} specification: {device}",
                    error_code="INVALID_DEVICE",
                    details={"error": str(e)}
                )
        elif not isinstance(device, torch.device):
            raise InvalidParameterError(
                f"{name} must be string or torch.device, got {type(device)}",
                error_code="INVALID_DEVICE_TYPE"
            )
        
        # Check if CUDA device is available when requested
        if device.type == 'cuda':
            if not torch.cuda.is_available():
                raise ValidationError(
                    f"CUDA not available, cannot use device {device}",
                    error_code="CUDA_NOT_AVAILABLE"
                )
            
            if device.index is not None and device.index >= torch.cuda.device_count():
                raise ValidationError(
                    f"CUDA device {device.index} not available, only {torch.cuda.device_count()} devices",
                    error_code="INVALID_CUDA_DEVICE"
                )
        
        return device


class ConfigValidator:
    """Validation for configuration dictionaries."""
    
    @staticmethod
    def validate_model_config(config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate model configuration.
        
        Args:
            config: Model configuration dictionary
            
        Returns:
            Validated configuration
        """
        required_fields = ['input_dim', 'hidden_dim', 'num_layers', 'modes']
        
        for field in required_fields:
            if field not in config:
                raise InvalidConfigError(
                    f"Missing required model config field: {field}",
                    error_code="MISSING_MODEL_CONFIG"
                )
        
        # Validate specific fields
        config['input_dim'] = InputValidator.validate_positive_number(
            config['input_dim'], 'input_dim'
        )
        config['hidden_dim'] = InputValidator.validate_positive_number(
            config['hidden_dim'], 'hidden_dim'
        )
        config['num_layers'] = InputValidator.validate_positive_number(
            config['num_layers'], 'num_layers'
        )
        config['modes'] = InputValidator.validate_positive_number(
            config['modes'], 'modes'
        )
        
        # Optional fields with validation
        if 'output_dim' in config:
            config['output_dim'] = InputValidator.validate_positive_number(
                config['output_dim'], 'output_dim'
            )
        
        if 'uncertainty_type' in config:
            config['uncertainty_type'] = InputValidator.validate_string(
                config['uncertainty_type'], 
                'uncertainty_type',
                allowed_values=['diagonal', 'full']
            )
        
        if 'activation' in config:
            config['activation'] = InputValidator.validate_string(
                config['activation'],
                'activation',
                allowed_values=['gelu', 'relu', 'tanh', 'silu']
            )
        
        return config
    
    @staticmethod
    def validate_training_config(config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate training configuration.
        
        Args:
            config: Training configuration dictionary
            
        Returns:
            Validated configuration
        """
        required_fields = ['epochs', 'batch_size', 'learning_rate']
        
        for field in required_fields:
            if field not in config:
                raise InvalidConfigError(
                    f"Missing required training config field: {field}",
                    error_code="MISSING_TRAINING_CONFIG"
                )
        
        # Validate fields
        config['epochs'] = int(InputValidator.validate_positive_number(
            config['epochs'], 'epochs'
        ))
        config['batch_size'] = int(InputValidator.validate_positive_number(
            config['batch_size'], 'batch_size'
        ))
        config['learning_rate'] = InputValidator.validate_positive_number(
            config['learning_rate'], 'learning_rate'
        )
        
        # Optional fields
        if 'weight_decay' in config:
            config['weight_decay'] = InputValidator.validate_positive_number(
                config['weight_decay'], 'weight_decay', allow_zero=True
            )
        
        if 'kl_weight' in config:
            config['kl_weight'] = InputValidator.validate_positive_number(
                config['kl_weight'], 'kl_weight', allow_zero=True
            )
        
        if 'gradient_clipping' in config:
            config['gradient_clipping'] = InputValidator.validate_positive_number(
                config['gradient_clipping'], 'gradient_clipping', allow_zero=True
            )
        
        if 'num_mc_samples' in config:
            config['num_mc_samples'] = int(InputValidator.validate_positive_number(
                config['num_mc_samples'], 'num_mc_samples'
            ))
        
        return config
    
    @staticmethod
    def validate_data_config(config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate data configuration.
        
        Args:
            config: Data configuration dictionary
            
        Returns:
            Validated configuration
        """
        required_fields = ['pde_name', 'resolution', 'num_samples']
        
        for field in required_fields:
            if field not in config:
                raise InvalidConfigError(
                    f"Missing required data config field: {field}",
                    error_code="MISSING_DATA_CONFIG"
                )
        
        # Validate PDE name
        allowed_pdes = [
            'navier_stokes_2d', 'darcy_flow_2d', 'burgers_1d', 'heat_3d'
        ]
        config['pde_name'] = InputValidator.validate_string(
            config['pde_name'],
            'pde_name',
            allowed_values=allowed_pdes
        )
        
        # Validate numerical fields
        config['resolution'] = int(InputValidator.validate_positive_number(
            config['resolution'], 'resolution'
        ))
        config['num_samples'] = int(InputValidator.validate_positive_number(
            config['num_samples'], 'num_samples'
        ))
        
        # Validate split ratios
        if 'val_split' in config:
            config['val_split'] = InputValidator.validate_probability(
                config['val_split'], 'val_split'
            )
        
        if 'test_split' in config:
            config['test_split'] = InputValidator.validate_probability(
                config['test_split'], 'test_split'
            )
        
        # Validate splits sum to <= 1
        val_split = config.get('val_split', 0.2)
        test_split = config.get('test_split', 0.1)
        if val_split + test_split >= 1.0:
            raise ValidationError(
                f"val_split + test_split must be < 1.0, got {val_split + test_split}",
                error_code="INVALID_SPLIT_RATIOS"
            )
        
        return config


class SecurityValidator:
    """Security-focused validation and sanitization."""
    
    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """Sanitize filename to prevent directory traversal attacks.
        
        Args:
            filename: Original filename
            
        Returns:
            Sanitized filename
        """
        # Remove directory components
        filename = Path(filename).name
        
        # Remove or replace dangerous characters
        dangerous_chars = ['<', '>', ':', '"', '|', '?', '*', '\x00']
        for char in dangerous_chars:
            filename = filename.replace(char, '_')
        
        # Limit length
        if len(filename) > 255:
            name, ext = Path(filename).stem, Path(filename).suffix
            filename = name[:255-len(ext)] + ext
        
        # Ensure not empty
        if not filename or filename in ['.', '..']:
            filename = 'file.txt'
        
        return filename
    
    @staticmethod
    def validate_yaml_safe(content: str) -> bool:
        """Check if YAML content is safe to load.
        
        Args:
            content: YAML content string
            
        Returns:
            True if safe, False otherwise
        """
        # Check for potentially dangerous YAML constructs
        dangerous_patterns = [
            r'!!python/',  # Python objects
            r'!!map\s*\{',  # Inline mappings that could be exploited
            r'&\w+',  # Anchors (can cause memory issues)
            r'\*\w+',  # Aliases
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                return False
        
        return True
    
    @staticmethod
    def validate_code_injection(input_string: str) -> str:
        """Validate string for potential code injection.
        
        Args:
            input_string: String to validate
            
        Returns:
            Validated string
            
        Raises:
            ValidationError: If dangerous patterns detected
        """
        dangerous_patterns = [
            r'__import__',
            r'exec\s*\(',
            r'eval\s*\(',
            r'compile\s*\(',
            r'open\s*\(',
            r'file\s*\(',
            r'subprocess',
            r'os\.system',
            r'os\.popen',
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, input_string, re.IGNORECASE):
                raise ValidationError(
                    f"Potentially dangerous code pattern detected: {pattern}",
                    error_code="CODE_INJECTION_ATTEMPT"
                )
        
        return input_string