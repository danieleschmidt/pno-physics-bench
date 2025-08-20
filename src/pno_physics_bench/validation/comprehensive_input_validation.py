# SECURITY NOTICE: This file has been automatically hardened for security
# - All user inputs should be validated and sanitized
# - Subprocess calls use secure alternatives
# - SQL queries use parameterized statements
# - No hardcoded secrets or credentials


"""Comprehensive Input Validation Framework"""

import numpy as np
import torch
from typing import Any, Union, Tuple, Optional, List
import re

class ValidationError(Exception):
    """Custom validation error"""
    pass

class InputValidator:
    """Comprehensive input validation for PNO components"""
    
    @staticmethod
    def validate_tensor_shape(tensor: torch.Tensor, expected_shape: Optional[Tuple] = None,
                            min_dims: int = 1, max_dims: int = 6) -> bool:
        """Validate tensor shape and dimensions"""
        if not isinstance(tensor, torch.Tensor):
            raise ValidationError(f"Expected torch.Tensor, got {type(tensor)}")
        
        if len(tensor.shape) < min_dims or len(tensor.shape) > max_dims:
            raise ValidationError(f"Tensor dimensions {len(tensor.shape)} outside valid range [{min_dims}, {max_dims}]")
        
        if expected_shape and tensor.shape != expected_shape:
            raise ValidationError(f"Expected shape {expected_shape}, got {tensor.shape}")
        
        return True
    
    @staticmethod
    def validate_numerical_range(value: Union[float, int, torch.Tensor, np.ndarray],
                                min_val: float = -1e6, max_val: float = 1e6) -> bool:
        """Validate numerical values are within reasonable range"""
        if isinstance(value, (torch.Tensor, np.ndarray)):
            if torch.any(torch.isnan(value)) if isinstance(value, torch.Tensor) else np.any(np.isnan(value)):
                raise ValidationError("Input contains NaN values")
            
            if torch.any(torch.isinf(value)) if isinstance(value, torch.Tensor) else np.any(np.isinf(value)):
                raise ValidationError("Input contains infinite values")
            
            min_check = torch.min(value) >= min_val if isinstance(value, torch.Tensor) else np.min(value) >= min_val
            max_check = torch.max(value) <= max_val if isinstance(value, torch.Tensor) else np.max(value) <= max_val
            
            if not min_check or not max_check:
                raise ValidationError(f"Values outside valid range [{min_val}, {max_val}]")
        else:
            if np.isnan(value) or np.isinf(value):
                raise ValidationError("Input contains NaN or infinite values")
            
            if not (min_val <= value <= max_val):
                raise ValidationError(f"Value {value} outside valid range [{min_val}, {max_val}]")
        
        return True
    
    @staticmethod
    def validate_probability_distribution(probs: Union[torch.Tensor, np.ndarray],
                                        dim: int = -1, tolerance: float = 1e-6) -> bool:
        """Validate probability distribution sums to 1"""
        if isinstance(probs, torch.Tensor):
            prob_sum = torch.sum(probs, dim=dim)
            valid = torch.all(torch.abs(prob_sum - 1.0) < tolerance)
        else:
            prob_sum = np.sum(probs, axis=dim)
            valid = np.all(np.abs(prob_sum - 1.0) < tolerance)
        
        if not valid:
            raise ValidationError("Probability distribution does not sum to 1")
        
        return True
    
    @staticmethod
    def validate_model_config(config: dict) -> bool:
        """Validate model configuration parameters"""
        required_keys = ['input_dim', 'hidden_dim', 'num_layers']
        for key in required_keys:
            if key not in config:
                raise ValidationError(f"Required config key '{key}' missing")
        
        # Validate ranges
        if config['input_dim'] <= 0 or config['input_dim'] > 1000:
            raise ValidationError(f"Invalid input_dim: {config['input_dim']}")
        
        if config['hidden_dim'] <= 0 or config['hidden_dim'] > 10000:
            raise ValidationError(f"Invalid hidden_dim: {config['hidden_dim']}")
        
        if config['num_layers'] <= 0 or config['num_layers'] > 100:
            raise ValidationError(f"Invalid num_layers: {config['num_layers']}")
        
        return True
    
    @staticmethod
    def sanitize_string_input(text: str, max_length: int = 1000,
                            allowed_chars: str = r'[a-zA-Z0-9\s\-_\.]') -> str:
        """Sanitize string inputs for security"""
        if len(text) > max_length:
            raise ValidationError(f"String length {len(text)} exceeds maximum {max_length}")
        
        if not re.match(f'^{allowed_chars}*$', text):
            raise ValidationError("String contains invalid characters")
        
        return text.strip()
    
    @staticmethod
    def validate_file_path(path: str, allowed_extensions: List[str] = None) -> bool:
        """Validate file paths for security"""
        # Prevent path traversal attacks
        if '..' in path or path.startswith('/'):
            raise ValidationError("Invalid file path - path traversal detected")
        
        if allowed_extensions:
            extension = path.split('.')[-1].lower()
            if extension not in allowed_extensions:
                raise ValidationError(f"File extension '{extension}' not allowed")
        
        return True

def validate_pno_inputs(func):
    """Decorator for automatic PNO input validation"""
    def wrapper(*args, **kwargs):
        validator = InputValidator()
        
        # Validate tensor inputs
        for arg in args:
            if isinstance(arg, torch.Tensor):
                validator.validate_tensor_shape(arg)
                validator.validate_numerical_range(arg)
        
        for value in kwargs.values():
            if isinstance(value, torch.Tensor):
                validator.validate_tensor_shape(value)
                validator.validate_numerical_range(value)
            elif isinstance(value, dict) and 'config' in str(type(value)).lower():
                validator.validate_model_config(value)
        
        return func(*args, **kwargs)
    return wrapper
