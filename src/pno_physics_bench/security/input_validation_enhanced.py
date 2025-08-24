"""
Enhanced Input Validation Module
===============================

Provides comprehensive input validation and sanitization for the pno-physics-bench package.
This module implements security best practices to prevent injection attacks and validate
all user inputs.
"""

import re
import json
from typing import Any, Dict, List, Union, Optional, Tuple
from pathlib import Path
import ast

class InputValidator:
    """Comprehensive input validator with security hardening."""
    
    def __init__(self):
        self.max_string_length = 10000
        self.max_list_length = 1000
        self.max_dict_keys = 100
        self.allowed_file_extensions = {'.json', '.yaml', '.yml', '.txt', '.csv'}
        
    def validate_string(self, value: Any, field_name: str = "input") -> Tuple[bool, Optional[str]]:
        """Validate string input with security checks."""
        if not isinstance(value, str):
            return False, f"{field_name} must be a string"
        
        if len(value) > self.max_string_length:
            return False, f"{field_name} exceeds maximum length of {self.max_string_length}"
        
        # Check for potentially dangerous patterns
        dangerous_patterns = [
            r'<script[^>]*>.*?</script>',  # Script tags
            r'javascript:',  # JavaScript URLs
            r'on\w+\s*=',  # Event handlers
            r'eval\s*\(',  # eval calls
            r'exec\s*\(',  # exec calls
            r'__import__',  # import calls
            r'\.\./.*',  # Path traversal
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, value, re.IGNORECASE):
                return False, f"{field_name} contains potentially dangerous content"
        
        return True, None
    
    def validate_numeric(self, value: Any, field_name: str = "input", 
                        min_val: float = None, max_val: float = None) -> Tuple[bool, Optional[str]]:
        """Validate numeric input with range checks."""
        try:
            if isinstance(value, str):
                # Try to convert string to number
                if '.' in value:
                    num_value = float(value)
                else:
                    num_value = int(value)
            elif isinstance(value, (int, float)):
                num_value = value
            else:
                return False, f"{field_name} must be a number"
            
            if min_val is not None and num_value < min_val:
                return False, f"{field_name} must be >= {min_val}"
            
            if max_val is not None and num_value > max_val:
                return False, f"{field_name} must be <= {max_val}"
            
            return True, None
            
        except (ValueError, OverflowError):
            return False, f"{field_name} is not a valid number"
    
    def validate_list(self, value: Any, field_name: str = "input",
                     max_length: int = None) -> Tuple[bool, Optional[str]]:
        """Validate list input with size limits."""
        if not isinstance(value, list):
            return False, f"{field_name} must be a list"
        
        max_len = max_length or self.max_list_length
        if len(value) > max_len:
            return False, f"{field_name} exceeds maximum length of {max_len}"
        
        return True, None
    
    def validate_dict(self, value: Any, field_name: str = "input",
                     required_keys: List[str] = None) -> Tuple[bool, Optional[str]]:
        """Validate dictionary input with key validation."""
        if not isinstance(value, dict):
            return False, f"{field_name} must be a dictionary"
        
        if len(value) > self.max_dict_keys:
            return False, f"{field_name} has too many keys (max: {self.max_dict_keys})"
        
        # Validate all keys are strings
        for key in value.keys():
            if not isinstance(key, str):
                return False, f"{field_name} contains non-string key: {key}"
            
            is_valid, error = self.validate_string(key, f"{field_name} key")
            if not is_valid:
                return False, error
        
        # Check required keys
        if required_keys:
            missing_keys = set(required_keys) - set(value.keys())
            if missing_keys:
                return False, f"{field_name} missing required keys: {list(missing_keys)}"
        
        return True, None
    
    def validate_file_path(self, value: Any, field_name: str = "path") -> Tuple[bool, Optional[str]]:
        """Validate file path with security checks."""
        is_valid, error = self.validate_string(value, field_name)
        if not is_valid:
            return False, error
        
        try:
            path = Path(value)
            
            # Check for path traversal
            if '..' in path.parts:
                return False, f"{field_name} contains invalid path traversal"
            
            # Check file extension if file
            if path.suffix and path.suffix.lower() not in self.allowed_file_extensions:
                return False, f"{field_name} has unauthorized file extension: {path.suffix}"
            
            return True, None
            
        except Exception:
            return False, f"{field_name} is not a valid path"
    
    def sanitize_string(self, value: str) -> str:
        """Sanitize string input by removing/escaping dangerous content."""
        if not isinstance(value, str):
            return str(value)
        
        # Remove dangerous HTML/JavaScript patterns
        value = re.sub(r'<script[^>]*>.*?</script>', '', value, flags=re.IGNORECASE)
        value = re.sub(r'javascript:', '', value, flags=re.IGNORECASE)
        value = re.sub(r'on\w+\s*=', '', value, flags=re.IGNORECASE)
        
        # Escape special characters
        value = value.replace('<', '&lt;').replace('>', '&gt;')
        value = value.replace('"', '&quot;').replace("'", '&#x27;')
        
        return value
    
    def validate_json_config(self, config: Union[str, dict], field_name: str = "config") -> Tuple[bool, Optional[str]]:
        """Validate JSON configuration with security checks."""
        try:
            if isinstance(config, str):
                # Parse JSON string safely
                parsed_config = json.loads(config)
            elif isinstance(config, dict):
                parsed_config = config
            else:
                return False, f"{field_name} must be JSON string or dictionary"
            
            # Validate the dictionary
            return self.validate_dict(parsed_config, field_name)
            
        except json.JSONDecodeError as e:
            return False, f"{field_name} is not valid JSON: {str(e)}"
    
    def validate_tensor_config(self, config: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Validate tensor/model configuration parameters."""
        required_keys = ['input_dim', 'output_dim']
        
        is_valid, error = self.validate_dict(config, "tensor_config", required_keys)
        if not is_valid:
            return False, error
        
        # Validate specific tensor parameters
        for key, value in config.items():
            if key in ['input_dim', 'output_dim', 'hidden_dim', 'num_layers']:
                is_valid, error = self.validate_numeric(value, key, min_val=1, max_val=10000)
                if not is_valid:
                    return False, error
            
            elif key in ['learning_rate', 'dropout_rate']:
                is_valid, error = self.validate_numeric(value, key, min_val=0.0, max_val=1.0)
                if not is_valid:
                    return False, error
            
            elif key == 'activation':
                allowed_activations = ['relu', 'tanh', 'sigmoid', 'gelu', 'swish']
                if value not in allowed_activations:
                    return False, f"activation must be one of: {allowed_activations}"
        
        return True, None

# Global validator instance
validator = InputValidator()

def validate_input(value: Any, validation_type: str = "string", **kwargs) -> Tuple[bool, Optional[str]]:
    """Convenient function for input validation."""
    if validation_type == "string":
        return validator.validate_string(value, **kwargs)
    elif validation_type == "numeric":
        return validator.validate_numeric(value, **kwargs)
    elif validation_type == "list":
        return validator.validate_list(value, **kwargs)
    elif validation_type == "dict":
        return validator.validate_dict(value, **kwargs)
    elif validation_type == "path":
        return validator.validate_file_path(value, **kwargs)
    elif validation_type == "json":
        return validator.validate_json_config(value, **kwargs)
    else:
        return False, f"Unknown validation type: {validation_type}"
