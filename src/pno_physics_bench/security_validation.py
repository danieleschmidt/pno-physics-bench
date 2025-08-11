"""Security validation and input sanitization for PNO systems."""

import re
import hashlib
import time
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from pathlib import Path
import json

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

import numpy as np


@dataclass
class SecurityThreat:
    """Represents a detected security threat."""
    threat_id: str
    threat_type: str
    severity: str  # critical, high, medium, low
    description: str
    source: str
    timestamp: float
    mitigation: str
    blocked: bool


class InputValidator:
    """Comprehensive input validation for PNO systems."""
    
    def __init__(self, 
                 max_tensor_size: int = 100_000_000,  # 100M elements
                 max_batch_size: int = 1000,
                 allowed_dtypes: Optional[List[str]] = None):
        
        self.max_tensor_size = max_tensor_size
        self.max_batch_size = max_batch_size
        self.allowed_dtypes = allowed_dtypes or ['float32', 'float64', 'int32', 'int64']
        
        self.validation_history = []
        self.threat_log = []
        self.logger = logging.getLogger(__name__)
    
    def validate_tensor_input(self, tensor: Any, input_name: str = "input") -> Tuple[bool, Optional[SecurityThreat]]:
        """Validate tensor inputs for security threats."""
        
        threat = None
        
        # Check if input is tensor-like
        if HAS_TORCH and isinstance(tensor, torch.Tensor):
            return self._validate_torch_tensor(tensor, input_name)
        elif isinstance(tensor, np.ndarray):
            return self._validate_numpy_array(tensor, input_name)
        elif isinstance(tensor, (list, tuple)):
            return self._validate_sequence(tensor, input_name)
        else:
            threat = SecurityThreat(
                threat_id=f"invalid_type_{int(time.time())}",
                threat_type="invalid_input_type",
                severity="medium",
                description=f"Input {input_name} has invalid type: {type(tensor)}",
                source="input_validation",
                timestamp=time.time(),
                mitigation="reject_input",
                blocked=True
            )
            return False, threat
        
        return True, None
    
    def _validate_torch_tensor(self, tensor: 'torch.Tensor', input_name: str) -> Tuple[bool, Optional[SecurityThreat]]:
        """Validate PyTorch tensor."""
        
        # Check tensor size
        if tensor.numel() > self.max_tensor_size:
            threat = SecurityThreat(
                threat_id=f"tensor_size_{int(time.time())}",
                threat_type="excessive_tensor_size",
                severity="high",
                description=f"Tensor {input_name} size {tensor.numel()} exceeds limit {self.max_tensor_size}",
                source="input_validation",
                timestamp=time.time(),
                mitigation="reject_input",
                blocked=True
            )
            return False, threat
        
        # Check batch size
        if len(tensor.shape) > 0 and tensor.shape[0] > self.max_batch_size:
            threat = SecurityThreat(
                threat_id=f"batch_size_{int(time.time())}",
                threat_type="excessive_batch_size",
                severity="medium",
                description=f"Batch size {tensor.shape[0]} exceeds limit {self.max_batch_size}",
                source="input_validation", 
                timestamp=time.time(),
                mitigation="reject_input",
                blocked=True
            )
            return False, threat
        
        # Check for malicious values
        if torch.isnan(tensor).any():
            threat = SecurityThreat(
                threat_id=f"nan_values_{int(time.time())}",
                threat_type="malicious_nan",
                severity="high",
                description=f"NaN values detected in {input_name}",
                source="input_validation",
                timestamp=time.time(),
                mitigation="sanitize_input",
                blocked=False  # Can be sanitized
            )
            return False, threat
        
        if torch.isinf(tensor).any():
            threat = SecurityThreat(
                threat_id=f"inf_values_{int(time.time())}",
                threat_type="malicious_inf",
                severity="high", 
                description=f"Infinite values detected in {input_name}",
                source="input_validation",
                timestamp=time.time(),
                mitigation="sanitize_input",
                blocked=False
            )
            return False, threat
        
        # Check for extremely large values (potential overflow attacks)
        max_abs_value = torch.abs(tensor).max().item()
        if max_abs_value > 1e10:
            threat = SecurityThreat(
                threat_id=f"large_values_{int(time.time())}",
                threat_type="potential_overflow",
                severity="medium",
                description=f"Extremely large values detected: {max_abs_value}",
                source="input_validation",
                timestamp=time.time(),
                mitigation="clip_values",
                blocked=False
            )
            return False, threat
        
        return True, None
    
    def _validate_numpy_array(self, array: np.ndarray, input_name: str) -> Tuple[bool, Optional[SecurityThreat]]:
        """Validate NumPy array."""
        
        # Check array size
        if array.size > self.max_tensor_size:
            threat = SecurityThreat(
                threat_id=f"array_size_{int(time.time())}",
                threat_type="excessive_array_size",
                severity="high",
                description=f"Array {input_name} size {array.size} exceeds limit {self.max_tensor_size}",
                source="input_validation",
                timestamp=time.time(),
                mitigation="reject_input",
                blocked=True
            )
            return False, threat
        
        # Check for malicious values
        if np.isnan(array).any():
            threat = SecurityThreat(
                threat_id=f"np_nan_{int(time.time())}",
                threat_type="malicious_nan",
                severity="high",
                description=f"NaN values detected in {input_name}",
                source="input_validation",
                timestamp=time.time(),
                mitigation="sanitize_input",
                blocked=False
            )
            return False, threat
        
        if np.isinf(array).any():
            threat = SecurityThreat(
                threat_id=f"np_inf_{int(time.time())}",
                threat_type="malicious_inf", 
                severity="high",
                description=f"Infinite values detected in {input_name}",
                source="input_validation",
                timestamp=time.time(),
                mitigation="sanitize_input",
                blocked=False
            )
            return False, threat
        
        return True, None
    
    def _validate_sequence(self, sequence: Union[List, Tuple], input_name: str) -> Tuple[bool, Optional[SecurityThreat]]:
        """Validate list/tuple inputs."""
        
        if len(sequence) > self.max_batch_size:
            threat = SecurityThreat(
                threat_id=f"seq_length_{int(time.time())}",
                threat_type="excessive_sequence_length",
                severity="medium",
                description=f"Sequence {input_name} length {len(sequence)} exceeds limit {self.max_batch_size}",
                source="input_validation",
                timestamp=time.time(),
                mitigation="reject_input",
                blocked=True
            )
            return False, threat
        
        return True, None
    
    def sanitize_input(self, tensor: Any, threat: SecurityThreat) -> Any:
        """Sanitize input based on threat type."""
        
        if threat.mitigation == "sanitize_input":
            if HAS_TORCH and isinstance(tensor, torch.Tensor):
                # Replace NaN/Inf with zeros
                tensor = torch.where(torch.isnan(tensor), torch.zeros_like(tensor), tensor)
                tensor = torch.where(torch.isinf(tensor), torch.zeros_like(tensor), tensor)
                return tensor
                
            elif isinstance(tensor, np.ndarray):
                tensor = np.where(np.isnan(tensor), 0, tensor)
                tensor = np.where(np.isinf(tensor), 0, tensor)
                return tensor
        
        elif threat.mitigation == "clip_values":
            if HAS_TORCH and isinstance(tensor, torch.Tensor):
                return torch.clamp(tensor, -1e6, 1e6)
            elif isinstance(tensor, np.ndarray):
                return np.clip(tensor, -1e6, 1e6)
        
        return tensor


class ModelAccessControl:
    """Access control for model operations."""
    
    def __init__(self, authorized_operations: Optional[List[str]] = None):
        self.authorized_operations = authorized_operations or [
            "forward", "predict", "predict_with_uncertainty", "evaluate"
        ]
        self.access_log = []
        self.blocked_attempts = []
        self.logger = logging.getLogger(__name__)
    
    def check_operation_permission(self, operation: str, context: Dict[str, Any] = None) -> bool:
        """Check if operation is permitted."""
        
        access_record = {
            "timestamp": time.time(),
            "operation": operation,
            "context": context or {},
            "authorized": operation in self.authorized_operations
        }
        
        self.access_log.append(access_record)
        
        if not access_record["authorized"]:
            self.blocked_attempts.append(access_record)
            self.logger.warning(f"Blocked unauthorized operation: {operation}")
        
        return access_record["authorized"]
    
    def get_access_summary(self) -> Dict[str, Any]:
        """Get access control summary."""
        
        return {
            "total_access_attempts": len(self.access_log),
            "blocked_attempts": len(self.blocked_attempts),
            "authorized_operations": self.authorized_operations,
            "recent_blocked": self.blocked_attempts[-5:] if self.blocked_attempts else []
        }


class PrivacyProtector:
    """Privacy protection for model inputs and outputs."""
    
    def __init__(self, 
                 enable_differential_privacy: bool = False,
                 epsilon: float = 1.0,
                 enable_data_anonymization: bool = True):
        
        self.enable_differential_privacy = enable_differential_privacy
        self.epsilon = epsilon
        self.enable_data_anonymization = enable_data_anonymization
        
        self.privacy_log = []
        self.logger = logging.getLogger(__name__)
    
    def add_differential_privacy_noise(self, tensor: Any, sensitivity: float = 1.0) -> Any:
        """Add differential privacy noise to tensor."""
        
        if not self.enable_differential_privacy:
            return tensor
        
        # Calculate noise scale based on epsilon and sensitivity
        noise_scale = sensitivity / self.epsilon
        
        if HAS_TORCH and isinstance(tensor, torch.Tensor):
            noise = torch.normal(0, noise_scale, tensor.shape, device=tensor.device)
            noisy_tensor = tensor + noise
            
            self.privacy_log.append({
                "timestamp": time.time(),
                "operation": "differential_privacy_noise",
                "epsilon": self.epsilon,
                "sensitivity": sensitivity,
                "noise_scale": noise_scale
            })
            
            return noisy_tensor
            
        elif isinstance(tensor, np.ndarray):
            noise = np.random.normal(0, noise_scale, tensor.shape)
            return tensor + noise
        
        return tensor
    
    def anonymize_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Anonymize sensitive data fields."""
        
        if not self.enable_data_anonymization:
            return data
        
        anonymized_data = data.copy()
        sensitive_fields = ['user_id', 'session_id', 'ip_address', 'timestamp']
        
        for field in sensitive_fields:
            if field in anonymized_data:
                # Hash sensitive fields
                original_value = str(anonymized_data[field])
                hashed_value = hashlib.sha256(original_value.encode()).hexdigest()[:8]
                anonymized_data[field] = f"anon_{hashed_value}"
        
        self.privacy_log.append({
            "timestamp": time.time(),
            "operation": "data_anonymization",
            "fields_anonymized": [f for f in sensitive_fields if f in data]
        })
        
        return anonymized_data


class SecureModelWrapper:
    """Secure wrapper for PNO models with comprehensive protection."""
    
    def __init__(self, 
                 model,
                 enable_input_validation: bool = True,
                 enable_access_control: bool = True,
                 enable_privacy_protection: bool = False,
                 security_config: Optional[Dict[str, Any]] = None):
        
        self.model = model
        self.security_config = security_config or {}
        
        # Initialize security components
        if enable_input_validation:
            self.input_validator = InputValidator(
                max_tensor_size=self.security_config.get('max_tensor_size', 100_000_000),
                max_batch_size=self.security_config.get('max_batch_size', 1000)
            )
        else:
            self.input_validator = None
        
        if enable_access_control:
            self.access_control = ModelAccessControl(
                authorized_operations=self.security_config.get('authorized_operations')
            )
        else:
            self.access_control = None
        
        if enable_privacy_protection:
            self.privacy_protector = PrivacyProtector(
                enable_differential_privacy=self.security_config.get('differential_privacy', False),
                epsilon=self.security_config.get('epsilon', 1.0)
            )
        else:
            self.privacy_protector = None
        
        self.security_incidents = []
        self.logger = logging.getLogger(__name__)
    
    def secure_predict(self, *args, **kwargs) -> Dict[str, Any]:
        """Secure prediction with comprehensive security checks."""
        
        # Check operation permission
        if self.access_control:
            if not self.access_control.check_operation_permission("predict"):
                raise PermissionError("Prediction operation not authorized")
        
        # Validate inputs
        validated_args = []
        for i, arg in enumerate(args):
            if self.input_validator:
                is_valid, threat = self.input_validator.validate_tensor_input(arg, f"arg_{i}")
                
                if not is_valid:
                    if threat.blocked:
                        self._record_security_incident(threat)
                        raise ValueError(f"Security threat detected: {threat.description}")
                    else:
                        # Sanitize input
                        arg = self.input_validator.sanitize_input(arg, threat)
                        self._record_security_incident(threat, resolved=True)
            
            validated_args.append(arg)
        
        # Perform prediction
        try:
            if hasattr(self.model, 'predict_with_uncertainty'):
                result = self.model.predict_with_uncertainty(*validated_args, **kwargs)
            else:
                result = self.model(*validated_args, **kwargs)
            
            # Apply privacy protection
            if self.privacy_protector:
                if isinstance(result, dict) and 'prediction' in result:
                    result['prediction'] = self.privacy_protector.add_differential_privacy_noise(
                        result['prediction']
                    )
        
        except Exception as e:
            self._record_security_incident(SecurityThreat(
                threat_id=f"prediction_error_{int(time.time())}",
                threat_type="prediction_failure",
                severity="medium",
                description=f"Prediction failed: {str(e)}",
                source="model_execution",
                timestamp=time.time(),
                mitigation="log_and_propagate",
                blocked=False
            ))
            raise e
        
        return result
    
    def _record_security_incident(self, threat: SecurityThreat, resolved: bool = False):
        """Record security incident."""
        
        incident = {
            "threat": threat,
            "resolved": resolved,
            "recorded_at": time.time()
        }
        
        self.security_incidents.append(incident)
        
        if threat.severity in ["critical", "high"]:
            self.logger.error(f"Security incident: {threat.description}")
        else:
            self.logger.warning(f"Security incident: {threat.description}")
    
    def get_security_report(self) -> Dict[str, Any]:
        """Generate comprehensive security report."""
        
        report = {
            "total_incidents": len(self.security_incidents),
            "resolved_incidents": sum(1 for inc in self.security_incidents if inc["resolved"]),
            "unresolved_incidents": sum(1 for inc in self.security_incidents if not inc["resolved"]),
            "threat_types": {},
            "severity_distribution": {},
            "recent_incidents": []
        }
        
        # Analyze incidents
        for incident in self.security_incidents:
            threat = incident["threat"]
            
            # Count threat types
            report["threat_types"][threat.threat_type] = \
                report["threat_types"].get(threat.threat_type, 0) + 1
            
            # Count severity levels
            report["severity_distribution"][threat.severity] = \
                report["severity_distribution"].get(threat.severity, 0) + 1
        
        # Recent incidents (last 10)
        recent_incidents = self.security_incidents[-10:] if len(self.security_incidents) > 10 else self.security_incidents
        report["recent_incidents"] = [
            {
                "threat_type": inc["threat"].threat_type,
                "severity": inc["threat"].severity,
                "description": inc["threat"].description,
                "resolved": inc["resolved"],
                "timestamp": inc["recorded_at"]
            }
            for inc in recent_incidents
        ]
        
        # Component-specific reports
        if self.input_validator:
            report["input_validation"] = {
                "threats_detected": len(self.input_validator.threat_log),
                "validations_performed": len(self.input_validator.validation_history)
            }
        
        if self.access_control:
            report["access_control"] = self.access_control.get_access_summary()
        
        if self.privacy_protector:
            report["privacy_protection"] = {
                "operations_performed": len(self.privacy_protector.privacy_log),
                "differential_privacy_enabled": self.privacy_protector.enable_differential_privacy,
                "epsilon": self.privacy_protector.epsilon
            }
        
        return report


def create_secure_pno_system(model, security_level: str = "standard") -> SecureModelWrapper:
    """Factory function to create secure PNO system with different security levels."""
    
    security_configs = {
        "minimal": {
            "enable_input_validation": True,
            "enable_access_control": False,
            "enable_privacy_protection": False
        },
        "standard": {
            "enable_input_validation": True,
            "enable_access_control": True,
            "enable_privacy_protection": False,
            "max_tensor_size": 50_000_000,
            "max_batch_size": 500
        },
        "high": {
            "enable_input_validation": True,
            "enable_access_control": True,
            "enable_privacy_protection": True,
            "max_tensor_size": 10_000_000,
            "max_batch_size": 100,
            "differential_privacy": True,
            "epsilon": 1.0
        }
    }
    
    config = security_configs.get(security_level, security_configs["standard"])
    
    return SecureModelWrapper(
        model,
        enable_input_validation=config["enable_input_validation"],
        enable_access_control=config["enable_access_control"], 
        enable_privacy_protection=config["enable_privacy_protection"],
        security_config=config
    )


if __name__ == "__main__":
    print("Security Validation System for PNO Models")
    print("=" * 45)
    
    # Example usage with mock model
    class MockSecureModel:
        def predict_with_uncertainty(self, x):
            return {"prediction": x * 0.95, "uncertainty": 0.05}
    
    # Create secure system
    mock_model = MockSecureModel()
    secure_system = create_secure_pno_system(mock_model, security_level="standard")
    
    # Test with valid input
    try:
        if HAS_TORCH:
            valid_input = torch.randn(2, 3, 16, 16)
        else:
            valid_input = np.random.randn(2, 3, 16, 16)
        
        result = secure_system.secure_predict(valid_input)
        print("✓ Valid input processed successfully")
        
    except Exception as e:
        print(f"✗ Valid input failed: {e}")
    
    # Test with malicious input (NaN values)
    try:
        if HAS_TORCH:
            malicious_input = torch.full((2, 3, 16, 16), float('nan'))
        else:
            malicious_input = np.full((2, 3, 16, 16), np.nan)
        
        result = secure_system.secure_predict(malicious_input)
        print("✓ Malicious input sanitized and processed")
        
    except Exception as e:
        print(f"✗ Malicious input blocked: {e}")
    
    # Get security report
    security_report = secure_system.get_security_report()
    print(f"\nSecurity Report:")
    print(f"- Total incidents: {security_report['total_incidents']}")
    print(f"- Resolved incidents: {security_report['resolved_incidents']}")
    print(f"- Threat types: {security_report['threat_types']}")
    print(f"- Severity distribution: {security_report['severity_distribution']}")
    
    print("\nSecurity validation system initialized successfully!")