# SECURITY NOTICE: This file has been automatically hardened for security
# - All user inputs should be validated and sanitized
# - Subprocess calls use secure alternatives
# - SQL queries use parameterized statements
# - No hardcoded secrets or credentials


"""Advanced Security Framework for PNO Physics Bench"""

import hashlib
import secrets
import logging
import time
from typing import Dict, Optional, Any
from datetime import datetime, timedelta
import threading

logger = logging.getLogger(__name__)

class SecurityValidator:
    """Comprehensive security validation and protection"""
    
    def __init__(self):
        self.failed_attempts = {}
        self.rate_limits = {}
        self.security_log = []
        self.max_attempts = 5
        self.lockout_duration = 300  # 5 minutes
        self.rate_limit_window = 60  # 1 minute
        self.max_requests_per_window = 100
        self._lock = threading.Lock()
    
    def validate_access_attempt(self, identifier: str, action: str) -> bool:
        """Validate access attempt with rate limiting and lockout"""
        with self._lock:
            current_time = time.time()
            
            # Check if identifier is locked out
            if identifier in self.failed_attempts:
                attempts, last_attempt = self.failed_attempts[identifier]
                if attempts >= self.max_attempts:
                    if current_time - last_attempt < self.lockout_duration:
                        self._log_security_event("ACCESS_DENIED_LOCKOUT", identifier, action)
                        return False
                    else:
                        # Reset after lockout period
                        del self.failed_attempts[identifier]
            
            # Check rate limiting
            if identifier in self.rate_limits:
                requests, window_start = self.rate_limits[identifier]
                if current_time - window_start < self.rate_limit_window:
                    if requests >= self.max_requests_per_window:
                        self._log_security_event("RATE_LIMIT_EXCEEDED", identifier, action)
                        return False
                    else:
                        self.rate_limits[identifier] = (requests + 1, window_start)
                else:
                    # New window
                    self.rate_limits[identifier] = (1, current_time)
            else:
                self.rate_limits[identifier] = (1, current_time)
            
            return True
    
    def record_failed_attempt(self, identifier: str, action: str):
        """Record failed access attempt"""
        with self._lock:
            current_time = time.time()
            if identifier in self.failed_attempts:
                attempts, _ = self.failed_attempts[identifier]
                self.failed_attempts[identifier] = (attempts + 1, current_time)
            else:
                self.failed_attempts[identifier] = (1, current_time)
            
            self._log_security_event("ACCESS_ATTEMPT_FAILED", identifier, action)
    
    def _log_security_event(self, event_type: str, identifier: str, action: str):
        """Log security events for monitoring"""
        event = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "identifier": hashlib.sha256(identifier.encode()).hexdigest()[:16],  # Anonymized
            "action": action,
            "severity": "HIGH" if "DENIED" in event_type else "MEDIUM"
        }
        self.security_log.append(event)
        logger.warning(f"Security event: {event}")
    
    def sanitize_model_parameters(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize model parameters to prevent injection"""
        sanitized = {}
        allowed_types = (int, float, bool, str, list, tuple)
        
        for key, value in params.items():
            # Validate key
            if not isinstance(key, str) or len(key) > 100:
                raise ValueError(f"Invalid parameter key: {key}")
            
            # Validate value type
            if not isinstance(value, allowed_types):
                raise ValueError(f"Invalid parameter type for {key}: {type(value)}")
            
            # Sanitize string values
            if isinstance(value, str):
                if len(value) > 1000:
                    raise ValueError(f"Parameter {key} string too long")
                # Remove potentially dangerous characters
                value = ''.join(c for c in value if c.isalnum() or c in '._-')
            
            sanitized[key] = value
        
        return sanitized
    
    def generate_secure_token(self, length: int = 32) -> str:
        """Generate cryptographically secure token"""
        return secrets.token_urlsafe(length)
    
    def hash_sensitive_data(self, data: str) -> str:
        """Hash sensitive data with salt"""
        salt = secrets.token_bytes(32)
        hash_obj = hashlib.pbkdf2_hmac('sha256', data.encode(), salt, 100000)
        return f"{salt.hex()}:{hash_obj.hex()}"
    
    def verify_hash(self, data: str, hashed: str) -> bool:
        """Verify hashed data"""
        try:
            salt_hex, hash_hex = hashed.split(':')
            salt = bytes.fromhex(salt_hex)
            expected_hash = bytes.fromhex(hash_hex)
            actual_hash = hashlib.pbkdf2_hmac('sha256', data.encode(), salt, 100000)
            return secrets.compare_digest(expected_hash, actual_hash)
        except Exception:
            return False

class SecureModelWrapper:
    """Secure wrapper for ML models"""
    
    def __init__(self, model, validator: SecurityValidator):
        self.model = model
        self.validator = validator
        self.session_token = validator.generate_secure_token()
    
    def predict(self, input_data, session_id: str = None):
        """Secure prediction with validation"""
        if not self.validator.validate_access_attempt(session_id or "anonymous", "predict"):
            raise PermissionError("Access denied - rate limit or lockout")
        
        try:
            # Validate input data
            if hasattr(input_data, 'shape') and len(input_data.shape) > 6:
                raise ValueError("Input tensor has too many dimensions")
            
            result = self.model.predict(input_data)
            return result
        
        except Exception as e:
            self.validator.record_failed_attempt(session_id or "anonymous", "predict")
            raise
    
    def update_parameters(self, params: Dict[str, Any], session_id: str = None):
        """Secure parameter update"""
        if not self.validator.validate_access_attempt(session_id or "anonymous", "update_params"):
            raise PermissionError("Access denied")
        
        sanitized_params = self.validator.sanitize_model_parameters(params)
        self.model.update_parameters(sanitized_params)

# Global security validator instance
security_validator = SecurityValidator()
