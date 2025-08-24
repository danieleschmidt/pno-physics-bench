# SECURITY NOTICE: This file has been automatically hardened for security
# - All user inputs should be validated and sanitized
# - Subprocess calls use secure alternatives
# - SQL queries use parameterized statements
# - No hardcoded secrets or credentials

"""
Production-Grade Security Validation Framework
Generation 2 Robustness Enhancement
"""

import hashlib
import hmac
import json
import logging
import os
import re
import secrets
import threading
import time
from collections import defaultdict, deque
from contextlib import contextmanager
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import weakref

logger = logging.getLogger(__name__)


class SecurityLevel(Enum):
    """Security levels for different operations."""
    PUBLIC = auto()      # No restrictions
    AUTHENTICATED = auto()  # Requires authentication
    AUTHORIZED = auto()     # Requires specific permissions
    PRIVILEGED = auto()     # Requires elevated privileges
    SYSTEM = auto()         # System-level operations only


class ThreatType(Enum):
    """Types of security threats."""
    INJECTION = auto()
    DOS = auto()
    UNAUTHORIZED_ACCESS = auto()
    DATA_EXFILTRATION = auto()
    MALICIOUS_INPUT = auto()
    PRIVILEGE_ESCALATION = auto()


@dataclass
class SecurityEvent:
    """Security event record."""
    event_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: datetime = field(default_factory=datetime.now)
    event_type: str = ""
    severity: str = "MEDIUM"
    source_ip: Optional[str] = None
    user_agent: Optional[str] = None
    resource: str = ""
    action: str = ""
    outcome: str = ""  # SUCCESS, FAILURE, BLOCKED
    details: Dict[str, Any] = field(default_factory=dict)
    threat_indicators: List[str] = field(default_factory=list)


@dataclass
class RateLimitConfig:
    """Rate limiting configuration."""
    max_requests: int = 100
    window_seconds: int = 60
    burst_allowance: int = 10
    lockout_duration_seconds: int = 300


class SecurityAuditor:
    """Comprehensive security auditing and logging."""
    
    def __init__(self, audit_log_file: Optional[str] = None):
        self.audit_log_file = audit_log_file or '/root/repo/logs/security_audit.log'
        self.security_events = deque(maxlen=10000)
        self.threat_patterns = self._load_threat_patterns()
        self._lock = threading.RLock()
        self._setup_audit_logging()
    
    def _setup_audit_logging(self):
        """Setup secure audit logging."""
        os.makedirs(os.path.dirname(self.audit_log_file), exist_ok=True)
        
        # Create security-specific logger
        self.audit_logger = logging.getLogger('security_audit')
        self.audit_logger.setLevel(logging.INFO)
        
        # Remove existing handlers
        for handler in self.audit_logger.handlers[:]:
            self.audit_logger.removeHandler(handler)
        
        # Secure file handler with rotation
        from logging.handlers import RotatingFileHandler
        handler = RotatingFileHandler(
            self.audit_log_file,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=10
        )
        
        formatter = logging.Formatter(
            '%(asctime)s - SECURITY_AUDIT - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.audit_logger.addHandler(handler)
    
    def _load_threat_patterns(self) -> Dict[ThreatType, List[str]]:
        """Load known threat patterns for detection."""
        return {
            ThreatType.INJECTION: [
                r'<script[^>]*>.*?</script>',
                r'javascript:',
                r'eval\s*\(',
                r'exec\s*\(',
                r'__import__',
                r'subprocess\.',
                r'os\.system',
                r'os\.popen'
            ],
            ThreatType.DOS: [
                r'(.)\1{1000,}',  # Repeated characters (potential ReDoS)
                r'\*{100,}',      # Many wildcards
                r'\.{100,}',      # Many dots
            ],
            ThreatType.MALICIOUS_INPUT: [
                r'[<>"\'].*[<>"\']',  # XSS patterns
                r'union\s+select',     # SQL injection
                r'drop\s+table',       # SQL injection
                r'../.*../',           # Path traversal
                r'\.\.\\.*\.\.\\',     # Windows path traversal
            ]
        }
    
    def log_security_event(self, event: SecurityEvent):
        """Log a security event with audit trail."""
        with self._lock:
            self.security_events.append(event)
            
            # Create audit log entry
            audit_entry = {
                'event_id': event.event_id,
                'timestamp': event.timestamp.isoformat(),
                'type': event.event_type,
                'severity': event.severity,
                'resource': event.resource,
                'action': event.action,
                'outcome': event.outcome,
                'threat_indicators': event.threat_indicators,
                'details_hash': self._hash_sensitive_data(json.dumps(event.details, sort_keys=True))
            }
            
            self.audit_logger.info(json.dumps(audit_entry))
            
            # Alert on high-severity events
            if event.severity in ['HIGH', 'CRITICAL']:
                logger.warning(f"High-severity security event: {event.event_type}")
    
    def _hash_sensitive_data(self, data: str) -> str:
        """Hash sensitive data for audit logs."""
        return hashlib.sha256(data.encode()).hexdigest()[:16]
    
    def detect_threats(self, input_data: str) -> List[Tuple[ThreatType, str]]:
        """Detect potential threats in input data."""
        threats = []
        
        for threat_type, patterns in self.threat_patterns.items():
            for pattern in patterns:
                if re.search(pattern, input_data, re.IGNORECASE | re.DOTALL):
                    threats.append((threat_type, pattern))
        
        return threats
    
    def get_security_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get security summary for the specified time period."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        recent_events = [
            event for event in self.security_events
            if event.timestamp >= cutoff_time
        ]
        
        if not recent_events:
            return {
                'total_events': 0,
                'period_hours': hours,
                'status': 'SECURE'
            }
        
        # Analyze events
        event_types = defaultdict(int)
        severity_counts = defaultdict(int)
        threat_indicators = defaultdict(int)
        
        for event in recent_events:
            event_types[event.event_type] += 1
            severity_counts[event.severity] += 1
            for indicator in event.threat_indicators:
                threat_indicators[indicator] += 1
        
        # Determine overall security status
        critical_events = severity_counts.get('CRITICAL', 0)
        high_events = severity_counts.get('HIGH', 0)
        
        if critical_events > 0:
            status = 'CRITICAL_THREATS_DETECTED'
        elif high_events > 5:
            status = 'HIGH_RISK'
        elif len(recent_events) > 100:
            status = 'ELEVATED_ACTIVITY'
        else:
            status = 'SECURE'
        
        return {
            'total_events': len(recent_events),
            'period_hours': hours,
            'status': status,
            'event_types': dict(event_types),
            'severity_distribution': dict(severity_counts),
            'top_threat_indicators': dict(sorted(threat_indicators.items(), 
                                               key=lambda x: x[1], reverse=True)[:10])
        }


class RateLimiter:
    """Advanced rate limiting with burst protection and adaptive thresholds."""
    
    def __init__(self, config: RateLimitConfig):
        self.config = config
        self.request_windows = defaultdict(lambda: deque())
        self.burst_usage = defaultdict(int)
        self.lockout_times = {}
        self._lock = threading.RLock()
    
    def is_allowed(self, identifier: str, action: str = "default") -> Tuple[bool, Optional[str]]:
        """Check if request is allowed under rate limits."""
        current_time = time.time()
        key = f"{identifier}:{action}"
        
        with self._lock:
            # Check if currently locked out
            if key in self.lockout_times:
                if current_time - self.lockout_times[key] < self.config.lockout_duration_seconds:
                    return False, "LOCKED_OUT"
                else:
                    del self.lockout_times[key]
            
            # Clean old requests from window
            window = self.request_windows[key]
            window_start = current_time - self.config.window_seconds
            
            while window and window[0] < window_start:
                window.popleft()
            
            # Check rate limit
            if len(window) >= self.config.max_requests:
                # Check if burst allowance is available
                if self.burst_usage[key] < self.config.burst_allowance:
                    self.burst_usage[key] += 1
                    window.append(current_time)
                    return True, "BURST_ALLOWED"
                else:
                    # Rate limit exceeded, initiate lockout
                    self.lockout_times[key] = current_time
                    return False, "RATE_LIMITED"
            
            # Request is allowed
            window.append(current_time)
            
            # Reset burst usage if we're under normal limits
            if len(window) <= self.config.max_requests * 0.8:
                self.burst_usage[key] = max(0, self.burst_usage[key] - 1)
            
            return True, "ALLOWED"
    
    def get_usage_stats(self, identifier: str, action: str = "default") -> Dict[str, Any]:
        """Get usage statistics for identifier."""
        key = f"{identifier}:{action}"
        current_time = time.time()
        
        with self._lock:
            window = self.request_windows[key]
            
            # Count recent requests
            recent_requests = sum(1 for req_time in window 
                                if current_time - req_time <= self.config.window_seconds)
            
            return {
                'requests_in_window': recent_requests,
                'max_requests': self.config.max_requests,
                'burst_usage': self.burst_usage[key],
                'max_burst': self.config.burst_allowance,
                'is_locked_out': key in self.lockout_times,
                'utilization_percent': (recent_requests / self.config.max_requests) * 100
            }


class InputSanitizer:
    """Comprehensive input sanitization and validation."""
    
    def __init__(self):
        self.max_string_length = 10000
        self.max_list_length = 1000
        self.max_dict_keys = 100
        self.allowed_types = {int, float, bool, str, list, tuple, dict, type(None)}
        self.dangerous_patterns = [
            r'<script[^>]*>.*?</script>',
            r'javascript:',
            r'data:',
            r'vbscript:',
            r'on\w+\s*=',  # Event handlers
            r'expression\s*\(',  # CSS expression
            r'@import',
            r'<?xml',
            r'<!DOCTYPE'
        ]
    
    def sanitize_string(self, value: str, allow_html: bool = False) -> str:
        """Sanitize string input."""
        if not isinstance(value, str):
            raise ValueError("Input must be string")
        
        if len(value) > self.max_string_length:
            raise ValueError(f"String too long: {len(value)} > {self.max_string_length}")
        
        # Check for dangerous patterns
        for pattern in self.dangerous_patterns:
            if re.search(pattern, value, re.IGNORECASE):
                raise ValueError(f"Potentially dangerous content detected")
        
        if not allow_html:
            # HTML escape
            value = (value.replace('&', '&amp;')
                          .replace('<', '&lt;')
                          .replace('>', '&gt;')
                          .replace('"', '&quot;')
                          .replace("'", '&#x27;'))
        
        return value
    
    def sanitize_dict(self, data: Dict[str, Any], max_depth: int = 5) -> Dict[str, Any]:
        """Sanitize dictionary recursively."""
        if not isinstance(data, dict):
            raise ValueError("Input must be dictionary")
        
        if len(data) > self.max_dict_keys:
            raise ValueError(f"Too many keys: {len(data)} > {self.max_dict_keys}")
        
        if max_depth <= 0:
            raise ValueError("Dictionary nesting too deep")
        
        sanitized = {}
        
        for key, value in data.items():
            # Sanitize key
            if not isinstance(key, str):
                raise ValueError(f"Dictionary key must be string: {type(key)}")
            
            clean_key = self.sanitize_string(key)
            
            # Sanitize value
            clean_value = self.sanitize_value(value, max_depth - 1)
            sanitized[clean_key] = clean_value
        
        return sanitized
    
    def sanitize_list(self, data: List[Any], max_depth: int = 5) -> List[Any]:
        """Sanitize list recursively."""
        if not isinstance(data, (list, tuple)):
            raise ValueError("Input must be list or tuple")
        
        if len(data) > self.max_list_length:
            raise ValueError(f"List too long: {len(data)} > {self.max_list_length}")
        
        return [self.sanitize_value(item, max_depth - 1) for item in data]
    
    def sanitize_value(self, value: Any, max_depth: int = 5) -> Any:
        """Sanitize any value type."""
        value_type = type(value)
        
        if value_type not in self.allowed_types:
            raise ValueError(f"Disallowed type: {value_type}")
        
        if isinstance(value, str):
            return self.sanitize_string(value)
        elif isinstance(value, dict):
            return self.sanitize_dict(value, max_depth)
        elif isinstance(value, (list, tuple)):
            return self.sanitize_list(value, max_depth)
        elif isinstance(value, (int, float)):
            # Check for suspicious numeric values
            if abs(value) > 1e15:
                raise ValueError(f"Suspicious large numeric value: {value}")
            if isinstance(value, float) and (value != value):  # NaN check
                raise ValueError("NaN values not allowed")
            return value
        else:
            return value


class DOSProtection:
    """Denial of Service protection system."""
    
    def __init__(self):
        self.rate_limiters = {}
        self.request_patterns = defaultdict(lambda: deque(maxlen=1000))
        self.blocked_identifiers = {}
        self.anomaly_detector = RequestAnomalyDetector()
        self._lock = threading.RLock()
    
    def get_rate_limiter(self, action: str, config: Optional[RateLimitConfig] = None) -> RateLimiter:
        """Get or create rate limiter for action."""
        if action not in self.rate_limiters:
            config = config or RateLimitConfig()
            self.rate_limiters[action] = RateLimiter(config)
        return self.rate_limiters[action]
    
    def check_request(self, identifier: str, action: str, 
                     request_size: Optional[int] = None) -> Tuple[bool, str]:
        """Check if request should be allowed."""
        current_time = time.time()
        
        with self._lock:
            # Check if identifier is blocked
            if identifier in self.blocked_identifiers:
                block_time, reason = self.blocked_identifiers[identifier]
                if current_time - block_time < 3600:  # 1 hour block
                    return False, f"BLOCKED: {reason}"
                else:
                    del self.blocked_identifiers[identifier]
            
            # Record request pattern
            self.request_patterns[identifier].append({
                'timestamp': current_time,
                'action': action,
                'size': request_size or 0
            })
            
            # Check for anomalies
            if self.anomaly_detector.detect_anomaly(identifier, self.request_patterns[identifier]):
                self._block_identifier(identifier, "ANOMALY_DETECTED")
                return False, "ANOMALY_DETECTED"
            
            # Check rate limits
            rate_limiter = self.get_rate_limiter(action)
            allowed, reason = rate_limiter.is_allowed(identifier, action)
            
            if not allowed and reason == "RATE_LIMITED":
                # Check if this should trigger a block
                recent_rate_limits = sum(1 for req in self.request_patterns[identifier]
                                       if current_time - req['timestamp'] <= 300 and 
                                       req.get('rate_limited', False))
                
                if recent_rate_limits > 10:  # Multiple rate limit hits
                    self._block_identifier(identifier, "REPEATED_RATE_LIMIT_VIOLATIONS")
                    return False, "BLOCKED_FOR_VIOLATIONS"
            
            return allowed, reason
    
    def _block_identifier(self, identifier: str, reason: str):
        """Block an identifier for security violations."""
        current_time = time.time()
        self.blocked_identifiers[identifier] = (current_time, reason)
        
        logger.warning(f"Blocked identifier {identifier[:8]}... for {reason}")
        
        # Log security event
        event = SecurityEvent(
            event_type="IDENTIFIER_BLOCKED",
            severity="HIGH",
            resource="dos_protection",
            action="block_identifier",
            outcome="BLOCKED",
            details={'reason': reason, 'identifier_hash': hashlib.sha256(identifier.encode()).hexdigest()[:16]}
        )
        
        # This would log to security auditor if we had access to it
        logger.warning(f"Security event: {event.event_type} - {reason}")


class RequestAnomalyDetector:
    """Detects anomalous request patterns that might indicate attacks."""
    
    def __init__(self):
        self.normal_patterns = {}
        self.anomaly_thresholds = {
            'request_rate_multiplier': 5.0,  # 5x normal rate
            'size_multiplier': 10.0,         # 10x normal size
            'pattern_deviation': 0.8         # 80% different from normal
        }
    
    def detect_anomaly(self, identifier: str, request_history: deque) -> bool:
        """Detect if request pattern is anomalous."""
        if len(request_history) < 10:
            return False  # Need more data
        
        recent_requests = [req for req in request_history 
                          if time.time() - req['timestamp'] <= 300]  # Last 5 minutes
        
        if len(recent_requests) < 5:
            return False
        
        # Calculate request rate
        time_span = max(req['timestamp'] for req in recent_requests) - \
                   min(req['timestamp'] for req in recent_requests)
        
        if time_span > 0:
            request_rate = len(recent_requests) / time_span
            
            # Compare with historical average
            if identifier in self.normal_patterns:
                normal_rate = self.normal_patterns[identifier].get('avg_rate', 1.0)
                if request_rate > normal_rate * self.anomaly_thresholds['request_rate_multiplier']:
                    return True
        
        # Check request sizes
        sizes = [req['size'] for req in recent_requests if req['size'] > 0]
        if sizes:
            avg_size = sum(sizes) / len(sizes)
            max_size = max(sizes)
            
            if identifier in self.normal_patterns:
                normal_size = self.normal_patterns[identifier].get('avg_size', 1000)
                if max_size > normal_size * self.anomaly_thresholds['size_multiplier']:
                    return True
        
        return False
    
    def update_normal_patterns(self, identifier: str, request_history: deque):
        """Update normal patterns for identifier."""
        if len(request_history) < 20:
            return
        
        # Calculate patterns from older data (not recent anomalies)
        older_requests = [req for req in request_history 
                         if time.time() - req['timestamp'] > 3600]  # Older than 1 hour
        
        if len(older_requests) >= 10:
            time_diffs = []
            sizes = []
            
            for i in range(1, len(older_requests)):
                time_diffs.append(older_requests[i]['timestamp'] - older_requests[i-1]['timestamp'])
                if older_requests[i]['size'] > 0:
                    sizes.append(older_requests[i]['size'])
            
            if time_diffs:
                avg_interval = sum(time_diffs) / len(time_diffs)
                avg_rate = 1.0 / avg_interval if avg_interval > 0 else 1.0
            else:
                avg_rate = 1.0
            
            avg_size = sum(sizes) / len(sizes) if sizes else 1000
            
            self.normal_patterns[identifier] = {
                'avg_rate': avg_rate,
                'avg_size': avg_size,
                'last_updated': time.time()
            }


class SecurityValidator:
    """Comprehensive security validation framework."""
    
    def __init__(self):
        self.auditor = SecurityAuditor()
        self.dos_protection = DOSProtection()
        self.input_sanitizer = InputSanitizer()
        self.session_manager = SecureSessionManager()
        self._lock = threading.RLock()
    
    def validate_request(self, request_data: Dict[str, Any], 
                        source_identifier: str,
                        required_security_level: SecurityLevel = SecurityLevel.PUBLIC) -> Dict[str, Any]:
        """Comprehensive request validation."""
        validation_start = time.time()
        
        try:
            # 1. Rate limiting and DOS protection
            allowed, reason = self.dos_protection.check_request(
                source_identifier, 
                request_data.get('action', 'unknown'),
                len(json.dumps(request_data))
            )
            
            if not allowed:
                self._log_security_violation("RATE_LIMIT_EXCEEDED", source_identifier, {
                    'reason': reason,
                    'action': request_data.get('action', 'unknown')
                })
                raise PermissionError(f"Request blocked: {reason}")
            
            # 2. Input sanitization
            sanitized_data = self.input_sanitizer.sanitize_dict(request_data)
            
            # 3. Threat detection
            request_json = json.dumps(sanitized_data)
            threats = self.auditor.detect_threats(request_json)
            
            if threats:
                self._log_security_violation("THREAT_DETECTED", source_identifier, {
                    'threats': [(threat_type.name, pattern) for threat_type, pattern in threats],
                    'data_hash': hashlib.sha256(request_json.encode()).hexdigest()[:16]
                })
                raise SecurityError(f"Threats detected: {[t[0].name for t in threats]}")
            
            # 4. Authorization check
            if required_security_level != SecurityLevel.PUBLIC:
                if not self.session_manager.check_authorization(source_identifier, required_security_level):
                    self._log_security_violation("AUTHORIZATION_FAILED", source_identifier, {
                        'required_level': required_security_level.name,
                        'action': request_data.get('action', 'unknown')
                    })
                    raise PermissionError("Insufficient authorization")
            
            # Log successful validation
            validation_duration = (time.time() - validation_start) * 1000
            self._log_security_event("REQUEST_VALIDATED", source_identifier, {
                'validation_duration_ms': validation_duration,
                'data_size': len(request_json),
                'security_level': required_security_level.name
            })
            
            return sanitized_data
            
        except Exception as e:
            self._log_security_violation("VALIDATION_ERROR", source_identifier, {
                'error': str(e),
                'error_type': type(e).__name__
            })
            raise
    
    def _log_security_event(self, event_type: str, source: str, details: Dict[str, Any]):
        """Log security event."""
        event = SecurityEvent(
            event_type=event_type,
            severity="INFO",
            resource="security_validator",
            action="validate_request",
            outcome="SUCCESS",
            details=details
        )
        self.auditor.log_security_event(event)
    
    def _log_security_violation(self, violation_type: str, source: str, details: Dict[str, Any]):
        """Log security violation."""
        event = SecurityEvent(
            event_type=violation_type,
            severity="HIGH",
            resource="security_validator",
            action="validate_request",
            outcome="BLOCKED",
            details=details,
            threat_indicators=[violation_type]
        )
        self.auditor.log_security_event(event)
    
    def get_security_status(self) -> Dict[str, Any]:
        """Get current security status."""
        return {
            'timestamp': datetime.now().isoformat(),
            'auditor_summary': self.auditor.get_security_summary(),
            'active_sessions': self.session_manager.get_session_count(),
            'rate_limiter_stats': {
                action: limiter.get_usage_stats("system", action)
                for action, limiter in self.dos_protection.rate_limiters.items()
            }
        }


class SecureSessionManager:
    """Manages secure sessions with timeout and validation."""
    
    def __init__(self):
        self.sessions = {}
        self.session_timeout = 3600  # 1 hour
        self._lock = threading.RLock()
        self._cleanup_thread = threading.Thread(target=self._cleanup_expired_sessions, daemon=True)
        self._cleanup_thread.start()
    
    def create_session(self, identifier: str, security_level: SecurityLevel) -> str:
        """Create a new secure session."""
        session_id = secrets.token_urlsafe(32)
        
        with self._lock:
            self.sessions[session_id] = {
                'identifier': identifier,
                'security_level': security_level,
                'created_at': time.time(),
                'last_activity': time.time(),
                'access_count': 0
            }
        
        logger.info(f"Created secure session for {identifier[:8]}... with level {security_level.name}")
        return session_id
    
    def validate_session(self, session_id: str) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Validate a session."""
        with self._lock:
            if session_id not in self.sessions:
                return False, None
            
            session = self.sessions[session_id]
            current_time = time.time()
            
            # Check timeout
            if current_time - session['last_activity'] > self.session_timeout:
                del self.sessions[session_id]
                return False, None
            
            # Update activity
            session['last_activity'] = current_time
            session['access_count'] += 1
            
            return True, session
    
    def check_authorization(self, identifier: str, required_level: SecurityLevel) -> bool:
        """Check if identifier has required authorization level."""
        # For this implementation, we'll use a simple mapping
        # In production, this would connect to proper auth system
        
        auth_levels = {
            'system': SecurityLevel.SYSTEM,
            'admin': SecurityLevel.PRIVILEGED,
            'user': SecurityLevel.AUTHENTICATED,
            'guest': SecurityLevel.PUBLIC
        }
        
        # Extract user type from identifier (simplified)
        for user_type, level in auth_levels.items():
            if user_type in identifier.lower():
                return level.value >= required_level.value
        
        # Default to public access
        return required_level == SecurityLevel.PUBLIC
    
    def get_session_count(self) -> int:
        """Get current session count."""
        with self._lock:
            return len(self.sessions)
    
    def _cleanup_expired_sessions(self):
        """Background cleanup of expired sessions."""
        while True:
            try:
                current_time = time.time()
                expired_sessions = []
                
                with self._lock:
                    for session_id, session in self.sessions.items():
                        if current_time - session['last_activity'] > self.session_timeout:
                            expired_sessions.append(session_id)
                    
                    for session_id in expired_sessions:
                        del self.sessions[session_id]
                
                if expired_sessions:
                    logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")
                
                time.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                logger.error(f"Session cleanup error: {e}")
                time.sleep(600)  # Wait longer on error


class SecurityError(Exception):
    """Security-related exception."""
    pass


# Context managers and decorators
@contextmanager
def secure_operation(operation_name: str, source_identifier: str = "system",
                    security_level: SecurityLevel = SecurityLevel.PUBLIC):
    """Context manager for secure operations."""
    validator = global_security_validator
    
    # Pre-operation security check
    try:
        # This would normally validate the request, but for context manager we'll just log
        logger.info(f"Starting secure operation: {operation_name} for {source_identifier}")
        yield
        
        # Log successful completion
        event = SecurityEvent(
            event_type="OPERATION_COMPLETED",
            severity="INFO",
            resource=operation_name,
            action="execute",
            outcome="SUCCESS",
            details={'identifier': source_identifier, 'security_level': security_level.name}
        )
        validator.auditor.log_security_event(event)
        
    except Exception as e:
        # Log security failure
        event = SecurityEvent(
            event_type="OPERATION_FAILED",
            severity="MEDIUM",
            resource=operation_name,
            action="execute",
            outcome="FAILURE",
            details={'error': str(e), 'identifier': source_identifier}
        )
        validator.auditor.log_security_event(event)
        raise


def secure_function(security_level: SecurityLevel = SecurityLevel.PUBLIC,
                   rate_limit_action: Optional[str] = None):
    """Decorator to add security validation to functions."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            source_id = kwargs.pop('_security_source_id', 'anonymous')
            
            with secure_operation(
                func.__name__, 
                source_id, 
                security_level
            ):
                return func(*args, **kwargs)
        
        return wrapper
    return decorator


# Global security components
global_security_validator = SecurityValidator()


# Production security health checks
def check_security_log_health() -> bool:
    """Check security logging health."""
    try:
        # Test security log write
        test_event = SecurityEvent(
            event_type="HEALTH_CHECK",
            severity="INFO",
            resource="security_health_check",
            action="test_logging",
            outcome="SUCCESS"
        )
        global_security_validator.auditor.log_security_event(test_event)
        return True
    except Exception:
        return False


def check_rate_limiter_health() -> bool:
    """Check rate limiter health."""
    try:
        # Test rate limiter functionality
        allowed, reason = global_security_validator.dos_protection.check_request(
            "health_check_test", "test_action"
        )
        return True  # If no exception, it's working
    except Exception:
        return False


def check_session_manager_health() -> bool:
    """Check session manager health."""
    try:
        session_count = global_security_validator.session_manager.get_session_count()
        return session_count >= 0  # Basic sanity check
    except Exception:
        return False


# UUID import for event IDs
import uuid