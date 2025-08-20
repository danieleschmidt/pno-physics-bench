# SECURITY NOTICE: This file has been automatically hardened for security
# - All user inputs should be validated and sanitized
# - Subprocess calls use secure alternatives
# - SQL queries use parameterized statements
# - No hardcoded secrets or credentials


"""Comprehensive audit logging and security monitoring for PNO systems."""

import json
import time
import threading
import hashlib
import uuid
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Union, Callable
from dataclasses import dataclass, asdict, field
from enum import Enum
import logging
import inspect
import functools
from pathlib import Path
import queue
import atexit

logger = logging.getLogger(__name__)


class AuditLevel(Enum):
    """Audit event severity levels."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    SECURITY = "security"


class EventType(Enum):
    """Types of audit events."""
    MODEL_TRAINING = "model_training"
    MODEL_INFERENCE = "model_inference"
    DATA_ACCESS = "data_access"
    PARAMETER_CHANGE = "parameter_change"
    SECURITY_EVENT = "security_event"
    SYSTEM_EVENT = "system_event"
    USER_ACTION = "user_action"
    ERROR_EVENT = "error_event"


@dataclass
class AuditEvent:
    """Audit event data structure."""
    event_id: str
    timestamp: str
    event_type: EventType
    level: AuditLevel
    message: str
    component: str
    
    # Event details
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    
    # Context information
    function_name: Optional[str] = None
    module_name: Optional[str] = None
    line_number: Optional[int] = None
    
    # Data and parameters
    input_data_hash: Optional[str] = None
    output_data_hash: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None
    
    # Performance metrics
    execution_time: Optional[float] = None
    memory_usage: Optional[float] = None
    gpu_usage: Optional[float] = None
    
    # Security context
    source_ip: Optional[str] = None
    user_agent: Optional[str] = None
    security_level: Optional[str] = None
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert audit event to dictionary."""
        event_dict = asdict(self)
        # Convert enums to strings
        event_dict['event_type'] = self.event_type.value
        event_dict['level'] = self.level.value
        return event_dict
    
    def to_json(self) -> str:
        """Convert audit event to JSON string."""
        return json.dumps(self.to_dict(), default=str)


class AuditLogger:
    """Comprehensive audit logging system."""
    
    def __init__(
        self,
        name: str = "pno_audit",
        log_file: Optional[str] = None,
        max_memory_events: int = 10000,
        enable_encryption: bool = False,
        encryption_key: Optional[bytes] = None,
        enable_async: bool = True
    ):
        """Initialize audit logger.
        
        Args:
            name: Logger name
            log_file: Path to audit log file
            max_memory_events: Maximum events to keep in memory
            enable_encryption: Whether to encrypt log entries
            encryption_key: Encryption key (if None, will generate)
            enable_async: Whether to use async logging
        """
        self.name = name
        self.log_file = Path(log_file) if log_file else None
        self.max_memory_events = max_memory_events
        self.enable_encryption = enable_encryption
        self.enable_async = enable_async
        
        # Event storage
        self.events: List[AuditEvent] = []
        self.event_lock = threading.RLock()
        
        # Async processing
        if self.enable_async:
            self.event_queue = queue.Queue()
            self.worker_thread = threading.Thread(target=self._process_events, daemon=True)
            self.worker_thread.start()
            atexit.register(self._cleanup)
        
        # Encryption setup
        if self.enable_encryption:
            self.encryption_key = encryption_key or self._generate_key()
        
        # Session tracking
        self.session_id = str(uuid.uuid4())
        self.start_time = time.time()
        
        # Performance tracking
        self.performance_stats = {
            'events_logged': 0,
            'events_per_second': 0,
            'avg_event_size': 0,
            'total_log_size': 0
        }
        
        logger.info(f"Initialized audit logger: {name}")
    
    def _generate_key(self) -> bytes:
        """Generate encryption key."""
        return hashlib.sha256(f"{self.name}_{time.time()}".encode()).digest()
    
    def _process_events(self):
        """Background thread to process audit events."""
        while True:
            try:
                event = self.event_queue.get(timeout=1.0)
                if event is None:  # Shutdown signal
                    break
                self._write_event(event)
                self.event_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error processing audit event: {e}")
    
    def _write_event(self, event: AuditEvent):
        """Write event to storage."""
        with self.event_lock:
            # Add to memory
            self.events.append(event)
            if len(self.events) > self.max_memory_events:
                self.events.pop(0)  # Remove oldest
            
            # Write to file
            if self.log_file:
                try:
                    self.log_file.parent.mkdir(parents=True, exist_ok=True)
                    with open(self.log_file, 'a', encoding='utf-8') as f:
                        log_line = event.to_json()
                        if self.enable_encryption:
                            log_line = self._encrypt(log_line)
                        f.write(log_line + '\n')
                except Exception as e:
                    logger.error(f"Failed to write audit log: {e}")
            
            # Update performance stats
            self.performance_stats['events_logged'] += 1
            self.performance_stats['events_per_second'] = (
                self.performance_stats['events_logged'] / 
                (time.time() - self.start_time)
            )
    
    def _encrypt(self, data: str) -> str:
        """Encrypt log data (placeholder implementation)."""
        # In a real implementation, use proper encryption like AES
        return hashlib.sha256(data.encode() + self.encryption_key).hexdigest()
    
    def _get_caller_info(self) -> Dict[str, Any]:
        """Get information about the calling function."""
        frame = inspect.currentframe()
        try:
            # Go up the stack to find the actual caller
            for _ in range(3):  # Skip this method and log method
                frame = frame.f_back
                if frame is None:
                    break
            
            if frame:
                return {
                    'function_name': frame.f_code.co_name,
                    'module_name': frame.f_globals.get('__name__', 'unknown'),
                    'line_number': frame.f_lineno
                }
        finally:
            del frame
        
        return {}
    
    def log_event(
        self,
        event_type: EventType,
        level: AuditLevel,
        message: str,
        component: str,
        **kwargs
    ) -> str:
        """Log an audit event.
        
        Args:
            event_type: Type of event
            level: Severity level
            message: Event message
            component: Component name
            **kwargs: Additional event data
            
        Returns:
            Event ID
        """
        # Generate event ID
        event_id = str(uuid.uuid4())
        
        # Get caller information
        caller_info = self._get_caller_info()
        
        # Create event
        event = AuditEvent(
            event_id=event_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            event_type=event_type,
            level=level,
            message=message,
            component=component,
            session_id=self.session_id,
            **caller_info,
            **kwargs
        )
        
        # Process event
        if self.enable_async:
            self.event_queue.put(event)
        else:
            self._write_event(event)
        
        return event_id
    
    def log_model_training(
        self,
        model_name: str,
        parameters: Dict[str, Any],
        message: str = "Model training event",
        **kwargs
    ) -> str:
        """Log model training event."""
        return self.log_event(
            event_type=EventType.MODEL_TRAINING,
            level=AuditLevel.INFO,
            message=message,
            component=f"training.{model_name}",
            parameters=parameters,
            **kwargs
        )
    
    def log_model_inference(
        self,
        model_name: str,
        input_shape: Optional[str] = None,
        output_shape: Optional[str] = None,
        message: str = "Model inference event",
        **kwargs
    ) -> str:
        """Log model inference event."""
        metadata = {}
        if input_shape:
            metadata['input_shape'] = input_shape
        if output_shape:
            metadata['output_shape'] = output_shape
        
        return self.log_event(
            event_type=EventType.MODEL_INFERENCE,
            level=AuditLevel.INFO,
            message=message,
            component=f"inference.{model_name}",
            metadata=metadata,
            **kwargs
        )
    
    def log_data_access(
        self,
        data_path: str,
        access_type: str,
        user_id: Optional[str] = None,
        message: str = "Data access event",
        **kwargs
    ) -> str:
        """Log data access event."""
        return self.log_event(
            event_type=EventType.DATA_ACCESS,
            level=AuditLevel.INFO,
            message=message,
            component="data_access",
            user_id=user_id,
            metadata={'data_path': data_path, 'access_type': access_type},
            **kwargs
        )
    
    def log_security_event(
        self,
        security_event: str,
        severity: AuditLevel = AuditLevel.SECURITY,
        source_ip: Optional[str] = None,
        user_id: Optional[str] = None,
        message: str = "Security event",
        **kwargs
    ) -> str:
        """Log security event."""
        return self.log_event(
            event_type=EventType.SECURITY_EVENT,
            level=severity,
            message=message,
            component="security",
            user_id=user_id,
            source_ip=source_ip,
            metadata={'security_event': security_event},
            **kwargs
        )
    
    def log_error(
        self,
        error: Exception,
        component: str,
        message: str = "Error occurred",
        **kwargs
    ) -> str:
        """Log error event."""
        return self.log_event(
            event_type=EventType.ERROR_EVENT,
            level=AuditLevel.ERROR,
            message=message,
            component=component,
            metadata={
                'error_type': type(error).__name__,
                'error_message': str(error),
                'error_traceback': str(error.__traceback__) if hasattr(error, '__traceback__') else None
            },
            **kwargs
        )
    
    def get_events(
        self,
        event_type: Optional[EventType] = None,
        level: Optional[AuditLevel] = None,
        component: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> List[AuditEvent]:
        """Get audit events with filtering.
        
        Args:
            event_type: Filter by event type
            level: Filter by level
            component: Filter by component
            start_time: Filter by start time
            end_time: Filter by end time
            limit: Limit number of results
            
        Returns:
            List of filtered audit events
        """
        with self.event_lock:
            filtered_events = []
            
            for event in self.events:
                # Apply filters
                if event_type and event.event_type != event_type:
                    continue
                if level and event.level != level:
                    continue
                if component and event.component != component:
                    continue
                
                # Time filtering
                if start_time or end_time:
                    event_time = datetime.fromisoformat(event.timestamp.replace('Z', '+00:00'))
                    if start_time and event_time < start_time:
                        continue
                    if end_time and event_time > end_time:
                        continue
                
                filtered_events.append(event)
                
                # Apply limit
                if limit and len(filtered_events) >= limit:
                    break
            
            return filtered_events
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get audit logger performance statistics."""
        current_time = time.time()
        uptime = current_time - self.start_time
        
        stats = self.performance_stats.copy()
        stats.update({
            'uptime_seconds': uptime,
            'memory_events': len(self.events),
            'session_id': self.session_id
        })
        
        return stats
    
    def export_events(self, file_path: str, format: str = 'json') -> bool:
        """Export audit events to file.
        
        Args:
            file_path: Output file path
            format: Export format ('json', 'csv')
            
        Returns:
            Success status
        """
        try:
            output_path = Path(file_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with self.event_lock:
                if format.lower() == 'json':
                    with open(output_path, 'w', encoding='utf-8') as f:
                        json.dump([event.to_dict() for event in self.events], f, indent=2, default=str)
                elif format.lower() == 'csv':
                    import csv
                    with open(output_path, 'w', newline='', encoding='utf-8') as f:
                        if self.events:
                            writer = csv.DictWriter(f, fieldnames=self.events[0].to_dict().keys())
                            writer.writeheader()
                            for event in self.events:
                                writer.writerow(event.to_dict())
                else:
                    raise ValueError(f"Unsupported format: {format}")
            
            logger.info(f"Exported {len(self.events)} audit events to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export audit events: {e}")
            return False
    
    def _cleanup(self):
        """Cleanup resources."""
        if self.enable_async:
            self.event_queue.put(None)  # Shutdown signal
            self.worker_thread.join(timeout=5.0)


class SecurityMonitor:
    """Security monitoring and alerting system."""
    
    def __init__(
        self,
        audit_logger: AuditLogger,
        alert_thresholds: Optional[Dict[str, int]] = None
    ):
        """Initialize security monitor.
        
        Args:
            audit_logger: Audit logger instance
            alert_thresholds: Thresholds for security alerts
        """
        self.audit_logger = audit_logger
        self.alert_thresholds = alert_thresholds or {
            'failed_logins': 5,
            'data_access_rate': 100,  # per minute
            'error_rate': 50,         # per minute
            'security_events': 1      # immediate alert
        }
        
        # Monitoring state
        self.monitoring_window = 300  # 5 minutes
        self.alert_callbacks: List[Callable[[str, Dict], None]] = []
        
        # Start monitoring
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
    
    def add_alert_callback(self, callback: Callable[[str, Dict], None]):
        """Add callback for security alerts."""
        self.alert_callbacks.append(callback)
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while True:
            try:
                self._check_security_metrics()
                time.sleep(60)  # Check every minute
            except Exception as e:
                logger.error(f"Error in security monitoring: {e}")
    
    def _check_security_metrics(self):
        """Check security metrics and trigger alerts."""
        current_time = datetime.now(timezone.utc)
        window_start = current_time - timedelta(seconds=self.monitoring_window)
        
        # Get recent events
        recent_events = self.audit_logger.get_events(
            start_time=window_start,
            end_time=current_time
        )
        
        # Analyze events
        metrics = self._analyze_events(recent_events)
        
        # Check thresholds and trigger alerts
        for metric, value in metrics.items():
            if metric in self.alert_thresholds and value >= self.alert_thresholds[metric]:
                self._trigger_alert(f"Threshold exceeded: {metric}", {
                    'metric': metric,
                    'value': value,
                    'threshold': self.alert_thresholds[metric],
                    'window_minutes': self.monitoring_window / 60
                })
    
    def _analyze_events(self, events: List[AuditEvent]) -> Dict[str, int]:
        """Analyze events for security metrics."""
        metrics = {
            'total_events': len(events),
            'error_events': 0,
            'security_events': 0,
            'data_access_events': 0,
            'failed_operations': 0
        }
        
        for event in events:
            if event.event_type == EventType.ERROR_EVENT:
                metrics['error_events'] += 1
            elif event.event_type == EventType.SECURITY_EVENT:
                metrics['security_events'] += 1
            elif event.event_type == EventType.DATA_ACCESS:
                metrics['data_access_events'] += 1
            
            if event.level in [AuditLevel.ERROR, AuditLevel.CRITICAL]:
                metrics['failed_operations'] += 1
        
        return metrics
    
    def _trigger_alert(self, alert_type: str, details: Dict[str, Any]):
        """Trigger security alert."""
        alert_data = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'alert_type': alert_type,
            'details': details
        }
        
        # Log the alert
        self.audit_logger.log_security_event(
            security_event=alert_type,
            severity=AuditLevel.CRITICAL,
            message=f"Security alert: {alert_type}",
            metadata=alert_data
        )
        
        # Notify callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert_type, alert_data)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")


def audit_decorator(
    event_type: EventType,
    component: str,
    level: AuditLevel = AuditLevel.INFO,
    capture_inputs: bool = False,
    capture_outputs: bool = False,
    audit_logger: Optional[AuditLogger] = None
):
    """Decorator for automatic audit logging of functions.
    
    Args:
        event_type: Type of event to log
        component: Component name
        level: Audit level
        capture_inputs: Whether to capture input parameters
        capture_outputs: Whether to capture outputs
        audit_logger: Audit logger instance (uses global if None)
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal audit_logger
            if audit_logger is None:
                audit_logger = global_audit_logger
            
            start_time = time.time()
            
            # Prepare audit data
            audit_data = {
                'function_name': func.__name__,
                'module_name': func.__module__
            }
            
            if capture_inputs:
                # Capture input parameters (sanitized)
                sig = inspect.signature(func)
                bound_args = sig.bind(*args, **kwargs)
                bound_args.apply_defaults()
                audit_data['parameters'] = {
                    k: str(v)[:1000] for k, v in bound_args.arguments.items()  # Limit size
                }
            
            try:
                # Execute function
                result = func(*args, **kwargs)
                
                # Calculate execution time
                execution_time = time.time() - start_time
                audit_data['execution_time'] = execution_time
                
                if capture_outputs:
                    audit_data['output_type'] = type(result).__name__
                    if hasattr(result, 'shape'):  # For tensors/arrays
                        audit_data['output_shape'] = str(result.shape)
                
                # Log successful execution
                audit_logger.log_event(
                    event_type=event_type,
                    level=level,
                    message=f"Function {func.__name__} executed successfully",
                    component=component,
                    **audit_data
                )
                
                return result
                
            except Exception as e:
                # Log error
                execution_time = time.time() - start_time
                audit_data['execution_time'] = execution_time
                audit_data['error_type'] = type(e).__name__
                audit_data['error_message'] = str(e)
                
                audit_logger.log_event(
                    event_type=EventType.ERROR_EVENT,
                    level=AuditLevel.ERROR,
                    message=f"Function {func.__name__} failed: {str(e)}",
                    component=component,
                    **audit_data
                )
                
                raise
        
        return wrapper
    return decorator


# Global audit logger instance
global_audit_logger = AuditLogger()


def set_global_audit_logger(audit_logger: AuditLogger):
    """Set the global audit logger instance."""
    global global_audit_logger
    global_audit_logger = audit_logger


def get_global_audit_logger() -> AuditLogger:
    """Get the global audit logger instance."""
    return global_audit_logger