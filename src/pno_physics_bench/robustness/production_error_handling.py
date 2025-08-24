# SECURITY NOTICE: This file has been automatically hardened for security
# - All user inputs should be validated and sanitized
# - Subprocess calls use secure alternatives
# - SQL queries use parameterized statements
# - No hardcoded secrets or credentials

"""
Production-Grade Error Handling and Recovery System for PNO Physics Bench
Generation 2 Robustness Enhancement
"""

import asyncio
import functools
import logging
import threading
import time
import traceback
import weakref
from collections import defaultdict, deque
from contextlib import contextmanager
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from dataclasses import dataclass, field

# Import existing robustness components
try:
    from .circuit_breaker import CircuitBreaker, CircuitBreakerException, CircuitState
    from .enhanced_error_handling import PNOError
except ImportError:
    # Fallback definitions for standalone operation
    class PNOError(Exception):
        def __init__(self, message: str, error_code: Optional[str] = None, context: Optional[Dict] = None):
            super().__init__(message)
            self.message = message
            self.error_code = error_code or "PNO_GENERIC_ERROR"
            self.context = context or {}
            self.timestamp = datetime.now().isoformat()

logger = logging.getLogger(__name__)


class RecoveryStrategy(Enum):
    """Recovery strategies for different types of failures."""
    IMMEDIATE_RETRY = auto()
    EXPONENTIAL_BACKOFF = auto()
    CIRCUIT_BREAKER = auto()
    GRACEFUL_DEGRADATION = auto()
    FAILOVER = auto()
    ABORT = auto()


class FailureCategory(Enum):
    """Categories of failures for intelligent handling."""
    TRANSIENT = auto()          # Temporary network/resource issues
    PERSISTENT = auto()         # Configuration or logic errors
    RESOURCE_EXHAUSTION = auto() # Memory, CPU, disk issues
    SECURITY = auto()           # Security violations
    CORRUPTION = auto()         # Data corruption issues
    EXTERNAL_DEPENDENCY = auto() # Third-party service failures


@dataclass
class RetryConfig:
    """Configuration for retry mechanisms."""
    max_attempts: int = 3
    initial_delay: float = 0.1
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    retry_exceptions: Tuple[type, ...] = (Exception,)
    abort_exceptions: Tuple[type, ...] = ()


@dataclass
class FailureRecord:
    """Record of a failure event."""
    timestamp: datetime
    error_type: str
    error_message: str
    category: FailureCategory
    context: Dict[str, Any]
    stack_trace: str
    recovery_attempted: bool = False
    recovery_successful: bool = False
    recovery_strategy: Optional[RecoveryStrategy] = None


class FailureAnalyzer:
    """Analyzes failures to categorize them and suggest recovery strategies."""
    
    def __init__(self):
        self.failure_patterns = {
            # Transient failures
            r'connection.*timeout|network.*unreachable|temporary.*failure': FailureCategory.TRANSIENT,
            r'cuda.*out.*of.*memory|memory.*error': FailureCategory.RESOURCE_EXHAUSTION,
            r'permission.*denied|access.*denied|unauthorized': FailureCategory.SECURITY,
            r'file.*not.*found|no.*such.*file': FailureCategory.PERSISTENT,
            r'checksum.*mismatch|corruption.*detected': FailureCategory.CORRUPTION,
            r'api.*error|service.*unavailable|external.*error': FailureCategory.EXTERNAL_DEPENDENCY,
        }
        
        self.recovery_strategies = {
            FailureCategory.TRANSIENT: RecoveryStrategy.EXPONENTIAL_BACKOFF,
            FailureCategory.RESOURCE_EXHAUSTION: RecoveryStrategy.GRACEFUL_DEGRADATION,
            FailureCategory.SECURITY: RecoveryStrategy.ABORT,
            FailureCategory.PERSISTENT: RecoveryStrategy.ABORT,
            FailureCategory.CORRUPTION: RecoveryStrategy.FAILOVER,
            FailureCategory.EXTERNAL_DEPENDENCY: RecoveryStrategy.CIRCUIT_BREAKER,
        }
    
    def analyze_failure(self, error: Exception, context: Dict[str, Any]) -> Tuple[FailureCategory, RecoveryStrategy]:
        """Analyze a failure and suggest recovery strategy."""
        error_message = str(error).lower()
        error_type = type(error).__name__.lower()
        
        # Check for specific patterns
        import re
        for pattern, category in self.failure_patterns.items():
            if re.search(pattern, error_message) or re.search(pattern, error_type):
                return category, self.recovery_strategies[category]
        
        # Default categorization based on exception type
        if isinstance(error, (ConnectionError, TimeoutError)):
            return FailureCategory.TRANSIENT, RecoveryStrategy.EXPONENTIAL_BACKOFF
        elif isinstance(error, MemoryError):
            return FailureCategory.RESOURCE_EXHAUSTION, RecoveryStrategy.GRACEFUL_DEGRADATION
        elif isinstance(error, PermissionError):
            return FailureCategory.SECURITY, RecoveryStrategy.ABORT
        else:
            return FailureCategory.PERSISTENT, RecoveryStrategy.IMMEDIATE_RETRY


class ProductionRetryHandler:
    """Production-grade retry handler with intelligent backoff and circuit breaking."""
    
    def __init__(self, config: Optional[RetryConfig] = None):
        self.config = config or RetryConfig()
        self.failure_analyzer = FailureAnalyzer()
        self.attempt_history = defaultdict(list)
        self.circuit_breakers = {}
        self._lock = threading.Lock()
    
    def get_circuit_breaker(self, operation_id: str) -> CircuitBreaker:
        """Get or create circuit breaker for operation."""
        if operation_id not in self.circuit_breakers:
            try:
                from .circuit_breaker import CircuitBreaker
                self.circuit_breakers[operation_id] = CircuitBreaker(
                    failure_threshold=5,
                    timeout=300.0,  # 5 minutes
                    name=f"retry_handler_{operation_id}"
                )
            except ImportError:
                # Fallback to simple circuit breaker
                self.circuit_breakers[operation_id] = SimpleCircuitBreaker()
        
        return self.circuit_breakers[operation_id]
    
    def calculate_delay(self, attempt: int, base_delay: float) -> float:
        """Calculate delay with exponential backoff and jitter."""
        delay = min(
            base_delay * (self.config.exponential_base ** (attempt - 1)),
            self.config.max_delay
        )
        
        if self.config.jitter:
            import random
            delay *= (0.5 + random.random() * 0.5)  # 50-100% of calculated delay
        
        return delay
    
    def should_retry(self, error: Exception, attempt: int, category: FailureCategory) -> bool:
        """Determine if operation should be retried."""
        if attempt >= self.config.max_attempts:
            return False
        
        if any(isinstance(error, exc) for exc in self.config.abort_exceptions):
            return False
        
        if category == FailureCategory.SECURITY:
            return False
        
        if category == FailureCategory.PERSISTENT and attempt > 1:
            return False
        
        return any(isinstance(error, exc) for exc in self.config.retry_exceptions)
    
    async def retry_async(self, func: Callable, *args, operation_id: str = None, **kwargs) -> Any:
        """Async retry wrapper with intelligent failure handling."""
        operation_id = operation_id or f"{func.__name__}_{id(func)}"
        circuit_breaker = self.get_circuit_breaker(operation_id)
        
        last_error = None
        
        for attempt in range(1, self.config.max_attempts + 1):
            try:
                # Check circuit breaker
                if hasattr(circuit_breaker, 'call'):
                    if attempt > 1:  # Skip circuit breaker on first attempt
                        return circuit_breaker.call(func, *args, **kwargs)
                    else:
                        result = func(*args, **kwargs)
                        if hasattr(circuit_breaker, '_record_success'):
                            circuit_breaker._record_success()
                        return result
                else:
                    return func(*args, **kwargs)
                    
            except Exception as error:
                last_error = error
                category, recovery_strategy = self.failure_analyzer.analyze_failure(
                    error, {"operation": operation_id, "attempt": attempt}
                )
                
                # Record failure
                with self._lock:
                    self.attempt_history[operation_id].append({
                        'timestamp': datetime.now(),
                        'attempt': attempt,
                        'error': str(error),
                        'category': category.name
                    })
                
                # Record circuit breaker failure
                if hasattr(circuit_breaker, '_record_failure'):
                    circuit_breaker._record_failure(error)
                
                logger.warning(
                    f"Attempt {attempt}/{self.config.max_attempts} failed for {operation_id}: "
                    f"{error} (category: {category.name})"
                )
                
                # Check if we should retry
                if not self.should_retry(error, attempt, category):
                    logger.error(f"Aborting {operation_id} after {attempt} attempts")
                    break
                
                # Apply recovery strategy
                if recovery_strategy == RecoveryStrategy.EXPONENTIAL_BACKOFF:
                    delay = self.calculate_delay(attempt, self.config.initial_delay)
                    logger.info(f"Retrying {operation_id} in {delay:.2f} seconds")
                    await asyncio.sleep(delay)
                elif recovery_strategy == RecoveryStrategy.CIRCUIT_BREAKER:
                    # Circuit breaker will handle the delay
                    pass
                elif recovery_strategy == RecoveryStrategy.ABORT:
                    logger.error(f"Aborting {operation_id} due to {category.name}")
                    break
        
        # All attempts failed
        if last_error:
            raise PNOError(
                f"Operation {operation_id} failed after {self.config.max_attempts} attempts",
                error_code="RETRY_EXHAUSTED",
                context={
                    'operation_id': operation_id,
                    'attempts': self.config.max_attempts,
                    'last_error': str(last_error),
                    'attempt_history': self.attempt_history[operation_id][-5:]  # Last 5 attempts
                }
            ) from last_error
    
    def retry_sync(self, func: Callable, *args, operation_id: str = None, **kwargs) -> Any:
        """Synchronous retry wrapper."""
        operation_id = operation_id or f"{func.__name__}_{id(func)}"
        circuit_breaker = self.get_circuit_breaker(operation_id)
        
        last_error = None
        
        for attempt in range(1, self.config.max_attempts + 1):
            try:
                # Check circuit breaker
                if hasattr(circuit_breaker, 'call'):
                    if attempt > 1:  # Skip circuit breaker on first attempt
                        return circuit_breaker.call(func, *args, **kwargs)
                    else:
                        result = func(*args, **kwargs)
                        if hasattr(circuit_breaker, '_record_success'):
                            circuit_breaker._record_success()
                        return result
                else:
                    return func(*args, **kwargs)
                    
            except Exception as error:
                last_error = error
                category, recovery_strategy = self.failure_analyzer.analyze_failure(
                    error, {"operation": operation_id, "attempt": attempt}
                )
                
                # Record failure
                with self._lock:
                    self.attempt_history[operation_id].append({
                        'timestamp': datetime.now(),
                        'attempt': attempt,
                        'error': str(error),
                        'category': category.name
                    })
                
                logger.warning(
                    f"Attempt {attempt}/{self.config.max_attempts} failed for {operation_id}: "
                    f"{error} (category: {category.name})"
                )
                
                # Check if we should retry
                if not self.should_retry(error, attempt, category):
                    logger.error(f"Aborting {operation_id} after {attempt} attempts")
                    break
                
                # Apply recovery strategy
                if recovery_strategy == RecoveryStrategy.EXPONENTIAL_BACKOFF:
                    delay = self.calculate_delay(attempt, self.config.initial_delay)
                    logger.info(f"Retrying {operation_id} in {delay:.2f} seconds")
                    time.sleep(delay)
                elif recovery_strategy == RecoveryStrategy.ABORT:
                    logger.error(f"Aborting {operation_id} due to {category.name}")
                    break
        
        # All attempts failed
        if last_error:
            raise PNOError(
                f"Operation {operation_id} failed after {self.config.max_attempts} attempts",
                error_code="RETRY_EXHAUSTED",
                context={
                    'operation_id': operation_id,
                    'attempts': self.config.max_attempts,
                    'last_error': str(last_error),
                    'attempt_history': self.attempt_history[operation_id][-5:]
                }
            ) from last_error


class SimpleCircuitBreaker:
    """Simple fallback circuit breaker when advanced one isn't available."""
    
    def __init__(self, failure_threshold: int = 5, timeout: float = 300.0):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'CLOSED'
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        if self.state == 'OPEN':
            if (self.last_failure_time and 
                time.time() - self.last_failure_time >= self.timeout):
                self.state = 'HALF_OPEN'
            else:
                raise Exception("Circuit breaker is OPEN")
        
        try:
            result = func(*args, **kwargs)
            self._record_success()
            return result
        except Exception as e:
            self._record_failure()
            raise
    
    def _record_success(self):
        self.failure_count = 0
        if self.state == 'HALF_OPEN':
            self.state = 'CLOSED'
    
    def _record_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= self.failure_threshold:
            self.state = 'OPEN'


class FaultToleranceManager:
    """Manages fault tolerance across the PNO system."""
    
    def __init__(self):
        self.retry_handler = ProductionRetryHandler()
        self.failure_history = deque(maxlen=1000)
        self.recovery_callbacks = defaultdict(list)
        self.shutdown_callbacks = []
        self.health_checks = {}
        self._shutdown_initiated = False
        self._lock = threading.Lock()
    
    def register_health_check(self, name: str, check_func: Callable[[], bool]):
        """Register a health check function."""
        self.health_checks[name] = check_func
    
    def register_recovery_callback(self, error_type: str, callback: Callable):
        """Register callback for specific error types."""
        self.recovery_callbacks[error_type].append(callback)
    
    def register_shutdown_callback(self, callback: Callable):
        """Register callback for graceful shutdown."""
        self.shutdown_callbacks.append(callback)
    
    def execute_health_checks(self) -> Dict[str, bool]:
        """Execute all registered health checks."""
        results = {}
        for name, check_func in self.health_checks.items():
            try:
                results[name] = check_func()
            except Exception as e:
                logger.error(f"Health check {name} failed: {e}")
                results[name] = False
        return results
    
    def handle_critical_failure(self, error: Exception, context: Dict[str, Any]):
        """Handle critical system failures."""
        failure_record = FailureRecord(
            timestamp=datetime.now(),
            error_type=type(error).__name__,
            error_message=str(error),
            category=FailureCategory.PERSISTENT,
            context=context,
            stack_trace=traceback.format_exc()
        )
        
        with self._lock:
            self.failure_history.append(failure_record)
        
        logger.critical(f"Critical failure detected: {error}")
        
        # Execute recovery callbacks
        error_type = type(error).__name__
        for callback in self.recovery_callbacks.get(error_type, []):
            try:
                callback(error, context)
            except Exception as callback_error:
                logger.error(f"Recovery callback failed: {callback_error}")
        
        # Check if graceful shutdown is needed
        if self._should_shutdown(failure_record):
            self.initiate_graceful_shutdown()
    
    def _should_shutdown(self, failure: FailureRecord) -> bool:
        """Determine if system should shutdown based on failure pattern."""
        # Check for critical failure patterns
        critical_patterns = [
            'out of memory',
            'disk full',
            'corruption detected',
            'security breach'
        ]
        
        error_msg = failure.error_message.lower()
        if any(pattern in error_msg for pattern in critical_patterns):
            return True
        
        # Check failure rate in recent history
        recent_failures = [
            f for f in self.failure_history 
            if datetime.now() - f.timestamp <= timedelta(minutes=5)
        ]
        
        if len(recent_failures) > 10:  # More than 10 failures in 5 minutes
            logger.warning("High failure rate detected, considering shutdown")
            return True
        
        return False
    
    def initiate_graceful_shutdown(self):
        """Initiate graceful system shutdown."""
        if self._shutdown_initiated:
            return
        
        self._shutdown_initiated = True
        logger.critical("Initiating graceful shutdown due to critical failures")
        
        # Execute shutdown callbacks
        for callback in self.shutdown_callbacks:
            try:
                callback()
            except Exception as e:
                logger.error(f"Shutdown callback failed: {e}")
        
        # Final system cleanup
        self._perform_final_cleanup()
    
    def _perform_final_cleanup(self):
        """Perform final cleanup before shutdown."""
        try:
            # Save failure history for post-mortem analysis
            import json
            failure_data = []
            for failure in self.failure_history:
                failure_data.append({
                    'timestamp': failure.timestamp.isoformat(),
                    'error_type': failure.error_type,
                    'error_message': failure.error_message,
                    'category': failure.category.name,
                    'context': failure.context
                })
            
            with open(f'/root/repo/logs/failure_history_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json', 'w') as f:
                json.dump(failure_data, f, indent=2)
            
            logger.info("Failure history saved for analysis")
            
        except Exception as e:
            logger.error(f"Failed to save failure history: {e}")


# Decorators for easy integration
def with_retry(config: Optional[RetryConfig] = None, operation_id: Optional[str] = None):
    """Decorator to add retry capability to functions."""
    retry_handler = ProductionRetryHandler(config)
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return retry_handler.retry_sync(
                func, *args, operation_id=operation_id or func.__name__, **kwargs
            )
        return wrapper
    return decorator


def with_async_retry(config: Optional[RetryConfig] = None, operation_id: Optional[str] = None):
    """Decorator to add async retry capability to functions."""
    retry_handler = ProductionRetryHandler(config)
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            return await retry_handler.retry_async(
                func, *args, operation_id=operation_id or func.__name__, **kwargs
            )
        return wrapper
    return decorator


def fault_tolerant(recovery_strategy: RecoveryStrategy = RecoveryStrategy.EXPONENTIAL_BACKOFF):
    """Decorator to make functions fault tolerant."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            fault_manager = FaultToleranceManager()
            
            try:
                return func(*args, **kwargs)
            except Exception as e:
                context = {
                    'function': func.__name__,
                    'args_count': len(args),
                    'kwargs_count': len(kwargs),
                    'recovery_strategy': recovery_strategy.name
                }
                
                fault_manager.handle_critical_failure(e, context)
                raise
                
        return wrapper
    return decorator


# Global instances
global_retry_handler = ProductionRetryHandler()
global_fault_manager = FaultToleranceManager()

# Health check functions
def check_memory_health() -> bool:
    """Check system memory health."""
    try:
        import psutil
        memory = psutil.virtual_memory()
        return memory.percent < 90  # Less than 90% usage
    except ImportError:
        return True  # Assume healthy if can't check


def check_disk_health() -> bool:
    """Check disk space health."""
    try:
        import psutil
        disk = psutil.disk_usage('/')
        return (disk.free / disk.total) > 0.1  # At least 10% free
    except ImportError:
        return True


def check_process_health() -> bool:
    """Check if critical processes are responsive."""
    try:
        # Basic responsiveness check
        import threading
        return threading.active_count() < 100  # Not too many threads
    except Exception:
        return True


# Register default health checks
global_fault_manager.register_health_check('memory', check_memory_health)
global_fault_manager.register_health_check('disk', check_disk_health)
global_fault_manager.register_health_check('processes', check_process_health)