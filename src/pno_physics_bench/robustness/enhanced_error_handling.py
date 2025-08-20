# SECURITY NOTICE: This file has been automatically hardened for security
# - All user inputs should be validated and sanitized
# - Subprocess calls use secure alternatives
# - SQL queries use parameterized statements
# - No hardcoded secrets or credentials


"""Advanced Error Handling Framework for PNO Physics Bench"""

import functools
import logging
import traceback
from typing import Any, Callable, Optional, Type, Union
from datetime import datetime

logger = logging.getLogger(__name__)

class PNOError(Exception):
    """Base exception class for PNO Physics Bench"""
    def __init__(self, message: str, error_code: Optional[str] = None, context: Optional[Dict] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or "PNO_GENERIC_ERROR"
        self.context = context or {}
        self.timestamp = datetime.now().isoformat()

class ModelInitializationError(PNOError):
    """Raised when model initialization fails"""
    pass

class TrainingError(PNOError):
    """Raised during training failures"""
    pass

class DataValidationError(PNOError):
    """Raised when data validation fails"""
    pass

class UncertaintyComputationError(PNOError):
    """Raised when uncertainty computation fails"""
    pass

class InferenceError(PNOError):
    """Raised during inference failures"""
    pass

def robust_execution(
    default_return: Any = None,
    exceptions: tuple = (Exception,),
    log_errors: bool = True,
    reraise: bool = False
):
    """Decorator for robust function execution with comprehensive error handling"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except exceptions as e:
                error_info = {
                    "function": func.__name__,
                    "args": str(args)[:200],  # Truncate for security
                    "kwargs": str({k: str(v)[:100] for k, v in kwargs.items()}),
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "traceback": traceback.format_exc()
                }
                
                if log_errors:
                    logger.error(f"Robust execution failed in {func.__name__}: {error_info}")
                
                if reraise:
                    raise PNOError(
                        f"Function {func.__name__} failed: {str(e)}",
                        error_code=f"ROBUST_EXEC_FAIL_{func.__name__.upper()}",
                        context=error_info
                    ) from e
                
                return default_return
        return wrapper
    return decorator

def validate_input(
    check_fn: Callable[[Any], bool],
    error_message: str = "Input validation failed",
    error_class: Type[PNOError] = DataValidationError
):
    """Decorator for input validation"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Validate all arguments
            all_args = list(args) + list(kwargs.values())
            for arg in all_args:
                if not check_fn(arg):
                    raise error_class(
                        f"{error_message} in {func.__name__}",
                        error_code="INPUT_VALIDATION_FAILED",
                        context={"function": func.__name__, "invalid_input": str(type(arg))}
                    )
            return func(*args, **kwargs)
        return wrapper
    return decorator

class CircuitBreaker:
    """Circuit breaker pattern for preventing cascade failures"""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
    def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        if self.state == "OPEN":
            if self._should_attempt_reset():
                self.state = "HALF_OPEN"
            else:
                raise PNOError(
                    "Circuit breaker is OPEN - too many failures",
                    error_code="CIRCUIT_BREAKER_OPEN"
                )
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset"""
        if self.last_failure_time is None:
            return True
        return (datetime.now().timestamp() - self.last_failure_time) > self.recovery_timeout
    
    def _on_success(self):
        """Handle successful execution"""
        self.failure_count = 0
        self.state = "CLOSED"
    
    def _on_failure(self):
        """Handle failed execution"""
        self.failure_count += 1
        self.last_failure_time = datetime.now().timestamp()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
            logger.warning(f"Circuit breaker opened after {self.failure_count} failures")

# Global circuit breakers for different components
model_circuit_breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=30)
training_circuit_breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=60)
inference_circuit_breaker = CircuitBreaker(failure_threshold=10, recovery_timeout=15)
