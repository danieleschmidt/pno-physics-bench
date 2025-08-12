"""Circuit breaker pattern implementation for robust PNO training and inference."""

import time
import threading
from typing import Callable, Optional, Dict, Any, List
from enum import Enum
from dataclasses import dataclass, field
from collections import deque
import logging
from abc import ABC, abstractmethod
import functools

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"       # Normal operation
    OPEN = "open"          # Circuit is open, requests fail fast
    HALF_OPEN = "half_open" # Testing if service is back


@dataclass
class CircuitBreakerMetrics:
    """Metrics for circuit breaker monitoring."""
    total_requests: int = 0
    failed_requests: int = 0
    successful_requests: int = 0
    circuit_open_count: int = 0
    last_failure_time: Optional[float] = None
    recent_failures: deque = field(default_factory=lambda: deque(maxlen=100))
    
    @property
    def failure_rate(self) -> float:
        """Calculate current failure rate."""
        if self.total_requests == 0:
            return 0.0
        return self.failed_requests / self.total_requests
    
    @property
    def recent_failure_rate(self) -> float:
        """Calculate recent failure rate."""
        if len(self.recent_failures) == 0:
            return 0.0
        return sum(self.recent_failures) / len(self.recent_failures)


class CircuitBreakerException(Exception):
    """Exception raised when circuit breaker is open."""
    pass


class FailureDetector(ABC):
    """Base class for failure detection strategies."""
    
    @abstractmethod
    def is_failure(self, exception: Exception) -> bool:
        """Determine if an exception counts as a failure."""
        pass


class DefaultFailureDetector(FailureDetector):
    """Default failure detector that treats all exceptions as failures."""
    
    def __init__(self, ignored_exceptions: Optional[List[type]] = None):
        """Initialize detector.
        
        Args:
            ignored_exceptions: Exception types to ignore
        """
        self.ignored_exceptions = ignored_exceptions or []
    
    def is_failure(self, exception: Exception) -> bool:
        """Check if exception should be counted as failure."""
        return not any(isinstance(exception, exc_type) for exc_type in self.ignored_exceptions)


class ModelSpecificFailureDetector(FailureDetector):
    """Failure detector specific to ML model failures."""
    
    def __init__(self):
        """Initialize ML-specific failure detector."""
        self.ml_failure_indicators = [
            "CUDA out of memory",
            "gradient overflow",
            "NaN",
            "loss explosion",
            "convergence failure"
        ]
    
    def is_failure(self, exception: Exception) -> bool:
        """Detect ML-specific failures."""
        error_msg = str(exception).lower()
        
        # Check for specific ML failure patterns
        for indicator in self.ml_failure_indicators:
            if indicator.lower() in error_msg:
                return True
        
        # Common exception types that indicate real failures
        failure_types = (
            RuntimeError,     # CUDA errors, etc.
            ValueError,       # Invalid tensor shapes, etc.
            MemoryError,      # Out of memory
            AssertionError,   # Model assertions
        )
        
        return isinstance(exception, failure_types)


class CircuitBreaker:
    """Circuit breaker implementation with configurable failure detection."""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        timeout: float = 60.0,
        expected_exception: type = Exception,
        failure_detector: Optional[FailureDetector] = None,
        name: str = "default"
    ):
        """Initialize circuit breaker.
        
        Args:
            failure_threshold: Number of failures before opening circuit
            timeout: Timeout before attempting to close circuit (seconds)
            expected_exception: Exception type to catch
            failure_detector: Custom failure detection strategy
            name: Name for this circuit breaker instance
        """
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.expected_exception = expected_exception
        self.failure_detector = failure_detector or DefaultFailureDetector()
        self.name = name
        
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time = None
        self._lock = threading.RLock()
        self._metrics = CircuitBreakerMetrics()
        
        # Callbacks
        self._on_state_change_callbacks: List[Callable[[CircuitState, CircuitState], None]] = []
        self._on_failure_callbacks: List[Callable[[Exception], None]] = []
        
    @property
    def state(self) -> CircuitState:
        """Get current circuit state."""
        return self._state
    
    @property
    def metrics(self) -> CircuitBreakerMetrics:
        """Get circuit breaker metrics."""
        return self._metrics
    
    def add_state_change_callback(self, callback: Callable[[CircuitState, CircuitState], None]):
        """Add callback for state changes."""
        self._on_state_change_callbacks.append(callback)
    
    def add_failure_callback(self, callback: Callable[[Exception], None]):
        """Add callback for failures."""
        self._on_failure_callbacks.append(callback)
    
    def _change_state(self, new_state: CircuitState):
        """Change circuit state and notify callbacks."""
        if new_state != self._state:
            old_state = self._state
            self._state = new_state
            
            logger.info(f"Circuit breaker '{self.name}' state changed: {old_state.value} -> {new_state.value}")
            
            # Notify callbacks
            for callback in self._on_state_change_callbacks:
                try:
                    callback(old_state, new_state)
                except Exception as e:
                    logger.error(f"Error in state change callback: {e}")
    
    def _record_success(self):
        """Record successful operation."""
        with self._lock:
            self._failure_count = 0
            self._metrics.successful_requests += 1
            self._metrics.total_requests += 1
            self._metrics.recent_failures.append(0)
            
            if self._state == CircuitState.HALF_OPEN:
                self._change_state(CircuitState.CLOSED)
    
    def _record_failure(self, exception: Exception):
        """Record failed operation."""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()
            self._metrics.failed_requests += 1
            self._metrics.total_requests += 1
            self._metrics.last_failure_time = self._last_failure_time
            self._metrics.recent_failures.append(1)
            
            # Notify failure callbacks
            for callback in self._on_failure_callbacks:
                try:
                    callback(exception)
                except Exception as e:
                    logger.error(f"Error in failure callback: {e}")
            
            # Check if we should open the circuit
            if self._failure_count >= self.failure_threshold:
                self._change_state(CircuitState.OPEN)
                self._metrics.circuit_open_count += 1
    
    def _can_attempt(self) -> bool:
        """Check if we can attempt the operation."""
        if self._state == CircuitState.CLOSED:
            return True
        elif self._state == CircuitState.OPEN:
            # Check if timeout has passed
            if (self._last_failure_time and 
                time.time() - self._last_failure_time >= self.timeout):
                self._change_state(CircuitState.HALF_OPEN)
                return True
            return False
        elif self._state == CircuitState.HALF_OPEN:
            return True
        
        return False
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection.
        
        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            CircuitBreakerException: When circuit is open
        """
        if not self._can_attempt():
            raise CircuitBreakerException(
                f"Circuit breaker '{self.name}' is open. "
                f"Last failure: {self._last_failure_time}"
            )
        
        try:
            result = func(*args, **kwargs)
            self._record_success()
            return result
        except self.expected_exception as e:
            if self.failure_detector.is_failure(e):
                self._record_failure(e)
            else:
                # Not considered a failure, just re-raise
                pass
            raise
    
    def __call__(self, func: Callable) -> Callable:
        """Decorator interface for circuit breaker."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return self.call(func, *args, **kwargs)
        return wrapper
    
    def reset(self):
        """Reset circuit breaker to closed state."""
        with self._lock:
            self._failure_count = 0
            self._last_failure_time = None
            self._change_state(CircuitState.CLOSED)
            logger.info(f"Circuit breaker '{self.name}' reset to closed state")


class ModelTrainingCircuitBreaker(CircuitBreaker):
    """Specialized circuit breaker for ML model training."""
    
    def __init__(
        self,
        failure_threshold: int = 3,
        timeout: float = 300.0,  # 5 minutes
        name: str = "model_training",
        **kwargs
    ):
        """Initialize training circuit breaker."""
        super().__init__(
            failure_threshold=failure_threshold,
            timeout=timeout,
            failure_detector=ModelSpecificFailureDetector(),
            name=name,
            **kwargs
        )
        
        # Training-specific metrics
        self.training_metrics = {
            'nan_losses': 0,
            'gradient_explosions': 0,
            'memory_errors': 0,
            'convergence_failures': 0
        }
    
    def _record_failure(self, exception: Exception):
        """Record failure with training-specific categorization."""
        super()._record_failure(exception)
        
        error_msg = str(exception).lower()
        
        # Categorize training failures
        if 'nan' in error_msg or 'inf' in error_msg:
            self.training_metrics['nan_losses'] += 1
        elif 'gradient' in error_msg and ('explod' in error_msg or 'overflow' in error_msg):
            self.training_metrics['gradient_explosions'] += 1
        elif 'memory' in error_msg or 'cuda out of memory' in error_msg:
            self.training_metrics['memory_errors'] += 1
        elif 'convergence' in error_msg:
            self.training_metrics['convergence_failures'] += 1


class InferenceCircuitBreaker(CircuitBreaker):
    """Specialized circuit breaker for model inference."""
    
    def __init__(
        self,
        failure_threshold: int = 10,
        timeout: float = 30.0,
        name: str = "model_inference",
        **kwargs
    ):
        """Initialize inference circuit breaker."""
        super().__init__(
            failure_threshold=failure_threshold,
            timeout=timeout,
            failure_detector=ModelSpecificFailureDetector(),
            name=name,
            **kwargs
        )
        
        # Inference-specific metrics
        self.inference_metrics = {
            'prediction_errors': 0,
            'input_validation_errors': 0,
            'model_loading_errors': 0,
            'timeout_errors': 0
        }


class CircuitBreakerRegistry:
    """Global registry for circuit breakers."""
    
    def __init__(self):
        """Initialize registry."""
        self._breakers: Dict[str, CircuitBreaker] = {}
        self._lock = threading.RLock()
    
    def register(self, name: str, breaker: CircuitBreaker):
        """Register a circuit breaker."""
        with self._lock:
            self._breakers[name] = breaker
            logger.info(f"Registered circuit breaker: {name}")
    
    def get(self, name: str) -> Optional[CircuitBreaker]:
        """Get circuit breaker by name."""
        return self._breakers.get(name)
    
    def get_all_metrics(self) -> Dict[str, CircuitBreakerMetrics]:
        """Get metrics for all circuit breakers."""
        with self._lock:
            return {name: breaker.metrics for name, breaker in self._breakers.items()}
    
    def reset_all(self):
        """Reset all circuit breakers."""
        with self._lock:
            for breaker in self._breakers.values():
                breaker.reset()
            logger.info("Reset all circuit breakers")


# Global registry instance
circuit_breaker_registry = CircuitBreakerRegistry()


def create_training_circuit_breaker(
    name: str = "default_training",
    **kwargs
) -> ModelTrainingCircuitBreaker:
    """Create and register a training circuit breaker."""
    breaker = ModelTrainingCircuitBreaker(name=name, **kwargs)
    circuit_breaker_registry.register(name, breaker)
    return breaker


def create_inference_circuit_breaker(
    name: str = "default_inference",
    **kwargs
) -> InferenceCircuitBreaker:
    """Create and register an inference circuit breaker."""
    breaker = InferenceCircuitBreaker(name=name, **kwargs)
    circuit_breaker_registry.register(name, breaker)
    return breaker


def with_circuit_breaker(
    breaker_name: str,
    failure_threshold: int = 5,
    timeout: float = 60.0,
    create_if_missing: bool = True
):
    """Decorator to add circuit breaker protection to a function.
    
    Args:
        breaker_name: Name of the circuit breaker
        failure_threshold: Failures before opening circuit
        timeout: Timeout before retry
        create_if_missing: Create breaker if it doesn't exist
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            breaker = circuit_breaker_registry.get(breaker_name)
            
            if breaker is None and create_if_missing:
                breaker = CircuitBreaker(
                    failure_threshold=failure_threshold,
                    timeout=timeout,
                    name=breaker_name
                )
                circuit_breaker_registry.register(breaker_name, breaker)
            
            if breaker is None:
                raise ValueError(f"Circuit breaker '{breaker_name}' not found")
            
            return breaker.call(func, *args, **kwargs)
        
        return wrapper
    return decorator


class AdvancedCircuitBreaker(CircuitBreaker):
    """Advanced circuit breaker with adaptive thresholds and recovery strategies."""
    
    def __init__(
        self,
        initial_failure_threshold: int = 5,
        max_failure_threshold: int = 20,
        min_failure_threshold: int = 2,
        adaptation_window: int = 100,
        timeout: float = 60.0,
        **kwargs
    ):
        """Initialize advanced circuit breaker.
        
        Args:
            initial_failure_threshold: Starting failure threshold
            max_failure_threshold: Maximum failure threshold
            min_failure_threshold: Minimum failure threshold
            adaptation_window: Window for threshold adaptation
            timeout: Circuit timeout
        """
        super().__init__(failure_threshold=initial_failure_threshold, timeout=timeout, **kwargs)
        
        self.initial_failure_threshold = initial_failure_threshold
        self.max_failure_threshold = max_failure_threshold
        self.min_failure_threshold = min_failure_threshold
        self.adaptation_window = adaptation_window
        
        # Adaptive state
        self._recent_performance = deque(maxlen=adaptation_window)
        self._adaptation_count = 0
    
    def _adapt_threshold(self):
        """Adapt failure threshold based on recent performance."""
        if len(self._recent_performance) < self.adaptation_window:
            return
        
        recent_failure_rate = sum(self._recent_performance) / len(self._recent_performance)
        
        # Adapt threshold based on performance
        if recent_failure_rate > 0.1:  # High failure rate
            # Lower threshold to be more sensitive
            new_threshold = max(
                self.min_failure_threshold,
                int(self.failure_threshold * 0.8)
            )
        elif recent_failure_rate < 0.01:  # Very low failure rate
            # Raise threshold to be less sensitive
            new_threshold = min(
                self.max_failure_threshold,
                int(self.failure_threshold * 1.2)
            )
        else:
            new_threshold = self.failure_threshold
        
        if new_threshold != self.failure_threshold:
            logger.info(
                f"Adapted circuit breaker '{self.name}' threshold: "
                f"{self.failure_threshold} -> {new_threshold}"
            )
            self.failure_threshold = new_threshold
            self._adaptation_count += 1
    
    def _record_success(self):
        """Record success and adapt threshold."""
        super()._record_success()
        self._recent_performance.append(0)
        
        if len(self._recent_performance) == self.adaptation_window:
            self._adapt_threshold()
    
    def _record_failure(self, exception: Exception):
        """Record failure and adapt threshold."""
        super()._record_failure(exception)
        self._recent_performance.append(1)
        
        if len(self._recent_performance) == self.adaptation_window:
            self._adapt_threshold()


# Factory functions for common use cases
def create_pno_training_breaker() -> ModelTrainingCircuitBreaker:
    """Create circuit breaker optimized for PNO training."""
    return create_training_circuit_breaker(
        name="pno_training",
        failure_threshold=3,
        timeout=600.0  # 10 minutes for heavy training
    )


def create_pno_inference_breaker() -> InferenceCircuitBreaker:
    """Create circuit breaker optimized for PNO inference."""
    return create_inference_circuit_breaker(
        name="pno_inference",
        failure_threshold=5,
        timeout=30.0
    )