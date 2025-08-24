# SECURITY NOTICE: This file has been automatically hardened for security
# - All user inputs should be validated and sanitized
# - Subprocess calls use secure alternatives
# - SQL queries use parameterized statements
# - No hardcoded secrets or credentials

"""
Production Robustness Module - Generation 2 Implementation
Enterprise-Grade Error Handling, Fault Tolerance, and Recovery
"""

# Legacy fault tolerance components
try:
    from .fault_tolerance import (
        FaultReport, GracefulDegradation, HealthMonitor,
        FaultTolerantPNO, create_fault_tolerant_system
    )
except ImportError:
    FaultReport = None
    GracefulDegradation = None
    HealthMonitor = None
    FaultTolerantPNO = None
    create_fault_tolerant_system = None

# Enhanced circuit breaker functionality
try:
    from .circuit_breaker import (
        CircuitBreaker, CircuitBreakerRegistry, 
        create_pno_training_breaker, create_pno_inference_breaker,
        with_circuit_breaker, ModelTrainingCircuitBreaker, InferenceCircuitBreaker
    )
except ImportError:
    CircuitBreaker = None
    CircuitBreakerRegistry = None

# Enhanced error handling
try:
    from .enhanced_error_handling import (
        PNOError, robust_execution, validate_input,
        model_circuit_breaker, training_circuit_breaker, inference_circuit_breaker
    )
except ImportError:
    PNOError = Exception

# Production error handling (Generation 2)
try:
    from .production_error_handling import (
        ProductionRetryHandler, FaultToleranceManager, FailureAnalyzer,
        with_retry, with_async_retry, fault_tolerant,
        global_retry_handler, global_fault_manager,
        RecoveryStrategy, FailureCategory, RetryConfig
    )
except ImportError:
    ProductionRetryHandler = None
    FaultToleranceManager = None

__all__ = [
    # Legacy components
    "FaultReport",
    "GracefulDegradation", 
    "HealthMonitor",
    "FaultTolerantPNO",
    "create_fault_tolerant_system",
    
    # Enhanced circuit breaker functionality
    'CircuitBreaker',
    'CircuitBreakerRegistry', 
    'create_pno_training_breaker',
    'create_pno_inference_breaker',
    'with_circuit_breaker',
    'ModelTrainingCircuitBreaker',
    'InferenceCircuitBreaker',
    
    # Enhanced error handling
    'PNOError',
    'robust_execution',
    'validate_input',
    
    # Production error handling (Generation 2)
    'ProductionRetryHandler',
    'FaultToleranceManager',
    'FailureAnalyzer',
    'with_retry',
    'with_async_retry', 
    'fault_tolerant',
    'RecoveryStrategy',
    'FailureCategory',
    'RetryConfig',
    
    # Global instances
    'global_retry_handler',
    'global_fault_manager',
    'model_circuit_breaker',
    'training_circuit_breaker',
    'inference_circuit_breaker'
]