"""Fault tolerance and reliability systems for PNO models."""

import time
import logging
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass
from pathlib import Path
import json
from contextlib import contextmanager
from functools import wraps
import traceback

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

import numpy as np


@dataclass
class FaultReport:
    """Report structure for system faults."""
    timestamp: str
    fault_type: str
    severity: str  # critical, high, medium, low
    component: str
    error_message: str
    stack_trace: str
    recovery_action: str
    success: bool
    performance_impact: Dict[str, float]


class CircuitBreaker:
    """Circuit breaker pattern for fault tolerance."""
    
    def __init__(self, 
                 failure_threshold: int = 5,
                 recovery_timeout: int = 60,
                 expected_exception: Exception = Exception):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        
        self.logger = logging.getLogger(__name__)
    
    def __call__(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if self.state == "OPEN":
                if self._should_attempt_reset():
                    self.state = "HALF_OPEN"
                else:
                    raise Exception(f"Circuit breaker is OPEN. Service unavailable.")
            
            try:
                result = func(*args, **kwargs)
                self._on_success()
                return result
                
            except self.expected_exception as e:
                self._on_failure()
                raise e
        
        return wrapper
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        return (time.time() - self.last_failure_time) >= self.recovery_timeout
    
    def _on_success(self):
        """Handle successful operation."""
        self.failure_count = 0
        if self.state == "HALF_OPEN":
            self.state = "CLOSED"
            self.logger.info("Circuit breaker reset to CLOSED")
    
    def _on_failure(self):
        """Handle failed operation."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
            self.logger.warning(f"Circuit breaker opened after {self.failure_count} failures")


class RetryStrategy:
    """Configurable retry strategy for operations."""
    
    def __init__(self, 
                 max_attempts: int = 3,
                 backoff_strategy: str = "exponential",
                 base_delay: float = 1.0,
                 max_delay: float = 60.0,
                 jitter: bool = True):
        self.max_attempts = max_attempts
        self.backoff_strategy = backoff_strategy
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.jitter = jitter
        
        self.logger = logging.getLogger(__name__)
    
    def __call__(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(self.max_attempts):
                try:
                    return func(*args, **kwargs)
                    
                except Exception as e:
                    last_exception = e
                    
                    if attempt == self.max_attempts - 1:
                        self.logger.error(f"All retry attempts failed for {func.__name__}")
                        break
                    
                    delay = self._calculate_delay(attempt)
                    self.logger.warning(f"Attempt {attempt + 1} failed, retrying in {delay:.2f}s")
                    time.sleep(delay)
            
            raise last_exception
        
        return wrapper
    
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay for next retry attempt."""
        if self.backoff_strategy == "fixed":
            delay = self.base_delay
        elif self.backoff_strategy == "linear":
            delay = self.base_delay * (attempt + 1)
        elif self.backoff_strategy == "exponential":
            delay = self.base_delay * (2 ** attempt)
        else:
            delay = self.base_delay
        
        # Apply maximum delay limit
        delay = min(delay, self.max_delay)
        
        # Add jitter to prevent thundering herd
        if self.jitter:
            delay *= (0.5 + np.random.random() * 0.5)
        
        return delay


class GracefulDegradation:
    """Graceful degradation for model operations."""
    
    def __init__(self, fallback_strategy: str = "simple_model"):
        self.fallback_strategy = fallback_strategy
        self.performance_history = []
        self.logger = logging.getLogger(__name__)
    
    def __call__(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                start_time = time.time()
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                
                # Monitor performance
                self._record_performance(execution_time, success=True)
                return result
                
            except Exception as e:
                self.logger.warning(f"Primary operation failed: {e}")
                execution_time = time.time() - start_time
                self._record_performance(execution_time, success=False)
                
                # Attempt graceful degradation
                return self._fallback_operation(*args, **kwargs)
        
        return wrapper
    
    def _record_performance(self, execution_time: float, success: bool):
        """Record performance metrics."""
        self.performance_history.append({
            "timestamp": time.time(),
            "execution_time": execution_time,
            "success": success
        })
        
        # Keep only recent history
        if len(self.performance_history) > 1000:
            self.performance_history = self.performance_history[-1000:]
    
    def _fallback_operation(self, *args, **kwargs):
        """Execute fallback operation based on strategy."""
        if self.fallback_strategy == "simple_model":
            return self._simple_model_fallback(*args, **kwargs)
        elif self.fallback_strategy == "cached_result":
            return self._cached_result_fallback(*args, **kwargs)
        elif self.fallback_strategy == "default_value":
            return self._default_value_fallback(*args, **kwargs)
        else:
            raise NotImplementedError(f"Unknown fallback strategy: {self.fallback_strategy}")
    
    def _simple_model_fallback(self, *args, **kwargs):
        """Use a simple model as fallback."""
        self.logger.info("Using simple model fallback")
        
        # Return simple prediction (placeholder)
        if args and hasattr(args[0], 'shape'):
            input_tensor = args[0]
            if hasattr(input_tensor, 'shape'):
                return input_tensor * 0.9  # Simple transformation
        
        return None
    
    def _cached_result_fallback(self, *args, **kwargs):
        """Return cached result if available."""
        self.logger.info("Using cached result fallback")
        # Implementation would depend on caching system
        return None
    
    def _default_value_fallback(self, *args, **kwargs):
        """Return safe default value."""
        self.logger.info("Using default value fallback")
        return {"prediction": None, "uncertainty": 1.0}


class HealthMonitor:
    """Health monitoring system for PNO models."""
    
    def __init__(self, 
                 check_interval: int = 30,
                 alert_thresholds: Optional[Dict[str, float]] = None):
        self.check_interval = check_interval
        self.alert_thresholds = alert_thresholds or {
            "cpu_usage": 90.0,
            "memory_usage": 85.0,
            "error_rate": 0.05,
            "response_time": 5.0
        }
        
        self.health_status = {"status": "healthy", "last_check": time.time()}
        self.alerts = []
        self.metrics_history = []
        
        self.logger = logging.getLogger(__name__)
    
    def check_health(self, model=None, recent_predictions=None) -> Dict[str, Any]:
        """Perform comprehensive health check."""
        
        health_report = {
            "timestamp": time.time(),
            "overall_status": "healthy",
            "checks": {},
            "alerts": [],
            "recommendations": []
        }
        
        # Check system resources
        system_health = self._check_system_health()
        health_report["checks"]["system"] = system_health
        
        # Check model health
        if model is not None:
            model_health = self._check_model_health(model)
            health_report["checks"]["model"] = model_health
        
        # Check prediction quality
        if recent_predictions is not None:
            prediction_health = self._check_prediction_health(recent_predictions)
            health_report["checks"]["predictions"] = prediction_health
        
        # Determine overall status
        all_checks = [check["status"] for check in health_report["checks"].values()]
        if "critical" in all_checks:
            health_report["overall_status"] = "critical"
        elif "warning" in all_checks:
            health_report["overall_status"] = "warning"
        
        # Generate alerts and recommendations
        health_report["alerts"] = self._generate_alerts(health_report["checks"])
        health_report["recommendations"] = self._generate_recommendations(health_report["checks"])
        
        # Store health report
        self.metrics_history.append(health_report)
        if len(self.metrics_history) > 100:
            self.metrics_history = self.metrics_history[-100:]
        
        return health_report
    
    def _check_system_health(self) -> Dict[str, Any]:
        """Check system resource health."""
        import psutil
        
        cpu_usage = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        status = "healthy"
        if cpu_usage > self.alert_thresholds["cpu_usage"]:
            status = "warning"
        if memory.percent > self.alert_thresholds["memory_usage"]:
            status = "critical" if status != "critical" else status
        
        return {
            "status": status,
            "cpu_usage": cpu_usage,
            "memory_usage": memory.percent,
            "disk_usage": disk.percent,
            "available_memory_gb": memory.available / (1024**3)
        }
    
    def _check_model_health(self, model) -> Dict[str, Any]:
        """Check model-specific health indicators."""
        
        health_indicators = {
            "status": "healthy",
            "parameters_finite": True,
            "gradients_finite": True,
            "model_size_mb": 0
        }
        
        if HAS_TORCH and isinstance(model, nn.Module):
            # Check for NaN/Inf parameters
            for name, param in model.named_parameters():
                if param is not None:
                    if not torch.isfinite(param).all():
                        health_indicators["parameters_finite"] = False
                        health_indicators["status"] = "critical"
                        break
            
            # Check gradients if available
            for name, param in model.named_parameters():
                if param.grad is not None:
                    if not torch.isfinite(param.grad).all():
                        health_indicators["gradients_finite"] = False
                        health_indicators["status"] = "warning"
                        break
            
            # Estimate model size
            param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
            health_indicators["model_size_mb"] = param_count * 4 / (1024**2)  # Assuming float32
        
        return health_indicators
    
    def _check_prediction_health(self, predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Check prediction quality and consistency."""
        
        if not predictions:
            return {"status": "unknown", "message": "No predictions to analyze"}
        
        health_indicators = {
            "status": "healthy",
            "num_predictions": len(predictions),
            "nan_predictions": 0,
            "inf_predictions": 0,
            "uncertainty_range": [0.0, 1.0]
        }
        
        # Analyze predictions
        nan_count = 0
        inf_count = 0
        uncertainties = []
        
        for pred in predictions:
            if 'prediction' in pred and pred['prediction'] is not None:
                if hasattr(pred['prediction'], 'isnan'):
                    if pred['prediction'].isnan().any():
                        nan_count += 1
                    if pred['prediction'].isinf().any():
                        inf_count += 1
                
                if 'uncertainty' in pred:
                    uncertainties.append(float(pred['uncertainty']))
        
        health_indicators["nan_predictions"] = nan_count
        health_indicators["inf_predictions"] = inf_count
        
        if uncertainties:
            health_indicators["uncertainty_range"] = [min(uncertainties), max(uncertainties)]
        
        # Determine status
        if nan_count > 0 or inf_count > 0:
            health_indicators["status"] = "critical"
        elif len(uncertainties) == 0:
            health_indicators["status"] = "warning"
        
        return health_indicators
    
    def _generate_alerts(self, checks: Dict[str, Dict]) -> List[str]:
        """Generate alerts based on health checks."""
        alerts = []
        
        for check_name, check_result in checks.items():
            if check_result["status"] == "critical":
                alerts.append(f"CRITICAL: {check_name} health check failed")
            elif check_result["status"] == "warning":
                alerts.append(f"WARNING: {check_name} health check shows issues")
        
        return alerts
    
    def _generate_recommendations(self, checks: Dict[str, Dict]) -> List[str]:
        """Generate recommendations based on health status."""
        recommendations = []
        
        # System recommendations
        if "system" in checks:
            system_check = checks["system"]
            if system_check.get("cpu_usage", 0) > 80:
                recommendations.append("Consider scaling compute resources or optimizing model")
            if system_check.get("memory_usage", 0) > 75:
                recommendations.append("Monitor memory usage and consider memory optimization")
        
        # Model recommendations
        if "model" in checks:
            model_check = checks["model"]
            if not model_check.get("parameters_finite", True):
                recommendations.append("URGENT: Model parameters contain NaN/Inf values - restart training")
            if not model_check.get("gradients_finite", True):
                recommendations.append("Check learning rate and gradient clipping settings")
        
        # Prediction recommendations
        if "predictions" in checks:
            pred_check = checks["predictions"]
            if pred_check.get("nan_predictions", 0) > 0:
                recommendations.append("Investigate input data quality and model stability")
            if pred_check.get("uncertainty_range", [0, 1])[1] > 2.0:
                recommendations.append("High uncertainty values detected - check model calibration")
        
        return recommendations


class FaultTolerantPNO:
    """Fault-tolerant wrapper for PNO models."""
    
    def __init__(self, 
                 base_model,
                 enable_circuit_breaker: bool = True,
                 enable_retry: bool = True,
                 enable_degradation: bool = True,
                 checkpoint_frequency: int = 100):
        
        self.base_model = base_model
        self.checkpoint_frequency = checkpoint_frequency
        self.operation_count = 0
        self.fault_reports = []
        
        # Initialize fault tolerance mechanisms
        if enable_circuit_breaker:
            self.circuit_breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=30)
        else:
            self.circuit_breaker = None
        
        if enable_retry:
            self.retry_strategy = RetryStrategy(max_attempts=2, base_delay=0.5)
        else:
            self.retry_strategy = None
        
        if enable_degradation:
            self.graceful_degradation = GracefulDegradation(fallback_strategy="simple_model")
        else:
            self.graceful_degradation = None
        
        self.health_monitor = HealthMonitor(check_interval=60)
        self.logger = logging.getLogger(__name__)
    
    @contextmanager
    def fault_tolerance_context(self):
        """Context manager for fault-tolerant operations."""
        start_time = time.time()
        try:
            yield
            self.operation_count += 1
            
            # Periodic health check
            if self.operation_count % 50 == 0:
                health_report = self.health_monitor.check_health(self.base_model)
                if health_report["overall_status"] != "healthy":
                    self.logger.warning(f"Health check alerts: {health_report['alerts']}")
            
            # Periodic checkpointing
            if self.operation_count % self.checkpoint_frequency == 0:
                self._create_checkpoint()
                
        except Exception as e:
            execution_time = time.time() - start_time
            fault_report = self._handle_fault(e, execution_time)
            self.fault_reports.append(fault_report)
            
            if fault_report.severity in ["critical", "high"]:
                self.logger.error(f"Critical fault detected: {fault_report.error_message}")
            
            raise e
    
    def predict_with_fault_tolerance(self, *args, **kwargs):
        """Fault-tolerant prediction method."""
        
        def _prediction_operation():
            with self.fault_tolerance_context():
                if hasattr(self.base_model, 'predict_with_uncertainty'):
                    return self.base_model.predict_with_uncertainty(*args, **kwargs)
                else:
                    return self.base_model(*args, **kwargs)
        
        # Apply fault tolerance decorators
        operation = _prediction_operation
        
        if self.graceful_degradation:
            operation = self.graceful_degradation(operation)
        
        if self.retry_strategy:
            operation = self.retry_strategy(operation)
        
        if self.circuit_breaker:
            operation = self.circuit_breaker(operation)
        
        return operation()
    
    def _handle_fault(self, exception: Exception, execution_time: float) -> FaultReport:
        """Handle and categorize faults."""
        
        # Categorize fault severity
        if "CUDA" in str(exception) or "memory" in str(exception).lower():
            severity = "critical"
            recovery_action = "restart_with_lower_memory"
        elif "NaN" in str(exception) or "Inf" in str(exception):
            severity = "high"
            recovery_action = "reduce_learning_rate"
        elif "timeout" in str(exception).lower():
            severity = "medium"
            recovery_action = "retry_with_backoff"
        else:
            severity = "low"
            recovery_action = "log_and_continue"
        
        fault_report = FaultReport(
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            fault_type=type(exception).__name__,
            severity=severity,
            component="prediction_system",
            error_message=str(exception),
            stack_trace=traceback.format_exc(),
            recovery_action=recovery_action,
            success=False,
            performance_impact={"execution_time": execution_time}
        )
        
        return fault_report
    
    def _create_checkpoint(self):
        """Create model checkpoint for recovery."""
        if HAS_TORCH and hasattr(self.base_model, 'state_dict'):
            checkpoint_path = f"checkpoint_{int(time.time())}.pt"
            try:
                torch.save({
                    'model_state_dict': self.base_model.state_dict(),
                    'operation_count': self.operation_count,
                    'timestamp': time.time()
                }, checkpoint_path)
                self.logger.info(f"Checkpoint created: {checkpoint_path}")
            except Exception as e:
                self.logger.error(f"Failed to create checkpoint: {e}")
    
    def get_fault_summary(self) -> Dict[str, Any]:
        """Get summary of all faults encountered."""
        
        if not self.fault_reports:
            return {"total_faults": 0, "healthy": True}
        
        severity_counts = {}
        fault_types = {}
        
        for report in self.fault_reports:
            severity_counts[report.severity] = severity_counts.get(report.severity, 0) + 1
            fault_types[report.fault_type] = fault_types.get(report.fault_type, 0) + 1
        
        return {
            "total_faults": len(self.fault_reports),
            "severity_distribution": severity_counts,
            "fault_types": fault_types,
            "healthy": len(self.fault_reports) == 0 or severity_counts.get("critical", 0) == 0,
            "recent_faults": self.fault_reports[-5:] if len(self.fault_reports) > 5 else self.fault_reports
        }


def create_fault_tolerant_system(model, config: Optional[Dict[str, Any]] = None) -> FaultTolerantPNO:
    """Factory function to create fault-tolerant PNO system."""
    
    if config is None:
        config = {
            "enable_circuit_breaker": True,
            "enable_retry": True,
            "enable_degradation": True,
            "checkpoint_frequency": 100
        }
    
    return FaultTolerantPNO(model, **config)


if __name__ == "__main__":
    print("Fault Tolerance System for PNO Models")
    print("=" * 40)
    
    # Example usage with mock model
    class MockModel:
        def predict_with_uncertainty(self, x):
            # Simulate occasional failures
            if np.random.random() < 0.1:
                raise RuntimeError("Simulated model failure")
            return {"prediction": x * 0.9, "uncertainty": 0.1}
    
    # Create fault-tolerant system
    mock_model = MockModel()
    ft_system = create_fault_tolerant_system(mock_model)
    
    # Test fault tolerance
    successes = 0
    failures = 0
    
    for i in range(50):
        try:
            result = ft_system.predict_with_fault_tolerance(np.random.randn(3, 32, 32))
            successes += 1
            print(f"✓ Prediction {i+1} successful")
        except Exception as e:
            failures += 1
            print(f"✗ Prediction {i+1} failed: {e}")
    
    # Get fault summary
    fault_summary = ft_system.get_fault_summary()
    print(f"\nFault Tolerance Results:")
    print(f"- Successful predictions: {successes}")
    print(f"- Failed predictions: {failures}")
    print(f"- Total faults handled: {fault_summary['total_faults']}")
    print(f"- System healthy: {fault_summary['healthy']}")
    
    if fault_summary['fault_types']:
        print(f"- Fault types: {fault_summary['fault_types']}")