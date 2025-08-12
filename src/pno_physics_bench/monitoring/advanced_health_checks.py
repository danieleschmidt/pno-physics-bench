"""Advanced health monitoring and diagnostics for PNO systems."""

import os
import time
import threading
import json
import subprocess
import platform
import contextlib
from typing import Dict, Any, List, Optional, Callable, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
import logging
import warnings

logger = logging.getLogger(__name__)

# Optional dependencies
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    warnings.warn("PyTorch not available - GPU monitoring disabled")

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    warnings.warn("psutil not available - system monitoring limited")


class HealthStatus(Enum):
    """Health check status levels."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class ComponentType(Enum):
    """Types of system components."""
    SYSTEM = "system"
    MEMORY = "memory"
    DISK = "disk"
    NETWORK = "network"
    GPU = "gpu"
    MODEL = "model"
    DATABASE = "database"
    SERVICE = "service"


@dataclass
class HealthCheckResult:
    """Result of a health check."""
    component: str
    component_type: ComponentType
    status: HealthStatus
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    metrics: Dict[str, Any] = field(default_factory=dict)
    suggestions: List[str] = field(default_factory=list)
    execution_time: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'component': self.component,
            'component_type': self.component_type.value,
            'status': self.status.value,
            'message': self.message,
            'timestamp': self.timestamp.isoformat(),
            'metrics': self.metrics,
            'suggestions': self.suggestions,
            'execution_time': self.execution_time
        }


class BaseHealthCheck(ABC):
    """Base class for health checks."""
    
    def __init__(
        self,
        name: str,
        component_type: ComponentType,
        timeout: float = 30.0,
        enabled: bool = True
    ):
        """Initialize health check.
        
        Args:
            name: Check name
            component_type: Type of component
            timeout: Check timeout in seconds
            enabled: Whether check is enabled
        """
        self.name = name
        self.component_type = component_type
        self.timeout = timeout
        self.enabled = enabled
        self.last_result: Optional[HealthCheckResult] = None
        self.check_count = 0
        self.failure_count = 0
    
    @abstractmethod
    def _perform_check(self) -> HealthCheckResult:
        """Perform the actual health check."""
        pass
    
    def check(self) -> HealthCheckResult:
        """Execute health check with timeout and error handling."""
        if not self.enabled:
            return HealthCheckResult(
                component=self.name,
                component_type=self.component_type,
                status=HealthStatus.UNKNOWN,
                message="Health check disabled"
            )
        
        start_time = time.time()
        self.check_count += 1
        
        try:
            # Use threading for timeout
            result_container = [None]
            exception_container = [None]
            
            def run_check():
                try:
                    result_container[0] = self._perform_check()
                except Exception as e:
                    exception_container[0] = e
            
            thread = threading.Thread(target=run_check)
            thread.daemon = True
            thread.start()
            thread.join(timeout=self.timeout)
            
            if thread.is_alive():
                # Timeout occurred
                self.failure_count += 1
                return HealthCheckResult(
                    component=self.name,
                    component_type=self.component_type,
                    status=HealthStatus.CRITICAL,
                    message=f"Health check timed out after {self.timeout}s",
                    execution_time=self.timeout
                )
            
            if exception_container[0]:
                # Exception occurred
                self.failure_count += 1
                return HealthCheckResult(
                    component=self.name,
                    component_type=self.component_type,
                    status=HealthStatus.CRITICAL,
                    message=f"Health check failed: {str(exception_container[0])}",
                    execution_time=time.time() - start_time
                )
            
            result = result_container[0]
            if result.status in [HealthStatus.WARNING, HealthStatus.CRITICAL]:
                self.failure_count += 1
            
            result.execution_time = time.time() - start_time
            self.last_result = result
            return result
            
        except Exception as e:
            self.failure_count += 1
            return HealthCheckResult(
                component=self.name,
                component_type=self.component_type,
                status=HealthStatus.CRITICAL,
                message=f"Unexpected error in health check: {str(e)}",
                execution_time=time.time() - start_time
            )
    
    @property
    def failure_rate(self) -> float:
        """Calculate failure rate."""
        if self.check_count == 0:
            return 0.0
        return self.failure_count / self.check_count


class SystemResourceCheck(BaseHealthCheck):
    """Check system resource usage."""
    
    def __init__(
        self,
        memory_threshold: float = 0.85,  # 85%
        cpu_threshold: float = 0.90,     # 90%
        disk_threshold: float = 0.90,    # 90%
        **kwargs
    ):
        """Initialize system resource check."""
        super().__init__("system_resources", ComponentType.SYSTEM, **kwargs)
        self.memory_threshold = memory_threshold
        self.cpu_threshold = cpu_threshold
        self.disk_threshold = disk_threshold
    
    def _perform_check(self) -> HealthCheckResult:
        """Check system resources."""
        issues = []
        suggestions = []
        metrics = {}
        status = HealthStatus.HEALTHY
        
        if HAS_PSUTIL:
            # Memory check
            memory = psutil.virtual_memory()
            memory_usage = memory.percent / 100.0
            metrics['memory_usage'] = memory_usage
            metrics['memory_available_gb'] = memory.available / (1024**3)
            
            if memory_usage > self.memory_threshold:
                issues.append(f"High memory usage: {memory_usage:.1%}")
                suggestions.append("Consider reducing batch size or model complexity")
                status = HealthStatus.WARNING if memory_usage < 0.95 else HealthStatus.CRITICAL
            
            # CPU check
            cpu_usage = psutil.cpu_percent(interval=1) / 100.0
            metrics['cpu_usage'] = cpu_usage
            metrics['cpu_count'] = psutil.cpu_count()
            
            if cpu_usage > self.cpu_threshold:
                issues.append(f"High CPU usage: {cpu_usage:.1%}")
                suggestions.append("Consider using GPU acceleration or reducing computational load")
                status = max(status, HealthStatus.WARNING if cpu_usage < 0.95 else HealthStatus.CRITICAL, key=lambda x: x.value)
            
            # Disk check
            disk = psutil.disk_usage('/')
            disk_usage = disk.used / disk.total
            metrics['disk_usage'] = disk_usage
            metrics['disk_free_gb'] = disk.free / (1024**3)
            
            if disk_usage > self.disk_threshold:
                issues.append(f"High disk usage: {disk_usage:.1%}")
                suggestions.append("Clean up temporary files or increase disk space")
                status = max(status, HealthStatus.WARNING if disk_usage < 0.95 else HealthStatus.CRITICAL, key=lambda x: x.value)
        else:
            # Fallback using basic OS commands
            try:
                # Get memory info from /proc/meminfo
                with open('/proc/meminfo', 'r') as f:
                    meminfo = f.read()
                
                for line in meminfo.split('\n'):
                    if 'MemTotal:' in line:
                        total_kb = int(line.split()[1])
                    elif 'MemAvailable:' in line:
                        available_kb = int(line.split()[1])
                
                memory_usage = 1 - (available_kb / total_kb)
                metrics['memory_usage'] = memory_usage
                
                if memory_usage > self.memory_threshold:
                    issues.append(f"High memory usage: {memory_usage:.1%}")
                    status = HealthStatus.WARNING
                    
            except Exception as e:
                issues.append(f"Could not check system resources: {e}")
                status = HealthStatus.WARNING
        
        message = "System resources healthy" if not issues else "; ".join(issues)
        
        return HealthCheckResult(
            component=self.name,
            component_type=self.component_type,
            status=status,
            message=message,
            metrics=metrics,
            suggestions=suggestions
        )


class GPUHealthCheck(BaseHealthCheck):
    """Check GPU status and memory."""
    
    def __init__(
        self,
        memory_threshold: float = 0.90,  # 90%
        temperature_threshold: float = 85.0,  # Celsius
        **kwargs
    ):
        """Initialize GPU health check."""
        super().__init__("gpu_status", ComponentType.GPU, **kwargs)
        self.memory_threshold = memory_threshold
        self.temperature_threshold = temperature_threshold
    
    def _perform_check(self) -> HealthCheckResult:
        """Check GPU health."""
        if not HAS_TORCH:
            return HealthCheckResult(
                component=self.name,
                component_type=self.component_type,
                status=HealthStatus.UNKNOWN,
                message="PyTorch not available - cannot check GPU"
            )
        
        if not torch.cuda.is_available():
            return HealthCheckResult(
                component=self.name,
                component_type=self.component_type,
                status=HealthStatus.WARNING,
                message="CUDA not available",
                suggestions=["Consider using CPU mode or install CUDA"]
            )
        
        issues = []
        suggestions = []
        metrics = {}
        status = HealthStatus.HEALTHY
        
        try:
            device_count = torch.cuda.device_count()
            metrics['device_count'] = device_count
            
            for i in range(device_count):
                device = f"cuda:{i}"
                
                # Memory check
                memory_reserved = torch.cuda.memory_reserved(i) / 1024**3  # GB
                memory_allocated = torch.cuda.memory_allocated(i) / 1024**3  # GB
                
                # Get total memory
                try:
                    total_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3  # GB
                    memory_usage = memory_allocated / total_memory
                    
                    metrics[f'gpu_{i}_memory_usage'] = memory_usage
                    metrics[f'gpu_{i}_memory_allocated_gb'] = memory_allocated
                    metrics[f'gpu_{i}_memory_reserved_gb'] = memory_reserved
                    metrics[f'gpu_{i}_memory_total_gb'] = total_memory
                    
                    if memory_usage > self.memory_threshold:
                        issues.append(f"GPU {i} high memory usage: {memory_usage:.1%}")
                        suggestions.append(f"Reduce batch size for GPU {i}")
                        status = max(status, HealthStatus.WARNING, key=lambda x: x.value)
                        
                except Exception as e:
                    issues.append(f"Could not get memory info for GPU {i}: {e}")
                
                # Device properties
                props = torch.cuda.get_device_properties(i)
                metrics[f'gpu_{i}_name'] = props.name
                metrics[f'gpu_{i}_capability'] = f"{props.major}.{props.minor}"
                
        except Exception as e:
            return HealthCheckResult(
                component=self.name,
                component_type=self.component_type,
                status=HealthStatus.CRITICAL,
                message=f"GPU health check failed: {str(e)}"
            )
        
        message = "GPU resources healthy" if not issues else "; ".join(issues)
        
        return HealthCheckResult(
            component=self.name,
            component_type=self.component_type,
            status=status,
            message=message,
            metrics=metrics,
            suggestions=suggestions
        )


class ModelHealthCheck(BaseHealthCheck):
    """Check model status and performance."""
    
    def __init__(
        self,
        model,
        test_input: Optional[Any] = None,
        expected_output_shape: Optional[Tuple] = None,
        performance_threshold: float = 1.0,  # seconds
        **kwargs
    ):
        """Initialize model health check.
        
        Args:
            model: Model to check
            test_input: Sample input for testing
            expected_output_shape: Expected output shape
            performance_threshold: Maximum inference time
        """
        super().__init__("model_status", ComponentType.MODEL, **kwargs)
        self.model = model
        self.test_input = test_input
        self.expected_output_shape = expected_output_shape
        self.performance_threshold = performance_threshold
    
    def _perform_check(self) -> HealthCheckResult:
        """Check model health."""
        issues = []
        suggestions = []
        metrics = {}
        status = HealthStatus.HEALTHY
        
        try:
            # Check if model is in eval mode for inference
            if hasattr(self.model, 'training'):
                metrics['model_training_mode'] = self.model.training
            
            # Count parameters
            if hasattr(self.model, 'parameters'):
                total_params = sum(p.numel() for p in self.model.parameters())
                trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
                metrics['total_parameters'] = total_params
                metrics['trainable_parameters'] = trainable_params
            
            # Test inference if test input provided
            if self.test_input is not None:
                start_time = time.time()
                
                # Set to eval mode for testing
                was_training = getattr(self.model, 'training', False)
                if hasattr(self.model, 'eval'):
                    self.model.eval()
                
                try:
                    with torch.no_grad() if HAS_TORCH else contextlib.nullcontext():
                        output = self.model(self.test_input)
                    
                    inference_time = time.time() - start_time
                    metrics['inference_time'] = inference_time
                    
                    # Check inference time
                    if inference_time > self.performance_threshold:
                        issues.append(f"Slow inference: {inference_time:.3f}s > {self.performance_threshold}s")
                        suggestions.append("Consider model optimization or hardware upgrade")
                        status = HealthStatus.WARNING
                    
                    # Check output shape
                    if self.expected_output_shape and hasattr(output, 'shape'):
                        actual_shape = tuple(output.shape)
                        metrics['output_shape'] = actual_shape
                        
                        if actual_shape != self.expected_output_shape:
                            issues.append(f"Output shape mismatch: {actual_shape} != {self.expected_output_shape}")
                            status = HealthStatus.CRITICAL
                    
                    # Check for NaN or Inf in output
                    if HAS_TORCH and isinstance(output, torch.Tensor):
                        has_nan = torch.isnan(output).any().item()
                        has_inf = torch.isinf(output).any().item()
                        
                        metrics['output_has_nan'] = has_nan
                        metrics['output_has_inf'] = has_inf
                        
                        if has_nan:
                            issues.append("Model output contains NaN values")
                            suggestions.append("Check for numerical instability in model")
                            status = HealthStatus.CRITICAL
                        
                        if has_inf:
                            issues.append("Model output contains infinite values")
                            suggestions.append("Check for numerical overflow in model")
                            status = HealthStatus.CRITICAL
                
                finally:
                    # Restore training mode
                    if hasattr(self.model, 'train') and was_training:
                        self.model.train()
            
        except Exception as e:
            return HealthCheckResult(
                component=self.name,
                component_type=self.component_type,
                status=HealthStatus.CRITICAL,
                message=f"Model health check failed: {str(e)}"
            )
        
        message = "Model healthy" if not issues else "; ".join(issues)
        
        return HealthCheckResult(
            component=self.name,
            component_type=self.component_type,
            status=status,
            message=message,
            metrics=metrics,
            suggestions=suggestions
        )


class AdvancedHealthMonitor:
    """Advanced health monitoring system."""
    
    def __init__(
        self,
        check_interval: float = 60.0,  # seconds
        alert_callback: Optional[Callable[[List[HealthCheckResult]], None]] = None
    ):
        """Initialize health monitor.
        
        Args:
            check_interval: Interval between health checks
            alert_callback: Callback for health alerts
        """
        self.check_interval = check_interval
        self.alert_callback = alert_callback
        
        # Health checks
        self.health_checks: Dict[str, BaseHealthCheck] = {}
        
        # Monitoring state
        self.monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.results_history: List[Dict[str, HealthCheckResult]] = []
        self.max_history = 1000
        
        # Add default health checks
        self._add_default_checks()
    
    def _add_default_checks(self):
        """Add default health checks."""
        # System resources
        self.add_health_check(SystemResourceCheck())
        
        # GPU check (if available)
        if HAS_TORCH:
            self.add_health_check(GPUHealthCheck())
    
    def add_health_check(self, health_check: BaseHealthCheck):
        """Add a health check."""
        self.health_checks[health_check.name] = health_check
        logger.info(f"Added health check: {health_check.name}")
    
    def remove_health_check(self, name: str):
        """Remove a health check."""
        if name in self.health_checks:
            del self.health_checks[name]
            logger.info(f"Removed health check: {name}")
    
    def run_checks(self) -> Dict[str, HealthCheckResult]:
        """Run all health checks."""
        results = {}
        
        for name, check in self.health_checks.items():
            try:
                result = check.check()
                results[name] = result
            except Exception as e:
                results[name] = HealthCheckResult(
                    component=name,
                    component_type=ComponentType.SYSTEM,
                    status=HealthStatus.CRITICAL,
                    message=f"Health check error: {str(e)}"
                )
        
        # Store in history
        self.results_history.append(results)
        if len(self.results_history) > self.max_history:
            self.results_history.pop(0)
        
        # Check for alerts
        self._check_alerts(results)
        
        return results
    
    def _check_alerts(self, results: Dict[str, HealthCheckResult]):
        """Check for alert conditions."""
        alert_results = []
        
        for result in results.values():
            if result.status in [HealthStatus.CRITICAL, HealthStatus.WARNING]:
                alert_results.append(result)
        
        if alert_results and self.alert_callback:
            try:
                self.alert_callback(alert_results)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")
    
    def start_monitoring(self):
        """Start continuous health monitoring."""
        if self.monitoring:
            logger.warning("Health monitoring already started")
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("Started health monitoring")
    
    def stop_monitoring(self):
        """Stop health monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        logger.info("Stopped health monitoring")
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.monitoring:
            try:
                self.run_checks()
                time.sleep(self.check_interval)
            except Exception as e:
                logger.error(f"Error in health monitoring loop: {e}")
                time.sleep(self.check_interval)
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get health summary."""
        if not self.results_history:
            return {"status": "no_data", "checks": []}
        
        latest_results = self.results_history[-1]
        
        # Overall status
        overall_status = HealthStatus.HEALTHY
        for result in latest_results.values():
            if result.status == HealthStatus.CRITICAL:
                overall_status = HealthStatus.CRITICAL
                break
            elif result.status == HealthStatus.WARNING:
                overall_status = HealthStatus.WARNING
        
        # Check statistics
        check_stats = {}
        for name, check in self.health_checks.items():
            check_stats[name] = {
                'total_checks': check.check_count,
                'failures': check.failure_count,
                'failure_rate': check.failure_rate,
                'enabled': check.enabled,
                'last_status': latest_results.get(name, {}).status.value if name in latest_results else 'unknown'
            }
        
        return {
            'overall_status': overall_status.value,
            'timestamp': datetime.now().isoformat(),
            'checks_count': len(latest_results),
            'monitoring_active': self.monitoring,
            'check_statistics': check_stats,
            'latest_results': {name: result.to_dict() for name, result in latest_results.items()}
        }
    
    def export_health_report(self, file_path: str) -> bool:
        """Export health report to file."""
        try:
            report = {
                'metadata': {
                    'export_time': datetime.now().isoformat(),
                    'monitoring_period': f"{len(self.results_history)} checks",
                    'check_interval': self.check_interval
                },
                'summary': self.get_health_summary(),
                'history': [
                    {name: result.to_dict() for name, result in results.items()}
                    for results in self.results_history
                ]
            }
            
            with open(file_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            logger.info(f"Exported health report to {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export health report: {e}")
            return False


# Helper function to create a comprehensive health monitor
def create_comprehensive_health_monitor(
    model=None,
    test_input=None,
    **kwargs
) -> AdvancedHealthMonitor:
    """Create a comprehensive health monitor with common checks.
    
    Args:
        model: Model to monitor
        test_input: Test input for model checks
        **kwargs: Additional arguments for AdvancedHealthMonitor
        
    Returns:
        Configured AdvancedHealthMonitor
    """
    monitor = AdvancedHealthMonitor(**kwargs)
    
    # Add model check if model provided
    if model is not None:
        model_check = ModelHealthCheck(
            model=model,
            test_input=test_input
        )
        monitor.add_health_check(model_check)
    
    return monitor