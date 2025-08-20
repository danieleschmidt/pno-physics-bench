# SECURITY NOTICE: This file has been automatically hardened for security
# - All user inputs should be validated and sanitized
# - Subprocess calls use secure alternatives
# - SQL queries use parameterized statements
# - No hardcoded secrets or credentials


"""Health checking system for production deployments."""

import torch
import psutil
import time
import logging
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import numpy as np


logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health check status levels."""
    HEALTHY = "healthy"
    WARNING = "warning" 
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class HealthCheckResult:
    """Result of a health check."""
    name: str
    status: HealthStatus
    message: str
    metrics: Dict[str, Any]
    timestamp: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'status': self.status.value,
            'message': self.message,
            'metrics': self.metrics,
            'timestamp': self.timestamp
        }


class HealthCheck:
    """Base class for health checks."""
    
    def __init__(self, name: str, warning_threshold: float = 0.8, critical_threshold: float = 0.9):
        self.name = name
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
    
    def check(self) -> HealthCheckResult:
        """Perform health check and return result."""
        try:
            metrics = self._collect_metrics()
            status, message = self._evaluate_health(metrics)
            
            return HealthCheckResult(
                name=self.name,
                status=status,
                message=message,
                metrics=metrics,
                timestamp=time.time()
            )
        except Exception as e:
            logger.error(f"Health check {self.name} failed: {e}")
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNKNOWN,
                message=f"Health check failed: {e}",
                metrics={},
                timestamp=time.time()
            )
    
    def _collect_metrics(self) -> Dict[str, Any]:
        """Collect metrics for health evaluation."""
        raise NotImplementedError
    
    def _evaluate_health(self, metrics: Dict[str, Any]) -> tuple[HealthStatus, str]:
        """Evaluate health based on metrics."""
        raise NotImplementedError


class CPUHealthCheck(HealthCheck):
    """Check CPU usage and performance."""
    
    def __init__(self, **kwargs):
        super().__init__("cpu_health", **kwargs)
    
    def _collect_metrics(self) -> Dict[str, Any]:
        # Get CPU usage over 1 second
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # Get load average (Unix systems)
        try:
            load_avg = psutil.getloadavg()
        except AttributeError:
            load_avg = (0, 0, 0)  # Windows doesn't have load average
        
        return {
            'cpu_percent': cpu_percent,
            'load_avg_1m': load_avg[0],
            'load_avg_5m': load_avg[1],
            'load_avg_15m': load_avg[2],
            'cpu_count': psutil.cpu_count()
        }
    
    def _evaluate_health(self, metrics: Dict[str, Any]) -> tuple[HealthStatus, str]:
        cpu_percent = metrics['cpu_percent']
        
        if cpu_percent > self.critical_threshold * 100:
            return HealthStatus.CRITICAL, f"High CPU usage: {cpu_percent:.1f}%"
        elif cpu_percent > self.warning_threshold * 100:
            return HealthStatus.WARNING, f"Elevated CPU usage: {cpu_percent:.1f}%"
        else:
            return HealthStatus.HEALTHY, f"CPU usage normal: {cpu_percent:.1f}%"


class MemoryHealthCheck(HealthCheck):
    """Check memory usage."""
    
    def __init__(self, **kwargs):
        super().__init__("memory_health", **kwargs)
    
    def _collect_metrics(self) -> Dict[str, Any]:
        memory = psutil.virtual_memory()
        
        return {
            'memory_percent': memory.percent,
            'memory_available': memory.available,
            'memory_total': memory.total,
            'memory_used': memory.used
        }
    
    def _evaluate_health(self, metrics: Dict[str, Any]) -> tuple[HealthStatus, str]:
        memory_percent = metrics['memory_percent']
        
        if memory_percent > self.critical_threshold * 100:
            return HealthStatus.CRITICAL, f"High memory usage: {memory_percent:.1f}%"
        elif memory_percent > self.warning_threshold * 100:
            return HealthStatus.WARNING, f"Elevated memory usage: {memory_percent:.1f}%"
        else:
            return HealthStatus.HEALTHY, f"Memory usage normal: {memory_percent:.1f}%"


class GPUHealthCheck(HealthCheck):
    """Check GPU health and memory usage."""
    
    def __init__(self, **kwargs):
        super().__init__("gpu_health", **kwargs)
    
    def _collect_metrics(self) -> Dict[str, Any]:
        if not torch.cuda.is_available():
            return {'cuda_available': False}
        
        metrics = {'cuda_available': True, 'devices': []}
        
        for i in range(torch.cuda.device_count()):
            device_metrics = {
                'device_id': i,
                'memory_allocated': torch.cuda.memory_allocated(i),
                'memory_reserved': torch.cuda.memory_reserved(i),
                'memory_total': torch.cuda.get_device_properties(i).total_memory,
            }
            
            # Calculate memory usage percentage
            device_metrics['memory_percent'] = (
                device_metrics['memory_allocated'] / device_metrics['memory_total'] * 100
            )
            
            metrics['devices'].append(device_metrics)
        
        return metrics
    
    def _evaluate_health(self, metrics: Dict[str, Any]) -> tuple[HealthStatus, str]:
        if not metrics['cuda_available']:
            return HealthStatus.HEALTHY, "No GPU available"
        
        max_memory_percent = 0
        for device in metrics['devices']:
            max_memory_percent = max(max_memory_percent, device['memory_percent'])
        
        if max_memory_percent > self.critical_threshold * 100:
            return HealthStatus.CRITICAL, f"High GPU memory usage: {max_memory_percent:.1f}%"
        elif max_memory_percent > self.warning_threshold * 100:
            return HealthStatus.WARNING, f"Elevated GPU memory usage: {max_memory_percent:.1f}%"
        else:
            return HealthStatus.HEALTHY, f"GPU memory usage normal: {max_memory_percent:.1f}%"


class ModelHealthMonitor:
    """Monitor model health during training and inference."""
    
    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.prediction_times = []
        self.loss_history = []
        self.gradient_norms = []
        
    def record_prediction_time(self, prediction_time: float):
        """Record prediction timing."""
        self.prediction_times.append(prediction_time)
        
        # Keep only recent history
        if len(self.prediction_times) > 1000:
            self.prediction_times = self.prediction_times[-1000:]
    
    def record_loss(self, loss: float):
        """Record training loss."""
        self.loss_history.append(loss)
        
        # Keep only recent history
        if len(self.loss_history) > 1000:
            self.loss_history = self.loss_history[-1000:]
    
    def record_gradient_norm(self, grad_norm: float):
        """Record gradient norm."""
        self.gradient_norms.append(grad_norm)
        
        # Keep only recent history
        if len(self.gradient_norms) > 1000:
            self.gradient_norms = self.gradient_norms[-1000:]
    
    def check_model_health(self) -> HealthCheckResult:
        """Comprehensive model health check."""
        metrics = {}
        issues = []
        
        # Check prediction times
        if self.prediction_times:
            recent_times = self.prediction_times[-100:]  # Last 100 predictions
            metrics['avg_prediction_time'] = np.mean(recent_times)
            metrics['prediction_time_std'] = np.std(recent_times)
            
            if metrics['avg_prediction_time'] > 5.0:  # 5 seconds per prediction
                issues.append("Slow prediction times")
        
        # Check loss trends
        if len(self.loss_history) > 50:
            recent_losses = self.loss_history[-50:]
            metrics['recent_loss_mean'] = np.mean(recent_losses)
            metrics['loss_trend'] = np.polyfit(range(len(recent_losses)), recent_losses, 1)[0]
            
            # Check for loss explosion
            if metrics['recent_loss_mean'] > 100:
                issues.append("Loss explosion detected")
            
            # Check for loss not decreasing
            if metrics['loss_trend'] > 0.01:
                issues.append("Loss not decreasing")
        
        # Check gradient norms
        if self.gradient_norms:
            recent_grads = self.gradient_norms[-100:]
            metrics['avg_gradient_norm'] = np.mean(recent_grads)
            
            if metrics['avg_gradient_norm'] > 10.0:
                issues.append("Large gradient norms")
            elif metrics['avg_gradient_norm'] < 1e-6:
                issues.append("Very small gradient norms (vanishing gradients)")
        
        # Check for NaN parameters
        nan_params = 0
        total_params = 0
        for param in self.model.parameters():
            if torch.isnan(param).any():
                nan_params += torch.isnan(param).sum().item()
            total_params += param.numel()
        
        metrics['nan_parameters'] = nan_params
        metrics['total_parameters'] = total_params
        
        if nan_params > 0:
            issues.append(f"NaN parameters detected: {nan_params}/{total_params}")
        
        # Determine overall health
        if any("explosion" in issue or "NaN" in issue for issue in issues):
            status = HealthStatus.CRITICAL
        elif len(issues) > 2:
            status = HealthStatus.WARNING
        elif len(issues) > 0:
            status = HealthStatus.WARNING
        else:
            status = HealthStatus.HEALTHY
        
        message = "Model healthy" if not issues else "; ".join(issues)
        
        return HealthCheckResult(
            name="model_health",
            status=status,
            message=message,
            metrics=metrics,
            timestamp=time.time()
        )


class HealthChecker:
    """Central health checking coordinator."""
    
    def __init__(self):
        self.checks: List[HealthCheck] = []
        self.model_monitors: Dict[str, ModelHealthMonitor] = {}
        self.check_history: List[Dict[str, HealthCheckResult]] = []
    
    def add_check(self, check: HealthCheck):
        """Add a health check."""
        self.checks.append(check)
        logger.info(f"Added health check: {check.name}")
    
    def add_model_monitor(self, name: str, model: torch.nn.Module):
        """Add model health monitor."""
        self.model_monitors[name] = ModelHealthMonitor(model)
        logger.info(f"Added model monitor: {name}")
    
    def run_all_checks(self) -> Dict[str, HealthCheckResult]:
        """Run all registered health checks."""
        results = {}
        
        # Run system health checks
        for check in self.checks:
            result = check.check()
            results[check.name] = result
        
        # Run model health checks
        for name, monitor in self.model_monitors.items():
            result = monitor.check_model_health()
            results[f"model_{name}"] = result
        
        # Store in history
        self.check_history.append(results)
        
        # Keep only recent history
        if len(self.check_history) > 100:
            self.check_history = self.check_history[-100:]
        
        return results
    
    def get_overall_health(self) -> HealthStatus:
        """Get overall system health."""
        results = self.run_all_checks()
        
        # Determine worst status
        worst_status = HealthStatus.HEALTHY
        for result in results.values():
            if result.status == HealthStatus.CRITICAL:
                return HealthStatus.CRITICAL
            elif result.status == HealthStatus.WARNING:
                worst_status = HealthStatus.WARNING
            elif result.status == HealthStatus.UNKNOWN and worst_status == HealthStatus.HEALTHY:
                worst_status = HealthStatus.UNKNOWN
        
        return worst_status
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get comprehensive health summary."""
        results = self.run_all_checks()
        
        summary = {
            'overall_status': self.get_overall_health().value,
            'timestamp': time.time(),
            'checks': {name: result.to_dict() for name, result in results.items()},
            'num_healthy': sum(1 for r in results.values() if r.status == HealthStatus.HEALTHY),
            'num_warning': sum(1 for r in results.values() if r.status == HealthStatus.WARNING),
            'num_critical': sum(1 for r in results.values() if r.status == HealthStatus.CRITICAL),
            'num_unknown': sum(1 for r in results.values() if r.status == HealthStatus.UNKNOWN),
        }
        
        return summary
    
    def setup_default_checks(self):
        """Setup default system health checks."""
        self.add_check(CPUHealthCheck())
        self.add_check(MemoryHealthCheck())
        self.add_check(GPUHealthCheck())
        logger.info("Default health checks configured")