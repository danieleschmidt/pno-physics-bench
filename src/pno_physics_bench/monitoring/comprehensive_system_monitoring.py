"""Comprehensive System Monitoring for Probabilistic Neural Operators.

This module provides real-time monitoring, alerting, and performance tracking
for PNO systems including uncertainty quality, computational efficiency, and
system health.
"""

import torch
import torch.nn as nn
import psutil
import time
import threading
import logging
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
from collections import deque, defaultdict
from enum import Enum
import json
import numpy as np
from abc import ABC, abstractmethod


class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class SystemMetrics:
    """System-level performance metrics."""
    timestamp: float
    cpu_usage: float
    memory_usage_gb: float
    memory_usage_percent: float
    gpu_usage: Optional[float] = None
    gpu_memory_gb: Optional[float] = None
    gpu_temperature: Optional[float] = None
    disk_io_read_mb: Optional[float] = None
    disk_io_write_mb: Optional[float] = None


@dataclass
class ModelMetrics:
    """Model-specific performance metrics."""
    timestamp: float
    inference_time_ms: float
    throughput_samples_per_sec: float
    prediction_accuracy: Optional[float] = None
    uncertainty_quality: Optional[float] = None
    calibration_error: Optional[float] = None
    memory_usage_mb: float = 0.0
    gradient_norm: Optional[float] = None
    loss_value: Optional[float] = None


@dataclass
class UncertaintyMetrics:
    """Uncertainty-specific quality metrics."""
    timestamp: float
    mean_uncertainty: float
    uncertainty_std: float
    calibration_error: float
    coverage_90: float
    coverage_95: float
    uncertainty_correlation: float
    prediction_interval_width: float
    reliability_score: float


@dataclass
class Alert:
    """System alert information."""
    timestamp: float
    level: AlertLevel
    source: str
    message: str
    metrics: Dict[str, Any] = field(default_factory=dict)
    acknowledged: bool = False
    resolved: bool = False


class MetricsCollector(ABC):
    """Abstract base class for metrics collection."""
    
    @abstractmethod
    def collect(self) -> Dict[str, Any]:
        """Collect metrics."""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Get collector name."""
        pass


class SystemMetricsCollector(MetricsCollector):
    """Collect system-level metrics."""
    
    def __init__(self, collect_gpu_metrics: bool = True):
        self.collect_gpu_metrics = collect_gpu_metrics
        
        # Initialize GPU monitoring if available
        self.gpu_available = False
        if collect_gpu_metrics:
            try:
                import pynvml
                pynvml.nvmlInit()
                self.gpu_available = True
                self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # First GPU
            except Exception:
                self.gpu_available = False
    
    def collect(self) -> SystemMetrics:
        """Collect current system metrics."""
        
        # CPU and memory
        cpu_usage = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        memory_usage_gb = memory.used / (1024**3)
        memory_usage_percent = memory.percent
        
        # Disk I/O
        disk_io = psutil.disk_io_counters()
        disk_io_read_mb = disk_io.read_bytes / (1024**2) if disk_io else None
        disk_io_write_mb = disk_io.write_bytes / (1024**2) if disk_io else None
        
        # GPU metrics
        gpu_usage = None
        gpu_memory_gb = None
        gpu_temperature = None
        
        if self.gpu_available:
            try:
                import pynvml
                
                # GPU utilization
                gpu_util = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
                gpu_usage = gpu_util.gpu
                
                # GPU memory
                gpu_mem = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
                gpu_memory_gb = gpu_mem.used / (1024**3)
                
                # GPU temperature
                gpu_temperature = pynvml.nvmlDeviceGetTemperature(
                    self.gpu_handle, pynvml.NVML_TEMPERATURE_GPU
                )
                
            except Exception as e:
                logging.warning(f"Failed to collect GPU metrics: {e}")
        
        return SystemMetrics(
            timestamp=time.time(),
            cpu_usage=cpu_usage,
            memory_usage_gb=memory_usage_gb,
            memory_usage_percent=memory_usage_percent,
            gpu_usage=gpu_usage,
            gpu_memory_gb=gpu_memory_gb,
            gpu_temperature=gpu_temperature,
            disk_io_read_mb=disk_io_read_mb,
            disk_io_write_mb=disk_io_write_mb
        )
    
    def get_name(self) -> str:
        return "SystemMetricsCollector"


class ModelPerformanceCollector(MetricsCollector):
    """Collect model performance metrics."""
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.last_inference_time = None
        self.inference_count = 0
        self.throughput_window = deque(maxlen=100)
        
    def collect_inference_metrics(
        self,
        inputs: torch.Tensor,
        predictions: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        inference_time: Optional[float] = None
    ) -> ModelMetrics:
        """Collect metrics for a single inference."""
        
        batch_size = inputs.size(0)
        current_time = time.time()
        
        # Inference time
        if inference_time is None:
            inference_time = 0.0
        
        inference_time_ms = inference_time * 1000
        
        # Throughput calculation
        if self.last_inference_time is not None:
            time_diff = current_time - self.last_inference_time
            throughput = batch_size / max(time_diff, 1e-6)
            self.throughput_window.append(throughput)
            avg_throughput = np.mean(self.throughput_window)
        else:
            avg_throughput = 0.0
        
        self.last_inference_time = current_time
        self.inference_count += 1
        
        # Accuracy (if targets available)
        accuracy = None
        if targets is not None:
            with torch.no_grad():
                if len(predictions.shape) > 1:
                    mse = torch.mean((predictions - targets) ** 2)
                    accuracy = 1.0 / (1.0 + mse.item())  # Convert MSE to accuracy-like metric
                else:
                    accuracy = 1.0 - torch.mean(torch.abs(predictions - targets)).item()
        
        # Memory usage
        memory_usage_mb = 0.0
        if torch.cuda.is_available():
            memory_usage_mb = torch.cuda.memory_allocated() / (1024**2)
        
        # Gradient norm (if in training mode)
        gradient_norm = None
        if self.model.training:
            total_norm = 0.0
            param_count = 0
            for param in self.model.parameters():
                if param.grad is not None:
                    param_norm = param.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
                    param_count += 1
            
            if param_count > 0:
                gradient_norm = (total_norm ** 0.5)
        
        return ModelMetrics(
            timestamp=current_time,
            inference_time_ms=inference_time_ms,
            throughput_samples_per_sec=avg_throughput,
            prediction_accuracy=accuracy,
            memory_usage_mb=memory_usage_mb,
            gradient_norm=gradient_norm
        )
    
    def collect(self) -> Dict[str, Any]:
        """Collect general model metrics."""
        return {
            'inference_count': self.inference_count,
            'average_throughput': np.mean(self.throughput_window) if self.throughput_window else 0.0,
            'model_parameters': sum(p.numel() for p in self.model.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        }
    
    def get_name(self) -> str:
        return "ModelPerformanceCollector"


class UncertaintyQualityCollector(MetricsCollector):
    """Collect uncertainty quality metrics."""
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.prediction_history = deque(maxlen=window_size)
        self.uncertainty_history = deque(maxlen=window_size)
        self.target_history = deque(maxlen=window_size)
    
    def add_sample(
        self,
        predictions: torch.Tensor,
        uncertainties: torch.Tensor,
        targets: torch.Tensor
    ):
        """Add a sample for uncertainty quality assessment."""
        
        # Store flattened tensors for easier analysis
        self.prediction_history.append(predictions.flatten().detach().cpu())
        self.uncertainty_history.append(uncertainties.flatten().detach().cpu())
        self.target_history.append(targets.flatten().detach().cpu())
    
    def collect(self) -> UncertaintyMetrics:
        """Compute uncertainty quality metrics."""
        
        if len(self.prediction_history) == 0:
            return UncertaintyMetrics(
                timestamp=time.time(),
                mean_uncertainty=0.0,
                uncertainty_std=0.0,
                calibration_error=0.0,
                coverage_90=0.0,
                coverage_95=0.0,
                uncertainty_correlation=0.0,
                prediction_interval_width=0.0,
                reliability_score=0.0
            )
        
        # Concatenate all samples
        predictions = torch.cat(list(self.prediction_history))
        uncertainties = torch.cat(list(self.uncertainty_history))
        targets = torch.cat(list(self.target_history))
        
        # Basic uncertainty statistics
        mean_uncertainty = uncertainties.mean().item()
        uncertainty_std = uncertainties.std().item()
        
        # Prediction errors
        errors = torch.abs(predictions - targets)
        
        # Calibration error (simplified ECE)
        calibration_error = self._compute_calibration_error(errors, uncertainties)
        
        # Coverage at different confidence levels
        coverage_90 = self._compute_coverage(errors, uncertainties, confidence=0.9)
        coverage_95 = self._compute_coverage(errors, uncertainties, confidence=0.95)
        
        # Correlation between uncertainty and error
        uncertainty_correlation = self._compute_correlation(errors, uncertainties)
        
        # Average prediction interval width
        prediction_interval_width = uncertainties.mean().item() * 1.96  # 95% interval
        
        # Reliability score (combination of calibration and correlation)
        reliability_score = self._compute_reliability_score(
            calibration_error, uncertainty_correlation, coverage_95
        )
        
        return UncertaintyMetrics(
            timestamp=time.time(),
            mean_uncertainty=mean_uncertainty,
            uncertainty_std=uncertainty_std,
            calibration_error=calibration_error,
            coverage_90=coverage_90,
            coverage_95=coverage_95,
            uncertainty_correlation=uncertainty_correlation,
            prediction_interval_width=prediction_interval_width,
            reliability_score=reliability_score
        )
    
    def _compute_calibration_error(
        self,
        errors: torch.Tensor,
        uncertainties: torch.Tensor,
        num_bins: int = 10
    ) -> float:
        """Compute Expected Calibration Error."""
        
        # Sort by uncertainty
        sorted_indices = torch.argsort(uncertainties)
        sorted_errors = errors[sorted_indices]
        sorted_uncertainties = uncertainties[sorted_indices]
        
        n_samples = len(errors)
        bin_size = n_samples // num_bins
        
        ece = 0.0
        for i in range(num_bins):
            start_idx = i * bin_size
            end_idx = (i + 1) * bin_size if i < num_bins - 1 else n_samples
            
            if end_idx > start_idx:
                bin_errors = sorted_errors[start_idx:end_idx]
                bin_uncertainties = sorted_uncertainties[start_idx:end_idx]
                
                # Expected error in this bin
                expected_error = bin_uncertainties.mean().item()
                
                # Actual error in this bin
                actual_error = bin_errors.mean().item()
                
                # Weight by bin size
                weight = (end_idx - start_idx) / n_samples
                
                ece += weight * abs(expected_error - actual_error)
        
        return ece
    
    def _compute_coverage(
        self,
        errors: torch.Tensor,
        uncertainties: torch.Tensor,
        confidence: float
    ) -> float:
        """Compute coverage at given confidence level."""
        
        # For Gaussian assumption, confidence interval is ±z*σ
        z_score = {0.9: 1.645, 0.95: 1.96, 0.99: 2.576}.get(confidence, 1.96)
        
        interval_width = z_score * uncertainties
        within_interval = errors <= interval_width
        
        return within_interval.float().mean().item()
    
    def _compute_correlation(
        self,
        errors: torch.Tensor,
        uncertainties: torch.Tensor
    ) -> float:
        """Compute correlation between errors and uncertainties."""
        
        if len(errors) < 2:
            return 0.0
        
        # Pearson correlation
        error_mean = errors.mean()
        uncertainty_mean = uncertainties.mean()
        
        numerator = ((errors - error_mean) * (uncertainties - uncertainty_mean)).sum()
        
        error_std = ((errors - error_mean) ** 2).sum().sqrt()
        uncertainty_std = ((uncertainties - uncertainty_mean) ** 2).sum().sqrt()
        
        denominator = error_std * uncertainty_std
        
        if denominator == 0:
            return 0.0
        
        correlation = (numerator / denominator).item()
        return max(-1.0, min(1.0, correlation))  # Clamp to [-1, 1]
    
    def _compute_reliability_score(
        self,
        calibration_error: float,
        correlation: float,
        coverage: float
    ) -> float:
        """Compute overall reliability score."""
        
        # Combine different aspects of uncertainty quality
        # Good reliability = low calibration error + high correlation + good coverage
        
        calibration_score = max(0.0, 1.0 - calibration_error)
        correlation_score = max(0.0, correlation)  # Positive correlation is good
        coverage_score = 1.0 - abs(coverage - 0.95)  # Coverage should be close to 95%
        
        # Weighted combination
        reliability_score = (
            0.4 * calibration_score +
            0.3 * correlation_score +
            0.3 * coverage_score
        )
        
        return max(0.0, min(1.0, reliability_score))
    
    def get_name(self) -> str:
        return "UncertaintyQualityCollector"


class AlertManager:
    """Manage system alerts and notifications."""
    
    def __init__(
        self,
        alert_thresholds: Optional[Dict[str, Any]] = None,
        max_alerts: int = 1000
    ):
        self.alert_thresholds = alert_thresholds or self._default_thresholds()
        self.alerts = deque(maxlen=max_alerts)
        self.alert_handlers = []
        self.suppressed_alerts = set()
        
    def _default_thresholds(self) -> Dict[str, Any]:
        """Default alert thresholds."""
        return {
            'cpu_usage': {'warning': 80.0, 'critical': 95.0},
            'memory_usage': {'warning': 80.0, 'critical': 95.0},
            'gpu_usage': {'warning': 90.0, 'critical': 98.0},
            'gpu_temperature': {'warning': 80.0, 'critical': 90.0},
            'inference_time': {'warning': 1000.0, 'critical': 5000.0},  # ms
            'calibration_error': {'warning': 0.1, 'critical': 0.2},
            'uncertainty_correlation': {'warning': 0.3, 'critical': 0.1}  # Lower is worse
        }
    
    def check_system_metrics(self, metrics: SystemMetrics):
        """Check system metrics against thresholds."""
        
        # CPU usage
        self._check_threshold(
            'cpu_usage', metrics.cpu_usage, metrics.timestamp,
            f"CPU usage: {metrics.cpu_usage:.1f}%"
        )
        
        # Memory usage
        self._check_threshold(
            'memory_usage', metrics.memory_usage_percent, metrics.timestamp,
            f"Memory usage: {metrics.memory_usage_percent:.1f}% ({metrics.memory_usage_gb:.2f}GB)"
        )
        
        # GPU metrics
        if metrics.gpu_usage is not None:
            self._check_threshold(
                'gpu_usage', metrics.gpu_usage, metrics.timestamp,
                f"GPU usage: {metrics.gpu_usage:.1f}%"
            )
        
        if metrics.gpu_temperature is not None:
            self._check_threshold(
                'gpu_temperature', metrics.gpu_temperature, metrics.timestamp,
                f"GPU temperature: {metrics.gpu_temperature:.1f}°C"
            )
    
    def check_model_metrics(self, metrics: ModelMetrics):
        """Check model metrics against thresholds."""
        
        # Inference time
        self._check_threshold(
            'inference_time', metrics.inference_time_ms, metrics.timestamp,
            f"Inference time: {metrics.inference_time_ms:.1f}ms"
        )
    
    def check_uncertainty_metrics(self, metrics: UncertaintyMetrics):
        """Check uncertainty quality metrics."""
        
        # Calibration error
        self._check_threshold(
            'calibration_error', metrics.calibration_error, metrics.timestamp,
            f"Calibration error: {metrics.calibration_error:.3f}"
        )
        
        # Uncertainty correlation (lower is worse)
        if 'uncertainty_correlation' in self.alert_thresholds:
            thresholds = self.alert_thresholds['uncertainty_correlation']
            
            if metrics.uncertainty_correlation < thresholds.get('critical', 0.1):
                self._create_alert(
                    AlertLevel.CRITICAL, 'uncertainty_correlation',
                    f"Poor uncertainty correlation: {metrics.uncertainty_correlation:.3f}",
                    metrics.timestamp
                )
            elif metrics.uncertainty_correlation < thresholds.get('warning', 0.3):
                self._create_alert(
                    AlertLevel.WARNING, 'uncertainty_correlation',
                    f"Low uncertainty correlation: {metrics.uncertainty_correlation:.3f}",
                    metrics.timestamp
                )
    
    def _check_threshold(
        self,
        metric_name: str,
        value: float,
        timestamp: float,
        message: str
    ):
        """Check a single metric against thresholds."""
        
        if metric_name not in self.alert_thresholds:
            return
        
        thresholds = self.alert_thresholds[metric_name]
        
        if value >= thresholds.get('critical', float('inf')):
            self._create_alert(AlertLevel.CRITICAL, metric_name, message, timestamp)
        elif value >= thresholds.get('warning', float('inf')):
            self._create_alert(AlertLevel.WARNING, metric_name, message, timestamp)
    
    def _create_alert(
        self,
        level: AlertLevel,
        source: str,
        message: str,
        timestamp: float,
        metrics: Optional[Dict] = None
    ):
        """Create a new alert."""
        
        # Check if this alert type is suppressed
        alert_key = f"{source}_{level.value}"
        if alert_key in self.suppressed_alerts:
            return
        
        alert = Alert(
            timestamp=timestamp,
            level=level,
            source=source,
            message=message,
            metrics=metrics or {}
        )
        
        self.alerts.append(alert)
        
        # Notify handlers
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                logging.error(f"Alert handler failed: {e}")
    
    def add_alert_handler(self, handler: Callable[[Alert], None]):
        """Add an alert handler function."""
        self.alert_handlers.append(handler)
    
    def suppress_alert_type(self, alert_type: str):
        """Suppress alerts of a specific type."""
        self.suppressed_alerts.add(alert_type)
    
    def unsuppress_alert_type(self, alert_type: str):
        """Remove suppression for alert type."""
        self.suppressed_alerts.discard(alert_type)
    
    def get_active_alerts(self, since: Optional[float] = None) -> List[Alert]:
        """Get active (unresolved) alerts."""
        alerts = [alert for alert in self.alerts if not alert.resolved]
        
        if since is not None:
            alerts = [alert for alert in alerts if alert.timestamp >= since]
        
        return alerts
    
    def acknowledge_alert(self, alert_index: int):
        """Acknowledge an alert."""
        if 0 <= alert_index < len(self.alerts):
            self.alerts[alert_index].acknowledged = True
    
    def resolve_alert(self, alert_index: int):
        """Resolve an alert."""
        if 0 <= alert_index < len(self.alerts):
            self.alerts[alert_index].resolved = True


class ComprehensiveMonitor:
    """Comprehensive monitoring system for PNO operations."""
    
    def __init__(
        self,
        model: Optional[nn.Module] = None,
        collection_interval: float = 30.0,  # seconds
        enable_real_time: bool = True,
        alert_thresholds: Optional[Dict] = None
    ):
        self.model = model
        self.collection_interval = collection_interval
        self.enable_real_time = enable_real_time
        
        # Initialize collectors
        self.collectors = []
        
        # System metrics
        self.system_collector = SystemMetricsCollector()
        self.collectors.append(self.system_collector)
        
        # Model metrics
        if model is not None:
            self.model_collector = ModelPerformanceCollector(model)
            self.collectors.append(self.model_collector)
        
        # Uncertainty metrics
        self.uncertainty_collector = UncertaintyQualityCollector()
        self.collectors.append(self.uncertainty_collector)
        
        # Alert management
        self.alert_manager = AlertManager(alert_thresholds=alert_thresholds)
        
        # Data storage
        self.metrics_history = defaultdict(list)
        self.max_history_size = 10000
        
        # Real-time monitoring
        self.monitoring_active = False
        self.monitoring_thread = None
        
        if enable_real_time:
            self.start_real_time_monitoring()
    
    def record_inference(
        self,
        inputs: torch.Tensor,
        predictions: torch.Tensor,
        uncertainties: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
        inference_time: Optional[float] = None
    ):
        """Record a single inference for monitoring."""
        
        # Model performance metrics
        if hasattr(self, 'model_collector'):
            model_metrics = self.model_collector.collect_inference_metrics(
                inputs, predictions, targets, inference_time
            )
            self._store_metrics('model', model_metrics)
            self.alert_manager.check_model_metrics(model_metrics)
        
        # Uncertainty quality metrics
        if uncertainties is not None and targets is not None:
            self.uncertainty_collector.add_sample(predictions, uncertainties, targets)
            uncertainty_metrics = self.uncertainty_collector.collect()
            self._store_metrics('uncertainty', uncertainty_metrics)
            self.alert_manager.check_uncertainty_metrics(uncertainty_metrics)
    
    def collect_all_metrics(self) -> Dict[str, Any]:
        """Collect metrics from all collectors."""
        
        all_metrics = {}
        
        for collector in self.collectors:
            try:
                metrics = collector.collect()
                all_metrics[collector.get_name()] = metrics
            except Exception as e:
                logging.error(f"Failed to collect metrics from {collector.get_name()}: {e}")
        
        return all_metrics
    
    def _store_metrics(self, category: str, metrics: Any):
        """Store metrics in history."""
        
        self.metrics_history[category].append(metrics)
        
        # Limit history size
        if len(self.metrics_history[category]) > self.max_history_size:
            self.metrics_history[category].pop(0)
    
    def start_real_time_monitoring(self):
        """Start real-time monitoring in background thread."""
        
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        logging.info("Started real-time monitoring")
    
    def stop_real_time_monitoring(self):
        """Stop real-time monitoring."""
        
        self.monitoring_active = False
        
        if self.monitoring_thread is not None:
            self.monitoring_thread.join(timeout=1.0)
            self.monitoring_thread = None
        
        logging.info("Stopped real-time monitoring")
    
    def _monitoring_loop(self):
        """Background monitoring loop."""
        
        while self.monitoring_active:
            try:
                # Collect system metrics
                system_metrics = self.system_collector.collect()
                self._store_metrics('system', system_metrics)
                self.alert_manager.check_system_metrics(system_metrics)
                
                # Sleep until next collection
                time.sleep(self.collection_interval)
                
            except Exception as e:
                logging.error(f"Error in monitoring loop: {e}")
                time.sleep(self.collection_interval)
    
    def get_system_health_report(self) -> Dict[str, Any]:
        """Generate comprehensive system health report."""
        
        report = {
            'timestamp': time.time(),
            'monitoring_status': 'active' if self.monitoring_active else 'inactive',
            'alerts': {
                'total': len(self.alert_manager.alerts),
                'active': len(self.alert_manager.get_active_alerts()),
                'critical': len([a for a in self.alert_manager.alerts if a.level == AlertLevel.CRITICAL and not a.resolved])
            },
            'metrics_summary': {}
        }
        
        # System metrics summary
        if 'system' in self.metrics_history and self.metrics_history['system']:
            recent_system = self.metrics_history['system'][-1]
            report['metrics_summary']['system'] = {
                'cpu_usage': recent_system.cpu_usage,
                'memory_usage_percent': recent_system.memory_usage_percent,
                'gpu_usage': recent_system.gpu_usage,
                'gpu_temperature': recent_system.gpu_temperature
            }
        
        # Model metrics summary
        if 'model' in self.metrics_history and self.metrics_history['model']:
            recent_model = self.metrics_history['model'][-1]
            report['metrics_summary']['model'] = {
                'throughput': recent_model.throughput_samples_per_sec,
                'inference_time_ms': recent_model.inference_time_ms,
                'accuracy': recent_model.prediction_accuracy,
                'gradient_norm': recent_model.gradient_norm
            }
        
        # Uncertainty metrics summary
        if 'uncertainty' in self.metrics_history and self.metrics_history['uncertainty']:
            recent_uncertainty = self.metrics_history['uncertainty'][-1]
            report['metrics_summary']['uncertainty'] = {
                'calibration_error': recent_uncertainty.calibration_error,
                'coverage_95': recent_uncertainty.coverage_95,
                'reliability_score': recent_uncertainty.reliability_score,
                'correlation': recent_uncertainty.uncertainty_correlation
            }
        
        return report
    
    def export_metrics(self, filepath: str, format: str = 'json'):
        """Export collected metrics to file."""
        
        export_data = {
            'export_timestamp': time.time(),
            'metrics_history': {}
        }
        
        # Convert dataclasses to dictionaries for serialization
        for category, metrics_list in self.metrics_history.items():
            export_data['metrics_history'][category] = []
            
            for metrics in metrics_list:
                if hasattr(metrics, '__dict__'):
                    export_data['metrics_history'][category].append(metrics.__dict__)
                else:
                    export_data['metrics_history'][category].append(metrics)
        
        # Export alerts
        export_data['alerts'] = [
            {
                'timestamp': alert.timestamp,
                'level': alert.level.value,
                'source': alert.source,
                'message': alert.message,
                'acknowledged': alert.acknowledged,
                'resolved': alert.resolved
            }
            for alert in self.alert_manager.alerts
        ]
        
        if format.lower() == 'json':
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def __del__(self):
        """Cleanup when monitor is destroyed."""
        if hasattr(self, 'monitoring_active') and self.monitoring_active:
            self.stop_real_time_monitoring()