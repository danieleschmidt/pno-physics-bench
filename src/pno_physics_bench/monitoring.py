# SECURITY NOTICE: This file has been automatically hardened for security
# - All user inputs should be validated and sanitized
# - Subprocess calls use secure alternatives
# - SQL queries use parameterized statements
# - No hardcoded secrets or credentials


"""Monitoring and metrics collection for PNO Physics Bench."""

import time
import psutil
import threading
from typing import Dict, Any, Optional, List
from contextlib import contextmanager
from prometheus_client import (
    Counter, Histogram, Gauge, Info, start_http_server,
    CollectorRegistry, REGISTRY
)
import torch


class MLMetrics:
    """Prometheus metrics for ML training and inference."""
    
    def __init__(self, registry: Optional[CollectorRegistry] = None):
        """Initialize ML metrics."""
        if registry is None:
            registry = REGISTRY
        
        # Training metrics
        self.training_loss = Gauge(
            'pno_training_loss',
            'Current training loss',
            registry=registry
        )
        
        self.validation_loss = Gauge(
            'pno_validation_loss',
            'Current validation loss',
            registry=registry
        )
        
        self.current_epoch = Gauge(
            'pno_current_epoch',
            'Current training epoch',
            registry=registry
        )
        
        self.learning_rate = Gauge(
            'pno_learning_rate',
            'Current learning rate',
            registry=registry
        )
        
        # Uncertainty metrics
        self.uncertainty_coverage_90 = Gauge(
            'pno_uncertainty_coverage_90',
            '90% uncertainty coverage',
            registry=registry
        )
        
        self.uncertainty_coverage_95 = Gauge(
            'pno_uncertainty_coverage_95',
            '95% uncertainty coverage',
            registry=registry
        )
        
        self.mean_uncertainty = Gauge(
            'pno_mean_uncertainty',
            'Mean prediction uncertainty',
            registry=registry
        )
        
        # Performance metrics
        self.training_step_duration = Histogram(
            'pno_training_step_duration_seconds',
            'Training step duration in seconds',
            buckets=[0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
            registry=registry
        )
        
        self.inference_duration = Histogram(
            'pno_inference_duration_seconds',
            'Inference duration in seconds',
            buckets=[0.001, 0.01, 0.1, 0.5, 1.0, 2.0],
            registry=registry
        )
        
        self.batch_size = Gauge(
            'pno_batch_size',
            'Current batch size',
            registry=registry
        )
        
        # GPU metrics (if available)
        if torch.cuda.is_available():
            self.gpu_memory_used = Gauge(
                'pno_gpu_memory_used_bytes',
                'GPU memory used in bytes',
                ['device'],
                registry=registry
            )
            
            self.gpu_memory_total = Gauge(
                'pno_gpu_memory_total_bytes',
                'Total GPU memory in bytes',
                ['device'],
                registry=registry
            )
            
            self.gpu_utilization = Gauge(
                'pno_gpu_utilization_percent',
                'GPU utilization percentage',
                ['device'],
                registry=registry
            )
        
        # Data metrics
        self.samples_processed = Counter(
            'pno_samples_processed_total',
            'Total number of samples processed',
            ['split'],  # train, val, test
            registry=registry
        )
        
        self.batches_processed = Counter(
            'pno_batches_processed_total',
            'Total number of batches processed',
            ['split'],
            registry=registry
        )
        
        # Model metrics
        self.model_parameters = Gauge(
            'pno_model_parameters_total',
            'Total number of model parameters',
            registry=registry
        )
        
        self.model_info = Info(
            'pno_model',
            'Model information',
            registry=registry
        )
        
        # Error counters
        self.training_errors = Counter(
            'pno_training_errors_total',
            'Total training errors',
            ['error_type'],
            registry=registry
        )
        
        self.inference_errors = Counter(
            'pno_inference_errors_total',
            'Total inference errors',
            ['error_type'],
            registry=registry
        )
    
    def update_training_metrics(self, epoch: int, train_loss: float, 
                              val_loss: Optional[float] = None, 
                              learning_rate: Optional[float] = None):
        """Update training metrics."""
        self.current_epoch.set(epoch)
        self.training_loss.set(train_loss)
        
        if val_loss is not None:
            self.validation_loss.set(val_loss)
        
        if learning_rate is not None:
            self.learning_rate.set(learning_rate)
    
    def update_uncertainty_metrics(self, coverage_90: float, coverage_95: float, 
                                 mean_uncertainty: float):
        """Update uncertainty quantification metrics."""
        self.uncertainty_coverage_90.set(coverage_90)
        self.uncertainty_coverage_95.set(coverage_95)
        self.mean_uncertainty.set(mean_uncertainty)
    
    def update_gpu_metrics(self):
        """Update GPU metrics if CUDA is available."""
        if not torch.cuda.is_available():
            return
        
        for i in range(torch.cuda.device_count()):
            device = f"cuda:{i}"
            
            # Memory metrics
            memory_used = torch.cuda.memory_allocated(i)
            memory_total = torch.cuda.get_device_properties(i).total_memory
            
            self.gpu_memory_used.labels(device=device).set(memory_used)
            self.gpu_memory_total.labels(device=device).set(memory_total)
            
            # Try to get utilization (requires nvidia-ml-py if available)
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                self.gpu_utilization.labels(device=device).set(utilization.gpu)
            except ImportError:
                # pynvml not available, skip utilization
                pass
    
    def record_batch_processed(self, split: str, batch_size: int):
        """Record that a batch was processed."""
        self.batches_processed.labels(split=split).inc()
        self.samples_processed.labels(split=split).inc(batch_size)
        self.batch_size.set(batch_size)
    
    def record_training_error(self, error_type: str):
        """Record a training error."""
        self.training_errors.labels(error_type=error_type).inc()
    
    def record_inference_error(self, error_type: str):
        """Record an inference error."""
        self.inference_errors.labels(error_type=error_type).inc()
    
    @contextmanager
    def time_training_step(self):
        """Context manager to time training steps."""
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            self.training_step_duration.observe(duration)
    
    @contextmanager
    def time_inference(self):
        """Context manager to time inference."""
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            self.inference_duration.observe(duration)


class SystemMetrics:
    """System-level metrics for monitoring."""
    
    def __init__(self, registry: Optional[CollectorRegistry] = None):
        """Initialize system metrics."""
        if registry is None:
            registry = REGISTRY
        
        self.cpu_usage = Gauge(
            'pno_cpu_usage_percent',
            'CPU usage percentage',
            registry=registry
        )
        
        self.memory_usage = Gauge(
            'pno_memory_usage_bytes',
            'Memory usage in bytes',
            registry=registry
        )
        
        self.memory_total = Gauge(
            'pno_memory_total_bytes',
            'Total memory in bytes',
            registry=registry
        )
        
        self.disk_usage = Gauge(
            'pno_disk_usage_bytes',
            'Disk usage in bytes',
            ['path'],
            registry=registry
        )
        
        self.disk_total = Gauge(
            'pno_disk_total_bytes',
            'Total disk space in bytes',
            ['path'],
            registry=registry
        )
        
        # Start background thread to update system metrics
        self._update_thread = threading.Thread(target=self._update_loop, daemon=True)
        self._running = True
        self._update_thread.start()
    
    def _update_loop(self):
        """Background loop to update system metrics."""
        while self._running:
            try:
                self.update_system_metrics()
                time.sleep(10)  # Update every 10 seconds
            except Exception as e:
                print(f"Error updating system metrics: {e}")
                time.sleep(30)  # Wait longer on error
    
    def update_system_metrics(self):
        """Update system metrics."""
        # CPU usage
        cpu_percent = psutil.cpu_percent()
        self.cpu_usage.set(cpu_percent)
        
        # Memory usage
        memory = psutil.virtual_memory()
        self.memory_usage.set(memory.used)
        self.memory_total.set(memory.total)
        
        # Disk usage for common paths
        paths = ['/']  # Add more paths as needed
        for path in paths:
            try:
                disk = psutil.disk_usage(path)
                self.disk_usage.labels(path=path).set(disk.used)
                self.disk_total.labels(path=path).set(disk.total)
            except Exception:
                # Path might not exist in container
                pass
    
    def stop(self):
        """Stop the background update thread."""
        self._running = False
        if self._update_thread.is_alive():
            self._update_thread.join()


class MetricsCollector:
    """Main metrics collector for PNO Physics Bench."""
    
    def __init__(self, port: int = 8080, registry: Optional[CollectorRegistry] = None):
        """Initialize metrics collector."""
        self.port = port
        self.registry = registry or REGISTRY
        
        self.ml_metrics = MLMetrics(registry=self.registry)
        self.system_metrics = SystemMetrics(registry=self.registry)
        self._server_started = False
    
    def start_server(self):
        """Start the Prometheus metrics server."""
        if not self._server_started:
            start_http_server(self.port, registry=self.registry)
            self._server_started = True
            print(f"Metrics server started on port {self.port}")
    
    def stop(self):
        """Stop the metrics collector."""
        self.system_metrics.stop()
    
    def set_model_info(self, model_name: str, version: str, parameters: int, 
                      architecture: str, uncertainty_type: str):
        """Set model information."""
        self.ml_metrics.model_info.info({
            'name': model_name,
            'version': version,
            'architecture': architecture,
            'uncertainty_type': uncertainty_type
        })
        self.ml_metrics.model_parameters.set(parameters)


# Global metrics collector instance
_metrics_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector instance."""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector


def start_metrics_server(port: int = 8080):
    """Start the metrics server."""
    collector = get_metrics_collector()
    collector.port = port
    collector.start_server()


def stop_metrics_collection():
    """Stop metrics collection."""
    global _metrics_collector
    if _metrics_collector is not None:
        _metrics_collector.stop()
        _metrics_collector = None


# Convenience functions
def record_training_step(epoch: int, train_loss: float, val_loss: Optional[float] = None,
                        learning_rate: Optional[float] = None):
    """Record training step metrics."""
    collector = get_metrics_collector()
    collector.ml_metrics.update_training_metrics(epoch, train_loss, val_loss, learning_rate)


def record_uncertainty_metrics(coverage_90: float, coverage_95: float, mean_uncertainty: float):
    """Record uncertainty metrics."""
    collector = get_metrics_collector()
    collector.ml_metrics.update_uncertainty_metrics(coverage_90, coverage_95, mean_uncertainty)


def record_batch_processed(split: str, batch_size: int):
    """Record batch processing."""
    collector = get_metrics_collector()
    collector.ml_metrics.record_batch_processed(split, batch_size)


def time_training_step():
    """Time a training step."""
    collector = get_metrics_collector()
    return collector.ml_metrics.time_training_step()


def time_inference():
    """Time inference."""
    collector = get_metrics_collector()
    return collector.ml_metrics.time_inference()