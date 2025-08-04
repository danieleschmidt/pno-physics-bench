"""Structured logging configuration with monitoring integration."""

import logging
import logging.config
import sys
import os
import json
import time
from typing import Dict, Any, Optional, Union
from pathlib import Path
from datetime import datetime
import threading
import queue
from contextlib import contextmanager

import structlog
from structlog.stdlib import LoggerFactory
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import coloredlogs


# Prometheus metrics for monitoring
LOGGING_METRICS = {
    'log_messages_total': Counter(
        'pno_log_messages_total',
        'Total number of log messages',
        ['level', 'logger_name']
    ),
    'error_messages_total': Counter(
        'pno_error_messages_total', 
        'Total number of error messages',
        ['error_type', 'component']
    ),
    'training_metrics': Gauge(
        'pno_training_metric',
        'Training metrics',
        ['metric_name', 'experiment_id']
    ),
    'model_inference_duration': Histogram(
        'pno_inference_duration_seconds',
        'Model inference duration',
        ['model_type', 'batch_size']
    ),
    'memory_usage_bytes': Gauge(
        'pno_memory_usage_bytes',
        'Memory usage in bytes',
        ['memory_type', 'device']
    ),
}


class MetricsHandler(logging.Handler):
    """Custom handler to send metrics to Prometheus."""
    
    def emit(self, record: logging.LogRecord) -> None:
        """Emit log record and update metrics."""
        try:
            # Count log messages by level
            LOGGING_METRICS['log_messages_total'].labels(
                level=record.levelname.lower(),
                logger_name=record.name
            ).inc()
            
            # Count errors with additional context
            if record.levelno >= logging.ERROR:
                error_type = getattr(record, 'error_type', 'unknown')
                component = getattr(record, 'component', record.name.split('.')[-1])
                LOGGING_METRICS['error_messages_total'].labels(
                    error_type=error_type,
                    component=component
                ).inc()
            
            # Extract training metrics if present
            if hasattr(record, 'training_metrics'):
                experiment_id = getattr(record, 'experiment_id', 'default')
                for metric_name, value in record.training_metrics.items():
                    if isinstance(value, (int, float)):
                        LOGGING_METRICS['training_metrics'].labels(
                            metric_name=metric_name,
                            experiment_id=experiment_id
                        ).set(value)
        
        except Exception:
            # Silently ignore metrics errors to avoid log spam
            pass


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_entry = {
            'timestamp': datetime.utcfromtimestamp(record.created).isoformat() + 'Z',
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }
        
        # Add exception information
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        # Add custom fields
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                          'filename', 'module', 'lineno', 'funcName', 'created',
                          'msecs', 'relativeCreated', 'thread', 'threadName',
                          'processName', 'process', 'getMessage', 'exc_info',
                          'exc_text', 'stack_info']:
                log_entry[key] = value
        
        return json.dumps(log_entry, default=str)


class ContextualFilter(logging.Filter):
    """Add contextual information to log records."""
    
    def __init__(self):
        super().__init__()
        self.context = threading.local()
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Add context to log record."""
        # Add thread-local context
        if hasattr(self.context, 'experiment_id'):
            record.experiment_id = self.context.experiment_id
        if hasattr(self.context, 'model_type'):
            record.model_type = self.context.model_type
        if hasattr(self.context, 'pde_name'):
            record.pde_name = self.context.pde_name
        
        return True
    
    def set_context(self, **kwargs):
        """Set context for current thread."""
        for key, value in kwargs.items():
            setattr(self.context, key, value)
    
    def clear_context(self):
        """Clear context for current thread."""
        self.context = threading.local()


# Global context filter instance
_context_filter = ContextualFilter()


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    json_format: bool = False,
    enable_metrics: bool = True,
    metrics_port: int = 8000,
    enable_colors: bool = True,
) -> None:
    """Setup comprehensive logging configuration.
    
    Args:
        log_level: Logging level
        log_file: Optional log file path
        json_format: Whether to use JSON formatting
        enable_metrics: Whether to enable Prometheus metrics
        metrics_port: Port for Prometheus metrics server
        enable_colors: Whether to use colored console output
    """
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer() if json_format else structlog.dev.ConsoleRenderer(),
        ],
        context_class=dict,
        logger_factory=LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Setup standard logging
    handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    if json_format:
        console_handler.setFormatter(JSONFormatter())
    elif enable_colors:
        coloredlogs.install(
            level=log_level,
            logger=logging.getLogger(),
            fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    else:
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
    
    console_handler.addFilter(_context_filter)
    handlers.append(console_handler)
    
    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_formatter = JSONFormatter() if json_format else logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        file_handler.addFilter(_context_filter)
        handlers.append(file_handler)
    
    # Metrics handler
    if enable_metrics:
        metrics_handler = MetricsHandler()
        metrics_handler.addFilter(_context_filter)
        handlers.append(metrics_handler)
        
        # Start metrics server
        try:
            start_http_server(metrics_port)
            logging.getLogger(__name__).info(f"Metrics server started on port {metrics_port}")
        except Exception as e:
            logging.getLogger(__name__).warning(f"Failed to start metrics server: {e}")
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        handlers=handlers,
        force=True,
    )
    
    # Set specific logger levels
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)


@contextmanager
def logging_context(**kwargs):
    """Context manager for adding contextual information to logs.
    
    Args:
        **kwargs: Context variables to add
    """
    _context_filter.set_context(**kwargs)
    try:
        yield
    finally:
        _context_filter.clear_context()


class PerformanceLogger:
    """Logger for performance metrics and profiling."""
    
    def __init__(self, logger_name: str = "performance"):
        self.logger = logging.getLogger(logger_name)
        self.start_times = {}
    
    def start_timer(self, operation: str) -> None:
        """Start timing an operation."""
        self.start_times[operation] = time.time()
    
    def end_timer(self, operation: str, **context) -> float:
        """End timing and log duration."""
        if operation not in self.start_times:
            self.logger.warning(f"Timer '{operation}' not started")
            return 0.0
        
        duration = time.time() - self.start_times[operation]
        del self.start_times[operation]
        
        self.logger.info(
            f"Operation '{operation}' completed",
            extra={
                'operation': operation,
                'duration_seconds': duration,
                'performance_metric': True,
                **context
            }
        )
        
        # Update Prometheus metrics
        if operation in ['inference', 'training_step']:
            model_type = context.get('model_type', 'unknown')
            batch_size = context.get('batch_size', 'unknown')
            LOGGING_METRICS['model_inference_duration'].labels(
                model_type=model_type,
                batch_size=str(batch_size)
            ).observe(duration)
        
        return duration
    
    @contextmanager
    def timer(self, operation: str, **context):
        """Context manager for timing operations."""
        self.start_timer(operation)
        try:
            yield
        finally:
            self.end_timer(operation, **context)


class TrainingLogger:
    """Specialized logger for training metrics and events."""
    
    def __init__(self, experiment_id: str, logger_name: str = "training"):
        self.experiment_id = experiment_id
        self.logger = logging.getLogger(logger_name)
        self.epoch_start_time = None
    
    def log_hyperparameters(self, hyperparams: Dict[str, Any]) -> None:
        """Log hyperparameters."""
        self.logger.info(
            "Training hyperparameters",
            extra={
                'hyperparameters': hyperparams,
                'experiment_id': self.experiment_id,
                'event_type': 'hyperparameters'
            }
        )
    
    def log_epoch_start(self, epoch: int) -> None:
        """Log epoch start."""
        self.epoch_start_time = time.time()
        self.logger.info(
            f"Starting epoch {epoch}",
            extra={
                'epoch': epoch,
                'experiment_id': self.experiment_id,
                'event_type': 'epoch_start'
            }
        )
    
    def log_epoch_end(self, epoch: int, metrics: Dict[str, float]) -> None:
        """Log epoch completion with metrics."""
        duration = None
        if self.epoch_start_time:
            duration = time.time() - self.epoch_start_time
            self.epoch_start_time = None
        
        self.logger.info(
            f"Completed epoch {epoch}",
            extra={
                'epoch': epoch,
                'training_metrics': metrics,
                'epoch_duration': duration,
                'experiment_id': self.experiment_id,
                'event_type': 'epoch_end'
            }
        )
        
        # Update Prometheus metrics
        for metric_name, value in metrics.items():
            if isinstance(value, (int, float)):
                LOGGING_METRICS['training_metrics'].labels(
                    metric_name=metric_name,
                    experiment_id=self.experiment_id
                ).set(value)
    
    def log_model_checkpoint(self, epoch: int, checkpoint_path: str, is_best: bool = False) -> None:
        """Log model checkpoint save."""
        self.logger.info(
            f"Model checkpoint saved",
            extra={
                'epoch': epoch,
                'checkpoint_path': checkpoint_path,
                'is_best': is_best,
                'experiment_id': self.experiment_id,
                'event_type': 'checkpoint'
            }
        )
    
    def log_early_stopping(self, epoch: int, reason: str) -> None:
        """Log early stopping."""
        self.logger.info(
            f"Early stopping at epoch {epoch}: {reason}",
            extra={
                'epoch': epoch,
                'reason': reason,
                'experiment_id': self.experiment_id,
                'event_type': 'early_stopping'
            }
        )


class MemoryLogger:
    """Logger for memory usage monitoring."""
    
    def __init__(self, logger_name: str = "memory"):
        self.logger = logging.getLogger(logger_name)
    
    def log_memory_usage(self, context: str = "general") -> None:
        """Log current memory usage."""
        try:
            import psutil
            import torch
            
            # System memory
            system_memory = psutil.virtual_memory()
            
            memory_info = {
                'context': context,
                'system_total_gb': system_memory.total / (1024**3),
                'system_available_gb': system_memory.available / (1024**3),
                'system_used_gb': system_memory.used / (1024**3),
                'system_percent': system_memory.percent,
            }
            
            # GPU memory
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    allocated = torch.cuda.memory_allocated(i) / (1024**3)
                    reserved = torch.cuda.memory_reserved(i) / (1024**3)
                    memory_info[f'gpu_{i}_allocated_gb'] = allocated
                    memory_info[f'gpu_{i}_reserved_gb'] = reserved
                    
                    # Update Prometheus metrics
                    LOGGING_METRICS['memory_usage_bytes'].labels(
                        memory_type='allocated',
                        device=f'cuda:{i}'
                    ).set(allocated * 1024**3)
                    
                    LOGGING_METRICS['memory_usage_bytes'].labels(
                        memory_type='reserved',
                        device=f'cuda:{i}'
                    ).set(reserved * 1024**3)
            
            # Update system memory metrics
            LOGGING_METRICS['memory_usage_bytes'].labels(
                memory_type='system_used',
                device='cpu'
            ).set(memory_info['system_used_gb'] * 1024**3)
            
            self.logger.info(
                "Memory usage",
                extra={
                    'memory_info': memory_info,
                    'event_type': 'memory_usage'
                }
            )
            
        except Exception as e:
            self.logger.warning(f"Failed to get memory usage: {e}")


# Convenience function for getting structured logger
def get_logger(name: str) -> logging.Logger:
    """Get a configured logger instance.
    
    Args:
        name: Logger name
        
    Returns:
        Configured logger
    """
    return structlog.get_logger(name)