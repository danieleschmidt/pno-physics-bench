# SECURITY NOTICE: This file has been automatically hardened for security
# - All user inputs should be validated and sanitized
# - Subprocess calls use secure alternatives
# - SQL queries use parameterized statements
# - No hardcoded secrets or credentials


"""Comprehensive Monitoring and Logging Framework"""

import logging
import json
import time
import psutil
import threading
from datetime import datetime
from typing import Dict, Any, Optional, List
from collections import deque, defaultdict
import os

class AdvancedMetricsCollector:
    """Advanced metrics collection for PNO systems"""
    
    def __init__(self, max_history: int = 1000):
        self.metrics_history = deque(maxlen=max_history)
        self.performance_counters = defaultdict(list)
        self.error_counts = defaultdict(int)
        self.start_time = time.time()
        self._lock = threading.Lock()
    
    def record_metric(self, metric_name: str, value: float, 
                     tags: Optional[Dict[str, str]] = None):
        """Record a metric with timestamp and optional tags"""
        with self._lock:
            metric = {
                "name": metric_name,
                "value": value,
                "timestamp": datetime.now().isoformat(),
                "tags": tags or {}
            }
            self.metrics_history.append(metric)
            self.performance_counters[metric_name].append((time.time(), value))
    
    def record_error(self, error_type: str, context: Optional[Dict] = None):
        """Record error occurrence"""
        with self._lock:
            self.error_counts[error_type] += 1
            self.record_metric(f"error_count_{error_type}", self.error_counts[error_type])
    
    def get_system_metrics(self) -> Dict[str, float]:
        """Get current system metrics"""
        return {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_usage_percent": psutil.disk_usage('/').percent,
            "uptime_seconds": time.time() - self.start_time
        }
    
    def get_performance_summary(self, metric_name: str, 
                              window_seconds: int = 300) -> Dict[str, float]:
        """Get performance summary for a metric over time window"""
        current_time = time.time()
        recent_values = []
        
        for timestamp, value in self.performance_counters[metric_name]:
            if current_time - timestamp <= window_seconds:
                recent_values.append(value)
        
        if not recent_values:
            return {}
        
        return {
            "count": len(recent_values),
            "mean": sum(recent_values) / len(recent_values),
            "min": min(recent_values),
            "max": max(recent_values),
            "latest": recent_values[-1] if recent_values else 0
        }

class AdvancedLogger:
    """Advanced logging with structured output and monitoring integration"""
    
    def __init__(self, name: str, log_file: Optional[str] = None):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        
        # Create formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
        )
        
        json_formatter = JsonFormatter()
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(detailed_formatter)
        self.logger.addHandler(console_handler)
        
        # File handler with JSON format
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(json_formatter)
            self.logger.addHandler(file_handler)
        
        self.metrics_collector = AdvancedMetricsCollector()
    
    def log_with_metrics(self, level: str, message: str, 
                        metrics: Optional[Dict[str, float]] = None,
                        context: Optional[Dict[str, Any]] = None):
        """Log message with associated metrics"""
        log_method = getattr(self.logger, level.lower())
        
        # Create structured log entry
        log_entry = {
            "message": message,
            "level": level,
            "timestamp": datetime.now().isoformat(),
            "context": context or {},
            "metrics": metrics or {}
        }
        
        log_method(json.dumps(log_entry))
        
        # Record metrics separately
        if metrics:
            for metric_name, value in metrics.items():
                self.metrics_collector.record_metric(metric_name, value, context)
    
    def log_performance(self, operation: str, duration: float, 
                       success: bool = True, context: Optional[Dict] = None):
        """Log performance metrics for operations"""
        self.metrics_collector.record_metric(f"operation_duration_{operation}", duration)
        
        if not success:
            self.metrics_collector.record_error(f"operation_failed_{operation}", context)
        
        self.log_with_metrics(
            "INFO",
            f"Operation {operation} completed in {duration:.3f}s",
            {"duration": duration, "success": success},
            context
        )

class JsonFormatter(logging.Formatter):
    """JSON formatter for structured logging"""
    
    def format(self, record):
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        
        return json.dumps(log_entry)

def monitor_function_performance(operation_name: str):
    """Decorator to monitor function performance"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            success = True
            error = None
            
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                success = False
                error = str(e)
                raise
            finally:
                duration = time.time() - start_time
                
                # Log performance metrics
                logger = AdvancedLogger(f"performance_{operation_name}")
                logger.log_performance(
                    operation_name,
                    duration,
                    success,
                    {"args_count": len(args), "kwargs_count": len(kwargs), "error": error}
                )
        
        return wrapper
    return decorator

# Global instances
metrics_collector = AdvancedMetricsCollector()
performance_logger = AdvancedLogger("pno_performance", "/root/repo/logs/performance.log")
