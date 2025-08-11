"""Comprehensive logging and monitoring system for PNO research."""

import logging
import json
import time
import threading
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, asdict
from pathlib import Path
from collections import defaultdict, deque
import traceback
from contextlib import contextmanager
import sys
import os

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

import numpy as np


@dataclass
class ExperimentMetric:
    """Structured experiment metric."""
    name: str
    value: Union[float, int, str]
    step: int
    timestamp: float
    experiment_id: str
    tags: Dict[str, str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = {}


@dataclass
class ModelEvent:
    """Model lifecycle event."""
    event_type: str  # training_start, epoch_complete, validation, error, etc.
    timestamp: float
    experiment_id: str
    details: Dict[str, Any]
    severity: str = "info"  # debug, info, warning, error, critical


class StructuredLogger:
    """Advanced structured logging for ML experiments."""
    
    def __init__(self, 
                 log_dir: str = "logs",
                 experiment_id: Optional[str] = None,
                 log_level: str = "INFO"):
        
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        self.experiment_id = experiment_id or f"exp_{int(time.time())}"
        self.log_level = getattr(logging, log_level.upper())
        
        # Initialize loggers
        self.setup_loggers()
        
        # Metric storage
        self.metrics_buffer = deque(maxlen=10000)
        self.events_buffer = deque(maxlen=5000)
        self.performance_metrics = defaultdict(list)
        
        # Threading for async logging
        self.async_logging = True
        self.log_queue = deque()
        self.log_thread = None
        self._start_async_logging()
    
    def setup_loggers(self):
        """Setup multiple specialized loggers."""
        
        # Main experiment logger
        self.main_logger = logging.getLogger(f"experiment_{self.experiment_id}")
        self.main_logger.setLevel(self.log_level)
        self.main_logger.handlers.clear()
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        self.main_logger.addHandler(console_handler)
        
        # File handler for main log
        main_log_file = self.log_dir / f"{self.experiment_id}_main.log"
        file_handler = logging.FileHandler(main_log_file)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        self.main_logger.addHandler(file_handler)
        
        # Metrics logger (JSON format)
        self.metrics_logger = logging.getLogger(f"metrics_{self.experiment_id}")
        self.metrics_logger.setLevel(logging.INFO)
        self.metrics_logger.handlers.clear()
        self.metrics_logger.propagate = False
        
        metrics_log_file = self.log_dir / f"{self.experiment_id}_metrics.jsonl"
        metrics_handler = logging.FileHandler(metrics_log_file)
        metrics_handler.setFormatter(logging.Formatter('%(message)s'))
        self.metrics_logger.addHandler(metrics_handler)
        
        # Error logger
        self.error_logger = logging.getLogger(f"errors_{self.experiment_id}")
        self.error_logger.setLevel(logging.WARNING)
        self.error_logger.handlers.clear()
        self.error_logger.propagate = False
        
        error_log_file = self.log_dir / f"{self.experiment_id}_errors.log"
        error_handler = logging.FileHandler(error_log_file)
        error_handler.setFormatter(logging.Formatter(
            '%(asctime)s - SEVERITY:%(levelname)s - %(message)s\n%(pathname)s:%(lineno)d\n'
        ))
        self.error_logger.addHandler(error_handler)
    
    def _start_async_logging(self):
        """Start asynchronous logging thread."""
        if self.async_logging:
            self.log_thread = threading.Thread(target=self._async_log_worker, daemon=True)
            self.log_thread.start()
    
    def _async_log_worker(self):
        """Worker thread for asynchronous logging."""
        while True:
            try:
                if self.log_queue:
                    log_entry = self.log_queue.popleft()
                    self._write_log_entry(log_entry)
                else:
                    time.sleep(0.1)  # Small delay when queue is empty
            except IndexError:
                time.sleep(0.1)
            except Exception as e:
                print(f"Async logging error: {e}")
    
    def _write_log_entry(self, log_entry: Dict[str, Any]):
        """Write log entry to appropriate logger."""
        log_type = log_entry.get("log_type", "main")
        message = log_entry.get("message", "")
        level = log_entry.get("level", "info")
        
        if log_type == "metrics":
            self.metrics_logger.info(json.dumps(log_entry))
        elif log_type == "error":
            getattr(self.error_logger, level)(message)
        else:
            getattr(self.main_logger, level)(message)
    
    def log_metric(self, name: str, value: Union[float, int, str], 
                   step: int = 0, tags: Optional[Dict[str, str]] = None):
        """Log a structured metric."""
        
        metric = ExperimentMetric(
            name=name,
            value=value,
            step=step,
            timestamp=time.time(),
            experiment_id=self.experiment_id,
            tags=tags or {}
        )
        
        self.metrics_buffer.append(metric)
        self.performance_metrics[name].append((step, value, time.time()))
        
        # Async logging
        if self.async_logging:
            log_entry = {
                "log_type": "metrics",
                "metric": asdict(metric)
            }
            self.log_queue.append(log_entry)
        else:
            self.metrics_logger.info(json.dumps(asdict(metric)))
    
    def log_event(self, event_type: str, details: Dict[str, Any], 
                  severity: str = "info"):
        """Log a model lifecycle event."""
        
        event = ModelEvent(
            event_type=event_type,
            timestamp=time.time(),
            experiment_id=self.experiment_id,
            details=details,
            severity=severity
        )
        
        self.events_buffer.append(event)
        
        # Format event message
        event_message = f"[{event_type}] {json.dumps(details, default=str)}"
        
        if self.async_logging:
            log_entry = {
                "log_type": "main" if severity in ["debug", "info"] else "error",
                "message": event_message,
                "level": severity
            }
            self.log_queue.append(log_entry)
        else:
            getattr(self.main_logger, severity)(event_message)
    
    def log_tensor_stats(self, tensor: Any, name: str = "tensor", step: int = 0):
        """Log comprehensive tensor statistics."""
        
        if HAS_TORCH and isinstance(tensor, torch.Tensor):
            stats = self._compute_torch_tensor_stats(tensor, name)
        elif isinstance(tensor, np.ndarray):
            stats = self._compute_numpy_stats(tensor, name)
        else:
            self.main_logger.warning(f"Cannot compute stats for tensor type: {type(tensor)}")
            return
        
        # Log individual stats as metrics
        for stat_name, stat_value in stats.items():
            self.log_metric(f"{name}_{stat_name}", stat_value, step)
        
        # Log summary event
        self.log_event("tensor_stats", {
            "tensor_name": name,
            "step": step,
            "stats": stats
        })
    
    def _compute_torch_tensor_stats(self, tensor: 'torch.Tensor', name: str) -> Dict[str, float]:
        """Compute comprehensive PyTorch tensor statistics."""
        
        stats = {
            "shape": str(list(tensor.shape)),
            "dtype": str(tensor.dtype),
            "device": str(tensor.device),
            "numel": int(tensor.numel()),
            "memory_mb": float(tensor.numel() * tensor.element_size() / 1024 / 1024)
        }
        
        if tensor.numel() > 0:
            stats.update({
                "mean": float(tensor.mean().item()),
                "std": float(tensor.std().item()),
                "min": float(tensor.min().item()),
                "max": float(tensor.max().item()),
                "abs_mean": float(tensor.abs().mean().item()),
                "zero_fraction": float((tensor == 0).float().mean().item()),
                "nan_count": int(torch.isnan(tensor).sum().item()),
                "inf_count": int(torch.isinf(tensor).sum().item()),
                "finite_fraction": float(torch.isfinite(tensor).float().mean().item())
            })
            
            # Gradient statistics if available
            if tensor.grad is not None:
                grad_stats = self._compute_torch_tensor_stats(tensor.grad, f"{name}_grad")
                stats.update({f"grad_{k}": v for k, v in grad_stats.items()})
        
        return stats
    
    def _compute_numpy_stats(self, array: np.ndarray, name: str) -> Dict[str, float]:
        """Compute comprehensive NumPy array statistics."""
        
        stats = {
            "shape": str(list(array.shape)),
            "dtype": str(array.dtype),
            "size": int(array.size),
            "memory_mb": float(array.nbytes / 1024 / 1024)
        }
        
        if array.size > 0:
            stats.update({
                "mean": float(np.mean(array)),
                "std": float(np.std(array)),
                "min": float(np.min(array)),
                "max": float(np.max(array)),
                "abs_mean": float(np.mean(np.abs(array))),
                "zero_fraction": float(np.mean(array == 0)),
                "nan_count": int(np.sum(np.isnan(array))),
                "inf_count": int(np.sum(np.isinf(array))),
                "finite_fraction": float(np.mean(np.isfinite(array)))
            })
        
        return stats
    
    @contextmanager
    def log_execution_time(self, operation_name: str, log_level: str = "info"):
        """Context manager to log execution time."""
        
        start_time = time.time()
        start_details = {
            "operation": operation_name,
            "start_time": start_time
        }
        
        self.log_event(f"{operation_name}_start", start_details, log_level)
        
        try:
            yield
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            end_details = {
                "operation": operation_name,
                "execution_time_seconds": execution_time,
                "success": True
            }
            
            self.log_event(f"{operation_name}_complete", end_details, log_level)
            self.log_metric(f"{operation_name}_duration", execution_time)
            
        except Exception as e:
            end_time = time.time()
            execution_time = end_time - start_time
            
            error_details = {
                "operation": operation_name,
                "execution_time_seconds": execution_time,
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__,
                "traceback": traceback.format_exc()
            }
            
            self.log_event(f"{operation_name}_error", error_details, "error")
            self.log_metric(f"{operation_name}_duration", execution_time)
            
            raise e
    
    def log_model_parameters(self, model, step: int = 0):
        """Log model parameter statistics."""
        
        if not HAS_TORCH or not isinstance(model, torch.nn.Module):
            self.main_logger.warning("Model parameter logging only available for PyTorch models")
            return
        
        total_params = 0
        trainable_params = 0
        parameter_stats = {}
        
        for name, param in model.named_parameters():
            if param is not None:
                total_params += param.numel()
                if param.requires_grad:
                    trainable_params += param.numel()
                
                # Log parameter statistics
                param_stats = self._compute_torch_tensor_stats(param, f"param_{name}")
                parameter_stats[name] = param_stats
                
                # Log as metrics
                for stat_name, stat_value in param_stats.items():
                    if isinstance(stat_value, (int, float)):
                        self.log_metric(f"model.{name}.{stat_name}", stat_value, step)
        
        # Log overall model stats
        model_stats = {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "non_trainable_parameters": total_params - trainable_params,
            "trainable_fraction": trainable_params / total_params if total_params > 0 else 0
        }
        
        for stat_name, stat_value in model_stats.items():
            self.log_metric(f"model.{stat_name}", stat_value, step)
        
        self.log_event("model_parameters", {
            "step": step,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "parameter_details": parameter_stats
        })
    
    def log_training_progress(self, epoch: int, batch: int, 
                            loss: float, metrics: Dict[str, float]):
        """Log training progress with rich context."""
        
        # Log loss and metrics
        self.log_metric("train_loss", loss, epoch * 1000 + batch)
        
        for metric_name, metric_value in metrics.items():
            self.log_metric(f"train_{metric_name}", metric_value, epoch * 1000 + batch)
        
        # Log training event
        self.log_event("training_step", {
            "epoch": epoch,
            "batch": batch,
            "loss": loss,
            "metrics": metrics
        })
    
    def log_validation_results(self, epoch: int, val_loss: float, 
                             val_metrics: Dict[str, float]):
        """Log validation results."""
        
        self.log_metric("val_loss", val_loss, epoch)
        
        for metric_name, metric_value in val_metrics.items():
            self.log_metric(f"val_{metric_name}", metric_value, epoch)
        
        self.log_event("validation", {
            "epoch": epoch,
            "val_loss": val_loss,
            "val_metrics": val_metrics
        })
    
    def log_hyperparameters(self, hyperparams: Dict[str, Any]):
        """Log experiment hyperparameters."""
        
        # Save hyperparameters to file
        hyperparams_file = self.log_dir / f"{self.experiment_id}_hyperparams.json"
        with open(hyperparams_file, 'w') as f:
            json.dump(hyperparams, f, indent=2, default=str)
        
        self.log_event("hyperparameters", hyperparams)
    
    def generate_experiment_summary(self) -> Dict[str, Any]:
        """Generate comprehensive experiment summary."""
        
        summary = {
            "experiment_id": self.experiment_id,
            "start_time": min(event.timestamp for event in self.events_buffer) if self.events_buffer else time.time(),
            "end_time": time.time(),
            "total_metrics": len(self.metrics_buffer),
            "total_events": len(self.events_buffer),
            "metrics_summary": {},
            "events_summary": {},
            "performance_summary": {}
        }
        
        # Metrics summary
        metric_names = set(metric.name for metric in self.metrics_buffer)
        for name in metric_names:
            values = [metric.value for metric in self.metrics_buffer if metric.name == name and isinstance(metric.value, (int, float))]
            if values:
                summary["metrics_summary"][name] = {
                    "count": len(values),
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "min": np.min(values),
                    "max": np.max(values),
                    "last_value": values[-1]
                }
        
        # Events summary
        event_types = defaultdict(int)
        for event in self.events_buffer:
            event_types[event.event_type] += 1
        summary["events_summary"] = dict(event_types)
        
        # Performance summary
        for metric_name, metric_history in self.performance_metrics.items():
            if len(metric_history) > 1:
                steps, values, timestamps = zip(*metric_history)
                if all(isinstance(v, (int, float)) for v in values):
                    summary["performance_summary"][metric_name] = {
                        "trend": "improving" if values[-1] < values[0] else "degrading",
                        "change": float(values[-1] - values[0]),
                        "change_percent": float((values[-1] - values[0]) / values[0] * 100) if values[0] != 0 else 0
                    }
        
        return summary
    
    def export_logs(self, export_format: str = "json") -> Path:
        """Export all logs in specified format."""
        
        export_data = {
            "experiment_id": self.experiment_id,
            "export_timestamp": time.time(),
            "metrics": [asdict(metric) for metric in self.metrics_buffer],
            "events": [asdict(event) for event in self.events_buffer],
            "summary": self.generate_experiment_summary()
        }
        
        if export_format == "json":
            export_file = self.log_dir / f"{self.experiment_id}_export.json"
            with open(export_file, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
        
        else:
            raise ValueError(f"Unsupported export format: {export_format}")
        
        return export_file
    
    def close(self):
        """Clean up logging resources."""
        
        # Stop async logging
        self.async_logging = False
        
        if self.log_thread:
            self.log_thread.join(timeout=5)
        
        # Flush remaining logs
        while self.log_queue:
            try:
                log_entry = self.log_queue.popleft()
                self._write_log_entry(log_entry)
            except IndexError:
                break
        
        # Close handlers
        for handler in self.main_logger.handlers:
            handler.close()
        for handler in self.metrics_logger.handlers:
            handler.close()
        for handler in self.error_logger.handlers:
            handler.close()


class ExperimentTracker:
    """High-level experiment tracking interface."""
    
    def __init__(self, experiment_name: str, log_dir: str = "experiments"):
        self.experiment_name = experiment_name
        self.experiment_id = f"{experiment_name}_{int(time.time())}"
        
        self.logger = StructuredLogger(
            log_dir=log_dir,
            experiment_id=self.experiment_id
        )
        
        self.start_time = time.time()
        self.logger.log_event("experiment_start", {
            "experiment_name": experiment_name,
            "experiment_id": self.experiment_id
        })
    
    def log_hyperparameters(self, **hyperparams):
        """Log experiment hyperparameters."""
        self.logger.log_hyperparameters(hyperparams)
    
    def log_metrics(self, step: int, **metrics):
        """Log multiple metrics at once."""
        for name, value in metrics.items():
            self.logger.log_metric(name, value, step)
    
    def log_model_checkpoint(self, model, epoch: int, metrics: Dict[str, float]):
        """Log model checkpoint with metrics."""
        
        self.logger.log_model_parameters(model, epoch)
        
        checkpoint_info = {
            "epoch": epoch,
            "metrics": metrics,
            "model_size_mb": 0  # Would calculate actual size
        }
        
        self.logger.log_event("checkpoint_saved", checkpoint_info)
    
    def finish_experiment(self) -> Dict[str, Any]:
        """Finish experiment and generate summary."""
        
        end_time = time.time()
        duration = end_time - self.start_time
        
        self.logger.log_event("experiment_end", {
            "duration_seconds": duration
        })
        
        summary = self.logger.generate_experiment_summary()
        
        # Export logs
        export_file = self.logger.export_logs()
        summary["export_file"] = str(export_file)
        
        # Close logger
        self.logger.close()
        
        return summary


if __name__ == "__main__":
    print("Comprehensive Logging System for PNO Research")
    print("=" * 50)
    
    # Example usage
    tracker = ExperimentTracker("pno_uncertainty_test")
    
    # Log hyperparameters
    tracker.log_hyperparameters(
        learning_rate=0.001,
        batch_size=32,
        num_layers=4,
        hidden_dim=256,
        uncertainty_type="variational"
    )
    
    # Simulate training loop
    for epoch in range(5):
        for batch in range(10):
            # Simulate training metrics
            loss = 1.0 / (epoch + 1) + np.random.random() * 0.1
            accuracy = 0.5 + (epoch * 0.1) + np.random.random() * 0.05
            
            tracker.log_metrics(
                step=epoch * 10 + batch,
                train_loss=loss,
                train_accuracy=accuracy
            )
            
            # Log tensor stats (simulated)
            if HAS_TORCH:
                fake_tensor = torch.randn(32, 256)
                tracker.logger.log_tensor_stats(fake_tensor, f"hidden_activations_epoch_{epoch}", epoch)
        
        # Validation
        val_loss = 0.8 / (epoch + 1) + np.random.random() * 0.05
        val_accuracy = 0.6 + (epoch * 0.08) + np.random.random() * 0.03
        
        tracker.logger.log_validation_results(epoch, val_loss, {
            "accuracy": val_accuracy,
            "calibration_error": np.random.random() * 0.1
        })
    
    # Finish experiment
    summary = tracker.finish_experiment()
    
    print("\nExperiment Summary:")
    print(f"- Experiment ID: {summary['experiment_id']}")
    print(f"- Duration: {summary['end_time'] - summary['start_time']:.2f} seconds")
    print(f"- Total metrics logged: {summary['total_metrics']}")
    print(f"- Total events: {summary['total_events']}")
    print(f"- Export file: {summary['export_file']}")
    
    print("\nComprehensive logging system initialized successfully!")