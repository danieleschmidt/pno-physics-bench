"""Advanced monitoring and error handling for PNO training."""

import logging
import time
import torch
import psutil
import warnings
from typing import Dict, Any, Optional, List
from contextlib import contextmanager
from pathlib import Path
import json


class TrainingMonitor:
    """Advanced training monitoring with error recovery."""
    
    def __init__(
        self,
        log_dir: str = "./logs",
        memory_threshold: float = 0.9,
        gpu_memory_threshold: float = 0.9,
        enable_profiling: bool = False
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.memory_threshold = memory_threshold
        self.gpu_memory_threshold = gpu_memory_threshold
        self.enable_profiling = enable_profiling
        
        # Setup logging
        self.logger = self._setup_logger()
        
        # Monitoring state
        self.start_time = None
        self.metrics_history = []
        self.error_count = 0
        self.warning_count = 0
        
    def _setup_logger(self) -> logging.Logger:
        """Setup dedicated training logger."""
        logger = logging.getLogger("pno_training_monitor")
        logger.setLevel(logging.INFO)
        
        # File handler
        log_file = self.log_dir / "training_monitor.log"
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.WARNING)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        logger.addHandler(fh)
        logger.addHandler(ch)
        
        return logger
    
    def check_system_health(self) -> Dict[str, Any]:
        """Check system resource usage."""
        health_status = {
            "timestamp": time.time(),
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_percent": psutil.virtual_memory().percent,
            "available_memory_gb": psutil.virtual_memory().available / (1024**3),
            "disk_usage_percent": psutil.disk_usage("/").percent,
        }
        
        # GPU monitoring if available
        if torch.cuda.is_available():
            try:
                health_status["gpu_count"] = torch.cuda.device_count()
                health_status["gpu_memory_allocated"] = []
                health_status["gpu_memory_cached"] = []
                
                for i in range(torch.cuda.device_count()):
                    allocated = torch.cuda.memory_allocated(i) / (1024**3)
                    cached = torch.cuda.memory_reserved(i) / (1024**3)
                    health_status["gpu_memory_allocated"].append(allocated)
                    health_status["gpu_memory_cached"].append(cached)
                    
            except Exception as e:
                self.logger.warning(f"Failed to get GPU info: {e}")
        
        # Check thresholds and warn
        if health_status["memory_percent"] > self.memory_threshold * 100:
            self.logger.warning(f"High memory usage: {health_status['memory_percent']:.1f}%")
            self.warning_count += 1
            
        if torch.cuda.is_available():
            for i, allocated in enumerate(health_status.get("gpu_memory_allocated", [])):
                if allocated > 10:  # More than 10GB allocated
                    self.logger.warning(f"High GPU memory usage on device {i}: {allocated:.2f}GB")
                    self.warning_count += 1
        
        return health_status
    
    @contextmanager
    def training_session(self, session_name: str = "training"):
        """Context manager for training session monitoring."""
        self.start_time = time.time()
        self.logger.info(f"Starting training session: {session_name}")
        
        # Log initial system state
        initial_health = self.check_system_health()
        self.logger.info(f"Initial system health: {initial_health}")
        
        try:
            yield self
        except KeyboardInterrupt:
            self.logger.info("Training interrupted by user")
            raise
        except torch.cuda.OutOfMemoryError as e:
            self.logger.error(f"CUDA OOM error: {e}")
            self.error_count += 1
            self._handle_cuda_oom()
            raise
        except Exception as e:
            self.logger.error(f"Training error: {e}")
            self.error_count += 1
            raise
        finally:
            if self.start_time:
                duration = time.time() - self.start_time
                self.logger.info(f"Training session ended. Duration: {duration:.2f}s")
                self.logger.info(f"Errors: {self.error_count}, Warnings: {self.warning_count}")
                
                # Save final metrics
                self._save_session_summary(session_name, duration)
    
    def _handle_cuda_oom(self):
        """Handle CUDA out of memory errors."""
        if torch.cuda.is_available():
            self.logger.info("Attempting to clear CUDA cache...")
            torch.cuda.empty_cache()
            
            # Log memory after cleanup
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / (1024**3)
                cached = torch.cuda.memory_reserved(i) / (1024**3)
                self.logger.info(f"GPU {i} - Allocated: {allocated:.2f}GB, Cached: {cached:.2f}GB")
    
    def log_metrics(self, epoch: int, metrics: Dict[str, Any]):
        """Log training metrics with health check."""
        # Add system health to metrics
        health = self.check_system_health()
        
        metric_entry = {
            "epoch": epoch,
            "timestamp": time.time(),
            "metrics": metrics,
            "system_health": health
        }
        
        self.metrics_history.append(metric_entry)
        
        # Log important metrics
        train_loss = metrics.get("train_loss", "N/A")
        val_loss = metrics.get("val_loss", "N/A")
        self.logger.info(f"Epoch {epoch} - Train Loss: {train_loss}, Val Loss: {val_loss}")
        
        # Check for potential issues
        self._check_training_health(metrics)
    
    def _check_training_health(self, metrics: Dict[str, Any]):
        """Check for training health issues."""
        train_loss = metrics.get("train_loss")
        val_loss = metrics.get("val_loss")
        
        if train_loss is not None:
            if torch.isnan(torch.tensor(train_loss)) or torch.isinf(torch.tensor(train_loss)):
                self.logger.error(f"Invalid train loss: {train_loss}")
                self.error_count += 1
                
            if train_loss > 1000:
                self.logger.warning(f"Very high train loss: {train_loss}")
                self.warning_count += 1
        
        if val_loss is not None and train_loss is not None:
            if val_loss > train_loss * 10:
                self.logger.warning(f"Potential overfitting: val_loss ({val_loss}) >> train_loss ({train_loss})")
                self.warning_count += 1
    
    def _save_session_summary(self, session_name: str, duration: float):
        """Save training session summary."""
        summary = {
            "session_name": session_name,
            "duration_seconds": duration,
            "total_epochs": len(self.metrics_history),
            "errors": self.error_count,
            "warnings": self.warning_count,
            "metrics_history": self.metrics_history[-10:],  # Last 10 entries
            "final_system_health": self.check_system_health()
        }
        
        summary_file = self.log_dir / f"session_{session_name}_{int(time.time())}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        self.logger.info(f"Session summary saved to {summary_file}")


class RobustTrainingWrapper:
    """Wrapper for robust training with automatic recovery."""
    
    def __init__(
        self,
        trainer,
        max_retries: int = 3,
        save_interval: int = 10,
        checkpoint_dir: str = "./checkpoints"
    ):
        self.trainer = trainer
        self.max_retries = max_retries
        self.save_interval = save_interval
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.monitor = TrainingMonitor()
        self.logger = logging.getLogger(__name__)
        
    def fit_robust(
        self,
        train_loader,
        val_loader=None,
        epochs: int = 100,
        **kwargs
    ) -> Dict[str, List[float]]:
        """Robust training with error recovery."""
        
        with self.monitor.training_session("robust_training"):
            retry_count = 0
            last_successful_epoch = 0
            
            while retry_count <= self.max_retries:
                try:
                    self.logger.info(f"Training attempt {retry_count + 1}/{self.max_retries + 1}")
                    
                    # Resume from last checkpoint if retrying
                    if retry_count > 0 and last_successful_epoch > 0:
                        checkpoint_path = self.checkpoint_dir / f"epoch_{last_successful_epoch}.pt"
                        if checkpoint_path.exists():
                            self.trainer.load_checkpoint(str(checkpoint_path))
                            self.logger.info(f"Resumed from epoch {last_successful_epoch}")
                    
                    # Start training
                    history = self.trainer.fit(
                        train_loader=train_loader,
                        val_loader=val_loader,
                        epochs=epochs,
                        **kwargs
                    )
                    
                    self.logger.info("Training completed successfully!")
                    return history
                    
                except torch.cuda.OutOfMemoryError as e:
                    self.logger.error(f"CUDA OOM on attempt {retry_count + 1}: {e}")
                    
                    # Try to reduce batch size
                    if hasattr(train_loader, 'batch_size') and train_loader.batch_size > 1:
                        new_batch_size = max(1, train_loader.batch_size // 2)
                        self.logger.info(f"Reducing batch size from {train_loader.batch_size} to {new_batch_size}")
                        train_loader.batch_size = new_batch_size
                        if val_loader and hasattr(val_loader, 'batch_size'):
                            val_loader.batch_size = new_batch_size
                    
                    retry_count += 1
                    
                except Exception as e:
                    self.logger.error(f"Training failed on attempt {retry_count + 1}: {e}")
                    retry_count += 1
                    
                    if retry_count <= self.max_retries:
                        self.logger.info(f"Retrying in 5 seconds...")
                        time.sleep(5)
            
            raise RuntimeError(f"Training failed after {self.max_retries + 1} attempts")
    
    def save_checkpoint(self, epoch: int, force: bool = False):
        """Save checkpoint if needed."""
        if force or epoch % self.save_interval == 0:
            checkpoint_path = self.checkpoint_dir / f"epoch_{epoch}.pt"
            self.trainer.save_checkpoint(str(checkpoint_path))
            self.monitor.logger.info(f"Checkpoint saved: {checkpoint_path}")


class GradientMonitor:
    """Monitor gradients for training stability."""
    
    def __init__(self, model, log_interval: int = 10):
        self.model = model
        self.log_interval = log_interval
        self.logger = logging.getLogger(__name__)
        self.step_count = 0
        
    def check_gradients(self) -> Dict[str, float]:
        """Check gradient statistics."""
        self.step_count += 1
        
        grad_stats = {
            "grad_norm_total": 0.0,
            "grad_norm_max": 0.0,
            "grad_norm_min": float('inf'),
            "num_zero_grads": 0,
            "num_nan_grads": 0
        }
        
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.data.norm().item()
                grad_stats["grad_norm_total"] += grad_norm
                grad_stats["grad_norm_max"] = max(grad_stats["grad_norm_max"], grad_norm)
                grad_stats["grad_norm_min"] = min(grad_stats["grad_norm_min"], grad_norm)
                
                if grad_norm == 0:
                    grad_stats["num_zero_grads"] += 1
                if torch.isnan(param.grad.data).any():
                    grad_stats["num_nan_grads"] += 1
                    self.logger.warning(f"NaN gradients detected in {name}")
        
        # Log periodically
        if self.step_count % self.log_interval == 0:
            self.logger.info(f"Step {self.step_count} - Gradient Stats: {grad_stats}")
        
        return grad_stats