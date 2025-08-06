"""Performance optimization engine for PNO training and inference."""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Callable
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache, wraps
import pickle
import hashlib
from pathlib import Path
import logging


logger = logging.getLogger(__name__)


class AdaptivePerformanceOptimizer:
    """Adaptive performance optimization for PNO models."""
    
    def __init__(
        self,
        model: nn.Module,
        cache_dir: str = "./cache",
        enable_mixed_precision: bool = True,
        enable_compilation: bool = True,
        max_cache_size: int = 1000
    ):
        self.model = model
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.enable_mixed_precision = enable_mixed_precision
        self.enable_compilation = enable_compilation
        self.max_cache_size = max_cache_size
        
        # Performance tracking
        self.performance_stats = {
            "forward_times": [],
            "memory_usage": [],
            "cache_hits": 0,
            "cache_misses": 0,
            "optimization_applied": []
        }
        
        # Initialize optimizations
        self._apply_optimizations()
        
        # Adaptive caching
        self.prediction_cache = {}
        self.cache_lock = threading.Lock()
        
        logger.info("Performance optimizer initialized")
    
    def _apply_optimizations(self):
        """Apply performance optimizations to the model."""
        optimizations_applied = []
        
        # Mixed precision
        if self.enable_mixed_precision and torch.cuda.is_available():
            try:
                # Enable automatic mixed precision
                self.scaler = torch.cuda.amp.GradScaler()
                optimizations_applied.append("mixed_precision")
            except Exception as e:
                logger.warning(f"Failed to enable mixed precision: {e}")
        
        # Model compilation (PyTorch 2.0+)
        if self.enable_compilation and hasattr(torch, 'compile'):
            try:
                # Compile model for faster inference
                self.model = torch.compile(self.model, mode="reduce-overhead")
                optimizations_applied.append("torch_compile")
            except Exception as e:
                logger.warning(f"Failed to compile model: {e}")
        
        # Memory optimization
        try:
            # Enable memory efficient attention if available
            if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
                torch.backends.cuda.enable_flash_sdp(True)
                optimizations_applied.append("flash_attention")
        except Exception as e:
            logger.debug(f"Flash attention not available: {e}")
        
        self.performance_stats["optimization_applied"] = optimizations_applied
        logger.info(f"Applied optimizations: {optimizations_applied}")
    
    def _get_cache_key(self, x: torch.Tensor, **kwargs) -> str:
        """Generate cache key for input tensor."""
        # Create hash from tensor properties and values (sample only for large tensors)
        key_data = {
            "shape": x.shape,
            "dtype": str(x.dtype),
            "device": str(x.device),
            "kwargs": kwargs
        }
        
        # For large tensors, sample a subset for hashing
        if x.numel() > 10000:
            indices = torch.randint(0, x.numel(), (1000,))
            sample_data = x.flatten()[indices]
        else:
            sample_data = x
            
        # Create hash
        key_str = str(key_data) + str(sample_data.cpu().numpy().tobytes())
        return hashlib.md5(key_str.encode()).hexdigest()
    
    @torch.no_grad()
    def cached_predict(
        self,
        x: torch.Tensor,
        cache_key: Optional[str] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Cached prediction with uncertainty."""
        if cache_key is None:
            cache_key = self._get_cache_key(x, **kwargs)
        
        # Check cache
        with self.cache_lock:
            if cache_key in self.prediction_cache:
                self.performance_stats["cache_hits"] += 1
                return self.prediction_cache[cache_key]
            
            self.performance_stats["cache_misses"] += 1
        
        # Compute prediction
        start_time = time.time()
        
        if hasattr(self.model, 'predict_with_uncertainty'):
            mean, std = self.model.predict_with_uncertainty(x, **kwargs)
        else:
            outputs = self.model(x)
            if isinstance(outputs, tuple):
                mean, log_var = outputs
                std = torch.exp(0.5 * log_var)
            else:
                mean = outputs
                std = torch.zeros_like(mean)
        
        forward_time = time.time() - start_time
        self.performance_stats["forward_times"].append(forward_time)
        
        # Store in cache (with size limit)
        with self.cache_lock:
            if len(self.prediction_cache) >= self.max_cache_size:
                # Remove oldest entry (simple FIFO)
                oldest_key = next(iter(self.prediction_cache))
                del self.prediction_cache[oldest_key]
            
            self.prediction_cache[cache_key] = (mean, std)
        
        return mean, std
    
    def batch_predict_parallel(
        self,
        inputs: List[torch.Tensor],
        num_workers: int = 4,
        **kwargs
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Parallel batch prediction for multiple inputs."""
        if len(inputs) <= num_workers:
            # For small batches, process sequentially
            return [self.cached_predict(x, **kwargs) for x in inputs]
        
        # Parallel processing
        results = [None] * len(inputs)
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Submit all tasks
            future_to_index = {
                executor.submit(self.cached_predict, inputs[i], **kwargs): i
                for i in range(len(inputs))
            }
            
            # Collect results
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    results[index] = future.result()
                except Exception as e:
                    logger.error(f"Error in parallel prediction {index}: {e}")
                    # Return zero tensors as fallback
                    results[index] = (
                        torch.zeros_like(inputs[index][:, :1]),
                        torch.zeros_like(inputs[index][:, :1])
                    )
        
        return results
    
    def optimize_for_inference(self):
        """Optimize model specifically for inference."""
        self.model.eval()
        
        # Disable gradient computation globally
        torch.set_grad_enabled(False)
        
        # Fuse operations where possible
        try:
            # Fuse conv-bn layers
            torch.quantization.fuse_modules_qat(self.model, [])
            logger.info("Applied operation fusion")
        except Exception as e:
            logger.debug(f"Operation fusion not applied: {e}")
        
        # Enable inference mode optimizations
        if hasattr(torch, 'inference_mode'):
            self.model = torch.inference_mode()(self.model)
            logger.info("Enabled inference mode optimizations")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        stats = self.performance_stats.copy()
        
        if self.performance_stats["forward_times"]:
            times = np.array(self.performance_stats["forward_times"])
            stats["avg_forward_time"] = float(np.mean(times))
            stats["min_forward_time"] = float(np.min(times))
            stats["max_forward_time"] = float(np.max(times))
        
        stats["cache_hit_rate"] = (
            self.performance_stats["cache_hits"] / 
            max(1, self.performance_stats["cache_hits"] + self.performance_stats["cache_misses"])
        )
        
        return stats
    
    def clear_cache(self):
        """Clear prediction cache."""
        with self.cache_lock:
            self.prediction_cache.clear()
        logger.info("Prediction cache cleared")


class DynamicBatchSizer:
    """Dynamic batch sizing based on available memory."""
    
    def __init__(
        self,
        initial_batch_size: int = 32,
        max_batch_size: int = 256,
        memory_threshold: float = 0.8,
        adaptation_rate: float = 0.1
    ):
        self.current_batch_size = initial_batch_size
        self.max_batch_size = max_batch_size
        self.memory_threshold = memory_threshold
        self.adaptation_rate = adaptation_rate
        
        self.success_count = 0
        self.failure_count = 0
        self.last_oom_size = None
        
        logger.info(f"Dynamic batch sizer initialized with batch_size={initial_batch_size}")
    
    def get_batch_size(self) -> int:
        """Get current optimal batch size."""
        return self.current_batch_size
    
    def report_success(self):
        """Report successful batch processing."""
        self.success_count += 1
        
        # Gradually increase batch size if consistently successful
        if self.success_count >= 10 and self.current_batch_size < self.max_batch_size:
            if self.last_oom_size is None or self.current_batch_size < self.last_oom_size * 0.9:
                old_size = self.current_batch_size
                self.current_batch_size = min(
                    self.max_batch_size,
                    int(self.current_batch_size * (1 + self.adaptation_rate))
                )
                if self.current_batch_size != old_size:
                    logger.info(f"Increased batch size: {old_size} -> {self.current_batch_size}")
                self.success_count = 0
    
    def report_oom(self):
        """Report out-of-memory error."""
        self.failure_count += 1
        self.last_oom_size = self.current_batch_size
        
        # Reduce batch size
        old_size = self.current_batch_size
        self.current_batch_size = max(1, int(self.current_batch_size * 0.7))
        logger.warning(f"OOM detected, reduced batch size: {old_size} -> {self.current_batch_size}")
        
        self.success_count = 0


class ModelParallelizer:
    """Model parallelization utilities."""
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.device_count = torch.cuda.device_count() if torch.cuda.is_available() else 1
        self.is_parallel = False
        
    def apply_data_parallel(self) -> nn.Module:
        """Apply DataParallel for multi-GPU training."""
        if self.device_count > 1 and not self.is_parallel:
            try:
                self.model = nn.DataParallel(self.model)
                self.is_parallel = True
                logger.info(f"Applied DataParallel across {self.device_count} GPUs")
            except Exception as e:
                logger.warning(f"Failed to apply DataParallel: {e}")
        
        return self.model
    
    def apply_distributed_parallel(self, local_rank: int) -> nn.Module:
        """Apply DistributedDataParallel for distributed training."""
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            try:
                self.model = nn.parallel.DistributedDataParallel(
                    self.model,
                    device_ids=[local_rank],
                    output_device=local_rank
                )
                self.is_parallel = True
                logger.info("Applied DistributedDataParallel")
            except Exception as e:
                logger.warning(f"Failed to apply DistributedDataParallel: {e}")
        
        return self.model


def performance_profiler(func: Callable) -> Callable:
    """Decorator for profiling function performance."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        start_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        
        result = func(*args, **kwargs)
        
        end_time = time.time()
        end_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        
        logger.debug(f"{func.__name__} - Time: {end_time - start_time:.4f}s, "
                    f"Memory: {(end_memory - start_memory) / 1024**2:.2f}MB")
        
        return result
    return wrapper


class AdaptiveMemoryManager:
    """Adaptive memory management for large-scale training."""
    
    def __init__(
        self,
        memory_fraction: float = 0.8,
        cleanup_threshold: float = 0.9
    ):
        self.memory_fraction = memory_fraction
        self.cleanup_threshold = cleanup_threshold
        self.cleanup_count = 0
        
    def setup_memory_management(self):
        """Setup memory management policies."""
        if torch.cuda.is_available():
            # Set memory fraction
            torch.cuda.set_per_process_memory_fraction(self.memory_fraction)
            
            # Enable memory pool for faster allocation
            try:
                torch.cuda.memory._set_allocator_settings("expandable_segments:True")
            except Exception:
                pass  # Not available in all PyTorch versions
            
            logger.info(f"Memory management setup with {self.memory_fraction*100}% memory fraction")
    
    def check_and_cleanup(self):
        """Check memory usage and cleanup if needed."""
        if not torch.cuda.is_available():
            return
        
        allocated = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()
        max_memory = torch.cuda.get_device_properties(0).total_memory
        
        usage_ratio = allocated / max_memory
        
        if usage_ratio > self.cleanup_threshold:
            # Perform cleanup
            torch.cuda.empty_cache()
            self.cleanup_count += 1
            
            new_allocated = torch.cuda.memory_allocated()
            freed_mb = (allocated - new_allocated) / 1024**2
            
            logger.info(f"Memory cleanup #{self.cleanup_count}: freed {freed_mb:.2f}MB")
    
    @contextmanager
    def memory_context(self):
        """Context manager for automatic memory management."""
        try:
            yield
        finally:
            self.check_and_cleanup()


class QuantizationOptimizer:
    """Model quantization for inference speedup."""
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.quantized_model = None
        
    def apply_dynamic_quantization(self) -> nn.Module:
        """Apply dynamic quantization for CPU inference."""
        try:
            self.quantized_model = torch.quantization.quantize_dynamic(
                self.model,
                {torch.nn.Linear, torch.nn.Conv2d},
                dtype=torch.qint8
            )
            logger.info("Applied dynamic quantization")
            return self.quantized_model
        except Exception as e:
            logger.warning(f"Dynamic quantization failed: {e}")
            return self.model
    
    def apply_static_quantization(self, calibration_loader) -> nn.Module:
        """Apply static quantization with calibration data."""
        try:
            # Prepare model for quantization
            self.model.eval()
            self.model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
            torch.quantization.prepare(self.model, inplace=True)
            
            # Calibrate with sample data
            with torch.no_grad():
                for i, (data, _) in enumerate(calibration_loader):
                    if i >= 10:  # Use limited samples for calibration
                        break
                    _ = self.model(data)
            
            # Convert to quantized model
            self.quantized_model = torch.quantization.convert(self.model, inplace=False)
            logger.info("Applied static quantization")
            return self.quantized_model
            
        except Exception as e:
            logger.warning(f"Static quantization failed: {e}")
            return self.model


# Export main classes
__all__ = [
    "AdaptivePerformanceOptimizer",
    "DynamicBatchSizer", 
    "ModelParallelizer",
    "AdaptiveMemoryManager",
    "QuantizationOptimizer",
    "performance_profiler"
]