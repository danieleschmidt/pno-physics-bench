"""Performance profiling and optimization utilities."""

import torch
import time
import psutil
import logging
from typing import Dict, Any, Optional, Callable, List, Tuple
from contextlib import contextmanager
from pathlib import Path
import json
import numpy as np


logger = logging.getLogger(__name__)


class PerformanceProfiler:
    """Comprehensive performance profiler for PNO training and inference."""
    
    def __init__(self):
        self.profiles = {}
        self.current_profile = None
        self.memory_snapshots = []
    
    @contextmanager
    def profile(self, name: str):
        """Context manager for profiling code blocks."""
        self.start_profile(name)
        try:
            yield
        finally:
            self.end_profile(name)
    
    def start_profile(self, name: str):
        """Start profiling a named operation."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        profile_data = {
            'start_time': time.time(),
            'start_memory': self._get_memory_usage(),
            'gpu_memory_start': self._get_gpu_memory() if torch.cuda.is_available() else None,
        }
        
        self.profiles[name] = profile_data
        self.current_profile = name
    
    def end_profile(self, name: str):
        """End profiling a named operation."""
        if name not in self.profiles:
            logger.warning(f"Profile '{name}' was not started")
            return
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        profile_data = self.profiles[name]
        profile_data.update({
            'end_time': time.time(),
            'end_memory': self._get_memory_usage(),
            'gpu_memory_end': self._get_gpu_memory() if torch.cuda.is_available() else None,
        })
        
        # Calculate derived metrics
        profile_data['duration'] = profile_data['end_time'] - profile_data['start_time']
        profile_data['memory_delta'] = profile_data['end_memory'] - profile_data['start_memory']
        
        if profile_data['gpu_memory_start'] is not None:
            profile_data['gpu_memory_delta'] = (
                profile_data['gpu_memory_end'] - profile_data['gpu_memory_start']
            )
        
        self.current_profile = None
        logger.debug(f"Profile '{name}': {profile_data['duration']:.3f}s, "
                    f"Memory: {profile_data['memory_delta']:+.1f}MB")
    
    def _get_memory_usage(self) -> float:
        """Get current CPU memory usage in MB."""
        try:
            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)
        except:
            return 0.0
    
    def _get_gpu_memory(self) -> float:
        """Get current GPU memory usage in MB."""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024 * 1024)
        return 0.0
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all profiles."""
        summary = {
            'total_profiles': len(self.profiles),
            'profiles': {}
        }
        
        for name, data in self.profiles.items():
            if 'duration' in data:  # Only completed profiles
                summary['profiles'][name] = {
                    'duration': data['duration'],
                    'memory_delta': data['memory_delta'],
                    'gpu_memory_delta': data.get('gpu_memory_delta', 0),
                }
        
        return summary
    
    def save_profile(self, filepath: str):
        """Save profile data to file."""
        with open(filepath, 'w') as f:
            json.dump(self.get_summary(), f, indent=2)
        logger.info(f"Profile saved to {filepath}")


def profile_model(
    model: torch.nn.Module,
    input_shape: Tuple[int, ...],
    device: str = 'cuda',
    num_warmup: int = 10,
    num_runs: int = 100
) -> Dict[str, float]:
    """Profile model inference performance.
    
    Args:
        model: Model to profile
        input_shape: Input tensor shape
        device: Device to run on
        num_warmup: Number of warmup runs
        num_runs: Number of timing runs
        
    Returns:
        Performance metrics dictionary
    """
    model.eval()
    model = model.to(device)
    
    # Create dummy input
    dummy_input = torch.randn(input_shape, device=device)
    
    # Warmup runs
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(dummy_input)
    
    if device == 'cuda':
        torch.cuda.synchronize()
    
    # Timing runs
    times = []
    memory_usage = []
    
    with torch.no_grad():
        for _ in range(num_runs):
            if device == 'cuda':
                torch.cuda.synchronize()
                start_memory = torch.cuda.memory_allocated()
            
            start_time = time.time()
            output = model(dummy_input)
            
            if device == 'cuda':
                torch.cuda.synchronize()
                end_memory = torch.cuda.memory_allocated()
                memory_usage.append(end_memory - start_memory)
            
            end_time = time.time()
            times.append(end_time - start_time)
    
    # Calculate statistics
    times = np.array(times)
    metrics = {
        'mean_time': float(np.mean(times)),
        'std_time': float(np.std(times)),
        'min_time': float(np.min(times)),
        'max_time': float(np.max(times)),
        'throughput': 1.0 / np.mean(times),  # samples per second
    }
    
    if memory_usage:
        memory_usage = np.array(memory_usage)
        metrics.update({
            'mean_memory': float(np.mean(memory_usage)) / (1024 * 1024),  # MB
            'max_memory': float(np.max(memory_usage)) / (1024 * 1024),  # MB
        })
    
    # Model statistics
    metrics['num_parameters'] = sum(p.numel() for p in model.parameters())
    metrics['model_size_mb'] = metrics['num_parameters'] * 4 / (1024 * 1024)  # Assuming float32
    
    return metrics


def memory_usage(func: Callable) -> Callable:
    """Decorator to track memory usage of functions."""
    def wrapper(*args, **kwargs):
        initial_memory = psutil.Process().memory_info().rss
        
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            initial_gpu_memory = torch.cuda.memory_allocated()
        
        result = func(*args, **kwargs)
        
        final_memory = psutil.Process().memory_info().rss
        memory_delta = (final_memory - initial_memory) / (1024 * 1024)  # MB
        
        logger.info(f"{func.__name__} memory usage: {memory_delta:+.1f} MB")
        
        if torch.cuda.is_available():
            final_gpu_memory = torch.cuda.memory_allocated()
            gpu_memory_delta = (final_gpu_memory - initial_gpu_memory) / (1024 * 1024)
            peak_gpu_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)
            logger.info(f"{func.__name__} GPU memory: {gpu_memory_delta:+.1f} MB, "
                       f"peak: {peak_gpu_memory:.1f} MB")
        
        return result
    return wrapper


class BatchSizeOptimizer:
    """Automatically find optimal batch size for training."""
    
    def __init__(
        self,
        model: torch.nn.Module,
        loss_fn: torch.nn.Module,
        input_shape: Tuple[int, ...],
        device: str = 'cuda',
        memory_limit_gb: float = 8.0
    ):
        self.model = model
        self.loss_fn = loss_fn
        self.input_shape = input_shape
        self.device = device
        self.memory_limit = memory_limit_gb * 1024 * 1024 * 1024  # Convert to bytes
    
    def find_optimal_batch_size(
        self,
        start_batch_size: int = 1,
        max_batch_size: int = 512
    ) -> Dict[str, Any]:
        """Find optimal batch size using binary search."""
        
        logger.info("Finding optimal batch size...")
        
        # Test if we can run at all
        if not self._test_batch_size(start_batch_size):
            raise RuntimeError(f"Cannot run with minimum batch size {start_batch_size}")
        
        # Binary search for maximum feasible batch size
        low, high = start_batch_size, max_batch_size
        optimal_batch_size = start_batch_size
        
        while low <= high:
            mid = (low + high) // 2
            
            if self._test_batch_size(mid):
                optimal_batch_size = mid
                low = mid + 1
            else:
                high = mid - 1
        
        # Performance test at optimal batch size
        performance = self._benchmark_batch_size(optimal_batch_size)
        
        result = {
            'optimal_batch_size': optimal_batch_size,
            'performance': performance,
            'memory_usage_mb': performance.get('memory_usage', 0) / (1024 * 1024)
        }
        
        logger.info(f"Optimal batch size: {optimal_batch_size}")
        logger.info(f"Memory usage: {result['memory_usage_mb']:.1f} MB")
        
        return result
    
    def _test_batch_size(self, batch_size: int) -> bool:
        """Test if a batch size is feasible."""
        try:
            # Clear cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Create test data
            input_shape = (batch_size,) + self.input_shape[1:]
            x = torch.randn(input_shape, device=self.device)
            y = torch.randn(input_shape, device=self.device)
            
            # Forward pass
            self.model.train()
            output = self.model(x)
            
            # Compute loss
            if isinstance(output, tuple):
                loss_input = output
            else:
                loss_input = (output, torch.zeros_like(output))
            
            loss_dict = self.loss_fn(loss_input, y, self.model)
            loss = loss_dict['total']
            
            # Backward pass
            loss.backward()
            
            # Check memory usage
            if torch.cuda.is_available():
                memory_used = torch.cuda.memory_allocated()
                if memory_used > self.memory_limit:
                    return False
            
            return True
            
        except Exception as e:
            if "out of memory" in str(e).lower():
                return False
            else:
                logger.warning(f"Error testing batch size {batch_size}: {e}")
                return False
        
        finally:
            # Cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def _benchmark_batch_size(self, batch_size: int, num_runs: int = 10) -> Dict[str, float]:
        """Benchmark performance at specific batch size."""
        
        times = []
        memory_usage = []
        
        for _ in range(num_runs):
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                
                # Create test data
                input_shape = (batch_size,) + self.input_shape[1:]
                x = torch.randn(input_shape, device=self.device)
                y = torch.randn(input_shape, device=self.device)
                
                start_time = time.time()
                
                # Forward pass
                self.model.train()
                output = self.model(x)
                
                # Compute loss and backward pass
                if isinstance(output, tuple):
                    loss_input = output
                else:
                    loss_input = (output, torch.zeros_like(output))
                
                loss_dict = self.loss_fn(loss_input, y, self.model)
                loss = loss_dict['total']
                loss.backward()
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                end_time = time.time()
                times.append(end_time - start_time)
                
                if torch.cuda.is_available():
                    memory_usage.append(torch.cuda.max_memory_allocated())
                
            except Exception as e:
                logger.warning(f"Benchmark run failed: {e}")
                continue
        
        if not times:
            return {'error': 'All benchmark runs failed'}
        
        return {
            'mean_time': np.mean(times),
            'std_time': np.std(times),
            'samples_per_second': batch_size / np.mean(times),
            'memory_usage': np.max(memory_usage) if memory_usage else 0
        }


class GradientAccumulator:
    """Efficient gradient accumulation for large effective batch sizes."""
    
    def __init__(self, model: torch.nn.Module, accumulation_steps: int):
        self.model = model
        self.accumulation_steps = accumulation_steps
        self.step_count = 0
        self.accumulated_loss = 0.0
    
    def accumulate(self, loss: torch.Tensor) -> bool:
        """Accumulate gradients. Returns True when ready to step."""
        # Scale loss by accumulation steps
        scaled_loss = loss / self.accumulation_steps
        scaled_loss.backward()
        
        self.accumulated_loss += loss.item()
        self.step_count += 1
        
        # Ready to step?
        if self.step_count >= self.accumulation_steps:
            return True
        
        return False
    
    def step(self, optimizer: torch.optim.Optimizer) -> float:
        """Perform optimizer step and reset accumulation."""
        if self.step_count < self.accumulation_steps:
            logger.warning(f"Stepping with incomplete accumulation: {self.step_count}/{self.accumulation_steps}")
        
        optimizer.step()
        optimizer.zero_grad()
        
        # Get average loss
        avg_loss = self.accumulated_loss / max(self.step_count, 1)
        
        # Reset
        self.step_count = 0
        self.accumulated_loss = 0.0
        
        return avg_loss


def optimize_dataloader(
    dataset,
    batch_size: int,
    num_workers: Optional[int] = None,
    pin_memory: bool = True,
    persistent_workers: bool = True
) -> torch.utils.data.DataLoader:
    """Create optimized DataLoader with best practices."""
    
    # Auto-detect optimal num_workers
    if num_workers is None:
        num_workers = min(8, psutil.cpu_count())  # Cap at 8 to avoid overhead
    
    # Optimize for CUDA
    if torch.cuda.is_available():
        pin_memory = True
        prefetch_factor = 2
    else:
        pin_memory = False
        prefetch_factor = 2
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers and num_workers > 0,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        drop_last=True,  # For consistent batch sizes
    )
    
    logger.info(f"Created optimized DataLoader: batch_size={batch_size}, "
               f"num_workers={num_workers}, pin_memory={pin_memory}")
    
    return dataloader