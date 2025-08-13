"""Performance optimization and auto-scaling for PNO systems."""

import time
import threading
import multiprocessing
import queue
import os
from typing import Any, Dict, List, Optional, Callable, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import logging
import psutil
import warnings
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from collections import deque, defaultdict

logger = logging.getLogger(__name__)

# Optional dependencies
try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    warnings.warn("PyTorch not available - GPU optimization disabled")

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


class OptimizationStrategy(Enum):
    """Performance optimization strategies."""
    AGGRESSIVE = "aggressive"
    BALANCED = "balanced"
    CONSERVATIVE = "conservative"
    ADAPTIVE = "adaptive"


class ResourceType(Enum):
    """Types of system resources."""
    CPU = "cpu"
    MEMORY = "memory"
    GPU = "gpu"
    DISK = "disk"
    NETWORK = "network"


@dataclass
class PerformanceMetrics:
    """Performance measurement data."""
    timestamp: float
    operation: str
    duration: float
    throughput: float
    memory_usage: float
    cpu_usage: float
    gpu_usage: Optional[float] = None
    cache_hit_rate: Optional[float] = None
    error_rate: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp,
            'operation': self.operation,
            'duration': self.duration,
            'throughput': self.throughput,
            'memory_usage': self.memory_usage,
            'cpu_usage': self.cpu_usage,
            'gpu_usage': self.gpu_usage,
            'cache_hit_rate': self.cache_hit_rate,
            'error_rate': self.error_rate
        }


@dataclass
class OptimizationResult:
    """Result of optimization attempt."""
    strategy: str
    applied: bool
    improvement: float
    metrics_before: PerformanceMetrics
    metrics_after: Optional[PerformanceMetrics] = None
    description: str = ""
    side_effects: List[str] = field(default_factory=list)


class PerformanceProfiler:
    """Performance profiling and monitoring."""
    
    def __init__(self, window_size: int = 1000):
        """Initialize performance profiler.
        
        Args:
            window_size: Size of metrics history window
        """
        self.window_size = window_size
        self.metrics_history = deque(maxlen=window_size)
        self.operation_stats = defaultdict(list)
        self._lock = threading.RLock()
        
    def record_metrics(self, metrics: PerformanceMetrics):
        """Record performance metrics."""
        with self._lock:
            self.metrics_history.append(metrics)
            self.operation_stats[metrics.operation].append(metrics)
            
            # Keep operation stats bounded
            if len(self.operation_stats[metrics.operation]) > self.window_size:
                self.operation_stats[metrics.operation].pop(0)
    
    def get_operation_baseline(self, operation: str) -> Optional[PerformanceMetrics]:
        """Get baseline performance for operation."""
        with self._lock:
            if operation not in self.operation_stats:
                return None
            
            stats = self.operation_stats[operation]
            if not stats:
                return None
            
            # Calculate median values as baseline
            durations = [m.duration for m in stats]
            throughputs = [m.throughput for m in stats]
            memory_usages = [m.memory_usage for m in stats]
            cpu_usages = [m.cpu_usage for m in stats]
            
            durations.sort()
            throughputs.sort()
            memory_usages.sort()
            cpu_usages.sort()
            
            median_idx = len(durations) // 2
            
            return PerformanceMetrics(
                timestamp=time.time(),
                operation=operation,
                duration=durations[median_idx],
                throughput=throughputs[median_idx],
                memory_usage=memory_usages[median_idx],
                cpu_usage=cpu_usages[median_idx]
            )
    
    def detect_performance_regression(
        self,
        operation: str,
        threshold: float = 0.2  # 20% degradation
    ) -> bool:
        """Detect performance regression for operation."""
        baseline = self.get_operation_baseline(operation)
        if not baseline:
            return False
        
        with self._lock:
            recent_stats = self.operation_stats[operation][-10:]  # Last 10 measurements
            if len(recent_stats) < 5:
                return False
            
            recent_avg_duration = sum(m.duration for m in recent_stats) / len(recent_stats)
            
            regression = (recent_avg_duration - baseline.duration) / baseline.duration
            return regression > threshold
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        with self._lock:
            if not self.metrics_history:
                return {}
            
            recent_metrics = list(self.metrics_history)[-100:]  # Last 100 measurements
            
            avg_duration = sum(m.duration for m in recent_metrics) / len(recent_metrics)
            avg_throughput = sum(m.throughput for m in recent_metrics) / len(recent_metrics)
            avg_memory = sum(m.memory_usage for m in recent_metrics) / len(recent_metrics)
            avg_cpu = sum(m.cpu_usage for m in recent_metrics) / len(recent_metrics)
            
            return {
                'average_duration': avg_duration,
                'average_throughput': avg_throughput,
                'average_memory_usage': avg_memory,
                'average_cpu_usage': avg_cpu,
                'total_operations': len(self.metrics_history),
                'operations_tracked': len(self.operation_stats)
            }


class BaseOptimizer(ABC):
    """Base class for performance optimizers."""
    
    def __init__(self, name: str, profiler: PerformanceProfiler):
        """Initialize optimizer.
        
        Args:
            name: Optimizer name
            profiler: Performance profiler instance
        """
        self.name = name
        self.profiler = profiler
        self.enabled = True
        self.optimization_history = []
    
    @abstractmethod
    def can_optimize(self, metrics: PerformanceMetrics) -> bool:
        """Check if optimization can be applied."""
        pass
    
    @abstractmethod
    def optimize(self, metrics: PerformanceMetrics) -> OptimizationResult:
        """Apply optimization."""
        pass
    
    def estimate_impact(self, metrics: PerformanceMetrics) -> float:
        """Estimate optimization impact (0-1 scale)."""
        return 0.1  # Default conservative estimate


class BatchSizeOptimizer(BaseOptimizer):
    """Optimizer for dynamic batch size adjustment."""
    
    def __init__(
        self,
        profiler: PerformanceProfiler,
        min_batch_size: int = 1,
        max_batch_size: int = 512,
        memory_threshold: float = 0.8
    ):
        """Initialize batch size optimizer."""
        super().__init__("batch_size", profiler)
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.memory_threshold = memory_threshold
        self.current_batch_size = 32  # Default
    
    def can_optimize(self, metrics: PerformanceMetrics) -> bool:
        """Check if batch size can be optimized."""
        # Can optimize if memory usage is low or high
        return (
            metrics.memory_usage < 0.5 or  # Underutilized
            metrics.memory_usage > self.memory_threshold  # Over threshold
        )
    
    def optimize(self, metrics: PerformanceMetrics) -> OptimizationResult:
        """Optimize batch size."""
        old_batch_size = self.current_batch_size
        
        if metrics.memory_usage < 0.5 and self.current_batch_size < self.max_batch_size:
            # Increase batch size
            new_batch_size = min(self.current_batch_size * 2, self.max_batch_size)
            improvement = 0.2  # Estimated 20% improvement
            description = f"Increased batch size from {old_batch_size} to {new_batch_size}"
            
        elif metrics.memory_usage > self.memory_threshold and self.current_batch_size > self.min_batch_size:
            # Decrease batch size
            new_batch_size = max(self.current_batch_size // 2, self.min_batch_size)
            improvement = 0.1  # Estimated 10% improvement (stability)
            description = f"Decreased batch size from {old_batch_size} to {new_batch_size}"
            
        else:
            return OptimizationResult(
                strategy=self.name,
                applied=False,
                improvement=0.0,
                metrics_before=metrics,
                description="No batch size optimization needed"
            )
        
        self.current_batch_size = new_batch_size
        
        return OptimizationResult(
            strategy=self.name,
            applied=True,
            improvement=improvement,
            metrics_before=metrics,
            description=description
        )


class ThreadPoolOptimizer(BaseOptimizer):
    """Optimizer for thread pool sizing."""
    
    def __init__(
        self,
        profiler: PerformanceProfiler,
        min_threads: int = 1,
        max_threads: int = None
    ):
        """Initialize thread pool optimizer."""
        super().__init__("thread_pool", profiler)
        self.min_threads = min_threads
        self.max_threads = max_threads or min(32, (os.cpu_count() or 1) + 4)
        self.current_threads = os.cpu_count() or 1
    
    def can_optimize(self, metrics: PerformanceMetrics) -> bool:
        """Check if thread pool can be optimized."""
        return (
            metrics.cpu_usage < 0.4 or  # CPU underutilized
            metrics.cpu_usage > 0.9     # CPU overutilized
        )
    
    def optimize(self, metrics: PerformanceMetrics) -> OptimizationResult:
        """Optimize thread pool size."""
        old_threads = self.current_threads
        
        if metrics.cpu_usage < 0.4 and self.current_threads > self.min_threads:
            # Reduce threads
            new_threads = max(self.current_threads - 1, self.min_threads)
            improvement = 0.05
            description = f"Reduced thread pool from {old_threads} to {new_threads}"
            
        elif metrics.cpu_usage > 0.9 and self.current_threads < self.max_threads:
            # Increase threads
            new_threads = min(self.current_threads + 2, self.max_threads)
            improvement = 0.15
            description = f"Increased thread pool from {old_threads} to {new_threads}"
            
        else:
            return OptimizationResult(
                strategy=self.name,
                applied=False,
                improvement=0.0,
                metrics_before=metrics,
                description="No thread pool optimization needed"
            )
        
        self.current_threads = new_threads
        
        return OptimizationResult(
            strategy=self.name,
            applied=True,
            improvement=improvement,
            metrics_before=metrics,
            description=description
        )


class MemoryOptimizer(BaseOptimizer):
    """Optimizer for memory usage."""
    
    def __init__(self, profiler: PerformanceProfiler):
        """Initialize memory optimizer."""
        super().__init__("memory", profiler)
        self.gc_enabled = True
    
    def can_optimize(self, metrics: PerformanceMetrics) -> bool:
        """Check if memory can be optimized."""
        return metrics.memory_usage > 0.7  # High memory usage
    
    def optimize(self, metrics: PerformanceMetrics) -> OptimizationResult:
        """Optimize memory usage."""
        if not self.can_optimize(metrics):
            return OptimizationResult(
                strategy=self.name,
                applied=False,
                improvement=0.0,
                metrics_before=metrics,
                description="No memory optimization needed"
            )
        
        # Force garbage collection
        import gc
        collected = gc.collect()
        
        # Clear PyTorch cache if available
        if HAS_TORCH and torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return OptimizationResult(
            strategy=self.name,
            applied=True,
            improvement=0.1,  # Estimated improvement
            metrics_before=metrics,
            description=f"Performed garbage collection, freed {collected} objects"
        )


class GPUOptimizer(BaseOptimizer):
    """Optimizer for GPU usage."""
    
    def __init__(self, profiler: PerformanceProfiler):
        """Initialize GPU optimizer."""
        super().__init__("gpu", profiler)
        self.mixed_precision_enabled = False
    
    def can_optimize(self, metrics: PerformanceMetrics) -> bool:
        """Check if GPU can be optimized."""
        return (
            HAS_TORCH and 
            torch.cuda.is_available() and
            metrics.gpu_usage is not None
        )
    
    def optimize(self, metrics: PerformanceMetrics) -> OptimizationResult:
        """Optimize GPU usage."""
        if not self.can_optimize(metrics):
            return OptimizationResult(
                strategy=self.name,
                applied=False,
                improvement=0.0,
                metrics_before=metrics,
                description="GPU optimization not available"
            )
        
        optimizations = []
        
        # Enable mixed precision if not already enabled
        if not self.mixed_precision_enabled:
            self.mixed_precision_enabled = True
            optimizations.append("Enabled mixed precision training")
        
        # Clear GPU cache
        if metrics.gpu_usage > 0.8:
            torch.cuda.empty_cache()
            optimizations.append("Cleared GPU cache")
        
        if not optimizations:
            return OptimizationResult(
                strategy=self.name,
                applied=False,
                improvement=0.0,
                metrics_before=metrics,
                description="No GPU optimization applied"
            )
        
        return OptimizationResult(
            strategy=self.name,
            applied=True,
            improvement=0.15,  # Estimated improvement
            metrics_before=metrics,
            description="; ".join(optimizations)
        )


class AutoScaler:
    """Automatic scaling system for PNO operations."""
    
    def __init__(
        self,
        profiler: PerformanceProfiler,
        strategy: OptimizationStrategy = OptimizationStrategy.BALANCED,
        optimization_interval: float = 60.0  # seconds
    ):
        """Initialize auto-scaler.
        
        Args:
            profiler: Performance profiler
            strategy: Optimization strategy
            optimization_interval: Time between optimization checks
        """
        self.profiler = profiler
        self.strategy = strategy
        self.optimization_interval = optimization_interval
        
        # Initialize optimizers
        self.optimizers: List[BaseOptimizer] = [
            BatchSizeOptimizer(profiler),
            ThreadPoolOptimizer(profiler),
            MemoryOptimizer(profiler),
            GPUOptimizer(profiler)
        ]
        
        # Auto-scaling state
        self.enabled = False
        self.scaling_thread: Optional[threading.Thread] = None
        self.optimization_history = []
        self._stop_event = threading.Event()
        
        # Performance thresholds based on strategy
        self._set_thresholds()
    
    def _set_thresholds(self):
        """Set performance thresholds based on strategy."""
        if self.strategy == OptimizationStrategy.AGGRESSIVE:
            self.performance_threshold = 0.05  # 5% degradation triggers optimization
            self.optimization_cooldown = 30.0   # 30 seconds between optimizations
        elif self.strategy == OptimizationStrategy.BALANCED:
            self.performance_threshold = 0.1   # 10% degradation
            self.optimization_cooldown = 60.0   # 1 minute
        elif self.strategy == OptimizationStrategy.CONSERVATIVE:
            self.performance_threshold = 0.2   # 20% degradation
            self.optimization_cooldown = 300.0  # 5 minutes
        else:  # ADAPTIVE
            self.performance_threshold = 0.1
            self.optimization_cooldown = 60.0
    
    def start(self):
        """Start auto-scaling."""
        if self.enabled:
            logger.warning("Auto-scaler already running")
            return
        
        self.enabled = True
        self._stop_event.clear()
        self.scaling_thread = threading.Thread(target=self._scaling_loop, daemon=True)
        self.scaling_thread.start()
        logger.info("Started auto-scaler")
    
    def stop(self):
        """Stop auto-scaling."""
        if not self.enabled:
            return
        
        self.enabled = False
        self._stop_event.set()
        
        if self.scaling_thread:
            self.scaling_thread.join(timeout=5.0)
        
        logger.info("Stopped auto-scaler")
    
    def _scaling_loop(self):
        """Main auto-scaling loop."""
        last_optimization = 0
        
        while not self._stop_event.wait(self.optimization_interval):
            try:
                current_time = time.time()
                
                # Check if enough time has passed since last optimization
                if current_time - last_optimization < self.optimization_cooldown:
                    continue
                
                # Get recent performance metrics
                summary = self.profiler.get_performance_summary()
                if not summary:
                    continue
                
                # Create synthetic metrics for optimization decisions
                current_metrics = PerformanceMetrics(
                    timestamp=current_time,
                    operation="auto_scaling",
                    duration=summary.get('average_duration', 0),
                    throughput=summary.get('average_throughput', 0),
                    memory_usage=summary.get('average_memory_usage', 0),
                    cpu_usage=summary.get('average_cpu_usage', 0)
                )
                
                # Check if optimization is needed
                if self._should_optimize(current_metrics):
                    self._perform_optimization(current_metrics)
                    last_optimization = current_time
                
            except Exception as e:
                logger.error(f"Error in auto-scaling loop: {e}")
    
    def _should_optimize(self, metrics: PerformanceMetrics) -> bool:
        """Determine if optimization should be performed."""
        # Check resource utilization
        high_memory = metrics.memory_usage > 0.8
        high_cpu = metrics.cpu_usage > 0.9
        low_throughput = metrics.throughput < 1.0  # Arbitrary threshold
        
        # Check for performance regression
        regression_detected = self.profiler.detect_performance_regression(
            "auto_scaling", self.performance_threshold
        )
        
        return high_memory or high_cpu or low_throughput or regression_detected
    
    def _perform_optimization(self, metrics: PerformanceMetrics):
        """Perform optimization using available optimizers."""
        applied_optimizations = []
        
        for optimizer in self.optimizers:
            if not optimizer.enabled:
                continue
            
            try:
                if optimizer.can_optimize(metrics):
                    result = optimizer.optimize(metrics)
                    if result.applied:
                        applied_optimizations.append(result)
                        logger.info(f"Applied {optimizer.name} optimization: {result.description}")
            
            except Exception as e:
                logger.error(f"Error in {optimizer.name} optimizer: {e}")
        
        # Record optimization history
        self.optimization_history.append({
            'timestamp': time.time(),
            'metrics': metrics.to_dict(),
            'optimizations': [opt.description for opt in applied_optimizations],
            'total_improvement': sum(opt.improvement for opt in applied_optimizations)
        })
        
        # Keep history bounded
        if len(self.optimization_history) > 1000:
            self.optimization_history.pop(0)
    
    def get_scaling_report(self) -> Dict[str, Any]:
        """Get auto-scaling performance report."""
        if not self.optimization_history:
            return {
                'enabled': self.enabled,
                'strategy': self.strategy.value,
                'optimizations_performed': 0
            }
        
        total_optimizations = len(self.optimization_history)
        total_improvement = sum(
            opt['total_improvement'] for opt in self.optimization_history
        )
        
        recent_optimizations = [
            opt for opt in self.optimization_history
            if time.time() - opt['timestamp'] < 3600  # Last hour
        ]
        
        optimizer_stats = {}
        for optimizer in self.optimizers:
            optimizer_stats[optimizer.name] = {
                'enabled': optimizer.enabled,
                'optimizations_applied': len(optimizer.optimization_history)
            }
        
        return {
            'enabled': self.enabled,
            'strategy': self.strategy.value,
            'total_optimizations': total_optimizations,
            'total_improvement_estimate': total_improvement,
            'recent_optimizations': len(recent_optimizations),
            'optimizer_stats': optimizer_stats,
            'performance_threshold': self.performance_threshold,
            'optimization_cooldown': self.optimization_cooldown
        }


class ParallelExecutor:
    """High-performance parallel execution system."""
    
    def __init__(
        self,
        max_workers: Optional[int] = None,
        use_processes: bool = False
    ):
        """Initialize parallel executor.
        
        Args:
            max_workers: Maximum number of workers
            use_processes: Whether to use processes instead of threads
        """
        self.max_workers = max_workers or os.cpu_count()
        self.use_processes = use_processes
        
        # Create executor
        if use_processes:
            self.executor = ProcessPoolExecutor(max_workers=self.max_workers)
        else:
            self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        
        # Performance tracking
        self.execution_stats = defaultdict(list)
    
    def parallel_map(
        self,
        func: Callable,
        iterable: List[Any],
        chunk_size: Optional[int] = None
    ) -> List[Any]:
        """Execute function in parallel over iterable.
        
        Args:
            func: Function to execute
            iterable: Input data
            chunk_size: Chunk size for processing
            
        Returns:
            List of results
        """
        start_time = time.time()
        
        if chunk_size is None:
            # Adaptive chunk sizing
            chunk_size = max(1, len(iterable) // (self.max_workers * 4))
        
        # Submit tasks
        futures = []
        for i in range(0, len(iterable), chunk_size):
            chunk = iterable[i:i + chunk_size]
            future = self.executor.submit(self._process_chunk, func, chunk)
            futures.append(future)
        
        # Collect results
        results = []
        for future in as_completed(futures):
            chunk_results = future.result()
            results.extend(chunk_results)
        
        # Record performance
        execution_time = time.time() - start_time
        throughput = len(iterable) / execution_time if execution_time > 0 else 0
        
        self.execution_stats[func.__name__].append({
            'execution_time': execution_time,
            'throughput': throughput,
            'input_size': len(iterable),
            'chunk_size': chunk_size,
            'workers_used': len(futures)
        })
        
        return results
    
    def _process_chunk(self, func: Callable, chunk: List[Any]) -> List[Any]:
        """Process a chunk of data."""
        return [func(item) for item in chunk]
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get parallel execution performance statistics."""
        stats = {}
        
        for func_name, executions in self.execution_stats.items():
            if executions:
                avg_time = sum(e['execution_time'] for e in executions) / len(executions)
                avg_throughput = sum(e['throughput'] for e in executions) / len(executions)
                total_items = sum(e['input_size'] for e in executions)
                
                stats[func_name] = {
                    'executions': len(executions),
                    'average_time': avg_time,
                    'average_throughput': avg_throughput,
                    'total_items_processed': total_items
                }
        
        return {
            'max_workers': self.max_workers,
            'use_processes': self.use_processes,
            'function_stats': stats
        }
    
    def shutdown(self):
        """Shutdown the executor."""
        self.executor.shutdown(wait=True)


# Global instances
global_profiler = PerformanceProfiler()
global_auto_scaler = AutoScaler(global_profiler)
global_parallel_executor = ParallelExecutor()


def enable_auto_scaling(strategy: OptimizationStrategy = OptimizationStrategy.BALANCED):
    """Enable global auto-scaling."""
    global_auto_scaler.strategy = strategy
    global_auto_scaler._set_thresholds()
    global_auto_scaler.start()


def disable_auto_scaling():
    """Disable global auto-scaling."""
    global_auto_scaler.stop()


def get_performance_report() -> Dict[str, Any]:
    """Get comprehensive performance report."""
    return {
        'profiler': global_profiler.get_performance_summary(),
        'auto_scaler': global_auto_scaler.get_scaling_report(),
        'parallel_executor': global_parallel_executor.get_performance_stats()
    }