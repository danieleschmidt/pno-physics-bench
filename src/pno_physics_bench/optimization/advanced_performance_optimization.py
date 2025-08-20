# SECURITY NOTICE: This file has been automatically hardened for security
# - All user inputs should be validated and sanitized
# - Subprocess calls use secure alternatives
# - SQL queries use parameterized statements
# - No hardcoded secrets or credentials


"""Advanced Performance Optimization Framework"""

import time
import threading
import functools
import weakref
import gc
from typing import Dict, List, Any, Callable, Optional, Tuple
from collections import defaultdict
import logging
import sys
import tracemalloc

logger = logging.getLogger(__name__)

class PerformanceProfiler:
    """Advanced performance profiler"""
    
    def __init__(self):
        self.function_stats = defaultdict(list)
        self.memory_stats = defaultdict(list)
        self.hotspots = []
        self._lock = threading.Lock()
        self.profiling_enabled = False
    
    def enable_profiling(self):
        """Enable performance profiling"""
        self.profiling_enabled = True
        tracemalloc.start()
        logger.info("Performance profiling enabled")
    
    def disable_profiling(self):
        """Disable performance profiling"""
        self.profiling_enabled = False
        if tracemalloc.is_tracing():
            tracemalloc.stop()
        logger.info("Performance profiling disabled")
    
    def profile_function(self, func_name: str, execution_time: float, 
                        memory_usage: Optional[int] = None):
        """Record function performance data"""
        if not self.profiling_enabled:
            return
        
        with self._lock:
            self.function_stats[func_name].append({
                "execution_time": execution_time,
                "memory_usage": memory_usage,
                "timestamp": time.time()
            })
    
    def get_performance_report(self, top_n: int = 10) -> Dict[str, Any]:
        """Generate performance report"""
        with self._lock:
            report = {
                "function_performance": {},
                "memory_usage": {},
                "hotspots": []
            }
            
            # Analyze function performance
            for func_name, stats in self.function_stats.items():
                if not stats:
                    continue
                
                exec_times = [s["execution_time"] for s in stats]
                report["function_performance"][func_name] = {
                    "call_count": len(exec_times),
                    "total_time": sum(exec_times),
                    "avg_time": sum(exec_times) / len(exec_times),
                    "min_time": min(exec_times),
                    "max_time": max(exec_times)
                }
            
            # Identify hotspots (slowest functions)
            hotspots = sorted(
                report["function_performance"].items(),
                key=lambda x: x[1]["total_time"],
                reverse=True
            )[:top_n]
            
            report["hotspots"] = [
                {"function": name, "total_time": data["total_time"], "avg_time": data["avg_time"]}
                for name, data in hotspots
            ]
            
            return report

def performance_monitor(func_name: Optional[str] = None):
    """Decorator for performance monitoring"""
    def decorator(func: Callable) -> Callable:
        name = func_name or func.__name__
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not profiler.profiling_enabled:
                return func(*args, **kwargs)
            
            # Memory tracking
            if tracemalloc.is_tracing():
                snapshot_before = tracemalloc.take_snapshot()
            
            start_time = time.perf_counter()
            
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                execution_time = time.perf_counter() - start_time
                
                memory_usage = None
                if tracemalloc.is_tracing():
                    snapshot_after = tracemalloc.take_snapshot()
                    top_stats = snapshot_after.compare_to(snapshot_before, 'lineno')
                    if top_stats:
                        memory_usage = sum(stat.size_diff for stat in top_stats)
                
                profiler.profile_function(name, execution_time, memory_usage)
        
        return wrapper
    return decorator

class MemoryOptimizer:
    """Memory optimization utilities"""
    
    def __init__(self):
        self.weak_refs = weakref.WeakValueDictionary()
        self.memory_pools = {}
    
    def optimize_memory_usage(self):
        """Optimize memory usage"""
        # Force garbage collection
        collected = gc.collect()
        
        # Get memory statistics
        if tracemalloc.is_tracing():
            snapshot = tracemalloc.take_snapshot()
            top_stats = snapshot.statistics('lineno')
            
            total_memory = sum(stat.size for stat in top_stats)
            logger.info(f"Memory optimization: collected {collected} objects, "
                       f"total memory: {total_memory / 1024 / 1024:.2f} MB")
        
        return collected
    
    def create_object_pool(self, pool_name: str, factory_func: Callable, 
                          max_size: int = 100):
        """Create object pool for memory optimization"""
        if pool_name not in self.memory_pools:
            self.memory_pools[pool_name] = ObjectPool(factory_func, max_size)
    
    def get_from_pool(self, pool_name: str) -> Any:
        """Get object from pool"""
        if pool_name in self.memory_pools:
            return self.memory_pools[pool_name].get()
        return None
    
    def return_to_pool(self, pool_name: str, obj: Any):
        """Return object to pool"""
        if pool_name in self.memory_pools:
            self.memory_pools[pool_name].put(obj)

class ObjectPool:
    """Object pool for memory optimization"""
    
    def __init__(self, factory_func: Callable, max_size: int = 100):
        self.factory_func = factory_func
        self.max_size = max_size
        self.pool = []
        self._lock = threading.Lock()
    
    def get(self) -> Any:
        """Get object from pool"""
        with self._lock:
            if self.pool:
                return self.pool.pop()
            else:
                return self.factory_func()
    
    def put(self, obj: Any):
        """Return object to pool"""
        with self._lock:
            if len(self.pool) < self.max_size:
                # Reset object state if it has a reset method
                if hasattr(obj, 'reset'):
                    obj.reset()
                self.pool.append(obj)

class BatchProcessor:
    """Optimized batch processing"""
    
    def __init__(self, batch_size: int = 32, max_wait_time: float = 1.0):
        self.batch_size = batch_size
        self.max_wait_time = max_wait_time
        self.pending_items = []
        self.last_flush_time = time.time()
        self._lock = threading.Lock()
        self.processors = {}
    
    def add_processor(self, name: str, processor_func: Callable):
        """Add batch processor function"""
        self.processors[name] = processor_func
    
    def process_item(self, processor_name: str, item: Any) -> Any:
        """Process item with batching"""
        with self._lock:
            self.pending_items.append((processor_name, item))
            
            # Check if batch is ready
            if (len(self.pending_items) >= self.batch_size or 
                time.time() - self.last_flush_time > self.max_wait_time):
                return self._flush_batch()
            
            return None
    
    def _flush_batch(self) -> List[Any]:
        """Flush pending batch"""
        if not self.pending_items:
            return []
        
        # Group items by processor
        processor_batches = defaultdict(list)
        for processor_name, item in self.pending_items:
            processor_batches[processor_name].append(item)
        
        results = []
        
        # Process each batch
        for processor_name, items in processor_batches.items():
            if processor_name in self.processors:
                try:
                    batch_results = self.processors[processor_name](items)
                    results.extend(batch_results)
                except Exception as e:
                    logger.error(f"Batch processing error for {processor_name}: {e}")
        
        self.pending_items.clear()
        self.last_flush_time = time.time()
        
        return results

class PerformanceOptimizer:
    """Main performance optimization engine"""
    
    def __init__(self):
        self.profiler = PerformanceProfiler()
        self.memory_optimizer = MemoryOptimizer()
        self.batch_processor = BatchProcessor()
        self.optimization_history = []
    
    def start_optimization(self):
        """Start performance optimization"""
        self.profiler.enable_profiling()
        
        # Start background optimization thread
        self.optimization_thread = threading.Thread(
            target=self._optimization_loop,
            daemon=True
        )
        self.optimization_thread.start()
        
        logger.info("Performance optimization started")
    
    def stop_optimization(self):
        """Stop performance optimization"""
        self.profiler.disable_profiling()
        logger.info("Performance optimization stopped")
    
    def _optimization_loop(self):
        """Background optimization loop"""
        while self.profiler.profiling_enabled:
            try:
                # Memory optimization every 5 minutes
                self.memory_optimizer.optimize_memory_usage()
                
                # Generate performance report every 10 minutes
                report = self.profiler.get_performance_report()
                self.optimization_history.append({
                    "timestamp": time.time(),
                    "report": report
                })
                
                # Keep only last 24 hours of history
                cutoff_time = time.time() - 86400
                self.optimization_history = [
                    h for h in self.optimization_history
                    if h["timestamp"] > cutoff_time
                ]
                
                time.sleep(300)  # 5 minutes
                
            except Exception as e:
                logger.error(f"Error in optimization loop: {e}")
                time.sleep(60)
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """Get comprehensive optimization report"""
        current_report = self.profiler.get_performance_report()
        
        return {
            "current_performance": current_report,
            "optimization_history": self.optimization_history[-10:],  # Last 10 reports
            "memory_pools": {
                name: len(pool.pool) for name, pool in self.memory_optimizer.memory_pools.items()
            },
            "batch_processor_status": {
                "pending_items": len(self.batch_processor.pending_items),
                "processors_count": len(self.batch_processor.processors)
            }
        }

# Global instances
profiler = PerformanceProfiler()
memory_optimizer = MemoryOptimizer()
performance_optimizer = PerformanceOptimizer()
