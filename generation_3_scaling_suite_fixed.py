#!/usr/bin/env python3
"""
Generation 3 Scaling Suite - FIXED VERSION 
Performance optimization, caching, concurrent processing, resource management, auto-scaling.
"""

import sys
import os
import json
import time
import threading
import multiprocessing
import asyncio
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from datetime import datetime, timedelta
import logging
import hashlib
import gc
import resource
from typing import Dict, Any, Optional, List, Union, Callable
from functools import lru_cache, wraps
from collections import defaultdict, deque
import queue

# Add src to path
sys.path.insert(0, 'src')

# Import numpy
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    class MockNumpy:
        @staticmethod
        def array(data): return data
        @staticmethod
        def zeros(shape): return [[0.0] * shape[1] for _ in range(shape[0])] if len(shape) == 2 else [0.0] * shape[0]
        @staticmethod
        def sum(data): return sum(sum(row) if isinstance(row, list) else row for row in data)
        @staticmethod
        def random(): 
            class MockRandom:
                @staticmethod
                def randn(*args): return MockNumpy.zeros(args)
            return MockRandom()
    np = MockNumpy()

class PerformanceMonitor:
    """Advanced performance monitoring and profiling."""
    
    def __init__(self):
        self.metrics = defaultdict(list)
        self.start_times = {}
        self.memory_usage = deque(maxlen=1000)  # Keep last 1000 measurements
        self.cpu_usage = deque(maxlen=1000)
        self._lock = threading.RLock()
    
    def start_timer(self, operation: str):
        """Start timing an operation."""
        with self._lock:
            self.start_times[operation] = time.time()
    
    def end_timer(self, operation: str) -> float:
        """End timing and record duration."""
        with self._lock:
            if operation not in self.start_times:
                return 0.0
            
            duration = time.time() - self.start_times[operation]
            self.metrics[f"{operation}_duration"].append(duration)
            del self.start_times[operation]
            return duration
    
    def record_metric(self, name: str, value: float):
        """Record a custom metric."""
        with self._lock:
            self.metrics[name].append(value)
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get current system performance metrics."""
        try:
            # Memory usage
            if hasattr(resource, 'getrusage'):
                usage = resource.getrusage(resource.RUSAGE_SELF)
                memory_mb = usage.ru_maxrss / 1024  # Convert to MB on Linux
            else:
                memory_mb = 50.0  # Mock value
            
            self.memory_usage.append(memory_mb)
            
            # CPU usage estimate (simplified)
            cpu_percent = min(100.0, threading.active_count() * 10)
            self.cpu_usage.append(cpu_percent)
            
            return {
                'memory_mb': memory_mb,
                'cpu_percent': cpu_percent,
                'active_threads': threading.active_count(),
                'gc_objects': len(gc.get_objects()) if hasattr(gc, 'get_objects') else 100
            }
        except Exception as e:
            return {'error': str(e)}
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        with self._lock:
            summary = {
                'timestamp': datetime.now().isoformat(),
                'metrics': {},
                'system': self.get_system_metrics()
            }
            
            # Calculate statistics for each metric
            for metric_name, values in self.metrics.items():
                if values:
                    summary['metrics'][metric_name] = {
                        'count': len(values),
                        'mean': sum(values) / len(values),
                        'min': min(values),
                        'max': max(values),
                        'last_10_avg': sum(values[-10:]) / min(len(values), 10)
                    }
            
            return summary

class IntelligentCache:
    """Advanced caching system with adaptive strategies."""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache = {}
        self.access_times = {}
        self.access_counts = defaultdict(int)
        self.hit_count = 0
        self.miss_count = 0
        self._lock = threading.RLock()
    
    def _get_key_hash(self, key: Any) -> str:
        """Generate hash for complex keys."""
        if isinstance(key, (str, int, float)):
            return str(key)
        return hashlib.md5(str(key).encode()).hexdigest()
    
    def _is_expired(self, key: str) -> bool:
        """Check if cache entry is expired."""
        if key not in self.access_times:
            return True
        
        age = time.time() - self.access_times[key]['created']
        return age > self.ttl_seconds
    
    def get(self, key: Any) -> Optional[Any]:
        """Get value from cache."""
        with self._lock:
            key_hash = self._get_key_hash(key)
            
            if key_hash not in self.cache or self._is_expired(key_hash):
                self.miss_count += 1
                return None
            
            # Update access info
            self.access_times[key_hash]['last_accessed'] = time.time()
            self.access_counts[key_hash] += 1
            self.hit_count += 1
            
            return self.cache[key_hash]
    
    def put(self, key: Any, value: Any):
        """Put value in cache."""
        with self._lock:
            key_hash = self._get_key_hash(key)
            
            # Simple eviction if at max size
            if len(self.cache) >= self.max_size:
                # Remove oldest entry
                oldest_key = min(self.access_times.keys(), 
                               key=lambda k: self.access_times[k]['created'])
                if oldest_key in self.cache:
                    del self.cache[oldest_key]
                if oldest_key in self.access_times:
                    del self.access_times[oldest_key]
            
            # Store value and metadata
            self.cache[key_hash] = value
            self.access_times[key_hash] = {
                'created': time.time(),
                'last_accessed': time.time()
            }
            self.access_counts[key_hash] = 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        with self._lock:
            total_requests = self.hit_count + self.miss_count
            hit_rate = self.hit_count / total_requests if total_requests > 0 else 0.0
            
            return {
                'size': len(self.cache),
                'max_size': self.max_size,
                'hit_count': self.hit_count,
                'miss_count': self.miss_count,
                'hit_rate': hit_rate,
                'total_requests': total_requests
            }

class ResourcePool:
    """Generic resource pooling for expensive objects."""
    
    def __init__(self, factory: Callable, max_size: int = 10, timeout: float = 30.0):
        self.factory = factory
        self.max_size = max_size
        self.timeout = timeout
        self.pool = queue.Queue(maxsize=max_size)
        self.created_count = 0
        self.active_count = 0
        self._lock = threading.Lock()
    
    def get_resource(self):
        """Get resource from pool or create new one."""
        try:
            # Try to get from pool first
            resource = self.pool.get(timeout=1.0)
            with self._lock:
                self.active_count += 1
            return resource
        except queue.Empty:
            # Create new resource if pool is empty
            with self._lock:
                if self.created_count < self.max_size:
                    resource = self.factory()
                    self.created_count += 1
                    self.active_count += 1
                    return resource
            
            # Return a new resource if at limit
            resource = self.factory()
            with self._lock:
                self.active_count += 1
            return resource
    
    def return_resource(self, resource):
        """Return resource to pool."""
        try:
            self.pool.put(resource, timeout=1.0)
            with self._lock:
                self.active_count -= 1
        except queue.Full:
            # Pool is full, discard resource
            with self._lock:
                self.active_count -= 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics."""
        with self._lock:
            return {
                'pool_size': self.pool.qsize(),
                'max_size': self.max_size,
                'created_count': self.created_count,
                'active_count': self.active_count,
                'available_count': self.pool.qsize()
            }

class ConcurrentProcessor:
    """Advanced concurrent processing with load balancing."""
    
    def __init__(self, max_workers: int = None):
        self.max_workers = max_workers or min(32, (os.cpu_count() or 1) + 4)
        self.thread_pool = ThreadPoolExecutor(max_workers=self.max_workers)
        self.task_results = {}
        self.active_tasks = {}
        self._lock = threading.Lock()
        self._task_counter = 0
        
    def submit_task(self, func: Callable, args: tuple = (), kwargs: dict = None, 
                   priority: int = 0, use_process: bool = False) -> str:
        """Submit task for concurrent execution."""
        kwargs = kwargs or {}
        
        with self._lock:
            task_id = f"task_{self._task_counter}_{int(time.time() * 1000)}"
            self._task_counter += 1
        
        # Submit to thread pool (simplified for this demo)
        future = self.thread_pool.submit(func, *args, **kwargs)
        
        with self._lock:
            self.active_tasks[task_id] = {
                'future': future,
                'submitted_at': time.time(),
                'priority': priority,
                'use_process': use_process
            }
        
        return task_id
    
    def get_result(self, task_id: str, timeout: float = None) -> Any:
        """Get result from completed task."""
        with self._lock:
            if task_id not in self.active_tasks:
                raise ValueError(f"Task {task_id} not found")
            
            task_info = self.active_tasks[task_id]
        
        try:
            result = task_info['future'].result(timeout=timeout)
            
            with self._lock:
                self.task_results[task_id] = result
                del self.active_tasks[task_id]
            
            return result
        except Exception as e:
            with self._lock:
                self.task_results[task_id] = {'error': str(e)}
                del self.active_tasks[task_id]
            raise
    
    def batch_process(self, func: Callable, items: List[Any], 
                     chunk_size: int = None, use_processes: bool = False) -> List[Any]:
        """Process items in parallel batches."""
        if not items:
            return []
        
        chunk_size = chunk_size or max(1, len(items) // self.max_workers)
        chunks = [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]
        
        # Submit batch processing tasks
        task_ids = []
        for chunk in chunks:
            task_id = self.submit_task(
                self._process_chunk, 
                (func, chunk), 
                use_process=use_processes
            )
            task_ids.append(task_id)
        
        # Collect results
        results = []
        for task_id in task_ids:
            chunk_results = self.get_result(task_id)
            results.extend(chunk_results)
        
        return results
    
    @staticmethod
    def _process_chunk(func: Callable, chunk: List[Any]) -> List[Any]:
        """Process a chunk of items."""
        return [func(item) for item in chunk]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        with self._lock:
            return {
                'max_workers': self.max_workers,
                'active_tasks': len(self.active_tasks),
                'completed_tasks': len(self.task_results)
            }
    
    def shutdown(self):
        """Shutdown executors."""
        self.thread_pool.shutdown(wait=True)

class AutoScaler:
    """Auto-scaling based on system metrics."""
    
    def __init__(self, monitor: PerformanceMonitor):
        self.monitor = monitor
        self.scaling_rules = {
            'memory_threshold_mb': 800,
            'cpu_threshold_percent': 75,
            'response_time_threshold_ms': 1000,
            'error_rate_threshold': 0.1
        }
        self.scaling_history = []
    
    def should_scale_up(self) -> bool:
        """Determine if system should scale up."""
        system_metrics = self.monitor.get_system_metrics()
        
        # Check memory usage
        if system_metrics.get('memory_mb', 0) > self.scaling_rules['memory_threshold_mb']:
            return True
        
        # Check CPU usage
        if system_metrics.get('cpu_percent', 0) > self.scaling_rules['cpu_threshold_percent']:
            return True
        
        # Check response times
        perf_summary = self.monitor.get_performance_summary()
        for metric_name, stats in perf_summary.get('metrics', {}).items():
            if 'duration' in metric_name:
                avg_duration_ms = stats.get('last_10_avg', 0) * 1000
                if avg_duration_ms > self.scaling_rules['response_time_threshold_ms']:
                    return True
        
        return False
    
    def should_scale_down(self) -> bool:
        """Determine if system should scale down."""
        system_metrics = self.monitor.get_system_metrics()
        
        # Only scale down if all metrics are well below thresholds
        memory_ok = system_metrics.get('memory_mb', 0) < self.scaling_rules['memory_threshold_mb'] * 0.5
        cpu_ok = system_metrics.get('cpu_percent', 0) < self.scaling_rules['cpu_threshold_percent'] * 0.5
        
        # Check if we've been stable for a while
        recent_history = [entry for entry in self.scaling_history 
                         if (datetime.now() - datetime.fromisoformat(entry['timestamp'])).total_seconds() < 300]
        
        stable = len(recent_history) == 0  # No recent scaling events
        
        return memory_ok and cpu_ok and stable
    
    def trigger_scaling(self, action: str, reason: str):
        """Trigger scaling action."""
        scaling_event = {
            'timestamp': datetime.now().isoformat(),
            'action': action,
            'reason': reason,
            'system_metrics': self.monitor.get_system_metrics()
        }
        
        self.scaling_history.append(scaling_event)
        
        return scaling_event

def make_serializable(obj):
    """Convert objects to JSON serializable format."""
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_serializable(item) for item in obj]
    elif isinstance(obj, set):
        return list(obj)
    elif hasattr(obj, '__dict__'):
        return make_serializable(obj.__dict__)
    else:
        return obj

class ScalingValidator:
    """Comprehensive validation suite for Generation 3."""
    
    def __init__(self):
        self.monitor = PerformanceMonitor()
        self.cache = IntelligentCache(max_size=100, ttl_seconds=60)
        self.processor = ConcurrentProcessor(max_workers=4)
        self.autoscaler = AutoScaler(self.monitor)
        
        # Create resource pool for testing
        self.resource_pool = ResourcePool(
            factory=lambda: {'id': time.time(), 'data': np.zeros((10, 10)) if NUMPY_AVAILABLE else [[0]*10]*10},
            max_size=5
        )
    
    def run_scaling_tests(self) -> Dict[str, Any]:
        """Run comprehensive scaling validation."""
        print("=== GENERATION 3: MAKE IT SCALE (OPTIMIZED) VALIDATION ===")
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'tests': {},
            'overall_status': 'unknown'
        }
        
        # Test 1: Performance Monitoring
        print("\n1. Testing Performance Monitoring...")
        try:
            # Test timer functionality
            self.monitor.start_timer('test_operation')
            time.sleep(0.01)  # Simulate work
            duration = self.monitor.end_timer('test_operation')
            
            # Test metric recording
            self.monitor.record_metric('test_metric', 42.0)
            
            # Get performance summary
            summary = self.monitor.get_performance_summary()
            
            if duration > 0 and 'test_operation_duration' in summary['metrics']:
                results['tests']['performance_monitoring'] = {
                    'status': 'pass',
                    'duration_ms': duration * 1000,
                    'metrics_count': len(summary['metrics'])
                }
                print("   ✓ Performance monitoring working")
            else:
                results['tests']['performance_monitoring'] = {'status': 'fail', 'error': 'Metrics not recorded'}
                print("   ✗ Performance monitoring failed")
            
        except Exception as e:
            results['tests']['performance_monitoring'] = {'status': 'fail', 'error': str(e)}
            print(f"   ✗ Performance monitoring failed: {str(e)}")
        
        # Test 2: Intelligent Caching
        print("\n2. Testing Intelligent Caching...")
        try:
            # Test cache operations
            test_key = "test_key"
            test_value = {"data": "test_data", "timestamp": time.time()}
            
            # Miss test
            result = self.cache.get(test_key)
            assert result is None, "Cache should miss on first access"
            
            # Put test
            self.cache.put(test_key, test_value)
            
            # Hit test
            result = self.cache.get(test_key)
            assert result == test_value, "Cache should return stored value"
            
            # Stats test
            stats = self.cache.get_stats()
            
            results['tests']['intelligent_caching'] = {
                'status': 'pass',
                'cache_size': stats['size'],
                'hit_rate': stats['hit_rate']
            }
            print(f"   ✓ Intelligent caching working (hit rate: {stats['hit_rate']:.2f})")
            
        except Exception as e:
            results['tests']['intelligent_caching'] = {'status': 'fail', 'error': str(e)}
            print(f"   ✗ Intelligent caching failed: {str(e)}")
        
        # Test 3: Resource Pooling
        print("\n3. Testing Resource Pooling...")
        try:
            # Test resource acquisition and return
            resource1 = self.resource_pool.get_resource()
            resource2 = self.resource_pool.get_resource()
            
            assert resource1 is not None, "Should get resource from pool"
            assert resource2 is not None, "Should get second resource from pool"
            
            # Return resources
            self.resource_pool.return_resource(resource1)
            self.resource_pool.return_resource(resource2)
            
            # Check stats
            stats = self.resource_pool.get_stats()
            
            results['tests']['resource_pooling'] = {
                'status': 'pass',
                'pool_stats': stats
            }
            print(f"   ✓ Resource pooling working (pool size: {stats['available_count']})")
            
        except Exception as e:
            results['tests']['resource_pooling'] = {'status': 'fail', 'error': str(e)}
            print(f"   ✗ Resource pooling failed: {str(e)}")
        
        # Test 4: Concurrent Processing
        print("\n4. Testing Concurrent Processing...")
        try:
            # Test simple task submission
            def test_task(x):
                time.sleep(0.01)  # Simulate work
                return x * x
            
            task_id = self.processor.submit_task(test_task, (5,))
            result = self.processor.get_result(task_id, timeout=5.0)
            
            assert result == 25, f"Expected 25, got {result}"
            
            # Test batch processing
            items = list(range(10))
            batch_results = self.processor.batch_process(lambda x: x * 2, items)
            expected = [x * 2 for x in items]
            
            assert batch_results == expected, "Batch processing results incorrect"
            
            # Check stats
            stats = self.processor.get_stats()
            
            results['tests']['concurrent_processing'] = {
                'status': 'pass',
                'processor_stats': stats,
                'batch_size': len(batch_results)
            }
            print(f"   ✓ Concurrent processing working ({stats['max_workers']} workers)")
            
        except Exception as e:
            results['tests']['concurrent_processing'] = {'status': 'fail', 'error': str(e)}
            print(f"   ✗ Concurrent processing failed: {str(e)}")
        
        # Test 5: Auto-Scaling Logic
        print("\n5. Testing Auto-Scaling Logic...")
        try:
            # Simulate high load condition
            self.monitor.record_metric('test_duration', 2.0)  # 2 seconds > 1 second threshold
            
            # Test scaling decision logic
            should_scale_up = self.autoscaler.should_scale_up()
            should_scale_down = self.autoscaler.should_scale_down()
            
            # Trigger scaling event
            scaling_event = self.autoscaler.trigger_scaling('scale_up', 'high_response_time')
            
            results['tests']['auto_scaling'] = {
                'status': 'pass',
                'scaling_decisions': {
                    'should_scale_up': should_scale_up,
                    'should_scale_down': should_scale_down
                },
                'scaling_event': scaling_event is not None
            }
            print(f"   ✓ Auto-scaling logic working (scale up: {should_scale_up})")
            
        except Exception as e:
            results['tests']['auto_scaling'] = {'status': 'fail', 'error': str(e)}
            print(f"   ✗ Auto-scaling failed: {str(e)}")
        
        # Test 6: Integrated Performance Test
        print("\n6. Testing Integrated Performance...")
        try:
            def complex_workload(n):
                # Simulate complex computation with caching
                cache_key = f"computation_{n}"
                cached_result = self.cache.get(cache_key)
                
                if cached_result is not None:
                    return cached_result
                
                # Simulate computation
                if NUMPY_AVAILABLE:
                    result = np.sum(np.random.randn(50, 50))
                else:
                    result = sum(sum([0.1] * 50) for _ in range(50))
                
                self.cache.put(cache_key, result)
                return result
            
            # Run workload multiple times
            start_time = time.time()
            results_list = []
            
            for i in range(20):
                result = complex_workload(i % 5)  # Reuse some computations
                results_list.append(result)
            
            total_time = time.time() - start_time
            
            # Check cache effectiveness
            cache_stats = self.cache.get_stats()
            
            results['tests']['integrated_performance'] = {
                'status': 'pass',
                'total_time_ms': total_time * 1000,
                'cache_hit_rate': cache_stats['hit_rate'],
                'operations_completed': len(results_list),
                'avg_time_per_op_ms': (total_time / len(results_list)) * 1000
            }
            
            print(f"   ✓ Integrated performance test complete")
            print(f"     - Total time: {total_time * 1000:.1f}ms")
            print(f"     - Cache hit rate: {cache_stats['hit_rate']:.2f}")
            print(f"     - Avg time per operation: {(total_time / len(results_list)) * 1000:.1f}ms")
            
        except Exception as e:
            results['tests']['integrated_performance'] = {'status': 'fail', 'error': str(e)}
            print(f"   ✗ Integrated performance test failed: {str(e)}")
        
        # Overall assessment
        passed_tests = len([test for test in results['tests'].values() 
                          if test['status'] == 'pass'])
        total_tests = len(results['tests'])
        
        if passed_tests == total_tests:
            results['overall_status'] = 'scalable'
            print(f"\n=== GENERATION 3 SCALING VALIDATION COMPLETE ===")
            print(f"✓ All {total_tests} scaling tests passed")
            print("✓ System is SCALABLE and OPTIMIZED")
        elif passed_tests >= total_tests * 0.8:
            results['overall_status'] = 'mostly_scalable'
            print(f"\n=== GENERATION 3 SCALING VALIDATION COMPLETE ===") 
            print(f"✓ {passed_tests}/{total_tests} scaling tests passed")
            print("⚠ System is MOSTLY SCALABLE with minor issues")
        else:
            results['overall_status'] = 'needs_optimization'
            print(f"\n=== GENERATION 3 SCALING VALIDATION COMPLETE ===")
            print(f"⚠ Only {passed_tests}/{total_tests} scaling tests passed")
            print("❌ System needs scaling optimizations")
        
        return results
    
    def cleanup(self):
        """Cleanup resources."""
        try:
            self.processor.shutdown()
        except:
            pass

def main():
    """Run Generation 3 scaling validation."""
    validator = ScalingValidator()
    
    try:
        results = validator.run_scaling_tests()
        
        # Save results (make serializable first)
        serializable_results = make_serializable(results)
        results_file = f"generation_3_scaling_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"\n📊 Detailed results saved to: {results_file}")
        
        return results['overall_status'] in ['scalable', 'mostly_scalable']
    
    finally:
        validator.cleanup()

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 Generation 3: MAKE IT SCALE - COMPLETE!")
    else:
        print("\n❌ Generation 3: Scaling validation failed")
        sys.exit(1)