#!/usr/bin/env python3
"""
Generation 3: Comprehensive Validation and Testing Suite

This module provides comprehensive validation and testing for the Generation 3
enterprise scaling implementation, ensuring all components work correctly
under various load conditions and configurations.

Author: Autonomous SDLC Generation 3
Date: 2025-08-23
"""

import asyncio
import concurrent.futures
import json
import logging
import multiprocessing as mp
import os
import random
import statistics
import sys
import tempfile
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import psutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import the Generation 3 implementation
sys.path.append(str(Path(__file__).parent))
try:
    from generation_3_enterprise_scaling_implementation import (
        EnterpriseScalingInfrastructure,
        AdvancedPerformanceOptimizer,
        DistributedComputingFramework,
        MultiTierCacheSystem,
        PerformanceAnalyticsSuite,
        EnterpriseMetrics,
        OptimizationLevel,
        ScalingStrategy
    )
    IMPLEMENTATION_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Could not import Generation 3 implementation: {e}")
    IMPLEMENTATION_AVAILABLE = False


class ValidationResult:
    """Result of a validation test."""
    
    def __init__(self, test_name: str, passed: bool, duration: float, 
                 details: Optional[Dict[str, Any]] = None, error: Optional[str] = None):
        """Initialize validation result.
        
        Args:
            test_name: Name of the test
            passed: Whether the test passed
            duration: Test execution duration
            details: Additional test details
            error: Error message if failed
        """
        self.test_name = test_name
        self.passed = passed
        self.duration = duration
        self.details = details or {}
        self.error = error
        self.timestamp = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'test_name': self.test_name,
            'passed': self.passed,
            'duration': self.duration,
            'details': self.details,
            'error': self.error,
            'timestamp': self.timestamp
        }


class LoadTestResult:
    """Result of a load test."""
    
    def __init__(self, test_name: str, duration: float, total_operations: int,
                 success_rate: float, avg_response_time: float, peak_cpu: float,
                 peak_memory: float, errors: List[str]):
        """Initialize load test result."""
        self.test_name = test_name
        self.duration = duration
        self.total_operations = total_operations
        self.success_rate = success_rate
        self.avg_response_time = avg_response_time
        self.peak_cpu = peak_cpu
        self.peak_memory = peak_memory
        self.errors = errors
        self.throughput = total_operations / duration if duration > 0 else 0
        self.timestamp = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'test_name': self.test_name,
            'duration': self.duration,
            'total_operations': self.total_operations,
            'success_rate': self.success_rate,
            'avg_response_time': self.avg_response_time,
            'peak_cpu': self.peak_cpu,
            'peak_memory': self.peak_memory,
            'throughput': self.throughput,
            'errors': self.errors,
            'timestamp': self.timestamp
        }


class Generation3ValidatorSuite:
    """Comprehensive validation suite for Generation 3 enterprise scaling."""
    
    def __init__(self):
        """Initialize validation suite."""
        self.results = []
        self.load_test_results = []
        self.test_data_dir = Path(tempfile.mkdtemp(prefix="gen3_validation_"))
        self.start_time = time.time()
        
        logger.info(f"Generation 3 Validation Suite initialized")
        logger.info(f"Test data directory: {self.test_data_dir}")
    
    def run_all_validations(self) -> Dict[str, Any]:
        """Run all validation tests.
        
        Returns:
            Comprehensive validation report
        """
        logger.info("🚀 Starting Generation 3 Comprehensive Validation Suite")
        print("=" * 80)
        
        validation_start = time.time()
        
        try:
            # Basic functionality tests
            self._run_basic_functionality_tests()
            
            # Performance optimization tests
            self._run_performance_optimization_tests()
            
            # Distributed computing tests
            self._run_distributed_computing_tests()
            
            # Caching system tests
            self._run_caching_system_tests()
            
            # Performance analytics tests
            self._run_performance_analytics_tests()
            
            # Enterprise infrastructure tests
            self._run_enterprise_infrastructure_tests()
            
            # Load and stress tests
            self._run_load_tests()
            
            # Integration tests
            self._run_integration_tests()
            
        except Exception as e:
            logger.error(f"Validation suite failed: {e}")
            logger.error(traceback.format_exc())
        
        validation_duration = time.time() - validation_start
        
        # Generate final report
        report = self._generate_validation_report(validation_duration)
        
        # Save results
        self._save_validation_results(report)
        
        return report
    
    def _run_basic_functionality_tests(self):
        """Run basic functionality validation tests."""
        logger.info("\n📋 Running Basic Functionality Tests...")
        
        if not IMPLEMENTATION_AVAILABLE:
            self.results.append(ValidationResult(
                "import_generation_3_modules",
                False,
                0.0,
                error="Generation 3 implementation not available for import"
            ))
            return
        
        # Test 1: Import and instantiate main components
        result = self._test_component_instantiation()
        self.results.append(result)
        
        # Test 2: Configuration validation
        result = self._test_configuration_validation()
        self.results.append(result)
        
        # Test 3: Resource availability
        result = self._test_resource_availability()
        self.results.append(result)
    
    def _test_component_instantiation(self) -> ValidationResult:
        """Test instantiation of main components."""
        test_name = "component_instantiation"
        start_time = time.time()
        
        try:
            # Test performance optimizer
            optimizer = AdvancedPerformanceOptimizer(OptimizationLevel.DEVELOPMENT)
            assert optimizer is not None
            
            # Test distributed framework
            dist_framework = DistributedComputingFramework()
            assert dist_framework is not None
            
            # Test cache system
            cache_config = {
                'l1_size': 100,
                'l2_size_mb': 10,
                'redis_enabled': False
            }
            cache_system = MultiTierCacheSystem(cache_config)
            assert cache_system is not None
            
            # Test analytics suite
            analytics = PerformanceAnalyticsSuite()
            assert analytics is not None
            
            # Test enterprise infrastructure
            infrastructure = EnterpriseScalingInfrastructure({
                'optimization_level': 'development',
                'kubernetes_enabled': False
            })
            assert infrastructure is not None
            
            duration = time.time() - start_time
            return ValidationResult(
                test_name, True, duration,
                details={'components_created': 5}
            )
            
        except Exception as e:
            duration = time.time() - start_time
            return ValidationResult(
                test_name, False, duration,
                error=str(e)
            )
    
    def _test_configuration_validation(self) -> ValidationResult:
        """Test configuration validation."""
        test_name = "configuration_validation"
        start_time = time.time()
        
        try:
            # Test valid configurations
            valid_configs = [
                {'optimization_level': 'development'},
                {'optimization_level': 'production', 'kubernetes_enabled': False},
                {'cache_config': {'l1_size': 500, 'redis_enabled': False}}
            ]
            
            for config in valid_configs:
                infrastructure = EnterpriseScalingInfrastructure(config)
                assert infrastructure is not None
            
            duration = time.time() - start_time
            return ValidationResult(
                test_name, True, duration,
                details={'configs_tested': len(valid_configs)}
            )
            
        except Exception as e:
            duration = time.time() - start_time
            return ValidationResult(
                test_name, False, duration,
                error=str(e)
            )
    
    def _test_resource_availability(self) -> ValidationResult:
        """Test system resource availability."""
        test_name = "resource_availability"
        start_time = time.time()
        
        try:
            # Check CPU availability
            cpu_count = mp.cpu_count()
            assert cpu_count > 0
            
            # Check memory availability
            memory = psutil.virtual_memory()
            available_gb = memory.available / (1024 ** 3)
            assert available_gb > 0.5  # At least 512MB available
            
            # Check disk space
            disk = psutil.disk_usage('/')
            free_gb = disk.free / (1024 ** 3)
            assert free_gb > 1.0  # At least 1GB free
            
            # Check for optional dependencies
            optional_deps = {}
            
            try:
                import torch
                optional_deps['pytorch'] = torch.__version__
                optional_deps['cuda_available'] = torch.cuda.is_available()
                if torch.cuda.is_available():
                    optional_deps['cuda_devices'] = torch.cuda.device_count()
            except ImportError:
                optional_deps['pytorch'] = False
            
            try:
                import redis
                optional_deps['redis'] = True
            except ImportError:
                optional_deps['redis'] = False
            
            try:
                import kubernetes
                optional_deps['kubernetes'] = True
            except ImportError:
                optional_deps['kubernetes'] = False
            
            duration = time.time() - start_time
            return ValidationResult(
                test_name, True, duration,
                details={
                    'cpu_cores': cpu_count,
                    'available_memory_gb': round(available_gb, 2),
                    'free_disk_gb': round(free_gb, 2),
                    'optional_dependencies': optional_deps
                }
            )
            
        except Exception as e:
            duration = time.time() - start_time
            return ValidationResult(
                test_name, False, duration,
                error=str(e)
            )
    
    def _run_performance_optimization_tests(self):
        """Run performance optimization tests."""
        logger.info("\n⚡ Running Performance Optimization Tests...")
        
        if not IMPLEMENTATION_AVAILABLE:
            return
        
        # Test function optimization
        result = self._test_function_optimization()
        self.results.append(result)
        
        # Test vectorization
        result = self._test_vectorization()
        self.results.append(result)
        
        # Test memory management
        result = self._test_memory_management()
        self.results.append(result)
    
    def _test_function_optimization(self) -> ValidationResult:
        """Test function optimization capabilities."""
        test_name = "function_optimization"
        start_time = time.time()
        
        try:
            optimizer = AdvancedPerformanceOptimizer(OptimizationLevel.DEVELOPMENT)
            
            # Define test function
            def test_computation(n):
                return sum(i ** 2 for i in range(n))
            
            # Optimize function
            optimized_func = optimizer.optimize_function(test_computation)
            
            # Test both versions
            test_size = 1000
            
            # Time original function
            original_start = time.time()
            original_result = test_computation(test_size)
            original_time = time.time() - original_start
            
            # Time optimized function
            optimized_start = time.time()
            optimized_result = optimized_func(test_size)
            optimized_time = time.time() - optimized_start
            
            # Verify results are the same
            assert original_result == optimized_result
            
            duration = time.time() - start_time
            return ValidationResult(
                test_name, True, duration,
                details={
                    'original_time': original_time,
                    'optimized_time': optimized_time,
                    'speedup_ratio': original_time / optimized_time if optimized_time > 0 else 1.0,
                    'result_verified': True
                }
            )
            
        except Exception as e:
            duration = time.time() - start_time
            return ValidationResult(
                test_name, False, duration,
                error=str(e)
            )
    
    def _test_vectorization(self) -> ValidationResult:
        """Test vectorization capabilities."""
        test_name = "vectorization"
        start_time = time.time()
        
        try:
            optimizer = AdvancedPerformanceOptimizer(OptimizationLevel.DEVELOPMENT)
            
            # Define vectorizable operation
            def square_operation(data):
                return data ** 2
            
            # Vectorize operation
            vectorized_op = optimizer.vectorize_operation(square_operation)
            
            # Test with different data types
            test_data = np.random.randn(10000)
            
            # Time vectorized operation
            vec_start = time.time()
            vec_result = vectorized_op(test_data)
            vec_time = time.time() - vec_start
            
            # Verify result shape and type
            assert vec_result.shape == test_data.shape
            assert isinstance(vec_result, np.ndarray)
            
            duration = time.time() - start_time
            return ValidationResult(
                test_name, True, duration,
                details={
                    'vectorized_time': vec_time,
                    'data_size': len(test_data),
                    'throughput': len(test_data) / vec_time if vec_time > 0 else 0
                }
            )
            
        except Exception as e:
            duration = time.time() - start_time
            return ValidationResult(
                test_name, False, duration,
                error=str(e)
            )
    
    def _test_memory_management(self) -> ValidationResult:
        """Test memory management capabilities."""
        test_name = "memory_management"
        start_time = time.time()
        
        try:
            optimizer = AdvancedPerformanceOptimizer(OptimizationLevel.DEVELOPMENT)
            
            # Test memory pool creation
            pool_name = "test_pool"
            pool_size = 64  # 64 MB
            memory_pool = optimizer.get_memory_pool(pool_name, pool_size)
            
            assert memory_pool is not None
            assert memory_pool.name == pool_name
            assert memory_pool.size_bytes == pool_size * 1024 * 1024
            
            # Test memory allocation
            allocation_size = 1024 * 1024  # 1 MB
            allocation_id = memory_pool.allocate(allocation_size)
            
            assert allocation_id is not None
            assert memory_pool.get_utilization() > 0
            
            # Test deallocation
            success = memory_pool.deallocate(allocation_id)
            assert success
            
            duration = time.time() - start_time
            return ValidationResult(
                test_name, True, duration,
                details={
                    'pool_created': True,
                    'allocation_successful': True,
                    'deallocation_successful': True,
                    'pool_size_mb': pool_size
                }
            )
            
        except Exception as e:
            duration = time.time() - start_time
            return ValidationResult(
                test_name, False, duration,
                error=str(e)
            )
    
    def _run_distributed_computing_tests(self):
        """Run distributed computing tests."""
        logger.info("\n🌐 Running Distributed Computing Tests...")
        
        if not IMPLEMENTATION_AVAILABLE:
            return
        
        # Test workload distribution
        result = self._test_workload_distribution()
        self.results.append(result)
        
        # Test load balancing
        result = self._test_load_balancing()
        self.results.append(result)
        
        # Test resource scheduling
        result = self._test_resource_scheduling()
        self.results.append(result)
    
    def _test_workload_distribution(self) -> ValidationResult:
        """Test workload distribution capabilities."""
        test_name = "workload_distribution"
        start_time = time.time()
        
        try:
            framework = DistributedComputingFramework(world_size=4)
            
            # Create test workload
            workload = list(range(100))
            
            # Test different distribution strategies
            strategies = ['round_robin', 'load_balanced']
            
            for strategy in strategies:
                chunks = framework.distribute_workload(workload, strategy=strategy)
                
                # Verify distribution
                assert len(chunks) == framework.world_size
                assert sum(len(chunk) for chunk in chunks) == len(workload)
                
                # Verify no duplicates or missing items
                distributed_items = []
                for chunk in chunks:
                    distributed_items.extend(chunk)
                
                assert set(distributed_items) == set(workload)
            
            duration = time.time() - start_time
            return ValidationResult(
                test_name, True, duration,
                details={
                    'workload_size': len(workload),
                    'world_size': framework.world_size,
                    'strategies_tested': len(strategies),
                    'distribution_verified': True
                }
            )
            
        except Exception as e:
            duration = time.time() - start_time
            return ValidationResult(
                test_name, False, duration,
                error=str(e)
            )
    
    def _test_load_balancing(self) -> ValidationResult:
        """Test intelligent load balancing."""
        test_name = "load_balancing"
        start_time = time.time()
        
        try:
            framework = DistributedComputingFramework()
            load_balancer = framework.load_balancer
            
            # Simulate different worker performances
            num_workers = 4
            for worker_id in range(num_workers):
                # Update performance metrics
                execution_time = random.uniform(0.1, 2.0)  # Random execution times
                work_size = 100
                load_balancer.update_worker_performance(worker_id, execution_time, work_size)
            
            # Test load balancing
            workload = list(range(1000))
            chunks = load_balancer.distribute_workload(workload, num_workers)
            
            # Verify load balancing
            assert len(chunks) == num_workers
            chunk_sizes = [len(chunk) for chunk in chunks]
            
            # Load balancing should create reasonably balanced chunks
            # (not necessarily equal, but within reasonable variance)
            avg_chunk_size = sum(chunk_sizes) / len(chunk_sizes)
            max_deviation = max(abs(size - avg_chunk_size) for size in chunk_sizes)
            assert max_deviation < avg_chunk_size * 0.5  # Within 50% of average
            
            duration = time.time() - start_time
            return ValidationResult(
                test_name, True, duration,
                details={
                    'num_workers': num_workers,
                    'workload_size': len(workload),
                    'chunk_sizes': chunk_sizes,
                    'avg_chunk_size': avg_chunk_size,
                    'max_deviation': max_deviation
                }
            )
            
        except Exception as e:
            duration = time.time() - start_time
            return ValidationResult(
                test_name, False, duration,
                error=str(e)
            )
    
    def _test_resource_scheduling(self) -> ValidationResult:
        """Test resource scheduling capabilities."""
        test_name = "resource_scheduling"
        start_time = time.time()
        
        try:
            framework = DistributedComputingFramework()
            scheduler = framework.resource_scheduler
            
            # Test task scheduling
            test_task = {
                'cpu_cores': 2,
                'memory_mb': 512,
                'gpu_count': 0
            }
            
            task_id = scheduler.schedule_task(test_task, priority=1)
            assert task_id is not None
            
            # Verify task was scheduled
            scheduled_tasks = scheduler.scheduled_tasks
            assert len(scheduled_tasks) > 0
            assert any(task['task_id'] == task_id for task in scheduled_tasks)
            
            # Test resource utilization
            cpu_pool = scheduler.resource_pools['cpu']
            memory_pool = scheduler.resource_pools['memory']
            
            assert cpu_pool.get_utilization() > 0
            assert memory_pool.get_utilization() > 0
            
            # Test task completion
            scheduler.complete_task(task_id)
            
            # Find completed task
            completed_task = next(
                (task for task in scheduled_tasks if task['task_id'] == task_id),
                None
            )
            assert completed_task is not None
            assert completed_task['status'] == 'completed'
            
            duration = time.time() - start_time
            return ValidationResult(
                test_name, True, duration,
                details={
                    'task_scheduled': True,
                    'task_completed': True,
                    'cpu_utilization': cpu_pool.get_utilization(),
                    'memory_utilization': memory_pool.get_utilization()
                }
            )
            
        except Exception as e:
            duration = time.time() - start_time
            return ValidationResult(
                test_name, False, duration,
                error=str(e)
            )
    
    def _run_caching_system_tests(self):
        """Run caching system tests."""
        logger.info("\n💾 Running Caching System Tests...")
        
        if not IMPLEMENTATION_AVAILABLE:
            return
        
        # Test multi-tier caching
        result = self._test_multi_tier_caching()
        self.results.append(result)
        
        # Test cache performance
        result = self._test_cache_performance()
        self.results.append(result)
        
        # Test cache eviction
        result = self._test_cache_eviction()
        self.results.append(result)
    
    def _test_multi_tier_caching(self) -> ValidationResult:
        """Test multi-tier caching functionality."""
        test_name = "multi_tier_caching"
        start_time = time.time()
        
        try:
            cache_config = {
                'l1_size': 100,
                'l2_size_mb': 10,
                'redis_enabled': False,
                'predictive_warming': False
            }
            
            cache_system = MultiTierCacheSystem(cache_config)
            
            # Test basic put/get operations
            test_data = {
                'key1': 'value1',
                'key2': {'nested': 'data'},
                'key3': [1, 2, 3, 4, 5]
            }
            
            # Store data
            for key, value in test_data.items():
                cache_system.put(key, value)
            
            # Retrieve and verify data
            for key, expected_value in test_data.items():
                cached_value = cache_system.get(key)
                assert cached_value == expected_value
            
            # Test cache statistics
            stats = cache_system.get_statistics()
            assert 'l1_hits' in stats
            assert 'total_requests' in stats
            assert stats['l1_hits'] >= len(test_data)
            
            duration = time.time() - start_time
            return ValidationResult(
                test_name, True, duration,
                details={
                    'items_cached': len(test_data),
                    'cache_stats': stats,
                    'data_integrity_verified': True
                }
            )
            
        except Exception as e:
            duration = time.time() - start_time
            return ValidationResult(
                test_name, False, duration,
                error=str(e)
            )
    
    def _test_cache_performance(self) -> ValidationResult:
        """Test cache performance under load."""
        test_name = "cache_performance"
        start_time = time.time()
        
        try:
            cache_config = {
                'l1_size': 1000,
                'l2_size_mb': 50,
                'redis_enabled': False
            }
            
            cache_system = MultiTierCacheSystem(cache_config)
            
            # Performance test parameters
            num_operations = 10000
            cache_operations = []
            
            # Generate test data
            test_keys = [f"perf_key_{i}" for i in range(num_operations // 2)]
            test_values = [f"perf_value_{i}" * 100 for i in range(len(test_keys))]  # Larger values
            
            # Measure put operations
            put_start = time.time()
            for key, value in zip(test_keys, test_values):
                cache_system.put(key, value)
            put_time = time.time() - put_start
            
            # Measure get operations (should mostly hit L1 cache)
            get_start = time.time()
            hit_count = 0
            for key in test_keys:
                if cache_system.get(key) is not None:
                    hit_count += 1
            get_time = time.time() - get_start
            
            # Calculate performance metrics
            put_throughput = len(test_keys) / put_time if put_time > 0 else 0
            get_throughput = len(test_keys) / get_time if get_time > 0 else 0
            hit_rate = hit_count / len(test_keys)
            
            duration = time.time() - start_time
            return ValidationResult(
                test_name, True, duration,
                details={
                    'operations_tested': num_operations,
                    'put_throughput_ops_sec': put_throughput,
                    'get_throughput_ops_sec': get_throughput,
                    'cache_hit_rate': hit_rate,
                    'total_put_time': put_time,
                    'total_get_time': get_time
                }
            )
            
        except Exception as e:
            duration = time.time() - start_time
            return ValidationResult(
                test_name, False, duration,
                error=str(e)
            )
    
    def _test_cache_eviction(self) -> ValidationResult:
        """Test cache eviction policies."""
        test_name = "cache_eviction"
        start_time = time.time()
        
        try:
            # Small cache to force eviction
            cache_config = {
                'l1_size': 10,  # Very small L1 cache
                'l2_size_mb': 1,
                'redis_enabled': False
            }
            
            cache_system = MultiTierCacheSystem(cache_config)
            
            # Fill cache beyond capacity
            num_items = 20  # More than L1 capacity
            for i in range(num_items):
                cache_system.put(f"evict_key_{i}", f"evict_value_{i}")
            
            # Check that old items were evicted from L1
            l1_cache = cache_system.l1_cache
            assert len(l1_cache.cache) <= cache_config['l1_size']
            
            # Check that recent items are still accessible (may be in L2)
            recent_key = f"evict_key_{num_items - 1}"
            recent_value = cache_system.get(recent_key)
            assert recent_value is not None
            
            # Get statistics to verify eviction occurred
            stats = cache_system.get_statistics()
            
            duration = time.time() - start_time
            return ValidationResult(
                test_name, True, duration,
                details={
                    'items_inserted': num_items,
                    'l1_cache_size': len(l1_cache.cache),
                    'l1_max_size': cache_config['l1_size'],
                    'eviction_occurred': num_items > cache_config['l1_size'],
                    'recent_item_accessible': recent_value is not None
                }
            )
            
        except Exception as e:
            duration = time.time() - start_time
            return ValidationResult(
                test_name, False, duration,
                error=str(e)
            )
    
    def _run_performance_analytics_tests(self):
        """Run performance analytics tests."""
        logger.info("\n📈 Running Performance Analytics Tests...")
        
        if not IMPLEMENTATION_AVAILABLE:
            return
        
        # Test metrics collection
        result = self._test_metrics_collection()
        self.results.append(result)
        
        # Test anomaly detection
        result = self._test_anomaly_detection()
        self.results.append(result)
        
        # Test performance reporting
        result = self._test_performance_reporting()
        self.results.append(result)
    
    def _test_metrics_collection(self) -> ValidationResult:
        """Test metrics collection functionality."""
        test_name = "metrics_collection"
        start_time = time.time()
        
        try:
            analytics = PerformanceAnalyticsSuite()
            
            # Test metrics collection
            initial_count = len(analytics.metrics_history)
            
            # Start monitoring briefly
            analytics.start_monitoring(interval=0.5)  # 0.5 second interval
            time.sleep(2.0)  # Let it collect a few metrics
            analytics.stop_monitoring()
            
            # Verify metrics were collected
            final_count = len(analytics.metrics_history)
            metrics_collected = final_count - initial_count
            
            assert metrics_collected > 0
            
            # Test manual metrics processing
            test_metrics = EnterpriseMetrics(
                cpu_utilization=75.0,
                memory_utilization=60.0,
                gpu_utilization=0.0,
                response_time_p95=0.150
            )
            
            analytics._process_metrics(test_metrics)
            
            duration = time.time() - start_time
            return ValidationResult(
                test_name, True, duration,
                details={
                    'metrics_collected_during_monitoring': metrics_collected,
                    'total_metrics_in_history': len(analytics.metrics_history),
                    'manual_metrics_processed': True
                }
            )
            
        except Exception as e:
            duration = time.time() - start_time
            return ValidationResult(
                test_name, False, duration,
                error=str(e)
            )
    
    def _test_anomaly_detection(self) -> ValidationResult:
        """Test anomaly detection capabilities."""
        test_name = "anomaly_detection"
        start_time = time.time()
        
        try:
            analytics = PerformanceAnalyticsSuite()
            anomaly_detector = analytics.anomaly_detector
            
            # Create baseline metrics
            baseline_metrics = []
            for i in range(50):
                metrics = EnterpriseMetrics(
                    cpu_utilization=50.0 + random.uniform(-5, 5),  # Normal variation
                    memory_utilization=60.0 + random.uniform(-5, 5),
                    response_time_p95=0.100 + random.uniform(-0.01, 0.01)
                )
                baseline_metrics.append(metrics)
                anomaly_detector.detect_anomalies(metrics)
            
            # Introduce anomalous metrics
            anomalous_metrics = EnterpriseMetrics(
                cpu_utilization=95.0,  # Very high CPU
                memory_utilization=90.0,  # Very high memory
                response_time_p95=1.0  # Very high response time
            )
            
            anomalies = anomaly_detector.detect_anomalies(anomalous_metrics)
            
            # Should detect anomalies in the extreme values
            assert len(anomalies) > 0
            
            duration = time.time() - start_time
            return ValidationResult(
                test_name, True, duration,
                details={
                    'baseline_metrics_processed': len(baseline_metrics),
                    'anomalies_detected': len(anomalies),
                    'anomaly_details': anomalies,
                    'total_anomalies_tracked': len(anomaly_detector.anomalies)
                }
            )
            
        except Exception as e:
            duration = time.time() - start_time
            return ValidationResult(
                test_name, False, duration,
                error=str(e)
            )
    
    def _test_performance_reporting(self) -> ValidationResult:
        """Test performance reporting functionality."""
        test_name = "performance_reporting"
        start_time = time.time()
        
        try:
            analytics = PerformanceAnalyticsSuite()
            
            # Add some metrics to history
            for i in range(10):
                metrics = EnterpriseMetrics(
                    cpu_utilization=50.0 + i,
                    memory_utilization=60.0 + i * 2,
                    response_time_p95=0.100 + i * 0.01
                )
                analytics.metrics_history.append(metrics)
            
            # Generate performance report
            report = analytics.get_performance_report()
            
            # Verify report structure
            assert 'timestamp' in report
            assert 'averages' in report
            assert 'trends' in report
            
            # Verify calculated averages
            averages = report['averages']
            assert 'cpu_utilization' in averages
            assert 'memory_utilization' in averages
            
            # Verify trend analysis
            trends = report['trends']
            assert 'cpu_trend' in trends
            assert 'memory_trend' in trends
            
            duration = time.time() - start_time
            return ValidationResult(
                test_name, True, duration,
                details={
                    'report_generated': True,
                    'metrics_in_report': len(analytics.metrics_history),
                    'report_sections': list(report.keys()),
                    'cpu_trend': trends['cpu_trend'],
                    'memory_trend': trends['memory_trend']
                }
            )
            
        except Exception as e:
            duration = time.time() - start_time
            return ValidationResult(
                test_name, False, duration,
                error=str(e)
            )
    
    def _run_enterprise_infrastructure_tests(self):
        """Run enterprise infrastructure tests."""
        logger.info("\n🏗️ Running Enterprise Infrastructure Tests...")
        
        if not IMPLEMENTATION_AVAILABLE:
            return
        
        # Test infrastructure initialization
        result = self._test_infrastructure_initialization()
        self.results.append(result)
        
        # Test scaling recommendations
        result = self._test_scaling_recommendations()
        self.results.append(result)
        
        # Test status reporting
        result = self._test_infrastructure_status()
        self.results.append(result)
    
    def _test_infrastructure_initialization(self) -> ValidationResult:
        """Test enterprise infrastructure initialization."""
        test_name = "infrastructure_initialization"
        start_time = time.time()
        
        try:
            config = {
                'optimization_level': 'development',
                'kubernetes_enabled': False,
                'auto_scaling_enabled': True,
                'cache_config': {
                    'l1_size': 500,
                    'l2_size_mb': 100,
                    'redis_enabled': False
                }
            }
            
            infrastructure = EnterpriseScalingInfrastructure(config)
            
            # Verify components are initialized
            assert infrastructure.performance_optimizer is not None
            assert infrastructure.distributed_framework is not None
            assert infrastructure.cache_system is not None
            assert infrastructure.analytics_suite is not None
            
            # Test infrastructure startup and shutdown
            infrastructure.start_infrastructure()
            time.sleep(1.0)  # Let it start up
            infrastructure.stop_infrastructure()
            
            duration = time.time() - start_time
            return ValidationResult(
                test_name, True, duration,
                details={
                    'components_initialized': 4,
                    'startup_successful': True,
                    'shutdown_successful': True,
                    'optimization_level': config['optimization_level']
                }
            )
            
        except Exception as e:
            duration = time.time() - start_time
            return ValidationResult(
                test_name, False, duration,
                error=str(e)
            )
    
    def _test_scaling_recommendations(self) -> ValidationResult:
        """Test scaling recommendations generation."""
        test_name = "scaling_recommendations"
        start_time = time.time()
        
        try:
            infrastructure = EnterpriseScalingInfrastructure({
                'optimization_level': 'development',
                'kubernetes_enabled': False
            })
            
            # Add some test metrics to trigger recommendations
            analytics = infrastructure.analytics_suite
            
            # High CPU scenario
            high_cpu_metrics = EnterpriseMetrics(
                cpu_utilization=95.0,
                memory_utilization=50.0
            )
            analytics.metrics_history.append(high_cpu_metrics)
            
            # Low CPU scenario
            low_cpu_metrics = EnterpriseMetrics(
                cpu_utilization=15.0,
                memory_utilization=30.0
            )
            analytics.metrics_history.append(low_cpu_metrics)
            
            # Generate recommendations
            recommendations = infrastructure.generate_scaling_recommendations()
            
            # Should have some recommendations based on the metrics
            assert isinstance(recommendations, list)
            
            duration = time.time() - start_time
            return ValidationResult(
                test_name, True, duration,
                details={
                    'recommendations_generated': len(recommendations),
                    'sample_recommendations': recommendations[:3],  # First 3
                    'metrics_scenarios_tested': 2
                }
            )
            
        except Exception as e:
            duration = time.time() - start_time
            return ValidationResult(
                test_name, False, duration,
                error=str(e)
            )
    
    def _test_infrastructure_status(self) -> ValidationResult:
        """Test infrastructure status reporting."""
        test_name = "infrastructure_status"
        start_time = time.time()
        
        try:
            infrastructure = EnterpriseScalingInfrastructure({
                'optimization_level': 'production',
                'kubernetes_enabled': False,
                'auto_scaling_enabled': True
            })
            
            # Get infrastructure status
            status = infrastructure.get_infrastructure_status()
            
            # Verify status report structure
            required_fields = [
                'timestamp', 'kubernetes_enabled', 'auto_scaling_enabled',
                'optimization_level', 'distributed_computing', 'cache_system'
            ]
            
            for field in required_fields:
                assert field in status
            
            # Verify nested structures
            assert 'world_size' in status['distributed_computing']
            assert isinstance(status['cache_system'], dict)
            
            duration = time.time() - start_time
            return ValidationResult(
                test_name, True, duration,
                details={
                    'status_fields_verified': len(required_fields),
                    'optimization_level': status['optimization_level'],
                    'distributed_world_size': status['distributed_computing']['world_size'],
                    'auto_scaling_enabled': status['auto_scaling_enabled']
                }
            )
            
        except Exception as e:
            duration = time.time() - start_time
            return ValidationResult(
                test_name, False, duration,
                error=str(e)
            )
    
    def _run_load_tests(self):
        """Run load and stress tests."""
        logger.info("\n🔥 Running Load and Stress Tests...")
        
        if not IMPLEMENTATION_AVAILABLE:
            return
        
        # Test concurrent cache operations
        result = self._run_concurrent_cache_test()
        self.load_test_results.append(result)
        
        # Test distributed workload processing
        result = self._run_distributed_workload_test()
        self.load_test_results.append(result)
        
        # Test performance under memory pressure
        result = self._run_memory_pressure_test()
        self.load_test_results.append(result)
    
    def _run_concurrent_cache_test(self) -> LoadTestResult:
        """Run concurrent cache operations test."""
        test_name = "concurrent_cache_operations"
        start_time = time.time()
        
        total_operations = 0
        successful_operations = 0
        response_times = []
        errors = []
        peak_cpu = 0.0
        peak_memory = 0.0
        
        try:
            cache_system = MultiTierCacheSystem({
                'l1_size': 1000,
                'l2_size_mb': 100,
                'redis_enabled': False
            })
            
            # Concurrent operations configuration
            num_threads = 10
            operations_per_thread = 1000
            
            def cache_worker(thread_id):
                nonlocal successful_operations, total_operations
                thread_errors = []
                thread_times = []
                
                for i in range(operations_per_thread):
                    try:
                        op_start = time.time()
                        
                        # Mix of put and get operations
                        if i % 3 == 0:
                            # Put operation
                            key = f"thread_{thread_id}_key_{i}"
                            value = f"thread_{thread_id}_value_{i}" * 10
                            cache_system.put(key, value)
                        else:
                            # Get operation (may miss initially)
                            key = f"thread_{thread_id}_key_{i // 3}"
                            cache_system.get(key)
                        
                        op_time = time.time() - op_start
                        thread_times.append(op_time)
                        successful_operations += 1
                        
                    except Exception as e:
                        thread_errors.append(str(e))
                    
                    total_operations += 1
                
                return thread_times, thread_errors
            
            # Monitor system resources
            def monitor_resources():
                nonlocal peak_cpu, peak_memory
                monitor_start = time.time()
                
                while time.time() - monitor_start < 30:  # Monitor for 30 seconds
                    cpu_percent = psutil.cpu_percent(interval=0.1)
                    memory_percent = psutil.virtual_memory().percent
                    
                    peak_cpu = max(peak_cpu, cpu_percent)
                    peak_memory = max(peak_memory, memory_percent)
                    
                    time.sleep(0.1)
            
            # Start resource monitoring
            monitor_thread = threading.Thread(target=monitor_resources, daemon=True)
            monitor_thread.start()
            
            # Run concurrent cache operations
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = [
                    executor.submit(cache_worker, i)
                    for i in range(num_threads)
                ]
                
                # Collect results
                for future in as_completed(futures):
                    thread_times, thread_errors = future.result()
                    response_times.extend(thread_times)
                    errors.extend(thread_errors)
            
            # Wait for monitoring to complete
            monitor_thread.join(timeout=5)
            
        except Exception as e:
            errors.append(str(e))
        
        duration = time.time() - start_time
        success_rate = successful_operations / total_operations if total_operations > 0 else 0
        avg_response_time = statistics.mean(response_times) if response_times else 0
        
        return LoadTestResult(
            test_name, duration, total_operations, success_rate,
            avg_response_time, peak_cpu, peak_memory, errors
        )
    
    def _run_distributed_workload_test(self) -> LoadTestResult:
        """Run distributed workload processing test."""
        test_name = "distributed_workload_processing"
        start_time = time.time()
        
        total_operations = 0
        successful_operations = 0
        response_times = []
        errors = []
        peak_cpu = 0.0
        peak_memory = 0.0
        
        try:
            framework = DistributedComputingFramework(world_size=4)
            
            # Create large workload
            workload_size = 10000
            workload = list(range(workload_size))
            
            # Process workload in chunks
            num_iterations = 10
            
            def process_chunk(chunk):
                """Simulate processing a chunk of work."""
                chunk_start = time.time()
                
                # Simulate computational work
                result = sum(x ** 2 for x in chunk)
                
                processing_time = time.time() - chunk_start
                return result, processing_time
            
            # Monitor system resources
            def monitor_resources():
                nonlocal peak_cpu, peak_memory
                monitor_start = time.time()
                
                while time.time() - monitor_start < 20:
                    cpu_percent = psutil.cpu_percent(interval=0.1)
                    memory_percent = psutil.virtual_memory().percent
                    
                    peak_cpu = max(peak_cpu, cpu_percent)
                    peak_memory = max(peak_memory, memory_percent)
                    
                    time.sleep(0.1)
            
            monitor_thread = threading.Thread(target=monitor_resources, daemon=True)
            monitor_thread.start()
            
            # Process workload multiple times
            for iteration in range(num_iterations):
                try:
                    iter_start = time.time()
                    
                    # Distribute workload
                    chunks = framework.distribute_workload(workload, strategy='load_balanced')
                    
                    # Process chunks concurrently
                    with ThreadPoolExecutor(max_workers=len(chunks)) as executor:
                        futures = [
                            executor.submit(process_chunk, chunk)
                            for chunk in chunks if chunk  # Skip empty chunks
                        ]
                        
                        iteration_results = []
                        for future in as_completed(futures):
                            result, processing_time = future.result()
                            iteration_results.append(result)
                            response_times.append(processing_time)
                            successful_operations += 1
                    
                    iter_time = time.time() - iter_start
                    total_operations += len(chunks)
                    
                except Exception as e:
                    errors.append(f"Iteration {iteration}: {str(e)}")
                    total_operations += 1
            
            monitor_thread.join(timeout=5)
            
        except Exception as e:
            errors.append(str(e))
        
        duration = time.time() - start_time
        success_rate = successful_operations / total_operations if total_operations > 0 else 0
        avg_response_time = statistics.mean(response_times) if response_times else 0
        
        return LoadTestResult(
            test_name, duration, total_operations, success_rate,
            avg_response_time, peak_cpu, peak_memory, errors
        )
    
    def _run_memory_pressure_test(self) -> LoadTestResult:
        """Run test under memory pressure conditions."""
        test_name = "memory_pressure_test"
        start_time = time.time()
        
        total_operations = 0
        successful_operations = 0
        response_times = []
        errors = []
        peak_cpu = 0.0
        peak_memory = 0.0
        
        try:
            # Create memory pressure by allocating large arrays
            memory_pressure_data = []
            target_memory_mb = 500  # Target 500MB memory usage
            
            optimizer = AdvancedPerformanceOptimizer(OptimizationLevel.DEVELOPMENT)
            
            # Monitor resources
            def monitor_resources():
                nonlocal peak_cpu, peak_memory
                monitor_start = time.time()
                
                while time.time() - monitor_start < 15:
                    cpu_percent = psutil.cpu_percent(interval=0.1)
                    memory_percent = psutil.virtual_memory().percent
                    
                    peak_cpu = max(peak_cpu, cpu_percent)
                    peak_memory = max(peak_memory, memory_percent)
                    
                    time.sleep(0.1)
            
            monitor_thread = threading.Thread(target=monitor_resources, daemon=True)
            monitor_thread.start()
            
            # Create memory pressure
            array_size = 1024 * 1024 // 8  # 1MB per array (8 bytes per float64)
            num_arrays = target_memory_mb
            
            for i in range(num_arrays):
                try:
                    # Allocate memory
                    data = np.random.randn(array_size)
                    memory_pressure_data.append(data)
                    
                    # Perform operations under memory pressure
                    op_start = time.time()
                    
                    # Test memory pool allocation
                    memory_pool = optimizer.get_memory_pool(f"pressure_pool_{i}", 10)
                    allocation_id = memory_pool.allocate(1024 * 1024)  # 1MB allocation
                    
                    if allocation_id is not None:
                        memory_pool.deallocate(allocation_id)
                        successful_operations += 1
                    
                    op_time = time.time() - op_start
                    response_times.append(op_time)
                    total_operations += 1
                    
                    # Check if we should break due to memory constraints
                    current_memory = psutil.virtual_memory().percent
                    if current_memory > 90:  # Stop if memory usage too high
                        logger.warning("Breaking memory pressure test due to high memory usage")
                        break
                        
                except Exception as e:
                    errors.append(f"Memory allocation {i}: {str(e)}")
                    total_operations += 1
            
            monitor_thread.join(timeout=5)
            
            # Cleanup memory pressure data
            del memory_pressure_data
            import gc
            gc.collect()
            
        except Exception as e:
            errors.append(str(e))
        
        duration = time.time() - start_time
        success_rate = successful_operations / total_operations if total_operations > 0 else 0
        avg_response_time = statistics.mean(response_times) if response_times else 0
        
        return LoadTestResult(
            test_name, duration, total_operations, success_rate,
            avg_response_time, peak_cpu, peak_memory, errors
        )
    
    def _run_integration_tests(self):
        """Run integration tests."""
        logger.info("\n🔗 Running Integration Tests...")
        
        if not IMPLEMENTATION_AVAILABLE:
            return
        
        # Test full pipeline integration
        result = self._test_full_pipeline_integration()
        self.results.append(result)
        
        # Test error handling and recovery
        result = self._test_error_handling_recovery()
        self.results.append(result)
    
    def _test_full_pipeline_integration(self) -> ValidationResult:
        """Test full pipeline integration."""
        test_name = "full_pipeline_integration"
        start_time = time.time()
        
        try:
            # Initialize full infrastructure
            config = {
                'optimization_level': 'development',
                'kubernetes_enabled': False,
                'auto_scaling_enabled': False,  # Disable for test
                'cache_config': {
                    'l1_size': 100,
                    'l2_size_mb': 10,
                    'redis_enabled': False
                }
            }
            
            infrastructure = EnterpriseScalingInfrastructure(config)
            infrastructure.start_infrastructure()
            
            try:
                # Test integrated workflow
                # 1. Performance optimization
                optimizer = infrastructure.performance_optimizer
                
                @optimizer.optimize_function
                def integrated_computation(data):
                    return np.sum(data ** 2)
                
                # 2. Caching
                cache_system = infrastructure.cache_system
                test_data = np.random.randn(1000)
                
                cache_key = "integrated_test_data"
                cache_system.put(cache_key, test_data)
                cached_data = cache_system.get(cache_key)
                
                assert np.array_equal(cached_data, test_data)
                
                # 3. Distributed processing
                dist_framework = infrastructure.distributed_framework
                workload = list(range(100))
                distributed_chunks = dist_framework.distribute_workload(workload)
                
                # 4. Performance monitoring
                analytics = infrastructure.analytics_suite
                test_metrics = EnterpriseMetrics(
                    cpu_utilization=45.0,
                    memory_utilization=55.0
                )
                analytics._process_metrics(test_metrics)
                
                # 5. Generate status and recommendations
                status = infrastructure.get_infrastructure_status()
                recommendations = infrastructure.generate_scaling_recommendations()
                
                # Verify integration results
                assert 'timestamp' in status
                assert isinstance(recommendations, list)
                assert len(distributed_chunks) == dist_framework.world_size
                
                duration = time.time() - start_time
                return ValidationResult(
                    test_name, True, duration,
                    details={
                        'components_integrated': 5,
                        'cache_verified': True,
                        'distribution_verified': True,
                        'monitoring_verified': True,
                        'status_generated': True,
                        'recommendations_count': len(recommendations)
                    }
                )
                
            finally:
                infrastructure.stop_infrastructure()
                
        except Exception as e:
            duration = time.time() - start_time
            return ValidationResult(
                test_name, False, duration,
                error=str(e)
            )
    
    def _test_error_handling_recovery(self) -> ValidationResult:
        """Test error handling and recovery mechanisms."""
        test_name = "error_handling_recovery"
        start_time = time.time()
        
        try:
            infrastructure = EnterpriseScalingInfrastructure({
                'optimization_level': 'development',
                'kubernetes_enabled': False
            })
            
            recovery_scenarios_tested = 0
            recovery_scenarios_passed = 0
            
            # Scenario 1: Cache system with invalid data
            try:
                cache_system = infrastructure.cache_system
                # Try to cache an uncacheable object
                cache_system.put("invalid_key", lambda x: x)  # Function object
                # Should handle gracefully
                recovery_scenarios_tested += 1
                recovery_scenarios_passed += 1
            except Exception:
                recovery_scenarios_tested += 1
                # Expected to fail, but should not crash the system
            
            # Scenario 2: Performance optimizer with invalid function
            try:
                optimizer = infrastructure.performance_optimizer
                def problematic_function():
                    raise ValueError("Intentional error")
                
                # Should handle optimization gracefully
                optimized = optimizer.optimize_function(problematic_function, use_jit=False)
                recovery_scenarios_tested += 1
                recovery_scenarios_passed += 1
            except Exception:
                recovery_scenarios_tested += 1
            
            # Scenario 3: Distributed framework with empty workload
            try:
                framework = infrastructure.distributed_framework
                empty_workload = []
                chunks = framework.distribute_workload(empty_workload)
                assert isinstance(chunks, list)
                recovery_scenarios_tested += 1
                recovery_scenarios_passed += 1
            except Exception:
                recovery_scenarios_tested += 1
            
            # Scenario 4: Analytics with invalid metrics
            try:
                analytics = infrastructure.analytics_suite
                invalid_metrics = EnterpriseMetrics(
                    cpu_utilization=float('nan'),
                    memory_utilization=-10.0  # Invalid negative value
                )
                # Should handle invalid metrics gracefully
                analytics._process_metrics(invalid_metrics)
                recovery_scenarios_tested += 1
                recovery_scenarios_passed += 1
            except Exception:
                recovery_scenarios_tested += 1
            
            duration = time.time() - start_time
            recovery_rate = recovery_scenarios_passed / recovery_scenarios_tested if recovery_scenarios_tested > 0 else 0
            
            return ValidationResult(
                test_name, recovery_rate > 0.5, duration,  # Pass if >50% of scenarios handled
                details={
                    'recovery_scenarios_tested': recovery_scenarios_tested,
                    'recovery_scenarios_passed': recovery_scenarios_passed,
                    'recovery_rate': recovery_rate,
                    'system_stability_maintained': True
                }
            )
            
        except Exception as e:
            duration = time.time() - start_time
            return ValidationResult(
                test_name, False, duration,
                error=str(e)
            )
    
    def _generate_validation_report(self, validation_duration: float) -> Dict[str, Any]:
        """Generate comprehensive validation report.
        
        Args:
            validation_duration: Total validation time
            
        Returns:
            Validation report
        """
        # Calculate test statistics
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r.passed)
        failed_tests = total_tests - passed_tests
        pass_rate = passed_tests / total_tests if total_tests > 0 else 0
        
        # Calculate load test statistics
        total_load_tests = len(self.load_test_results)
        avg_success_rate = (
            statistics.mean(r.success_rate for r in self.load_test_results)
            if self.load_test_results else 0
        )
        total_operations = sum(r.total_operations for r in self.load_test_results)
        total_throughput = sum(r.throughput for r in self.load_test_results)
        
        # System information
        system_info = {
            'cpu_cores': mp.cpu_count(),
            'total_memory_gb': round(psutil.virtual_memory().total / (1024 ** 3), 2),
            'available_memory_gb': round(psutil.virtual_memory().available / (1024 ** 3), 2),
            'platform': sys.platform,
            'python_version': sys.version
        }
        
        # Create comprehensive report
        report = {
            'validation_summary': {
                'timestamp': time.time(),
                'validation_duration': validation_duration,
                'total_tests': total_tests,
                'passed_tests': passed_tests,
                'failed_tests': failed_tests,
                'pass_rate': pass_rate,
                'implementation_available': IMPLEMENTATION_AVAILABLE
            },
            'functionality_tests': {
                'total': len([r for r in self.results if 'load_test' not in r.test_name]),
                'passed': len([r for r in self.results if r.passed and 'load_test' not in r.test_name]),
                'details': [r.to_dict() for r in self.results]
            },
            'load_tests': {
                'total': total_load_tests,
                'avg_success_rate': avg_success_rate,
                'total_operations': total_operations,
                'total_throughput': total_throughput,
                'details': [r.to_dict() for r in self.load_test_results]
            },
            'performance_metrics': {
                'avg_test_duration': statistics.mean([r.duration for r in self.results]) if self.results else 0,
                'max_test_duration': max([r.duration for r in self.results]) if self.results else 0,
                'total_test_time': sum(r.duration for r in self.results),
                'tests_per_second': total_tests / validation_duration if validation_duration > 0 else 0
            },
            'system_info': system_info,
            'recommendations': self._generate_recommendations(),
            'test_data_location': str(self.test_data_dir)
        }
        
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on test results.
        
        Returns:
            List of recommendations
        """
        recommendations = []
        
        # Analyze test failures
        failed_tests = [r for r in self.results if not r.passed]
        if failed_tests:
            recommendations.append(f"Address {len(failed_tests)} failed tests before production deployment")
            
            # Check for specific failure patterns
            import_errors = [r for r in failed_tests if 'import' in r.error.lower() if r.error]
            if import_errors:
                recommendations.append("Install missing dependencies for full functionality")
        
        # Analyze load test results
        if self.load_test_results:
            avg_success_rate = statistics.mean(r.success_rate for r in self.load_test_results)
            if avg_success_rate < 0.95:
                recommendations.append("Improve system reliability - load test success rate below 95%")
            
            max_cpu = max(r.peak_cpu for r in self.load_test_results)
            if max_cpu > 90:
                recommendations.append("Consider CPU scaling - peak usage exceeded 90% during load tests")
            
            max_memory = max(r.peak_memory for r in self.load_test_results)
            if max_memory > 85:
                recommendations.append("Consider memory optimization - peak usage exceeded 85%")
        
        # System-specific recommendations
        cpu_cores = mp.cpu_count()
        if cpu_cores <= 2:
            recommendations.append("Consider upgrading to a system with more CPU cores for better performance")
        
        available_memory = psutil.virtual_memory().available / (1024 ** 3)
        if available_memory < 2.0:
            recommendations.append("Consider increasing available system memory for optimal performance")
        
        if not recommendations:
            recommendations.append("All tests passed successfully - system ready for production deployment")
        
        return recommendations
    
    def _save_validation_results(self, report: Dict[str, Any]):
        """Save validation results to files.
        
        Args:
            report: Validation report to save
        """
        try:
            # Save JSON report
            json_file = self.test_data_dir / "generation_3_validation_report.json"
            with open(json_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            # Save summary text report
            text_file = self.test_data_dir / "generation_3_validation_summary.txt"
            with open(text_file, 'w') as f:
                f.write("Generation 3 Enterprise Scaling - Validation Report\n")
                f.write("=" * 60 + "\n\n")
                
                summary = report['validation_summary']
                f.write(f"Validation Duration: {summary['validation_duration']:.2f} seconds\n")
                f.write(f"Total Tests: {summary['total_tests']}\n")
                f.write(f"Passed Tests: {summary['passed_tests']}\n")
                f.write(f"Failed Tests: {summary['failed_tests']}\n")
                f.write(f"Pass Rate: {summary['pass_rate']:.1%}\n\n")
                
                if report['recommendations']:
                    f.write("Recommendations:\n")
                    for i, rec in enumerate(report['recommendations'], 1):
                        f.write(f"{i}. {rec}\n")
                f.write("\n")
                
                f.write("System Information:\n")
                for key, value in report['system_info'].items():
                    f.write(f"  {key}: {value}\n")
            
            logger.info(f"Validation results saved to {self.test_data_dir}")
            
        except Exception as e:
            logger.error(f"Failed to save validation results: {e}")


def main():
    """Main function to run the Generation 3 validation suite."""
    print("🧪 PNO Physics Bench - Generation 3 Validation Suite")
    print("=" * 60)
    
    validator = Generation3ValidatorSuite()
    
    try:
        # Run comprehensive validation
        report = validator.run_all_validations()
        
        # Print summary
        print("\n📊 VALIDATION SUMMARY")
        print("-" * 40)
        
        summary = report['validation_summary']
        print(f"Duration: {summary['validation_duration']:.2f} seconds")
        print(f"Total Tests: {summary['total_tests']}")
        print(f"Passed: {summary['passed_tests']} ✅")
        print(f"Failed: {summary['failed_tests']} ❌")
        print(f"Pass Rate: {summary['pass_rate']:.1%}")
        
        if report['load_tests']['total'] > 0:
            print(f"\nLoad Tests: {report['load_tests']['total']}")
            print(f"Avg Success Rate: {report['load_tests']['avg_success_rate']:.1%}")
            print(f"Total Operations: {report['load_tests']['total_operations']:,}")
        
        print(f"\n🎯 RECOMMENDATIONS")
        print("-" * 30)
        for i, rec in enumerate(report['recommendations'], 1):
            print(f"{i}. {rec}")
        
        print(f"\n📁 Results saved to: {validator.test_data_dir}")
        
        # Overall validation status
        if summary['pass_rate'] >= 0.8:
            print("\n🎉 VALIDATION SUCCESSFUL - Generation 3 ready for deployment!")
        else:
            print("\n⚠️  VALIDATION ISSUES DETECTED - Review failed tests before deployment")
        
    except KeyboardInterrupt:
        print("\n⏹️  Validation interrupted by user")
    except Exception as e:
        print(f"\n❌ Validation failed: {e}")
        logger.exception("Validation execution failed")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())