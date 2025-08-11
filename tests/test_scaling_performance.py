"""Comprehensive tests for scaling and performance optimization components."""

import pytest
import numpy as np
import time
import threading
from pathlib import Path
import sys
from concurrent.futures import ThreadPoolExecutor

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

from pno_physics_bench.scaling.distributed_computing import (
    ComputeNode,
    DistributedTask,
    LoadBalancer,
    DistributedTaskQueue,
    DistributedPNOWorker,
    DistributedPNOCoordinator,
    DistributedPNOCluster,
    create_distributed_pno_system
)

from pno_physics_bench.performance_optimization import (
    MemoryPool,
    ComputeCache,
    AdaptiveOptimizer,
    PerformanceProfiler,
    PerformanceOptimizedPNO,
    create_optimized_pno_system
)


class TestDistributedComputing:
    """Test suite for distributed computing components."""
    
    def test_compute_node_creation(self):
        """Test ComputeNode creation and attributes."""
        
        node = ComputeNode(
            node_id="worker_001",
            hostname="localhost",
            port=8080,
            capabilities={"cpu_cores": 8, "memory_gb": 16, "gpu_available": True},
            status="idle",
            load_factor=0.0
        )
        
        assert node.node_id == "worker_001"
        assert node.hostname == "localhost"
        assert node.port == 8080
        assert node.capabilities["cpu_cores"] == 8
        assert node.status == "idle"
        assert node.load_factor == 0.0
    
    def test_distributed_task_creation(self):
        """Test DistributedTask creation and attributes."""
        
        task = DistributedTask(
            task_id="task_001",
            task_type="pno_forward",
            payload={"input_data": [1, 2, 3], "model_params": {}},
            priority=1,
            status="pending"
        )
        
        assert task.task_id == "task_001"
        assert task.task_type == "pno_forward"
        assert task.payload["input_data"] == [1, 2, 3]
        assert task.priority == 1
        assert task.status == "pending"
        assert task.assigned_node is None
    
    def test_load_balancer_creation(self):
        """Test LoadBalancer creation and node registration."""
        
        load_balancer = LoadBalancer(balancing_strategy="round_robin")
        
        assert load_balancer.balancing_strategy == "round_robin"
        assert len(load_balancer.nodes) == 0
        
        # Register nodes
        node1 = ComputeNode("worker_1", "host1", 8001, {"cpu": 4}, status="idle", load_factor=0.1)
        node2 = ComputeNode("worker_2", "host2", 8002, {"cpu": 8}, status="idle", load_factor=0.3)
        
        load_balancer.register_node(node1)
        load_balancer.register_node(node2)
        
        assert len(load_balancer.nodes) == 2
        assert "worker_1" in load_balancer.nodes
        assert "worker_2" in load_balancer.nodes
    
    def test_load_balancer_round_robin_selection(self):
        """Test round-robin node selection."""
        
        load_balancer = LoadBalancer(balancing_strategy="round_robin")
        
        # Add nodes
        nodes = [
            ComputeNode(f"worker_{i}", f"host{i}", 8000+i, {"cpu": 4}, status="idle")
            for i in range(3)
        ]
        
        for node in nodes:
            load_balancer.register_node(node)
        
        # Create test task
        task = DistributedTask("test_task", "pno_forward", {})
        
        # Test round-robin selection
        selected_nodes = []
        for i in range(6):  # Test 2 full rounds
            selected = load_balancer.select_node(task)
            selected_nodes.append(selected.node_id)
        
        # Should cycle through nodes
        assert selected_nodes[0] == selected_nodes[3]  # Same node after 3 selections
        assert selected_nodes[1] == selected_nodes[4]
        assert selected_nodes[2] == selected_nodes[5]
    
    def test_load_balancer_least_loaded_selection(self):
        """Test least-loaded node selection."""
        
        load_balancer = LoadBalancer(balancing_strategy="least_loaded")
        
        # Add nodes with different load factors
        node1 = ComputeNode("worker_1", "host1", 8001, {"cpu": 4}, status="idle", load_factor=0.8)
        node2 = ComputeNode("worker_2", "host2", 8002, {"cpu": 4}, status="idle", load_factor=0.2)
        node3 = ComputeNode("worker_3", "host3", 8003, {"cpu": 4}, status="idle", load_factor=0.5)
        
        for node in [node1, node2, node3]:
            load_balancer.register_node(node)
        
        task = DistributedTask("test_task", "pno_forward", {})
        
        # Should select least loaded node (worker_2)
        selected = load_balancer.select_node(task)
        assert selected.node_id == "worker_2"
    
    def test_load_balancer_capability_based_selection(self):
        """Test capability-based node selection."""
        
        load_balancer = LoadBalancer(balancing_strategy="capability_based")
        
        # Add nodes with different capabilities
        node1 = ComputeNode("cpu_worker", "host1", 8001, {"cpu_cores": 4, "memory_gb": 8}, status="idle")
        node2 = ComputeNode("gpu_worker", "host2", 8002, {"cpu_cores": 8, "memory_gb": 16, "gpu_memory_gb": 8}, status="idle")
        
        for node in [node1, node2]:
            load_balancer.register_node(node)
        
        # Task requiring GPU
        gpu_task = DistributedTask(
            "gpu_task", 
            "pno_forward", 
            {"required_capabilities": {"gpu_memory_gb": 4}}
        )
        
        selected = load_balancer.select_node(gpu_task)
        assert selected.node_id == "gpu_worker"
        
        # Task not requiring GPU (should select least loaded)
        cpu_task = DistributedTask("cpu_task", "pno_forward", {})
        
        selected = load_balancer.select_node(cpu_task)
        # Should select one of the available nodes
        assert selected.node_id in ["cpu_worker", "gpu_worker"]
    
    def test_distributed_task_queue_creation(self):
        """Test DistributedTaskQueue creation and basic operations."""
        
        queue = DistributedTaskQueue(max_size=100)
        
        assert queue.max_size == 100
        assert queue.tasks.empty()
        assert len(queue.task_registry) == 0
    
    def test_distributed_task_queue_submission_retrieval(self):
        """Test task submission and retrieval from queue."""
        
        queue = DistributedTaskQueue(max_size=10)
        
        # Submit tasks with different priorities
        task1 = DistributedTask("task_1", "pno_forward", {}, priority=1)
        task2 = DistributedTask("task_2", "uncertainty_sampling", {}, priority=3)
        task3 = DistributedTask("task_3", "pno_training_step", {}, priority=2)
        
        assert queue.submit_task(task1) is True
        assert queue.submit_task(task2) is True
        assert queue.submit_task(task3) is True
        
        assert len(queue.task_registry) == 3
        assert queue.tasks.qsize() == 3
        
        # Retrieve tasks (should come out in priority order)
        retrieved_task1 = queue.get_task()  # Highest priority (3)
        assert retrieved_task1.task_id == "task_2"
        assert retrieved_task1.status == "running"
        
        retrieved_task2 = queue.get_task()  # Medium priority (2)
        assert retrieved_task2.task_id == "task_3"
        
        retrieved_task3 = queue.get_task()  # Lowest priority (1)
        assert retrieved_task3.task_id == "task_1"
        
        # Queue should be empty now
        assert queue.get_task(timeout=0.1) is None
    
    def test_distributed_task_queue_completion(self):
        """Test task completion tracking."""
        
        queue = DistributedTaskQueue()
        
        task = DistributedTask("completion_test", "pno_forward", {})
        queue.submit_task(task)
        
        # Complete task successfully
        queue.complete_task("completion_test", result={"prediction": [1, 2, 3]})
        
        completed_task = queue.get_task_status("completion_test")
        assert completed_task.status == "completed"
        assert completed_task.result["prediction"] == [1, 2, 3]
        assert completed_task.error is None
        
        # Test failed completion
        failed_task = DistributedTask("failed_test", "pno_forward", {})
        queue.submit_task(failed_task)
        queue.complete_task("failed_test", error="Model failed")
        
        failed_completed = queue.get_task_status("failed_test")
        assert failed_completed.status == "failed"
        assert failed_completed.error == "Model failed"
    
    def test_distributed_task_queue_stats(self):
        """Test queue statistics."""
        
        queue = DistributedTaskQueue()
        
        # Add tasks with different statuses
        tasks = [
            DistributedTask(f"task_{i}", "pno_forward", {}) 
            for i in range(5)
        ]
        
        for task in tasks:
            queue.submit_task(task)
        
        # Complete some tasks
        queue.complete_task("task_0", result={})
        queue.complete_task("task_1", error="Failed")
        
        # Get one task (sets to running)
        queue.get_task()
        
        stats = queue.get_queue_stats()
        
        assert stats["total_tasks"] == 5
        assert stats["status_breakdown"]["completed"] == 1
        assert stats["status_breakdown"]["failed"] == 1
        assert stats["status_breakdown"]["running"] == 1
        assert stats["status_breakdown"]["pending"] == 2
    
    def test_distributed_pno_worker_creation(self):
        """Test DistributedPNOWorker creation."""
        
        worker = DistributedPNOWorker(
            worker_id="test_worker",
            capabilities={"cpu_cores": 4, "memory_gb": 8},
            heartbeat_interval=10.0
        )
        
        assert worker.worker_id == "test_worker"
        assert worker.capabilities["cpu_cores"] == 4
        assert worker.heartbeat_interval == 10.0
        assert worker.is_running is False
        assert worker.current_task is None
        assert worker.performance_metrics["tasks_completed"] == 0
    
    def test_distributed_pno_worker_task_execution(self):
        """Test worker task execution simulation."""
        
        worker = DistributedPNOWorker("test_worker", {})
        
        # Test PNO forward execution
        forward_task = DistributedTask(
            "forward_test",
            "pno_forward",
            {"input_data": [[1, 2], [3, 4]], "model_params": {}}
        )
        
        result = worker._execute_pno_forward(forward_task)
        
        assert isinstance(result, dict)
        assert "prediction" in result
        assert "uncertainty" in result
        assert "computation_time" in result
        assert len(result["prediction"]) == len(forward_task.payload["input_data"])
        
        # Test training step execution
        training_task = DistributedTask(
            "training_test",
            "pno_training_step",
            {"batch_data": [1, 2, 3], "learning_rate": 0.001}
        )
        
        result = worker._execute_training_step(training_task)
        
        assert isinstance(result, dict)
        assert "loss" in result
        assert "gradients" in result
        assert "batch_size" in result
        
        # Test uncertainty sampling execution
        sampling_task = DistributedTask(
            "sampling_test",
            "uncertainty_sampling",
            {"input_data": [1, 2, 3], "num_samples": 10}
        )
        
        result = worker._execute_uncertainty_sampling(sampling_task)
        
        assert isinstance(result, dict)
        assert "samples" in result
        assert "mean" in result
        assert "std" in result
        assert "num_samples" in result
        assert len(result["samples"]) == 10
    
    def test_distributed_pno_coordinator_creation(self):
        """Test DistributedPNOCoordinator creation."""
        
        coordinator = DistributedPNOCoordinator(port=8000)
        
        assert coordinator.port == 8000
        assert isinstance(coordinator.load_balancer, LoadBalancer)
        assert isinstance(coordinator.task_queue, DistributedTaskQueue)
        assert coordinator.is_running is False
        assert len(coordinator.workers) == 0
        assert coordinator.coordinator_stats["tasks_submitted"] == 0
    
    def test_distributed_pno_coordinator_task_submission(self):
        """Test coordinator task submission."""
        
        coordinator = DistributedPNOCoordinator()
        
        # Submit task
        task_id = coordinator.submit_task(
            task_type="pno_forward",
            payload={"input_data": [1, 2, 3]},
            priority=1
        )
        
        assert isinstance(task_id, str)
        assert task_id.startswith("task_")
        assert coordinator.coordinator_stats["tasks_submitted"] == 1
        
        # Check task is in queue
        task_status = coordinator.task_queue.get_task_status(task_id)
        assert task_status is not None
        assert task_status.task_type == "pno_forward"
        assert task_status.status == "pending"
    
    def test_distributed_pno_coordinator_worker_registration(self):
        """Test worker registration with coordinator."""
        
        coordinator = DistributedPNOCoordinator()
        
        # Register worker
        worker_capabilities = {"cpu_cores": 4, "memory_gb": 8, "gpu_available": False}
        coordinator.register_worker("test_worker", worker_capabilities)
        
        assert "test_worker" in coordinator.workers
        assert "test_worker" in coordinator.load_balancer.nodes
        
        # Check system status
        status = coordinator.get_system_status()
        assert status["active_workers"] == 1
        assert "test_worker" in status["worker_details"]
    
    def test_distributed_pno_cluster_creation(self):
        """Test DistributedPNOCluster creation."""
        
        cluster = DistributedPNOCluster(num_workers=3)
        
        assert cluster.num_workers == 3
        assert isinstance(cluster.coordinator, DistributedPNOCoordinator)
        assert len(cluster.workers) == 0  # Not started yet
    
    def test_distributed_pno_cluster_startup_simulation(self):
        """Test cluster startup simulation (without actual threads)."""
        
        cluster = DistributedPNOCluster(num_workers=2)
        
        # Simulate cluster start (without actually starting threads)
        cluster.coordinator.start()
        
        # Create worker objects (without starting them)
        for i in range(cluster.num_workers):
            worker = DistributedPNOWorker(
                worker_id=f"worker_{i}",
                capabilities={
                    "cpu_cores": 4,
                    "memory_gb": 8,
                    "gpu_available": False,
                    "supports_uncertainty": True
                }
            )
            cluster.workers.append(worker)
            cluster.coordinator.register_worker(worker.worker_id, worker.capabilities)
        
        assert len(cluster.workers) == 2
        assert cluster.coordinator.get_system_status()["active_workers"] == 2
        
        # Test cluster status
        status = cluster.get_cluster_status()
        assert isinstance(status, dict)
        assert "coordinator_status" in status
        assert "worker_metrics" in status
        assert status["cluster_size"] == 2
    
    def test_create_distributed_pno_system_factory(self):
        """Test distributed PNO system factory function."""
        
        config = {"num_workers": 4}
        cluster = create_distributed_pno_system(config)
        
        assert isinstance(cluster, DistributedPNOCluster)
        assert cluster.num_workers == 4


class TestPerformanceOptimization:
    """Test suite for performance optimization components."""
    
    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
    def test_memory_pool_creation(self):
        """Test MemoryPool creation."""
        
        pool = MemoryPool(max_size_mb=100)
        
        assert pool.max_size_bytes == 100 * 1024 * 1024
        assert pool.allocated_size == 0
        assert len(pool.free_tensors) == 0
        assert len(pool.used_tensors) == 0
    
    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
    def test_memory_pool_tensor_allocation(self):
        """Test tensor allocation from memory pool."""
        
        pool = MemoryPool(max_size_mb=10)  # Small pool for testing
        
        # Get tensor from pool
        tensor1 = pool.get_tensor((3, 4), dtype=torch.float32)
        
        assert tensor1 is not None
        assert tensor1.shape == (3, 4)
        assert tensor1.dtype == torch.float32
        assert len(pool.used_tensors) == 1
        assert pool.allocation_stats["pool_misses"] == 1  # First allocation
        
        # Return tensor to pool
        pool.return_tensor(tensor1)
        
        assert len(pool.used_tensors) == 0
        assert len(pool.free_tensors[tensor1.numel() * tensor1.element_size()]) == 1
        
        # Get tensor with same size (should reuse)
        tensor2 = pool.get_tensor((3, 4), dtype=torch.float32)
        
        assert tensor2 is not None
        assert pool.allocation_stats["pool_hits"] == 1  # Reused tensor
    
    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
    def test_memory_pool_size_limits(self):
        """Test memory pool size limits."""
        
        pool = MemoryPool(max_size_mb=1)  # Very small pool
        
        # Try to allocate large tensor
        large_tensor = pool.get_tensor((1000, 1000), dtype=torch.float32)  # ~4MB
        
        # Should return None due to size limit
        assert large_tensor is None
    
    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
    def test_memory_pool_stats(self):
        """Test memory pool statistics."""
        
        pool = MemoryPool(max_size_mb=10)
        
        # Allocate some tensors
        tensor1 = pool.get_tensor((2, 3))
        tensor2 = pool.get_tensor((4, 5))
        
        stats = pool.get_stats()
        
        assert isinstance(stats, dict)
        assert stats["allocated_size_mb"] > 0
        assert stats["utilization"] > 0
        assert stats["stats"]["total_allocations"] == 2
        assert stats["used_tensors_count"] == 2
    
    def test_compute_cache_creation(self):
        """Test ComputeCache creation."""
        
        cache = ComputeCache(max_entries=100, ttl_seconds=3600)
        
        assert cache.max_entries == 100
        assert cache.ttl_seconds == 3600
        assert len(cache.cache) == 0
        assert cache.stats["hits"] == 0
        assert cache.stats["misses"] == 0
    
    def test_compute_cache_basic_operations(self):
        """Test basic cache operations."""
        
        cache = ComputeCache(max_entries=10, ttl_seconds=1.0)
        
        # Cache miss
        result = cache.get("test_key")
        assert result is None
        assert cache.stats["misses"] == 1
        
        # Store value
        cache.put("test_key", {"result": "test_value"})
        
        # Cache hit
        result = cache.get("test_key")
        assert result is not None
        assert result["result"] == "test_value"
        assert cache.stats["hits"] == 1
    
    def test_compute_cache_ttl_expiration(self):
        """Test cache TTL expiration."""
        
        cache = ComputeCache(max_entries=10, ttl_seconds=0.1)  # Very short TTL
        
        # Store value
        cache.put("expire_test", {"value": 123})
        
        # Should be available immediately
        result = cache.get("expire_test")
        assert result is not None
        
        # Wait for expiration
        time.sleep(0.2)
        
        # Should be expired
        result = cache.get("expire_test")
        assert result is None
    
    def test_compute_cache_eviction(self):
        """Test cache LRU eviction."""
        
        cache = ComputeCache(max_entries=2, ttl_seconds=3600)  # Small cache
        
        # Fill cache
        cache.put("key1", "value1")
        cache.put("key2", "value2")
        
        # Access key1 to make it more recent
        cache.get("key1")
        
        # Add third item (should evict key2, the LRU)
        cache.put("key3", "value3")
        
        assert cache.get("key1") is not None  # Should still be there
        assert cache.get("key2") is None      # Should be evicted
        assert cache.get("key3") is not None  # Should be there
        assert cache.stats["evictions"] == 1
    
    def test_compute_cache_stats(self):
        """Test cache statistics."""
        
        cache = ComputeCache()
        
        # Perform some operations
        cache.get("miss1")  # Miss
        cache.get("miss2")  # Miss
        cache.put("hit_key", "value")
        cache.get("hit_key")  # Hit
        
        stats = cache.get_stats()
        
        assert isinstance(stats, dict)
        assert stats["stats"]["hits"] == 1
        assert stats["stats"]["misses"] == 2
        assert stats["stats"]["total_requests"] == 3
        assert stats["hit_rate"] == 1/3
        assert stats["size"] == 1
    
    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
    def test_adaptive_optimizer_creation(self):
        """Test AdaptiveOptimizer creation."""
        
        # Create base optimizer
        model = torch.nn.Linear(10, 1)
        base_optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        
        adaptive_opt = AdaptiveOptimizer(base_optimizer, adaptation_frequency=10)
        
        assert adaptive_opt.base_optimizer is base_optimizer
        assert adaptive_opt.adaptation_frequency == 10
        assert adaptive_opt.step_count == 0
        assert len(adaptive_opt.loss_history) == 0
    
    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
    def test_adaptive_optimizer_step(self):
        """Test adaptive optimizer step."""
        
        model = torch.nn.Linear(10, 1)
        base_optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        
        adaptive_opt = AdaptiveOptimizer(base_optimizer, adaptation_frequency=5)
        
        initial_lr = base_optimizer.param_groups[0]['lr']
        
        # Simulate plateau (constant loss)
        for i in range(25):  # Trigger adaptation
            adaptive_opt.step(loss=0.5)  # Constant loss (plateau)
        
        # Should have adapted learning rate
        final_lr = base_optimizer.param_groups[0]['lr']
        assert final_lr < initial_lr  # Learning rate should be reduced
        assert adaptive_opt.step_count == 25
    
    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
    def test_adaptive_optimizer_stats(self):
        """Test adaptive optimizer statistics."""
        
        model = torch.nn.Linear(10, 1)
        base_optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        
        adaptive_opt = AdaptiveOptimizer(base_optimizer)
        
        # Take some steps
        for i in range(5):
            adaptive_opt.step(loss=0.5 - i * 0.1)  # Decreasing loss
        
        stats = adaptive_opt.get_adaptation_stats()
        
        assert isinstance(stats, dict)
        assert stats["step_count"] == 5
        assert stats["current_lr"] == 0.01
        assert stats["recent_loss"] == 0.1
        assert stats["loss_trend"] == "decreasing"
    
    @pytest.mark.skipif(not HAS_PSUTIL, reason="psutil not available")
    def test_performance_profiler_creation(self):
        """Test PerformanceProfiler creation."""
        
        profiler = PerformanceProfiler()
        
        assert len(profiler.profiles) == 0
        assert len(profiler.active_timers) == 0
        assert len(profiler.call_stack) == 0
    
    @pytest.mark.skipif(not HAS_PSUTIL, reason="psutil not available")
    def test_performance_profiler_decoration(self):
        """Test performance profiling decoration."""
        
        profiler = PerformanceProfiler()
        
        @profiler.profile_operation("test_function")
        def test_function(x, y):
            time.sleep(0.01)  # Simulate work
            return x + y
        
        # Call function multiple times
        for i in range(3):
            result = test_function(i, i+1)
            assert result == 2*i + 1
        
        # Check profiling results
        assert "test_function" in profiler.profiles
        profile = profiler.profiles["test_function"]
        
        assert profile.num_calls == 3
        assert profile.total_time > 0.03  # At least 3 * 0.01 seconds
        assert profile.avg_time > 0.01
        assert profile.min_time > 0
        assert profile.max_time > profile.min_time
    
    @pytest.mark.skipif(not HAS_PSUTIL, reason="psutil not available")
    def test_performance_profiler_error_handling(self):
        """Test profiler error handling."""
        
        profiler = PerformanceProfiler()
        
        @profiler.profile_operation("error_function")
        def error_function():
            raise ValueError("Test error")
        
        # Function should still raise error
        with pytest.raises(ValueError, match="Test error"):
            error_function()
        
        # But profiling should have recorded the attempt
        assert "error_function" in profiler.profiles
        profile = profiler.profiles["error_function"]
        assert profile.num_calls == 1
        assert profile.total_time > 0
    
    @pytest.mark.skipif(not HAS_PSUTIL, reason="psutil not available")
    def test_performance_profiler_report(self):
        """Test performance report generation."""
        
        profiler = PerformanceProfiler()
        
        @profiler.profile_operation("fast_op")
        def fast_op():
            time.sleep(0.001)
        
        @profiler.profile_operation("slow_op")
        def slow_op():
            time.sleep(0.01)
        
        # Execute operations
        fast_op()
        slow_op()
        slow_op()  # Execute slow_op twice
        
        report = profiler.get_performance_report()
        
        assert isinstance(report, dict)
        assert report["total_operations"] == 2
        assert "operation_profiles" in report
        
        profiles = report["operation_profiles"]
        assert "fast_op" in profiles
        assert "slow_op" in profiles
        
        # slow_op should have higher total time
        assert profiles["slow_op"]["total_time"] > profiles["fast_op"]["total_time"]
        assert profiles["slow_op"]["num_calls"] == 2
        assert profiles["fast_op"]["num_calls"] == 1
    
    def test_performance_optimized_pno_creation(self):
        """Test PerformanceOptimizedPNO creation."""
        
        # Mock base model
        class MockModel:
            def predict_with_uncertainty(self, x, num_samples=100):
                return {"prediction": x * 0.9, "uncertainty": 0.1}
        
        base_model = MockModel()
        
        optimized_pno = PerformanceOptimizedPNO(
            base_model=base_model,
            enable_memory_pool=True,
            enable_compute_cache=True,
            enable_profiling=True
        )
        
        assert optimized_pno.base_model is base_model
        assert optimized_pno.memory_pool is not None
        assert optimized_pno.compute_cache is not None
        assert optimized_pno.profiler is not None
        assert optimized_pno.optimization_stats["cache_enabled"] is True
    
    def test_performance_optimized_pno_prediction(self):
        """Test optimized prediction functionality."""
        
        class MockModel:
            def predict_with_uncertainty(self, x, num_samples=100):
                time.sleep(0.001)  # Simulate computation
                if hasattr(x, 'shape'):
                    if hasattr(x, 'device'):  # PyTorch tensor
                        return {
                            "prediction": x * 0.9,
                            "uncertainty": torch.abs(x * 0.1)
                        }
                    else:  # NumPy array
                        return {
                            "prediction": x * 0.9,
                            "uncertainty": np.abs(x * 0.1)
                        }
                return {"prediction": x * 0.9, "uncertainty": 0.1}
        
        base_model = MockModel()
        optimized_pno = PerformanceOptimizedPNO(
            base_model,
            enable_compute_cache=True
        )
        
        # Test with NumPy array
        test_input = np.random.randn(2, 3, 4, 4)
        
        # First prediction (cache miss)
        start_time = time.time()
        result1 = optimized_pno.predict_with_uncertainty(test_input, num_samples=50)
        time1 = time.time() - start_time
        
        assert isinstance(result1, dict)
        assert "prediction" in result1
        assert "uncertainty" in result1
        
        # Second prediction with same input (cache hit)
        start_time = time.time()
        result2 = optimized_pno.predict_with_uncertainty(test_input, num_samples=50)
        time2 = time.time() - start_time
        
        # Second call should be faster due to caching
        assert time2 < time1
        assert optimized_pno.optimization_stats["optimized_calls"] > 0
    
    def test_performance_optimized_pno_stats(self):
        """Test optimization statistics collection."""
        
        class MockModel:
            def predict_with_uncertainty(self, x, num_samples=100):
                return {"prediction": x * 0.9, "uncertainty": 0.1}
        
        optimized_pno = PerformanceOptimizedPNO(
            MockModel(),
            enable_memory_pool=True,
            enable_compute_cache=True,
            enable_profiling=False  # Disable profiling to avoid psutil dependency
        )
        
        # Make some predictions
        test_input = np.random.randn(2, 3)
        for i in range(3):
            optimized_pno.predict_with_uncertainty(test_input)
        
        stats = optimized_pno.get_optimization_stats()
        
        assert isinstance(stats, dict)
        assert "cache_enabled" in stats
        assert "memory_pool_enabled" in stats
        assert "profiling_enabled" in stats
        
        if stats["memory_pool_enabled"]:
            assert "memory_pool" in stats
            assert isinstance(stats["memory_pool"], dict)
        
        if stats["cache_enabled"]:
            assert "compute_cache" in stats
            assert isinstance(stats["compute_cache"], dict)
    
    def test_create_optimized_pno_system_factory(self):
        """Test optimized PNO system factory function."""
        
        class MockModel:
            def predict_with_uncertainty(self, x, num_samples=100):
                return {"prediction": x * 0.9, "uncertainty": 0.1}
        
        model = MockModel()
        
        # Default optimization
        optimized_system = create_optimized_pno_system(model)
        
        assert isinstance(optimized_system, PerformanceOptimizedPNO)
        assert optimized_system.base_model is model
        
        # Custom optimization config
        custom_config = {
            "enable_memory_pool": False,
            "enable_compute_cache": True,
            "enable_profiling": False
        }
        
        custom_system = create_optimized_pno_system(model, custom_config)
        
        assert custom_system.memory_pool is None
        assert custom_system.compute_cache is not None
        assert custom_system.profiler is None


class TestIntegrationScalingPerformance:
    """Integration tests combining scaling and performance components."""
    
    def test_distributed_performance_integration(self):
        """Test integration of distributed computing with performance optimization."""
        
        # Create optimized model
        class MockOptimizedModel:
            def predict_with_uncertainty(self, x, num_samples=100):
                time.sleep(0.001)  # Simulate computation
                return {"prediction": x * 0.9, "uncertainty": 0.1}
        
        base_model = MockOptimizedModel()
        optimized_model = create_optimized_pno_system(
            base_model,
            {"enable_compute_cache": True, "enable_profiling": False}
        )
        
        # Create distributed system with optimized model
        cluster = DistributedPNOCluster(num_workers=2)
        
        # Simulate adding optimized model to workers
        for i in range(2):
            worker = DistributedPNOWorker(
                worker_id=f"optimized_worker_{i}",
                capabilities={
                    "cpu_cores": 4,
                    "memory_gb": 8,
                    "optimization_enabled": True,
                    "cache_enabled": True
                }
            )
            cluster.workers.append(worker)
        
        assert len(cluster.workers) == 2
        assert all(w.capabilities.get("optimization_enabled") for w in cluster.workers)
    
    def test_performance_monitoring_across_distributed_system(self):
        """Test performance monitoring across distributed components."""
        
        # Create coordinator with performance tracking
        coordinator = DistributedPNOCoordinator()
        
        # Register workers with performance capabilities
        for i in range(3):
            worker_capabilities = {
                "cpu_cores": 4,
                "memory_gb": 8,
                "performance_monitoring": True,
                "cache_size_mb": 100
            }
            coordinator.register_worker(f"perf_worker_{i}", worker_capabilities)
        
        # Submit tasks
        task_ids = []
        for i in range(5):
            task_id = coordinator.submit_task(
                task_type="pno_forward",
                payload={"input_data": list(range(i*10, (i+1)*10))},
                priority=1
            )
            task_ids.append(task_id)
        
        # Check system status includes performance info
        status = coordinator.get_system_status()
        
        assert status["active_workers"] == 3
        assert len(status["worker_details"]) == 3
        assert status["coordinator_stats"]["tasks_submitted"] == 5
        
        # Workers should have performance monitoring capabilities
        for worker_id in status["worker_details"]:
            assert worker_id in coordinator.load_balancer.nodes
            node = coordinator.load_balancer.nodes[worker_id]
            assert node.capabilities.get("performance_monitoring") is True
    
    def test_scalable_caching_system(self):
        """Test caching system scalability."""
        
        # Create multiple cache instances (simulating distributed caches)
        caches = [ComputeCache(max_entries=50, ttl_seconds=60) for _ in range(3)]
        
        # Simulate distributed cache operations
        test_data = [(f"key_{i}", f"value_{i}") for i in range(100)]
        
        # Distribute data across caches (simple hash-based distribution)
        for key, value in test_data:
            cache_index = hash(key) % len(caches)
            caches[cache_index].put(key, value)
        
        # Verify data distribution
        total_entries = sum(len(cache.cache) for cache in caches)
        assert total_entries <= 100  # Some may be evicted due to size limits
        
        # Test cache hit rates
        hits = 0
        total_requests = 0
        
        for key, expected_value in test_data[:50]:  # Test first 50
            cache_index = hash(key) % len(caches)
            result = caches[cache_index].get(key)
            total_requests += 1
            
            if result is not None:
                hits += 1
                assert result == expected_value
        
        hit_rate = hits / total_requests if total_requests > 0 else 0
        
        # Should have reasonable hit rate (accounting for evictions)
        assert hit_rate >= 0.3  # At least 30% hit rate
        
        # Get combined stats
        combined_stats = {
            "total_caches": len(caches),
            "total_entries": sum(len(cache.cache) for cache in caches),
            "combined_hit_rate": hit_rate,
            "individual_stats": [cache.get_stats() for cache in caches]
        }
        
        assert isinstance(combined_stats, dict)
        assert combined_stats["total_caches"] == 3
        assert len(combined_stats["individual_stats"]) == 3


def test_concurrent_operations():
    """Test concurrent operations across scaling and performance components."""
    
    # Test concurrent cache operations
    cache = ComputeCache(max_entries=100, ttl_seconds=3600)
    
    def cache_worker(worker_id, num_operations):
        for i in range(num_operations):
            key = f"worker_{worker_id}_key_{i}"
            value = f"worker_{worker_id}_value_{i}"
            
            # Put and get operations
            cache.put(key, value)
            result = cache.get(key)
            assert result == value
    
    # Run concurrent cache operations
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [
            executor.submit(cache_worker, i, 20) 
            for i in range(3)
        ]
        
        # Wait for completion
        for future in futures:
            future.result()  # Will raise exception if worker failed
    
    # Check final cache state
    stats = cache.get_stats()
    assert stats["stats"]["total_requests"] == 3 * 20  # 3 workers × 20 gets each
    assert stats["stats"]["hits"] + stats["stats"]["misses"] == stats["stats"]["total_requests"]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])