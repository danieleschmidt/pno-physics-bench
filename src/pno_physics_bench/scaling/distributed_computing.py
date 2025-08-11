"""Distributed computing and high-performance scaling for PNO systems."""

import time
import json
import logging
import threading
import multiprocessing as mp
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
from dataclasses import dataclass
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import queue
import socket
import pickle

try:
    import torch
    import torch.nn as nn
    import torch.distributed as dist
    import torch.multiprocessing as torch_mp
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import ray
    HAS_RAY = True
except ImportError:
    HAS_RAY = False

import numpy as np


@dataclass 
class ComputeNode:
    """Represents a compute node in distributed system."""
    node_id: str
    hostname: str
    port: int
    capabilities: Dict[str, Any]
    status: str = "idle"  # idle, busy, error, offline
    load_factor: float = 0.0
    last_heartbeat: float = 0.0


@dataclass
class DistributedTask:
    """Represents a distributed computation task."""
    task_id: str
    task_type: str
    payload: Dict[str, Any]
    priority: int = 0
    assigned_node: Optional[str] = None
    status: str = "pending"  # pending, running, completed, failed
    created_at: float = 0.0
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    result: Any = None
    error: Optional[str] = None


class LoadBalancer:
    """Intelligent load balancer for distributed PNO computations."""
    
    def __init__(self, balancing_strategy: str = "round_robin"):
        self.balancing_strategy = balancing_strategy
        self.nodes = {}
        self.node_stats = defaultdict(lambda: {"tasks_completed": 0, "avg_completion_time": 0.0})
        self.current_node_index = 0
        self.logger = logging.getLogger(__name__)
    
    def register_node(self, node: ComputeNode):
        """Register a new compute node."""
        self.nodes[node.node_id] = node
        self.logger.info(f"Registered compute node: {node.node_id}")
    
    def unregister_node(self, node_id: str):
        """Unregister a compute node."""
        if node_id in self.nodes:
            del self.nodes[node_id]
            self.logger.info(f"Unregistered compute node: {node_id}")
    
    def select_node(self, task: DistributedTask) -> Optional[ComputeNode]:
        """Select optimal node for task execution."""
        
        available_nodes = [node for node in self.nodes.values() 
                          if node.status == "idle"]
        
        if not available_nodes:
            return None
        
        if self.balancing_strategy == "round_robin":
            return self._round_robin_selection(available_nodes)
        elif self.balancing_strategy == "least_loaded":
            return self._least_loaded_selection(available_nodes)
        elif self.balancing_strategy == "capability_based":
            return self._capability_based_selection(available_nodes, task)
        elif self.balancing_strategy == "performance_based":
            return self._performance_based_selection(available_nodes)
        else:
            return available_nodes[0]  # Fallback
    
    def _round_robin_selection(self, nodes: List[ComputeNode]) -> ComputeNode:
        """Round-robin node selection."""
        selected_node = nodes[self.current_node_index % len(nodes)]
        self.current_node_index += 1
        return selected_node
    
    def _least_loaded_selection(self, nodes: List[ComputeNode]) -> ComputeNode:
        """Select node with lowest load."""
        return min(nodes, key=lambda node: node.load_factor)
    
    def _capability_based_selection(self, nodes: List[ComputeNode], 
                                  task: DistributedTask) -> ComputeNode:
        """Select node based on required capabilities."""
        
        # Check if task requires specific capabilities
        required_capabilities = task.payload.get("required_capabilities", {})
        
        suitable_nodes = []
        for node in nodes:
            suitable = True
            for capability, required_value in required_capabilities.items():
                if capability not in node.capabilities:
                    suitable = False
                    break
                if node.capabilities[capability] < required_value:
                    suitable = False
                    break
            
            if suitable:
                suitable_nodes.append(node)
        
        if not suitable_nodes:
            return nodes[0]  # Fallback to any available node
        
        # Among suitable nodes, select least loaded
        return min(suitable_nodes, key=lambda node: node.load_factor)
    
    def _performance_based_selection(self, nodes: List[ComputeNode]) -> ComputeNode:
        """Select node based on historical performance."""
        
        best_node = None
        best_score = float('inf')
        
        for node in nodes:
            stats = self.node_stats[node.node_id]
            # Score based on average completion time and current load
            score = stats["avg_completion_time"] * (1 + node.load_factor)
            
            if score < best_score:
                best_score = score
                best_node = node
        
        return best_node or nodes[0]
    
    def update_node_stats(self, node_id: str, completion_time: float):
        """Update node performance statistics."""
        stats = self.node_stats[node_id]
        stats["tasks_completed"] += 1
        
        # Exponential moving average for completion time
        alpha = 0.1
        if stats["avg_completion_time"] == 0:
            stats["avg_completion_time"] = completion_time
        else:
            stats["avg_completion_time"] = (
                alpha * completion_time + (1 - alpha) * stats["avg_completion_time"]
            )


class DistributedTaskQueue:
    """Thread-safe distributed task queue with priority support."""
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.tasks = queue.PriorityQueue(maxsize=max_size)
        self.task_registry = {}  # task_id -> task
        self.lock = threading.Lock()
        self.logger = logging.getLogger(__name__)
    
    def submit_task(self, task: DistributedTask) -> bool:
        """Submit a task to the queue."""
        try:
            with self.lock:
                if len(self.task_registry) >= self.max_size:
                    self.logger.warning("Task queue is full")
                    return False
                
                # Priority queue uses negative priority for max-heap behavior
                self.tasks.put((-task.priority, time.time(), task))
                self.task_registry[task.task_id] = task
                task.created_at = time.time()
                
                self.logger.debug(f"Task submitted: {task.task_id}")
                return True
                
        except queue.Full:
            self.logger.error("Failed to submit task: queue full")
            return False
    
    def get_task(self, timeout: Optional[float] = None) -> Optional[DistributedTask]:
        """Get next task from queue."""
        try:
            _, _, task = self.tasks.get(timeout=timeout)
            task.status = "running"
            task.started_at = time.time()
            return task
            
        except queue.Empty:
            return None
    
    def complete_task(self, task_id: str, result: Any = None, error: str = None):
        """Mark task as completed."""
        with self.lock:
            if task_id in self.task_registry:
                task = self.task_registry[task_id]
                task.completed_at = time.time()
                task.result = result
                task.error = error
                task.status = "completed" if error is None else "failed"
                
                self.logger.debug(f"Task completed: {task_id}")
    
    def get_task_status(self, task_id: str) -> Optional[DistributedTask]:
        """Get task status."""
        with self.lock:
            return self.task_registry.get(task_id)
    
    def get_queue_stats(self) -> Dict[str, int]:
        """Get queue statistics."""
        with self.lock:
            status_counts = defaultdict(int)
            for task in self.task_registry.values():
                status_counts[task.status] += 1
            
            return {
                "total_tasks": len(self.task_registry),
                "queued_tasks": self.tasks.qsize(),
                "status_breakdown": dict(status_counts)
            }


class DistributedPNOWorker:
    """Worker node for distributed PNO computations."""
    
    def __init__(self, 
                 worker_id: str,
                 capabilities: Dict[str, Any],
                 heartbeat_interval: float = 30.0):
        
        self.worker_id = worker_id
        self.capabilities = capabilities
        self.heartbeat_interval = heartbeat_interval
        
        self.is_running = False
        self.current_task = None
        self.task_history = []
        
        self.logger = logging.getLogger(f"worker_{worker_id}")
        
        # Performance monitoring
        self.performance_metrics = {
            "tasks_completed": 0,
            "total_compute_time": 0.0,
            "avg_task_time": 0.0,
            "error_count": 0
        }
    
    def start(self, coordinator_host: str = "localhost", coordinator_port: int = 8000):
        """Start worker and connect to coordinator."""
        self.is_running = True
        self.coordinator_host = coordinator_host
        self.coordinator_port = coordinator_port
        
        # Start heartbeat thread
        heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        heartbeat_thread.start()
        
        # Start main work loop
        self._work_loop()
    
    def stop(self):
        """Stop worker."""
        self.is_running = False
        self.logger.info(f"Worker {self.worker_id} stopped")
    
    def _heartbeat_loop(self):
        """Send periodic heartbeat to coordinator."""
        while self.is_running:
            try:
                self._send_heartbeat()
                time.sleep(self.heartbeat_interval)
            except Exception as e:
                self.logger.error(f"Heartbeat error: {e}")
                time.sleep(5)  # Retry after 5 seconds
    
    def _send_heartbeat(self):
        """Send heartbeat message to coordinator."""
        heartbeat_data = {
            "worker_id": self.worker_id,
            "timestamp": time.time(),
            "status": "busy" if self.current_task else "idle",
            "capabilities": self.capabilities,
            "performance_metrics": self.performance_metrics
        }
        
        # In real implementation, would send via network
        self.logger.debug(f"Heartbeat sent: {heartbeat_data}")
    
    def _work_loop(self):
        """Main work loop to process tasks."""
        while self.is_running:
            try:
                # In real implementation, would request tasks from coordinator
                task = self._request_task()
                
                if task:
                    self._execute_task(task)
                else:
                    time.sleep(1)  # Wait before checking again
                    
            except Exception as e:
                self.logger.error(f"Work loop error: {e}")
                self.performance_metrics["error_count"] += 1
                time.sleep(5)
    
    def _request_task(self) -> Optional[DistributedTask]:
        """Request task from coordinator."""
        # Placeholder - in real implementation would make network request
        return None
    
    def _execute_task(self, task: DistributedTask):
        """Execute a distributed task."""
        self.current_task = task
        start_time = time.time()
        
        try:
            self.logger.info(f"Executing task: {task.task_id}")
            
            # Execute task based on type
            if task.task_type == "pno_forward":
                result = self._execute_pno_forward(task)
            elif task.task_type == "pno_training_step":
                result = self._execute_training_step(task)
            elif task.task_type == "uncertainty_sampling":
                result = self._execute_uncertainty_sampling(task)
            else:
                raise ValueError(f"Unknown task type: {task.task_type}")
            
            # Update performance metrics
            execution_time = time.time() - start_time
            self.performance_metrics["tasks_completed"] += 1
            self.performance_metrics["total_compute_time"] += execution_time
            self.performance_metrics["avg_task_time"] = (
                self.performance_metrics["total_compute_time"] / 
                self.performance_metrics["tasks_completed"]
            )
            
            self.task_history.append({
                "task_id": task.task_id,
                "execution_time": execution_time,
                "success": True
            })
            
            # Send result back to coordinator
            self._send_result(task.task_id, result)
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Task execution failed: {e}")
            
            self.performance_metrics["error_count"] += 1
            self.task_history.append({
                "task_id": task.task_id,
                "execution_time": execution_time,
                "success": False,
                "error": str(e)
            })
            
            self._send_error(task.task_id, str(e))
        
        finally:
            self.current_task = None
    
    def _execute_pno_forward(self, task: DistributedTask) -> Dict[str, Any]:
        """Execute PNO forward pass."""
        input_data = task.payload.get("input_data")
        model_params = task.payload.get("model_params")
        
        # Simulate PNO forward computation
        if isinstance(input_data, (list, tuple)):
            input_data = np.array(input_data)
        
        # Mock computation - replace with actual PNO forward pass
        output = input_data * 0.9 + np.random.normal(0, 0.1, input_data.shape)
        uncertainty = np.abs(output * 0.1)
        
        return {
            "prediction": output.tolist(),
            "uncertainty": uncertainty.tolist(),
            "computation_time": time.time()
        }
    
    def _execute_training_step(self, task: DistributedTask) -> Dict[str, Any]:
        """Execute distributed training step."""
        batch_data = task.payload.get("batch_data")
        gradients = task.payload.get("gradients")
        learning_rate = task.payload.get("learning_rate", 0.001)
        
        # Simulate training computation
        time.sleep(0.1)  # Simulate computation time
        
        # Mock gradient computation
        loss = np.random.random() * 0.5
        new_gradients = {f"layer_{i}": np.random.randn(10, 10) for i in range(3)}
        
        return {
            "loss": loss,
            "gradients": new_gradients,
            "batch_size": len(batch_data) if batch_data else 0
        }
    
    def _execute_uncertainty_sampling(self, task: DistributedTask) -> Dict[str, Any]:
        """Execute uncertainty sampling computation."""
        num_samples = task.payload.get("num_samples", 100)
        input_data = task.payload.get("input_data")
        
        # Simulate uncertainty sampling
        samples = []
        for _ in range(num_samples):
            if isinstance(input_data, (list, tuple)):
                sample = np.array(input_data) + np.random.normal(0, 0.05, np.array(input_data).shape)
            else:
                sample = input_data + np.random.normal(0, 0.05)
            samples.append(sample.tolist() if hasattr(sample, 'tolist') else sample)
        
        # Compute statistics
        if samples:
            samples_array = np.array(samples)
            mean_prediction = np.mean(samples_array, axis=0)
            std_prediction = np.std(samples_array, axis=0)
        else:
            mean_prediction = input_data
            std_prediction = 0.0
        
        return {
            "samples": samples,
            "mean": mean_prediction.tolist() if hasattr(mean_prediction, 'tolist') else mean_prediction,
            "std": std_prediction.tolist() if hasattr(std_prediction, 'tolist') else std_prediction,
            "num_samples": len(samples)
        }
    
    def _send_result(self, task_id: str, result: Dict[str, Any]):
        """Send task result to coordinator."""
        # Placeholder - in real implementation would send via network
        self.logger.debug(f"Sending result for task {task_id}")
    
    def _send_error(self, task_id: str, error: str):
        """Send task error to coordinator."""
        # Placeholder - in real implementation would send via network
        self.logger.error(f"Sending error for task {task_id}: {error}")


class DistributedPNOCoordinator:
    """Coordinator for distributed PNO computations."""
    
    def __init__(self, port: int = 8000):
        self.port = port
        self.load_balancer = LoadBalancer(balancing_strategy="performance_based")
        self.task_queue = DistributedTaskQueue()
        
        self.workers = {}  # worker_id -> last_heartbeat
        self.active_tasks = {}  # task_id -> (worker_id, start_time)
        
        self.is_running = False
        self.coordinator_stats = {
            "tasks_submitted": 0,
            "tasks_completed": 0,
            "tasks_failed": 0,
            "active_workers": 0
        }
        
        self.logger = logging.getLogger(__name__)
    
    def start(self):
        """Start coordinator."""
        self.is_running = True
        
        # Start task scheduler thread
        scheduler_thread = threading.Thread(target=self._task_scheduler, daemon=True)
        scheduler_thread.start()
        
        # Start worker monitor thread
        monitor_thread = threading.Thread(target=self._worker_monitor, daemon=True)
        monitor_thread.start()
        
        self.logger.info(f"Coordinator started on port {self.port}")
    
    def stop(self):
        """Stop coordinator."""
        self.is_running = False
        self.logger.info("Coordinator stopped")
    
    def submit_task(self, task_type: str, payload: Dict[str, Any], 
                   priority: int = 0) -> str:
        """Submit task for distributed execution."""
        
        task_id = f"task_{int(time.time())}_{np.random.randint(1000, 9999)}"
        task = DistributedTask(
            task_id=task_id,
            task_type=task_type,
            payload=payload,
            priority=priority
        )
        
        if self.task_queue.submit_task(task):
            self.coordinator_stats["tasks_submitted"] += 1
            self.logger.info(f"Task submitted: {task_id}")
            return task_id
        else:
            raise RuntimeError("Failed to submit task: queue full")
    
    def get_task_result(self, task_id: str, timeout: float = 60.0) -> Optional[Dict[str, Any]]:
        """Get task result (blocking)."""
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            task = self.task_queue.get_task_status(task_id)
            
            if task and task.status in ["completed", "failed"]:
                if task.status == "completed":
                    return {
                        "success": True,
                        "result": task.result,
                        "execution_time": task.completed_at - task.started_at if task.started_at else 0
                    }
                else:
                    return {
                        "success": False,
                        "error": task.error,
                        "execution_time": task.completed_at - task.started_at if task.started_at else 0
                    }
            
            time.sleep(0.1)
        
        return None  # Timeout
    
    def _task_scheduler(self):
        """Schedule tasks to available workers."""
        while self.is_running:
            try:
                task = self.task_queue.get_task(timeout=1.0)
                
                if task:
                    # Find available worker
                    selected_node = self.load_balancer.select_node(task)
                    
                    if selected_node:
                        task.assigned_node = selected_node.node_id
                        self.active_tasks[task.task_id] = (selected_node.node_id, time.time())
                        
                        # In real implementation, would send task to worker
                        self._dispatch_task_to_worker(task, selected_node)
                    else:
                        # No available workers, put task back
                        self.task_queue.submit_task(task)
                        time.sleep(1)
                
            except Exception as e:
                self.logger.error(f"Task scheduler error: {e}")
                time.sleep(1)
    
    def _dispatch_task_to_worker(self, task: DistributedTask, node: ComputeNode):
        """Dispatch task to worker node."""
        # Placeholder - in real implementation would send via network
        self.logger.debug(f"Dispatching task {task.task_id} to worker {node.node_id}")
        
        # Update node status
        node.status = "busy"
        node.load_factor = min(1.0, node.load_factor + 0.1)
    
    def _worker_monitor(self):
        """Monitor worker health and cleanup stale workers."""
        while self.is_running:
            try:
                current_time = time.time()
                stale_workers = []
                
                for worker_id, last_heartbeat in self.workers.items():
                    if current_time - last_heartbeat > 60:  # 1 minute timeout
                        stale_workers.append(worker_id)
                
                for worker_id in stale_workers:
                    self.logger.warning(f"Removing stale worker: {worker_id}")
                    del self.workers[worker_id]
                    self.load_balancer.unregister_node(worker_id)
                
                self.coordinator_stats["active_workers"] = len(self.workers)
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Worker monitor error: {e}")
                time.sleep(30)
    
    def register_worker(self, worker_id: str, capabilities: Dict[str, Any]):
        """Register a new worker."""
        
        node = ComputeNode(
            node_id=worker_id,
            hostname="localhost",  # Would be actual hostname
            port=0,  # Would be actual port
            capabilities=capabilities
        )
        
        self.load_balancer.register_node(node)
        self.workers[worker_id] = time.time()
        
        self.logger.info(f"Worker registered: {worker_id}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        
        queue_stats = self.task_queue.get_queue_stats()
        
        return {
            "coordinator_stats": self.coordinator_stats,
            "queue_stats": queue_stats,
            "active_workers": len(self.workers),
            "worker_details": list(self.workers.keys()),
            "load_balancer_strategy": self.load_balancer.balancing_strategy,
            "active_tasks": len(self.active_tasks)
        }


class DistributedPNOCluster:
    """High-level interface for distributed PNO cluster."""
    
    def __init__(self, num_workers: int = 4):
        self.num_workers = num_workers
        self.coordinator = DistributedPNOCoordinator()
        self.workers = []
        
        self.logger = logging.getLogger(__name__)
    
    def start_cluster(self):
        """Start distributed cluster."""
        
        # Start coordinator
        self.coordinator.start()
        
        # Start workers
        for i in range(self.num_workers):
            worker = DistributedPNOWorker(
                worker_id=f"worker_{i}",
                capabilities={
                    "cpu_cores": 4,
                    "memory_gb": 8,
                    "gpu_available": False,
                    "supports_uncertainty": True
                }
            )
            
            self.workers.append(worker)
            
            # Register worker with coordinator
            self.coordinator.register_worker(worker.worker_id, worker.capabilities)
        
        self.logger.info(f"Distributed cluster started with {len(self.workers)} workers")
    
    def stop_cluster(self):
        """Stop distributed cluster."""
        
        for worker in self.workers:
            worker.stop()
        
        self.coordinator.stop()
        self.logger.info("Distributed cluster stopped")
    
    def distributed_predict(self, input_batches: List[np.ndarray], 
                          num_uncertainty_samples: int = 100) -> Dict[str, Any]:
        """Perform distributed prediction with uncertainty quantification."""
        
        task_ids = []
        
        # Submit forward pass tasks
        for i, batch in enumerate(input_batches):
            task_id = self.coordinator.submit_task(
                task_type="pno_forward",
                payload={
                    "input_data": batch.tolist(),
                    "batch_id": i
                },
                priority=1
            )
            task_ids.append(task_id)
        
        # Submit uncertainty sampling tasks
        uncertainty_task_ids = []
        for i, batch in enumerate(input_batches):
            task_id = self.coordinator.submit_task(
                task_type="uncertainty_sampling",
                payload={
                    "input_data": batch.tolist(),
                    "num_samples": num_uncertainty_samples,
                    "batch_id": i
                },
                priority=0  # Lower priority than forward passes
            )
            uncertainty_task_ids.append(task_id)
        
        # Collect results
        predictions = []
        uncertainties = []
        
        # Get forward pass results
        for task_id in task_ids:
            result = self.coordinator.get_task_result(task_id, timeout=30.0)
            if result and result["success"]:
                predictions.append(result["result"]["prediction"])
            else:
                self.logger.error(f"Forward pass task failed: {task_id}")
                predictions.append(None)
        
        # Get uncertainty sampling results
        for task_id in uncertainty_task_ids:
            result = self.coordinator.get_task_result(task_id, timeout=60.0)
            if result and result["success"]:
                uncertainties.append(result["result"])
            else:
                self.logger.error(f"Uncertainty sampling task failed: {task_id}")
                uncertainties.append(None)
        
        return {
            "predictions": predictions,
            "uncertainties": uncertainties,
            "num_batches": len(input_batches),
            "num_samples_per_batch": num_uncertainty_samples
        }
    
    def get_cluster_status(self) -> Dict[str, Any]:
        """Get cluster status and performance metrics."""
        
        coordinator_status = self.coordinator.get_system_status()
        
        worker_metrics = []
        for worker in self.workers:
            worker_metrics.append({
                "worker_id": worker.worker_id,
                "performance_metrics": worker.performance_metrics,
                "is_running": worker.is_running,
                "current_task": worker.current_task.task_id if worker.current_task else None
            })
        
        return {
            "coordinator_status": coordinator_status,
            "worker_metrics": worker_metrics,
            "cluster_size": len(self.workers)
        }


def create_distributed_pno_system(config: Dict[str, Any]) -> DistributedPNOCluster:
    """Factory function to create distributed PNO system."""
    
    num_workers = config.get("num_workers", 4)
    cluster = DistributedPNOCluster(num_workers=num_workers)
    
    return cluster


if __name__ == "__main__":
    print("Distributed Computing System for PNO Models")
    print("=" * 50)
    
    # Example usage
    config = {
        "num_workers": 3,
        "balancing_strategy": "performance_based"
    }
    
    # Create and start cluster
    cluster = create_distributed_pno_system(config)
    cluster.start_cluster()
    
    print(f"Started cluster with {cluster.num_workers} workers")
    
    # Simulate distributed computation
    input_batches = [
        np.random.randn(2, 3, 16, 16),
        np.random.randn(2, 3, 16, 16),
        np.random.randn(2, 3, 16, 16)
    ]
    
    print("Submitting distributed prediction tasks...")
    
    # Would perform actual distributed computation
    # results = cluster.distributed_predict(input_batches, num_uncertainty_samples=50)
    
    # Get cluster status
    status = cluster.get_cluster_status()
    print(f"\nCluster Status:")
    print(f"- Active workers: {status['coordinator_status']['active_workers']}")
    print(f"- Tasks submitted: {status['coordinator_status']['coordinator_stats']['tasks_submitted']}")
    print(f"- Tasks completed: {status['coordinator_status']['coordinator_stats']['tasks_completed']}")
    
    # Stop cluster
    cluster.stop_cluster()
    print("\nDistributed computing system demonstration completed!")