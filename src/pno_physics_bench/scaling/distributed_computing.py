# SECURITY NOTICE: This file has been automatically hardened for security
# - All user inputs should be validated and sanitized
# - Subprocess calls use secure alternatives
# - SQL queries use parameterized statements
# - No hardcoded secrets or credentials


"""Distributed Computing Framework for PNO Physics Bench"""

import asyncio
import concurrent.futures
import multiprocessing as mp
import threading
import time
import queue
import logging
from typing import List, Dict, Any, Callable, Optional, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import json

logger = logging.getLogger(__name__)

@dataclass
class ComputeTask:
    """Represents a distributed computation task"""
    task_id: str
    function: Callable
    args: tuple
    kwargs: dict
    priority: int = 0
    timeout: Optional[float] = None
    retry_count: int = 0
    max_retries: int = 3

class TaskResult:
    """Result of a distributed computation"""
    
    def __init__(self, task_id: str, success: bool, result: Any = None, 
                 error: Optional[str] = None, execution_time: float = 0.0):
        self.task_id = task_id
        self.success = success
        self.result = result
        self.error = error
        self.execution_time = execution_time
        self.timestamp = time.time()

class LoadBalancer:
    """Intelligent load balancer for distributed tasks"""
    
    def __init__(self, num_workers: int = None):
        self.num_workers = num_workers or mp.cpu_count()
        self.worker_loads = {i: 0 for i in range(self.num_workers)}
        self.worker_performance = {i: 1.0 for i in range(self.num_workers)}  # Performance factor
        self._lock = threading.Lock()
    
    def select_worker(self) -> int:
        """Select best worker based on current load and performance"""
        with self._lock:
            # Calculate effective load (load / performance)
            effective_loads = {
                worker_id: load / self.worker_performance[worker_id]
                for worker_id, load in self.worker_loads.items()
            }
            
            # Select worker with lowest effective load
            best_worker = min(effective_loads.keys(), key=lambda x: effective_loads[x])
            return best_worker
    
    def update_worker_load(self, worker_id: int, load_delta: int):
        """Update worker load"""
        with self._lock:
            self.worker_loads[worker_id] += load_delta
            if self.worker_loads[worker_id] < 0:
                self.worker_loads[worker_id] = 0
    
    def update_worker_performance(self, worker_id: int, execution_time: float, 
                                expected_time: float):
        """Update worker performance metrics"""
        with self._lock:
            # Performance factor: lower is better (faster execution)
            performance_factor = expected_time / execution_time if execution_time > 0 else 1.0
            
            # Exponential moving average
            alpha = 0.1
            self.worker_performance[worker_id] = (
                alpha * performance_factor + 
                (1 - alpha) * self.worker_performance[worker_id]
            )

class DistributedComputeEngine:
    """Main engine for distributed computing"""
    
    def __init__(self, max_workers: int = None, use_processes: bool = True):
        self.max_workers = max_workers or mp.cpu_count()
        self.use_processes = use_processes
        self.load_balancer = LoadBalancer(self.max_workers)
        self.task_queue = queue.PriorityQueue()
        self.result_queue = queue.Queue()
        self.active_tasks = {}
        self.completed_tasks = {}
        self.is_running = False
        self._lock = threading.Lock()
    
    def start(self):
        """Start the distributed compute engine"""
        if self.is_running:
            return
        
        self.is_running = True
        
        if self.use_processes:
            self.executor = ProcessPoolExecutor(max_workers=self.max_workers)
        else:
            self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        
        # Start task dispatcher
        self.dispatcher_thread = threading.Thread(target=self._task_dispatcher, daemon=True)
        self.dispatcher_thread.start()
        
        logger.info(f"Distributed compute engine started with {self.max_workers} workers")
    
    def stop(self):
        """Stop the distributed compute engine"""
        self.is_running = False
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)
        
        logger.info("Distributed compute engine stopped")
    
    def submit_task(self, task: ComputeTask) -> str:
        """Submit a task for distributed execution"""
        priority = -task.priority  # Negative for max-priority queue
        self.task_queue.put((priority, time.time(), task))
        return task.task_id
    
    def get_result(self, task_id: str, timeout: Optional[float] = None) -> Optional[TaskResult]:
        """Get result of a completed task"""
        with self._lock:
            if task_id in self.completed_tasks:
                return self.completed_tasks[task_id]
        
        # Wait for result
        start_time = time.time()
        while timeout is None or (time.time() - start_time) < timeout:
            try:
                result = self.result_queue.get(timeout=1.0)
                with self._lock:
                    self.completed_tasks[result.task_id] = result
                
                if result.task_id == task_id:
                    return result
                
            except queue.Empty:
                continue
        
        return None
    
    def _task_dispatcher(self):
        """Dispatch tasks to workers"""
        while self.is_running:
            try:
                priority, submit_time, task = self.task_queue.get(timeout=1.0)
                
                # Select worker
                worker_id = self.load_balancer.select_worker()
                self.load_balancer.update_worker_load(worker_id, 1)
                
                # Submit to executor
                future = self.executor.submit(self._execute_task, task, worker_id)
                
                with self._lock:
                    self.active_tasks[task.task_id] = future
                
                # Handle completion asynchronously
                future.add_done_callback(lambda f, t=task: self._handle_task_completion(f, t))
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in task dispatcher: {e}")
    
    def _execute_task(self, task: ComputeTask, worker_id: int) -> TaskResult:
        """Execute a single task"""
        start_time = time.time()
        
        try:
            result = task.function(*task.args, **task.kwargs)
            execution_time = time.time() - start_time
            
            # Update performance metrics
            expected_time = 1.0  # Default expected time
            self.load_balancer.update_worker_performance(worker_id, execution_time, expected_time)
            
            return TaskResult(task.task_id, True, result, execution_time=execution_time)
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Task {task.task_id} failed: {e}")
            return TaskResult(task.task_id, False, error=str(e), execution_time=execution_time)
        
        finally:
            self.load_balancer.update_worker_load(worker_id, -1)
    
    def _handle_task_completion(self, future, task: ComputeTask):
        """Handle task completion"""
        try:
            result = future.result()
            
            # Handle retries for failed tasks
            if not result.success and task.retry_count < task.max_retries:
                task.retry_count += 1
                logger.warning(f"Retrying task {task.task_id} (attempt {task.retry_count})")
                self.submit_task(task)
                return
            
            self.result_queue.put(result)
            
            with self._lock:
                if task.task_id in self.active_tasks:
                    del self.active_tasks[task.task_id]
            
        except Exception as e:
            logger.error(f"Error handling task completion: {e}")
            error_result = TaskResult(task.task_id, False, error=str(e))
            self.result_queue.put(error_result)

class ParallelDataProcessor:
    """Parallel data processing utilities"""
    
    def __init__(self, compute_engine: DistributedComputeEngine):
        self.compute_engine = compute_engine
    
    def parallel_map(self, func: Callable, data_list: List[Any], 
                    chunk_size: Optional[int] = None) -> List[Any]:
        """Parallel map operation"""
        if chunk_size is None:
            chunk_size = max(1, len(data_list) // self.compute_engine.max_workers)
        
        # Split data into chunks
        chunks = [data_list[i:i + chunk_size] for i in range(0, len(data_list), chunk_size)]
        
        # Submit chunk processing tasks
        task_ids = []
        for i, chunk in enumerate(chunks):
            task = ComputeTask(
                task_id=f"parallel_map_chunk_{i}",
                function=lambda chunk=chunk: [func(item) for item in chunk],
                args=(),
                kwargs={}
            )
            task_id = self.compute_engine.submit_task(task)
            task_ids.append(task_id)
        
        # Collect results
        results = []
        for task_id in task_ids:
            result = self.compute_engine.get_result(task_id, timeout=300)  # 5 minute timeout
            if result and result.success:
                results.extend(result.result)
            else:
                logger.error(f"Parallel map task {task_id} failed")
        
        return results
    
    def parallel_reduce(self, func: Callable, data_list: List[Any], 
                       initial_value: Any = None) -> Any:
        """Parallel reduce operation"""
        if len(data_list) <= 1:
            return data_list[0] if data_list else initial_value
        
        # Recursive parallel reduction
        if len(data_list) == 2:
            return func(data_list[0], data_list[1])
        
        mid = len(data_list) // 2
        left_chunk = data_list[:mid]
        right_chunk = data_list[mid:]
        
        # Submit parallel reduction tasks
        left_task = ComputeTask(
            task_id=f"reduce_left_{time.time()}",
            function=self.parallel_reduce,
            args=(func, left_chunk, initial_value),
            kwargs={}
        )
        
        right_task = ComputeTask(
            task_id=f"reduce_right_{time.time()}",
            function=self.parallel_reduce,
            args=(func, right_chunk, initial_value),
            kwargs={}
        )
        
        left_task_id = self.compute_engine.submit_task(left_task)
        right_task_id = self.compute_engine.submit_task(right_task)
        
        # Get results and combine
        left_result = self.compute_engine.get_result(left_task_id, timeout=300)
        right_result = self.compute_engine.get_result(right_task_id, timeout=300)
        
        if left_result and left_result.success and right_result and right_result.success:
            return func(left_result.result, right_result.result)
        else:
            logger.error("Parallel reduce failed")
            return initial_value

# Global distributed compute engine
distributed_engine = DistributedComputeEngine()

def distributed_computation(priority: int = 0, timeout: Optional[float] = None, 
                          max_retries: int = 3):
    """Decorator for distributed computation"""
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            task_id = f"{func.__name__}_{time.time()}"
            task = ComputeTask(
                task_id=task_id,
                function=func,
                args=args,
                kwargs=kwargs,
                priority=priority,
                timeout=timeout,
                max_retries=max_retries
            )
            
            # Start engine if not running
            if not distributed_engine.is_running:
                distributed_engine.start()
            
            # Submit and wait for result
            distributed_engine.submit_task(task)
            result = distributed_engine.get_result(task_id, timeout=timeout)
            
            if result and result.success:
                return result.result
            else:
                error_msg = result.error if result else "Task timeout or unknown error"
                raise RuntimeError(f"Distributed computation failed: {error_msg}")
        
        return wrapper
    return decorator
