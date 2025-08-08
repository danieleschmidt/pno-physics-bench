"""Distributed training system for PNO models."""

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import os
import logging
from typing import Dict, Any, Optional, Callable, List
import time
from contextlib import contextmanager
import socket
import subprocess


logger = logging.getLogger(__name__)


class DistributedTrainingManager:
    """Manager for distributed training across multiple nodes/GPUs."""
    
    def __init__(
        self,
        backend: str = "nccl",
        init_method: str = "env://",
        timeout_minutes: int = 30
    ):
        self.backend = backend
        self.init_method = init_method
        self.timeout = timeout_minutes * 60  # Convert to seconds
        
        self.world_size = None
        self.rank = None
        self.local_rank = None
        self.is_distributed = False
        
    def setup_distributed(
        self,
        rank: int,
        world_size: int,
        local_rank: int
    ):
        """Setup distributed training environment."""
        self.rank = rank
        self.world_size = world_size
        self.local_rank = local_rank
        
        # Set CUDA device
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        
        try:
            # Initialize process group
            dist.init_process_group(
                backend=self.backend,
                init_method=self.init_method,
                rank=rank,
                world_size=world_size,
                timeout=torch.distributed.default_pg_timeout
            )
            
            self.is_distributed = True
            
            if self.rank == 0:
                logger.info(f"Distributed training initialized: {world_size} processes, backend={self.backend}")
                
        except Exception as e:
            logger.error(f"Failed to initialize distributed training: {e}")
            raise
    
    def cleanup_distributed(self):
        """Cleanup distributed training."""
        if self.is_distributed:
            dist.destroy_process_group()
            self.is_distributed = False
            logger.info("Distributed training cleanup completed")
    
    def wrap_model(self, model: torch.nn.Module) -> torch.nn.Module:
        """Wrap model for distributed training."""
        if not self.is_distributed:
            return model
        
        # Move model to GPU
        if torch.cuda.is_available():
            model = model.to(self.local_rank)
        
        # Wrap with DDP
        model = DDP(
            model,
            device_ids=[self.local_rank] if torch.cuda.is_available() else None,
            output_device=self.local_rank if torch.cuda.is_available() else None,
            find_unused_parameters=True  # Useful for complex models
        )
        
        return model
    
    def create_distributed_sampler(self, dataset, shuffle: bool = True):
        """Create distributed sampler for dataset."""
        if not self.is_distributed:
            return None
        
        return DistributedSampler(
            dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=shuffle
        )
    
    def all_reduce_metrics(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """All-reduce metrics across all processes."""
        if not self.is_distributed:
            return metrics
        
        reduced_metrics = {}
        
        for key, value in metrics.items():
            tensor = torch.tensor(value, dtype=torch.float32)
            if torch.cuda.is_available():
                tensor = tensor.to(self.local_rank)
            
            # All-reduce sum and average
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            reduced_metrics[key] = (tensor / self.world_size).item()
        
        return reduced_metrics
    
    def barrier(self):
        """Synchronization barrier across all processes."""
        if self.is_distributed:
            dist.barrier()
    
    def is_main_process(self) -> bool:
        """Check if this is the main process (rank 0)."""
        return self.rank == 0 or not self.is_distributed


class AutoScalingTrainer:
    """Auto-scaling trainer that adapts to available resources."""
    
    def __init__(
        self,
        base_trainer_class,
        min_workers: int = 1,
        max_workers: int = 8,
        scaling_strategy: str = "gpu_count"
    ):
        self.base_trainer_class = base_trainer_class
        self.min_workers = min_workers
        self.max_workers = max_workers
        self.scaling_strategy = scaling_strategy
        
        self.optimal_workers = self._determine_optimal_workers()
        self.distributed_manager = DistributedTrainingManager()
        
    def _determine_optimal_workers(self) -> int:
        """Determine optimal number of workers based on resources."""
        if self.scaling_strategy == "gpu_count":
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                return min(max(gpu_count, self.min_workers), self.max_workers)
        
        elif self.scaling_strategy == "cpu_count":
            import psutil
            cpu_count = psutil.cpu_count(logical=False)  # Physical cores
            return min(max(cpu_count // 2, self.min_workers), self.max_workers)
        
        elif self.scaling_strategy == "memory_based":
            import psutil
            memory_gb = psutil.virtual_memory().total / (1024**3)
            # Estimate 4GB per worker
            memory_workers = max(1, int(memory_gb // 4))
            return min(max(memory_workers, self.min_workers), self.max_workers)
        
        return self.min_workers
    
    def launch_distributed_training(
        self,
        train_fn: Callable,
        *args,
        **kwargs
    ):
        """Launch distributed training using multiprocessing."""
        world_size = self.optimal_workers
        
        if world_size == 1:
            # Single process training
            logger.info("Running single-process training")
            return train_fn(0, world_size, 0, *args, **kwargs)
        
        # Multi-process training
        logger.info(f"Launching distributed training with {world_size} processes")
        
        try:
            mp.spawn(
                self._train_worker,
                args=(world_size, train_fn, args, kwargs),
                nprocs=world_size,
                join=True
            )
        except Exception as e:
            logger.error(f"Distributed training failed: {e}")
            raise
    
    def _train_worker(
        self,
        rank: int,
        world_size: int,
        train_fn: Callable,
        args: tuple,
        kwargs: dict
    ):
        """Worker function for distributed training."""
        local_rank = rank  # Assuming single-node training
        
        # Setup distributed environment
        self.distributed_manager.setup_distributed(rank, world_size, local_rank)
        
        try:
            # Run training function
            result = train_fn(rank, world_size, local_rank, *args, **kwargs)
            
            # Synchronize before cleanup
            self.distributed_manager.barrier()
            
            return result
            
        finally:
            self.distributed_manager.cleanup_distributed()


class ElasticTraining:
    """Elastic training that can adapt to changing resource availability."""
    
    def __init__(
        self,
        trainer,
        checkpoint_interval: int = 10,
        health_check_interval: int = 60
    ):
        self.trainer = trainer
        self.checkpoint_interval = checkpoint_interval
        self.health_check_interval = health_check_interval
        
        self.current_workers = 1
        self.last_health_check = time.time()
        self.training_active = False
        
    def monitor_resources(self) -> Dict[str, Any]:
        """Monitor available resources."""
        resources = {
            "timestamp": time.time(),
            "gpu_available": torch.cuda.is_available(),
            "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "memory_available_gb": 0,
            "cpu_usage": 0
        }
        
        try:
            import psutil
            memory = psutil.virtual_memory()
            resources["memory_available_gb"] = memory.available / (1024**3)
            resources["cpu_usage"] = psutil.cpu_percent(interval=1)
        except ImportError:
            pass
        
        return resources
    
    def should_scale_up(self, resources: Dict[str, Any]) -> bool:
        """Determine if we should scale up training."""
        # Scale up if more GPUs become available
        if resources["gpu_count"] > self.current_workers:
            return True
        
        # Scale up if memory usage is low and CPU usage is low
        if (resources["memory_available_gb"] > 10 and 
            resources["cpu_usage"] < 50):
            return True
        
        return False
    
    def should_scale_down(self, resources: Dict[str, Any]) -> bool:
        """Determine if we should scale down training."""
        # Scale down if memory is getting low
        if resources["memory_available_gb"] < 2:
            return True
        
        # Scale down if fewer GPUs are available
        if resources["gpu_count"] < self.current_workers:
            return True
        
        return False
    
    @contextmanager
    def elastic_training_session(self):
        """Context manager for elastic training."""
        self.training_active = True
        
        try:
            yield self
        finally:
            self.training_active = False
    
    def adaptive_checkpoint(self, epoch: int, force: bool = False):
        """Adaptive checkpointing based on resource changes."""
        current_time = time.time()
        
        # Regular checkpointing
        if epoch % self.checkpoint_interval == 0 or force:
            checkpoint_path = f"checkpoint_elastic_{epoch}_{int(current_time)}.pt"
            self.trainer.save_checkpoint(checkpoint_path)
            logger.info(f"Elastic checkpoint saved: {checkpoint_path}")
        
        # Health check and potential scaling
        if current_time - self.last_health_check > self.health_check_interval:
            resources = self.monitor_resources()
            
            if self.should_scale_up(resources):
                logger.info("Resource scaling up detected - saving checkpoint for restart")
                checkpoint_path = f"checkpoint_scale_up_{int(current_time)}.pt"
                self.trainer.save_checkpoint(checkpoint_path)
                
            elif self.should_scale_down(resources):
                logger.warning("Resource scaling down detected - saving checkpoint")
                checkpoint_path = f"checkpoint_scale_down_{int(current_time)}.pt"
                self.trainer.save_checkpoint(checkpoint_path)
            
            self.last_health_check = current_time


class HierarchicalTraining:
    """Hierarchical training for multi-scale PNO models."""
    
    def __init__(
        self,
        model,
        scales: List[int] = [64, 128, 256],
        scale_weights: List[float] = [0.5, 0.3, 0.2]
    ):
        self.model = model
        self.scales = scales
        self.scale_weights = scale_weights
        
        if len(scales) != len(scale_weights):
            raise ValueError("scales and scale_weights must have same length")
    
    def multi_scale_forward(self, x: torch.Tensor) -> Dict[int, torch.Tensor]:
        """Forward pass at multiple scales."""
        results = {}
        
        original_size = x.shape[-2:]  # Height, width
        
        for scale in self.scales:
            # Resize input to current scale
            if scale != original_size[0]:  # Assuming square inputs
                scale_ratio = scale / original_size[0]
                scaled_x = torch.nn.functional.interpolate(
                    x, 
                    scale_factor=scale_ratio, 
                    mode='bilinear', 
                    align_corners=False
                )
            else:
                scaled_x = x
            
            # Forward pass at this scale
            output = self.model(scaled_x)
            
            # Resize output back to original size if needed
            if output.shape[-2:] != original_size:
                output = torch.nn.functional.interpolate(
                    output,
                    size=original_size,
                    mode='bilinear',
                    align_corners=False
                )
            
            results[scale] = output
        
        return results
    
    def compute_multi_scale_loss(
        self,
        predictions: Dict[int, torch.Tensor],
        targets: torch.Tensor,
        loss_fn: Callable
    ) -> torch.Tensor:
        """Compute weighted multi-scale loss."""
        total_loss = 0.0
        
        for scale, weight in zip(self.scales, self.scale_weights):
            if scale in predictions:
                scale_loss = loss_fn(predictions[scale], targets)
                total_loss += weight * scale_loss
        
        return total_loss


def find_free_port() -> int:
    """Find a free port for distributed training."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port


def setup_distributed_environment(
    rank: int = 0,
    world_size: int = 1,
    master_addr: str = "localhost",
    master_port: Optional[int] = None
):
    """Setup environment variables for distributed training."""
    if master_port is None:
        master_port = find_free_port()
    
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    
    if torch.cuda.is_available():
        os.environ["LOCAL_RANK"] = str(rank % torch.cuda.device_count())


class PerformanceProfiler:
    """Performance profiler for distributed training."""
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.metrics = {
            "forward_time": [],
            "backward_time": [],
            "communication_time": [],
            "memory_usage": []
        }
        
    @contextmanager
    def profile_forward(self):
        """Profile forward pass."""
        if not self.enabled:
            yield
            return
        
        start_time = time.time()
        yield
        forward_time = time.time() - start_time
        self.metrics["forward_time"].append(forward_time)
    
    @contextmanager
    def profile_backward(self):
        """Profile backward pass."""
        if not self.enabled:
            yield
            return
        
        start_time = time.time()
        yield
        backward_time = time.time() - start_time
        self.metrics["backward_time"].append(backward_time)
    
    def get_summary(self) -> Dict[str, float]:
        """Get performance summary."""
        summary = {}
        
        for metric_name, values in self.metrics.items():
            if values:
                import numpy as np
                summary[f"{metric_name}_mean"] = float(np.mean(values))
                summary[f"{metric_name}_std"] = float(np.std(values))
                summary[f"{metric_name}_min"] = float(np.min(values))
                summary[f"{metric_name}_max"] = float(np.max(values))
        
        return summary


# Export main classes
__all__ = [
    "DistributedTrainingManager",
    "AutoScalingTrainer",
    "ElasticTraining",
    "HierarchicalTraining",
    "PerformanceProfiler",
    "setup_distributed_environment",
    "find_free_port"
]