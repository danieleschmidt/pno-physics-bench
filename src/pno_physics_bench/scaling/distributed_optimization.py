"""
Distributed Optimization Framework for Large-Scale PNO Training.

This module implements advanced distributed training strategies including
gradient compression, adaptive learning rates, dynamic load balancing,
and fault-tolerant training for massive-scale PNO deployments.
"""

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import numpy as np
import time
import logging
import json
import os
import pickle
import gzip
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
import threading
import queue
import socket
import psutil

from ..models import ProbabilisticNeuralOperator


@dataclass
class DistributedConfig:
    """Configuration for distributed training."""
    world_size: int
    rank: int
    local_rank: int
    backend: str = "nccl"
    init_method: str = "env://"
    gradient_compression: str = "none"  # "none", "quantization", "sparsification"
    compression_ratio: float = 0.1
    async_gradient_reduction: bool = True
    fault_tolerance: bool = True
    checkpoint_frequency: int = 100
    load_balancing: bool = True
    adaptive_batch_size: bool = True


@dataclass
class TrainingMetrics:
    """Training metrics for monitoring."""
    epoch: int
    step: int
    loss: float
    learning_rate: float
    gradient_norm: float
    compute_time: float
    communication_time: float
    memory_usage: float
    throughput: float
    timestamp: float


class GradientCompressor(ABC):
    """Abstract base class for gradient compression."""
    
    @abstractmethod
    def compress(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, Any]:
        """Compress gradient tensor."""
        pass
    
    @abstractmethod
    def decompress(self, compressed_tensor: torch.Tensor, metadata: Any) -> torch.Tensor:
        """Decompress gradient tensor."""
        pass


class QuantizationCompressor(GradientCompressor):
    """Gradient quantization compressor."""
    
    def __init__(self, num_bits: int = 8):
        self.num_bits = num_bits
        self.levels = 2 ** num_bits - 1
    
    def compress(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Quantize gradient to specified bit precision."""
        original_shape = tensor.shape
        flat_tensor = tensor.flatten()
        
        # Compute quantization parameters
        min_val = flat_tensor.min()
        max_val = flat_tensor.max()
        scale = (max_val - min_val) / self.levels
        
        # Quantize
        quantized = torch.round((flat_tensor - min_val) / scale).clamp(0, self.levels)
        quantized = quantized.to(torch.uint8)
        
        metadata = {
            'shape': original_shape,
            'min_val': min_val.item(),
            'max_val': max_val.item(),
            'scale': scale.item()
        }
        
        return quantized, metadata
    
    def decompress(self, compressed_tensor: torch.Tensor, metadata: Dict[str, Any]) -> torch.Tensor:
        """Dequantize gradient."""
        # Dequantize
        dequantized = compressed_tensor.float() * metadata['scale'] + metadata['min_val']
        
        # Reshape
        return dequantized.view(metadata['shape'])


class SparsificationCompressor(GradientCompressor):
    """Top-k sparsification compressor."""
    
    def __init__(self, compression_ratio: float = 0.1):
        self.compression_ratio = compression_ratio
    
    def compress(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Keep only top-k gradients by magnitude."""
        original_shape = tensor.shape
        flat_tensor = tensor.flatten()
        
        # Find top-k elements
        k = max(1, int(len(flat_tensor) * self.compression_ratio))
        _, top_k_indices = torch.topk(flat_tensor.abs(), k)
        
        # Create sparse representation
        top_k_values = flat_tensor[top_k_indices]
        
        metadata = {
            'shape': original_shape,
            'indices': top_k_indices,
            'size': len(flat_tensor)
        }
        
        return top_k_values, metadata
    
    def decompress(self, compressed_tensor: torch.Tensor, metadata: Dict[str, Any]) -> torch.Tensor:
        """Reconstruct sparse gradient."""
        # Create full tensor
        full_tensor = torch.zeros(metadata['size'], device=compressed_tensor.device)
        full_tensor[metadata['indices']] = compressed_tensor
        
        # Reshape
        return full_tensor.view(metadata['shape'])


class AdaptiveBatchSizeController:
    """
    Adaptive batch size controller for optimal GPU utilization.
    
    Dynamically adjusts batch size based on memory usage and throughput.
    """
    
    def __init__(
        self,
        initial_batch_size: int = 32,
        min_batch_size: int = 8,
        max_batch_size: int = 256,
        memory_threshold: float = 0.85,
        throughput_window: int = 10
    ):
        self.current_batch_size = initial_batch_size
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.memory_threshold = memory_threshold
        self.throughput_window = throughput_window
        
        # Tracking
        self.throughput_history = []
        self.memory_history = []
        self.adjustment_history = []
    
    def get_memory_usage(self) -> float:
        """Get current GPU memory usage ratio."""
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated()
            memory_reserved = torch.cuda.memory_reserved()
            return memory_allocated / memory_reserved if memory_reserved > 0 else 0.0
        return 0.0
    
    def should_adjust_batch_size(self, throughput: float) -> Tuple[bool, int]:
        """Determine if batch size should be adjusted."""
        self.throughput_history.append(throughput)
        self.memory_history.append(self.get_memory_usage())
        
        # Keep only recent history
        if len(self.throughput_history) > self.throughput_window:
            self.throughput_history = self.throughput_history[-self.throughput_window:]
            self.memory_history = self.memory_history[-self.throughput_window:]
        
        # Need sufficient history
        if len(self.throughput_history) < self.throughput_window:
            return False, self.current_batch_size
        
        current_memory = self.memory_history[-1]
        avg_throughput = np.mean(self.throughput_history)
        throughput_trend = np.polyfit(range(len(self.throughput_history)), self.throughput_history, 1)[0]
        
        # Decision logic
        new_batch_size = self.current_batch_size
        
        # Increase batch size if memory allows and throughput is good
        if (current_memory < self.memory_threshold * 0.8 and 
            throughput_trend >= 0 and 
            self.current_batch_size < self.max_batch_size):
            new_batch_size = min(self.max_batch_size, int(self.current_batch_size * 1.2))
        
        # Decrease batch size if memory is too high or throughput is declining
        elif (current_memory > self.memory_threshold or 
              throughput_trend < -0.1 * avg_throughput and 
              self.current_batch_size > self.min_batch_size):
            new_batch_size = max(self.min_batch_size, int(self.current_batch_size * 0.8))
        
        should_adjust = new_batch_size != self.current_batch_size
        
        if should_adjust:
            self.adjustment_history.append({
                'old_batch_size': self.current_batch_size,
                'new_batch_size': new_batch_size,
                'memory_usage': current_memory,
                'throughput': throughput,
                'timestamp': time.time()
            })
            self.current_batch_size = new_batch_size
        
        return should_adjust, new_batch_size


class LoadBalancer:
    """
    Dynamic load balancer for distributed training.
    
    Monitors worker performance and redistributes work accordingly.
    """
    
    def __init__(self, world_size: int, rebalance_frequency: int = 50):
        self.world_size = world_size
        self.rebalance_frequency = rebalance_frequency
        
        # Performance tracking
        self.worker_metrics = {i: [] for i in range(world_size)}
        self.step_count = 0
        self.load_distribution = [1.0] * world_size  # Relative load factors
    
    def update_worker_metrics(self, rank: int, metrics: TrainingMetrics):
        """Update performance metrics for a worker."""
        self.worker_metrics[rank].append({
            'throughput': metrics.throughput,
            'compute_time': metrics.compute_time,
            'memory_usage': metrics.memory_usage,
            'timestamp': metrics.timestamp
        })
        
        # Keep only recent metrics
        if len(self.worker_metrics[rank]) > 100:
            self.worker_metrics[rank] = self.worker_metrics[rank][-50:]
    
    def should_rebalance(self) -> bool:
        """Check if load rebalancing is needed."""
        self.step_count += 1
        return self.step_count % self.rebalance_frequency == 0
    
    def compute_load_distribution(self) -> List[float]:
        """Compute optimal load distribution based on worker performance."""
        if not any(self.worker_metrics.values()):
            return self.load_distribution
        
        # Compute average performance for each worker
        worker_performance = []
        for rank in range(self.world_size):
            if self.worker_metrics[rank]:
                recent_metrics = self.worker_metrics[rank][-10:]  # Last 10 measurements
                avg_throughput = np.mean([m['throughput'] for m in recent_metrics])
                avg_compute_time = np.mean([m['compute_time'] for m in recent_metrics])
                
                # Performance score (higher is better)
                score = avg_throughput / (avg_compute_time + 1e-6)
                worker_performance.append(score)
            else:
                worker_performance.append(1.0)  # Default performance
        
        # Normalize to create load distribution
        total_performance = sum(worker_performance)
        if total_performance > 0:
            self.load_distribution = [p / total_performance * self.world_size 
                                    for p in worker_performance]
        
        return self.load_distribution
    
    def get_worker_batch_size(self, rank: int, base_batch_size: int) -> int:
        """Get adjusted batch size for specific worker."""
        load_factor = self.load_distribution[rank]
        adjusted_batch_size = max(1, int(base_batch_size * load_factor))
        return adjusted_batch_size


class FaultTolerantTrainer:
    """
    Fault-tolerant training coordinator.
    
    Handles worker failures, checkpoint recovery, and elastic scaling.
    """
    
    def __init__(
        self,
        config: DistributedConfig,
        model: ProbabilisticNeuralOperator,
        checkpoint_dir: str = "./checkpoints"
    ):
        self.config = config
        self.model = model
        self.checkpoint_dir = checkpoint_dir
        
        # State tracking
        self.epoch = 0
        self.step = 0
        self.active_workers = set(range(config.world_size))
        self.failed_workers = set()
        
        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Heartbeat monitoring
        self.last_heartbeat = {i: time.time() for i in range(config.world_size)}
        self.heartbeat_timeout = 60.0  # seconds
        
        # Gradient compression
        self.gradient_compressor = self._create_gradient_compressor()
        
        # Load balancer
        self.load_balancer = LoadBalancer(config.world_size)
        
        # Batch size controller
        self.batch_controller = AdaptiveBatchSizeController()
        
        # Metrics tracking
        self.training_metrics = []
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
    
    def _create_gradient_compressor(self) -> Optional[GradientCompressor]:
        """Create gradient compressor based on config."""
        if self.config.gradient_compression == "quantization":
            return QuantizationCompressor(num_bits=8)
        elif self.config.gradient_compression == "sparsification":
            return SparsificationCompressor(self.config.compression_ratio)
        return None
    
    def save_checkpoint(self, optimizer: torch.optim.Optimizer, additional_state: Dict = None):
        """Save training checkpoint."""
        checkpoint = {
            'epoch': self.epoch,
            'step': self.step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': asdict(self.config),
            'active_workers': list(self.active_workers),
            'failed_workers': list(self.failed_workers),
            'training_metrics': self.training_metrics[-100:],  # Last 100 metrics
            'timestamp': time.time()
        }
        
        if additional_state:
            checkpoint.update(additional_state)
        
        checkpoint_path = os.path.join(
            self.checkpoint_dir, 
            f"checkpoint_epoch_{self.epoch}_step_{self.step}.pt"
        )
        
        torch.save(checkpoint, checkpoint_path)
        
        # Create symlink to latest checkpoint
        latest_path = os.path.join(self.checkpoint_dir, "latest_checkpoint.pt")
        if os.path.exists(latest_path):
            os.unlink(latest_path)
        os.symlink(os.path.basename(checkpoint_path), latest_path)
        
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, optimizer: torch.optim.Optimizer, checkpoint_path: str = None) -> Dict:
        """Load training checkpoint."""
        if checkpoint_path is None:
            checkpoint_path = os.path.join(self.checkpoint_dir, "latest_checkpoint.pt")
        
        if not os.path.exists(checkpoint_path):
            self.logger.warning(f"Checkpoint not found: {checkpoint_path}")
            return {}
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Restore state
        self.epoch = checkpoint.get('epoch', 0)
        self.step = checkpoint.get('step', 0)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        self.active_workers = set(checkpoint.get('active_workers', range(self.config.world_size)))
        self.failed_workers = set(checkpoint.get('failed_workers', []))
        
        if 'training_metrics' in checkpoint:
            self.training_metrics = checkpoint['training_metrics']
        
        self.logger.info(f"Checkpoint loaded: {checkpoint_path}")
        return checkpoint
    
    def update_heartbeat(self, rank: int):
        """Update heartbeat for worker."""
        self.last_heartbeat[rank] = time.time()
    
    def check_worker_health(self) -> List[int]:
        """Check for failed workers based on heartbeat timeout."""
        current_time = time.time()
        newly_failed = []
        
        for rank in list(self.active_workers):
            if current_time - self.last_heartbeat[rank] > self.heartbeat_timeout:
                self.logger.warning(f"Worker {rank} appears to have failed (heartbeat timeout)")
                self.active_workers.remove(rank)
                self.failed_workers.add(rank)
                newly_failed.append(rank)
        
        return newly_failed
    
    def redistribute_work(self, failed_workers: List[int]) -> Dict[int, Dict]:
        """Redistribute work from failed workers."""
        if not failed_workers:
            return {}
        
        # Simple redistribution: divide failed workers' load among active workers
        redistribution_plan = {}
        
        for failed_rank in failed_workers:
            # Distribute this worker's load
            load_per_worker = 1.0 / len(self.active_workers)
            
            for active_rank in self.active_workers:
                if active_rank not in redistribution_plan:
                    redistribution_plan[active_rank] = {'additional_load': 0.0}
                
                redistribution_plan[active_rank]['additional_load'] += load_per_worker
        
        self.logger.info(f"Redistributed work from failed workers {failed_workers}")
        return redistribution_plan
    
    def compress_gradients(self, gradients: Dict[str, torch.Tensor]) -> Dict[str, Tuple[torch.Tensor, Any]]:
        """Compress gradients for communication."""
        if self.gradient_compressor is None:
            return {name: (grad, None) for name, grad in gradients.items()}
        
        compressed = {}
        for name, grad in gradients.items():
            comp_grad, metadata = self.gradient_compressor.compress(grad)
            compressed[name] = (comp_grad, metadata)
        
        return compressed
    
    def decompress_gradients(self, compressed_gradients: Dict[str, Tuple[torch.Tensor, Any]]) -> Dict[str, torch.Tensor]:
        """Decompress gradients after communication."""
        if self.gradient_compressor is None:
            return {name: grad for name, (grad, _) in compressed_gradients.items()}
        
        decompressed = {}
        for name, (comp_grad, metadata) in compressed_gradients.items():
            grad = self.gradient_compressor.decompress(comp_grad, metadata)
            decompressed[name] = grad
        
        return decompressed
    
    def all_reduce_gradients(self, model: nn.Module) -> float:
        """Perform gradient all-reduce with fault tolerance."""
        start_time = time.time()
        
        # Check for failed workers
        failed_workers = self.check_worker_health()
        if failed_workers:
            self.redistribute_work(failed_workers)
        
        # Get gradients
        gradients = {name: param.grad for name, param in model.named_parameters() 
                    if param.grad is not None}
        
        # Compress gradients
        if self.gradient_compressor:
            compressed_gradients = self.compress_gradients(gradients)
        else:
            compressed_gradients = {name: (grad, None) for name, grad in gradients.items()}
        
        # All-reduce (simplified - in practice would use proper distributed operations)
        # This is a placeholder for actual distributed gradient reduction
        try:
            for name, (grad, metadata) in compressed_gradients.items():
                if dist.is_initialized():
                    dist.all_reduce(grad)
                    grad /= len(self.active_workers)
        except Exception as e:
            self.logger.error(f"Gradient reduction failed: {e}")
            # Fallback: continue with local gradients
        
        # Decompress gradients
        if self.gradient_compressor:
            final_gradients = self.decompress_gradients(compressed_gradients)
            
            # Update model gradients
            for name, param in model.named_parameters():
                if name in final_gradients:
                    param.grad = final_gradients[name]
        
        communication_time = time.time() - start_time
        return communication_time
    
    def train_step(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        data_batch: torch.Tensor,
        target_batch: torch.Tensor,
        criterion: nn.Module
    ) -> TrainingMetrics:
        """Execute one training step with fault tolerance."""
        start_time = time.time()
        
        # Update heartbeat
        self.update_heartbeat(self.config.rank)
        
        # Adaptive batch size
        if self.config.adaptive_batch_size:
            current_throughput = len(data_batch) / (time.time() - start_time + 1e-6)
            adjusted, new_batch_size = self.batch_controller.should_adjust_batch_size(current_throughput)
            
            if adjusted:
                self.logger.info(f"Adjusted batch size: {self.batch_controller.current_batch_size}")
        
        # Forward pass
        compute_start = time.time()
        prediction = model(data_batch)
        loss = criterion(prediction, target_batch)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Compute gradient norm
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        compute_time = time.time() - compute_start
        
        # Gradient synchronization
        communication_time = self.all_reduce_gradients(model)
        
        # Optimizer step
        optimizer.step()
        
        # Calculate metrics
        total_time = time.time() - start_time
        throughput = len(data_batch) / total_time
        memory_usage = torch.cuda.memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0
        
        metrics = TrainingMetrics(
            epoch=self.epoch,
            step=self.step,
            loss=loss.item(),
            learning_rate=optimizer.param_groups[0]['lr'],
            gradient_norm=grad_norm.item(),
            compute_time=compute_time,
            communication_time=communication_time,
            memory_usage=memory_usage,
            throughput=throughput,
            timestamp=time.time()
        )
        
        # Update load balancer
        self.load_balancer.update_worker_metrics(self.config.rank, metrics)
        
        # Store metrics
        self.training_metrics.append(asdict(metrics))
        
        # Checkpoint periodically
        if self.step % self.config.checkpoint_frequency == 0:
            self.save_checkpoint(optimizer)
        
        self.step += 1
        
        return metrics


class DistributedPNOTrainer:
    """
    High-level distributed PNO trainer with advanced optimization.
    
    Orchestrates fault-tolerant distributed training with automatic
    scaling and optimization.
    """
    
    def __init__(
        self,
        model: ProbabilisticNeuralOperator,
        config: DistributedConfig,
        checkpoint_dir: str = "./checkpoints"
    ):
        self.model = model
        self.config = config
        
        # Initialize distributed training
        self._setup_distributed()
        
        # Wrap model with DDP
        self.ddp_model = DDP(model, device_ids=[config.local_rank])
        
        # Create fault-tolerant trainer
        self.fault_tolerant_trainer = FaultTolerantTrainer(
            config=config,
            model=model,
            checkpoint_dir=checkpoint_dir
        )
        
        # Performance monitoring
        self.performance_monitor = PerformanceMonitor()
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
    
    def _setup_distributed(self):
        """Setup distributed training environment."""
        os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', 'localhost')
        os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '12355')
        
        dist.init_process_group(
            backend=self.config.backend,
            init_method=self.config.init_method,
            world_size=self.config.world_size,
            rank=self.config.rank
        )
        
        # Set device
        if torch.cuda.is_available():
            torch.cuda.set_device(self.config.local_rank)
            self.device = torch.device(f'cuda:{self.config.local_rank}')
        else:
            self.device = torch.device('cpu')
        
        self.model.to(self.device)
    
    def train(
        self,
        train_loader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        num_epochs: int,
        val_loader: Optional[torch.utils.data.DataLoader] = None
    ) -> Dict[str, List]:
        """
        Execute distributed training.
        
        Returns:
            Training history dictionary
        """
        history = {
            'train_loss': [],
            'train_metrics': [],
            'val_loss': [],
            'performance_metrics': []
        }
        
        # Load checkpoint if exists
        checkpoint_state = self.fault_tolerant_trainer.load_checkpoint(optimizer)
        start_epoch = checkpoint_state.get('epoch', 0)
        
        self.logger.info(f"Starting distributed training from epoch {start_epoch}")
        
        for epoch in range(start_epoch, num_epochs):
            self.fault_tolerant_trainer.epoch = epoch
            
            # Training
            self.ddp_model.train()
            epoch_metrics = []
            
            # Create distributed sampler
            if hasattr(train_loader.sampler, 'set_epoch'):
                train_loader.sampler.set_epoch(epoch)
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                # Execute training step with fault tolerance
                step_metrics = self.fault_tolerant_trainer.train_step(
                    model=self.ddp_model,
                    optimizer=optimizer,
                    data_batch=data,
                    target_batch=target,
                    criterion=criterion
                )
                
                epoch_metrics.append(step_metrics)
                
                # Performance monitoring
                perf_metrics = self.performance_monitor.collect_metrics()
                history['performance_metrics'].append(perf_metrics)
                
                # Logging
                if batch_idx % 100 == 0 and self.config.rank == 0:
                    self.logger.info(
                        f"Epoch {epoch}, Batch {batch_idx}: "
                        f"Loss={step_metrics.loss:.4f}, "
                        f"Throughput={step_metrics.throughput:.2f} samples/s, "
                        f"Memory={step_metrics.memory_usage:.2f}GB"
                    )
            
            # Aggregate epoch metrics
            if epoch_metrics:
                avg_loss = np.mean([m.loss for m in epoch_metrics])
                history['train_loss'].append(avg_loss)
                history['train_metrics'].extend([asdict(m) for m in epoch_metrics])
            
            # Validation
            if val_loader and self.config.rank == 0:
                val_loss = self._validate(val_loader, criterion)
                history['val_loss'].append(val_loss)
                self.logger.info(f"Epoch {epoch}: Train Loss={avg_loss:.4f}, Val Loss={val_loss:.4f}")
            
            # Save checkpoint
            if epoch % 5 == 0:  # Every 5 epochs
                self.fault_tolerant_trainer.save_checkpoint(
                    optimizer, 
                    additional_state={'history': history}
                )
        
        self.logger.info("Distributed training completed")
        return history
    
    def _validate(self, val_loader: torch.utils.data.DataLoader, criterion: nn.Module) -> float:
        """Run validation."""
        self.ddp_model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                prediction = self.ddp_model(data)
                loss = criterion(prediction, target)
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def cleanup(self):
        """Cleanup distributed training."""
        if dist.is_initialized():
            dist.destroy_process_group()


class PerformanceMonitor:
    """Monitor system and training performance metrics."""
    
    def __init__(self):
        self.start_time = time.time()
    
    def collect_metrics(self) -> Dict[str, float]:
        """Collect comprehensive performance metrics."""
        metrics = {
            'timestamp': time.time(),
            'uptime': time.time() - self.start_time
        }
        
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=None)
        cpu_count = psutil.cpu_count()
        load_avg = os.getloadavg() if hasattr(os, 'getloadavg') else (0, 0, 0)
        
        metrics.update({
            'cpu_percent': cpu_percent,
            'cpu_count': cpu_count,
            'load_avg_1m': load_avg[0],
            'load_avg_5m': load_avg[1],
            'load_avg_15m': load_avg[2]
        })
        
        # Memory metrics
        memory = psutil.virtual_memory()
        metrics.update({
            'memory_total': memory.total / (1024**3),  # GB
            'memory_available': memory.available / (1024**3),
            'memory_percent': memory.percent,
            'memory_used': memory.used / (1024**3)
        })
        
        # GPU metrics (if available)
        if torch.cuda.is_available():
            gpu_memory_allocated = torch.cuda.memory_allocated() / (1024**3)
            gpu_memory_reserved = torch.cuda.memory_reserved() / (1024**3)
            gpu_memory_cached = torch.cuda.memory_cached() / (1024**3)
            
            metrics.update({
                'gpu_memory_allocated': gpu_memory_allocated,
                'gpu_memory_reserved': gpu_memory_reserved,
                'gpu_memory_cached': gpu_memory_cached,
                'gpu_memory_percent': (gpu_memory_allocated / gpu_memory_reserved * 100) if gpu_memory_reserved > 0 else 0
            })
        
        # Disk I/O
        disk_io = psutil.disk_io_counters()
        if disk_io:
            metrics.update({
                'disk_read_bytes': disk_io.read_bytes,
                'disk_write_bytes': disk_io.write_bytes,
                'disk_read_count': disk_io.read_count,
                'disk_write_count': disk_io.write_count
            })
        
        # Network I/O
        network_io = psutil.net_io_counters()
        if network_io:
            metrics.update({
                'network_bytes_sent': network_io.bytes_sent,
                'network_bytes_recv': network_io.bytes_recv,
                'network_packets_sent': network_io.packets_sent,
                'network_packets_recv': network_io.packets_recv
            })
        
        return metrics


def launch_distributed_training(
    model_fn: Callable,
    config: DistributedConfig,
    train_data_fn: Callable,
    **training_kwargs
):
    """
    Launch distributed training across multiple processes.
    
    Args:
        model_fn: Function that returns model instance
        config: Distributed training configuration
        train_data_fn: Function that returns training data loader
        **training_kwargs: Additional training arguments
    """
    def train_worker(rank: int, world_size: int):
        """Worker function for distributed training."""
        # Update config for this worker
        worker_config = DistributedConfig(
            world_size=world_size,
            rank=rank,
            local_rank=rank % torch.cuda.device_count() if torch.cuda.is_available() else 0,
            backend=config.backend,
            init_method=config.init_method,
            gradient_compression=config.gradient_compression,
            compression_ratio=config.compression_ratio,
            async_gradient_reduction=config.async_gradient_reduction,
            fault_tolerance=config.fault_tolerance,
            checkpoint_frequency=config.checkpoint_frequency,
            load_balancing=config.load_balancing,
            adaptive_batch_size=config.adaptive_batch_size
        )
        
        # Create model and trainer
        model = model_fn()
        trainer = DistributedPNOTrainer(model, worker_config)
        
        # Create data loader
        train_loader = train_data_fn(rank, world_size)
        
        # Setup optimizer and criterion
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()
        
        try:
            # Run training
            history = trainer.train(
                train_loader=train_loader,
                optimizer=optimizer,
                criterion=criterion,
                **training_kwargs
            )
            
            # Save final results
            if rank == 0:
                results_path = f"distributed_training_results_rank_{rank}.json"
                with open(results_path, 'w') as f:
                    json.dump(history, f, indent=2, default=str)
                
                print(f"Training completed. Results saved to {results_path}")
        
        except Exception as e:
            print(f"Training failed on rank {rank}: {e}")
            raise
        
        finally:
            trainer.cleanup()
    
    # Launch training processes
    if config.world_size > 1:
        mp.spawn(
            train_worker,
            args=(config.world_size,),
            nprocs=config.world_size,
            join=True
        )
    else:
        train_worker(0, 1)


def create_distributed_training_example():
    """Create example distributed training setup."""
    
    def create_model():
        return ProbabilisticNeuralOperator(
            input_dim=3,
            hidden_dim=128,
            num_layers=4,
            modes=16
        )
    
    def create_data_loader(rank: int, world_size: int):
        # Create synthetic data
        dataset_size = 1000
        data = torch.randn(dataset_size, 3, 64, 64)
        targets = torch.randn(dataset_size, 1, 64, 64)
        
        dataset = torch.utils.data.TensorDataset(data, targets)
        
        # Create distributed sampler
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True
        )
        
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=32,
            sampler=sampler,
            num_workers=2,
            pin_memory=True
        )
    
    # Configuration
    config = DistributedConfig(
        world_size=torch.cuda.device_count() if torch.cuda.is_available() else 1,
        rank=0,
        local_rank=0,
        gradient_compression="quantization",
        compression_ratio=0.1,
        fault_tolerance=True,
        adaptive_batch_size=True
    )
    
    return create_model, create_data_loader, config


if __name__ == "__main__":
    # Create distributed training example
    model_fn, data_fn, config = create_distributed_training_example()
    
    print("🚀 Distributed Optimization Framework ready!")
    print(f"Configuration: {config}")
    
    # Optionally run training
    # launch_distributed_training(
    #     model_fn=model_fn,
    #     config=config,
    #     train_data_fn=data_fn,
    #     num_epochs=10
    # )