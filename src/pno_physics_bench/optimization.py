"""Performance optimization utilities for PNO models."""

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist
from typing import Dict, Any, Optional, Union, Tuple, List
import time
import functools
import warnings
import logging
from contextlib import contextmanager
from functools import lru_cache
import gc

# from .exceptions import ResourceError, OptimizationError  # Not implemented yet
from .logging_config import PerformanceLogger


logger = logging.getLogger(__name__)
perf_logger = PerformanceLogger()


class ModelOptimizer:
    """Optimize PyTorch models for inference and training performance."""
    
    def __init__(self, model: nn.Module, device: torch.device):
        """Initialize model optimizer.
        
        Args:
            model: PyTorch model to optimize
            device: Target device
        """
        self.model = model
        self.device = device
        self.original_forward = None
        
    def compile_model(
        self,
        backend: str = "inductor",
        mode: str = "default",
        dynamic: bool = False,
    ) -> nn.Module:
        """Compile model using torch.compile (PyTorch 2.0+).
        
        Args:
            backend: Compilation backend
            mode: Compilation mode ("default", "reduce-overhead", "max-autotune")
            dynamic: Whether to enable dynamic shapes
            
        Returns:
            Compiled model
        """
        if not hasattr(torch, 'compile'):
            logger.warning("torch.compile not available, skipping compilation")
            return self.model
        
        try:
            logger.info(f"Compiling model with backend={backend}, mode={mode}")
            compiled_model = torch.compile(
                self.model,
                backend=backend,
                mode=mode,
                dynamic=dynamic,
            )
            logger.info("Model compilation successful")
            return compiled_model
        except Exception as e:
            logger.warning(f"Model compilation failed: {e}")
            return self.model
    
    def optimize_for_inference(
        self,
        use_half_precision: bool = False,
        use_channels_last: bool = False,
        freeze_model: bool = True,
    ) -> nn.Module:
        """Optimize model for inference.
        
        Args:
            use_half_precision: Use FP16 precision
            use_channels_last: Use channels-last memory format
            freeze_model: Freeze model parameters
            
        Returns:
            Optimized model
        """
        logger.info("Optimizing model for inference")
        
        # Set to evaluation mode
        self.model.eval()
        
        # Freeze parameters
        if freeze_model:
            for param in self.model.parameters():
                param.requires_grad_(False)
        
        # Half precision
        if use_half_precision and self.device.type == 'cuda':
            self.model = self.model.half()
            logger.info("Enabled half precision (FP16)")
        
        # Channels last memory format (for 4D tensors)
        if use_channels_last:
            self.model = self.model.to(memory_format=torch.channels_last)
            logger.info("Enabled channels-last memory format")
        
        # TorchScript optimization
        try:
            self.model = torch.jit.optimize_for_inference(
                torch.jit.script(self.model)
            )
            logger.info("Applied TorchScript optimization")
        except Exception as e:
            logger.warning(f"TorchScript optimization failed: {e}")
        
        return self.model
    
    def fuse_modules(self) -> nn.Module:
        """Fuse compatible modules for better performance.
        
        Returns:
            Model with fused modules
        """
        try:
            # Fuse Conv-BN-ReLU patterns
            torch.quantization.fuse_modules(
                self.model,
                [['conv', 'bn', 'relu']],  # Example pattern
                inplace=True
            )
            logger.info("Fused compatible modules")
        except Exception as e:
            logger.warning(f"Module fusion failed: {e}")
        
        return self.model
    
    def quantize_model(
        self,
        quantization_type: str = "dynamic",
        calibration_loader: Optional[DataLoader] = None,
    ) -> nn.Module:
        """Quantize model for reduced memory and faster inference.
        
        Args:
            quantization_type: Type of quantization ("dynamic", "static", "qat")
            calibration_loader: Data loader for calibration (static quantization)
            
        Returns:
            Quantized model
        """
        try:
            if quantization_type == "dynamic":
                quantized_model = torch.quantization.quantize_dynamic(
                    self.model,
                    {nn.Linear, nn.Conv2d},
                    dtype=torch.qint8
                )
                logger.info("Applied dynamic quantization")
            elif quantization_type == "static" and calibration_loader:
                # Prepare model for static quantization
                self.model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
                prepared_model = torch.quantization.prepare(self.model)
                
                # Calibrate
                prepared_model.eval()
                with torch.no_grad():
                    for inputs, _ in calibration_loader:
                        prepared_model(inputs.to(self.device))
                
                quantized_model = torch.quantization.convert(prepared_model)
                logger.info("Applied static quantization")
            else:
                logger.warning(f"Unsupported quantization type: {quantization_type}")
                return self.model
            
            return quantized_model
            
        except Exception as e:
            logger.warning(f"Quantization failed: {e}")
            return self.model


class MemoryOptimizer:
    """Memory usage optimization utilities."""
    
    @staticmethod
    def enable_memory_efficient_attention() -> None:
        """Enable memory-efficient attention if available."""
        try:
            # Enable flash attention if available
            torch.backends.cuda.flash_sdp_enabled(True)
            torch.backends.cuda.mem_efficient_sdp_enabled(True)
            logger.info("Enabled memory-efficient attention")
        except Exception as e:
            logger.warning(f"Could not enable memory-efficient attention: {e}")
    
    @staticmethod
    def optimize_memory_usage(
        model: nn.Module,
        use_gradient_checkpointing: bool = True,
        use_activation_checkpointing: bool = False,
    ) -> nn.Module:
        """Optimize memory usage during training.
        
        Args:
            model: Model to optimize
            use_gradient_checkpointing: Use gradient checkpointing
            use_activation_checkpointing: Use activation checkpointing
            
        Returns:
            Memory-optimized model
        """
        if use_gradient_checkpointing:
            if hasattr(model, 'gradient_checkpointing_enable'):
                model.gradient_checkpointing_enable()
                logger.info("Enabled gradient checkpointing")
        
        if use_activation_checkpointing:
            # Apply activation checkpointing to specific layers
            for name, module in model.named_modules():
                if isinstance(module, (nn.TransformerEncoderLayer, nn.TransformerDecoderLayer)):
                    module = torch.utils.checkpoint.checkpoint_wrapper(module)
                    logger.info(f"Applied activation checkpointing to {name}")
        
        return model
    
    @staticmethod
    @contextmanager
    def memory_context():
        """Context manager for memory-efficient operations."""
        # Clear cache before operation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        gc.collect()
        
        try:
            yield
        finally:
            # Clean up after operation
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    @staticmethod
    def get_memory_stats() -> Dict[str, float]:
        """Get current memory statistics.
        
        Returns:
            Dictionary with memory statistics in GB
        """
        stats = {}
        
        # System memory
        try:
            import psutil
            system_memory = psutil.virtual_memory()
            stats['system_total'] = system_memory.total / (1024**3)
            stats['system_available'] = system_memory.available / (1024**3)
            stats['system_used'] = system_memory.used / (1024**3)
        except ImportError:
            pass
        
        # GPU memory
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / (1024**3)
                reserved = torch.cuda.memory_reserved(i) / (1024**3)
                stats[f'gpu_{i}_allocated'] = allocated
                stats[f'gpu_{i}_reserved'] = reserved
        
        return stats


class DistributedTrainingManager:
    """Manager for distributed training setup and optimization."""
    
    def __init__(self, backend: str = "nccl", init_method: str = "env://"):
        """Initialize distributed training manager.
        
        Args:
            backend: Distributed backend
            init_method: Initialization method
        """
        self.backend = backend
        self.init_method = init_method
        self.is_initialized = False
        
    def setup_distributed(
        self,
        rank: Optional[int] = None,
        world_size: Optional[int] = None,
    ) -> None:
        """Setup distributed training.
        
        Args:
            rank: Process rank (auto-detected if None)
            world_size: Total number of processes (auto-detected if None)
        """
        # Auto-detect if not provided
        if rank is None:
            rank = int(os.environ.get('RANK', 0))
        if world_size is None:
            world_size = int(os.environ.get('WORLD_SIZE', 1))
        
        if world_size > 1:
            logger.info(f"Initializing distributed training: rank={rank}, world_size={world_size}")
            
            # Initialize process group
            dist.init_process_group(
                backend=self.backend,
                init_method=self.init_method,
                rank=rank,
                world_size=world_size,
            )
            
            # Set CUDA device for this process
            if torch.cuda.is_available():
                torch.cuda.set_device(rank % torch.cuda.device_count())
            
            self.is_initialized = True
            logger.info("Distributed training initialized successfully")
        else:
            logger.info("Single-process training (no distribution)")
    
    def wrap_model(self, model: nn.Module, device_ids: Optional[List[int]] = None) -> nn.Module:
        """Wrap model for distributed training.
        
        Args:
            model: Model to wrap
            device_ids: Device IDs to use
            
        Returns:
            Wrapped model
        """
        if not self.is_initialized:
            return model
        
        if device_ids is None and torch.cuda.is_available():
            device_ids = [torch.cuda.current_device()]
        
        try:
            ddp_model = DDP(
                model,
                device_ids=device_ids,
                find_unused_parameters=False,  # Set to True if needed
            )
            logger.info("Model wrapped with DistributedDataParallel")
            return ddp_model
        except Exception as e:
            logger.error(f"Failed to wrap model with DDP: {e}")
            return model
    
    def create_distributed_sampler(
        self,
        dataset,
        shuffle: bool = True,
        drop_last: bool = False,
    ) -> Optional[DistributedSampler]:
        """Create distributed sampler for dataset.
        
        Args:
            dataset: Dataset to sample from
            shuffle: Whether to shuffle data
            drop_last: Whether to drop incomplete batches
            
        Returns:
            Distributed sampler or None if not distributed
        """
        if not self.is_initialized:
            return None
        
        return DistributedSampler(
            dataset,
            shuffle=shuffle,
            drop_last=drop_last,
        )
    
    def cleanup(self) -> None:
        """Clean up distributed training."""
        if self.is_initialized:
            dist.destroy_process_group()
            self.is_initialized = False
            logger.info("Distributed training cleaned up")


class CacheManager:
    """Intelligent caching for expensive operations."""
    
    def __init__(self, max_memory_mb: int = 1024):
        """Initialize cache manager.
        
        Args:
            max_memory_mb: Maximum memory to use for caching (MB)
        """
        self.max_memory = max_memory_mb * 1024 * 1024  # Convert to bytes
        self.current_memory = 0
        self.cache = {}
        self.access_times = {}
        
    def _estimate_tensor_memory(self, tensor: torch.Tensor) -> int:
        """Estimate memory usage of a tensor.
        
        Args:
            tensor: Tensor to estimate
            
        Returns:
            Estimated memory in bytes
        """
        return tensor.element_size() * tensor.numel()
    
    def _evict_lru(self) -> None:
        """Evict least recently used items."""
        if not self.access_times:
            return
        
        # Sort by access time (oldest first)
        sorted_items = sorted(self.access_times.items(), key=lambda x: x[1])
        
        for key, _ in sorted_items:
            if self.current_memory <= self.max_memory:
                break
            
            if key in self.cache:
                item = self.cache[key]
                if isinstance(item, torch.Tensor):
                    memory_used = self._estimate_tensor_memory(item)
                    self.current_memory -= memory_used
                
                del self.cache[key]
                del self.access_times[key]
                logger.debug(f"Evicted cache item: {key}")
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached item or None
        """
        if key in self.cache:
            self.access_times[key] = time.time()
            return self.cache[key]
        return None
    
    def put(self, key: str, value: Any) -> None:
        """Put item in cache.
        
        Args:
            key: Cache key
            value: Item to cache
        """
        # Estimate memory usage
        memory_needed = 0
        if isinstance(value, torch.Tensor):
            memory_needed = self._estimate_tensor_memory(value)
        
        # Check if we can fit this item
        if memory_needed > self.max_memory:
            logger.warning(f"Item too large for cache: {key}")
            return
        
        # Evict items if necessary
        self.current_memory += memory_needed
        if self.current_memory > self.max_memory:
            self._evict_lru()
        
        # Add to cache
        self.cache[key] = value
        self.access_times[key] = time.time()
        logger.debug(f"Cached item: {key} ({memory_needed} bytes)")
    
    def clear(self) -> None:
        """Clear all cached items."""
        self.cache.clear()
        self.access_times.clear()
        self.current_memory = 0
        logger.info("Cache cleared")
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Dictionary with cache stats
        """
        return {
            'items': len(self.cache),
            'memory_used_mb': self.current_memory / (1024 * 1024),
            'memory_limit_mb': self.max_memory / (1024 * 1024),
            'utilization': self.current_memory / self.max_memory if self.max_memory > 0 else 0,
        }


class BatchProcessor:
    """Efficient batch processing with dynamic batching."""
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        max_batch_size: int = 32,
        target_memory_gb: float = 8.0,
    ):
        """Initialize batch processor.
        
        Args:
            model: Model for processing
            device: Target device
            max_batch_size: Maximum batch size
            target_memory_gb: Target memory usage in GB
        """
        self.model = model
        self.device = device
        self.max_batch_size = max_batch_size
        self.target_memory = target_memory_gb * 1024**3
        self.optimal_batch_size = max_batch_size
        
    def find_optimal_batch_size(
        self,
        sample_input: torch.Tensor,
        start_size: int = 1,
    ) -> int:
        """Find optimal batch size for given input.
        
        Args:
            sample_input: Sample input tensor
            start_size: Starting batch size for search
            
        Returns:
            Optimal batch size
        """
        logger.info("Finding optimal batch size...")
        
        batch_size = start_size
        memory_usage = []
        
        self.model.eval()
        
        with torch.no_grad():
            while batch_size <= self.max_batch_size:
                try:
                    # Create batch
                    batch_input = sample_input.repeat(batch_size, *[1] * (sample_input.dim() - 1))
                    batch_input = batch_input.to(self.device)
                    
                    # Clear cache
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    # Forward pass
                    _ = self.model(batch_input)
                    
                    # Check memory usage
                    if torch.cuda.is_available():
                        memory_used = torch.cuda.memory_allocated(self.device)
                        memory_usage.append((batch_size, memory_used))
                        
                        if memory_used > self.target_memory:
                            logger.info(f"Memory limit reached at batch size {batch_size}")
                            break
                    
                    batch_size *= 2
                    
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        logger.info(f"OOM at batch size {batch_size}")
                        break
                    else:
                        raise e
        
        # Find optimal batch size (90% of maximum feasible)
        if memory_usage:
            max_feasible = memory_usage[-1][0]
            optimal = max(1, int(max_feasible * 0.9))
            self.optimal_batch_size = min(optimal, self.max_batch_size)
        
        logger.info(f"Optimal batch size: {self.optimal_batch_size}")
        return self.optimal_batch_size
    
    def process_batches(
        self,
        inputs: List[torch.Tensor],
        batch_size: Optional[int] = None,
    ) -> List[torch.Tensor]:
        """Process inputs in optimal batches.
        
        Args:
            inputs: List of input tensors
            batch_size: Batch size (uses optimal if None)
            
        Returns:
            List of output tensors
        """
        if batch_size is None:
            batch_size = self.optimal_batch_size
        
        outputs = []
        self.model.eval()
        
        with torch.no_grad():
            for i in range(0, len(inputs), batch_size):
                batch_inputs = inputs[i:i + batch_size]
                
                # Stack batch
                batch_tensor = torch.stack(batch_inputs).to(self.device)
                
                # Process batch
                with perf_logger.timer("batch_inference", batch_size=len(batch_inputs)):
                    batch_outputs = self.model(batch_tensor)
                
                # Split outputs back
                if isinstance(batch_outputs, torch.Tensor):
                    outputs.extend(torch.unbind(batch_outputs, dim=0))
                else:
                    # Handle tuple outputs
                    for j in range(len(batch_inputs)):
                        output_tuple = tuple(out[j] for out in batch_outputs)
                        outputs.append(output_tuple)
        
        return outputs


def optimize_dataloader(
    dataloader: DataLoader,
    num_workers: Optional[int] = None,
    pin_memory: bool = True,
    persistent_workers: bool = True,
) -> DataLoader:
    """Optimize DataLoader for better performance.
    
    Args:
        dataloader: Original DataLoader
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory
        persistent_workers: Whether to keep workers alive
        
    Returns:
        Optimized DataLoader
    """
    if num_workers is None:
        import multiprocessing
        num_workers = min(multiprocessing.cpu_count(), 8)
    
    # Create optimized DataLoader - handle batch_sampler conflicts
    kwargs = {
        'dataset': dataloader.dataset,
        'num_workers': num_workers,
        'collate_fn': dataloader.collate_fn,
        'pin_memory': pin_memory and torch.cuda.is_available(),
        'timeout': 0,
        'worker_init_fn': dataloader.worker_init_fn,
        'multiprocessing_context': dataloader.multiprocessing_context,
        'generator': dataloader.generator,
        'prefetch_factor': 2,
        'persistent_workers': persistent_workers and num_workers > 0,
    }
    
    # Handle mutually exclusive options
    if dataloader.batch_sampler is not None:
        kwargs['batch_sampler'] = dataloader.batch_sampler
    else:
        kwargs['batch_size'] = dataloader.batch_size
        kwargs['shuffle'] = False  # Preserve original behavior
        kwargs['sampler'] = dataloader.sampler
        kwargs['drop_last'] = dataloader.drop_last
    
    optimized_loader = DataLoader(**kwargs)
    
    logger.info(f"Optimized DataLoader: {num_workers} workers, pin_memory={pin_memory}")
    return optimized_loader