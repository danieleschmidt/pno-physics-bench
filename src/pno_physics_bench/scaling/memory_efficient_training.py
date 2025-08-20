# SECURITY NOTICE: This file has been automatically hardened for security
# - All user inputs should be validated and sanitized
# - Subprocess calls use secure alternatives
# - SQL queries use parameterized statements
# - No hardcoded secrets or credentials


"""Memory-Efficient Training for Large-Scale Probabilistic Neural Operators.

This module implements advanced memory optimization techniques including
gradient checkpointing, mixed precision training, and dynamic memory management.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
import logging
import time
import gc
from contextlib import contextmanager
from dataclasses import dataclass
import numpy as np
from collections import defaultdict


@dataclass
class MemoryStats:
    """Memory usage statistics."""
    allocated_mb: float
    cached_mb: float
    reserved_mb: float
    max_allocated_mb: float
    peak_memory_mb: float
    timestamp: float


class MemoryProfiler:
    """Advanced memory profiling and monitoring."""
    
    def __init__(self, enable_detailed_logging: bool = False):
        self.enable_detailed_logging = enable_detailed_logging
        self.memory_history = []
        self.peak_memory = 0.0
        self.baseline_memory = 0.0
        self.logger = logging.getLogger("MemoryProfiler")
        
        # Track memory by operation type
        self.operation_memory = defaultdict(list)
        
    def record_baseline(self):
        """Record baseline memory usage."""
        if torch.cuda.is_available():
            self.baseline_memory = torch.cuda.memory_allocated() / (1024**2)
        else:
            self.baseline_memory = 0.0
            
    def get_current_stats(self) -> MemoryStats:
        """Get current memory statistics."""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024**2)
            cached = torch.cuda.memory_cached() / (1024**2)
            reserved = torch.cuda.memory_reserved() / (1024**2)
            max_allocated = torch.cuda.max_memory_allocated() / (1024**2)
            
            current_peak = max(self.peak_memory, allocated)
            self.peak_memory = current_peak
            
            return MemoryStats(
                allocated_mb=allocated,
                cached_mb=cached,
                reserved_mb=reserved,
                max_allocated_mb=max_allocated,
                peak_memory_mb=current_peak,
                timestamp=time.time()
            )
        else:
            return MemoryStats(0, 0, 0, 0, 0, time.time())
    
    @contextmanager
    def profile_operation(self, operation_name: str):
        """Profile memory usage for a specific operation."""
        start_stats = self.get_current_stats()
        start_time = time.time()
        
        try:
            yield
        finally:
            end_stats = self.get_current_stats()
            duration = time.time() - start_time
            
            memory_delta = end_stats.allocated_mb - start_stats.allocated_mb
            
            self.operation_memory[operation_name].append({
                'memory_delta_mb': memory_delta,
                'duration_sec': duration,
                'peak_during_op': end_stats.peak_memory_mb - start_stats.peak_memory_mb
            })
            
            if self.enable_detailed_logging:
                self.logger.info(f"{operation_name}: {memory_delta:+.1f} MB, duration: {duration:.3f}s")
    
    def get_memory_report(self) -> Dict[str, Any]:
        """Generate comprehensive memory usage report."""
        current_stats = self.get_current_stats()
        
        report = {
            'current_memory_mb': current_stats.allocated_mb,
            'peak_memory_mb': self.peak_memory,
            'baseline_memory_mb': self.baseline_memory,
            'memory_overhead_mb': current_stats.allocated_mb - self.baseline_memory,
            'operation_statistics': {}
        }
        
        # Operation-specific statistics
        for op_name, measurements in self.operation_memory.items():
            if measurements:
                memory_deltas = [m['memory_delta_mb'] for m in measurements]
                durations = [m['duration_sec'] for m in measurements]
                
                report['operation_statistics'][op_name] = {
                    'avg_memory_delta_mb': np.mean(memory_deltas),
                    'max_memory_delta_mb': np.max(memory_deltas),
                    'total_memory_allocated_mb': np.sum([d for d in memory_deltas if d > 0]),
                    'avg_duration_sec': np.mean(durations),
                    'call_count': len(measurements)
                }
        
        return report


class GradientCheckpointManager:
    """Intelligent gradient checkpointing for memory optimization."""
    
    def __init__(
        self,
        memory_threshold_mb: float = 8000.0,  # 8GB
        checkpoint_ratio: float = 0.5,  # Checkpoint 50% of activations
        adaptive_checkpointing: bool = True
    ):
        self.memory_threshold_mb = memory_threshold_mb
        self.checkpoint_ratio = checkpoint_ratio
        self.adaptive_checkpointing = adaptive_checkpointing
        
        self.checkpointed_layers = set()
        self.memory_profiler = MemoryProfiler()
        self.logger = logging.getLogger("GradientCheckpointManager")
    
    def should_checkpoint(self, layer_name: str, current_memory_mb: float) -> bool:
        """Decide whether to checkpoint a specific layer."""
        
        if not self.adaptive_checkpointing:
            # Simple ratio-based checkpointing
            return hash(layer_name) % 100 < (self.checkpoint_ratio * 100)
        
        # Adaptive checkpointing based on memory usage
        if current_memory_mb > self.memory_threshold_mb:
            return True
        
        # Checkpoint expensive operations
        expensive_ops = ['conv', 'linear', 'attention', 'transformer']
        layer_lower = layer_name.lower()
        
        if any(op in layer_lower for op in expensive_ops):
            memory_pressure = current_memory_mb / self.memory_threshold_mb
            return memory_pressure > 0.7
        
        return False
    
    def checkpoint_forward(self, func: Callable, *args, **kwargs):
        """Apply gradient checkpointing to a forward function."""
        current_memory = self.memory_profiler.get_current_stats().allocated_mb
        
        if current_memory > self.memory_threshold_mb:
            # Use checkpointing
            return checkpoint(func, *args, **kwargs)
        else:
            # Regular forward pass
            return func(*args, **kwargs)


class MemoryEfficientLayer(nn.Module):
    """Base class for memory-efficient layer implementations."""
    
    def __init__(self, enable_checkpointing: bool = True):
        super().__init__()
        self.enable_checkpointing = enable_checkpointing
        self.checkpoint_manager = GradientCheckpointManager()
        
    def checkpointed_forward(self, func: Callable, *args, **kwargs):
        """Forward with optional checkpointing."""
        if self.enable_checkpointing and self.training:
            return self.checkpoint_manager.checkpoint_forward(func, *args, **kwargs)
        else:
            return func(*args, **kwargs)


class MemoryEfficientSpectralConv2d(MemoryEfficientLayer):
    """Memory-efficient spectral convolution with gradient checkpointing."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes1: int,
        modes2: int,
        enable_checkpointing: bool = True,
        use_mixed_precision: bool = True
    ):
        super().__init__(enable_checkpointing)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.use_mixed_precision = use_mixed_precision
        
        self.scale = 1 / (in_channels * out_channels)
        
        # Use half precision for weights if mixed precision is enabled
        dtype = torch.complex64 if use_mixed_precision else torch.complex128
        
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=dtype)
        )
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=dtype)
        )
    
    def spectral_conv_op(self, x: torch.Tensor) -> torch.Tensor:
        """Core spectral convolution operation."""
        batch_size = x.shape[0]
        
        # Use mixed precision if enabled
        if self.use_mixed_precision and x.dtype == torch.float32:
            with torch.cuda.amp.autocast():
                return self._spectral_conv_impl(x)
        else:
            return self._spectral_conv_impl(x)
    
    def _spectral_conv_impl(self, x: torch.Tensor) -> torch.Tensor:
        """Implementation of spectral convolution."""
        batch_size = x.shape[0]
        
        # Compute Fourier coefficients
        x_ft = torch.fft.rfft2(x)
        
        # Multiply relevant Fourier modes
        out_ft = torch.zeros(
            batch_size, self.out_channels, x.size(-2), x.size(-1)//2 + 1,
            dtype=torch.complex64 if self.use_mixed_precision else torch.complex128,
            device=x.device
        )
        
        # First set of modes
        out_ft[:, :, :self.modes1, :self.modes2] = torch.einsum(
            "bixy,ioxy->boxy",
            x_ft[:, :, :self.modes1, :self.modes2],
            self.weights1
        )
        
        # Second set of modes
        out_ft[:, :, -self.modes1:, :self.modes2] = torch.einsum(
            "bixy,ioxy->boxy",
            x_ft[:, :, -self.modes1:, :self.modes2],
            self.weights2
        )
        
        # Return to physical space
        return torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with optional checkpointing."""
        return self.checkpointed_forward(self.spectral_conv_op, x)


class MemoryEfficientPNOBlock(MemoryEfficientLayer):
    """Memory-efficient PNO block with advanced optimizations."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes1: int,
        modes2: int,
        enable_checkpointing: bool = True,
        use_mixed_precision: bool = True,
        activation_offloading: bool = False
    ):
        super().__init__(enable_checkpointing)
        
        self.activation_offloading = activation_offloading
        
        # Spectral convolution
        self.spectral_conv = MemoryEfficientSpectralConv2d(
            in_channels, out_channels, modes1, modes2,
            enable_checkpointing, use_mixed_precision
        )
        
        # Local convolution with potential memory optimization
        self.local_conv = nn.Conv2d(in_channels, out_channels, 1)
        
        # Activation function
        self.activation = nn.GELU()
        
        # Normalization (with memory-efficient variants)
        self.norm = nn.GroupNorm(min(32, out_channels), out_channels)
        
    def block_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Core block forward operation."""
        
        # Spectral branch
        spectral_out = self.spectral_conv(x)
        
        # Local branch
        local_out = self.local_conv(x)
        
        # Combine
        out = spectral_out + local_out
        
        # Normalization and activation
        out = self.norm(out)
        out = self.activation(out)
        
        return out
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with memory optimizations."""
        
        if self.activation_offloading and not self.training:
            # Offload activations to CPU during inference
            return self._forward_with_offloading(x)
        else:
            return self.checkpointed_forward(self.block_forward, x)
    
    def _forward_with_offloading(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with activation offloading."""
        
        # Move to CPU after each major operation
        device = x.device
        
        # Spectral convolution
        x_cpu = x.cpu()
        spectral_out = self.spectral_conv(x_cpu.to(device)).cpu()
        
        # Local convolution
        local_out = self.local_conv(x_cpu.to(device)).cpu()
        
        # Combine and final operations
        out = (spectral_out + local_out).to(device)
        out = self.norm(out)
        out = self.activation(out)
        
        return out


class DynamicMemoryManager:
    """Dynamic memory management for training optimization."""
    
    def __init__(
        self,
        target_memory_gb: float = 10.0,
        cleanup_threshold: float = 0.9,
        aggressive_cleanup: bool = False
    ):
        self.target_memory_bytes = target_memory_gb * (1024**3)
        self.cleanup_threshold = cleanup_threshold
        self.aggressive_cleanup = aggressive_cleanup
        
        self.memory_profiler = MemoryProfiler()
        self.cleanup_count = 0
        self.logger = logging.getLogger("DynamicMemoryManager")
    
    def check_and_cleanup(self, force: bool = False) -> bool:
        """Check memory usage and cleanup if necessary."""
        
        if not torch.cuda.is_available():
            return False
        
        current_memory = torch.cuda.memory_allocated()
        memory_usage_ratio = current_memory / self.target_memory_bytes
        
        if force or memory_usage_ratio > self.cleanup_threshold:
            return self._perform_cleanup()
        
        return False
    
    def _perform_cleanup(self) -> bool:
        """Perform memory cleanup operations."""
        
        initial_memory = torch.cuda.memory_allocated() / (1024**2)
        
        # Clear cache
        torch.cuda.empty_cache()
        
        # Force garbage collection
        gc.collect()
        
        if self.aggressive_cleanup:
            # Additional aggressive cleanup
            self._aggressive_memory_cleanup()
        
        final_memory = torch.cuda.memory_allocated() / (1024**2)
        memory_freed = initial_memory - final_memory
        
        self.cleanup_count += 1
        self.logger.info(f"Memory cleanup #{self.cleanup_count}: freed {memory_freed:.1f} MB")
        
        return memory_freed > 0
    
    def _aggressive_memory_cleanup(self):
        """Aggressive memory cleanup operations."""
        
        # Clear all caches
        if hasattr(torch.cuda, 'ipc_collect'):
            torch.cuda.ipc_collect()
        
        # Force synchronization
        torch.cuda.synchronize()
        
        # Multiple cache clears
        for _ in range(3):
            torch.cuda.empty_cache()
            gc.collect()
    
    @contextmanager
    def managed_memory_context(self):
        """Context manager for automatic memory management."""
        
        initial_stats = self.memory_profiler.get_current_stats()
        
        try:
            yield self
        finally:
            # Check if cleanup is needed
            self.check_and_cleanup()
            
            final_stats = self.memory_profiler.get_current_stats()
            memory_change = final_stats.allocated_mb - initial_stats.allocated_mb
            
            if abs(memory_change) > 100:  # Log significant memory changes
                self.logger.debug(f"Memory context: {memory_change:+.1f} MB change")


class MemoryOptimizedTrainer:
    """Training system with comprehensive memory optimizations."""
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        enable_mixed_precision: bool = True,
        enable_gradient_checkpointing: bool = True,
        max_memory_gb: float = 10.0,
        accumulate_gradients: int = 1
    ):
        self.model = model
        self.optimizer = optimizer
        self.enable_mixed_precision = enable_mixed_precision
        self.enable_gradient_checkpointing = enable_gradient_checkpointing
        self.accumulate_gradients = accumulate_gradients
        
        # Memory management
        self.memory_manager = DynamicMemoryManager(target_memory_gb=max_memory_gb)
        self.memory_profiler = MemoryProfiler(enable_detailed_logging=True)
        
        # Mixed precision scaler
        if enable_mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None
        
        # Gradient checkpointing setup
        if enable_gradient_checkpointing:
            self._setup_gradient_checkpointing()
        
        self.training_step = 0
        self.logger = logging.getLogger("MemoryOptimizedTrainer")
    
    def _setup_gradient_checkpointing(self):
        """Setup gradient checkpointing for model layers."""
        
        checkpoint_manager = GradientCheckpointManager()
        
        for name, module in self.model.named_modules():
            if isinstance(module, MemoryEfficientLayer):
                # Already has checkpointing capability
                continue
            elif isinstance(module, (nn.TransformerEncoderLayer, nn.TransformerDecoderLayer)):
                # Wrap transformer layers
                module._forward = checkpoint_wrapper(module._forward)
            elif isinstance(module, (nn.Conv2d, nn.Linear)) and 'large' in name.lower():
                # Checkpoint large layers
                module.forward = checkpoint_wrapper(module.forward)
    
    def train_step(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        loss_fn: Callable
    ) -> Dict[str, float]:
        """Memory-optimized training step."""
        
        self.training_step += 1
        
        with self.memory_manager.managed_memory_context():
            with self.memory_profiler.profile_operation("forward_pass"):
                # Forward pass with optional mixed precision
                if self.enable_mixed_precision and self.scaler is not None:
                    with torch.cuda.amp.autocast():
                        predictions = self.model(inputs)
                        loss = loss_fn(predictions, targets) / self.accumulate_gradients
                else:
                    predictions = self.model(inputs)
                    loss = loss_fn(predictions, targets) / self.accumulate_gradients
            
            with self.memory_profiler.profile_operation("backward_pass"):
                # Backward pass
                if self.enable_mixed_precision and self.scaler is not None:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
            
            # Gradient accumulation
            if self.training_step % self.accumulate_gradients == 0:
                with self.memory_profiler.profile_operation("optimizer_step"):
                    if self.enable_mixed_precision and self.scaler is not None:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    
                    self.optimizer.zero_grad()
            
            # Memory cleanup check
            if self.training_step % 10 == 0:
                self.memory_manager.check_and_cleanup()
        
        # Return metrics
        return {
            'loss': loss.item() * self.accumulate_gradients,
            'memory_allocated_mb': torch.cuda.memory_allocated() / (1024**2) if torch.cuda.is_available() else 0,
            'peak_memory_mb': self.memory_profiler.peak_memory
        }
    
    def train_epoch(
        self,
        dataloader: torch.utils.data.DataLoader,
        loss_fn: Callable,
        validation_loader: Optional[torch.utils.data.DataLoader] = None
    ) -> Dict[str, Any]:
        """Train for one epoch with memory monitoring."""
        
        self.model.train()
        epoch_losses = []
        epoch_memory_stats = []
        
        self.memory_profiler.record_baseline()
        
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            # Move to device
            inputs = inputs.cuda() if torch.cuda.is_available() else inputs
            targets = targets.cuda() if torch.cuda.is_available() else targets
            
            # Training step
            step_metrics = self.train_step(inputs, targets, loss_fn)
            
            epoch_losses.append(step_metrics['loss'])
            epoch_memory_stats.append(step_metrics['memory_allocated_mb'])
            
            # Log progress
            if batch_idx % 100 == 0:
                self.logger.info(
                    f"Batch {batch_idx}: Loss={step_metrics['loss']:.4f}, "
                    f"Memory={step_metrics['memory_allocated_mb']:.1f}MB"
                )
        
        # Validation
        val_metrics = {}
        if validation_loader is not None:
            val_metrics = self.validate(validation_loader, loss_fn)
        
        # Generate comprehensive report
        memory_report = self.memory_profiler.get_memory_report()
        
        return {
            'train_loss': np.mean(epoch_losses),
            'avg_memory_mb': np.mean(epoch_memory_stats),
            'max_memory_mb': np.max(epoch_memory_stats),
            'memory_report': memory_report,
            'validation_metrics': val_metrics
        }
    
    def validate(
        self,
        dataloader: torch.utils.data.DataLoader,
        loss_fn: Callable
    ) -> Dict[str, float]:
        """Memory-efficient validation."""
        
        self.model.eval()
        val_losses = []
        
        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs = inputs.cuda() if torch.cuda.is_available() else inputs
                targets = targets.cuda() if torch.cuda.is_available() else targets
                
                if self.enable_mixed_precision:
                    with torch.cuda.amp.autocast():
                        predictions = self.model(inputs)
                        loss = loss_fn(predictions, targets)
                else:
                    predictions = self.model(inputs)
                    loss = loss_fn(predictions, targets)
                
                val_losses.append(loss.item())
        
        return {
            'validation_loss': np.mean(val_losses),
            'validation_memory_mb': torch.cuda.memory_allocated() / (1024**2) if torch.cuda.is_available() else 0
        }


def checkpoint_wrapper(forward_fn: Callable) -> Callable:
    """Wrapper to add gradient checkpointing to any forward function."""
    
    def wrapped_forward(*args, **kwargs):
        if torch.is_grad_enabled():
            return checkpoint(forward_fn, *args, **kwargs)
        else:
            return forward_fn(*args, **kwargs)
    
    return wrapped_forward


class AdaptiveMemoryScheduler:
    """Adaptive scheduler for memory-conscious training."""
    
    def __init__(
        self,
        initial_batch_size: int = 32,
        memory_threshold_gb: float = 8.0,
        batch_size_step: int = 4
    ):
        self.initial_batch_size = initial_batch_size
        self.current_batch_size = initial_batch_size
        self.memory_threshold_bytes = memory_threshold_gb * (1024**3)
        self.batch_size_step = batch_size_step
        
        self.memory_history = deque(maxlen=10)
        self.adjustment_count = 0
        self.logger = logging.getLogger("AdaptiveMemoryScheduler")
    
    def adjust_batch_size(self, current_memory: float, loss_trend: float) -> int:
        """Dynamically adjust batch size based on memory usage and training progress."""
        
        self.memory_history.append(current_memory)
        
        # Calculate memory trend
        if len(self.memory_history) >= 5:
            recent_memory = np.mean(list(self.memory_history)[-5:])
            memory_pressure = recent_memory / self.memory_threshold_bytes
            
            # Adjust batch size based on memory pressure
            if memory_pressure > 0.9:
                # High memory pressure - reduce batch size
                new_batch_size = max(4, self.current_batch_size - self.batch_size_step)
                if new_batch_size != self.current_batch_size:
                    self.adjustment_count += 1
                    self.logger.info(f"Reducing batch size: {self.current_batch_size} -> {new_batch_size}")
            elif memory_pressure < 0.6 and loss_trend < 0:
                # Low memory pressure and good training progress - increase batch size
                new_batch_size = min(256, self.current_batch_size + self.batch_size_step)
                if new_batch_size != self.current_batch_size:
                    self.adjustment_count += 1
                    self.logger.info(f"Increasing batch size: {self.current_batch_size} -> {new_batch_size}")
            else:
                new_batch_size = self.current_batch_size
            
            self.current_batch_size = new_batch_size
        
        return self.current_batch_size