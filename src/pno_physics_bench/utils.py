"""Utility functions for PNO Physics Bench."""

import logging
import random
import numpy as np
import torch
import os
from typing import Optional, Union
import coloredlogs


def setup_logging(
    level: int = logging.INFO,
    format_string: Optional[str] = None,
    use_colors: bool = True,
) -> None:
    """Setup logging configuration.
    
    Args:
        level: Logging level
        format_string: Custom format string
        use_colors: Whether to use colored logs
    """
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    if use_colors:
        coloredlogs.install(
            level=level,
            fmt=format_string,
            field_styles={
                'asctime': {'color': 'green'},
                'hostname': {'color': 'magenta'},
                'levelname': {'bold': True, 'color': 'black'},
                'name': {'color': 'blue'},
                'programname': {'color': 'cyan'},
                'username': {'color': 'yellow'},
            }
        )
    else:
        logging.basicConfig(level=level, format=format_string)


def set_random_seed(seed: int) -> None:
    """Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # For deterministic behavior (may reduce performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set environment variable for Python hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)


def count_parameters(model: torch.nn.Module) -> int:
    """Count the number of trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_device(gpu_id: Optional[int] = None) -> torch.device:
    """Get torch device for computation.
    
    Args:
        gpu_id: GPU device ID (None for auto-select)
        
    Returns:
        Torch device
    """
    if torch.cuda.is_available():
        if gpu_id is not None:
            return torch.device(f'cuda:{gpu_id}')
        else:
            return torch.device('cuda')
    else:
        return torch.device('cpu')


def format_time(seconds: float) -> str:
    """Format time in seconds to human readable string.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def format_number(number: Union[int, float]) -> str:
    """Format large numbers with appropriate suffixes.
    
    Args:
        number: Number to format
        
    Returns:
        Formatted number string
    """
    if isinstance(number, int):
        if number >= 1_000_000:
            return f"{number / 1_000_000:.1f}M"
        elif number >= 1_000:
            return f"{number / 1_000:.1f}K"
        else:
            return str(number)
    else:
        if abs(number) >= 1_000_000:
            return f"{number / 1_000_000:.1f}M"
        elif abs(number) >= 1_000:
            return f"{number / 1_000:.1f}K"
        else:
            return f"{number:.3f}"


def ensure_dir(path: Union[str, os.PathLike]) -> None:
    """Ensure directory exists, create if not.
    
    Args:
        path: Directory path
    """
    os.makedirs(path, exist_ok=True)


def get_memory_usage() -> dict:
    """Get current memory usage statistics.
    
    Returns:
        Dictionary with memory usage info
    """
    import psutil
    
    # System memory
    system_memory = psutil.virtual_memory()
    
    memory_info = {
        'system_total_gb': system_memory.total / (1024**3),
        'system_available_gb': system_memory.available / (1024**3),
        'system_used_gb': system_memory.used / (1024**3),
        'system_percent': system_memory.percent,
    }
    
    # GPU memory if available
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / (1024**3)
            reserved = torch.cuda.memory_reserved(i) / (1024**3)
            memory_info[f'gpu_{i}_allocated_gb'] = allocated
            memory_info[f'gpu_{i}_reserved_gb'] = reserved
    
    return memory_info


def log_memory_usage(logger: logging.Logger) -> None:
    """Log current memory usage.
    
    Args:
        logger: Logger instance
    """
    try:
        memory_info = get_memory_usage()
        logger.info(f"System Memory: {memory_info['system_used_gb']:.1f}GB / "
                   f"{memory_info['system_total_gb']:.1f}GB ({memory_info['system_percent']:.1f}%)")
        
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                if f'gpu_{i}_allocated_gb' in memory_info:
                    logger.info(f"GPU {i} Memory: {memory_info[f'gpu_{i}_allocated_gb']:.1f}GB allocated, "
                               f"{memory_info[f'gpu_{i}_reserved_gb']:.1f}GB reserved")
    except Exception as e:
        logger.warning(f"Failed to get memory usage: {e}")


class AverageMeter:
    """Computes and stores the average and current value."""
    
    def __init__(self, name: str, fmt: str = ':f'):
        self.name = name
        self.fmt = fmt
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter:
    """Display progress of training/evaluation."""
    
    def __init__(self, num_batches: int, meters: list, prefix: str = ""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
    
    def display(self, batch: int, logger: logging.Logger):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        logger.info('\t'.join(entries))
    
    def _get_batch_fmtstr(self, num_batches: int):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def save_checkpoint(
    state: dict,
    is_best: bool,
    filename: str = 'checkpoint.pth.tar',
    best_filename: str = 'model_best.pth.tar'
) -> None:
    """Save training checkpoint.
    
    Args:
        state: State dictionary to save
        is_best: Whether this is the best checkpoint so far
        filename: Checkpoint filename
        best_filename: Best checkpoint filename
    """
    torch.save(state, filename)
    if is_best:
        import shutil
        shutil.copyfile(filename, best_filename)


def load_checkpoint(
    filename: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[torch.device] = None
) -> dict:
    """Load training checkpoint.
    
    Args:
        filename: Checkpoint filename
        model: Model to load state into
        optimizer: Optimizer to load state into
        device: Device to map tensors to
        
    Returns:
        Checkpoint dictionary
    """
    if device is None:
        checkpoint = torch.load(filename)
    else:
        checkpoint = torch.load(filename, map_location=device)
    
    model.load_state_dict(checkpoint['state_dict'])
    
    if optimizer is not None and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
    
    return checkpoint


def compute_gradient_norm(model: torch.nn.Module) -> float:
    """Compute the L2 norm of model gradients.
    
    Args:
        model: PyTorch model
        
    Returns:
        Gradient norm
    """
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm


def freeze_model(model: torch.nn.Module) -> None:
    """Freeze all parameters in a model.
    
    Args:
        model: Model to freeze
    """
    for param in model.parameters():
        param.requires_grad = False


def unfreeze_model(model: torch.nn.Module) -> None:
    """Unfreeze all parameters in a model.
    
    Args:
        model: Model to unfreeze
    """
    for param in model.parameters():
        param.requires_grad = True


def get_learning_rate(optimizer: torch.optim.Optimizer) -> float:
    """Get current learning rate from optimizer.
    
    Args:
        optimizer: PyTorch optimizer
        
    Returns:
        Current learning rate
    """
    for param_group in optimizer.param_groups:
        return param_group['lr']
    return 0.0


def warmup_learning_rate(
    optimizer: torch.optim.Optimizer,
    current_step: int,
    warmup_steps: int,
    base_lr: float
) -> None:
    """Apply learning rate warmup.
    
    Args:
        optimizer: PyTorch optimizer
        current_step: Current training step
        warmup_steps: Number of warmup steps
        base_lr: Base learning rate
    """
    if current_step < warmup_steps:
        lr = base_lr * (current_step + 1) / warmup_steps
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr