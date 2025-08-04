"""Reproducibility utilities for consistent results."""

import torch
import numpy as np
import random
import os
import logging
from typing import Optional, Dict, Any
import hashlib
import json


logger = logging.getLogger(__name__)


def set_random_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Set random seeds for reproducible results.
    
    Args:
        seed: Random seed value
        deterministic: Whether to use deterministic algorithms (slower but reproducible)
    """
    # Python random
    random.seed(seed)
    
    # NumPy random
    np.random.seed(seed)
    
    # PyTorch random
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # Multi-GPU
    
    # Set deterministic behavior
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # Set environment variable for additional determinism
        os.environ['PYTHONHASHSEED'] = str(seed)
        
        # Use deterministic algorithms where available
        try:
            torch.use_deterministic_algorithms(True)
        except AttributeError:
            # Older PyTorch versions
            torch.set_deterministic(True)
        except Exception as e:
            logger.warning(f"Could not enable deterministic algorithms: {e}")
    
    logger.info(f"Random seed set to {seed} (deterministic={deterministic})")


def get_random_state() -> Dict[str, Any]:
    """Get current random state for all generators."""
    state = {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
    }
    
    if torch.cuda.is_available():
        state['torch_cuda'] = torch.cuda.get_rng_state()
        if torch.cuda.device_count() > 1:
            state['torch_cuda_all'] = torch.cuda.get_rng_state_all()
    
    return state


def set_random_state(state: Dict[str, Any]) -> None:
    """Restore random state for all generators."""
    if 'python' in state:
        random.setstate(state['python'])
    
    if 'numpy' in state:
        np.random.set_state(state['numpy'])
    
    if 'torch' in state:
        torch.set_rng_state(state['torch'])
    
    if torch.cuda.is_available():
        if 'torch_cuda' in state:
            torch.cuda.set_rng_state(state['torch_cuda'])
        if 'torch_cuda_all' in state and torch.cuda.device_count() > 1:
            torch.cuda.set_rng_state_all(state['torch_cuda_all'])


def create_experiment_hash(config: Dict[str, Any], exclude_keys: Optional[list] = None) -> str:
    """Create deterministic hash from experiment configuration.
    
    Args:
        config: Experiment configuration dictionary
        exclude_keys: Keys to exclude from hash (e.g., output paths, random seeds)
        
    Returns:
        Hexadecimal hash string
    """
    if exclude_keys is None:
        exclude_keys = ['output_dir', 'log_dir', 'checkpoint_path', 'random_seed', 'device']
    
    # Create filtered config
    filtered_config = {k: v for k, v in config.items() if k not in exclude_keys}
    
    # Convert to deterministic JSON string
    config_str = json.dumps(filtered_config, sort_keys=True, separators=(',', ':'))
    
    # Create hash
    hash_obj = hashlib.sha256(config_str.encode('utf-8'))
    return hash_obj.hexdigest()[:16]  # Use first 16 characters


def get_device_info() -> Dict[str, Any]:
    """Get comprehensive device and environment information."""
    info = {
        'python_version': os.sys.version,
        'torch_version': torch.__version__,
        'numpy_version': np.__version__,
        'cuda_available': torch.cuda.is_available(),
        'device_count': 0,
        'devices': []
    }
    
    if torch.cuda.is_available():
        info['device_count'] = torch.cuda.device_count()
        info['cuda_version'] = torch.version.cuda
        info['cudnn_version'] = torch.backends.cudnn.version()
        
        for i in range(info['device_count']):
            props = torch.cuda.get_device_properties(i)
            device_info = {
                'device_id': i,
                'name': props.name,
                'memory_total': props.total_memory,
                'memory_allocated': torch.cuda.memory_allocated(i),
                'memory_reserved': torch.cuda.memory_reserved(i),
                'compute_capability': f"{props.major}.{props.minor}",
                'multiprocessor_count': props.multi_processor_count
            }
            info['devices'].append(device_info)
    
    # CPU information
    try:
        import psutil
        info['cpu_count'] = psutil.cpu_count()
        info['memory_total'] = psutil.virtual_memory().total
        info['memory_available'] = psutil.virtual_memory().available
    except ImportError:
        info['cpu_count'] = os.cpu_count()
    
    return info


class ReproducibilityManager:
    """Manager for ensuring reproducible experiments."""
    
    def __init__(self, base_seed: int = 42):
        self.base_seed = base_seed
        self.experiment_seeds = {}
        self.random_states = {}
    
    def get_seed_for_experiment(self, experiment_name: str) -> int:
        """Get deterministic seed for a named experiment."""
        if experiment_name not in self.experiment_seeds:
            # Create deterministic seed from name
            hash_obj = hashlib.sha256(f"{experiment_name}_{self.base_seed}".encode())
            seed = int(hash_obj.hexdigest()[:8], 16) % (2**31)  # Ensure positive 32-bit int
            self.experiment_seeds[experiment_name] = seed
        
        return self.experiment_seeds[experiment_name]
    
    def setup_experiment(self, experiment_name: str, deterministic: bool = True) -> int:
        """Setup reproducible environment for experiment."""
        seed = self.get_seed_for_experiment(experiment_name)
        
        # Save current state
        self.random_states[experiment_name] = get_random_state()
        
        # Set new seed
        set_random_seed(seed, deterministic)
        
        logger.info(f"Experiment '{experiment_name}' setup with seed {seed}")
        return seed
    
    def restore_state(self, experiment_name: str) -> None:
        """Restore random state for experiment."""
        if experiment_name in self.random_states:
            set_random_state(self.random_states[experiment_name])
            logger.info(f"Random state restored for experiment '{experiment_name}'")
    
    def create_reproducible_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Add reproducibility information to configuration."""
        repro_config = config.copy()
        
        # Add environment info
        repro_config['reproducibility'] = {
            'device_info': get_device_info(),
            'random_seed': self.base_seed,
            'experiment_hash': create_experiment_hash(config),
            'torch_version': torch.__version__,
            'deterministic': torch.backends.cudnn.deterministic if torch.cuda.is_available() else None
        }
        
        return repro_config


def ensure_deterministic_dataloader(dataloader, worker_init_fn=None):
    """Ensure DataLoader uses deterministic behavior."""
    
    def deterministic_worker_init_fn(worker_id):
        # Set worker-specific seed
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        
        # Call custom worker init function if provided
        if worker_init_fn is not None:
            worker_init_fn(worker_id)
    
    # Update dataloader settings for determinism
    if hasattr(dataloader, 'worker_init_fn'):
        dataloader.worker_init_fn = deterministic_worker_init_fn
    
    # Ensure generator is set for reproducible sampling
    if hasattr(dataloader, 'generator') and dataloader.generator is None:
        dataloader.generator = torch.Generator()
        dataloader.generator.manual_seed(42)
    
    return dataloader


def log_environment_info():
    """Log comprehensive environment information."""
    info = get_device_info()
    
    logger.info("=== ENVIRONMENT INFO ===")
    logger.info(f"Python: {info['python_version']}")
    logger.info(f"PyTorch: {info['torch_version']}")
    logger.info(f"NumPy: {info['numpy_version']}")
    
    if info['cuda_available']:
        logger.info(f"CUDA: {info['cuda_version']}")
        logger.info(f"cuDNN: {info['cudnn_version']}")
        logger.info(f"GPU devices: {info['device_count']}")
        
        for device in info['devices']:
            memory_gb = device['memory_total'] / (1024**3)
            logger.info(f"  GPU {device['device_id']}: {device['name']} ({memory_gb:.1f}GB)")
    else:
        logger.info("CUDA: Not available")
    
    logger.info(f"CPU cores: {info.get('cpu_count', 'Unknown')}")
    if 'memory_total' in info:
        memory_gb = info['memory_total'] / (1024**3)
        logger.info(f"System memory: {memory_gb:.1f}GB")
    
    logger.info("=== END ENVIRONMENT INFO ===")


def format_time(seconds: float) -> str:
    """Format seconds into human-readable time string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"