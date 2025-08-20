# SECURITY NOTICE: This file has been automatically hardened for security
# - All user inputs should be validated and sanitized
# - Subprocess calls use secure alternatives
# - SQL queries use parameterized statements
# - No hardcoded secrets or credentials


"""Input validation and model configuration checks."""

import torch
import numpy as np
from typing import Dict, Any, Optional, Tuple, Union, List
import logging

from .error_handling import ModelConfigError, DataError, validate_tensor


logger = logging.getLogger(__name__)


def validate_input_shapes(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    model_config: Dict[str, Any]
) -> None:
    """Validate input/target shapes against model configuration."""
    
    # Basic tensor validation
    validate_tensor(inputs, "inputs")
    validate_tensor(targets, "targets")
    
    # Shape validation
    if len(inputs.shape) != 4:
        raise DataError(
            f"Inputs must be 4D (batch, channels, height, width), got shape {inputs.shape}"
        )
    
    if len(targets.shape) != 4:
        raise DataError(
            f"Targets must be 4D (batch, channels, height, width), got shape {targets.shape}"
        )
    
    batch_size, input_channels, height, width = inputs.shape
    target_batch, target_channels, target_height, target_width = targets.shape
    
    # Batch size consistency
    if batch_size != target_batch:
        raise DataError(
            f"Input and target batch sizes don't match: {batch_size} vs {target_batch}"
        )
    
    # Spatial dimensions consistency
    if height != target_height or width != target_width:
        raise DataError(
            f"Input and target spatial dimensions don't match: "
            f"({height}, {width}) vs ({target_height}, {target_width})"
        )
    
    # Model configuration consistency
    expected_input_dim = model_config.get('input_dim')
    expected_output_dim = model_config.get('output_dim')
    
    if expected_input_dim is not None and input_channels != expected_input_dim:
        raise DataError(
            f"Input channels ({input_channels}) don't match model input_dim ({expected_input_dim})"
        )
    
    if expected_output_dim is not None and target_channels != expected_output_dim:
        raise DataError(
            f"Target channels ({target_channels}) don't match model output_dim ({expected_output_dim})"
        )
    
    # Spatial resolution validation
    expected_size = model_config.get('input_size')
    if expected_size is not None:
        if isinstance(expected_size, (list, tuple)) and len(expected_size) == 2:
            exp_h, exp_w = expected_size
            if height != exp_h or width != exp_w:
                logger.warning(
                    f"Input size ({height}, {width}) doesn't match expected ({exp_h}, {exp_w}). "
                    f"Model may need to resize position encodings."
                )


def validate_model_config(config: Dict[str, Any], model_type: str = "PNO") -> Dict[str, Any]:
    """Validate and normalize model configuration."""
    
    validated_config = config.copy()
    
    # Required parameters
    required_params = {
        "PNO": ['input_dim', 'output_dim', 'hidden_dim'],
        "FNO": ['input_dim', 'output_dim', 'hidden_dim'],
        "DeepONet": ['input_dim', 'output_dim', 'grid_size']
    }
    
    if model_type not in required_params:
        raise ModelConfigError(f"Unknown model type: {model_type}")
    
    for param in required_params[model_type]:
        if param not in config:
            raise ModelConfigError(f"Missing required parameter: {param}")
        
        if not isinstance(config[param], int) or config[param] <= 0:
            raise ModelConfigError(f"Parameter {param} must be a positive integer, got {config[param]}")
    
    # Model-specific validation
    if model_type in ["PNO", "FNO"]:
        # Validate modes parameter
        modes = validated_config.get('modes', 20)
        if not isinstance(modes, int) or modes <= 0:
            raise ModelConfigError(f"modes must be a positive integer, got {modes}")
        validated_config['modes'] = modes
        
        # Validate num_layers
        num_layers = validated_config.get('num_layers', 4)
        if not isinstance(num_layers, int) or num_layers <= 0:
            raise ModelConfigError(f"num_layers must be a positive integer, got {num_layers}")
        validated_config['num_layers'] = num_layers
        
        # Validate input_size
        input_size = validated_config.get('input_size', (64, 64))
        if not isinstance(input_size, (list, tuple)) or len(input_size) != 2:
            raise ModelConfigError(f"input_size must be a tuple of 2 integers, got {input_size}")
        if any(not isinstance(x, int) or x <= 0 for x in input_size):
            raise ModelConfigError(f"input_size values must be positive integers, got {input_size}")
        validated_config['input_size'] = tuple(input_size)
    
    elif model_type == "DeepONet":
        # Validate grid_size
        grid_size = validated_config['grid_size']
        if not isinstance(grid_size, int) or grid_size <= 0:
            raise ModelConfigError(f"grid_size must be a positive integer, got {grid_size}")
        
        # Validate hidden dimensions
        branch_dims = validated_config.get('branch_hidden_dims', (128, 128, 128))
        trunk_dims = validated_config.get('trunk_hidden_dims', (128, 128, 128))
        
        for dims, name in [(branch_dims, 'branch_hidden_dims'), (trunk_dims, 'trunk_hidden_dims')]:
            if not isinstance(dims, (list, tuple)):
                raise ModelConfigError(f"{name} must be a list or tuple")
            if any(not isinstance(x, int) or x <= 0 for x in dims):
                raise ModelConfigError(f"{name} must contain positive integers, got {dims}")
        
        validated_config['branch_hidden_dims'] = tuple(branch_dims)
        validated_config['trunk_hidden_dims'] = tuple(trunk_dims)
    
    # PNO-specific validation
    if model_type == "PNO":
        uncertainty_type = validated_config.get('uncertainty_type', 'diagonal')
        if uncertainty_type not in ['diagonal', 'full']:
            raise ModelConfigError(f"uncertainty_type must be 'diagonal' or 'full', got {uncertainty_type}")
        validated_config['uncertainty_type'] = uncertainty_type
    
    logger.info(f"Model configuration validated for {model_type}")
    return validated_config


def validate_training_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate training configuration parameters."""
    
    validated_config = config.copy()
    
    # Learning rate validation
    lr = validated_config.get('learning_rate', 1e-3)
    if not isinstance(lr, (int, float)) or lr <= 0:
        raise ModelConfigError(f"learning_rate must be positive, got {lr}")
    if lr > 1.0:
        logger.warning(f"High learning rate ({lr}) may cause training instability")
    validated_config['learning_rate'] = float(lr)
    
    # Batch size validation
    batch_size = validated_config.get('batch_size', 32)
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ModelConfigError(f"batch_size must be a positive integer, got {batch_size}")
    validated_config['batch_size'] = batch_size
    
    # Epochs validation
    epochs = validated_config.get('epochs', 100)
    if not isinstance(epochs, int) or epochs <= 0:
        raise ModelConfigError(f"epochs must be a positive integer, got {epochs}")
    validated_config['epochs'] = epochs
    
    # Loss function parameters
    kl_weight = validated_config.get('kl_weight', 1e-4)
    if not isinstance(kl_weight, (int, float)) or kl_weight < 0:
        raise ModelConfigError(f"kl_weight must be non-negative, got {kl_weight}")
    validated_config['kl_weight'] = float(kl_weight)
    
    # Gradient clipping
    grad_clip = validated_config.get('gradient_clipping', 1.0)
    if grad_clip is not None:
        if not isinstance(grad_clip, (int, float)) or grad_clip <= 0:
            raise ModelConfigError(f"gradient_clipping must be positive or None, got {grad_clip}")
        validated_config['gradient_clipping'] = float(grad_clip)
    
    # Early stopping patience
    patience = validated_config.get('early_stopping_patience', 10)
    if not isinstance(patience, int) or patience <= 0:
        raise ModelConfigError(f"early_stopping_patience must be a positive integer, got {patience}")
    validated_config['early_stopping_patience'] = patience
    
    logger.info("Training configuration validated")
    return validated_config


def validate_dataset_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate dataset configuration."""
    
    validated_config = config.copy()
    
    # PDE name validation
    pde_name = validated_config.get('pde_name')
    if pde_name is None:
        raise DataError("pde_name is required")
    
    valid_pdes = ['navier_stokes_2d', 'darcy_flow_2d', 'burgers_1d']
    if pde_name not in valid_pdes:
        raise DataError(f"pde_name must be one of {valid_pdes}, got {pde_name}")
    
    # Resolution validation
    resolution = validated_config.get('resolution', 64)
    if not isinstance(resolution, int) or resolution <= 0:
        raise DataError(f"resolution must be a positive integer, got {resolution}")
    if resolution < 8:
        logger.warning(f"Very low resolution ({resolution}) may affect model performance")
    if resolution > 512:
        logger.warning(f"High resolution ({resolution}) will require significant memory")
    validated_config['resolution'] = resolution
    
    # Number of samples validation
    num_samples = validated_config.get('num_samples', 1000)
    if not isinstance(num_samples, int) or num_samples <= 0:
        raise DataError(f"num_samples must be a positive integer, got {num_samples}")
    validated_config['num_samples'] = num_samples
    
    # Split validation
    val_split = validated_config.get('val_split', 0.2)
    test_split = validated_config.get('test_split', 0.1)
    
    for split, name in [(val_split, 'val_split'), (test_split, 'test_split')]:
        if not isinstance(split, (int, float)) or split < 0 or split >= 1:
            raise DataError(f"{name} must be between 0 and 1, got {split}")
    
    if val_split + test_split >= 1.0:
        raise DataError(f"val_split + test_split must be < 1.0, got {val_split + test_split}")
    
    validated_config['val_split'] = float(val_split)
    validated_config['test_split'] = float(test_split)
    
    logger.info(f"Dataset configuration validated for {pde_name}")
    return validated_config


def check_computational_requirements(
    model_config: Dict[str, Any],
    dataset_config: Dict[str, Any],
    training_config: Dict[str, Any]
) -> Dict[str, Any]:
    """Estimate computational requirements and warn about potential issues."""
    
    # Estimate model parameters
    hidden_dim = model_config.get('hidden_dim', 64)
    num_layers = model_config.get('num_layers', 4)
    input_dim = model_config.get('input_dim', 3)
    output_dim = model_config.get('output_dim', 3)
    
    # Rough parameter estimation for PNO
    estimated_params = (
        input_dim * hidden_dim +  # input projection
        num_layers * hidden_dim * hidden_dim * 4 +  # PNO blocks  
        hidden_dim * output_dim  # output head
    )
    
    # Memory estimation
    batch_size = training_config.get('batch_size', 32)
    resolution = dataset_config.get('resolution', 64)
    
    # Rough memory estimation (in MB)
    model_memory = estimated_params * 4 / (1024 * 1024)  # 4 bytes per float32
    data_memory = batch_size * input_dim * resolution * resolution * 4 / (1024 * 1024)
    gradient_memory = model_memory  # Roughly same as model
    
    total_memory = model_memory + data_memory + gradient_memory
    
    requirements = {
        'estimated_parameters': estimated_params,
        'model_memory_mb': model_memory,
        'data_memory_mb': data_memory,
        'total_memory_mb': total_memory,
        'warnings': []
    }
    
    # Generate warnings
    if estimated_params > 10_000_000:  # 10M parameters
        requirements['warnings'].append("Large model (>10M parameters) may require significant GPU memory")
    
    if total_memory > 4000:  # 4GB
        requirements['warnings'].append("High memory usage (>4GB) estimated - consider reducing batch size or model size")
    
    if batch_size * resolution * resolution > 100_000:
        requirements['warnings'].append("Large batch*resolution may cause memory issues")
    
    # Log requirements
    logger.info(f"Estimated parameters: {estimated_params:,}")
    logger.info(f"Estimated memory usage: {total_memory:.1f} MB")
    
    for warning in requirements['warnings']:
        logger.warning(warning)
    
    return requirements


def validate_gpu_compatibility(model_config: Dict[str, Any]) -> Dict[str, Any]:
    """Check GPU compatibility and provide recommendations."""
    
    gpu_info = {
        'cuda_available': torch.cuda.is_available(),
        'device_count': 0,
        'devices': [],
        'recommendations': []
    }
    
    if torch.cuda.is_available():
        gpu_info['device_count'] = torch.cuda.device_count()
        
        for i in range(gpu_info['device_count']):
            props = torch.cuda.get_device_properties(i)
            device_info = {
                'device_id': i,
                'name': props.name,
                'memory_total': props.total_memory,
                'compute_capability': (props.major, props.minor)
            }
            gpu_info['devices'].append(device_info)
            
            # Memory recommendations
            memory_gb = props.total_memory / (1024**3)
            if memory_gb < 4:
                gpu_info['recommendations'].append(
                    f"GPU {i} has limited memory ({memory_gb:.1f}GB) - use small batch sizes"
                )
            elif memory_gb >= 16:
                gpu_info['recommendations'].append(
                    f"GPU {i} has ample memory ({memory_gb:.1f}GB) - can use larger models/batches"
                )
    else:
        gpu_info['recommendations'].append("No GPU available - training will be slow on CPU")
        gpu_info['recommendations'].append("Consider using smaller models and batch sizes for CPU")
    
    return gpu_info