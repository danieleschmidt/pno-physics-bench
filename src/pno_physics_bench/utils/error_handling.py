"""Enhanced error handling and custom exceptions."""

import torch
import numpy as np
from typing import Any, Optional, Dict, Union
import logging
import traceback
from pathlib import Path


logger = logging.getLogger(__name__)


class PNOError(Exception):
    """Base exception for PNO Physics Bench."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}
        
        # Log the error
        logger.error(f"PNOError: {message}")
        if self.details:
            logger.error(f"Details: {self.details}")


class ModelConfigError(PNOError):
    """Exception for model configuration issues."""
    pass


class DataError(PNOError):
    """Exception for data-related issues."""
    pass


class TrainingError(PNOError):
    """Exception for training-related issues."""
    pass


class UncertaintyError(PNOError):
    """Exception for uncertainty quantification issues."""
    pass


def safe_tensor_operation(func, *args, **kwargs):
    """Safely execute tensor operations with error handling."""
    try:
        return func(*args, **kwargs)
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            logger.error("GPU out of memory. Try reducing batch size or model size.")
            raise PNOError(
                "GPU memory error during tensor operation",
                details={
                    "original_error": str(e),
                    "suggestion": "Reduce batch size or use CPU"
                }
            )
        elif "size mismatch" in str(e).lower():
            logger.error(f"Tensor size mismatch: {e}")
            raise PNOError(
                "Tensor dimension mismatch",
                details={
                    "original_error": str(e),
                    "args_shapes": [getattr(arg, 'shape', 'N/A') for arg in args if hasattr(arg, 'shape')]
                }
            )
        else:
            raise PNOError(f"Tensor operation failed: {e}")


def validate_tensor(
    tensor: torch.Tensor,
    name: str,
    expected_shape: Optional[tuple] = None,
    expected_dtype: Optional[torch.dtype] = None,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
    allow_nan: bool = False,
    allow_inf: bool = False
) -> None:
    """Comprehensive tensor validation."""
    
    if not isinstance(tensor, torch.Tensor):
        raise DataError(f"{name} must be a torch.Tensor, got {type(tensor)}")
    
    # Shape validation
    if expected_shape is not None:
        if tensor.shape != expected_shape:
            raise DataError(
                f"{name} shape mismatch",
                details={
                    "expected": expected_shape,
                    "actual": tensor.shape
                }
            )
    
    # Dtype validation
    if expected_dtype is not None:
        if tensor.dtype != expected_dtype:
            raise DataError(
                f"{name} dtype mismatch",
                details={
                    "expected": expected_dtype,
                    "actual": tensor.dtype
                }
            )
    
    # NaN validation
    if not allow_nan and torch.isnan(tensor).any():
        nan_count = torch.isnan(tensor).sum().item()
        raise DataError(
            f"{name} contains NaN values",
            details={"nan_count": nan_count}
        )
    
    # Inf validation
    if not allow_inf and torch.isinf(tensor).any():
        inf_count = torch.isinf(tensor).sum().item()
        raise DataError(
            f"{name} contains infinite values",
            details={"inf_count": inf_count}
        )
    
    # Value range validation
    if min_val is not None:
        if tensor.min().item() < min_val:
            raise DataError(
                f"{name} contains values below minimum",
                details={
                    "min_allowed": min_val,
                    "actual_min": tensor.min().item()
                }
            )
    
    if max_val is not None:
        if tensor.max().item() > max_val:
            raise DataError(
                f"{name} contains values above maximum",
                details={
                    "max_allowed": max_val,
                    "actual_max": tensor.max().item()
                }
            )


def handle_training_errors(func):
    """Decorator for handling training-related errors."""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                raise TrainingError(
                    "Training failed due to memory constraints",
                    details={
                        "suggestion": "Reduce batch size, model size, or use gradient accumulation",
                        "original_error": str(e)
                    }
                )
            elif "backward" in str(e).lower():
                raise TrainingError(
                    "Gradient computation failed",
                    details={
                        "suggestion": "Check for NaN/Inf in loss or use gradient clipping",
                        "original_error": str(e)
                    }
                )
            else:
                raise TrainingError(f"Training error: {e}")
        except ValueError as e:
            raise TrainingError(
                "Training configuration error",
                details={
                    "original_error": str(e),
                    "suggestion": "Check model configuration and data shapes"
                }
            )
    return wrapper


def safe_checkpoint_save(checkpoint: Dict[str, Any], filepath: str) -> bool:
    """Safely save checkpoint with backup and validation."""
    filepath = Path(filepath)
    backup_path = filepath.with_suffix(filepath.suffix + '.backup')
    
    try:
        # Create backup if file exists
        if filepath.exists():
            import shutil
            shutil.copy2(filepath, backup_path)
        
        # Save checkpoint
        torch.save(checkpoint, filepath)
        
        # Validate saved checkpoint
        try:
            loaded = torch.load(filepath, map_location='cpu')
            required_keys = ['model_state_dict', 'epoch']
            for key in required_keys:
                if key not in loaded:
                    raise ValueError(f"Missing key: {key}")
        except Exception as e:
            # Restore backup if validation fails
            if backup_path.exists():
                import shutil
                shutil.copy2(backup_path, filepath)
            raise PNOError(f"Checkpoint validation failed: {e}")
        
        # Clean up backup
        if backup_path.exists():
            backup_path.unlink()
        
        logger.info(f"Checkpoint saved successfully: {filepath}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to save checkpoint: {e}")
        raise PNOError(f"Checkpoint save failed: {e}")


def validate_uncertainty_outputs(
    mean: torch.Tensor,
    uncertainty: torch.Tensor,
    targets: Optional[torch.Tensor] = None
) -> None:
    """Validate uncertainty prediction outputs."""
    
    # Basic tensor validation
    validate_tensor(mean, "prediction_mean", allow_nan=False, allow_inf=False)
    validate_tensor(uncertainty, "prediction_uncertainty", min_val=0.0, allow_nan=False, allow_inf=False)
    
    # Shape consistency
    if mean.shape != uncertainty.shape:
        raise UncertaintyError(
            "Mean and uncertainty shape mismatch",
            details={
                "mean_shape": mean.shape,
                "uncertainty_shape": uncertainty.shape
            }
        )
    
    # Target compatibility
    if targets is not None:
        validate_tensor(targets, "targets", allow_nan=False, allow_inf=False)
        if mean.shape != targets.shape:
            raise UncertaintyError(
                "Prediction and target shape mismatch",  
                details={
                    "prediction_shape": mean.shape,
                    "target_shape": targets.shape
                }
            )
    
    # Uncertainty quality checks
    if uncertainty.mean().item() == 0.0:
        logger.warning("All uncertainties are zero - model may be overconfident")
    
    if uncertainty.std().item() < 1e-6:
        logger.warning("Very low uncertainty variance - predictions may be poorly calibrated")


class ErrorRecovery:
    """Handles error recovery strategies during training."""
    
    def __init__(self):
        self.error_count = 0
        self.max_errors = 5
        self.recovery_strategies = []
    
    def add_strategy(self, strategy_func):
        """Add a recovery strategy function."""
        self.recovery_strategies.append(strategy_func)
    
    def handle_error(self, error: Exception, context: Dict[str, Any]) -> bool:
        """Try to recover from error using registered strategies."""
        self.error_count += 1
        
        if self.error_count > self.max_errors:
            logger.error(f"Too many errors ({self.error_count}), giving up")
            return False
        
        logger.warning(f"Attempting error recovery (attempt {self.error_count})")
        
        for i, strategy in enumerate(self.recovery_strategies):
            try:
                if strategy(error, context):
                    logger.info(f"Recovery strategy {i} succeeded")
                    return True
            except Exception as e:
                logger.warning(f"Recovery strategy {i} failed: {e}")
        
        logger.error("All recovery strategies failed")
        return False
    
    def reset(self):
        """Reset error counter."""
        self.error_count = 0


def memory_cleanup():
    """Clean up GPU/CPU memory."""
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Force garbage collection
        import gc
        gc.collect()
        
        logger.info("Memory cleanup completed")
    except Exception as e:
        logger.warning(f"Memory cleanup failed: {e}")


def create_error_report(
    error: Exception,
    context: Optional[Dict[str, Any]] = None,
    save_path: Optional[str] = None
) -> Dict[str, Any]:
    """Create detailed error report for debugging."""
    
    report = {
        "error_type": type(error).__name__,
        "error_message": str(error),
        "traceback": traceback.format_exc(),
        "context": context or {},
        "system_info": {
            "pytorch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        }
    }
    
    # Add memory info if available
    try:
        if torch.cuda.is_available():
            report["gpu_memory"] = {
                "allocated": torch.cuda.memory_allocated(),
                "cached": torch.cuda.memory_reserved(),
            }
    except:
        pass
    
    if save_path:
        import json
        with open(save_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        logger.info(f"Error report saved to {save_path}")
    
    return report