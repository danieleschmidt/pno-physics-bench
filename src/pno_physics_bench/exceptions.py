"""Custom exceptions for PNO Physics Bench."""

from typing import Optional, Any, Dict


class PNOError(Exception):
    """Base exception class for PNO Physics Bench."""
    
    def __init__(self, message: str, error_code: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}
    
    def __str__(self) -> str:
        if self.error_code:
            return f"[{self.error_code}] {self.message}"
        return self.message


class ModelError(PNOError):
    """Exceptions related to model operations."""
    pass


class DataError(PNOError):
    """Exceptions related to data operations."""
    pass


class TrainingError(PNOError):
    """Exceptions related to training operations."""
    pass


class ConfigurationError(PNOError):
    """Exceptions related to configuration issues."""
    pass


class ValidationError(PNOError):
    """Exceptions related to input validation."""
    pass


class UncertaintyError(PNOError):
    """Exceptions related to uncertainty computation."""
    pass


class BenchmarkError(PNOError):
    """Exceptions related to benchmarking operations."""
    pass


class ResourceError(PNOError):
    """Exceptions related to resource management."""
    pass


# Specific error types
class ModelArchitectureError(ModelError):
    """Error in model architecture definition."""
    pass


class ModelLoadError(ModelError):
    """Error loading model from checkpoint."""
    pass


class DataGenerationError(DataError):
    """Error generating PDE data."""
    pass


class DataLoadError(DataError):
    """Error loading data from file."""
    pass


class DataValidationError(DataError):
    """Error in data validation."""
    pass


class TrainingSetupError(TrainingError):
    """Error setting up training configuration."""
    pass


class TrainingFailureError(TrainingError):
    """Error during training execution."""
    pass


class OptimizationError(TrainingError):
    """Error in optimization process."""
    pass


class InvalidConfigError(ConfigurationError):
    """Invalid configuration parameters."""
    pass


class MissingConfigError(ConfigurationError):
    """Missing required configuration."""
    pass


class InvalidParameterError(ValidationError):
    """Invalid parameter value."""
    pass


class DimensionMismatchError(ValidationError):
    """Dimension mismatch in tensor operations."""
    pass


class UncertaintyComputationError(UncertaintyError):
    """Error computing uncertainty estimates."""
    pass


class CalibrationError(UncertaintyError):
    """Error in uncertainty calibration."""
    pass


class BenchmarkSetupError(BenchmarkError):
    """Error setting up benchmark."""
    pass


class BenchmarkExecutionError(BenchmarkError):
    """Error during benchmark execution."""
    pass


class InsufficientResourcesError(ResourceError):
    """Insufficient computational resources."""
    pass


class MemoryError(ResourceError):
    """Memory allocation error."""
    pass


class DeviceError(ResourceError):
    """Device-related error."""
    pass