"""Advanced Error Handling and Recovery for Probabilistic Neural Operators.

This module implements comprehensive error handling, graceful degradation,
and automatic recovery mechanisms for robust PNO operations.
"""

import torch
import torch.nn as nn
import traceback
import logging
import time
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from enum import Enum
from dataclasses import dataclass
from contextlib import contextmanager
import threading
from collections import deque
import numpy as np


class ErrorSeverity(Enum):
    """Error severity levels for graduated response."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RecoveryStrategy(Enum):
    """Available recovery strategies."""
    RETRY = "retry"
    FALLBACK = "fallback"
    DEGRADED_MODE = "degraded_mode"
    ABORT = "abort"
    AUTO_REPAIR = "auto_repair"


@dataclass
class ErrorContext:
    """Context information for error handling."""
    error_type: str
    severity: ErrorSeverity
    timestamp: float
    stack_trace: str
    model_state: Dict[str, Any]
    input_characteristics: Dict[str, Any]
    recovery_attempts: int = 0
    max_recovery_attempts: int = 3


class PNOException(Exception):
    """Base exception class for PNO-specific errors."""
    
    def __init__(
        self,
        message: str,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        recoverable: bool = True,
        context: Optional[Dict] = None
    ):
        super().__init__(message)
        self.severity = severity
        self.recoverable = recoverable
        self.context = context or {}
        self.timestamp = time.time()


class NumericalInstabilityError(PNOException):
    """Error for numerical instability issues."""
    pass


class ConvergenceError(PNOException):
    """Error when training or inference fails to converge."""
    pass


class UncertaintyCalibrationError(PNOException):
    """Error in uncertainty calibration process."""
    pass


class MemoryExhaustionError(PNOException):
    """Error when system runs out of memory."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.severity = ErrorSeverity.HIGH


class ErrorRecoverySystem:
    """Comprehensive error recovery and handling system."""
    
    def __init__(
        self,
        max_recovery_attempts: int = 3,
        fallback_models: Optional[List[nn.Module]] = None,
        enable_auto_repair: bool = True
    ):
        self.max_recovery_attempts = max_recovery_attempts
        self.fallback_models = fallback_models or []
        self.enable_auto_repair = enable_auto_repair
        
        # Error tracking
        self.error_history = deque(maxlen=1000)
        self.recovery_statistics = {}
        self.model_health_metrics = {}
        
        # Recovery strategies
        self.recovery_strategies = {
            NumericalInstabilityError: self._handle_numerical_instability,
            ConvergenceError: self._handle_convergence_error,
            UncertaintyCalibrationError: self._handle_calibration_error,
            MemoryExhaustionError: self._handle_memory_exhaustion,
            torch.cuda.OutOfMemoryError: self._handle_cuda_oom,
            RuntimeError: self._handle_runtime_error
        }
        
        # Auto-repair mechanisms
        self.repair_mechanisms = {
            'gradient_explosion': self._repair_gradient_explosion,
            'nan_values': self._repair_nan_values,
            'infinite_values': self._repair_infinite_values,
            'memory_leak': self._repair_memory_leak
        }
        
        self.logger = logging.getLogger(__name__)
    
    def handle_error(
        self,
        error: Exception,
        model: nn.Module,
        inputs: torch.Tensor,
        context: Optional[Dict] = None
    ) -> Tuple[bool, Any]:
        """Main error handling entry point."""
        
        # Create error context
        error_ctx = self._create_error_context(error, model, inputs, context)
        self.error_history.append(error_ctx)
        
        # Log error
        self._log_error(error_ctx)
        
        # Determine recovery strategy
        recovery_success = False
        result = None
        
        try:
            # Get appropriate handler
            handler = self.recovery_strategies.get(type(error), self._default_error_handler)
            
            # Attempt recovery
            recovery_success, result = handler(error, model, inputs, error_ctx)
            
            # Update statistics
            self._update_recovery_statistics(error_ctx, recovery_success)
            
        except Exception as recovery_error:
            self.logger.error(f"Recovery failed: {recovery_error}")
            recovery_success = False
        
        return recovery_success, result
    
    def _create_error_context(
        self,
        error: Exception,
        model: nn.Module,
        inputs: torch.Tensor,
        context: Optional[Dict] = None
    ) -> ErrorContext:
        """Create comprehensive error context."""
        
        # Analyze input characteristics
        input_characteristics = {
            'shape': list(inputs.shape),
            'dtype': str(inputs.dtype),
            'device': str(inputs.device),
            'has_nan': torch.isnan(inputs).any().item(),
            'has_inf': torch.isinf(inputs).any().item(),
            'min_val': inputs.min().item() if inputs.numel() > 0 else 0,
            'max_val': inputs.max().item() if inputs.numel() > 0 else 0,
            'mean': inputs.mean().item() if inputs.numel() > 0 else 0,
            'std': inputs.std().item() if inputs.numel() > 0 else 0
        }
        
        # Analyze model state
        model_state = {}
        try:
            for name, param in model.named_parameters():
                if param is not None:
                    model_state[name] = {
                        'has_nan': torch.isnan(param).any().item(),
                        'has_inf': torch.isinf(param).any().item(),
                        'grad_norm': param.grad.norm().item() if param.grad is not None else 0
                    }
        except Exception:
            model_state['analysis_failed'] = True
        
        # Determine severity
        severity = self._determine_error_severity(error, input_characteristics, model_state)
        
        return ErrorContext(
            error_type=type(error).__name__,
            severity=severity,
            timestamp=time.time(),
            stack_trace=traceback.format_exc(),
            model_state=model_state,
            input_characteristics=input_characteristics
        )
    
    def _determine_error_severity(
        self,
        error: Exception,
        input_chars: Dict,
        model_state: Dict
    ) -> ErrorSeverity:
        """Determine error severity based on error type and context."""
        
        # Critical errors
        if isinstance(error, (MemoryExhaustionError, torch.cuda.OutOfMemoryError)):
            return ErrorSeverity.CRITICAL
        
        # High severity conditions
        if (input_chars.get('has_nan', False) or 
            input_chars.get('has_inf', False) or
            any(state.get('has_nan', False) for state in model_state.values() if isinstance(state, dict))):
            return ErrorSeverity.HIGH
        
        # Medium severity for convergence issues
        if isinstance(error, (ConvergenceError, NumericalInstabilityError)):
            return ErrorSeverity.MEDIUM
        
        # Default to low severity
        return ErrorSeverity.LOW
    
    def _handle_numerical_instability(
        self,
        error: NumericalInstabilityError,
        model: nn.Module,
        inputs: torch.Tensor,
        context: ErrorContext
    ) -> Tuple[bool, Any]:
        """Handle numerical instability errors."""
        
        self.logger.info("Attempting to recover from numerical instability")
        
        # Strategy 1: Gradient clipping and parameter regularization
        if context.recovery_attempts == 0:
            try:
                # Apply gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                # Regularize parameters with extreme values
                with torch.no_grad():
                    for param in model.parameters():
                        if param is not None:
                            # Clamp extreme values
                            param.data = torch.clamp(param.data, -10, 10)
                            
                            # Add small regularization
                            param.data *= 0.999
                
                # Retry forward pass
                result = model(inputs)
                return True, result
                
            except Exception:
                pass
        
        # Strategy 2: Reduce precision and scale inputs
        elif context.recovery_attempts == 1:
            try:
                # Scale down inputs
                scaled_inputs = inputs * 0.1
                
                # Use mixed precision
                with torch.cuda.amp.autocast():
                    result = model(scaled_inputs)
                
                # Scale back result
                if isinstance(result, tuple):
                    result = tuple(r * 10 for r in result)
                else:
                    result = result * 10
                
                return True, result
                
            except Exception:
                pass
        
        # Strategy 3: Fallback to simpler computation
        elif context.recovery_attempts == 2:
            return self._fallback_computation(model, inputs, context)
        
        return False, None
    
    def _handle_convergence_error(
        self,
        error: ConvergenceError,
        model: nn.Module,
        inputs: torch.Tensor,
        context: ErrorContext
    ) -> Tuple[bool, Any]:
        """Handle convergence errors."""
        
        # Strategy 1: Reduce learning rate and retry
        if context.recovery_attempts == 0:
            try:
                # Apply learning rate decay to all optimizable parameters
                with torch.no_grad():
                    for param in model.parameters():
                        if param.grad is not None:
                            param.grad *= 0.1
                
                result = model(inputs)
                return True, result
                
            except Exception:
                pass
        
        # Strategy 2: Reset problematic layers
        elif context.recovery_attempts == 1:
            try:
                # Identify and reset layers with extreme gradients
                self._reset_problematic_layers(model, context)
                result = model(inputs)
                return True, result
                
            except Exception:
                pass
        
        # Strategy 3: Use ensemble fallback
        return self._ensemble_fallback(inputs)
    
    def _handle_calibration_error(
        self,
        error: UncertaintyCalibrationError,
        model: nn.Module,
        inputs: torch.Tensor,
        context: ErrorContext
    ) -> Tuple[bool, Any]:
        """Handle uncertainty calibration errors."""
        
        # Strategy 1: Use default uncertainty estimates
        try:
            # Get basic prediction
            with torch.no_grad():
                if hasattr(model, 'forward_mean'):
                    predictions = model.forward_mean(inputs)
                else:
                    predictions = model(inputs)
                    if isinstance(predictions, tuple):
                        predictions = predictions[0]
                
                # Generate default uncertainties
                default_uncertainty = torch.ones_like(predictions) * 0.1
                
                return True, (predictions, default_uncertainty)
                
        except Exception:
            pass
        
        # Strategy 2: Fallback to deterministic prediction
        try:
            with torch.no_grad():
                result = model(inputs)
                if isinstance(result, tuple):
                    return True, (result[0], torch.zeros_like(result[0]))
                else:
                    return True, (result, torch.zeros_like(result))
        except Exception:
            pass
        
        return False, None
    
    def _handle_memory_exhaustion(
        self,
        error: MemoryExhaustionError,
        model: nn.Module,
        inputs: torch.Tensor,
        context: ErrorContext
    ) -> Tuple[bool, Any]:
        """Handle memory exhaustion errors."""
        
        # Strategy 1: Batch size reduction
        if context.recovery_attempts == 0:
            try:
                # Process in smaller chunks
                batch_size = inputs.size(0)
                chunk_size = max(1, batch_size // 4)
                
                results = []
                for i in range(0, batch_size, chunk_size):
                    chunk = inputs[i:i+chunk_size]
                    with torch.no_grad():
                        chunk_result = model(chunk)
                    results.append(chunk_result)
                
                # Combine results
                if isinstance(results[0], tuple):
                    combined_result = tuple(torch.cat([r[i] for r in results], dim=0) for i in range(len(results[0])))
                else:
                    combined_result = torch.cat(results, dim=0)
                
                return True, combined_result
                
            except Exception:
                pass
        
        # Strategy 2: Move to CPU
        elif context.recovery_attempts == 1:
            try:
                # Clear GPU memory
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Move to CPU
                model_cpu = model.cpu()
                inputs_cpu = inputs.cpu()
                
                with torch.no_grad():
                    result = model_cpu(inputs_cpu)
                
                # Move result back to original device
                if inputs.is_cuda:
                    if isinstance(result, tuple):
                        result = tuple(r.cuda() for r in result)
                    else:
                        result = result.cuda()
                
                return True, result
                
            except Exception:
                pass
        
        return False, None
    
    def _handle_cuda_oom(
        self,
        error: torch.cuda.OutOfMemoryError,
        model: nn.Module,
        inputs: torch.Tensor,
        context: ErrorContext
    ) -> Tuple[bool, Any]:
        """Handle CUDA out of memory errors."""
        
        # Clear cache and retry
        torch.cuda.empty_cache()
        
        # Use memory exhaustion handler
        return self._handle_memory_exhaustion(
            MemoryExhaustionError("CUDA OOM", severity=ErrorSeverity.HIGH),
            model, inputs, context
        )
    
    def _handle_runtime_error(
        self,
        error: RuntimeError,
        model: nn.Module,
        inputs: torch.Tensor,
        context: ErrorContext
    ) -> Tuple[bool, Any]:
        """Handle generic runtime errors."""
        
        error_msg = str(error).lower()
        
        # Handle specific runtime error patterns
        if 'nan' in error_msg or 'inf' in error_msg:
            if self.enable_auto_repair:
                self._auto_repair_nan_inf(model, inputs)
                try:
                    result = model(inputs)
                    return True, result
                except Exception:
                    pass
        
        # Fallback to simpler computation
        return self._fallback_computation(model, inputs, context)
    
    def _default_error_handler(
        self,
        error: Exception,
        model: nn.Module,
        inputs: torch.Tensor,
        context: ErrorContext
    ) -> Tuple[bool, Any]:
        """Default error handler for unknown error types."""
        
        # Try fallback computation
        return self._fallback_computation(model, inputs, context)
    
    def _fallback_computation(
        self,
        model: nn.Module,
        inputs: torch.Tensor,
        context: ErrorContext
    ) -> Tuple[bool, Any]:
        """Fallback to simplified computation."""
        
        # Try fallback models
        for fallback_model in self.fallback_models:
            try:
                with torch.no_grad():
                    result = fallback_model(inputs)
                return True, result
            except Exception:
                continue
        
        # Last resort: return input-based estimate
        try:
            # Simple linear transformation as fallback
            fallback_result = torch.mean(inputs, dim=-1, keepdim=True)
            fallback_uncertainty = torch.ones_like(fallback_result) * 0.5
            
            return True, (fallback_result, fallback_uncertainty)
            
        except Exception:
            return False, None
    
    def _ensemble_fallback(self, inputs: torch.Tensor) -> Tuple[bool, Any]:
        """Use ensemble of fallback models."""
        
        if not self.fallback_models:
            return False, None
        
        try:
            predictions = []
            for model in self.fallback_models:
                with torch.no_grad():
                    pred = model(inputs)
                    if isinstance(pred, tuple):
                        pred = pred[0]
                    predictions.append(pred)
            
            if predictions:
                ensemble_pred = torch.stack(predictions).mean(dim=0)
                ensemble_uncertainty = torch.stack(predictions).std(dim=0)
                return True, (ensemble_pred, ensemble_uncertainty)
                
        except Exception:
            pass
        
        return False, None
    
    def _auto_repair_nan_inf(self, model: nn.Module, inputs: torch.Tensor):
        """Automatically repair NaN and Inf values."""
        
        # Repair model parameters
        with torch.no_grad():
            for param in model.parameters():
                if param is not None:
                    # Replace NaN with zeros
                    param.data = torch.where(torch.isnan(param.data), torch.zeros_like(param.data), param.data)
                    
                    # Replace Inf with large but finite values
                    param.data = torch.where(torch.isinf(param.data), torch.sign(param.data) * 10.0, param.data)
        
        # Repair inputs (if needed)
        inputs.data = torch.where(torch.isnan(inputs), torch.zeros_like(inputs), inputs)
        inputs.data = torch.where(torch.isinf(inputs), torch.sign(inputs) * 10.0, inputs)
    
    def _reset_problematic_layers(self, model: nn.Module, context: ErrorContext):
        """Reset layers with problematic gradients or parameters."""
        
        with torch.no_grad():
            for name, param in model.named_parameters():
                # Check for problematic gradients
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    if grad_norm > 100 or torch.isnan(param.grad).any():
                        # Reset this parameter
                        param.data.normal_(0, 0.02)
                        param.grad.zero_()
    
    def _log_error(self, context: ErrorContext):
        """Log error with context information."""
        
        log_level = {
            ErrorSeverity.LOW: logging.INFO,
            ErrorSeverity.MEDIUM: logging.WARNING,
            ErrorSeverity.HIGH: logging.ERROR,
            ErrorSeverity.CRITICAL: logging.CRITICAL
        }[context.severity]
        
        self.logger.log(
            log_level,
            f"PNO Error [{context.error_type}] - Severity: {context.severity.value} - "
            f"Attempt: {context.recovery_attempts}/{self.max_recovery_attempts}"
        )
    
    def _update_recovery_statistics(self, context: ErrorContext, success: bool):
        """Update recovery statistics for monitoring."""
        
        error_type = context.error_type
        if error_type not in self.recovery_statistics:
            self.recovery_statistics[error_type] = {
                'total_attempts': 0,
                'successful_recoveries': 0,
                'failed_recoveries': 0
            }
        
        self.recovery_statistics[error_type]['total_attempts'] += 1
        if success:
            self.recovery_statistics[error_type]['successful_recoveries'] += 1
        else:
            self.recovery_statistics[error_type]['failed_recoveries'] += 1
    
    def get_health_report(self) -> Dict[str, Any]:
        """Generate comprehensive health report."""
        
        recent_errors = [err for err in self.error_history if time.time() - err.timestamp < 3600]  # Last hour
        
        return {
            'total_errors': len(self.error_history),
            'recent_errors': len(recent_errors),
            'error_types': list(set(err.error_type for err in self.error_history)),
            'recovery_statistics': self.recovery_statistics,
            'error_rate_per_hour': len(recent_errors),
            'most_common_errors': self._get_most_common_errors(),
            'average_recovery_success_rate': self._calculate_avg_recovery_rate()
        }
    
    def _get_most_common_errors(self) -> List[Tuple[str, int]]:
        """Get most common error types."""
        
        error_counts = {}
        for err in self.error_history:
            error_counts[err.error_type] = error_counts.get(err.error_type, 0) + 1
        
        return sorted(error_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    
    def _calculate_avg_recovery_rate(self) -> float:
        """Calculate average recovery success rate."""
        
        if not self.recovery_statistics:
            return 0.0
        
        total_attempts = sum(stats['total_attempts'] for stats in self.recovery_statistics.values())
        total_successes = sum(stats['successful_recoveries'] for stats in self.recovery_statistics.values())
        
        return total_successes / total_attempts if total_attempts > 0 else 0.0


@contextmanager
def robust_pno_execution(
    model: nn.Module,
    recovery_system: Optional[ErrorRecoverySystem] = None,
    max_retries: int = 3
):
    """Context manager for robust PNO execution."""
    
    if recovery_system is None:
        recovery_system = ErrorRecoverySystem()
    
    retry_count = 0
    
    while retry_count <= max_retries:
        try:
            yield recovery_system
            break  # Success, exit the retry loop
            
        except Exception as e:
            retry_count += 1
            
            if retry_count > max_retries:
                # Final attempt failed
                raise PNOException(
                    f"Max retries exceeded. Last error: {str(e)}",
                    severity=ErrorSeverity.CRITICAL,
                    recoverable=False
                )
            
            # Log retry attempt
            logging.warning(f"Execution failed, attempting retry {retry_count}/{max_retries}: {str(e)}")
            time.sleep(0.1 * retry_count)  # Exponential backoff


class RobustPNOWrapper(nn.Module):
    """Wrapper that adds robustness features to any PNO model."""
    
    def __init__(
        self,
        base_model: nn.Module,
        enable_error_recovery: bool = True,
        fallback_models: Optional[List[nn.Module]] = None
    ):
        super().__init__()
        
        self.base_model = base_model
        self.enable_error_recovery = enable_error_recovery
        
        if enable_error_recovery:
            self.recovery_system = ErrorRecoverySystem(fallback_models=fallback_models)
        else:
            self.recovery_system = None
    
    def forward(self, *args, **kwargs):
        """Robust forward pass with error handling."""
        
        if not self.enable_error_recovery:
            return self.base_model(*args, **kwargs)
        
        try:
            return self.base_model(*args, **kwargs)
            
        except Exception as e:
            # Attempt error recovery
            success, result = self.recovery_system.handle_error(
                e, self.base_model, args[0] if args else None
            )
            
            if success:
                return result
            else:
                # Re-raise if recovery failed
                raise e
    
    def predict_with_uncertainty(self, *args, **kwargs):
        """Robust uncertainty prediction."""
        
        try:
            if hasattr(self.base_model, 'predict_with_uncertainty'):
                return self.base_model.predict_with_uncertainty(*args, **kwargs)
            else:
                # Fallback to regular forward pass
                pred = self.forward(*args, **kwargs)
                if isinstance(pred, tuple) and len(pred) == 2:
                    return pred
                else:
                    return pred, torch.ones_like(pred) * 0.1
                    
        except Exception as e:
            if self.enable_error_recovery:
                success, result = self.recovery_system.handle_error(
                    e, self.base_model, args[0] if args else None
                )
                
                if success:
                    return result
            
            raise e
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get model health status."""
        
        if self.recovery_system:
            return self.recovery_system.get_health_report()
        else:
            return {'status': 'no_monitoring', 'errors': 0}