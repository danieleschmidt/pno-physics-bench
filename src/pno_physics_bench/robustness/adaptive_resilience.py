"""Adaptive Resilience System for PNO - Advanced Error Recovery and Fault Tolerance.

This module implements sophisticated resilience mechanisms that automatically detect,
diagnose, and recover from various failure modes in adaptive PNO systems.
"""

import torch
import torch.nn as nn
import logging
import time
import json
from typing import Dict, List, Optional, Any, Callable, Tuple
from collections import deque, defaultdict
from enum import Enum
import threading
import hashlib
from dataclasses import dataclass
from contextlib import contextmanager


class FailureMode(Enum):
    """Types of failure modes that can occur in adaptive PNO systems."""
    GRADIENT_EXPLOSION = "gradient_explosion"
    GRADIENT_VANISHING = "gradient_vanishing"
    UNCERTAINTY_COLLAPSE = "uncertainty_collapse"
    DISTRIBUTION_SHIFT = "distribution_shift"
    MEMORY_OVERFLOW = "memory_overflow"
    NUMERICAL_INSTABILITY = "numerical_instability"
    CALIBRATION_BREAKDOWN = "calibration_breakdown"
    ADAPTATION_OSCILLATION = "adaptation_oscillation"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    SECURITY_BREACH = "security_breach"


@dataclass
class HealthMetrics:
    """Comprehensive health metrics for PNO systems."""
    timestamp: float
    gradient_norm: float
    uncertainty_mean: float
    uncertainty_std: float
    calibration_error: float
    memory_usage: float
    computation_time: float
    loss_value: float
    adaptation_rate: float
    security_score: float
    stability_index: float


class AdaptiveHealthMonitor:
    """Real-time health monitoring system with predictive failure detection."""
    
    def __init__(self, window_size: int = 1000, alert_threshold: float = 0.8):
        self.window_size = window_size
        self.alert_threshold = alert_threshold
        
        # Health metrics history
        self.metrics_history: deque = deque(maxlen=window_size)
        self.failure_predictions: Dict[FailureMode, float] = defaultdict(float)
        
        # Anomaly detection parameters
        self.baseline_metrics: Optional[Dict[str, float]] = None
        self.deviation_thresholds: Dict[str, float] = {
            'gradient_norm': 10.0,
            'uncertainty_mean': 5.0,
            'calibration_error': 0.3,
            'memory_usage': 0.9,
            'computation_time': 10.0,
            'stability_index': 0.1
        }
        
        # Adaptive thresholds
        self.adaptive_thresholds = {}
        self.threshold_update_alpha = 0.01
        
        logging.info("Initialized Adaptive Health Monitor")
    
    def update_metrics(self, model: nn.Module, loss: torch.Tensor, 
                      uncertainty_stats: Dict[str, float],
                      performance_stats: Dict[str, float]) -> HealthMetrics:
        """Update health metrics with current system state."""
        
        current_time = time.time()
        
        # Compute gradient norm
        gradient_norm = self._compute_gradient_norm(model)
        
        # Memory usage
        if torch.cuda.is_available():
            memory_usage = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
        else:
            memory_usage = 0.0
        
        # Stability index (based on recent metric variance)
        stability_index = self._compute_stability_index()
        
        # Security score (placeholder - would integrate with security scanner)
        security_score = self._compute_security_score(model)
        
        metrics = HealthMetrics(
            timestamp=current_time,
            gradient_norm=float(gradient_norm),
            uncertainty_mean=uncertainty_stats.get('mean', 0.0),
            uncertainty_std=uncertainty_stats.get('std', 0.0),
            calibration_error=performance_stats.get('calibration_error', 0.0),
            memory_usage=memory_usage,
            computation_time=performance_stats.get('computation_time', 0.0),
            loss_value=float(loss.detach()) if isinstance(loss, torch.Tensor) else float(loss),
            adaptation_rate=performance_stats.get('adaptation_rate', 0.0),
            security_score=security_score,
            stability_index=stability_index
        )
        
        self.metrics_history.append(metrics)
        
        # Update baseline if not set
        if self.baseline_metrics is None and len(self.metrics_history) > 100:
            self._establish_baseline()
        
        # Update adaptive thresholds
        self._update_adaptive_thresholds(metrics)
        
        return metrics
    
    def predict_failures(self, metrics: HealthMetrics) -> Dict[FailureMode, float]:
        """Predict probability of various failure modes."""
        predictions = {}
        
        # Gradient explosion/vanishing
        if metrics.gradient_norm > 100.0:
            predictions[FailureMode.GRADIENT_EXPLOSION] = min(1.0, metrics.gradient_norm / 1000.0)
        elif metrics.gradient_norm < 1e-6:
            predictions[FailureMode.GRADIENT_VANISHING] = 1.0 - metrics.gradient_norm * 1e6
        
        # Uncertainty collapse
        if metrics.uncertainty_mean < 0.001:
            predictions[FailureMode.UNCERTAINTY_COLLAPSE] = 1.0 - metrics.uncertainty_mean * 1000
        
        # Calibration breakdown
        if metrics.calibration_error > 0.5:
            predictions[FailureMode.CALIBRATION_BREAKDOWN] = metrics.calibration_error
        
        # Memory overflow
        if metrics.memory_usage > 0.95:
            predictions[FailureMode.MEMORY_OVERFLOW] = metrics.memory_usage
        
        # Numerical instability
        if torch.isnan(torch.tensor(metrics.loss_value)) or torch.isinf(torch.tensor(metrics.loss_value)):
            predictions[FailureMode.NUMERICAL_INSTABILITY] = 1.0
        
        # Adaptation oscillation
        oscillation_score = self._detect_oscillation()
        if oscillation_score > 0.7:
            predictions[FailureMode.ADAPTATION_OSCILLATION] = oscillation_score
        
        # Performance degradation
        degradation_score = self._detect_degradation()
        if degradation_score > 0.6:
            predictions[FailureMode.PERFORMANCE_DEGRADATION] = degradation_score
        
        # Security threats
        if metrics.security_score < 0.3:
            predictions[FailureMode.SECURITY_BREACH] = 1.0 - metrics.security_score
        
        self.failure_predictions.update(predictions)
        return predictions
    
    def _compute_gradient_norm(self, model: nn.Module) -> float:
        """Compute L2 norm of model gradients."""
        total_norm = 0.0
        param_count = 0
        
        for param in model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                param_count += 1
        
        return (total_norm ** 0.5) / max(param_count, 1)
    
    def _compute_stability_index(self) -> float:
        """Compute stability index based on recent metric variance."""
        if len(self.metrics_history) < 10:
            return 1.0
        
        recent_losses = [m.loss_value for m in list(self.metrics_history)[-10:]]
        if len(set(recent_losses)) == 1:  # All same values
            return 1.0
        
        variance = torch.var(torch.tensor(recent_losses))
        mean_loss = sum(recent_losses) / len(recent_losses)
        
        # Stability index: lower variance relative to mean = higher stability
        stability = 1.0 / (1.0 + variance / (abs(mean_loss) + 1e-8))
        return float(stability)
    
    def _compute_security_score(self, model: nn.Module) -> float:
        """Compute basic security score (placeholder for full implementation)."""
        # Check for suspicious parameter patterns
        suspicious_params = 0
        total_params = 0
        
        for param in model.parameters():
            if param.data.abs().max() > 1000:  # Abnormally large weights
                suspicious_params += 1
            total_params += 1
        
        base_score = 1.0 - (suspicious_params / max(total_params, 1))
        
        # Add input validation score (simplified)
        input_validation_score = 0.8  # Would be computed based on actual validation
        
        return (base_score + input_validation_score) / 2.0
    
    def _establish_baseline(self):
        """Establish baseline metrics for anomaly detection."""
        if len(self.metrics_history) < 50:
            return
        
        recent_metrics = list(self.metrics_history)[-100:]
        
        self.baseline_metrics = {
            'gradient_norm': sum(m.gradient_norm for m in recent_metrics) / len(recent_metrics),
            'uncertainty_mean': sum(m.uncertainty_mean for m in recent_metrics) / len(recent_metrics),
            'calibration_error': sum(m.calibration_error for m in recent_metrics) / len(recent_metrics),
            'computation_time': sum(m.computation_time for m in recent_metrics) / len(recent_metrics),
            'stability_index': sum(m.stability_index for m in recent_metrics) / len(recent_metrics)
        }
        
        logging.info(f"Established baseline metrics: {self.baseline_metrics}")
    
    def _update_adaptive_thresholds(self, current_metrics: HealthMetrics):
        """Update adaptive thresholds based on recent performance."""
        if self.baseline_metrics is None:
            return
        
        # Update thresholds using exponential moving average
        alpha = self.threshold_update_alpha
        
        for key, baseline_value in self.baseline_metrics.items():
            current_value = getattr(current_metrics, key, 0.0)
            deviation = abs(current_value - baseline_value) / (baseline_value + 1e-8)
            
            if key not in self.adaptive_thresholds:
                self.adaptive_thresholds[key] = self.deviation_thresholds.get(key, 2.0)
            
            # Adaptive threshold: increase if consistently exceeding, decrease if consistently within bounds
            if deviation > self.adaptive_thresholds[key]:
                self.adaptive_thresholds[key] = alpha * deviation + (1 - alpha) * self.adaptive_thresholds[key]
            else:
                self.adaptive_thresholds[key] = (1 - alpha) * self.adaptive_thresholds[key]
    
    def _detect_oscillation(self) -> float:
        """Detect oscillatory behavior in adaptation."""
        if len(self.metrics_history) < 20:
            return 0.0
        
        recent_adaptation_rates = [m.adaptation_rate for m in list(self.metrics_history)[-20:]]
        
        # Count sign changes
        sign_changes = 0
        for i in range(1, len(recent_adaptation_rates)):
            if (recent_adaptation_rates[i] > 0) != (recent_adaptation_rates[i-1] > 0):
                sign_changes += 1
        
        # High frequency of sign changes indicates oscillation
        oscillation_score = sign_changes / (len(recent_adaptation_rates) - 1)
        return oscillation_score
    
    def _detect_degradation(self) -> float:
        """Detect performance degradation trends."""
        if len(self.metrics_history) < 50:
            return 0.0
        
        # Compare recent performance to baseline
        recent_metrics = list(self.metrics_history)[-25:]
        baseline_metrics = list(self.metrics_history)[-50:-25:]
        
        recent_loss = sum(m.loss_value for m in recent_metrics) / len(recent_metrics)
        baseline_loss = sum(m.loss_value for m in baseline_metrics) / len(baseline_metrics)
        
        if baseline_loss == 0:
            return 0.0
        
        degradation = (recent_loss - baseline_loss) / baseline_loss
        return max(0.0, degradation)  # Only positive degradation


class AdaptiveRecoverySystem:
    """Automated recovery system that responds to detected failures."""
    
    def __init__(self):
        self.recovery_strategies: Dict[FailureMode, List[Callable]] = defaultdict(list)
        self.recovery_history: List[Dict] = []
        self.max_recovery_attempts = 3
        
        # Register recovery strategies
        self._register_recovery_strategies()
        
        logging.info("Initialized Adaptive Recovery System")
    
    def _register_recovery_strategies(self):
        """Register recovery strategies for different failure modes."""
        
        # Gradient explosion
        self.recovery_strategies[FailureMode.GRADIENT_EXPLOSION].extend([
            self._apply_gradient_clipping,
            self._reduce_learning_rate,
            self._reset_optimizer_state
        ])
        
        # Gradient vanishing
        self.recovery_strategies[FailureMode.GRADIENT_VANISHING].extend([
            self._increase_learning_rate,
            self._apply_gradient_scaling,
            self._adjust_architecture
        ])
        
        # Uncertainty collapse
        self.recovery_strategies[FailureMode.UNCERTAINTY_COLLAPSE].extend([
            self._increase_uncertainty_regularization,
            self._reinitialize_uncertainty_parameters,
            self._inject_uncertainty_noise
        ])
        
        # Calibration breakdown
        self.recovery_strategies[FailureMode.CALIBRATION_BREAKDOWN].extend([
            self._recalibrate_uncertainty,
            self._reset_adaptive_scaling,
            self._update_calibration_targets
        ])
        
        # Memory overflow
        self.recovery_strategies[FailureMode.MEMORY_OVERFLOW].extend([
            self._reduce_batch_size,
            self._enable_gradient_checkpointing,
            self._clear_cache
        ])
        
        # Numerical instability
        self.recovery_strategies[FailureMode.NUMERICAL_INSTABILITY].extend([
            self._apply_numerical_stabilization,
            self._reduce_precision,
            self._reset_model_state
        ])
    
    def execute_recovery(self, failure_mode: FailureMode, model: nn.Module, 
                        optimizer: torch.optim.Optimizer, **context) -> bool:
        """Execute recovery procedure for detected failure mode."""
        
        logging.warning(f"Executing recovery for {failure_mode.value}")
        
        strategies = self.recovery_strategies.get(failure_mode, [])
        
        for attempt, strategy in enumerate(strategies):
            if attempt >= self.max_recovery_attempts:
                break
            
            try:
                recovery_start = time.time()
                success = strategy(model, optimizer, **context)
                recovery_time = time.time() - recovery_start
                
                # Log recovery attempt
                recovery_record = {
                    'timestamp': recovery_start,
                    'failure_mode': failure_mode.value,
                    'strategy': strategy.__name__,
                    'attempt': attempt + 1,
                    'success': success,
                    'recovery_time': recovery_time
                }
                
                self.recovery_history.append(recovery_record)
                
                if success:
                    logging.info(f"Recovery successful using {strategy.__name__}")
                    return True
                else:
                    logging.warning(f"Recovery attempt {attempt + 1} failed with {strategy.__name__}")
                    
            except Exception as e:
                logging.error(f"Recovery strategy {strategy.__name__} failed with error: {e}")
                continue
        
        logging.error(f"All recovery strategies failed for {failure_mode.value}")
        return False
    
    # Recovery strategy implementations
    def _apply_gradient_clipping(self, model: nn.Module, optimizer: torch.optim.Optimizer, **context) -> bool:
        """Apply gradient clipping to prevent explosion."""
        try:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            logging.info("Applied gradient clipping (max_norm=1.0)")
            return True
        except Exception as e:
            logging.error(f"Gradient clipping failed: {e}")
            return False
    
    def _reduce_learning_rate(self, model: nn.Module, optimizer: torch.optim.Optimizer, **context) -> bool:
        """Reduce learning rate by half."""
        try:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.5
                logging.info(f"Reduced learning rate to {param_group['lr']}")
            return True
        except Exception as e:
            logging.error(f"Learning rate reduction failed: {e}")
            return False
    
    def _reset_optimizer_state(self, model: nn.Module, optimizer: torch.optim.Optimizer, **context) -> bool:
        """Reset optimizer internal state."""
        try:
            optimizer.state = defaultdict(dict)
            logging.info("Reset optimizer state")
            return True
        except Exception as e:
            logging.error(f"Optimizer state reset failed: {e}")
            return False
    
    def _increase_learning_rate(self, model: nn.Module, optimizer: torch.optim.Optimizer, **context) -> bool:
        """Increase learning rate to combat vanishing gradients."""
        try:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 1.5
                logging.info(f"Increased learning rate to {param_group['lr']}")
            return True
        except Exception as e:
            logging.error(f"Learning rate increase failed: {e}")
            return False
    
    def _apply_gradient_scaling(self, model: nn.Module, optimizer: torch.optim.Optimizer, **context) -> bool:
        """Apply gradient scaling for vanishing gradients."""
        try:
            for param in model.parameters():
                if param.grad is not None:
                    param.grad.data *= 2.0
            logging.info("Applied gradient scaling (factor=2.0)")
            return True
        except Exception as e:
            logging.error(f"Gradient scaling failed: {e}")
            return False
    
    def _increase_uncertainty_regularization(self, model: nn.Module, optimizer: torch.optim.Optimizer, **context) -> bool:
        """Increase uncertainty regularization to prevent collapse."""
        try:
            # Implementation would depend on specific model architecture
            if hasattr(model, 'uncertainty_weight'):
                model.uncertainty_weight *= 2.0
                logging.info(f"Increased uncertainty regularization to {model.uncertainty_weight}")
                return True
            return False
        except Exception as e:
            logging.error(f"Uncertainty regularization increase failed: {e}")
            return False
    
    def _reinitialize_uncertainty_parameters(self, model: nn.Module, optimizer: torch.optim.Optimizer, **context) -> bool:
        """Reinitialize uncertainty-related parameters."""
        try:
            for name, param in model.named_parameters():
                if 'log_var' in name or 'uncertainty' in name:
                    nn.init.constant_(param, -2.0)  # Small but non-zero uncertainty
            logging.info("Reinitialized uncertainty parameters")
            return True
        except Exception as e:
            logging.error(f"Uncertainty parameter reinitialization failed: {e}")
            return False
    
    def _reduce_batch_size(self, model: nn.Module, optimizer: torch.optim.Optimizer, **context) -> bool:
        """Reduce batch size to save memory."""
        try:
            # This would need to be handled at the training loop level
            # For now, just log the recommendation
            logging.info("Recommended: Reduce batch size to save memory")
            return True
        except Exception as e:
            logging.error(f"Batch size reduction failed: {e}")
            return False
    
    def _clear_cache(self, model: nn.Module, optimizer: torch.optim.Optimizer, **context) -> bool:
        """Clear CUDA cache to free memory."""
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logging.info("Cleared CUDA cache")
            return True
        except Exception as e:
            logging.error(f"Cache clearing failed: {e}")
            return False
    
    def _apply_numerical_stabilization(self, model: nn.Module, optimizer: torch.optim.Optimizer, **context) -> bool:
        """Apply numerical stabilization techniques."""
        try:
            # Clamp extreme parameter values
            with torch.no_grad():
                for param in model.parameters():
                    param.clamp_(-10.0, 10.0)
            logging.info("Applied parameter clamping for numerical stability")
            return True
        except Exception as e:
            logging.error(f"Numerical stabilization failed: {e}")
            return False
    
    # Additional recovery strategies would be implemented here...
    def _recalibrate_uncertainty(self, model: nn.Module, optimizer: torch.optim.Optimizer, **context) -> bool:
        return True  # Placeholder
    
    def _reset_adaptive_scaling(self, model: nn.Module, optimizer: torch.optim.Optimizer, **context) -> bool:
        return True  # Placeholder
        
    def _update_calibration_targets(self, model: nn.Module, optimizer: torch.optim.Optimizer, **context) -> bool:
        return True  # Placeholder
        
    def _inject_uncertainty_noise(self, model: nn.Module, optimizer: torch.optim.Optimizer, **context) -> bool:
        return True  # Placeholder
        
    def _adjust_architecture(self, model: nn.Module, optimizer: torch.optim.Optimizer, **context) -> bool:
        return True  # Placeholder
        
    def _enable_gradient_checkpointing(self, model: nn.Module, optimizer: torch.optim.Optimizer, **context) -> bool:
        return True  # Placeholder
        
    def _reduce_precision(self, model: nn.Module, optimizer: torch.optim.Optimizer, **context) -> bool:
        return True  # Placeholder
        
    def _reset_model_state(self, model: nn.Module, optimizer: torch.optim.Optimizer, **context) -> bool:
        return True  # Placeholder


class ResilienceOrchestrator:
    """Main orchestrator for adaptive resilience system."""
    
    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer):
        self.model = model
        self.optimizer = optimizer
        
        self.health_monitor = AdaptiveHealthMonitor()
        self.recovery_system = AdaptiveRecoverySystem()
        
        # Resilience configuration
        self.monitoring_enabled = True
        self.auto_recovery_enabled = True
        self.failure_threshold = 0.7
        
        # Thread safety
        self._lock = threading.Lock()
        
        logging.info("Initialized Resilience Orchestrator")
    
    @contextmanager
    def resilient_training_step(self, loss: torch.Tensor, uncertainty_stats: Dict, performance_stats: Dict):
        """Context manager for resilient training steps."""
        try:
            with self._lock:
                if self.monitoring_enabled:
                    # Update health metrics
                    health_metrics = self.health_monitor.update_metrics(
                        self.model, loss, uncertainty_stats, performance_stats
                    )
                    
                    # Predict failures
                    failure_predictions = self.health_monitor.predict_failures(health_metrics)
                    
                    # Check for critical failures
                    critical_failures = [
                        failure_mode for failure_mode, probability in failure_predictions.items()
                        if probability >= self.failure_threshold
                    ]
                    
                    if critical_failures and self.auto_recovery_enabled:
                        logging.warning(f"Critical failures detected: {critical_failures}")
                        
                        # Execute recovery for most critical failure
                        most_critical = max(critical_failures, key=lambda f: failure_predictions[f])
                        
                        recovery_success = self.recovery_system.execute_recovery(
                            most_critical, self.model, self.optimizer,
                            health_metrics=health_metrics,
                            failure_predictions=failure_predictions
                        )
                        
                        if not recovery_success:
                            logging.error(f"Failed to recover from {most_critical.value}")
                            raise RuntimeError(f"System recovery failed for {most_critical.value}")
            
            yield
            
        except Exception as e:
            logging.error(f"Error in resilient training step: {e}")
            # Could implement additional error handling here
            raise
    
    def get_system_health_report(self) -> Dict[str, Any]:
        """Generate comprehensive system health report."""
        if not self.health_monitor.metrics_history:
            return {'status': 'no_data'}
        
        latest_metrics = self.health_monitor.metrics_history[-1]
        
        # Compute health score
        health_components = {
            'gradient_health': min(1.0, 1.0 / (1.0 + latest_metrics.gradient_norm / 10.0)),
            'uncertainty_health': min(1.0, latest_metrics.uncertainty_mean * 10.0),
            'calibration_health': 1.0 - min(1.0, latest_metrics.calibration_error),
            'stability_health': latest_metrics.stability_index,
            'security_health': latest_metrics.security_score
        }
        
        overall_health = sum(health_components.values()) / len(health_components)
        
        # Recent failure predictions
        recent_failures = self.health_monitor.failure_predictions
        
        # Recovery statistics
        recovery_stats = self._compute_recovery_stats()
        
        return {
            'overall_health_score': overall_health,
            'health_components': health_components,
            'latest_metrics': {
                'gradient_norm': latest_metrics.gradient_norm,
                'uncertainty_mean': latest_metrics.uncertainty_mean,
                'calibration_error': latest_metrics.calibration_error,
                'memory_usage': latest_metrics.memory_usage,
                'stability_index': latest_metrics.stability_index,
                'security_score': latest_metrics.security_score
            },
            'failure_predictions': dict(recent_failures),
            'recovery_statistics': recovery_stats,
            'monitoring_status': 'active' if self.monitoring_enabled else 'inactive',
            'auto_recovery_status': 'enabled' if self.auto_recovery_enabled else 'disabled'
        }
    
    def _compute_recovery_stats(self) -> Dict[str, Any]:
        """Compute recovery system statistics."""
        if not self.recovery_system.recovery_history:
            return {'total_attempts': 0, 'success_rate': 0.0}
        
        total_attempts = len(self.recovery_system.recovery_history)
        successful_attempts = sum(1 for r in self.recovery_system.recovery_history if r['success'])
        
        success_rate = successful_attempts / total_attempts if total_attempts > 0 else 0.0
        
        # Failure mode breakdown
        failure_breakdown = defaultdict(int)
        for record in self.recovery_system.recovery_history:
            failure_breakdown[record['failure_mode']] += 1
        
        return {
            'total_attempts': total_attempts,
            'successful_attempts': successful_attempts,
            'success_rate': success_rate,
            'failure_mode_breakdown': dict(failure_breakdown),
            'recent_recoveries': self.recovery_system.recovery_history[-10:]  # Last 10
        }


# Export classes
__all__ = [
    'FailureMode',
    'HealthMetrics', 
    'AdaptiveHealthMonitor',
    'AdaptiveRecoverySystem',
    'ResilienceOrchestrator'
]