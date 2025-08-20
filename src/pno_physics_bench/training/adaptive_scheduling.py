# SECURITY NOTICE: This file has been automatically hardened for security
# - All user inputs should be validated and sanitized
# - Subprocess calls use secure alternatives
# - SQL queries use parameterized statements
# - No hardcoded secrets or credentials


"""Adaptive learning rate scheduling for PNO training with uncertainty-aware optimization."""

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import _LRScheduler
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import logging
from collections import deque
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class AdaptiveScheduler(_LRScheduler, ABC):
    """Base class for adaptive learning rate schedulers."""
    
    def __init__(self, optimizer, patience: int = 10, min_lr: float = 1e-7, **kwargs):
        self.patience = patience
        self.min_lr = min_lr
        self.wait = 0
        self.best_loss = float('inf')
        super().__init__(optimizer, **kwargs)
    
    @abstractmethod
    def should_adjust(self, metrics: Dict[str, float]) -> bool:
        """Determine if learning rate should be adjusted."""
        pass
    
    def step(self, metrics: Optional[Dict[str, float]] = None):
        """Step the scheduler with optional metrics."""
        if metrics is not None and self.should_adjust(metrics):
            self._adjust_lr()
        super().step()
    
    def _adjust_lr(self):
        """Adjust learning rate based on implementation."""
        for param_group in self.optimizer.param_groups:
            old_lr = param_group['lr']
            new_lr = max(old_lr * 0.5, self.min_lr)
            param_group['lr'] = new_lr
            logger.info(f"Reduced learning rate: {old_lr:.2e} -> {new_lr:.2e}")


class UncertaintyAwareLRScheduler(AdaptiveScheduler):
    """Learning rate scheduler that adapts based on uncertainty metrics."""
    
    def __init__(
        self, 
        optimizer,
        uncertainty_threshold: float = 0.1,
        calibration_weight: float = 0.3,
        coverage_target: float = 0.9,
        patience: int = 15,
        **kwargs
    ):
        """Initialize uncertainty-aware scheduler.
        
        Args:
            uncertainty_threshold: Threshold for uncertainty-based adjustments
            calibration_weight: Weight for calibration error in decisions
            coverage_target: Target coverage for uncertainty estimates
            patience: Patience before reducing learning rate
        """
        super().__init__(optimizer, patience=patience, **kwargs)
        self.uncertainty_threshold = uncertainty_threshold
        self.calibration_weight = calibration_weight
        self.coverage_target = coverage_target
        self.uncertainty_history = deque(maxlen=patience)
        self.calibration_history = deque(maxlen=patience)
        
    def should_adjust(self, metrics: Dict[str, float]) -> bool:
        """Adjust LR based on uncertainty and calibration metrics."""
        uncertainty = metrics.get('avg_uncertainty', 0.0)
        calibration_error = metrics.get('calibration_error', 0.0)
        coverage = metrics.get('coverage_90', 0.9)
        
        self.uncertainty_history.append(uncertainty)
        self.calibration_history.append(calibration_error)
        
        # Check if uncertainty is increasing (overfitting)
        if len(self.uncertainty_history) >= self.patience:
            uncertainty_trend = np.mean(list(self.uncertainty_history)[-5:]) - np.mean(list(self.uncertainty_history)[:5])
            calibration_trend = np.mean(list(self.calibration_history)[-5:]) - np.mean(list(self.calibration_history)[:5])
            
            # Reduce LR if uncertainty is increasing or calibration is degrading
            if (uncertainty_trend > self.uncertainty_threshold or 
                calibration_trend > 0.05 or 
                abs(coverage - self.coverage_target) > 0.1):
                return True
                
        return False


class AdaptiveMomentumScheduler:
    """Adaptive momentum scheduler based on uncertainty convergence."""
    
    def __init__(
        self,
        optimizer,
        uncertainty_window: int = 10,
        momentum_range: Tuple[float, float] = (0.8, 0.99),
        convergence_threshold: float = 0.01
    ):
        """Initialize adaptive momentum scheduler.
        
        Args:
            optimizer: The optimizer to modify
            uncertainty_window: Window for measuring uncertainty convergence
            momentum_range: (min, max) momentum values
            convergence_threshold: Threshold for uncertainty convergence
        """
        self.optimizer = optimizer
        self.uncertainty_window = uncertainty_window
        self.min_momentum, self.max_momentum = momentum_range
        self.convergence_threshold = convergence_threshold
        self.uncertainty_history = deque(maxlen=uncertainty_window)
        
    def update_momentum(self, uncertainty_metrics: Dict[str, float]):
        """Update momentum based on uncertainty convergence."""
        avg_uncertainty = uncertainty_metrics.get('avg_uncertainty', 0.0)
        self.uncertainty_history.append(avg_uncertainty)
        
        if len(self.uncertainty_history) >= self.uncertainty_window:
            # Calculate uncertainty variance as convergence measure
            uncertainty_var = np.var(list(self.uncertainty_history))
            
            # Higher variance -> lower momentum (more exploration)
            # Lower variance -> higher momentum (faster convergence)
            if uncertainty_var > self.convergence_threshold:
                momentum = self.min_momentum
            else:
                momentum = self.max_momentum
                
            # Update optimizer momentum
            for param_group in self.optimizer.param_groups:
                if 'betas' in param_group:  # Adam-like optimizers
                    param_group['betas'] = (momentum, param_group['betas'][1])
                elif 'momentum' in param_group:  # SGD-like optimizers
                    param_group['momentum'] = momentum


class HyperbolicLRScheduler(AdaptiveScheduler):
    """Hyperbolic learning rate decay with uncertainty-aware restarts."""
    
    def __init__(
        self,
        optimizer,
        T_max: int = 100,
        eta_min: float = 1e-6,
        restart_factor: float = 2.0,
        uncertainty_restart_threshold: float = 0.05,
        **kwargs
    ):
        """Initialize hyperbolic scheduler with uncertainty restarts.
        
        Args:
            T_max: Maximum number of iterations
            eta_min: Minimum learning rate
            restart_factor: Factor to multiply T_max on restart
            uncertainty_restart_threshold: Uncertainty threshold for restarts
        """
        super().__init__(optimizer, **kwargs)
        self.T_max = T_max
        self.eta_min = eta_min
        self.restart_factor = restart_factor
        self.uncertainty_restart_threshold = uncertainty_restart_threshold
        self.T_cur = 0
        self.base_lrs = [group['lr'] for group in self.optimizer.param_groups]
        self.uncertainty_baseline = None
        
    def get_lr(self):
        """Calculate hyperbolic learning rate."""
        return [
            self.eta_min + (base_lr - self.eta_min) * 
            (1 + np.cosh(self.T_cur / self.T_max)) / (2 * np.cosh(1))
            for base_lr in self.base_lrs
        ]
    
    def should_adjust(self, metrics: Dict[str, float]) -> bool:
        """Check if restart is needed based on uncertainty plateau."""
        uncertainty = metrics.get('avg_uncertainty', 0.0)
        
        if self.uncertainty_baseline is None:
            self.uncertainty_baseline = uncertainty
            return False
            
        # Restart if uncertainty has plateaued
        uncertainty_improvement = (self.uncertainty_baseline - uncertainty) / self.uncertainty_baseline
        
        if uncertainty_improvement < self.uncertainty_restart_threshold and self.T_cur > self.T_max * 0.5:
            self._restart()
            return True
            
        return False
    
    def _restart(self):
        """Restart the scheduler with increased period."""
        logger.info(f"Restarting scheduler: T_max {self.T_max} -> {int(self.T_max * self.restart_factor)}")
        self.T_max = int(self.T_max * self.restart_factor)
        self.T_cur = 0
        self.uncertainty_baseline = None
    
    def step(self, metrics: Optional[Dict[str, float]] = None):
        """Step with restart logic."""
        self.T_cur += 1
        super().step(metrics)


class AdaptiveWarmupScheduler:
    """Adaptive warmup scheduler that adjusts based on uncertainty stabilization."""
    
    def __init__(
        self,
        optimizer,
        base_scheduler: _LRScheduler,
        warmup_epochs: int = 10,
        uncertainty_stabilization_threshold: float = 0.02,
        min_warmup_epochs: int = 5,
        max_warmup_epochs: int = 50
    ):
        """Initialize adaptive warmup scheduler.
        
        Args:
            optimizer: The optimizer
            base_scheduler: Base scheduler to use after warmup
            warmup_epochs: Initial warmup epochs
            uncertainty_stabilization_threshold: Threshold for uncertainty stabilization
            min_warmup_epochs: Minimum warmup epochs
            max_warmup_epochs: Maximum warmup epochs
        """
        self.optimizer = optimizer
        self.base_scheduler = base_scheduler
        self.warmup_epochs = warmup_epochs
        self.uncertainty_threshold = uncertainty_stabilization_threshold
        self.min_warmup = min_warmup_epochs
        self.max_warmup = max_warmup_epochs
        
        self.current_epoch = 0
        self.in_warmup = True
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.uncertainty_history = deque(maxlen=10)
        
    def step(self, metrics: Optional[Dict[str, float]] = None):
        """Step the scheduler with adaptive warmup."""
        self.current_epoch += 1
        
        if self.in_warmup:
            self._warmup_step(metrics)
        else:
            self.base_scheduler.step()
    
    def _warmup_step(self, metrics: Optional[Dict[str, float]] = None):
        """Perform warmup step with uncertainty monitoring."""
        # Standard warmup LR scaling
        warmup_factor = min(1.0, self.current_epoch / self.warmup_epochs)
        
        for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            param_group['lr'] = base_lr * warmup_factor
        
        # Check if we should exit warmup early based on uncertainty
        if metrics and self.current_epoch >= self.min_warmup:
            uncertainty = metrics.get('avg_uncertainty', 0.0)
            self.uncertainty_history.append(uncertainty)
            
            if len(self.uncertainty_history) >= 5:
                uncertainty_std = np.std(list(self.uncertainty_history))
                if uncertainty_std < self.uncertainty_threshold:
                    logger.info(f"Early warmup exit at epoch {self.current_epoch} (uncertainty stabilized)")
                    self.in_warmup = False
        
        # Force exit if maximum warmup reached
        if self.current_epoch >= self.max_warmup:
            logger.info(f"Warmup completed at maximum epochs: {self.max_warmup}")
            self.in_warmup = False


class MultiCriteriaScheduler:
    """Multi-criteria scheduler combining loss, uncertainty, and calibration metrics."""
    
    def __init__(
        self,
        optimizer,
        schedulers: Dict[str, _LRScheduler],
        weights: Dict[str, float],
        patience: int = 20,
        min_improvement: float = 1e-4
    ):
        """Initialize multi-criteria scheduler.
        
        Args:
            optimizer: The optimizer
            schedulers: Dictionary of named schedulers
            weights: Weights for each criterion in decision making
            patience: Patience before adjustment
            min_improvement: Minimum improvement threshold
        """
        self.optimizer = optimizer
        self.schedulers = schedulers
        self.weights = weights
        self.patience = patience
        self.min_improvement = min_improvement
        
        self.wait = 0
        self.best_score = float('inf')
        self.metric_history = {name: deque(maxlen=patience) for name in weights.keys()}
        
    def step(self, metrics: Dict[str, float]):
        """Step based on weighted combination of metrics."""
        # Calculate weighted score
        score = 0.0
        for metric_name, weight in self.weights.items():
            if metric_name in metrics:
                self.metric_history[metric_name].append(metrics[metric_name])
                score += weight * metrics[metric_name]
        
        # Check for improvement
        if score < self.best_score - self.min_improvement:
            self.best_score = score
            self.wait = 0
        else:
            self.wait += 1
        
        # Adjust if no improvement
        if self.wait >= self.patience:
            self._adjust_all_schedulers()
            self.wait = 0
    
    def _adjust_all_schedulers(self):
        """Adjust all schedulers based on current metrics."""
        for name, scheduler in self.schedulers.items():
            if hasattr(scheduler, 'step'):
                scheduler.step()
        
        logger.info("Adjusted learning rate via multi-criteria scheduler")


def create_adaptive_scheduler(
    optimizer,
    scheduler_type: str = "uncertainty_aware",
    **kwargs
) -> _LRScheduler:
    """Factory function to create adaptive schedulers.
    
    Args:
        optimizer: The optimizer
        scheduler_type: Type of adaptive scheduler
        **kwargs: Additional arguments for scheduler
        
    Returns:
        Configured adaptive scheduler
    """
    if scheduler_type == "uncertainty_aware":
        return UncertaintyAwareLRScheduler(optimizer, **kwargs)
    elif scheduler_type == "hyperbolic":
        return HyperbolicLRScheduler(optimizer, **kwargs)
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")