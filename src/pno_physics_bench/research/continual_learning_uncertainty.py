# SECURITY NOTICE: This file has been automatically hardened for security
# - All user inputs should be validated and sanitized
# - Subprocess calls use secure alternatives
# - SQL queries use parameterized statements
# - No hardcoded secrets or credentials


"""Continual Learning with Uncertainty-Aware Adaptation for Neural Operators.

This module implements continual learning techniques that maintain uncertainty
estimates across sequential learning tasks while preventing catastrophic forgetting.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
from collections import defaultdict
from abc import ABC, abstractmethod


class ElasticWeightConsolidation(nn.Module):
    """EWC with uncertainty-aware importance weighting."""
    
    def __init__(
        self,
        model: nn.Module,
        lambda_ewc: float = 1000.0,
        uncertainty_weighting: bool = True,
        importance_threshold: float = 0.01
    ):
        super().__init__()
        
        self.model = model
        self.lambda_ewc = lambda_ewc
        self.uncertainty_weighting = uncertainty_weighting
        self.importance_threshold = importance_threshold
        
        # Storage for task-specific parameters and importance
        self.task_params = {}
        self.fisher_information = {}
        self.uncertainty_importances = {}
        self.current_task_id = 0
        
    def compute_fisher_information(
        self,
        dataloader: torch.utils.data.DataLoader,
        task_id: int,
        num_samples: Optional[int] = None
    ):
        """Compute Fisher Information Matrix with uncertainty weighting."""
        
        self.model.eval()
        fisher_dict = {}
        uncertainty_dict = {}
        
        # Initialize Fisher information
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                fisher_dict[name] = torch.zeros_like(param)
                uncertainty_dict[name] = torch.zeros_like(param)
        
        sample_count = 0
        total_samples = num_samples or len(dataloader.dataset)
        
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            if num_samples and sample_count >= num_samples:
                break
                
            inputs, targets = inputs.to(next(self.model.parameters()).device), targets.to(next(self.model.parameters()).device)
            
            # Forward pass with uncertainty
            if hasattr(self.model, 'forward') and len(inspect.signature(self.model.forward).parameters) > 1:
                outputs, uncertainties = self.model(inputs, return_uncertainty=True)
            else:
                outputs = self.model(inputs)
                uncertainties = torch.ones_like(outputs) * 0.1  # Default uncertainty
            
            # Compute loss gradients
            loss = F.mse_loss(outputs, targets)
            self.model.zero_grad()
            loss.backward(retain_graph=True)
            
            # Uncertainty-weighted importance
            if self.uncertainty_weighting:
                # Higher uncertainty -> lower importance for that parameter
                uncertainty_weight = 1.0 / (uncertainties.mean() + 1e-8)
            else:
                uncertainty_weight = 1.0
            
            # Accumulate Fisher information
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    fisher_dict[name] += uncertainty_weight * param.grad.data ** 2
                    uncertainty_dict[name] += uncertainties.mean().item()
            
            sample_count += inputs.size(0)
        
        # Normalize by number of samples
        for name in fisher_dict:
            fisher_dict[name] /= sample_count
            uncertainty_dict[name] /= len(dataloader)
        
        # Store Fisher information and uncertainties for this task
        self.fisher_information[task_id] = fisher_dict
        self.uncertainty_importances[task_id] = uncertainty_dict
        
        # Store current parameters
        self.task_params[task_id] = {
            name: param.clone().detach()
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }
    
    def ewc_loss(self, task_id: Optional[int] = None) -> torch.Tensor:
        """Compute EWC regularization loss."""
        loss = torch.tensor(0.0, device=next(self.model.parameters()).device)
        
        # If no specific task, use all previous tasks
        if task_id is None:
            task_ids = list(self.task_params.keys())
        else:
            task_ids = [task_id] if task_id in self.task_params else []
        
        for tid in task_ids:
            for name, param in self.model.named_parameters():
                if name in self.fisher_information[tid]:
                    # Standard EWC term
                    fisher = self.fisher_information[tid][name]
                    old_param = self.task_params[tid][name]
                    
                    # Uncertainty-aware weighting
                    if self.uncertainty_weighting and name in self.uncertainty_importances[tid]:
                        uncertainty_factor = 1.0 / (self.uncertainty_importances[tid][name] + 1e-8)
                    else:
                        uncertainty_factor = 1.0
                    
                    # Only penalize important parameters
                    importance_mask = fisher > self.importance_threshold
                    
                    ewc_term = fisher * importance_mask * uncertainty_factor * (param - old_param) ** 2
                    loss += ewc_term.sum()
        
        return self.lambda_ewc * loss
    
    def set_task(self, task_id: int):
        """Set current task ID."""
        self.current_task_id = task_id
    
    def forward(self, *args, **kwargs):
        """Forward pass through the model."""
        return self.model(*args, **kwargs)


class UncertaintyAwareContinualLearner(nn.Module):
    """Continual learning system with uncertainty-based task detection and adaptation."""
    
    def __init__(
        self,
        base_model: nn.Module,
        memory_size: int = 1000,
        uncertainty_threshold: float = 0.5,
        adaptation_rate: float = 0.01,
        regularization_strength: float = 100.0
    ):
        super().__init__()
        
        self.base_model = base_model
        self.memory_size = memory_size
        self.uncertainty_threshold = uncertainty_threshold
        self.adaptation_rate = adaptation_rate
        self.regularization_strength = regularization_strength
        
        # Episodic memory for rehearsal
        self.episodic_memory = EpisodicMemory(memory_size)
        
        # Task detection based on uncertainty
        self.task_detector = UncertaintyBasedTaskDetector(
            uncertainty_threshold=uncertainty_threshold
        )
        
        # Meta-learning adaptation
        self.meta_adapter = MetaLearningAdapter(base_model)
        
        # Task-specific statistics
        self.task_statistics = {}
        self.current_task_id = 0
        
    def detect_task_change(
        self,
        inputs: torch.Tensor,
        predictions: Optional[torch.Tensor] = None,
        uncertainties: Optional[torch.Tensor] = None
    ) -> bool:
        """Detect if current inputs represent a new task."""
        
        if uncertainties is None:
            # Get uncertainty from model if not provided
            with torch.no_grad():
                if hasattr(self.base_model, 'predict_with_uncertainty'):
                    _, uncertainties = self.base_model.predict_with_uncertainty(inputs)
                else:
                    pred = self.base_model(inputs)
                    uncertainties = torch.ones_like(pred) * 0.1
        
        return self.task_detector.detect_task_change(inputs, uncertainties)
    
    def adapt_to_new_task(
        self,
        new_data_loader: torch.utils.data.DataLoader,
        task_id: Optional[int] = None
    ):
        """Adapt model to new task while preserving previous knowledge."""
        
        if task_id is None:
            task_id = self.current_task_id + 1
        
        # Store statistics for current task before adaptation
        if self.current_task_id in self.task_statistics:
            prev_stats = self.task_statistics[self.current_task_id]
        else:
            prev_stats = None
        
        # Fast adaptation using meta-learning
        adapted_params = self.meta_adapter.adapt(
            new_data_loader, 
            previous_stats=prev_stats
        )
        
        # Update model parameters
        self._update_model_params(adapted_params)
        
        # Update episodic memory with representative samples
        self._update_episodic_memory(new_data_loader, task_id)
        
        # Update task statistics
        self._update_task_statistics(new_data_loader, task_id)
        
        self.current_task_id = task_id
    
    def continual_training_step(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        optimizer: torch.optim.Optimizer
    ) -> Dict[str, float]:
        """Single training step with continual learning."""
        
        # Check for task change
        task_changed = self.detect_task_change(inputs)
        
        # Forward pass
        if hasattr(self.base_model, 'predict_with_uncertainty'):
            predictions, uncertainties = self.base_model.predict_with_uncertainty(inputs)
        else:
            predictions = self.base_model(inputs)
            uncertainties = torch.ones_like(predictions) * 0.1
        
        # Primary loss
        primary_loss = F.mse_loss(predictions, targets)
        
        # Regularization losses
        reg_losses = {}
        
        # EWC loss for previous tasks
        if hasattr(self, 'ewc') and len(self.ewc.task_params) > 0:
            reg_losses['ewc'] = self.ewc.ewc_loss()
        
        # Rehearsal loss using episodic memory
        if len(self.episodic_memory) > 0:
            memory_inputs, memory_targets = self.episodic_memory.sample_batch(inputs.size(0) // 2)
            memory_preds = self.base_model(memory_inputs)
            reg_losses['rehearsal'] = F.mse_loss(memory_preds, memory_targets)
        
        # Uncertainty regularization (encourage confident predictions on old tasks)
        if len(self.episodic_memory) > 0:
            memory_inputs, _ = self.episodic_memory.sample_batch(inputs.size(0) // 4)
            if hasattr(self.base_model, 'predict_with_uncertainty'):
                _, memory_uncertainties = self.base_model.predict_with_uncertainty(memory_inputs)
                reg_losses['uncertainty_reg'] = memory_uncertainties.mean()
        
        # Total loss
        total_loss = primary_loss
        for reg_name, reg_loss in reg_losses.items():
            total_loss += self.regularization_strength * reg_loss
        
        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        # Return metrics
        metrics = {
            'total_loss': total_loss.item(),
            'primary_loss': primary_loss.item(),
            'avg_uncertainty': uncertainties.mean().item(),
            'task_changed': task_changed
        }
        
        for reg_name, reg_loss in reg_losses.items():
            metrics[f'{reg_name}_loss'] = reg_loss.item()
        
        return metrics
    
    def _update_model_params(self, new_params: Dict[str, torch.Tensor]):
        """Update model parameters with adaptation rate."""
        
        for name, param in self.base_model.named_parameters():
            if name in new_params:
                param.data = (1 - self.adaptation_rate) * param.data + self.adaptation_rate * new_params[name]
    
    def _update_episodic_memory(
        self,
        data_loader: torch.utils.data.DataLoader,
        task_id: int
    ):
        """Update episodic memory with representative samples."""
        
        # Select diverse samples based on uncertainty
        selected_samples = []
        
        with torch.no_grad():
            for inputs, targets in data_loader:
                if hasattr(self.base_model, 'predict_with_uncertainty'):
                    _, uncertainties = self.base_model.predict_with_uncertainty(inputs)
                else:
                    uncertainties = torch.ones(inputs.size(0))
                
                # Select samples with high uncertainty (more informative)
                high_uncertainty_indices = uncertainties.mean(dim=tuple(range(1, len(uncertainties.shape)))) > self.uncertainty_threshold
                
                if high_uncertainty_indices.any():
                    selected_inputs = inputs[high_uncertainty_indices]
                    selected_targets = targets[high_uncertainty_indices]
                    
                    for inp, tgt in zip(selected_inputs, selected_targets):
                        selected_samples.append((inp, tgt, task_id))
                        
                        if len(selected_samples) >= self.memory_size // (self.current_task_id + 1):
                            break
                
                if len(selected_samples) >= self.memory_size // (self.current_task_id + 1):
                    break
        
        # Add to episodic memory
        for sample in selected_samples:
            self.episodic_memory.add_sample(*sample)
    
    def _update_task_statistics(
        self,
        data_loader: torch.utils.data.DataLoader,
        task_id: int
    ):
        """Update task-specific statistics."""
        
        stats = {
            'mean_input': torch.zeros(1),
            'std_input': torch.zeros(1),
            'mean_uncertainty': 0.0,
            'num_samples': 0
        }
        
        with torch.no_grad():
            for inputs, targets in data_loader:
                if hasattr(self.base_model, 'predict_with_uncertainty'):
                    _, uncertainties = self.base_model.predict_with_uncertainty(inputs)
                else:
                    uncertainties = torch.ones_like(targets) * 0.1
                
                # Update statistics
                if stats['num_samples'] == 0:
                    stats['mean_input'] = inputs.mean(dim=0)
                    stats['std_input'] = inputs.std(dim=0)
                else:
                    # Running average
                    alpha = inputs.size(0) / (stats['num_samples'] + inputs.size(0))
                    stats['mean_input'] = (1 - alpha) * stats['mean_input'] + alpha * inputs.mean(dim=0)
                    stats['std_input'] = (1 - alpha) * stats['std_input'] + alpha * inputs.std(dim=0)
                
                stats['mean_uncertainty'] += uncertainties.mean().item() * inputs.size(0)
                stats['num_samples'] += inputs.size(0)
        
        stats['mean_uncertainty'] /= stats['num_samples']
        self.task_statistics[task_id] = stats


class EpisodicMemory:
    """Episodic memory for rehearsal-based continual learning."""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.memory = []
        self.task_counts = defaultdict(int)
        
    def add_sample(self, input_sample: torch.Tensor, target: torch.Tensor, task_id: int):
        """Add sample to episodic memory."""
        
        if len(self.memory) >= self.max_size:
            # Remove oldest sample from most represented task
            most_represented_task = max(self.task_counts.items(), key=lambda x: x[1])[0]
            
            # Find and remove sample from most represented task
            for i, (_, _, tid) in enumerate(self.memory):
                if tid == most_represented_task:
                    removed_sample = self.memory.pop(i)
                    self.task_counts[removed_sample[2]] -= 1
                    break
        
        self.memory.append((input_sample.cpu(), target.cpu(), task_id))
        self.task_counts[task_id] += 1
    
    def sample_batch(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample a batch from episodic memory."""
        
        if len(self.memory) == 0:
            return torch.empty(0), torch.empty(0)
        
        # Stratified sampling to maintain task balance
        samples_per_task = max(1, batch_size // len(self.task_counts))
        selected_samples = []
        
        for task_id in self.task_counts:
            task_samples = [(inp, tgt) for inp, tgt, tid in self.memory if tid == task_id]
            
            if task_samples:
                sampled_indices = torch.randperm(len(task_samples))[:samples_per_task]
                selected_samples.extend([task_samples[i] for i in sampled_indices])
        
        # Fill remaining slots with random samples
        while len(selected_samples) < batch_size and len(self.memory) > len(selected_samples):
            remaining_samples = [(inp, tgt) for inp, tgt, _ in self.memory]
            random_idx = torch.randint(0, len(remaining_samples), (1,)).item()
            sample = remaining_samples[random_idx]
            
            if sample not in selected_samples:
                selected_samples.append(sample)
        
        if not selected_samples:
            return torch.empty(0), torch.empty(0)
        
        # Stack samples
        inputs = torch.stack([sample[0] for sample in selected_samples])
        targets = torch.stack([sample[1] for sample in selected_samples])
        
        return inputs, targets
    
    def __len__(self):
        return len(self.memory)


class UncertaintyBasedTaskDetector:
    """Detect task changes based on uncertainty patterns."""
    
    def __init__(
        self,
        uncertainty_threshold: float = 0.5,
        window_size: int = 100,
        detection_threshold: float = 0.7
    ):
        self.uncertainty_threshold = uncertainty_threshold
        self.window_size = window_size
        self.detection_threshold = detection_threshold
        
        self.uncertainty_history = []
        self.baseline_uncertainty = None
        
    def detect_task_change(
        self,
        inputs: torch.Tensor,
        uncertainties: torch.Tensor
    ) -> bool:
        """Detect task change based on uncertainty spike."""
        
        avg_uncertainty = uncertainties.mean().item()
        self.uncertainty_history.append(avg_uncertainty)
        
        # Maintain sliding window
        if len(self.uncertainty_history) > self.window_size:
            self.uncertainty_history.pop(0)
        
        # Initialize baseline
        if self.baseline_uncertainty is None:
            if len(self.uncertainty_history) >= 10:
                self.baseline_uncertainty = np.mean(self.uncertainty_history)
            return False
        
        # Detect significant increase in uncertainty
        if len(self.uncertainty_history) >= 10:
            recent_uncertainty = np.mean(self.uncertainty_history[-10:])
            
            # Task change if recent uncertainty significantly higher than baseline
            if recent_uncertainty > self.baseline_uncertainty * (1 + self.detection_threshold):
                # Update baseline for new task
                self.baseline_uncertainty = recent_uncertainty
                return True
        
        return False


class MetaLearningAdapter:
    """Meta-learning based parameter adaptation for new tasks."""
    
    def __init__(
        self,
        model: nn.Module,
        inner_lr: float = 0.01,
        num_inner_steps: int = 5
    ):
        self.model = model
        self.inner_lr = inner_lr
        self.num_inner_steps = num_inner_steps
        
    def adapt(
        self,
        support_loader: torch.utils.data.DataLoader,
        previous_stats: Optional[Dict] = None
    ) -> Dict[str, torch.Tensor]:
        """MAML-style adaptation to new task."""
        
        # Initialize adapted parameters
        adapted_params = {name: param.clone() for name, param in self.model.named_parameters()}
        
        # Few-shot adaptation
        for step in range(self.num_inner_steps):
            # Sample support batch
            support_inputs, support_targets = next(iter(support_loader))
            
            # Forward pass with adapted parameters
            loss = self._forward_with_params(support_inputs, support_targets, adapted_params)
            
            # Compute gradients
            grads = torch.autograd.grad(loss, adapted_params.values(), create_graph=True)
            
            # Update adapted parameters
            for (name, param), grad in zip(adapted_params.items(), grads):
                adapted_params[name] = param - self.inner_lr * grad
        
        return adapted_params
    
    def _forward_with_params(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        params: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Forward pass with given parameters."""
        
        # This is a simplified version - in practice, you'd need to implement
        # functional forward pass with given parameters
        predictions = self.model(inputs)
        loss = F.mse_loss(predictions, targets)
        
        return loss


class UncertaintyAwareGradientEpisodicMemory(EpisodicMemory):
    """GEM with uncertainty-weighted importance for sample selection."""
    
    def __init__(self, max_size: int = 1000, gamma: float = 0.5):
        super().__init__(max_size)
        self.gamma = gamma  # Uncertainty weighting factor
        self.sample_importances = []
    
    def add_sample(
        self,
        input_sample: torch.Tensor,
        target: torch.Tensor,
        task_id: int,
        uncertainty: Optional[float] = None
    ):
        """Add sample with uncertainty-based importance."""
        
        importance = 1.0 + self.gamma * (uncertainty or 0.1)
        
        if len(self.memory) >= self.max_size:
            # Remove sample with lowest importance
            min_importance_idx = min(
                range(len(self.sample_importances)),
                key=lambda i: self.sample_importances[i]
            )
            
            removed_sample = self.memory.pop(min_importance_idx)
            self.sample_importances.pop(min_importance_idx)
            self.task_counts[removed_sample[2]] -= 1
        
        self.memory.append((input_sample.cpu(), target.cpu(), task_id))
        self.sample_importances.append(importance)
        self.task_counts[task_id] += 1
    
    def sample_batch(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Importance-weighted sampling."""
        
        if len(self.memory) == 0:
            return torch.empty(0), torch.empty(0)
        
        # Importance-based sampling probabilities
        importances = torch.tensor(self.sample_importances)
        probabilities = F.softmax(importances, dim=0)
        
        # Sample indices based on importance
        sampled_indices = torch.multinomial(
            probabilities, 
            min(batch_size, len(self.memory)), 
            replacement=False
        )
        
        # Get samples
        selected_samples = [self.memory[i] for i in sampled_indices]
        
        inputs = torch.stack([sample[0] for sample in selected_samples])
        targets = torch.stack([sample[1] for sample in selected_samples])
        
        return inputs, targets


def uncertainty_aware_continual_training(
    model: nn.Module,
    task_data_loaders: List[torch.utils.data.DataLoader],
    num_epochs_per_task: int = 10,
    memory_size: int = 1000,
    uncertainty_threshold: float = 0.5
) -> Dict[str, Any]:
    """Complete uncertainty-aware continual learning pipeline."""
    
    # Initialize continual learner
    continual_learner = UncertaintyAwareContinualLearner(
        base_model=model,
        memory_size=memory_size,
        uncertainty_threshold=uncertainty_threshold
    )
    
    # Initialize EWC
    ewc = ElasticWeightConsolidation(model)
    continual_learner.ewc = ewc
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    results = {
        'task_metrics': [],
        'memory_efficiency': [],
        'uncertainty_evolution': []
    }
    
    # Train on each task sequentially
    for task_id, data_loader in enumerate(task_data_loaders):
        print(f"Training on Task {task_id}")
        
        # Compute Fisher information for previous task
        if task_id > 0:
            ewc.compute_fisher_information(task_data_loaders[task_id-1], task_id-1)
        
        # Adapt to new task
        continual_learner.adapt_to_new_task(data_loader, task_id)
        
        # Training loop for current task
        task_metrics = []
        
        for epoch in range(num_epochs_per_task):
            epoch_metrics = []
            
            for inputs, targets in data_loader:
                metrics = continual_learner.continual_training_step(
                    inputs, targets, optimizer
                )
                epoch_metrics.append(metrics)
            
            # Average metrics for epoch
            avg_metrics = {
                key: np.mean([m[key] for m in epoch_metrics if key in m])
                for key in epoch_metrics[0].keys()
            }
            
            task_metrics.append(avg_metrics)
        
        results['task_metrics'].append(task_metrics)
        
        # Evaluate memory efficiency
        memory_usage = len(continual_learner.episodic_memory) / memory_size
        results['memory_efficiency'].append(memory_usage)
        
        # Track uncertainty evolution
        with torch.no_grad():
            sample_inputs, _ = next(iter(data_loader))
            if hasattr(model, 'predict_with_uncertainty'):
                _, uncertainties = model.predict_with_uncertainty(sample_inputs)
                results['uncertainty_evolution'].append(uncertainties.mean().item())
    
    return results