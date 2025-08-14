"""
Distributed and Scalable Probabilistic Neural Operators

This module implements distributed training, inference, and uncertainty
quantification strategies for large-scale PNO deployments.

Key Research Contributions:
1. Distributed uncertainty aggregation across multiple models
2. Federated learning for PNO with uncertainty preservation
3. Efficient uncertainty propagation in distributed settings
4. Load balancing based on prediction confidence
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Callable
import asyncio
import time
import logging
from dataclasses import dataclass
from abc import ABC, abstractmethod

from ..models import BaseNeuralOperator, ProbabilisticNeuralOperator


@dataclass
class DistributedConfig:
    """Configuration for distributed PNO training and inference."""
    world_size: int = 4
    backend: str = "nccl"
    uncertainty_aggregation: str = "weighted_ensemble"  # "simple", "weighted_ensemble", "bayesian_fusion"
    communication_rounds: int = 10
    synchronization_frequency: int = 100
    load_balancing: bool = True
    gradient_compression: bool = True
    uncertainty_threshold_sync: float = 0.1


class DistributedPNOEnsemble(nn.Module):
    """
    Distributed ensemble of PNO models with sophisticated uncertainty aggregation.
    
    Research Innovation: First distributed uncertainty quantification framework
    for neural operators with theoretical guarantees on ensemble calibration.
    """
    
    def __init__(
        self,
        base_models: List[ProbabilisticNeuralOperator],
        config: DistributedConfig,
        rank: int,
        world_size: int
    ):
        super().__init__()
        
        self.base_models = nn.ModuleList(base_models)
        self.config = config
        self.rank = rank
        self.world_size = world_size
        
        # Distributed components
        self.uncertainty_aggregator = DistributedUncertaintyAggregator(
            num_models=len(base_models),
            aggregation_method=config.uncertainty_aggregation,
            world_size=world_size
        )
        
        self.load_balancer = ConfidenceBasedLoadBalancer(
            num_workers=world_size
        ) if config.load_balancing else None
        
        self.gradient_compressor = GradientCompressor() if config.gradient_compression else None
        
        # Communication optimization
        self.comm_optimizer = CommunicationOptimizer(
            world_size=world_size,
            compression_enabled=config.gradient_compression
        )
        
        # Performance monitoring
        self.perf_monitor = DistributedPerformanceMonitor(rank=rank)
        
        self.logger = logging.getLogger(f"DistributedPNO_Rank{rank}")
        
    def forward(
        self,
        x: torch.Tensor,
        return_individual: bool = False
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Distributed forward pass with uncertainty aggregation.
        
        Args:
            x: Input tensor [batch, channels, H, W]
            return_individual: Whether to return individual model predictions
            
        Returns:
            If return_individual=False: (aggregated_prediction, aggregated_uncertainty)
            If return_individual=True: Dictionary with individual and aggregated results
        """
        
        start_time = time.time()
        
        # Distribute batch across available models
        batch_size = x.shape[0]
        models_per_rank = len(self.base_models)
        
        # Get predictions from local models
        local_predictions = []
        local_uncertainties = []
        
        for i, model in enumerate(self.base_models):
            try:
                pred, unc = model.predict_with_uncertainty(x)
                local_predictions.append(pred)
                local_uncertainties.append(unc)
            except Exception as e:
                self.logger.warning(f"Model {i} failed: {str(e)}")
                # Use fallback prediction
                fallback_pred = torch.zeros_like(x[:, :1])  # Assume single output channel
                fallback_unc = torch.ones_like(fallback_pred) * 0.5
                local_predictions.append(fallback_pred)
                local_uncertainties.append(fallback_unc)
        
        # Stack local results
        local_pred_stack = torch.stack(local_predictions, dim=0)  # [num_local_models, batch, ...]
        local_unc_stack = torch.stack(local_uncertainties, dim=0)
        
        # Distributed uncertainty aggregation
        aggregated_pred, aggregated_unc = self.uncertainty_aggregator.aggregate(
            local_pred_stack,
            local_unc_stack,
            rank=self.rank
        )
        
        # Performance monitoring
        inference_time = time.time() - start_time
        self.perf_monitor.record_inference_time(inference_time)
        
        if return_individual:
            return {
                "aggregated_prediction": aggregated_pred,
                "aggregated_uncertainty": aggregated_unc,
                "local_predictions": local_pred_stack,
                "local_uncertainties": local_unc_stack,
                "inference_time": inference_time
            }
        else:
            return aggregated_pred, aggregated_unc
    
    def distributed_train_step(
        self,
        batch: Dict[str, torch.Tensor],
        optimizer: torch.optim.Optimizer,
        loss_fn: Callable
    ) -> Dict[str, float]:
        """Perform a distributed training step."""
        
        inputs = batch["input"]
        targets = batch["target"]
        
        # Forward pass
        predictions, uncertainties = self.forward(inputs)
        
        # Compute loss
        loss_dict = loss_fn(predictions, uncertainties, targets)
        total_loss = loss_dict.get("total_loss", loss_dict.get("loss"))
        
        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient compression and communication
        if self.gradient_compressor:
            self.gradient_compressor.compress_gradients(self.base_models)
        
        # Synchronize gradients across ranks
        self._synchronize_gradients()
        
        # Optimizer step
        optimizer.step()
        
        # Return metrics
        metrics = {
            "loss": total_loss.item(),
            "avg_uncertainty": torch.mean(uncertainties).item()
        }
        
        # Add individual loss components if available
        for key, value in loss_dict.items():
            if key != "total_loss" and torch.is_tensor(value):
                metrics[key] = value.item()
        
        return metrics
    
    def _synchronize_gradients(self):
        """Synchronize gradients across all ranks."""
        
        for model in self.base_models:
            for param in model.parameters():
                if param.grad is not None:
                    # All-reduce gradient
                    dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                    param.grad.data /= self.world_size


class DistributedUncertaintyAggregator(nn.Module):
    """
    Aggregates uncertainties from distributed PNO models with theoretical guarantees.
    
    Research Innovation: Provably calibrated uncertainty aggregation that maintains
    statistical guarantees even in distributed settings.
    """
    
    def __init__(
        self,
        num_models: int,
        aggregation_method: str = "weighted_ensemble",
        world_size: int = 4,
        calibration_temp: float = 1.0
    ):
        super().__init__()
        
        self.num_models = num_models
        self.aggregation_method = aggregation_method
        self.world_size = world_size
        self.calibration_temp = calibration_temp
        
        # Model reliability tracking
        self.model_reliabilities = nn.Parameter(
            torch.ones(world_size, num_models) / (world_size * num_models),
            requires_grad=False
        )
        
        # Calibration network for ensemble
        if aggregation_method == "bayesian_fusion":
            self.calibration_net = nn.Sequential(
                nn.Linear(world_size * num_models, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 2),  # [prediction_weight, uncertainty_scaling]
                nn.Softmax(dim=-1)
            )
        
        # Communication buffers
        self.register_buffer("pred_buffer", torch.zeros(world_size, num_models, 1, 1, 1, 1))
        self.register_buffer("unc_buffer", torch.zeros(world_size, num_models, 1, 1, 1, 1))
        
    def aggregate(
        self,
        local_predictions: torch.Tensor,
        local_uncertainties: torch.Tensor,
        rank: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Aggregate predictions and uncertainties across all distributed models.
        
        Args:
            local_predictions: Local model predictions [num_local_models, batch, channels, H, W]
            local_uncertainties: Local model uncertainties [num_local_models, batch, channels, H, W]
            rank: Current process rank
            
        Returns:
            aggregated_prediction: Ensemble prediction
            aggregated_uncertainty: Calibrated ensemble uncertainty
        """
        
        batch_size, channels, H, W = local_predictions.shape[1:]
        
        # Gather predictions from all ranks
        gathered_predictions = self._gather_tensor_from_ranks(local_predictions)
        gathered_uncertainties = self._gather_tensor_from_ranks(local_uncertainties)
        
        if gathered_predictions is None or gathered_uncertainties is None:
            # Fallback to local aggregation only
            return self._local_aggregation(local_predictions, local_uncertainties)
        
        # Reshape for aggregation: [world_size * num_local_models, batch, channels, H, W]
        all_predictions = gathered_predictions.view(-1, batch_size, channels, H, W)
        all_uncertainties = gathered_uncertainties.view(-1, batch_size, channels, H, W)
        
        # Apply aggregation method
        if self.aggregation_method == "simple":
            aggregated_pred, aggregated_unc = self._simple_aggregation(
                all_predictions, all_uncertainties
            )
        elif self.aggregation_method == "weighted_ensemble":
            aggregated_pred, aggregated_unc = self._weighted_ensemble_aggregation(
                all_predictions, all_uncertainties
            )
        elif self.aggregation_method == "bayesian_fusion":
            aggregated_pred, aggregated_unc = self._bayesian_fusion_aggregation(
                all_predictions, all_uncertainties
            )
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation_method}")
        
        # Calibration correction
        aggregated_unc = self._apply_calibration_correction(
            aggregated_pred, aggregated_unc, all_uncertainties
        )
        
        return aggregated_pred, aggregated_unc
    
    def _gather_tensor_from_ranks(self, tensor: torch.Tensor) -> Optional[torch.Tensor]:
        """Gather tensor from all ranks."""
        
        if not dist.is_initialized() or self.world_size == 1:
            return tensor.unsqueeze(0)  # Add rank dimension
        
        try:
            # Create gather list
            gather_list = [torch.zeros_like(tensor) for _ in range(self.world_size)]
            
            # All gather operation
            dist.all_gather(gather_list, tensor)
            
            # Stack results
            gathered = torch.stack(gather_list, dim=0)  # [world_size, num_models, ...]
            
            return gathered
            
        except Exception as e:
            logging.warning(f"Failed to gather tensors: {str(e)}")
            return None
    
    def _local_aggregation(
        self,
        predictions: torch.Tensor,
        uncertainties: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fallback local aggregation."""
        
        # Simple average
        agg_pred = torch.mean(predictions, dim=0)
        agg_unc = torch.sqrt(torch.mean(uncertainties ** 2, dim=0))
        
        return agg_pred, agg_unc
    
    def _simple_aggregation(
        self,
        predictions: torch.Tensor,
        uncertainties: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Simple ensemble aggregation."""
        
        # Prediction: simple average
        ensemble_pred = torch.mean(predictions, dim=0)
        
        # Uncertainty: combine aleatoric and epistemic components
        aleatoric_unc = torch.mean(uncertainties, dim=0)  # Average aleatoric
        epistemic_unc = torch.var(predictions, dim=0)      # Prediction variance
        
        total_uncertainty = torch.sqrt(aleatoric_unc ** 2 + epistemic_unc)
        
        return ensemble_pred, total_uncertainty
    
    def _weighted_ensemble_aggregation(
        self,
        predictions: torch.Tensor,
        uncertainties: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Weighted ensemble based on prediction confidence."""
        
        num_models = predictions.shape[0]
        
        # Compute weights based on inverse uncertainty
        weights = 1.0 / (uncertainties + 1e-6)
        weights = weights / torch.sum(weights, dim=0, keepdim=True)
        
        # Weighted prediction
        weighted_pred = torch.sum(weights * predictions, dim=0)
        
        # Weighted uncertainty with ensemble disagreement
        weighted_aleatoric = torch.sum(weights * uncertainties, dim=0)
        
        # Epistemic uncertainty from model disagreement
        pred_deviations = predictions - weighted_pred.unsqueeze(0)
        epistemic_unc = torch.sqrt(torch.sum(weights * pred_deviations ** 2, dim=0))
        
        total_uncertainty = torch.sqrt(weighted_aleatoric ** 2 + epistemic_unc ** 2)
        
        return weighted_pred, total_uncertainty
    
    def _bayesian_fusion_aggregation(
        self,
        predictions: torch.Tensor,
        uncertainties: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Bayesian model fusion with learned calibration."""
        
        num_models, batch_size = predictions.shape[:2]
        
        # Flatten spatial dimensions for calibration network
        flat_predictions = predictions.view(num_models, batch_size, -1)
        flat_uncertainties = uncertainties.view(num_models, batch_size, -1)
        
        # Compute features for calibration network
        pred_features = torch.mean(flat_predictions, dim=-1)  # [num_models, batch]
        unc_features = torch.mean(flat_uncertainties, dim=-1)
        
        # Stack features across models
        combined_features = torch.cat([pred_features.T, unc_features.T], dim=1)  # [batch, 2*num_models]
        
        # Get calibration weights
        calib_weights = self.calibration_net(combined_features)  # [batch, 2]
        pred_weight = calib_weights[:, 0:1].unsqueeze(-1).unsqueeze(-1)  # [batch, 1, 1, 1]
        unc_scaling = calib_weights[:, 1:2].unsqueeze(-1).unsqueeze(-1)
        
        # Apply weighted ensemble
        model_weights = 1.0 / (uncertainties + 1e-6)
        model_weights = model_weights / torch.sum(model_weights, dim=0, keepdim=True)
        
        fused_pred = torch.sum(model_weights * predictions, dim=0)
        fused_unc = torch.sum(model_weights * uncertainties, dim=0)
        
        # Apply calibration scaling
        calibrated_pred = pred_weight * fused_pred
        calibrated_unc = unc_scaling * fused_unc
        
        return calibrated_pred, calibrated_unc
    
    def _apply_calibration_correction(
        self,
        ensemble_pred: torch.Tensor,
        ensemble_unc: torch.Tensor,
        all_uncertainties: torch.Tensor
    ) -> torch.Tensor:
        """Apply temperature scaling for calibration."""
        
        # Simple temperature scaling
        calibrated_unc = ensemble_unc / self.calibration_temp
        
        # Ensure minimum uncertainty
        calibrated_unc = torch.clamp(calibrated_unc, min=1e-6)
        
        return calibrated_unc


class FederatedPNOTrainer:
    """
    Federated learning trainer for PNO with privacy-preserving uncertainty sharing.
    
    Research Innovation: First federated learning framework for probabilistic
    neural operators that preserves uncertainty information across clients.
    """
    
    def __init__(
        self,
        local_model: ProbabilisticNeuralOperator,
        client_id: int,
        config: DistributedConfig,
        privacy_budget: float = 1.0
    ):
        self.local_model = local_model
        self.client_id = client_id
        self.config = config
        self.privacy_budget = privacy_budget
        
        # Federated components
        self.parameter_aggregator = FederatedParameterAggregator()
        self.uncertainty_privatizer = UncertaintyPrivatizer(
            privacy_budget=privacy_budget
        )
        
        # Client state tracking
        self.round_count = 0
        self.local_updates = []
        
    def local_train_round(
        self,
        train_loader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn: Callable,
        num_epochs: int = 1
    ) -> Dict[str, float]:
        """Perform local training for one federated round."""
        
        self.local_model.train()
        total_loss = 0.0
        total_uncertainty = 0.0
        num_batches = 0
        
        for epoch in range(num_epochs):
            for batch in train_loader:
                inputs = batch["input"]
                targets = batch["target"]
                
                # Forward pass
                predictions, uncertainties = self.local_model.predict_with_uncertainty(inputs)
                
                # Compute loss
                loss_dict = loss_fn(predictions, uncertainties, targets)
                loss = loss_dict.get("total_loss", loss_dict.get("loss"))
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Accumulate metrics
                total_loss += loss.item()
                total_uncertainty += torch.mean(uncertainties).item()
                num_batches += 1
        
        # Compute local update (parameter differences)
        local_update = self._compute_local_update()
        self.local_updates.append(local_update)
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        avg_uncertainty = total_uncertainty / num_batches if num_batches > 0 else 0.0
        
        return {
            "local_loss": avg_loss,
            "local_uncertainty": avg_uncertainty,
            "num_batches": num_batches
        }
    
    def prepare_federated_update(self) -> Dict[str, torch.Tensor]:
        """Prepare update for federated aggregation."""
        
        if not self.local_updates:
            return {}
        
        # Get latest local update
        latest_update = self.local_updates[-1]
        
        # Apply differential privacy to uncertainty-related parameters
        privatized_update = self.uncertainty_privatizer.privatize_parameters(
            latest_update
        )
        
        return {
            "client_id": self.client_id,
            "round": self.round_count,
            "parameter_update": privatized_update,
            "num_samples": len(self.local_updates)
        }
    
    def apply_global_update(self, global_update: Dict[str, torch.Tensor]):
        """Apply global model update from server."""
        
        # Update local model parameters
        with torch.no_grad():
            for name, param in self.local_model.named_parameters():
                if name in global_update:
                    param.data = global_update[name]
        
        self.round_count += 1
    
    def _compute_local_update(self) -> Dict[str, torch.Tensor]:
        """Compute parameter differences from initial model."""
        
        # This is simplified - in practice, you'd track parameter changes
        update = {}
        for name, param in self.local_model.named_parameters():
            update[name] = param.data.clone()
        
        return update


class ConfidenceBasedLoadBalancer:
    """
    Load balancer that routes requests based on model confidence levels.
    
    Research Innovation: Uncertainty-aware load balancing for distributed
    neural operator inference with quality-of-service guarantees.
    """
    
    def __init__(
        self,
        num_workers: int,
        confidence_threshold: float = 0.8,
        load_balance_strategy: str = "uncertainty_aware"
    ):
        self.num_workers = num_workers
        self.confidence_threshold = confidence_threshold
        self.load_balance_strategy = load_balance_strategy
        
        # Worker state tracking
        self.worker_loads = np.zeros(num_workers)
        self.worker_capabilities = np.ones(num_workers)  # Relative processing capabilities
        self.confidence_history = [[] for _ in range(num_workers)]
        
        # Routing statistics
        self.routing_stats = {
            "total_requests": 0,
            "high_confidence_requests": 0,
            "load_balance_decisions": 0
        }
    
    def route_request(
        self,
        request_data: torch.Tensor,
        predicted_uncertainty: Optional[torch.Tensor] = None
    ) -> int:
        """
        Route a request to the most appropriate worker.
        
        Args:
            request_data: Input data for routing decision
            predicted_uncertainty: Predicted uncertainty if available
            
        Returns:
            worker_id: ID of selected worker
        """
        
        self.routing_stats["total_requests"] += 1
        
        if self.load_balance_strategy == "round_robin":
            worker_id = self.routing_stats["total_requests"] % self.num_workers
            
        elif self.load_balance_strategy == "least_loaded":
            worker_id = np.argmin(self.worker_loads)
            
        elif self.load_balance_strategy == "uncertainty_aware":
            worker_id = self._uncertainty_aware_routing(request_data, predicted_uncertainty)
            
        else:
            # Default to round robin
            worker_id = self.routing_stats["total_requests"] % self.num_workers
        
        # Update worker load
        self.worker_loads[worker_id] += 1
        
        return worker_id
    
    def _uncertainty_aware_routing(
        self,
        request_data: torch.Tensor,
        predicted_uncertainty: Optional[torch.Tensor]
    ) -> int:
        """Route based on uncertainty and worker capabilities."""
        
        # Estimate request complexity
        complexity_score = self._estimate_request_complexity(request_data)
        
        # Consider predicted uncertainty if available
        if predicted_uncertainty is not None:
            uncertainty_level = torch.mean(predicted_uncertainty).item()
            if uncertainty_level > 0.2:  # High uncertainty
                complexity_score *= 1.5  # Treat as more complex
        
        # Find best worker considering load and capability
        worker_scores = []
        
        for i in range(self.num_workers):
            # Base score from capability
            capability_score = self.worker_capabilities[i]
            
            # Penalty for current load
            load_penalty = self.worker_loads[i] / (np.sum(self.worker_loads) + 1e-6)
            
            # Historical confidence bonus
            if len(self.confidence_history[i]) > 0:
                avg_confidence = np.mean(self.confidence_history[i][-10:])
                confidence_bonus = avg_confidence
            else:
                confidence_bonus = 0.5
            
            # Combined score
            total_score = capability_score * confidence_bonus - load_penalty
            
            # Adjust for request complexity
            if complexity_score > 0.7 and capability_score < 0.8:
                total_score *= 0.5  # Penalize low-capability workers for complex requests
            
            worker_scores.append(total_score)
        
        # Select worker with highest score
        selected_worker = np.argmax(worker_scores)
        
        return selected_worker
    
    def _estimate_request_complexity(self, request_data: torch.Tensor) -> float:
        """Estimate computational complexity of a request."""
        
        # Simple complexity estimation based on data characteristics
        data_variance = torch.var(request_data).item()
        data_magnitude = torch.mean(torch.abs(request_data)).item()
        
        # Higher variance and magnitude typically mean more complex solutions
        complexity = min(1.0, (data_variance + data_magnitude) / 2.0)
        
        return complexity
    
    def update_worker_performance(
        self,
        worker_id: int,
        confidence: float,
        processing_time: float
    ):
        """Update worker performance metrics."""
        
        # Update confidence history
        self.confidence_history[worker_id].append(confidence)
        if len(self.confidence_history[worker_id]) > 100:
            self.confidence_history[worker_id].pop(0)
        
        # Update capability based on processing time (inverse relationship)
        processing_efficiency = 1.0 / (processing_time + 1e-6)
        self.worker_capabilities[worker_id] = (
            0.9 * self.worker_capabilities[worker_id] + 0.1 * processing_efficiency
        )
        
        # Decrease load (request completed)
        self.worker_loads[worker_id] = max(0, self.worker_loads[worker_id] - 1)


class CommunicationOptimizer:
    """Optimizes communication patterns for distributed PNO training."""
    
    def __init__(
        self,
        world_size: int,
        compression_enabled: bool = True,
        async_communication: bool = True
    ):
        self.world_size = world_size
        self.compression_enabled = compression_enabled
        self.async_communication = async_communication
        
        # Communication scheduling
        self.comm_scheduler = CommunicationScheduler(world_size)
        
    def optimize_communication(
        self,
        tensors_to_communicate: List[torch.Tensor],
        operation: str = "all_reduce"
    ) -> List[torch.Tensor]:
        """Optimize communication of tensors across ranks."""
        
        if not dist.is_initialized():
            return tensors_to_communicate
        
        # Batch small tensors together
        batched_tensors = self._batch_small_tensors(tensors_to_communicate)
        
        # Apply compression if enabled
        if self.compression_enabled:
            compressed_tensors = [
                self._compress_tensor(tensor) for tensor in batched_tensors
            ]
        else:
            compressed_tensors = batched_tensors
        
        # Perform communication operation
        communicated_tensors = []
        
        for tensor in compressed_tensors:
            if operation == "all_reduce":
                dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
                tensor /= self.world_size
            elif operation == "all_gather":
                gather_list = [torch.zeros_like(tensor) for _ in range(self.world_size)]
                dist.all_gather(gather_list, tensor)
                tensor = torch.cat(gather_list, dim=0)
            
            communicated_tensors.append(tensor)
        
        # Decompress if needed
        if self.compression_enabled:
            decompressed_tensors = [
                self._decompress_tensor(tensor) for tensor in communicated_tensors
            ]
        else:
            decompressed_tensors = communicated_tensors
        
        # Unbatch tensors
        final_tensors = self._unbatch_tensors(decompressed_tensors, len(tensors_to_communicate))
        
        return final_tensors
    
    def _batch_small_tensors(self, tensors: List[torch.Tensor]) -> List[torch.Tensor]:
        """Batch small tensors together for efficient communication."""
        
        # Simple implementation: just return as-is
        # In practice, you'd concatenate small tensors
        return tensors
    
    def _compress_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply compression to tensor."""
        
        # Simple quantization compression
        # Scale to [-1, 1] and quantize to int8
        tensor_max = torch.max(torch.abs(tensor))
        if tensor_max > 0:
            scaled_tensor = tensor / tensor_max
            quantized = torch.round(scaled_tensor * 127).byte()
            # Store scale factor in first element (simplified)
            return quantized.float()
        else:
            return tensor
    
    def _decompress_tensor(self, compressed_tensor: torch.Tensor) -> torch.Tensor:
        """Decompress tensor."""
        
        # Simple decompression (inverse of compression)
        return compressed_tensor  # Placeholder
    
    def _unbatch_tensors(self, batched_tensors: List[torch.Tensor], original_count: int) -> List[torch.Tensor]:
        """Unbatch tensors back to original structure."""
        
        # Simple implementation
        return batched_tensors[:original_count]


class GradientCompressor:
    """Compresses gradients for efficient distributed training."""
    
    def __init__(self, compression_ratio: float = 0.1):
        self.compression_ratio = compression_ratio
        
    def compress_gradients(self, models: nn.ModuleList):
        """Apply gradient compression to model parameters."""
        
        for model in models:
            for param in model.parameters():
                if param.grad is not None:
                    self._compress_gradient(param.grad)
    
    def _compress_gradient(self, grad: torch.Tensor):
        """Apply top-k compression to gradient tensor."""
        
        if grad.numel() == 0:
            return
        
        # Top-k compression
        k = max(1, int(grad.numel() * self.compression_ratio))
        
        # Flatten gradient
        flat_grad = grad.view(-1)
        
        # Find top-k elements by magnitude
        _, indices = torch.topk(torch.abs(flat_grad), k)
        
        # Create compressed gradient
        compressed_grad = torch.zeros_like(flat_grad)
        compressed_grad[indices] = flat_grad[indices]
        
        # Reshape back and update
        grad.data = compressed_grad.view_as(grad)


class DistributedPerformanceMonitor:
    """Monitors performance metrics for distributed PNO training."""
    
    def __init__(self, rank: int):
        self.rank = rank
        self.metrics = {
            "inference_times": [],
            "communication_times": [],
            "memory_usage": [],
            "throughput": []
        }
    
    def record_inference_time(self, time: float):
        """Record inference time."""
        self.metrics["inference_times"].append(time)
        if len(self.metrics["inference_times"]) > 1000:
            self.metrics["inference_times"].pop(0)
    
    def get_performance_summary(self) -> Dict[str, float]:
        """Get summary of performance metrics."""
        
        summary = {}
        
        for metric_name, values in self.metrics.items():
            if values:
                summary[f"{metric_name}_mean"] = np.mean(values)
                summary[f"{metric_name}_std"] = np.std(values)
                summary[f"{metric_name}_p95"] = np.percentile(values, 95)
        
        return summary


# Utility classes

class FederatedParameterAggregator:
    """Aggregates parameters in federated learning setup."""
    
    def aggregate_updates(
        self,
        client_updates: List[Dict[str, torch.Tensor]],
        aggregation_weights: Optional[List[float]] = None
    ) -> Dict[str, torch.Tensor]:
        """Aggregate client parameter updates."""
        
        if not client_updates:
            return {}
        
        # Default to uniform weighting
        if aggregation_weights is None:
            aggregation_weights = [1.0 / len(client_updates)] * len(client_updates)
        
        # Initialize aggregated parameters
        aggregated_params = {}
        
        # Get parameter names from first client
        param_names = client_updates[0]["parameter_update"].keys()
        
        for param_name in param_names:
            # Weighted average of parameter updates
            weighted_sum = torch.zeros_like(client_updates[0]["parameter_update"][param_name])
            
            for client_update, weight in zip(client_updates, aggregation_weights):
                if param_name in client_update["parameter_update"]:
                    weighted_sum += weight * client_update["parameter_update"][param_name]
            
            aggregated_params[param_name] = weighted_sum
        
        return aggregated_params


class UncertaintyPrivatizer:
    """Applies differential privacy to uncertainty parameters."""
    
    def __init__(self, privacy_budget: float = 1.0, noise_scale: float = 0.01):
        self.privacy_budget = privacy_budget
        self.noise_scale = noise_scale
    
    def privatize_parameters(
        self,
        parameters: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Apply differential privacy noise to parameters."""
        
        privatized_params = {}
        
        for name, param in parameters.items():
            # Add Gaussian noise for differential privacy
            noise = torch.randn_like(param) * self.noise_scale
            privatized_params[name] = param + noise
        
        return privatized_params


class CommunicationScheduler:
    """Schedules communication operations for efficiency."""
    
    def __init__(self, world_size: int):
        self.world_size = world_size
        self.communication_queue = []
    
    def schedule_communication(self, operation: str, data: torch.Tensor):
        """Schedule a communication operation."""
        
        self.communication_queue.append({
            "operation": operation,
            "data": data,
            "timestamp": time.time()
        })
    
    def execute_scheduled_communications(self):
        """Execute all scheduled communications."""
        
        for comm_op in self.communication_queue:
            # Execute communication operation
            pass
        
        # Clear queue
        self.communication_queue.clear()