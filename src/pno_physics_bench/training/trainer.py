"""Training system for Probabilistic Neural Operators."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from typing import Dict, List, Optional, Callable, Any, Union
import logging
import time
from pathlib import Path
import wandb
from tqdm import tqdm

from .losses import PNOLoss, ELBOLoss
from .callbacks import Callback
from ..metrics import CalibrationMetrics


logger = logging.getLogger(__name__)


class PNOTrainer:
    """Trainer for Probabilistic Neural Operators with uncertainty quantification."""
    
    def __init__(
        self,
        model: nn.Module,
        loss_fn: Optional[nn.Module] = None,
        optimizer: Optional[Optimizer] = None,
        scheduler: Optional[_LRScheduler] = None,
        device: Optional[str] = None,
        
        # Training parameters
        gradient_clipping: Optional[float] = 1.0,
        mixed_precision: bool = False,
        
        # Uncertainty parameters
        num_samples: int = 5,
        kl_weight: float = 1e-4,
        
        # Logging
        log_interval: int = 10,
        use_wandb: bool = False,
        wandb_project: Optional[str] = None,
        
        # Callbacks
        callbacks: Optional[List[Callback]] = None,
    ):
        """Initialize PNO trainer.
        
        Args:
            model: PNO model to train
            loss_fn: Loss function (defaults to ELBOLoss)
            optimizer: Optimizer (defaults to AdamW)
            scheduler: Learning rate scheduler
            device: Training device
            gradient_clipping: Max gradient norm for clipping
            mixed_precision: Whether to use mixed precision training
            num_samples: Number of MC samples for training
            kl_weight: Weight for KL divergence term
            log_interval: Logging frequency (epochs)
            use_wandb: Whether to use Weights & Biases logging
            wandb_project: W&B project name
            callbacks: List of training callbacks
        """
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Move model to device
        self.model = model.to(self.device)
        
        # Set up loss function
        if loss_fn is None:
            self.loss_fn = ELBOLoss(kl_weight=kl_weight, num_samples=num_samples)
        else:
            self.loss_fn = loss_fn
        
        # Set up optimizer
        if optimizer is None:
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=1e-3,
                weight_decay=1e-4
            )
        else:
            self.optimizer = optimizer
        
        self.scheduler = scheduler
        self.gradient_clipping = gradient_clipping
        self.mixed_precision = mixed_precision
        self.num_samples = num_samples
        
        # Logging setup
        self.log_interval = log_interval
        self.use_wandb = use_wandb
        if use_wandb and wandb_project:
            wandb.init(project=wandb_project)
            wandb.watch(self.model)
        
        # Callbacks
        self.callbacks = callbacks or []
        
        # Mixed precision scaler
        if mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': [],
        }
        
        # Metrics
        self.calibration_metrics = CalibrationMetrics()
        
        logger.info(f"PNO Trainer initialized on {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 100,
        resume_from: Optional[str] = None,
        save_dir: Optional[str] = None,
    ) -> Dict[str, List[float]]:
        """Train the model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of training epochs
            resume_from: Path to checkpoint to resume from
            save_dir: Directory to save checkpoints
            
        Returns:
            Training history dictionary
        """
        if resume_from:
            self.load_checkpoint(resume_from)
        
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Starting training for {epochs} epochs")
        logger.info(f"Training samples: {len(train_loader.dataset)}")
        if val_loader:
            logger.info(f"Validation samples: {len(val_loader.dataset)}")
        
        # Callback: on_train_begin
        for callback in self.callbacks:
            callback.on_train_begin(self)
        
        start_time = time.time()
        
        for epoch in range(self.current_epoch, epochs):
            self.current_epoch = epoch
            
            # Callback: on_epoch_begin
            for callback in self.callbacks:
                callback.on_epoch_begin(epoch, self)
            
            # Training phase
            train_metrics = self._train_epoch(train_loader)
            
            # Validation phase
            val_metrics = {}
            if val_loader is not None:
                val_metrics = self._validate_epoch(val_loader)
            
            # Update learning rate
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    if val_metrics:
                        self.scheduler.step(val_metrics['loss'])
                    else:
                        self.scheduler.step(train_metrics['loss'])
                else:
                    self.scheduler.step()
            
            # Update training history
            self.training_history['train_loss'].append(train_metrics['loss'])
            if val_metrics:
                self.training_history['val_loss'].append(val_metrics['loss'])
            self.training_history['learning_rate'].append(
                self.optimizer.param_groups[0]['lr']
            )
            
            # Logging
            if (epoch + 1) % self.log_interval == 0:
                self._log_metrics(epoch, train_metrics, val_metrics)
            
            # Save checkpoint
            if save_dir and val_metrics and val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                checkpoint_path = save_dir / f"best_model_epoch_{epoch+1}.pt"
                self.save_checkpoint(str(checkpoint_path))
            
            # Callback: on_epoch_end
            for callback in self.callbacks:
                if callback.on_epoch_end(epoch, train_metrics, val_metrics, self):
                    logger.info(f"Training stopped by callback {callback.__class__.__name__}")
                    break
        
        training_time = time.time() - start_time
        logger.info(f"Training completed in {training_time:.2f} seconds")
        
        # Callback: on_train_end
        for callback in self.callbacks:
            callback.on_train_end(self)
        
        return self.training_history
    
    def _train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        epoch_losses = {}
        total_samples = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch+1} [Train]")
        
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            batch_size = inputs.shape[0]
            
            # Callback: on_batch_begin
            for callback in self.callbacks:
                callback.on_batch_begin(batch_idx, inputs, targets, self)
            
            self.optimizer.zero_grad()
            
            if self.mixed_precision:
                with torch.cuda.amp.autocast():
                    # Forward pass with uncertainty sampling
                    if hasattr(self.model, 'forward'):
                        outputs = self.model(inputs, sample=True, return_kl=True)
                    else:
                        outputs = self.model(inputs)
                    
                    # Compute loss
                    losses = self.loss_fn(outputs, targets, self.model)
                
                # Backward pass
                self.scaler.scale(losses['total']).backward()
                
                # Gradient clipping
                if self.gradient_clipping:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.gradient_clipping
                    )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Forward pass
                if hasattr(self.model, 'forward'):
                    outputs = self.model(inputs, sample=True, return_kl=True)
                else:
                    outputs = self.model(inputs)
                
                # Compute loss
                losses = self.loss_fn(outputs, targets, self.model)
                
                # Backward pass
                losses['total'].backward()
                
                # Gradient clipping
                if self.gradient_clipping:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.gradient_clipping
                    )
                
                self.optimizer.step()
            
            # Accumulate losses
            for key, value in losses.items():
                if key not in epoch_losses:
                    epoch_losses[key] = 0.0
                epoch_losses[key] += value.item() * batch_size
            
            total_samples += batch_size
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{losses['total'].item():.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}"
            })
            
            # Callback: on_batch_end
            for callback in self.callbacks:
                callback.on_batch_end(batch_idx, losses, self)
        
        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= total_samples
        
        return epoch_losses
    
    def _validate_epoch(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        
        epoch_losses = {}
        total_samples = 0
        
        # For uncertainty metrics
        all_predictions = []
        all_uncertainties = []
        all_targets = []
        
        pbar = tqdm(val_loader, desc=f"Epoch {self.current_epoch+1} [Val]")
        
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(pbar):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                batch_size = inputs.shape[0]
                
                # Forward pass without sampling for validation
                if hasattr(self.model, 'predict_with_uncertainty'):
                    pred_mean, pred_std = self.model.predict_with_uncertainty(
                        inputs, num_samples=self.num_samples
                    )
                    outputs = (pred_mean, 2 * torch.log(pred_std))  # Convert to log_var
                    
                    # Store for metrics
                    all_predictions.append(pred_mean.cpu())
                    all_uncertainties.append(pred_std.cpu())
                    all_targets.append(targets.cpu())
                else:
                    outputs = self.model(inputs, sample=False)
                    if isinstance(outputs, tuple):
                        all_predictions.append(outputs[0].cpu())
                        if len(outputs) > 1:
                            all_uncertainties.append(torch.exp(0.5 * outputs[1]).cpu())
                        all_targets.append(targets.cpu())
                
                # Compute loss
                losses = self.loss_fn(outputs, targets, self.model)
                
                # Accumulate losses
                for key, value in losses.items():
                    if key not in epoch_losses:
                        epoch_losses[key] = 0.0
                    epoch_losses[key] += value.item() * batch_size
                
                total_samples += batch_size
                
                pbar.set_postfix({'val_loss': f"{losses['total'].item():.4f}"})
        
        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= total_samples
        
        # Compute uncertainty metrics
        if all_predictions and all_uncertainties and all_targets:
            try:
                predictions = torch.cat(all_predictions, dim=0)
                uncertainties = torch.cat(all_uncertainties, dim=0)
                targets = torch.cat(all_targets, dim=0)
                
                # Calibration metrics
                ece = self.calibration_metrics.expected_calibration_error(
                    predictions, uncertainties, targets
                )
                coverage_90 = self.calibration_metrics.coverage_at_confidence(
                    predictions, uncertainties, targets, confidence=0.9
                )
                
                epoch_losses['ece'] = ece
                epoch_losses['coverage_90'] = coverage_90
            except Exception as e:
                logger.warning(f"Failed to compute uncertainty metrics: {e}")
        
        return epoch_losses
    
    def _log_metrics(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float]
    ) -> None:
        """Log training metrics."""
        log_msg = f"Epoch {epoch+1:3d} | "
        log_msg += f"Train Loss: {train_metrics['loss']:.4f} | "
        
        if val_metrics:
            log_msg += f"Val Loss: {val_metrics['loss']:.4f} | "
            if 'ece' in val_metrics:
                log_msg += f"ECE: {val_metrics['ece']:.4f} | "
            if 'coverage_90' in val_metrics:
                log_msg += f"Coverage@90: {val_metrics['coverage_90']:.3f} | "
        
        log_msg += f"LR: {self.optimizer.param_groups[0]['lr']:.2e}"
        
        logger.info(log_msg)
        
        # Weights & Biases logging
        if self.use_wandb:
            log_dict = {
                'epoch': epoch + 1,
                'train/loss': train_metrics['loss'],
                'train/lr': self.optimizer.param_groups[0]['lr'],
            }
            
            # Add detailed training losses
            for key, value in train_metrics.items():
                if key != 'loss':
                    log_dict[f'train/{key}'] = value
            
            # Add validation metrics
            if val_metrics:
                for key, value in val_metrics.items():
                    log_dict[f'val/{key}'] = value
            
            wandb.log(log_dict)
    
    def save_checkpoint(
        self,
        filepath: str,
        include_optimizer: bool = True,
        metadata: Optional[Dict] = None
    ) -> None:
        """Save training checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'best_val_loss': self.best_val_loss,
            'training_history': self.training_history,
        }
        
        if include_optimizer:
            checkpoint['optimizer_state_dict'] = self.optimizer.state_dict()
            if self.scheduler:
                checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        if metadata:
            checkpoint['metadata'] = metadata
        
        torch.save(checkpoint, filepath)
        logger.info(f"Checkpoint saved to {filepath}")
    
    def load_checkpoint(
        self,
        filepath: str,
        load_optimizer: bool = True,
        device: Optional[str] = None
    ) -> None:
        """Load training checkpoint."""
        if device is None:
            device = self.device
        
        checkpoint = torch.load(filepath, map_location=device)
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load training state
        self.current_epoch = checkpoint.get('epoch', 0)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.training_history = checkpoint.get('training_history', {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': [],
        })
        
        # Load optimizer state
        if load_optimizer and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if load_optimizer and self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        logger.info(f"Checkpoint loaded from {filepath}")
        logger.info(f"Resuming from epoch {self.current_epoch}")
    
    def evaluate(
        self,
        test_loader: DataLoader,
        num_uncertainty_samples: int = 100
    ) -> Dict[str, float]:
        """Evaluate model on test set with comprehensive metrics."""
        self.model.eval()
        
        all_predictions = []
        all_uncertainties = []
        all_targets = []
        total_loss = 0.0
        total_samples = 0
        
        logger.info("Evaluating model...")
        
        with torch.no_grad():
            for inputs, targets in tqdm(test_loader, desc="Evaluation"):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                batch_size = inputs.shape[0]
                
                # Get predictions with uncertainty
                if hasattr(self.model, 'predict_with_uncertainty'):
                    pred_mean, pred_std = self.model.predict_with_uncertainty(
                        inputs, num_samples=num_uncertainty_samples
                    )
                else:
                    outputs = self.model(inputs, sample=False)
                    if isinstance(outputs, tuple):
                        pred_mean, pred_log_var = outputs[:2]
                        pred_std = torch.exp(0.5 * pred_log_var)
                    else:
                        pred_mean = outputs
                        pred_std = torch.zeros_like(pred_mean)
                
                # Compute loss
                if hasattr(self.model, 'predict_with_uncertainty'):
                    outputs = (pred_mean, 2 * torch.log(pred_std + 1e-8))
                else:
                    outputs = (pred_mean, pred_log_var) if isinstance(outputs, tuple) else pred_mean
                
                losses = self.loss_fn(outputs, targets, self.model)
                total_loss += losses['total'].item() * batch_size
                total_samples += batch_size
                
                # Store predictions
                all_predictions.append(pred_mean.cpu())
                all_uncertainties.append(pred_std.cpu())
                all_targets.append(targets.cpu())
        
        # Concatenate all results
        predictions = torch.cat(all_predictions, dim=0)
        uncertainties = torch.cat(all_uncertainties, dim=0)
        targets = torch.cat(all_targets, dim=0)
        
        # Compute comprehensive metrics
        metrics = {
            'test_loss': total_loss / total_samples,
            'rmse': torch.sqrt(torch.mean((predictions - targets) ** 2)).item(),
            'mae': torch.mean(torch.abs(predictions - targets)).item(),
        }
        
        # Uncertainty metrics
        try:
            metrics.update(self.calibration_metrics.compute_all_metrics(
                predictions, uncertainties, targets
            ))
        except Exception as e:
            logger.warning(f"Failed to compute uncertainty metrics: {e}")
        
        # Log results
        logger.info("Evaluation Results:")
        for key, value in metrics.items():
            logger.info(f"  {key}: {value:.4f}")
        
        return metrics