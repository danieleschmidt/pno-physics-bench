"""Training callbacks for PNO training."""

import torch
import numpy as np
from typing import Dict, Optional, Any, List
import logging
from pathlib import Path
import matplotlib.pyplot as plt
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


logger = logging.getLogger(__name__)


class Callback:
    """Base class for training callbacks."""
    
    def on_train_begin(self, trainer) -> None:
        """Called at the beginning of training."""
        pass
    
    def on_train_end(self, trainer) -> None:
        """Called at the end of training."""
        pass
    
    def on_epoch_begin(self, epoch: int, trainer) -> None:
        """Called at the beginning of each epoch."""
        pass
    
    def on_epoch_end(
        self, 
        epoch: int, 
        train_metrics: Dict[str, float], 
        val_metrics: Dict[str, float], 
        trainer
    ) -> bool:
        """Called at the end of each epoch.
        
        Returns:
            True if training should stop, False to continue
        """
        return False
    
    def on_batch_begin(self, batch_idx: int, inputs: torch.Tensor, targets: torch.Tensor, trainer) -> None:
        """Called at the beginning of each batch."""
        pass
    
    def on_batch_end(self, batch_idx: int, losses: Dict[str, torch.Tensor], trainer) -> None:
        """Called at the end of each batch."""
        pass


class EarlyStopping(Callback):
    """Early stopping callback to prevent overfitting."""
    
    def __init__(
        self,
        monitor: str = 'val_loss',
        patience: int = 10,
        min_delta: float = 1e-6,
        restore_best_weights: bool = True,
        verbose: bool = True,
    ):
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.verbose = verbose
        
        self.best_value = None
        self.best_weights = None
        self.wait = 0
        self.stopped_epoch = 0
    
    def on_train_begin(self, trainer) -> None:
        self.best_value = np.inf if 'loss' in self.monitor else -np.inf
        self.wait = 0
        self.stopped_epoch = 0
        if self.restore_best_weights:
            self.best_weights = trainer.model.state_dict().copy()
    
    def on_epoch_end(
        self, 
        epoch: int, 
        train_metrics: Dict[str, float], 
        val_metrics: Dict[str, float], 
        trainer
    ) -> bool:
        # Get monitored value
        if self.monitor.startswith('val_'):
            metric_name = self.monitor[4:]  # Remove 'val_' prefix
            current_value = val_metrics.get(metric_name, val_metrics.get('loss', np.inf))
        else:
            metric_name = self.monitor
            current_value = train_metrics.get(metric_name, train_metrics.get('loss', np.inf))
        
        # Check for improvement
        if 'loss' in self.monitor or 'error' in self.monitor:
            # Lower is better
            if current_value < self.best_value - self.min_delta:
                self.best_value = current_value
                self.wait = 0
                if self.restore_best_weights:
                    self.best_weights = trainer.model.state_dict().copy()
            else:
                self.wait += 1
        else:
            # Higher is better
            if current_value > self.best_value + self.min_delta:
                self.best_value = current_value
                self.wait = 0
                if self.restore_best_weights:
                    self.best_weights = trainer.model.state_dict().copy()
            else:
                self.wait += 1
        
        # Check if should stop
        if self.wait >= self.patience:
            self.stopped_epoch = epoch
            if self.verbose:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                logger.info(f"Best {self.monitor}: {self.best_value:.6f}")
            
            if self.restore_best_weights and self.best_weights:
                trainer.model.load_state_dict(self.best_weights)
                if self.verbose:
                    logger.info("Restored best model weights")
            
            return True
        
        return False


class ModelCheckpoint(Callback):
    """Save model checkpoints during training."""
    
    def __init__(
        self,
        filepath: str,
        monitor: str = 'val_loss',
        save_best_only: bool = True,
        save_weights_only: bool = False,
        verbose: bool = True,
    ):
        self.filepath = filepath
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.verbose = verbose
        
        self.best_value = None
    
    def on_train_begin(self, trainer) -> None:
        self.best_value = np.inf if 'loss' in self.monitor else -np.inf
        
        # Create directory if it doesn't exist
        Path(self.filepath).parent.mkdir(parents=True, exist_ok=True)
    
    def on_epoch_end(
        self, 
        epoch: int, 
        train_metrics: Dict[str, float], 
        val_metrics: Dict[str, float], 
        trainer
    ) -> bool:
        # Get monitored value
        if self.monitor.startswith('val_'):
            metric_name = self.monitor[4:]  # Remove 'val_' prefix
            current_value = val_metrics.get(metric_name, val_metrics.get('loss', np.inf))
        else:
            metric_name = self.monitor
            current_value = train_metrics.get(metric_name, train_metrics.get('loss', np.inf))
        
        # Check if should save
        should_save = False
        if not self.save_best_only:
            should_save = True
        else:
            if 'loss' in self.monitor or 'error' in self.monitor:
                # Lower is better
                if current_value < self.best_value:
                    self.best_value = current_value
                    should_save = True
            else:
                # Higher is better
                if current_value > self.best_value:
                    self.best_value = current_value
                    should_save = True
        
        if should_save:
            filepath = self.filepath.format(epoch=epoch + 1, **train_metrics, **val_metrics)
            
            if self.save_weights_only:
                torch.save(trainer.model.state_dict(), filepath)
            else:
                trainer.save_checkpoint(filepath)
            
            if self.verbose:
                logger.info(f"Saved checkpoint to {filepath}")
        
        return False


class UncertaintyVisualization(Callback):
    """Visualize uncertainty predictions during training."""
    
    def __init__(
        self,
        val_dataset,
        save_dir: str,
        frequency: int = 10,
        num_samples: int = 3,
        num_uncertainty_samples: int = 50,
    ):
        self.val_dataset = val_dataset
        self.save_dir = Path(save_dir)
        self.frequency = frequency
        self.num_samples = num_samples
        self.num_uncertainty_samples = num_uncertainty_samples
        
        # Create save directory
        self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def on_epoch_end(
        self, 
        epoch: int, 
        train_metrics: Dict[str, float], 
        val_metrics: Dict[str, float], 
        trainer
    ) -> bool:
        if (epoch + 1) % self.frequency != 0:
            return False
        
        trainer.model.eval()
        
        with torch.no_grad():
            for i in range(self.num_samples):
                # Get sample
                inputs, targets = self.val_dataset[i]
                inputs = inputs.unsqueeze(0).to(trainer.device)
                targets = targets.unsqueeze(0).to(trainer.device)
                
                # Get predictions with uncertainty
                if hasattr(trainer.model, 'predict_with_uncertainty'):
                    pred_mean, pred_std = trainer.model.predict_with_uncertainty(
                        inputs, num_samples=self.num_uncertainty_samples
                    )
                else:
                    outputs = trainer.model(inputs, sample=False)
                    if isinstance(outputs, tuple):
                        pred_mean, pred_log_var = outputs[:2]
                        pred_std = torch.exp(0.5 * pred_log_var)
                    else:
                        pred_mean = outputs
                        pred_std = torch.zeros_like(pred_mean)
                
                # Create visualization
                fig = self._create_uncertainty_plot(
                    inputs[0].cpu(), targets[0].cpu(), pred_mean[0].cpu(), pred_std[0].cpu()
                )
                
                # Save plot
                save_path = self.save_dir / f"uncertainty_epoch_{epoch+1:03d}_sample_{i}.png"
                fig.savefig(save_path, dpi=150, bbox_inches='tight')
                plt.close(fig)
        
        return False
    
    def _create_uncertainty_plot(
        self, 
        inputs: torch.Tensor, 
        targets: torch.Tensor, 
        pred_mean: torch.Tensor, 
        pred_std: torch.Tensor
    ):
        """Create uncertainty visualization plot."""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Handle different dimensionalities
        if inputs.dim() == 3:  # (H, W, C)
            input_vis = inputs[..., 0] if inputs.shape[-1] > 1 else inputs.squeeze(-1)
            target_vis = targets[..., 0] if targets.shape[-1] > 1 else targets.squeeze(-1)
            pred_vis = pred_mean[..., 0] if pred_mean.shape[-1] > 1 else pred_mean.squeeze(-1)
            std_vis = pred_std[..., 0] if pred_std.shape[-1] > 1 else pred_std.squeeze(-1)
        else:  # 1D case
            input_vis = inputs
            target_vis = targets
            pred_vis = pred_mean
            std_vis = pred_std
        
        # Plot input
        if input_vis.dim() == 2:
            im1 = axes[0, 0].imshow(input_vis, cmap='viridis')
            axes[0, 0].set_title('Input')
            plt.colorbar(im1, ax=axes[0, 0])
        else:
            axes[0, 0].plot(input_vis)
            axes[0, 0].set_title('Input')
        
        # Plot target
        if target_vis.dim() == 2:
            im2 = axes[0, 1].imshow(target_vis, cmap='viridis')
            axes[0, 1].set_title('Target')
            plt.colorbar(im2, ax=axes[0, 1])
        else:
            axes[0, 1].plot(target_vis)
            axes[0, 1].set_title('Target')
        
        # Plot prediction
        if pred_vis.dim() == 2:
            im3 = axes[0, 2].imshow(pred_vis, cmap='viridis')
            axes[0, 2].set_title('Prediction (Mean)')
            plt.colorbar(im3, ax=axes[0, 2])
        else:
            axes[0, 2].plot(pred_vis)
            axes[0, 2].set_title('Prediction (Mean)')
        
        # Plot uncertainty
        if std_vis.dim() == 2:
            im4 = axes[1, 0].imshow(std_vis, cmap='hot')
            axes[1, 0].set_title('Uncertainty (Std)')
            plt.colorbar(im4, ax=axes[1, 0])
        else:
            axes[1, 0].plot(std_vis)
            axes[1, 0].set_title('Uncertainty (Std)')
        
        # Plot error
        error = torch.abs(pred_vis - target_vis)
        if error.dim() == 2:
            im5 = axes[1, 1].imshow(error, cmap='Reds')
            axes[1, 1].set_title('Absolute Error')
            plt.colorbar(im5, ax=axes[1, 1])
        else:
            axes[1, 1].plot(error)
            axes[1, 1].set_title('Absolute Error')
        
        # Plot uncertainty vs error scatter
        if error.dim() == 2:
            error_flat = error.flatten()
            std_flat = std_vis.flatten()
            axes[1, 2].scatter(std_flat, error_flat, alpha=0.5, s=1)
            axes[1, 2].set_xlabel('Predicted Uncertainty')
            axes[1, 2].set_ylabel('Actual Error')
            axes[1, 2].set_title('Uncertainty vs Error')
            
            # Add perfect calibration line
            max_val = max(error_flat.max(), std_flat.max())
            axes[1, 2].plot([0, max_val], [0, max_val], 'r--', alpha=0.7, label='Perfect Calibration')
            axes[1, 2].legend()
        
        plt.tight_layout()
        return fig


class MetricsLogger(Callback):
    """Log detailed metrics during training."""
    
    def __init__(
        self,
        log_file: Optional[str] = None,
        log_to_wandb: bool = False,
    ):
        self.log_file = log_file
        self.log_to_wandb = log_to_wandb
        
        if log_file:
            self.log_path = Path(log_file)
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write header
            with open(self.log_path, 'w') as f:
                f.write("epoch,train_loss,val_loss,learning_rate,ece,coverage_90\n")
    
    def on_epoch_end(
        self, 
        epoch: int, 
        train_metrics: Dict[str, float], 
        val_metrics: Dict[str, float], 
        trainer
    ) -> bool:
        # Log to file
        if self.log_file:
            with open(self.log_path, 'a') as f:
                lr = trainer.optimizer.param_groups[0]['lr']
                val_loss = val_metrics.get('loss', 0.0)
                ece = val_metrics.get('ece', 0.0)
                coverage_90 = val_metrics.get('coverage_90', 0.0)
                
                f.write(f"{epoch+1},{train_metrics['loss']:.6f},{val_loss:.6f},"
                       f"{lr:.2e},{ece:.6f},{coverage_90:.6f}\n")
        
        # Log to W&B
        if self.log_to_wandb and HAS_WANDB and wandb.run is not None:
            log_dict = {
                'epoch': epoch + 1,
                'learning_rate': trainer.optimizer.param_groups[0]['lr'],
            }
            
            # Add all metrics with prefixes
            for key, value in train_metrics.items():
                log_dict[f'train/{key}'] = value
            
            for key, value in val_metrics.items():
                log_dict[f'val/{key}'] = value
            
            wandb.log(log_dict)
        
        return False


class LearningRateScheduler(Callback):
    """Custom learning rate scheduling callback."""
    
    def __init__(
        self,
        schedule_type: str = 'cosine',
        warmup_epochs: int = 5,
        max_lr: float = 1e-3,
        min_lr: float = 1e-6,
        cycle_length: Optional[int] = None,
    ):
        self.schedule_type = schedule_type
        self.warmup_epochs = warmup_epochs
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.cycle_length = cycle_length
        
        self.initial_lr = None
    
    def on_train_begin(self, trainer) -> None:
        self.initial_lr = trainer.optimizer.param_groups[0]['lr']
    
    def on_epoch_begin(self, epoch: int, trainer) -> None:
        if self.schedule_type == 'warmup_cosine':
            if epoch < self.warmup_epochs:
                # Warmup phase
                lr = self.min_lr + (self.max_lr - self.min_lr) * epoch / self.warmup_epochs
            else:
                # Cosine decay
                remaining_epochs = epoch - self.warmup_epochs
                lr = self.min_lr + (self.max_lr - self.min_lr) * 0.5 * (
                    1 + np.cos(np.pi * remaining_epochs / (100 - self.warmup_epochs))
                )
        elif self.schedule_type == 'cosine_restart':
            if self.cycle_length is None:
                self.cycle_length = 50
            
            cycle_epoch = epoch % self.cycle_length
            lr = self.min_lr + (self.max_lr - self.min_lr) * 0.5 * (
                1 + np.cos(np.pi * cycle_epoch / self.cycle_length)
            )
        else:
            return  # Use default scheduler
        
        # Update learning rate
        for param_group in trainer.optimizer.param_groups:
            param_group['lr'] = lr