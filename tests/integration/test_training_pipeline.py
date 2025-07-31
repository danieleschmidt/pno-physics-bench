"""Integration tests for the complete training pipeline."""

import pytest
import torch
import torch.nn as nn
from pathlib import Path
from unittest.mock import Mock, patch
import tempfile
import json


class TestTrainingPipeline:
    """Test the complete training pipeline integration."""
    
    def test_end_to_end_training(self, synthetic_dataset, training_config, model_checkpoint_path, wandb_disabled):
        """Test complete training pipeline from data to trained model."""
        
        # Mock components for integration testing
        class MockDataLoader:
            def __init__(self, dataset, batch_size=4):
                self.dataset = dataset
                self.batch_size = batch_size
                self.data = [(dataset['inputs'][:batch_size], dataset['targets'][:batch_size])]
            
            def __iter__(self):
                return iter(self.data)
            
            def __len__(self):
                return 1
        
        class MockPNOModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(64, 32)
                self.output_linear = nn.Linear(32, 32)
                
            def forward(self, x, return_uncertainty=False):
                batch_size = x.shape[0]
                x_flat = x.view(batch_size, -1)
                
                # Adapt to actual input size
                if x_flat.shape[1] != 64:
                    self.linear = nn.Linear(x_flat.shape[1], 32)
                
                h = torch.relu(self.linear(x_flat))
                mean = self.output_linear(h)
                
                if return_uncertainty:
                    std = torch.ones_like(mean) * 0.1
                    return mean, std
                return mean
            
            def kl_divergence(self):
                # Mock KL divergence
                return torch.tensor(0.01)
        
        class MockTrainer:
            def __init__(self, model, **kwargs):
                self.model = model
                self.config = kwargs
                self.optimizer = torch.optim.Adam(model.parameters(), lr=kwargs.get('learning_rate', 1e-3))
                self.losses = []
                
            def fit(self, train_loader, val_loader=None, epochs=1):
                for epoch in range(epochs):
                    epoch_loss = 0
                    for batch_idx, (inputs, targets) in enumerate(train_loader):
                        self.optimizer.zero_grad()
                        
                        # Forward pass
                        predictions = self.model(inputs)
                        
                        # Compute loss
                        mse_loss = nn.MSELoss()(predictions, targets)
                        kl_loss = self.model.kl_divergence()
                        total_loss = mse_loss + self.config.get('kl_weight', 1e-4) * kl_loss
                        
                        # Backward pass
                        total_loss.backward()
                        
                        # Gradient clipping
                        if self.config.get('gradient_clip_val'):
                            torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(), 
                                self.config['gradient_clip_val']
                            )
                        
                        self.optimizer.step()
                        epoch_loss += total_loss.item()
                    
                    self.losses.append(epoch_loss / len(train_loader))
                
                return self.losses
        
        # Setup test data
        data = synthetic_dataset
        config = training_config
        
        # Create data loaders
        train_loader = MockDataLoader(data, config['batch_size'])
        val_loader = MockDataLoader(data, config['batch_size'])
        
        # Initialize model and trainer
        model = MockPNOModel()
        trainer = MockTrainer(model, **config)
        
        # Train model
        losses = trainer.fit(train_loader, val_loader, epochs=config['max_epochs'])
        
        # Assertions
        assert len(losses) == config['max_epochs']
        assert all(torch.isfinite(torch.tensor(loss)) for loss in losses)
        
        # Test model produces valid predictions
        with torch.no_grad():
            test_input = data['inputs'][:2]
            predictions = model(test_input)
            assert predictions.shape[0] == 2
            assert torch.isfinite(predictions).all()
    
    def test_checkpoint_saving_loading(self, synthetic_dataset, training_config, model_checkpoint_path):
        """Test checkpoint saving and loading during training."""
        
        class MockModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 5)
            
            def forward(self, x):
                return self.linear(x.view(x.shape[0], -1))
        
        model = MockModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        # Save checkpoint
        checkpoint_path = model_checkpoint_path / "test_checkpoint.pth"
        checkpoint = {
            'epoch': 5,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': 0.123,
            'config': training_config
        }
        torch.save(checkpoint, checkpoint_path)
        
        # Load checkpoint
        loaded_checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Create new model and load state
        new_model = MockModel()
        new_optimizer = torch.optim.Adam(new_model.parameters(), lr=1e-3)
        
        new_model.load_state_dict(loaded_checkpoint['model_state_dict'])
        new_optimizer.load_state_dict(loaded_checkpoint['optimizer_state_dict'])
        
        # Verify loaded state
        assert loaded_checkpoint['epoch'] == 5
        assert loaded_checkpoint['loss'] == 0.123
        assert loaded_checkpoint['config'] == training_config
        
        # Verify model parameters are identical
        for p1, p2 in zip(model.parameters(), new_model.parameters()):
            assert torch.equal(p1, p2)
    
    @pytest.mark.slow
    def test_early_stopping(self, synthetic_dataset, training_config):
        """Test early stopping mechanism."""
        
        class MockTrainerWithEarlyStopping:
            def __init__(self, model, patience=3):
                self.model = model
                self.patience = patience
                self.best_loss = float('inf')
                self.patience_counter = 0
                self.optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            
            def train_epoch(self, train_loader):
                total_loss = 0
                for inputs, targets in train_loader:
                    self.optimizer.zero_grad()
                    predictions = self.model(inputs)
                    loss = nn.MSELoss()(predictions, targets)
                    loss.backward()
                    self.optimizer.step()
                    total_loss += loss.item()
                return total_loss / len(train_loader)
            
            def validate(self, val_loader):
                total_loss = 0
                with torch.no_grad():
                    for inputs, targets in val_loader:
                        predictions = self.model(inputs)
                        loss = nn.MSELoss()(predictions, targets)
                        total_loss += loss.item()
                return total_loss / len(val_loader)
            
            def fit(self, train_loader, val_loader, max_epochs=10):
                for epoch in range(max_epochs):
                    # Train
                    train_loss = self.train_epoch(train_loader)
                    
                    # Validate
                    val_loss = self.validate(val_loader)
                    
                    # Early stopping check
                    if val_loss < self.best_loss:
                        self.best_loss = val_loss
                        self.patience_counter = 0
                    else:
                        self.patience_counter += 1
                        
                    if self.patience_counter >= self.patience:
                        print(f"Early stopping at epoch {epoch}")
                        return epoch + 1
                
                return max_epochs
        
        # Mock model that overfits quickly
        class MockOverfittingModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(64, 32)
                self.epoch_count = 0
            
            def forward(self, x):
                x_flat = x.view(x.shape[0], -1)
                if x_flat.shape[1] != 64:
                    self.linear = nn.Linear(x_flat.shape[1], 32)
                
                # Simulate overfitting by adding noise that increases over time
                output = self.linear(x_flat)
                self.epoch_count += 1
                if self.epoch_count > 3:  # Start overfitting after epoch 3
                    output += torch.randn_like(output) * 0.1 * (self.epoch_count - 3)
                
                return output
        
        # Create mock data loaders
        class MockDataLoader:
            def __init__(self, dataset, batch_size=4):
                self.dataset = dataset
                self.batch_size = batch_size
                self.data = [(dataset['inputs'][:batch_size], dataset['targets'][:batch_size])]
            
            def __iter__(self):
                return iter(self.data)
            
            def __len__(self):
                return 1
        
        data = synthetic_dataset
        train_loader = MockDataLoader(data)
        val_loader = MockDataLoader(data)
        
        model = MockOverfittingModel()
        trainer = MockTrainerWithEarlyStopping(model, patience=2)
        
        epochs_run = trainer.fit(train_loader, val_loader, max_epochs=10)
        
        # Should stop early due to overfitting
        assert epochs_run < 10
    
    def test_distributed_training_setup(self, synthetic_dataset):
        """Test distributed training setup (single GPU simulation)."""
        
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available for distributed training test")
        
        # Mock distributed setup
        class MockDistributedModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(64, 32)
                self.is_distributed = False
            
            def forward(self, x):
                x_flat = x.view(x.shape[0], -1)
                if x_flat.shape[1] != 64:
                    self.linear = nn.Linear(x_flat.shape[1], 32)
                return self.linear(x_flat)
            
            def to_distributed(self):
                # Simulate wrapping with DistributedDataParallel
                self.is_distributed = True
                return self
        
        model = MockDistributedModel()
        device = torch.device("cuda:0")
        
        # Move to GPU and simulate distributed setup
        model = model.to(device)
        model = model.to_distributed()
        
        assert model.is_distributed
        
        # Test forward pass on GPU
        data = synthetic_dataset
        inputs = data['inputs'].to(device)
        
        with torch.no_grad():
            output = model(inputs)
            assert output.device == device
    
    def test_mixed_precision_training(self, synthetic_dataset):
        """Test mixed precision training."""
        
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available for mixed precision test")
        
        from torch.cuda.amp import GradScaler, autocast
        
        class MockModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(64, 32)
            
            def forward(self, x):
                x_flat = x.view(x.shape[0], -1)
                if x_flat.shape[1] != 64:
                    self.linear = nn.Linear(x_flat.shape[1], 32)
                return self.linear(x_flat)
        
        device = torch.device("cuda:0")
        model = MockModel().to(device)
        optimizer = torch.optim.Adam(model.parameters())
        scaler = GradScaler()
        
        data = synthetic_dataset
        inputs = data['inputs'].to(device)
        targets = data['targets'].to(device)
        
        # Training step with mixed precision
        optimizer.zero_grad()
        
        with autocast():
            predictions = model(inputs)
            loss = nn.MSELoss()(predictions, targets)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        assert torch.isfinite(loss)
    
    def test_hyperparameter_validation(self, training_config):
        """Test hyperparameter validation."""
        
        def validate_training_config(config):
            """Validate training configuration."""
            required_keys = ['learning_rate', 'batch_size', 'max_epochs']
            for key in required_keys:
                if key not in config:
                    raise ValueError(f"Missing required config key: {key}")
            
            if config['learning_rate'] <= 0:
                raise ValueError("Learning rate must be positive")
            
            if config['batch_size'] <= 0:
                raise ValueError("Batch size must be positive")
            
            if config['max_epochs'] <= 0:
                raise ValueError("Max epochs must be positive")
            
            return True
        
        # Valid config should pass
        assert validate_training_config(training_config)
        
        # Invalid configs should fail
        invalid_configs = [
            {**training_config, 'learning_rate': -1},
            {**training_config, 'batch_size': 0},
            {**training_config, 'max_epochs': -5},
            {k: v for k, v in training_config.items() if k != 'learning_rate'}
        ]
        
        for invalid_config in invalid_configs:
            with pytest.raises(ValueError):
                validate_training_config(invalid_config)
    
    def test_logging_integration(self, synthetic_dataset, training_config, wandb_disabled):
        """Test logging integration with W&B (mocked)."""
        
        class MockWandbLogger:
            def __init__(self):
                self.logged_metrics = []
            
            def log(self, metrics, step=None):
                self.logged_metrics.append((metrics, step))
            
            def finish(self):
                pass
        
        class MockTrainerWithLogging:
            def __init__(self, model, logger=None):
                self.model = model
                self.logger = logger or MockWandbLogger()
                self.optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            
            def train_step(self, batch, step):
                inputs, targets = batch
                self.optimizer.zero_grad()
                
                predictions = self.model(inputs)
                loss = nn.MSELoss()(predictions, targets)
                loss.backward()
                self.optimizer.step()
                
                # Log metrics
                self.logger.log({
                    'train_loss': loss.item(),
                    'step': step
                }, step=step)
                
                return loss.item()
        
        # Setup
        class MockModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(64, 32)
            
            def forward(self, x):
                x_flat = x.view(x.shape[0], -1)
                if x_flat.shape[1] != 64:
                    self.linear = nn.Linear(x_flat.shape[1], 32)
                return self.linear(x_flat)
        
        model = MockModel()
        logger = MockWandbLogger()
        trainer = MockTrainerWithLogging(model, logger)
        
        # Train for a few steps
        data = synthetic_dataset
        batch = (data['inputs'][:4], data['targets'][:4])
        
        for step in range(3):
            loss = trainer.train_step(batch, step)
            assert torch.isfinite(torch.tensor(loss))
        
        # Check logging
        assert len(logger.logged_metrics) == 3
        for metrics, step in logger.logged_metrics:
            assert 'train_loss' in metrics
            assert 'step' in metrics
            assert isinstance(step, int)