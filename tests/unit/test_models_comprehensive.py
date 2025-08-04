"""Comprehensive unit tests for PNO models."""

import pytest
import torch
import torch.nn as nn
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import warnings
import tempfile
from pathlib import Path

from src.pno_physics_bench.models import (
    ProbabilisticNeuralOperator,
    FourierNeuralOperator, 
    DeepONet
)
from src.pno_physics_bench.models.layers import (
    SpectralConv2d_Probabilistic,
    PNOBlock,
    LiftProjectionLayer
)
from src.pno_physics_bench.exceptions import ModelError, ValidationError


class TestSpectralConv2dProbabilistic:
    """Test probabilistic spectral convolution layer."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.layer = SpectralConv2d_Probabilistic(
            in_channels=2,
            out_channels=4,
            modes1=8,
            modes2=8,
            prior_std=1.0
        )
        self.input_tensor = torch.randn(2, 2, 32, 32)
    
    def test_initialization(self):
        """Test layer initialization."""
        assert self.layer.in_channels == 2
        assert self.layer.out_channels == 4
        assert self.layer.modes1 == 8
        assert self.layer.modes2 == 8
        assert self.layer.prior_std == 1.0
        
        # Check parameter shapes
        assert self.layer.weights1_mean.shape == (2, 4, 8, 8)
        assert self.layer.weights2_mean.shape == (2, 4, 8, 8)
        assert self.layer.weights1_log_var.shape == (2, 4, 8, 8)
        assert self.layer.weights2_log_var.shape == (2, 4, 8, 8)
    
    def test_forward_sampling(self):
        """Test forward pass with sampling."""
        output = self.layer(self.input_tensor, sample=True)
        
        assert output.shape == (2, 4, 32, 32)
        assert output.dtype == torch.float32
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_forward_no_sampling(self):
        """Test forward pass without sampling."""
        output = self.layer(self.input_tensor, sample=False)
        
        assert output.shape == (2, 4, 32, 32)
        assert output.dtype == torch.float32
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_kl_divergence(self):
        """Test KL divergence computation."""
        kl = self.layer.kl_divergence()
        
        assert isinstance(kl, torch.Tensor)
        assert kl.shape == ()
        assert kl.item() >= 0
        assert not torch.isnan(kl)
        assert not torch.isinf(kl)
    
    def test_different_modes(self):
        """Test with different mode sizes."""
        layer = SpectralConv2d_Probabilistic(2, 4, 4, 6)
        output = layer(self.input_tensor)
        assert output.shape == (2, 4, 32, 32)
    
    def test_training_vs_eval_mode(self):
        """Test behavior difference between training and eval modes."""
        self.layer.train()
        output_train = self.layer(self.input_tensor, sample=True)
        
        self.layer.eval()
        output_eval = self.layer(self.input_tensor, sample=True)
        
        # Outputs should be different due to sampling
        assert not torch.allclose(output_train, output_eval, atol=1e-6)


class TestPNOBlock:
    """Test PNO block."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.block = PNOBlock(
            in_channels=4,
            out_channels=4,
            modes1=8,
            modes2=8,
            activation="gelu"
        )
        self.input_tensor = torch.randn(2, 4, 32, 32)
    
    def test_initialization(self):
        """Test block initialization."""
        assert isinstance(self.block.conv, SpectralConv2d_Probabilistic)
        assert isinstance(self.block.w_mean, nn.Conv2d)
        assert isinstance(self.block.w_log_var, nn.Conv2d)
    
    def test_forward(self):
        """Test forward pass."""
        output = self.block(self.input_tensor)
        
        assert output.shape == (2, 4, 32, 32)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_kl_divergence(self):
        """Test KL divergence computation."""
        kl = self.block.kl_divergence()
        
        assert isinstance(kl, torch.Tensor)
        assert kl.item() >= 0
        assert not torch.isnan(kl)
    
    def test_different_activations(self):
        """Test different activation functions."""
        activations = ["gelu", "relu", "tanh"]
        
        for activation in activations:
            block = PNOBlock(4, 4, 8, 8, activation=activation)
            output = block(self.input_tensor)
            assert output.shape == (2, 4, 32, 32)
    
    def test_invalid_activation(self):
        """Test invalid activation function."""
        with pytest.raises(ValueError):
            PNOBlock(4, 4, 8, 8, activation="invalid")


class TestLiftProjectionLayer:
    """Test lift/projection layer."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.layer = LiftProjectionLayer(
            in_dim=3,
            out_dim=64,
            hidden_dim=32,
            probabilistic=True
        )
        self.input_tensor = torch.randn(2, 16, 16, 3)
    
    def test_probabilistic_forward(self):
        """Test probabilistic forward pass."""
        output = self.layer(self.input_tensor)
        
        assert output.shape == (2, 16, 16, 64)
        assert not torch.isnan(output).any()
    
    def test_deterministic_forward(self):
        """Test deterministic forward pass."""
        layer = LiftProjectionLayer(3, 64, 32, probabilistic=False)
        output = layer(self.input_tensor)
        
        assert output.shape == (2, 16, 16, 64)
        assert not torch.isnan(output).any()
    
    def test_kl_divergence_probabilistic(self):
        """Test KL divergence for probabilistic layer."""
        kl = self.layer.kl_divergence()
        assert kl.item() >= 0
    
    def test_kl_divergence_deterministic(self):
        """Test KL divergence for deterministic layer."""
        layer = LiftProjectionLayer(3, 64, 32, probabilistic=False)
        kl = layer.kl_divergence()
        assert kl.item() == 0.0


class TestProbabilisticNeuralOperator:
    """Test main PNO model."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.model = ProbabilisticNeuralOperator(
            input_dim=2,
            hidden_dim=32,
            num_layers=2,
            modes=8,
            output_dim=1,
            uncertainty_type="diagonal"
        )
        self.input_tensor = torch.randn(4, 16, 16, 2)
    
    def test_initialization(self):
        """Test model initialization."""
        assert self.model.input_dim == 2
        assert self.model.hidden_dim == 32
        assert self.model.num_layers == 2
        assert self.model.modes == 8
        assert self.model.output_dim == 1
        assert self.model.uncertainty_type == "diagonal"
    
    def test_forward_with_uncertainty(self):
        """Test forward pass with uncertainty output."""
        output = self.model(self.input_tensor, sample=True, return_kl=False)
        
        assert isinstance(output, tuple)
        assert len(output) == 2
        
        mean, log_var = output
        assert mean.shape == (4, 16, 16, 1)
        assert log_var.shape == (4, 16, 16, 1)
        assert not torch.isnan(mean).any()
        assert not torch.isnan(log_var).any()
    
    def test_forward_with_kl(self):
        """Test forward pass with KL divergence."""
        output = self.model(self.input_tensor, sample=True, return_kl=True)
        
        assert isinstance(output, tuple)
        assert len(output) == 3
        
        mean, log_var, kl = output
        assert mean.shape == (4, 16, 16, 1)
        assert log_var.shape == (4, 16, 16, 1)
        assert isinstance(kl, torch.Tensor)
        assert kl.item() >= 0
    
    def test_predict_with_uncertainty(self):
        """Test uncertainty prediction."""
        mean, std = self.model.predict_with_uncertainty(
            self.input_tensor, num_samples=10
        )
        
        assert mean.shape == (4, 16, 16, 1)
        assert std.shape == (4, 16, 16, 1)
        assert (std >= 0).all()
        assert not torch.isnan(mean).any()
        assert not torch.isnan(std).any()
    
    def test_predict_distributional(self):
        """Test distributional prediction."""
        mean, total_std = self.model.predict_distributional(
            self.input_tensor, num_samples=10
        )
        
        assert mean.shape == (4, 16, 16, 1)
        assert total_std.shape == (4, 16, 16, 1)
        assert (total_std >= 0).all()
    
    def test_decompose_uncertainty(self):
        """Test uncertainty decomposition."""
        uncertainties = self.model.decompose_uncertainty(
            self.input_tensor, num_samples=10
        )
        
        assert 'aleatoric' in uncertainties
        assert 'epistemic' in uncertainties
        assert 'total' in uncertainties
        
        for key, unc in uncertainties.items():
            assert unc.shape == (4, 16, 16, 1)
            assert (unc >= 0).all()
    
    def test_sample_predictions(self):
        """Test prediction sampling."""
        samples = self.model.sample_predictions(
            self.input_tensor, num_samples=5
        )
        
        assert samples.shape == (5, 4, 16, 16, 1)
        assert not torch.isnan(samples).any()
    
    def test_compute_kl_divergence(self):
        """Test KL divergence computation."""
        kl = self.model.compute_kl_divergence()
        
        assert isinstance(kl, torch.Tensor)
        assert kl.item() >= 0
        assert not torch.isnan(kl)
    
    def test_different_input_formats(self):
        """Test different input tensor formats."""
        # Test (batch, input_dim, height, width) format
        input_chw = torch.randn(4, 2, 16, 16)
        output = self.model(input_chw)
        assert isinstance(output, tuple)
        
        # Test (batch, height, width, input_dim) format
        input_hwc = torch.randn(4, 16, 16, 2)
        output = self.model(input_hwc)
        assert isinstance(output, tuple)
    
    def test_invalid_uncertainty_type(self):
        """Test invalid uncertainty type."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = ProbabilisticNeuralOperator(
                input_dim=2,
                uncertainty_type="full"  # Should fall back to diagonal
            )
            assert model.uncertainty_type == "diagonal"
    
    def test_save_load_checkpoint(self):
        """Test model checkpoint save/load."""
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "test_model.pt"
            
            # Save checkpoint
            config = {
                'input_dim': 2,
                'hidden_dim': 32,
                'num_layers': 2,
                'modes': 8,
                'output_dim': 1,
            }
            
            self.model.save_checkpoint(
                str(checkpoint_path),
                config=config,
                epoch=10,
                loss=0.5
            )
            
            assert checkpoint_path.exists()
            
            # Load checkpoint
            loaded_model = ProbabilisticNeuralOperator.load_checkpoint(
                str(checkpoint_path)
            )
            
            assert loaded_model.input_dim == 2
            assert loaded_model.hidden_dim == 32
            
            # Test that loaded model produces same output
            loaded_model.eval()
            self.model.eval()
            
            with torch.no_grad():
                output1 = self.model(self.input_tensor, sample=False)
                output2 = loaded_model(self.input_tensor, sample=False)
                
                torch.testing.assert_close(output1[0], output2[0], atol=1e-6, rtol=1e-6)


class TestBaslineModels:
    """Test baseline models (FNO, DeepONet)."""
    
    def test_fourier_neural_operator(self):
        """Test FNO model."""
        model = FourierNeuralOperator(
            input_dim=2,
            hidden_dim=32,
            num_layers=2,
            modes=8,
            output_dim=1
        )
        
        input_tensor = torch.randn(4, 16, 16, 2)
        output = model(input_tensor)
        
        assert output.shape == (4, 16, 16, 1)
        assert not torch.isnan(output).any()
    
    def test_deeponet(self):
        """Test DeepONet model."""
        model = DeepONet(
            input_dim=2,
            spatial_dim=2,
            grid_size=16,
            hidden_dim=64,
            num_layers=3,
            output_dim=1
        )
        
        input_tensor = torch.randn(4, 16, 16, 2)
        output = model(input_tensor)
        
        assert output.shape == (4, 16, 16, 1)
        assert not torch.isnan(output).any()
    
    def test_deeponet_with_coordinates(self):
        """Test DeepONet with custom coordinates."""
        model = DeepONet(
            input_dim=2,
            spatial_dim=2,
            grid_size=16,
            hidden_dim=64,
            num_layers=3,
            output_dim=1
        )
        
        input_tensor = torch.randn(4, 16, 16, 2)
        coords = torch.rand(4, 16, 16, 2)
        output = model(input_tensor, coords)
        
        assert output.shape == (4, 16, 16, 1)
        assert not torch.isnan(output).any()


class TestModelEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_zero_input_dim(self):
        """Test invalid input dimension."""
        with pytest.raises((ValueError, RuntimeError)):
            ProbabilisticNeuralOperator(input_dim=0)
    
    def test_very_small_input(self):
        """Test with very small input tensor."""
        model = ProbabilisticNeuralOperator(input_dim=1, modes=4)
        small_input = torch.randn(1, 4, 4, 1)
        
        output = model(small_input)
        assert output[0].shape == (1, 4, 4, 1)
    
    def test_large_modes(self):
        """Test with modes larger than input size."""
        model = ProbabilisticNeuralOperator(input_dim=1, modes=100)
        input_tensor = torch.randn(1, 8, 8, 1)
        
        # Should handle gracefully
        output = model(input_tensor)
        assert output[0].shape == (1, 8, 8, 1)
    
    def test_device_consistency(self):
        """Test device consistency."""
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
            model = ProbabilisticNeuralOperator(input_dim=2).to(device)
            input_tensor = torch.randn(2, 8, 8, 2).to(device)
            
            output = model(input_tensor)
            assert output[0].device == device
            assert output[1].device == device
    
    def test_gradient_flow(self):
        """Test gradient flow through model."""
        model = ProbabilisticNeuralOperator(input_dim=2)
        input_tensor = torch.randn(2, 8, 8, 2, requires_grad=True)
        
        output = model(input_tensor, return_kl=True)
        loss = output[0].sum() + output[2]  # mean + kl
        loss.backward()
        
        # Check gradients exist
        assert input_tensor.grad is not None
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None
    
    def test_numerical_stability(self):
        """Test numerical stability with extreme values."""
        model = ProbabilisticNeuralOperator(input_dim=2)
        
        # Test with very large values
        large_input = torch.randn(2, 8, 8, 2) * 1000
        output = model(large_input)
        assert not torch.isnan(output[0]).any()
        assert not torch.isinf(output[0]).any()
        
        # Test with very small values
        small_input = torch.randn(2, 8, 8, 2) * 1e-6
        output = model(small_input)
        assert not torch.isnan(output[0]).any()
        assert not torch.isinf(output[0]).any()


@pytest.mark.parametrize("input_dim,hidden_dim,num_layers,modes", [
    (1, 16, 1, 4),
    (2, 32, 2, 8),
    (3, 64, 3, 16),
    (4, 128, 4, 32),
])
def test_model_configurations(input_dim, hidden_dim, num_layers, modes):
    """Test various model configurations."""
    model = ProbabilisticNeuralOperator(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        modes=modes
    )
    
    input_tensor = torch.randn(2, 16, 16, input_dim)
    output = model(input_tensor)
    
    assert output[0].shape == (2, 16, 16, 1)
    assert output[1].shape == (2, 16, 16, 1)


@pytest.mark.parametrize("uncertainty_type", ["diagonal"])
def test_uncertainty_types(uncertainty_type):
    """Test different uncertainty types."""
    model = ProbabilisticNeuralOperator(
        input_dim=2,
        uncertainty_type=uncertainty_type
    )
    
    input_tensor = torch.randn(2, 8, 8, 2)
    uncertainties = model.decompose_uncertainty(input_tensor, num_samples=5)
    
    assert 'aleatoric' in uncertainties
    assert 'epistemic' in uncertainties
    assert 'total' in uncertainties