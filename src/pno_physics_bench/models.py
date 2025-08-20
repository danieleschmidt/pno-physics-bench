# SECURITY NOTICE: This file has been automatically hardened for security
# - All user inputs should be validated and sanitized
# - Subprocess calls use secure alternatives
# - SQL queries use parameterized statements
# - No hardcoded secrets or credentials


"""Core neural operator models with uncertainty quantification."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict, Any
from abc import ABC, abstractmethod


class SpectralConv2d(nn.Module):
    """Spectral convolution layer for Fourier Neural Operators."""
    
    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        
        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    def compl_mul2d(self, input: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """Complex multiplication in frequency domain."""
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        
        # Compute Fourier coefficients
        x_ft = torch.fft.rfft2(x)
        
        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batch_size, self.out_channels, x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        # Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x


class SpectralConv2d_Probabilistic(SpectralConv2d):
    """Probabilistic spectral convolution with variational parameters."""
    
    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int):
        super().__init__(in_channels, out_channels, modes1, modes2)
        
        # Variational parameters for weights
        self.weights1_log_var = nn.Parameter(torch.log(torch.ones_like(self.weights1.real) * 0.01))
        self.weights2_log_var = nn.Parameter(torch.log(torch.ones_like(self.weights2.real) * 0.01))
        
    def sample_weights(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample weights using reparameterization trick."""
        if self.training:
            # Sample from variational posterior
            eps1 = torch.randn_like(self.weights1_log_var)
            eps2 = torch.randn_like(self.weights2_log_var)
            
            std1 = torch.exp(0.5 * self.weights1_log_var)
            std2 = torch.exp(0.5 * self.weights2_log_var)
            
            w1_real = self.weights1.real + std1 * eps1
            w1_imag = self.weights1.imag + std1 * torch.randn_like(eps1)
            w2_real = self.weights2.real + std2 * eps2
            w2_imag = self.weights2.imag + std2 * torch.randn_like(eps2)
            
            return (w1_real + 1j * w1_imag, w2_real + 1j * w2_imag)
        else:
            return (self.weights1, self.weights2)
    
    def forward(self, x: torch.Tensor, sample: bool = True) -> torch.Tensor:
        batch_size = x.shape[0]
        
        # Sample or use mean weights
        w1, w2 = self.sample_weights() if sample else (self.weights1, self.weights2)
        
        # Compute Fourier coefficients
        x_ft = torch.fft.rfft2(x)
        
        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batch_size, self.out_channels, x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], w1)
        out_ft[:, :, -self.modes1:, :self.modes2] = self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], w2)

        # Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x
    
    def kl_divergence(self) -> torch.Tensor:
        """Compute KL divergence for variational weights."""
        kl1 = -0.5 * torch.sum(1 + self.weights1_log_var - self.weights1.real.pow(2) - self.weights1.imag.pow(2) - self.weights1_log_var.exp())
        kl2 = -0.5 * torch.sum(1 + self.weights2_log_var - self.weights2.real.pow(2) - self.weights2.imag.pow(2) - self.weights2_log_var.exp())
        return kl1 + kl2


class BaseNeuralOperator(nn.Module, ABC):
    """Abstract base class for neural operators."""
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the operator."""
        pass
    
    @abstractmethod
    def predict_with_uncertainty(self, x: torch.Tensor, num_samples: int = 100) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict with uncertainty estimation."""
        pass


class FourierNeuralOperator(BaseNeuralOperator):
    """Standard Fourier Neural Operator (FNO) implementation."""
    
    def __init__(
        self,
        input_dim: int = 3,
        hidden_dim: int = 256,
        num_layers: int = 4,
        modes: int = 20,
        activation: str = "gelu"
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.modes = modes
        
        # Activation function
        if activation == "gelu":
            self.activation = F.gelu
        elif activation == "relu":
            self.activation = F.relu
        else:
            self.activation = F.gelu
        
        # Lifting layer
        self.lift = nn.Conv2d(input_dim, hidden_dim, 1)
        
        # Fourier layers
        self.fourier_layers = nn.ModuleList([
            SpectralConv2d(hidden_dim, hidden_dim, modes, modes)
            for _ in range(num_layers)
        ])
        
        # Local linear layers
        self.linear_layers = nn.ModuleList([
            nn.Conv2d(hidden_dim, hidden_dim, 1)
            for _ in range(num_layers)
        ])
        
        # Projection layer
        self.project = nn.Conv2d(hidden_dim, 1, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through FNO."""
        # Lift to higher dimension
        x = self.lift(x)
        
        # Apply Fourier layers
        for fourier, linear in zip(self.fourier_layers, self.linear_layers):
            x = self.activation(fourier(x) + linear(x))
        
        # Project to output
        x = self.project(x)
        return x
    
    def predict_with_uncertainty(self, x: torch.Tensor, num_samples: int = 100) -> Tuple[torch.Tensor, torch.Tensor]:
        """FNO doesn't have native uncertainty - return zeros."""
        with torch.no_grad():
            mean = self.forward(x)
            std = torch.zeros_like(mean)
        return mean, std


class ProbabilisticNeuralOperator(BaseNeuralOperator):
    """Probabilistic Neural Operator with uncertainty quantification."""
    
    def __init__(
        self,
        input_dim: int = 3,
        hidden_dim: int = 256,
        num_layers: int = 4,
        modes: int = 20,
        uncertainty_type: str = "full",
        posterior: str = "variational",
        activation: str = "gelu"
    ):
        # Input validation
        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}")
        if modes <= 0:
            raise ValueError(f"modes must be positive, got {modes}")
        if uncertainty_type not in ["full", "diagonal", "scalar"]:
            raise ValueError(f"uncertainty_type must be one of ['full', 'diagonal', 'scalar'], got {uncertainty_type}")
        if posterior not in ["variational", "ensemble", "mc_dropout"]:
            raise ValueError(f"posterior must be one of ['variational', 'ensemble', 'mc_dropout'], got {posterior}")
        if activation not in ["gelu", "relu", "tanh", "swish"]:
            raise ValueError(f"activation must be one of ['gelu', 'relu', 'tanh', 'swish'], got {activation}")
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.modes = modes
        self.uncertainty_type = uncertainty_type
        self.posterior = posterior
        
        # Activation function
        if activation == "gelu":
            self.activation = F.gelu
        elif activation == "relu":
            self.activation = F.relu
        else:
            self.activation = F.gelu
        
        # Lifting layer with variational parameters
        self.lift_mean = nn.Conv2d(input_dim, hidden_dim, 1)
        self.lift_log_var = nn.Conv2d(input_dim, hidden_dim, 1)
        
        # Probabilistic Fourier layers
        self.fourier_layers = nn.ModuleList([
            SpectralConv2d_Probabilistic(hidden_dim, hidden_dim, modes, modes)
            for _ in range(num_layers)
        ])
        
        # Variational linear layers
        self.linear_mean = nn.ModuleList([
            nn.Conv2d(hidden_dim, hidden_dim, 1)
            for _ in range(num_layers)
        ])
        self.linear_log_var = nn.ModuleList([
            nn.Conv2d(hidden_dim, hidden_dim, 1)
            for _ in range(num_layers)
        ])
        
        # Output projection with uncertainty
        self.project_mean = nn.Conv2d(hidden_dim, 1, 1)
        self.project_log_var = nn.Conv2d(hidden_dim, 1, 1)
        
        # Initialize log_var parameters to small values for numerical stability
        for layer in self.linear_log_var:
            nn.init.constant_(layer.weight, -3.0)  # Less extreme initialization
            nn.init.constant_(layer.bias, -3.0)
        nn.init.constant_(self.lift_log_var.weight, -3.0)
        nn.init.constant_(self.lift_log_var.bias, -3.0)
        nn.init.constant_(self.project_log_var.weight, -3.0)
        nn.init.constant_(self.project_log_var.bias, -3.0)
        
    def reparameterize(self, mean: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick for sampling."""
        if self.training:
            # Clamp log_var to prevent numerical instability
            log_var = torch.clamp(log_var, min=-10.0, max=10.0)
            std = torch.exp(0.5 * log_var)
            # Add small epsilon for numerical stability
            std = torch.clamp(std, min=1e-6)
            eps = torch.randn_like(std)
            return mean + eps * std
        else:
            return mean
    
    def forward(self, x: torch.Tensor, sample: bool = True) -> torch.Tensor:
        """Forward pass through PNO."""
        # Input validation
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(x)}")
        if x.dim() != 4:
            raise ValueError(f"Expected 4D tensor (batch, channels, height, width), got {x.dim()}D tensor")
        if x.size(1) != self.input_dim:
            raise ValueError(f"Expected {self.input_dim} input channels, got {x.size(1)}")
        
        # Probabilistic lift
        lift_mean = self.lift_mean(x)
        lift_log_var = self.lift_log_var(x)
        x = self.reparameterize(lift_mean, lift_log_var) if sample else lift_mean
        
        # Apply probabilistic Fourier layers
        for i, fourier in enumerate(self.fourier_layers):
            # Fourier branch
            fourier_out = fourier(x, sample=sample)
            
            # Local linear branch
            linear_mean = self.linear_mean[i](x)
            if sample and self.training:
                linear_log_var = self.linear_log_var[i](x)
                linear_out = self.reparameterize(linear_mean, linear_log_var)
            else:
                linear_out = linear_mean
            
            # Combine and activate
            x = self.activation(fourier_out + linear_out)
        
        # Probabilistic output projection
        output_mean = self.project_mean(x)
        if sample and self.training:
            output_log_var = self.project_log_var(x)
            output = self.reparameterize(output_mean, output_log_var)
        else:
            output = output_mean
            
        return output
    
    def predict_distributional(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get predictive mean and standard deviation."""
        self.eval()
        with torch.no_grad():
            mean = self.forward(x, sample=False)
            # Compute predictive variance (epistemic + aleatoric)
            log_var = self.project_log_var(self.get_final_features(x))
            # Clamp for numerical stability
            log_var = torch.clamp(log_var, min=-10.0, max=10.0)
            std = torch.exp(0.5 * log_var)
            std = torch.clamp(std, min=1e-6)  # Ensure positive std
        return mean, std
    
    def get_final_features(self, x: torch.Tensor) -> torch.Tensor:
        """Get features before final projection."""
        # Lift
        x = self.lift_mean(x)
        
        # Apply Fourier layers  
        for i, fourier in enumerate(self.fourier_layers):
            fourier_out = fourier(x, sample=False)
            linear_out = self.linear_mean[i](x)
            x = self.activation(fourier_out + linear_out)
        
        return x
    
    def predict_with_uncertainty(self, x: torch.Tensor, num_samples: int = 100) -> Tuple[torch.Tensor, torch.Tensor]:
        """Monte Carlo estimation of predictive uncertainty."""
        # Input validation
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(x)}")
        if num_samples <= 0:
            raise ValueError(f"num_samples must be positive, got {num_samples}")
        if num_samples > 1000:
            import warnings
            warnings.warn(f"Large num_samples ({num_samples}) may be slow and memory-intensive")
        
        # Store original training mode
        was_training = self.training
        self.train()  # Enable dropout/sampling
        samples = []
        
        try:
            with torch.no_grad():
                for _ in range(num_samples):
                    sample = self.forward(x, sample=True)
                    samples.append(sample)
            
            samples = torch.stack(samples)
            mean = samples.mean(dim=0)
            std = samples.std(dim=0)
        
        except torch.cuda.OutOfMemoryError as e:
            raise RuntimeError(f"CUDA out of memory during uncertainty estimation. "
                             f"Try reducing num_samples or batch_size. Original error: {e}")
        except Exception as e:
            raise RuntimeError(f"Error during uncertainty estimation: {e}")
        finally:
            # Restore original training mode
            self.train(was_training)
        
        return mean, std
    
    def kl_divergence(self) -> torch.Tensor:
        """Compute total KL divergence for all variational parameters."""
        kl = torch.tensor(0.0, device=next(self.parameters()).device)
        
        # KL from Fourier layers
        for layer in self.fourier_layers:
            kl += layer.kl_divergence()
            
        return kl


class DeepONet(BaseNeuralOperator):
    """Deep Operator Network implementation."""
    
    def __init__(
        self,
        branch_net_dims: list = [100, 512, 512, 512],
        trunk_net_dims: list = [2, 512, 512, 512],
        activation: str = "tanh"
    ):
        super().__init__()
        
        self.branch_dims = branch_net_dims
        self.trunk_dims = trunk_net_dims
        
        # Activation function
        if activation == "tanh":
            self.activation = torch.tanh
        elif activation == "relu":
            self.activation = F.relu
        elif activation == "gelu":
            self.activation = F.gelu
        else:
            self.activation = torch.tanh
        
        # Branch network (processes input function)
        branch_layers = []
        for i in range(len(branch_net_dims) - 1):
            branch_layers.append(nn.Linear(branch_net_dims[i], branch_net_dims[i+1]))
            if i < len(branch_net_dims) - 2:
                branch_layers.append(nn.ReLU())
        self.branch_net = nn.Sequential(*branch_layers)
        
        # Trunk network (processes evaluation coordinates)
        trunk_layers = []
        for i in range(len(trunk_net_dims) - 1):
            trunk_layers.append(nn.Linear(trunk_net_dims[i], trunk_net_dims[i+1]))
            if i < len(trunk_net_dims) - 2:
                trunk_layers.append(nn.ReLU())
        self.trunk_net = nn.Sequential(*trunk_layers)
        
        # Bias term
        self.bias = nn.Parameter(torch.zeros(1))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through DeepONet."""
        batch_size, channels, height, width = x.shape
        
        # Flatten spatial dimensions for branch network
        branch_input = x.view(batch_size, -1)  # (B, C*H*W)
        branch_out = self.branch_net(branch_input)  # (B, branch_dim)
        
        # Create coordinate grid for trunk network
        y, x_coords = torch.meshgrid(
            torch.linspace(0, 1, height, device=x.device),
            torch.linspace(0, 1, width, device=x.device),
            indexing='ij'
        )
        coords = torch.stack([x_coords, y], dim=-1).view(-1, 2)  # (H*W, 2)
        
        # Trunk network
        trunk_out = self.trunk_net(coords)  # (H*W, trunk_dim)
        
        # Dot product between branch and trunk outputs
        output = torch.einsum('bi,ji->bj', branch_out, trunk_out)  # (B, H*W)
        output = output + self.bias
        
        # Reshape back to spatial dimensions
        output = output.view(batch_size, 1, height, width)
        
        return output
    
    def predict_with_uncertainty(self, x: torch.Tensor, num_samples: int = 100) -> Tuple[torch.Tensor, torch.Tensor]:
        """DeepONet doesn't have native uncertainty - return zeros."""
        with torch.no_grad():
            mean = self.forward(x)
            std = torch.zeros_like(mean)
        return mean, std


# Export main classes
__all__ = [
    "BaseNeuralOperator",
    "FourierNeuralOperator", 
    "ProbabilisticNeuralOperator",
    "DeepONet",
    "SpectralConv2d",
    "SpectralConv2d_Probabilistic"
]