"""Dataset utilities for PDE data loading and preprocessing."""

import torch
import torch.utils.data as data
import numpy as np
import h5py
from pathlib import Path
from typing import Tuple, Dict, Optional, List, Union, Any
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class BasePDEDataset(data.Dataset, ABC):
    """Abstract base class for PDE datasets."""
    
    def __init__(
        self,
        data_path: Union[str, Path],
        resolution: int = 64,
        normalize: bool = True,
        device: str = "cpu"
    ):
        # Input validation
        if resolution <= 0:
            raise ValueError(f"resolution must be positive, got {resolution}")
        if resolution > 512:
            import warnings
            warnings.warn(f"Large resolution ({resolution}) may be memory-intensive")
        if device not in ["cpu", "cuda"] and not device.startswith("cuda:"):
            raise ValueError(f"Invalid device '{device}'. Use 'cpu', 'cuda', or 'cuda:N'")
            
        self.data_path = Path(data_path)
        self.resolution = resolution
        self.normalize = normalize
        self.device = device
        
        # Data containers
        self.inputs = None
        self.outputs = None
        self.normalization_stats = {}
        self._data_loaded = False
        
    @abstractmethod
    def load_data(self) -> None:
        """Load dataset from file."""
        pass
    
    def __len__(self) -> int:
        return len(self.inputs) if self.inputs is not None else 0
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a single data sample."""
        if not self._data_loaded or self.inputs is None or self.outputs is None:
            raise RuntimeError("Dataset not loaded. Call load_data() first.")
        
        # Index validation
        if idx < 0 or idx >= len(self.inputs):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.inputs)}")
            
        try:
            input_tensor = torch.tensor(self.inputs[idx], dtype=torch.float32)
            output_tensor = torch.tensor(self.outputs[idx], dtype=torch.float32)
            
            # Validate tensor shapes
            if input_tensor.dim() < 2 or output_tensor.dim() < 2:
                raise ValueError(f"Invalid tensor dimensions: input {input_tensor.shape}, output {output_tensor.shape}")
            
        except Exception as e:
            raise RuntimeError(f"Error loading data sample {idx}: {e}")
        
        return input_tensor, output_tensor
    
    def normalize_data(self) -> None:
        """Normalize input and output data."""
        if self.inputs is not None:
            # Input normalization
            input_mean = np.mean(self.inputs, axis=(0, 2, 3), keepdims=True)
            input_std = np.std(self.inputs, axis=(0, 2, 3), keepdims=True)
            input_std = np.where(input_std == 0, 1, input_std)  # Avoid division by zero
            
            self.inputs = (self.inputs - input_mean) / input_std
            self.normalization_stats['input_mean'] = input_mean
            self.normalization_stats['input_std'] = input_std
        
        if self.outputs is not None:
            # Output normalization
            output_mean = np.mean(self.outputs, axis=(0, 2, 3), keepdims=True)
            output_std = np.std(self.outputs, axis=(0, 2, 3), keepdims=True)
            output_std = np.where(output_std == 0, 1, output_std)
            
            self.outputs = (self.outputs - output_mean) / output_std
            self.normalization_stats['output_mean'] = output_mean
            self.normalization_stats['output_std'] = output_std
    
    def denormalize_outputs(self, outputs: torch.Tensor) -> torch.Tensor:
        """Denormalize model outputs."""
        if 'output_mean' in self.normalization_stats:
            mean = torch.tensor(self.normalization_stats['output_mean'], device=outputs.device)
            std = torch.tensor(self.normalization_stats['output_std'], device=outputs.device)
            return outputs * std + mean
        return outputs


class SyntheticNavierStokesDataset(BasePDEDataset):
    """Synthetic Navier-Stokes 2D dataset."""
    
    def __init__(
        self,
        num_samples: int = 1000,
        resolution: int = 64,
        viscosity: float = 1e-3,
        time_steps: int = 50,
        dt: float = 0.01,
        normalize: bool = True,
        device: str = "cpu"
    ):
        # Create synthetic path
        data_path = Path("synthetic_navier_stokes")
        super().__init__(data_path, resolution, normalize, device)
        
        self.num_samples = num_samples
        self.viscosity = viscosity
        self.time_steps = time_steps
        self.dt = dt
        
        self.load_data()
        
    def load_data(self) -> None:
        """Generate synthetic Navier-Stokes data."""
        logger.info(f"Generating synthetic Navier-Stokes dataset with {self.num_samples} samples")
        
        # Generate initial conditions (vorticity fields)
        inputs = []
        outputs = []
        
        for i in range(self.num_samples):
            # Random initial vorticity field
            initial_vorticity = self._generate_initial_vorticity()
            
            # Solve NS equations (simplified)
            final_vorticity = self._solve_ns_simplified(initial_vorticity)
            
            # Create input (initial + coordinates + parameters)
            x, y = np.meshgrid(np.linspace(0, 2*np.pi, self.resolution),
                              np.linspace(0, 2*np.pi, self.resolution))
            
            input_field = np.stack([
                initial_vorticity,
                x / (2*np.pi),  # Normalized x coordinates
                y / (2*np.pi),  # Normalized y coordinates
            ], axis=0)
            
            inputs.append(input_field)
            outputs.append(final_vorticity[np.newaxis, ...])  # Add channel dimension
        
        try:
            self.inputs = np.array(inputs)
            self.outputs = np.array(outputs)
            
            # Validate data shapes
            if len(self.inputs) == 0 or len(self.outputs) == 0:
                raise ValueError("Generated empty dataset")
            if self.inputs.shape[0] != self.outputs.shape[0]:
                raise ValueError(f"Mismatch in number of samples: inputs {self.inputs.shape[0]}, outputs {self.outputs.shape[0]}")
            
            if self.normalize:
                self.normalize_data()
                
            self._data_loaded = True
            logger.info(f"Dataset loaded: {len(self.inputs)} samples, resolution {self.resolution}x{self.resolution}")
            
        except Exception as e:
            logger.error(f"Failed to create dataset arrays: {e}")
            raise RuntimeError(f"Dataset creation failed: {e}")
    
    def _generate_initial_vorticity(self) -> np.ndarray:
        """Generate random initial vorticity field."""
        # Create random Fourier modes
        modes = np.random.randn(8, 8) + 1j * np.random.randn(8, 8)
        
        # Create full spectrum
        full_modes = np.zeros((self.resolution, self.resolution//2 + 1), dtype=complex)
        full_modes[:8, :8] = modes
        
        # Convert to physical space
        vorticity = np.fft.irfft2(full_modes)
        
        # Normalize
        vorticity = (vorticity - vorticity.mean()) / vorticity.std()
        return vorticity
    
    def _solve_ns_simplified(self, initial_vorticity: np.ndarray) -> np.ndarray:
        """Simplified NS solver (spectral method)."""
        # This is a simplified version - in practice, you'd use a proper NS solver
        vorticity = initial_vorticity.copy()
        
        for _ in range(self.time_steps):
            # Apply viscous diffusion (simplified)
            vorticity_ft = np.fft.rfft2(vorticity)
            
            # Wavenumbers
            kx = np.fft.fftfreq(self.resolution, 1.0/self.resolution)
            ky = np.fft.rfftfreq(self.resolution, 1.0/self.resolution)
            KX, KY = np.meshgrid(kx, ky, indexing='ij')
            K2 = KX**2 + KY**2
            K2[0, 0] = 1  # Avoid division by zero
            
            # Viscous term
            vorticity_ft *= np.exp(-self.viscosity * K2 * self.dt)
            
            # Add small nonlinear term (simplified)
            vorticity = np.fft.irfft2(vorticity_ft)
            vorticity += 0.01 * self.dt * np.sin(vorticity)
        
        return vorticity


class SyntheticDarcyFlowDataset(BasePDEDataset):
    """Synthetic Darcy Flow 2D dataset."""
    
    def __init__(
        self,
        num_samples: int = 1000,
        resolution: int = 64,
        normalize: bool = True,
        device: str = "cpu"
    ):
        data_path = Path("synthetic_darcy_flow")
        super().__init__(data_path, resolution, normalize, device)
        
        self.num_samples = num_samples
        self.load_data()
        
    def load_data(self) -> None:
        """Generate synthetic Darcy flow data."""
        logger.info(f"Generating synthetic Darcy flow dataset with {self.num_samples} samples")
        
        inputs = []
        outputs = []
        
        for i in range(self.num_samples):
            # Generate random permeability field
            permeability = self._generate_permeability_field()
            
            # Solve Darcy equation
            pressure = self._solve_darcy(permeability)
            
            # Create input (permeability + coordinates)
            x, y = np.meshgrid(np.linspace(0, 1, self.resolution),
                              np.linspace(0, 1, self.resolution))
            
            input_field = np.stack([
                permeability,
                x,
                y,
            ], axis=0)
            
            inputs.append(input_field)
            outputs.append(pressure[np.newaxis, ...])
        
        try:
            self.inputs = np.array(inputs)
            self.outputs = np.array(outputs)
            
            # Validate data shapes
            if len(self.inputs) == 0 or len(self.outputs) == 0:
                raise ValueError("Generated empty dataset")
            if self.inputs.shape[0] != self.outputs.shape[0]:
                raise ValueError(f"Mismatch in number of samples: inputs {self.inputs.shape[0]}, outputs {self.outputs.shape[0]}")
            
            if self.normalize:
                self.normalize_data()
                
            self._data_loaded = True
            logger.info(f"Dataset loaded: {len(self.inputs)} samples, resolution {self.resolution}x{self.resolution}")
            
        except Exception as e:
            logger.error(f"Failed to create dataset arrays: {e}")
            raise RuntimeError(f"Dataset creation failed: {e}")
    
    def _generate_permeability_field(self) -> np.ndarray:
        """Generate random log-permeability field."""
        # Generate correlated random field using FFT
        modes = np.random.randn(8, 8) + 1j * np.random.randn(8, 8)
        
        full_modes = np.zeros((self.resolution, self.resolution//2 + 1), dtype=complex)
        full_modes[:8, :8] = modes
        
        # Apply correlation
        kx = np.fft.fftfreq(self.resolution)[:, np.newaxis]
        ky = np.fft.rfftfreq(self.resolution)[np.newaxis, :]
        k2 = kx**2 + ky**2
        k2[0, 0] = 1
        
        # Power spectrum with correlation length
        correlation_length = 0.1
        power_spectrum = np.exp(-k2 * correlation_length**2)
        full_modes *= np.sqrt(power_spectrum)
        
        # Convert to physical space
        log_perm = np.fft.irfft2(full_modes)
        return np.exp(log_perm)
    
    def _solve_darcy(self, permeability: np.ndarray) -> np.ndarray:
        """Solve Darcy equation using finite differences."""
        # Simplified Darcy solver: -div(k * grad(p)) = f
        # With zero Dirichlet boundary conditions and unit source
        
        n = self.resolution
        h = 1.0 / (n - 1)
        
        # Create system matrix (simplified)
        pressure = np.zeros_like(permeability)
        
        # Iterative solver (Gauss-Seidel)
        for iteration in range(100):
            pressure_old = pressure.copy()
            
            for i in range(1, n-1):
                for j in range(1, n-1):
                    # Average permeabilities at cell faces
                    k_e = 0.5 * (permeability[i, j] + permeability[i, j+1]) if j+1 < n else permeability[i, j]
                    k_w = 0.5 * (permeability[i, j] + permeability[i, j-1]) if j-1 >= 0 else permeability[i, j]
                    k_n = 0.5 * (permeability[i, j] + permeability[i-1, j]) if i-1 >= 0 else permeability[i, j]
                    k_s = 0.5 * (permeability[i, j] + permeability[i+1, j]) if i+1 < n else permeability[i, j]
                    
                    # Source term (unit source at center)
                    source = 1.0 if (n//4 < i < 3*n//4 and n//4 < j < 3*n//4) else 0.0
                    
                    # Update pressure
                    pressure[i, j] = (
                        k_e * pressure[i, j+1] + k_w * pressure[i, j-1] +
                        k_n * pressure[i-1, j] + k_s * pressure[i+1, j] +
                        h**2 * source
                    ) / (k_e + k_w + k_n + k_s)
            
            # Check convergence
            if np.max(np.abs(pressure - pressure_old)) < 1e-6:
                break
        
        return pressure


class PDEDataset:
    """Factory class for creating PDE datasets."""
    
    @staticmethod
    def load(
        name: str,
        resolution: int = 64,
        num_samples: int = 1000,
        normalize: bool = True,
        device: str = "cpu",
        **kwargs
    ) -> BasePDEDataset:
        """Load a PDE dataset by name."""
        # Input validation
        if not isinstance(name, str) or not name:
            raise ValueError("Dataset name must be a non-empty string")
        if resolution <= 0:
            raise ValueError(f"resolution must be positive, got {resolution}")
        if num_samples <= 0:
            raise ValueError(f"num_samples must be positive, got {num_samples}")
        if num_samples > 100000:
            import warnings
            warnings.warn(f"Large num_samples ({num_samples}) may consume significant memory and time")
        
        # Available datasets
        available_datasets = ["navier_stokes_2d", "darcy_flow_2d"]
        
        if name == "navier_stokes_2d":
            try:
                return SyntheticNavierStokesDataset(
                    num_samples=num_samples,
                    resolution=resolution,
                    normalize=normalize,
                    device=device,
                    **kwargs
                )
            except Exception as e:
                raise RuntimeError(f"Failed to create Navier-Stokes dataset: {e}")
        elif name == "darcy_flow_2d":
            try:
                return SyntheticDarcyFlowDataset(
                    num_samples=num_samples,
                    resolution=resolution,
                    normalize=normalize,
                    device=device,
                    **kwargs
                )
            except Exception as e:
                raise RuntimeError(f"Failed to create Darcy flow dataset: {e}")
        else:
            raise ValueError(f"Unknown dataset: '{name}'. Available datasets: {available_datasets}")
    
    @staticmethod
    def get_loaders(
        dataset: BasePDEDataset,
        batch_size: int = 32,
        train_split: float = 0.7,
        val_split: float = 0.15,
        test_split: float = 0.15,
        shuffle: bool = True,
        num_workers: int = 0
    ) -> Tuple[data.DataLoader, data.DataLoader, data.DataLoader]:
        """Create train/val/test data loaders."""
        
        assert abs(train_split + val_split + test_split - 1.0) < 1e-6, \
            "Splits must sum to 1.0"
        
        total_size = len(dataset)
        train_size = int(train_split * total_size)
        val_size = int(val_split * total_size)
        test_size = total_size - train_size - val_size
        
        # Split dataset
        train_dataset, val_dataset, test_dataset = data.random_split(
            dataset, 
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        # Create data loaders
        train_loader = data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True
        )
        
        val_loader = data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        test_loader = data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        return train_loader, val_loader, test_loader


# Add get_loaders method to BasePDEDataset for convenience
def get_loaders(self, **kwargs) -> Tuple[data.DataLoader, data.DataLoader, data.DataLoader]:
    """Get data loaders for this dataset."""
    return PDEDataset.get_loaders(self, **kwargs)

BasePDEDataset.get_loaders = get_loaders


# Export main classes
__all__ = [
    "BasePDEDataset",
    "PDEDataset", 
    "SyntheticNavierStokesDataset",
    "SyntheticDarcyFlowDataset"
]