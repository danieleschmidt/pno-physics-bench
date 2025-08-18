"""Quantum-Enhanced Uncertainty Quantification for Neural Operators.

This module implements quantum-inspired algorithms for improved uncertainty
estimation in probabilistic neural operators, leveraging quantum superposition
and entanglement concepts for enhanced predictive modeling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import math
from abc import ABC, abstractmethod


class QuantumStatePreparation(nn.Module):
    """Quantum-inspired state preparation for uncertainty encoding."""
    
    def __init__(
        self,
        num_qubits: int = 8,
        feature_dim: int = 256,
        entanglement_depth: int = 3
    ):
        super().__init__()
        
        self.num_qubits = num_qubits
        self.hilbert_dim = 2 ** num_qubits
        self.entanglement_depth = entanglement_depth
        
        # Quantum state encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(feature_dim, 4 * num_qubits),
            nn.Tanh(),  # Bounded activations for quantum amplitudes
            nn.Linear(4 * num_qubits, 2 * self.hilbert_dim)  # Real and imaginary parts
        )
        
        # Entanglement gates parameters
        self.entanglement_params = nn.Parameter(
            torch.randn(entanglement_depth, num_qubits, 4)
        )
        
        # Measurement operators
        self.measurement_ops = nn.Parameter(
            torch.randn(num_qubits, 4, 4)  # Pauli operators
        )
        
    def prepare_quantum_state(self, classical_features: torch.Tensor) -> torch.Tensor:
        """Prepare quantum state from classical features."""
        batch_size = classical_features.size(0)
        
        # Encode classical features to quantum amplitudes
        amplitudes = self.state_encoder(classical_features)
        real_part, imag_part = amplitudes.chunk(2, dim=-1)
        
        # Normalize to valid quantum state
        complex_amplitudes = torch.complex(real_part, imag_part)
        normalized_amplitudes = F.normalize(complex_amplitudes, p=2, dim=-1)
        
        return normalized_amplitudes.view(batch_size, self.hilbert_dim)
    
    def apply_entanglement(self, quantum_state: torch.Tensor) -> torch.Tensor:
        """Apply entanglement operations to quantum state."""
        batch_size = quantum_state.size(0)
        current_state = quantum_state
        
        for depth in range(self.entanglement_depth):
            # Apply controlled rotations between adjacent qubits
            for i in range(self.num_qubits - 1):
                # Extract entanglement parameters for this qubit pair
                theta, phi, lambda_, gamma = self.entanglement_params[depth, i]
                
                # Create controlled rotation matrix
                cnot_matrix = self._create_cnot_matrix(i, i + 1)
                rotation_matrix = self._create_rotation_matrix(theta, phi, lambda_, gamma)
                
                # Apply entanglement gate
                entanglement_gate = torch.matmul(cnot_matrix, rotation_matrix)
                current_state = torch.matmul(current_state.unsqueeze(-2), entanglement_gate.unsqueeze(0))
                current_state = current_state.squeeze(-2)
        
        return current_state
    
    def _create_cnot_matrix(self, control: int, target: int) -> torch.Tensor:
        """Create controlled-NOT gate matrix."""
        device = self.entanglement_params.device
        
        # Initialize identity matrix
        cnot = torch.eye(self.hilbert_dim, dtype=torch.complex64, device=device)
        
        # Apply controlled-X operation
        for state in range(self.hilbert_dim):
            binary_state = format(state, f'0{self.num_qubits}b')
            
            # Check if control qubit is 1
            if binary_state[control] == '1':
                # Flip target qubit
                target_state = list(binary_state)
                target_state[target] = '0' if target_state[target] == '1' else '1'
                target_index = int(''.join(target_state), 2)
                
                # Swap rows in the matrix
                cnot[state], cnot[target_index] = cnot[target_index].clone(), cnot[state].clone()
        
        return cnot
    
    def _create_rotation_matrix(
        self, 
        theta: torch.Tensor, 
        phi: torch.Tensor, 
        lambda_: torch.Tensor, 
        gamma: torch.Tensor
    ) -> torch.Tensor:
        """Create parameterized rotation matrix."""
        device = theta.device
        
        # Single qubit rotation matrix
        cos_half = torch.cos(theta / 2)
        sin_half = torch.sin(theta / 2)
        exp_phi = torch.exp(1j * phi)
        exp_lambda = torch.exp(1j * lambda_)
        exp_gamma = torch.exp(1j * gamma)
        
        rotation_2x2 = torch.stack([
            torch.stack([cos_half, -exp_lambda * sin_half]),
            torch.stack([exp_phi * sin_half, exp_gamma * cos_half])
        ])
        
        # Extend to full Hilbert space (simplified version)
        rotation_full = torch.eye(self.hilbert_dim, dtype=torch.complex64, device=device)
        rotation_full[:2, :2] = rotation_2x2
        
        return rotation_full
    
    def measure_quantum_state(self, quantum_state: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Perform quantum measurements to extract uncertainty information."""
        batch_size = quantum_state.size(0)
        measurements = {}
        
        # Pauli measurements for each qubit
        pauli_names = ['I', 'X', 'Y', 'Z']
        
        for qubit in range(self.num_qubits):
            qubit_measurements = {}
            
            for pauli_idx, pauli_name in enumerate(pauli_names):
                # Extract measurement operator for this qubit and Pauli operator
                measurement_op = self.measurement_ops[qubit, pauli_idx]
                
                # Compute expectation value
                expectation = torch.real(
                    torch.sum(
                        quantum_state.conj() * torch.matmul(quantum_state.unsqueeze(-2), measurement_op.unsqueeze(0)).squeeze(-2),
                        dim=-1
                    )
                )
                
                qubit_measurements[pauli_name] = expectation
            
            measurements[f'qubit_{qubit}'] = qubit_measurements
        
        # Compute quantum uncertainty measures
        measurements['quantum_entropy'] = self._compute_von_neumann_entropy(quantum_state)
        measurements['entanglement_measure'] = self._compute_entanglement_measure(quantum_state)
        
        return measurements
    
    def _compute_von_neumann_entropy(self, quantum_state: torch.Tensor) -> torch.Tensor:
        """Compute von Neumann entropy as uncertainty measure."""
        batch_size = quantum_state.size(0)
        
        # Compute density matrix
        density_matrix = torch.matmul(
            quantum_state.unsqueeze(-1),
            quantum_state.unsqueeze(-2).conj()
        )
        
        # Eigenvalue decomposition
        eigenvalues = torch.linalg.eigvals(density_matrix).real
        eigenvalues = torch.clamp(eigenvalues, min=1e-12)  # Avoid log(0)
        
        # Von Neumann entropy: -Tr(ρ log ρ)
        entropy = -torch.sum(eigenvalues * torch.log(eigenvalues), dim=-1)
        
        return entropy
    
    def _compute_entanglement_measure(self, quantum_state: torch.Tensor) -> torch.Tensor:
        """Compute entanglement measure between qubits."""
        batch_size = quantum_state.size(0)
        
        # Simplified entanglement measure based on Schmidt decomposition
        # Reshape state for bipartition
        half_qubits = self.num_qubits // 2
        left_dim = 2 ** half_qubits
        right_dim = 2 ** (self.num_qubits - half_qubits)
        
        # Reshape quantum state for bipartition
        reshaped_state = quantum_state.view(batch_size, left_dim, right_dim)
        
        # SVD for Schmidt decomposition
        U, S, Vh = torch.linalg.svd(reshaped_state)
        
        # Schmidt coefficients
        schmidt_coeffs = S
        schmidt_coeffs = torch.clamp(schmidt_coeffs, min=1e-12)
        
        # Entanglement entropy
        entanglement = -torch.sum(schmidt_coeffs**2 * torch.log(schmidt_coeffs**2), dim=-1)
        
        return entanglement


class QuantumUncertaintyNeuralOperator(nn.Module):
    """Neural operator with quantum-enhanced uncertainty quantification."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_qubits: int = 8,
        quantum_depth: int = 3,
        modes: int = 20
    ):
        super().__init__()
        
        self.quantum_state_prep = QuantumStatePreparation(
            num_qubits=num_qubits,
            feature_dim=hidden_dim,
            entanglement_depth=quantum_depth
        )
        
        # Classical neural operator layers
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        self.operator_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.operator_layers.append(
                QuantumEnhancedSpectralLayer(hidden_dim, hidden_dim, modes, num_qubits)
            )
        
        # Output projections
        self.mean_projection = nn.Linear(hidden_dim, input_dim)
        self.uncertainty_projection = nn.Linear(hidden_dim + num_qubits * 4, input_dim)
        
    def forward(
        self, 
        x: torch.Tensor,
        return_quantum_info: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, Dict]]:
        """Forward pass with quantum-enhanced uncertainty."""
        
        # Initial projection
        features = self.input_projection(x)
        
        quantum_info = {}
        total_quantum_uncertainty = torch.zeros(x.size(0), device=x.device)
        
        # Apply quantum-enhanced operator layers
        for i, layer in enumerate(self.operator_layers):
            features, layer_quantum_info = layer(features)
            quantum_info[f'layer_{i}'] = layer_quantum_info
            
            # Accumulate quantum uncertainty
            total_quantum_uncertainty += layer_quantum_info['quantum_entropy']
        
        # Mean prediction
        mean_pred = self.mean_projection(features)
        
        # Quantum-enhanced uncertainty prediction
        # Combine classical features with quantum measurements
        quantum_features = []
        for layer_info in quantum_info.values():
            for qubit_info in layer_info['measurements'].values():
                if isinstance(qubit_info, dict):
                    quantum_features.extend(qubit_info.values())
                else:
                    quantum_features.append(qubit_info)
        
        quantum_features = torch.stack(quantum_features, dim=-1)
        combined_features = torch.cat([features, quantum_features], dim=-1)
        uncertainty_pred = F.softplus(self.uncertainty_projection(combined_features))
        
        if return_quantum_info:
            return mean_pred, uncertainty_pred, quantum_info
        else:
            return mean_pred, uncertainty_pred


class QuantumEnhancedSpectralLayer(nn.Module):
    """Spectral convolution layer with quantum enhancement."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes: int,
        num_qubits: int = 8
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        
        # Classical spectral weights
        self.spectral_weights = nn.Parameter(
            torch.randn(in_channels, out_channels, modes, dtype=torch.complex64)
        )
        
        # Quantum state preparation for spectral modes
        self.quantum_state_prep = QuantumStatePreparation(
            num_qubits=num_qubits,
            feature_dim=in_channels,
            entanglement_depth=2
        )
        
        # Local convolution with quantum modulation
        self.local_conv = nn.Conv1d(in_channels, out_channels, 1)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """Forward pass with quantum-enhanced spectral convolution."""
        batch_size, seq_len, channels = x.shape
        
        # Fourier transform
        x_ft = torch.fft.rfft(x, dim=1)
        
        # Quantum enhancement of spectral weights
        # Use quantum state to modulate spectral convolution
        quantum_state = self.quantum_state_prep.prepare_quantum_state(
            x.mean(dim=1)  # Global features for quantum state
        )
        quantum_state = self.quantum_state_prep.apply_entanglement(quantum_state)
        quantum_measurements = self.quantum_state_prep.measure_quantum_state(quantum_state)
        
        # Extract quantum modulation factors
        quantum_modulation = torch.stack([
            measurements['X'] if isinstance(measurements, dict) else measurements
            for measurements in quantum_measurements['measurements'].values() 
            if isinstance(measurements, (dict, torch.Tensor))
        ]).mean(dim=0)
        
        # Apply quantum-modulated spectral convolution
        out_ft = torch.zeros(
            batch_size, x_ft.size(1), self.out_channels,
            dtype=torch.complex64, device=x.device
        )
        
        # Modulate spectral weights with quantum amplitudes
        modulated_weights = self.spectral_weights * (1 + 0.1 * quantum_modulation.unsqueeze(-1).unsqueeze(-1))
        
        for i in range(min(self.modes, x_ft.size(1))):
            out_ft[:, i, :] = torch.einsum('bi,io->bo', x_ft[:, i, :], modulated_weights[:, :, i])
        
        # Inverse FFT
        out_spectral = torch.fft.irfft(out_ft, n=seq_len, dim=1)
        
        # Local convolution
        out_local = self.local_conv(x.transpose(1, 2)).transpose(1, 2)
        
        # Combine spectral and local paths
        output = out_spectral + out_local
        
        return output, quantum_measurements


class QuantumInspiredUncertaintyDecomposition(nn.Module):
    """Quantum-inspired decomposition of uncertainty into fundamental components."""
    
    def __init__(
        self,
        feature_dim: int = 256,
        num_uncertainty_modes: int = 4,
        num_qubits: int = 6
    ):
        super().__init__()
        
        self.num_uncertainty_modes = num_uncertainty_modes
        self.quantum_decomposer = QuantumStatePreparation(
            num_qubits=num_qubits,
            feature_dim=feature_dim,
            entanglement_depth=2
        )
        
        # Uncertainty mode extractors
        self.mode_extractors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feature_dim, feature_dim // 2),
                nn.ReLU(),
                nn.Linear(feature_dim // 2, 1),
                nn.Softplus()
            )
            for _ in range(num_uncertainty_modes)
        ])
        
        # Quantum coherence analyzer
        self.coherence_analyzer = nn.Sequential(
            nn.Linear(2**num_qubits, feature_dim//4),
            nn.ReLU(),
            nn.Linear(feature_dim//4, num_uncertainty_modes),
            nn.Softmax(dim=-1)
        )
        
    def forward(
        self, 
        features: torch.Tensor,
        predictions: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Decompose uncertainty using quantum-inspired methods."""
        
        # Prepare quantum state from features
        quantum_state = self.quantum_decomposer.prepare_quantum_state(features)
        quantum_state = self.quantum_decomposer.apply_entanglement(quantum_state)
        
        # Extract uncertainty modes
        uncertainty_modes = {}
        mode_names = ['aleatoric', 'epistemic', 'spatial', 'temporal']
        
        for i, (extractor, mode_name) in enumerate(zip(self.mode_extractors, mode_names)):
            if i < len(mode_names):
                uncertainty_modes[mode_name] = extractor(features)
        
        # Quantum coherence weights for mode combination
        quantum_amplitude_features = torch.abs(quantum_state)**2
        coherence_weights = self.coherence_analyzer(quantum_amplitude_features)
        
        # Quantum-weighted uncertainty combination
        total_uncertainty = sum(
            weight.unsqueeze(-1) * uncertainty
            for weight, uncertainty in zip(coherence_weights.T, uncertainty_modes.values())
        )
        
        # Add quantum uncertainty measures
        quantum_measurements = self.quantum_decomposer.measure_quantum_state(quantum_state)
        uncertainty_modes['quantum_entropy'] = quantum_measurements['quantum_entropy'].unsqueeze(-1)
        uncertainty_modes['entanglement'] = quantum_measurements['entanglement_measure'].unsqueeze(-1)
        uncertainty_modes['total'] = total_uncertainty
        uncertainty_modes['coherence_weights'] = coherence_weights
        
        return uncertainty_modes


class QuantumVariationalInference(nn.Module):
    """Quantum-inspired variational inference for Bayesian neural operators."""
    
    def __init__(
        self,
        param_dim: int,
        num_qubits: int = 8,
        num_variational_samples: int = 10
    ):
        super().__init__()
        
        self.param_dim = param_dim
        self.num_qubits = num_qubits
        self.num_samples = num_variational_samples
        
        # Quantum state preparation for parameter distribution
        self.quantum_param_encoder = QuantumStatePreparation(
            num_qubits=num_qubits,
            feature_dim=param_dim,
            entanglement_depth=3
        )
        
        # Variational parameters
        self.mean_params = nn.Parameter(torch.randn(param_dim))
        self.log_var_params = nn.Parameter(torch.zeros(param_dim))
        
        # Quantum amplitude decoder
        self.amplitude_decoder = nn.Sequential(
            nn.Linear(2**num_qubits, param_dim * 2),
            nn.Tanh(),
            nn.Linear(param_dim * 2, param_dim)
        )
        
    def sample_quantum_parameters(self, num_samples: int = None) -> torch.Tensor:
        """Sample parameters using quantum-inspired variational distribution."""
        num_samples = num_samples or self.num_samples
        
        samples = []
        
        for _ in range(num_samples):
            # Classical variational sample
            eps = torch.randn_like(self.mean_params)
            classical_sample = self.mean_params + torch.exp(0.5 * self.log_var_params) * eps
            
            # Quantum enhancement
            quantum_state = self.quantum_param_encoder.prepare_quantum_state(
                classical_sample.unsqueeze(0)
            )
            quantum_state = self.quantum_param_encoder.apply_entanglement(quantum_state)
            
            # Decode quantum amplitudes to parameter corrections
            quantum_correction = self.amplitude_decoder(torch.abs(quantum_state)**2)
            
            # Combine classical and quantum contributions
            enhanced_sample = classical_sample + 0.1 * quantum_correction.squeeze(0)
            samples.append(enhanced_sample)
        
        return torch.stack(samples)
    
    def kl_divergence(self) -> torch.Tensor:
        """Compute KL divergence with quantum corrections."""
        
        # Classical KL divergence
        classical_kl = -0.5 * torch.sum(
            1 + self.log_var_params - self.mean_params**2 - self.log_var_params.exp()
        )
        
        # Quantum correction term
        # Use quantum state entropy as regularization
        quantum_state = self.quantum_param_encoder.prepare_quantum_state(
            self.mean_params.unsqueeze(0)
        )
        quantum_entropy = self.quantum_param_encoder._compute_von_neumann_entropy(quantum_state)
        
        # Total KL with quantum regularization
        total_kl = classical_kl - 0.1 * quantum_entropy.sum()
        
        return total_kl
    
    def forward(self, num_samples: int = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample parameters and compute KL divergence."""
        samples = self.sample_quantum_parameters(num_samples)
        kl_div = self.kl_divergence()
        
        return samples, kl_div


def quantum_uncertainty_propagation(
    model: nn.Module,
    inputs: torch.Tensor,
    num_quantum_samples: int = 50,
    num_qubits: int = 8
) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
    """Propagate uncertainty through neural operator using quantum sampling."""
    
    batch_size = inputs.size(0)
    device = inputs.device
    
    # Quantum state preparation for input uncertainty
    quantum_processor = QuantumStatePreparation(
        num_qubits=num_qubits,
        feature_dim=inputs.size(-1),
        entanglement_depth=2
    ).to(device)
    
    predictions = []
    quantum_info = []
    
    for _ in range(num_quantum_samples):
        # Prepare quantum state from inputs
        input_features = inputs.view(batch_size, -1)
        quantum_state = quantum_processor.prepare_quantum_state(input_features)
        quantum_state = quantum_processor.apply_entanglement(quantum_state)
        
        # Measure quantum state to get input perturbation
        measurements = quantum_processor.measure_quantum_state(quantum_state)
        
        # Apply quantum-inspired noise to inputs
        quantum_noise = torch.stack([
            measurements[f'qubit_{i}']['X'] for i in range(num_qubits)
        ], dim=-1).mean(dim=-1, keepdim=True)
        
        perturbed_inputs = inputs + 0.01 * quantum_noise.unsqueeze(-1) * torch.randn_like(inputs)
        
        # Forward pass with perturbed inputs
        with torch.no_grad():
            if hasattr(model, 'forward'):
                pred = model(perturbed_inputs)
                if isinstance(pred, tuple):
                    pred = pred[0]  # Take mean prediction
            else:
                pred = model(perturbed_inputs)
            
        predictions.append(pred)
        quantum_info.append(measurements)
    
    # Combine predictions
    predictions = torch.stack(predictions)
    mean_prediction = predictions.mean(dim=0)
    uncertainty = predictions.std(dim=0)
    
    # Aggregate quantum information
    aggregated_quantum_info = {
        'quantum_entropy': torch.stack([info['quantum_entropy'] for info in quantum_info]).mean(dim=0),
        'entanglement': torch.stack([info['entanglement_measure'] for info in quantum_info]).mean(dim=0)
    }
    
    return mean_prediction, uncertainty, aggregated_quantum_info