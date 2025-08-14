"""
Test suite for hierarchical uncertainty decomposition research module.

Tests novel research contributions:
- Multi-scale uncertainty decomposition
- Physics-informed uncertainty weighting
- Cross-scale coupling analysis
- Adaptive uncertainty propagation
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from unittest.mock import Mock, patch

# Import modules to test
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))

from pno_physics_bench.research.hierarchical_uncertainty import (
    HierarchicalUncertaintyDecomposer,
    UncertaintyEstimator,
    CrossScaleCouplingNet,
    AdaptiveUncertaintyPropagator
)
from pno_physics_bench.models import BaseNeuralOperator


class MockBaseNeuralOperator(BaseNeuralOperator):
    """Mock base neural operator for testing."""
    
    def __init__(self, input_dim=3, hidden_dim=64):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        self.encoder = nn.Sequential(
            nn.Conv2d(input_dim, hidden_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)
        )
    
    def forward(self, x):
        return self.encoder(x)
    
    def encode(self, x):
        return self.encoder(x)


class TestHierarchicalUncertaintyDecomposer:
    """Test hierarchical uncertainty decomposition functionality."""
    
    @pytest.fixture
    def mock_base_model(self):
        return MockBaseNeuralOperator(input_dim=3, hidden_dim=64)
    
    @pytest.fixture
    def decomposer(self, mock_base_model):
        return HierarchicalUncertaintyDecomposer(
            base_model=mock_base_model,
            scales=[1, 4, 16],
            uncertainty_types=["physics", "boundary", "numerical"],
            coupling_analysis=True
        )
    
    @pytest.fixture
    def sample_data(self):
        batch_size = 2
        channels = 3
        height, width = 32, 32
        
        prediction = torch.randn(batch_size, channels, height, width)
        uncertainty = torch.rand(batch_size, channels, height, width) * 0.1 + 0.01
        
        return prediction, uncertainty
    
    def test_initialization(self, mock_base_model):
        """Test proper initialization of hierarchical decomposer."""
        decomposer = HierarchicalUncertaintyDecomposer(
            base_model=mock_base_model,
            scales=[1, 2, 4],
            uncertainty_types=["physics", "boundary"]
        )
        
        assert len(decomposer.scales) == 3
        assert len(decomposer.uncertainty_types) == 2
        assert len(decomposer.scale_estimators) == 3
        assert len(decomposer.physics_weights) == 2
        
    def test_forward_pass_basic(self, decomposer, sample_data):
        """Test basic forward pass functionality."""
        prediction, uncertainty = sample_data
        
        # Should not raise any exceptions
        results = decomposer(prediction, uncertainty)
        
        # Check required output keys
        required_keys = [
            "scale_uncertainties", "type_uncertainties", 
            "coupling_matrix", "total_uncertainty"
        ]
        
        for key in required_keys:
            assert key in results, f"Missing required output key: {key}"
        
        # Check output shapes
        assert results["total_uncertainty"].shape == prediction.shape
        
    def test_forward_with_physics_params(self, decomposer, sample_data):
        """Test forward pass with physics parameters."""
        prediction, uncertainty = sample_data
        
        physics_params = {
            "reynolds": torch.tensor([1000.0, 2000.0])  # Reynolds numbers
        }
        
        results = decomposer(prediction, uncertainty, physics_params)
        
        # Should complete without errors
        assert "total_uncertainty" in results
        assert results["total_uncertainty"].shape == prediction.shape
        
    def test_scale_uncertainties_shape(self, decomposer, sample_data):
        """Test that scale uncertainties have correct shapes."""
        prediction, uncertainty = sample_data
        results = decomposer(prediction, uncertainty)
        
        scale_uncertainties = results["scale_uncertainties"]
        
        # Should have entry for each scale
        assert len(scale_uncertainties) == len(decomposer.scales)
        
        for scale in decomposer.scales:
            scale_key = f"scale_{scale}"
            assert scale_key in scale_uncertainties
            assert scale_uncertainties[scale_key].shape == prediction.shape
            
    def test_type_uncertainties_output(self, decomposer, sample_data):
        """Test uncertainty type decomposition."""
        prediction, uncertainty = sample_data
        results = decomposer(prediction, uncertainty)
        
        type_uncertainties = results["type_uncertainties"]
        
        # Should have entry for each uncertainty type
        assert len(type_uncertainties) == len(decomposer.uncertainty_types)
        
        for unc_type in decomposer.uncertainty_types:
            assert unc_type in type_uncertainties
            assert type_uncertainties[unc_type].shape == prediction.shape
            
    def test_coupling_matrix_shape(self, decomposer, sample_data):
        """Test cross-scale coupling matrix shape."""
        prediction, uncertainty = sample_data
        results = decomposer(prediction, uncertainty)
        
        coupling_matrix = results["coupling_matrix"]
        batch_size = prediction.shape[0]
        num_scales = len(decomposer.scales)
        
        expected_shape = (batch_size, num_scales, num_scales, 1, 1)
        assert coupling_matrix.shape == expected_shape
        
    def test_physics_scale_weight_computation(self, decomposer):
        """Test physics-informed scale weighting."""
        
        # Test different uncertainty types
        physics_params = {"reynolds": torch.tensor([1000.0])}
        
        # Boundary uncertainty should favor large scales
        boundary_weight = decomposer._compute_physics_scale_weight(
            "boundary", scale=16, physics_params=None
        )
        boundary_weight_small = decomposer._compute_physics_scale_weight(
            "boundary", scale=1, physics_params=None
        )
        
        assert boundary_weight > boundary_weight_small
        
        # Numerical uncertainty should favor small scales
        numerical_weight = decomposer._compute_physics_scale_weight(
            "numerical", scale=1, physics_params=None
        )
        numerical_weight_large = decomposer._compute_physics_scale_weight(
            "numerical", scale=16, physics_params=None
        )
        
        assert numerical_weight > numerical_weight_large
        
    def test_uncertainty_aggregation(self, decomposer, sample_data):
        """Test uncertainty aggregation with coupling effects."""
        prediction, uncertainty = sample_data
        
        # Create mock scale and type uncertainties
        scale_uncertainties = {
            f"scale_{s}": uncertainty * (s / 16) for s in decomposer.scales
        }
        type_uncertainties = {
            unc_type: uncertainty * 0.5 for unc_type in decomposer.uncertainty_types
        }
        
        # Mock coupling matrix
        batch_size = prediction.shape[0]
        num_scales = len(decomposer.scales)
        coupling_matrix = torch.rand(batch_size, num_scales, num_scales, 1, 1) * 0.1
        
        total_unc = decomposer._aggregate_uncertainties(
            scale_uncertainties, type_uncertainties, coupling_matrix
        )
        
        assert total_unc.shape == prediction.shape
        assert torch.all(total_unc > 0)  # Should be positive
        
    @pytest.mark.parametrize("num_scales", [2, 3, 5])
    def test_different_scale_counts(self, mock_base_model, num_scales):
        """Test decomposer with different numbers of scales."""
        scales = [2**i for i in range(num_scales)]
        
        decomposer = HierarchicalUncertaintyDecomposer(
            base_model=mock_base_model,
            scales=scales,
            uncertainty_types=["physics", "boundary"]
        )
        
        prediction = torch.randn(1, 3, 32, 32)
        uncertainty = torch.rand(1, 3, 32, 32) * 0.1
        
        results = decomposer(prediction, uncertainty)
        
        assert len(results["scale_uncertainties"]) == num_scales
        
    def test_gradient_flow(self, decomposer, sample_data):
        """Test that gradients flow properly through the decomposer."""
        prediction, uncertainty = sample_data
        prediction.requires_grad_(True)
        uncertainty.requires_grad_(True)
        
        results = decomposer(prediction, uncertainty)
        loss = torch.sum(results["total_uncertainty"])
        
        loss.backward()
        
        # Check that gradients exist
        assert prediction.grad is not None
        assert uncertainty.grad is not None
        assert torch.any(prediction.grad != 0)


class TestUncertaintyEstimator:
    """Test uncertainty estimator component."""
    
    @pytest.fixture
    def estimator(self):
        return UncertaintyEstimator(
            input_dim=64,
            scale_factor=4,
            uncertainty_dim=3
        )
    
    @pytest.fixture
    def sample_features(self):
        return torch.randn(2, 64, 16, 16)
    
    def test_initialization(self):
        """Test proper initialization."""
        estimator = UncertaintyEstimator(32, 2, 4)
        
        assert estimator.scale_factor == 2
        assert hasattr(estimator, 'feature_extractor')
        assert hasattr(estimator, 'scale_norm')
        
    def test_forward_pass(self, estimator, sample_features):
        """Test forward pass produces correct output shape."""
        output = estimator(sample_features)
        
        expected_shape = (2, 3, 16, 16)  # batch, uncertainty_dim, H, W
        assert output.shape == expected_shape
        
    def test_positive_uncertainty(self, estimator, sample_features):
        """Test that output uncertainties are positive."""
        output = estimator(sample_features)
        
        assert torch.all(output > 0), "All uncertainties should be positive"
        
    def test_different_input_sizes(self, estimator):
        """Test with different input sizes."""
        sizes = [(1, 64, 8, 8), (4, 64, 32, 32), (2, 64, 64, 64)]
        
        for size in sizes:
            input_tensor = torch.randn(*size)
            output = estimator(input_tensor)
            
            expected_shape = (size[0], 3, size[2], size[3])
            assert output.shape == expected_shape


class TestCrossScaleCouplingNet:
    """Test cross-scale coupling network."""
    
    @pytest.fixture
    def coupling_net(self):
        return CrossScaleCouplingNet(
            scales=[1, 4, 16],
            hidden_dim=64
        )
    
    @pytest.fixture
    def scale_features(self):
        """Mock scale features dictionary."""
        features = {}
        for scale in [1, 4, 16]:
            features[f"scale_{scale}"] = torch.randn(2, 64, 8, 8)
        return features
    
    def test_initialization(self):
        """Test proper initialization."""
        scales = [1, 2, 4, 8]
        coupling_net = CrossScaleCouplingNet(scales, hidden_dim=128)
        
        assert coupling_net.num_scales == 4
        assert hasattr(coupling_net, 'cross_attention')
        assert hasattr(coupling_net, 'coupling_predictor')
        
    def test_forward_pass(self, coupling_net, scale_features):
        """Test forward pass with scale features."""
        coupling_matrix = coupling_net(scale_features)
        
        batch_size = 2
        num_scales = 3
        expected_shape = (batch_size, num_scales, num_scales, 1, 1)
        
        assert coupling_matrix.shape == expected_shape
        
    def test_coupling_matrix_properties(self, coupling_net, scale_features):
        """Test properties of coupling matrix."""
        coupling_matrix = coupling_net(scale_features)
        
        # Should be in valid range [0, 1] due to sigmoid
        assert torch.all(coupling_matrix >= 0)
        assert torch.all(coupling_matrix <= 1)
        
    def test_attention_mechanism(self, coupling_net, scale_features):
        """Test that attention mechanism works correctly."""
        
        # Mock the cross_attention forward method to verify it's called
        original_forward = coupling_net.cross_attention.forward
        
        call_count = 0
        def mock_forward(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return original_forward(*args, **kwargs)
        
        coupling_net.cross_attention.forward = mock_forward
        
        coupling_matrix = coupling_net(scale_features)
        
        assert call_count == 1, "Cross attention should be called once"
        
    def test_missing_scale_features(self, coupling_net):
        """Test behavior with missing scale features."""
        incomplete_features = {
            "scale_1": torch.randn(2, 64, 8, 8),
            "scale_4": torch.randn(2, 64, 8, 8)
            # Missing scale_16
        }
        
        # Should handle missing features gracefully or raise informative error
        with pytest.raises((KeyError, RuntimeError)):
            coupling_net(incomplete_features)


class TestAdaptiveUncertaintyPropagator:
    """Test adaptive uncertainty propagation."""
    
    @pytest.fixture
    def mock_base_model(self):
        model = MockBaseNeuralOperator(input_dim=3, hidden_dim=64)
        
        # Add predict method for propagation
        def forward_step(x):
            # Simple forward step for testing
            return model(x)
        
        model.forward_step = forward_step
        return model
    
    @pytest.fixture
    def propagator(self, mock_base_model):
        return AdaptiveUncertaintyPropagator(
            base_model=mock_base_model,
            propagation_steps=5,
            adaptation_threshold=0.1
        )
    
    @pytest.fixture
    def initial_conditions(self):
        initial_state = torch.randn(2, 3, 16, 16)
        initial_uncertainty = torch.rand(2, 3, 16, 16) * 0.05 + 0.01
        return initial_state, initial_uncertainty
    
    def test_initialization(self, mock_base_model):
        """Test proper initialization."""
        propagator = AdaptiveUncertaintyPropagator(
            mock_base_model, propagation_steps=10
        )
        
        assert propagator.propagation_steps == 10
        assert hasattr(propagator, 'propagation_controller')
        assert hasattr(propagator, 'uncertainty_predictor')
        
    def test_forward_propagation(self, propagator, initial_conditions):
        """Test forward propagation produces correct outputs."""
        initial_state, initial_uncertainty = initial_conditions
        
        states, uncertainties = propagator(initial_state, initial_uncertainty)
        
        # Should have correct number of time steps
        assert len(states) == propagator.propagation_steps + 1  # Including initial
        assert len(uncertainties) == propagator.propagation_steps + 1
        
        # All states should have correct shape
        for state in states:
            assert state.shape == initial_state.shape
            
        for uncertainty in uncertainties:
            assert uncertainty.shape == initial_uncertainty.shape
            
    def test_uncertainty_evolution(self, propagator, initial_conditions):
        """Test that uncertainty evolves over time steps."""
        initial_state, initial_uncertainty = initial_conditions
        
        states, uncertainties = propagator(initial_state, initial_uncertainty)
        
        # Uncertainties should remain positive
        for unc in uncertainties:
            assert torch.all(unc > 0), "Uncertainties must remain positive"
            
        # Uncertainties should be bounded
        for unc in uncertainties:
            assert torch.all(unc <= 10.0), "Uncertainties should be bounded"
            
    def test_adaptive_refinement_trigger(self, propagator, initial_conditions):
        """Test that adaptive refinement is triggered for high uncertainty."""
        initial_state, initial_uncertainty = initial_conditions
        
        # Set high initial uncertainty to trigger refinement
        high_uncertainty = torch.ones_like(initial_uncertainty) * 0.2
        
        states, uncertainties = propagator(initial_state, high_uncertainty)
        
        # Should still produce valid outputs
        assert len(states) > 0
        assert len(uncertainties) > 0
        
    def test_gradient_computation(self, propagator, initial_conditions):
        """Test gradient computation in solution evolution."""
        initial_state, initial_uncertainty = initial_conditions
        
        # Test gradient computation method
        test_solution = torch.randn(1, 3, 16, 16)
        gradient_mag = propagator._compute_solution_gradient(test_solution)
        
        assert gradient_mag.shape == test_solution.shape
        assert torch.all(gradient_mag >= 0), "Gradient magnitude should be non-negative"
        
    def test_physics_parameters(self, propagator, initial_conditions):
        """Test propagation with physics parameters."""
        initial_state, initial_uncertainty = initial_conditions
        
        physics_params = {
            "viscosity": torch.tensor([0.01, 0.02]),
            "reynolds": torch.tensor([1000.0, 2000.0])
        }
        
        states, uncertainties = propagator(
            initial_state, initial_uncertainty, physics_params
        )
        
        # Should complete without errors
        assert len(states) == propagator.propagation_steps + 1
        
    @pytest.mark.parametrize("num_steps", [1, 5, 10])
    def test_different_step_counts(self, mock_base_model, initial_conditions, num_steps):
        """Test propagation with different step counts."""
        propagator = AdaptiveUncertaintyPropagator(
            mock_base_model, propagation_steps=num_steps
        )
        
        initial_state, initial_uncertainty = initial_conditions
        states, uncertainties = propagator(initial_state, initial_uncertainty)
        
        assert len(states) == num_steps + 1
        assert len(uncertainties) == num_steps + 1


class TestIntegration:
    """Integration tests for hierarchical uncertainty components."""
    
    @pytest.fixture
    def full_system(self):
        """Create a complete hierarchical uncertainty system."""
        base_model = MockBaseNeuralOperator(input_dim=3, hidden_dim=64)
        
        decomposer = HierarchicalUncertaintyDecomposer(
            base_model=base_model,
            scales=[1, 4],
            uncertainty_types=["physics", "numerical"]
        )
        
        return decomposer, base_model
    
    def test_end_to_end_pipeline(self, full_system):
        """Test complete end-to-end pipeline."""
        decomposer, base_model = full_system
        
        # Create test data
        prediction = torch.randn(1, 3, 16, 16)
        uncertainty = torch.rand(1, 3, 16, 16) * 0.1 + 0.01
        
        # Run complete analysis
        results = decomposer(prediction, uncertainty)
        
        # Verify all components work together
        assert "scale_uncertainties" in results
        assert "type_uncertainties" in results
        assert "coupling_matrix" in results
        assert "total_uncertainty" in results
        
        # Check numerical stability
        assert torch.all(torch.isfinite(results["total_uncertainty"]))
        
    def test_memory_efficiency(self, full_system):
        """Test memory efficiency with larger inputs."""
        decomposer, base_model = full_system
        
        # Large input to test memory handling
        prediction = torch.randn(4, 3, 64, 64)
        uncertainty = torch.rand(4, 3, 64, 64) * 0.1 + 0.01
        
        # Should complete without memory errors
        results = decomposer(prediction, uncertainty)
        
        assert results["total_uncertainty"].shape == prediction.shape
        
    def test_batch_processing(self, full_system):
        """Test batch processing capabilities."""
        decomposer, base_model = full_system
        
        batch_sizes = [1, 2, 4, 8]
        
        for batch_size in batch_sizes:
            prediction = torch.randn(batch_size, 3, 16, 16)
            uncertainty = torch.rand(batch_size, 3, 16, 16) * 0.1 + 0.01
            
            results = decomposer(prediction, uncertainty)
            
            assert results["total_uncertainty"].shape[0] == batch_size


# Performance benchmarking tests
class TestPerformance:
    """Performance tests for hierarchical uncertainty decomposition."""
    
    @pytest.mark.slow
    def test_computational_complexity(self):
        """Test computational complexity scaling."""
        import time
        
        base_model = MockBaseNeuralOperator(input_dim=3, hidden_dim=64)
        decomposer = HierarchicalUncertaintyDecomposer(
            base_model=base_model,
            scales=[1, 2, 4],
            uncertainty_types=["physics"]
        )
        
        sizes = [16, 32, 64]
        times = []
        
        for size in sizes:
            prediction = torch.randn(1, 3, size, size)
            uncertainty = torch.rand(1, 3, size, size) * 0.1 + 0.01
            
            start_time = time.time()
            _ = decomposer(prediction, uncertainty)
            end_time = time.time()
            
            times.append(end_time - start_time)
        
        # Check that time scaling is reasonable (not exponential)
        # This is a basic sanity check
        assert times[2] < times[0] * 20, "Computational complexity seems too high"
        
    @pytest.mark.slow
    def test_memory_scaling(self):
        """Test memory usage scaling."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available for memory testing")
        
        base_model = MockBaseNeuralOperator(input_dim=3, hidden_dim=64).cuda()
        decomposer = HierarchicalUncertaintyDecomposer(
            base_model=base_model,
            scales=[1, 4],
            uncertainty_types=["physics"]
        ).cuda()
        
        sizes = [16, 32]
        memories = []
        
        for size in sizes:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            prediction = torch.randn(1, 3, size, size).cuda()
            uncertainty = torch.rand(1, 3, size, size).cuda() * 0.1 + 0.01
            
            _ = decomposer(prediction, uncertainty)
            
            peak_memory = torch.cuda.max_memory_allocated()
            memories.append(peak_memory)
        
        # Memory should scale reasonably with input size
        assert memories[1] < memories[0] * 10, "Memory scaling seems excessive"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])