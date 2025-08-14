"""
Test suite for multi-fidelity probabilistic neural operators.

Tests research contributions:
- Adaptive fidelity selection
- Information-theoretic fusion
- Cross-fidelity uncertainty propagation
- Cost-aware multi-fidelity training
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from unittest.mock import Mock, MagicMock
from dataclasses import dataclass

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))

from pno_physics_bench.research.multi_fidelity import (
    MultiFidelityPNO,
    FidelityLevel,
    FidelitySelector,
    CrossFidelityUncertaintyPropagator,
    UncertaintyFusionNet,
    AdaptiveRefinementController
)
from pno_physics_bench.models import ProbabilisticNeuralOperator


class MockProbabilisticNeuralOperator(nn.Module):
    """Mock PNO for testing."""
    
    def __init__(self, input_dim=3, hidden_dim=64, resolution=32):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.resolution = resolution
        
        # Simple mock architecture
        self.encoder = nn.Sequential(
            nn.Conv2d(input_dim, hidden_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, input_dim, 3, padding=1)
        )
        
        self.uncertainty_head = nn.Sequential(
            nn.Conv2d(input_dim, hidden_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, input_dim, 1),
            nn.Softplus()
        )
    
    def forward(self, x):
        return self.encoder(x)
    
    def predict_with_uncertainty(self, x):
        prediction = self.forward(x)
        uncertainty = self.uncertainty_head(prediction) + 0.01
        return prediction, uncertainty


class TestFidelityLevel:
    """Test FidelityLevel dataclass."""
    
    def test_fidelity_level_creation(self):
        """Test creating fidelity level specifications."""
        fidelity = FidelityLevel(
            name="low_fidelity",
            resolution=32,
            physics_approximation="euler",
            computational_cost=1.0,
            accuracy_estimate=0.8
        )
        
        assert fidelity.name == "low_fidelity"
        assert fidelity.resolution == 32
        assert fidelity.computational_cost == 1.0
        assert fidelity.accuracy_estimate == 0.8


class TestFidelitySelector:
    """Test fidelity selection mechanism."""
    
    @pytest.fixture
    def selector(self):
        return FidelitySelector(
            input_dim=3,
            num_fidelities=3,
            cost_budget=5.0
        )
    
    @pytest.fixture
    def sample_input(self):
        return torch.randn(4, 3, 32, 32)
    
    def test_initialization(self):
        """Test proper initialization."""
        selector = FidelitySelector(
            input_dim=3,
            num_fidelities=2,
            cost_budget=3.0
        )
        
        assert selector.num_fidelities == 2
        assert selector.cost_budget == 3.0
        assert hasattr(selector, 'feature_extractor')
        assert hasattr(selector, 'fidelity_predictor')
        
    def test_forward_pass(self, selector, sample_input):
        """Test forward pass produces valid fidelity weights."""
        weights = selector(sample_input)
        
        batch_size = sample_input.shape[0]
        expected_shape = (batch_size, selector.num_fidelities)
        
        assert weights.shape == expected_shape
        
        # Weights should sum to 1 (softmax output)
        weight_sums = torch.sum(weights, dim=1)
        assert torch.allclose(weight_sums, torch.ones(batch_size), atol=1e-6)
        
        # Weights should be non-negative
        assert torch.all(weights >= 0)
        
    def test_target_accuracy_adjustment(self, selector, sample_input):
        """Test adjustment for target accuracy requirements."""
        
        # High accuracy should bias toward higher fidelity models
        high_accuracy = 0.9
        weights_high = selector(sample_input, target_accuracy=high_accuracy)
        
        # Low accuracy should allow lower fidelity models
        low_accuracy = 0.5
        weights_low = selector(sample_input, target_accuracy=low_accuracy)
        
        # Higher fidelity models should get more weight for high accuracy
        # (assuming higher indices correspond to higher fidelity)
        high_fidelity_weight_high = weights_high[:, -1].mean()
        high_fidelity_weight_low = weights_low[:, -1].mean()
        
        assert high_fidelity_weight_high >= high_fidelity_weight_low
        
    def test_cost_constraint_handling(self, selector, sample_input):
        """Test cost constraint handling."""
        
        # Test with very low cost constraint
        low_cost = 0.5
        weights_low_cost = selector(sample_input, cost_constraint=low_cost)
        
        # Test with high cost constraint
        high_cost = 10.0
        weights_high_cost = selector(sample_input, cost_constraint=high_cost)
        
        # Should produce valid outputs in both cases
        assert weights_low_cost.shape == weights_high_cost.shape
        assert torch.all(weights_low_cost >= 0)
        assert torch.all(weights_high_cost >= 0)
        
    def test_complexity_estimation(self, selector):
        """Test input complexity estimation functionality."""
        
        # Simple input
        simple_input = torch.zeros(1, 3, 32, 32)
        simple_input[0, 0, 16, 16] = 1.0  # Single point
        
        # Complex input  
        complex_input = torch.randn(1, 3, 32, 32) * 2.0
        
        weights_simple = selector(simple_input)
        weights_complex = selector(complex_input)
        
        # Both should produce valid outputs
        assert weights_simple.shape == (1, selector.num_fidelities)
        assert weights_complex.shape == (1, selector.num_fidelities)


class TestCrossFidelityUncertaintyPropagator:
    """Test cross-fidelity uncertainty propagation."""
    
    @pytest.fixture
    def fidelity_levels(self):
        return [
            FidelityLevel("low", 16, "euler", 1.0, 0.7),
            FidelityLevel("med", 32, "runge_kutta", 2.0, 0.85),
            FidelityLevel("high", 64, "full_navier_stokes", 4.0, 0.95)
        ]
    
    @pytest.fixture
    def propagator(self, fidelity_levels):
        return CrossFidelityUncertaintyPropagator(fidelity_levels)
    
    @pytest.fixture
    def sample_predictions_uncertainties(self):
        predictions = {
            "low": torch.randn(2, 3, 32, 32),
            "med": torch.randn(2, 3, 32, 32),
            "high": torch.randn(2, 3, 32, 32)
        }
        
        uncertainties = {
            "low": torch.rand(2, 3, 32, 32) * 0.2 + 0.01,
            "med": torch.rand(2, 3, 32, 32) * 0.15 + 0.01,
            "high": torch.rand(2, 3, 32, 32) * 0.1 + 0.01
        }
        
        return predictions, uncertainties
    
    def test_initialization(self, fidelity_levels):
        """Test proper initialization."""
        propagator = CrossFidelityUncertaintyPropagator(fidelity_levels)
        
        assert propagator.num_fidelities == len(fidelity_levels)
        assert hasattr(propagator, 'correlation_net')
        assert hasattr(propagator, 'propagation_weights')
        
    def test_forward_propagation(self, propagator, sample_predictions_uncertainties):
        """Test forward propagation of uncertainties."""
        predictions, uncertainties = sample_predictions_uncertainties
        
        propagated = propagator(predictions, uncertainties)
        
        # Should return dictionary with same keys
        assert set(propagated.keys()) == set(uncertainties.keys())
        
        # Shapes should be preserved
        for key in uncertainties.keys():
            assert propagated[key].shape == uncertainties[key].shape
            
        # Uncertainties should remain positive
        for key in propagated.keys():
            assert torch.all(propagated[key] > 0)
            
    def test_correlation_computation(self, propagator, sample_predictions_uncertainties):
        """Test uncertainty correlation computation."""
        predictions, uncertainties = sample_predictions_uncertainties
        
        if len(predictions) < 2:
            pytest.skip("Need at least 2 fidelity levels for correlation test")
            
        # Run propagation
        propagated = propagator(predictions, uncertainties)
        
        # Should complete without errors
        assert len(propagated) >= 2
        
    def test_empty_predictions(self, propagator):
        """Test handling of empty prediction dictionaries."""
        empty_predictions = {}
        empty_uncertainties = {}
        
        result = propagator(empty_predictions, empty_uncertainties)
        
        assert result == empty_uncertainties
        
    def test_single_fidelity(self, propagator):
        """Test behavior with single fidelity level."""
        single_predictions = {"low": torch.randn(1, 3, 16, 16)}
        single_uncertainties = {"low": torch.rand(1, 3, 16, 16) * 0.1 + 0.01}
        
        result = propagator(single_predictions, single_uncertainties)
        
        # Should return the input uncertainties (no propagation needed)
        assert "low" in result
        assert torch.allclose(result["low"], single_uncertainties["low"], atol=1e-4)


class TestUncertaintyFusionNet:
    """Test information-theoretic uncertainty fusion."""
    
    @pytest.fixture
    def fusion_net(self):
        return UncertaintyFusionNet(
            num_fidelities=3,
            hidden_dim=64
        )
    
    @pytest.fixture
    def fusion_inputs(self):
        predictions = {
            "low": torch.randn(2, 3, 16, 16),
            "med": torch.randn(2, 3, 16, 16),  
            "high": torch.randn(2, 3, 16, 16)
        }
        
        uncertainties = {
            "low": torch.rand(2, 3, 16, 16) * 0.2 + 0.01,
            "med": torch.rand(2, 3, 16, 16) * 0.15 + 0.01,
            "high": torch.rand(2, 3, 16, 16) * 0.1 + 0.01
        }
        
        fidelity_weights = torch.rand(2, 3)  # [batch, num_fidelities]
        fidelity_weights = fidelity_weights / fidelity_weights.sum(dim=1, keepdim=True)
        
        return predictions, uncertainties, fidelity_weights
    
    def test_initialization(self):
        """Test proper initialization."""
        fusion_net = UncertaintyFusionNet(num_fidelities=2, hidden_dim=32)
        
        assert fusion_net.num_fidelities == 2
        assert len(fusion_net.info_estimators) == 2
        assert hasattr(fusion_net, 'fusion_net')
        assert hasattr(fusion_net, 'weight_generator')
        
    def test_forward_fusion(self, fusion_net, fusion_inputs):
        """Test forward fusion process."""
        predictions, uncertainties, fidelity_weights = fusion_inputs
        
        result = fusion_net(predictions, uncertainties, fidelity_weights)
        
        # Check required output keys
        required_keys = ["prediction", "uncertainty", "information_weights", "fusion_confidence"]
        for key in required_keys:
            assert key in result
            
        # Check output shapes
        expected_shape = predictions["low"].shape
        assert result["prediction"].shape == expected_shape
        assert result["uncertainty"].shape == expected_shape
        
        # Information weights should have correct shape
        batch_size, num_fidelities = fidelity_weights.shape
        spatial_shape = predictions["low"].shape[2:]
        expected_info_shape = (batch_size, num_fidelities, *spatial_shape)
        assert result["information_weights"].shape == expected_info_shape
        
    def test_information_content_estimation(self, fusion_net, fusion_inputs):
        """Test information content estimation for each fidelity."""
        predictions, uncertainties, fidelity_weights = fusion_inputs
        
        # Test individual information estimators
        for i, (fidelity_name, pred) in enumerate(predictions.items()):
            unc = uncertainties[fidelity_name]
            
            # Create input for information estimator
            pred_unc_input = torch.cat([pred, unc], dim=1)
            info_content = fusion_net.info_estimators[i](pred_unc_input)
            
            # Should be positive and bounded
            assert torch.all(info_content >= 0)
            assert torch.all(info_content <= 1)
            assert info_content.shape[1] == 1  # Single output channel
            
    def test_fusion_with_different_qualities(self, fusion_net):
        """Test fusion behavior with different quality inputs."""
        
        # High quality prediction with low uncertainty
        high_quality_pred = torch.zeros(1, 3, 16, 16)
        high_quality_unc = torch.ones(1, 3, 16, 16) * 0.01
        
        # Low quality prediction with high uncertainty
        low_quality_pred = torch.randn(1, 3, 16, 16) * 2.0
        low_quality_unc = torch.ones(1, 3, 16, 16) * 0.5
        
        predictions = {"high": high_quality_pred, "low": low_quality_pred}
        uncertainties = {"high": high_quality_unc, "low": low_quality_unc}
        
        # Equal weights initially
        fidelity_weights = torch.tensor([[0.5, 0.5]])
        
        result = fusion_net(predictions, uncertainties, fidelity_weights)
        
        # Should produce reasonable fusion
        assert result["prediction"].shape == high_quality_pred.shape
        assert torch.all(result["uncertainty"] > 0)
        
    def test_empty_predictions_error(self, fusion_net):
        """Test error handling for empty predictions."""
        empty_predictions = {}
        empty_uncertainties = {}
        fidelity_weights = torch.rand(1, 0)  # Empty weights
        
        with pytest.raises(ValueError, match="No predictions provided"):
            fusion_net(empty_predictions, empty_uncertainties, fidelity_weights)


class TestMultiFidelityPNO:
    """Test complete multi-fidelity PNO system."""
    
    @pytest.fixture
    def fidelity_levels(self):
        return [
            FidelityLevel("low", 16, "euler", 1.0, 0.7),
            FidelityLevel("med", 32, "runge_kutta", 2.0, 0.85),
            FidelityLevel("high", 64, "full_ns", 4.0, 0.95)
        ]
    
    @pytest.fixture
    def multi_fidelity_pno(self, fidelity_levels):
        return MultiFidelityPNO(
            fidelity_levels=fidelity_levels,
            base_model_class=MockProbabilisticNeuralOperator,
            fusion_strategy="information_theoretic",
            cost_budget=5.0
        )
    
    @pytest.fixture
    def sample_input(self):
        return torch.randn(2, 3, 32, 32)
    
    def test_initialization(self, fidelity_levels):
        """Test proper initialization of multi-fidelity system."""
        mf_pno = MultiFidelityPNO(
            fidelity_levels=fidelity_levels,
            base_model_class=MockProbabilisticNeuralOperator
        )
        
        assert len(mf_pno.fidelity_models) == len(fidelity_levels)
        assert hasattr(mf_pno, 'fidelity_selector')
        assert hasattr(mf_pno, 'uncertainty_propagator')
        
    def test_forward_pass_basic(self, multi_fidelity_pno, sample_input):
        """Test basic forward pass functionality."""
        result = multi_fidelity_pno(sample_input)
        
        # Should return prediction and uncertainty
        assert len(result) == 2
        prediction, uncertainty = result
        
        # Check shapes
        assert prediction.shape == sample_input.shape[:1] + sample_input.shape[1:2] + sample_input.shape[2:]
        assert uncertainty.shape == prediction.shape
        
        # Check validity
        assert torch.all(torch.isfinite(prediction))
        assert torch.all(torch.isfinite(uncertainty))
        assert torch.all(uncertainty > 0)
        
    def test_forward_with_constraints(self, multi_fidelity_pno, sample_input):
        """Test forward pass with accuracy and cost constraints."""
        result = multi_fidelity_pno(
            sample_input,
            target_accuracy=0.9,
            cost_constraint=3.0
        )
        
        prediction, uncertainty = result
        
        # Should still produce valid outputs
        assert prediction.shape == sample_input.shape[:1] + (1,) + sample_input.shape[2:]
        assert uncertainty.shape == prediction.shape
        assert torch.all(uncertainty > 0)
        
    def test_detailed_output(self, multi_fidelity_pno, sample_input):
        """Test detailed output with fidelity information."""
        # Modify forward method to test with detailed output
        # This would require modifying the forward signature in actual implementation
        
        # For now, test that basic forward pass works
        result = multi_fidelity_pno(sample_input)
        assert len(result) == 2
        
    def test_different_fusion_strategies(self, fidelity_levels):
        """Test different fusion strategies."""
        strategies = ["simple", "weighted_ensemble", "information_theoretic"]
        
        for strategy in strategies:
            mf_pno = MultiFidelityPNO(
                fidelity_levels=fidelity_levels,
                base_model_class=MockProbabilisticNeuralOperator,
                fusion_strategy=strategy
            )
            
            sample_input = torch.randn(1, 3, 16, 16)
            result = mf_pno(sample_input)
            
            # Should work for all strategies
            assert len(result) == 2
            prediction, uncertainty = result
            assert torch.all(torch.isfinite(prediction))
            assert torch.all(uncertainty > 0)
            
    def test_input_resolution_adaptation(self, multi_fidelity_pno):
        """Test adaptation to different input resolutions."""
        resolutions = [16, 32, 64]
        
        for resolution in resolutions:
            sample_input = torch.randn(1, 3, resolution, resolution)
            
            # Should handle different resolutions
            result = multi_fidelity_pno(sample_input)
            prediction, uncertainty = result
            
            # Output should match input spatial dimensions
            assert prediction.shape[2:] == sample_input.shape[2:]
            assert uncertainty.shape[2:] == sample_input.shape[2:]
            
    def test_cost_tracking(self, multi_fidelity_pno, sample_input):
        """Test computational cost tracking."""
        # This would require implementing cost tracking in the actual forward method
        # For now, test that the system initializes with cost budget
        
        assert hasattr(multi_fidelity_pno, 'cost_budget')
        assert multi_fidelity_pno.cost_budget == 5.0
        
        # Basic forward pass should work
        result = multi_fidelity_pno(sample_input)
        assert len(result) == 2


class TestAdaptiveRefinementController:
    """Test adaptive refinement for accuracy targets."""
    
    @pytest.fixture
    def fidelity_levels(self):
        return [
            FidelityLevel("low", 16, "euler", 1.0, 0.7),
            FidelityLevel("high", 64, "full_ns", 4.0, 0.95)
        ]
    
    @pytest.fixture
    def controller(self, fidelity_levels):
        return AdaptiveRefinementController(
            fidelity_levels=fidelity_levels,
            threshold=0.1
        )
    
    def test_initialization(self, fidelity_levels):
        """Test proper initialization."""
        controller = AdaptiveRefinementController(fidelity_levels)
        
        assert controller.fidelity_levels == fidelity_levels
        assert controller.threshold == 0.05  # default
        
    def test_refinement_decision(self, controller):
        """Test refinement decision making."""
        
        # Test input
        x = torch.randn(2, 3, 32, 32)
        current_pred = torch.randn(2, 3, 32, 32)
        
        # Low uncertainty - should not refine much
        low_uncertainty = torch.ones(2, 3, 32, 32) * 0.05
        
        result_low = controller.refine(
            x, current_pred, low_uncertainty, target_accuracy=0.9
        )
        
        # High uncertainty - should refine more
        high_uncertainty = torch.ones(2, 3, 32, 32) * 0.2
        
        result_high = controller.refine(
            x, current_pred, high_uncertainty, target_accuracy=0.9
        )
        
        # Both should return valid results
        for result in [result_low, result_high]:
            assert "prediction" in result
            assert "uncertainty" in result
            assert "additional_cost" in result
            
            assert result["prediction"].shape == current_pred.shape
            assert result["uncertainty"].shape == current_pred.shape
            assert result["additional_cost"] >= 0
            
        # High uncertainty case should have higher cost
        assert result_high["additional_cost"] >= result_low["additional_cost"]
        
    def test_no_refinement_needed(self, controller):
        """Test case where no refinement is needed."""
        
        x = torch.randn(1, 3, 16, 16)
        current_pred = torch.randn(1, 3, 16, 16)
        
        # Very low uncertainty
        very_low_unc = torch.ones(1, 3, 16, 16) * 0.01
        
        result = controller.refine(x, current_pred, very_low_unc, target_accuracy=0.8)
        
        # Should return original prediction with no additional cost
        assert torch.allclose(result["prediction"], current_pred, atol=1e-6)
        assert result["additional_cost"] == 0.0


class TestIntegration:
    """Integration tests for multi-fidelity system."""
    
    @pytest.fixture
    def complete_system(self):
        """Create complete multi-fidelity system."""
        fidelity_levels = [
            FidelityLevel("low", 16, "euler", 1.0, 0.7),
            FidelityLevel("med", 32, "runge_kutta", 2.0, 0.85),
            FidelityLevel("high", 64, "full_ns", 4.0, 0.95)
        ]
        
        return MultiFidelityPNO(
            fidelity_levels=fidelity_levels,
            base_model_class=MockProbabilisticNeuralOperator,
            fusion_strategy="weighted_ensemble",
            cost_budget=10.0,
            adaptive_threshold=0.1
        )
    
    def test_end_to_end_pipeline(self, complete_system):
        """Test complete end-to-end multi-fidelity pipeline."""
        
        # Various input conditions
        test_cases = [
            # (input_shape, target_accuracy, cost_constraint)
            ((1, 3, 16, 16), None, None),
            ((2, 3, 32, 32), 0.9, 5.0),
            ((1, 3, 64, 64), 0.8, 15.0),
        ]
        
        for input_shape, target_acc, cost_const in test_cases:
            sample_input = torch.randn(*input_shape)
            
            result = complete_system(
                sample_input,
                target_accuracy=target_acc,
                cost_constraint=cost_const
            )
            
            prediction, uncertainty = result
            
            # Verify outputs
            assert prediction.shape[0] == input_shape[0]  # Batch dimension
            assert prediction.shape[2:] == input_shape[2:]  # Spatial dimensions
            assert uncertainty.shape == prediction.shape
            
            # Verify validity
            assert torch.all(torch.isfinite(prediction))
            assert torch.all(torch.isfinite(uncertainty))
            assert torch.all(uncertainty > 0)
            
    def test_gradient_flow(self, complete_system):
        """Test gradient flow through multi-fidelity system."""
        
        sample_input = torch.randn(1, 3, 16, 16, requires_grad=True)
        
        prediction, uncertainty = complete_system(sample_input)
        loss = torch.sum(prediction) + torch.sum(uncertainty)
        
        loss.backward()
        
        # Check that gradients flow back to input
        assert sample_input.grad is not None
        assert torch.any(sample_input.grad != 0)
        
    def test_memory_efficiency(self, complete_system):
        """Test memory efficiency with larger inputs."""
        
        # Large input to test memory handling
        large_input = torch.randn(4, 3, 64, 64)
        
        # Should complete without memory errors
        result = complete_system(large_input)
        prediction, uncertainty = result
        
        assert prediction.shape[0] == 4  # Batch size preserved
        assert torch.all(torch.isfinite(prediction))
        
    @pytest.mark.parametrize("batch_size", [1, 2, 4, 8])
    def test_batch_processing(self, complete_system, batch_size):
        """Test batch processing capabilities."""
        
        batch_input = torch.randn(batch_size, 3, 32, 32)
        
        prediction, uncertainty = complete_system(batch_input)
        
        assert prediction.shape[0] == batch_size
        assert uncertainty.shape[0] == batch_size
        
    def test_different_cost_budgets(self, complete_system):
        """Test behavior with different cost budgets."""
        
        sample_input = torch.randn(1, 3, 32, 32)
        
        cost_budgets = [1.0, 5.0, 10.0, 20.0]
        
        for budget in cost_budgets:
            result = complete_system(
                sample_input,
                cost_constraint=budget
            )
            
            prediction, uncertainty = result
            
            # Should produce valid results regardless of budget
            assert torch.all(torch.isfinite(prediction))
            assert torch.all(uncertainty > 0)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])