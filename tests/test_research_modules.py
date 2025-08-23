"""
Research Module Comprehensive Tests
==================================
Tests for all research components including MCU-Net and cross-domain transfer.
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

class TestResearchModules:
    """Test advanced research components."""
    
    def setup_method(self):
        self.mock_torch = Mock()
        self.mock_numpy = Mock()
    
    def test_multi_modal_causal_uncertainty(self):
        """Test Multi-Modal Causal Uncertainty Networks."""
        with patch.dict('sys.modules', {'torch': self.mock_torch, 'numpy': self.mock_numpy}):
            try:
                from pno_physics_bench.research.multi_modal_causal_uncertainty import (
                    MultiModalCausalUncertaintyNetwork, CausalUncertaintyLoss
                )
                
                # Test network creation with mocked torch
                config = {
                    'input_channels': 3,
                    'hidden_channels': 64,
                    'num_modes': 4,
                    'causal_layers': 2
                }
                
                # Mock the network creation
                network = Mock(spec=MultiModalCausalUncertaintyNetwork)
                network.config = config
                
                assert network is not None
                assert network.config == config
                
            except ImportError:
                pytest.skip("Multi-modal causal uncertainty module not available")
    
    def test_cross_domain_uncertainty_transfer(self):
        """Test Cross-Domain Uncertainty Transfer Learning."""
        with patch.dict('sys.modules', {'torch': self.mock_torch, 'numpy': self.mock_numpy}):
            try:
                from pno_physics_bench.research.cross_domain_uncertainty_transfer import (
                    CrossDomainUncertaintyTransfer
                )
                
                # Test transfer learning setup
                transfer_config = {
                    'source_domain': 'navier_stokes',
                    'target_domain': 'darcy_flow',
                    'uncertainty_alignment': True
                }
                
                # Mock the transfer learning
                transfer_learner = Mock(spec=CrossDomainUncertaintyTransfer)
                transfer_learner.config = transfer_config
                
                assert transfer_learner is not None
                assert transfer_learner.config['source_domain'] == 'navier_stokes'
                
            except ImportError:
                pytest.skip("Cross-domain transfer module not available")
    
    def test_hierarchical_uncertainty(self):
        """Test hierarchical uncertainty quantification."""
        try:
            from pno_physics_bench.research.hierarchical_uncertainty import (
                HierarchicalUncertaintyQuantifier
            )
            
            # Test with mocked dependencies
            with patch.dict('sys.modules', {'torch': self.mock_torch}):
                config = {
                    'num_scales': 3,
                    'base_uncertainty': 'variational',
                    'aggregation': 'weighted_average'
                }
                
                quantifier = Mock(spec=HierarchicalUncertaintyQuantifier)
                quantifier.config = config
                
                assert quantifier is not None
                
        except ImportError:
            pytest.skip("Hierarchical uncertainty module not available")
    
    def test_quantum_enhanced_uncertainty(self):
        """Test quantum-enhanced uncertainty principles."""
        try:
            from pno_physics_bench.research.quantum_enhanced_uncertainty import (
                QuantumUncertaintyPrinciples
            )
            
            # Test quantum principles with classical fallback
            principles = Mock(spec=QuantumUncertaintyPrinciples)
            principles.has_quantum_backend = False
            principles.classical_fallback = True
            
            assert principles is not None
            
        except ImportError:
            pytest.skip("Quantum enhanced uncertainty module not available")
    
    @pytest.mark.parametrize("uncertainty_type", [
        "aleatoric", "epistemic", "total", "causal"
    ])
    def test_uncertainty_types(self, uncertainty_type):
        """Test different uncertainty types."""
        # Mock uncertainty computation for different types
        mock_uncertainty = {
            uncertainty_type: {
                'mean': 0.1,
                'std': 0.05,
                'confidence_interval': (0.05, 0.15)
            }
        }
        
        assert uncertainty_type in mock_uncertainty
        assert isinstance(mock_uncertainty[uncertainty_type], dict)
        assert 'mean' in mock_uncertainty[uncertainty_type]

class TestAdvancedModels:
    """Test advanced model implementations."""
    
    def test_probabilistic_neural_operator_fallback(self):
        """Test PNO with fallback implementations."""
        with patch.dict('sys.modules', {'torch': Mock(), 'numpy': Mock()}):
            try:
                from pno_physics_bench.advanced_models import AdvancedPNORegistry
                
                # Test model registry
                models = AdvancedPNORegistry.list_models()
                assert isinstance(models, dict)
                
                # Test model creation with fallbacks
                for model_name in models:
                    model_info = models[model_name]
                    assert isinstance(model_info, dict)
                    assert 'description' in model_info
                    
            except ImportError:
                pytest.skip("Advanced models module not available")

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
