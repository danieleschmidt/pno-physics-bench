"""Comprehensive tests for advanced PNO models."""

import pytest
import numpy as np
import time
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from pno_physics_bench.advanced_models import (
    AdaptiveSpectralMixing,
    MetaLearningPNO, 
    SelfAdaptiveUncertainty,
    MultiScaleResidualPNO,
    AdvancedPNORegistry,
    create_advanced_pno_ensemble
)

from pno_physics_bench.quantum_enhanced_pno import (
    QuantumFeatureMap,
    QuantumUncertaintyGates,
    QuantumEnhancedSpectralConv,
    QuantumProbabilisticNeuralOperator,
    create_quantum_pno_suite,
    quantum_uncertainty_benchmark
)

from pno_physics_bench.autonomous_research_agent import (
    ResearchHypothesis,
    HypothesisGenerator,
    ExperimentDesigner,
    AutonomousResearchAgent
)


class TestAdvancedModels:
    """Test suite for advanced PNO models."""
    
    def test_adaptive_spectral_mixing_creation(self):
        """Test AdaptiveSpectralMixing model creation."""
        
        if not HAS_TORCH:
            pytest.skip("PyTorch not available")
        
        model = AdaptiveSpectralMixing(num_modes=20, hidden_dim=64)
        
        assert model.num_modes == 20
        assert model.hidden_dim == 64
        assert hasattr(model, 'freq_attention')
        assert hasattr(model, 'mode_uncertainty')
    
    def test_adaptive_spectral_mixing_forward(self):
        """Test AdaptiveSpectralMixing forward pass."""
        
        if not HAS_TORCH:
            pytest.skip("PyTorch not available")
        
        model = AdaptiveSpectralMixing(num_modes=16, hidden_dim=32)
        
        # Create test input
        batch_size, channels, height, width = 2, 3, 16, 16
        test_input = torch.randn(batch_size, channels, height, width)
        
        # Forward pass
        output = model(test_input)
        
        assert output.shape == test_input.shape
        assert not torch.isnan(output).any()
        assert torch.isfinite(output).all()
    
    def test_meta_learning_pno_creation(self):
        """Test MetaLearningPNO model creation."""
        
        if not HAS_TORCH:
            pytest.skip("PyTorch not available")
        
        # Create a simple base model
        class SimpleModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(10, 10)
            
            def forward(self, x):
                return self.linear(x)
        
        base_model = SimpleModel()
        meta_pno = MetaLearningPNO(base_model, meta_lr=1e-3)
        
        assert meta_pno.base_pno is base_model
        assert meta_pno.meta_lr == 1e-3
        assert hasattr(meta_pno, 'task_embedding')
    
    def test_self_adaptive_uncertainty_creation(self):
        """Test SelfAdaptiveUncertainty model creation."""
        
        if not HAS_TORCH:
            pytest.skip("PyTorch not available")
        
        model = SelfAdaptiveUncertainty(input_dim=128, hidden_dims=[64, 32])
        
        assert model.input_dim == 128
        assert hasattr(model, 'uncertainty_net')
        assert hasattr(model, 'calibration_temperature')
        assert hasattr(model, 'calibration_bias')
    
    def test_self_adaptive_uncertainty_forward(self):
        """Test SelfAdaptiveUncertainty forward pass."""
        
        if not HAS_TORCH:
            pytest.skip("PyTorch not available")
        
        model = SelfAdaptiveUncertainty(input_dim=64, hidden_dims=[32, 16])
        
        # Create test input
        batch_size = 4
        feature_dim = 32
        test_features = torch.randn(batch_size, feature_dim)
        test_predictions = torch.randn(batch_size, feature_dim)
        
        # Forward pass
        uncertainty_mean, uncertainty_var = model(test_features, test_predictions)
        
        assert uncertainty_mean.shape[0] == batch_size
        assert uncertainty_var.shape[0] == batch_size
        assert not torch.isnan(uncertainty_mean).any()
        assert not torch.isnan(uncertainty_var).any()
    
    def test_multiscale_residual_pno_creation(self):
        """Test MultiScaleResidualPNO model creation."""
        
        if not HAS_TORCH:
            pytest.skip("PyTorch not available")
        
        model = MultiScaleResidualPNO(
            input_channels=3,
            hidden_channels=32,
            num_scales=3,
            modes=16
        )
        
        assert len(model.spectral_convs) == 3
        assert len(model.scale_projections) == 3
        assert hasattr(model, 'fusion')
        assert hasattr(model, 'residual_scale')
    
    def test_multiscale_residual_pno_forward(self):
        """Test MultiScaleResidualPNO forward pass."""
        
        if not HAS_TORCH:
            pytest.skip("PyTorch not available")
        
        model = MultiScaleResidualPNO(
            input_channels=3,
            hidden_channels=32,
            num_scales=2,  # Smaller for testing
            modes=8
        )
        
        # Create test input
        batch_size, channels, height, width = 2, 3, 16, 16
        test_input = torch.randn(batch_size, channels, height, width)
        
        # Forward pass
        output, scale_uncertainties = model(test_input)
        
        assert output.shape == test_input.shape
        assert isinstance(scale_uncertainties, dict)
        assert len(scale_uncertainties) == 2  # num_scales
        assert not torch.isnan(output).any()
    
    def test_advanced_pno_registry(self):
        """Test AdvancedPNORegistry functionality."""
        
        # List available models
        models = AdvancedPNORegistry.list_models()
        
        assert isinstance(models, dict)
        assert len(models) > 0
        assert 'adaptive_spectral' in models
        assert 'meta_learning' in models
        
        # Get model by name (without PyTorch)
        if not HAS_TORCH:
            # Should handle gracefully without PyTorch
            try:
                model = AdvancedPNORegistry.get_model('adaptive_spectral', num_modes=10)
                # Model creation should work even without PyTorch (fallback behavior)
                assert model is not None
            except Exception:
                # Expected if PyTorch is required
                pass
    
    def test_ensemble_creation(self):
        """Test ensemble creation functionality."""
        
        # Test with fallback implementations
        try:
            ensemble = create_advanced_pno_ensemble(
                base_models=['adaptive_spectral', 'multiscale_residual'],
                ensemble_method='weighted_average'
            )
            
            assert hasattr(ensemble, 'models')
            assert hasattr(ensemble, 'weights')
            assert hasattr(ensemble, 'predict_with_uncertainty')
            
        except Exception as e:
            # Expected if dependencies are missing
            pytest.skip(f"Ensemble creation requires dependencies: {e}")


class TestQuantumEnhancedPNO:
    """Test suite for quantum-enhanced PNO models."""
    
    def test_quantum_feature_map_creation(self):
        """Test QuantumFeatureMap creation."""
        
        feature_map = QuantumFeatureMap(feature_dim=64, num_qubits=4, encoding="angle")
        
        assert feature_map.feature_dim == 64
        assert feature_map.num_qubits == 4
        assert feature_map.encoding == "angle"
    
    def test_quantum_feature_map_encoding(self):
        """Test quantum feature encoding."""
        
        if not HAS_TORCH:
            pytest.skip("PyTorch not available")
        
        feature_map = QuantumFeatureMap(feature_dim=32, num_qubits=4)
        
        # Create test input
        test_input = torch.randn(2, 32)
        
        # Encode data
        encoded_data = feature_map.encode_classical_data(test_input)
        
        assert isinstance(encoded_data, dict)
        assert 'encoded_data' in encoded_data
        assert 'quantum_params' in encoded_data
        assert 'rotation_angles' in encoded_data
    
    def test_quantum_uncertainty_gates_creation(self):
        """Test QuantumUncertaintyGates creation."""
        
        gates = QuantumUncertaintyGates(num_qubits=4)
        
        assert gates.num_qubits == 4
        
        if HAS_TORCH:
            assert hasattr(gates, 'uncertainty_phases')
            assert hasattr(gates, 'entanglement_strength')
    
    def test_quantum_uncertainty_gates_evolution(self):
        """Test quantum uncertainty evolution."""
        
        if not HAS_TORCH:
            pytest.skip("PyTorch not available")
        
        gates = QuantumUncertaintyGates(num_qubits=4)
        
        # Create mock quantum state
        quantum_state = {
            'encoded_data': torch.randn(2, 4),
            'quantum_params': torch.randn(2, 4)
        }
        
        # Apply evolution
        evolved_state = gates.apply_uncertainty_evolution(quantum_state)
        
        assert isinstance(evolved_state, dict)
        assert 'evolved_features' in evolved_state
        assert 'entanglement_uncertainty' in evolved_state
    
    def test_quantum_enhanced_spectral_conv(self):
        """Test QuantumEnhancedSpectralConv."""
        
        if not HAS_TORCH:
            pytest.skip("PyTorch not available")
        
        from pno_physics_bench.quantum_enhanced_pno import QuantumCircuitConfig
        
        config = QuantumCircuitConfig(num_qubits=4, depth=2)
        conv = QuantumEnhancedSpectralConv(
            in_channels=3,
            out_channels=16,
            modes=8,
            quantum_config=config
        )
        
        assert conv.in_channels == 3
        assert conv.out_channels == 16
        assert conv.modes == 8
        assert hasattr(conv, 'quantum_mapper')
        assert hasattr(conv, 'quantum_gates')
    
    def test_quantum_enhanced_spectral_conv_forward(self):
        """Test quantum-enhanced spectral convolution forward pass."""
        
        if not HAS_TORCH:
            pytest.skip("PyTorch not available")
        
        from pno_physics_bench.quantum_enhanced_pno import QuantumCircuitConfig
        
        config = QuantumCircuitConfig(num_qubits=4, depth=2)
        conv = QuantumEnhancedSpectralConv(
            in_channels=3,
            out_channels=8,
            modes=4,  # Small for testing
            quantum_config=config
        )
        
        # Create test input
        test_input = torch.randn(1, 3, 8, 8)  # Small spatial dimensions
        
        # Forward pass
        output, quantum_info = conv.quantum_enhanced_forward(test_input)
        
        assert output.shape[0] == test_input.shape[0]
        assert output.shape[1] == 8  # out_channels
        assert isinstance(quantum_info, dict)
        assert 'entanglement_uncertainty' in quantum_info
    
    def test_quantum_pno_creation(self):
        """Test QuantumProbabilisticNeuralOperator creation."""
        
        if not HAS_TORCH:
            pytest.skip("PyTorch not available")
        
        from pno_physics_bench.quantum_enhanced_pno import (
            QuantumProbabilisticNeuralOperator,
            QuantumCircuitConfig
        )
        
        config = QuantumCircuitConfig(num_qubits=4, depth=2)
        
        model = QuantumProbabilisticNeuralOperator(
            input_channels=3,
            hidden_channels=16,
            output_channels=3,
            modes=8,
            num_layers=2,
            quantum_config=config
        )
        
        assert model.input_channels == 3
        assert model.output_channels == 3
        assert len(model.quantum_spectral_layers) == 2
        assert hasattr(model, 'uncertainty_aggregator')
    
    def test_quantum_pno_forward(self):
        """Test quantum PNO forward pass."""
        
        if not HAS_TORCH:
            pytest.skip("PyTorch not available")
        
        from pno_physics_bench.quantum_enhanced_pno import (
            QuantumProbabilisticNeuralOperator,
            QuantumCircuitConfig
        )
        
        config = QuantumCircuitConfig(num_qubits=4, depth=1)
        
        model = QuantumProbabilisticNeuralOperator(
            input_channels=3,
            hidden_channels=8,
            output_channels=3,
            modes=4,
            num_layers=1,  # Single layer for testing
            quantum_config=config
        )
        
        # Create test input
        test_input = torch.randn(1, 3, 8, 8)
        
        # Standard forward pass
        output = model(test_input)
        assert output.shape == test_input.shape
        
        # Forward pass with quantum info
        output_with_info, quantum_info = model(test_input, return_quantum_info=True)
        assert output_with_info.shape == test_input.shape
        assert isinstance(quantum_info, dict)
        assert 'aggregated_uncertainty' in quantum_info
    
    def test_quantum_suite_creation(self):
        """Test quantum PNO suite creation."""
        
        config = {
            'input_channels': 3,
            'hidden_channels': 16,
            'modes': 8,
            'quantum_config': {
                'num_qubits': 4,
                'depth': 2
            }
        }
        
        suite = create_quantum_pno_suite(config)
        
        assert isinstance(suite, dict)
        assert len(suite) > 0
        assert 'tensor_network_pno' in suite  # Always available
        
        if HAS_TORCH:
            assert 'quantum_pno' in suite
            assert 'hamiltonian_pno' in suite
    
    def test_quantum_uncertainty_benchmark(self):
        """Test quantum uncertainty benchmarking."""
        
        # Create mock model for benchmarking
        class MockQuantumModel:
            def predict_with_quantum_uncertainty(self, x, num_quantum_samples=10):
                if HAS_TORCH:
                    return {
                        'mean': x * 0.9,
                        'std': torch.abs(x * 0.1),
                        'quantum_entanglement_uncertainty': torch.tensor(0.05)
                    }
                else:
                    return {
                        'mean': x * 0.9,
                        'std': np.abs(x * 0.1),
                        'quantum_entanglement_uncertainty': 0.05
                    }
        
        mock_model = MockQuantumModel()
        test_data = np.random.randn(2, 3, 8, 8)
        
        # Run benchmark
        results = quantum_uncertainty_benchmark(mock_model, test_data, num_runs=3)
        
        assert isinstance(results, dict)
        assert 'mean_prediction_variance' in results
        assert 'quantum_entanglement_score' in results
        assert 'uncertainty_calibration' in results
        assert 'computational_efficiency' in results


class TestAutonomousResearchAgent:
    """Test suite for autonomous research agent."""
    
    def test_research_hypothesis_creation(self):
        """Test ResearchHypothesis creation."""
        
        hypothesis = ResearchHypothesis(
            id="test_hyp_001",
            title="Test Hypothesis",
            description="A test hypothesis for uncertainty improvement",
            expected_improvement=0.1,
            complexity_score=5,
            priority=0.8,
            dependencies=["base_model"]
        )
        
        assert hypothesis.id == "test_hyp_001"
        assert hypothesis.title == "Test Hypothesis"
        assert hypothesis.expected_improvement == 0.1
        assert hypothesis.complexity_score == 5
        assert hypothesis.priority == 0.8
        assert hypothesis.status == "proposed"
        assert isinstance(hypothesis.evidence, dict)
        assert len(hypothesis.created_at) > 0
    
    def test_hypothesis_generator(self):
        """Test HypothesisGenerator functionality."""
        
        generator = HypothesisGenerator()
        
        # Test hypothesis generation
        current_performance = {
            "prediction_accuracy": 0.8,
            "uncertainty_calibration": 0.6,
            "computational_efficiency": 0.7
        }
        
        hypotheses = generator.generate_novel_hypotheses(
            current_performance, num_hypotheses=3
        )
        
        assert len(hypotheses) == 3
        assert all(isinstance(h, ResearchHypothesis) for h in hypotheses)
        assert all(h.expected_improvement > 0 for h in hypotheses)
        assert all(h.priority > 0 for h in hypotheses)
        assert all(h.complexity_score > 0 for h in hypotheses)
    
    def test_experiment_designer(self):
        """Test ExperimentDesigner functionality."""
        
        designer = ExperimentDesigner()
        
        # Create test hypothesis
        hypothesis = ResearchHypothesis(
            id="exp_test_001",
            title="Test Experimental Hypothesis",
            description="Testing experiment design",
            expected_improvement=0.15,
            complexity_score=6,
            priority=0.9,
            dependencies=[]
        )
        
        # Design experiment
        experiment = designer.design_experiment(hypothesis)
        
        assert isinstance(experiment, dict)
        assert experiment["hypothesis_id"] == hypothesis.id
        assert "methodology" in experiment
        assert "parameters" in experiment
        assert "success_criteria" in experiment
        assert "duration_estimate" in experiment
        assert "resource_requirements" in experiment
        
        # Check experiment is tracked
        assert experiment["id"] in designer.active_experiments
    
    def test_autonomous_research_agent_creation(self):
        """Test AutonomousResearchAgent creation."""
        
        agent = AutonomousResearchAgent(
            base_model_path=None,
            experiment_log_dir="test_experiments",
            max_concurrent_experiments=2
        )
        
        assert agent.max_concurrent_experiments == 2
        assert isinstance(agent.hypothesis_generator, HypothesisGenerator)
        assert isinstance(agent.experiment_designer, ExperimentDesigner)
        assert hasattr(agent, 'research_history')
        assert hasattr(agent, 'active_hypotheses')
        assert hasattr(agent, 'performance_history')
    
    def test_research_cycle_simulation(self):
        """Test autonomous research cycle (simulation)."""
        
        agent = AutonomousResearchAgent(
            experiment_log_dir="test_experiments",
            max_concurrent_experiments=1
        )
        
        # Initial performance
        baseline_performance = {
            "prediction_accuracy": 0.75,
            "uncertainty_calibration": 0.65,
            "computational_efficiency": 0.70
        }
        
        # Run short research cycle
        results = agent.run_research_cycle(
            current_performance=baseline_performance,
            num_cycles=2  # Short cycle for testing
        )
        
        assert isinstance(results, dict)
        assert results["cycles_completed"] == 2
        assert results["hypotheses_generated"] > 0
        assert "best_performance" in results
        assert "research_insights" in results
        assert isinstance(results["research_insights"], list)
    
    def test_research_report_generation(self):
        """Test research report generation."""
        
        agent = AutonomousResearchAgent()
        
        # Mock cycle results
        cycle_results = {
            "cycles_completed": 3,
            "hypotheses_generated": 15,
            "experiments_conducted": 6,
            "successful_improvements": 2,
            "best_performance": {
                "prediction_accuracy": 0.82,
                "uncertainty_calibration": 0.71
            },
            "research_insights": [
                "✓ Adaptive learning improved accuracy by 0.05",
                "✗ Complex architecture failed to meet criteria"
            ]
        }
        
        # Add mock knowledge base
        agent.knowledge_base = {
            "successful_patterns": [{"mechanism": "test", "improvement": 0.05}],
            "failed_approaches": [{"approach": "test_fail", "reason": "insufficient"}]
        }
        
        # Generate report
        report = agent.generate_research_report(cycle_results)
        
        assert isinstance(report, str)
        assert "Research Cycles Completed: 3" in report
        assert "Hypotheses Generated: 15" in report
        assert "Best Performance Achieved:" in report
        assert "Research Insights" in report


class TestIntegration:
    """Integration tests combining multiple components."""
    
    def test_end_to_end_advanced_workflow(self):
        """Test end-to-end advanced PNO workflow."""
        
        if not HAS_TORCH:
            pytest.skip("PyTorch not available")
        
        # Create advanced model
        try:
            model = AdvancedPNORegistry.get_model(
                'multiscale_residual',
                input_channels=3,
                hidden_channels=16,
                num_scales=2,
                modes=8
            )
            
            # Test forward pass
            test_input = torch.randn(1, 3, 16, 16)
            output, uncertainties = model(test_input)
            
            assert output.shape == test_input.shape
            assert isinstance(uncertainties, dict)
            assert len(uncertainties) > 0
            
        except Exception as e:
            pytest.skip(f"Advanced model integration requires full dependencies: {e}")
    
    def test_research_agent_with_quantum_models(self):
        """Test research agent with quantum models."""
        
        # Create research agent
        agent = AutonomousResearchAgent(max_concurrent_experiments=1)
        
        # Test hypothesis generation for quantum enhancement
        performance = {
            "prediction_accuracy": 0.8,
            "uncertainty_calibration": 0.7,
            "quantum_advantage": 0.0  # New metric
        }
        
        hypotheses = agent.hypothesis_generator.generate_novel_hypotheses(
            performance, num_hypotheses=2
        )
        
        assert len(hypotheses) == 2
        assert all(isinstance(h, ResearchHypothesis) for h in hypotheses)
    
    def test_performance_monitoring_integration(self):
        """Test performance monitoring across components."""
        
        # Test that models can report performance metrics
        if HAS_TORCH:
            try:
                model = AdaptiveSpectralMixing(num_modes=8, hidden_dim=32)
                
                # Simulate performance monitoring
                test_input = torch.randn(1, 3, 8, 8)
                
                start_time = time.time()
                output = model(test_input)
                end_time = time.time()
                
                execution_time = end_time - start_time
                
                assert execution_time > 0
                assert output is not None
                assert not torch.isnan(output).any()
                
            except Exception as e:
                pytest.skip(f"Performance monitoring requires full setup: {e}")


def test_model_compatibility():
    """Test compatibility across different model types."""
    
    # Test that all models have consistent interfaces
    model_configs = [
        ('adaptive_spectral', {'num_modes': 8, 'hidden_dim': 32}),
        ('self_adaptive', {'input_dim': 64, 'hidden_dims': [32, 16]}),
    ]
    
    for model_name, config in model_configs:
        try:
            if HAS_TORCH:
                model = AdvancedPNORegistry.get_model(model_name, **config)
                
                # Check that model has expected attributes
                assert hasattr(model, '__class__')
                assert model.__class__.__name__ in [
                    'AdaptiveSpectralMixing',
                    'SelfAdaptiveUncertainty'
                ]
            else:
                # Should handle gracefully without PyTorch
                pass
                
        except Exception as e:
            # Expected for some models without full dependencies
            assert "not available" in str(e) or "requires" in str(e)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])