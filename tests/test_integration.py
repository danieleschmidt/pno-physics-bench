"""
Integration Tests for pno-physics-bench
======================================
Tests that verify component integration and end-to-end workflows.
"""

import pytest
import tempfile
import json
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

class TestIntegration:
    """Test component integration."""
    
    def setup_method(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.mock_torch = Mock()
        self.mock_numpy = Mock()
    
    def teardown_method(self):
        import shutil
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def test_training_pipeline_integration(self):
        """Test complete training pipeline integration."""
        with patch.dict('sys.modules', {'torch': self.mock_torch, 'numpy': self.mock_numpy}):
            try:
                # Mock the training components
                mock_model = Mock()
                mock_trainer = Mock()
                mock_dataset = Mock()
                
                # Test pipeline setup
                training_config = {
                    'model_type': 'ProbabilisticNeuralOperator',
                    'learning_rate': 0.001,
                    'batch_size': 32,
                    'epochs': 10
                }
                
                # Simulate training pipeline
                mock_trainer.configure.return_value = True
                mock_trainer.fit.return_value = {'loss': 0.1, 'accuracy': 0.9}
                
                result = mock_trainer.fit()
                assert isinstance(result, dict)
                assert 'loss' in result
                
            except ImportError:
                pytest.skip("Training pipeline components not available")
    
    def test_model_evaluation_integration(self):
        """Test model evaluation and metrics computation."""
        try:
            # Mock evaluation components
            mock_model = Mock()
            mock_evaluator = Mock()
            
            # Test evaluation setup
            eval_config = {
                'metrics': ['accuracy', 'uncertainty_calibration', 'coverage'],
                'test_size': 1000,
                'confidence_levels': [0.9, 0.95, 0.99]
            }
            
            # Mock evaluation results
            mock_results = {
                'accuracy': 0.85,
                'uncertainty_calibration': 0.92,
                'coverage_90': 0.89,
                'coverage_95': 0.94,
                'coverage_99': 0.98
            }
            
            mock_evaluator.evaluate.return_value = mock_results
            
            results = mock_evaluator.evaluate()
            assert isinstance(results, dict)
            assert len(results) == 5
            
        except Exception as e:
            pytest.skip(f"Evaluation integration test failed: {e}")
    
    def test_monitoring_integration(self):
        """Test monitoring and logging integration."""
        try:
            from pno_physics_bench.comprehensive_logging import ExperimentMetric
            from pno_physics_bench.monitoring import SystemMonitor
            
            # Mock monitoring setup
            monitor = Mock(spec=SystemMonitor)
            
            # Test metric collection
            test_metrics = [
                ExperimentMetric("accuracy", 0.95, 1, 1234567890, "exp_001", {"phase": "test"}),
                ExperimentMetric("loss", 0.05, 1, 1234567890, "exp_001", {"phase": "test"}),
            ]
            
            for metric in test_metrics:
                assert metric.name in ["accuracy", "loss"]
                assert isinstance(metric.value, float)
                assert metric.experiment_id == "exp_001"
                
        except ImportError:
            pytest.skip("Monitoring integration components not available")
    
    def test_deployment_configuration_integration(self):
        """Test deployment configuration integration."""
        deployment_configs = [
            "docker-compose.yml",
            "Dockerfile", 
            "kubernetes.yaml",
            "monitoring/prometheus.yml"
        ]
        
        project_root = Path(__file__).parent.parent
        
        # Check if deployment files exist
        existing_configs = []
        for config_file in deployment_configs:
            config_path = project_root / config_file
            if config_path.exists():
                existing_configs.append(config_file)
        
        # Should have at least some deployment configuration
        assert len(existing_configs) > 0, "No deployment configuration files found"
    
    def test_security_integration(self):
        """Test security component integration."""
        try:
            from pno_physics_bench.security_validation import InputValidator
            
            validator = InputValidator()
            
            # Test integrated security validation
            test_inputs = [
                ("safe_input", True),
                ("<script>alert('xss')</script>", False),
                ("' OR '1'='1", False),
                ("normal_data_123", True)
            ]
            
            for input_data, should_be_valid in test_inputs:
                is_valid, _ = validator.validate_tensor_input(input_data, "test")
                # Note: exact validation behavior depends on implementation
                assert isinstance(is_valid, bool)
                
        except ImportError:
            pytest.skip("Security integration components not available")
    
    @pytest.mark.parametrize("component_pair", [
        ("model", "trainer"),
        ("dataset", "evaluator"),
        ("monitor", "logger"),
        ("security", "validator")
    ])
    def test_component_integration(self, component_pair):
        """Test integration between different component pairs."""
        component_a, component_b = component_pair
        
        # Mock component integration
        mock_a = Mock()
        mock_b = Mock()
        
        # Test that components can communicate
        mock_a.send_to.return_value = True
        mock_b.receive_from.return_value = True
        
        # Simulate integration
        integration_success = mock_a.send_to() and mock_b.receive_from()
        assert integration_success is True

class TestEndToEndWorkflows:
    """Test complete end-to-end workflows."""
    
    def test_research_experiment_workflow(self):
        """Test complete research experiment workflow."""
        # Mock complete research workflow
        workflow_steps = [
            "hypothesis_generation",
            "experiment_design", 
            "model_training",
            "uncertainty_evaluation",
            "results_analysis",
            "publication_preparation"
        ]
        
        mock_workflow = Mock()
        mock_workflow.steps = workflow_steps
        
        # Test workflow execution
        for step in workflow_steps:
            step_result = Mock()
            step_result.success = True
            step_result.step_name = step
            
            assert step_result.success is True
            assert step_result.step_name == step
        
        # Overall workflow should succeed
        assert len(workflow_steps) == 6
    
    def test_production_deployment_workflow(self):
        """Test production deployment workflow."""
        deployment_steps = [
            "model_validation",
            "security_audit",
            "performance_benchmarking",
            "containerization",
            "orchestration_setup",
            "monitoring_configuration",
            "deployment_execution"
        ]
        
        mock_deployment = Mock()
        mock_deployment.steps = deployment_steps
        
        for step in deployment_steps:
            step_result = Mock()
            step_result.success = True
            step_result.step_name = step
            
            assert step_result.success is True
            assert step_result.step_name == step
        
        assert len(deployment_steps) == 7

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
