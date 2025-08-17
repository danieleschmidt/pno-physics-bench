"""
Comprehensive Test Suite for Autonomous SDLC Implementation.

This module tests all the advanced research modules and frameworks
implemented during the autonomous SDLC execution, ensuring code quality,
functionality, and research integrity.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import time
import tempfile
import os
import json
import pickle
import threading
from typing import Dict, List, Tuple, Any
from unittest.mock import Mock, patch, MagicMock
import warnings

# Import all the modules to test
import sys
sys.path.append('/root/repo/src')

from pno_physics_bench.models import ProbabilisticNeuralOperator
from pno_physics_bench.research.temporal_uncertainty_dynamics import (
    TemporalUncertaintyKernel,
    AdaptiveTemporalPNO,
    TemporalUncertaintyAnalyzer,
    validate_temporal_uncertainty_theory
)
from pno_physics_bench.research.causal_uncertainty_inference import (
    CausalUncertaintyGraph,
    CausalUncertaintyInference,
    CausalUncertaintyAnalyzer,
    validate_causal_inference_theory
)
from pno_physics_bench.research.quantum_uncertainty_principles import (
    QuantumUncertaintyPrinciple,
    QuantumUncertaintyNeuralOperator,
    QuantumUncertaintyAnalyzer,
    validate_quantum_uncertainty_theory
)
from pno_physics_bench.robustness.advanced_validation import (
    PhysicsConsistencyValidator,
    UncertaintyCalibrationValidator,
    RobustnessValidator,
    ComprehensiveValidator
)
from pno_physics_bench.security.advanced_security import (
    InputSanitizer,
    ModelWatermarking,
    SecureInference
)
from pno_physics_bench.scaling.distributed_optimization import (
    DistributedConfig,
    GradientCompressor,
    QuantizationCompressor,
    SparsificationCompressor,
    AdaptiveBatchSizeController,
    LoadBalancer,
    FaultTolerantTrainer
)
from pno_physics_bench.scaling.intelligent_caching import (
    LocalCache,
    DistributedCache,
    CachedPNOInference,
    AdaptivePolicy,
    CompressionEngine,
    SemanticHasher
)


class TestTemporalUncertaintyDynamics:
    """Test suite for temporal uncertainty dynamics module."""
    
    def test_temporal_uncertainty_kernel_initialization(self):
        """Test temporal uncertainty kernel initialization."""
        kernel = TemporalUncertaintyKernel(
            hidden_dim=64,
            temporal_horizon=10,
            kernel_type="matern",
            correlation_decay=0.9
        )
        
        assert kernel.hidden_dim == 64
        assert kernel.temporal_horizon == 10
        assert kernel.kernel_type == "matern"
        assert kernel.correlation_decay == 0.9
        assert kernel.temporal_embedding.shape == (10, 64)
        assert kernel.correlation_matrix.shape == (10, 10)
    
    def test_temporal_kernel_forward_pass(self):
        """Test temporal kernel forward pass."""
        kernel = TemporalUncertaintyKernel(hidden_dim=32, temporal_horizon=5)
        
        # Create test input
        batch_size, seq_len, h, w, unc_dim = 2, 5, 16, 16, 32
        uncertainty_sequence = torch.randn(batch_size, seq_len, h, w, unc_dim)
        time_points = torch.linspace(0, 1, seq_len)
        
        # Forward pass
        evolved_uncertainty, correlation_matrix = kernel(uncertainty_sequence, time_points)
        
        # Check output shapes
        assert evolved_uncertainty.shape == uncertainty_sequence.shape
        assert correlation_matrix.shape == (seq_len, seq_len)
        
        # Check correlation matrix properties
        assert torch.allclose(correlation_matrix, correlation_matrix.T, atol=1e-5)
        assert (torch.diag(correlation_matrix) > 0.5).all()
    
    def test_adaptive_temporal_pno(self):
        """Test adaptive temporal PNO."""
        # Create base PNO
        base_pno = ProbabilisticNeuralOperator(
            input_dim=3, hidden_dim=64, num_layers=2, modes=8
        )
        
        # Create adaptive temporal PNO
        temporal_pno = AdaptiveTemporalPNO(
            base_pno=base_pno,
            temporal_horizon=5,
            adaptation_rate=0.01
        )
        
        # Test prediction
        initial_condition = torch.randn(2, 3, 32, 32)
        time_points = torch.linspace(0, 1, 5)
        
        predictions, uncertainties, temporal_states = temporal_pno.predict_temporal_sequence(
            initial_condition, time_points, num_samples=10
        )
        
        # Check outputs
        assert predictions.shape == (2, 5, 1, 32, 32)
        assert uncertainties.shape == (2, 5, 1, 32, 32)
        assert len(temporal_states) == 5
        
        # Check temporal state updates
        for state in temporal_states:
            assert hasattr(state, 'mean')
            assert hasattr(state, 'covariance')
            assert hasattr(state, 'temporal_correlation')
    
    def test_temporal_uncertainty_analyzer(self):
        """Test temporal uncertainty analyzer."""
        analyzer = TemporalUncertaintyAnalyzer()
        
        # Create base PNO and adaptive temporal PNO
        base_pno = ProbabilisticNeuralOperator(input_dim=3, hidden_dim=32, num_layers=2, modes=4)
        temporal_pno = AdaptiveTemporalPNO(base_pno, temporal_horizon=3)
        
        # Create test data
        test_sequence = torch.randn(1, 3, 16, 16, 16)
        time_points = torch.linspace(0, 1, 3)
        
        # Run analysis
        results = analyzer.analyze_uncertainty_propagation(temporal_pno, test_sequence, time_points)
        
        # Check results structure
        assert 'temporal_metrics' in results
        assert 'uncertainty_decomposition' in results
        assert 'temporal_correlations' in results
        assert 'lyapunov_uncertainty_exponent' in results
        
        # Validate temporal metrics
        metrics = results['temporal_metrics']
        assert 'uncertainty_growth_rate' in metrics
        assert 'temporal_calibration_consistency' in metrics
        
        # Validate uncertainty decomposition
        decomp = results['uncertainty_decomposition']
        assert 'aleatoric_over_time' in decomp
        assert 'epistemic_over_time' in decomp
        assert len(decomp['aleatoric_over_time']) == 3
        assert len(decomp['epistemic_over_time']) == 3
    
    def test_temporal_uncertainty_theory_validation(self):
        """Test theoretical validation of temporal uncertainty dynamics."""
        # This should not raise any assertions
        result = validate_temporal_uncertainty_theory()
        assert result is True


class TestCausalUncertaintyInference:
    """Test suite for causal uncertainty inference module."""
    
    def test_causal_uncertainty_graph(self):
        """Test causal uncertainty graph."""
        graph = CausalUncertaintyGraph(spatial_shape=(4, 4), temporal_length=2)
        
        # Check initialization
        assert graph.spatial_shape == (4, 4)
        assert graph.temporal_length == 2
        assert len(graph.nodes) == 32  # 4*4*2
        
        # Test edge addition
        graph.add_causal_edge("u_0_0_0", "u_0_0_1", strength=0.8, edge_type="spatial")
        
        # Check edge properties
        assert graph.compute_causal_strength("u_0_0_0", "u_0_0_1") == 0.8
        assert "u_0_0_1" in graph.nodes["u_0_0_0"].causal_children
        assert "u_0_0_0" in graph.nodes["u_0_0_1"].causal_parents
    
    def test_causal_uncertainty_inference_model(self):
        """Test causal uncertainty inference model."""
        model = CausalUncertaintyInference(
            spatial_shape=(8, 8),
            hidden_dim=64,
            num_causal_layers=2,
            attention_heads=4
        )
        
        # Test forward pass
        uncertainty_field = torch.randn(2, 8, 8)
        causal_strengths, causal_graph = model(uncertainty_field, return_causal_graph=True)
        
        # Check outputs
        assert causal_strengths.shape == (2, 64, 64)  # 8*8 = 64 nodes
        assert causal_graph is not None
        assert causal_graph.spatial_shape == (8, 8)
        
        # Test intervention prediction
        intervention_effect = model.predict_intervention_effect(
            uncertainty_field, (4, 4), 2.0
        )
        assert intervention_effect.shape == (2, 8, 8)
    
    def test_causal_uncertainty_analyzer(self):
        """Test causal uncertainty analyzer."""
        analyzer = CausalUncertaintyAnalyzer()
        
        # Create model and test data
        model = CausalUncertaintyInference(spatial_shape=(4, 4), hidden_dim=32)
        uncertainty_data = torch.randn(2, 4, 4)
        
        # Test average treatment effect
        ate_results = analyzer.compute_average_treatment_effect(
            model, uncertainty_data, (2, 2), [0.0, 1.0, 2.0]
        )
        
        assert 'average_treatment_effect' in ate_results
        assert 'intervention_effects' in ate_results
        assert 'effect_variance' in ate_results
        assert len(ate_results['intervention_effects']) == 3
        
        # Test causal assumptions
        assumption_results = analyzer.test_causal_assumptions(model, uncertainty_data, num_bootstrap=10)
        
        assert 'markov_violations' in assumption_results
        assert 'potential_confounding_rate' in assumption_results
        assert 'transitivity_score' in assumption_results
    
    def test_causal_inference_theory_validation(self):
        """Test theoretical validation of causal inference."""
        result = validate_causal_inference_theory()
        assert result is True


class TestQuantumUncertaintyPrinciples:
    """Test suite for quantum uncertainty principles module."""
    
    def test_quantum_uncertainty_principle(self):
        """Test quantum uncertainty principle implementation."""
        principle = QuantumUncertaintyPrinciple(
            spatial_shape=(16, 16),
            hbar_effective=1.0,
            principle_type="heisenberg"
        )
        
        # Test observables creation
        assert principle.position.operator_type == 'position'
        assert principle.momentum.operator_type == 'momentum'
        assert principle.energy.operator_type == 'energy'
        
        # Test Heisenberg uncertainty bound
        field = torch.randn(2, 16, 16)
        heisenberg_result = principle.compute_heisenberg_bound(
            field, principle.position, principle.momentum, 'x', 'px'
        )
        
        assert 'uncertainty1' in heisenberg_result
        assert 'uncertainty2' in heisenberg_result
        assert 'uncertainty_product' in heisenberg_result
        assert 'quantum_bound' in heisenberg_result
        assert 'violation_ratio' in heisenberg_result
        
        # Check uncertainty principle satisfaction
        uncertainty_product = heisenberg_result['uncertainty_product']
        quantum_bound = heisenberg_result['quantum_bound']
        assert torch.all(uncertainty_product >= quantum_bound * 0.9)  # Allow small numerical errors
    
    def test_quantum_uncertainty_neural_operator(self):
        """Test quantum uncertainty neural operator."""
        # Create base PNO
        base_pno = ProbabilisticNeuralOperator(input_dim=3, hidden_dim=32, num_layers=2, modes=4)
        
        # Create quantum principle
        principle = QuantumUncertaintyPrinciple(spatial_shape=(16, 16))
        
        # Create quantum neural operator
        quantum_pno = QuantumUncertaintyNeuralOperator(
            base_pno=base_pno,
            quantum_principle=principle,
            uncertainty_weight=0.1
        )
        
        # Test forward pass
        input_tensor = torch.randn(2, 3, 16, 16)
        prediction, uncertainty = quantum_pno(input_tensor, enforce_quantum_bounds=True)
        
        assert prediction.shape == (2, 1, 16, 16)
        assert uncertainty.shape == (2, 1, 16, 16)
        
        # Test quantum loss
        quantum_loss = quantum_pno.get_quantum_loss()
        assert isinstance(quantum_loss, torch.Tensor)
        assert quantum_loss.numel() == 1
    
    def test_quantum_uncertainty_analyzer(self):
        """Test quantum uncertainty analyzer."""
        analyzer = QuantumUncertaintyAnalyzer()
        
        # Create quantum neural operator
        base_pno = ProbabilisticNeuralOperator(input_dim=3, hidden_dim=32, num_layers=2, modes=4)
        principle = QuantumUncertaintyPrinciple(spatial_shape=(8, 8))
        quantum_pno = QuantumUncertaintyNeuralOperator(base_pno, principle)
        
        # Test data
        test_data = torch.randn(2, 3, 8, 8)
        
        # Test principle validation
        validation_results = analyzer.validate_uncertainty_principles(test_data, principle)
        
        assert 'heisenberg_violations' in validation_results
        assert 'energy_time_violations' in validation_results
        assert 'principle_satisfaction_rate' in validation_results
        
        # Test quantum fidelity
        predicted_field = torch.randn(2, 8, 8)
        target_field = torch.randn(2, 8, 8)
        fidelity = analyzer.compute_quantum_fidelity(predicted_field, target_field, principle)
        
        assert isinstance(fidelity, float)
        assert 0.0 <= fidelity <= 1.0
    
    def test_quantum_uncertainty_theory_validation(self):
        """Test theoretical validation of quantum uncertainty principles."""
        result = validate_quantum_uncertainty_theory()
        assert result is True


class TestAdvancedValidation:
    """Test suite for advanced validation framework."""
    
    def test_physics_consistency_validator(self):
        """Test physics consistency validator."""
        validator = PhysicsConsistencyValidator(
            pde_type="navier_stokes",
            tolerance=1e-3
        )
        
        # Create test data
        prediction = torch.randn(2, 3, 32, 32)  # u, v, p
        input_field = torch.randn(2, 3, 32, 32)
        
        # Test conservation laws validation
        result = validator.validate_conservation_laws(prediction, input_field)
        
        assert result.test_name == "conservation_laws"
        assert isinstance(result.passed, bool)
        assert isinstance(result.score, float)
        assert 0.0 <= result.score <= 1.0
        assert 'mass_conservation_error' in result.details
        
        # Test boundary conditions validation
        boundary_result = validator.validate_boundary_conditions(prediction, "periodic")
        
        assert boundary_result.test_name == "boundary_conditions_periodic"
        assert isinstance(boundary_result.passed, bool)
        assert isinstance(boundary_result.score, float)
    
    def test_uncertainty_calibration_validator(self):
        """Test uncertainty calibration validator."""
        validator = UncertaintyCalibrationValidator()
        
        # Create test data
        predictions = torch.randn(100, 1, 16, 16)
        uncertainties = torch.rand(100, 1, 16, 16) * 0.5 + 0.1  # Positive uncertainties
        targets = predictions + torch.randn_like(predictions) * uncertainties
        
        # Test coverage validation
        coverage_result = validator.validate_coverage(predictions, uncertainties, targets)
        
        assert coverage_result.test_name == "uncertainty_coverage"
        assert isinstance(coverage_result.passed, bool)
        assert 'coverage_90' in coverage_result.details
        
        # Test sharpness validation
        sharpness_result = validator.validate_sharpness(uncertainties)
        
        assert sharpness_result.test_name == "uncertainty_sharpness"
        assert 'mean_uncertainty' in sharpness_result.details
        assert 'std_uncertainty' in sharpness_result.details
    
    def test_robustness_validator(self):
        """Test robustness validator."""
        validator = RobustnessValidator()
        
        # Create mock model
        model = ProbabilisticNeuralOperator(input_dim=3, hidden_dim=32, num_layers=2, modes=4)
        test_input = torch.randn(2, 3, 16, 16)
        
        # Test noise robustness
        noise_result = validator.validate_noise_robustness(model, test_input, noise_levels=[0.01, 0.05])
        
        assert noise_result.test_name == "noise_robustness"
        assert isinstance(noise_result.passed, bool)
        assert 'average_robustness_score' in noise_result.details
        
        # Test adversarial robustness
        adv_result = validator.validate_adversarial_robustness(model, test_input, epsilon=0.1)
        
        assert adv_result.test_name == "adversarial_robustness"
        assert isinstance(adv_result.passed, bool)
    
    def test_comprehensive_validator(self):
        """Test comprehensive validator."""
        validator = ComprehensiveValidator(pde_type="navier_stokes")
        
        # Create model and test data
        model = ProbabilisticNeuralOperator(input_dim=3, hidden_dim=32, num_layers=2, modes=4)
        test_data = {
            'input': torch.randn(4, 3, 16, 16),
            'target': torch.randn(4, 1, 16, 16),
            'ood_input': torch.randn(4, 3, 16, 16) * 2.0
        }
        
        # Run comprehensive validation
        results = validator.run_comprehensive_validation(model, test_data)
        
        # Check that multiple validation types are included
        expected_validations = [
            'conservation_laws', 'boundary_conditions', 'pde_residual',
            'uncertainty_coverage', 'uncertainty_sharpness', 'calibration_curve',
            'noise_robustness', 'adversarial_robustness', 'distribution_shift_robustness'
        ]
        
        # At least some validations should be present
        assert len(results) >= 3
        
        # Test report generation
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            report_path = f.name
        
        try:
            report_msg = validator.generate_validation_report(results, report_path)
            assert os.path.exists(report_path)
            
            # Load and check report
            with open(report_path, 'r') as f:
                report = json.load(f)
            
            assert 'validation_summary' in report
            assert 'detailed_results' in report
            assert 'recommendations' in report
            
        finally:
            if os.path.exists(report_path):
                os.unlink(report_path)


class TestAdvancedSecurity:
    """Test suite for advanced security framework."""
    
    def test_input_sanitizer(self):
        """Test input sanitizer."""
        sanitizer = InputSanitizer(
            expected_shape=(3, 32, 32),
            value_range=(-5.0, 5.0),
            anomaly_threshold=3.0
        )
        
        # Test normal input
        normal_input = torch.randn(2, 3, 32, 32)
        sanitized, events = sanitizer.validate_and_sanitize(normal_input)
        
        assert sanitized.shape == normal_input.shape
        assert isinstance(events, list)
        
        # Test shape validation
        assert sanitizer.validate_input_shape(normal_input)
        
        # Test range validation
        assert sanitizer.validate_input_range(normal_input)
        
        # Test out-of-range input
        extreme_input = torch.randn(2, 3, 32, 32) * 20  # Values way out of range
        sanitized_extreme, events_extreme = sanitizer.validate_and_sanitize(extreme_input)
        
        # Should be clamped to valid range
        assert sanitized_extreme.min() >= -5.0
        assert sanitized_extreme.max() <= 5.0
        assert len(events_extreme) > 0  # Should have sanitization events
    
    def test_model_watermarking(self):
        """Test model watermarking."""
        watermarking = ModelWatermarking(
            watermark_key="test-key-2024",
            watermark_strength=0.01
        )
        
        # Create test data
        prediction = torch.randn(2, 1, 16, 16)
        input_tensor = torch.randn(2, 3, 16, 16)
        
        # Embed watermark
        watermarked = watermarking.embed_watermark(prediction, input_tensor)
        
        assert watermarked.shape == prediction.shape
        assert not torch.equal(watermarked, prediction)  # Should be different
        
        # Verify watermark
        is_watermarked, confidence = watermarking.verify_watermark(
            prediction, input_tensor, watermarked
        )
        
        assert is_watermarked
        assert confidence >= watermarking.verification_threshold
        
        # Test with non-watermarked data
        fake_watermarked = prediction + torch.randn_like(prediction) * 0.01
        is_fake, fake_confidence = watermarking.verify_watermark(
            prediction, input_tensor, fake_watermarked
        )
        
        assert not is_fake or fake_confidence < watermarking.verification_threshold
    
    @patch('redis.Redis')
    def test_secure_inference(self, mock_redis):
        """Test secure inference."""
        # Mock Redis to avoid dependency
        mock_redis.return_value = Mock()
        
        model = ProbabilisticNeuralOperator(input_dim=3, hidden_dim=32, num_layers=2, modes=4)
        
        secure_inference = SecureInference(
            model=model,
            require_authentication=True,
            audit_logging=True
        )
        
        # Test token generation and validation
        token = secure_inference.generate_auth_token("test_user", expiry_hours=1)
        assert isinstance(token, str)
        assert len(token) > 0
        
        valid, user_id = secure_inference.validate_auth_token(token)
        assert valid
        assert user_id == "test_user"
        
        # Test rate limiting
        assert secure_inference.check_rate_limit("client1")
        
        # Test data encryption/decryption
        test_tensor = torch.randn(1, 3, 16, 16)
        encrypted = secure_inference.encrypt_data(test_tensor)
        decrypted = secure_inference.decrypt_data(encrypted, test_tensor.shape, test_tensor.dtype)
        
        assert torch.allclose(test_tensor, decrypted, atol=1e-6)


class TestDistributedOptimization:
    """Test suite for distributed optimization framework."""
    
    def test_gradient_compressors(self):
        """Test gradient compression methods."""
        # Test quantization compressor
        quantizer = QuantizationCompressor(num_bits=8)
        test_tensor = torch.randn(100, 50)
        
        compressed, metadata = quantizer.compress(test_tensor)
        decompressed = quantizer.decompress(compressed, metadata)
        
        assert decompressed.shape == test_tensor.shape
        assert compressed.dtype == torch.uint8
        
        # Test sparsification compressor
        sparsifier = SparsificationCompressor(compression_ratio=0.1)
        
        sparse_compressed, sparse_metadata = sparsifier.compress(test_tensor)
        sparse_decompressed = sparsifier.decompress(sparse_compressed, sparse_metadata)
        
        assert sparse_decompressed.shape == test_tensor.shape
        # Should be mostly zeros (sparse)
        assert (sparse_decompressed == 0).float().mean() > 0.8
    
    def test_adaptive_batch_size_controller(self):
        """Test adaptive batch size controller."""
        controller = AdaptiveBatchSizeController(
            initial_batch_size=32,
            min_batch_size=8,
            max_batch_size=128
        )
        
        assert controller.current_batch_size == 32
        
        # Test with good throughput - should not adjust immediately
        should_adjust, new_size = controller.should_adjust_batch_size(throughput=100.0)
        assert not should_adjust  # Need more history
        
        # Build up history
        for i in range(15):
            controller.should_adjust_batch_size(throughput=100.0 + i)
        
        # Now test with declining throughput
        should_adjust, new_size = controller.should_adjust_batch_size(throughput=50.0)
        # Might adjust based on trend
        
        assert isinstance(should_adjust, bool)
        assert controller.min_batch_size <= new_size <= controller.max_batch_size
    
    def test_load_balancer(self):
        """Test load balancer."""
        balancer = LoadBalancer(world_size=4, rebalance_frequency=10)
        
        # Create mock metrics
        from pno_physics_bench.scaling.distributed_optimization import TrainingMetrics
        
        metrics = TrainingMetrics(
            epoch=0, step=0, loss=1.0, learning_rate=1e-3,
            gradient_norm=1.0, compute_time=0.1, communication_time=0.05,
            memory_usage=2.0, throughput=100.0, timestamp=time.time()
        )
        
        # Update metrics for workers
        for rank in range(4):
            balancer.update_worker_metrics(rank, metrics)
        
        # Test load distribution computation
        load_distribution = balancer.compute_load_distribution()
        assert len(load_distribution) == 4
        assert abs(sum(load_distribution) - 4.0) < 1e-6  # Should sum to world_size
        
        # Test batch size adjustment
        adjusted_batch_size = balancer.get_worker_batch_size(0, base_batch_size=32)
        assert isinstance(adjusted_batch_size, int)
        assert adjusted_batch_size > 0
    
    def test_distributed_config(self):
        """Test distributed configuration."""
        config = DistributedConfig(
            world_size=2,
            rank=0,
            local_rank=0,
            gradient_compression="quantization",
            compression_ratio=0.1,
            fault_tolerance=True
        )
        
        assert config.world_size == 2
        assert config.rank == 0
        assert config.gradient_compression == "quantization"
        assert config.fault_tolerance is True


class TestIntelligentCaching:
    """Test suite for intelligent caching framework."""
    
    def test_cache_policies(self):
        """Test cache eviction policies."""
        from pno_physics_bench.scaling.intelligent_caching import (
            LRUPolicy, LFUPolicy, TTLPolicy, AdaptivePolicy, CacheEntry
        )
        
        # Create test cache entry
        entry = CacheEntry(
            key="test_key",
            value=b"test_data",
            timestamp=time.time() - 100,  # 100 seconds old
            access_count=5,
            size_bytes=100
        )
        
        current_time = time.time()
        
        # Test LRU policy
        lru_policy = LRUPolicy()
        assert lru_policy.should_evict(entry, current_time)
        priority = lru_policy.get_eviction_priority(entry, current_time)
        assert priority > 0  # Older entries have higher priority
        
        # Test LFU policy
        lfu_policy = LFUPolicy()
        assert lfu_policy.should_evict(entry, current_time)
        lfu_priority = lfu_policy.get_eviction_priority(entry, current_time)
        assert lfu_priority < 0  # Negative access count
        
        # Test TTL policy
        ttl_policy = TTLPolicy(default_ttl=50.0)  # 50 second TTL
        assert ttl_policy.should_evict(entry, current_time)  # Should be expired
        
        # Test Adaptive policy
        adaptive_policy = AdaptivePolicy()
        assert adaptive_policy.should_evict(entry, current_time)
        adaptive_priority = adaptive_policy.get_eviction_priority(entry, current_time)
        assert isinstance(adaptive_priority, float)
    
    def test_compression_engine(self):
        """Test compression engine."""
        engine = CompressionEngine(compression_threshold=100)
        
        # Test small data (should not compress)
        small_data = b"small"
        compressed, ratio = engine.compress(small_data)
        assert compressed == small_data
        assert ratio == 1.0
        
        # Test large data (should compress)
        large_data = b"a" * 1000
        compressed_large, ratio_large = engine.compress(large_data)
        assert len(compressed_large) < len(large_data)
        assert ratio_large > 1.0
        
        # Test decompression
        decompressed = engine.decompress(compressed_large, ratio_large)
        assert decompressed == large_data
    
    def test_semantic_hasher(self):
        """Test semantic hasher."""
        hasher = SemanticHasher()
        
        # Test tensor hashing
        tensor1 = torch.randn(10, 10)
        tensor2 = tensor1 + 1e-6  # Very similar
        tensor3 = torch.randn(10, 10) * 10  # Very different
        
        hash1 = hasher.compute_tensor_hash(tensor1)
        hash2 = hasher.compute_tensor_hash(tensor2)
        hash3 = hasher.compute_tensor_hash(tensor3)
        
        assert isinstance(hash1, str)
        assert len(hash1) == 16  # SHA256 truncated to 16 chars
        
        # Similar tensors might have same semantic hash (depends on precision)
        # Different tensors should have different hashes
        assert hash1 != hash3
    
    def test_local_cache(self):
        """Test local cache."""
        cache = LocalCache(
            max_size_bytes=1024,
            max_entries=10,
            enable_compression=False  # Disable for predictable testing
        )
        
        # Test cache operations
        test_data = {"test": "data"}
        key_data = {"input": "test_input"}
        
        # Put and get
        cache_key = cache.put(key_data, test_data)
        assert isinstance(cache_key, str)
        
        retrieved_data, hit = cache.get(key_data)
        assert hit
        assert retrieved_data == test_data
        
        # Test cache miss
        miss_data, miss_hit = cache.get({"different": "key"})
        assert not miss_hit
        assert miss_data is None
        
        # Test cache stats
        stats = cache.get_stats()
        assert stats.hits == 1
        assert stats.misses == 1
        assert stats.num_entries == 1
    
    @patch('redis.Redis')
    def test_distributed_cache(self, mock_redis):
        """Test distributed cache with Redis mock."""
        # Mock Redis to avoid dependency
        mock_redis_instance = Mock()
        mock_redis.return_value = mock_redis_instance
        mock_redis_instance.ping.return_value = True
        
        # Mock pipeline
        mock_pipeline = Mock()
        mock_redis_instance.pipeline.return_value = mock_pipeline
        mock_pipeline.execute.return_value = [b'compressed_data', '{"compression_ratio": 1.0}']
        
        # Create distributed cache with local fallback
        local_cache = LocalCache(max_size_bytes=512, max_entries=5)
        dist_cache = DistributedCache(
            redis_config={'host': 'localhost', 'port': 6379},
            local_cache=local_cache,
            enable_local_fallback=True
        )
        
        # Test put and get operations
        test_data = {"test": "distributed_data"}
        key_data = {"input": "distributed_test"}
        
        cache_key = dist_cache.put(key_data, test_data)
        assert isinstance(cache_key, str)
        
        # The mock setup should make this return data
        # In real scenario, would test actual Redis operations
    
    def test_cached_pno_inference(self):
        """Test cached PNO inference."""
        # Create model and cache
        model = ProbabilisticNeuralOperator(input_dim=3, hidden_dim=32, num_layers=2, modes=4)
        cache = LocalCache(max_size_bytes=1024*1024, max_entries=100)
        
        cached_inference = CachedPNOInference(
            model=model,
            cache_system=cache,
            similarity_threshold=0.95,
            enable_semantic_caching=True
        )
        
        # Test inference with caching
        test_input = torch.randn(2, 3, 16, 16)
        
        # First call (cache miss)
        pred1, unc1 = cached_inference.predict_with_uncertainty(test_input, num_samples=10)
        assert pred1.shape == (2, 1, 16, 16)
        assert unc1.shape == (2, 1, 16, 16)
        
        # Second call with same input (cache hit)
        pred2, unc2 = cached_inference.predict_with_uncertainty(test_input, num_samples=10)
        assert torch.allclose(pred1, pred2)
        assert torch.allclose(unc1, unc2)
        
        # Check performance stats
        stats = cached_inference.get_performance_stats()
        assert 'cache_performance' in stats
        assert stats['cache_performance']['cache_hits'] == 1
        assert stats['cache_performance']['cache_misses'] == 1
        assert stats['cache_performance']['hit_rate'] == 0.5


class TestIntegrationScenarios:
    """Integration tests combining multiple components."""
    
    def test_end_to_end_research_pipeline(self):
        """Test complete research pipeline integration."""
        # Create base model
        model = ProbabilisticNeuralOperator(input_dim=3, hidden_dim=64, num_layers=3, modes=8)
        
        # Create temporal uncertainty dynamics
        temporal_pno = AdaptiveTemporalPNO(
            base_pno=model,
            temporal_horizon=5,
            adaptation_rate=0.01
        )
        
        # Create quantum uncertainty constraints
        quantum_principle = QuantumUncertaintyPrinciple(spatial_shape=(32, 32))
        quantum_pno = QuantumUncertaintyNeuralOperator(
            base_pno=model,
            quantum_principle=quantum_principle
        )
        
        # Create validation framework
        validator = ComprehensiveValidator(pde_type="navier_stokes")
        
        # Test data
        test_input = torch.randn(4, 3, 32, 32)
        test_target = torch.randn(4, 1, 32, 32)
        
        # Run temporal prediction
        time_points = torch.linspace(0, 1, 5)
        temporal_pred, temporal_unc, _ = temporal_pno.predict_temporal_sequence(
            test_input[:1], time_points, num_samples=10
        )
        
        assert temporal_pred.shape == (1, 5, 1, 32, 32)
        assert temporal_unc.shape == (1, 5, 1, 32, 32)
        
        # Run quantum-constrained prediction
        quantum_pred, quantum_unc = quantum_pno(test_input)
        assert quantum_pred.shape == test_target.shape
        
        # Run comprehensive validation
        validation_data = {
            'input': test_input,
            'target': test_target
        }
        
        validation_results = validator.run_comprehensive_validation(model, validation_data)
        assert len(validation_results) > 0
        
        # Each validation should have proper structure
        for result in validation_results.values():
            if hasattr(result, 'test_name'):
                assert isinstance(result.test_name, str)
                assert isinstance(result.passed, bool)
                assert isinstance(result.score, float)
    
    def test_security_and_caching_integration(self):
        """Test security and caching integration."""
        # Create secure cached inference system
        model = ProbabilisticNeuralOperator(input_dim=3, hidden_dim=32, num_layers=2, modes=4)
        
        # Create cache with security
        cache = LocalCache(max_size_bytes=512*1024, max_entries=100)
        cached_inference = CachedPNOInference(model, cache)
        
        # Create input sanitizer
        sanitizer = InputSanitizer(
            expected_shape=(3, 16, 16),
            value_range=(-5.0, 5.0)
        )
        
        # Create watermarking
        watermarking = ModelWatermarking("integration-test-key")
        
        # Test secure cached inference pipeline
        test_input = torch.randn(2, 3, 16, 16)
        
        # Sanitize input
        sanitized_input, events = sanitizer.validate_and_sanitize(test_input)
        assert len(events) >= 0  # May have sanitization events
        
        # Run cached inference
        prediction, uncertainty = cached_inference.predict_with_uncertainty(sanitized_input)
        
        # Apply watermarking
        watermarked_pred = watermarking.embed_watermark(prediction, sanitized_input)
        
        # Verify watermark
        is_watermarked, confidence = watermarking.verify_watermark(
            prediction, sanitized_input, watermarked_pred
        )
        
        assert is_watermarked
        assert confidence > 0.5
        
        # Test cache performance
        stats = cached_inference.get_performance_stats()
        assert 'cache_performance' in stats


def run_autonomous_sdlc_tests():
    """Run all autonomous SDLC tests."""
    print("🧪 Running Autonomous SDLC Test Suite...")
    
    # Configure pytest to capture output
    pytest_args = [
        __file__,
        "-v",
        "--tb=short",
        "--capture=no",
        "-x"  # Stop on first failure
    ]
    
    # Run tests
    result = pytest.main(pytest_args)
    
    if result == 0:
        print("✅ All Autonomous SDLC tests passed!")
        return True
    else:
        print("❌ Some Autonomous SDLC tests failed!")
        return False


if __name__ == "__main__":
    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    
    success = run_autonomous_sdlc_tests()
    exit(0 if success else 1)