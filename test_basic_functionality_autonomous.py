"""
Basic Functionality Tests for Autonomous SDLC Implementation.

This module tests core functionality without external dependencies.
"""

import torch
import torch.nn as nn
import numpy as np
import time
import tempfile
import os
import json
import sys
import traceback

# Add src to path
sys.path.append('/root/repo/src')

def test_temporal_uncertainty_kernel():
    """Test temporal uncertainty kernel."""
    print("Testing Temporal Uncertainty Kernel...")
    
    try:
        from pno_physics_bench.research.temporal_uncertainty_dynamics import TemporalUncertaintyKernel
        
        kernel = TemporalUncertaintyKernel(
            hidden_dim=64,
            temporal_horizon=10,
            kernel_type="matern",
            correlation_decay=0.9
        )
        
        # Test basic properties
        assert kernel.hidden_dim == 64
        assert kernel.temporal_horizon == 10
        assert kernel.kernel_type == "matern"
        
        # Test forward pass
        batch_size, seq_len, h, w, unc_dim = 2, 5, 16, 16, 64
        uncertainty_sequence = torch.randn(batch_size, seq_len, h, w, unc_dim)
        time_points = torch.linspace(0, 1, seq_len)
        
        evolved_uncertainty, correlation_matrix = kernel(uncertainty_sequence, time_points)
        
        assert evolved_uncertainty.shape == uncertainty_sequence.shape
        assert correlation_matrix.shape == (seq_len, seq_len)
        
        print("✅ Temporal Uncertainty Kernel test passed")
        return True
        
    except Exception as e:
        print(f"❌ Temporal Uncertainty Kernel test failed: {e}")
        traceback.print_exc()
        return False


def test_causal_uncertainty_inference():
    """Test causal uncertainty inference."""
    print("Testing Causal Uncertainty Inference...")
    
    try:
        from pno_physics_bench.research.causal_uncertainty_inference import (
            CausalUncertaintyGraph, CausalUncertaintyInference
        )
        
        # Test graph
        graph = CausalUncertaintyGraph(spatial_shape=(4, 4), temporal_length=2)
        assert graph.spatial_shape == (4, 4)
        assert graph.temporal_length == 2
        assert len(graph.nodes) == 32  # 4*4*2
        
        # Test inference model
        model = CausalUncertaintyInference(
            spatial_shape=(8, 8),
            hidden_dim=64,
            num_causal_layers=2,
            attention_heads=4
        )
        
        uncertainty_field = torch.randn(2, 8, 8)
        causal_strengths, causal_graph = model(uncertainty_field, return_causal_graph=True)
        
        assert causal_strengths.shape == (2, 64, 64)
        assert causal_graph is not None
        
        print("✅ Causal Uncertainty Inference test passed")
        return True
        
    except Exception as e:
        print(f"❌ Causal Uncertainty Inference test failed: {e}")
        traceback.print_exc()
        return False


def test_quantum_uncertainty_principles():
    """Test quantum uncertainty principles."""
    print("Testing Quantum Uncertainty Principles...")
    
    try:
        from pno_physics_bench.research.quantum_uncertainty_principles import (
            QuantumUncertaintyPrinciple, QuantumObservable
        )
        
        principle = QuantumUncertaintyPrinciple(
            spatial_shape=(16, 16),
            hbar_effective=1.0,
            principle_type="heisenberg"
        )
        
        # Test observables
        assert principle.position.operator_type == 'position'
        assert principle.momentum.operator_type == 'momentum'
        assert principle.energy.operator_type == 'energy'
        
        # Test quantum state analysis
        field = torch.randn(2, 16, 16)
        quantum_state = principle.analyze_quantum_state(field)
        
        assert hasattr(quantum_state, 'position_uncertainty')
        assert hasattr(quantum_state, 'momentum_uncertainty')
        assert hasattr(quantum_state, 'complementarity_product')
        
        print("✅ Quantum Uncertainty Principles test passed")
        return True
        
    except Exception as e:
        print(f"❌ Quantum Uncertainty Principles test failed: {e}")
        traceback.print_exc()
        return False


def test_advanced_validation():
    """Test advanced validation framework."""
    print("Testing Advanced Validation Framework...")
    
    try:
        from pno_physics_bench.robustness.advanced_validation import (
            PhysicsConsistencyValidator, UncertaintyCalibrationValidator
        )
        
        # Test physics validator
        validator = PhysicsConsistencyValidator(
            pde_type="navier_stokes",
            tolerance=1e-3
        )
        
        prediction = torch.randn(2, 3, 32, 32)  # u, v, p
        input_field = torch.randn(2, 3, 32, 32)
        
        result = validator.validate_conservation_laws(prediction, input_field)
        
        assert result.test_name == "conservation_laws"
        assert isinstance(result.passed, bool)
        assert isinstance(result.score, float)
        assert 0.0 <= result.score <= 1.0
        
        # Test uncertainty validator
        unc_validator = UncertaintyCalibrationValidator()
        
        predictions = torch.randn(100, 1, 16, 16)
        uncertainties = torch.rand(100, 1, 16, 16) * 0.5 + 0.1
        targets = predictions + torch.randn_like(predictions) * uncertainties
        
        coverage_result = unc_validator.validate_coverage(predictions, uncertainties, targets)
        
        assert coverage_result.test_name == "uncertainty_coverage"
        assert isinstance(coverage_result.passed, bool)
        
        print("✅ Advanced Validation Framework test passed")
        return True
        
    except Exception as e:
        print(f"❌ Advanced Validation Framework test failed: {e}")
        traceback.print_exc()
        return False


def test_security_framework():
    """Test security framework."""
    print("Testing Security Framework...")
    
    try:
        from pno_physics_bench.security.advanced_security import (
            InputSanitizer, ModelWatermarking
        )
        
        # Test input sanitizer
        sanitizer = InputSanitizer(
            expected_shape=(3, 32, 32),
            value_range=(-5.0, 5.0),
            anomaly_threshold=3.0
        )
        
        normal_input = torch.randn(2, 3, 32, 32)
        sanitized, events = sanitizer.validate_and_sanitize(normal_input)
        
        assert sanitized.shape == normal_input.shape
        assert isinstance(events, list)
        
        # Test watermarking
        watermarking = ModelWatermarking(
            watermark_key="test-key-2024",
            watermark_strength=0.01
        )
        
        prediction = torch.randn(2, 1, 16, 16)
        input_tensor = torch.randn(2, 3, 16, 16)
        
        watermarked = watermarking.embed_watermark(prediction, input_tensor)
        assert watermarked.shape == prediction.shape
        assert not torch.equal(watermarked, prediction)
        
        is_watermarked, confidence = watermarking.verify_watermark(
            prediction, input_tensor, watermarked
        )
        assert is_watermarked
        assert confidence >= watermarking.verification_threshold
        
        print("✅ Security Framework test passed")
        return True
        
    except Exception as e:
        print(f"❌ Security Framework test failed: {e}")
        traceback.print_exc()
        return False


def test_distributed_optimization():
    """Test distributed optimization framework."""
    print("Testing Distributed Optimization Framework...")
    
    try:
        from pno_physics_bench.scaling.distributed_optimization import (
            DistributedConfig, QuantizationCompressor, SparsificationCompressor,
            AdaptiveBatchSizeController, LoadBalancer
        )
        
        # Test configuration
        config = DistributedConfig(
            world_size=2,
            rank=0,
            local_rank=0,
            gradient_compression="quantization",
            fault_tolerance=True
        )
        
        assert config.world_size == 2
        assert config.gradient_compression == "quantization"
        
        # Test gradient compression
        quantizer = QuantizationCompressor(num_bits=8)
        test_tensor = torch.randn(100, 50)
        
        compressed, metadata = quantizer.compress(test_tensor)
        decompressed = quantizer.decompress(compressed, metadata)
        
        assert decompressed.shape == test_tensor.shape
        assert compressed.dtype == torch.uint8
        
        # Test adaptive batch size controller
        controller = AdaptiveBatchSizeController(
            initial_batch_size=32,
            min_batch_size=8,
            max_batch_size=128
        )
        
        assert controller.current_batch_size == 32
        
        # Test load balancer
        balancer = LoadBalancer(world_size=4, rebalance_frequency=10)
        load_distribution = balancer.compute_load_distribution()
        assert len(load_distribution) == 4
        
        print("✅ Distributed Optimization Framework test passed")
        return True
        
    except Exception as e:
        print(f"❌ Distributed Optimization Framework test failed: {e}")
        traceback.print_exc()
        return False


def test_intelligent_caching():
    """Test intelligent caching framework."""
    print("Testing Intelligent Caching Framework...")
    
    try:
        from pno_physics_bench.scaling.intelligent_caching import (
            LocalCache, CompressionEngine, SemanticHasher, CachedPNOInference
        )
        from pno_physics_bench.models import ProbabilisticNeuralOperator
        
        # Test compression engine
        engine = CompressionEngine(compression_threshold=100)
        
        small_data = b"small"
        compressed, ratio = engine.compress(small_data)
        assert compressed == small_data
        assert ratio == 1.0
        
        # Test semantic hasher
        hasher = SemanticHasher()
        tensor = torch.randn(10, 10)
        hash_val = hasher.compute_tensor_hash(tensor)
        assert isinstance(hash_val, str)
        assert len(hash_val) == 16
        
        # Test local cache
        cache = LocalCache(
            max_size_bytes=1024,
            max_entries=10,
            enable_compression=False
        )
        
        test_data = {"test": "data"}
        key_data = {"input": "test_input"}
        
        cache_key = cache.put(key_data, test_data)
        assert isinstance(cache_key, str)
        
        retrieved_data, hit = cache.get(key_data)
        assert hit
        assert retrieved_data == test_data
        
        # Test cached inference
        model = ProbabilisticNeuralOperator(input_dim=3, hidden_dim=32, num_layers=2, modes=4)
        cached_inference = CachedPNOInference(model, cache)
        
        test_input = torch.randn(2, 3, 16, 16)
        pred1, unc1 = cached_inference.predict_with_uncertainty(test_input, num_samples=10)
        pred2, unc2 = cached_inference.predict_with_uncertainty(test_input, num_samples=10)
        
        assert torch.allclose(pred1, pred2)
        assert torch.allclose(unc1, unc2)
        
        print("✅ Intelligent Caching Framework test passed")
        return True
        
    except Exception as e:
        print(f"❌ Intelligent Caching Framework test failed: {e}")
        traceback.print_exc()
        return False


def test_research_theory_validation():
    """Test research theory validation functions."""
    print("Testing Research Theory Validation...")
    
    try:
        from pno_physics_bench.research.temporal_uncertainty_dynamics import validate_temporal_uncertainty_theory
        from pno_physics_bench.research.causal_uncertainty_inference import validate_causal_inference_theory
        from pno_physics_bench.research.quantum_uncertainty_principles import validate_quantum_uncertainty_theory
        
        # Test temporal theory validation
        temporal_result = validate_temporal_uncertainty_theory()
        assert temporal_result is True
        
        # Test causal theory validation
        causal_result = validate_causal_inference_theory()
        assert causal_result is True
        
        # Test quantum theory validation
        quantum_result = validate_quantum_uncertainty_theory()
        assert quantum_result is True
        
        print("✅ Research Theory Validation test passed")
        return True
        
    except Exception as e:
        print(f"❌ Research Theory Validation test failed: {e}")
        traceback.print_exc()
        return False


def test_integration_scenario():
    """Test integration scenario."""
    print("Testing Integration Scenario...")
    
    try:
        from pno_physics_bench.models import ProbabilisticNeuralOperator
        from pno_physics_bench.research.temporal_uncertainty_dynamics import AdaptiveTemporalPNO
        from pno_physics_bench.research.quantum_uncertainty_principles import (
            QuantumUncertaintyPrinciple, QuantumUncertaintyNeuralOperator
        )
        from pno_physics_bench.scaling.intelligent_caching import LocalCache, CachedPNOInference
        
        # Create base model
        model = ProbabilisticNeuralOperator(input_dim=3, hidden_dim=32, num_layers=2, modes=4)
        
        # Create temporal model
        temporal_pno = AdaptiveTemporalPNO(
            base_pno=model,
            temporal_horizon=3,
            adaptation_rate=0.01
        )
        
        # Create quantum model
        quantum_principle = QuantumUncertaintyPrinciple(spatial_shape=(16, 16))
        quantum_pno = QuantumUncertaintyNeuralOperator(model, quantum_principle)
        
        # Create cached inference
        cache = LocalCache(max_size_bytes=1024*1024, max_entries=100)
        cached_inference = CachedPNOInference(model, cache)
        
        # Test data
        test_input = torch.randn(2, 3, 16, 16)
        
        # Test temporal prediction
        time_points = torch.linspace(0, 1, 3)
        temporal_pred, temporal_unc, _ = temporal_pno.predict_temporal_sequence(
            test_input[:1], time_points, num_samples=5
        )
        assert temporal_pred.shape == (1, 3, 1, 16, 16)
        
        # Test quantum prediction
        quantum_pred, quantum_unc = quantum_pno(test_input)
        assert quantum_pred.shape == (2, 1, 16, 16)
        
        # Test cached inference
        cached_pred, cached_unc = cached_inference.predict_with_uncertainty(test_input)
        assert cached_pred.shape == (2, 1, 16, 16)
        
        print("✅ Integration Scenario test passed")
        return True
        
    except Exception as e:
        print(f"❌ Integration Scenario test failed: {e}")
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all basic functionality tests."""
    print("🧪 Running Autonomous SDLC Basic Functionality Tests...")
    print("=" * 60)
    
    tests = [
        test_temporal_uncertainty_kernel,
        test_causal_uncertainty_inference,
        test_quantum_uncertainty_principles,
        test_advanced_validation,
        test_security_framework,
        test_distributed_optimization,
        test_intelligent_caching,
        test_research_theory_validation,
        test_integration_scenario
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            failed += 1
        print("-" * 40)
    
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed! Autonomous SDLC implementation is working correctly.")
        return True
    else:
        print(f"⚠️  {failed} test(s) failed. Please review the implementation.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)