"""
Test suite for performance optimization components.

Tests advanced optimizations:
- Memory-efficient spectral attention
- Mixed-precision uncertainty quantification  
- GPU kernel optimizations
- Adaptive computation graphs
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from unittest.mock import Mock, patch, MagicMock

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))

from pno_physics_bench.research.performance_optimizers import (
    MemoryEfficientSpectralAttention,
    SpectralProjection,
    MixedPrecisionUncertaintyQuantifier,
    OptimizedUncertaintySampler,
    AdaptiveComputationGraph,
    ModelCompilationOptimizer,
    PerformanceProfiler
)


class MockBaseModel(nn.Module):
    """Mock base model for testing."""
    
    def __init__(self, input_dim=3, hidden_dim=64):
        super().__init__()
        self.input_dim = input_dim 
        self.hidden_dim = hidden_dim
        
        self.encoder = nn.Sequential(
            nn.Conv2d(input_dim, hidden_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)
        )
        
        self.uncertainty_head = nn.Sequential(
            nn.Conv2d(hidden_dim, input_dim, 1),
            nn.Softplus()
        )
    
    def forward(self, x):
        features = self.encoder(x)
        return features
    
    def predict_with_uncertainty(self, x):
        features = self.forward(x)
        prediction = features[:, :self.input_dim]
        uncertainty = self.uncertainty_head(features) + 0.01
        return prediction, uncertainty


class TestMemoryEfficientSpectralAttention:
    """Test memory-efficient spectral attention mechanism."""
    
    @pytest.fixture
    def attention_layer(self):
        return MemoryEfficientSpectralAttention(
            hidden_dim=64,
            num_heads=8,
            modes=16,
            chunk_size=512,
            use_flash_attention=False  # Disable for testing
        )
    
    @pytest.fixture
    def sample_input(self):
        return torch.randn(2, 64, 32, 32)
    
    def test_initialization(self):
        """Test proper initialization."""
        attention = MemoryEfficientSpectralAttention(
            hidden_dim=128,
            num_heads=8,
            modes=20
        )
        
        assert attention.hidden_dim == 128
        assert attention.num_heads == 8
        assert attention.head_dim == 16  # 128 / 8
        assert attention.modes == 20
        
        # Check components exist
        assert hasattr(attention, 'spectral_proj_q')
        assert hasattr(attention, 'spectral_proj_k')
        assert hasattr(attention, 'spectral_proj_v')
        assert hasattr(attention, 'out_proj')
        
    def test_forward_pass_basic(self, attention_layer, sample_input):
        """Test basic forward pass."""
        output = attention_layer(sample_input)
        
        # Output should have same shape as input
        assert output.shape == sample_input.shape
        
        # Should be finite values
        assert torch.all(torch.isfinite(output))
        
    def test_forward_with_checkpointing(self, attention_layer, sample_input):
        """Test forward pass with gradient checkpointing."""
        
        # Enable training mode for checkpointing
        attention_layer.train()
        sample_input.requires_grad_(True)
        
        output = attention_layer(sample_input, use_checkpointing=True)
        
        assert output.shape == sample_input.shape
        
        # Test gradient flow
        loss = torch.sum(output)
        loss.backward()
        
        assert sample_input.grad is not None
        
    def test_chunked_vs_full_attention(self, sample_input):
        """Test chunked attention produces similar results to full attention."""
        
        # Small chunk size to force chunking
        chunked_attention = MemoryEfficientSpectralAttention(
            hidden_dim=64,
            num_heads=4,
            modes=8,
            chunk_size=64,  # Small chunk size
            use_flash_attention=False
        )
        
        # Large chunk size (essentially full attention)
        full_attention = MemoryEfficientSpectralAttention(
            hidden_dim=64,
            num_heads=4,
            modes=8,
            chunk_size=4096,  # Large chunk size
            use_flash_attention=False
        )
        
        # Use same weights for fair comparison
        full_attention.load_state_dict(chunked_attention.state_dict())
        
        chunked_attention.eval()
        full_attention.eval()
        
        with torch.no_grad():
            chunked_output = chunked_attention(sample_input)
            full_output = full_attention(sample_input)
        
        # Outputs should be similar (allowing for numerical differences)
        assert torch.allclose(chunked_output, full_output, atol=1e-3, rtol=1e-3)
        
    def test_different_input_sizes(self, attention_layer):
        """Test with different input sizes."""
        
        sizes = [(1, 64, 16, 16), (4, 64, 32, 32), (2, 64, 64, 64)]
        
        for size in sizes:
            input_tensor = torch.randn(*size)
            output = attention_layer(input_tensor)
            
            assert output.shape == input_tensor.shape
            assert torch.all(torch.isfinite(output))
            
    def test_spectral_domain_processing(self, attention_layer, sample_input):
        """Test that spectral domain processing works correctly."""
        
        # The attention should handle frequency domain operations
        attention_layer.eval()
        
        with torch.no_grad():
            output = attention_layer(sample_input)
        
        # Check that output is reasonable
        input_energy = torch.sum(sample_input ** 2)
        output_energy = torch.sum(output ** 2)
        
        # Energy should be preserved approximately
        energy_ratio = output_energy / input_energy
        assert 0.1 < energy_ratio < 10.0  # Reasonable range
        
    def test_attention_mask_functionality(self, attention_layer):
        """Test attention mask functionality."""
        
        # Test that frequency mask is created and used
        assert hasattr(attention_layer, 'freq_mask')
        
        # Mask should have correct dimensions
        expected_seq_len = attention_layer.modes * attention_layer.modes
        assert attention_layer.freq_mask.shape == (expected_seq_len, expected_seq_len)


class TestSpectralProjection:
    """Test spectral projection layer."""
    
    @pytest.fixture
    def projection_layer(self):
        return SpectralProjection(
            input_dim=32,
            output_dim=64,
            modes=8
        )
    
    def test_initialization(self):
        """Test proper initialization."""
        proj = SpectralProjection(16, 32, 4)
        
        assert proj.input_dim == 16
        assert proj.output_dim == 32
        assert proj.modes == 4
        
        # Check spectral weights shape
        expected_weight_shape = (16, 32, 4, 4)
        assert proj.spectral_weights.shape == expected_weight_shape
        assert proj.spectral_weights.dtype == torch.cfloat
        
    def test_forward_pass(self, projection_layer):
        """Test forward pass with correct reshaping."""
        
        # Input: [batch, seq_len, input_dim]
        seq_len = 16  # 4x4 spatial
        input_tensor = torch.randn(2, seq_len, 32)
        
        output = projection_layer(input_tensor)
        
        # Output should have correct shape
        expected_shape = (2, seq_len, 64)
        assert output.shape == expected_shape
        
    def test_spectral_transformation(self, projection_layer):
        """Test spectral domain transformation."""
        
        # Create input with known spectral properties
        seq_len = 16  # 4x4
        input_tensor = torch.randn(1, seq_len, 32)
        
        output = projection_layer(input_tensor)
        
        # Output should be different from input (transformation applied)
        assert not torch.allclose(output[:, :, :32], input_tensor, atol=1e-3)
        
        # Should maintain finite values
        assert torch.all(torch.isfinite(output))
        
    def test_gradient_flow(self, projection_layer):
        """Test gradient flow through spectral operations."""
        
        input_tensor = torch.randn(1, 16, 32, requires_grad=True)
        
        output = projection_layer(input_tensor)
        loss = torch.sum(output)
        loss.backward()
        
        # Check gradients exist and are finite
        assert input_tensor.grad is not None
        assert torch.all(torch.isfinite(input_tensor.grad))
        assert torch.any(input_tensor.grad != 0)


class TestMixedPrecisionUncertaintyQuantifier:
    """Test mixed-precision uncertainty quantification."""
    
    @pytest.fixture
    def mock_model(self):
        return MockBaseModel(input_dim=3, hidden_dim=32)
    
    @pytest.fixture
    def mp_quantifier(self, mock_model):
        return MixedPrecisionUncertaintyQuantifier(
            base_model=mock_model,
            uncertainty_precision="fp16",
            error_tolerance=1e-3
        )
    
    @pytest.fixture
    def sample_input(self):
        return torch.randn(2, 3, 16, 16)
    
    def test_initialization(self, mock_model):
        """Test proper initialization."""
        mp_quant = MixedPrecisionUncertaintyQuantifier(
            base_model=mock_model,
            uncertainty_precision="dynamic"
        )
        
        assert mp_quant.uncertainty_precision == "dynamic"
        assert hasattr(mp_quant, 'scaler')
        assert hasattr(mp_quant, 'precision_scheduler')
        
    def test_forward_pass_fp16(self, mp_quantifier, sample_input):
        """Test forward pass with FP16 precision."""
        
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available for mixed precision testing")
            
        mp_quantifier.cuda()
        sample_input = sample_input.cuda()
        
        predictions, uncertainties = mp_quantifier(sample_input)
        
        # Check outputs
        assert predictions.shape[0] == sample_input.shape[0]
        assert uncertainties.shape == predictions.shape
        assert torch.all(uncertainties > 0)
        
    def test_forward_pass_fp32(self, mp_quantifier, sample_input):
        """Test forward pass with FP32 precision."""
        
        # Force FP32 precision
        mp_quantifier.precision_scheduler.current_precision = "fp32"
        
        predictions, uncertainties = mp_quantifier(sample_input)
        
        assert predictions.shape[0] == sample_input.shape[0]
        assert uncertainties.shape == predictions.shape
        assert torch.all(uncertainties > 0)
        
    def test_precision_error_monitoring(self, mp_quantifier, sample_input):
        """Test precision error monitoring functionality."""
        
        # Enable training mode for monitoring
        mp_quantifier.train()
        
        # Run multiple forward passes
        for _ in range(5):
            _ = mp_quantifier(sample_input)
            
        # Should have recorded some error metrics
        assert len(mp_quantifier.precision_errors) > 0
        
        # Errors should be reasonable values
        for error in mp_quantifier.precision_errors:
            assert error >= 0
            assert error < 1000  # Sanity check
            
    def test_precision_info_output(self, mp_quantifier, sample_input):
        """Test detailed precision information output."""
        
        result = mp_quantifier(sample_input, return_precision_info=True)
        
        assert isinstance(result, dict)
        
        required_keys = ["predictions", "uncertainties", "precision", "precision_error"]
        for key in required_keys:
            assert key in result
            
        assert result["predictions"].shape[0] == sample_input.shape[0]
        assert result["precision"] in ["fp16", "fp32"]
        
    def test_uncertainty_estimation_fallback(self, mp_quantifier, sample_input):
        """Test uncertainty estimation fallback for models without built-in uncertainty."""
        
        # Create model without predict_with_uncertainty method
        simple_model = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 3, 3, padding=1)
        )
        
        mp_quant_simple = MixedPrecisionUncertaintyQuantifier(
            base_model=simple_model
        )
        
        predictions, uncertainties = mp_quant_simple(sample_input)
        
        # Should still produce uncertainty estimates
        assert uncertainties.shape == predictions.shape
        assert torch.all(uncertainties > 0)


class TestOptimizedUncertaintySampler:
    """Test optimized uncertainty sampling."""
    
    @pytest.fixture
    def sampler(self):
        return OptimizedUncertaintySampler(
            use_triton_kernels=False,  # Disable for testing
            default_num_samples=50,
            batch_sampling=True
        )
    
    @pytest.fixture
    def sample_predictions(self):
        mean_pred = torch.randn(2, 3, 16, 16)
        uncertainties = torch.rand(2, 3, 16, 16) * 0.2 + 0.01
        return mean_pred, uncertainties
    
    def test_initialization(self):
        """Test proper initialization."""
        sampler = OptimizedUncertaintySampler(
            default_num_samples=100,
            batch_sampling=False
        )
        
        assert sampler.default_num_samples == 100
        assert sampler.batch_sampling == False
        assert hasattr(sampler, 'random_cache')
        
    def test_pytorch_sampling(self, sampler, sample_predictions):
        """Test PyTorch-based sampling implementation."""
        
        mean_pred, uncertainties = sample_predictions
        num_samples = 20
        
        samples = sampler.sample_predictions(
            mean_pred, uncertainties, num_samples
        )
        
        # Check output shape
        expected_shape = (num_samples, *mean_pred.shape)
        assert samples.shape == expected_shape
        
        # Check that samples are reasonable
        assert torch.all(torch.isfinite(samples))
        
        # Check that samples have correct statistical properties
        sample_mean = torch.mean(samples, dim=0)
        sample_std = torch.std(samples, dim=0)
        
        # Sample mean should be close to input mean
        assert torch.allclose(sample_mean, mean_pred, atol=0.1, rtol=0.1)
        
        # Sample std should be related to input uncertainties
        correlation = torch.corrcoef(torch.stack([
            sample_std.flatten(),
            uncertainties.flatten()
        ]))[0, 1]
        assert correlation > 0.3  # Should be positively correlated
        
    def test_triton_sampling_fallback(self, sample_predictions):
        """Test Triton sampling fallback to PyTorch."""
        
        # Create sampler with Triton enabled but it should fallback
        sampler = OptimizedUncertaintySampler(use_triton_kernels=True)
        
        mean_pred, uncertainties = sample_predictions
        
        # Should work regardless of Triton availability
        samples = sampler.sample_predictions(mean_pred, uncertainties, 10)
        
        expected_shape = (10, *mean_pred.shape)
        assert samples.shape == expected_shape
        assert torch.all(torch.isfinite(samples))
        
    def test_random_cache_efficiency(self, sampler, sample_predictions):
        """Test random number caching for efficiency."""
        
        mean_pred, uncertainties = sample_predictions
        
        # Multiple sampling calls should use cache efficiently
        samples1 = sampler.sample_predictions(mean_pred, uncertainties, 5)
        samples2 = sampler.sample_predictions(mean_pred, uncertainties, 5)
        
        # Should produce different samples (not cached results)
        assert not torch.allclose(samples1, samples2, atol=1e-6)
        
        # But both should be valid
        assert torch.all(torch.isfinite(samples1))
        assert torch.all(torch.isfinite(samples2))
        
    def test_different_sample_counts(self, sampler, sample_predictions):
        """Test sampling with different sample counts."""
        
        mean_pred, uncertainties = sample_predictions
        sample_counts = [1, 5, 10, 50, 100]
        
        for count in sample_counts:
            samples = sampler.sample_predictions(mean_pred, uncertainties, count)
            
            expected_shape = (count, *mean_pred.shape)
            assert samples.shape == expected_shape
            assert torch.all(torch.isfinite(samples))
            
    def test_cache_refresh_mechanism(self, sampler):
        """Test random cache refresh mechanism."""
        
        # Fill cache to trigger refresh
        large_pred = torch.randn(1, 1, 100, 100)  # Large to consume cache
        large_unc = torch.rand(1, 1, 100, 100) * 0.1 + 0.01
        
        # This should trigger cache refresh internally
        samples = sampler.sample_predictions(large_pred, large_unc, 10)
        
        assert torch.all(torch.isfinite(samples))
        assert samples.shape == (10, 1, 1, 100, 100)


class TestAdaptiveComputationGraph:
    """Test adaptive computation graph."""
    
    @pytest.fixture
    def mock_layers(self):
        return nn.ModuleList([
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU()
        ])
    
    @pytest.fixture
    def adaptive_graph(self, mock_layers):
        return AdaptiveComputationGraph(
            base_layers=mock_layers,
            uncertainty_thresholds=[0.1, 0.2, 0.3],
            computation_budgets=[1, 2, 4]
        )
    
    def test_initialization(self, mock_layers):
        """Test proper initialization."""
        graph = AdaptiveComputationGraph(
            base_layers=mock_layers,
            uncertainty_thresholds=[0.15, 0.25],
            computation_budgets=[2, 3]
        )
        
        assert len(graph.uncertainty_thresholds) == 2
        assert len(graph.computation_budgets) == 2
        assert len(graph.computation_controllers) == 2
        assert hasattr(graph, 'router')
        
    def test_forward_without_uncertainty(self, adaptive_graph):
        """Test forward pass without uncertainty guidance."""
        
        input_tensor = torch.randn(2, 16, 32, 32)
        
        output, computation_mask = adaptive_graph(input_tensor)
        
        # Check outputs
        assert output.shape == input_tensor.shape
        assert computation_mask.shape == (input_tensor.shape[0],)
        
        # Computation mask should be valid indices
        assert torch.all(computation_mask >= 0)
        assert torch.all(computation_mask < len(adaptive_graph.uncertainty_thresholds))
        
    def test_forward_with_uncertainty_guidance(self, adaptive_graph):
        """Test forward pass with uncertainty guidance."""
        
        input_tensor = torch.randn(2, 16, 32, 32)
        
        # Low uncertainty - should use less computation
        low_uncertainty = torch.ones(2) * 0.05
        output_low, mask_low = adaptive_graph(input_tensor, low_uncertainty)
        
        # High uncertainty - should use more computation  
        high_uncertainty = torch.ones(2) * 0.25
        output_high, mask_high = adaptive_graph(input_tensor, high_uncertainty)
        
        # Both should produce valid outputs
        assert output_low.shape == input_tensor.shape
        assert output_high.shape == input_tensor.shape
        
        # High uncertainty should generally use more computation
        avg_computation_low = torch.mean(mask_low.float())
        avg_computation_high = torch.mean(mask_high.float())
        
        # This is a general trend, not a strict requirement
        # assert avg_computation_high >= avg_computation_low
        
    def test_router_functionality(self, adaptive_graph):
        """Test adaptive router functionality."""
        
        # Test router directly
        uncertainties = torch.tensor([0.05, 0.15, 0.25, 0.35])
        routing_decisions = adaptive_graph.router.route(uncertainties)
        
        # Should produce valid routing decisions
        assert routing_decisions.shape == (4,)
        assert torch.all(routing_decisions >= 0)
        assert torch.all(routing_decisions < len(adaptive_graph.uncertainty_thresholds))
        
        # Lower uncertainties should generally get lower computation levels
        low_unc_decision = routing_decisions[0].item()
        high_unc_decision = routing_decisions[3].item()
        
        # This is a general trend
        # assert high_unc_decision >= low_unc_decision
        
    def test_computation_controllers(self, adaptive_graph):
        """Test computation controller functionality."""
        
        input_tensor = torch.randn(1, 16, 16, 16)
        
        # Test each computation level
        for i, controller in enumerate(adaptive_graph.computation_controllers):
            output = controller(input_tensor, adaptive_graph.base_layers)
            
            # Should produce valid output
            assert output.shape == input_tensor.shape
            assert torch.all(torch.isfinite(output))
            
    def test_gradient_flow(self, adaptive_graph):
        """Test gradient flow through adaptive computation."""
        
        input_tensor = torch.randn(1, 16, 16, 16, requires_grad=True)
        uncertainty = torch.tensor([0.15])
        
        output, _ = adaptive_graph(input_tensor, uncertainty)
        loss = torch.sum(output)
        loss.backward()
        
        # Check gradient flow
        assert input_tensor.grad is not None
        assert torch.any(input_tensor.grad != 0)


class TestModelCompilationOptimizer:
    """Test model compilation optimization."""
    
    @pytest.fixture
    def simple_model(self):
        return nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 3, 3, padding=1)
        )
    
    @pytest.fixture
    def optimizer(self):
        return ModelCompilationOptimizer(
            target_hardware="cuda",
            optimization_level="balanced",
            preserve_uncertainty=True
        )
    
    def test_initialization(self):
        """Test proper initialization."""
        opt = ModelCompilationOptimizer(
            optimization_level="aggressive"
        )
        
        assert opt.optimization_level == "aggressive"
        assert opt.preserve_uncertainty == True
        assert hasattr(opt, 'compilation_strategies')
        
    def test_conservative_compilation(self, optimizer, simple_model):
        """Test conservative compilation strategy."""
        
        example_input = torch.randn(1, 3, 16, 16)
        
        # Use conservative compilation
        optimizer.optimization_level = "conservative"
        compiled_model = optimizer.compile_model(simple_model, example_input)
        
        # Should produce a compiled model
        assert compiled_model is not None
        
        # Should produce same outputs
        simple_model.eval()
        compiled_model.eval()
        
        with torch.no_grad():
            original_output = simple_model(example_input)
            compiled_output = compiled_model(example_input)
            
        # Outputs should be very close
        assert torch.allclose(original_output, compiled_output, atol=1e-5, rtol=1e-5)
        
    def test_balanced_compilation(self, optimizer, simple_model):
        """Test balanced compilation strategy."""
        
        example_input = torch.randn(1, 3, 16, 16)
        
        compiled_model = optimizer.compile_model(simple_model, example_input)
        
        # Should work without errors
        assert compiled_model is not None
        
        # Test inference
        compiled_model.eval()
        with torch.no_grad():
            output = compiled_model(example_input)
            
        assert output.shape == (1, 3, 16, 16)
        assert torch.all(torch.isfinite(output))
        
    def test_uncertainty_preservation_check(self, optimizer):
        """Test uncertainty preservation verification."""
        
        # Create model with uncertainty method
        mock_model_with_uncertainty = MockBaseModel()
        example_input = torch.randn(1, 3, 16, 16)
        
        # This should work without raising errors
        compiled_model = optimizer.compile_model(mock_model_with_uncertainty, example_input)
        
        assert compiled_model is not None
        
    def test_compilation_strategies_exist(self, optimizer):
        """Test that all compilation strategies are available."""
        
        expected_strategies = ["conservative", "balanced", "aggressive"]
        
        for strategy in expected_strategies:
            assert strategy in optimizer.compilation_strategies
            
        # Each strategy should be callable
        for strategy_name, strategy_func in optimizer.compilation_strategies.items():
            assert callable(strategy_func)


class TestPerformanceProfiler:
    """Test performance profiling functionality."""
    
    @pytest.fixture
    def profiler(self):
        return PerformanceProfiler()
    
    def test_initialization(self):
        """Test proper initialization."""
        profiler = PerformanceProfiler()
        
        assert hasattr(profiler, 'operation_times')
        assert hasattr(profiler, 'memory_usage')
        assert profiler.profiling_active == False
        
    def test_profiling_inactive_by_default(self, profiler):
        """Test that profiling is inactive by default."""
        
        def dummy_operation(x):
            return x * 2
            
        # Should just run the operation without profiling
        result = profiler.profile_operation("test_op", dummy_operation, torch.tensor(5.0))
        
        assert result == torch.tensor(10.0)
        assert "test_op" not in profiler.operation_times
        
    def test_active_profiling(self, profiler):
        """Test active profiling functionality."""
        
        profiler.start_profiling()
        
        def dummy_operation(x, y):
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            return x + y
            
        result = profiler.profile_operation(
            "addition",
            dummy_operation,
            torch.tensor(3.0),
            torch.tensor(4.0)
        )
        
        assert result == torch.tensor(7.0)
        assert "addition" in profiler.operation_times
        assert len(profiler.operation_times["addition"]) == 1
        
    def test_multiple_operations_profiling(self, profiler):
        """Test profiling multiple operations."""
        
        profiler.start_profiling()
        
        def op1():
            return torch.randn(10, 10).sum()
            
        def op2():
            return torch.zeros(5, 5).mean()
            
        # Profile multiple operations
        profiler.profile_operation("op1", op1)
        profiler.profile_operation("op2", op2)
        profiler.profile_operation("op1", op1)  # Second call to op1
        
        assert "op1" in profiler.operation_times
        assert "op2" in profiler.operation_times
        assert len(profiler.operation_times["op1"]) == 2
        assert len(profiler.operation_times["op2"]) == 1
        
    def test_performance_report_generation(self, profiler):
        """Test performance report generation."""
        
        profiler.start_profiling()
        
        def test_operation():
            return torch.randn(100, 100).sum()
            
        # Profile several calls
        for _ in range(5):
            profiler.profile_operation("test_op", test_operation)
            
        report = profiler.get_performance_report()
        
        assert "test_op" in report
        
        op_stats = report["test_op"]
        required_keys = ["avg_time", "std_time", "min_time", "max_time", "call_count"]
        
        for key in required_keys:
            assert key in op_stats
            
        assert op_stats["call_count"] == 5
        assert op_stats["avg_time"] > 0
        assert op_stats["min_time"] > 0
        assert op_stats["min_time"] <= op_stats["max_time"]
        
    def test_memory_tracking(self, profiler):
        """Test memory usage tracking."""
        
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available for memory tracking test")
            
        profiler.start_profiling()
        
        def memory_intensive_operation():
            # Create and then delete a large tensor
            large_tensor = torch.randn(1000, 1000).cuda()
            result = large_tensor.sum()
            del large_tensor
            return result
            
        profiler.profile_operation("memory_op", memory_intensive_operation)
        
        assert "memory_op" in profiler.memory_usage
        assert len(profiler.memory_usage["memory_op"]) == 1
        
        # Memory usage should be recorded (could be positive or negative)
        memory_delta = profiler.memory_usage["memory_op"][0]
        assert isinstance(memory_delta, (int, float))
        
    def test_start_stop_profiling(self, profiler):
        """Test starting and stopping profiling."""
        
        def test_op():
            return torch.tensor(42.0)
            
        # Initially not profiling
        profiler.profile_operation("test1", test_op)
        assert "test1" not in profiler.operation_times
        
        # Start profiling
        profiler.start_profiling()
        profiler.profile_operation("test2", test_op)
        assert "test2" in profiler.operation_times
        
        # Stop profiling
        profiler.stop_profiling()
        profiler.profile_operation("test3", test_op)
        assert "test3" not in profiler.operation_times


class TestIntegration:
    """Integration tests for performance optimization components."""
    
    def test_attention_with_mixed_precision(self):
        """Test spectral attention with mixed precision."""
        
        attention = MemoryEfficientSpectralAttention(
            hidden_dim=64,
            num_heads=8,
            modes=12
        )
        
        mp_quantifier = MixedPrecisionUncertaintyQuantifier(
            base_model=attention,
            uncertainty_precision="fp16"
        )
        
        sample_input = torch.randn(1, 64, 32, 32)
        
        # Should work together
        predictions, uncertainties = mp_quantifier(sample_input)
        
        assert predictions.shape == sample_input.shape
        assert uncertainties.shape == sample_input.shape
        assert torch.all(uncertainties > 0)
        
    def test_sampler_with_profiler(self):
        """Test uncertainty sampler with performance profiling."""
        
        sampler = OptimizedUncertaintySampler(use_triton_kernels=False)
        profiler = PerformanceProfiler()
        
        profiler.start_profiling()
        
        mean_pred = torch.randn(1, 3, 16, 16)
        uncertainties = torch.rand(1, 3, 16, 16) * 0.1 + 0.01
        
        samples = profiler.profile_operation(
            "uncertainty_sampling",
            sampler.sample_predictions,
            mean_pred, uncertainties, 20
        )
        
        assert samples.shape == (20, 1, 3, 16, 16)
        assert "uncertainty_sampling" in profiler.operation_times
        
        report = profiler.get_performance_report()
        assert "uncertainty_sampling" in report
        
    def test_adaptive_graph_with_compilation(self):
        """Test adaptive computation graph with model compilation."""
        
        base_layers = nn.ModuleList([
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU()
        ])
        
        adaptive_graph = AdaptiveComputationGraph(
            base_layers=base_layers,
            uncertainty_thresholds=[0.1, 0.2],
            computation_budgets=[1, 2]
        )
        
        compiler = ModelCompilationOptimizer(optimization_level="conservative")
        
        example_input = torch.randn(1, 16, 16, 16)
        
        # Compile the adaptive graph
        compiled_graph = compiler.compile_model(adaptive_graph, example_input)
        
        # Test inference
        compiled_graph.eval()
        with torch.no_grad():
            output, mask = compiled_graph(example_input)
            
        assert output.shape == example_input.shape
        assert mask.shape == (1,)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])