#!/usr/bin/env python3
"""Test scaling functionality for Generation 3."""

import torch
import numpy as np
import sys
import os
import time

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_model_optimization():
    """Test model optimization features."""
    print("\nTesting model optimization...")
    
    try:
        from pno_physics_bench.optimization import ModelOptimizer, MemoryOptimizer
        from pno_physics_bench.models import ProbabilisticNeuralOperator
        
        # Create model
        model = ProbabilisticNeuralOperator(
            input_dim=3,
            hidden_dim=8,
            num_layers=2,
            modes=4
        )
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        # Test model optimizer
        optimizer = ModelOptimizer(model, device)
        print("✓ Model optimizer created")
        
        # Test inference optimization
        optimized_model = optimizer.optimize_for_inference(
            use_half_precision=False,  # Keep FP32 for CPU
            use_channels_last=False,
            freeze_model=True
        )
        print("✓ Model optimized for inference")
        
        # Test memory optimization
        memory_optimized_model = MemoryOptimizer.optimize_memory_usage(
            model,
            use_gradient_checkpointing=False,  # Not available for custom models
            use_activation_checkpointing=False
        )
        print("✓ Memory optimization applied")
        
        # Test memory context
        with MemoryOptimizer.memory_context():
            test_input = torch.randn(2, 3, 16, 16).to(device)
            output = optimized_model(test_input, sample=False)
            print(f"✓ Memory context: output shape {output.shape}")
        
        # Test memory stats
        memory_stats = MemoryOptimizer.get_memory_stats()
        print(f"✓ Memory stats: {len(memory_stats)} metrics")
        
        return True
        
    except ImportError as e:
        print(f"✗ Optimization import failed: {e}")
        return False
    except Exception as e:
        print(f"✗ Optimization test failed: {e}")
        return False

def test_caching_system():
    """Test intelligent caching."""
    print("\nTesting caching system...")
    
    try:
        from pno_physics_bench.optimization import CacheManager
        
        # Create cache manager
        cache = CacheManager(max_memory_mb=100)  # 100MB cache
        print("✓ Cache manager created")
        
        # Test caching tensors
        test_tensor = torch.randn(10, 10)
        cache.put("test_tensor", test_tensor)
        print("✓ Tensor cached")
        
        # Test retrieval
        cached_tensor = cache.get("test_tensor")
        if cached_tensor is not None and torch.equal(cached_tensor, test_tensor):
            print("✓ Tensor retrieved correctly")
        else:
            print("✗ Tensor retrieval failed")
            return False
        
        # Test cache miss
        missing = cache.get("missing_key")
        if missing is None:
            print("✓ Cache miss handled correctly")
        
        # Test cache stats
        stats = cache.stats()
        print(f"✓ Cache stats: {stats['items']} items, {stats['memory_used_mb']:.2f} MB used")
        
        # Test eviction with large tensor
        large_tensor = torch.randn(1000, 1000)  # ~4MB tensor
        cache.put("large_tensor", large_tensor)
        
        # Test cache clearing
        cache.clear()
        if cache.stats()['items'] == 0:
            print("✓ Cache cleared successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ Caching test failed: {e}")
        return False

def test_batch_processing():
    """Test efficient batch processing."""
    print("\nTesting batch processing...")
    
    try:
        from pno_physics_bench.optimization import BatchProcessor
        from pno_physics_bench.models import ProbabilisticNeuralOperator
        
        # Create model
        model = ProbabilisticNeuralOperator(
            input_dim=3,
            hidden_dim=4,
            num_layers=1,
            modes=2
        )
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        # Create batch processor
        batch_processor = BatchProcessor(
            model=model,
            device=device,
            max_batch_size=8,
            target_memory_gb=2.0
        )
        print("✓ Batch processor created")
        
        # Test optimal batch size finding
        sample_input = torch.randn(1, 3, 8, 8)
        optimal_size = batch_processor.find_optimal_batch_size(sample_input, start_size=1)
        print(f"✓ Optimal batch size found: {optimal_size}")
        
        # Test batch processing
        test_inputs = [torch.randn(3, 8, 8) for _ in range(10)]
        outputs = batch_processor.process_batches(test_inputs, batch_size=4)
        
        if len(outputs) == len(test_inputs):
            print(f"✓ Batch processing successful: {len(outputs)} outputs")
        else:
            print(f"✗ Batch processing failed: {len(outputs)}/{len(test_inputs)} outputs")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ Batch processing test failed: {e}")
        return False

def test_dataloader_optimization():
    """Test DataLoader optimization."""
    print("\nTesting DataLoader optimization...")
    
    try:
        from pno_physics_bench.optimization import optimize_dataloader
        from pno_physics_bench.datasets import PDEDataset
        
        # Create dataset
        dataset = PDEDataset.load("navier_stokes_2d", resolution=8, num_samples=8)
        train_loader, _, _ = PDEDataset.get_loaders(dataset, batch_size=2)
        
        # Test DataLoader optimization  
        from torch.utils.data import DataLoader
        # Create a simpler loader without batch_sampler
        simple_loader = DataLoader(dataset, batch_size=2, shuffle=False)
        
        optimized_loader = optimize_dataloader(
            simple_loader,
            num_workers=2,
            pin_memory=False,  # Disable for testing
            persistent_workers=False
        )
        print("✓ DataLoader optimized")
        
        # Test that optimized loader works
        for inputs, targets in optimized_loader:
            print(f"✓ Optimized loader: {inputs.shape}, {targets.shape}")
            break
        
        return True
        
    except Exception as e:
        print(f"✗ DataLoader optimization test failed: {e}")
        return False

def test_distributed_components():
    """Test distributed training components."""
    print("\nTesting distributed components...")
    
    try:
        from pno_physics_bench.distributed_training import (
            DistributedTrainingManager, AutoScalingTrainer, find_free_port, 
            setup_distributed_environment
        )
        from pno_physics_bench.models import ProbabilisticNeuralOperator
        
        # Test port finding
        port = find_free_port()
        print(f"✓ Free port found: {port}")
        
        # Test environment setup
        setup_distributed_environment(rank=0, world_size=1, master_port=port)
        print("✓ Distributed environment setup")
        
        # Test distributed manager (single process)
        dist_manager = DistributedTrainingManager()
        print("✓ Distributed training manager created")
        
        # Test auto-scaling trainer
        class MockTrainer:
            pass
        
        auto_trainer = AutoScalingTrainer(
            MockTrainer,
            min_workers=1,
            max_workers=2,
            scaling_strategy="cpu_count"
        )
        print(f"✓ Auto-scaling trainer: {auto_trainer.optimal_workers} optimal workers")
        
        return True
        
    except ImportError as e:
        print(f"✗ Distributed components import failed: {e}")
        return False
    except Exception as e:
        print(f"✗ Distributed components test failed: {e}")
        return False

def test_performance_profiling():
    """Test performance profiling."""
    print("\nTesting performance profiling...")
    
    try:
        from pno_physics_bench.distributed_training import PerformanceProfiler
        from pno_physics_bench.logging_config import PerformanceLogger
        
        # Test distributed profiler
        profiler = PerformanceProfiler(enabled=True)
        
        # Test forward profiling
        with profiler.profile_forward():
            time.sleep(0.01)  # Simulate work
        
        # Test backward profiling
        with profiler.profile_backward():
            time.sleep(0.005)  # Simulate work
        
        summary = profiler.get_summary()
        print(f"✓ Performance profiler: {len(summary)} metrics")
        
        # Test performance logger
        perf_logger = PerformanceLogger()
        
        with perf_logger.timer("test_operation"):
            time.sleep(0.01)
        
        print("✓ Performance logger working")
        
        return True
        
    except Exception as e:
        print(f"✗ Performance profiling test failed: {e}")
        return False

def test_memory_efficiency():
    """Test memory efficiency improvements."""
    print("\nTesting memory efficiency...")
    
    try:
        from pno_physics_bench.optimization import MemoryOptimizer
        from pno_physics_bench.models import ProbabilisticNeuralOperator
        
        # Get initial memory stats
        initial_stats = MemoryOptimizer.get_memory_stats()
        print(f"✓ Initial memory stats: {len(initial_stats)} metrics")
        
        # Test memory-efficient operations
        with MemoryOptimizer.memory_context():
            # Create model and data
            model = ProbabilisticNeuralOperator(
                input_dim=3, hidden_dim=16, num_layers=2, modes=4
            )
            
            # Large batch to test memory management
            large_batch = torch.randn(8, 3, 32, 32)
            
            # Forward pass
            with torch.no_grad():
                output = model(large_batch, sample=False)
            
            print(f"✓ Memory-efficient forward pass: {output.shape}")
        
        # Get final memory stats
        final_stats = MemoryOptimizer.get_memory_stats()
        print(f"✓ Final memory stats: {len(final_stats)} metrics")
        
        return True
        
    except Exception as e:
        print(f"✗ Memory efficiency test failed: {e}")
        return False

def test_hierarchical_training():
    """Test hierarchical multi-scale training."""
    print("\nTesting hierarchical training...")
    
    try:
        from pno_physics_bench.distributed_training import HierarchicalTraining
        from pno_physics_bench.models import ProbabilisticNeuralOperator
        
        # Create model
        model = ProbabilisticNeuralOperator(
            input_dim=3, hidden_dim=4, num_layers=1, modes=2
        )
        
        # Create hierarchical trainer
        hierarchical = HierarchicalTraining(
            model=model,
            scales=[16, 32],  # Smaller scales for testing
            scale_weights=[0.6, 0.4]
        )
        print("✓ Hierarchical trainer created")
        
        # Test multi-scale forward pass
        test_input = torch.randn(2, 3, 32, 32)
        multi_scale_outputs = hierarchical.multi_scale_forward(test_input)
        
        if len(multi_scale_outputs) == 2:
            print(f"✓ Multi-scale forward: {len(multi_scale_outputs)} scales")
            for scale, output in multi_scale_outputs.items():
                print(f"  Scale {scale}: {output.shape}")
        else:
            print(f"✗ Multi-scale forward failed: {len(multi_scale_outputs)} scales")
            return False
        
        # Test multi-scale loss
        def simple_loss(pred, target):
            return torch.nn.functional.mse_loss(pred, target)
        
        targets = torch.randn(2, 1, 32, 32)
        loss = hierarchical.compute_multi_scale_loss(multi_scale_outputs, targets, simple_loss)
        print(f"✓ Multi-scale loss: {loss.item():.4f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Hierarchical training test failed: {e}")
        return False

def main():
    """Run all scaling functionality tests."""
    print("=" * 60)
    print("PNO PHYSICS BENCH - GENERATION 3 SCALING FUNCTIONALITY TEST")
    print("=" * 60)
    
    tests = [
        test_model_optimization,
        test_caching_system,
        test_batch_processing,
        test_dataloader_optimization,
        test_distributed_components,
        test_performance_profiling,
        test_memory_efficiency,
        test_hierarchical_training,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} crashed: {e}")
    
    print("\n" + "=" * 60)
    print(f"RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 GENERATION 3 COMPLETE - SYSTEM SCALES!")
        print("All performance optimization and scaling features working correctly.")
    else:
        print("⚠️  Some scaling tests failed - system needs optimization.")
    
    print("=" * 60)
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)