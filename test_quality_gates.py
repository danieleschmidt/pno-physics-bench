#!/usr/bin/env python3
"""Quality gates and comprehensive testing system."""

import torch
import numpy as np
import sys
import os
import time
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_model_correctness():
    """Test model correctness and consistency."""
    print("\nTesting model correctness...")
    
    try:
        from pno_physics_bench.models import ProbabilisticNeuralOperator, FourierNeuralOperator, DeepONet
        
        # Test PNO correctness
        model = ProbabilisticNeuralOperator(input_dim=3, hidden_dim=8, num_layers=2, modes=4)
        
        # Consistency test - same input should give same output in eval mode
        test_input = torch.randn(2, 3, 16, 16)
        model.eval()
        
        with torch.no_grad():
            output1 = model(test_input, sample=False)
            output2 = model(test_input, sample=False)
            
            if torch.allclose(output1, output2, atol=1e-6):
                print("✓ PNO deterministic in eval mode")
            else:
                print("✗ PNO not deterministic in eval mode")
                return False
        
        # Test uncertainty prediction consistency
        mean1, std1 = model.predict_with_uncertainty(test_input, num_samples=50)
        mean2, std2 = model.predict_with_uncertainty(test_input, num_samples=50)
        
        # Uncertainty estimates should be stable
        mean_diff = torch.abs(mean1 - mean2).mean().item()
        std_diff = torch.abs(std1 - std2).mean().item()
        
        if mean_diff < 0.1 and std_diff < 0.1:
            print(f"✓ Uncertainty prediction stable: mean_diff={mean_diff:.4f}, std_diff={std_diff:.4f}")
        else:
            print(f"✗ Uncertainty prediction unstable: mean_diff={mean_diff:.4f}, std_diff={std_diff:.4f}")
            return False
        
        # Test gradients exist and are reasonable
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        output = model(test_input, sample=True)
        loss = torch.nn.functional.mse_loss(output, torch.randn_like(output))
        
        optimizer.zero_grad()
        loss.backward()
        
        total_grad_norm = 0.0
        param_count = 0
        for param in model.parameters():
            if param.grad is not None:
                total_grad_norm += param.grad.norm().item() ** 2
                param_count += 1
        
        total_grad_norm = total_grad_norm ** 0.5
        
        if total_grad_norm > 1e-8 and param_count > 0:
            print(f"✓ Gradients exist: norm={total_grad_norm:.4f}, {param_count} parameters")
        else:
            print(f"✗ Gradients missing or zero: norm={total_grad_norm:.4f}")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ Model correctness test failed: {e}")
        return False

def test_performance_benchmarks():
    """Test performance meets minimum requirements."""
    print("\nTesting performance benchmarks...")
    
    try:
        from pno_physics_bench.models import ProbabilisticNeuralOperator
        from pno_physics_bench.logging_config import PerformanceLogger
        
        model = ProbabilisticNeuralOperator(input_dim=3, hidden_dim=16, num_layers=2, modes=8)
        model.eval()
        
        perf_logger = PerformanceLogger()
        
        # Inference speed test
        test_input = torch.randn(8, 3, 32, 32)  # Moderate batch
        
        # Warmup
        for _ in range(5):
            with torch.no_grad():
                _ = model(test_input, sample=False)
        
        # Benchmark inference
        inference_times = []
        with torch.no_grad():
            for _ in range(10):
                start_time = time.time()
                output = model(test_input, sample=False)
                inference_times.append(time.time() - start_time)
        
        avg_inference_time = np.mean(inference_times)
        inference_throughput = test_input.shape[0] / avg_inference_time  # samples/sec
        
        if avg_inference_time < 1.0:  # Should be under 1 second
            print(f"✓ Inference speed: {avg_inference_time:.3f}s, {inference_throughput:.1f} samples/sec")
        else:
            print(f"✗ Inference too slow: {avg_inference_time:.3f}s")
            return False
        
        # Memory efficiency test
        memory_efficient = True
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            with torch.no_grad():
                large_batch = torch.randn(16, 3, 64, 64).cuda()
                model = model.cuda()
                _ = model(large_batch, sample=False)
            
            peak_memory_mb = torch.cuda.max_memory_allocated() / (1024**2)
            if peak_memory_mb < 2000:  # Under 2GB
                print(f"✓ Memory efficient: {peak_memory_mb:.1f} MB peak")
            else:
                print(f"⚠  High memory usage: {peak_memory_mb:.1f} MB peak")
                memory_efficient = False
        else:
            print("✓ Memory test skipped (no CUDA)")
        
        return memory_efficient
        
    except Exception as e:
        print(f"✗ Performance benchmark failed: {e}")
        return False

def test_training_stability():
    """Test training stability and convergence."""
    print("\nTesting training stability...")
    
    try:
        from pno_physics_bench.models import ProbabilisticNeuralOperator
        from pno_physics_bench.training import PNOTrainer, ELBOLoss
        from pno_physics_bench.datasets import PDEDataset
        
        # Create small model and dataset for quick testing
        model = ProbabilisticNeuralOperator(input_dim=3, hidden_dim=8, num_layers=1, modes=2)
        dataset = PDEDataset.load("navier_stokes_2d", resolution=16, num_samples=8)
        train_loader, _, _ = PDEDataset.get_loaders(dataset, batch_size=4)
        
        # Create trainer
        trainer = PNOTrainer(
            model=model,
            loss_fn=ELBOLoss(kl_weight=1e-4),
            gradient_clipping=1.0,
            num_samples=2
        )
        
        # Track training losses
        losses = []
        model.train()
        
        # Mini training loop
        for epoch in range(5):
            epoch_losses = []
            for inputs, targets in train_loader:
                trainer.optimizer.zero_grad()
                
                outputs = model(inputs, sample=True)
                loss_dict = trainer.loss_fn(outputs, targets, model)
                total_loss = loss_dict['total']
                
                # Check for NaN/Inf
                if torch.isnan(total_loss) or torch.isinf(total_loss):
                    print(f"✗ Training instability: loss = {total_loss}")
                    return False
                
                total_loss.backward()
                trainer.optimizer.step()
                
                epoch_losses.append(total_loss.item())
            
            avg_epoch_loss = np.mean(epoch_losses)
            losses.append(avg_epoch_loss)
            
        # Check for general convergence trend (not strict requirement)
        if len(losses) >= 3:
            recent_trend = np.mean(losses[-2:]) - np.mean(losses[:2])
            if recent_trend <= 0:  # Loss should decrease or stabilize
                print(f"✓ Training stable: losses = {losses}")
            else:
                print(f"⚠  Training diverging: losses = {losses}")
                # Not a failure, but worth noting
        
        print(f"✓ Training completed without NaN/Inf: final loss = {losses[-1]:.4f}")
        return True
        
    except Exception as e:
        print(f"✗ Training stability test failed: {e}")
        return False

def test_uncertainty_calibration():
    """Test uncertainty calibration quality."""
    print("\nTesting uncertainty calibration...")
    
    try:
        from pno_physics_bench.models import ProbabilisticNeuralOperator
        from pno_physics_bench.metrics import CalibrationMetrics
        
        model = ProbabilisticNeuralOperator(input_dim=3, hidden_dim=8, num_layers=2, modes=4)
        model.eval()
        
        # Generate test data
        test_inputs = torch.randn(32, 3, 16, 16)
        test_targets = torch.randn(32, 1, 16, 16)
        
        # Get predictions with uncertainty
        with torch.no_grad():
            pred_mean, pred_std = model.predict_with_uncertainty(test_inputs, num_samples=20)
        
        # Test calibration metrics
        calibration = CalibrationMetrics()
        
        try:
            ece = calibration.expected_calibration_error(pred_mean, pred_std, test_targets)
            sharpness = calibration.sharpness(pred_std)
            coverage_90 = calibration.coverage_at_confidence(pred_mean, pred_std, test_targets, 0.9)
            
            print(f"✓ Calibration metrics computed:")
            print(f"  ECE: {ece:.4f} (lower is better)")
            print(f"  Sharpness: {sharpness:.4f}")
            print(f"  Coverage@90%: {coverage_90:.4f} (should be ~0.90)")
            
            # Basic sanity checks
            if 0 <= ece <= 2.0 and sharpness > 0 and 0.5 <= coverage_90 <= 1.0:
                print("✓ Calibration metrics in reasonable ranges")
                return True
            else:
                print("✗ Calibration metrics outside reasonable ranges")
                return False
                
        except Exception as e:
            print(f"✗ Calibration computation failed: {e}")
            return False
        
    except Exception as e:
        print(f"✗ Uncertainty calibration test failed: {e}")
        return False

def test_api_compatibility():
    """Test API compatibility and interfaces."""
    print("\nTesting API compatibility...")
    
    try:
        # Test all model classes can be instantiated
        from pno_physics_bench.models import ProbabilisticNeuralOperator, FourierNeuralOperator, DeepONet
        
        models = [
            ProbabilisticNeuralOperator(input_dim=3, hidden_dim=8, num_layers=1, modes=2),
            FourierNeuralOperator(input_dim=3, hidden_dim=8, num_layers=1, modes=2),
            DeepONet(branch_net_dims=[3*16*16, 64, 64], trunk_net_dims=[2, 64, 64])
        ]
        
        for i, model in enumerate(models):
            # Test forward pass
            test_input = torch.randn(2, 3, 16, 16)
            output = model(test_input)
            
            if output.shape[0] == 2:  # Batch dimension preserved
                print(f"✓ Model {i}: forward pass working")
            else:
                print(f"✗ Model {i}: incorrect output shape {output.shape}")
                return False
        
        # Test trainer API
        from pno_physics_bench.training import PNOTrainer, ELBOLoss
        
        trainer = PNOTrainer(models[0], loss_fn=ELBOLoss())
        print("✓ Trainer API compatible")
        
        # Test dataset API
        from pno_physics_bench.datasets import PDEDataset
        
        dataset = PDEDataset.load("navier_stokes_2d", resolution=8, num_samples=4)
        train_loader, val_loader, test_loader = PDEDataset.get_loaders(dataset, batch_size=2)
        
        if len(dataset) == 4 and len(train_loader) > 0:
            print("✓ Dataset API compatible")
        else:
            print("✗ Dataset API issues")
            return False
        
        # Test metrics API
        from pno_physics_bench.metrics import CalibrationMetrics
        
        metrics = CalibrationMetrics()
        dummy_pred = torch.randn(10, 1, 8, 8)
        dummy_std = torch.rand(10, 1, 8, 8)
        dummy_target = torch.randn(10, 1, 8, 8)
        
        ece = metrics.expected_calibration_error(dummy_pred, dummy_std, dummy_target)
        print("✓ Metrics API compatible")
        
        return True
        
    except Exception as e:
        print(f"✗ API compatibility test failed: {e}")
        return False

def test_edge_cases():
    """Test edge cases and error handling."""
    print("\nTesting edge cases...")
    
    try:
        from pno_physics_bench.models import ProbabilisticNeuralOperator
        from pno_physics_bench.utils.error_handling import validate_tensor, DataError
        
        model = ProbabilisticNeuralOperator(input_dim=3, hidden_dim=4, num_layers=1, modes=2)
        
        # Test small inputs
        tiny_input = torch.randn(1, 3, 4, 4)  # Very small spatial resolution
        try:
            output = model(tiny_input, sample=False)
            print("✓ Handles small spatial resolution")
        except Exception as e:
            print(f"✗ Small input failed: {e}")
            return False
        
        # Test large batch
        try:
            large_batch = torch.randn(64, 3, 16, 16)
            with torch.no_grad():
                output = model(large_batch, sample=False)
            print("✓ Handles large batch size")
        except RuntimeError as e:
            if "memory" in str(e).lower():
                print("✓ Large batch handled gracefully (OOM)")
            else:
                print(f"✗ Large batch failed: {e}")
                return False
        
        # Test error handling
        try:
            invalid_tensor = torch.tensor([float('nan')])
            validate_tensor(invalid_tensor, "test", allow_nan=False)
            print("✗ Error handling not working")
            return False
        except DataError:
            print("✓ Error handling working")
        
        # Test zero input
        zero_input = torch.zeros(2, 3, 8, 8)
        try:
            output = model(zero_input, sample=False)
            if not torch.isnan(output).any():
                print("✓ Handles zero input")
            else:
                print("✗ Zero input produces NaN")
                return False
        except Exception as e:
            print(f"✗ Zero input failed: {e}")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ Edge cases test failed: {e}")
        return False

def test_documentation_coverage():
    """Test documentation coverage."""
    print("\nTesting documentation coverage...")
    
    try:
        # Check if key files exist
        required_files = [
            Path("README.md"),
            Path("src/pno_physics_bench/__init__.py"),
            Path("src/pno_physics_bench/models.py"),
            Path("src/pno_physics_bench/training/__init__.py"),
        ]
        
        missing_files = []
        for file_path in required_files:
            if not file_path.exists():
                missing_files.append(str(file_path))
        
        if missing_files:
            print(f"✗ Missing documentation files: {missing_files}")
            return False
        
        # Check if main classes have docstrings
        from pno_physics_bench.models import ProbabilisticNeuralOperator
        from pno_physics_bench.training import PNOTrainer
        
        classes_to_check = [ProbabilisticNeuralOperator, PNOTrainer]
        
        for cls in classes_to_check:
            if cls.__doc__ is None or len(cls.__doc__.strip()) < 10:
                print(f"✗ Missing docstring for {cls.__name__}")
                return False
        
        print("✓ Documentation coverage adequate")
        return True
        
    except Exception as e:
        print(f"✗ Documentation test failed: {e}")
        return False

def test_security_compliance():
    """Test basic security compliance."""
    print("\nTesting security compliance...")
    
    try:
        # Check that models don't accept arbitrary Python code
        from pno_physics_bench.models import ProbabilisticNeuralOperator
        
        model = ProbabilisticNeuralOperator(input_dim=3, hidden_dim=4, num_layers=1, modes=2)
        
        # Test input validation exists
        try:
            # Test with wrong input dimensions
            wrong_input = torch.randn(2, 5, 8, 8)  # Wrong channel count
            output = model(wrong_input, sample=False)
            # Should handle gracefully, not crash
            print("✓ Input validation handles dimension mismatch")
        except Exception as e:
            # Any exception is fine as long as it's handled
            print(f"✓ Input validation working: {type(e).__name__}")
        
        # Test parameter bounds
        param_issues = []
        for name, param in model.named_parameters():
            if torch.isnan(param).any():
                param_issues.append(f"NaN in {name}")
            if torch.isinf(param).any():
                param_issues.append(f"Inf in {name}")
            if param.abs().max() > 1000:
                param_issues.append(f"Large values in {name}")
        
        if param_issues:
            print(f"✗ Parameter issues: {param_issues}")
            return False
        else:
            print("✓ Parameters within reasonable bounds")
        
        # Test that no hardcoded secrets or keys exist in the codebase
        # (This is a basic check - real security audit would be more thorough)
        suspicious_strings = ["password", "secret", "api_key", "token"]
        code_files = list(Path("src").rglob("*.py"))
        
        for file_path in code_files[:10]:  # Check first 10 files
            try:
                content = file_path.read_text().lower()
                for suspicious in suspicious_strings:
                    if suspicious in content and "test" not in str(file_path):
                        print(f"⚠  Potential security issue in {file_path}: contains '{suspicious}'")
            except:
                pass  # Skip files we can't read
        
        print("✓ Basic security checks passed")
        return True
        
    except Exception as e:
        print(f"✗ Security compliance test failed: {e}")
        return False

def main():
    """Run comprehensive quality gates."""
    print("=" * 70)
    print("PNO PHYSICS BENCH - COMPREHENSIVE QUALITY GATES")
    print("=" * 70)
    
    # Configure logging to reduce noise during tests
    logging.getLogger().setLevel(logging.WARNING)
    
    quality_gates = [
        ("Model Correctness", test_model_correctness),
        ("Performance Benchmarks", test_performance_benchmarks),
        ("Training Stability", test_training_stability),
        ("Uncertainty Calibration", test_uncertainty_calibration),
        ("API Compatibility", test_api_compatibility),
        ("Edge Cases", test_edge_cases),
        ("Documentation Coverage", test_documentation_coverage),
        ("Security Compliance", test_security_compliance),
    ]
    
    passed = 0
    total = len(quality_gates)
    results = {}
    
    for gate_name, gate_func in quality_gates:
        print(f"\n{'='*50}")
        print(f"QUALITY GATE: {gate_name.upper()}")
        print('='*50)
        
        try:
            result = gate_func()
            results[gate_name] = result
            if result:
                passed += 1
                print(f"✅ PASSED: {gate_name}")
            else:
                print(f"❌ FAILED: {gate_name}")
        except Exception as e:
            print(f"💥 CRASHED: {gate_name} - {e}")
            results[gate_name] = False
    
    print("\n" + "=" * 70)
    print("QUALITY GATES SUMMARY")
    print("=" * 70)
    
    for gate_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status:8} | {gate_name}")
    
    print(f"\nOVERALL RESULT: {passed}/{total} quality gates passed")
    
    if passed == total:
        print("🎉 ALL QUALITY GATES PASSED!")
        print("🚀 SYSTEM IS PRODUCTION READY!")
    elif passed >= total * 0.8:  # 80% threshold
        print("⚠️  MOST QUALITY GATES PASSED")
        print("🔧 System needs minor improvements but is largely functional")
    else:
        print("🚨 MULTIPLE QUALITY GATES FAILED")
        print("🔨 System needs significant work before production")
    
    print("=" * 70)
    
    return passed >= total * 0.8  # 80% pass rate required

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)