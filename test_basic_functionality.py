#!/usr/bin/env python3
"""Basic functionality test for PNO Physics Bench."""

import torch
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_basic_imports():
    """Test that core components can be imported."""
    print("Testing basic imports...")
    
    try:
        from pno_physics_bench.models import ProbabilisticNeuralOperator, FourierNeuralOperator, DeepONet
        from pno_physics_bench.datasets import PDEDataset
        from pno_physics_bench.training import PNOTrainer, ELBOLoss
        from pno_physics_bench.metrics import CalibrationMetrics
        print("✓ All imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False

def test_model_creation():
    """Test model instantiation."""
    print("\nTesting model creation...")
    
    try:
        from pno_physics_bench.models import ProbabilisticNeuralOperator, FourierNeuralOperator, DeepONet
        
        # Test PNO
        pno = ProbabilisticNeuralOperator(
            input_dim=3,
            output_dim=3,
            hidden_dim=32,
            num_layers=2,
            modes=8
        )
        print(f"✓ PNO created: {sum(p.numel() for p in pno.parameters())} parameters")
        
        # Test FNO
        fno = FourierNeuralOperator(
            input_dim=3,
            output_dim=3,
            hidden_dim=32,
            num_layers=2,
            modes=8
        )
        print(f"✓ FNO created: {sum(p.numel() for p in fno.parameters())} parameters")
        
        # Test DeepONet
        deeponet = DeepONet(
            input_dim=3,
            output_dim=3,
            grid_size=32
        )
        print(f"✓ DeepONet created: {sum(p.numel() for p in deeponet.parameters())} parameters")
        
        return True
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        return False

def test_forward_pass():
    """Test forward pass through models."""
    print("\nTesting forward passes...")
    
    try:
        from pno_physics_bench.models import ProbabilisticNeuralOperator, FourierNeuralOperator, DeepONet
        
        # Create test input
        batch_size = 2
        input_dim = 3
        height, width = 32, 32
        x = torch.randn(batch_size, input_dim, height, width)
        
        # Test PNO
        pno = ProbabilisticNeuralOperator(
            input_dim=input_dim,
            output_dim=input_dim,
            hidden_dim=16,
            num_layers=2,
            modes=4,
            input_size=(height, width)
        )
        
        mean, log_var = pno(x, sample=False)
        print(f"✓ PNO forward pass: {mean.shape}, {log_var.shape}")
        
        # Test uncertainty prediction
        pred_mean, pred_std = pno.predict_with_uncertainty(x, num_samples=5)
        print(f"✓ PNO uncertainty prediction: {pred_mean.shape}, {pred_std.shape}")
        
        # Test FNO
        fno = FourierNeuralOperator(
            input_dim=input_dim,
            output_dim=input_dim,
            hidden_dim=16,
            num_layers=2,
            modes=4,
            input_size=(height, width)
        )
        
        output = fno(x)
        print(f"✓ FNO forward pass: {output.shape}")
        
        # Test DeepONet
        deeponet = DeepONet(
            input_dim=input_dim,
            output_dim=input_dim,
            grid_size=height
        )
        
        output = deeponet(x)
        print(f"✓ DeepONet forward pass: {output.shape}")
        
        return True
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_dataset_creation():
    """Test dataset creation and sampling."""
    print("\nTesting dataset creation...")
    
    try:
        from pno_physics_bench.datasets import PDEDataset
        
        # Test Navier-Stokes dataset
        dataset = PDEDataset(
            pde_name="navier_stokes_2d",
            resolution=16,  # Small for testing
            num_samples=5,
            generate_on_demand=True
        )
        
        print(f"✓ Dataset created: {len(dataset)} samples")
        
        # Test sampling
        input_data, target_data = dataset[0]
        print(f"✓ Sample shapes: input {input_data.shape}, target {target_data.shape}")
        
        # Test data loaders
        train_loader, val_loader, test_loader = dataset.get_loaders(batch_size=2)
        print(f"✓ Data loaders created: {len(train_loader)} train batches")
        
        # Test batch
        for inputs, targets in train_loader:
            print(f"✓ Batch shapes: {inputs.shape}, {targets.shape}")
            break
        
        return True
    except Exception as e:
        print(f"✗ Dataset creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_loss_functions():
    """Test loss function computation."""
    print("\nTesting loss functions...")
    
    try:
        from pno_physics_bench.training import ELBOLoss
        
        # Create test data
        batch_size = 2
        channels = 3
        height, width = 16, 16
        
        pred_mean = torch.randn(batch_size, channels, height, width)
        pred_log_var = torch.randn(batch_size, channels, height, width) * 0.1  # Small variance
        targets = torch.randn(batch_size, channels, height, width)
        
        # Test ELBO loss
        elbo_loss = ELBOLoss(kl_weight=1e-4)
        losses = elbo_loss((pred_mean, pred_log_var), targets)
        
        print(f"✓ ELBO loss computed: {losses['total'].item():.4f}")
        print(f"  - Reconstruction: {losses['reconstruction'].item():.4f}")
        if 'kl_divergence' in losses:
            print(f"  - KL divergence: {losses['kl_divergence'].item():.4f}")
        
        return True
    except Exception as e:
        print(f"✗ Loss function test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_metrics():
    """Test calibration metrics."""
    print("\nTesting calibration metrics...")
    
    try:
        from pno_physics_bench.metrics import CalibrationMetrics
        
        # Create test data
        predictions = torch.randn(100, 3, 8, 8)
        uncertainties = torch.abs(torch.randn(100, 3, 8, 8)) + 0.1  # Positive uncertainties
        targets = predictions + 0.1 * torch.randn_like(predictions)  # Add some noise
        
        metrics = CalibrationMetrics()
        
        # Test basic metrics
        ece = metrics.expected_calibration_error(predictions, uncertainties, targets)
        coverage = metrics.coverage_at_confidence(predictions, uncertainties, targets, 0.9)
        sharpness = metrics.sharpness(uncertainties)
        
        print(f"✓ ECE: {ece:.4f}")
        print(f"✓ Coverage@90%: {coverage:.4f}")
        print(f"✓ Sharpness: {sharpness:.4f}")
        
        # Test comprehensive metrics
        all_metrics = metrics.compute_all_metrics(predictions, uncertainties, targets)
        print(f"✓ Computed {len(all_metrics)} total metrics")
        
        return True
    except Exception as e:
        print(f"✗ Metrics test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_training_setup():
    """Test training setup without actual training."""
    print("\nTesting training setup...")
    
    try:
        from pno_physics_bench.models import ProbabilisticNeuralOperator
        from pno_physics_bench.training import PNOTrainer, ELBOLoss
        from pno_physics_bench.datasets import PDEDataset
        
        # Create small model and dataset
        model = ProbabilisticNeuralOperator(
            input_dim=3,
            output_dim=3,
            hidden_dim=8,
            num_layers=1,
            modes=2,
            input_size=(8, 8)
        )
        
        dataset = PDEDataset(
            pde_name="navier_stokes_2d",
            resolution=8,
            num_samples=4,
            generate_on_demand=True
        )
        
        train_loader, val_loader, _ = dataset.get_loaders(batch_size=2)
        
        # Create trainer
        trainer = PNOTrainer(
            model=model,
            loss_fn=ELBOLoss(kl_weight=1e-4),
            num_samples=2,
            log_interval=1,
            use_wandb=False
        )
        
        print(f"✓ Trainer created with {sum(p.numel() for p in model.parameters())} parameters")
        
        # Test single training step
        for inputs, targets in train_loader:
            outputs = model(inputs, sample=True, return_kl=True)
            losses = trainer.loss_fn(outputs, targets, model)
            print(f"✓ Training step: loss = {losses['total'].item():.4f}")
            break
        
        return True
    except Exception as e:
        print(f"✗ Training setup failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all basic functionality tests."""
    print("=" * 60)
    print("PNO PHYSICS BENCH - BASIC FUNCTIONALITY TEST")
    print("=" * 60)
    
    tests = [
        test_basic_imports,
        test_model_creation,
        test_forward_pass,
        test_dataset_creation,
        test_loss_functions,
        test_metrics,
        test_training_setup,
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
        print("🎉 ALL TESTS PASSED - GENERATION 1 COMPLETE!")
        print("Basic functionality is working correctly.")
    else:
        print("⚠️  Some tests failed - debugging needed.")
    
    print("=" * 60)
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)