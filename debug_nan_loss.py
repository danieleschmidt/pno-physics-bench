#!/usr/bin/env python3
"""Debug NaN loss issue in PNO training."""

import torch
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from pno_physics_bench.models import ProbabilisticNeuralOperator
from pno_physics_bench.training import PNOTrainer, ELBOLoss
from pno_physics_bench.datasets import PDEDataset

def debug_nan_loss():
    """Debug the NaN loss issue."""
    print("=" * 60)
    print("DEBUGGING NaN LOSS ISSUE")
    print("=" * 60)
    
    try:
        # Create minimal setup
        model = ProbabilisticNeuralOperator(
            input_dim=3,
            hidden_dim=4,  # Very small for debugging
            num_layers=1,
            modes=2
        )
        
        # Create single sample manually to avoid dataset complexity
        batch_size = 1
        input_data = torch.randn(batch_size, 3, 8, 8) * 0.1  # Small values
        target_data = torch.randn(batch_size, 1, 8, 8) * 0.1
        
        print(f"Input range: [{input_data.min():.4f}, {input_data.max():.4f}]")
        print(f"Target range: [{target_data.min():.4f}, {target_data.max():.4f}]")
        
        # Test model forward pass
        print("\n1. Testing model forward pass...")
        model.eval()
        with torch.no_grad():
            output = model(input_data, sample=False)
            print(f"Model output shape: {output.shape}")
            print(f"Model output range: [{output.min():.4f}, {output.max():.4f}]")
            print(f"Model output has NaN: {torch.isnan(output).any()}")
            print(f"Model output has Inf: {torch.isinf(output).any()}")
        
        # Test uncertainty prediction
        print("\n2. Testing uncertainty prediction...")
        with torch.no_grad():
            pred_mean, pred_std = model.predict_with_uncertainty(input_data, num_samples=2)
            print(f"Pred mean shape: {pred_mean.shape}, range: [{pred_mean.min():.4f}, {pred_mean.max():.4f}]")
            print(f"Pred std shape: {pred_std.shape}, range: [{pred_std.min():.4f}, {pred_std.max():.4f}]")
            print(f"Pred mean has NaN: {torch.isnan(pred_mean).any()}")
            print(f"Pred std has NaN: {torch.isnan(pred_std).any()}")
            print(f"Pred std has zero/negative values: {(pred_std <= 0).any()}")
        
        # Test loss computation
        print("\n3. Testing loss computation...")
        loss_fn = ELBOLoss(kl_weight=1e-6)  # Very small KL weight
        
        model.train()
        output = model(input_data, sample=True)
        print(f"Training output shape: {output.shape}")
        print(f"Training output range: [{output.min():.4f}, {output.max():.4f}]")
        print(f"Training output has NaN: {torch.isnan(output).any()}")
        
        losses = loss_fn(output, target_data, model)
        print(f"Loss components:")
        for key, value in losses.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {value.item():.6f} (NaN: {torch.isnan(value).any()})")
            else:
                print(f"  {key}: {value}")
        
        # Test KL divergence
        print("\n4. Testing KL divergence...")
        if hasattr(model, 'kl_divergence'):
            kl = model.kl_divergence()
            print(f"Model KL divergence: {kl.item():.6f} (NaN: {torch.isnan(kl).any()})")
        
        # Check individual layers for NaN
        print("\n5. Checking model parameters...")
        for name, param in model.named_parameters():
            if torch.isnan(param).any():
                print(f"FOUND NaN in parameter: {name}")
                print(f"  Shape: {param.shape}")
                print(f"  NaN count: {torch.isnan(param).sum()}")
            if torch.isinf(param).any():
                print(f"FOUND Inf in parameter: {name}")
                print(f"  Shape: {param.shape}")  
                print(f"  Inf count: {torch.isinf(param).sum()}")
        
        print("\n6. Testing with very simple inputs...")
        # Test with all zeros
        zero_input = torch.zeros(1, 3, 8, 8)
        zero_target = torch.zeros(1, 1, 8, 8)
        
        with torch.no_grad():
            zero_output = model(zero_input, sample=False)
            print(f"Zero input output: {zero_output.mean():.6f} (NaN: {torch.isnan(zero_output).any()})")
        
        # Test with ones
        ones_input = torch.ones(1, 3, 8, 8)
        ones_target = torch.ones(1, 1, 8, 8)
        
        with torch.no_grad():
            ones_output = model(ones_input, sample=False)
            print(f"Ones input output: {ones_output.mean():.6f} (NaN: {torch.isnan(ones_output).any()})")
        
        return True
        
    except Exception as e:
        print(f"Error during debugging: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    debug_nan_loss()