#!/usr/bin/env python3
"""Debug gradient NaN issue in PNO training."""

import torch
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from pno_physics_bench.models import ProbabilisticNeuralOperator
from pno_physics_bench.training import PNOTrainer, ELBOLoss
from pno_physics_bench.datasets import PDEDataset

def debug_gradient_nan():
    """Debug gradient NaN during backpropagation."""
    print("=" * 60)
    print("DEBUGGING GRADIENT NaN ISSUE")
    print("=" * 60)
    
    try:
        # Create model and optimizer
        model = ProbabilisticNeuralOperator(
            input_dim=3,
            hidden_dim=4,
            num_layers=1,
            modes=2
        )
        
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)  # Small learning rate
        loss_fn = ELBOLoss(kl_weight=1e-6)
        
        # Create data
        input_data = torch.randn(1, 3, 8, 8) * 0.1
        target_data = torch.randn(1, 1, 8, 8) * 0.1
        
        print("Before training step:")
        print(f"Input range: [{input_data.min():.4f}, {input_data.max():.4f}]")
        print(f"Target range: [{target_data.min():.4f}, {target_data.max():.4f}]")
        
        # Training step
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        print("\n1. Forward pass...")
        output = model(input_data, sample=True)
        print(f"Output range: [{output.min():.4f}, {output.max():.4f}]")
        print(f"Output has NaN: {torch.isnan(output).any()}")
        
        # Loss computation
        print("\n2. Loss computation...")
        losses = loss_fn(output, target_data, model)
        total_loss = losses['total']
        print(f"Total loss: {total_loss.item():.6f}")
        print(f"Loss has NaN: {torch.isnan(total_loss).any()}")
        
        # Backward pass
        print("\n3. Backward pass...")
        total_loss.backward()
        
        # Check gradients
        print("\n4. Checking gradients...")
        grad_norm = 0.0
        nan_grads = []
        inf_grads = []
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm += param.grad.norm().item() ** 2
                if torch.isnan(param.grad).any():
                    nan_grads.append(name)
                    print(f"NaN gradient in {name}: {torch.isnan(param.grad).sum()} NaNs")
                if torch.isinf(param.grad).any():
                    inf_grads.append(name)
                    print(f"Inf gradient in {name}: {torch.isinf(param.grad).sum()} Infs")
        
        grad_norm = grad_norm ** 0.5
        print(f"Total gradient norm: {grad_norm:.6f}")
        
        if nan_grads:
            print(f"Parameters with NaN gradients: {nan_grads}")
        if inf_grads:
            print(f"Parameters with Inf gradients: {inf_grads}")
        
        # Optimizer step
        print("\n5. Optimizer step...")
        if grad_norm < 1000:  # Only step if gradients are reasonable
            optimizer.step()
            print("Optimizer step completed")
        else:
            print(f"Skipping optimizer step due to large gradient norm: {grad_norm}")
        
        # Check parameters after step
        print("\n6. Checking parameters after step...")
        for name, param in model.named_parameters():
            if torch.isnan(param).any():
                print(f"NaN parameter after step: {name}")
            if torch.isinf(param).any():
                print(f"Inf parameter after step: {name}")
        
        # Test forward pass after step
        print("\n7. Forward pass after optimizer step...")
        with torch.no_grad():
            new_output = model(input_data, sample=False)
            print(f"New output range: [{new_output.min():.4f}, {new_output.max():.4f}]")
            print(f"New output has NaN: {torch.isnan(new_output).any()}")
        
        return True
        
    except Exception as e:
        print(f"Error during gradient debugging: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    debug_gradient_nan()