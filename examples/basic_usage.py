#!/usr/bin/env python3
"""Basic usage example for PNO Physics Bench."""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    import torch
    import numpy as np
    from pno_physics_bench.models import ProbabilisticNeuralOperator, FourierNeuralOperator
    from pno_physics_bench.datasets import PDEDataset
    from pno_physics_bench.training import PNOTrainer
    from pno_physics_bench.training.losses import ELBOLoss
    
    def main():
        print("🌊 PNO Physics Bench - Basic Usage Example")
        print("=" * 50)
        
        # Check device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        # 1. Load dataset
        print("\n📊 Loading Navier-Stokes dataset...")
        dataset = PDEDataset.load(
            "navier_stokes_2d", 
            resolution=64,
            num_samples=100,  # Small for demo
            normalize=True
        )
        
        # Get data loaders
        train_loader, val_loader, test_loader = dataset.get_loaders(
            batch_size=8,
            train_split=0.6,
            val_split=0.2,
            test_split=0.2
        )
        
        print(f"Dataset loaded: {len(dataset)} samples")
        print(f"Train: {len(train_loader.dataset)} samples")
        print(f"Val: {len(val_loader.dataset)} samples") 
        print(f"Test: {len(test_loader.dataset)} samples")
        
        # 2. Initialize models
        print("\n🧠 Initializing models...")
        
        # Standard FNO
        fno = FourierNeuralOperator(
            input_dim=3,
            hidden_dim=64,  # Smaller for demo
            num_layers=2,
            modes=12,
            activation="gelu"
        ).to(device)
        
        # Probabilistic Neural Operator
        pno = ProbabilisticNeuralOperator(
            input_dim=3,
            hidden_dim=64,
            num_layers=2,
            modes=12,
            uncertainty_type="full",
            posterior="variational"
        ).to(device)
        
        print(f"FNO parameters: {sum(p.numel() for p in fno.parameters()):,}")
        print(f"PNO parameters: {sum(p.numel() for p in pno.parameters()):,}")
        
        # 3. Test forward pass
        print("\n🔬 Testing forward passes...")
        
        # Get a sample batch
        sample_input, sample_target = next(iter(train_loader))
        sample_input = sample_input.to(device)
        sample_target = sample_target.to(device)
        
        print(f"Input shape: {sample_input.shape}")
        print(f"Target shape: {sample_target.shape}")
        
        # FNO forward pass
        with torch.no_grad():
            fno_output = fno(sample_input)
            print(f"FNO output shape: {fno_output.shape}")
            
            # PNO forward pass
            pno_output = pno(sample_input, sample=False)
            print(f"PNO output shape: {pno_output.shape}")
            
            # PNO with uncertainty
            mean, std = pno.predict_with_uncertainty(sample_input, num_samples=10)
            print(f"PNO uncertainty - Mean: {mean.shape}, Std: {std.shape}")
            print(f"Average uncertainty: {std.mean().item():.6f}")
        
        # 4. Setup training
        print("\n🚀 Setting up training...")
        
        # Loss function
        loss_fn = ELBOLoss(
            kl_weight=1e-4,
            num_samples=5,
            reconstruction_weight=1.0
        )
        
        # Trainer
        trainer = PNOTrainer(
            model=pno,
            loss_fn=loss_fn,
            optimizer=torch.optim.AdamW(pno.parameters(), lr=1e-3),
            device=device,
            num_samples=5,
            log_interval=1
        )
        
        print("Trainer initialized successfully!")
        
        # 5. Short training demo (just 2 epochs)
        print("\n🏋️ Running short training demo...")
        
        history = trainer.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=2
        )
        
        print("Training completed!")
        print(f"Final train loss: {history['train_loss'][-1]:.6f}")
        print(f"Final val loss: {history['val_loss'][-1]:.6f}")
        
        # 6. Evaluation
        print("\n📏 Evaluating model...")
        
        metrics = trainer.evaluate(test_loader, num_uncertainty_samples=20)
        
        print("Evaluation metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.6f}")
        
        print("\n✅ Basic usage example completed successfully!")
        
    if __name__ == "__main__":
        main()
        
except ImportError as e:
    print(f"❌ Missing dependencies: {e}")
    print("\nTo run this example, install the required dependencies:")
    print("pip install torch numpy scipy matplotlib seaborn h5py tqdm omegaconf hydra-core wandb tensorboard")
    
except Exception as e:
    print(f"❌ Error during execution: {e}")
    import traceback
    traceback.print_exc()