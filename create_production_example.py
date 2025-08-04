#!/usr/bin/env python3
"""
Production-ready example demonstrating PNO Physics Bench capabilities.
This showcases all three generations: Make it Work, Make it Reliable, Make it Scale.
"""

import torch
import sys
import os
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def production_training_example():
    """Complete production training example with all features."""
    
    print("🚀 PNO PHYSICS BENCH - PRODUCTION EXAMPLE")
    print("=" * 60)
    
    # Generation 1: Make it Work - Core functionality
    print("\n📈 GENERATION 1: CORE FUNCTIONALITY")
    
    from pno_physics_bench.models import ProbabilisticNeuralOperator
    from pno_physics_bench.datasets import PDEDataset
    from pno_physics_bench.training import PNOTrainer, ELBOLoss
    from pno_physics_bench.metrics import CalibrationMetrics
    
    # Create model with proper configuration
    model_config = {
        'input_dim': 3,
        'output_dim': 3,
        'hidden_dim': 32,  # Small for demo
        'num_layers': 2,
        'modes': 8,
        'input_size': (32, 32),
        'uncertainty_type': 'diagonal'
    }
    
    model = ProbabilisticNeuralOperator(**model_config)
    print(f"✅ Model created: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Create dataset
    dataset_config = {
        'pde_name': 'navier_stokes_2d',
        'resolution': 32,  # Small for demo
        'num_samples': 100,
        'generate_on_demand': True
    }
    
    dataset = PDEDataset(**dataset_config)
    train_loader, val_loader, test_loader = dataset.get_loaders(batch_size=8)
    print(f"✅ Dataset created: {len(dataset)} samples")
    
    # Generation 2: Make it Reliable - Add robustness
    print("\n🛡️ GENERATION 2: RELIABILITY & VALIDATION")
    
    from pno_physics_bench.utils import validate_model_config, validate_training_config
    from pno_physics_bench.utils import set_random_seed, get_device_info
    
    # Validate configurations
    validated_model_config = validate_model_config(model_config, "PNO")
    print("✅ Model configuration validated")
    
    training_config = {
        'learning_rate': 1e-3,
        'batch_size': 8,
        'epochs': 5,  # Short for demo
        'kl_weight': 1e-4,
        'gradient_clipping': 1.0,
        'early_stopping_patience': 3
    }
    
    validated_training_config = validate_training_config(training_config)
    print("✅ Training configuration validated")
    
    # Setup reproducibility
    set_random_seed(42, deterministic=True)
    device_info = get_device_info()
    print(f"✅ Reproducibility setup: Device = {'GPU' if device_info['cuda_available'] else 'CPU'}")
    
    # Generation 3: Make it Scale - Performance optimization
    print("\n⚡ GENERATION 3: PERFORMANCE & MONITORING")
    
    from pno_physics_bench.utils import profile_model, PerformanceProfiler
    from pno_physics_bench.monitoring import HealthChecker
    
    # Profile model performance
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_metrics = profile_model(
        model, 
        input_shape=(1, 3, 32, 32),
        device=device,
        num_runs=10
    )
    print(f"✅ Model profiled: {model_metrics['mean_time']*1000:.2f}ms per forward pass")
    print(f"   Throughput: {model_metrics['throughput']:.1f} samples/sec")
    
    # Setup monitoring
    health_checker = HealthChecker()
    health_checker.setup_default_checks()
    health_checker.add_model_monitor('pno_model', model)
    print("✅ Health monitoring configured")
    
    # Enhanced training with monitoring
    print("\n🔥 TRAINING WITH FULL MONITORING")
    
    from pno_physics_bench.training.callbacks import EarlyStopping, ModelCheckpoint, MetricsLogger
    
    # Setup callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=3, verbose=True),
        ModelCheckpoint(
            filepath='./checkpoints/best_model_epoch_{epoch}.pt',
            monitor='val_loss',
            save_best_only=True,
            verbose=True
        ),
        MetricsLogger(log_file='training_metrics.csv')
    ]
    
    # Create trainer with enhanced loss
    trainer = PNOTrainer(
        model=model,
        loss_fn=ELBOLoss(kl_weight=1e-4, num_samples=3),
        gradient_clipping=1.0,
        mixed_precision=torch.cuda.is_available(),
        num_samples=3,
        log_interval=1,
        use_wandb=False,  # Set to True if wandb is available
        callbacks=callbacks
    )
    
    print("✅ Enhanced trainer configured with callbacks")
    
    # Performance profiler for training
    profiler = PerformanceProfiler()
    
    # Training loop with monitoring
    print("🚂 Starting monitored training...")
    
    try:
        with profiler.profile("full_training"):
            # Monitor health before training
            health_summary = health_checker.get_health_summary()
            print(f"Pre-training health: {health_summary['overall_status']}")
            
            # Train model
            training_history = trainer.fit(
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=5,
                save_dir='./checkpoints'
            )
            
            # Monitor health after training
            health_summary = health_checker.get_health_summary()
            print(f"Post-training health: {health_summary['overall_status']}")
    
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return False
    
    # Comprehensive evaluation
    print("\n📊 COMPREHENSIVE EVALUATION")
    
    try:
        # Model evaluation with uncertainty
        test_metrics = trainer.evaluate(test_loader, num_uncertainty_samples=20)
        
        print("📈 Test Results:")
        print(f"  RMSE: {test_metrics.get('rmse', 0):.4f}")
        print(f"  NLL: {test_metrics.get('nll', 0):.4f}")
        print(f"  ECE: {test_metrics.get('ece', 0):.4f}")
        print(f"  Coverage@90%: {test_metrics.get('coverage_90', 0):.3f}")
        
        # Performance summary
        perf_summary = profiler.get_summary()
        training_time = perf_summary['profiles']['full_training']['duration']
        print(f"⏱️  Total training time: {training_time:.2f}s")
        
        # Final health check
        final_health = health_checker.get_health_summary()
        print(f"🏥 Final system health: {final_health['overall_status']}")
        print(f"   Healthy checks: {final_health['num_healthy']}")
        print(f"   Warning checks: {final_health['num_warning']}")
        print(f"   Critical checks: {final_health['num_critical']}")
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("🎉 PRODUCTION EXAMPLE COMPLETED SUCCESSFULLY!")
    print("✅ All three generations working:")
    print("   - Generation 1: Core functionality (models, training, datasets)")
    print("   - Generation 2: Reliability (validation, error handling, reproducibility)")
    print("   - Generation 3: Scalability (profiling, monitoring, optimization)")
    print("=" * 60)
    
    return True


if __name__ == "__main__":
    success = production_training_example()
    sys.exit(0 if success else 1)