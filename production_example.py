#!/usr/bin/env python3
"""Production-ready example demonstrating the complete PNO Physics Bench workflow."""

import sys
import os
import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

def production_training_pipeline(
    config: Dict[str, Any],
    use_distributed: bool = False,
    enable_monitoring: bool = True,
    enable_optimization: bool = True
) -> Dict[str, Any]:
    """Production training pipeline with all optimizations enabled."""
    
    logger.info("🚀 Starting Production PNO Training Pipeline")
    
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader
        
        # Import PNO components
        from pno_physics_bench.models import ProbabilisticNeuralOperator
        from pno_physics_bench.datasets import PDEDataset
        from pno_physics_bench.training import PNOTrainer
        from pno_physics_bench.training.losses import ELBOLoss
        
        if enable_optimization:
            from pno_physics_bench.optimization_engine import (
                AdaptivePerformanceOptimizer,
                DynamicBatchSizer,
                AdaptiveMemoryManager
            )
        
        if enable_monitoring:
            from pno_physics_bench.monitoring_advanced import TrainingMonitor, RobustTrainingWrapper
        
        if use_distributed:
            from pno_physics_bench.distributed_training import AutoScalingTrainer
            
    except ImportError as e:
        logger.error(f"Missing dependencies: {e}")
        logger.info("Please install required dependencies:")
        logger.info("pip install torch numpy scipy matplotlib seaborn h5py tqdm omegaconf wandb")
        return {"status": "failed", "error": "Missing dependencies"}
    
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    results = {"status": "running", "metrics": {}}
    
    try:
        # 1. Setup monitoring
        if enable_monitoring:
            monitor = TrainingMonitor(
                log_dir=config.get("log_dir", "./production_logs"),
                memory_threshold=0.85,
                enable_profiling=True
            )
        
        # 2. Setup memory management
        if enable_optimization:
            memory_manager = AdaptiveMemoryManager(memory_fraction=0.8)
            memory_manager.setup_memory_management()
            
            # Dynamic batch sizing
            batch_sizer = DynamicBatchSizer(
                initial_batch_size=config.get("batch_size", 32),
                max_batch_size=config.get("max_batch_size", 256),
                memory_threshold=0.8
            )
        
        # 3. Create dataset
        logger.info("📊 Creating dataset...")
        dataset = PDEDataset.load(
            name=config.get("dataset_name", "navier_stokes_2d"),
            resolution=config.get("resolution", 64),
            num_samples=config.get("num_samples", 1000),
            normalize=True
        )
        
        # Get data loaders
        train_loader, val_loader, test_loader = dataset.get_loaders(
            batch_size=config.get("batch_size", 32),
            train_split=0.7,
            val_split=0.15,
            test_split=0.15,
            shuffle=True,
            num_workers=config.get("num_workers", 4)
        )
        
        logger.info(f"Dataset loaded: Train={len(train_loader.dataset)}, "
                   f"Val={len(val_loader.dataset)}, Test={len(test_loader.dataset)}")
        
        # 4. Create model
        logger.info("🧠 Creating model...")
        model = ProbabilisticNeuralOperator(
            input_dim=config.get("input_dim", 3),
            hidden_dim=config.get("hidden_dim", 256),
            num_layers=config.get("num_layers", 4),
            modes=config.get("modes", 20),
            uncertainty_type="full",
            posterior="variational"
        ).to(device)
        
        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # 5. Setup performance optimization
        if enable_optimization:
            logger.info("⚡ Applying performance optimizations...")
            perf_optimizer = AdaptivePerformanceOptimizer(
                model=model,
                enable_mixed_precision=True,
                enable_compilation=True,
                max_cache_size=1000
            )
        
        # 6. Create loss function and optimizer
        loss_fn = ELBOLoss(
            kl_weight=config.get("kl_weight", 1e-4),
            num_samples=config.get("mc_samples", 5),
            reconstruction_weight=1.0
        )
        
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.get("learning_rate", 1e-3),
            weight_decay=config.get("weight_decay", 1e-4)
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.get("epochs", 100),
            eta_min=1e-6
        )
        
        # 7. Create trainer
        logger.info("🏋️ Setting up trainer...")
        trainer = PNOTrainer(
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            gradient_clipping=1.0,
            mixed_precision=enable_optimization,
            num_samples=config.get("mc_samples", 5),
            log_interval=config.get("log_interval", 10),
            use_wandb=config.get("use_wandb", False),
            wandb_project=config.get("wandb_project", "pno-production")
        )
        
        # 8. Wrap with robust training if monitoring enabled
        if enable_monitoring:
            logger.info("🛡️ Enabling robust training...")
            robust_trainer = RobustTrainingWrapper(
                trainer=trainer,
                max_retries=3,
                save_interval=config.get("checkpoint_interval", 10),
                checkpoint_dir=config.get("checkpoint_dir", "./production_checkpoints")
            )
        
        # 9. Training with monitoring
        logger.info("🚀 Starting training...")
        start_time = time.time()
        
        if enable_monitoring:
            with monitor.training_session("production_training"):
                if enable_monitoring:
                    history = robust_trainer.fit_robust(
                        train_loader=train_loader,
                        val_loader=val_loader,
                        epochs=config.get("epochs", 10),  # Reduced for demo
                    )
                else:
                    history = trainer.fit(
                        train_loader=train_loader,
                        val_loader=val_loader,
                        epochs=config.get("epochs", 10),
                    )
        else:
            history = trainer.fit(
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=config.get("epochs", 10),
            )
        
        training_time = time.time() - start_time
        
        # 10. Evaluation
        logger.info("📏 Evaluating model...")
        eval_metrics = trainer.evaluate(test_loader, num_uncertainty_samples=50)
        
        # 11. Performance statistics
        perf_stats = {}
        if enable_optimization:
            perf_stats = perf_optimizer.get_performance_stats()
        
        # Compile results
        results.update({
            "status": "completed",
            "training_time": training_time,
            "final_train_loss": history["train_loss"][-1],
            "final_val_loss": history["val_loss"][-1],
            "eval_metrics": eval_metrics,
            "performance_stats": perf_stats,
            "model_parameters": sum(p.numel() for p in model.parameters()),
        })
        
        logger.info("✅ Training completed successfully!")
        logger.info(f"Training time: {training_time:.2f}s")
        logger.info(f"Final validation loss: {history['val_loss'][-1]:.6f}")
        logger.info(f"Test RMSE: {eval_metrics.get('rmse', 'N/A'):.6f}")
        logger.info(f"Test Coverage@90: {eval_metrics.get('coverage_90', 'N/A'):.3f}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        results.update({
            "status": "failed", 
            "error": str(e)
        })
        import traceback
        traceback.print_exc()
    
    return results


def research_workflow_demo():
    """Demonstrate advanced research features."""
    logger.info("🔬 Starting Research Workflow Demo")
    
    try:
        import torch
        from pno_physics_bench.research_extensions import (
            AttentionBasedUncertainty,
            HierarchicalUncertaintyPNO,
            CausalPhysicsInformedPNO
        )
        from pno_physics_bench.models import ProbabilisticNeuralOperator
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 1. Attention-based uncertainty
        logger.info("🎯 Testing attention-based uncertainty...")
        attention_unc = AttentionBasedUncertainty(
            input_dim=256,
            hidden_dim=256,
            num_heads=8
        ).to(device)
        
        # Test input
        test_features = torch.randn(2, 256, 32, 32).to(device)
        unc_results = attention_unc(test_features, return_attention=True)
        
        logger.info(f"Attention uncertainty computed: {list(unc_results['uncertainties'].keys())}")
        
        # 2. Hierarchical uncertainty
        logger.info("🏗️ Testing hierarchical uncertainty...")
        hierarchical_pno = HierarchicalUncertaintyPNO(
            input_dim=3,
            scales=[32, 64, 128],
            hidden_dims=[64, 128, 256],
            fusion_method="attention"
        ).to(device)
        
        test_input = torch.randn(2, 3, 64, 64).to(device)
        hier_results = hierarchical_pno(test_input)
        
        logger.info(f"Hierarchical prediction shape: {hier_results['mean'].shape}")
        logger.info(f"Scale outputs: {list(hier_results['scale_outputs'].keys())}")
        
        # 3. Physics-informed PNO
        logger.info("🧪 Testing physics-informed PNO...")
        base_model = ProbabilisticNeuralOperator(input_dim=3, hidden_dim=128).to(device)
        physics_pno = CausalPhysicsInformedPNO(
            base_model=base_model,
            physics_loss_weight=0.1,
            causal_weight=0.05
        ).to(device)
        
        physics_results = physics_pno(test_input, return_physics_loss=True)
        
        logger.info(f"Physics prediction shape: {physics_results['prediction'].shape}")
        logger.info(f"Physics loss: {physics_results['physics_loss'].item():.6f}")
        
        logger.info("✅ Research workflow demo completed!")
        
    except ImportError as e:
        logger.warning(f"Research demo skipped due to missing dependencies: {e}")
    except Exception as e:
        logger.error(f"Research workflow demo failed: {e}")


def main():
    """Main execution function."""
    logger.info("🌊 PNO Physics Bench - Production Example")
    logger.info("=" * 60)
    
    # Production configuration
    production_config = {
        # Dataset configuration
        "dataset_name": "navier_stokes_2d",
        "resolution": 64,
        "num_samples": 200,  # Small for demo
        
        # Model configuration
        "input_dim": 3,
        "hidden_dim": 128,  # Reduced for demo
        "num_layers": 3,
        "modes": 12,
        
        # Training configuration
        "batch_size": 16,  # Small for demo
        "max_batch_size": 64,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "epochs": 5,  # Very short for demo
        "kl_weight": 1e-4,
        "mc_samples": 5,
        
        # Optimization configuration
        "num_workers": 2,
        "log_interval": 1,
        "checkpoint_interval": 2,
        
        # Paths
        "log_dir": "./demo_logs",
        "checkpoint_dir": "./demo_checkpoints",
        
        # Experiment tracking
        "use_wandb": False,  # Disabled for demo
        "wandb_project": "pno-demo"
    }
    
    # Run production training pipeline
    results = production_training_pipeline(
        config=production_config,
        use_distributed=False,  # Single machine demo
        enable_monitoring=True,
        enable_optimization=True
    )
    
    # Print results
    logger.info("\n📊 PRODUCTION PIPELINE RESULTS")
    logger.info("=" * 60)
    
    if results["status"] == "completed":
        logger.info(f"✅ Status: {results['status']}")
        logger.info(f"⏱️  Training Time: {results['training_time']:.2f}s")
        logger.info(f"📉 Final Train Loss: {results['final_train_loss']:.6f}")
        logger.info(f"📉 Final Val Loss: {results['final_val_loss']:.6f}")
        logger.info(f"🎯 Test RMSE: {results['eval_metrics'].get('rmse', 'N/A'):.6f}")
        logger.info(f"🎯 Test Coverage@90: {results['eval_metrics'].get('coverage_90', 'N/A'):.3f}")
        logger.info(f"🧠 Model Parameters: {results['model_parameters']:,}")
        
        # Performance stats
        perf_stats = results.get("performance_stats", {})
        if perf_stats:
            logger.info(f"⚡ Cache Hit Rate: {perf_stats.get('cache_hit_rate', 0):.3f}")
            logger.info(f"⚡ Avg Forward Time: {perf_stats.get('avg_forward_time', 0):.4f}s")
    else:
        logger.error(f"❌ Status: {results['status']}")
        if "error" in results:
            logger.error(f"Error: {results['error']}")
    
    # Run research demo
    logger.info("\n🔬 RESEARCH FEATURES DEMO")
    logger.info("=" * 60)
    research_workflow_demo()
    
    logger.info("\n🎉 Production example completed!")
    logger.info("For full production deployment, see DEPLOYMENT.md")
    
    return results["status"] == "completed"


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)