"""Command-line interface for PNO Physics Bench."""

import argparse
import sys
import logging
from pathlib import Path
from typing import Dict, Any
import torch
import yaml
from omegaconf import OmegaConf, DictConfig

from .models import ProbabilisticNeuralOperator
from .datasets import PDEDataset
from .training import PNOTrainer, ELBOLoss
from .training.callbacks import EarlyStopping, ModelCheckpoint, UncertaintyVisualization
from .benchmarks import PNOBenchmark
from .utils import setup_logging, set_random_seed


logger = logging.getLogger(__name__)


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for CLI."""
    parser = argparse.ArgumentParser(
        description="PNO Physics Bench - Training & benchmarking suite for Probabilistic Neural Operators",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train a PNO model')
    train_parser.add_argument('--config', type=str, required=True,
                             help='Path to training configuration file')
    train_parser.add_argument('--output-dir', type=str, default='./outputs',
                             help='Output directory for checkpoints and logs')
    train_parser.add_argument('--resume', type=str, default=None,
                             help='Path to checkpoint to resume from')
    train_parser.add_argument('--gpu', type=int, default=None,
                             help='GPU device ID (None for CPU)')
    train_parser.add_argument('--seed', type=int, default=42,
                             help='Random seed for reproducibility')
    train_parser.add_argument('--wandb', action='store_true',
                             help='Use Weights & Biases logging')
    train_parser.add_argument('--wandb-project', type=str, default='pno-physics-bench',
                             help='W&B project name')
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate a trained model')
    eval_parser.add_argument('--model', type=str, required=True,
                            help='Path to trained model checkpoint')
    eval_parser.add_argument('--config', type=str, required=True,
                            help='Path to configuration file')
    eval_parser.add_argument('--data', type=str, default=None,
                            help='Path to test dataset')
    eval_parser.add_argument('--output-dir', type=str, default='./eval_outputs',
                            help='Output directory for evaluation results')
    eval_parser.add_argument('--num-samples', type=int, default=100,
                            help='Number of MC samples for uncertainty estimation')
    eval_parser.add_argument('--visualize', action='store_true',
                            help='Generate uncertainty visualizations')
    
    # Benchmark command
    benchmark_parser = subparsers.add_parser('benchmark', help='Run comprehensive benchmarks')
    benchmark_parser.add_argument('--config', type=str, required=True,
                                 help='Path to benchmark configuration file')
    benchmark_parser.add_argument('--output-dir', type=str, default='./benchmark_outputs',
                                 help='Output directory for benchmark results')
    benchmark_parser.add_argument('--pdes', nargs='+', 
                                 default=['navier_stokes_2d', 'darcy_flow_2d'],
                                 help='PDE types to benchmark')
    benchmark_parser.add_argument('--methods', nargs='+',
                                 default=['PNO', 'FNO', 'DeepONet'],
                                 help='Methods to compare')
    benchmark_parser.add_argument('--num-seeds', type=int, default=3,
                                 help='Number of random seeds for each experiment')
    
    # Generate data command
    data_parser = subparsers.add_parser('generate-data', help='Generate PDE datasets')
    data_parser.add_argument('--pde', type=str, required=True,
                            help='PDE type to generate')
    data_parser.add_argument('--num-samples', type=int, default=1000,
                            help='Number of samples to generate')
    data_parser.add_argument('--resolution', type=int, default=64,
                            help='Spatial resolution')
    data_parser.add_argument('--output', type=str, required=True,
                            help='Output file path (HDF5 format)')
    data_parser.add_argument('--splits', nargs=3, type=float, 
                            default=[0.7, 0.2, 0.1],
                            help='Train/val/test split ratios')
    
    # Config template command
    config_parser = subparsers.add_parser('create-config', help='Create configuration template')
    config_parser.add_argument('--type', choices=['train', 'benchmark'], required=True,
                              help='Type of configuration to create')
    config_parser.add_argument('--output', type=str, required=True,
                              help='Output configuration file path')
    
    return parser


def load_config(config_path: str) -> DictConfig:
    """Load configuration from YAML file."""
    try:
        config = OmegaConf.load(config_path)
        return config
    except Exception as e:
        logger.error(f"Failed to load configuration from {config_path}: {e}")
        sys.exit(1)


def train_command(args: argparse.Namespace) -> None:
    """Execute training command."""
    logger.info("Starting PNO training")
    
    # Set random seed
    set_random_seed(args.seed)
    
    # Load configuration
    config = load_config(args.config)
    
    # Set device
    if args.gpu is not None:
        device = f'cuda:{args.gpu}'
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    logger.info(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    OmegaConf.save(config, output_dir / 'config.yaml')
    
    # Load dataset
    logger.info(f"Loading dataset: {config.data.pde_name}")
    dataset = PDEDataset(
        pde_name=config.data.pde_name,
        resolution=config.data.resolution,
        num_samples=config.data.num_samples,
        normalize=config.data.get('normalize', True),
        data_path=config.data.get('data_path', None),
    )
    
    # Create data loaders
    train_loader, val_loader, test_loader = dataset.get_loaders(
        batch_size=config.training.batch_size,
        val_split=config.data.get('val_split', 0.2),
        test_split=config.data.get('test_split', 0.1),
        num_workers=config.training.get('num_workers', 0),
    )
    
    logger.info(f"Dataset loaded: {len(train_loader.dataset)} train, "
               f"{len(val_loader.dataset)} val, {len(test_loader.dataset)} test samples")
    
    # Create model
    logger.info("Creating PNO model")
    model = ProbabilisticNeuralOperator(
        input_dim=dataset.input_dim,
        hidden_dim=config.model.hidden_dim,
        num_layers=config.model.num_layers,
        modes=config.model.modes,
        output_dim=dataset.output_dim,
        uncertainty_type=config.model.get('uncertainty_type', 'diagonal'),
        activation=config.model.get('activation', 'gelu'),
    )
    
    logger.info(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.get('weight_decay', 1e-4),
    )
    
    # Create scheduler
    scheduler = None
    if config.training.get('scheduler', None):
        scheduler_config = config.training.scheduler
        if scheduler_config.type == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=config.training.epochs
            )
        elif scheduler_config.type == 'plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', patience=scheduler_config.get('patience', 10)
            )
    
    # Create loss function
    loss_fn = ELBOLoss(
        kl_weight=config.training.get('kl_weight', 1e-4),
        num_samples=config.training.get('num_mc_samples', 5),
    )
    
    # Create callbacks
    callbacks = []
    
    # Early stopping
    if config.training.get('early_stopping', True):
        callbacks.append(EarlyStopping(
            patience=config.training.get('patience', 20),
            restore_best_weights=True,
        ))
    
    # Model checkpointing
    callbacks.append(ModelCheckpoint(
        filepath=str(output_dir / 'checkpoints' / 'model_epoch_{epoch:03d}.pt'),
        save_best_only=True,
    ))
    
    # Uncertainty visualization
    if config.training.get('visualize_uncertainty', True):
        callbacks.append(UncertaintyVisualization(
            val_dataset=val_loader.dataset,
            save_dir=str(output_dir / 'visualizations'),
            frequency=config.training.get('vis_frequency', 10),
        ))
    
    # Create trainer
    trainer = PNOTrainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        gradient_clipping=config.training.get('gradient_clipping', 1.0),
        mixed_precision=config.training.get('mixed_precision', False),
        num_samples=config.training.get('num_mc_samples', 5),
        log_interval=config.training.get('log_interval', 10),
        use_wandb=args.wandb,
        wandb_project=args.wandb_project,
        callbacks=callbacks,
    )
    
    # Train model
    logger.info("Starting training")
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config.training.epochs,
        resume_from=args.resume,
        save_dir=str(output_dir / 'checkpoints'),
    )
    
    # Final evaluation
    logger.info("Running final evaluation")
    test_metrics = trainer.evaluate(test_loader)
    
    # Save results
    results = {
        'training_history': history,
        'test_metrics': test_metrics,
        'config': OmegaConf.to_container(config),
    }
    
    torch.save(results, output_dir / 'training_results.pt')
    
    logger.info("Training completed successfully")
    logger.info(f"Results saved to {output_dir}")
    
    # Print final metrics
    logger.info("Final Test Metrics:")
    for key, value in test_metrics.items():
        logger.info(f"  {key}: {value:.4f}")


def evaluate_command(args: argparse.Namespace) -> None:
    """Execute evaluation command."""
    logger.info("Starting model evaluation")
    
    # Load configuration
    config = load_config(args.config)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    logger.info(f"Loading model from {args.model}")
    model = ProbabilisticNeuralOperator.load_checkpoint(args.model)
    model.eval()
    
    # Load test dataset
    if args.data:
        dataset = PDEDataset(
            pde_name=config.data.pde_name,
            data_path=args.data,
            split='test',
            normalize=config.data.get('normalize', True),
        )
    else:
        dataset = PDEDataset(
            pde_name=config.data.pde_name,
            resolution=config.data.resolution,
            num_samples=config.data.get('test_samples', 200),
            split='test',
            normalize=config.data.get('normalize', True),
        )
    
    test_loader = torch.utils.data.DataLoader(
        dataset, batch_size=32, shuffle=False, num_workers=0
    )
    
    logger.info(f"Test dataset loaded: {len(dataset)} samples")
    
    # Run evaluation
    from .uncertainty import UncertaintyDecomposer
    from .metrics import CalibrationMetrics
    
    uncertainty_decomposer = UncertaintyDecomposer()
    calibration_metrics = CalibrationMetrics()
    
    # Collect predictions
    all_predictions = []
    all_uncertainties = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            pred_mean, pred_std = model.predict_with_uncertainty(
                inputs, num_samples=args.num_samples
            )
            
            all_predictions.append(pred_mean)
            all_uncertainties.append(pred_std)
            all_targets.append(targets)
    
    # Concatenate results
    predictions = torch.cat(all_predictions, dim=0)
    uncertainties = torch.cat(all_uncertainties, dim=0)
    targets = torch.cat(all_targets, dim=0)
    
    # Compute comprehensive metrics
    metrics = calibration_metrics.compute_all_metrics(
        predictions, uncertainties, targets
    )
    
    # Uncertainty analysis
    correlation_results = uncertainty_decomposer.uncertainty_correlation_analysis(
        model, test_loader
    )
    
    # Save results
    results = {
        'metrics': metrics,
        'correlation_analysis': correlation_results,
        'config': OmegaConf.to_container(config),
    }
    
    torch.save(results, output_dir / 'evaluation_results.pt')
    
    # Generate visualizations
    if args.visualize:
        logger.info("Generating visualizations")
        
        # Reliability diagram
        fig = calibration_metrics.plot_reliability_diagram(
            predictions, uncertainties, targets
        )
        fig.savefig(output_dir / 'reliability_diagram.png', dpi=300, bbox_inches='tight')
        
        # Uncertainty correlation
        fig = uncertainty_decomposer.plot_uncertainty_correlation(correlation_results)
        fig.savefig(output_dir / 'uncertainty_correlation.png', dpi=300, bbox_inches='tight')
    
    logger.info("Evaluation completed")
    logger.info(f"Results saved to {output_dir}")
    
    # Print metrics
    logger.info("Evaluation Metrics:")
    for key, value in metrics.items():
        logger.info(f"  {key}: {value:.4f}")


def benchmark_command(args: argparse.Namespace) -> None:
    """Execute benchmark command."""
    logger.info("Starting comprehensive benchmarking")
    
    # Load configuration
    config = load_config(args.config)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create benchmark instance
    benchmark = PNOBenchmark(
        output_dir=str(output_dir),
        num_seeds=args.num_seeds,
    )
    
    # Run benchmarks
    results = {}
    for pde in args.pdes:
        logger.info(f"Benchmarking {pde}")
        
        pde_results = benchmark.compare_methods(
            pde_name=pde,
            methods=args.methods,
            config=config,
        )
        
        results[pde] = pde_results
    
    # Generate comparison report
    benchmark.generate_comparison_report(results, output_dir / 'benchmark_report.html')
    
    logger.info("Benchmarking completed")
    logger.info(f"Results saved to {output_dir}")


def generate_data_command(args: argparse.Namespace) -> None:
    """Execute data generation command."""
    logger.info(f"Generating {args.pde} dataset")
    
    # Create dataset with generation
    dataset = PDEDataset(
        pde_name=args.pde,
        resolution=args.resolution,
        num_samples=args.num_samples,
        generate_on_demand=True,
    )
    
    # Save dataset
    dataset.save_to_file(args.output)
    
    logger.info(f"Dataset saved to {args.output}")


def create_config_command(args: argparse.Namespace) -> None:
    """Create configuration template."""
    if args.type == 'train':
        config = {
            'data': {
                'pde_name': 'navier_stokes_2d',
                'resolution': 64,
                'num_samples': 1000,
                'normalize': True,
                'val_split': 0.2,
                'test_split': 0.1,
            },
            'model': {
                'hidden_dim': 64,
                'num_layers': 4,
                'modes': 20,
                'uncertainty_type': 'diagonal',
                'activation': 'gelu',
            },
            'training': {
                'epochs': 100,
                'batch_size': 32,
                'learning_rate': 1e-3,
                'weight_decay': 1e-4,
                'kl_weight': 1e-4,
                'num_mc_samples': 5,
                'gradient_clipping': 1.0,
                'mixed_precision': False,
                'early_stopping': True,
                'patience': 20,
                'log_interval': 10,
                'visualize_uncertainty': True,
                'vis_frequency': 10,
                'scheduler': {
                    'type': 'cosine',
                },
            },
        }
    elif args.type == 'benchmark':
        config = {
            'pdes': ['navier_stokes_2d', 'darcy_flow_2d'],
            'methods': ['PNO', 'FNO', 'DeepONet'],
            'resolutions': [32, 64],
            'num_samples': 1000,
            'training': {
                'epochs': 50,
                'batch_size': 32,
                'learning_rate': 1e-3,
            },
            'evaluation': {
                'num_mc_samples': 100,
                'confidence_levels': [0.5, 0.8, 0.9, 0.95],
            },
        }
    
    # Save configuration
    with open(args.output, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    
    logger.info(f"Configuration template created: {args.output}")


def main() -> None:
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Setup logging
    setup_logging(level=logging.INFO)
    
    # Execute command
    try:
        if args.command == 'train':
            train_command(args)
        elif args.command == 'evaluate':
            evaluate_command(args)
        elif args.command == 'benchmark':
            benchmark_command(args)
        elif args.command == 'generate-data':
            generate_data_command(args)
        elif args.command == 'create-config':
            create_config_command(args)
        else:
            logger.error(f"Unknown command: {args.command}")
            sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error executing command: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()