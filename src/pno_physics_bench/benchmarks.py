"""Benchmarking utilities for comparing PNO methods."""

import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from omegaconf import DictConfig
import time

from .models import ProbabilisticNeuralOperator, FourierNeuralOperator, DeepONet
from .datasets import PDEDataset
from .training import PNOTrainer, ELBOLoss
from .metrics import CalibrationMetrics
from .uncertainty import UncertaintyDecomposer
from .utils import set_random_seed, format_time


logger = logging.getLogger(__name__)


class PNOBenchmark:
    """Comprehensive benchmarking suite for PNO methods."""
    
    def __init__(
        self,
        output_dir: str = "./benchmark_outputs",
        num_seeds: int = 3,
        device: Optional[str] = None,
    ):
        """Initialize benchmark suite.
        
        Args:
            output_dir: Directory to save benchmark results
            num_seeds: Number of random seeds for each experiment
            device: Device for computation
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.num_seeds = num_seeds
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize metrics
        self.calibration_metrics = CalibrationMetrics()
        self.uncertainty_decomposer = UncertaintyDecomposer()
        
        logger.info(f"Benchmark suite initialized, output dir: {self.output_dir}")
    
    def compare_methods(
        self,
        pde_name: str,
        methods: List[str] = ['PNO', 'FNO', 'DeepONet'],
        config: Optional[DictConfig] = None,
        resolution: int = 64,
        num_samples: int = 1000,
        epochs: int = 50,
    ) -> Dict[str, Dict[str, Any]]:
        """Compare different methods on a specific PDE.
        
        Args:
            pde_name: Name of PDE to benchmark
            methods: List of methods to compare
            config: Configuration dictionary
            resolution: Spatial resolution
            num_samples: Number of training samples
            epochs: Number of training epochs
            
        Returns:
            Dictionary with results for each method
        """
        logger.info(f"Benchmarking methods {methods} on {pde_name}")
        
        results = {}
        
        for method in methods:
            logger.info(f"Running {method} experiments")
            method_results = self._run_method_experiments(
                method=method,
                pde_name=pde_name,
                config=config,
                resolution=resolution,
                num_samples=num_samples,
                epochs=epochs,
            )
            results[method] = method_results
        
        # Save comparison results
        self._save_comparison_results(results, pde_name)
        
        return results
    
    def _run_method_experiments(
        self,
        method: str,
        pde_name: str,
        config: Optional[DictConfig],
        resolution: int,
        num_samples: int,
        epochs: int,
    ) -> Dict[str, Any]:
        """Run experiments for a single method across multiple seeds.
        
        Args:
            method: Method name
            pde_name: PDE name
            config: Configuration
            resolution: Spatial resolution
            num_samples: Number of samples
            epochs: Number of epochs
            
        Returns:
            Aggregated results across seeds
        """
        seed_results = []
        
        for seed in range(self.num_seeds):
            logger.info(f"Running {method} with seed {seed}")
            
            try:
                result = self._run_single_experiment(
                    method=method,
                    pde_name=pde_name,
                    config=config,
                    resolution=resolution,
                    num_samples=num_samples,
                    epochs=epochs,
                    seed=seed,
                )
                seed_results.append(result)
            except Exception as e:
                logger.error(f"Failed to run {method} with seed {seed}: {e}")
                continue
        
        if not seed_results:
            logger.error(f"All experiments failed for {method}")
            return {}
        
        # Aggregate results across seeds
        aggregated = self._aggregate_seed_results(seed_results)
        
        return aggregated
    
    def _run_single_experiment(
        self,
        method: str,
        pde_name: str,
        config: Optional[DictConfig],
        resolution: int,
        num_samples: int,
        epochs: int,
        seed: int,
    ) -> Dict[str, Any]:
        """Run a single experiment for one method and seed.
        
        Args:
            method: Method name
            pde_name: PDE name
            config: Configuration
            resolution: Spatial resolution
            num_samples: Number of samples
            epochs: Number of epochs
            seed: Random seed
            
        Returns:
            Experiment results
        """
        # Set random seed
        set_random_seed(seed)
        
        # Create dataset
        dataset = PDEDataset(
            pde_name=pde_name,
            resolution=resolution,
            num_samples=num_samples,
            normalize=True,
            generate_on_demand=True,
        )
        
        # Create data loaders
        train_loader, val_loader, test_loader = dataset.get_loaders(
            batch_size=32,
            val_split=0.2,
            test_split=0.1,
        )
        
        # Create model based on method
        if method == 'PNO':
            model = ProbabilisticNeuralOperator(
                input_dim=dataset.input_dim,
                hidden_dim=64,
                num_layers=4,
                modes=20,
                output_dim=dataset.output_dim,
                uncertainty_type='diagonal',
            )
            loss_fn = ELBOLoss(kl_weight=1e-4)
        elif method == 'FNO':
            model = FourierNeuralOperator(
                input_dim=dataset.input_dim,
                hidden_dim=64,
                num_layers=4,
                modes=20,
                output_dim=dataset.output_dim,
            )
            loss_fn = torch.nn.MSELoss()
        elif method == 'DeepONet':
            model = DeepONet(
                input_dim=dataset.input_dim,
                grid_size=resolution,
                hidden_dim=128,
                num_layers=4,
                output_dim=dataset.output_dim,
            )
            loss_fn = torch.nn.MSELoss()
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Create optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=1e-3,
            weight_decay=1e-4,
        )
        
        # Create trainer (for PNO) or simple training loop (for baselines)
        start_time = time.time()
        
        if method == 'PNO':
            trainer = PNOTrainer(
                model=model,
                loss_fn=loss_fn,
                optimizer=optimizer,
                device=self.device,
                num_samples=5,
                log_interval=epochs + 1,  # Suppress logging
                use_wandb=False,
            )
            
            # Train
            trainer.fit(
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=epochs,
            )
            
            # Evaluate
            test_metrics = trainer.evaluate(test_loader, num_uncertainty_samples=50)
        else:
            # Simple training for baselines
            model.to(self.device)
            model.train()
            
            for epoch in range(epochs):
                epoch_loss = 0.0
                for inputs, targets in train_loader:
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                    
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = loss_fn(outputs, targets)
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item()
            
            # Evaluate
            model.eval()
            test_metrics = self._evaluate_baseline(model, test_loader)
        
        training_time = time.time() - start_time
        
        # Add timing information
        test_metrics['training_time'] = training_time
        test_metrics['seed'] = seed
        
        return test_metrics
    
    def _evaluate_baseline(self, model: torch.nn.Module, test_loader) -> Dict[str, float]:
        """Evaluate baseline model (FNO, DeepONet) without uncertainty.
        
        Args:
            model: Trained baseline model
            test_loader: Test data loader
            
        Returns:
            Evaluation metrics
        """
        model.eval()
        
        all_predictions = []
        all_targets = []
        total_loss = 0.0
        total_samples = 0
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                outputs = model(inputs)
                loss = torch.nn.functional.mse_loss(outputs, targets)
                
                total_loss += loss.item() * inputs.shape[0]
                total_samples += inputs.shape[0]
                
                all_predictions.append(outputs.cpu())
                all_targets.append(targets.cpu())
        
        # Concatenate results
        predictions = torch.cat(all_predictions, dim=0)
        targets = torch.cat(all_targets, dim=0)
        
        # Compute basic metrics
        metrics = {
            'test_loss': total_loss / total_samples,
            'rmse': torch.sqrt(torch.mean((predictions - targets) ** 2)).item(),
            'mae': torch.mean(torch.abs(predictions - targets)).item(),
        }
        
        # No uncertainty metrics for baselines
        metrics.update({
            'ece': np.nan,
            'nll': np.nan,
            'coverage_90': np.nan,
            'sharpness': np.nan,
        })
        
        return metrics
    
    def _aggregate_seed_results(self, seed_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate results across multiple seeds.
        
        Args:
            seed_results: List of results from different seeds
            
        Returns:
            Aggregated statistics
        """
        if not seed_results:
            return {}
        
        # Collect all metrics
        metric_names = set()
        for result in seed_results:
            metric_names.update(result.keys())
        
        aggregated = {}
        
        for metric in metric_names:
            if metric == 'seed':
                continue
                
            values = []
            for result in seed_results:
                if metric in result and not np.isnan(result[metric]):
                    values.append(result[metric])
            
            if values:
                aggregated[f'{metric}_mean'] = np.mean(values)
                aggregated[f'{metric}_std'] = np.std(values)
                aggregated[f'{metric}_min'] = np.min(values)
                aggregated[f'{metric}_max'] = np.max(values)
            else:
                aggregated[f'{metric}_mean'] = np.nan
                aggregated[f'{metric}_std'] = np.nan
                aggregated[f'{metric}_min'] = np.nan
                aggregated[f'{metric}_max'] = np.nan
        
        aggregated['num_seeds'] = len(seed_results)
        
        return aggregated
    
    def _save_comparison_results(self, results: Dict[str, Dict[str, Any]], pde_name: str) -> None:
        """Save comparison results to files.
        
        Args:
            results: Results dictionary
            pde_name: PDE name
        """
        # Create DataFrame for easy analysis
        rows = []
        for method, method_results in results.items():
            if not method_results:
                continue
                
            row = {'method': method, 'pde': pde_name}
            row.update(method_results)
            rows.append(row)
        
        if rows:
            df = pd.DataFrame(rows)
            csv_path = self.output_dir / f'{pde_name}_results.csv'
            df.to_csv(csv_path, index=False)
            logger.info(f"Results saved to {csv_path}")
    
    def generate_comparison_table(
        self,
        results: Dict[str, Dict[str, Dict[str, Any]]],
        output_path: str = "benchmark_table.tex",
    ) -> None:
        """Generate LaTeX comparison table.
        
        Args:
            results: Nested results dictionary {pde: {method: metrics}}
            output_path: Output file path
        """
        # Collect data for table
        table_data = []
        
        for pde, pde_results in results.items():
            for method, method_results in pde_results.items():
                if not method_results:
                    continue
                    
                row = {
                    'PDE': pde.replace('_', ' ').title(),
                    'Method': method,
                    'RMSE': f"{method_results.get('rmse_mean', np.nan):.4f} ± {method_results.get('rmse_std', 0):.4f}",
                    'NLL': f"{method_results.get('nll_mean', np.nan):.2f} ± {method_results.get('nll_std', 0):.2f}",
                    'ECE': f"{method_results.get('ece_mean', np.nan):.4f} ± {method_results.get('ece_std', 0):.4f}",
                    'Coverage@90': f"{method_results.get('coverage_90_mean', np.nan):.3f} ± {method_results.get('coverage_90_std', 0):.3f}",
                    'Time (s)': f"{method_results.get('training_time_mean', np.nan):.1f}",
                }
                table_data.append(row)
        
        # Create DataFrame and save as LaTeX
        df = pd.DataFrame(table_data)
        
        latex_table = df.to_latex(
            index=False,
            escape=False,
            column_format='l|l|c|c|c|c|c',
            caption='Comparison of neural operator methods across different PDEs',
            label='tab:pno_comparison',
        )
        
        with open(output_path, 'w') as f:
            f.write(latex_table)
        
        logger.info(f"LaTeX table saved to {output_path}")
    
    def generate_comparison_report(
        self,
        results: Dict[str, Dict[str, Dict[str, Any]]],
        output_path: str = "benchmark_report.html",
    ) -> None:
        """Generate comprehensive HTML benchmark report.
        
        Args:
            results: Nested results dictionary
            output_path: Output HTML file path
        """
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>PNO Benchmark Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: center; }}
                th {{ background-color: #f2f2f2; }}
                .best {{ background-color: #90EE90; }}
                .header {{ color: #333; }}
            </style>
        </head>
        <body>
            <h1 class="header">PNO Physics Bench - Benchmark Report</h1>
            <p>Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <h2>Summary</h2>
            <p>This report compares different neural operator methods across various PDEs.</p>
            <ul>
                <li><strong>PNO</strong>: Probabilistic Neural Operator with uncertainty quantification</li>
                <li><strong>FNO</strong>: Standard Fourier Neural Operator (deterministic)</li>
                <li><strong>DeepONet</strong>: Deep Operator Network (deterministic)</li>
            </ul>
        """
        
        # Add comparison tables for each PDE
        for pde, pde_results in results.items():
            html_content += f"""
            <h2>{pde.replace('_', ' ').title()} Results</h2>
            <table>
                <tr>
                    <th>Method</th>
                    <th>RMSE</th>
                    <th>MAE</th>
                    <th>NLL</th>
                    <th>ECE</th>
                    <th>Coverage@90%</th>
                    <th>Training Time (s)</th>
                </tr>
            """
            
            # Find best values for highlighting
            best_rmse = min([r.get('rmse_mean', float('inf')) for r in pde_results.values() if r])
            best_mae = min([r.get('mae_mean', float('inf')) for r in pde_results.values() if r])
            
            for method, method_results in pde_results.items():
                if not method_results:
                    continue
                
                # Highlight best values
                rmse_class = 'best' if method_results.get('rmse_mean', float('inf')) == best_rmse else ''
                mae_class = 'best' if method_results.get('mae_mean', float('inf')) == best_mae else ''
                
                html_content += f"""
                <tr>
                    <td><strong>{method}</strong></td>
                    <td class="{rmse_class}">{method_results.get('rmse_mean', 'N/A'):.4f} ± {method_results.get('rmse_std', 0):.4f}</td>
                    <td class="{mae_class}">{method_results.get('mae_mean', 'N/A'):.4f} ± {method_results.get('mae_std', 0):.4f}</td>
                    <td>{method_results.get('nll_mean', 'N/A') if not np.isnan(method_results.get('nll_mean', np.nan)) else 'N/A'}</td>
                    <td>{method_results.get('ece_mean', 'N/A') if not np.isnan(method_results.get('ece_mean', np.nan)) else 'N/A'}</td>
                    <td>{method_results.get('coverage_90_mean', 'N/A') if not np.isnan(method_results.get('coverage_90_mean', np.nan)) else 'N/A'}</td>
                    <td>{method_results.get('training_time_mean', 'N/A'):.1f}</td>
                </tr>
                """
            
            html_content += "</table>"
        
        html_content += """
            <h2>Notes</h2>
            <ul>
                <li>Results are averaged over multiple random seeds</li>
                <li>NLL, ECE, and Coverage metrics are only available for PNO (uncertainty-aware method)</li>
                <li>Green highlighting indicates best performance for that metric</li>
                <li>Error bars show standard deviation across seeds</li>
            </ul>
        </body>
        </html>
        """
        
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        logger.info(f"HTML report saved to {output_path}")
    
    def plot_comparison_results(
        self,
        results: Dict[str, Dict[str, Dict[str, Any]]],
        metrics: List[str] = ['rmse_mean', 'mae_mean', 'nll_mean', 'ece_mean'],
        save_dir: Optional[str] = None,
    ) -> Dict[str, plt.Figure]:
        """Create comparison plots for benchmark results.
        
        Args:
            results: Nested results dictionary
            metrics: List of metrics to plot
            save_dir: Directory to save plots
            
        Returns:
            Dictionary of matplotlib figures
        """
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
        
        figures = {}
        
        # Prepare data for plotting
        plot_data = []
        for pde, pde_results in results.items():
            for method, method_results in pde_results.items():
                if not method_results:
                    continue
                
                row = {
                    'PDE': pde.replace('_', ' ').title(),
                    'Method': method,
                }
                row.update(method_results)
                plot_data.append(row)
        
        df = pd.DataFrame(plot_data)
        
        # Create plots for each metric
        for metric in metrics:
            if metric not in df.columns or df[metric].isna().all():
                continue
            
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Bar plot with error bars
            sns.barplot(
                data=df,
                x='PDE',
                y=metric,
                hue='Method',
                ax=ax,
                capsize=0.1,
            )
            
            ax.set_title(f'Comparison of {metric.replace("_", " ").title()}')
            ax.set_ylabel(metric.replace('_', ' ').title())
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            figures[metric] = fig
            
            if save_dir:
                fig.savefig(save_dir / f'{metric}_comparison.png', dpi=300, bbox_inches='tight')
        
        return figures