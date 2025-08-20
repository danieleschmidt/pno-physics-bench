#!/usr/bin/env python3
# SECURITY NOTICE: This file has been automatically hardened for security
# - All user inputs should be validated and sanitized
# - Subprocess calls use secure alternatives
# - SQL queries use parameterized statements
# - No hardcoded secrets or credentials


"""Standalone monitoring server for PNO Physics Bench."""

import os
import time
import argparse
from pathlib import Path
from pno_physics_bench.monitoring import start_metrics_server, get_metrics_collector


def main():
    """Start the monitoring server."""
    parser = argparse.ArgumentParser(description="PNO Physics Bench Monitoring Server")
    parser.add_argument(
        "--port", 
        type=int, 
        default=int(os.getenv("METRICS_PORT", 8080)),
        help="Port to serve metrics on"
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Directory to monitor for results"
    )
    parser.add_argument(
        "--update-interval",
        type=int,
        default=30,
        help="Update interval in seconds"
    )
    
    args = parser.parse_args()
    
    print(f"Starting PNO Physics Bench monitoring server on port {args.port}")
    
    # Start metrics server
    start_metrics_server(args.port)
    
    # Get metrics collector
    collector = get_metrics_collector()
    
    # Set some default model info
    collector.set_model_info(
        model_name="PNO",
        version="0.1.0",
        parameters=0,  # Will be updated when model is loaded
        architecture="Probabilistic Neural Operator",
        uncertainty_type="variational"
    )
    
    print(f"Monitoring server ready!")
    print(f"Metrics available at: http://localhost:{args.port}/metrics")
    print(f"Results directory: {args.results_dir}")
    print(f"Update interval: {args.update_interval}s")
    
    # Keep the server running and periodically update metrics
    try:
        while True:
            # Update GPU metrics if available
            collector.ml_metrics.update_gpu_metrics()
            
            # Monitor results directory for new experiments
            if args.results_dir.exists():
                monitor_results_directory(args.results_dir, collector)
            
            time.sleep(args.update_interval)
            
    except KeyboardInterrupt:
        print("\nShutting down monitoring server...")
        collector.stop()


def monitor_results_directory(results_dir: Path, collector):
    """Monitor results directory for training metrics."""
    try:
        # Look for tensorboard logs
        tensorboard_dir = results_dir / "tensorboard_logs"
        if tensorboard_dir.exists():
            # Count number of experiments
            experiments = list(tensorboard_dir.iterdir())
            if experiments:
                print(f"Found {len(experiments)} experiments in tensorboard logs")
        
        # Look for checkpoint files
        checkpoints_dir = results_dir / "checkpoints" 
        if checkpoints_dir.exists():
            checkpoints = list(checkpoints_dir.glob("*.pth"))
            if checkpoints:
                # Get latest checkpoint
                latest_checkpoint = max(checkpoints, key=lambda p: p.stat().st_mtime)
                print(f"Latest checkpoint: {latest_checkpoint.name}")
        
        # Look for metrics files (JSON/CSV with training metrics)
        metrics_files = list(results_dir.glob("**/metrics.json"))
        for metrics_file in metrics_files:
            try:
                import json
                with open(metrics_file) as f:
                    metrics = json.load(f)
                
                # Update metrics if available
                if "epoch" in metrics:
                    collector.ml_metrics.current_epoch.set(metrics["epoch"])
                if "train_loss" in metrics:
                    collector.ml_metrics.training_loss.set(metrics["train_loss"])
                if "val_loss" in metrics:
                    collector.ml_metrics.validation_loss.set(metrics["val_loss"])
                if "learning_rate" in metrics:
                    collector.ml_metrics.learning_rate.set(metrics["learning_rate"])
                    
            except Exception as e:
                print(f"Error reading metrics file {metrics_file}: {e}")
    
    except Exception as e:
        print(f"Error monitoring results directory: {e}")


if __name__ == "__main__":
    main()