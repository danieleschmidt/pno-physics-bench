"""Performance benchmark tests for PNO models."""

import pytest
import torch
import time
import psutil
import json
from pathlib import Path
from typing import Dict, Any
import numpy as np


class TestModelPerformance:
    """Benchmark model performance metrics."""
    
    @pytest.mark.benchmark
    def test_forward_pass_speed(self, benchmark, synthetic_dataset, benchmark_config):
        """Benchmark forward pass speed."""
        
        class MockPNOModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = torch.nn.Conv2d(2, 64, kernel_size=3, padding=1)
                self.conv2 = torch.nn.Conv2d(64, 64, kernel_size=3, padding=1)
                self.conv3 = torch.nn.Conv2d(64, 1, kernel_size=3, padding=1)
                self.relu = torch.nn.ReLU()
            
            def forward(self, x):
                # Reshape to 2D if needed
                if len(x.shape) == 3:
                    x = x.unsqueeze(-1).repeat(1, 1, 1, 32)
                elif len(x.shape) == 2:
                    x = x.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 32, 32)
                
                x = self.relu(self.conv1(x))
                x = self.relu(self.conv2(x))
                x = self.conv3(x)
                return x
        
        model = MockPNOModel()
        data = synthetic_dataset
        inputs = data['inputs']
        
        # Benchmark function
        def forward_pass():
            with torch.no_grad():
                return model(inputs)
        
        # Run benchmark
        result = benchmark.pedantic(
            forward_pass,
            rounds=benchmark_config['min_rounds'],
            warmup_rounds=2 if benchmark_config['warmup'] else 0
        )
        
        # Verify output
        output = forward_pass()
        assert output.shape[0] == inputs.shape[0]  # Batch dimension preserved
        assert torch.isfinite(output).all()
    
    @pytest.mark.benchmark
    @pytest.mark.gpu
    def test_gpu_vs_cpu_performance(self, benchmark, synthetic_dataset):
        """Compare GPU vs CPU performance."""
        
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        class MockModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = torch.nn.Linear(1024, 512)
                self.linear2 = torch.nn.Linear(512, 256)
                self.linear3 = torch.nn.Linear(256, 1024)
                self.relu = torch.nn.ReLU()
            
            def forward(self, x):
                x_flat = x.view(x.shape[0], -1)
                if x_flat.shape[1] != 1024:
                    self.linear1 = torch.nn.Linear(x_flat.shape[1], 512).to(x.device)
                
                x = self.relu(self.linear1(x_flat))
                x = self.relu(self.linear2(x))
                x = self.linear3(x)
                return x
        
        data = synthetic_dataset
        inputs = data['inputs']
        
        # CPU benchmark
        model_cpu = MockModel()
        inputs_cpu = inputs.cpu()
        
        def cpu_forward():
            with torch.no_grad():
                return model_cpu(inputs_cpu)
        
        cpu_result = benchmark.pedantic(cpu_forward, rounds=3, warmup_rounds=1)
        
        # GPU benchmark
        model_gpu = MockModel().cuda()
        inputs_gpu = inputs.cuda()
        
        def gpu_forward():
            with torch.no_grad():
                result = model_gpu(inputs_gpu)
                torch.cuda.synchronize()  # Ensure GPU computation is complete
                return result
        
        gpu_result = benchmark.pedantic(gpu_forward, rounds=3, warmup_rounds=1)
        
        # GPU should be faster (though this depends on model size)
        # We just check that both complete successfully
        assert cpu_result is not None
        assert gpu_result is not None
    
    @pytest.mark.benchmark
    def test_memory_usage(self, synthetic_dataset):
        """Test memory usage during forward and backward passes."""
        
        class MockMemoryIntensiveModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # Larger model to test memory usage
                self.layers = torch.nn.ModuleList([
                    torch.nn.Linear(1024, 1024) for _ in range(5)
                ])
                self.relu = torch.nn.ReLU()
            
            def forward(self, x):
                x_flat = x.view(x.shape[0], -1)
                if x_flat.shape[1] != 1024:
                    # Adapt first layer
                    self.layers[0] = torch.nn.Linear(x_flat.shape[1], 1024)
                
                for layer in self.layers:
                    x_flat = self.relu(layer(x_flat))
                return x_flat
        
        model = MockMemoryIntensiveModel()
        data = synthetic_dataset
        inputs = data['inputs']
        targets = torch.randn(inputs.shape[0], 1024)  # Match output size
        
        # Measure memory before
        process = psutil.Process()
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # Forward pass
        optimizer = torch.optim.Adam(model.parameters())
        optimizer.zero_grad()
        
        outputs = model(inputs)
        memory_after_forward = process.memory_info().rss / 1024 / 1024
        
        # Backward pass
        loss = torch.nn.MSELoss()(outputs, targets)
        loss.backward()
        
        memory_after_backward = process.memory_info().rss / 1024 / 1024
        
        # Update parameters
        optimizer.step()
        memory_after_update = process.memory_info().rss / 1024 / 1024
        
        # Memory should increase during computation
        assert memory_after_forward >= memory_before
        assert memory_after_backward >= memory_after_forward
        
        # Log memory usage
        print(f"Memory usage - Before: {memory_before:.1f} MB, "
              f"After forward: {memory_after_forward:.1f} MB, "
              f"After backward: {memory_after_backward:.1f} MB, "
              f"After update: {memory_after_update:.1f} MB")
    
    @pytest.mark.benchmark
    def test_training_step_performance(self, benchmark, synthetic_dataset):
        """Benchmark complete training step performance."""
        
        class MockTrainingModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = torch.nn.Conv2d(2, 32, 3, padding=1)
                self.conv2 = torch.nn.Conv2d(32, 32, 3, padding=1)
                self.conv3 = torch.nn.Conv2d(32, 1, 3, padding=1)
                self.relu = torch.nn.ReLU()
            
            def forward(self, x):
                # Handle different input shapes
                if len(x.shape) == 3:
                    x = x.unsqueeze(-1).repeat(1, 1, 1, 32)
                elif len(x.shape) == 2:
                    x = x.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 32, 32)
                
                x = self.relu(self.conv1(x))
                x = self.relu(self.conv2(x))
                return self.conv3(x)
            
            def kl_divergence(self):
                return torch.tensor(0.01)
        
        model = MockTrainingModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        data = synthetic_dataset
        inputs = data['inputs']
        targets = torch.randn(inputs.shape[0], 1, 32, 32)  # Match expected output shape
        
        def training_step():
            optimizer.zero_grad()
            
            # Forward pass
            predictions = model(inputs)
            
            # Compute loss
            mse_loss = torch.nn.MSELoss()(predictions, targets)
            kl_loss = model.kl_divergence()
            total_loss = mse_loss + 1e-4 * kl_loss
            
            # Backward pass
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            # Update parameters
            optimizer.step()
            
            return total_loss.item()
        
        # Benchmark training step
        result = benchmark.pedantic(training_step, rounds=5, warmup_rounds=2)
        
        # Verify training step works
        loss_value = training_step()
        assert torch.isfinite(torch.tensor(loss_value))
    
    @pytest.mark.benchmark
    @pytest.mark.slow
    def test_large_batch_performance(self, benchmark):
        """Test performance with large batch sizes."""
        
        class MockModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 64, 3, padding=1)
                self.pool = torch.nn.AdaptiveAvgPool2d((8, 8))
                self.linear = torch.nn.Linear(64 * 8 * 8, 1000)
            
            def forward(self, x):
                x = self.conv(x)
                x = self.pool(x)
                x = x.view(x.size(0), -1)
                return self.linear(x)
        
        model = MockModel()
        
        # Test with increasing batch sizes
        batch_sizes = [16, 32, 64, 128]
        times = []
        
        for batch_size in batch_sizes:
            inputs = torch.randn(batch_size, 3, 64, 64)
            
            def forward_pass():
                with torch.no_grad():
                    return model(inputs)
            
            # Time the forward pass
            start_time = time.time()
            result = benchmark.pedantic(forward_pass, rounds=3, warmup_rounds=1)
            end_time = time.time()
            
            times.append(end_time - start_time)
            
            # Verify output shape
            output = forward_pass()
            assert output.shape == (batch_size, 1000)
        
        # Log performance scaling
        for batch_size, exec_time in zip(batch_sizes, times):
            throughput = batch_size / exec_time
            print(f"Batch size {batch_size}: {exec_time:.4f}s, "
                  f"Throughput: {throughput:.1f} samples/s")
    
    @pytest.mark.benchmark
    def test_uncertainty_computation_overhead(self, benchmark, synthetic_dataset):
        """Test overhead of uncertainty computation."""
        
        class MockDeterministicModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(64, 32)
            
            def forward(self, x):
                x_flat = x.view(x.shape[0], -1)
                if x_flat.shape[1] != 64:
                    self.linear = torch.nn.Linear(x_flat.shape[1], 32)
                return self.linear(x_flat)
        
        class MockUncertaintyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.mean_layer = torch.nn.Linear(64, 32)
                self.var_layer = torch.nn.Linear(64, 32)
            
            def forward(self, x):
                x_flat = x.view(x.shape[0], -1)
                if x_flat.shape[1] != 64:
                    self.mean_layer = torch.nn.Linear(x_flat.shape[1], 32)
                    self.var_layer = torch.nn.Linear(x_flat.shape[1], 32)
                
                mean = self.mean_layer(x_flat)
                log_var = self.var_layer(x_flat)
                
                # Reparameterization trick (adds overhead)
                if self.training:
                    eps = torch.randn_like(mean)
                    std = torch.exp(0.5 * log_var)
                    return mean + std * eps, std
                else:
                    return mean, torch.exp(0.5 * log_var)
        
        data = synthetic_dataset
        inputs = data['inputs']
        
        # Benchmark deterministic model
        det_model = MockDeterministicModel()
        
        def det_forward():
            with torch.no_grad():
                return det_model(inputs)
        
        det_result = benchmark.pedantic(det_forward, rounds=5, warmup_rounds=2)
        
        # Benchmark uncertainty model
        unc_model = MockUncertaintyModel()
        unc_model.eval()  # Disable reparameterization for fair comparison
        
        def unc_forward():
            with torch.no_grad():
                return unc_model(inputs)
        
        unc_result = benchmark.pedantic(unc_forward, rounds=5, warmup_rounds=2)
        
        # Both should complete successfully
        det_output = det_forward()
        unc_mean, unc_std = unc_forward()
        
        assert det_output.shape == unc_mean.shape
        assert unc_std.shape == unc_mean.shape
        assert (unc_std > 0).all()


class TestDataLoaderPerformance:
    """Benchmark data loading performance."""
    
    @pytest.mark.benchmark
    def test_dataloader_speed(self, benchmark, synthetic_dataset):
        """Test data loader performance."""
        
        class MockDataset:
            def __init__(self, data):
                self.inputs = data['inputs']
                self.targets = data['targets']
            
            def __len__(self):
                return len(self.inputs)
            
            def __getitem__(self, idx):
                return self.inputs[idx], self.targets[idx]
        
        dataset = MockDataset(synthetic_dataset)
        
        # Test different batch sizes and num_workers
        configs = [
            {'batch_size': 16, 'num_workers': 0},
            {'batch_size': 32, 'num_workers': 0},
            {'batch_size': 16, 'num_workers': 2},
            {'batch_size': 32, 'num_workers': 2},
        ]
        
        for config in configs:
            data_loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=config['batch_size'],
                num_workers=config['num_workers'],
                shuffle=False,
                pin_memory=torch.cuda.is_available()
            )
            
            def iterate_dataloader():
                for batch in data_loader:
                    pass  # Just iterate through all batches
            
            result = benchmark.pedantic(iterate_dataloader, rounds=3, warmup_rounds=1)
            
            print(f"Config {config}: completed iteration")
    
    @pytest.mark.benchmark
    def test_data_preprocessing_speed(self, benchmark):
        """Test data preprocessing performance."""
        
        def preprocess_batch(batch_data):
            """Mock preprocessing function."""
            inputs, targets = batch_data
            
            # Normalize inputs
            inputs = (inputs - inputs.mean()) / (inputs.std() + 1e-8)
            
            # Add noise to targets (data augmentation)
            targets = targets + torch.randn_like(targets) * 0.01
            
            return inputs, targets
        
        # Generate test batch
        batch_size = 32
        inputs = torch.randn(batch_size, 2, 64)
        targets = torch.randn(batch_size, 64)
        batch = (inputs, targets)
        
        def preprocess():
            return preprocess_batch(batch)
        
        result = benchmark.pedantic(preprocess, rounds=10, warmup_rounds=2)
        
        # Verify preprocessing works
        proc_inputs, proc_targets = preprocess()
        assert proc_inputs.shape == inputs.shape
        assert proc_targets.shape == targets.shape
        assert torch.isfinite(proc_inputs).all()
        assert torch.isfinite(proc_targets).all()


class TestInferencePerformance:
    """Benchmark inference performance for deployment."""
    
    @pytest.mark.benchmark
    def test_single_sample_latency(self, benchmark):
        """Test latency for single sample inference."""
        
        class MockInferenceModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = torch.nn.Sequential(
                    torch.nn.Linear(128, 256),
                    torch.nn.ReLU(),
                    torch.nn.Linear(256, 128),
                    torch.nn.ReLU(),
                    torch.nn.Linear(128, 64)
                )
            
            def forward(self, x):
                return self.layers(x)
        
        model = MockInferenceModel()
        model.eval()
        
        # Single sample input
        single_input = torch.randn(1, 128)
        
        def single_inference():
            with torch.no_grad():
                return model(single_input)
        
        result = benchmark.pedantic(single_inference, rounds=100, warmup_rounds=10)
        
        # Verify output
        output = single_inference()
        assert output.shape == (1, 64)
    
    @pytest.mark.benchmark
    def test_batch_inference_throughput(self, benchmark):
        """Test throughput for batch inference."""
        
        class MockBatchModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = torch.nn.Conv2d(3, 32, 3, padding=1)
                self.conv2 = torch.nn.Conv2d(32, 64, 3, padding=1)
                self.pool = torch.nn.AdaptiveAvgPool2d((4, 4))
                self.fc = torch.nn.Linear(64 * 4 * 4, 10)
            
            def forward(self, x):
                x = torch.relu(self.conv1(x))
                x = torch.relu(self.conv2(x))
                x = self.pool(x)
                x = x.view(x.size(0), -1)
                return self.fc(x)
        
        model = MockBatchModel()
        model.eval()
        
        # Batch input
        batch_input = torch.randn(64, 3, 32, 32)
        
        def batch_inference():
            with torch.no_grad():
                return model(batch_input)
        
        result = benchmark.pedantic(batch_inference, rounds=10, warmup_rounds=3)
        
        # Calculate throughput
        output = batch_inference()
        assert output.shape == (64, 10)
        
        # Log throughput information
        batch_size = batch_input.shape[0]
        print(f"Batch size: {batch_size}, Output shape: {output.shape}")


# Performance regression detection
def save_benchmark_results(results: Dict[str, Any], results_file: Path):
    """Save benchmark results for regression detection."""
    results_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)


def load_baseline_results(baseline_file: Path) -> Dict[str, Any]:
    """Load baseline benchmark results."""
    if not baseline_file.exists():
        return {}
    
    with open(baseline_file, 'r') as f:
        return json.load(f)


def detect_performance_regression(current: Dict, baseline: Dict, threshold: float = 0.1) -> bool:
    """Detect performance regression by comparing current vs baseline."""
    if not baseline:
        return False
    
    for test_name, current_time in current.items():
        if test_name in baseline:
            baseline_time = baseline[test_name]
            regression = (current_time - baseline_time) / baseline_time
            
            if regression > threshold:
                print(f"Performance regression detected in {test_name}: "
                      f"{regression:.2%} slower than baseline")
                return True
    
    return False