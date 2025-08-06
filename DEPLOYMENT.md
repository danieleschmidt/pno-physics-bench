# 🚀 PNO Physics Bench - Production Deployment Guide

This guide provides comprehensive instructions for deploying PNO Physics Bench in production environments with scaling, monitoring, and reliability features.

## 📋 Prerequisites

### System Requirements
- **Python**: 3.9+ (recommended: 3.11)
- **Memory**: Minimum 8GB RAM, Recommended 32GB+
- **GPU**: Optional but recommended (NVIDIA with CUDA 11.8+)
- **Storage**: 50GB+ available disk space
- **Network**: High-bandwidth for distributed training

### Dependencies
```bash
# Core ML dependencies
pip install torch>=2.0.0 torchvision torchaudio
pip install numpy>=1.21.0 scipy>=1.7.0

# Scientific computing
pip install matplotlib>=3.5.0 seaborn>=0.11.0 h5py>=3.6.0

# Utilities and configuration
pip install tqdm>=4.62.0 omegaconf>=2.1.0 hydra-core>=1.1.0

# Experiment tracking
pip install wandb>=0.12.0 tensorboard>=2.8.0

# Optional: JAX backend
pip install jax>=0.4.0 jaxlib>=0.4.0 optax>=0.1.0
```

## 🏗️ Installation

### 1. Standard Installation
```bash
# Clone repository
git clone https://github.com/yourusername/pno-physics-bench.git
cd pno-physics-bench

# Install in development mode
pip install -e .

# Verify installation
python -c "import pno_physics_bench; print('Installation successful!')"
```

### 2. Docker Installation
```bash
# Build Docker image
docker build -t pno-physics-bench .

# Run container
docker run --gpus all -it pno-physics-bench
```

### 3. Kubernetes Deployment
```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: pno-training
spec:
  replicas: 3
  selector:
    matchLabels:
      app: pno-training
  template:
    metadata:
      labels:
        app: pno-training
    spec:
      containers:
      - name: pno-trainer
        image: pno-physics-bench:latest
        resources:
          requests:
            nvidia.com/gpu: 1
            memory: "16Gi"
            cpu: "4"
          limits:
            nvidia.com/gpu: 1
            memory: "32Gi"
            cpu: "8"
```

## ⚡ Performance Optimization

### 1. Enable Performance Features
```python
from pno_physics_bench.optimization_engine import AdaptivePerformanceOptimizer

# Initialize performance optimizer
optimizer = AdaptivePerformanceOptimizer(
    model=your_model,
    enable_mixed_precision=True,
    enable_compilation=True,
    max_cache_size=1000
)

# Optimize for inference
optimizer.optimize_for_inference()

# Use cached predictions
mean, std = optimizer.cached_predict(input_tensor)
```

### 2. Dynamic Batch Sizing
```python
from pno_physics_bench.optimization_engine import DynamicBatchSizer

batch_sizer = DynamicBatchSizer(
    initial_batch_size=32,
    max_batch_size=256,
    memory_threshold=0.8
)

# In training loop
current_batch_size = batch_sizer.get_batch_size()
try:
    # Training step
    loss = train_step(batch_size=current_batch_size)
    batch_sizer.report_success()
except torch.cuda.OutOfMemoryError:
    batch_sizer.report_oom()
```

## 🌐 Distributed Training

### 1. Single-Node Multi-GPU
```python
from pno_physics_bench.distributed_training import AutoScalingTrainer

def train_function(rank, world_size, local_rank, model, dataset):
    # Your training logic here
    trainer = PNOTrainer(model, device=f"cuda:{local_rank}")
    return trainer.fit(dataset)

# Auto-scale based on available GPUs
auto_trainer = AutoScalingTrainer(
    PNOTrainer,
    scaling_strategy="gpu_count"
)

auto_trainer.launch_distributed_training(train_function, model, dataset)
```

### 2. Multi-Node Training
```bash
# Node 0 (master)
python -m torch.distributed.launch \
    --nproc_per_node=8 \
    --nnodes=4 \
    --node_rank=0 \
    --master_addr="192.168.1.100" \
    --master_port=12345 \
    train_distributed.py

# Node 1
python -m torch.distributed.launch \
    --nproc_per_node=8 \
    --nnodes=4 \
    --node_rank=1 \
    --master_addr="192.168.1.100" \
    --master_port=12345 \
    train_distributed.py
```

## 📊 Monitoring and Observability

### 1. Advanced Monitoring Setup
```python
from pno_physics_bench.monitoring_advanced import TrainingMonitor

monitor = TrainingMonitor(
    log_dir="./production_logs",
    memory_threshold=0.85,
    gpu_memory_threshold=0.9
)

with monitor.training_session("production_training"):
    # Your training code here
    for epoch in range(epochs):
        metrics = train_epoch()
        monitor.log_metrics(epoch, metrics)
```

### 2. Prometheus Monitoring
```yaml
# prometheus-config.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'pno-training'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: /metrics
    scrape_interval: 10s
```

### 3. Grafana Dashboard
Import the provided dashboard from `monitoring/grafana-dashboards/pno/ml-training-dashboard.json`

Key metrics tracked:
- Training loss and validation accuracy
- GPU memory usage and utilization
- Batch processing times
- Model uncertainty calibration metrics
- System resource usage

## 🔧 Configuration Management

### 1. Hydra Configuration
```yaml
# config/train_config.yaml
model:
  type: ProbabilisticNeuralOperator
  input_dim: 3
  hidden_dim: 256
  num_layers: 4
  modes: 20

dataset:
  name: navier_stokes_2d
  resolution: 128
  num_samples: 10000
  normalize: true

training:
  batch_size: 32
  learning_rate: 1e-3
  epochs: 100
  mixed_precision: true
  
monitoring:
  wandb_project: pno-production
  log_interval: 10
  checkpoint_interval: 25
```

### 2. Environment Variables
```bash
# Production environment
export PNO_ENV=production
export PNO_LOG_LEVEL=INFO
export PNO_WANDB_KEY=your_wandb_key
export PNO_CHECKPOINT_DIR=/shared/checkpoints
export PNO_DATA_DIR=/shared/datasets
```

## 🚨 Error Handling and Recovery

### 1. Robust Training Wrapper
```python
from pno_physics_bench.monitoring_advanced import RobustTrainingWrapper

wrapper = RobustTrainingWrapper(
    trainer=your_trainer,
    max_retries=3,
    save_interval=10,
    checkpoint_dir="./resilient_checkpoints"
)

# Automatically handles failures and retries
history = wrapper.fit_robust(
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=100
)
```

### 2. Health Checks
```python
# Health check endpoint
@app.route("/health")
def health_check():
    health_status = monitor.check_system_health()
    
    if health_status["memory_percent"] < 90:
        return {"status": "healthy", "details": health_status}
    else:
        return {"status": "unhealthy", "details": health_status}, 503
```

## 📈 Scaling Patterns

### 1. Horizontal Scaling
```python
from pno_physics_bench.distributed_training import ElasticTraining

elastic_trainer = ElasticTraining(
    trainer=base_trainer,
    checkpoint_interval=5,
    health_check_interval=30
)

with elastic_trainer.elastic_training_session():
    # Training that adapts to resource changes
    trainer.fit(train_loader, val_loader)
```

### 2. Auto-scaling with Kubernetes HPA
```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: pno-training-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: pno-training
  minReplicas: 1
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

## 🔒 Security and Compliance

### 1. Model Security
- Encrypt model checkpoints at rest
- Use secure channels for distributed training
- Implement access controls for model artifacts
- Regular security audits of dependencies

### 2. Data Protection
- Implement data encryption in transit and at rest
- Follow GDPR/CCPA compliance for sensitive data
- Audit trail for all data access
- Anonymization of training data when possible

## 🌍 Multi-Region Deployment

### 1. Global Load Balancing
```yaml
# Global load balancer configuration
apiVersion: v1
kind: Service
metadata:
  name: pno-inference-global
  annotations:
    cloud.google.com/global-load-balancer: "true"
spec:
  type: LoadBalancer
  ports:
  - port: 80
    targetPort: 8080
  selector:
    app: pno-inference
```

### 2. Data Locality
- Deploy training clusters close to data sources
- Use regional data replication strategies
- Implement smart routing based on data locality
- Monitor cross-region transfer costs

## 📊 Performance Benchmarks

### Expected Performance Metrics
- **Training Throughput**: 100-1000 samples/sec (depending on resolution and hardware)
- **Inference Latency**: <100ms for 64x64 resolution
- **Memory Usage**: 4-16GB GPU memory for training
- **Uncertainty Calibration**: ECE < 0.05 for well-calibrated models

### Optimization Targets
- 95th percentile latency < 200ms
- 99.9% availability
- Memory efficiency: <20GB per training job
- Cost optimization: <$0.10 per training sample

## 🔧 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size using DynamicBatchSizer
   - Enable gradient checkpointing
   - Use mixed precision training

2. **Poor Uncertainty Calibration**
   - Increase KL divergence weight
   - Use calibration loss during training
   - Validate on held-out calibration set

3. **Slow Training**
   - Enable model compilation with PyTorch 2.0
   - Use DataParallel or DistributedDataParallel
   - Optimize data loading with multiple workers

4. **Gradient Instability**
   - Use gradient clipping
   - Monitor gradients with GradientMonitor
   - Adjust learning rate schedule

## 📚 Additional Resources

- [Model Architecture Guide](docs/tutorials/01_pno_intro.md)
- [Training Best Practices](docs/tutorials/03_training.md)
- [API Reference](docs/api_reference.md)
- [Performance Tuning Guide](docs/performance_tuning.md)
- [Troubleshooting Guide](docs/troubleshooting.md)

## 📞 Support

For production deployment support:
- Create an issue on GitHub
- Check the troubleshooting documentation
- Join our Discord community
- Contact enterprise support for commercial deployments

---

**Last Updated**: February 2025  
**Version**: 0.1.0  
**Status**: Production Ready 🚀