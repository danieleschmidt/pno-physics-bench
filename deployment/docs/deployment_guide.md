# PNO Physics Bench - Production Deployment Guide

## Overview
This guide covers deploying the PNO Physics Bench system to production environments.

## Prerequisites
- Docker 20.x or higher
- Kubernetes 1.20+ (for container orchestration)
- Python 3.12+ (for local development)
- 8GB+ RAM per instance
- GPU support optional but recommended

## Quick Start

1. **Build the application**:
   ```bash
   cd deployment
   ./scripts/build.sh
   ```

2. **Deploy to Kubernetes**:
   ```bash
   ./scripts/deploy.sh production.json
   ```

3. **Monitor deployment**:
   ```bash
   ./scripts/monitor.sh
   ```

## Configuration

### Production Config
The production configuration is located at `configs/production.json`. Key settings:

- `model.device`: Set to "auto" for automatic GPU detection
- `training.distributed`: Enable for multi-GPU training
- `monitoring.enable_metrics`: Enable Prometheus metrics
- `scaling.auto_scaling`: Enable automatic worker scaling

### Environment Variables
- `CONFIG_PATH`: Path to configuration file
- `CUDA_VISIBLE_DEVICES`: GPU selection
- `PYTHONPATH`: Should include `/app/src`

## Monitoring

### Health Endpoints
- `http://service:8001/health` - Basic health check
- `http://service:8001/ready` - Readiness probe
- `http://service:8001/metrics` - Prometheus metrics

### Metrics
The system exposes the following metrics:
- Model inference latency
- Memory usage (GPU/CPU)
- Training loss and accuracy
- Request throughput
- Error rates

### Logs
Structured JSON logs are available via kubectl:
```bash
kubectl logs -l app=pno-physics-bench
```

## Scaling

### Horizontal Scaling
Increase replicas in the Kubernetes deployment:
```bash
kubectl scale deployment pno-physics-bench --replicas=4
```

### Vertical Scaling
Adjust resource limits in `kubernetes.yaml`:
```yaml
resources:
  limits:
    memory: "16Gi"
    cpu: "8000m"
```

### Auto-scaling
Enable Horizontal Pod Autoscaler:
```bash
kubectl autoscale deployment pno-physics-bench --cpu-percent=70 --min=2 --max=10
```

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce batch size or increase memory limits
2. **Slow inference**: Enable model compilation and optimization
3. **Training instability**: Check gradient clipping and learning rate
4. **GPU not detected**: Verify CUDA installation and device visibility

### Debug Commands
```bash
# Check pod logs
kubectl logs pno-physics-bench-xxx

# Get pod details
kubectl describe pod pno-physics-bench-xxx

# Execute into pod
kubectl exec -it pno-physics-bench-xxx -- /bin/bash

# Port forward for local access
kubectl port-forward service/pno-physics-bench-service 8000:8000
```

## Security

### Best Practices
- Use non-root containers
- Enable network policies
- Regularly update base images
- Monitor for vulnerabilities
- Use secrets for sensitive configuration

### Input Validation
All model inputs are validated for:
- Correct tensor dimensions
- Reasonable value ranges
- NaN/Inf detection
- Memory bounds checking

## Backup and Recovery

### Model Checkpoints
Models are automatically checkpointed during training to persistent storage.

### Configuration Backup
Store configurations in version control and use ConfigMaps for deployment.

## Performance Tuning

### Model Optimization
- Enable model compilation: `model.optimization.compile = true`
- Use mixed precision: `training.mixed_precision = true`
- Optimize batch sizes: `model.optimization.batch_size_optimization = true`

### System Optimization
- Use CPU affinity for worker processes
- Configure appropriate resource requests/limits
- Enable persistent volumes for model storage

## Support

For production support:
1. Check the monitoring dashboard
2. Review system logs
3. Run diagnostic scripts
4. Contact the development team with deployment details
