#!/usr/bin/env python3
"""Production deployment preparation and validation."""

import torch
import sys
import os
import json
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def prepare_deployment_artifacts():
    """Prepare all artifacts needed for production deployment."""
    print("\nPreparing deployment artifacts...")
    
    try:
        # Create deployment directory structure
        deployment_dir = Path("deployment")
        deployment_dir.mkdir(exist_ok=True)
        
        subdirs = ["models", "configs", "scripts", "docs", "monitoring"]
        for subdir in subdirs:
            (deployment_dir / subdir).mkdir(exist_ok=True)
        
        print(f"✓ Deployment directory structure created: {deployment_dir}")
        
        # Create model checkpoint
        from pno_physics_bench.models import ProbabilisticNeuralOperator
        
        model = ProbabilisticNeuralOperator(
            input_dim=3, 
            hidden_dim=32, 
            num_layers=3, 
            modes=16
        )
        
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'model_config': {
                'input_dim': 3,
                'hidden_dim': 32,
                'num_layers': 3,
                'modes': 16
            },
            'deployment_info': {
                'created_at': datetime.now().isoformat(),
                'pytorch_version': torch.__version__,
                'model_class': 'ProbabilisticNeuralOperator',
                'total_parameters': sum(p.numel() for p in model.parameters())
            }
        }
        
        checkpoint_path = deployment_dir / "models" / "production_model.pt"
        torch.save(checkpoint, checkpoint_path)
        print(f"✓ Model checkpoint saved: {checkpoint_path}")
        
        return True
        
    except Exception as e:
        print(f"✗ Deployment preparation failed: {e}")
        return False

def create_deployment_configs():
    """Create production configuration files."""
    print("\nCreating deployment configurations...")
    
    try:
        deployment_dir = Path("deployment")
        
        # Production configuration
        prod_config = {
            "environment": "production",
            "model": {
                "checkpoint_path": "models/production_model.pt",
                "device": "auto",
                "optimization": {
                    "compile": True,
                    "half_precision": False,
                    "batch_size_optimization": True
                }
            },
            "training": {
                "distributed": True,
                "gradient_clipping": 1.0,
                "mixed_precision": True,
                "checkpoint_interval": 100
            },
            "monitoring": {
                "enable_metrics": True,
                "metrics_port": 8000,
                "log_level": "INFO",
                "health_checks": True
            },
            "scaling": {
                "auto_scaling": True,
                "min_workers": 1,
                "max_workers": 8,
                "target_memory_gb": 8.0
            },
            "security": {
                "input_validation": True,
                "parameter_bounds_check": True,
                "memory_limits": True
            }
        }
        
        config_path = deployment_dir / "configs" / "production.json"
        with open(config_path, 'w') as f:
            json.dump(prod_config, f, indent=2)
        
        print(f"✓ Production config created: {config_path}")
        
        # Docker configuration
        dockerfile_content = """FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc g++ \\
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ src/
COPY deployment/ deployment/

# Set environment variables
ENV PYTHONPATH=/app/src
ENV CUDA_VISIBLE_DEVICES=""

# Expose ports
EXPOSE 8000 8001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s \\
  CMD python -c "import torch; import src.pno_physics_bench; print('OK')"

# Run application
CMD ["python", "-m", "pno_physics_bench.server"]
"""
        
        dockerfile_path = deployment_dir / "Dockerfile"
        dockerfile_path.write_text(dockerfile_content)
        print(f"✓ Dockerfile created: {dockerfile_path}")
        
        # Kubernetes deployment
        k8s_config = """apiVersion: apps/v1
kind: Deployment
metadata:
  name: pno-physics-bench
  labels:
    app: pno-physics-bench
spec:
  replicas: 2
  selector:
    matchLabels:
      app: pno-physics-bench
  template:
    metadata:
      labels:
        app: pno-physics-bench
    spec:
      containers:
      - name: pno-physics-bench
        image: pno-physics-bench:latest
        ports:
        - containerPort: 8000
        - containerPort: 8001
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "8Gi"
            cpu: "4000m"
        env:
        - name: CONFIG_PATH
          value: "/app/deployment/configs/production.json"
        livenessProbe:
          httpGet:
            path: /health
            port: 8001
          initialDelaySeconds: 60
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /ready
            port: 8001
          initialDelaySeconds: 30
          periodSeconds: 10
---
apiVersion: v1
kind: Service
metadata:
  name: pno-physics-bench-service
spec:
  selector:
    app: pno-physics-bench
  ports:
  - name: api
    port: 8000
    targetPort: 8000
  - name: metrics
    port: 8001
    targetPort: 8001
  type: LoadBalancer
"""
        
        k8s_path = deployment_dir / "configs" / "kubernetes.yaml"
        k8s_path.write_text(k8s_config)
        print(f"✓ Kubernetes config created: {k8s_path}")
        
        return True
        
    except Exception as e:
        print(f"✗ Config creation failed: {e}")
        return False

def create_deployment_scripts():
    """Create deployment and management scripts."""
    print("\nCreating deployment scripts...")
    
    try:
        deployment_dir = Path("deployment")
        scripts_dir = deployment_dir / "scripts"
        
        # Build script
        build_script = """#!/bin/bash
set -e

echo "Building PNO Physics Bench for production..."

# Build Docker image
docker build -t pno-physics-bench:latest .

# Tag for registry
docker tag pno-physics-bench:latest registry.example.com/pno-physics-bench:latest

# Run tests in container
docker run --rm pno-physics-bench:latest python test_basic_functionality.py
docker run --rm pno-physics-bench:latest python test_robust_functionality.py
docker run --rm pno-physics-bench:latest python test_scaling_functionality.py

echo "Build completed successfully!"
"""
        
        build_path = scripts_dir / "build.sh"
        build_path.write_text(build_script)
        build_path.chmod(0o755)
        print(f"✓ Build script created: {build_path}")
        
        # Deploy script
        deploy_script = """#!/bin/bash
set -e

CONFIG_FILE=${1:-production.json}
NAMESPACE=${2:-default}

echo "Deploying PNO Physics Bench with config: $CONFIG_FILE"

# Apply Kubernetes configuration
kubectl apply -f configs/kubernetes.yaml -n $NAMESPACE

# Wait for deployment to be ready
kubectl wait --for=condition=available --timeout=300s deployment/pno-physics-bench -n $NAMESPACE

# Verify deployment
kubectl get pods -l app=pno-physics-bench -n $NAMESPACE
kubectl get services -l app=pno-physics-bench -n $NAMESPACE

echo "Deployment completed successfully!"
"""
        
        deploy_path = scripts_dir / "deploy.sh"
        deploy_path.write_text(deploy_script)
        deploy_path.chmod(0o755)
        print(f"✓ Deploy script created: {deploy_path}")
        
        # Monitoring script
        monitor_script = """#!/bin/bash

NAMESPACE=${1:-default}

echo "PNO Physics Bench - Monitoring Dashboard"
echo "========================================"

# Pod status
echo "Pod Status:"
kubectl get pods -l app=pno-physics-bench -n $NAMESPACE

# Service status
echo -e "\\nService Status:"
kubectl get services -l app=pno-physics-bench -n $NAMESPACE

# Resource usage
echo -e "\\nResource Usage:"
kubectl top pods -l app=pno-physics-bench -n $NAMESPACE 2>/dev/null || echo "Metrics server not available"

# Recent logs
echo -e "\\nRecent Logs:"
kubectl logs -l app=pno-physics-bench --tail=10 -n $NAMESPACE

# Health check
echo -e "\\nHealth Check:"
SERVICE_IP=$(kubectl get service pno-physics-bench-service -n $NAMESPACE -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "localhost")
curl -s http://$SERVICE_IP:8001/health || echo "Health check unavailable"
"""
        
        monitor_path = scripts_dir / "monitor.sh"
        monitor_path.write_text(monitor_script)
        monitor_path.chmod(0o755)
        print(f"✓ Monitor script created: {monitor_path}")
        
        return True
        
    except Exception as e:
        print(f"✗ Script creation failed: {e}")
        return False

def create_production_documentation():
    """Create production deployment documentation."""
    print("\nCreating production documentation...")
    
    try:
        deployment_dir = Path("deployment")
        docs_dir = deployment_dir / "docs"
        
        # Deployment guide
        deployment_guide = """# PNO Physics Bench - Production Deployment Guide

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
"""
        
        guide_path = docs_dir / "deployment_guide.md"
        guide_path.write_text(deployment_guide)
        print(f"✓ Deployment guide created: {guide_path}")
        
        # API documentation
        api_docs = """# PNO Physics Bench - API Reference

## Model Inference API

### POST /predict
Perform model inference with uncertainty quantification.

**Request Body:**
```json
{
  "input": "tensor data as nested array",
  "num_samples": 10,
  "return_uncertainty": true
}
```

**Response:**
```json
{
  "prediction": "tensor data",
  "uncertainty": "uncertainty tensor",
  "inference_time": 0.045,
  "model_info": {
    "version": "1.0.0",
    "parameters": 1250000
  }
}
```

## Training API

### POST /train
Start model training with specified configuration.

### GET /training/status
Get current training status and metrics.

### POST /training/stop
Stop current training job.

## Management API

### GET /health
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-08-08T20:00:00Z",
  "checks": {
    "model_loaded": true,
    "gpu_available": false,
    "memory_usage": 0.65
  }
}
```

### GET /metrics
Prometheus metrics endpoint.

### GET /info
System information.

**Response:**
```json
{
  "version": "1.0.0",
  "pytorch_version": "2.1.0",
  "cuda_version": null,
  "model_info": {
    "type": "ProbabilisticNeuralOperator",
    "parameters": 1250000,
    "last_updated": "2025-08-08T20:00:00Z"
  }
}
```

## Error Codes

- `400` - Bad Request (invalid input)
- `422` - Unprocessable Entity (model error)
- `500` - Internal Server Error
- `503` - Service Unavailable (model not loaded)
"""
        
        api_path = docs_dir / "api_reference.md"
        api_path.write_text(api_docs)
        print(f"✓ API documentation created: {api_path}")
        
        return True
        
    except Exception as e:
        print(f"✗ Documentation creation failed: {e}")
        return False

def validate_deployment():
    """Validate deployment readiness."""
    print("\nValidating deployment readiness...")
    
    try:
        deployment_dir = Path("deployment")
        
        # Check all required files exist
        required_files = [
            "models/production_model.pt",
            "configs/production.json",
            "configs/kubernetes.yaml",
            "scripts/build.sh",
            "scripts/deploy.sh",
            "scripts/monitor.sh",
            "docs/deployment_guide.md",
            "docs/api_reference.md",
            "Dockerfile"
        ]
        
        missing_files = []
        for file_path in required_files:
            full_path = deployment_dir / file_path
            if not full_path.exists():
                missing_files.append(str(file_path))
        
        if missing_files:
            print(f"✗ Missing deployment files: {missing_files}")
            return False
        
        print("✓ All deployment files present")
        
        # Validate model checkpoint
        checkpoint_path = deployment_dir / "models/production_model.pt"
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            required_keys = ['model_state_dict', 'model_config', 'deployment_info']
            
            for key in required_keys:
                if key not in checkpoint:
                    print(f"✗ Missing checkpoint key: {key}")
                    return False
            
            print(f"✓ Model checkpoint valid: {checkpoint['deployment_info']['total_parameters']} parameters")
            
        except Exception as e:
            print(f"✗ Invalid model checkpoint: {e}")
            return False
        
        # Validate configuration
        config_path = deployment_dir / "configs/production.json"
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            required_sections = ['environment', 'model', 'monitoring', 'scaling']
            for section in required_sections:
                if section not in config:
                    print(f"✗ Missing config section: {section}")
                    return False
            
            print("✓ Configuration valid")
            
        except Exception as e:
            print(f"✗ Invalid configuration: {e}")
            return False
        
        # Check script permissions
        scripts = ['build.sh', 'deploy.sh', 'monitor.sh']
        for script in scripts:
            script_path = deployment_dir / "scripts" / script
            if not os.access(script_path, os.X_OK):
                print(f"✗ Script not executable: {script}")
                return False
        
        print("✓ All scripts executable")
        
        print("✓ Deployment validation completed successfully")
        return True
        
    except Exception as e:
        print(f"✗ Deployment validation failed: {e}")
        return False

def main():
    """Complete production deployment preparation."""
    print("=" * 70)
    print("PNO PHYSICS BENCH - PRODUCTION DEPLOYMENT PREPARATION")
    print("=" * 70)
    
    tasks = [
        ("Deployment Artifacts", prepare_deployment_artifacts),
        ("Configuration Files", create_deployment_configs),
        ("Deployment Scripts", create_deployment_scripts),
        ("Documentation", create_production_documentation),
        ("Deployment Validation", validate_deployment),
    ]
    
    passed = 0
    total = len(tasks)
    
    for task_name, task_func in tasks:
        print(f"\n{'='*50}")
        print(f"TASK: {task_name.upper()}")
        print('='*50)
        
        try:
            if task_func():
                passed += 1
                print(f"✅ COMPLETED: {task_name}")
            else:
                print(f"❌ FAILED: {task_name}")
        except Exception as e:
            print(f"💥 CRASHED: {task_name} - {e}")
    
    print("\n" + "=" * 70)
    print("PRODUCTION DEPLOYMENT SUMMARY")
    print("=" * 70)
    
    if passed == total:
        print("🎉 PRODUCTION DEPLOYMENT READY!")
        print("🚀 All deployment artifacts created successfully")
        print(f"📦 Deployment package: {Path('deployment').absolute()}")
        
        print("\nNext Steps:")
        print("1. Review deployment configurations")
        print("2. Run: cd deployment && ./scripts/build.sh")
        print("3. Deploy: ./scripts/deploy.sh production.json")
        print("4. Monitor: ./scripts/monitor.sh")
        
    else:
        print(f"⚠️  DEPLOYMENT PREPARATION ISSUES: {passed}/{total} tasks completed")
        print("🔧 Some deployment artifacts need attention")
    
    print("=" * 70)
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)