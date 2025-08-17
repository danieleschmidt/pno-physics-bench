"""
Production Deployment Manifest for Autonomous SDLC Implementation.

This module provides comprehensive production deployment capabilities
including Docker containers, Kubernetes manifests, monitoring setup,
and automated deployment scripts.
"""

import os
import json
# import yaml  # Not available in environment
from typing import Dict, List, Any
from dataclasses import dataclass, asdict


@dataclass
class DeploymentConfig:
    """Production deployment configuration."""
    environment: str = "production"
    replicas: int = 3
    cpu_limit: str = "2000m"
    memory_limit: str = "4Gi"
    gpu_limit: int = 1
    storage_size: str = "20Gi"
    redis_enabled: bool = True
    monitoring_enabled: bool = True
    security_enabled: bool = True
    auto_scaling: bool = True
    max_replicas: int = 10
    min_replicas: int = 2


class ProductionDeploymentGenerator:
    """Generator for production deployment artifacts."""
    
    def __init__(self, config: DeploymentConfig):
        self.config = config
        
    def generate_dockerfile(self) -> str:
        """Generate optimized production Dockerfile."""
        return """# Production Dockerfile for PNO Physics Bench
FROM nvidia/cuda:12.1-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6+PTX"

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    python3 \\
    python3-pip \\
    python3-dev \\
    git \\
    curl \\
    wget \\
    build-essential \\
    cmake \\
    libnuma1 \\
    libnuma-dev \\
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
COPY pyproject.toml .

# Install Python dependencies
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel
RUN pip3 install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
RUN pip3 install --no-cache-dir -r requirements.txt
RUN pip3 install --no-cache-dir redis lz4 cryptography

# Copy source code
COPY src/ src/
COPY examples/ examples/
COPY tests/ tests/

# Install package in production mode
RUN pip3 install --no-cache-dir -e .

# Create non-root user for security
RUN useradd -m -u 1000 pnouser && \\
    chown -R pnouser:pnouser /app
USER pnouser

# Expose application port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD python3 -c "import pno_physics_bench; print('OK')" || exit 1

# Default command
CMD ["python3", "-m", "pno_physics_bench.cli", "serve", "--host", "0.0.0.0", "--port", "8080"]
"""

    def generate_kubernetes_deployment(self) -> str:
        """Generate Kubernetes deployment manifest."""
        manifest = {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": "pno-physics-bench",
                "labels": {
                    "app": "pno-physics-bench",
                    "version": "v1.0.0",
                    "component": "inference-server"
                }
            },
            "spec": {
                "replicas": self.config.replicas,
                "selector": {
                    "matchLabels": {
                        "app": "pno-physics-bench"
                    }
                },
                "template": {
                    "metadata": {
                        "labels": {
                            "app": "pno-physics-bench",
                            "version": "v1.0.0"
                        },
                        "annotations": {
                            "prometheus.io/scrape": "true",
                            "prometheus.io/port": "9090",
                            "prometheus.io/path": "/metrics"
                        }
                    },
                    "spec": {
                        "securityContext": {
                            "runAsNonRoot": True,
                            "runAsUser": 1000,
                            "fsGroup": 1000
                        },
                        "containers": [
                            {
                                "name": "pno-server",
                                "image": "pno-physics-bench:latest",
                                "imagePullPolicy": "Always",
                                "ports": [
                                    {"containerPort": 8080, "name": "http"},
                                    {"containerPort": 9090, "name": "metrics"}
                                ],
                                "env": [
                                    {"name": "ENVIRONMENT", "value": self.config.environment},
                                    {"name": "REDIS_ENABLED", "value": str(self.config.redis_enabled)},
                                    {"name": "MONITORING_ENABLED", "value": str(self.config.monitoring_enabled)},
                                    {"name": "CUDA_VISIBLE_DEVICES", "value": "0"}
                                ],
                                "resources": {
                                    "requests": {
                                        "cpu": "500m",
                                        "memory": "2Gi",
                                        "nvidia.com/gpu": 1
                                    },
                                    "limits": {
                                        "cpu": self.config.cpu_limit,
                                        "memory": self.config.memory_limit,
                                        "nvidia.com/gpu": self.config.gpu_limit
                                    }
                                },
                                "livenessProbe": {
                                    "httpGet": {
                                        "path": "/health",
                                        "port": 8080
                                    },
                                    "initialDelaySeconds": 30,
                                    "periodSeconds": 10,
                                    "timeoutSeconds": 5,
                                    "failureThreshold": 3
                                },
                                "readinessProbe": {
                                    "httpGet": {
                                        "path": "/ready",
                                        "port": 8080
                                    },
                                    "initialDelaySeconds": 15,
                                    "periodSeconds": 5,
                                    "timeoutSeconds": 3,
                                    "failureThreshold": 3
                                },
                                "volumeMounts": [
                                    {
                                        "name": "model-storage",
                                        "mountPath": "/app/models"
                                    },
                                    {
                                        "name": "cache-storage",
                                        "mountPath": "/app/cache"
                                    }
                                ]
                            }
                        ],
                        "volumes": [
                            {
                                "name": "model-storage",
                                "persistentVolumeClaim": {
                                    "claimName": "pno-model-pvc"
                                }
                            },
                            {
                                "name": "cache-storage",
                                "emptyDir": {
                                    "sizeLimit": "10Gi"
                                }
                            }
                        ],
                        "nodeSelector": {
                            "accelerator": "nvidia-tesla-v100"
                        },
                        "tolerations": [
                            {
                                "key": "nvidia.com/gpu",
                                "operator": "Exists",
                                "effect": "NoSchedule"
                            }
                        ]
                    }
                }
            }
        }
        
        return json.dumps(manifest, indent=2)

    def generate_service_manifest(self) -> str:
        """Generate Kubernetes service manifest."""
        manifest = {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {
                "name": "pno-physics-bench-service",
                "labels": {
                    "app": "pno-physics-bench"
                }
            },
            "spec": {
                "selector": {
                    "app": "pno-physics-bench"
                },
                "ports": [
                    {
                        "name": "http",
                        "port": 80,
                        "targetPort": 8080,
                        "protocol": "TCP"
                    },
                    {
                        "name": "metrics",
                        "port": 9090,
                        "targetPort": 9090,
                        "protocol": "TCP"
                    }
                ],
                "type": "LoadBalancer"
            }
        }
        
        return json.dumps(manifest, indent=2)

    def generate_hpa_manifest(self) -> str:
        """Generate Horizontal Pod Autoscaler manifest."""
        if not self.config.auto_scaling:
            return ""
            
        manifest = {
            "apiVersion": "autoscaling/v2",
            "kind": "HorizontalPodAutoscaler",
            "metadata": {
                "name": "pno-physics-bench-hpa"
            },
            "spec": {
                "scaleTargetRef": {
                    "apiVersion": "apps/v1",
                    "kind": "Deployment",
                    "name": "pno-physics-bench"
                },
                "minReplicas": self.config.min_replicas,
                "maxReplicas": self.config.max_replicas,
                "metrics": [
                    {
                        "type": "Resource",
                        "resource": {
                            "name": "cpu",
                            "target": {
                                "type": "Utilization",
                                "averageUtilization": 70
                            }
                        }
                    },
                    {
                        "type": "Resource",
                        "resource": {
                            "name": "memory",
                            "target": {
                                "type": "Utilization",
                                "averageUtilization": 80
                            }
                        }
                    }
                ],
                "behavior": {
                    "scaleUp": {
                        "stabilizationWindowSeconds": 300,
                        "policies": [
                            {
                                "type": "Percent",
                                "value": 100,
                                "periodSeconds": 15
                            }
                        ]
                    },
                    "scaleDown": {
                        "stabilizationWindowSeconds": 300,
                        "policies": [
                            {
                                "type": "Percent",
                                "value": 10,
                                "periodSeconds": 60
                            }
                        ]
                    }
                }
            }
        }
        
        return json.dumps(manifest, indent=2)

    def generate_redis_deployment(self) -> str:
        """Generate Redis deployment for caching."""
        if not self.config.redis_enabled:
            return ""
            
        manifest = {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": "redis-cache",
                "labels": {
                    "app": "redis-cache"
                }
            },
            "spec": {
                "replicas": 1,
                "selector": {
                    "matchLabels": {
                        "app": "redis-cache"
                    }
                },
                "template": {
                    "metadata": {
                        "labels": {
                            "app": "redis-cache"
                        }
                    },
                    "spec": {
                        "containers": [
                            {
                                "name": "redis",
                                "image": "redis:7-alpine",
                                "ports": [
                                    {"containerPort": 6379}
                                ],
                                "args": [
                                    "redis-server",
                                    "--maxmemory", "2gb",
                                    "--maxmemory-policy", "allkeys-lru",
                                    "--save", "900", "1",
                                    "--save", "300", "10",
                                    "--save", "60", "10000"
                                ],
                                "resources": {
                                    "requests": {
                                        "cpu": "100m",
                                        "memory": "256Mi"
                                    },
                                    "limits": {
                                        "cpu": "500m",
                                        "memory": "2Gi"
                                    }
                                },
                                "volumeMounts": [
                                    {
                                        "name": "redis-storage",
                                        "mountPath": "/data"
                                    }
                                ]
                            }
                        ],
                        "volumes": [
                            {
                                "name": "redis-storage",
                                "persistentVolumeClaim": {
                                    "claimName": "redis-pvc"
                                }
                            }
                        ]
                    }
                }
            }
        }
        
        return json.dumps(manifest, indent=2)

    def generate_monitoring_config(self) -> str:
        """Generate Prometheus monitoring configuration."""
        if not self.config.monitoring_enabled:
            return ""
            
        config = {
            "global": {
                "scrape_interval": "15s",
                "evaluation_interval": "15s"
            },
            "rule_files": [
                "pno_rules.yml"
            ],
            "scrape_configs": [
                {
                    "job_name": "pno-physics-bench",
                    "kubernetes_sd_configs": [
                        {
                            "role": "pod"
                        }
                    ],
                    "relabel_configs": [
                        {
                            "source_labels": ["__meta_kubernetes_pod_annotation_prometheus_io_scrape"],
                            "action": "keep",
                            "regex": True
                        },
                        {
                            "source_labels": ["__meta_kubernetes_pod_annotation_prometheus_io_path"],
                            "action": "replace",
                            "target_label": "__metrics_path__",
                            "regex": "(.+)"
                        }
                    ]
                }
            ],
            "alerting": {
                "alertmanagers": [
                    {
                        "kubernetes_sd_configs": [
                            {
                                "role": "pod"
                            }
                        ],
                        "relabel_configs": [
                            {
                                "source_labels": ["__meta_kubernetes_pod_label_app"],
                                "action": "keep",
                                "regex": "alertmanager"
                            }
                        ]
                    }
                ]
            }
        }
        
        return json.dumps(config, indent=2)

    def generate_deployment_script(self) -> str:
        """Generate automated deployment script."""
        return """#!/bin/bash
# Automated Production Deployment Script for PNO Physics Bench

set -e

echo "🚀 Starting PNO Physics Bench Production Deployment..."

# Configuration
NAMESPACE="pno-production"
DOCKER_REGISTRY="your-registry.com"
IMAGE_TAG="latest"

# Create namespace if it doesn't exist
kubectl create namespace $NAMESPACE --dry-run=client -o yaml | kubectl apply -f -

# Build and push Docker image
echo "📦 Building Docker image..."
docker build -t $DOCKER_REGISTRY/pno-physics-bench:$IMAGE_TAG .
docker push $DOCKER_REGISTRY/pno-physics-bench:$IMAGE_TAG

# Apply Kubernetes manifests
echo "🎯 Deploying to Kubernetes..."

# Deploy Redis cache
if [[ "$REDIS_ENABLED" == "true" ]]; then
    echo "   • Deploying Redis cache..."
    kubectl apply -f redis-deployment.yaml -n $NAMESPACE
    kubectl apply -f redis-service.yaml -n $NAMESPACE
    kubectl apply -f redis-pvc.yaml -n $NAMESPACE
fi

# Deploy main application
echo "   • Deploying PNO Physics Bench..."
kubectl apply -f pno-deployment.yaml -n $NAMESPACE
kubectl apply -f pno-service.yaml -n $NAMESPACE
kubectl apply -f pno-pvc.yaml -n $NAMESPACE

# Deploy autoscaling
if [[ "$AUTO_SCALING" == "true" ]]; then
    echo "   • Configuring autoscaling..."
    kubectl apply -f pno-hpa.yaml -n $NAMESPACE
fi

# Deploy monitoring
if [[ "$MONITORING_ENABLED" == "true" ]]; then
    echo "   • Setting up monitoring..."
    kubectl apply -f prometheus-config.yaml -n $NAMESPACE
    kubectl apply -f grafana-deployment.yaml -n $NAMESPACE
fi

# Wait for deployment
echo "⏳ Waiting for deployment to be ready..."
kubectl rollout status deployment/pno-physics-bench -n $NAMESPACE --timeout=600s

# Verify deployment
echo "✅ Verifying deployment..."
kubectl get pods -n $NAMESPACE -l app=pno-physics-bench
kubectl get services -n $NAMESPACE

# Get service endpoint
EXTERNAL_IP=$(kubectl get service pno-physics-bench-service -n $NAMESPACE -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
if [[ -n "$EXTERNAL_IP" ]]; then
    echo "🌐 Service available at: http://$EXTERNAL_IP"
else
    echo "🔄 Waiting for external IP assignment..."
fi

# Run health check
echo "🏥 Running health check..."
kubectl port-forward -n $NAMESPACE service/pno-physics-bench-service 8080:80 &
PF_PID=$!
sleep 5

if curl -f http://localhost:8080/health > /dev/null 2>&1; then
    echo "✅ Health check passed!"
else
    echo "❌ Health check failed!"
    kill $PF_PID
    exit 1
fi

kill $PF_PID

echo "🎉 Deployment completed successfully!"
echo "📊 Access metrics at: http://$EXTERNAL_IP/metrics"
echo "📚 Access docs at: http://$EXTERNAL_IP/docs"
"""

    def generate_all_manifests(self, output_dir: str = "/root/repo/deployment"):
        """Generate all deployment manifests and scripts."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate all deployment artifacts
        artifacts = {
            "Dockerfile": self.generate_dockerfile(),
            "pno-deployment.yaml": self.generate_kubernetes_deployment(),
            "pno-service.yaml": self.generate_service_manifest(),
            "pno-hpa.yaml": self.generate_hpa_manifest(),
            "redis-deployment.yaml": self.generate_redis_deployment(),
            "prometheus-config.yaml": self.generate_monitoring_config(),
            "deploy.sh": self.generate_deployment_script()
        }
        
        # Write files
        for filename, content in artifacts.items():
            if content:  # Skip empty content
                file_path = os.path.join(output_dir, filename)
                with open(file_path, 'w') as f:
                    f.write(content)
                
                # Make scripts executable
                if filename.endswith('.sh'):
                    os.chmod(file_path, 0o755)
        
        # Generate deployment configuration
        config_path = os.path.join(output_dir, "deployment-config.json")
        with open(config_path, 'w') as f:
            json.dump(asdict(self.config), f, indent=2)
        
        return output_dir


def create_production_deployment():
    """Create complete production deployment."""
    print("🚀 Generating Production Deployment Manifests...")
    
    # Create production configuration
    config = DeploymentConfig(
        environment="production",
        replicas=3,
        cpu_limit="2000m",
        memory_limit="4Gi",
        gpu_limit=1,
        storage_size="20Gi",
        redis_enabled=True,
        monitoring_enabled=True,
        security_enabled=True,
        auto_scaling=True,
        max_replicas=10,
        min_replicas=2
    )
    
    # Generate deployment artifacts
    generator = ProductionDeploymentGenerator(config)
    output_dir = generator.generate_all_manifests()
    
    print(f"✅ Production deployment artifacts generated in: {output_dir}")
    print("\nGenerated files:")
    for file in os.listdir(output_dir):
        print(f"   • {file}")
    
    print("\n🎯 Deployment Instructions:")
    print("1. Update Docker registry in deploy.sh")
    print("2. Configure kubectl context for target cluster")
    print("3. Run: ./deployment/deploy.sh")
    print("4. Monitor deployment: kubectl get pods -n pno-production")
    
    return output_dir


if __name__ == "__main__":
    deployment_dir = create_production_deployment()
    print(f"\n🎉 Production deployment ready at: {deployment_dir}")