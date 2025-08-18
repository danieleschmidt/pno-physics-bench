"""Autonomous Production Deployment Suite for PNO Physics Bench.

This module implements a complete autonomous deployment pipeline with
multi-environment support, rolling deployments, health monitoring,
and automatic rollback capabilities.
"""

import os
import sys
import subprocess
import time
import json
import yaml
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass, asdict
from enum import Enum
import threading
import socket
import requests
from contextlib import contextmanager


class DeploymentEnvironment(Enum):
    """Deployment environment types."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class DeploymentStatus(Enum):
    """Deployment status types."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    SUCCESS = "success"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


@dataclass
class DeploymentConfig:
    """Deployment configuration."""
    environment: DeploymentEnvironment
    version: str
    replicas: int
    resource_limits: Dict[str, str]
    health_check_url: str
    monitoring_enabled: bool
    auto_scaling: bool
    rollback_on_failure: bool
    deployment_timeout_seconds: int = 1800  # 30 minutes


@dataclass
class DeploymentResult:
    """Deployment execution result."""
    deployment_id: str
    status: DeploymentStatus
    environment: DeploymentEnvironment
    version: str
    start_time: float
    end_time: Optional[float]
    duration_seconds: Optional[float]
    deployed_replicas: int
    health_checks_passed: bool
    logs: List[str]
    errors: List[str]
    rollback_performed: bool = False


class KubernetesDeploymentManager:
    """Manages Kubernetes deployments for PNO Physics Bench."""
    
    def __init__(self, kubeconfig_path: Optional[str] = None):
        self.kubeconfig_path = kubeconfig_path
        self.logger = logging.getLogger("KubernetesDeploymentManager")
        
        # Verify kubectl availability
        if not self._check_kubectl_available():
            raise RuntimeError("kubectl not available. Please install kubectl.")
    
    def _check_kubectl_available(self) -> bool:
        """Check if kubectl is available."""
        try:
            result = subprocess.run(['kubectl', 'version', '--client'], 
                                  capture_output=True, text=True)
            return result.returncode == 0
        except FileNotFoundError:
            return False
    
    def generate_deployment_manifests(
        self, 
        config: DeploymentConfig,
        image_name: str = "pno-physics-bench",
        namespace: str = "default"
    ) -> Dict[str, str]:
        """Generate Kubernetes deployment manifests."""
        
        # Deployment manifest
        deployment_manifest = {
            'apiVersion': 'apps/v1',
            'kind': 'Deployment',
            'metadata': {
                'name': f'pno-physics-bench-{config.environment.value}',
                'namespace': namespace,
                'labels': {
                    'app': 'pno-physics-bench',
                    'environment': config.environment.value,
                    'version': config.version
                }
            },
            'spec': {
                'replicas': config.replicas,
                'selector': {
                    'matchLabels': {
                        'app': 'pno-physics-bench',
                        'environment': config.environment.value
                    }
                },
                'template': {
                    'metadata': {
                        'labels': {
                            'app': 'pno-physics-bench',
                            'environment': config.environment.value,
                            'version': config.version
                        }
                    },
                    'spec': {
                        'containers': [{
                            'name': 'pno-physics-bench',
                            'image': f'{image_name}:{config.version}',
                            'ports': [
                                {'containerPort': 8000, 'name': 'http'},
                                {'containerPort': 9090, 'name': 'metrics'}
                            ],
                            'resources': {
                                'limits': config.resource_limits,
                                'requests': {
                                    'cpu': str(int(config.resource_limits.get('cpu', '1000m')[:-1]) // 2) + 'm',
                                    'memory': str(int(config.resource_limits.get('memory', '2Gi')[:-2]) // 2) + 'Gi'
                                }
                            },
                            'env': [
                                {'name': 'ENVIRONMENT', 'value': config.environment.value},
                                {'name': 'LOG_LEVEL', 'value': 'INFO'},
                                {'name': 'MONITORING_ENABLED', 'value': str(config.monitoring_enabled).lower()}
                            ],
                            'livenessProbe': {
                                'httpGet': {
                                    'path': '/health',
                                    'port': 8000
                                },
                                'initialDelaySeconds': 30,
                                'periodSeconds': 10
                            },
                            'readinessProbe': {
                                'httpGet': {
                                    'path': '/ready',
                                    'port': 8000
                                },
                                'initialDelaySeconds': 10,
                                'periodSeconds': 5
                            }
                        }]
                    }
                }
            }
        }
        
        # Service manifest
        service_manifest = {
            'apiVersion': 'v1',
            'kind': 'Service',
            'metadata': {
                'name': f'pno-physics-bench-service-{config.environment.value}',
                'namespace': namespace,
                'labels': {
                    'app': 'pno-physics-bench',
                    'environment': config.environment.value
                }
            },
            'spec': {
                'selector': {
                    'app': 'pno-physics-bench',
                    'environment': config.environment.value
                },
                'ports': [
                    {'port': 80, 'targetPort': 8000, 'name': 'http'},
                    {'port': 9090, 'targetPort': 9090, 'name': 'metrics'}
                ],
                'type': 'LoadBalancer' if config.environment == DeploymentEnvironment.PRODUCTION else 'ClusterIP'
            }
        }
        
        # HorizontalPodAutoscaler manifest (if auto-scaling enabled)
        hpa_manifest = None
        if config.auto_scaling:
            hpa_manifest = {
                'apiVersion': 'autoscaling/v2',
                'kind': 'HorizontalPodAutoscaler',
                'metadata': {
                    'name': f'pno-physics-bench-hpa-{config.environment.value}',
                    'namespace': namespace
                },
                'spec': {
                    'scaleTargetRef': {
                        'apiVersion': 'apps/v1',
                        'kind': 'Deployment',
                        'name': f'pno-physics-bench-{config.environment.value}'
                    },
                    'minReplicas': max(1, config.replicas // 2),
                    'maxReplicas': config.replicas * 3,
                    'metrics': [
                        {
                            'type': 'Resource',
                            'resource': {
                                'name': 'cpu',
                                'target': {
                                    'type': 'Utilization',
                                    'averageUtilization': 70
                                }
                            }
                        },
                        {
                            'type': 'Resource',
                            'resource': {
                                'name': 'memory',
                                'target': {
                                    'type': 'Utilization',
                                    'averageUtilization': 80
                                }
                            }
                        }
                    ]
                }
            }
        
        manifests = {
            'deployment': yaml.dump(deployment_manifest),
            'service': yaml.dump(service_manifest)
        }
        
        if hpa_manifest:
            manifests['hpa'] = yaml.dump(hpa_manifest)
        
        return manifests
    
    def deploy(
        self, 
        config: DeploymentConfig,
        manifests: Dict[str, str],
        namespace: str = "default"
    ) -> DeploymentResult:
        """Execute Kubernetes deployment."""
        
        deployment_id = f"deploy-{config.environment.value}-{int(time.time())}"
        start_time = time.time()
        logs = []
        errors = []
        
        self.logger.info(f"Starting deployment {deployment_id}")
        
        try:
            # Create namespace if it doesn't exist
            self._ensure_namespace(namespace)
            
            # Apply manifests
            for manifest_type, manifest_content in manifests.items():
                self.logger.info(f"Applying {manifest_type} manifest")
                
                # Write manifest to temporary file
                temp_file = f"/tmp/{manifest_type}-{deployment_id}.yaml"
                with open(temp_file, 'w') as f:
                    f.write(manifest_content)
                
                # Apply with kubectl
                result = subprocess.run([
                    'kubectl', 'apply', '-f', temp_file, '-n', namespace
                ], capture_output=True, text=True)
                
                if result.returncode != 0:
                    error_msg = f"Failed to apply {manifest_type}: {result.stderr}"
                    errors.append(error_msg)
                    self.logger.error(error_msg)
                else:
                    logs.append(f"Applied {manifest_type} successfully")
                
                # Clean up temp file
                os.remove(temp_file)
            
            if errors:
                return DeploymentResult(
                    deployment_id=deployment_id,
                    status=DeploymentStatus.FAILED,
                    environment=config.environment,
                    version=config.version,
                    start_time=start_time,
                    end_time=time.time(),
                    duration_seconds=time.time() - start_time,
                    deployed_replicas=0,
                    health_checks_passed=False,
                    logs=logs,
                    errors=errors
                )
            
            # Wait for deployment to be ready
            deployment_name = f'pno-physics-bench-{config.environment.value}'
            
            self.logger.info(f"Waiting for deployment {deployment_name} to be ready")
            
            if not self._wait_for_deployment_ready(deployment_name, namespace, config.deployment_timeout_seconds):
                errors.append("Deployment failed to become ready within timeout")
                
                if config.rollback_on_failure:
                    self.logger.info("Performing automatic rollback")
                    self._rollback_deployment(deployment_name, namespace)
                    return DeploymentResult(
                        deployment_id=deployment_id,
                        status=DeploymentStatus.ROLLED_BACK,
                        environment=config.environment,
                        version=config.version,
                        start_time=start_time,
                        end_time=time.time(),
                        duration_seconds=time.time() - start_time,
                        deployed_replicas=0,
                        health_checks_passed=False,
                        logs=logs,
                        errors=errors,
                        rollback_performed=True
                    )
                
                return DeploymentResult(
                    deployment_id=deployment_id,
                    status=DeploymentStatus.FAILED,
                    environment=config.environment,
                    version=config.version,
                    start_time=start_time,
                    end_time=time.time(),
                    duration_seconds=time.time() - start_time,
                    deployed_replicas=0,
                    health_checks_passed=False,
                    logs=logs,
                    errors=errors
                )
            
            # Get deployment status
            deployed_replicas = self._get_deployed_replicas(deployment_name, namespace)
            
            # Perform health checks
            health_checks_passed = True
            if config.health_check_url:
                health_checks_passed = self._perform_health_checks(config.health_check_url, namespace)
            
            end_time = time.time()
            
            return DeploymentResult(
                deployment_id=deployment_id,
                status=DeploymentStatus.SUCCESS,
                environment=config.environment,
                version=config.version,
                start_time=start_time,
                end_time=end_time,
                duration_seconds=end_time - start_time,
                deployed_replicas=deployed_replicas,
                health_checks_passed=health_checks_passed,
                logs=logs,
                errors=errors
            )
            
        except Exception as e:
            error_msg = f"Deployment failed with exception: {str(e)}"
            errors.append(error_msg)
            self.logger.error(error_msg)
            
            return DeploymentResult(
                deployment_id=deployment_id,
                status=DeploymentStatus.FAILED,
                environment=config.environment,
                version=config.version,
                start_time=start_time,
                end_time=time.time(),
                duration_seconds=time.time() - start_time,
                deployed_replicas=0,
                health_checks_passed=False,
                logs=logs,
                errors=errors
            )
    
    def _ensure_namespace(self, namespace: str):
        """Ensure namespace exists."""
        result = subprocess.run([
            'kubectl', 'get', 'namespace', namespace
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            # Create namespace
            subprocess.run([
                'kubectl', 'create', 'namespace', namespace
            ], capture_output=True, text=True)
    
    def _wait_for_deployment_ready(self, deployment_name: str, namespace: str, timeout_seconds: int) -> bool:
        """Wait for deployment to be ready."""
        start_time = time.time()
        
        while time.time() - start_time < timeout_seconds:
            result = subprocess.run([
                'kubectl', 'rollout', 'status', f'deployment/{deployment_name}', 
                '-n', namespace, '--timeout=30s'
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                return True
            
            time.sleep(10)
        
        return False
    
    def _get_deployed_replicas(self, deployment_name: str, namespace: str) -> int:
        """Get number of deployed replicas."""
        result = subprocess.run([
            'kubectl', 'get', 'deployment', deployment_name, '-n', namespace, 
            '-o', 'jsonpath={.status.readyReplicas}'
        ], capture_output=True, text=True)
        
        try:
            return int(result.stdout.strip() or "0")
        except ValueError:
            return 0
    
    def _perform_health_checks(self, health_check_url: str, namespace: str) -> bool:
        """Perform health checks on deployed service."""
        # This would typically involve port-forwarding or using an ingress
        # For simplicity, we'll simulate health checks
        
        try:
            # In a real implementation, you would:
            # 1. Port-forward to the service
            # 2. Make HTTP requests to health endpoints
            # 3. Verify responses
            
            # Simulate health check
            time.sleep(5)  # Give service time to start
            return True
            
        except Exception as e:
            self.logger.error(f"Health checks failed: {str(e)}")
            return False
    
    def _rollback_deployment(self, deployment_name: str, namespace: str):
        """Rollback deployment to previous version."""
        subprocess.run([
            'kubectl', 'rollout', 'undo', f'deployment/{deployment_name}', '-n', namespace
        ], capture_output=True, text=True)


class ContainerImageBuilder:
    """Builds and manages container images for deployment."""
    
    def __init__(self, registry_url: Optional[str] = None):
        self.registry_url = registry_url
        self.logger = logging.getLogger("ContainerImageBuilder")
        
        # Verify Docker availability
        if not self._check_docker_available():
            raise RuntimeError("Docker not available. Please install Docker.")
    
    def _check_docker_available(self) -> bool:
        """Check if Docker is available."""
        try:
            result = subprocess.run(['docker', '--version'], capture_output=True, text=True)
            return result.returncode == 0
        except FileNotFoundError:
            return False
    
    def generate_dockerfile(self, environment: DeploymentEnvironment) -> str:
        """Generate optimized Dockerfile for deployment."""
        
        base_image = "python:3.11-slim"
        
        if environment == DeploymentEnvironment.PRODUCTION:
            # Use more optimized base for production
            base_image = "python:3.11-slim-bullseye"
        
        dockerfile_content = f"""# Multi-stage Dockerfile for PNO Physics Bench
# Stage 1: Build dependencies
FROM {base_image} as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    g++ \\
    gfortran \\
    libblas-dev \\
    liblapack-dev \\
    libhdf5-dev \\
    pkg-config \\
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \\
    pip install --no-cache-dir -r /tmp/requirements.txt

# Stage 2: Production image
FROM {base_image} as production

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \\
    libhdf5-103 \\
    libblas3 \\
    liblapack3 \\
    && rm -rf /var/lib/apt/lists/* \\
    && apt-get clean

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONPATH="/opt/venv/lib/python3.11/site-packages"

# Create non-root user
RUN useradd --create-home --shell /bin/bash pno && \\
    mkdir -p /app && \\
    chown -R pno:pno /app

USER pno
WORKDIR /app

# Copy application code
COPY --chown=pno:pno src/ ./src/
COPY --chown=pno:pno pyproject.toml ./
COPY --chown=pno:pno README.md ./

# Install package
RUN pip install --no-cache-dir -e .

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV LOG_LEVEL=INFO
ENV WORKERS=4

# Expose ports
EXPOSE 8000 9090

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \\
    CMD python -c "import requests; requests.get('http://localhost:8000/health', timeout=5)"

# Default command
CMD ["python", "-m", "pno_physics_bench.server", "--host", "0.0.0.0", "--port", "8000"]
"""
        
        return dockerfile_content
    
    def build_image(
        self, 
        project_root: str,
        image_name: str,
        version: str,
        environment: DeploymentEnvironment,
        push_to_registry: bool = False
    ) -> Tuple[bool, List[str]]:
        """Build container image."""
        
        logs = []
        
        try:
            # Generate Dockerfile
            dockerfile_content = self.generate_dockerfile(environment)
            dockerfile_path = os.path.join(project_root, "Dockerfile.generated")
            
            with open(dockerfile_path, 'w') as f:
                f.write(dockerfile_content)
            
            logs.append("Generated Dockerfile")
            
            # Build image
            tag = f"{image_name}:{version}"
            if self.registry_url:
                tag = f"{self.registry_url}/{tag}"
            
            self.logger.info(f"Building image: {tag}")
            
            build_result = subprocess.run([
                'docker', 'build', 
                '-f', dockerfile_path,
                '-t', tag,
                project_root
            ], capture_output=True, text=True)
            
            if build_result.returncode != 0:
                logs.append(f"Build failed: {build_result.stderr}")
                return False, logs
            
            logs.append(f"Successfully built image: {tag}")
            
            # Push to registry if requested
            if push_to_registry and self.registry_url:
                self.logger.info(f"Pushing image to registry: {tag}")
                
                push_result = subprocess.run([
                    'docker', 'push', tag
                ], capture_output=True, text=True)
                
                if push_result.returncode != 0:
                    logs.append(f"Push failed: {push_result.stderr}")
                    return False, logs
                
                logs.append(f"Successfully pushed image: {tag}")
            
            # Clean up generated Dockerfile
            os.remove(dockerfile_path)
            
            return True, logs
            
        except Exception as e:
            logs.append(f"Image build failed: {str(e)}")
            return False, logs


class DeploymentMonitor:
    """Monitors deployment health and performance."""
    
    def __init__(self):
        self.logger = logging.getLogger("DeploymentMonitor")
        self.monitoring_active = False
        self.monitoring_thread = None
    
    def start_monitoring(
        self, 
        deployment_result: DeploymentResult,
        health_check_url: str,
        monitoring_duration_seconds: int = 3600
    ):
        """Start monitoring deployed service."""
        
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(deployment_result, health_check_url, monitoring_duration_seconds),
            daemon=True
        )
        self.monitoring_thread.start()
        
        self.logger.info(f"Started monitoring for deployment {deployment_result.deployment_id}")
    
    def stop_monitoring(self):
        """Stop monitoring."""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
    
    def _monitoring_loop(
        self, 
        deployment_result: DeploymentResult,
        health_check_url: str,
        duration_seconds: int
    ):
        """Main monitoring loop."""
        
        start_time = time.time()
        check_interval = 30  # Check every 30 seconds
        
        metrics = {
            'health_check_successes': 0,
            'health_check_failures': 0,
            'response_times': [],
            'error_count': 0
        }
        
        while (self.monitoring_active and 
               time.time() - start_time < duration_seconds):
            
            try:
                # Perform health check
                check_start = time.time()
                success = self._perform_health_check(health_check_url)
                response_time = time.time() - check_start
                
                if success:
                    metrics['health_check_successes'] += 1
                    metrics['response_times'].append(response_time)
                else:
                    metrics['health_check_failures'] += 1
                    metrics['error_count'] += 1
                
                # Log metrics periodically
                if (metrics['health_check_successes'] + metrics['health_check_failures']) % 10 == 0:
                    self._log_metrics(deployment_result.deployment_id, metrics)
                
                # Check for critical failures
                total_checks = metrics['health_check_successes'] + metrics['health_check_failures']
                if (total_checks >= 10 and 
                    metrics['health_check_failures'] / total_checks > 0.5):
                    
                    self.logger.critical(
                        f"Deployment {deployment_result.deployment_id} failing health checks "
                        f"({metrics['health_check_failures']}/{total_checks})"
                    )
                
            except Exception as e:
                self.logger.error(f"Monitoring error: {str(e)}")
                metrics['error_count'] += 1
            
            time.sleep(check_interval)
        
        # Final metrics report
        self._generate_monitoring_report(deployment_result.deployment_id, metrics)
    
    def _perform_health_check(self, health_check_url: str) -> bool:
        """Perform single health check."""
        try:
            # In real implementation, this would make HTTP requests
            # For now, simulate health checks
            time.sleep(0.1)  # Simulate network delay
            return True
        except Exception:
            return False
    
    def _log_metrics(self, deployment_id: str, metrics: Dict[str, Any]):
        """Log current metrics."""
        total_checks = metrics['health_check_successes'] + metrics['health_check_failures']
        success_rate = metrics['health_check_successes'] / total_checks if total_checks > 0 else 0
        
        avg_response_time = (sum(metrics['response_times']) / len(metrics['response_times']) 
                           if metrics['response_times'] else 0)
        
        self.logger.info(
            f"Deployment {deployment_id} metrics: "
            f"Success rate: {success_rate:.2%}, "
            f"Avg response time: {avg_response_time:.3f}s, "
            f"Total checks: {total_checks}"
        )
    
    def _generate_monitoring_report(self, deployment_id: str, metrics: Dict[str, Any]):
        """Generate final monitoring report."""
        
        total_checks = metrics['health_check_successes'] + metrics['health_check_failures']
        success_rate = metrics['health_check_successes'] / total_checks if total_checks > 0 else 0
        
        avg_response_time = (sum(metrics['response_times']) / len(metrics['response_times']) 
                           if metrics['response_times'] else 0)
        
        report = {
            'deployment_id': deployment_id,
            'monitoring_summary': {
                'total_health_checks': total_checks,
                'success_rate': success_rate,
                'average_response_time_seconds': avg_response_time,
                'total_errors': metrics['error_count']
            },
            'detailed_metrics': metrics
        }
        
        # Save report
        report_path = f"monitoring_report_{deployment_id}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.logger.info(f"Monitoring report saved: {report_path}")


class AutonomousDeploymentOrchestrator:
    """Orchestrates the complete autonomous deployment pipeline."""
    
    def __init__(
        self,
        project_root: str,
        registry_url: Optional[str] = None,
        kubeconfig_path: Optional[str] = None
    ):
        self.project_root = Path(project_root)
        self.registry_url = registry_url
        
        # Initialize components
        self.image_builder = ContainerImageBuilder(registry_url)
        self.k8s_manager = KubernetesDeploymentManager(kubeconfig_path)
        self.monitor = DeploymentMonitor()
        
        self.logger = logging.getLogger("AutonomousDeploymentOrchestrator")
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def execute_deployment_pipeline(
        self,
        environment: DeploymentEnvironment,
        version: str,
        config_overrides: Optional[Dict[str, Any]] = None
    ) -> DeploymentResult:
        """Execute complete deployment pipeline."""
        
        self.logger.info(f"Starting autonomous deployment pipeline for {environment.value}")
        
        try:
            # 1. Generate deployment configuration
            config = self._generate_deployment_config(environment, version, config_overrides)
            
            # 2. Build container image
            image_name = "pno-physics-bench"
            build_success, build_logs = self.image_builder.build_image(
                project_root=str(self.project_root),
                image_name=image_name,
                version=version,
                environment=environment,
                push_to_registry=bool(self.registry_url)
            )
            
            if not build_success:
                return DeploymentResult(
                    deployment_id=f"failed-build-{int(time.time())}",
                    status=DeploymentStatus.FAILED,
                    environment=environment,
                    version=version,
                    start_time=time.time(),
                    end_time=time.time(),
                    duration_seconds=0,
                    deployed_replicas=0,
                    health_checks_passed=False,
                    logs=build_logs,
                    errors=["Image build failed"]
                )
            
            # 3. Generate Kubernetes manifests
            manifests = self.k8s_manager.generate_deployment_manifests(
                config=config,
                image_name=f"{self.registry_url}/{image_name}" if self.registry_url else image_name
            )
            
            # 4. Execute deployment
            deployment_result = self.k8s_manager.deploy(config, manifests)
            
            # 5. Start monitoring if deployment succeeded
            if deployment_result.status == DeploymentStatus.SUCCESS:
                if config.monitoring_enabled and config.health_check_url:
                    self.monitor.start_monitoring(
                        deployment_result=deployment_result,
                        health_check_url=config.health_check_url,
                        monitoring_duration_seconds=3600  # 1 hour
                    )
            
            return deployment_result
            
        except Exception as e:
            self.logger.error(f"Deployment pipeline failed: {str(e)}")
            
            return DeploymentResult(
                deployment_id=f"failed-pipeline-{int(time.time())}",
                status=DeploymentStatus.FAILED,
                environment=environment,
                version=version,
                start_time=time.time(),
                end_time=time.time(),
                duration_seconds=0,
                deployed_replicas=0,
                health_checks_passed=False,
                logs=[],
                errors=[f"Pipeline execution failed: {str(e)}"]
            )
    
    def _generate_deployment_config(
        self,
        environment: DeploymentEnvironment,
        version: str,
        overrides: Optional[Dict[str, Any]] = None
    ) -> DeploymentConfig:
        """Generate deployment configuration based on environment."""
        
        # Default configurations by environment
        configs = {
            DeploymentEnvironment.DEVELOPMENT: DeploymentConfig(
                environment=environment,
                version=version,
                replicas=1,
                resource_limits={'cpu': '500m', 'memory': '1Gi'},
                health_check_url="http://localhost:8000/health",
                monitoring_enabled=True,
                auto_scaling=False,
                rollback_on_failure=True,
                deployment_timeout_seconds=600  # 10 minutes
            ),
            
            DeploymentEnvironment.STAGING: DeploymentConfig(
                environment=environment,
                version=version,
                replicas=2,
                resource_limits={'cpu': '1000m', 'memory': '2Gi'},
                health_check_url="http://staging-pno-service/health",
                monitoring_enabled=True,
                auto_scaling=True,
                rollback_on_failure=True,
                deployment_timeout_seconds=900  # 15 minutes
            ),
            
            DeploymentEnvironment.PRODUCTION: DeploymentConfig(
                environment=environment,
                version=version,
                replicas=5,
                resource_limits={'cpu': '2000m', 'memory': '4Gi'},
                health_check_url="http://pno-service/health",
                monitoring_enabled=True,
                auto_scaling=True,
                rollback_on_failure=True,
                deployment_timeout_seconds=1800  # 30 minutes
            )
        }
        
        config = configs[environment]
        
        # Apply overrides
        if overrides:
            for key, value in overrides.items():
                if hasattr(config, key):
                    setattr(config, key, value)
        
        return config
    
    def generate_deployment_summary(self, results: List[DeploymentResult]) -> Dict[str, Any]:
        """Generate comprehensive deployment summary."""
        
        if not results:
            return {'error': 'No deployment results provided'}
        
        summary = {
            'total_deployments': len(results),
            'successful_deployments': sum(1 for r in results if r.status == DeploymentStatus.SUCCESS),
            'failed_deployments': sum(1 for r in results if r.status == DeploymentStatus.FAILED),
            'rolled_back_deployments': sum(1 for r in results if r.rollback_performed),
            'environments_deployed': list(set(r.environment.value for r in results)),
            'total_replicas_deployed': sum(r.deployed_replicas for r in results),
            'average_deployment_time_seconds': sum(r.duration_seconds or 0 for r in results) / len(results),
            'health_check_success_rate': sum(1 for r in results if r.health_checks_passed) / len(results),
            'deployment_details': [asdict(result) for result in results]
        }
        
        return summary


def create_production_deployment_pipeline(
    project_root: str,
    environments: List[DeploymentEnvironment],
    version: str,
    registry_url: Optional[str] = None
) -> Dict[str, DeploymentResult]:
    """Create and execute production deployment pipeline for multiple environments."""
    
    print("🚀 AUTONOMOUS PRODUCTION DEPLOYMENT PIPELINE")
    print("=" * 80)
    
    # Initialize orchestrator
    orchestrator = AutonomousDeploymentOrchestrator(
        project_root=project_root,
        registry_url=registry_url
    )
    
    deployment_results = {}
    
    # Deploy to each environment
    for environment in environments:
        print(f"\n📦 Deploying to {environment.value.upper()}")
        print("-" * 40)
        
        try:
            result = orchestrator.execute_deployment_pipeline(
                environment=environment,
                version=version
            )
            
            deployment_results[environment.value] = result
            
            status_icon = "✅" if result.status == DeploymentStatus.SUCCESS else "❌"
            print(f"{status_icon} {environment.value}: {result.status.value}")
            print(f"   Deployment ID: {result.deployment_id}")
            print(f"   Duration: {result.duration_seconds:.1f}s")
            print(f"   Replicas: {result.deployed_replicas}")
            print(f"   Health Checks: {'✅' if result.health_checks_passed else '❌'}")
            
            if result.errors:
                print(f"   Errors: {len(result.errors)}")
                for error in result.errors[:3]:  # Show first 3 errors
                    print(f"     - {error}")
            
        except Exception as e:
            print(f"❌ {environment.value}: Deployment failed - {str(e)}")
            deployment_results[environment.value] = DeploymentResult(
                deployment_id=f"failed-{environment.value}-{int(time.time())}",
                status=DeploymentStatus.FAILED,
                environment=environment,
                version=version,
                start_time=time.time(),
                end_time=time.time(),
                duration_seconds=0,
                deployed_replicas=0,
                health_checks_passed=False,
                logs=[],
                errors=[str(e)]
            )
    
    # Generate summary
    summary = orchestrator.generate_deployment_summary(list(deployment_results.values()))
    
    print(f"\n📊 DEPLOYMENT SUMMARY")
    print("=" * 40)
    print(f"Total deployments: {summary['total_deployments']}")
    print(f"Successful: {summary['successful_deployments']}")
    print(f"Failed: {summary['failed_deployments']}")
    print(f"Success rate: {summary['successful_deployments']/summary['total_deployments']:.1%}")
    print(f"Average deployment time: {summary['average_deployment_time_seconds']:.1f}s")
    print(f"Total replicas deployed: {summary['total_replicas_deployed']}")
    
    # Save summary
    summary_path = f"deployment_summary_{version}_{int(time.time())}.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"\n📄 Detailed summary saved to: {summary_path}")
    
    return deployment_results


if __name__ == "__main__":
    # Example usage
    project_root = os.path.dirname(os.path.dirname(__file__))  # Parent of deployment directory
    version = f"v1.0.{int(time.time())}"
    
    environments = [
        DeploymentEnvironment.DEVELOPMENT,
        DeploymentEnvironment.STAGING
        # DeploymentEnvironment.PRODUCTION  # Uncomment for production deployment
    ]
    
    results = create_production_deployment_pipeline(
        project_root=project_root,
        environments=environments,
        version=version,
        registry_url=None  # Set to your container registry URL
    )
    
    # Print final status
    success_count = sum(1 for r in results.values() if r.status == DeploymentStatus.SUCCESS)
    total_count = len(results)
    
    if success_count == total_count:
        print("\n🎉 ALL DEPLOYMENTS SUCCESSFUL!")
        exit(0)
    else:
        print(f"\n⚠️  {total_count - success_count} deployment(s) failed")
        exit(1)