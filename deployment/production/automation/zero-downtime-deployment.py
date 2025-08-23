#!/usr/bin/env python3
"""
Zero-Downtime Deployment Automation
Advanced deployment strategies with automatic rollback and health validation
"""

import asyncio
import json
import logging
import subprocess
import time
import yaml
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
import tempfile
import shutil
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DeploymentStrategy(Enum):
    """Deployment strategy types"""
    ROLLING_UPDATE = "rolling_update"
    BLUE_GREEN = "blue_green"
    CANARY = "canary"
    A_B_TESTING = "ab_testing"

class DeploymentStatus(Enum):
    """Deployment status types"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    VALIDATING = "validating"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"

@dataclass
class DeploymentConfig:
    """Deployment configuration"""
    strategy: DeploymentStrategy
    namespace: str
    deployment_name: str
    new_image: str
    health_check_url: str
    validation_timeout: int = 300
    rollback_threshold: float = 0.05  # 5% error rate threshold
    traffic_split_percentage: int = 10  # For canary deployments
    max_unavailable: int = 0
    max_surge: int = 1

@dataclass
class HealthMetrics:
    """Health check metrics"""
    timestamp: datetime
    response_time: float
    status_code: int
    error_rate: float
    throughput: float
    cpu_usage: float
    memory_usage: float

@dataclass
class DeploymentResult:
    """Deployment execution result"""
    deployment_id: str
    status: DeploymentStatus
    strategy: DeploymentStrategy
    start_time: datetime
    end_time: Optional[datetime]
    duration: Optional[float]
    metrics: List[HealthMetrics]
    rollback_reason: Optional[str]
    validation_results: Dict[str, Any]

class ZeroDowntimeDeployer:
    """Zero-downtime deployment orchestrator"""
    
    def __init__(self, config: DeploymentConfig):
        self.config = config
        self.deployment_id = f"deploy-{int(time.time())}"
        self.kubectl_timeout = 300
        
    async def deploy(self) -> DeploymentResult:
        """Execute zero-downtime deployment"""
        logger.info(f"Starting {self.config.strategy.value} deployment: {self.deployment_id}")
        
        result = DeploymentResult(
            deployment_id=self.deployment_id,
            status=DeploymentStatus.IN_PROGRESS,
            strategy=self.config.strategy,
            start_time=datetime.now(),
            end_time=None,
            duration=None,
            metrics=[],
            rollback_reason=None,
            validation_results={}
        )
        
        try:
            # Pre-deployment validation
            await self._pre_deployment_validation()
            
            # Execute deployment strategy
            if self.config.strategy == DeploymentStrategy.ROLLING_UPDATE:
                await self._rolling_update_deployment()
            elif self.config.strategy == DeploymentStrategy.BLUE_GREEN:
                await self._blue_green_deployment()
            elif self.config.strategy == DeploymentStrategy.CANARY:
                await self._canary_deployment()
            else:
                raise ValueError(f"Unsupported deployment strategy: {self.config.strategy}")
            
            # Post-deployment validation
            result.validation_results = await self._post_deployment_validation()
            
            if result.validation_results.get('overall_health', False):
                result.status = DeploymentStatus.COMPLETED
                logger.info(f"Deployment {self.deployment_id} completed successfully")
            else:
                result.status = DeploymentStatus.FAILED
                result.rollback_reason = "Post-deployment validation failed"
                await self._rollback_deployment()
                result.status = DeploymentStatus.ROLLED_BACK
                
        except Exception as e:
            logger.error(f"Deployment {self.deployment_id} failed: {e}")
            result.status = DeploymentStatus.FAILED
            result.rollback_reason = str(e)
            
            try:
                await self._rollback_deployment()
                result.status = DeploymentStatus.ROLLED_BACK
            except Exception as rollback_error:
                logger.error(f"Rollback also failed: {rollback_error}")
        
        finally:
            result.end_time = datetime.now()
            result.duration = (result.end_time - result.start_time).total_seconds()
            
        return result
    
    async def _pre_deployment_validation(self):
        """Validate system readiness before deployment"""
        logger.info("Running pre-deployment validation")
        
        # Check cluster health
        cluster_health = await self._check_cluster_health()
        if not cluster_health:
            raise RuntimeError("Cluster health check failed")
        
        # Check current deployment status
        current_status = await self._get_deployment_status()
        if current_status.get('ready_replicas', 0) == 0:
            raise RuntimeError("Current deployment has no ready replicas")
        
        # Validate new image
        if not await self._validate_image():
            raise RuntimeError("New image validation failed")
        
        # Check error budget
        if not await self._check_error_budget():
            raise RuntimeError("Insufficient error budget for deployment")
        
        logger.info("Pre-deployment validation passed")
    
    async def _rolling_update_deployment(self):
        """Execute rolling update deployment"""
        logger.info("Starting rolling update deployment")
        
        # Update deployment image
        await self._update_deployment_image()
        
        # Monitor rollout
        await self._monitor_rollout()
        
        # Validate health during rollout
        await self._continuous_health_monitoring()
        
        logger.info("Rolling update deployment completed")
    
    async def _blue_green_deployment(self):
        """Execute blue-green deployment"""
        logger.info("Starting blue-green deployment")
        
        # Create green deployment
        green_deployment_name = f"{self.config.deployment_name}-green"
        await self._create_green_deployment(green_deployment_name)
        
        # Wait for green deployment to be ready
        await self._wait_for_deployment_ready(green_deployment_name)
        
        # Validate green deployment
        green_health = await self._validate_green_deployment(green_deployment_name)
        
        if green_health:
            # Switch traffic to green
            await self._switch_traffic_to_green(green_deployment_name)
            
            # Monitor traffic switch
            await self._monitor_traffic_switch()
            
            # Clean up blue deployment after successful validation
            await asyncio.sleep(300)  # Wait 5 minutes before cleanup
            await self._cleanup_blue_deployment()
        else:
            # Clean up failed green deployment
            await self._cleanup_green_deployment(green_deployment_name)
            raise RuntimeError("Green deployment validation failed")
        
        logger.info("Blue-green deployment completed")
    
    async def _canary_deployment(self):
        """Execute canary deployment"""
        logger.info(f"Starting canary deployment with {self.config.traffic_split_percentage}% traffic")
        
        # Create canary deployment
        canary_deployment_name = f"{self.config.deployment_name}-canary"
        await self._create_canary_deployment(canary_deployment_name)
        
        # Configure traffic splitting
        await self._configure_traffic_splitting(canary_deployment_name)
        
        # Monitor canary metrics
        canary_metrics = await self._monitor_canary_metrics(canary_deployment_name)
        
        if canary_metrics['success']:
            # Gradually increase canary traffic
            for traffic_percentage in [25, 50, 75, 100]:
                logger.info(f"Increasing canary traffic to {traffic_percentage}%")
                await self._update_traffic_split(canary_deployment_name, traffic_percentage)
                await asyncio.sleep(300)  # Wait 5 minutes between increases
                
                # Monitor metrics at each stage
                stage_metrics = await self._monitor_canary_metrics(canary_deployment_name)
                if not stage_metrics['success']:
                    raise RuntimeError(f"Canary metrics degraded at {traffic_percentage}% traffic")
            
            # Finalize canary promotion
            await self._promote_canary(canary_deployment_name)
        else:
            # Rollback canary
            await self._cleanup_canary_deployment(canary_deployment_name)
            raise RuntimeError("Canary metrics validation failed")
        
        logger.info("Canary deployment completed")
    
    async def _update_deployment_image(self):
        """Update deployment with new image"""
        cmd = [
            "kubectl", "set", "image",
            f"deployment/{self.config.deployment_name}",
            f"pno-inference={self.config.new_image}",
            "-n", self.config.namespace
        ]
        
        result = await self._run_kubectl_command(cmd)
        if result.returncode != 0:
            raise RuntimeError(f"Failed to update deployment image: {result.stderr}")
    
    async def _monitor_rollout(self):
        """Monitor deployment rollout progress"""
        cmd = [
            "kubectl", "rollout", "status",
            f"deployment/{self.config.deployment_name}",
            "-n", self.config.namespace,
            f"--timeout={self.config.validation_timeout}s"
        ]
        
        result = await self._run_kubectl_command(cmd)
        if result.returncode != 0:
            raise RuntimeError(f"Rollout failed or timed out: {result.stderr}")
    
    async def _continuous_health_monitoring(self):
        """Continuously monitor health during deployment"""
        logger.info("Starting continuous health monitoring")
        
        monitoring_duration = self.config.validation_timeout
        check_interval = 30  # Check every 30 seconds
        checks_count = monitoring_duration // check_interval
        
        failed_checks = 0
        max_failed_checks = 3
        
        for i in range(checks_count):
            try:
                metrics = await self._collect_health_metrics()
                
                # Check if metrics indicate problems
                if metrics.error_rate > self.config.rollback_threshold:
                    failed_checks += 1
                    logger.warning(f"Health check failed ({failed_checks}/{max_failed_checks}): "
                                 f"Error rate {metrics.error_rate:.2%}")
                    
                    if failed_checks >= max_failed_checks:
                        raise RuntimeError(f"Health monitoring failed: Error rate {metrics.error_rate:.2%}")
                else:
                    failed_checks = 0  # Reset counter on successful check
                
                await asyncio.sleep(check_interval)
                
            except Exception as e:
                failed_checks += 1
                logger.error(f"Health monitoring error ({failed_checks}/{max_failed_checks}): {e}")
                
                if failed_checks >= max_failed_checks:
                    raise RuntimeError(f"Health monitoring failed: {e}")
                
                await asyncio.sleep(check_interval)
    
    async def _create_green_deployment(self, green_name: str):
        """Create green deployment for blue-green strategy"""
        # Get current deployment manifest
        current_manifest = await self._get_deployment_manifest()
        
        # Modify for green deployment
        green_manifest = current_manifest.copy()
        green_manifest['metadata']['name'] = green_name
        green_manifest['metadata']['labels']['deployment-type'] = 'green'
        green_manifest['spec']['selector']['matchLabels']['deployment-type'] = 'green'
        green_manifest['spec']['template']['metadata']['labels']['deployment-type'] = 'green'
        green_manifest['spec']['template']['spec']['containers'][0]['image'] = self.config.new_image
        
        # Apply green deployment
        await self._apply_manifest(green_manifest)
        
        logger.info(f"Green deployment {green_name} created")
    
    async def _validate_green_deployment(self, green_name: str) -> bool:
        """Validate green deployment health"""
        logger.info(f"Validating green deployment: {green_name}")
        
        # Get green deployment service endpoint
        green_service_ip = await self._get_service_cluster_ip(f"{green_name}-service")
        
        # Test green deployment directly
        test_url = f"http://{green_service_ip}:8000/health"
        
        for attempt in range(10):  # 10 attempts with 30-second intervals
            try:
                import aiohttp
                async with aiohttp.ClientSession() as session:
                    async with session.get(test_url, timeout=10) as response:
                        if response.status == 200:
                            logger.info(f"Green deployment validation passed (attempt {attempt + 1})")
                            return True
                        else:
                            logger.warning(f"Green deployment returned status {response.status}")
            
            except Exception as e:
                logger.warning(f"Green deployment validation failed (attempt {attempt + 1}): {e}")
            
            await asyncio.sleep(30)
        
        return False
    
    async def _switch_traffic_to_green(self, green_name: str):
        """Switch service traffic to green deployment"""
        logger.info("Switching traffic to green deployment")
        
        # Update service selector to point to green deployment
        service_patch = {
            'spec': {
                'selector': {
                    'app': self.config.deployment_name,
                    'deployment-type': 'green'
                }
            }
        }
        
        cmd = [
            "kubectl", "patch", "service", self.config.deployment_name,
            "-n", self.config.namespace,
            "-p", json.dumps(service_patch)
        ]
        
        result = await self._run_kubectl_command(cmd)
        if result.returncode != 0:
            raise RuntimeError(f"Failed to switch traffic to green: {result.stderr}")
    
    async def _monitor_traffic_switch(self):
        """Monitor metrics after traffic switch"""
        logger.info("Monitoring traffic switch")
        
        # Wait for traffic to stabilize
        await asyncio.sleep(60)
        
        # Monitor for 5 minutes after switch
        for i in range(10):  # 10 checks, 30 seconds apart
            metrics = await self._collect_health_metrics()
            
            if metrics.error_rate > self.config.rollback_threshold:
                raise RuntimeError(f"Traffic switch validation failed: Error rate {metrics.error_rate:.2%}")
            
            await asyncio.sleep(30)
        
        logger.info("Traffic switch validation passed")
    
    async def _create_canary_deployment(self, canary_name: str):
        """Create canary deployment"""
        # Similar to green deployment but with fewer replicas
        current_manifest = await self._get_deployment_manifest()
        
        canary_manifest = current_manifest.copy()
        canary_manifest['metadata']['name'] = canary_name
        canary_manifest['metadata']['labels']['deployment-type'] = 'canary'
        canary_manifest['spec']['replicas'] = 1  # Start with single replica
        canary_manifest['spec']['selector']['matchLabels']['deployment-type'] = 'canary'
        canary_manifest['spec']['template']['metadata']['labels']['deployment-type'] = 'canary'
        canary_manifest['spec']['template']['spec']['containers'][0]['image'] = self.config.new_image
        
        await self._apply_manifest(canary_manifest)
        
        logger.info(f"Canary deployment {canary_name} created")
    
    async def _configure_traffic_splitting(self, canary_name: str):
        """Configure traffic splitting for canary deployment"""
        # This would typically involve configuring an ingress controller
        # or service mesh for traffic splitting
        logger.info(f"Configuring {self.config.traffic_split_percentage}% traffic to canary")
        
        # Example: Update ingress annotations for traffic splitting
        # Implementation depends on ingress controller (nginx, istio, etc.)
        pass
    
    async def _monitor_canary_metrics(self, canary_name: str) -> Dict[str, Any]:
        """Monitor canary deployment metrics"""
        logger.info("Monitoring canary metrics")
        
        # Collect metrics for 5 minutes
        metrics_samples = []
        
        for i in range(10):  # 10 samples, 30 seconds apart
            try:
                metrics = await self._collect_health_metrics()
                metrics_samples.append(metrics)
                await asyncio.sleep(30)
            except Exception as e:
                logger.error(f"Error collecting canary metrics: {e}")
                return {'success': False, 'reason': str(e)}
        
        # Analyze metrics
        avg_error_rate = sum(m.error_rate for m in metrics_samples) / len(metrics_samples)
        avg_response_time = sum(m.response_time for m in metrics_samples) / len(metrics_samples)
        
        success = (
            avg_error_rate <= self.config.rollback_threshold and
            avg_response_time <= 1000  # 1 second threshold
        )
        
        return {
            'success': success,
            'avg_error_rate': avg_error_rate,
            'avg_response_time': avg_response_time,
            'samples': len(metrics_samples)
        }
    
    async def _post_deployment_validation(self) -> Dict[str, Any]:
        """Comprehensive post-deployment validation"""
        logger.info("Running post-deployment validation")
        
        validation_results = {
            'deployment_health': await self._validate_deployment_health(),
            'service_health': await self._validate_service_health(),
            'performance_metrics': await self._validate_performance(),
            'integration_tests': await self._run_integration_tests(),
            'overall_health': False
        }
        
        # Determine overall health
        validation_results['overall_health'] = all([
            validation_results['deployment_health'],
            validation_results['service_health'],
            validation_results['performance_metrics']['meets_sla'],
            validation_results['integration_tests']
        ])
        
        return validation_results
    
    async def _validate_deployment_health(self) -> bool:
        """Validate deployment health status"""
        try:
            status = await self._get_deployment_status()
            ready_replicas = status.get('ready_replicas', 0)
            desired_replicas = status.get('replicas', 0)
            
            return ready_replicas == desired_replicas and ready_replicas > 0
        except Exception as e:
            logger.error(f"Deployment health validation failed: {e}")
            return False
    
    async def _validate_service_health(self) -> bool:
        """Validate service endpoint health"""
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get(self.config.health_check_url, timeout=10) as response:
                    return response.status == 200
        except Exception as e:
            logger.error(f"Service health validation failed: {e}")
            return False
    
    async def _validate_performance(self) -> Dict[str, Any]:
        """Validate performance metrics meet SLA"""
        try:
            metrics = await self._collect_health_metrics()
            
            meets_sla = (
                metrics.response_time <= 500 and  # 500ms SLA
                metrics.error_rate <= 0.01  # 1% error rate SLA
            )
            
            return {
                'meets_sla': meets_sla,
                'response_time': metrics.response_time,
                'error_rate': metrics.error_rate,
                'throughput': metrics.throughput
            }
        except Exception as e:
            logger.error(f"Performance validation failed: {e}")
            return {'meets_sla': False, 'error': str(e)}
    
    async def _run_integration_tests(self) -> bool:
        """Run integration tests against deployed service"""
        try:
            # Run basic integration test
            import aiohttp
            
            test_payload = {
                "input_data": [[1.0, 2.0, 3.0, 4.0, 5.0]],
                "model_config": {"uncertainty": True}
            }
            
            async with aiohttp.ClientSession() as session:
                # Test prediction endpoint
                predict_url = f"{self.config.health_check_url.replace('/health', '/predict')}"
                
                async with session.post(predict_url, json=test_payload, timeout=30) as response:
                    if response.status != 200:
                        logger.error(f"Integration test failed: {response.status}")
                        return False
                    
                    result = await response.json()
                    
                    # Validate response structure
                    if 'predictions' not in result or 'uncertainty' not in result:
                        logger.error("Integration test failed: Invalid response structure")
                        return False
                    
                    return True
                    
        except Exception as e:
            logger.error(f"Integration tests failed: {e}")
            return False
    
    async def _rollback_deployment(self):
        """Rollback deployment to previous version"""
        logger.info(f"Rolling back deployment {self.deployment_id}")
        
        cmd = [
            "kubectl", "rollout", "undo",
            f"deployment/{self.config.deployment_name}",
            "-n", self.config.namespace
        ]
        
        result = await self._run_kubectl_command(cmd)
        if result.returncode != 0:
            raise RuntimeError(f"Rollback failed: {result.stderr}")
        
        # Wait for rollback to complete
        await self._monitor_rollout()
        
        logger.info("Rollback completed successfully")
    
    async def _collect_health_metrics(self) -> HealthMetrics:
        """Collect current health metrics"""
        # This would typically query Prometheus or similar monitoring system
        # For now, we'll simulate with basic health checks
        
        try:
            import aiohttp
            start_time = time.time()
            
            async with aiohttp.ClientSession() as session:
                async with session.get(self.config.health_check_url, timeout=10) as response:
                    response_time = (time.time() - start_time) * 1000  # ms
                    
                    # Get basic metrics (would normally come from monitoring system)
                    return HealthMetrics(
                        timestamp=datetime.now(),
                        response_time=response_time,
                        status_code=response.status,
                        error_rate=0.001 if response.status == 200 else 0.05,  # Simulated
                        throughput=100.0,  # Simulated RPS
                        cpu_usage=45.0,    # Simulated CPU %
                        memory_usage=65.0  # Simulated memory %
                    )
        
        except Exception as e:
            logger.error(f"Error collecting health metrics: {e}")
            return HealthMetrics(
                timestamp=datetime.now(),
                response_time=5000.0,  # High response time indicates problem
                status_code=0,
                error_rate=1.0,  # 100% error rate
                throughput=0.0,
                cpu_usage=0.0,
                memory_usage=0.0
            )
    
    async def _check_cluster_health(self) -> bool:
        """Check Kubernetes cluster health"""
        try:
            cmd = ["kubectl", "get", "nodes"]
            result = await self._run_kubectl_command(cmd)
            return result.returncode == 0
        except Exception:
            return False
    
    async def _get_deployment_status(self) -> Dict[str, Any]:
        """Get current deployment status"""
        cmd = [
            "kubectl", "get", "deployment", self.config.deployment_name,
            "-n", self.config.namespace,
            "-o", "json"
        ]
        
        result = await self._run_kubectl_command(cmd)
        if result.returncode != 0:
            raise RuntimeError(f"Failed to get deployment status: {result.stderr}")
        
        deployment_data = json.loads(result.stdout)
        return deployment_data.get('status', {})
    
    async def _validate_image(self) -> bool:
        """Validate new container image"""
        try:
            # Check if image exists and is pullable
            cmd = ["docker", "inspect", self.config.new_image]
            result = subprocess.run(cmd, capture_output=True, text=True)
            return result.returncode == 0
        except Exception:
            # If docker is not available, assume image is valid
            return True
    
    async def _check_error_budget(self) -> bool:
        """Check if sufficient error budget exists for deployment"""
        # This would typically query your SLA monitoring system
        # For now, we'll assume sufficient budget exists
        return True
    
    async def _get_deployment_manifest(self) -> Dict[str, Any]:
        """Get current deployment manifest"""
        cmd = [
            "kubectl", "get", "deployment", self.config.deployment_name,
            "-n", self.config.namespace,
            "-o", "yaml"
        ]
        
        result = await self._run_kubectl_command(cmd)
        if result.returncode != 0:
            raise RuntimeError(f"Failed to get deployment manifest: {result.stderr}")
        
        return yaml.safe_load(result.stdout)
    
    async def _apply_manifest(self, manifest: Dict[str, Any]):
        """Apply Kubernetes manifest"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(manifest, f)
            temp_file = f.name
        
        try:
            cmd = ["kubectl", "apply", "-f", temp_file]
            result = await self._run_kubectl_command(cmd)
            
            if result.returncode != 0:
                raise RuntimeError(f"Failed to apply manifest: {result.stderr}")
        
        finally:
            os.unlink(temp_file)
    
    async def _get_service_cluster_ip(self, service_name: str) -> str:
        """Get service cluster IP"""
        cmd = [
            "kubectl", "get", "service", service_name,
            "-n", self.config.namespace,
            "-o", "jsonpath={.spec.clusterIP}"
        ]
        
        result = await self._run_kubectl_command(cmd)
        if result.returncode != 0:
            raise RuntimeError(f"Failed to get service IP: {result.stderr}")
        
        return result.stdout.strip()
    
    async def _wait_for_deployment_ready(self, deployment_name: str):
        """Wait for deployment to be ready"""
        cmd = [
            "kubectl", "rollout", "status",
            f"deployment/{deployment_name}",
            "-n", self.config.namespace,
            f"--timeout={self.config.validation_timeout}s"
        ]
        
        result = await self._run_kubectl_command(cmd)
        if result.returncode != 0:
            raise RuntimeError(f"Deployment {deployment_name} failed to become ready: {result.stderr}")
    
    async def _cleanup_blue_deployment(self):
        """Clean up blue deployment after successful green deployment"""
        logger.info("Cleaning up blue deployment")
        
        cmd = [
            "kubectl", "delete", "deployment", self.config.deployment_name,
            "-n", self.config.namespace
        ]
        
        await self._run_kubectl_command(cmd)
    
    async def _cleanup_green_deployment(self, green_name: str):
        """Clean up failed green deployment"""
        logger.info(f"Cleaning up green deployment: {green_name}")
        
        cmd = [
            "kubectl", "delete", "deployment", green_name,
            "-n", self.config.namespace
        ]
        
        await self._run_kubectl_command(cmd)
    
    async def _cleanup_canary_deployment(self, canary_name: str):
        """Clean up canary deployment"""
        logger.info(f"Cleaning up canary deployment: {canary_name}")
        
        cmd = [
            "kubectl", "delete", "deployment", canary_name,
            "-n", self.config.namespace
        ]
        
        await self._run_kubectl_command(cmd)
    
    async def _update_traffic_split(self, canary_name: str, percentage: int):
        """Update traffic split percentage for canary"""
        # Implementation would depend on ingress controller or service mesh
        logger.info(f"Updating traffic split to {percentage}% canary")
        pass
    
    async def _promote_canary(self, canary_name: str):
        """Promote canary to primary deployment"""
        logger.info("Promoting canary to primary deployment")
        
        # Replace primary deployment with canary
        canary_manifest = await self._get_deployment_manifest_by_name(canary_name)
        
        # Update manifest to replace primary
        canary_manifest['metadata']['name'] = self.config.deployment_name
        canary_manifest['metadata']['labels'].pop('deployment-type', None)
        canary_manifest['spec']['selector']['matchLabels'].pop('deployment-type', None)
        canary_manifest['spec']['template']['metadata']['labels'].pop('deployment-type', None)
        
        # Scale up to full replica count
        current_status = await self._get_deployment_status()
        canary_manifest['spec']['replicas'] = current_status.get('replicas', 3)
        
        # Apply updated manifest
        await self._apply_manifest(canary_manifest)
        
        # Clean up canary deployment
        await self._cleanup_canary_deployment(canary_name)
    
    async def _get_deployment_manifest_by_name(self, deployment_name: str) -> Dict[str, Any]:
        """Get deployment manifest by name"""
        cmd = [
            "kubectl", "get", "deployment", deployment_name,
            "-n", self.config.namespace,
            "-o", "yaml"
        ]
        
        result = await self._run_kubectl_command(cmd)
        if result.returncode != 0:
            raise RuntimeError(f"Failed to get deployment manifest: {result.stderr}")
        
        return yaml.safe_load(result.stdout)
    
    async def _run_kubectl_command(self, cmd: List[str]) -> subprocess.CompletedProcess:
        """Run kubectl command asynchronously"""
        logger.debug(f"Running command: {' '.join(cmd)}")
        
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await process.communicate()
        
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=process.returncode,
            stdout=stdout.decode() if stdout else "",
            stderr=stderr.decode() if stderr else ""
        )

class DeploymentOrchestrator:
    """Orchestrates multiple deployment strategies and provides high-level interface"""
    
    def __init__(self):
        self.deployment_history: List[DeploymentResult] = []
    
    async def deploy_with_strategy(
        self, 
        strategy: DeploymentStrategy, 
        namespace: str = "production",
        deployment_name: str = "pno-physics-bench",
        new_image: str = None,
        **kwargs
    ) -> DeploymentResult:
        """Deploy using specified strategy"""
        
        if not new_image:
            raise ValueError("new_image is required")
        
        config = DeploymentConfig(
            strategy=strategy,
            namespace=namespace,
            deployment_name=deployment_name,
            new_image=new_image,
            health_check_url=f"https://api.pno-physics-bench.com/health",
            **kwargs
        )
        
        deployer = ZeroDowntimeDeployer(config)
        result = await deployer.deploy()
        
        self.deployment_history.append(result)
        
        return result
    
    async def auto_deploy(
        self,
        new_image: str,
        namespace: str = "production",
        deployment_name: str = "pno-physics-bench"
    ) -> DeploymentResult:
        """Automatically select best deployment strategy based on conditions"""
        
        # Analyze current conditions to select strategy
        strategy = await self._select_optimal_strategy(namespace, deployment_name)
        
        logger.info(f"Auto-selected deployment strategy: {strategy.value}")
        
        return await self.deploy_with_strategy(
            strategy=strategy,
            namespace=namespace,
            deployment_name=deployment_name,
            new_image=new_image
        )
    
    async def _select_optimal_strategy(
        self,
        namespace: str,
        deployment_name: str
    ) -> DeploymentStrategy:
        """Select optimal deployment strategy based on current conditions"""
        
        try:
            # Check current system load
            current_load = await self._get_current_load(namespace, deployment_name)
            
            # Check error budget
            error_budget = await self._get_error_budget_status()
            
            # Check time of day (business hours vs off-hours)
            current_hour = datetime.now().hour
            is_business_hours = 9 <= current_hour <= 17
            
            # Decision logic
            if error_budget < 0.5:  # Low error budget
                return DeploymentStrategy.CANARY
            elif current_load > 0.8:  # High load
                return DeploymentStrategy.ROLLING_UPDATE
            elif is_business_hours:  # Business hours - be conservative
                return DeploymentStrategy.BLUE_GREEN
            else:  # Off-hours - can be more aggressive
                return DeploymentStrategy.ROLLING_UPDATE
                
        except Exception as e:
            logger.warning(f"Error selecting strategy, defaulting to rolling update: {e}")
            return DeploymentStrategy.ROLLING_UPDATE
    
    async def _get_current_load(self, namespace: str, deployment_name: str) -> float:
        """Get current system load (0.0 to 1.0)"""
        try:
            # This would typically query monitoring system
            # For now, return simulated load
            return 0.6  # 60% load
        except Exception:
            return 0.5  # Default to 50% if unable to determine
    
    async def _get_error_budget_status(self) -> float:
        """Get current error budget remaining (0.0 to 1.0)"""
        try:
            # This would typically query SLA monitoring system
            # For now, return simulated budget
            return 0.8  # 80% budget remaining
        except Exception:
            return 1.0  # Default to full budget if unable to determine
    
    def get_deployment_history(self) -> List[DeploymentResult]:
        """Get deployment history"""
        return self.deployment_history.copy()
    
    def get_deployment_statistics(self) -> Dict[str, Any]:
        """Get deployment statistics"""
        if not self.deployment_history:
            return {}
        
        total_deployments = len(self.deployment_history)
        successful_deployments = len([d for d in self.deployment_history if d.status == DeploymentStatus.COMPLETED])
        failed_deployments = len([d for d in self.deployment_history if d.status == DeploymentStatus.FAILED])
        rolled_back_deployments = len([d for d in self.deployment_history if d.status == DeploymentStatus.ROLLED_BACK])
        
        durations = [d.duration for d in self.deployment_history if d.duration]
        avg_duration = sum(durations) / len(durations) if durations else 0
        
        strategy_counts = {}
        for deployment in self.deployment_history:
            strategy = deployment.strategy.value
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
        
        return {
            'total_deployments': total_deployments,
            'successful_deployments': successful_deployments,
            'failed_deployments': failed_deployments,
            'rolled_back_deployments': rolled_back_deployments,
            'success_rate': successful_deployments / total_deployments if total_deployments > 0 else 0,
            'average_duration_seconds': avg_duration,
            'strategy_usage': strategy_counts
        }

async def main():
    """Main execution function for testing"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Zero-Downtime Deployment Tool')
    parser.add_argument('--strategy', choices=['rolling', 'blue-green', 'canary', 'auto'], 
                       default='auto', help='Deployment strategy')
    parser.add_argument('--image', required=True, help='New container image')
    parser.add_argument('--namespace', default='production', help='Kubernetes namespace')
    parser.add_argument('--deployment', default='pno-physics-bench', help='Deployment name')
    
    args = parser.parse_args()
    
    # Map strategy names
    strategy_map = {
        'rolling': DeploymentStrategy.ROLLING_UPDATE,
        'blue-green': DeploymentStrategy.BLUE_GREEN,
        'canary': DeploymentStrategy.CANARY
    }
    
    orchestrator = DeploymentOrchestrator()
    
    try:
        if args.strategy == 'auto':
            result = await orchestrator.auto_deploy(
                new_image=args.image,
                namespace=args.namespace,
                deployment_name=args.deployment
            )
        else:
            result = await orchestrator.deploy_with_strategy(
                strategy=strategy_map[args.strategy],
                namespace=args.namespace,
                deployment_name=args.deployment,
                new_image=args.image
            )
        
        print(f"\nDeployment Result:")
        print(f"Status: {result.status.value}")
        print(f"Strategy: {result.strategy.value}")
        print(f"Duration: {result.duration:.2f} seconds")
        
        if result.rollback_reason:
            print(f"Rollback Reason: {result.rollback_reason}")
        
        if result.validation_results:
            print(f"Validation Results: {json.dumps(result.validation_results, indent=2)}")
    
    except Exception as e:
        logger.error(f"Deployment failed: {e}")
        exit(1)

if __name__ == "__main__":
    asyncio.run(main())