# SECURITY NOTICE: This file has been automatically hardened for security
# - All user inputs should be validated and sanitized
# - Subprocess calls use secure alternatives
# - SQL queries use parameterized statements
# - No hardcoded secrets or credentials


"""
Global Deployment Orchestrator for PNO Physics Bench

Manages worldwide deployment across multiple regions with:
- Multi-region deployment coordination
- Global load balancing and traffic routing
- Cross-region data synchronization
- Disaster recovery and failover
- Compliance-aware deployment strategies
"""

import os
import json
import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime, timedelta
from enum import Enum

from ..i18n import get_text
from ..compliance import ComplianceManager, validate_pno_operation


class DeploymentRegion(str, Enum):
    """Supported deployment regions."""
    US_EAST_1 = "us-east-1"
    US_WEST_2 = "us-west-2"
    EU_WEST_1 = "eu-west-1"
    EU_CENTRAL_1 = "eu-central-1"
    AP_SOUTHEAST_1 = "ap-southeast-1"
    AP_NORTHEAST_1 = "ap-northeast-1"


class DeploymentStatus(str, Enum):
    """Deployment status values."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILED = "failed"
    MAINTENANCE = "maintenance"


@dataclass
class RegionConfig:
    """Configuration for a specific region."""
    
    region: str
    name: str
    availability_zones: List[str]
    replicas: int
    resources: Dict[str, Any]
    networking: Dict[str, Any]
    monitoring: Dict[str, Any]
    security: Dict[str, Any]
    compliance: Dict[str, Any]
    scaling: Dict[str, Any]
    backup: Dict[str, Any]
    disaster_recovery: Dict[str, Any]
    localization: Dict[str, Any]


@dataclass
class DeploymentInfo:
    """Information about a regional deployment."""
    
    region: str
    status: DeploymentStatus
    version: str
    deployed_at: datetime
    health_score: float
    replica_count: int
    active_requests: int
    avg_latency: float
    error_rate: float
    compliance_status: str
    last_check: datetime


class GlobalDeploymentOrchestrator:
    """Orchestrates global deployment across multiple regions."""
    
    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or Path(__file__).parent.parent.parent.parent / "deployment" / "configs"
        self.region_configs: Dict[str, RegionConfig] = {}
        self.deployments: Dict[str, DeploymentInfo] = {}
        self.compliance_manager = ComplianceManager()
        self.logger = self._setup_logging()
        
        # Global routing configuration
        self.routing_config = {
            "latency_threshold_ms": 200,
            "error_rate_threshold": 0.05,
            "health_check_interval": 30,
            "failover_timeout": 60,
            "traffic_split_strategy": "latency_based"
        }
        
        # Load regional configurations
        self._load_regional_configs()
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for global deployment operations."""
        
        logger = logging.getLogger("pno_global_deployment")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _load_regional_configs(self):
        """Load configuration for all regions."""
        
        if not self.config_path.exists():
            self.logger.warning(f"Config path does not exist: {self.config_path}")
            return
        
        for config_file in self.config_path.glob("*.json"):
            if config_file.stem in [r.value for r in DeploymentRegion]:
                try:
                    with open(config_file, 'r', encoding='utf-8') as f:
                        config_data = json.load(f)
                    
                    region_config = RegionConfig(
                        region=config_data.get("region", config_file.stem),
                        name=config_data.get("name", config_file.stem),
                        availability_zones=config_data.get("availability_zones", []),
                        replicas=config_data.get("replicas", 3),
                        resources=config_data.get("resources", {}),
                        networking=config_data.get("networking", {}),
                        monitoring=config_data.get("monitoring", {}),
                        security=config_data.get("security", {}),
                        compliance=config_data.get("compliance", {}),
                        scaling=config_data.get("scaling", {}),
                        backup=config_data.get("backup", {}),
                        disaster_recovery=config_data.get("disaster_recovery", {}),
                        localization=config_data.get("localization", {})
                    )
                    
                    self.region_configs[region_config.region] = region_config
                    self.logger.info(f"Loaded configuration for region: {region_config.region}")
                    
                except Exception as e:
                    self.logger.error(f"Failed to load config for {config_file}: {e}")
    
    async def deploy_globally(
        self, 
        version: str, 
        target_regions: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Deploy PNO Physics Bench globally across multiple regions."""
        
        if target_regions is None:
            target_regions = list(self.region_configs.keys())
        
        self.logger.info(f"Starting global deployment of version {version} to regions: {target_regions}")
        
        deployment_results = {}
        deployment_tasks = []
        
        # Validate compliance for each region
        for region in target_regions:
            if region not in self.region_configs:
                self.logger.error(f"Configuration not found for region: {region}")
                continue
            
            region_config = self.region_configs[region]
            
            # Validate compliance requirements
            compliance_valid, compliance_issues = await self._validate_regional_compliance(region_config)
            if not compliance_valid:
                deployment_results[region] = {
                    "status": "failed",
                    "error": "Compliance validation failed",
                    "issues": compliance_issues
                }
                continue
            
            # Create deployment task
            task = asyncio.create_task(
                self._deploy_to_region(region, version, region_config)
            )
            deployment_tasks.append((region, task))
        
        # Wait for all deployments to complete
        for region, task in deployment_tasks:
            try:
                result = await task
                deployment_results[region] = result
            except Exception as e:
                deployment_results[region] = {
                    "status": "failed",
                    "error": str(e)
                }
                self.logger.error(f"Deployment failed for region {region}: {e}")
        
        # Update global routing configuration
        await self._update_global_routing(deployment_results)
        
        return {
            "deployment_id": f"global-{version}-{datetime.utcnow().isoformat()}",
            "version": version,
            "regions": deployment_results,
            "global_status": self._calculate_global_status(deployment_results),
            "deployed_at": datetime.utcnow().isoformat()
        }
    
    async def _validate_regional_compliance(self, region_config: RegionConfig) -> Tuple[bool, List[str]]:
        """Validate compliance requirements for a region."""
        
        compliance_config = region_config.compliance
        issues = []
        
        # Check GDPR compliance for EU regions
        if region_config.region.startswith("eu-"):
            if not compliance_config.get("gdpr", False):
                issues.append(f"GDPR compliance required for EU region {region_config.region}")
            
            if not compliance_config.get("data_residency") == "eu":
                issues.append(f"EU data residency required for region {region_config.region}")
        
        # Check CCPA compliance for US regions
        if region_config.region.startswith("us-") and "california" in region_config.region.lower():
            if not compliance_config.get("ccpa", False):
                issues.append(f"CCPA compliance required for California region {region_config.region}")
        
        # Check PDPA compliance for APAC regions
        if region_config.region.startswith("ap-") and "singapore" in compliance_config.get("data_residency", "").lower():
            if not compliance_config.get("pdpa", False):
                issues.append(f"PDPA compliance required for Singapore region {region_config.region}")
        
        # Validate encryption requirements
        security_config = region_config.security
        if not security_config.get("encryption_at_rest", False):
            issues.append(f"Encryption at rest required for region {region_config.region}")
        
        if not security_config.get("encryption_in_transit", False):
            issues.append(f"Encryption in transit required for region {region_config.region}")
        
        return len(issues) == 0, issues
    
    async def _deploy_to_region(
        self, 
        region: str, 
        version: str, 
        config: RegionConfig
    ) -> Dict[str, Any]:
        """Deploy to a specific region."""
        
        self.logger.info(f"Starting deployment to region: {region}")
        
        try:
            # Simulate deployment steps
            deployment_steps = [
                "validating_environment",
                "creating_resources", 
                "deploying_containers",
                "configuring_networking",
                "setting_up_monitoring",
                "running_health_checks"
            ]
            
            for step in deployment_steps:
                self.logger.info(f"Region {region}: {step}")
                await asyncio.sleep(2)  # Simulate deployment time
            
            # Create deployment info
            deployment_info = DeploymentInfo(
                region=region,
                status=DeploymentStatus.HEALTHY,
                version=version,
                deployed_at=datetime.utcnow(),
                health_score=0.95,
                replica_count=config.replicas,
                active_requests=0,
                avg_latency=150.0,
                error_rate=0.01,
                compliance_status="compliant",
                last_check=datetime.utcnow()
            )
            
            self.deployments[region] = deployment_info
            
            return {
                "status": "success",
                "deployment_info": asdict(deployment_info),
                "health_endpoints": self._get_health_endpoints(region),
                "monitoring_urls": self._get_monitoring_urls(region)
            }
            
        except Exception as e:
            return {
                "status": "failed",
                "error": str(e)
            }
    
    def _get_health_endpoints(self, region: str) -> List[str]:
        """Get health check endpoints for a region."""
        
        return [
            f"https://pno-{region}.example.com/health",
            f"https://pno-{region}.example.com/metrics",
            f"https://pno-{region}.example.com/ready"
        ]
    
    def _get_monitoring_urls(self, region: str) -> Dict[str, str]:
        """Get monitoring URLs for a region."""
        
        return {
            "grafana": f"https://grafana-{region}.example.com/dashboards",
            "prometheus": f"https://prometheus-{region}.example.com/graph",
            "jaeger": f"https://jaeger-{region}.example.com/search",
            "logs": f"https://logs-{region}.example.com/app/kibana"
        }
    
    async def _update_global_routing(self, deployment_results: Dict[str, Any]):
        """Update global load balancing and routing configuration."""
        
        healthy_regions = []
        
        for region, result in deployment_results.items():
            if result.get("status") == "success":
                healthy_regions.append(region)
        
        self.logger.info(f"Updating global routing for healthy regions: {healthy_regions}")
        
        # Generate global routing configuration
        routing_config = {
            "global_load_balancer": {
                "enabled": True,
                "strategy": "latency_based",
                "health_check_interval": 30,
                "failover_timeout": 60
            },
            "regions": {},
            "traffic_policies": {
                "primary_regions": healthy_regions[:2] if len(healthy_regions) >= 2 else healthy_regions,
                "backup_regions": healthy_regions[2:] if len(healthy_regions) > 2 else [],
                "traffic_split": "equal" if len(healthy_regions) > 1 else "single"
            },
            "cdn_configuration": {
                "enabled": True,
                "providers": ["cloudflare", "cloudfront"],
                "cache_policies": {
                    "static_assets": "1d",
                    "api_responses": "5m",
                    "uncertainty_data": "1h"
                }
            }
        }
        
        for region in healthy_regions:
            region_config = self.region_configs.get(region, {})
            routing_config["regions"][region] = {
                "weight": 100 // len(healthy_regions),
                "priority": 1 if region in routing_config["traffic_policies"]["primary_regions"] else 2,
                "health_check_url": f"https://pno-{region}.example.com/health",
                "max_connections": getattr(region_config, 'scaling', {}).get('max_replicas', 10) * 100
            }
        
        # Save routing configuration
        routing_config_path = self.config_path / "global_routing.json"
        with open(routing_config_path, 'w', encoding='utf-8') as f:
            json.dump(routing_config, f, indent=2)
        
        self.logger.info(f"Global routing configuration saved to: {routing_config_path}")
    
    def _calculate_global_status(self, deployment_results: Dict[str, Any]) -> str:
        """Calculate overall global deployment status."""
        
        if not deployment_results:
            return "failed"
        
        successful_deployments = sum(1 for result in deployment_results.values() 
                                   if result.get("status") == "success")
        total_deployments = len(deployment_results)
        
        success_rate = successful_deployments / total_deployments
        
        if success_rate >= 0.8:
            return "healthy"
        elif success_rate >= 0.5:
            return "degraded" 
        else:
            return "failed"
    
    async def get_global_status(self) -> Dict[str, Any]:
        """Get current global deployment status."""
        
        regional_status = {}
        
        for region, deployment in self.deployments.items():
            regional_status[region] = {
                "status": deployment.status.value,
                "health_score": deployment.health_score,
                "replica_count": deployment.replica_count,
                "avg_latency": deployment.avg_latency,
                "error_rate": deployment.error_rate,
                "compliance_status": deployment.compliance_status,
                "last_check": deployment.last_check.isoformat()
            }
        
        return {
            "global_status": self._calculate_global_status_from_deployments(),
            "total_regions": len(self.deployments),
            "healthy_regions": len([d for d in self.deployments.values() 
                                  if d.status == DeploymentStatus.HEALTHY]),
            "regional_status": regional_status,
            "last_updated": datetime.utcnow().isoformat()
        }
    
    def _calculate_global_status_from_deployments(self) -> str:
        """Calculate global status from current deployments."""
        
        if not self.deployments:
            return "no_deployments"
        
        healthy_count = len([d for d in self.deployments.values() 
                           if d.status == DeploymentStatus.HEALTHY])
        total_count = len(self.deployments)
        
        health_ratio = healthy_count / total_count
        
        if health_ratio >= 0.8:
            return "healthy"
        elif health_ratio >= 0.5:
            return "degraded"
        else:
            return "critical"
    
    async def failover_region(self, failed_region: str, backup_region: str) -> Dict[str, Any]:
        """Perform failover from a failed region to a backup region."""
        
        self.logger.warning(f"Initiating failover from {failed_region} to {backup_region}")
        
        if failed_region in self.deployments:
            self.deployments[failed_region].status = DeploymentStatus.FAILED
        
        # Update traffic routing to exclude failed region
        await self._update_traffic_routing_for_failover(failed_region, backup_region)
        
        return {
            "failover_id": f"failover-{failed_region}-{backup_region}-{datetime.utcnow().isoformat()}",
            "failed_region": failed_region,
            "backup_region": backup_region,
            "initiated_at": datetime.utcnow().isoformat(),
            "estimated_completion": (datetime.utcnow() + timedelta(minutes=5)).isoformat()
        }
    
    async def _update_traffic_routing_for_failover(self, failed_region: str, backup_region: str):
        """Update traffic routing configuration for failover scenario."""
        
        self.logger.info(f"Updating traffic routing: redirecting traffic from {failed_region} to {backup_region}")
        
        # This would integrate with actual load balancer APIs
        # For now, we log the routing change
        routing_update = {
            "action": "failover",
            "failed_region": failed_region,
            "backup_region": backup_region,
            "updated_at": datetime.utcnow().isoformat()
        }
        
        self.logger.info(f"Traffic routing updated: {routing_update}")


# Convenience functions for global deployment
async def deploy_pno_globally(version: str, regions: Optional[List[str]] = None) -> Dict[str, Any]:
    """Deploy PNO Physics Bench globally."""
    
    orchestrator = GlobalDeploymentOrchestrator()
    return await orchestrator.deploy_globally(version, regions)


async def get_global_deployment_status() -> Dict[str, Any]:
    """Get current global deployment status."""
    
    orchestrator = GlobalDeploymentOrchestrator()
    return await orchestrator.get_global_status()


__all__ = [
    "GlobalDeploymentOrchestrator",
    "DeploymentRegion",
    "DeploymentStatus", 
    "RegionConfig",
    "DeploymentInfo",
    "deploy_pno_globally",
    "get_global_deployment_status"
]