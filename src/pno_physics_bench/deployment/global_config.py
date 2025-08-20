# SECURITY NOTICE: This file has been automatically hardened for security
# - All user inputs should be validated and sanitized
# - Subprocess calls use secure alternatives
# - SQL queries use parameterized statements
# - No hardcoded secrets or credentials


"""
Global Configuration Management for Cross-Platform Deployment

Handles configuration for different deployment environments, regions,
and compliance requirements across cloud providers and on-premises infrastructure.

Key Features:
- Environment-specific configurations (dev, staging, production)
- Region-aware settings for data localization
- Multi-cloud deployment support
- Compliance-aware configuration templates
- Automated environment detection
"""

import os
import json
import yaml
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict, field
from pathlib import Path
from enum import Enum
import logging


class DeploymentEnvironment(Enum):
    """Deployment environment types."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"


class CloudProvider(Enum):
    """Supported cloud providers."""
    AWS = "aws"
    AZURE = "azure"
    GCP = "gcp"
    ON_PREMISES = "on_premises"
    KUBERNETES = "kubernetes"


class ComplianceRegion(Enum):
    """Compliance regions with specific requirements."""
    EU = "eu"           # GDPR
    US = "us"           # Various state laws
    CA_US = "ca_us"     # CCPA
    SG = "sg"           # PDPA
    APAC = "apac"       # General APAC
    GLOBAL = "global"   # Multi-region


@dataclass
class RegionConfig:
    """Configuration for a specific deployment region."""
    
    name: str
    compliance_requirements: List[str] = field(default_factory=list)
    data_residency_required: bool = False
    allowed_data_centers: List[str] = field(default_factory=list)
    encryption_requirements: Dict[str, str] = field(default_factory=dict)
    audit_retention_days: int = 2555  # Default 7 years
    
    # Performance settings
    preferred_instance_types: List[str] = field(default_factory=list)
    auto_scaling_enabled: bool = True
    max_concurrent_requests: int = 1000
    
    # Networking
    vpc_requirements: Dict[str, Any] = field(default_factory=dict)
    ingress_restrictions: List[str] = field(default_factory=list)


@dataclass 
class SecurityConfig:
    """Security configuration settings."""
    
    encryption_at_rest: bool = True
    encryption_in_transit: bool = True
    key_rotation_days: int = 90
    
    # Authentication
    auth_provider: str = "oauth2"
    multi_factor_required: bool = False
    session_timeout_minutes: int = 480  # 8 hours
    
    # Network security
    firewall_enabled: bool = True
    intrusion_detection: bool = True
    ddos_protection: bool = True
    
    # Secrets management
    secrets_provider: str = "aws_secrets_manager"
    secrets_encryption: bool = True


@dataclass
class PerformanceConfig:
    """Performance optimization settings."""
    
    # Model serving
    batch_size_default: int = 32
    max_batch_size: int = 256
    model_cache_size_gb: int = 4
    
    # GPU settings
    gpu_memory_fraction: float = 0.8
    mixed_precision_enabled: bool = True
    tensor_parallelism: bool = False
    
    # Caching
    redis_enabled: bool = False
    memcached_enabled: bool = False
    cache_ttl_seconds: int = 3600
    
    # Monitoring
    metrics_collection_interval: int = 30
    profiling_enabled: bool = False
    distributed_tracing: bool = True


@dataclass
class GlobalDeploymentConfig:
    """Complete global deployment configuration."""
    
    # Environment
    environment: DeploymentEnvironment = DeploymentEnvironment.PRODUCTION
    cloud_provider: CloudProvider = CloudProvider.AWS
    
    # Regional settings
    primary_region: str = "us-west-2"
    regions: Dict[str, RegionConfig] = field(default_factory=dict)
    
    # Core configuration
    security: SecurityConfig = field(default_factory=SecurityConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    
    # Application settings
    app_name: str = "pno-physics-bench"
    app_version: str = "1.0.0"
    
    # Infrastructure
    container_registry: str = ""
    load_balancer_type: str = "application"
    auto_scaling_group: Dict[str, Any] = field(default_factory=dict)
    
    # Monitoring and logging
    log_level: str = "INFO"
    centralized_logging: bool = True
    distributed_monitoring: bool = True
    
    # Feature flags
    feature_flags: Dict[str, bool] = field(default_factory=dict)


class GlobalConfigManager:
    """Manages global configuration across different deployment environments."""
    
    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or Path("configs")
        self.current_config: Optional[GlobalDeploymentConfig] = None
        self.environment_overrides = {}
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Load environment detection
        self.detected_environment = self._detect_environment()
        
    def _detect_environment(self) -> DeploymentEnvironment:
        """Detect current deployment environment."""
        
        # Check environment variables
        env_name = os.getenv("PNO_ENVIRONMENT", "").lower()
        if env_name:
            for env_type in DeploymentEnvironment:
                if env_name == env_type.value:
                    return env_type
        
        # Check for common CI/CD indicators
        if os.getenv("CI") or os.getenv("GITHUB_ACTIONS"):
            return DeploymentEnvironment.TESTING
        
        # Check for cloud provider metadata (simplified)
        if os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION"):
            return DeploymentEnvironment.PRODUCTION
        
        # Default to development
        return DeploymentEnvironment.DEVELOPMENT
    
    def load_config(
        self,
        environment: Optional[DeploymentEnvironment] = None,
        region: Optional[str] = None
    ) -> GlobalDeploymentConfig:
        """Load configuration for specified environment and region."""
        
        target_env = environment or self.detected_environment
        
        # Load base configuration
        base_config = self._load_base_config(target_env)
        
        # Apply region-specific overrides
        if region:
            base_config = self._apply_region_overrides(base_config, region)
        
        # Apply environment-specific overrides
        base_config = self._apply_environment_overrides(base_config, target_env)
        
        # Apply runtime overrides from environment variables
        base_config = self._apply_runtime_overrides(base_config)
        
        self.current_config = base_config
        return base_config
    
    def _load_base_config(self, environment: DeploymentEnvironment) -> GlobalDeploymentConfig:
        """Load base configuration for environment."""
        
        config_file = self.config_path / f"{environment.value}.yaml"
        
        if config_file.exists():
            with open(config_file, 'r') as f:
                config_data = yaml.safe_load(f)
            
            # Convert to GlobalDeploymentConfig
            return self._dict_to_config(config_data, environment)
        else:
            # Return default configuration
            return self._get_default_config(environment)
    
    def _dict_to_config(
        self, 
        config_data: Dict[str, Any], 
        environment: DeploymentEnvironment
    ) -> GlobalDeploymentConfig:
        """Convert dictionary to GlobalDeploymentConfig."""
        
        # Extract main config
        config = GlobalDeploymentConfig(environment=environment)
        
        # Update fields from config data
        if "cloud_provider" in config_data:
            config.cloud_provider = CloudProvider(config_data["cloud_provider"])
        
        if "primary_region" in config_data:
            config.primary_region = config_data["primary_region"]
            
        if "app_name" in config_data:
            config.app_name = config_data["app_name"]
            
        if "app_version" in config_data:
            config.app_version = config_data["app_version"]
        
        # Security configuration
        if "security" in config_data:
            sec_data = config_data["security"]
            config.security = SecurityConfig(**sec_data)
        
        # Performance configuration
        if "performance" in config_data:
            perf_data = config_data["performance"] 
            config.performance = PerformanceConfig(**perf_data)
        
        # Regional configurations
        if "regions" in config_data:
            for region_name, region_data in config_data["regions"].items():
                config.regions[region_name] = RegionConfig(
                    name=region_name,
                    **region_data
                )
        
        return config
    
    def _get_default_config(self, environment: DeploymentEnvironment) -> GlobalDeploymentConfig:
        """Get default configuration for environment."""
        
        config = GlobalDeploymentConfig(environment=environment)
        
        # Environment-specific defaults
        if environment == DeploymentEnvironment.DEVELOPMENT:
            config.security.multi_factor_required = False
            config.performance.profiling_enabled = True
            config.log_level = "DEBUG"
            
        elif environment == DeploymentEnvironment.PRODUCTION:
            config.security.multi_factor_required = True
            config.performance.distributed_tracing = True
            config.log_level = "INFO"
            
        # Add default regions
        config.regions = self._get_default_regions()
        
        return config
    
    def _get_default_regions(self) -> Dict[str, RegionConfig]:
        """Get default regional configurations."""
        
        return {
            "us-west-2": RegionConfig(
                name="us-west-2",
                compliance_requirements=["SOC2", "HIPAA"],
                preferred_instance_types=["c5.xlarge", "c5.2xlarge"],
                max_concurrent_requests=2000
            ),
            "eu-west-1": RegionConfig(
                name="eu-west-1", 
                compliance_requirements=["GDPR", "ISO27001"],
                data_residency_required=True,
                preferred_instance_types=["c5.xlarge", "c5.2xlarge"],
                audit_retention_days=2555
            ),
            "ap-southeast-1": RegionConfig(
                name="ap-southeast-1",
                compliance_requirements=["PDPA"],
                preferred_instance_types=["c5.large", "c5.xlarge"],
                max_concurrent_requests=1500
            )
        }
    
    def _apply_region_overrides(
        self, 
        config: GlobalDeploymentConfig, 
        region: str
    ) -> GlobalDeploymentConfig:
        """Apply region-specific configuration overrides."""
        
        if region in config.regions:
            region_config = config.regions[region]
            
            # Update primary region
            config.primary_region = region
            
            # Apply region-specific security settings
            if "GDPR" in region_config.compliance_requirements:
                config.security.encryption_at_rest = True
                config.security.key_rotation_days = 30  # More frequent rotation
                
            if region_config.data_residency_required:
                config.feature_flags["data_localization"] = True
        
        return config
    
    def _apply_environment_overrides(
        self,
        config: GlobalDeploymentConfig,
        environment: DeploymentEnvironment
    ) -> GlobalDeploymentConfig:
        """Apply environment-specific overrides."""
        
        if environment == DeploymentEnvironment.DEVELOPMENT:
            # Development optimizations
            config.security.encryption_at_rest = False
            config.performance.profiling_enabled = True
            config.performance.cache_ttl_seconds = 60  # Short cache for development
            
        elif environment == DeploymentEnvironment.PRODUCTION:
            # Production hardening
            config.security.intrusion_detection = True
            config.security.ddos_protection = True
            config.performance.model_cache_size_gb = 8
            
        return config
    
    def _apply_runtime_overrides(
        self,
        config: GlobalDeploymentConfig
    ) -> GlobalDeploymentConfig:
        """Apply runtime overrides from environment variables."""
        
        # Override from environment variables
        if os.getenv("PNO_LOG_LEVEL"):
            config.log_level = os.getenv("PNO_LOG_LEVEL")
            
        if os.getenv("PNO_BATCH_SIZE"):
            try:
                config.performance.batch_size_default = int(os.getenv("PNO_BATCH_SIZE"))
            except ValueError:
                pass
                
        if os.getenv("PNO_GPU_MEMORY_FRACTION"):
            try:
                config.performance.gpu_memory_fraction = float(os.getenv("PNO_GPU_MEMORY_FRACTION"))
            except ValueError:
                pass
        
        return config
    
    def save_config(
        self,
        config: GlobalDeploymentConfig,
        environment: Optional[DeploymentEnvironment] = None
    ):
        """Save configuration to file."""
        
        target_env = environment or config.environment
        config_file = self.config_path / f"{target_env.value}.yaml"
        
        # Ensure config directory exists
        config_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to dictionary and save
        config_dict = self._config_to_dict(config)
        
        with open(config_file, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
    
    def _config_to_dict(self, config: GlobalDeploymentConfig) -> Dict[str, Any]:
        """Convert GlobalDeploymentConfig to dictionary for serialization."""
        
        config_dict = {
            "environment": config.environment.value,
            "cloud_provider": config.cloud_provider.value,
            "primary_region": config.primary_region,
            "app_name": config.app_name,
            "app_version": config.app_version,
            "log_level": config.log_level,
            "centralized_logging": config.centralized_logging,
            "distributed_monitoring": config.distributed_monitoring,
            "security": asdict(config.security),
            "performance": asdict(config.performance),
            "feature_flags": config.feature_flags,
            "regions": {}
        }
        
        # Convert regions
        for region_name, region_config in config.regions.items():
            config_dict["regions"][region_name] = asdict(region_config)
        
        return config_dict
    
    def get_region_config(self, region: str) -> Optional[RegionConfig]:
        """Get configuration for specific region."""
        
        if self.current_config and region in self.current_config.regions:
            return self.current_config.regions[region]
        
        return None
    
    def validate_config(self, config: GlobalDeploymentConfig) -> List[str]:
        """Validate configuration and return list of issues."""
        
        issues = []
        
        # Security validation
        if config.environment == DeploymentEnvironment.PRODUCTION:
            if not config.security.encryption_at_rest:
                issues.append("Production environments must enable encryption at rest")
            
            if not config.security.intrusion_detection:
                issues.append("Production environments should enable intrusion detection")
        
        # Performance validation
        if config.performance.batch_size_default > config.performance.max_batch_size:
            issues.append("Default batch size cannot exceed maximum batch size")
        
        if config.performance.gpu_memory_fraction > 1.0:
            issues.append("GPU memory fraction cannot exceed 1.0")
        
        # Regional validation
        for region_name, region_config in config.regions.items():
            if region_config.data_residency_required and not region_config.allowed_data_centers:
                issues.append(f"Region {region_name} requires data residency but no allowed data centers specified")
        
        return issues
    
    def create_deployment_manifest(
        self,
        config: GlobalDeploymentConfig,
        target_platform: str = "kubernetes"
    ) -> Dict[str, Any]:
        """Create deployment manifest for target platform."""
        
        if target_platform == "kubernetes":
            return self._create_k8s_manifest(config)
        elif target_platform == "docker-compose":
            return self._create_docker_compose_manifest(config)
        else:
            raise ValueError(f"Unsupported deployment platform: {target_platform}")
    
    def _create_k8s_manifest(self, config: GlobalDeploymentConfig) -> Dict[str, Any]:
        """Create Kubernetes deployment manifest."""
        
        manifest = {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": config.app_name,
                "labels": {
                    "app": config.app_name,
                    "version": config.app_version,
                    "environment": config.environment.value
                }
            },
            "spec": {
                "replicas": 3,  # Default replicas
                "selector": {
                    "matchLabels": {
                        "app": config.app_name
                    }
                },
                "template": {
                    "metadata": {
                        "labels": {
                            "app": config.app_name,
                            "version": config.app_version
                        }
                    },
                    "spec": {
                        "containers": [{
                            "name": config.app_name,
                            "image": f"{config.container_registry}/{config.app_name}:{config.app_version}",
                            "ports": [{
                                "containerPort": 8000,
                                "name": "http"
                            }],
                            "env": [
                                {
                                    "name": "PNO_ENVIRONMENT",
                                    "value": config.environment.value
                                },
                                {
                                    "name": "PNO_LOG_LEVEL", 
                                    "value": config.log_level
                                },
                                {
                                    "name": "PNO_BATCH_SIZE",
                                    "value": str(config.performance.batch_size_default)
                                }
                            ],
                            "resources": {
                                "limits": {
                                    "memory": "4Gi",
                                    "cpu": "2000m"
                                },
                                "requests": {
                                    "memory": "2Gi", 
                                    "cpu": "1000m"
                                }
                            }
                        }]
                    }
                }
            }
        }
        
        return manifest
    
    def _create_docker_compose_manifest(self, config: GlobalDeploymentConfig) -> Dict[str, Any]:
        """Create Docker Compose manifest."""
        
        manifest = {
            "version": "3.8",
            "services": {
                config.app_name: {
                    "image": f"{config.container_registry}/{config.app_name}:{config.app_version}",
                    "ports": ["8000:8000"],
                    "environment": {
                        "PNO_ENVIRONMENT": config.environment.value,
                        "PNO_LOG_LEVEL": config.log_level,
                        "PNO_BATCH_SIZE": config.performance.batch_size_default
                    },
                    "deploy": {
                        "resources": {
                            "limits": {
                                "memory": "4G",
                                "cpus": "2.0"
                            }
                        }
                    }
                }
            }
        }
        
        # Add Redis if caching enabled
        if config.performance.redis_enabled:
            manifest["services"]["redis"] = {
                "image": "redis:7-alpine",
                "ports": ["6379:6379"]
            }
        
        return manifest


# Convenience functions
def load_global_config(
    environment: Optional[str] = None,
    region: Optional[str] = None,
    config_path: Optional[str] = None
) -> GlobalDeploymentConfig:
    """Load global configuration with convenience function."""
    
    config_manager = GlobalConfigManager(Path(config_path) if config_path else None)
    
    env_enum = None
    if environment:
        for env_type in DeploymentEnvironment:
            if environment.lower() == env_type.value:
                env_enum = env_type
                break
    
    return config_manager.load_config(env_enum, region)


def create_region_aware_config(
    base_region: str,
    compliance_requirements: List[str]
) -> RegionConfig:
    """Create region-aware configuration with compliance requirements."""
    
    return RegionConfig(
        name=base_region,
        compliance_requirements=compliance_requirements,
        data_residency_required="GDPR" in compliance_requirements,
        audit_retention_days=2555 if "GDPR" in compliance_requirements else 365
    )


__all__ = [
    "DeploymentEnvironment",
    "CloudProvider", 
    "ComplianceRegion",
    "RegionConfig",
    "SecurityConfig",
    "PerformanceConfig", 
    "GlobalDeploymentConfig",
    "GlobalConfigManager",
    "load_global_config",
    "create_region_aware_config"
]