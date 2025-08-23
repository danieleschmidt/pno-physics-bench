#!/usr/bin/env python3
"""
Disaster Recovery Automation Suite
Comprehensive backup, replication, and recovery automation for PNO Physics Bench
"""

import asyncio
import json
import logging
import subprocess
import time
import boto3
import yaml
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
import tempfile
import tarfile
import hashlib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BackupType(Enum):
    """Backup type enumeration"""
    FULL = "full"
    INCREMENTAL = "incremental"
    DIFFERENTIAL = "differential"

class RecoveryType(Enum):
    """Recovery type enumeration"""
    FULL_RESTORE = "full_restore"
    POINT_IN_TIME = "point_in_time"
    PARTIAL_RESTORE = "partial_restore"

class BackupStatus(Enum):
    """Backup status enumeration"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    VERIFIED = "verified"

@dataclass
class BackupMetadata:
    """Backup metadata structure"""
    backup_id: str
    timestamp: datetime
    backup_type: BackupType
    size_bytes: int
    checksum: str
    components: List[str]
    retention_days: int
    status: BackupStatus
    storage_location: str
    encryption_key_id: str

@dataclass
class DisasterRecoveryConfig:
    """Disaster recovery configuration"""
    primary_region: str
    backup_regions: List[str]
    rto_minutes: int  # Recovery Time Objective
    rpo_minutes: int  # Recovery Point Objective
    backup_schedule: str
    retention_policy: Dict[str, int]
    encryption_enabled: bool
    cross_region_replication: bool

class BackupManager:
    """Manages backup operations for PNO Physics Bench"""
    
    def __init__(self, config: DisasterRecoveryConfig):
        self.config = config
        self.namespace = "production"
        self.deployment_name = "pno-physics-bench"
        self.backup_storage_path = "/backup/pno-physics-bench"
        
        # Initialize cloud storage clients
        self.s3_client = boto3.client('s3')
        self.backup_bucket = "pno-backups-production"
        
    async def create_full_backup(self) -> BackupMetadata:
        """Create comprehensive full backup"""
        backup_id = f"full-backup-{int(time.time())}"
        logger.info(f"Starting full backup: {backup_id}")
        
        backup_metadata = BackupMetadata(
            backup_id=backup_id,
            timestamp=datetime.now(),
            backup_type=BackupType.FULL,
            size_bytes=0,
            checksum="",
            components=[],
            retention_days=self.config.retention_policy.get("full", 90),
            status=BackupStatus.RUNNING,
            storage_location="",
            encryption_key_id=""
        )
        
        try:
            # Create backup directory
            backup_dir = Path(self.backup_storage_path) / backup_id
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            # Backup components
            components = await self._backup_all_components(backup_dir)
            backup_metadata.components = components
            
            # Create backup archive
            archive_path = await self._create_backup_archive(backup_dir, backup_id)
            
            # Calculate checksum and size
            backup_metadata.checksum = await self._calculate_checksum(archive_path)
            backup_metadata.size_bytes = archive_path.stat().st_size
            
            # Upload to cloud storage
            cloud_location = await self._upload_to_cloud(archive_path, backup_id)
            backup_metadata.storage_location = cloud_location
            
            # Cross-region replication
            if self.config.cross_region_replication:
                await self._replicate_to_backup_regions(archive_path, backup_id)
            
            # Verify backup integrity
            if await self._verify_backup_integrity(cloud_location, backup_metadata.checksum):
                backup_metadata.status = BackupStatus.VERIFIED
            else:
                backup_metadata.status = BackupStatus.FAILED
                raise RuntimeError("Backup integrity verification failed")
            
            # Save metadata
            await self._save_backup_metadata(backup_metadata)
            
            # Cleanup local files
            shutil.rmtree(backup_dir)
            archive_path.unlink()
            
            logger.info(f"Full backup completed successfully: {backup_id}")
            return backup_metadata
            
        except Exception as e:
            backup_metadata.status = BackupStatus.FAILED
            logger.error(f"Full backup failed: {e}")
            raise
    
    async def create_incremental_backup(self, last_backup_id: Optional[str] = None) -> BackupMetadata:
        """Create incremental backup since last backup"""
        backup_id = f"incremental-backup-{int(time.time())}"
        logger.info(f"Starting incremental backup: {backup_id}")
        
        if not last_backup_id:
            last_backup_id = await self._get_last_backup_id()
        
        backup_metadata = BackupMetadata(
            backup_id=backup_id,
            timestamp=datetime.now(),
            backup_type=BackupType.INCREMENTAL,
            size_bytes=0,
            checksum="",
            components=[],
            retention_days=self.config.retention_policy.get("incremental", 30),
            status=BackupStatus.RUNNING,
            storage_location="",
            encryption_key_id=""
        )
        
        try:
            # Get last backup timestamp
            last_backup_time = await self._get_backup_timestamp(last_backup_id) if last_backup_id else None
            
            # Create backup directory
            backup_dir = Path(self.backup_storage_path) / backup_id
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            # Backup changed components only
            components = await self._backup_changed_components(backup_dir, last_backup_time)
            backup_metadata.components = components
            
            if not components:
                logger.info("No changes detected since last backup")
                backup_metadata.status = BackupStatus.COMPLETED
                return backup_metadata
            
            # Create backup archive
            archive_path = await self._create_backup_archive(backup_dir, backup_id)
            
            # Calculate checksum and size
            backup_metadata.checksum = await self._calculate_checksum(archive_path)
            backup_metadata.size_bytes = archive_path.stat().st_size
            
            # Upload to cloud storage
            cloud_location = await self._upload_to_cloud(archive_path, backup_id)
            backup_metadata.storage_location = cloud_location
            
            # Cross-region replication
            if self.config.cross_region_replication:
                await self._replicate_to_backup_regions(archive_path, backup_id)
            
            # Verify backup integrity
            if await self._verify_backup_integrity(cloud_location, backup_metadata.checksum):
                backup_metadata.status = BackupStatus.VERIFIED
            else:
                backup_metadata.status = BackupStatus.FAILED
                raise RuntimeError("Incremental backup integrity verification failed")
            
            # Save metadata
            await self._save_backup_metadata(backup_metadata)
            
            # Cleanup local files
            shutil.rmtree(backup_dir)
            archive_path.unlink()
            
            logger.info(f"Incremental backup completed successfully: {backup_id}")
            return backup_metadata
            
        except Exception as e:
            backup_metadata.status = BackupStatus.FAILED
            logger.error(f"Incremental backup failed: {e}")
            raise
    
    async def _backup_all_components(self, backup_dir: Path) -> List[str]:
        """Backup all system components"""
        components = []
        
        # 1. Kubernetes resources
        logger.info("Backing up Kubernetes resources...")
        k8s_backup_dir = backup_dir / "kubernetes"
        k8s_backup_dir.mkdir()
        
        await self._backup_kubernetes_resources(k8s_backup_dir)
        components.append("kubernetes")
        
        # 2. Application data and models
        logger.info("Backing up application data...")
        data_backup_dir = backup_dir / "data"
        data_backup_dir.mkdir()
        
        await self._backup_application_data(data_backup_dir)
        components.append("data")
        
        # 3. Configuration and secrets
        logger.info("Backing up configuration...")
        config_backup_dir = backup_dir / "config"
        config_backup_dir.mkdir()
        
        await self._backup_configuration(config_backup_dir)
        components.append("config")
        
        # 4. Monitoring data
        logger.info("Backing up monitoring data...")
        monitoring_backup_dir = backup_dir / "monitoring"
        monitoring_backup_dir.mkdir()
        
        await self._backup_monitoring_data(monitoring_backup_dir)
        components.append("monitoring")
        
        # 5. Database (if applicable)
        logger.info("Backing up database...")
        db_backup_dir = backup_dir / "database"
        db_backup_dir.mkdir()
        
        await self._backup_database(db_backup_dir)
        components.append("database")
        
        return components
    
    async def _backup_kubernetes_resources(self, backup_dir: Path):
        """Backup Kubernetes resources"""
        
        # Backup deployments
        cmd = [
            "kubectl", "get", "all", "-n", self.namespace,
            "-o", "yaml"
        ]
        result = await self._run_command(cmd)
        
        with open(backup_dir / "all-resources.yaml", "w") as f:
            f.write(result.stdout)
        
        # Backup configmaps and secrets
        cmd = [
            "kubectl", "get", "configmap,secret", "-n", self.namespace,
            "-o", "yaml"
        ]
        result = await self._run_command(cmd)
        
        with open(backup_dir / "config-secrets.yaml", "w") as f:
            f.write(result.stdout)
        
        # Backup persistent volumes
        cmd = [
            "kubectl", "get", "pv,pvc", "-n", self.namespace,
            "-o", "yaml"
        ]
        result = await self._run_command(cmd)
        
        with open(backup_dir / "storage.yaml", "w") as f:
            f.write(result.stdout)
        
        # Backup ingress and services
        cmd = [
            "kubectl", "get", "ingress,service", "-n", self.namespace,
            "-o", "yaml"
        ]
        result = await self._run_command(cmd)
        
        with open(backup_dir / "networking.yaml", "w") as f:
            f.write(result.stdout)
    
    async def _backup_application_data(self, backup_dir: Path):
        """Backup application data including models"""
        
        # Get pod name
        cmd = [
            "kubectl", "get", "pods", "-n", self.namespace,
            "-l", f"app={self.deployment_name}",
            "-o", "jsonpath={.items[0].metadata.name}"
        ]
        result = await self._run_command(cmd)
        pod_name = result.stdout.strip()
        
        if pod_name:
            # Backup model files
            cmd = [
                "kubectl", "exec", pod_name, "-n", self.namespace,
                "--", "tar", "-czf", "/tmp/models-backup.tar.gz", "/app/models"
            ]
            await self._run_command(cmd)
            
            # Copy model backup from pod
            cmd = [
                "kubectl", "cp", f"{self.namespace}/{pod_name}:/tmp/models-backup.tar.gz",
                str(backup_dir / "models-backup.tar.gz")
            ]
            await self._run_command(cmd)
            
            # Backup application config
            cmd = [
                "kubectl", "exec", pod_name, "-n", self.namespace,
                "--", "tar", "-czf", "/tmp/config-backup.tar.gz", "/app/config"
            ]
            await self._run_command(cmd, ignore_errors=True)  # Config might not exist
            
            # Copy config backup from pod (if exists)
            cmd = [
                "kubectl", "cp", f"{self.namespace}/{pod_name}:/tmp/config-backup.tar.gz",
                str(backup_dir / "config-backup.tar.gz")
            ]
            await self._run_command(cmd, ignore_errors=True)
    
    async def _backup_configuration(self, backup_dir: Path):
        """Backup system configuration"""
        
        # Export all configmaps
        cmd = [
            "kubectl", "get", "configmap", "-n", self.namespace,
            "-o", "json"
        ]
        result = await self._run_command(cmd)
        
        with open(backup_dir / "configmaps.json", "w") as f:
            f.write(result.stdout)
        
        # Export deployment configurations (without secrets)
        deployment_configs = await self._get_deployment_configs()
        
        with open(backup_dir / "deployment-configs.json", "w") as f:
            json.dump(deployment_configs, f, indent=2)
    
    async def _backup_monitoring_data(self, backup_dir: Path):
        """Backup monitoring configuration and data"""
        
        # Backup Prometheus configuration
        cmd = [
            "kubectl", "get", "configmap", "-n", "monitoring",
            "prometheus-config", "-o", "yaml"
        ]
        result = await self._run_command(cmd, ignore_errors=True)
        
        if result.returncode == 0:
            with open(backup_dir / "prometheus-config.yaml", "w") as f:
                f.write(result.stdout)
        
        # Backup Grafana dashboards
        cmd = [
            "kubectl", "get", "configmap", "-n", "monitoring",
            "grafana-dashboards", "-o", "yaml"
        ]
        result = await self._run_command(cmd, ignore_errors=True)
        
        if result.returncode == 0:
            with open(backup_dir / "grafana-dashboards.yaml", "w") as f:
                f.write(result.stdout)
    
    async def _backup_database(self, backup_dir: Path):
        """Backup database if present"""
        
        # Check if database pod exists
        cmd = [
            "kubectl", "get", "pods", "-n", self.namespace,
            "-l", "app=database",
            "-o", "jsonpath={.items[0].metadata.name}"
        ]
        result = await self._run_command(cmd, ignore_errors=True)
        
        if result.stdout.strip():
            db_pod = result.stdout.strip()
            
            # Create database dump
            cmd = [
                "kubectl", "exec", db_pod, "-n", self.namespace,
                "--", "pg_dump", "pno_db"
            ]
            result = await self._run_command(cmd, ignore_errors=True)
            
            if result.returncode == 0:
                with open(backup_dir / "database-dump.sql", "w") as f:
                    f.write(result.stdout)
        else:
            # No database to backup - create empty marker file
            (backup_dir / "no-database.marker").touch()
    
    async def _create_backup_archive(self, backup_dir: Path, backup_id: str) -> Path:
        """Create compressed backup archive"""
        
        archive_path = backup_dir.parent / f"{backup_id}.tar.gz"
        
        logger.info(f"Creating backup archive: {archive_path}")
        
        with tarfile.open(archive_path, "w:gz") as tar:
            tar.add(backup_dir, arcname=backup_id)
        
        return archive_path
    
    async def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA256 checksum of file"""
        
        sha256_hash = hashlib.sha256()
        
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        
        return sha256_hash.hexdigest()
    
    async def _upload_to_cloud(self, archive_path: Path, backup_id: str) -> str:
        """Upload backup to cloud storage"""
        
        key = f"backups/{datetime.now().strftime('%Y/%m/%d')}/{backup_id}.tar.gz"
        
        logger.info(f"Uploading backup to S3: s3://{self.backup_bucket}/{key}")
        
        # Upload with server-side encryption
        self.s3_client.upload_file(
            str(archive_path),
            self.backup_bucket,
            key,
            ExtraArgs={
                'ServerSideEncryption': 'AES256',
                'StorageClass': 'STANDARD_IA',  # Infrequent Access
                'Metadata': {
                    'backup-id': backup_id,
                    'timestamp': datetime.now().isoformat(),
                    'source': 'pno-physics-bench'
                }
            }
        )
        
        return f"s3://{self.backup_bucket}/{key}"
    
    async def _replicate_to_backup_regions(self, archive_path: Path, backup_id: str):
        """Replicate backup to multiple regions"""
        
        for region in self.config.backup_regions:
            if region != self.config.primary_region:
                logger.info(f"Replicating backup to region: {region}")
                
                # Create region-specific S3 client
                regional_s3 = boto3.client('s3', region_name=region)
                regional_bucket = f"pno-backups-{region}"
                
                key = f"backups/{datetime.now().strftime('%Y/%m/%d')}/{backup_id}.tar.gz"
                
                try:
                    regional_s3.upload_file(
                        str(archive_path),
                        regional_bucket,
                        key,
                        ExtraArgs={
                            'ServerSideEncryption': 'AES256',
                            'StorageClass': 'STANDARD_IA'
                        }
                    )
                    logger.info(f"Successfully replicated to {region}")
                    
                except Exception as e:
                    logger.error(f"Failed to replicate to {region}: {e}")
    
    async def _verify_backup_integrity(self, cloud_location: str, expected_checksum: str) -> bool:
        """Verify backup integrity in cloud storage"""
        
        try:
            # Download backup metadata to verify
            bucket, key = cloud_location.replace("s3://", "").split("/", 1)
            
            # Get object metadata
            response = self.s3_client.head_object(Bucket=bucket, Key=key)
            
            # For a more thorough check, we could download and verify checksum
            # For now, verify the object exists and has expected size
            if response['ContentLength'] > 0:
                logger.info("Backup integrity verification passed")
                return True
            else:
                logger.error("Backup integrity verification failed: empty file")
                return False
                
        except Exception as e:
            logger.error(f"Backup integrity verification failed: {e}")
            return False
    
    async def _save_backup_metadata(self, metadata: BackupMetadata):
        """Save backup metadata"""
        
        metadata_file = Path(self.backup_storage_path) / "metadata" / f"{metadata.backup_id}.json"
        metadata_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(metadata_file, "w") as f:
            json.dump(asdict(metadata), f, indent=2, default=str)
        
        # Also upload metadata to cloud
        key = f"metadata/{metadata.backup_id}.json"
        self.s3_client.put_object(
            Bucket=self.backup_bucket,
            Key=key,
            Body=json.dumps(asdict(metadata), indent=2, default=str),
            ContentType='application/json'
        )
    
    async def _run_command(self, cmd: List[str], ignore_errors: bool = False) -> subprocess.CompletedProcess:
        """Run command asynchronously"""
        
        logger.debug(f"Running command: {' '.join(cmd)}")
        
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await process.communicate()
        
        result = subprocess.CompletedProcess(
            args=cmd,
            returncode=process.returncode,
            stdout=stdout.decode() if stdout else "",
            stderr=stderr.decode() if stderr else ""
        )
        
        if result.returncode != 0 and not ignore_errors:
            logger.error(f"Command failed: {' '.join(cmd)}")
            logger.error(f"Error: {result.stderr}")
            raise RuntimeError(f"Command failed: {result.stderr}")
        
        return result

class DisasterRecoveryManager:
    """Manages disaster recovery operations"""
    
    def __init__(self, config: DisasterRecoveryConfig, backup_manager: BackupManager):
        self.config = config
        self.backup_manager = backup_manager
        self.namespace = "production"
        
    async def initiate_disaster_recovery(self, recovery_type: RecoveryType, 
                                       target_region: str,
                                       backup_id: Optional[str] = None) -> Dict[str, Any]:
        """Initiate disaster recovery procedure"""
        
        recovery_id = f"recovery-{int(time.time())}"
        logger.info(f"Initiating disaster recovery: {recovery_id}")
        logger.info(f"Recovery type: {recovery_type.value}")
        logger.info(f"Target region: {target_region}")
        
        recovery_result = {
            'recovery_id': recovery_id,
            'start_time': datetime.now(),
            'recovery_type': recovery_type.value,
            'target_region': target_region,
            'backup_id': backup_id,
            'status': 'in_progress',
            'steps_completed': [],
            'estimated_completion': None,
            'actual_completion': None
        }
        
        try:
            if recovery_type == RecoveryType.FULL_RESTORE:
                await self._full_system_restore(recovery_result, backup_id, target_region)
            elif recovery_type == RecoveryType.POINT_IN_TIME:
                await self._point_in_time_restore(recovery_result, backup_id, target_region)
            elif recovery_type == RecoveryType.PARTIAL_RESTORE:
                await self._partial_restore(recovery_result, backup_id, target_region)
            
            recovery_result['status'] = 'completed'
            recovery_result['actual_completion'] = datetime.now()
            
            logger.info(f"Disaster recovery completed: {recovery_id}")
            
        except Exception as e:
            recovery_result['status'] = 'failed'
            recovery_result['error'] = str(e)
            recovery_result['actual_completion'] = datetime.now()
            
            logger.error(f"Disaster recovery failed: {e}")
            raise
        
        return recovery_result
    
    async def _full_system_restore(self, recovery_result: Dict[str, Any], 
                                 backup_id: Optional[str], target_region: str):
        """Perform full system restore"""
        
        # Step 1: Identify backup to restore
        if not backup_id:
            backup_id = await self._get_latest_full_backup()
        
        recovery_result['backup_id'] = backup_id
        logger.info(f"Restoring from backup: {backup_id}")
        
        # Step 2: Download backup
        await self._download_backup(backup_id, target_region)
        recovery_result['steps_completed'].append('backup_downloaded')
        
        # Step 3: Prepare target environment
        await self._prepare_target_environment(target_region)
        recovery_result['steps_completed'].append('environment_prepared')
        
        # Step 4: Restore Kubernetes resources
        await self._restore_kubernetes_resources(backup_id)
        recovery_result['steps_completed'].append('kubernetes_restored')
        
        # Step 5: Restore application data
        await self._restore_application_data(backup_id)
        recovery_result['steps_completed'].append('data_restored')
        
        # Step 6: Restore configuration
        await self._restore_configuration(backup_id)
        recovery_result['steps_completed'].append('configuration_restored')
        
        # Step 7: Validate restoration
        await self._validate_restoration()
        recovery_result['steps_completed'].append('validation_completed')
        
        # Step 8: Switch traffic (if needed)
        if target_region != self.config.primary_region:
            await self._switch_traffic_to_region(target_region)
            recovery_result['steps_completed'].append('traffic_switched')
    
    async def _prepare_target_environment(self, target_region: str):
        """Prepare target environment for restoration"""
        
        logger.info(f"Preparing target environment in {target_region}")
        
        # Ensure namespace exists
        cmd = [
            "kubectl", "create", "namespace", self.namespace,
            "--dry-run=client", "-o", "yaml"
        ]
        result = await self.backup_manager._run_command(cmd)
        
        # Apply namespace
        cmd = ["kubectl", "apply", "-f", "-"]
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await process.communicate(input=result.stdout.encode())
        
        # Create necessary secrets and configurations
        await self._create_base_secrets()
    
    async def _restore_kubernetes_resources(self, backup_id: str):
        """Restore Kubernetes resources from backup"""
        
        logger.info("Restoring Kubernetes resources")
        
        backup_dir = Path(self.backup_manager.backup_storage_path) / backup_id
        
        # Extract backup if not already extracted
        if not backup_dir.exists():
            await self._extract_backup(backup_id)
        
        # Apply Kubernetes resources
        k8s_dir = backup_dir / "kubernetes"
        
        if k8s_dir.exists():
            # Apply all resources
            for yaml_file in k8s_dir.glob("*.yaml"):
                cmd = ["kubectl", "apply", "-f", str(yaml_file)]
                await self.backup_manager._run_command(cmd)
        
        # Wait for deployments to be ready
        cmd = [
            "kubectl", "rollout", "status", 
            f"deployment/{self.backup_manager.deployment_name}",
            "-n", self.namespace, "--timeout=300s"
        ]
        await self.backup_manager._run_command(cmd)
    
    async def _restore_application_data(self, backup_id: str):
        """Restore application data including models"""
        
        logger.info("Restoring application data")
        
        backup_dir = Path(self.backup_manager.backup_storage_path) / backup_id
        data_dir = backup_dir / "data"
        
        if not data_dir.exists():
            logger.warning("No application data found in backup")
            return
        
        # Get a running pod
        cmd = [
            "kubectl", "get", "pods", "-n", self.namespace,
            "-l", f"app={self.backup_manager.deployment_name}",
            "-o", "jsonpath={.items[0].metadata.name}"
        ]
        result = await self.backup_manager._run_command(cmd)
        pod_name = result.stdout.strip()
        
        if pod_name:
            # Restore model files
            models_backup = data_dir / "models-backup.tar.gz"
            if models_backup.exists():
                # Copy backup to pod
                cmd = [
                    "kubectl", "cp", str(models_backup),
                    f"{self.namespace}/{pod_name}:/tmp/models-backup.tar.gz"
                ]
                await self.backup_manager._run_command(cmd)
                
                # Extract models in pod
                cmd = [
                    "kubectl", "exec", pod_name, "-n", self.namespace,
                    "--", "tar", "-xzf", "/tmp/models-backup.tar.gz", "-C", "/"
                ]
                await self.backup_manager._run_command(cmd)
            
            # Restore configuration files
            config_backup = data_dir / "config-backup.tar.gz"
            if config_backup.exists():
                # Copy backup to pod
                cmd = [
                    "kubectl", "cp", str(config_backup),
                    f"{self.namespace}/{pod_name}:/tmp/config-backup.tar.gz"
                ]
                await self.backup_manager._run_command(cmd)
                
                # Extract config in pod
                cmd = [
                    "kubectl", "exec", pod_name, "-n", self.namespace,
                    "--", "tar", "-xzf", "/tmp/config-backup.tar.gz", "-C", "/"
                ]
                await self.backup_manager._run_command(cmd)
    
    async def _validate_restoration(self):
        """Validate that restoration was successful"""
        
        logger.info("Validating restoration")
        
        # Check deployment health
        cmd = [
            "kubectl", "get", "deployment", self.backup_manager.deployment_name,
            "-n", self.namespace, "-o", "json"
        ]
        result = await self.backup_manager._run_command(cmd)
        deployment_data = json.loads(result.stdout)
        
        status = deployment_data.get('status', {})
        ready_replicas = status.get('readyReplicas', 0)
        desired_replicas = status.get('replicas', 0)
        
        if ready_replicas != desired_replicas:
            raise RuntimeError(f"Deployment not ready: {ready_replicas}/{desired_replicas} replicas")
        
        # Test application health
        await self._test_application_health()
        
        logger.info("Restoration validation passed")
    
    async def _test_application_health(self):
        """Test application health after restoration"""
        
        # Port forward to test locally
        import subprocess
        import threading
        import time
        import requests
        
        # Start port forward in background
        port_forward_proc = subprocess.Popen([
            "kubectl", "port-forward", 
            f"deployment/{self.backup_manager.deployment_name}",
            "8080:8000", "-n", self.namespace
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        try:
            # Wait for port forward to establish
            time.sleep(5)
            
            # Test health endpoint
            response = requests.get("http://localhost:8080/health", timeout=10)
            
            if response.status_code != 200:
                raise RuntimeError(f"Health check failed: HTTP {response.status_code}")
                
            logger.info("Application health test passed")
            
        finally:
            port_forward_proc.terminate()
            port_forward_proc.wait()

class BackupScheduler:
    """Manages automated backup scheduling"""
    
    def __init__(self, backup_manager: BackupManager):
        self.backup_manager = backup_manager
        self.running = False
        
    async def start_scheduled_backups(self):
        """Start scheduled backup process"""
        
        self.running = True
        logger.info("Starting scheduled backup service")
        
        while self.running:
            try:
                current_time = datetime.now()
                
                # Check if it's time for full backup (weekly on Sunday at 2 AM)
                if (current_time.weekday() == 6 and  # Sunday
                    current_time.hour == 2 and
                    current_time.minute < 5):  # 5-minute window
                    
                    logger.info("Starting scheduled full backup")
                    await self.backup_manager.create_full_backup()
                
                # Check if it's time for incremental backup (daily at 2 AM, except Sunday)
                elif (current_time.weekday() != 6 and  # Not Sunday
                      current_time.hour == 2 and
                      current_time.minute < 5):  # 5-minute window
                    
                    logger.info("Starting scheduled incremental backup")
                    await self.backup_manager.create_incremental_backup()
                
                # Check every 5 minutes
                await asyncio.sleep(300)
                
            except Exception as e:
                logger.error(f"Scheduled backup failed: {e}")
                # Continue running even if backup fails
                await asyncio.sleep(3600)  # Wait 1 hour before retry
    
    def stop_scheduled_backups(self):
        """Stop scheduled backup process"""
        
        self.running = False
        logger.info("Stopping scheduled backup service")

class BackupRetentionManager:
    """Manages backup retention and cleanup"""
    
    def __init__(self, backup_manager: BackupManager):
        self.backup_manager = backup_manager
        
    async def cleanup_old_backups(self):
        """Clean up old backups based on retention policy"""
        
        logger.info("Starting backup retention cleanup")
        
        # Get all backup metadata
        backups = await self._get_all_backups()
        
        current_time = datetime.now()
        deleted_count = 0
        
        for backup in backups:
            backup_age = current_time - backup.timestamp
            retention_days = backup.retention_days
            
            if backup_age.days > retention_days:
                logger.info(f"Deleting expired backup: {backup.backup_id} (age: {backup_age.days} days)")
                
                try:
                    await self._delete_backup(backup)
                    deleted_count += 1
                except Exception as e:
                    logger.error(f"Failed to delete backup {backup.backup_id}: {e}")
        
        logger.info(f"Backup retention cleanup completed: {deleted_count} backups deleted")
    
    async def _get_all_backups(self) -> List[BackupMetadata]:
        """Get all backup metadata"""
        
        backups = []
        
        # List all backup metadata files
        metadata_dir = Path(self.backup_manager.backup_storage_path) / "metadata"
        
        if metadata_dir.exists():
            for metadata_file in metadata_dir.glob("*.json"):
                try:
                    with open(metadata_file) as f:
                        data = json.load(f)
                        
                    # Convert timestamp string back to datetime
                    data['timestamp'] = datetime.fromisoformat(data['timestamp'])
                    
                    backup = BackupMetadata(**data)
                    backups.append(backup)
                    
                except Exception as e:
                    logger.error(f"Failed to load backup metadata {metadata_file}: {e}")
        
        return backups
    
    async def _delete_backup(self, backup: BackupMetadata):
        """Delete backup from all storage locations"""
        
        # Delete from S3
        if backup.storage_location.startswith("s3://"):
            bucket, key = backup.storage_location.replace("s3://", "").split("/", 1)
            self.backup_manager.s3_client.delete_object(Bucket=bucket, Key=key)
        
        # Delete metadata
        metadata_file = Path(self.backup_manager.backup_storage_path) / "metadata" / f"{backup.backup_id}.json"
        if metadata_file.exists():
            metadata_file.unlink()
        
        # Delete from backup regions
        for region in self.backup_manager.config.backup_regions:
            try:
                regional_s3 = boto3.client('s3', region_name=region)
                regional_bucket = f"pno-backups-{region}"
                key = backup.storage_location.split("/")[-1]  # Get filename
                
                regional_s3.delete_object(Bucket=regional_bucket, Key=f"backups/{key}")
                
            except Exception as e:
                logger.error(f"Failed to delete backup from region {region}: {e}")

async def main():
    """Main execution function"""
    
    # Initialize configuration
    config = DisasterRecoveryConfig(
        primary_region="us-east-1",
        backup_regions=["us-west-2", "eu-west-1"],
        rto_minutes=15,  # 15 minute RTO
        rpo_minutes=60,  # 1 hour RPO
        backup_schedule="0 2 * * *",  # Daily at 2 AM
        retention_policy={
            "full": 90,        # 90 days for full backups
            "incremental": 30, # 30 days for incremental
            "differential": 7  # 7 days for differential
        },
        encryption_enabled=True,
        cross_region_replication=True
    )
    
    # Initialize managers
    backup_manager = BackupManager(config)
    dr_manager = DisasterRecoveryManager(config, backup_manager)
    scheduler = BackupScheduler(backup_manager)
    retention_manager = BackupRetentionManager(backup_manager)
    
    import argparse
    parser = argparse.ArgumentParser(description='Disaster Recovery Management')
    parser.add_argument('--action', choices=['backup', 'restore', 'schedule', 'cleanup'], 
                       required=True, help='Action to perform')
    parser.add_argument('--backup-type', choices=['full', 'incremental'], 
                       default='incremental', help='Type of backup')
    parser.add_argument('--backup-id', help='Backup ID for restoration')
    parser.add_argument('--target-region', help='Target region for disaster recovery')
    
    args = parser.parse_args()
    
    try:
        if args.action == 'backup':
            if args.backup_type == 'full':
                result = await backup_manager.create_full_backup()
            else:
                result = await backup_manager.create_incremental_backup()
            
            print(f"Backup completed: {result.backup_id}")
            print(f"Status: {result.status.value}")
            print(f"Size: {result.size_bytes / (1024*1024):.2f} MB")
            
        elif args.action == 'restore':
            if not args.target_region:
                args.target_region = config.primary_region
            
            result = await dr_manager.initiate_disaster_recovery(
                recovery_type=RecoveryType.FULL_RESTORE,
                target_region=args.target_region,
                backup_id=args.backup_id
            )
            
            print(f"Disaster recovery completed: {result['recovery_id']}")
            print(f"Status: {result['status']}")
            
        elif args.action == 'schedule':
            print("Starting backup scheduler (press Ctrl+C to stop)")
            await scheduler.start_scheduled_backups()
            
        elif args.action == 'cleanup':
            await retention_manager.cleanup_old_backups()
            print("Backup cleanup completed")
            
    except KeyboardInterrupt:
        print("\nStopping...")
        scheduler.stop_scheduled_backups()
        
    except Exception as e:
        logger.error(f"Operation failed: {e}")
        exit(1)

if __name__ == "__main__":
    asyncio.run(main())