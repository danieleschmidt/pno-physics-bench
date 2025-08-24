# SECURITY NOTICE: This file has been automatically hardened for security
# - All user inputs should be validated and sanitized
# - Subprocess calls use secure alternatives
# - SQL queries use parameterized statements
# - No hardcoded secrets or credentials

"""
Production Infrastructure Management and Robustness
Generation 2 Robustness Enhancement
"""

import atexit
import gc
import json
import logging
import os
import signal
import threading
import time
import weakref
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import uuid

logger = logging.getLogger(__name__)


class ShutdownPhase(Enum):
    """Phases of graceful shutdown process."""
    NORMAL_OPERATION = auto()
    SHUTDOWN_INITIATED = auto()
    DRAINING_REQUESTS = auto()
    CLEANUP_RESOURCES = auto()
    FINAL_CLEANUP = auto()
    SHUTDOWN_COMPLETE = auto()


class ResourceType(Enum):
    """Types of system resources to manage."""
    MEMORY = auto()
    CPU = auto()
    GPU = auto()
    DISK = auto()
    NETWORK = auto()
    FILE_HANDLES = auto()
    THREADS = auto()


@dataclass
class ResourceUsage:
    """Resource usage tracking."""
    resource_type: ResourceType
    current_usage: float
    peak_usage: float
    limit: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConfigurationSchema:
    """Schema for configuration validation."""
    required_keys: Set[str] = field(default_factory=set)
    optional_keys: Set[str] = field(default_factory=set)
    type_constraints: Dict[str, type] = field(default_factory=dict)
    value_constraints: Dict[str, Callable[[Any], bool]] = field(default_factory=dict)
    custom_validators: List[Callable[[Dict[str, Any]], List[str]]] = field(default_factory=list)


class GracefulShutdownManager:
    """Manages graceful shutdown of the entire system."""
    
    def __init__(self, shutdown_timeout: float = 30.0):
        self.shutdown_timeout = shutdown_timeout
        self.shutdown_phase = ShutdownPhase.NORMAL_OPERATION
        self.shutdown_callbacks = defaultdict(list)  # phase -> callbacks
        self.active_requests = set()
        self.shutdown_initiated = False
        self._lock = threading.RLock()
        self._shutdown_event = threading.Event()
        
        # Register signal handlers
        self._setup_signal_handlers()
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating graceful shutdown")
            self.initiate_shutdown()
        
        # Register shutdown signals
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
        
        # Register cleanup on normal exit
        atexit.register(self._emergency_cleanup)
    
    def register_shutdown_callback(self, phase: ShutdownPhase, callback: Callable[[], None],
                                 priority: int = 0):
        """Register callback for specific shutdown phase.
        
        Args:
            phase: Shutdown phase to run callback in
            callback: Function to call during shutdown
            priority: Priority (higher numbers run first)
        """
        self.shutdown_callbacks[phase].append((priority, callback))
        # Sort by priority (descending)
        self.shutdown_callbacks[phase].sort(key=lambda x: x[0], reverse=True)
    
    def register_active_request(self, request_id: str):
        """Register an active request."""
        with self._lock:
            self.active_requests.add(request_id)
    
    def unregister_active_request(self, request_id: str):
        """Unregister an active request."""
        with self._lock:
            self.active_requests.discard(request_id)
    
    def initiate_shutdown(self):
        """Initiate graceful shutdown process."""
        if self.shutdown_initiated:
            return
        
        with self._lock:
            self.shutdown_initiated = True
            self.shutdown_phase = ShutdownPhase.SHUTDOWN_INITIATED
        
        logger.info("Graceful shutdown initiated")
        
        # Run shutdown in separate thread to avoid blocking
        shutdown_thread = threading.Thread(target=self._execute_shutdown, daemon=False)
        shutdown_thread.start()
    
    def _execute_shutdown(self):
        """Execute the complete shutdown process."""
        try:
            # Phase 1: Stop accepting new requests
            self._change_phase(ShutdownPhase.SHUTDOWN_INITIATED)
            
            # Phase 2: Drain existing requests
            self._change_phase(ShutdownPhase.DRAINING_REQUESTS)
            self._drain_active_requests()
            
            # Phase 3: Cleanup resources
            self._change_phase(ShutdownPhase.CLEANUP_RESOURCES)
            
            # Phase 4: Final cleanup
            self._change_phase(ShutdownPhase.FINAL_CLEANUP)
            
            # Phase 5: Shutdown complete
            self._change_phase(ShutdownPhase.SHUTDOWN_COMPLETE)
            
            logger.info("Graceful shutdown completed successfully")
            
        except Exception as e:
            logger.error(f"Error during graceful shutdown: {e}")
            self._emergency_cleanup()
    
    def _change_phase(self, new_phase: ShutdownPhase):
        """Change shutdown phase and execute callbacks."""
        with self._lock:
            old_phase = self.shutdown_phase
            self.shutdown_phase = new_phase
        
        logger.info(f"Shutdown phase: {old_phase.name} -> {new_phase.name}")
        
        # Execute callbacks for this phase
        for priority, callback in self.shutdown_callbacks[new_phase]:
            try:
                logger.debug(f"Executing shutdown callback (priority {priority})")
                callback()
            except Exception as e:
                logger.error(f"Shutdown callback failed: {e}")
    
    def _drain_active_requests(self):
        """Wait for active requests to complete."""
        drain_start = time.time()
        
        while self.active_requests and time.time() - drain_start < self.shutdown_timeout:
            logger.info(f"Waiting for {len(self.active_requests)} active requests to complete")
            time.sleep(1.0)
        
        if self.active_requests:
            logger.warning(f"Shutdown timeout reached, {len(self.active_requests)} requests still active")
    
    def _emergency_cleanup(self):
        """Emergency cleanup when graceful shutdown fails."""
        logger.warning("Performing emergency cleanup")
        
        try:
            # Force cleanup of critical resources
            gc.collect()
            
            # Clear GPU memory if available
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
            
        except Exception as e:
            logger.error(f"Emergency cleanup failed: {e}")
    
    @contextmanager
    def track_request(self, request_id: Optional[str] = None):
        """Context manager to track active requests."""
        request_id = request_id or str(uuid.uuid4())[:8]
        
        if self.shutdown_initiated:
            raise RuntimeError("System is shutting down, not accepting new requests")
        
        self.register_active_request(request_id)
        try:
            yield request_id
        finally:
            self.unregister_active_request(request_id)


class MemoryManager:
    """Advanced memory management and monitoring."""
    
    def __init__(self):
        self.memory_pools = {}
        self.allocation_tracking = defaultdict(list)
        self.memory_limits = {
            'max_total_mb': 4096,      # 4GB
            'max_single_allocation_mb': 1024,  # 1GB
            'gc_threshold_mb': 3072,   # 3GB
            'warning_threshold_mb': 2048  # 2GB
        }
        self.gc_stats = {'forced_collections': 0, 'objects_collected': 0}
        self._lock = threading.RLock()
        self._monitoring_thread = None
        self.start_monitoring()
    
    def start_monitoring(self):
        """Start memory monitoring thread."""
        if self._monitoring_thread is None or not self._monitoring_thread.is_alive():
            self._monitoring_thread = threading.Thread(
                target=self._memory_monitoring_worker,
                daemon=True
            )
            self._monitoring_thread.start()
    
    def allocate_memory_pool(self, pool_name: str, size_mb: float) -> bool:
        """Allocate a named memory pool."""
        if size_mb > self.memory_limits['max_single_allocation_mb']:
            logger.error(f"Allocation too large: {size_mb}MB > {self.memory_limits['max_single_allocation_mb']}MB")
            return False
        
        current_usage = self.get_current_memory_usage()
        if current_usage + size_mb > self.memory_limits['max_total_mb']:
            logger.warning(f"Allocation would exceed memory limit, triggering garbage collection")
            self.force_garbage_collection()
            
            # Check again after GC
            current_usage = self.get_current_memory_usage()
            if current_usage + size_mb > self.memory_limits['max_total_mb']:
                logger.error(f"Cannot allocate {size_mb}MB, would exceed limit")
                return False
        
        with self._lock:
            self.memory_pools[pool_name] = {
                'size_mb': size_mb,
                'allocated_at': datetime.now(),
                'last_accessed': datetime.now()
            }
            
            self.allocation_tracking[pool_name].append({
                'timestamp': datetime.now(),
                'action': 'allocate',
                'size_mb': size_mb
            })
        
        logger.info(f"Allocated memory pool '{pool_name}': {size_mb}MB")
        return True
    
    def release_memory_pool(self, pool_name: str):
        """Release a named memory pool."""
        with self._lock:
            if pool_name in self.memory_pools:
                pool_info = self.memory_pools.pop(pool_name)
                self.allocation_tracking[pool_name].append({
                    'timestamp': datetime.now(),
                    'action': 'release',
                    'size_mb': pool_info['size_mb']
                })
                logger.info(f"Released memory pool '{pool_name}': {pool_info['size_mb']}MB")
                
                # Trigger garbage collection after large releases
                if pool_info['size_mb'] > 100:
                    self.force_garbage_collection()
    
    def get_current_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / (1024**2)
        except ImportError:
            # Fallback to sum of tracked pools
            with self._lock:
                return sum(pool['size_mb'] for pool in self.memory_pools.values())
    
    def force_garbage_collection(self) -> Dict[str, int]:
        """Force garbage collection and return statistics."""
        logger.info("Forcing garbage collection")
        
        # Clear GPU memory first if available
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.debug("Cleared GPU memory cache")
        except ImportError:
            pass
        
        # Run garbage collection
        collected_objects = gc.collect()
        self.gc_stats['forced_collections'] += 1
        self.gc_stats['objects_collected'] += collected_objects
        
        logger.info(f"Garbage collection completed: {collected_objects} objects collected")
        
        return {
            'objects_collected': collected_objects,
            'total_collections': self.gc_stats['forced_collections'],
            'total_objects_collected': self.gc_stats['objects_collected']
        }
    
    def get_memory_report(self) -> Dict[str, Any]:
        """Get comprehensive memory usage report."""
        current_usage = self.get_current_memory_usage()
        
        with self._lock:
            pool_usage = {
                name: info['size_mb'] 
                for name, info in self.memory_pools.items()
            }
        
        return {
            'timestamp': datetime.now().isoformat(),
            'current_usage_mb': current_usage,
            'usage_percent': (current_usage / self.memory_limits['max_total_mb']) * 100,
            'memory_pools': pool_usage,
            'limits': self.memory_limits,
            'gc_stats': self.gc_stats.copy(),
            'status': self._get_memory_status(current_usage)
        }
    
    def _get_memory_status(self, usage_mb: float) -> str:
        """Get memory status based on usage."""
        if usage_mb > self.memory_limits['gc_threshold_mb']:
            return 'CRITICAL'
        elif usage_mb > self.memory_limits['warning_threshold_mb']:
            return 'WARNING'
        else:
            return 'HEALTHY'
    
    def _memory_monitoring_worker(self):
        """Background worker for memory monitoring."""
        while True:
            try:
                usage = self.get_current_memory_usage()
                
                # Check if we need to trigger garbage collection
                if usage > self.memory_limits['gc_threshold_mb']:
                    logger.warning(f"Memory usage high ({usage:.1f}MB), triggering GC")
                    self.force_garbage_collection()
                elif usage > self.memory_limits['warning_threshold_mb']:
                    logger.info(f"Memory usage elevated: {usage:.1f}MB")
                
                # Clean up old allocations
                self._cleanup_stale_pools()
                
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Memory monitoring error: {e}")
                time.sleep(60)  # Wait longer on error
    
    def _cleanup_stale_pools(self):
        """Clean up stale memory pools that haven't been accessed."""
        stale_threshold = datetime.now() - timedelta(hours=1)
        stale_pools = []
        
        with self._lock:
            for pool_name, pool_info in self.memory_pools.items():
                if pool_info['last_accessed'] < stale_threshold:
                    stale_pools.append(pool_name)
        
        for pool_name in stale_pools:
            logger.info(f"Cleaning up stale memory pool: {pool_name}")
            self.release_memory_pool(pool_name)


class ConfigurationManager:
    """Advanced configuration management with validation and hot reloading."""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file or '/root/repo/config/production.json'
        self.config = {}
        self.config_schema = None
        self.config_history = deque(maxlen=100)
        self.validation_callbacks = []
        self.change_callbacks = []
        self._lock = threading.RLock()
        self._file_watcher_thread = None
        
        # Load initial configuration
        self.load_configuration()
        self.start_file_watching()
    
    def set_schema(self, schema: ConfigurationSchema):
        """Set configuration schema for validation."""
        self.config_schema = schema
        
        # Validate current configuration against schema
        if self.config:
            validation_errors = self.validate_configuration(self.config)
            if validation_errors:
                logger.warning(f"Current configuration has issues: {validation_errors}")
    
    def load_configuration(self) -> bool:
        """Load configuration from file."""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    new_config = json.load(f)
                
                # Validate new configuration
                if self.config_schema:
                    validation_errors = self.validate_configuration(new_config)
                    if validation_errors:
                        logger.error(f"Configuration validation failed: {validation_errors}")
                        return False
                
                with self._lock:
                    old_config = self.config.copy()
                    self.config = new_config
                    
                    # Record configuration change
                    self.config_history.append({
                        'timestamp': datetime.now(),
                        'action': 'loaded',
                        'changes': self._get_config_changes(old_config, new_config)
                    })
                
                # Notify change callbacks
                for callback in self.change_callbacks:
                    try:
                        callback(old_config, new_config)
                    except Exception as e:
                        logger.error(f"Configuration change callback failed: {e}")
                
                logger.info("Configuration loaded successfully")
                return True
            else:
                logger.warning(f"Configuration file not found: {self.config_file}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            return False
    
    def validate_configuration(self, config: Dict[str, Any]) -> List[str]:
        """Validate configuration against schema."""
        if not self.config_schema:
            return []
        
        errors = []
        
        # Check required keys
        for key in self.config_schema.required_keys:
            if key not in config:
                errors.append(f"Missing required key: {key}")
        
        # Check unknown keys
        known_keys = self.config_schema.required_keys | self.config_schema.optional_keys
        for key in config:
            if key not in known_keys:
                errors.append(f"Unknown configuration key: {key}")
        
        # Check type constraints
        for key, expected_type in self.config_schema.type_constraints.items():
            if key in config and not isinstance(config[key], expected_type):
                errors.append(f"Type error for {key}: expected {expected_type.__name__}, got {type(config[key]).__name__}")
        
        # Check value constraints
        for key, validator in self.config_schema.value_constraints.items():
            if key in config:
                try:
                    if not validator(config[key]):
                        errors.append(f"Value constraint failed for {key}")
                except Exception as e:
                    errors.append(f"Value validation error for {key}: {e}")
        
        # Run custom validators
        for validator in self.config_schema.custom_validators:
            try:
                custom_errors = validator(config)
                errors.extend(custom_errors)
            except Exception as e:
                errors.append(f"Custom validator error: {e}")
        
        return errors
    
    def get_config_value(self, key: str, default: Any = None) -> Any:
        """Get configuration value with default."""
        with self._lock:
            return self.config.get(key, default)
    
    def update_config_value(self, key: str, value: Any) -> bool:
        """Update a single configuration value."""
        with self._lock:
            old_config = self.config.copy()
            new_config = self.config.copy()
            new_config[key] = value
            
            # Validate updated configuration
            if self.config_schema:
                validation_errors = self.validate_configuration(new_config)
                if validation_errors:
                    logger.error(f"Configuration update validation failed: {validation_errors}")
                    return False
            
            self.config = new_config
            
            # Record change
            self.config_history.append({
                'timestamp': datetime.now(),
                'action': 'updated',
                'key': key,
                'old_value': old_config.get(key),
                'new_value': value
            })
        
        logger.info(f"Configuration updated: {key} = {value}")
        return True
    
    def _get_config_changes(self, old_config: Dict, new_config: Dict) -> List[Dict[str, Any]]:
        """Get list of configuration changes."""
        changes = []
        
        # Check for added/modified keys
        for key, value in new_config.items():
            if key not in old_config:
                changes.append({'type': 'added', 'key': key, 'new_value': value})
            elif old_config[key] != value:
                changes.append({
                    'type': 'modified',
                    'key': key,
                    'old_value': old_config[key],
                    'new_value': value
                })
        
        # Check for removed keys
        for key in old_config:
            if key not in new_config:
                changes.append({'type': 'removed', 'key': key, 'old_value': old_config[key]})
        
        return changes
    
    def start_file_watching(self):
        """Start watching configuration file for changes."""
        if self._file_watcher_thread is None or not self._file_watcher_thread.is_alive():
            self._file_watcher_thread = threading.Thread(
                target=self._file_watcher_worker,
                daemon=True
            )
            self._file_watcher_thread.start()
    
    def _file_watcher_worker(self):
        """Background worker for file watching."""
        last_mtime = None
        
        while True:
            try:
                if os.path.exists(self.config_file):
                    current_mtime = os.path.getmtime(self.config_file)
                    
                    if last_mtime is not None and current_mtime != last_mtime:
                        logger.info("Configuration file changed, reloading")
                        self.load_configuration()
                    
                    last_mtime = current_mtime
                
                time.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                logger.error(f"Configuration file watching error: {e}")
                time.sleep(30)  # Wait longer on error
    
    def register_change_callback(self, callback: Callable[[Dict, Dict], None]):
        """Register callback for configuration changes."""
        self.change_callbacks.append(callback)


class ResourceMonitor:
    """Monitors and manages system resources."""
    
    def __init__(self):
        self.resource_usage = defaultdict(lambda: deque(maxlen=1000))
        self.resource_limits = {
            ResourceType.MEMORY: 90.0,      # 90% memory usage
            ResourceType.CPU: 85.0,         # 85% CPU usage
            ResourceType.DISK: 85.0,        # 85% disk usage
            ResourceType.GPU: 90.0,         # 90% GPU memory
        }
        self.alert_callbacks = defaultdict(list)
        self._monitoring_active = False
        self._lock = threading.RLock()
    
    def start_monitoring(self):
        """Start resource monitoring."""
        if not self._monitoring_active:
            self._monitoring_active = True
            monitoring_thread = threading.Thread(
                target=self._monitoring_worker,
                daemon=True
            )
            monitoring_thread.start()
    
    def stop_monitoring(self):
        """Stop resource monitoring."""
        self._monitoring_active = False
    
    def record_resource_usage(self, resource_type: ResourceType, usage: float,
                            metadata: Optional[Dict[str, Any]] = None):
        """Record resource usage measurement."""
        usage_record = ResourceUsage(
            resource_type=resource_type,
            current_usage=usage,
            peak_usage=usage,  # Will be updated by monitoring
            metadata=metadata or {}
        )
        
        with self._lock:
            self.resource_usage[resource_type].append(usage_record)
            
            # Check for alerts
            limit = self.resource_limits.get(resource_type)
            if limit and usage > limit:
                self._trigger_resource_alert(resource_type, usage, limit)
    
    def _trigger_resource_alert(self, resource_type: ResourceType, usage: float, limit: float):
        """Trigger resource usage alert."""
        logger.warning(f"Resource alert: {resource_type.name} usage {usage:.1f}% > {limit:.1f}%")
        
        for callback in self.alert_callbacks[resource_type]:
            try:
                callback(resource_type, usage, limit)
            except Exception as e:
                logger.error(f"Resource alert callback failed: {e}")
    
    def get_resource_summary(self) -> Dict[str, Any]:
        """Get summary of resource usage."""
        summary = {}
        
        with self._lock:
            for resource_type in ResourceType:
                if resource_type in self.resource_usage:
                    recent_usage = list(self.resource_usage[resource_type])[-10:]  # Last 10 measurements
                    
                    if recent_usage:
                        current_values = [r.current_usage for r in recent_usage]
                        summary[resource_type.name] = {
                            'current': recent_usage[-1].current_usage,
                            'average': sum(current_values) / len(current_values),
                            'peak': max(current_values),
                            'limit': self.resource_limits.get(resource_type),
                            'measurements': len(recent_usage),
                            'status': 'OK' if recent_usage[-1].current_usage <= self.resource_limits.get(resource_type, 100) else 'ALERT'
                        }
        
        return summary
    
    def _monitoring_worker(self):
        """Background worker for resource monitoring."""
        while self._monitoring_active:
            try:
                # Monitor system resources
                self._collect_system_resources()
                
                # Monitor GPU resources
                self._collect_gpu_resources()
                
                time.sleep(15)  # Monitor every 15 seconds
                
            except Exception as e:
                logger.error(f"Resource monitoring error: {e}")
                time.sleep(60)  # Wait longer on error
    
    def _collect_system_resources(self):
        """Collect system resource measurements."""
        try:
            import psutil
            
            # Memory
            memory = psutil.virtual_memory()
            self.record_resource_usage(ResourceType.MEMORY, memory.percent)
            
            # CPU
            cpu_percent = psutil.cpu_percent(interval=1)
            self.record_resource_usage(ResourceType.CPU, cpu_percent)
            
            # Disk
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            self.record_resource_usage(ResourceType.DISK, disk_percent)
            
        except ImportError:
            logger.debug("psutil not available for system resource monitoring")
    
    def _collect_gpu_resources(self):
        """Collect GPU resource measurements."""
        try:
            import torch
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    allocated = torch.cuda.memory_allocated(i)
                    reserved = torch.cuda.memory_reserved(i)
                    
                    # Get total memory
                    props = torch.cuda.get_device_properties(i)
                    total_memory = props.total_memory
                    
                    usage_percent = (allocated / total_memory) * 100
                    
                    self.record_resource_usage(
                        ResourceType.GPU,
                        usage_percent,
                        {
                            'device_id': i,
                            'allocated_mb': allocated / (1024**2),
                            'reserved_mb': reserved / (1024**2),
                            'total_mb': total_memory / (1024**2)
                        }
                    )
        except ImportError:
            pass  # GPU monitoring not available


class ProductionInfrastructureManager:
    """Comprehensive production infrastructure management."""
    
    def __init__(self):
        self.shutdown_manager = GracefulShutdownManager()
        self.memory_manager = MemoryManager()
        self.config_manager = ConfigurationManager()
        self.resource_monitor = ResourceMonitor()
        
        self.infrastructure_health = {}
        self.startup_time = datetime.now()
        self._setup_infrastructure_monitoring()
        self._setup_default_configuration_schema()
    
    def _setup_infrastructure_monitoring(self):
        """Setup infrastructure monitoring and callbacks."""
        # Register shutdown callbacks
        self.shutdown_manager.register_shutdown_callback(
            ShutdownPhase.CLEANUP_RESOURCES,
            self._cleanup_memory_resources,
            priority=100
        )
        
        self.shutdown_manager.register_shutdown_callback(
            ShutdownPhase.FINAL_CLEANUP,
            self._save_final_state,
            priority=50
        )
        
        # Register resource alert callbacks
        self.resource_monitor.alert_callbacks[ResourceType.MEMORY].append(
            self._handle_memory_alert
        )
        
        # Start monitoring
        self.resource_monitor.start_monitoring()
    
    def _setup_default_configuration_schema(self):
        """Setup default configuration validation schema."""
        schema = ConfigurationSchema(
            required_keys={'model_type', 'input_dim', 'hidden_dim'},
            optional_keys={'batch_size', 'learning_rate', 'epochs', 'device'},
            type_constraints={
                'input_dim': int,
                'hidden_dim': int,
                'batch_size': int,
                'learning_rate': float,
                'epochs': int,
                'model_type': str
            },
            value_constraints={
                'input_dim': lambda x: x > 0,
                'hidden_dim': lambda x: x > 0,
                'batch_size': lambda x: x > 0,
                'learning_rate': lambda x: 0 < x < 1,
                'epochs': lambda x: x > 0
            }
        )
        
        # Add custom validators
        def validate_model_type(config):
            valid_types = ['PNO', 'FNO', 'DeepONet', 'MCU-Net']
            model_type = config.get('model_type', '')
            if model_type not in valid_types:
                return [f"Invalid model_type: {model_type}. Must be one of {valid_types}"]
            return []
        
        schema.custom_validators.append(validate_model_type)
        self.config_manager.set_schema(schema)
    
    def _cleanup_memory_resources(self):
        """Cleanup memory resources during shutdown."""
        logger.info("Cleaning up memory resources")
        
        # Release all memory pools
        for pool_name in list(self.memory_manager.memory_pools.keys()):
            self.memory_manager.release_memory_pool(pool_name)
        
        # Force final garbage collection
        self.memory_manager.force_garbage_collection()
    
    def _save_final_state(self):
        """Save final system state for analysis."""
        logger.info("Saving final system state")
        
        try:
            final_state = {
                'shutdown_time': datetime.now().isoformat(),
                'uptime_seconds': (datetime.now() - self.startup_time).total_seconds(),
                'memory_report': self.memory_manager.get_memory_report(),
                'resource_summary': self.resource_monitor.get_resource_summary(),
                'configuration': self.config_manager.config.copy(),
                'infrastructure_health': self.get_infrastructure_health()
            }
            
            state_file = f'/root/repo/logs/final_state_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
            os.makedirs(os.path.dirname(state_file), exist_ok=True)
            
            with open(state_file, 'w') as f:
                json.dump(final_state, f, indent=2)
            
            logger.info(f"Final state saved: {state_file}")
            
        except Exception as e:
            logger.error(f"Failed to save final state: {e}")
    
    def _handle_memory_alert(self, resource_type: ResourceType, usage: float, limit: float):
        """Handle memory usage alerts."""
        if usage > 95:  # Critical memory usage
            logger.critical(f"Critical memory usage: {usage:.1f}%")
            self.memory_manager.force_garbage_collection()
            
            # If still critical after GC, consider emergency measures
            time.sleep(1)  # Brief pause for GC to complete
            current_usage = self.memory_manager.get_current_memory_usage()
            memory_percent = (current_usage / self.memory_manager.memory_limits['max_total_mb']) * 100
            
            if memory_percent > 95:
                logger.critical("Memory usage still critical after GC, initiating graceful shutdown")
                self.shutdown_manager.initiate_shutdown()
    
    def get_infrastructure_health(self) -> Dict[str, Any]:
        """Get comprehensive infrastructure health status."""
        return {
            'timestamp': datetime.now().isoformat(),
            'uptime_seconds': (datetime.now() - self.startup_time).total_seconds(),
            'shutdown_status': self.shutdown_manager.shutdown_phase.name,
            'memory_health': self.memory_manager.get_memory_report(),
            'resource_health': self.resource_monitor.get_resource_summary(),
            'configuration_status': {
                'loaded': bool(self.config_manager.config),
                'file_exists': os.path.exists(self.config_manager.config_file),
                'last_change': self.config_manager.config_history[-1]['timestamp'].isoformat() 
                              if self.config_manager.config_history else None
            },
            'active_requests': len(self.shutdown_manager.active_requests)
        }
    
    @contextmanager
    def managed_request(self, request_id: Optional[str] = None):
        """Context manager for managed request execution."""
        with self.shutdown_manager.track_request(request_id) as req_id:
            yield req_id
    
    def prepare_for_deployment(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare infrastructure for new deployment."""
        logger.info("Preparing infrastructure for deployment")
        
        preparation_results = {
            'timestamp': datetime.now().isoformat(),
            'checks': {},
            'actions': [],
            'ready': True
        }
        
        # Validate configuration
        config_errors = self.config_manager.validate_configuration(config)
        preparation_results['checks']['configuration'] = {
            'status': 'PASS' if not config_errors else 'FAIL',
            'errors': config_errors
        }
        
        if config_errors:
            preparation_results['ready'] = False
        
        # Check resource availability
        resource_summary = self.resource_monitor.get_resource_summary()
        high_usage_resources = [
            name for name, info in resource_summary.items()
            if info.get('status') == 'ALERT'
        ]
        
        preparation_results['checks']['resources'] = {
            'status': 'PASS' if not high_usage_resources else 'WARN',
            'high_usage': high_usage_resources
        }
        
        # Preemptive garbage collection
        if high_usage_resources:
            gc_stats = self.memory_manager.force_garbage_collection()
            preparation_results['actions'].append(f"Forced GC: {gc_stats['objects_collected']} objects collected")
        
        # Create deployment checkpoint
        try:
            checkpoint_data = {
                'pre_deployment_state': self.get_infrastructure_health(),
                'deployment_config': config
            }
            
            checkpoint_file = f'/root/repo/logs/deployment_checkpoint_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
            with open(checkpoint_file, 'w') as f:
                json.dump(checkpoint_data, f, indent=2)
            
            preparation_results['actions'].append(f"Created deployment checkpoint: {checkpoint_file}")
            
        except Exception as e:
            logger.error(f"Failed to create deployment checkpoint: {e}")
            preparation_results['checks']['checkpoint'] = {'status': 'FAIL', 'error': str(e)}
            preparation_results['ready'] = False
        
        return preparation_results


# Global infrastructure manager
global_infrastructure_manager = ProductionInfrastructureManager()


# Context managers and decorators
@contextmanager
def infrastructure_context():
    """Context manager for infrastructure-aware operations."""
    with global_infrastructure_manager.managed_request() as request_id:
        yield {
            'request_id': request_id,
            'infrastructure': global_infrastructure_manager
        }


def infrastructure_managed(operation_name: str = None):
    """Decorator to add infrastructure management to functions."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            op_name = operation_name or func.__name__
            
            with infrastructure_context() as context:
                # Add infrastructure context to kwargs
                kwargs['_infrastructure_context'] = context
                
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    # Log infrastructure state on error
                    health = global_infrastructure_manager.get_infrastructure_health()
                    logger.error(f"Operation {op_name} failed, infrastructure health: {health}")
                    raise
        
        return wrapper
    return decorator


# Health check functions for infrastructure
def check_infrastructure_health() -> bool:
    """Check overall infrastructure health."""
    try:
        health = global_infrastructure_manager.get_infrastructure_health()
        
        # Check for critical conditions
        if health['shutdown_status'] != 'NORMAL_OPERATION':
            return False
        
        memory_status = health['memory_health']['status']
        if memory_status == 'CRITICAL':
            return False
        
        return True
        
    except Exception:
        return False


def check_configuration_health() -> bool:
    """Check configuration system health."""
    try:
        config_manager = global_infrastructure_manager.config_manager
        return bool(config_manager.config) and os.path.exists(config_manager.config_file)
    except Exception:
        return False


def check_memory_management_health() -> bool:
    """Check memory management health."""
    try:
        memory_report = global_infrastructure_manager.memory_manager.get_memory_report()
        return memory_report['status'] in ['HEALTHY', 'WARNING']
    except Exception:
        return False