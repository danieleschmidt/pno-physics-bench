"""Advanced resource management system for PNO operations."""

import os
import time
import threading
import queue
import weakref
from typing import Any, Dict, List, Optional, Callable, Tuple, Union, Set
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import logging
import psutil
import warnings
from concurrent.futures import Future
from collections import defaultdict, deque
import tempfile
import shutil

logger = logging.getLogger(__name__)

# Optional dependencies
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    warnings.warn("PyTorch not available - GPU resource management disabled")


class ResourceType(Enum):
    """Types of system resources."""
    CPU = "cpu"
    MEMORY = "memory" 
    GPU = "gpu"
    DISK = "disk"
    NETWORK = "network"
    COMPUTE = "compute"


class ResourcePriority(Enum):
    """Resource allocation priorities."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


class AllocationStrategy(Enum):
    """Resource allocation strategies."""
    GREEDY = "greedy"
    FAIR_SHARE = "fair_share"
    PRIORITY_BASED = "priority_based"
    ADAPTIVE = "adaptive"


@dataclass
class ResourceRequest:
    """Resource allocation request."""
    id: str
    resource_type: ResourceType
    amount: float
    priority: ResourcePriority
    requester: str
    timeout: Optional[float] = None
    callback: Optional[Callable] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_time: float = field(default_factory=time.time)
    
    @property
    def age(self) -> float:
        """Age of request in seconds."""
        return time.time() - self.created_time
    
    @property
    def is_expired(self) -> bool:
        """Check if request has expired."""
        if self.timeout is None:
            return False
        return self.age > self.timeout


@dataclass
class ResourceAllocation:
    """Active resource allocation."""
    request_id: str
    resource_type: ResourceType
    amount: float
    allocated_time: float
    requester: str
    priority: ResourcePriority
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def duration(self) -> float:
        """Duration of allocation in seconds."""
        return time.time() - self.allocated_time


@dataclass
class ResourceStatus:
    """Current resource status."""
    resource_type: ResourceType
    total_capacity: float
    allocated_amount: float
    available_amount: float
    utilization: float
    active_allocations: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'resource_type': self.resource_type.value,
            'total_capacity': self.total_capacity,
            'allocated_amount': self.allocated_amount,
            'available_amount': self.available_amount,
            'utilization': self.utilization,
            'active_allocations': self.active_allocations
        }


class ResourceMonitor:
    """System resource monitoring."""
    
    def __init__(self, update_interval: float = 1.0):
        """Initialize resource monitor.
        
        Args:
            update_interval: Resource update interval in seconds
        """
        self.update_interval = update_interval
        self.resource_data = {}
        self.monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()
        
        # History tracking
        self.history_size = 1000
        self.resource_history = defaultdict(lambda: deque(maxlen=self.history_size))
    
    def start_monitoring(self):
        """Start resource monitoring."""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("Started resource monitoring")
    
    def stop_monitoring(self):
        """Stop resource monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        logger.info("Stopped resource monitoring")
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.monitoring:
            try:
                self._update_resources()
                time.sleep(self.update_interval)
            except Exception as e:
                logger.error(f"Error in resource monitoring: {e}")
    
    def _update_resources(self):
        """Update resource information."""
        with self._lock:
            current_time = time.time()
            
            # CPU monitoring
            cpu_percent = psutil.cpu_percent(interval=None)
            cpu_count = psutil.cpu_count()
            self.resource_data[ResourceType.CPU] = {
                'total': cpu_count,
                'used': cpu_percent / 100.0 * cpu_count,
                'available': cpu_count - (cpu_percent / 100.0 * cpu_count),
                'utilization': cpu_percent / 100.0,
                'timestamp': current_time
            }
            
            # Memory monitoring
            memory = psutil.virtual_memory()
            self.resource_data[ResourceType.MEMORY] = {
                'total': memory.total / (1024**3),  # GB
                'used': memory.used / (1024**3),
                'available': memory.available / (1024**3),
                'utilization': memory.percent / 100.0,
                'timestamp': current_time
            }
            
            # Disk monitoring
            disk = psutil.disk_usage('/')
            self.resource_data[ResourceType.DISK] = {
                'total': disk.total / (1024**3),  # GB
                'used': disk.used / (1024**3),
                'available': disk.free / (1024**3),
                'utilization': disk.used / disk.total,
                'timestamp': current_time
            }
            
            # GPU monitoring (if available)
            if HAS_TORCH and torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                total_memory = 0
                used_memory = 0
                
                for i in range(gpu_count):
                    props = torch.cuda.get_device_properties(i)
                    total_memory += props.total_memory / (1024**3)  # GB
                    used_memory += torch.cuda.memory_allocated(i) / (1024**3)
                
                self.resource_data[ResourceType.GPU] = {
                    'total': total_memory,
                    'used': used_memory,
                    'available': total_memory - used_memory,
                    'utilization': used_memory / total_memory if total_memory > 0 else 0,
                    'device_count': gpu_count,
                    'timestamp': current_time
                }
            
            # Update history
            for resource_type, data in self.resource_data.items():
                self.resource_history[resource_type].append({
                    'timestamp': current_time,
                    'utilization': data['utilization'],
                    'available': data['available']
                })
    
    def get_resource_status(self, resource_type: ResourceType) -> Optional[ResourceStatus]:
        """Get current resource status."""
        with self._lock:
            if resource_type not in self.resource_data:
                return None
            
            data = self.resource_data[resource_type]
            return ResourceStatus(
                resource_type=resource_type,
                total_capacity=data['total'],
                allocated_amount=data['used'],
                available_amount=data['available'],
                utilization=data['utilization'],
                active_allocations=0  # Will be updated by resource manager
            )
    
    def get_resource_trend(
        self,
        resource_type: ResourceType,
        window_minutes: int = 10
    ) -> Dict[str, float]:
        """Get resource utilization trend."""
        with self._lock:
            if resource_type not in self.resource_history:
                return {}
            
            current_time = time.time()
            window_seconds = window_minutes * 60
            
            # Filter history to window
            recent_data = [
                entry for entry in self.resource_history[resource_type]
                if current_time - entry['timestamp'] <= window_seconds
            ]
            
            if len(recent_data) < 2:
                return {}
            
            # Calculate trend
            utilizations = [entry['utilization'] for entry in recent_data]
            
            avg_utilization = sum(utilizations) / len(utilizations)
            min_utilization = min(utilizations)
            max_utilization = max(utilizations)
            
            # Simple linear trend
            if len(utilizations) >= 2:
                x_values = list(range(len(utilizations)))
                y_values = utilizations
                
                # Calculate slope
                n = len(x_values)
                sum_x = sum(x_values)
                sum_y = sum(y_values)
                sum_xy = sum(x * y for x, y in zip(x_values, y_values))
                sum_x2 = sum(x * x for x in x_values)
                
                slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
                trend_direction = "increasing" if slope > 0.01 else "decreasing" if slope < -0.01 else "stable"
            else:
                slope = 0
                trend_direction = "stable"
            
            return {
                'average_utilization': avg_utilization,
                'min_utilization': min_utilization,
                'max_utilization': max_utilization,
                'trend_slope': slope,
                'trend_direction': trend_direction,
                'data_points': len(recent_data)
            }


class ResourcePool:
    """Manages a pool of specific resource type."""
    
    def __init__(
        self,
        resource_type: ResourceType,
        total_capacity: float,
        allocation_strategy: AllocationStrategy = AllocationStrategy.FAIR_SHARE
    ):
        """Initialize resource pool.
        
        Args:
            resource_type: Type of resource
            total_capacity: Total capacity of resource
            allocation_strategy: Allocation strategy
        """
        self.resource_type = resource_type
        self.total_capacity = total_capacity
        self.allocation_strategy = allocation_strategy
        
        # Allocation tracking
        self.allocations: Dict[str, ResourceAllocation] = {}
        self.pending_requests: queue.PriorityQueue = queue.PriorityQueue()
        self.allocation_history = []
        
        # Synchronization
        self._lock = threading.RLock()
        
        # Statistics
        self.total_requests = 0
        self.successful_allocations = 0
        self.failed_allocations = 0
    
    @property
    def allocated_amount(self) -> float:
        """Total allocated amount."""
        with self._lock:
            return sum(alloc.amount for alloc in self.allocations.values())
    
    @property
    def available_amount(self) -> float:
        """Available amount."""
        return self.total_capacity - self.allocated_amount
    
    @property
    def utilization(self) -> float:
        """Current utilization ratio."""
        return self.allocated_amount / self.total_capacity if self.total_capacity > 0 else 0
    
    def request_resource(self, request: ResourceRequest) -> bool:
        """Request resource allocation.
        
        Args:
            request: Resource request
            
        Returns:
            True if immediately allocated, False if queued
        """
        with self._lock:
            self.total_requests += 1
            
            # Check if resource can be allocated immediately
            if self.available_amount >= request.amount:
                return self._allocate_resource(request)
            else:
                # Queue the request
                priority_score = self._calculate_priority_score(request)
                self.pending_requests.put((priority_score, request))
                return False
    
    def release_resource(self, request_id: str) -> bool:
        """Release allocated resource.
        
        Args:
            request_id: Request ID to release
            
        Returns:
            Success status
        """
        with self._lock:
            if request_id not in self.allocations:
                return False
            
            allocation = self.allocations.pop(request_id)
            
            # Record in history
            self.allocation_history.append({
                'request_id': request_id,
                'amount': allocation.amount,
                'duration': allocation.duration,
                'released_time': time.time()
            })
            
            # Keep history bounded
            if len(self.allocation_history) > 10000:
                self.allocation_history.pop(0)
            
            # Process pending requests
            self._process_pending_requests()
            
            logger.debug(f"Released {allocation.amount} units of {self.resource_type.value}")
            return True
    
    def _allocate_resource(self, request: ResourceRequest) -> bool:
        """Allocate resource to request."""
        if self.available_amount < request.amount:
            self.failed_allocations += 1
            return False
        
        # Create allocation
        allocation = ResourceAllocation(
            request_id=request.id,
            resource_type=request.resource_type,
            amount=request.amount,
            allocated_time=time.time(),
            requester=request.requester,
            priority=request.priority,
            metadata=request.metadata
        )
        
        self.allocations[request.id] = allocation
        self.successful_allocations += 1
        
        # Call callback if provided
        if request.callback:
            try:
                request.callback(allocation)
            except Exception as e:
                logger.error(f"Error in allocation callback: {e}")
        
        logger.debug(f"Allocated {request.amount} units of {self.resource_type.value} to {request.requester}")
        return True
    
    def _calculate_priority_score(self, request: ResourceRequest) -> float:
        """Calculate priority score for request (lower = higher priority)."""
        base_priority = -request.priority.value  # Negative for min-heap
        
        if self.allocation_strategy == AllocationStrategy.PRIORITY_BASED:
            return base_priority
        elif self.allocation_strategy == AllocationStrategy.FAIR_SHARE:
            # Consider requester's current allocations
            requester_allocations = sum(
                alloc.amount for alloc in self.allocations.values()
                if alloc.requester == request.requester
            )
            return base_priority + requester_allocations * 0.1
        elif self.allocation_strategy == AllocationStrategy.ADAPTIVE:
            # Consider request age and size
            age_factor = request.age * 0.01
            size_factor = request.amount / self.total_capacity * 0.1
            return base_priority + age_factor - size_factor
        else:  # GREEDY
            return base_priority - request.amount  # Larger requests first
    
    def _process_pending_requests(self):
        """Process pending requests after resource release."""
        processed = []
        
        # Extract all pending requests
        while not self.pending_requests.empty():
            try:
                priority_score, request = self.pending_requests.get_nowait()
                
                # Skip expired requests
                if request.is_expired:
                    self.failed_allocations += 1
                    continue
                
                # Try to allocate
                if self.available_amount >= request.amount:
                    self._allocate_resource(request)
                else:
                    processed.append((priority_score, request))
            except queue.Empty:
                break
        
        # Re-queue unprocessed requests
        for item in processed:
            self.pending_requests.put(item)
    
    def get_status(self) -> ResourceStatus:
        """Get current resource pool status."""
        with self._lock:
            return ResourceStatus(
                resource_type=self.resource_type,
                total_capacity=self.total_capacity,
                allocated_amount=self.allocated_amount,
                available_amount=self.available_amount,
                utilization=self.utilization,
                active_allocations=len(self.allocations)
            )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get resource pool statistics."""
        with self._lock:
            if self.total_requests > 0:
                success_rate = self.successful_allocations / self.total_requests
            else:
                success_rate = 0.0
            
            if self.allocation_history:
                avg_duration = sum(h['duration'] for h in self.allocation_history[-100:]) / min(100, len(self.allocation_history))
            else:
                avg_duration = 0.0
            
            return {
                'total_requests': self.total_requests,
                'successful_allocations': self.successful_allocations,
                'failed_allocations': self.failed_allocations,
                'success_rate': success_rate,
                'pending_requests': self.pending_requests.qsize(),
                'active_allocations': len(self.allocations),
                'average_allocation_duration': avg_duration,
                'current_utilization': self.utilization
            }


class ResourceManager:
    """Advanced resource management system."""
    
    def __init__(
        self,
        resource_monitor: Optional[ResourceMonitor] = None,
        auto_scaling: bool = True
    ):
        """Initialize resource manager.
        
        Args:
            resource_monitor: Resource monitor instance
            auto_scaling: Whether to enable auto-scaling
        """
        self.resource_monitor = resource_monitor or ResourceMonitor()
        self.auto_scaling = auto_scaling
        
        # Resource pools
        self.resource_pools: Dict[ResourceType, ResourcePool] = {}
        
        # Initialize pools based on system resources
        self._initialize_resource_pools()
        
        # Management state
        self.active = False
        self.management_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        
        # Request tracking
        self.active_requests: Dict[str, ResourceRequest] = {}
        self._lock = threading.RLock()
    
    def _initialize_resource_pools(self):
        """Initialize resource pools based on system capacity."""
        # Start monitoring to get resource info
        self.resource_monitor.start_monitoring()
        time.sleep(1)  # Wait for initial monitoring data
        
        # CPU pool
        cpu_status = self.resource_monitor.get_resource_status(ResourceType.CPU)
        if cpu_status:
            self.resource_pools[ResourceType.CPU] = ResourcePool(
                ResourceType.CPU,
                cpu_status.total_capacity
            )
        
        # Memory pool (reserve 20% for system)
        memory_status = self.resource_monitor.get_resource_status(ResourceType.MEMORY)
        if memory_status:
            available_memory = memory_status.total_capacity * 0.8
            self.resource_pools[ResourceType.MEMORY] = ResourcePool(
                ResourceType.MEMORY,
                available_memory
            )
        
        # GPU pool
        gpu_status = self.resource_monitor.get_resource_status(ResourceType.GPU)
        if gpu_status:
            self.resource_pools[ResourceType.GPU] = ResourcePool(
                ResourceType.GPU,
                gpu_status.total_capacity
            )
        
        # Disk pool (reserve significant space for system)
        disk_status = self.resource_monitor.get_resource_status(ResourceType.DISK)
        if disk_status:
            available_disk = disk_status.total_capacity * 0.5  # Conservative
            self.resource_pools[ResourceType.DISK] = ResourcePool(
                ResourceType.DISK,
                available_disk
            )
        
        logger.info(f"Initialized {len(self.resource_pools)} resource pools")
    
    def start(self):
        """Start resource management."""
        if self.active:
            return
        
        self.active = True
        self._stop_event.clear()
        
        # Start monitoring
        self.resource_monitor.start_monitoring()
        
        # Start management thread
        if self.auto_scaling:
            self.management_thread = threading.Thread(target=self._management_loop, daemon=True)
            self.management_thread.start()
        
        logger.info("Started resource management")
    
    def stop(self):
        """Stop resource management."""
        if not self.active:
            return
        
        self.active = False
        self._stop_event.set()
        
        # Stop monitoring
        self.resource_monitor.stop_monitoring()
        
        # Stop management thread
        if self.management_thread:
            self.management_thread.join(timeout=5.0)
        
        logger.info("Stopped resource management")
    
    def _management_loop(self):
        """Main resource management loop."""
        while not self._stop_event.wait(30.0):  # Check every 30 seconds
            try:
                self._auto_scale_resources()
                self._cleanup_expired_requests()
            except Exception as e:
                logger.error(f"Error in resource management loop: {e}")
    
    def _auto_scale_resources(self):
        """Automatically scale resource pools based on usage."""
        for resource_type, pool in self.resource_pools.items():
            trend = self.resource_monitor.get_resource_trend(resource_type)
            
            if not trend:
                continue
            
            # Scale up if high utilization and increasing trend
            if (trend['average_utilization'] > 0.8 and 
                trend['trend_direction'] == 'increasing'):
                
                # Increase pool capacity by 10%
                new_capacity = pool.total_capacity * 1.1
                max_capacity = self.resource_monitor.get_resource_status(resource_type).total_capacity * 0.9
                
                if new_capacity <= max_capacity:
                    pool.total_capacity = new_capacity
                    logger.info(f"Scaled up {resource_type.value} pool to {new_capacity:.2f}")
            
            # Scale down if low utilization and stable/decreasing trend
            elif (trend['average_utilization'] < 0.3 and 
                  trend['trend_direction'] in ['stable', 'decreasing']):
                
                # Decrease pool capacity by 5%
                new_capacity = pool.total_capacity * 0.95
                min_capacity = self.resource_monitor.get_resource_status(resource_type).total_capacity * 0.2
                
                if new_capacity >= min_capacity and new_capacity >= pool.allocated_amount:
                    pool.total_capacity = new_capacity
                    logger.info(f"Scaled down {resource_type.value} pool to {new_capacity:.2f}")
    
    def _cleanup_expired_requests(self):
        """Clean up expired requests."""
        with self._lock:
            expired_requests = [
                request_id for request_id, request in self.active_requests.items()
                if request.is_expired
            ]
            
            for request_id in expired_requests:
                del self.active_requests[request_id]
                
                # Release any allocations
                for pool in self.resource_pools.values():
                    pool.release_resource(request_id)
            
            if expired_requests:
                logger.debug(f"Cleaned up {len(expired_requests)} expired requests")
    
    def allocate_resource(
        self,
        resource_type: ResourceType,
        amount: float,
        requester: str,
        priority: ResourcePriority = ResourcePriority.NORMAL,
        timeout: Optional[float] = None,
        callback: Optional[Callable] = None
    ) -> Optional[str]:
        """Allocate resource.
        
        Args:
            resource_type: Type of resource
            amount: Amount to allocate
            requester: Name of requester
            priority: Allocation priority
            timeout: Request timeout
            callback: Allocation callback
            
        Returns:
            Request ID if successful, None otherwise
        """
        if resource_type not in self.resource_pools:
            logger.warning(f"Resource type {resource_type.value} not available")
            return None
        
        # Create request
        request_id = f"{requester}_{resource_type.value}_{int(time.time())}"
        request = ResourceRequest(
            id=request_id,
            resource_type=resource_type,
            amount=amount,
            priority=priority,
            requester=requester,
            timeout=timeout,
            callback=callback
        )
        
        with self._lock:
            self.active_requests[request_id] = request
        
        # Try to allocate
        pool = self.resource_pools[resource_type]
        if pool.request_resource(request):
            logger.debug(f"Immediately allocated {amount} {resource_type.value} to {requester}")
            return request_id
        else:
            logger.debug(f"Queued allocation request for {amount} {resource_type.value} from {requester}")
            return request_id
    
    def release_resource(self, request_id: str) -> bool:
        """Release allocated resource.
        
        Args:
            request_id: Request ID to release
            
        Returns:
            Success status
        """
        with self._lock:
            if request_id not in self.active_requests:
                return False
            
            request = self.active_requests.pop(request_id)
        
        # Release from appropriate pool
        if request.resource_type in self.resource_pools:
            return self.resource_pools[request.resource_type].release_resource(request_id)
        
        return False
    
    def get_resource_status(self) -> Dict[ResourceType, ResourceStatus]:
        """Get status of all resources."""
        status = {}
        for resource_type, pool in self.resource_pools.items():
            pool_status = pool.get_status()
            
            # Update with monitor data
            monitor_status = self.resource_monitor.get_resource_status(resource_type)
            if monitor_status:
                # Use system utilization for better accuracy
                pool_status.utilization = monitor_status.utilization
            
            status[resource_type] = pool_status
        
        return status
    
    def get_management_report(self) -> Dict[str, Any]:
        """Get comprehensive resource management report."""
        # Resource status
        resource_status = {}
        for resource_type, status in self.get_resource_status().items():
            resource_status[resource_type.value] = status.to_dict()
        
        # Pool statistics
        pool_stats = {}
        for resource_type, pool in self.resource_pools.items():
            pool_stats[resource_type.value] = pool.get_statistics()
        
        # Active requests
        with self._lock:
            active_request_count = len(self.active_requests)
            request_by_type = defaultdict(int)
            for request in self.active_requests.values():
                request_by_type[request.resource_type.value] += 1
        
        return {
            'active': self.active,
            'auto_scaling': self.auto_scaling,
            'resource_status': resource_status,
            'pool_statistics': pool_stats,
            'active_requests': active_request_count,
            'requests_by_type': dict(request_by_type),
            'resource_pools': len(self.resource_pools)
        }


# Context manager for resource allocation
class ResourceContext:
    """Context manager for automatic resource management."""
    
    def __init__(
        self,
        resource_manager: ResourceManager,
        resource_type: ResourceType,
        amount: float,
        requester: str,
        **kwargs
    ):
        """Initialize resource context.
        
        Args:
            resource_manager: Resource manager instance
            resource_type: Type of resource
            amount: Amount to allocate
            requester: Requester name
            **kwargs: Additional allocation arguments
        """
        self.resource_manager = resource_manager
        self.resource_type = resource_type
        self.amount = amount
        self.requester = requester
        self.kwargs = kwargs
        self.request_id: Optional[str] = None
    
    def __enter__(self):
        """Allocate resource on context entry."""
        self.request_id = self.resource_manager.allocate_resource(
            self.resource_type,
            self.amount,
            self.requester,
            **self.kwargs
        )
        
        if self.request_id is None:
            raise RuntimeError(f"Failed to allocate {self.amount} {self.resource_type.value}")
        
        return self.request_id
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Release resource on context exit."""
        if self.request_id:
            self.resource_manager.release_resource(self.request_id)


# Global resource manager instance
global_resource_manager = ResourceManager()


def allocate_resource(
    resource_type: ResourceType,
    amount: float,
    requester: str,
    **kwargs
) -> Optional[str]:
    """Allocate resource using global manager."""
    return global_resource_manager.allocate_resource(
        resource_type, amount, requester, **kwargs
    )


def release_resource(request_id: str) -> bool:
    """Release resource using global manager."""
    return global_resource_manager.release_resource(request_id)


def get_resource_status() -> Dict[ResourceType, ResourceStatus]:
    """Get global resource status."""
    return global_resource_manager.get_resource_status()


def start_resource_management():
    """Start global resource management."""
    global_resource_manager.start()


def stop_resource_management():
    """Stop global resource management."""
    global_resource_manager.stop()