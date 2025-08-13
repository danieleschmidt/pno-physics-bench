"""Intelligent caching system for PNO operations with adaptive strategies."""

import time
import threading
import hashlib
import pickle
import json
import weakref
from typing import Any, Dict, List, Optional, Callable, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
from collections import OrderedDict, defaultdict
import logging
from pathlib import Path
import tempfile
import os
import warnings

logger = logging.getLogger(__name__)

# Optional dependencies
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    warnings.warn("PyTorch not available - tensor caching disabled")

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    warnings.warn("NumPy not available - array caching limited")


class CacheEvictionPolicy(Enum):
    """Cache eviction policies."""
    LRU = "lru"           # Least Recently Used
    LFU = "lfu"           # Least Frequently Used
    FIFO = "fifo"         # First In First Out
    ADAPTIVE = "adaptive"  # Adaptive based on access patterns
    TTL = "ttl"           # Time To Live


class CacheLocation(Enum):
    """Cache storage locations."""
    MEMORY = "memory"
    DISK = "disk"
    HYBRID = "hybrid"     # Memory + Disk
    DISTRIBUTED = "distributed"


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    key: str
    value: Any
    size_bytes: int
    created_time: float
    last_accessed: float
    access_count: int = 0
    hit_count: int = 0
    computation_time: Optional[float] = None
    ttl: Optional[float] = None
    priority: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def age(self) -> float:
        """Age of cache entry in seconds."""
        return time.time() - self.created_time
    
    @property
    def is_expired(self) -> bool:
        """Check if entry is expired."""
        if self.ttl is None:
            return False
        return self.age > self.ttl
    
    @property
    def access_frequency(self) -> float:
        """Calculate access frequency."""
        if self.age == 0:
            return 0
        return self.access_count / self.age
    
    def update_access(self):
        """Update access statistics."""
        self.last_accessed = time.time()
        self.access_count += 1


class CacheStatistics:
    """Cache performance statistics."""
    
    def __init__(self):
        """Initialize statistics."""
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.size_bytes = 0
        self.entry_count = 0
        self.start_time = time.time()
        
        # Performance tracking
        self.total_lookup_time = 0.0
        self.total_store_time = 0.0
        self.total_computation_savings = 0.0
    
    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
    
    @property
    def uptime(self) -> float:
        """Cache uptime in seconds."""
        return time.time() - self.start_time
    
    @property
    def average_lookup_time(self) -> float:
        """Average lookup time."""
        total_lookups = self.hits + self.misses
        return self.total_lookup_time / total_lookups if total_lookups > 0 else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert statistics to dictionary."""
        return {
            'hits': self.hits,
            'misses': self.misses,
            'evictions': self.evictions,
            'hit_rate': self.hit_rate,
            'size_bytes': self.size_bytes,
            'size_mb': self.size_bytes / (1024 * 1024),
            'entry_count': self.entry_count,
            'uptime_seconds': self.uptime,
            'average_lookup_time_ms': self.average_lookup_time * 1000,
            'computation_savings_seconds': self.total_computation_savings
        }


class BaseCache(ABC):
    """Base class for caching implementations."""
    
    def __init__(
        self,
        max_size_bytes: int = 1024 * 1024 * 1024,  # 1GB
        max_entries: int = 10000,
        eviction_policy: CacheEvictionPolicy = CacheEvictionPolicy.LRU,
        default_ttl: Optional[float] = None
    ):
        """Initialize base cache.
        
        Args:
            max_size_bytes: Maximum cache size in bytes
            max_entries: Maximum number of entries
            eviction_policy: Eviction policy
            default_ttl: Default time to live in seconds
        """
        self.max_size_bytes = max_size_bytes
        self.max_entries = max_entries
        self.eviction_policy = eviction_policy
        self.default_ttl = default_ttl
        
        self.entries: Dict[str, CacheEntry] = {}
        self.statistics = CacheStatistics()
        self._lock = threading.RLock()
        
        # Eviction policy data structures
        if eviction_policy == CacheEvictionPolicy.LRU:
            self._lru_order = OrderedDict()
        elif eviction_policy == CacheEvictionPolicy.LFU:
            self._frequency_map = defaultdict(int)
        
    @abstractmethod
    def _store_value(self, entry: CacheEntry) -> bool:
        """Store value in cache backend."""
        pass
    
    @abstractmethod
    def _retrieve_value(self, key: str) -> Optional[Any]:
        """Retrieve value from cache backend."""
        pass
    
    @abstractmethod
    def _remove_value(self, key: str) -> bool:
        """Remove value from cache backend."""
        pass
    
    def _calculate_size(self, value: Any) -> int:
        """Calculate size of value in bytes."""
        try:
            if HAS_TORCH and isinstance(value, torch.Tensor):
                return value.element_size() * value.numel()
            elif HAS_NUMPY and isinstance(value, np.ndarray):
                return value.nbytes
            else:
                # Fallback to pickle size estimation
                return len(pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL))
        except Exception:
            # Conservative estimate
            return 1024
    
    def _generate_key(self, *args, **kwargs) -> str:
        """Generate cache key from arguments."""
        # Create a stable hash of the arguments
        key_data = {
            'args': args,
            'kwargs': sorted(kwargs.items()) if kwargs else {}
        }
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.sha256(key_str.encode()).hexdigest()
    
    def _should_evict(self) -> bool:
        """Check if eviction is needed."""
        return (
            len(self.entries) >= self.max_entries or
            self.statistics.size_bytes >= self.max_size_bytes
        )
    
    def _select_eviction_candidates(self, count: int = 1) -> List[str]:
        """Select entries for eviction based on policy."""
        if not self.entries:
            return []
        
        candidates = []
        
        if self.eviction_policy == CacheEvictionPolicy.LRU:
            # Least recently used
            sorted_entries = sorted(
                self.entries.items(),
                key=lambda x: x[1].last_accessed
            )
            candidates = [key for key, _ in sorted_entries[:count]]
            
        elif self.eviction_policy == CacheEvictionPolicy.LFU:
            # Least frequently used
            sorted_entries = sorted(
                self.entries.items(),
                key=lambda x: x[1].access_frequency
            )
            candidates = [key for key, _ in sorted_entries[:count]]
            
        elif self.eviction_policy == CacheEvictionPolicy.FIFO:
            # First in, first out
            sorted_entries = sorted(
                self.entries.items(),
                key=lambda x: x[1].created_time
            )
            candidates = [key for key, _ in sorted_entries[:count]]
            
        elif self.eviction_policy == CacheEvictionPolicy.TTL:
            # Expired entries first
            expired = [
                key for key, entry in self.entries.items()
                if entry.is_expired
            ]
            candidates = expired[:count]
            
            # If not enough expired entries, fall back to LRU
            if len(candidates) < count:
                remaining = count - len(candidates)
                non_expired = [
                    key for key in self.entries.keys()
                    if key not in candidates
                ]
                sorted_non_expired = sorted(
                    [(key, self.entries[key]) for key in non_expired],
                    key=lambda x: x[1].last_accessed
                )
                candidates.extend([key for key, _ in sorted_non_expired[:remaining]])
                
        elif self.eviction_policy == CacheEvictionPolicy.ADAPTIVE:
            # Adaptive eviction based on multiple factors
            scored_entries = []
            current_time = time.time()
            
            for key, entry in self.entries.items():
                # Score based on recency, frequency, and size
                recency_score = 1.0 / (current_time - entry.last_accessed + 1)
                frequency_score = entry.access_frequency
                size_score = 1.0 / (entry.size_bytes + 1)
                
                # Weighted combination
                total_score = (
                    0.4 * recency_score +
                    0.4 * frequency_score +
                    0.2 * size_score
                )
                
                scored_entries.append((key, total_score))
            
            # Sort by score (lower is better for eviction)
            scored_entries.sort(key=lambda x: x[1])
            candidates = [key for key, _ in scored_entries[:count]]
        
        return candidates
    
    def _evict_entries(self, keys: List[str]):
        """Evict specified entries."""
        for key in keys:
            if key in self.entries:
                entry = self.entries[key]
                self._remove_value(key)
                self.statistics.size_bytes -= entry.size_bytes
                self.statistics.entry_count -= 1
                self.statistics.evictions += 1
                del self.entries[key]
                
                logger.debug(f"Evicted cache entry: {key}")
    
    def put(
        self,
        key: str,
        value: Any,
        ttl: Optional[float] = None,
        priority: int = 0,
        computation_time: Optional[float] = None
    ) -> bool:
        """Store value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds
            priority: Priority level
            computation_time: Original computation time
            
        Returns:
            Success status
        """
        start_time = time.time()
        
        with self._lock:
            try:
                # Calculate size
                size_bytes = self._calculate_size(value)
                
                # Create entry
                entry = CacheEntry(
                    key=key,
                    value=value,
                    size_bytes=size_bytes,
                    created_time=time.time(),
                    last_accessed=time.time(),
                    ttl=ttl or self.default_ttl,
                    priority=priority,
                    computation_time=computation_time
                )
                
                # Check if eviction needed
                while self._should_evict():
                    candidates = self._select_eviction_candidates(
                        max(1, len(self.entries) // 10)  # Evict 10% or at least 1
                    )
                    if not candidates:
                        break
                    self._evict_entries(candidates)
                
                # Store value
                if self._store_value(entry):
                    self.entries[key] = entry
                    self.statistics.size_bytes += size_bytes
                    self.statistics.entry_count += 1
                    
                    # Update eviction policy structures
                    if self.eviction_policy == CacheEvictionPolicy.LRU:
                        self._lru_order[key] = True
                    
                    store_time = time.time() - start_time
                    self.statistics.total_store_time += store_time
                    
                    logger.debug(f"Cached entry: {key} ({size_bytes} bytes)")
                    return True
                
                return False
                
            except Exception as e:
                logger.error(f"Failed to cache entry {key}: {e}")
                return False
    
    def get(self, key: str) -> Optional[Any]:
        """Retrieve value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None
        """
        start_time = time.time()
        
        with self._lock:
            try:
                # Check if entry exists
                if key not in self.entries:
                    self.statistics.misses += 1
                    return None
                
                entry = self.entries[key]
                
                # Check if expired
                if entry.is_expired:
                    self._evict_entries([key])
                    self.statistics.misses += 1
                    return None
                
                # Retrieve value
                value = self._retrieve_value(key)
                if value is None:
                    # Entry exists in metadata but not in storage
                    self._evict_entries([key])
                    self.statistics.misses += 1
                    return None
                
                # Update access statistics
                entry.update_access()
                entry.hit_count += 1
                self.statistics.hits += 1
                
                # Update eviction policy structures
                if self.eviction_policy == CacheEvictionPolicy.LRU:
                    self._lru_order.move_to_end(key)
                
                lookup_time = time.time() - start_time
                self.statistics.total_lookup_time += lookup_time
                
                if entry.computation_time:
                    self.statistics.total_computation_savings += entry.computation_time
                
                logger.debug(f"Cache hit: {key}")
                return value
                
            except Exception as e:
                logger.error(f"Failed to retrieve cache entry {key}: {e}")
                self.statistics.misses += 1
                return None
    
    def invalidate(self, key: str) -> bool:
        """Invalidate cache entry.
        
        Args:
            key: Cache key
            
        Returns:
            Success status
        """
        with self._lock:
            if key in self.entries:
                entry = self.entries[key]
                self._remove_value(key)
                self.statistics.size_bytes -= entry.size_bytes
                self.statistics.entry_count -= 1
                del self.entries[key]
                
                # Update eviction policy structures
                if self.eviction_policy == CacheEvictionPolicy.LRU:
                    self._lru_order.pop(key, None)
                
                logger.debug(f"Invalidated cache entry: {key}")
                return True
            
            return False
    
    def clear(self):
        """Clear all cache entries."""
        with self._lock:
            for key in list(self.entries.keys()):
                self._remove_value(key)
            
            self.entries.clear()
            self.statistics = CacheStatistics()
            
            if self.eviction_policy == CacheEvictionPolicy.LRU:
                self._lru_order.clear()
            elif self.eviction_policy == CacheEvictionPolicy.LFU:
                self._frequency_map.clear()
            
            logger.info("Cleared cache")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return self.statistics.to_dict()


class MemoryCache(BaseCache):
    """In-memory cache implementation."""
    
    def __init__(self, **kwargs):
        """Initialize memory cache."""
        super().__init__(**kwargs)
        self._storage: Dict[str, Any] = {}
    
    def _store_value(self, entry: CacheEntry) -> bool:
        """Store value in memory."""
        self._storage[entry.key] = entry.value
        return True
    
    def _retrieve_value(self, key: str) -> Optional[Any]:
        """Retrieve value from memory."""
        return self._storage.get(key)
    
    def _remove_value(self, key: str) -> bool:
        """Remove value from memory."""
        return self._storage.pop(key, None) is not None


class DiskCache(BaseCache):
    """Disk-based cache implementation."""
    
    def __init__(
        self,
        cache_dir: Optional[str] = None,
        compression: bool = True,
        **kwargs
    ):
        """Initialize disk cache.
        
        Args:
            cache_dir: Cache directory path
            compression: Whether to compress cached data
        """
        super().__init__(**kwargs)
        
        if cache_dir is None:
            cache_dir = os.path.join(tempfile.gettempdir(), "pno_cache")
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.compression = compression
    
    def _get_file_path(self, key: str) -> Path:
        """Get file path for cache key."""
        # Use first 2 chars as subdirectory for better file system performance
        subdir = self.cache_dir / key[:2]
        subdir.mkdir(exist_ok=True)
        return subdir / f"{key}.pkl"
    
    def _store_value(self, entry: CacheEntry) -> bool:
        """Store value to disk."""
        try:
            file_path = self._get_file_path(entry.key)
            
            if self.compression:
                import gzip
                with gzip.open(file_path, 'wb') as f:
                    pickle.dump(entry.value, f, protocol=pickle.HIGHEST_PROTOCOL)
            else:
                with open(file_path, 'wb') as f:
                    pickle.dump(entry.value, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to store to disk: {e}")
            return False
    
    def _retrieve_value(self, key: str) -> Optional[Any]:
        """Retrieve value from disk."""
        try:
            file_path = self._get_file_path(key)
            
            if not file_path.exists():
                return None
            
            if self.compression:
                import gzip
                with gzip.open(file_path, 'rb') as f:
                    return pickle.load(f)
            else:
                with open(file_path, 'rb') as f:
                    return pickle.load(f)
                    
        except Exception as e:
            logger.error(f"Failed to retrieve from disk: {e}")
            return None
    
    def _remove_value(self, key: str) -> bool:
        """Remove value from disk."""
        try:
            file_path = self._get_file_path(key)
            if file_path.exists():
                file_path.unlink()
                return True
            return False
            
        except Exception as e:
            logger.error(f"Failed to remove from disk: {e}")
            return False


class HybridCache(BaseCache):
    """Hybrid memory + disk cache."""
    
    def __init__(
        self,
        memory_ratio: float = 0.3,  # 30% in memory, 70% on disk
        **kwargs
    ):
        """Initialize hybrid cache.
        
        Args:
            memory_ratio: Ratio of cache to keep in memory
        """
        super().__init__(**kwargs)
        
        memory_size = int(self.max_size_bytes * memory_ratio)
        disk_size = self.max_size_bytes - memory_size
        
        # Create memory and disk caches
        self.memory_cache = MemoryCache(
            max_size_bytes=memory_size,
            max_entries=int(self.max_entries * memory_ratio),
            eviction_policy=self.eviction_policy
        )
        
        self.disk_cache = DiskCache(
            max_size_bytes=disk_size,
            max_entries=self.max_entries - self.memory_cache.max_entries,
            eviction_policy=self.eviction_policy
        )
    
    def _store_value(self, entry: CacheEntry) -> bool:
        """Store value in hybrid cache."""
        # High priority or frequently accessed items go to memory
        if entry.priority > 0 or entry.access_count > 5:
            return self.memory_cache._store_value(entry)
        else:
            return self.disk_cache._store_value(entry)
    
    def _retrieve_value(self, key: str) -> Optional[Any]:
        """Retrieve value from hybrid cache."""
        # Try memory first
        value = self.memory_cache._retrieve_value(key)
        if value is not None:
            return value
        
        # Try disk
        value = self.disk_cache._retrieve_value(key)
        if value is not None:
            # Promote to memory if accessed frequently
            if key in self.entries:
                entry = self.entries[key]
                if entry.access_count > 3:
                    self.memory_cache._store_value(entry)
        
        return value
    
    def _remove_value(self, key: str) -> bool:
        """Remove value from hybrid cache."""
        memory_removed = self.memory_cache._remove_value(key)
        disk_removed = self.disk_cache._remove_value(key)
        return memory_removed or disk_removed


class IntelligentCacheManager:
    """Intelligent cache manager with adaptive strategies."""
    
    def __init__(
        self,
        cache_type: CacheLocation = CacheLocation.HYBRID,
        auto_tuning: bool = True,
        **cache_kwargs
    ):
        """Initialize intelligent cache manager.
        
        Args:
            cache_type: Type of cache to use
            auto_tuning: Whether to enable automatic tuning
            **cache_kwargs: Cache configuration arguments
        """
        self.cache_type = cache_type
        self.auto_tuning = auto_tuning
        
        # Create cache instance
        if cache_type == CacheLocation.MEMORY:
            self.cache = MemoryCache(**cache_kwargs)
        elif cache_type == CacheLocation.DISK:
            self.cache = DiskCache(**cache_kwargs)
        elif cache_type == CacheLocation.HYBRID:
            self.cache = HybridCache(**cache_kwargs)
        else:
            self.cache = MemoryCache(**cache_kwargs)
        
        # Auto-tuning state
        self._tuning_history = []
        self._last_tuning = time.time()
        self._tuning_interval = 300  # 5 minutes
        
        # Performance tracking
        self._operation_times = defaultdict(list)
        
        if self.auto_tuning:
            self._start_auto_tuning()
    
    def _start_auto_tuning(self):
        """Start auto-tuning thread."""
        def tuning_loop():
            while True:
                time.sleep(self._tuning_interval)
                self._auto_tune()
        
        tuning_thread = threading.Thread(target=tuning_loop, daemon=True)
        tuning_thread.start()
    
    def _auto_tune(self):
        """Perform automatic cache tuning."""
        stats = self.cache.get_statistics()
        
        # Analyze performance
        hit_rate = stats['hit_rate']
        avg_lookup_time = stats['average_lookup_time_ms']
        
        # Tuning decisions
        if hit_rate < 0.5 and len(self._tuning_history) > 0:
            # Low hit rate - consider increasing cache size or changing policy
            logger.info(f"Low hit rate detected: {hit_rate:.2%}")
            
            if self.cache.eviction_policy != CacheEvictionPolicy.ADAPTIVE:
                self.cache.eviction_policy = CacheEvictionPolicy.ADAPTIVE
                logger.info("Switched to adaptive eviction policy")
        
        elif hit_rate > 0.8 and avg_lookup_time > 10:  # > 10ms
            # High hit rate but slow lookups - consider optimization
            logger.info(f"Slow lookup times detected: {avg_lookup_time:.1f}ms")
        
        # Record tuning state
        self._tuning_history.append({
            'timestamp': time.time(),
            'hit_rate': hit_rate,
            'lookup_time': avg_lookup_time,
            'cache_size': stats['size_mb']
        })
        
        # Keep limited history
        if len(self._tuning_history) > 100:
            self._tuning_history.pop(0)
    
    def cached_computation(
        self,
        func: Callable,
        cache_key: Optional[str] = None,
        ttl: Optional[float] = None,
        priority: int = 0
    ):
        """Decorator for caching function results.
        
        Args:
            func: Function to cache
            cache_key: Custom cache key
            ttl: Time to live
            priority: Cache priority
        """
        def decorator(*args, **kwargs):
            # Generate cache key
            if cache_key:
                key = cache_key
            else:
                key = self.cache._generate_key(func.__name__, *args, **kwargs)
            
            # Try to get from cache
            start_time = time.time()
            cached_result = self.cache.get(key)
            
            if cached_result is not None:
                return cached_result
            
            # Compute result
            computation_start = time.time()
            result = func(*args, **kwargs)
            computation_time = time.time() - computation_start
            
            # Cache result
            self.cache.put(
                key=key,
                value=result,
                ttl=ttl,
                priority=priority,
                computation_time=computation_time
            )
            
            # Track performance
            total_time = time.time() - start_time
            self._operation_times[func.__name__].append({
                'total_time': total_time,
                'computation_time': computation_time,
                'cache_hit': False
            })
            
            return result
        
        return decorator
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        cache_stats = self.cache.get_statistics()
        
        # Calculate operation statistics
        operation_stats = {}
        for func_name, times in self._operation_times.items():
            if times:
                total_times = [t['total_time'] for t in times]
                comp_times = [t['computation_time'] for t in times]
                cache_hits = sum(1 for t in times if t['cache_hit'])
                
                operation_stats[func_name] = {
                    'call_count': len(times),
                    'cache_hit_rate': cache_hits / len(times),
                    'avg_total_time': sum(total_times) / len(total_times),
                    'avg_computation_time': sum(comp_times) / len(comp_times),
                    'total_time_saved': sum(comp_times) - sum(total_times)
                }
        
        return {
            'cache_stats': cache_stats,
            'operation_stats': operation_stats,
            'auto_tuning_enabled': self.auto_tuning,
            'tuning_history_length': len(self._tuning_history)
        }


# Global cache manager instance
global_cache_manager = IntelligentCacheManager()


def cached(
    ttl: Optional[float] = None,
    priority: int = 0,
    cache_key: Optional[str] = None
):
    """Decorator for caching function results.
    
    Args:
        ttl: Time to live in seconds
        priority: Cache priority
        cache_key: Custom cache key
    """
    def decorator(func: Callable) -> Callable:
        return global_cache_manager.cached_computation(
            func, cache_key=cache_key, ttl=ttl, priority=priority
        )
    
    return decorator


def clear_cache():
    """Clear the global cache."""
    global_cache_manager.cache.clear()


def get_cache_stats() -> Dict[str, Any]:
    """Get global cache statistics."""
    return global_cache_manager.get_performance_report()