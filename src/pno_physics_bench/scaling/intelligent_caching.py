"""
Intelligent Caching System for Probabilistic Neural Operators.

This module implements advanced caching strategies including adaptive caching,
semantic caching, hierarchical cache management, and distributed cache
coordination for high-performance PNO inference.
"""

import torch
import torch.nn as nn
import numpy as np
import hashlib
import pickle
import time
import threading
import queue
import os
import json
import redis
import sqlite3
import lz4.frame
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
import logging
from collections import OrderedDict, defaultdict
import weakref
from concurrent.futures import ThreadPoolExecutor, as_completed
import psutil

from ..models import ProbabilisticNeuralOperator


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    key: str
    value: Any
    timestamp: float
    access_count: int
    size_bytes: int
    ttl: Optional[float] = None
    compression_ratio: float = 1.0
    semantic_hash: Optional[str] = None


@dataclass
class CacheStats:
    """Cache performance statistics."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    size_bytes: int = 0
    num_entries: int = 0
    avg_access_time: float = 0.0
    compression_ratio: float = 1.0
    memory_efficiency: float = 0.0


class CachePolicy(ABC):
    """Abstract base class for cache eviction policies."""
    
    @abstractmethod
    def should_evict(self, entry: CacheEntry, current_time: float) -> bool:
        """Determine if entry should be evicted."""
        pass
    
    @abstractmethod
    def get_eviction_priority(self, entry: CacheEntry, current_time: float) -> float:
        """Get eviction priority (higher = more likely to evict)."""
        pass


class LRUPolicy(CachePolicy):
    """Least Recently Used eviction policy."""
    
    def should_evict(self, entry: CacheEntry, current_time: float) -> bool:
        return True  # LRU always allows eviction
    
    def get_eviction_priority(self, entry: CacheEntry, current_time: float) -> float:
        return current_time - entry.timestamp


class LFUPolicy(CachePolicy):
    """Least Frequently Used eviction policy."""
    
    def should_evict(self, entry: CacheEntry, current_time: float) -> bool:
        return True
    
    def get_eviction_priority(self, entry: CacheEntry, current_time: float) -> float:
        return -entry.access_count  # Negative because lower access count = higher priority


class TTLPolicy(CachePolicy):
    """Time-To-Live eviction policy."""
    
    def __init__(self, default_ttl: float = 3600.0):
        self.default_ttl = default_ttl
    
    def should_evict(self, entry: CacheEntry, current_time: float) -> bool:
        ttl = entry.ttl if entry.ttl is not None else self.default_ttl
        return current_time - entry.timestamp > ttl
    
    def get_eviction_priority(self, entry: CacheEntry, current_time: float) -> float:
        ttl = entry.ttl if entry.ttl is not None else self.default_ttl
        time_left = ttl - (current_time - entry.timestamp)
        return -time_left  # Negative because less time left = higher priority


class AdaptivePolicy(CachePolicy):
    """Adaptive policy that combines multiple strategies."""
    
    def __init__(self, weights: Dict[str, float] = None):
        self.weights = weights or {
            'recency': 0.4,
            'frequency': 0.3,
            'size': 0.2,
            'semantic': 0.1
        }
        self.access_pattern_history = defaultdict(list)
    
    def should_evict(self, entry: CacheEntry, current_time: float) -> bool:
        return True
    
    def get_eviction_priority(self, entry: CacheEntry, current_time: float) -> float:
        # Recency component (LRU-like)
        recency_score = current_time - entry.timestamp
        
        # Frequency component (LFU-like)
        frequency_score = -entry.access_count
        
        # Size component (favor evicting larger entries)
        size_score = entry.size_bytes
        
        # Semantic component (based on access patterns)
        semantic_score = self._compute_semantic_score(entry)
        
        # Weighted combination
        total_score = (
            self.weights['recency'] * recency_score +
            self.weights['frequency'] * frequency_score +
            self.weights['size'] * size_score +
            self.weights['semantic'] * semantic_score
        )
        
        return total_score
    
    def _compute_semantic_score(self, entry: CacheEntry) -> float:
        """Compute semantic score based on access patterns."""
        if entry.semantic_hash not in self.access_pattern_history:
            return 0.0
        
        history = self.access_pattern_history[entry.semantic_hash]
        if len(history) < 2:
            return 0.0
        
        # Compute access frequency trend
        recent_accesses = [t for t in history if time.time() - t < 3600]  # Last hour
        return -len(recent_accesses)  # More recent accesses = lower eviction priority


class CompressionEngine:
    """Data compression engine for cache optimization."""
    
    def __init__(self, compression_threshold: int = 1024):
        self.compression_threshold = compression_threshold
    
    def should_compress(self, data: bytes) -> bool:
        """Determine if data should be compressed."""
        return len(data) > self.compression_threshold
    
    def compress(self, data: bytes) -> Tuple[bytes, float]:
        """Compress data and return compression ratio."""
        if not self.should_compress(data):
            return data, 1.0
        
        try:
            compressed = lz4.frame.compress(data)
            compression_ratio = len(data) / len(compressed)
            return compressed, compression_ratio
        except Exception:
            return data, 1.0
    
    def decompress(self, compressed_data: bytes, compression_ratio: float) -> bytes:
        """Decompress data."""
        if compression_ratio <= 1.01:  # Not compressed
            return compressed_data
        
        try:
            return lz4.frame.decompress(compressed_data)
        except Exception:
            return compressed_data


class SemanticHasher:
    """Generate semantic hashes for cache entries."""
    
    def __init__(self, similarity_threshold: float = 0.95):
        self.similarity_threshold = similarity_threshold
    
    def compute_tensor_hash(self, tensor: torch.Tensor, precision: int = 3) -> str:
        """Compute semantic hash for tensor based on statistical properties."""
        # Use statistical moments for semantic similarity
        mean = tensor.mean().item()
        std = tensor.std().item()
        min_val = tensor.min().item()
        max_val = tensor.max().item()
        
        # Quantize for similarity
        mean = round(mean, precision)
        std = round(std, precision)
        min_val = round(min_val, precision)
        max_val = round(max_val, precision)
        
        # Shape information
        shape_str = '_'.join(map(str, tensor.shape))
        
        # Create semantic signature
        semantic_data = f"{shape_str}_{mean}_{std}_{min_val}_{max_val}"
        
        return hashlib.sha256(semantic_data.encode()).hexdigest()[:16]
    
    def compute_input_hash(self, input_data: Dict[str, Any]) -> str:
        """Compute semantic hash for general input data."""
        if isinstance(input_data, torch.Tensor):
            return self.compute_tensor_hash(input_data)
        
        # For general data, use serialization
        try:
            serialized = json.dumps(input_data, sort_keys=True, default=str)
            return hashlib.sha256(serialized.encode()).hexdigest()[:16]
        except:
            return hashlib.sha256(str(input_data).encode()).hexdigest()[:16]


class LocalCache:
    """Local in-memory cache with intelligent eviction."""
    
    def __init__(
        self,
        max_size_bytes: int = 1024**3,  # 1GB
        max_entries: int = 10000,
        policy: CachePolicy = None,
        enable_compression: bool = True
    ):
        self.max_size_bytes = max_size_bytes
        self.max_entries = max_entries
        self.policy = policy or AdaptivePolicy()
        self.enable_compression = enable_compression
        
        # Storage
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.stats = CacheStats()
        
        # Components
        self.compressor = CompressionEngine()
        self.semantic_hasher = SemanticHasher()
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Background cleanup
        self.cleanup_thread = threading.Thread(target=self._background_cleanup, daemon=True)
        self.cleanup_thread.start()
        
        # Logging
        self.logger = logging.getLogger(__name__)
    
    def _serialize_value(self, value: Any) -> bytes:
        """Serialize value for storage."""
        return pickle.dumps(value)
    
    def _deserialize_value(self, data: bytes) -> Any:
        """Deserialize value from storage."""
        return pickle.loads(data)
    
    def _compute_cache_key(self, key_data: Dict[str, Any]) -> str:
        """Compute cache key from input data."""
        serialized = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode()).hexdigest()
    
    def _should_evict_entry(self, entry: CacheEntry) -> bool:
        """Check if entry should be evicted."""
        current_time = time.time()
        
        # Check TTL expiration
        if isinstance(self.policy, TTLPolicy) or hasattr(self.policy, 'default_ttl'):
            if self.policy.should_evict(entry, current_time):
                return True
        
        # Check size limits
        if self.stats.size_bytes > self.max_size_bytes or self.stats.num_entries > self.max_entries:
            return True
        
        return False
    
    def _evict_entries(self, target_size: Optional[int] = None):
        """Evict entries based on policy."""
        if not self.cache:
            return
        
        current_time = time.time()
        target_size = target_size or int(self.max_size_bytes * 0.8)  # Evict to 80% capacity
        
        # Get eviction candidates
        candidates = []
        for key, entry in self.cache.items():
            if self._should_evict_entry(entry):
                priority = self.policy.get_eviction_priority(entry, current_time)
                candidates.append((priority, key, entry))
        
        # Sort by eviction priority
        candidates.sort(reverse=True)
        
        # Evict until target size is reached
        for priority, key, entry in candidates:
            if self.stats.size_bytes <= target_size and self.stats.num_entries <= self.max_entries:
                break
            
            self._remove_entry(key)
            self.stats.evictions += 1
    
    def _remove_entry(self, key: str):
        """Remove entry from cache."""
        if key in self.cache:
            entry = self.cache[key]
            del self.cache[key]
            self.stats.size_bytes -= entry.size_bytes
            self.stats.num_entries -= 1
    
    def _background_cleanup(self):
        """Background thread for cache cleanup."""
        while True:
            try:
                time.sleep(60)  # Cleanup every minute
                with self.lock:
                    self._evict_entries()
            except Exception as e:
                self.logger.error(f"Background cleanup error: {e}")
    
    def put(
        self,
        key_data: Dict[str, Any],
        value: Any,
        ttl: Optional[float] = None
    ) -> str:
        """Store value in cache."""
        with self.lock:
            # Compute cache key
            cache_key = self._compute_cache_key(key_data)
            
            # Serialize value
            serialized_value = self._serialize_value(value)
            
            # Compress if enabled
            compressed_value = serialized_value
            compression_ratio = 1.0
            
            if self.enable_compression:
                compressed_value, compression_ratio = self.compressor.compress(serialized_value)
            
            # Compute semantic hash
            semantic_hash = self.semantic_hasher.compute_input_hash(key_data)
            
            # Create cache entry
            entry = CacheEntry(
                key=cache_key,
                value=compressed_value,
                timestamp=time.time(),
                access_count=0,
                size_bytes=len(compressed_value),
                ttl=ttl,
                compression_ratio=compression_ratio,
                semantic_hash=semantic_hash
            )
            
            # Remove existing entry if present
            if cache_key in self.cache:
                self._remove_entry(cache_key)
            
            # Check if we need to evict
            if (self.stats.size_bytes + entry.size_bytes > self.max_size_bytes or
                self.stats.num_entries >= self.max_entries):
                self._evict_entries(self.max_size_bytes - entry.size_bytes)
            
            # Add new entry
            self.cache[cache_key] = entry
            self.stats.size_bytes += entry.size_bytes
            self.stats.num_entries += 1
            
            self.logger.debug(f"Cached entry: {cache_key[:8]}... (size: {entry.size_bytes} bytes)")
            
            return cache_key
    
    def get(self, key_data: Dict[str, Any]) -> Tuple[Optional[Any], bool]:
        """Retrieve value from cache."""
        with self.lock:
            start_time = time.time()
            
            # Compute cache key
            cache_key = self._compute_cache_key(key_data)
            
            if cache_key not in self.cache:
                self.stats.misses += 1
                return None, False
            
            entry = self.cache[cache_key]
            
            # Check TTL expiration
            current_time = time.time()
            if self._should_evict_entry(entry):
                self._remove_entry(cache_key)
                self.stats.misses += 1
                return None, False
            
            # Update access statistics
            entry.access_count += 1
            entry.timestamp = current_time
            
            # Move to end (for LRU behavior)
            self.cache.move_to_end(cache_key)
            
            # Decompress value
            decompressed_value = entry.value
            if entry.compression_ratio > 1.01:
                decompressed_value = self.compressor.decompress(entry.value, entry.compression_ratio)
            
            # Deserialize value
            value = self._deserialize_value(decompressed_value)
            
            # Update statistics
            self.stats.hits += 1
            access_time = time.time() - start_time
            self.stats.avg_access_time = (
                (self.stats.avg_access_time * (self.stats.hits - 1) + access_time) / self.stats.hits
            )
            
            return value, True
    
    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        with self.lock:
            hit_rate = self.stats.hits / (self.stats.hits + self.stats.misses) if (self.stats.hits + self.stats.misses) > 0 else 0.0
            
            total_uncompressed_size = sum(
                entry.size_bytes * entry.compression_ratio
                for entry in self.cache.values()
            )
            
            compression_ratio = total_uncompressed_size / self.stats.size_bytes if self.stats.size_bytes > 0 else 1.0
            
            memory_efficiency = hit_rate * compression_ratio
            
            stats = CacheStats(
                hits=self.stats.hits,
                misses=self.stats.misses,
                evictions=self.stats.evictions,
                size_bytes=self.stats.size_bytes,
                num_entries=self.stats.num_entries,
                avg_access_time=self.stats.avg_access_time,
                compression_ratio=compression_ratio,
                memory_efficiency=memory_efficiency
            )
            
            return stats
    
    def clear(self):
        """Clear all cache entries."""
        with self.lock:
            self.cache.clear()
            self.stats = CacheStats()


class DistributedCache:
    """Distributed cache using Redis backend."""
    
    def __init__(
        self,
        redis_config: Dict[str, Any] = None,
        local_cache: Optional[LocalCache] = None,
        enable_local_fallback: bool = True
    ):
        self.redis_config = redis_config or {
            'host': 'localhost',
            'port': 6379,
            'db': 0,
            'decode_responses': False
        }
        
        self.local_cache = local_cache
        self.enable_local_fallback = enable_local_fallback
        
        # Initialize Redis connection
        try:
            self.redis_client = redis.Redis(**self.redis_config)
            self.redis_client.ping()  # Test connection
            self.redis_available = True
        except Exception as e:
            self.redis_available = False
            logging.warning(f"Redis not available: {e}")
        
        # Components
        self.compressor = CompressionEngine()
        self.semantic_hasher = SemanticHasher()
        
        # Statistics
        self.stats = CacheStats()
        
        # Logging
        self.logger = logging.getLogger(__name__)
    
    def _compute_cache_key(self, key_data: Dict[str, Any]) -> str:
        """Compute cache key from input data."""
        serialized = json.dumps(key_data, sort_keys=True, default=str)
        return f"pno_cache:{hashlib.sha256(serialized.encode()).hexdigest()}"
    
    def put(
        self,
        key_data: Dict[str, Any],
        value: Any,
        ttl: Optional[int] = None
    ) -> str:
        """Store value in distributed cache."""
        cache_key = self._compute_cache_key(key_data)
        
        # Serialize and compress
        serialized_value = pickle.dumps(value)
        compressed_value, compression_ratio = self.compressor.compress(serialized_value)
        
        # Store in Redis
        if self.redis_available:
            try:
                # Create metadata
                metadata = {
                    'compression_ratio': compression_ratio,
                    'semantic_hash': self.semantic_hasher.compute_input_hash(key_data),
                    'timestamp': time.time(),
                    'size_bytes': len(compressed_value)
                }
                
                # Store data and metadata separately
                pipe = self.redis_client.pipeline()
                pipe.set(cache_key, compressed_value, ex=ttl)
                pipe.set(f"{cache_key}:meta", json.dumps(metadata), ex=ttl)
                pipe.execute()
                
                self.logger.debug(f"Stored in Redis: {cache_key}")
                
            except Exception as e:
                self.logger.error(f"Redis storage failed: {e}")
                # Fallback to local cache
                if self.local_cache and self.enable_local_fallback:
                    return self.local_cache.put(key_data, value, ttl)
        
        # Fallback to local cache
        elif self.local_cache and self.enable_local_fallback:
            return self.local_cache.put(key_data, value, ttl)
        
        return cache_key
    
    def get(self, key_data: Dict[str, Any]) -> Tuple[Optional[Any], bool]:
        """Retrieve value from distributed cache."""
        cache_key = self._compute_cache_key(key_data)
        
        # Try Redis first
        if self.redis_available:
            try:
                # Get data and metadata
                pipe = self.redis_client.pipeline()
                pipe.get(cache_key)
                pipe.get(f"{cache_key}:meta")
                results = pipe.execute()
                
                compressed_value, metadata_str = results
                
                if compressed_value is not None and metadata_str is not None:
                    # Parse metadata
                    metadata = json.loads(metadata_str)
                    compression_ratio = metadata.get('compression_ratio', 1.0)
                    
                    # Decompress and deserialize
                    decompressed_value = self.compressor.decompress(compressed_value, compression_ratio)
                    value = pickle.loads(decompressed_value)
                    
                    self.stats.hits += 1
                    return value, True
                
                else:
                    self.stats.misses += 1
                    
            except Exception as e:
                self.logger.error(f"Redis retrieval failed: {e}")
        
        # Fallback to local cache
        if self.local_cache and self.enable_local_fallback:
            return self.local_cache.get(key_data)
        
        self.stats.misses += 1
        return None, False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get distributed cache statistics."""
        stats = {
            'local_stats': self.local_cache.get_stats() if self.local_cache else None,
            'redis_available': self.redis_available,
            'distributed_hits': self.stats.hits,
            'distributed_misses': self.stats.misses
        }
        
        if self.redis_available:
            try:
                redis_info = self.redis_client.info('memory')
                stats['redis_memory'] = {
                    'used_memory': redis_info.get('used_memory', 0),
                    'used_memory_human': redis_info.get('used_memory_human', '0B'),
                    'maxmemory': redis_info.get('maxmemory', 0)
                }
            except Exception:
                pass
        
        return stats


class CachedPNOInference:
    """PNO inference with intelligent caching."""
    
    def __init__(
        self,
        model: ProbabilisticNeuralOperator,
        cache_system: Union[LocalCache, DistributedCache],
        similarity_threshold: float = 0.95,
        enable_semantic_caching: bool = True
    ):
        self.model = model
        self.cache_system = cache_system
        self.similarity_threshold = similarity_threshold
        self.enable_semantic_caching = enable_semantic_caching
        
        # Semantic hasher for input similarity
        self.semantic_hasher = SemanticHasher(similarity_threshold)
        
        # Performance tracking
        self.inference_stats = {
            'cache_hits': 0,
            'cache_misses': 0,
            'semantic_hits': 0,
            'total_inference_time': 0.0,
            'total_cache_time': 0.0
        }
        
        # Logging
        self.logger = logging.getLogger(__name__)
    
    def predict_with_uncertainty(
        self,
        input_tensor: torch.Tensor,
        num_samples: int = 100,
        use_cache: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict with uncertainty using intelligent caching."""
        cache_start_time = time.time()
        
        if use_cache:
            # Compute cache key features
            input_features = {
                'shape': list(input_tensor.shape),
                'dtype': str(input_tensor.dtype),
                'device': str(input_tensor.device),
                'semantic_hash': self.semantic_hasher.compute_tensor_hash(input_tensor),
                'num_samples': num_samples,
                'model_hash': 'model_v1'
            }
            
            # Check cache
            cached_result, cache_hit = self.cache_system.get(input_features)
            
            if cache_hit:
                self.inference_stats['cache_hits'] += 1
                self.inference_stats['total_cache_time'] += time.time() - cache_start_time
                self.logger.debug("Cache hit - returning cached result")
                return cached_result
        
        cache_time = time.time() - cache_start_time
        self.inference_stats['total_cache_time'] += cache_time
        
        # Cache miss - run inference
        inference_start_time = time.time()
        
        with torch.no_grad():
            prediction, uncertainty = self.model.predict_with_uncertainty(
                input_tensor, num_samples=num_samples
            )
        
        inference_time = time.time() - inference_start_time
        self.inference_stats['total_inference_time'] += inference_time
        
        # Store result in cache
        if use_cache:
            result = (prediction, uncertainty)
            
            # Calculate TTL based on result stability
            ttl = self._compute_adaptive_ttl(uncertainty, inference_time)
            
            self.cache_system.put(input_features, result, ttl=ttl)
            
            self.inference_stats['cache_misses'] += 1
            self.logger.debug(f"Inference completed in {inference_time:.3f}s, cached with TTL {ttl}s")
        
        return prediction, uncertainty
    
    def _compute_adaptive_ttl(self, uncertainty: torch.Tensor, inference_time: float) -> int:
        """Compute adaptive TTL based on uncertainty and computation cost."""
        base_ttl = 3600  # 1 hour
        uncertainty_factor = 1.0 / (uncertainty.mean().item() + 0.1)
        cost_factor = min(10.0, inference_time / 0.1)
        adaptive_ttl = int(base_ttl * uncertainty_factor * cost_factor)
        return max(300, min(86400, adaptive_ttl))  # 5 minutes to 24 hours
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        total_requests = self.inference_stats['cache_hits'] + self.inference_stats['cache_misses']
        
        cache_hit_rate = (
            self.inference_stats['cache_hits'] / total_requests
            if total_requests > 0 else 0.0
        )
        
        return {
            'cache_performance': {
                'hit_rate': cache_hit_rate,
                'total_requests': total_requests,
                'cache_hits': self.inference_stats['cache_hits'],
                'cache_misses': self.inference_stats['cache_misses']
            },
            'cache_system_stats': self.cache_system.get_stats()
        }


def create_intelligent_caching_example():
    """Create example intelligent caching setup."""
    from ..models import ProbabilisticNeuralOperator
    
    # Create model
    model = ProbabilisticNeuralOperator(
        input_dim=3,
        hidden_dim=128,
        num_layers=4,
        modes=16
    )
    
    # Create cache
    cache = LocalCache(
        max_size_bytes=512 * 1024 * 1024,  # 512MB
        max_entries=5000,
        policy=AdaptivePolicy(),
        enable_compression=True
    )
    
    # Create cached inference system
    cached_inference = CachedPNOInference(
        model=model,
        cache_system=cache,
        similarity_threshold=0.95,
        enable_semantic_caching=True
    )
    
    return cached_inference


if __name__ == "__main__":
    print("🧠 Intelligent Caching System ready for production!")

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