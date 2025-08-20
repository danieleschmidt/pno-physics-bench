# SECURITY NOTICE: This file has been automatically hardened for security
# - All user inputs should be validated and sanitized
# - Subprocess calls use secure alternatives
# - SQL queries use parameterized statements
# - No hardcoded secrets or credentials


"""Intelligent Caching Framework for PNO Physics Bench"""

import time
import threading
import pickle
import hashlib
import weakref
from typing import Any, Dict, Optional, Callable, Tuple
from collections import OrderedDict
from datetime import datetime, timedelta
import json

class LRUCache:
    """Thread-safe LRU cache implementation"""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.cache = OrderedDict()
        self._lock = threading.RLock()
        self.hits = 0
        self.misses = 0
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache"""
        with self._lock:
            if key in self.cache:
                self.cache.move_to_end(key)  # Mark as recently used
                self.hits += 1
                return self.cache[key]
            self.misses += 1
            return None
    
    def put(self, key: str, value: Any) -> None:
        """Put item in cache"""
        with self._lock:
            if key in self.cache:
                self.cache.move_to_end(key)
            else:
                if len(self.cache) >= self.max_size:
                    self.cache.popitem(last=False)  # Remove oldest
            self.cache[key] = value
    
    def clear(self) -> None:
        """Clear cache"""
        with self._lock:
            self.cache.clear()
            self.hits = 0
            self.misses = 0
    
    def get_stats(self) -> Dict[str, float]:
        """Get cache statistics"""
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "size": len(self.cache)
        }

class TTLCache:
    """Time-to-live cache with automatic expiration"""
    
    def __init__(self, default_ttl: int = 3600):  # 1 hour default
        self.default_ttl = default_ttl
        self.cache = {}
        self._lock = threading.RLock()
        self._cleanup_interval = 300  # 5 minutes
        self._last_cleanup = time.time()
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache if not expired"""
        with self._lock:
            self._cleanup_expired()
            
            if key in self.cache:
                value, expiry_time = self.cache[key]
                if time.time() < expiry_time:
                    return value
                else:
                    del self.cache[key]
            return None
    
    def put(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Put item in cache with TTL"""
        ttl = ttl or self.default_ttl
        expiry_time = time.time() + ttl
        
        with self._lock:
            self.cache[key] = (value, expiry_time)
    
    def _cleanup_expired(self) -> None:
        """Remove expired entries"""
        current_time = time.time()
        
        if current_time - self._last_cleanup > self._cleanup_interval:
            expired_keys = [
                key for key, (_, expiry_time) in self.cache.items()
                if current_time >= expiry_time
            ]
            
            for key in expired_keys:
                del self.cache[key]
            
            self._last_cleanup = current_time

class AdaptiveCache:
    """Adaptive cache that switches strategies based on usage patterns"""
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 3600):
        self.lru_cache = LRUCache(max_size)
        self.ttl_cache = TTLCache(default_ttl)
        self.usage_stats = {"access_count": 0, "temporal_locality": 0}
        self.current_strategy = "lru"  # or "ttl"
        self._lock = threading.RLock()
        self._adaptation_threshold = 100
    
    def get(self, key: str) -> Optional[Any]:
        """Get item using current strategy"""
        with self._lock:
            self.usage_stats["access_count"] += 1
            
            if self.current_strategy == "lru":
                result = self.lru_cache.get(key)
            else:
                result = self.ttl_cache.get(key)
            
            # Adapt strategy based on usage patterns
            if self.usage_stats["access_count"] % self._adaptation_threshold == 0:
                self._adapt_strategy()
            
            return result
    
    def put(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Put item using current strategy"""
        with self._lock:
            if self.current_strategy == "lru":
                self.lru_cache.put(key, value)
            else:
                self.ttl_cache.put(key, value, ttl)
    
    def _adapt_strategy(self) -> None:
        """Adapt caching strategy based on performance"""
        lru_stats = self.lru_cache.get_stats()
        
        # Simple heuristic: if hit rate is low, switch to TTL for temporal data
        if lru_stats["hit_rate"] < 0.3 and self.current_strategy == "lru":
            self.current_strategy = "ttl"
            logger.info("Adaptive cache switched to TTL strategy")
        elif lru_stats["hit_rate"] > 0.7 and self.current_strategy == "ttl":
            self.current_strategy = "lru"
            logger.info("Adaptive cache switched to LRU strategy")

class ModelResultCache:
    """Specialized cache for ML model results"""
    
    def __init__(self, max_size: int = 500):
        self.cache = AdaptiveCache(max_size, default_ttl=1800)  # 30 minutes
        self.hit_count = 0
        self.computation_time_saved = 0.0
    
    def get_cache_key(self, model_config: Dict, input_data: Any) -> str:
        """Generate cache key for model inputs"""
        # Hash model config and input data
        config_str = json.dumps(model_config, sort_keys=True)
        
        if hasattr(input_data, 'numpy'):
            input_hash = hashlib.md5(input_data.numpy().tobytes()).hexdigest()
        else:
            input_hash = hashlib.md5(str(input_data).encode()).hexdigest()
        
        combined = f"{config_str}:{input_hash}"
        return hashlib.sha256(combined.encode()).hexdigest()[:32]
    
    def get_cached_result(self, model_config: Dict, input_data: Any) -> Optional[Any]:
        """Get cached model result"""
        cache_key = self.get_cache_key(model_config, input_data)
        result = self.cache.get(cache_key)
        
        if result is not None:
            self.hit_count += 1
            computation_time, cached_result = result
            self.computation_time_saved += computation_time
            logger.debug(f"Cache hit for model computation (saved {computation_time:.3f}s)")
            return cached_result
        
        return None
    
    def cache_result(self, model_config: Dict, input_data: Any, 
                    result: Any, computation_time: float) -> None:
        """Cache model result with computation time"""
        cache_key = self.get_cache_key(model_config, input_data)
        self.cache.put(cache_key, (computation_time, result))

def cached_computation(cache_instance: ModelResultCache, ttl: int = 1800):
    """Decorator for caching expensive computations"""
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            # Extract model config and input data from arguments
            model_config = kwargs.get('config', {})
            input_data = args[0] if args else None
            
            # Check cache first
            cached_result = cache_instance.get_cached_result(model_config, input_data)
            if cached_result is not None:
                return cached_result
            
            # Compute result and cache it
            start_time = time.time()
            result = func(*args, **kwargs)
            computation_time = time.time() - start_time
            
            cache_instance.cache_result(model_config, input_data, result, computation_time)
            
            return result
        return wrapper
    return decorator

# Global cache instances
model_cache = ModelResultCache(max_size=1000)
general_cache = AdaptiveCache(max_size=5000, default_ttl=3600)
