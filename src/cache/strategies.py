"""
Cache strategies implementation for SmartRAG.

This module provides various caching strategies including TTL, LRU, LFU, FIFO,
and a unified cache manager for strategy orchestration.
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from dataclasses import dataclass, field
from collections import OrderedDict, defaultdict
import heapq
import hashlib

from .redis_client import RedisClient

logger = logging.getLogger(__name__)


@dataclass
class CacheItem:
    """Represents a cached item with metadata."""
    key: str
    value: Any
    created_at: float = field(default_factory=time.time)
    accessed_at: float = field(default_factory=time.time)
    access_count: int = 0
    ttl: Optional[int] = None
    size: int = 0
    
    @property
    def is_expired(self) -> bool:
        """Check if item is expired."""
        if self.ttl is None:
            return False
        return time.time() - self.created_at > self.ttl
    
    def touch(self):
        """Update access metadata."""
        self.accessed_at = time.time()
        self.access_count += 1


class CacheStrategy(ABC):
    """Abstract base class for cache strategies."""
    
    def __init__(self, max_size: int = 1000):
        """
        Initialize cache strategy.
        
        Args:
            max_size: Maximum number of items to cache
        """
        self.max_size = max_size
        self.items: Dict[str, CacheItem] = {}
        self._lock = asyncio.Lock()
        
        logger.debug(f"Initialized {self.__class__.__name__} with max_size={max_size}")
    
    @abstractmethod
    async def should_evict(self) -> bool:
        """Check if eviction is needed."""
        pass
    
    @abstractmethod
    async def select_eviction_candidates(self, count: int = 1) -> List[str]:
        """Select items for eviction."""
        pass
    
    async def get(self, key: str) -> Optional[CacheItem]:
        """Get item from cache."""
        async with self._lock:
            item = self.items.get(key)
            if item is None:
                return None
            
            if item.is_expired:
                del self.items[key]
                return None
            
            item.touch()
            await self._on_access(key, item)
            return item
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set item in cache."""
        async with self._lock:
            # Check if eviction is needed
            if key not in self.items and await self.should_evict():
                candidates = await self.select_eviction_candidates()
                for candidate_key in candidates:
                    if candidate_key in self.items:
                        del self.items[candidate_key]
                        logger.debug(f"Evicted key: {candidate_key}")
            
            # Create cache item
            size = self._calculate_size(value)
            item = CacheItem(
                key=key,
                value=value,
                ttl=ttl,
                size=size
            )
            
            self.items[key] = item
            await self._on_set(key, item)
            return True
    
    async def delete(self, key: str) -> bool:
        """Delete item from cache."""
        async with self._lock:
            if key in self.items:
                del self.items[key]
                await self._on_delete(key)
                return True
            return False
    
    async def clear(self):
        """Clear all items from cache."""
        async with self._lock:
            self.items.clear()
            await self._on_clear()
    
    def _calculate_size(self, value: Any) -> int:
        """Calculate approximate size of value."""
        try:
            import sys
            return sys.getsizeof(value)
        except Exception:
            return 1  # Fallback size
    
    async def _on_access(self, key: str, item: CacheItem):
        """Hook called when item is accessed."""
        pass
    
    async def _on_set(self, key: str, item: CacheItem):
        """Hook called when item is set."""
        pass
    
    async def _on_delete(self, key: str):
        """Hook called when item is deleted."""
        pass
    
    async def _on_clear(self):
        """Hook called when cache is cleared."""
        pass
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_size = sum(item.size for item in self.items.values())
        return {
            "item_count": len(self.items),
            "max_size": self.max_size,
            "total_size": total_size,
            "utilization": len(self.items) / self.max_size if self.max_size > 0 else 0.0
        }


class TTLStrategy(CacheStrategy):
    """Time-To-Live cache strategy."""
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 3600):
        """
        Initialize TTL strategy.
        
        Args:
            max_size: Maximum number of items
            default_ttl: Default TTL in seconds
        """
        super().__init__(max_size)
        self.default_ttl = default_ttl
        self._cleanup_task: Optional[asyncio.Task] = None
        self._start_cleanup_task()
    
    def _start_cleanup_task(self):
        """Start background cleanup task."""
        self._cleanup_task = asyncio.create_task(self._cleanup_expired())
    
    async def _cleanup_expired(self):
        """Background task to clean up expired items."""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute
                
                async with self._lock:
                    expired_keys = [
                        key for key, item in self.items.items()
                        if item.is_expired
                    ]
                    
                    for key in expired_keys:
                        del self.items[key]
                    
                    if expired_keys:
                        logger.debug(f"Cleaned up {len(expired_keys)} expired items")
                        
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in TTL cleanup: {e}")
    
    async def should_evict(self) -> bool:
        """Check if eviction is needed."""
        return len(self.items) >= self.max_size
    
    async def select_eviction_candidates(self, count: int = 1) -> List[str]:
        """Select expired or oldest items for eviction."""
        current_time = time.time()
        
        # First, try to evict expired items
        expired_candidates = [
            key for key, item in self.items.items()
            if item.is_expired
        ]
        
        if len(expired_candidates) >= count:
            return expired_candidates[:count]
        
        # If not enough expired items, evict oldest
        remaining_count = count - len(expired_candidates)
        oldest_candidates = sorted(
            self.items.keys(),
            key=lambda k: self.items[k].created_at
        )[:remaining_count]
        
        return expired_candidates + oldest_candidates
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set item with TTL."""
        if ttl is None:
            ttl = self.default_ttl
        return await super().set(key, value, ttl)
    
    async def close(self):
        """Close TTL strategy and cleanup."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass


class LRUStrategy(CacheStrategy):
    """Least Recently Used cache strategy."""
    
    def __init__(self, max_size: int = 1000):
        """Initialize LRU strategy."""
        super().__init__(max_size)
        self.access_order: OrderedDict[str, None] = OrderedDict()
    
    async def should_evict(self) -> bool:
        """Check if eviction is needed."""
        return len(self.items) >= self.max_size
    
    async def select_eviction_candidates(self, count: int = 1) -> List[str]:
        """Select least recently used items."""
        candidates = []
        for key in self.access_order:
            if key in self.items:
                candidates.append(key)
                if len(candidates) >= count:
                    break
        return candidates
    
    async def _on_access(self, key: str, item: CacheItem):
        """Update access order on item access."""
        self.access_order.move_to_end(key, last=True)
    
    async def _on_set(self, key: str, item: CacheItem):
        """Update access order on item set."""
        self.access_order[key] = None
        self.access_order.move_to_end(key, last=True)
    
    async def _on_delete(self, key: str):
        """Remove from access order on delete."""
        self.access_order.pop(key, None)
    
    async def _on_clear(self):
        """Clear access order."""
        self.access_order.clear()


class LFUStrategy(CacheStrategy):
    """Least Frequently Used cache strategy."""
    
    def __init__(self, max_size: int = 1000):
        """Initialize LFU strategy."""
        super().__init__(max_size)
    
    async def should_evict(self) -> bool:
        """Check if eviction is needed."""
        return len(self.items) >= self.max_size
    
    async def select_eviction_candidates(self, count: int = 1) -> List[str]:
        """Select least frequently used items."""
        # Sort by access count, then by creation time
        sorted_items = sorted(
            self.items.items(),
            key=lambda x: (x[1].access_count, x[1].created_at)
        )
        
        return [key for key, _ in sorted_items[:count]]


class FIFOStrategy(CacheStrategy):
    """First In, First Out cache strategy."""
    
    def __init__(self, max_size: int = 1000):
        """Initialize FIFO strategy."""
        super().__init__(max_size)
    
    async def should_evict(self) -> bool:
        """Check if eviction is needed."""
        return len(self.items) >= self.max_size
    
    async def select_eviction_candidates(self, count: int = 1) -> List[str]:
        """Select oldest items (first in)."""
        sorted_items = sorted(
            self.items.items(),
            key=lambda x: x[1].created_at
        )
        
        return [key for key, _ in sorted_items[:count]]


class AdaptiveStrategy(CacheStrategy):
    """Adaptive cache strategy that switches between LRU and LFU."""
    
    def __init__(self, max_size: int = 1000, adaptation_window: int = 1000):
        """
        Initialize adaptive strategy.
        
        Args:
            max_size: Maximum number of items
            adaptation_window: Window size for adaptation decisions
        """
        super().__init__(max_size)
        self.adaptation_window = adaptation_window
        self.lru_strategy = LRUStrategy(max_size)
        self.lfu_strategy = LFUStrategy(max_size)
        self.current_strategy = self.lru_strategy
        
        # Performance tracking
        self.lru_hits = 0
        self.lru_misses = 0
        self.lfu_hits = 0
        self.lfu_misses = 0
        self.decision_count = 0
    
    async def _adapt_strategy(self):
        """Adapt strategy based on performance."""
        if self.decision_count % self.adaptation_window == 0:
            lru_hit_rate = (
                self.lru_hits / (self.lru_hits + self.lru_misses)
                if (self.lru_hits + self.lru_misses) > 0 else 0
            )
            lfu_hit_rate = (
                self.lfu_hits / (self.lfu_hits + self.lfu_misses)
                if (self.lfu_hits + self.lfu_misses) > 0 else 0
            )
            
            # Switch to better performing strategy
            old_strategy = self.current_strategy
            if lfu_hit_rate > lru_hit_rate:
                self.current_strategy = self.lfu_strategy
            else:
                self.current_strategy = self.lru_strategy
            
            if old_strategy != self.current_strategy:
                strategy_name = "LFU" if self.current_strategy == self.lfu_strategy else "LRU"
                logger.info(f"Adapted to {strategy_name} strategy")
    
    async def should_evict(self) -> bool:
        """Check if eviction is needed."""
        return await self.current_strategy.should_evict()
    
    async def select_eviction_candidates(self, count: int = 1) -> List[str]:
        """Select eviction candidates using current strategy."""
        self.decision_count += 1
        await self._adapt_strategy()
        return await self.current_strategy.select_eviction_candidates(count)


class CacheManager:
    """Unified cache manager with Redis backend and local strategy."""
    
    def __init__(
        self,
        redis_client: Optional[RedisClient] = None,
        local_strategy: Optional[CacheStrategy] = None,
        enable_local_cache: bool = True,
        enable_write_through: bool = True,
        enable_write_behind: bool = False,
        write_behind_delay: float = 1.0
    ):
        """
        Initialize cache manager.
        
        Args:
            redis_client: Redis client for distributed caching
            local_strategy: Local cache strategy
            enable_local_cache: Whether to use local cache
            enable_write_through: Whether to use write-through caching
            enable_write_behind: Whether to use write-behind caching
            write_behind_delay: Delay for write-behind operations
        """
        self.redis_client = redis_client
        self.local_strategy = local_strategy or LRUStrategy()
        self.enable_local_cache = enable_local_cache
        self.enable_write_through = enable_write_through
        self.enable_write_behind = enable_write_behind
        self.write_behind_delay = write_behind_delay
        
        # Write-behind queue
        self.write_behind_queue: List[Tuple[str, Any, Optional[int]]] = []
        self.write_behind_task: Optional[asyncio.Task] = None
        
        if enable_write_behind:
            self.write_behind_task = asyncio.create_task(self._write_behind_loop())
        
        # Performance metrics
        self.local_hits = 0
        self.redis_hits = 0
        self.misses = 0
        
        logger.info(f"Initialized cache manager with local={enable_local_cache}, redis={redis_client is not None}")
    
    def _make_cache_key(self, key: str, namespace: str = "default") -> str:
        """Create namespaced cache key."""
        return f"{namespace}:{key}"
    
    async def get(self, key: str, namespace: str = "default") -> Optional[Any]:
        """
        Get value from cache (local first, then Redis).
        
        Args:
            key: Cache key
            namespace: Cache namespace
            
        Returns:
            Cached value or None
        """
        cache_key = self._make_cache_key(key, namespace)
        
        # Try local cache first
        if self.enable_local_cache:
            local_item = await self.local_strategy.get(cache_key)
            if local_item is not None:
                self.local_hits += 1
                return local_item.value
        
        # Try Redis cache
        if self.redis_client:
            redis_value = await self.redis_client.get(cache_key)
            if redis_value is not None:
                self.redis_hits += 1
                
                # Update local cache
                if self.enable_local_cache:
                    await self.local_strategy.set(cache_key, redis_value)
                
                return redis_value
        
        self.misses += 1
        return None
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        namespace: str = "default"
    ) -> bool:
        """
        Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds
            namespace: Cache namespace
            
        Returns:
            True if successful
        """
        cache_key = self._make_cache_key(key, namespace)
        
        # Update local cache
        if self.enable_local_cache:
            await self.local_strategy.set(cache_key, value, ttl)
        
        # Update Redis cache
        if self.redis_client:
            if self.enable_write_through:
                # Write-through: immediate write to Redis
                return await self.redis_client.set(cache_key, value, ttl)
            elif self.enable_write_behind:
                # Write-behind: queue for later write
                self.write_behind_queue.append((cache_key, value, ttl))
                return True
        
        return True
    
    async def delete(self, key: str, namespace: str = "default") -> bool:
        """
        Delete value from cache.
        
        Args:
            key: Cache key
            namespace: Cache namespace
            
        Returns:
            True if deleted
        """
        cache_key = self._make_cache_key(key, namespace)
        
        results = []
        
        # Delete from local cache
        if self.enable_local_cache:
            results.append(await self.local_strategy.delete(cache_key))
        
        # Delete from Redis cache
        if self.redis_client:
            redis_result = await self.redis_client.delete(cache_key)
            results.append(redis_result > 0)
        
        return any(results)
    
    async def clear(self, namespace: str = "default"):
        """
        Clear cache namespace.
        
        Args:
            namespace: Cache namespace to clear
        """
        # Clear local cache
        if self.enable_local_cache:
            await self.local_strategy.clear()
        
        # Clear Redis cache by pattern
        if self.redis_client:
            pattern = f"{namespace}:*"
            keys = await self.redis_client.keys(pattern)
            if keys:
                await self.redis_client.delete(*keys)
    
    async def get_multi(
        self,
        keys: List[str],
        namespace: str = "default"
    ) -> Dict[str, Any]:
        """
        Get multiple values from cache.
        
        Args:
            keys: List of cache keys
            namespace: Cache namespace
            
        Returns:
            Dictionary of key-value pairs
        """
        cache_keys = [self._make_cache_key(key, namespace) for key in keys]
        results = {}
        missing_keys = []
        
        # Try local cache first
        if self.enable_local_cache:
            for i, cache_key in enumerate(cache_keys):
                local_item = await self.local_strategy.get(cache_key)
                if local_item is not None:
                    results[keys[i]] = local_item.value
                    self.local_hits += 1
                else:
                    missing_keys.append(i)
        else:
            missing_keys = list(range(len(keys)))
        
        # Try Redis for missing keys
        if self.redis_client and missing_keys:
            missing_cache_keys = [cache_keys[i] for i in missing_keys]
            redis_values = await self.redis_client.get_multi(*missing_cache_keys)
            
            for i, value in enumerate(redis_values):
                key_idx = missing_keys[i]
                if value is not None:
                    results[keys[key_idx]] = value
                    self.redis_hits += 1
                    
                    # Update local cache
                    if self.enable_local_cache:
                        await self.local_strategy.set(cache_keys[key_idx], value)
                else:
                    self.misses += 1
        
        return results
    
    async def set_multi(
        self,
        mapping: Dict[str, Any],
        ttl: Optional[int] = None,
        namespace: str = "default"
    ) -> bool:
        """
        Set multiple values in cache.
        
        Args:
            mapping: Key-value mapping
            ttl: Time to live in seconds
            namespace: Cache namespace
            
        Returns:
            True if successful
        """
        cache_mapping = {
            self._make_cache_key(key, namespace): value
            for key, value in mapping.items()
        }
        
        # Update local cache
        if self.enable_local_cache:
            for cache_key, value in cache_mapping.items():
                await self.local_strategy.set(cache_key, value, ttl)
        
        # Update Redis cache
        if self.redis_client:
            if self.enable_write_through:
                return await self.redis_client.set_multi(cache_mapping, ttl)
            elif self.enable_write_behind:
                for cache_key, value in cache_mapping.items():
                    self.write_behind_queue.append((cache_key, value, ttl))
                return True
        
        return True
    
    async def _write_behind_loop(self):
        """Background task for write-behind operations."""
        while True:
            try:
                await asyncio.sleep(self.write_behind_delay)
                
                if not self.write_behind_queue or not self.redis_client:
                    continue
                
                # Process queued writes
                batch = self.write_behind_queue.copy()
                self.write_behind_queue.clear()
                
                if batch:
                    # Group by TTL for batch operations
                    ttl_groups = defaultdict(dict)
                    for key, value, ttl in batch:
                        ttl_groups[ttl][key] = value
                    
                    # Execute batch writes
                    for ttl, mapping in ttl_groups.items():
                        await self.redis_client.set_multi(mapping, ttl)
                    
                    logger.debug(f"Processed {len(batch)} write-behind operations")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in write-behind loop: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.local_hits + self.redis_hits + self.misses
        
        stats = {
            "local_hits": self.local_hits,
            "redis_hits": self.redis_hits,
            "misses": self.misses,
            "total_requests": total_requests,
            "hit_rate": (self.local_hits + self.redis_hits) / total_requests if total_requests > 0 else 0.0,
            "local_hit_rate": self.local_hits / total_requests if total_requests > 0 else 0.0,
        }
        
        if self.enable_local_cache:
            stats["local_cache"] = self.local_strategy.get_stats()
        
        if self.redis_client:
            stats["redis_cache"] = self.redis_client.get_stats()
        
        if self.enable_write_behind:
            stats["write_behind_queue_size"] = len(self.write_behind_queue)
        
        return stats
    
    def reset_stats(self):
        """Reset cache statistics."""
        self.local_hits = 0
        self.redis_hits = 0
        self.misses = 0
        
        if self.redis_client:
            self.redis_client.reset_stats()
    
    async def close(self):
        """Close cache manager."""
        if self.write_behind_task:
            self.write_behind_task.cancel()
            try:
                await self.write_behind_task
            except asyncio.CancelledError:
                pass
        
        if hasattr(self.local_strategy, 'close'):
            await self.local_strategy.close()
        
        if self.redis_client:
            await self.redis_client.close()


# Factory functions for common configurations

async def create_ttl_cache(
    max_size: int = 1000,
    default_ttl: int = 3600,
    redis_client: Optional[RedisClient] = None
) -> CacheManager:
    """Create TTL-based cache manager."""
    strategy = TTLStrategy(max_size, default_ttl)
    return CacheManager(redis_client, strategy)


async def create_lru_cache(
    max_size: int = 1000,
    redis_client: Optional[RedisClient] = None
) -> CacheManager:
    """Create LRU-based cache manager."""
    strategy = LRUStrategy(max_size)
    return CacheManager(redis_client, strategy)


async def create_adaptive_cache(
    max_size: int = 1000,
    redis_client: Optional[RedisClient] = None
) -> CacheManager:
    """Create adaptive cache manager."""
    strategy = AdaptiveStrategy(max_size)
    return CacheManager(redis_client, strategy)